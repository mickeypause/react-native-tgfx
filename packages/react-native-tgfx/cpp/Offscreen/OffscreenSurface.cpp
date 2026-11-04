#include "OffscreenSurface.hpp"

#include <cmath>
#include <condition_variable>
#include <deque>
#include <stdexcept>
#include <thread>

#include "OffscreenRender.hpp"
#include "Viewport.hpp"
#include "tgfx/core/Buffer.h"
#include "tgfx/core/Canvas.h"
#include "tgfx/core/Color.h"
#include "tgfx/core/ColorSpace.h"
#include "tgfx/core/Data.h"
#include "tgfx/core/Image.h"
#include "tgfx/core/ImageInfo.h"
#include "tgfx/core/Surface.h"
#include "tgfx/core/Task.h"
#include "tgfx/gpu/Context.h"

namespace tgfxrn {

namespace {

/**
 * One process-wide worker for every offscreen surface. More threads would buy
 * nothing: all offscreen GPU work serializes on the shared device's
 * lockContext anyway, and this also mirrors Android's single render thread.
 */
// ponytail: process-lifetime detached worker; add shutdown if tests ever need it.
class OffscreenWorker {
 public:
  static OffscreenWorker& instance() {
    // Leaked deliberately: a static's destructor would run at exit while the
    // detached thread still waits on the condition variable.
    static auto* worker = new OffscreenWorker();
    return *worker;
  }

  void enqueue(std::weak_ptr<OffscreenSurface> surface) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push_back(std::move(surface));
      if (!started_) {
        started_ = true;
        std::thread([this] { run(); }).detach();
      }
    }
    wake_.notify_one();
  }

 private:
  void run() {
    for (;;) {
      std::weak_ptr<OffscreenSurface> next;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [this] { return !queue_.empty(); });
        next = std::move(queue_.front());
        queue_.pop_front();
      }
      if (auto surface = next.lock()) {
        surface->renderNow();
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::weak_ptr<OffscreenSurface>> queue_;
  bool started_ = false;
};

}  // namespace

OffscreenSurface::OffscreenSurface(int width, int height, float pixelRatio, bool opaque,
                                   bool displayP3)
    : width_(width),
      height_(height),
      pixelRatio_(std::isfinite(pixelRatio) && pixelRatio > 0 ? pixelRatio : 1),
      opaque_(opaque),
      displayP3_(displayP3) {}

OffscreenSurface::~OffscreenSurface() {
  // No worker hop: whoever dropped the last reference gets the teardown, and
  // pending callbacks must not vanish silently.
  std::vector<std::pair<SnapshotRequest, SnapshotCallback>> saves;
  std::vector<ImageCallback> images;
  takeOutputs(saves, images);
  failOutputs(saves, images, "[tgfx] offscreen surface: disposed before the frame landed");
  if (surface_ != nullptr) {
    // A GPU object must die under its context's lock, not wherever GC ran.
    OffscreenContext locked;
    surface_ = nullptr;
  }
}

void OffscreenSurface::setScene(std::span<const double> words,
                                const std::vector<std::string>& packedStrings) {
  {
    // Ahead of the word checks: the delta carries no resync, so a dropped one desyncs it.
    std::lock_guard<std::mutex> lock(mutex_);
    if (applyStringTable(strings_, packedStrings)) {
      stringsHash_ = hashStrings(*strings_);
    }
  }
  if (words.size() < 2) {
    return;
  }
  // Casting a NaN or out-of-range double to size_t is undefined behaviour.
  double usedWord = words[1];
  if (!std::isfinite(usedWord) || usedWord < 2 ||
      usedWord > static_cast<double>(words.size())) {
    return;
  }
  words = words.first(static_cast<size_t>(usedWord));
  uint64_t wordsHash = hashWords(words);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t hash = hashScene(wordsHash, stringsHash_);
    if (hash == sceneHash_) {
      return;
    }
    sceneHash_ = hash;
    pendingScene_.assign(words.begin(), words.end());
    pendingStrings_ = strings_;
    scenePending_ = true;
    // A re-encode bakes current values as literals and may reassign slot
    // indices, so stale pushes must not outrank the new program's words.
    std::fill(stringPushed_.begin(), stringPushed_.end(), 0);
    stringGen_++;
  }
  // Staged, not rendered: nothing observes a headless frame until flush(),
  // toImage() or save() asks — a per-frame driver pushing setValues must not
  // spin the shared offscreen device.
}

void OffscreenSurface::setValues(std::vector<double> values) {
  std::lock_guard<std::mutex> lock(mutex_);
  values_ = std::move(values);
}

void OffscreenSurface::setStringValues(const std::vector<double>& indices,
                                       const std::vector<std::string>& values) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = std::min(indices.size(), values.size());
    for (size_t i = 0; i < count; i++) {
      double at = indices[i];
      if (!(at >= 0) || at >= 65536) {
        continue;
      }
      auto slot = static_cast<size_t>(at);
      if (slot >= stringValues_.size()) {
        stringValues_.resize(slot + 1);
        stringPushed_.resize(slot + 1, 0);
      }
      stringValues_[slot] = values[i];
      stringPushed_[slot] = 1;
    }
    stringGen_++;
  }
}

bool OffscreenSurface::setViewport(float zoom, float offsetX, float offsetY) {
  // Rejected whole rather than clamped, exactly as the view does.
  if (!std::isfinite(zoom) || zoom <= 0.0f || !std::isfinite(offsetX) ||
      !std::isfinite(offsetY)) {
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (zoom == viewportZoom_ && offsetX == viewportX_ && offsetY == viewportY_) {
      return false;
    }
    viewportZoom_ = zoom;
    viewportX_ = offsetX;
    viewportY_ = offsetY;
  }
  return true;
}

std::vector<double> OffscreenSurface::hitTest(float x, float y, bool precise) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (dead_) {
      return {};
    }
  }
  // The staged program may not have drained yet; compile it here so a
  // setScene-then-hitTest sequence answers instead of missing everything.
  applyPendingScene();
  std::vector<double> snapshot;
  std::vector<std::string> strings;
  std::vector<uint8_t> pushed;
  float zoom = 1.0f;
  float offsetX = 0.0f;
  float offsetY = 0.0f;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot = values_;
    strings = stringValues_;
    pushed = stringPushed_;
    zoom = viewportZoom_;
    offsetX = viewportX_;
    offsetY = viewportY_;
  }
  // Canvas points, staged state — the view's contract; SceneRenderer::hitTest
  // takes the program under its own lock, so this is safe off the worker.
  return scene_.hitTest(x, y, precise, snapshot, strings, pushed,
                        rootMatrix(1.0f, zoom, offsetX, offsetY));
}

void OffscreenSurface::flush() {
  wake();
}

void OffscreenSurface::requestSave(SnapshotRequest request, SnapshotCallback callback) {
  bool dead = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    dead = dead_;
    if (!dead) {
      pendingSaves_.emplace_back(std::move(request), std::move(callback));
    }
  }
  if (dead) {
    if (callback) {
      callback(nullptr, 0, 0, "[tgfx] offscreen surface: already disposed");
    }
    return;
  }
  wake();
}

void OffscreenSurface::requestImage(ImageCallback callback) {
  bool dead = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    dead = dead_;
    if (!dead) {
      pendingImages_.push_back(std::move(callback));
    }
  }
  if (dead) {
    if (callback) {
      callback({}, "[tgfx] offscreen surface: already disposed");
    }
    return;
  }
  wake();
}

void OffscreenSurface::dispose() {
  std::vector<std::pair<SnapshotRequest, SnapshotCallback>> saves;
  std::vector<ImageCallback> images;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (dead_) {
      return;
    }
    dead_ = true;
    saves.swap(pendingSaves_);
    images.swap(pendingImages_);
  }
  failOutputs(saves, images, "[tgfx] offscreen surface: disposed before the frame landed");
  // The worker frees the GPU surface on its next drain, under the context
  // lock; touching worker-owned surface_ from here would race a live render.
  OffscreenWorker::instance().enqueue(weak_from_this());
}

void OffscreenSurface::applyPendingScene() {
  // One critical section for claim and publish; see sceneApplyMutex_.
  std::lock_guard<std::mutex> apply(sceneApplyMutex_);
  bool hasScene = false;
  std::vector<double> sceneWords;
  std::shared_ptr<const StringTable> sceneStrings;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (scenePending_) {
      sceneWords.swap(pendingScene_);
      sceneStrings = std::move(pendingStrings_);
      scenePending_ = false;
      hasScene = true;
    }
  }
  if (hasScene) {
    scene_.setScene(sceneWords, *sceneStrings);
  }
}

void OffscreenSurface::wake() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (dead_) {
      return;
    }
  }
  // One queue entry per surface at a time: a burst of flushes coalesces into
  // the next drain, which renders the latest staged state anyway.
  if (queued_.exchange(true, std::memory_order_acq_rel)) {
    return;
  }
  OffscreenWorker::instance().enqueue(weak_from_this());
}

bool OffscreenSurface::takeOutputs(
    std::vector<std::pair<SnapshotRequest, SnapshotCallback>>& saves,
    std::vector<ImageCallback>& images) {
  std::lock_guard<std::mutex> lock(mutex_);
  saves.swap(pendingSaves_);
  images.swap(pendingImages_);
  return !saves.empty() || !images.empty();
}

void OffscreenSurface::failOutputs(
    std::vector<std::pair<SnapshotRequest, SnapshotCallback>>& saves,
    std::vector<ImageCallback>& images, const std::string& error) {
  for (auto& [request, callback] : saves) {
    if (callback) {
      callback(nullptr, 0, 0, error);
    }
  }
  for (auto& callback : images) {
    if (callback) {
      callback({}, error);
    }
  }
  saves.clear();
  images.clear();
}

void OffscreenSurface::renderNow() {
  // Cleared before staging: a setter landing mid-drain re-queues a fresh pass.
  queued_.store(false, std::memory_order_release);

  bool dead = false;
  float zoom = 1.0f;
  float offsetX = 0.0f;
  float offsetY = 0.0f;
  std::vector<std::pair<SnapshotRequest, SnapshotCallback>> saves;
  std::vector<ImageCallback> images;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    dead = dead_;
    if (!dead) {
      drawValues_.assign(values_.begin(), values_.end());
      if (drawStringGen_ != stringGen_) {
        drawStringValues_.assign(stringValues_.begin(), stringValues_.end());
        drawStringPushed_.assign(stringPushed_.begin(), stringPushed_.end());
        drawStringGen_ = stringGen_;
      }
      zoom = viewportZoom_;
      offsetX = viewportX_;
      offsetY = viewportY_;
      saves.swap(pendingSaves_);
      images.swap(pendingImages_);
    }
  }
  if (dead) {
    // dispose()'s deferred teardown: free the GPU surface under its context.
    if (surface_ != nullptr) {
      try {
        OffscreenContext locked;
        surface_ = nullptr;
      } catch (const std::exception&) {
        surface_ = nullptr;  // no GPU left to free it under
      }
    }
    return;
  }

  // CPU-only, ahead of the GPU lock: a context failure must not eat the scene.
  applyPendingScene();

  // Everything the callbacks need, produced under the context lock and
  // delivered after it: a callback may re-enter this surface or block.
  std::shared_ptr<tgfx::Data> pixels;
  tgfx::ImageInfo info;
  std::string failure;
  try {
    OffscreenContext locked;
    auto* context = locked.get();
    {
      // dispose() may have staged between our copy and here; rendering on
      // would recreate a surface only the worker could ever free again.
      std::lock_guard<std::mutex> lock(mutex_);
      dead = dead_;
    }
    if (dead) {
      surface_ = nullptr;
      failure = "[tgfx] offscreen surface: already disposed";
    } else {
      if (surface_ == nullptr) {
        auto space =
            displayP3_ ? tgfx::ColorSpace::DisplayP3() : tgfx::ColorSpace::SRGB();
        double scaledWidth = static_cast<double>(width_) * pixelRatio_;
        double scaledHeight = static_cast<double>(height_) * pixelRatio_;
        surface_ = tgfx::Surface::Make(context, static_cast<int>(scaledWidth + 0.5),
                                       static_cast<int>(scaledHeight + 0.5), false, 1,
                                       false, 0, space);
        if (surface_ == nullptr) {
          throw std::runtime_error(
              "[tgfx] offscreen surface: could not allocate a GPU surface");
        }
      }
      auto* canvas = surface_->getCanvas();
      canvas->clear(opaque_ ? tgfx::Color::Black() : tgfx::Color::Transparent());
      canvas->save();
      canvas->concat(rootMatrix(pixelRatio_, zoom, offsetX, offsetY));
      scene_.render(canvas, drawValues_, drawStringValues_, drawStringGen_,
                    drawStringPushed_);
      canvas->restore();

      if (saves.empty() && images.empty()) {
        // GPU-resident between flushes: submit without waiting, no readback.
        context->flushAndSubmit(false);
        return;
      }
      // A readback was asked for: wait for the pass, then read the frame once.
      context->flushAndSubmit(true);
      info = tgfx::ImageInfo::Make(
          surface_->width(), surface_->height(), tgfx::ColorType::RGBA_8888,
          opaque_ ? tgfx::AlphaType::Opaque : tgfx::AlphaType::Premultiplied, 0,
          displayP3_ ? tgfx::ColorSpace::DisplayP3() : tgfx::ColorSpace::SRGB());
      tgfx::Buffer buffer(info.byteSize());
      if (buffer.isEmpty() || !surface_->readPixels(info, buffer.data())) {
        throw std::runtime_error("[tgfx] offscreen surface: could not read the frame back");
      }
      pixels = buffer.release();
    }
  } catch (const std::exception& e) {
    failure = e.what();
  }
  // The context lock is released; now the callbacks.
  if (!failure.empty() || pixels == nullptr) {
    failOutputs(saves, images,
                failure.empty() ? "[tgfx] offscreen surface: no pixels" : failure);
    return;
  }
  for (auto& callback : images) {
    // CPU pixels, not the texture: the caller draws in a different context.
    auto image = tgfx::Image::MakeFrom(info, pixels);
    auto handle = ImageRegistry::instance().addExternal(std::move(image), info.byteSize());
    if (handle.handle == 0) {
      callback({}, "[tgfx] offscreen surface: could not register the frame");
    } else {
      callback(handle, "");
    }
  }
  for (auto& [request, callback] : saves) {
    // The encode tail off this thread: another surface may be waiting to render.
    tgfx::Task::Run(
        [pixels, info, request = std::move(request), callback = std::move(callback)]() {
          finishSnapshot(pixels, info, request, callback);
        });
  }
}

}  // namespace tgfxrn
