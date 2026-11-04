#include "ImageRegistry.hpp"

#include "Log.hpp"
#include "tgfx/core/Image.h"
#include "tgfx/core/ImageBuffer.h"
#include "tgfx/core/ImageCodec.h"

namespace tgfxrn {

namespace {

/// Decode into a fully-decoded Image. Synchronous by construction: never call from a render thread.
std::shared_ptr<tgfx::Image> decode(const tgfx::ImageCodec& codec) {
  auto buffer = codec.makeBuffer();
  if (buffer == nullptr) {
    return nullptr;
  }
  auto image = tgfx::Image::MakeFrom(std::move(buffer));
  if (image == nullptr) {
    return nullptr;
  }
  // MakeFrom(ImageBuffer) knows nothing about EXIF, which MakeFromEncoded applied for us.
  return image->makeOriented(codec.orientation());
}

size_t byteSizeOf(const tgfx::Image& image) {
  auto pixels = static_cast<size_t>(image.width()) * static_cast<size_t>(image.height());
  return image.isAlphaOnly() ? pixels : pixels * 4;
}

}  // namespace

ImageRegistry& ImageRegistry::instance() {
  static ImageRegistry registry;
  return registry;
}

ImageRegistry::~ImageRegistry() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  wake_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

ImageHandle ImageRegistry::add(std::shared_ptr<tgfx::ImageCodec> codec) {
  if (codec == nullptr) {
    return {};
  }
  auto image = decode(*codec);
  if (image == nullptr) {
    TGFXRN_LOGW("failed to decode a %dx%d image", codec->width(), codec->height());
    return {};
  }
  std::lock_guard<std::mutex> lock(mutex_);
  int32_t handle = nextHandle_++;
  Entry entry;
  entry.codec = std::move(codec);
  entry.width = image->width();
  entry.height = image->height();
  entry.bytes = byteSizeOf(*image);
  entry.decoded = std::move(image);
  decodeCount_++;
  auto& stored = entries_.emplace(handle, std::move(entry)).first->second;
  touchLocked(handle, stored);
  return {handle, stored.width, stored.height};
}

ImageHandle ImageRegistry::addExternal(std::shared_ptr<tgfx::Image> image, size_t ownedBytes,
                                       bool live) {
  if (image == nullptr) {
    return {};
  }
  std::lock_guard<std::mutex> lock(mutex_);
  int32_t handle = nextHandle_++;
  Entry entry;
  entry.width = image->width();
  entry.height = image->height();
  entry.decoded = std::move(image);
  // Reported to the JS GC but never touched into the LRU, so it is never evicted.
  entry.bytes = ownedBytes;
  entry.live = live;
  auto& stored = entries_.emplace(handle, std::move(entry)).first->second;
  return {handle, stored.width, stored.height};
}

ImageHandle ImageRegistry::addStream(std::function<std::shared_ptr<tgfx::Image>()> produce,
                                     int width, int height) {
  if (produce == nullptr) {
    return {};
  }
  std::lock_guard<std::mutex> lock(mutex_);
  int32_t handle = nextHandle_++;
  Entry entry;
  entry.width = width;
  entry.height = height;
  entry.produce = std::move(produce);
  // Never touched into the LRU: the frames are the producer's, not ours to budget.
  auto& stored = entries_.emplace(handle, std::move(entry)).first->second;
  return {handle, stored.width, stored.height};
}

void ImageRegistry::invalidate() {
  bumpEpoch();
}

bool ImageRegistry::replaceExternal(int32_t handle, std::shared_ptr<tgfx::Image> image,
                                    size_t ownedBytes) {
  if (image == nullptr) {
    return false;
  }
  // Released after the lock: the drop hands the slot back, and must not hold the render thread.
  std::shared_ptr<tgfx::Image> previous;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(handle);
    if (it == entries_.end() || !it->second.external() || it->second.produce != nullptr) {
      return false;
    }
    it->second.width = image->width();
    it->second.height = image->height();
    it->second.bytes = ownedBytes;
    previous = std::move(it->second.decoded);
    it->second.decoded = std::move(image);
  }
  bumpEpoch();
  return true;
}

void ImageRegistry::remove(int32_t handle) {
  std::shared_ptr<tgfx::Image> previous;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(handle);
    if (it == entries_.end()) {
      return;
    }
    if (it->second.resident) {
      lru_.erase(it->second.at);
      cachedBytes_ -= it->second.bytes;
    }
    previous = std::move(it->second.decoded);
    wanted_.erase(handle);
    entries_.erase(it);
  }
  bumpEpoch();
}

std::shared_ptr<tgfx::Image> ImageRegistry::acquire(int32_t handle, bool* external) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = entries_.find(handle);
  if (it == entries_.end()) {
    return nullptr;
  }
  auto& entry = it->second;
  if (external != nullptr) {
    *external = entry.external();
  }
  if (entry.produce != nullptr) {
    // Outside the lock: the producer takes its own, and this runs on the render thread.
    auto produce = entry.produce;
    lock.unlock();
    return produce();
  }
  if (entry.external()) {
    return entry.decoded;
  }
  if (entry.decoded == nullptr) {
    // Only after the budget dropped these pixels. The caller is the render thread, so the worker re-decodes.
    scheduleDecodeLocked(handle);
    return nullptr;
  }
  // On a hit too: recency that only records insertion is FIFO, which evicts what is on screen.
  touchLocked(handle, entry);
  return entry.decoded;
}

std::shared_ptr<tgfx::Image> ImageRegistry::sourceImage(int32_t handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(handle);
  if (it == entries_.end()) {
    return nullptr;
  }
  auto& entry = it->second;
  if (entry.produce != nullptr || entry.live) {
    // Whatever frame is current is not a stable source, and its buffer is recycled.
    return nullptr;
  }
  if (entry.decoded != nullptr) {
    // The pixels the renderer already holds: deriving from the codec would decode a
    // second time, on the render thread, and upload a second full-size texture.
    return entry.decoded;
  }
  if (entry.codec != nullptr && entry.lazy == nullptr) {
    // Evicted, so rebuild from the codec. Cached because MakeFrom mints a fresh GPU key.
    entry.lazy = tgfx::Image::MakeFrom(entry.codec);
  }
  return entry.lazy;
}

uint64_t ImageRegistry::addWakeListener(std::function<void()> wake) {
  std::lock_guard<std::mutex> lock(listenersMutex_);
  uint64_t token = nextListener_++;
  listeners_.emplace_back(token, std::move(wake));
  return token;
}

void ImageRegistry::removeWakeListener(uint64_t token) {
  std::lock_guard<std::mutex> lock(listenersMutex_);
  for (auto it = listeners_.begin(); it != listeners_.end(); ++it) {
    if (it->first == token) {
      listeners_.erase(it);
      return;
    }
  }
}

void ImageRegistry::wakeAll() {
  // Under the listener lock, not through a copy: a wake only posts work and must not allocate.
  std::lock_guard<std::mutex> lock(listenersMutex_);
  for (auto& [token, wake] : listeners_) {
    (void)token;
    wake();
  }
}

void ImageRegistry::bumpEpoch() {
  epoch_.fetch_add(1, std::memory_order_release);
  wakeAll();
}

void ImageRegistry::scheduleDecodeLocked(int32_t handle) {
  if (!wanted_.insert(handle).second) {
    return;
  }
  if (!worker_.joinable()) {
    worker_ = std::thread([this] { decodeWorker(); });
  }
  wake_.notify_one();
}

void ImageRegistry::decodeWorker() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (true) {
    wake_.wait(lock, [this] { return stopping_ || !wanted_.empty(); });
    if (stopping_) {
      return;
    }
    int32_t handle = *wanted_.begin();
    auto it = entries_.find(handle);
    if (it == entries_.end() || it->second.codec == nullptr ||
        it->second.decoded != nullptr) {
      wanted_.erase(handle);
      wake_.notify_all();
      continue;
    }
    auto codec = it->second.codec;
    TGFXRN_LOGW("re-decoding image %d, evicted under a %zu-byte budget", handle,
                cacheLimit_);
    lock.unlock();
    auto image = decode(*codec);
    lock.lock();
    wanted_.erase(handle);
    auto at = entries_.find(handle);
    // The handle can have been disposed while the lock was down; `codec` identity proves it.
    bool landed = false;
    if (image != nullptr && at != entries_.end() && at->second.codec == codec &&
        at->second.decoded == nullptr) {
      at->second.bytes = byteSizeOf(*image);
      at->second.decoded = std::move(image);
      decodeCount_++;
      touchLocked(handle, at->second);
      landed = true;
    }
    wake_.notify_all();
    if (landed) {
      // Outside `mutex_`: a woken render thread resolving the handle would deadlock on it.
      lock.unlock();
      bumpEpoch();
      lock.lock();
    }
  }
}

void ImageRegistry::touchLocked(int32_t handle, Entry& entry) {
  if (entry.resident) {
    if (lru_.front() != handle) {
      lru_.splice(lru_.begin(), lru_, entry.at);
    }
    return;
  }
  lru_.push_front(handle);
  entry.at = lru_.begin();
  entry.resident = true;
  cachedBytes_ += entry.bytes;
  evictLocked();
}

void ImageRegistry::evictLocked() {
  while (cachedBytes_ > cacheLimit_ && lru_.size() > 1) {
    int32_t victim = lru_.back();
    lru_.pop_back();
    auto it = entries_.find(victim);
    if (it != entries_.end()) {
      cachedBytes_ -= it->second.bytes;
      it->second.decoded = nullptr;
      it->second.resident = false;
    }
  }
}

void ImageRegistry::purgeResident() {
  // The pixels die outside the lock; a decode worker touching the registry in
  // an Image destructor path must not find it held.
  std::vector<std::shared_ptr<tgfx::Image>> dropped;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int32_t handle : lru_) {
      auto it = entries_.find(handle);
      if (it == entries_.end()) {
        continue;
      }
      cachedBytes_ -= it->second.bytes;
      dropped.push_back(std::move(it->second.decoded));
      dropped.push_back(std::move(it->second.lazy));
      it->second.resident = false;
    }
    lru_.clear();
  }
  dropped.clear();
  // Canvases redraw and re-decode what is actually on screen.
  bumpEpoch();
}

void ImageRegistry::setCacheLimit(size_t bytes) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    cacheLimit_ = bytes;
    evictLocked();
  }
  bumpEpoch();
}

size_t ImageRegistry::cacheLimit() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return cacheLimit_;
}

size_t ImageRegistry::cachedBytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return cachedBytes_;
}

size_t ImageRegistry::byteSize(int32_t handle) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(handle);
  return it == entries_.end() ? 0 : it->second.bytes;
}

size_t ImageRegistry::decodeCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return decodeCount_;
}

void ImageRegistry::clear() {
  waitForDecodes();
  std::lock_guard<std::mutex> lock(mutex_);
  entries_.clear();
  lru_.clear();
  cachedBytes_ = 0;
  decodeCount_ = 0;
}

void ImageRegistry::waitForDecodes() {
  std::unique_lock<std::mutex> lock(mutex_);
  wake_.wait(lock, [this] { return wanted_.empty(); });
}

}  // namespace tgfxrn
