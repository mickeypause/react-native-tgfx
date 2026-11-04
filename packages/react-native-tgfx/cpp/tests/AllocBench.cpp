// Heap allocations per frame, and the memoised <Text> layout. See docs/text-layout.md.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <span>
#include <string>
#include <vector>

#include <cstring>

#include "FontRegistry.hpp"
#include "ImageRegistry.hpp"
#include "TgfxFrameSink.hpp"
#include "TgfxScene.hpp"
#include "tgfx/core/Canvas.h"
#include "tgfx/core/Color.h"
#include "tgfx/core/Data.h"
#include "tgfx/core/Font.h"
#include "tgfx/core/Image.h"
#include "tgfx/core/ImageCodec.h"
#include "tgfx/core/ImageInfo.h"
#include "tgfx/core/Paint.h"
#include "tgfx/core/Pixmap.h"
#include "tgfx/core/Surface.h"
#include "tgfx/gpu/Context.h"
#include "tgfx/gpu/Device.h"
#include "tgfx/gpu/metal/MetalDevice.h"

#if defined(__APPLE__)
#include <CoreVideo/CoreVideo.h>
#endif

static std::atomic<size_t> gAllocs{0};
static std::atomic<bool> gCounting{false};

void* operator new(size_t size) {
  if (gCounting.load(std::memory_order_relaxed)) {
    gAllocs.fetch_add(1, std::memory_order_relaxed);
  }
  void* p = std::malloc(size ? size : 1);
  if (p == nullptr)
    throw std::bad_alloc();
  return p;
}
void operator delete(void* p) noexcept {
  std::free(p);
}
void operator delete(void* p, size_t) noexcept {
  std::free(p);
}

namespace {

constexpr int kWidth = 256;
constexpr int kHeight = 256;
constexpr int kNodes = 200;

struct Writer {
  std::vector<double> words{0, 0, 0};
  std::vector<double> patches;
  void push(double v) { words.push_back(v); }
  void pushAnim(double literal, int slot = -1) {
    if (slot >= 0) {
      patches.push_back(static_cast<double>(words.size()));
      patches.push_back(slot);
    }
    words.push_back(literal);
  }
  std::vector<double> finish() {
    auto patchStart = words.size();
    words.push_back(static_cast<double>(patches.size() / 2));
    for (double p : patches)
      words.push_back(p);
    words.push_back(0);  // empty string-patch table
    words[0] = tgfxrn::kSceneVersion;
    words[1] = static_cast<double>(words.size());
    words[2] = static_cast<double>(patchStart);
    return words;
  }
};

std::vector<double> scene(bool filters) {
  Writer w;
  w.push(kNodes);
  for (int i = 0; i < kNodes; i++) {
    w.push(1);  // Rect
    w.pushAnim((i * 7) % 200);
    w.pushAnim((i * 13) % 200);
    w.pushAnim(24);
    w.pushAnim(24);
    w.push(0xff3366cc);
    w.push(filters ? (2 | 4) : 2);
    w.pushAnim(1);
    w.push(0);
    w.push(0);
    w.pushAnim(1, 0);  // animated alpha: applyPaint runs every frame
    if (filters) {
      w.push(24);  // multiply
      w.push(0);   // colour matrix
      const double m[20] = {0.2, 0.7, 0.1, 0, 0, 0.2, 0.7, 0.1, 0, 0,
                            0.2, 0.7, 0.1, 0, 0, 0, 0, 0, 1, 0};
      for (double v : m)
        w.push(v);
      w.push(-1);  // no image filter
    }
    w.push(-1);  // no shader
  }
  return w.finish();
}

tgfx::Device* device() {
  static auto shared = tgfx::MetalDevice::Make();
  return shared.get();
}

size_t frameAllocations(tgfxrn::SceneRenderer& renderer, bool count,
                        size_t* drawn) {
  auto* context = device()->lockContext();
  auto surface = tgfx::Surface::Make(context, kWidth, kHeight);
  auto* canvas = surface->getCanvas();
  canvas->clear();
  gAllocs.store(0);
  gCounting.store(count);
  const double values[1] = {0.9};
  renderer.render(canvas, values);
  gCounting.store(false);
  context->flushAndSubmit(true);
  size_t allocations = gAllocs.load();

  auto info = tgfx::ImageInfo::Make(kWidth, kHeight, tgfx::ColorType::RGBA_8888);
  std::vector<uint8_t> pixels(static_cast<size_t>(info.byteSize()));
  surface->readPixels(info, pixels.data());
  device()->unlock();
  *drawn = 0;
  for (size_t at = 3; at < pixels.size(); at += 4) {
    if (pixels[at] != 0) {
      (*drawn)++;
    }
  }
  return allocations;
}

size_t measure(bool filters, size_t* drawn) {
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(scene(filters), {});
  for (int i = 0; i < 3; i++) {
    frameAllocations(renderer, false, drawn);  // warm up
  }
  size_t best = SIZE_MAX;
  for (int i = 0; i < 5; i++) {
    best = std::min(best, frameAllocations(renderer, true, drawn));
  }
  return best;
}

// Child shaders settled at setScene must cost no more per frame than none; the GPU probe is memoryUsage().

constexpr int kShaderNodes = 50;

std::vector<double> childShaderScene(int32_t childHandle, bool animatedUniform,
                                     bool bare = false) {
  Writer w;
  w.push(kShaderNodes);
  for (int i = 0; i < kShaderNodes; i++) {
    w.push(1);  // Rect
    w.pushAnim((i * 7) % 200);
    w.pushAnim((i * 13) % 200);
    w.pushAnim(24);
    w.pushAnim(24);
    w.push(0xffffffff);
    w.push(2);
    w.pushAnim(1);
    w.push(0);
    w.push(0);
    w.pushAnim(1, 0);  // animated alpha: applyPaint runs every frame
    if (bare) {
      w.push(7);  // <ImageShader> paint, no runtime shader above it
      w.push(childHandle);
      w.push(0);
      w.push(0);
      w.push(1);
      w.push(0);
      continue;
    }
    w.push(6);  // runtime shader
    w.push(0);  // body
    w.push(1);  // empty preamble
    w.push(0);  // no functions
    w.push(1);  // one uniform
    w.push(2);  // "uLevel"
    w.push(1);
    w.pushAnim(0.5, animatedUniform ? 0 : -1);
    if (childHandle == 0) {
      w.push(0);  // no children
    } else {
      w.push(1);
      w.push(7);  // <ImageShader> child
      w.push(childHandle);
      w.push(0);  // clamp
      w.push(0);  // clamp
      w.push(1);  // linear
      w.push(0);  // no fit rect
    }
  }
  return w.finish();
}

size_t measureShader(const std::vector<double>& words,
                     const std::vector<std::string>& strings, size_t* drawn) {
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(words, strings);
  for (int i = 0; i < 3; i++) {
    frameAllocations(renderer, false, drawn);
  }
  size_t best = SIZE_MAX;
  for (int i = 0; i < 5; i++) {
    best = std::min(best, frameAllocations(renderer, true, drawn));
  }
  return best;
}

size_t gpuBytesAfter(tgfxrn::SceneRenderer& renderer, int frames, size_t* drawn) {
  for (int i = 0; i < frames; i++) {
    frameAllocations(renderer, false, drawn);
  }
  auto* context = device()->lockContext();
  size_t bytes = context->memoryUsage();
  device()->unlock();
  return bytes;
}

tgfxrn::ImageHandle registerSolid() {
  std::vector<uint8_t> pixels(4 * 4 * 4, 0);
  for (size_t at = 0; at < pixels.size(); at += 4) {
    pixels[at] = 40;
    pixels[at + 1] = 160;
    pixels[at + 2] = 220;
    pixels[at + 3] = 255;
  }
  auto info = tgfx::ImageInfo::Make(4, 4, tgfx::ColorType::RGBA_8888,
                                    tgfx::AlphaType::Premultiplied);
  tgfx::Pixmap pixmap(info, pixels.data());
  auto data = tgfx::ImageCodec::Encode(pixmap, tgfx::EncodedFormat::PNG, 100);
  if (data == nullptr) {
    return {};
  }
  return tgfxrn::ImageRegistry::instance().add(tgfx::ImageCodec::MakeFrom(std::move(data)));
}

bool reportChildShaders() {
  const std::vector<std::string> strings = {"return TGFX_CHILD_0(fragCoord) * uLevel;", "",
                                            "uLevel"};
  const std::vector<std::string> childless = {"return vec4(uLevel, 0.2, 0.4, 1.0);", "",
                                              "uLevel"};
  auto image = registerSolid();
  if (image.handle == 0) {
    std::printf("could not register the bench image\n");
    return false;
  }
  auto& registry = tgfxrn::ImageRegistry::instance();
  auto external = registry.addExternal(registry.acquire(image.handle));

  size_t bareDrawn = 0;
  size_t bare = measureShader(childShaderScene(image.handle, false, true), strings, &bareDrawn);
  size_t noChildDrawn = 0, staticDrawn = 0, dynamicDrawn = 0, animatedDrawn = 0;
  size_t noChild = measureShader(childShaderScene(0, false), childless, &noChildDrawn);
  size_t staticChild = measureShader(childShaderScene(image.handle, false), strings, &staticDrawn);
  size_t dynamicChild =
      measureShader(childShaderScene(external.handle, false), strings, &dynamicDrawn);
  size_t animated = measureShader(childShaderScene(image.handle, true), strings, &animatedDrawn);
  if (bareDrawn == 0 || noChildDrawn == 0 || staticDrawn == 0 || dynamicDrawn == 0 ||
      animatedDrawn == 0) {
    std::printf("nothing was drawn (%zu / %zu / %zu / %zu / %zu opaque pixels)\n", bareDrawn,
                noChildDrawn, staticDrawn, dynamicDrawn, animatedDrawn);
    return false;
  }
  std::printf("%d animated-alpha nodes, allocations during render():\n", kShaderNodes);
  std::printf("  <ImageShader>, no shader     %zu  (%.2f per node)\n", bare,
              double(bare) / kShaderNodes);
  std::printf("  <Shader>, no child           %zu  (%.2f per node)\n", noChild,
              double(noChild) / kShaderNodes);
  std::printf("  static <ImageShader> child   %zu  (%.2f per node)\n", staticChild,
              double(staticChild) / kShaderNodes);
  std::printf("  hardware-buffer child        %zu  (%.2f per node)\n", dynamicChild,
              double(dynamicChild) / kShaderNodes);
  std::printf("  animated uniform             %zu  (%.2f per node)\n", animated,
              double(animated) / kShaderNodes);

  // The GPU probe: the imported texture must not be uploaded again to be sampled as a child.
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(childShaderScene(external.handle, false), strings);
  size_t warm = gpuBytesAfter(renderer, 5, &dynamicDrawn);
  size_t later = gpuBytesAfter(renderer, 60, &dynamicDrawn);
  std::printf("  GPU bytes after 5 frames %zu, after 65 %zu%s\n", warm, later,
              later > warm ? "  <-- GREW" : "");
  registry.remove(external.handle);
  return dynamicDrawn != 0 && later <= warm;
}

// Text: the mean over enough frames for the rate to matter, since a moving size costs more.

constexpr int kTextNodes = 100;
constexpr int kTextFrames = 37;
constexpr float kBaseSize = 18;

std::vector<double> textScene() {
  Writer w;
  w.push(kTextNodes);
  for (int i = 0; i < kTextNodes; i++) {
    w.push(8);  // Text
    w.pushAnim(2);
    w.pushAnim(12 + (i % 4) * 12);
    w.pushAnim(kBaseSize, 0);
    w.push(1);  // "Helvetica"
    w.push(400);
    w.push(0);  // style bits
    w.push(0);  // letterSpacing
    w.push(1);  // one run
    w.push(0);  // "Benchmark"
    w.push(2);  // run family ("")
    w.push(0);  // run weight (inherit)
    w.push(0);  // run italic (inherit)
    w.push(0);  // run letterSpacing
    w.push(0);  // run size (inherit)
    w.push(0);  // run hasColor
    w.push(0);  // run color
    w.push(0);  // no wrap
    w.push(0);  // align left
    w.push(0);  // natural line height
    w.push(0);  // no maxLines
    w.push(2);  // empty ellipsis
    w.push(0xff000000);
    w.push(2);
    w.pushAnim(1);
    w.push(0);
    w.push(0);
    w.pushAnim(1);
    w.push(-1);  // no shader
  }
  return w.finish();
}

/// The pre-#142 path: a Font re-sized per draw, rebuilding a TextBlob inside drawSimpleText.
void drawOldPath(tgfx::Canvas* canvas, const tgfx::Font& base, const tgfx::Paint& paint,
                 float size) {
  for (int i = 0; i < kTextNodes; i++) {
    tgfx::Font sized = base;
    sized.setSize(size);
    canvas->drawSimpleText("Benchmark", 2, 12 + static_cast<float>(i % 4) * 12, sized, paint);
  }
}

template <typename Draw>
size_t meanAllocations(Draw&& draw, size_t* drawn) {
  size_t total = 0;
  for (int frame = 0; frame < kTextFrames; frame++) {
    auto* context = device()->lockContext();
    auto surface = tgfx::Surface::Make(context, kWidth, kHeight);
    auto* canvas = surface->getCanvas();
    canvas->clear();
    gAllocs.store(0);
    gCounting.store(true);
    draw(canvas, frame);
    gCounting.store(false);
    total += gAllocs.load();
    context->flushAndSubmit(true);
    auto info = tgfx::ImageInfo::Make(kWidth, kHeight, tgfx::ColorType::RGBA_8888);
    std::vector<uint8_t> pixels(static_cast<size_t>(info.byteSize()));
    surface->readPixels(info, pixels.data());
    device()->unlock();
    *drawn = 0;
    for (size_t at = 3; at < pixels.size(); at += 4) {
      if (pixels[at] != 0) {
        (*drawn)++;
      }
    }
  }
  return total / kTextFrames;
}

bool reportText() {
  tgfxrn::FontRegistry::instance().clear();
  auto match = tgfxrn::FontRegistry::instance().match("Helvetica", 400, false);
  if (match.typeface == nullptr) {
    std::printf("no typeface for the text bench\n");
    return false;
  }
  tgfx::Font base(match.typeface, 1);
  tgfx::Paint paint;
  paint.setColor(tgfx::Color::Black());
  auto words = textScene();
  const std::vector<std::string> strings{"Benchmark", "Helvetica", ""};

  std::printf("%d <Text> nodes, mean allocations per frame over %d frames:\n", kTextNodes,
              kTextFrames);
  std::printf("  fontSize          old (drawSimpleText)   new (memoised blob)\n");
  for (float step : {0.0f, 0.05f, 0.10f, 0.25f, 1.00f}) {
    size_t oldDrawn = 0, newDrawn = 0;
    size_t oldAllocs = meanAllocations(
        [&](tgfx::Canvas* canvas, int frame) {
          drawOldPath(canvas, base, paint, kBaseSize + step * static_cast<float>(frame));
        },
        &oldDrawn);
    tgfxrn::SceneRenderer renderer;
    renderer.setScene(words, strings);
    size_t newAllocs = meanAllocations(
        [&](tgfx::Canvas* canvas, int frame) {
          double size = kBaseSize + static_cast<double>(step) * frame;
          renderer.render(canvas, std::span<const double>(&size, 1));
        },
        &newDrawn);
    if (oldDrawn == 0 || newDrawn == 0) {
      std::printf("nothing was drawn (%zu / %zu opaque pixels)\n", oldDrawn, newDrawn);
      return false;
    }
    char label[32];
    if (step == 0) {
      std::snprintf(label, sizeof(label), "static");
    } else {
      std::snprintf(label, sizeof(label), "%.2f px/frame", static_cast<double>(step));
    }
    std::printf("  %-16s  %10zu             %10zu\n", label, oldAllocs, newAllocs);
  }
  return true;
}

// Camera frames (#132): no decode, no texture per frame. RGBA stand-ins cover the draw path only.

constexpr int kCameraFrames = 60;
constexpr int kOverlays = 8;

std::shared_ptr<tgfx::Image> solidImage(int size, uint8_t value) {
  auto info = tgfx::ImageInfo::Make(size, size, tgfx::ColorType::RGBA_8888,
                                    tgfx::AlphaType::Premultiplied);
  auto pixels = tgfx::Data::MakeWithCopy(
      std::vector<uint8_t>(static_cast<size_t>(info.byteSize()), value).data(),
      static_cast<size_t>(info.byteSize()));
  return tgfx::Image::MakeFrom(info, std::move(pixels));
}

std::vector<double> cameraScene(int32_t handle) {
  Writer w;
  w.push(1);
  w.push(11);  // CameraFrame
  w.push(handle);
  w.push(2);  // cover
  w.push(1);  // linear
  w.push(0);  // no rotation
  w.push(0);  // not mirrored
  w.push(0xffffffff);
  w.push(2);
  w.pushAnim(1);
  w.push(0);
  w.push(0);
  w.pushAnim(1, 0);  // animated alpha: the paint really is resolved per frame
  w.push(-1);
  w.push(kOverlays);
  for (int i = 0; i < kOverlays; i++) {
    w.push(1);  // Rect
    w.pushAnim(8 + (i % 4) * 40);
    w.pushAnim(8 + (i / 4) * 40);
    w.pushAnim(32);
    w.pushAnim(32);
    w.push(0xff3366cc);
    w.push(2);
    w.pushAnim(1);
    w.push(0);
    w.push(0);
    w.pushAnim(1, 0);
    w.push(-1);
  }
  return w.finish();
}

bool reportCamera() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  std::shared_ptr<tgfx::Image> buffers[2] = {solidImage(kWidth, 90),
                                             solidImage(kWidth, 200)};
  if (buffers[0] == nullptr || buffers[1] == nullptr) {
    std::printf("could not build the camera bench's frames\n");
    return false;
  }
  auto external = registry.addExternal(buffers[0]);
  if (external.handle == 0) {
    std::printf("could not register an external frame\n");
    return false;
  }
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(cameraScene(external.handle), {});

  size_t drawn = 0;
  size_t decodesBefore = registry.decodeCount();
  for (int i = 0; i < 4; i++) {
    registry.replaceExternal(external.handle, buffers[i % 2]);
    frameAllocations(renderer, false, &drawn);
  }
  auto* context = device()->lockContext();
  size_t memoryBefore = context->memoryUsage();
  device()->unlock();

  size_t worst = 0;
  size_t total = 0;
  for (int i = 0; i < kCameraFrames; i++) {
    registry.replaceExternal(external.handle, buffers[i % 2]);
    size_t allocations = frameAllocations(renderer, true, &drawn);
    worst = std::max(worst, allocations);
    total += allocations;
  }
  context = device()->lockContext();
  size_t memoryAfter = context->memoryUsage();
  device()->unlock();
  size_t decodes = registry.decodeCount() - decodesBefore;
  registry.remove(external.handle);

  if (drawn == 0) {
    std::printf("nothing was drawn (%zu opaque pixels)\n", drawn);
    return false;
  }
  std::printf("a <CameraFrame> + %d overlays, %d frames, buffer swapped each one:\n",
              kOverlays, kCameraFrames);
  std::printf("  opaque pixels drawn        %zu\n", drawn);
  std::printf("  decodes on the frame path  %zu\n", decodes);
  std::printf("  GPU bytes before / after   %zu / %zu  (delta %+lld)\n", memoryBefore,
              memoryAfter, static_cast<long long>(memoryAfter) - static_cast<long long>(memoryBefore));
  std::printf("  allocations per render()   mean %zu, worst %zu  (%d nodes)\n",
              total / kCameraFrames, worst, kOverlays + 1);
  std::printf("\n");
  if (decodes != 0) {
    std::printf("FAIL: the frame path decoded %zu times\n", decodes);
    return false;
  }
  if (memoryAfter > memoryBefore) {
    std::printf("FAIL: GPU memory grew by %zu bytes over %d frames\n",
                memoryAfter - memoryBefore, kCameraFrames);
    return false;
  }
  return true;
}

// Hardware buffers: a real NV12 IOSurface through setFrame. A flat cache means no per-frame pre-pass.
#if defined(__APPLE__)

constexpr int kFrameWidth = 720;
constexpr int kFrameHeight = 1280;
/// Matches the on-device run PR #148 reported, so the two are comparable.
constexpr int kHardwareFrames = 780;

CVPixelBufferRef makeYUVBuffer(uint8_t luma, uint8_t chroma) {
  auto empty = CFDictionaryCreate(kCFAllocatorDefault, nullptr, nullptr, 0,
                                  &kCFTypeDictionaryKeyCallBacks,
                                  &kCFTypeDictionaryValueCallBacks);
  const void* keys[] = {kCVPixelBufferIOSurfacePropertiesKey};
  const void* values[] = {empty};
  auto options =
      CFDictionaryCreate(kCFAllocatorDefault, keys, values, 1,
                         &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  CVPixelBufferRef buffer = nullptr;
  auto status = CVPixelBufferCreate(kCFAllocatorDefault, kFrameWidth, kFrameHeight,
                                    kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                                    options, &buffer);
  CFRelease(options);
  CFRelease(empty);
  if (status != kCVReturnSuccess || buffer == nullptr) {
    return nullptr;
  }
  CVPixelBufferLockBaseAddress(buffer, 0);
  auto* y = static_cast<uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(buffer, 0));
  auto yStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0);
  for (int row = 0; row < kFrameHeight; row++) {
    std::memset(y + static_cast<size_t>(row) * yStride, luma, kFrameWidth);
  }
  auto* uv = static_cast<uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(buffer, 1));
  auto uvStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1);
  for (int row = 0; row < kFrameHeight / 2; row++) {
    std::memset(uv + static_cast<size_t>(row) * uvStride, chroma, kFrameWidth);
  }
  CVPixelBufferUnlockBaseAddress(buffer, 0);
  return buffer;
}

uint8_t drawAndSampleCentre(tgfxrn::SceneRenderer& renderer) {
  auto* context = device()->lockContext();
  auto surface = tgfx::Surface::Make(context, kWidth, kHeight);
  auto* canvas = surface->getCanvas();
  canvas->clear();
  const double values[1] = {1.0};
  renderer.render(canvas, values);
  context->flushAndSubmit(true);
  auto info = tgfx::ImageInfo::Make(kWidth, kHeight, tgfx::ColorType::RGBA_8888);
  std::vector<uint8_t> pixels(static_cast<size_t>(info.byteSize()));
  surface->readPixels(info, pixels.data());
  device()->unlock();
  return pixels[(static_cast<size_t>(kHeight / 2) * kWidth + kWidth / 2) * 4];
}

bool reportHardwareFrame() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  CVPixelBufferRef buffers[2] = {makeYUVBuffer(40, 128), makeYUVBuffer(200, 128)};
  if (buffers[0] == nullptr || buffers[1] == nullptr) {
    std::printf("could not allocate an NV12 CVPixelBuffer\n");
    return false;
  }
  auto pointer = [&](int i) { return reinterpret_cast<uint64_t>(buffers[i]); };

  auto first = tgfxrn::importHardwareBuffer(pointer(0), tgfxrn::FrameFormat::BT601Limited);
  if (first == nullptr) {
    std::printf("FAIL: tgfx would not import a planar YUV hardware buffer\n");
    return false;
  }
  auto external = registry.addExternal(first);
  first.reset();

  tgfxrn::SceneRenderer renderer;
  renderer.setScene(cameraScene(external.handle), {});

  size_t drawn = 0;
  size_t decodesBefore = registry.decodeCount();
  for (int i = 0; i < 4; i++) {
    tgfxrn::setFrame(external.handle, pointer(i % 2), tgfxrn::FrameFormat::BT601Limited);
    frameAllocations(renderer, false, &drawn);
  }

  // The counters cannot see this: the pixels must follow the buffer that was swapped in.
  tgfxrn::setFrame(external.handle, pointer(0), tgfxrn::FrameFormat::BT601Limited);
  auto dark = drawAndSampleCentre(renderer);
  tgfxrn::setFrame(external.handle, pointer(1), tgfxrn::FrameFormat::BT601Limited);
  auto light = drawAndSampleCentre(renderer);

  // Limited expands (Y-16)*255/219; a *fresh* buffer, since tgfx binds the space at first import.
  auto* fresh = makeYUVBuffer(200, 128);
  tgfxrn::setFrame(external.handle, reinterpret_cast<uint64_t>(fresh),
                   tgfxrn::FrameFormat::BT601Full);
  auto lightFullRange = drawAndSampleCentre(renderer);
  CVPixelBufferRelease(fresh);

  auto* context = device()->lockContext();
  size_t memoryBefore = context->memoryUsage();
  device()->unlock();

  size_t worst = 0;
  size_t total = 0;
  size_t rejected = 0;
  for (int i = 0; i < kHardwareFrames; i++) {
    if (!tgfxrn::setFrame(external.handle, pointer(i % 2),
                          tgfxrn::FrameFormat::BT601Limited)) {
      rejected++;
    }
    size_t allocations = frameAllocations(renderer, true, &drawn);
    worst = std::max(worst, allocations);
    total += allocations;
  }
  context = device()->lockContext();
  size_t memoryAfter = context->memoryUsage();
  device()->unlock();
  size_t decodes = registry.decodeCount() - decodesBefore;
  registry.remove(external.handle);
  CVPixelBufferRelease(buffers[0]);
  CVPixelBufferRelease(buffers[1]);

  std::printf("a %dx%d NV12 hardware buffer through tgfxrn::setFrame, %d frames:\n",
              kFrameWidth, kFrameHeight, kHardwareFrames);
  std::printf("  opaque pixels drawn        %zu\n", drawn);
  std::printf("  centre luma, dark / light  %u / %u\n", dark, light);
  std::printf("  same luma as full range    %u  (limited read %u)\n", lightFullRange,
              light);
  std::printf("  frames the registry took   %d\n", kHardwareFrames - int(rejected));
  std::printf("  decodes on the frame path  %zu\n", decodes);
  std::printf("  GPU bytes before / after   %zu / %zu  (delta %+lld)\n", memoryBefore,
              memoryAfter,
              static_cast<long long>(memoryAfter) - static_cast<long long>(memoryBefore));
  std::printf("  allocations per render()   mean %zu, worst %zu\n",
              total / kHardwareFrames, worst);
  std::printf("\n");

  if (drawn == 0) {
    std::printf("FAIL: the frame never painted\n");
    return false;
  }
  if (rejected != 0) {
    std::printf("FAIL: the registry rejected %zu of %d frames\n", rejected,
                kHardwareFrames);
    return false;
  }
  if (dark >= light) {
    std::printf("FAIL: the drawn pixels did not follow the swapped buffer\n");
    return false;
  }
  if (lightFullRange == light) {
    std::printf(
        "FAIL: full-range and limited-range imports of the same luma read "
        "identically — the format argument is not reaching the import\n");
    return false;
  }
  if (decodes != 0) {
    std::printf("FAIL: the hardware-buffer path decoded %zu times\n", decodes);
    return false;
  }
  if (memoryAfter > memoryBefore) {
    std::printf(
        "FAIL: GPU memory grew by %zu bytes over %d frames — a per-frame "
        "texture, i.e. a YUV->RGB pre-pass or a re-upload\n",
        memoryAfter - memoryBefore, kHardwareFrames);
    return false;
  }
  return true;
}

#endif  // __APPLE__

}  // namespace

int main() {
  if (device() == nullptr) {
    std::printf("no GPU device\n");
    return 1;
  }
  size_t plainPixels = 0, filteredPixels = 0;
  size_t plain = measure(false, &plainPixels);
  size_t filtered = measure(true, &filteredPixels);
  if (plainPixels == 0 || filteredPixels == 0) {
    std::printf("nothing was drawn (%zu / %zu opaque pixels)\n", plainPixels,
                filteredPixels);
    return 1;
  }
  std::printf("opaque pixels drawn: unfiltered %zu, filtered %zu\n", plainPixels,
              filteredPixels);
  std::printf("%d animated rects, allocations during render():\n", kNodes);
  std::printf("  unfiltered            %zu  (%.2f per node)\n", plain,
              double(plain) / kNodes);
  std::printf("  colour matrix + blend %zu  (%.2f per node)\n", filtered,
              double(filtered) / kNodes);
  std::printf("  filters built at compile: %zu\n", tgfxrn::filterBuildCount());
  std::printf("\n");
  if (!reportChildShaders()) {
    return 1;
  }
  std::printf("\n");
  if (!reportCamera()) {
    return 1;
  }
#if defined(__APPLE__)
  if (!reportHardwareFrame()) {
    return 1;
  }
#endif
  return reportText() ? 0 : 1;
}
