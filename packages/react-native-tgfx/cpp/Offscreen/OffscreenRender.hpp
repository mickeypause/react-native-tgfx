// Draws a compiled program to a GPU surface that no view owns, for renderToImage.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "tgfx/core/ImageInfo.h"

namespace tgfx {
class Context;
class Data;
class Device;
class Image;
}  // namespace tgfx

namespace tgfxrn {

/**
 * Drains what Metal autoreleases inside its scope: offscreen GPU work runs on
 * long-lived C++ threads (the surface worker, Nitro's promise pool) that never
 * drain a pool of their own. A no-op off Apple.
 */
class AutoreleasePool {
 public:
#if defined(__APPLE__)
  AutoreleasePool();
  ~AutoreleasePool();
#else
  AutoreleasePool() = default;
#endif
  AutoreleasePool(const AutoreleasePool&) = delete;
  AutoreleasePool& operator=(const AutoreleasePool&) = delete;

 private:
#if defined(__APPLE__)
  void* pool_;
#endif
};

/**
 * Holds the shared windowless GPU context for as long as it lives, replacing a
 * device that lost its context for good. Throws if there is no usable GPU.
 */
class OffscreenContext {
 public:
  OffscreenContext();
  ~OffscreenContext();
  OffscreenContext(const OffscreenContext&) = delete;
  OffscreenContext& operator=(const OffscreenContext&) = delete;

  tgfx::Context* get() const {
    return context_;
  }

 private:
  AutoreleasePool pool_;
  std::shared_ptr<tgfx::Device> device_;
  tgfx::Context* context_ = nullptr;
};

/**
 * Renders `words` at `width` x `height` logical points scaled by `pixelRatio`,
 * reporting the pixels allocated in `outBytes`.
 *
 * `displayP3` picks the surface's colour space, the way `<Canvas colorSpace>`
 * does for a view. The space is carried on both ends: an untagged surface makes
 * tgfx skip conversion on draw, so a wide-gamut image would land with its own
 * primaries reinterpreted, and untagged result pixels would then be taken for
 * the destination's space by whatever draws them next.
 *
 * The result is read back to CPU pixels rather than kept as a texture: the
 * offscreen context is not the one a view draws in, and a texture cannot cross
 * between them. Throws std::runtime_error if there is no usable GPU.
 */
std::shared_ptr<tgfx::Image> renderSceneOffscreen(std::span<const double> words,
                                                  const std::vector<std::string>& strings,
                                                  int width, int height, float pixelRatio,
                                                  bool opaque, bool displayP3, size_t& outBytes);

struct RasterizedImage {
  std::shared_ptr<tgfx::Data> pixels;  // premultiplied RGBA_8888
  tgfx::ImageInfo info;
};

/**
 * The registry image drawn at its native size and read back to CPU pixels —
 * the input `finishSnapshot` encodes. One path for every kind of entry: even a
 * `fromScene` result sits behind the registry as a `tgfx::Image`, and a
 * derived image (subset/scaled/blurred) rasterizes what it draws. Throws for a
 * disposed handle and for live camera/video frames, which have no still.
 */
RasterizedImage rasterizeRegisteredImage(int32_t handle);

/// GPU bytes the shared offscreen context holds; 0 when it was never created.
size_t offscreenGpuBytes();

/// Drop the shared offscreen context's unlocked GPU resources now. Never creates it.
void purgeOffscreenGpu();

}  // namespace tgfxrn
