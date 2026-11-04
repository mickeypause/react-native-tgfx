// Android video frames.
//
// OpenGL builds: a tgfx SurfaceTextureReader the decoder renders into — a video
// decoder hands back vendor-private buffer formats (0x123 on Exynos, and so on)
// that GL cannot import, so the frames come through a SurfaceTexture instead.
//
// Vulkan builds (TGFX_RN_VULKAN): SurfaceTexture is a GL-external-texture
// construct and cannot exist without a GL context, so the decoder renders into
// an AImageReader configured for YUV_420_888 + GPU sampling. That forces the
// decoder to produce standard Y8Cb8Cr8_420 hardware buffers, which tgfx imports
// as external-memory VkImages with sampler YCbCr conversion.
#include <jni.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>

#include "ImageRegistry.hpp"
#include "tgfx/core/Image.h"
#include "tgfx/core/ImageBuffer.h"
#include "tgfx/core/YUVData.h"

#ifdef TGFX_RN_VULKAN
#include <dlfcn.h>

#include <android/api-level.h>
#include <android/hardware_buffer.h>
#include <android/log.h>
#include <media/NdkImageReader.h>
#else
#include "tgfx/platform/android/SurfaceTextureReader.h"
#endif

namespace {

#ifdef TGFX_RN_VULKAN

/**
 * The AImageReader/ANativeWindow_toSurface entry points are API 26+, but this
 * library compiles against minSdk 24 — so resolve them at runtime, the same way
 * tgfx resolves AHardwareBuffer. Video simply stays unavailable on a pre-26
 * device (which cannot run the Vulkan build's swapchain path anyway).
 */
struct NdkMedia {
  media_status_t (*readerNewWithUsage)(int32_t, int32_t, int32_t, uint64_t, int32_t,
                                       AImageReader**) = nullptr;
  void (*readerDelete)(AImageReader*) = nullptr;
  media_status_t (*readerSetImageListener)(AImageReader*, AImageReader_ImageListener*) = nullptr;
  media_status_t (*readerGetWindow)(AImageReader*, ANativeWindow**) = nullptr;
  media_status_t (*readerAcquireLatestImage)(AImageReader*, AImage**) = nullptr;
  void (*imageDelete)(AImage*) = nullptr;
  media_status_t (*imageGetHardwareBuffer)(const AImage*, AHardwareBuffer**) = nullptr;
  jobject (*windowToSurface)(JNIEnv*, ANativeWindow*) = nullptr;
  void (*bufferDescribe)(const AHardwareBuffer*, AHardwareBuffer_Desc*) = nullptr;
  /// API 29+. The documented way to read a hardware buffer's planes on the CPU:
  /// it returns pointers and strides for the mapping it just made, rather than
  /// whatever the producer chose to advertise.
  int (*bufferLockPlanes)(AHardwareBuffer*, uint64_t, int32_t, const ARect*,
                          AHardwareBuffer_Planes*) = nullptr;
  int (*bufferUnlock)(AHardwareBuffer*, int32_t*) = nullptr;
  // API 34+; optional — absent just means the BT601_LIMITED default stands.
  media_status_t (*imageGetDataSpace)(const AImage*, int32_t*) = nullptr;
  bool valid = false;

  static const NdkMedia& Get() {
    static const NdkMedia instance;
    return instance;
  }

 private:
  NdkMedia() {
    auto* media = dlopen("libmediandk.so", RTLD_NOW | RTLD_LOCAL);
    auto* android = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
    if (media == nullptr || android == nullptr) {
      return;
    }
    auto load = [](void* lib, const char* name) { return dlsym(lib, name); };
    readerNewWithUsage = reinterpret_cast<decltype(readerNewWithUsage)>(
        load(media, "AImageReader_newWithUsage"));
    readerDelete = reinterpret_cast<decltype(readerDelete)>(load(media, "AImageReader_delete"));
    readerSetImageListener = reinterpret_cast<decltype(readerSetImageListener)>(
        load(media, "AImageReader_setImageListener"));
    readerGetWindow =
        reinterpret_cast<decltype(readerGetWindow)>(load(media, "AImageReader_getWindow"));
    readerAcquireLatestImage = reinterpret_cast<decltype(readerAcquireLatestImage)>(
        load(media, "AImageReader_acquireLatestImage"));
    imageDelete = reinterpret_cast<decltype(imageDelete)>(load(media, "AImage_delete"));
    imageGetHardwareBuffer = reinterpret_cast<decltype(imageGetHardwareBuffer)>(
        load(media, "AImage_getHardwareBuffer"));
    windowToSurface =
        reinterpret_cast<decltype(windowToSurface)>(load(android, "ANativeWindow_toSurface"));
    bufferDescribe =
        reinterpret_cast<decltype(bufferDescribe)>(load(android, "AHardwareBuffer_describe"));
    bufferLockPlanes =
        reinterpret_cast<decltype(bufferLockPlanes)>(load(android, "AHardwareBuffer_lockPlanes"));
    bufferUnlock =
        reinterpret_cast<decltype(bufferUnlock)>(load(android, "AHardwareBuffer_unlock"));
    imageGetDataSpace =
        reinterpret_cast<decltype(imageGetDataSpace)>(load(media, "AImage_getDataSpace"));
    valid = readerNewWithUsage && readerDelete && readerSetImageListener && readerGetWindow &&
            readerAcquireLatestImage && imageDelete && imageGetHardwareBuffer && windowToSurface;
  }
};

/**
 * The decoder stamps each frame with an ADataSpace; the YCbCr matrix and range
 * must follow it or full-range video renders washed out (and vice versa).
 * ADataSpace packs the standard in bits 16-21 and the range in bits 27-29.
 */
tgfx::YUVColorSpace colorSpaceFromDataSpace(int32_t dataSpace) {
  const int32_t standard = (dataSpace >> 16) & 0x3F;
  const bool full = ((dataSpace >> 27) & 0x7) == 1;  // ADATASPACE_RANGE_FULL
  switch (standard) {
    case 1:  // STANDARD_BT709
      return full ? tgfx::YUVColorSpace::BT709_FULL : tgfx::YUVColorSpace::BT709_LIMITED;
    case 6:  // STANDARD_BT2020
    case 7:  // STANDARD_BT2020_CONSTANT_LUMINANCE
      return full ? tgfx::YUVColorSpace::BT2020_FULL : tgfx::YUVColorSpace::BT2020_LIMITED;
    default:  // the BT601 family, and unspecified
      return full ? tgfx::YUVColorSpace::BT601_FULL : tgfx::YUVColorSpace::BT601_LIMITED;
  }
}

/** AImageReader-backed stand-in for tgfx::SurfaceTextureReader. */
class HardwareBufferReader {
 public:
  static std::shared_ptr<HardwareBufferReader> Make(int width, int height) {
    const auto& ndk = NdkMedia::Get();
    if (!ndk.valid) {
      return nullptr;
    }
    AImageReader* reader = nullptr;
    // AImageReader's maxImages caps the WHOLE BufferQueue (setMaxBufferCount),
    // and hardware decoders pipeline deeply — this Exynos H264 decoder asks for
    // 17 output buffers and starves after one frame on a small queue. 24 covers
    // codec depth + one acquired frame + slack; ~350 KB per 640x360 buffer.
    // Some decoders ignore the requested format and emit private buffers;
    // tgfx imports those as Vulkan external formats. CPU_READ permits planar
    // access where available and keeps decoder output flowing.
    auto status = ndk.readerNewWithUsage(
        width, height, AIMAGE_FORMAT_YUV_420_888,
        AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, 24,
        &reader);
    if (status != AMEDIA_OK || reader == nullptr) {
      return nullptr;
    }
    auto result = std::shared_ptr<HardwareBufferReader>(new HardwareBufferReader(reader));
    AImageReader_ImageListener listener = {result.get(), OnImageAvailable};
    ndk.readerSetImageListener(reader, &listener);
    return result;
  }

  ~HardwareBufferReader() {
    const auto& ndk = NdkMedia::Get();
    ndk.readerSetImageListener(reader, nullptr);
    {
      std::lock_guard<std::mutex> lock(mutex);
      lastImage = nullptr;
      if (heldImage != nullptr) {
        ndk.imageDelete(heldImage);
        heldImage = nullptr;
      }
    }
    ndk.readerDelete(reader);
  }

  jobject createInputSurface(JNIEnv* env) {
    const auto& ndk = NdkMedia::Get();
    ANativeWindow* window = nullptr;
    if (ndk.readerGetWindow(reader, &window) != AMEDIA_OK || window == nullptr) {
      return nullptr;
    }
    // The reader owns the window; ANativeWindow_toSurface adds a Java-side ref.
    return ndk.windowToSurface(env, window);
  }

  /**
   * I420 copy of a frame's planes; tgfx applies the (range-correct) matrix itself.
   *
   * On API 29+, the planes come from AHardwareBuffer_lockPlanes rather than
   * AImage_getPlaneData. API 26-28 retains the original AImage plane path.
   * Some decoders report plane geometry that does not match CPU-accessible
   * memory. lockPlanes provides mapped plane information; when it exposes
   * fewer than three planes, pumpFrames() imports the buffer directly.
   */
  std::shared_ptr<tgfx::Image> makePlanarImage(AImage* aimage) {
    const auto& ndk = NdkMedia::Get();
    static const int apiLevel = android_get_device_api_level();
    int32_t width = 0, height = 0;
    if (AImage_getWidth(aimage, &width) != AMEDIA_OK ||
        AImage_getHeight(aimage, &height) != AMEDIA_OK || width < 2 || height < 2) {
      return nullptr;
    }
    AHardwareBuffer_Planes planes = {};
    AHardwareBuffer* lockedBuffer = nullptr;
    if (apiLevel >= 26 && apiLevel < 29) {
      int32_t planeCount = 0;
      if (AImage_getNumberOfPlanes(aimage, &planeCount) != AMEDIA_OK || planeCount != 3) {
        return nullptr;
      }
      planes.planeCount = 3;
      for (int32_t p = 0; p < 3; p++) {
        uint8_t* data = nullptr;
        int dataLength = 0;
        int32_t rowStride = 0, pixelStride = 0;
        if (AImage_getPlaneData(aimage, p, &data, &dataLength) != AMEDIA_OK || data == nullptr ||
            AImage_getPlaneRowStride(aimage, p, &rowStride) != AMEDIA_OK ||
            AImage_getPlanePixelStride(aimage, p, &pixelStride) != AMEDIA_OK || rowStride < 1 ||
            pixelStride < 1) {
          return nullptr;
        }
        planes.planes[p] = {data, static_cast<uint32_t>(pixelStride),
                            static_cast<uint32_t>(rowStride)};
      }
    } else {
      if (ndk.bufferLockPlanes == nullptr || ndk.bufferUnlock == nullptr ||
          ndk.imageGetHardwareBuffer(aimage, &lockedBuffer) != AMEDIA_OK ||
          lockedBuffer == nullptr) {
        return nullptr;
      }
      if (ndk.bufferLockPlanes(lockedBuffer, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1, nullptr,
                               &planes) != 0) {
        return nullptr;
      }
      if (planes.planeCount != 3) {
        ndk.bufferUnlock(lockedBuffer, nullptr);
        return nullptr;
      }
    }

    const int32_t chromaW = width / 2, chromaH = height / 2;
    const size_t ySize = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t cSize = static_cast<size_t>(chromaW) * static_cast<size_t>(chromaH);
    auto* block = static_cast<uint8_t*>(malloc(ySize + 2 * cSize));
    if (block == nullptr) {
      if (lockedBuffer != nullptr) {
        ndk.bufferUnlock(lockedBuffer, nullptr);
      }
      return nullptr;
    }
    uint8_t* dst[3] = {block, block + ySize, block + ySize + cSize};
    const size_t dstRow[3] = {static_cast<size_t>(width), static_cast<size_t>(chromaW),
                              static_cast<size_t>(chromaW)};
    const int32_t planeW[3] = {width, chromaW, chromaW};
    const int32_t planeH[3] = {height, chromaH, chromaH};
    for (int32_t p = 0; p < 3; p++) {
      const auto& plane = planes.planes[p];
      const auto* src = static_cast<const uint8_t*>(plane.data);
      const uint32_t rowStride = plane.rowStride;
      const uint32_t pixelStride = plane.pixelStride;
      if (src == nullptr || pixelStride < 1) {
        free(block);
        if (lockedBuffer != nullptr) {
          ndk.bufferUnlock(lockedBuffer, nullptr);
        }
        return nullptr;
      }
      for (int32_t row = 0; row < planeH[p]; row++) {
        const uint8_t* in = src + static_cast<size_t>(row) * static_cast<size_t>(rowStride);
        uint8_t* out = dst[p] + static_cast<size_t>(row) * dstRow[p];
        if (pixelStride == 1) {
          memcpy(out, in, static_cast<size_t>(planeW[p]));
        } else {
          // Semi-planar chroma: every pixelStride-th byte belongs to this plane.
          for (int32_t col = 0; col < planeW[p]; col++) {
            out[col] = in[static_cast<size_t>(col) * static_cast<size_t>(pixelStride)];
          }
        }
      }
    }
    if (lockedBuffer != nullptr) {
      ndk.bufferUnlock(lockedBuffer, nullptr);
    }

    const void* data[3] = {dst[0], dst[1], dst[2]};
    auto yuvData = tgfx::YUVData::MakeFrom(
        width, height, data, dstRow, tgfx::YUVData::I420_PLANE_COUNT,
        [](void* context, const void**, size_t) { free(context); }, block);
    if (yuvData == nullptr) {
      free(block);
      return nullptr;
    }
    return tgfx::Image::MakeFrom(tgfx::ImageBuffer::MakeI420(std::move(yuvData), frameColorSpace));
  }

  /**
   * Drain the reader and import the newest frame. Runs from the UI-tick pump,
   * NOT at draw time: draining here keeps the decoder's BufferQueue moving even
   * while the canvas is offscreen and draw-gated — otherwise the queue fills,
   * the decoder stalls, callbacks stop, and nothing ever re-wakes the canvas.
   * Unlike SurfaceTexture generations, an AHardwareBuffer-backed Image stays
   * valid, so drawing the cached frame later is safe. Returns true on a new frame.
   */
  bool pumpFrames() {
    if (!frameArrived.exchange(false, std::memory_order_acq_rel)) {
      return false;
    }
    const auto& ndk = NdkMedia::Get();
    std::lock_guard<std::mutex> lock(mutex);
    AImage* next = nullptr;
    auto acquireStatus = ndk.readerAcquireLatestImage(reader, &next);
    if (acquireStatus == AMEDIA_OK && next != nullptr) {
      if (!colorSpaceKnown) {
        int32_t dataSpace = 0;
        if (ndk.imageGetDataSpace != nullptr &&
            ndk.imageGetDataSpace(next, &dataSpace) == AMEDIA_OK) {
          frameColorSpace = colorSpaceFromDataSpace(dataSpace);
        }
        colorSpaceKnown = true;
      }
      // Preferred: copy the planes and let tgfx run its own YUV->RGB shader.
      // Zero-copy external-format import is broken on at least Mali: the driver
      // bakes its suggested narrow-range conversion into the sampler and
      // ignores VkSamplerYcbcrConversionCreateInfo overrides, so full-range
      // video renders washed out. A 640x360 frame copies in well under a
      // millisecond.
      // ponytail: CPU copy per frame; revisit zero-copy per-driver if 4K video matters.
      std::shared_ptr<tgfx::Image> image = makePlanarImage(next);
      AHardwareBuffer* buffer = nullptr;
      if (image == nullptr && ndk.imageGetHardwareBuffer(next, &buffer) == AMEDIA_OK &&
          buffer != nullptr) {
        image = tgfx::Image::MakeFrom(buffer, frameColorSpace);
        if (image == nullptr && !importFailureLogged) {
          importFailureLogged = true;
          AHardwareBuffer_Desc desc = {};
          if (ndk.bufferDescribe != nullptr) {
            ndk.bufferDescribe(buffer, &desc);
          }
          __android_log_print(ANDROID_LOG_WARN, "TgfxVideo",
                              "vulkan video import failed: buffer format=0x%x %ux%u usage=0x%llx",
                              desc.format, desc.width, desc.height,
                              static_cast<unsigned long long>(desc.usage));
        }
      } else if (image == nullptr && !importFailureLogged) {
        importFailureLogged = true;
        __android_log_print(ANDROID_LOG_WARN, "TgfxVideo",
                            "vulkan video: AImage has no hardware buffer");
      }
      if (image != nullptr) {
        // Release the previous frame only after the new one is importable, so a
        // failed import keeps the last good frame on screen.
        if (heldImage != nullptr) {
          ndk.imageDelete(heldImage);
        }
        heldImage = next;
        lastImage = std::move(image);
        return true;
      }
      ndk.imageDelete(next);
    }
    return false;
  }

  /** The most recently imported frame; what the scene draws. */
  std::shared_ptr<tgfx::Image> currentImage() {
    std::lock_guard<std::mutex> lock(mutex);
    return lastImage;
  }

  /** Set by the decoder callback, cleared by the pump — mirrors the GL
      SurfaceTexture listener → videoStreamNotify flow. */
  std::atomic<bool> frameArrived{false};

 private:
  explicit HardwareBufferReader(AImageReader* reader) : reader(reader) {
  }

  static void OnImageAvailable(void* context, AImageReader*) {
    static_cast<HardwareBufferReader*>(context)->frameArrived.store(true,
                                                                    std::memory_order_release);
  }

  AImageReader* reader = nullptr;
  std::mutex mutex;
  AImage* heldImage = nullptr;
  std::shared_ptr<tgfx::Image> lastImage;
  bool importFailureLogged = false;
  // Queried from the first frame; constant for the stream's life.
  tgfx::YUVColorSpace frameColorSpace = tgfx::YUVColorSpace::BT601_LIMITED;
  bool colorSpaceKnown = false;
};

using FrameReader = HardwareBufferReader;

std::shared_ptr<FrameReader> makeReader(int width, int height, jobject /*listener*/) {
  return HardwareBufferReader::Make(width, height);
}

#else

using FrameReader = tgfx::SurfaceTextureReader;

std::shared_ptr<FrameReader> makeReader(int width, int height, jobject listener) {
  return tgfx::SurfaceTextureReader::Make(width, height, listener);
}

#endif

struct VideoStream {
  std::mutex mutex;
  // Null between a background and the rebuild that follows the next resume.
  std::shared_ptr<FrameReader> reader;
  int width = 0;
  int height = 0;
  // Stable for the video's life: the scene refers to it, so it must not move.
  int32_t handle = 0;
  // Set by the decoder, cleared by the pump, and the only thing that schedules a
  // redraw. Without it a 30 fps video would repaint every canvas at display rate.
  std::atomic<bool> available{false};

  std::shared_ptr<FrameReader> current() {
    std::lock_guard<std::mutex> lock(mutex);
    return reader;
  }
};

/// A shared_ptr so a draw already inside the producer keeps the stream alive.
std::shared_ptr<VideoStream>* asHolder(jlong ptr) {
  return reinterpret_cast<std::shared_ptr<VideoStream>*>(ptr);
}

std::shared_ptr<VideoStream> asStream(jlong ptr) {
  auto* holder = asHolder(ptr);
  return holder != nullptr ? *holder : nullptr;
}

std::shared_ptr<tgfx::Image> produceImage(const std::shared_ptr<FrameReader>& reader) {
#ifdef TGFX_RN_VULKAN
  return reader->currentImage();
#else
  auto buffer = reader->acquireNextBuffer();
  return buffer != nullptr ? tgfx::Image::MakeFrom(std::move(buffer)) : nullptr;
#endif
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_com_margelo_nitro_tgfx_TgfxNative_videoStreamCreate(
    JNIEnv*, jobject, jint width, jint height, jobject listener) {
  auto reader = makeReader(width, height, listener);
  if (reader == nullptr) {
    return 0;
  }
  auto stream = std::make_shared<VideoStream>();
  stream->reader = std::move(reader);
  stream->width = width;
  stream->height = height;
  // Produced at draw time, not pushed: a SurfaceTexture buffer is good for one
  // texture generation, so an image made ahead of the draw expires unused — and a
  // paused video, whose last frame is then redrawn, would go black.
  stream->handle = tgfxrn::ImageRegistry::instance()
                       .addStream(
                           [stream]() -> std::shared_ptr<tgfx::Image> {
                             auto reader = stream->current();
                             if (reader == nullptr) {
                               return nullptr;
                             }
                             return produceImage(reader);
                           },
                           width, height)
                       .handle;
  return reinterpret_cast<jlong>(new std::shared_ptr<VideoStream>(std::move(stream)));
}

JNIEXPORT jobject JNICALL Java_com_margelo_nitro_tgfx_TgfxNative_videoStreamSurface(
    JNIEnv* env, jobject, jlong ptr) {
  auto stream = asStream(ptr);
  if (stream == nullptr) {
    return nullptr;
  }
  auto reader = stream->current();
  if (reader == nullptr) {
    return nullptr;
  }
#ifdef TGFX_RN_VULKAN
  return reader->createInputSurface(env);
#else
  (void)env;
  return reader->createInputSurface();
#endif
}

JNIEXPORT void JNICALL Java_com_margelo_nitro_tgfx_TgfxNative_videoStreamNotify(JNIEnv*, jobject,
                                                                                jlong ptr) {
  auto stream = asStream(ptr);
  if (stream == nullptr) {
    return;
  }
  auto reader = stream->current();
  if (reader != nullptr) {
#ifndef TGFX_RN_VULKAN
    reader->notifyFrameAvailable();
#endif
    stream->available.store(true, std::memory_order_release);
  }
}

JNIEXPORT jint JNICALL Java_com_margelo_nitro_tgfx_TgfxNative_videoStreamUpdate(JNIEnv*, jobject,
                                                                                jlong ptr) {
  auto stream = asStream(ptr);
  if (stream == nullptr) {
    return 0;
  }
#ifdef TGFX_RN_VULKAN
  // No Java SurfaceTexture listener exists on this path; the AImageReader
  // callback raises a flag the pump drains here. Draining off the draw path is
  // what keeps the decoder's queue moving while the canvas is draw-gated.
  auto reader = stream->current();
  if (reader != nullptr && reader->pumpFrames()) {
    stream->available.store(true, std::memory_order_release);
  }
#endif
  // Only a frame the decoder actually produced costs a redraw.
  if (stream->available.exchange(false, std::memory_order_acq_rel)) {
    tgfxrn::ImageRegistry::instance().invalidate();
  }
  return stream->handle;
}

/**
 * Drop the reader, for a canvas whose GPU context is about to go away.
 *
 * Its buffers are bound to the context that first drew them, so drawing one
 * against the context a resumed app builds instead dereferences a null context
 * deep inside tgfx. Nothing is drawn until videoStreamRebuild() lands.
 */
JNIEXPORT void JNICALL Java_com_margelo_nitro_tgfx_TgfxNative_videoStreamRelease(JNIEnv*, jobject,
                                                                                 jlong ptr) {
  auto stream = asStream(ptr);
  if (stream == nullptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(stream->mutex);
    stream->reader = nullptr;
  }
  stream->available.store(false, std::memory_order_release);
  tgfxrn::ImageRegistry::instance().invalidate();
}

/// A fresh reader on the same handle. Returns its Surface; the decoder must be re-pointed at it.
JNIEXPORT jobject JNICALL Java_com_margelo_nitro_tgfx_TgfxNative_videoStreamRebuild(
    JNIEnv* env, jobject, jlong ptr, jobject listener) {
  auto stream = asStream(ptr);
  if (stream == nullptr) {
    return nullptr;
  }
  auto reader = makeReader(stream->width, stream->height, listener);
  if (reader == nullptr) {
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(stream->mutex);
    stream->reader = reader;
  }
#ifdef TGFX_RN_VULKAN
  return reader->createInputSurface(env);
#else
  (void)env;
  return reader->createInputSurface();
#endif
}

JNIEXPORT void JNICALL Java_com_margelo_nitro_tgfx_TgfxNative_videoStreamDestroy(JNIEnv*, jobject,
                                                                                 jlong ptr) {
  auto* holder = asHolder(ptr);
  if (holder == nullptr) {
    return;
  }
  auto stream = *holder;
  if (stream != nullptr) {
    {
      std::lock_guard<std::mutex> lock(stream->mutex);
      stream->reader = nullptr;
    }
    // Drops the registry's reference to the producer, and with it the stream.
    tgfxrn::ImageRegistry::instance().remove(stream->handle);
  }
  delete holder;
}

}  // extern "C"
