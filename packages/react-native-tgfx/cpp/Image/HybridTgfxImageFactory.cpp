#include "HybridTgfxImageFactory.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include <algorithm>

#include "Base64.hpp"
#include "HybridTgfxImage.hpp"
#include "ImageRegistry.hpp"
#include "OffscreenRender.hpp"
#include "Snapshot.hpp"
#include "PdfExport.hpp"
#include "PlatformAsset.hpp"
#include "SvgExport.hpp"
#include "Log.hpp"
#include "SvgImage.hpp"
#include "TgfxFrameSink.hpp"
#include "TgfxRenderer.hpp"
#include "tgfx/core/Data.h"
#include "tgfx/core/Image.h"
#include "tgfx/core/ImageCodec.h"
#include "tgfx/core/ImageInfo.h"
#include "tgfx/core/ImageFilter.h"
#include "tgfx/core/Orientation.h"
#include "tgfx/core/Rect.h"
#include "tgfx/core/SamplingOptions.h"

namespace margelo::nitro::tgfx {

/// The writer hands over its whole backing store, which doubles on growth and
/// never shrinks; the program's real length is header word [1]. Copy only that.
static size_t usedWords(const double* data, size_t capacity) {
  if (capacity < 2) {
    return capacity;
  }
  const double used = data[1];
  if (std::isfinite(used) && used >= 2 && used <= static_cast<double>(capacity)) {
    return static_cast<size_t>(used);
  }
  return capacity;
}

/// Exhaustive with no `default:`, so -Wswitch names the omission when nitrogen adds a value.
tgfxrn::FrameFormat toFrameFormat(PixelFormat format) {
  switch (format) {
    case PixelFormat::RGBA: return tgfxrn::FrameFormat::RGBA;
    case PixelFormat::BT601_LIMITED: return tgfxrn::FrameFormat::BT601Limited;
    case PixelFormat::BT601_FULL: return tgfxrn::FrameFormat::BT601Full;
    case PixelFormat::BT709_LIMITED: return tgfxrn::FrameFormat::BT709Limited;
    case PixelFormat::BT709_FULL: return tgfxrn::FrameFormat::BT709Full;
    case PixelFormat::BT2020_LIMITED: return tgfxrn::FrameFormat::BT2020Limited;
    case PixelFormat::BT2020_FULL: return tgfxrn::FrameFormat::BT2020Full;
    case PixelFormat::JPEG_FULL: return tgfxrn::FrameFormat::JPEGFull;
  }
  return tgfxrn::FrameFormat::BT601Limited;
}

std::shared_ptr<::tgfx::Image> imageFromNativeBuffer(uint64_t buffer, PixelFormat format) {
  return tgfxrn::importHardwareBuffer(buffer, toFrameFormat(format));
}

namespace {

using ImagePromise = Promise<std::shared_ptr<HybridTgfxImageSpec>>;

std::shared_ptr<HybridTgfxImageSpec> registerCodec(
    std::shared_ptr<::tgfx::ImageCodec> codec, const std::string& what) {
  if (codec == nullptr) {
    throw std::runtime_error("tgfx: " + what + " is not a supported image format");
  }
  auto handle = tgfxrn::ImageRegistry::instance().add(std::move(codec));
  if (handle.handle == 0) {
    throw std::runtime_error("tgfx: failed to decode " + what);
  }
  return std::make_shared<HybridTgfxImage>(handle);
}

/// Every byte-shaped entry point lands here, so an `.svg` works wherever a PNG does.
std::shared_ptr<HybridTgfxImageSpec> registerData(std::shared_ptr<::tgfx::Data> data,
                                                  const std::string& what) {
  if (data == nullptr || data->empty()) {
    throw std::runtime_error("tgfx: " + what + " is not a supported image format");
  }
  if (!tgfxrn::looksLikeSvg(*data)) {
    return registerCodec(::tgfx::ImageCodec::MakeFrom(std::move(data)), what);
  }
  auto image = tgfxrn::svgImage(data);
  if (image == nullptr) {
    throw std::runtime_error("tgfx: " + what + " is not a renderable SVG document");
  }
  // External: there are no decoded pixels to budget, and the picture replays at whatever
  // size it is drawn, so the handle's own size is nominal rather than a resolution cap.
  auto handle = tgfxrn::ImageRegistry::instance().addExternal(std::move(image));
  if (handle.handle == 0) {
    throw std::runtime_error("tgfx: failed to render " + what);
  }
  return std::make_shared<HybridTgfxImage>(handle);
}

std::string stripFileScheme(const std::string& path) {
  constexpr std::string_view kScheme = "file://";
  if (path.rfind(kScheme, 0) == 0) {
    return path.substr(kScheme.size());
  }
  return path;
}

}  // namespace

std::shared_ptr<ImagePromise> HybridTgfxImageFactory::fromFile(const std::string& path) {
  return ImagePromise::async([file = stripFileScheme(path)]() {
    // A packaged resource has a name and no file, so it must be read before any path attempt.
    if (tgfxrn::looksLikeAssetName(file)) {
      if (auto data = tgfxrn::platformAssetData(file)) {
        return registerData(std::move(data), file);
      }
    }
    // Codec first: it keeps the bytes on disk and re-reads them after an eviction, which
    // reading the file up front would give up. Only a non-raster falls through to the bytes.
    if (auto codec = ::tgfx::ImageCodec::MakeFrom(file)) {
      return registerCodec(std::move(codec), file);
    }
    return registerData(::tgfx::Data::MakeFromFile(file), file);
  });
}

std::shared_ptr<ImagePromise> HybridTgfxImageFactory::fromBytes(
    const std::shared_ptr<ArrayBuffer>& bytes) {
  // A JS-owned ArrayBuffer is only valid for this call, so the copy cannot go to the pool.
  auto data = bytes == nullptr || bytes->size() == 0
                  ? nullptr
                  : ::tgfx::Data::MakeWithCopy(bytes->data(), bytes->size());
  return ImagePromise::async([data = std::move(data)]() mutable {
    if (data == nullptr) {
      throw std::runtime_error("tgfx: fromBytes was given an empty buffer");
    }
    return registerData(std::move(data), "the given bytes");
  });
}

std::shared_ptr<ImagePromise> HybridTgfxImageFactory::fromBase64(const std::string& input) {
  return ImagePromise::async([input]() {
    auto decoded = tgfxrn::base64Decode(input);
    if (decoded.empty()) {
      throw std::runtime_error("tgfx: fromBase64 was given no decodable bytes");
    }
    auto data = ::tgfx::Data::MakeWithCopy(decoded.data(), decoded.size());
    return registerData(std::move(data), "the given base64 payload");
  });
}

std::shared_ptr<ImagePromise> HybridTgfxImageFactory::fromSvg(const std::string& markup) {
  return ImagePromise::async([markup]() {
    // Straight to the SVG path: markup that does not open like XML is a caller
    // mistake, and falling through to the codecs would report it as an
    // unsupported image format rather than as bad markup.
    auto data = ::tgfx::Data::MakeWithCopy(markup.data(), markup.size());
    if (data == nullptr || data->empty() || !tgfxrn::looksLikeSvg(*data)) {
      throw std::runtime_error("tgfx: fromSvg was not given an SVG document");
    }
    auto image = tgfxrn::svgImage(data);
    if (image == nullptr) {
      throw std::runtime_error("tgfx: the given markup is not a renderable SVG document");
    }
    auto handle = tgfxrn::ImageRegistry::instance().addExternal(std::move(image));
    if (handle.handle == 0) {
      throw std::runtime_error("tgfx: failed to render the given markup");
    }
    return std::static_pointer_cast<HybridTgfxImageSpec>(
        std::make_shared<HybridTgfxImage>(handle));
  });
}

std::shared_ptr<Promise<std::string>> HybridTgfxImageFactory::toSvg(
    const std::shared_ptr<ArrayBuffer>& words, const std::vector<std::string>& strings,
    double width, double height, bool textAsPaths) {
  if (words != nullptr && words->size() % sizeof(double) != 0) {
    throw std::runtime_error("tgfx: renderToSvg was given a truncated draw program");
  }
  // Copied here, not on the worker: a JS-owned ArrayBuffer is only valid for this call.
  std::vector<double> program;
  if (words != nullptr && words->size() >= sizeof(double)) {
    const auto* data = reinterpret_cast<const double*>(words->data());
    program.assign(data, data + usedWords(data, words->size() / sizeof(double)));
  }
  if (program.empty()) {
    throw std::runtime_error("tgfx: renderToSvg was given an empty draw program");
  }
  if (!std::isfinite(width) || !std::isfinite(height) || width <= 0 || height <= 0 ||
      width > 65536 || height > 65536) {
    throw std::runtime_error("tgfx: renderToSvg needs a finite width and height above zero");
  }
  return Promise<std::string>::async(
      [program = std::move(program), strings, width, height, textAsPaths]() {
        return tgfxrn::renderSceneToSvg(program, strings, static_cast<float>(width),
                                        static_cast<float>(height), textAsPaths);
      });
}

std::shared_ptr<Promise<void>> HybridTgfxImageFactory::savePdf(
    const std::string& path, const std::vector<std::shared_ptr<ArrayBuffer>>& pages,
    const std::vector<std::string>& strings, const std::vector<double>& stringCounts,
    const std::vector<double>& sizes, const std::vector<std::string>& metadata) {
  if (pages.empty() || sizes.size() != pages.size() * 2 || stringCounts.size() != pages.size()) {
    throw std::runtime_error("tgfx: renderToPdf was given a malformed page list");
  }
  // Unpacked here, not on the worker: a JS-owned ArrayBuffer is only valid for this call.
  std::vector<tgfxrn::PdfPage> unpacked;
  unpacked.reserve(pages.size());
  size_t stringAt = 0;
  for (size_t i = 0; i < pages.size(); i++) {
    tgfxrn::PdfPage page;
    const auto& buffer = pages[i];
    if (buffer != nullptr && buffer->size() % sizeof(double) != 0) {
      throw std::runtime_error("tgfx: renderToPdf was given a truncated draw program");
    }
    if (buffer != nullptr && buffer->size() >= sizeof(double)) {
      const auto* data = reinterpret_cast<const double*>(buffer->data());
      page.words.assign(data, data + usedWords(data, buffer->size() / sizeof(double)));
    }
    if (page.words.empty()) {
      throw std::runtime_error("tgfx: renderToPdf was given a page with no draw program");
    }
    // Compared as a double: a huge count saturates the cast and wraps the sum below.
    if (!(stringCounts[i] >= 0) ||
        stringCounts[i] > static_cast<double>(strings.size() - stringAt)) {
      throw std::runtime_error("tgfx: renderToPdf was given a malformed string table");
    }
    size_t count = static_cast<size_t>(stringCounts[i]);
    page.strings.assign(strings.begin() + static_cast<long>(stringAt),
                        strings.begin() + static_cast<long>(stringAt + count));
    stringAt += count;
    page.width = static_cast<float>(sizes[i * 2]);
    page.height = static_cast<float>(sizes[i * 2 + 1]);
    unpacked.push_back(std::move(page));
  }
  tgfxrn::PdfMetadata info;
  if (metadata.size() >= 4) {
    info.title = metadata[0];
    info.author = metadata[1];
    info.subject = metadata[2];
    info.creator = metadata[3];
  }
  return Promise<void>::async([path, unpacked = std::move(unpacked), info]() {
    tgfxrn::writePdf(path, unpacked, info);
  });
}

std::shared_ptr<ImagePromise> HybridTgfxImageFactory::fromScene(
    const std::shared_ptr<ArrayBuffer>& words, const std::vector<std::string>& strings,
    double width, double height, double pixelRatio, bool opaque,
    const std::string& colorSpace) {
  // Copied here, not on the worker: a JS-owned ArrayBuffer is only valid for this call.
  std::vector<double> program;
  if (words != nullptr && words->size() >= sizeof(double)) {
    const auto* data = reinterpret_cast<const double*>(words->data());
    program.assign(data, data + usedWords(data, words->size() / sizeof(double)));
  }
  if (program.empty()) {
    throw std::runtime_error("tgfx: renderToImage was given an empty draw program");
  }
  if (!std::isfinite(width) || !std::isfinite(height) || !std::isfinite(pixelRatio) ||
      width <= 0 || height <= 0 || width > 65536 || height > 65536) {
    throw std::runtime_error("tgfx: renderToImage needs a finite width and height above zero");
  }
  bool displayP3 = colorSpace == "display-p3";
  return ImagePromise::async([program = std::move(program), strings, width, height, pixelRatio,
                              opaque, displayP3]() {
    size_t bytes = 0;
    auto image = tgfxrn::renderSceneOffscreen(program, strings, static_cast<int>(width),
                                              static_cast<int>(height),
                                              static_cast<float>(pixelRatio), opaque, displayP3,
                                              bytes);
    if (image == nullptr) {
      throw std::runtime_error("tgfx: renderToImage produced no image");
    }
    // External so it is never evicted, but its bytes are ours and the GC is told.
    auto handle = tgfxrn::ImageRegistry::instance().addExternal(std::move(image), bytes);
    if (handle.handle == 0) {
      throw std::runtime_error("tgfx: renderToImage could not register the result");
    }
    return std::static_pointer_cast<HybridTgfxImageSpec>(std::make_shared<HybridTgfxImage>(handle));
  });
}

static std::shared_ptr<::tgfx::Image> imageFromPixelBytes(
    const std::shared_ptr<ArrayBuffer>& pixels, double width, double height,
    std::optional<double> rowBytes, size_t& outBytes) {
  if (pixels == nullptr || pixels->size() == 0) {
    throw std::runtime_error("tgfx: fromPixels was given an empty buffer");
  }
  if (!std::isfinite(width) || !std::isfinite(height) || width < 1 || height < 1 ||
      width > 65536 || height > 65536) {
    throw std::runtime_error("tgfx: fromPixels needs a width and height between 1 and 65536");
  }
  const double stride = rowBytes.value_or(0);
  // Bounded (unchecked it wraps byteSize()) and in whole pixels: GPU uploads take rows of pixels.
  if (!std::isfinite(stride) || stride < 0 || stride > static_cast<double>(1u << 30) ||
      std::fmod(stride, 4) != 0) {
    throw std::runtime_error("tgfx: fromPixels needs a rowBytes that is a multiple of 4, below 1 GiB");
  }
  const auto w = static_cast<int>(width);
  const auto h = static_cast<int>(height);
  auto info = ::tgfx::ImageInfo::Make(w, h, ::tgfx::ColorType::RGBA_8888,
                                      ::tgfx::AlphaType::Premultiplied,
                                      static_cast<size_t>(stride));
  if (info.isEmpty()) {
    throw std::runtime_error("tgfx: fromPixels was given a size no image can have");
  }
  if (pixels->size() < info.byteSize()) {
    throw std::runtime_error("tgfx: fromPixels was given " + std::to_string(pixels->size()) +
                             " bytes for an image needing " + std::to_string(info.byteSize()));
  }

  std::shared_ptr<::tgfx::Data> data;
  if (pixels->isOwner()) {
    auto* retained = new std::shared_ptr<ArrayBuffer>(pixels);
    data = ::tgfx::Data::MakeAdopted(
        pixels->data(), info.byteSize(),
        [](const void*, void* context) { delete static_cast<std::shared_ptr<ArrayBuffer>*>(context); },
        retained);
    if (data == nullptr) {
      delete retained;
    }
  } else {
    data = ::tgfx::Data::MakeWithCopy(pixels->data(), info.byteSize());
  }
  if (data == nullptr) {
    throw std::runtime_error("tgfx: fromPixels could not take the pixel data");
  }

  auto image = ::tgfx::Image::MakeFrom(info, std::move(data));
  if (image == nullptr) {
    throw std::runtime_error("tgfx: could not build an image from those pixels");
  }
  outBytes = info.byteSize();
  return image;
}

std::shared_ptr<HybridTgfxImageSpec> HybridTgfxImageFactory::fromPixels(
    const std::shared_ptr<ArrayBuffer>& pixels, double width, double height,
    std::optional<double> rowBytes) {
  size_t bytes = 0;
  auto image = imageFromPixelBytes(pixels, width, height, rowBytes, bytes);
  // The handle pins the pixels either way — its copy, or the borrowed buffer it retains.
  auto handle = tgfxrn::ImageRegistry::instance().addExternal(std::move(image), bytes, true);
  return std::make_shared<HybridTgfxImage>(handle);
}

void HybridTgfxImage::setPixels(const std::shared_ptr<ArrayBuffer>& pixels, double width,
                                double height, std::optional<double> rowBytes) {
  size_t bytes = 0;
  auto image = imageFromPixelBytes(pixels, width, height, rowBytes, bytes);
  if (!tgfxrn::ImageRegistry::instance().replaceExternal(handle_.handle, image, bytes)) {
    TGFXRN_LOGW("setPixels on image %d ignored: not an external handle", handle_.handle);
    return;
  }
  handle_.width = image->width();
  handle_.height = image->height();
}

std::shared_ptr<HybridTgfxImageSpec> HybridTgfxImageFactory::fromNativeBuffer(
    uint64_t buffer, PixelFormat format) {
  // Synchronous: nothing to decode, and a per-frame path cannot spend a promise round trip.
  auto image = imageFromNativeBuffer(buffer, format);
  if (image == nullptr) {
    throw std::runtime_error("tgfx: the hardware buffer could not be imported");
  }
  auto handle = tgfxrn::ImageRegistry::instance().addExternal(std::move(image), 0, true);
  return std::make_shared<HybridTgfxImage>(handle);
}

namespace {

::tgfx::SamplingOptions samplingOf(std::optional<ImageSampling> sampling) {
  switch (sampling.value_or(ImageSampling::LINEAR)) {
    case ImageSampling::NEAREST:
      return ::tgfx::SamplingOptions(::tgfx::FilterMode::Nearest, ::tgfx::MipmapMode::None);
    case ImageSampling::MIPMAP:
      return ::tgfx::SamplingOptions(::tgfx::FilterMode::Linear, ::tgfx::MipmapMode::Linear);
    case ImageSampling::LINEAR:
      break;
  }
  return ::tgfx::SamplingOptions(::tgfx::FilterMode::Linear, ::tgfx::MipmapMode::None);
}

::tgfx::Orientation orientationOf(Orientation orientation) {
  switch (orientation) {
    case Orientation::TOPLEFT: return ::tgfx::Orientation::TopLeft;
    case Orientation::TOPRIGHT: return ::tgfx::Orientation::TopRight;
    case Orientation::BOTTOMRIGHT: return ::tgfx::Orientation::BottomRight;
    case Orientation::BOTTOMLEFT: return ::tgfx::Orientation::BottomLeft;
    case Orientation::LEFTTOP: return ::tgfx::Orientation::LeftTop;
    case Orientation::RIGHTTOP: return ::tgfx::Orientation::RightTop;
    case Orientation::RIGHTBOTTOM: return ::tgfx::Orientation::RightBottom;
    case Orientation::LEFTBOTTOM: return ::tgfx::Orientation::LeftBottom;
  }
  return ::tgfx::Orientation::TopLeft;
}

/// Charged the pixels it keeps alive: a derivation holds a strong ref past the parent's eviction.
std::shared_ptr<HybridTgfxImageSpec> registerDerived(std::shared_ptr<::tgfx::Image> image,
                                                     const char* what, size_t ownedBytes = 0) {
  if (image == nullptr) {
    throw std::runtime_error(std::string("tgfx: ") + what + " produced no image");
  }
  auto handle = tgfxrn::ImageRegistry::instance().addExternal(std::move(image), ownedBytes);
  if (handle.handle == 0) {
    throw std::runtime_error(std::string("tgfx: ") + what + " could not be registered");
  }
  return std::make_shared<HybridTgfxImage>(handle);
}

/// What a derivation keeps alive, so an abandoned handle still tells the GC what it costs.
size_t sourceBytes(int32_t handle) {
  return tgfxrn::ImageRegistry::instance().byteSize(handle);
}

std::shared_ptr<::tgfx::Image> sourceOf(int32_t handle, const char* what) {
  auto image = tgfxrn::ImageRegistry::instance().sourceImage(handle);
  if (image == nullptr) {
    throw std::runtime_error(std::string("tgfx: ") + what +
                             " needs a decoded image, and this handle has none");
  }
  return image;
}

}  // namespace

std::shared_ptr<HybridTgfxImageSpec> HybridTgfxImage::subset(double x, double y, double width,
                                                             double height) {
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(width) ||
      !std::isfinite(height) || !(width > 0) || !(height > 0)) {
    throw std::runtime_error("tgfx: subset needs a finite rect with a positive size");
  }
  auto source = sourceOf(handle_.handle, "subset");
  auto rect = ::tgfx::Rect::MakeXYWH(static_cast<float>(x), static_cast<float>(y),
                                     static_cast<float>(width), static_cast<float>(height));
  return registerDerived(source->makeSubset(rect), "subset", sourceBytes(handle_.handle));
}

std::shared_ptr<HybridTgfxImageSpec> HybridTgfxImage::scaled(
    double width, double height, std::optional<ImageSampling> sampling) {
  if (!(width >= 1) || !(height >= 1) || width > 65536 || height > 65536) {
    throw std::runtime_error("tgfx: scaled needs a size of at least one pixel");
  }
  auto source = sourceOf(handle_.handle, "scaled");
  return registerDerived(source->makeScaled(static_cast<int>(width), static_cast<int>(height),
                                            samplingOf(sampling)),
                         "scaled", sourceBytes(handle_.handle));
}

std::shared_ptr<HybridTgfxImageSpec> HybridTgfxImage::oriented(Orientation orientation) {
  auto source = sourceOf(handle_.handle, "oriented");
  return registerDerived(source->makeOriented(orientationOf(orientation)), "oriented",
                         sourceBytes(handle_.handle));
}

std::shared_ptr<HybridTgfxImageSpec> HybridTgfxImage::blurred(double blurX, double blurY) {
  if (!std::isfinite(blurX) || !std::isfinite(blurY) || blurX < 0 || blurY < 0) {
    throw std::runtime_error("tgfx: blurred needs finite blur radii of zero or more");
  }
  auto source = sourceOf(handle_.handle, "blurred");
  // Bounds grow by 2 sigma a side, and an unclamped one overflows the int the size lands in.
  double grownWidth = static_cast<double>(source->width()) + 4 * blurX;
  double grownHeight = static_cast<double>(source->height()) + 4 * blurY;
  if (grownWidth > 65536 || grownHeight > 65536) {
    throw std::runtime_error("tgfx: blurred is capped at 65536 pixels on a side");
  }
  auto filter = ::tgfx::ImageFilter::Blur(static_cast<float>(blurX), static_cast<float>(blurY));
  auto blurred = source->makeWithFilter(std::move(filter));
  // Its own pixels on top of the source it holds: a filter renders a new surface.
  size_t bytes = blurred == nullptr
                     ? 0
                     : sourceBytes(handle_.handle) + static_cast<size_t>(blurred->width()) *
                                                         static_cast<size_t>(blurred->height()) * 4;
  return registerDerived(std::move(blurred), "blurred", bytes);
}

namespace {

/// Shared by encode and save: rasterize the handle, run the snapshot tail.
tgfxrn::SnapshotRequest encodeRequest(const std::optional<std::string>& format,
                                      std::optional<double> quality) {
  tgfxrn::SnapshotRequest request;
  request.format = format.value_or("png");
  double q = quality.value_or(100);
  // Clamped here: unlike snapshot(), there is no TS wrapper in front of this.
  request.quality = static_cast<int>(std::min(100.0, std::max(1.0, std::isfinite(q) ? q : 100)));
  return request;
}

std::shared_ptr<::tgfx::Data> encodeRegisteredImage(int32_t handle,
                                                    const tgfxrn::SnapshotRequest& request) {
  auto raster = tgfxrn::rasterizeRegisteredImage(handle);
  std::shared_ptr<::tgfx::Data> out;
  std::string error;
  tgfxrn::finishSnapshot(raster.pixels, raster.info, request,
                         [&](std::shared_ptr<::tgfx::Data> data, int, int, const std::string& err) {
                           out = std::move(data);
                           error = err;
                         });
  if (!error.empty()) {
    throw std::runtime_error(error);
  }
  return out;
}

}  // namespace

std::shared_ptr<Promise<std::shared_ptr<ArrayBuffer>>> HybridTgfxImage::encode(
    const std::optional<std::string>& format, std::optional<double> quality) {
  auto request = encodeRequest(format, quality);
  int32_t handle = handle_.handle;
  // Nitro's shared promise pool, like fromScene; it also runs every decode, so
  // a burst of encodes queues behind them rather than growing threads.
  return Promise<std::shared_ptr<ArrayBuffer>>::async([handle, request]() {
    auto out = encodeRegisteredImage(handle, request);
    if (out == nullptr) {
      throw std::runtime_error("tgfx: encoding produced no bytes");
    }
    // Zero-copy: the buffer borrows the encoded Data, and the deleter's
    // capture keeps it alive until the JS side lets go.
    auto* bytes = const_cast<uint8_t*>(static_cast<const uint8_t*>(out->data()));
    return ArrayBuffer::wrap(bytes, out->size(), [out]() {});
  });
}

std::shared_ptr<Promise<void>> HybridTgfxImage::save(const std::string& path,
                                                     const std::optional<std::string>& format,
                                                     std::optional<double> quality) {
  auto request = encodeRequest(format, quality);
  request.path = path;
  int32_t handle = handle_.handle;
  return Promise<void>::async([handle, request]() {
    // Inside the async body so an empty path rejects like every other failure.
    if (request.path.empty()) {
      throw std::runtime_error("tgfx: save needs a file path");
    }
    // finishSnapshot writes the file itself; the bytes never enter JS.
    encodeRegisteredImage(handle, request);
  });
}

void HybridTgfxImage::setNativeBuffer(uint64_t buffer, PixelFormat format) {
  auto image = imageFromNativeBuffer(buffer, format);
  if (image == nullptr ||
      !tgfxrn::ImageRegistry::instance().replaceExternal(handle_.handle, image)) {
    TGFXRN_LOGW("setNativeBuffer on image %d ignored: not a native-buffer handle",
                handle_.handle);
    return;
  }
  handle_.width = image->width();
  handle_.height = image->height();
}

void HybridTgfxImageFactory::setCacheLimit(double bytes) {
  tgfxrn::ImageRegistry::instance().setCacheLimit(
      bytes > 0 ? static_cast<size_t>(bytes) : 0);
}

double HybridTgfxImageFactory::getCachedBytes() {
  return static_cast<double>(tgfxrn::ImageRegistry::instance().cachedBytes());
}

CacheInfo HybridTgfxImageFactory::getCacheInfo() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  return CacheInfo(static_cast<double>(registry.cachedBytes()),
                   static_cast<double>(registry.cacheLimit()),
                   static_cast<double>(tgfxrn::TgfxRenderer::liveGpuBytes()),
                   static_cast<double>(tgfxrn::offscreenGpuBytes()));
}

void HybridTgfxImageFactory::purgeCaches() {
  // Generation first, so every canvas the wake below reaches sees the purge.
  tgfxrn::TgfxRenderer::requestPurge();
  // Frees the decoded CPU pixels on this thread. Its epoch bump wakes every
  // canvas — the wake listener also fires on a pending purge, image-free
  // canvases included — and each drops its GPU cache on that frame.
  tgfxrn::ImageRegistry::instance().purgeResident();
  tgfxrn::purgeOffscreenGpu();
}

}  // namespace margelo::nitro::tgfx
