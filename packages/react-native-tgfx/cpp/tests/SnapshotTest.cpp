// Host test for the readback tail in cpp/Snapshot.hpp, gated on real drawn pixels throughout.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "Snapshot.hpp"
#include "tgfx/core/Canvas.h"
#include "tgfx/core/Color.h"
#include "tgfx/core/Image.h"
#include "tgfx/core/ImageCodec.h"
#include "tgfx/core/Paint.h"
#include "tgfx/core/Rect.h"
#include "tgfx/core/ColorSpace.h"
#include "tgfx/core/Surface.h"
#include "tgfx/core/SurfaceReadback.h"
#include "tgfx/gpu/Device.h"
#include "tgfx/gpu/metal/MetalDevice.h"

namespace {

int failures = 0;

#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      failures++;                                                   \
    }                                                               \
  } while (0)

constexpr int kWidth = 64;
constexpr int kHeight = 48;
// Half-transparent, so a readback that skipped the straight-alpha conversion comes back darker.
constexpr uint8_t kR = 200;
constexpr uint8_t kG = 60;
constexpr uint8_t kB = 30;
constexpr uint8_t kA = 128;

tgfx::Device* device() {
  static auto shared = tgfx::MetalDevice::Make();
  return shared.get();
}

std::shared_ptr<tgfx::Surface> drawFrame(tgfx::Context* context,
                                         std::shared_ptr<tgfx::ColorSpace> space = nullptr) {
  auto surface = space == nullptr
                     ? tgfx::Surface::Make(context, kWidth, kHeight)
                     : tgfx::Surface::Make(context, kWidth, kHeight, false, 1, false, 0,
                                           std::move(space));
  auto* canvas = surface->getCanvas();
  canvas->clear();
  tgfx::Paint paint;
  paint.setColor(tgfx::Color::FromRGBA(kR, kG, kB, kA));
  canvas->drawRect(tgfx::Rect::MakeWH(kWidth / 2, kHeight), paint);
  return surface;
}

std::shared_ptr<tgfx::Data> readBack(tgfx::Context* context,
                                     const std::shared_ptr<tgfx::Surface>& surface,
                                     tgfx::ImageInfo* info) {
  auto readback = surface->asyncReadPixels(
      tgfx::Rect::MakeWH(surface->width(), surface->height()));
  if (readback == nullptr) {
    return nullptr;
  }
  context->flushAndSubmit(true);
  CHECK(readback->isReady(context));
  *info = readback->info();
  const void* pixels =
      readback->lockPixels(context, surface->origin() == tgfx::ImageOrigin::BottomLeft);
  auto copy = pixels != nullptr ? tgfx::Data::MakeWithCopy(pixels, info->byteSize())
                                : nullptr;
  readback->unlockPixels(context);
  return copy;
}

std::vector<uint8_t> decode(const std::shared_ptr<tgfx::Data>& bytes, int* width,
                            int* height) {
  auto codec = tgfx::ImageCodec::MakeFrom(bytes);
  if (codec == nullptr) {
    return {};
  }
  *width = codec->width();
  *height = codec->height();
  auto info = tgfx::ImageInfo::Make(codec->width(), codec->height(),
                                    tgfx::ColorType::RGBA_8888,
                                    tgfx::AlphaType::Unpremultiplied);
  std::vector<uint8_t> pixels(info.byteSize());
  if (!codec->readPixels(info, pixels.data())) {
    return {};
  }
  return pixels;
}

bool looksLikeTheFrame(const std::vector<uint8_t>& pixels, int width, int height,
                       int tolerance) {
  if (pixels.size() != static_cast<size_t>(width) * height * 4) {
    return false;
  }
  auto at = [&](int x, int y, int channel) {
    return static_cast<int>(pixels[(static_cast<size_t>(y) * width + x) * 4 + channel]);
  };
  auto near = [&](int value, int expected) {
    return value >= expected - tolerance && value <= expected + tolerance;
  };
  int insideX = width / 4;
  int outsideX = width - width / 4;
  int y = height / 2;
  return near(at(insideX, y, 0), kR) && near(at(insideX, y, 1), kG) &&
         near(at(insideX, y, 2), kB) && near(at(insideX, y, 3), kA) &&
         at(outsideX, y, 3) == 0;
}

void snapshotRoundTripsThroughPng() {
  auto* context = device()->lockContext();
  auto surface = drawFrame(context);
  tgfx::ImageInfo info;
  auto pixels = readBack(context, surface, &info);
  device()->unlock();
  CHECK(pixels != nullptr);
  if (pixels == nullptr) {
    return;
  }

  tgfxrn::SnapshotRequest rawRequest;
  rawRequest.format = "raw";
  std::shared_ptr<tgfx::Data> raw;
  int rawWidth = 0;
  int rawHeight = 0;
  tgfxrn::finishSnapshot(pixels, info, rawRequest,
                         [&](std::shared_ptr<tgfx::Data> data, int w, int h,
                             const std::string& error) {
                           CHECK(error.empty());
                           raw = std::move(data);
                           rawWidth = w;
                           rawHeight = h;
                         });
  CHECK(rawWidth == kWidth);
  CHECK(rawHeight == kHeight);
  CHECK(raw != nullptr);
  if (raw == nullptr) {
    return;
  }
  std::vector<uint8_t> rawPixels(static_cast<const uint8_t*>(raw->data()),
                                 static_cast<const uint8_t*>(raw->data()) + raw->size());
  CHECK(looksLikeTheFrame(rawPixels, kWidth, kHeight, 2));

  tgfxrn::SnapshotRequest pngRequest;
  std::shared_ptr<tgfx::Data> png;
  tgfxrn::finishSnapshot(pixels, info, pngRequest,
                         [&](std::shared_ptr<tgfx::Data> data, int, int,
                             const std::string& error) {
                           CHECK(error.empty());
                           png = std::move(data);
                         });
  CHECK(png != nullptr);
  if (png == nullptr) {
    return;
  }
  int width = 0;
  int height = 0;
  auto decoded = decode(png, &width, &height);
  CHECK(width == kWidth);
  CHECK(height == kHeight);
  CHECK(looksLikeTheFrame(decoded, width, height, 2));
}

void saveWritesADecodableFile() {
  auto* context = device()->lockContext();
  auto surface = drawFrame(context);
  tgfx::ImageInfo info;
  auto pixels = readBack(context, surface, &info);
  device()->unlock();
  CHECK(pixels != nullptr);
  if (pixels == nullptr) {
    return;
  }
  tgfxrn::SnapshotRequest request;
  request.path = std::string(std::tmpnam(nullptr)) + ".png";
  std::shared_ptr<tgfx::Data> handedBack = nullptr;
  tgfxrn::finishSnapshot(pixels, info, request,
                         [&](std::shared_ptr<tgfx::Data> data, int, int,
                             const std::string& error) {
                           CHECK(error.empty());
                           handedBack = std::move(data);
                         });
  CHECK(handedBack == nullptr);

  auto onDisk = tgfx::Data::MakeFromFile(request.path);
  CHECK(onDisk != nullptr);
  if (onDisk == nullptr) {
    return;
  }
  int width = 0;
  int height = 0;
  auto decoded = decode(onDisk, &width, &height);
  CHECK(width == kWidth);
  CHECK(height == kHeight);
  CHECK(looksLikeTheFrame(decoded, width, height, 2));
  std::remove(request.path.c_str());
}

void badRequestsReportInsteadOfProducingABlankImage() {
  tgfx::ImageInfo info;
  std::string error;
  tgfxrn::finishSnapshot(nullptr, info, {},
                         [&](std::shared_ptr<tgfx::Data>, int, int,
                             const std::string& message) { error = message; });
  CHECK(!error.empty());

  auto* context = device()->lockContext();
  auto surface = drawFrame(context);
  auto pixels = readBack(context, surface, &info);
  device()->unlock();
  tgfxrn::SnapshotRequest request;
  request.path = "/definitely/not/a/directory/out.png";
  error.clear();
  tgfxrn::finishSnapshot(pixels, info, request,
                         [&](std::shared_ptr<tgfx::Data>, int, int,
                             const std::string& message) { error = message; });
  CHECK(error.find("save") != std::string::npos);
}

void fractionalRectsRoundInsteadOfTruncating() {
  auto bounds = tgfx::Rect::MakeWH(kWidth, kHeight);
  CHECK(tgfxrn::snapshotSourceRect({}, 2.0f, bounds) == bounds);

  tgfxrn::SnapshotRequest request;
  request.x = 4.2f;
  request.y = 3.1f;
  request.width = 12.4f;
  request.height = 9.3f;
  // Device rect (8.4, 6.2)-(33.2, 24.8); unrounded this reads 24x18 and resolves with those.
  auto source = tgfxrn::snapshotSourceRect(request, 2.0f, bounds);
  CHECK(source == tgfx::Rect::MakeLTRB(8, 6, 33, 25));
  CHECK(source.width() == std::round(source.width()));
  CHECK(source.height() == std::round(source.height()));

  tgfxrn::SnapshotRequest clipped;
  clipped.x = 24;
  clipped.y = 0;
  clipped.width = 40;
  clipped.height = 10;
  CHECK(tgfxrn::snapshotSourceRect(clipped, 2.0f, bounds) ==
        tgfx::Rect::MakeLTRB(48, 0, 64, 20));

  tgfxrn::SnapshotRequest outside;
  outside.x = 200;
  outside.y = 0;
  outside.width = 10;
  outside.height = 10;
  CHECK(tgfxrn::snapshotSourceRect(outside, 2.0f, bounds).isEmpty());
  tgfxrn::SnapshotRequest sliver;
  sliver.x = 1;
  sliver.y = 1;
  sliver.width = 0.2f;
  sliver.height = 0.2f;
  CHECK(tgfxrn::snapshotSourceRect(sliver, 1.0f, bounds).isEmpty());
}

void cropAndScaleLandWhereTheyWereAsked() {
  auto* context = device()->lockContext();
  auto surface = drawFrame(context);
  auto bounds = tgfx::Rect::MakeWH(kWidth, kHeight);

  tgfxrn::SnapshotRequest request;
  request.format = "raw";
  request.x = 8;
  request.y = 0;
  request.width = 24;
  request.height = 24;
  request.scale = 1.5f;
  // Ratio 2 gives device (16, 0)-(64, 48): the colour's right third lands at the crop's left.
  auto source = tgfxrn::snapshotSourceRect(request, 2.0f, bounds);
  CHECK(source == tgfx::Rect::MakeLTRB(16, 0, 64, 48));

  int width = static_cast<int>(std::lround(source.width() * request.scale));
  int height = static_cast<int>(std::lround(source.height() * request.scale));
  CHECK(width == 72);
  CHECK(height == 72);
  // As TgfxRenderer::startSnapshots does: an untagged scratch drops the profile
  // the unscaled path keeps, and this test is the only host-side shadow of it.
  auto scratch = tgfx::Surface::Make(context, width, height, false, 1, false, 0,
                                     surface->colorSpace());
  auto image = surface->makeImageSnapshot();
  CHECK(scratch != nullptr);
  CHECK(image != nullptr);
  if (scratch == nullptr || image == nullptr) {
    device()->unlock();
    return;
  }
  auto* canvas = scratch->getCanvas();
  canvas->clear();
  canvas->scale(request.scale, request.scale);
  canvas->translate(-source.x(), -source.y());
  canvas->drawImage(image);

  tgfx::ImageInfo info;
  auto pixels = readBack(context, scratch, &info);
  device()->unlock();
  CHECK(pixels != nullptr);
  if (pixels == nullptr) {
    return;
  }

  std::shared_ptr<tgfx::Data> raw;
  int outWidth = 0;
  int outHeight = 0;
  tgfxrn::finishSnapshot(pixels, info, request,
                         [&](std::shared_ptr<tgfx::Data> data, int w, int h,
                             const std::string& error) {
                           CHECK(error.empty());
                           raw = std::move(data);
                           outWidth = w;
                           outHeight = h;
                         });
  CHECK(outWidth == 72);
  CHECK(outHeight == 72);
  CHECK(raw != nullptr);
  if (raw == nullptr) {
    return;
  }
  std::vector<uint8_t> out(static_cast<const uint8_t*>(raw->data()),
                           static_cast<const uint8_t*>(raw->data()) + raw->size());
  CHECK(out.size() == static_cast<size_t>(outWidth) * outHeight * 4);
  auto at = [&](int x, int y, int channel) {
    return static_cast<int>(out[(static_cast<size_t>(y) * outWidth + x) * 4 + channel]);
  };
  auto near = [](int value, int expected) {
    return value >= expected - 2 && value <= expected + 2;
  };
  int y = outHeight / 2;
  CHECK(near(at(8, y, 0), kR));
  CHECK(near(at(8, y, 1), kG));
  CHECK(near(at(8, y, 2), kB));
  CHECK(near(at(8, y, 3), kA));
  CHECK(at(30, y, 3) == 0);
  CHECK(at(60, y, 3) == 0);
  CHECK(near(at(20, y, 3), kA));
}

}  // namespace

/// The scaled path renders through a scratch surface of its own, so it is the one
/// that can silently drop the colour space the unscaled path gets for free — and
/// with it the ICC profile the encoder writes only when the pixmap has one.
void aScaledReadbackKeepsTheSurfaceProfile() {
  auto* context = device()->lockContext();
  CHECK(context != nullptr);
  if (context == nullptr) {
    return;
  }
  auto surface = drawFrame(context, tgfx::ColorSpace::DisplayP3());
  CHECK(surface != nullptr && surface->colorSpace() != nullptr);

  auto scratch = tgfx::Surface::Make(context, kWidth, kHeight, false, 1, false, 0,
                                     surface->colorSpace());
  auto image = surface->makeImageSnapshot();
  CHECK(scratch != nullptr && image != nullptr);
  if (scratch == nullptr || image == nullptr) {
    device()->unlock();
    return;
  }
  scratch->getCanvas()->clear();
  scratch->getCanvas()->drawImage(image);

  tgfx::ImageInfo info;
  auto pixels = readBack(context, scratch, &info);
  device()->unlock();
  CHECK(pixels != nullptr);
  // The readback carries the scratch's space, which is what finishSnapshot tags with.
  CHECK(info.colorSpace() != nullptr);
  if (pixels == nullptr) {
    return;
  }

  tgfxrn::SnapshotRequest request;
  std::shared_ptr<tgfx::Data> png;
  tgfxrn::finishSnapshot(pixels, info, request,
                         [&](std::shared_ptr<tgfx::Data> data, int, int,
                             const std::string& error) {
                           CHECK(error.empty());
                           png = std::move(data);
                         });
  CHECK(png != nullptr);
  if (png == nullptr) {
    return;
  }
  // libpng writes iCCP only when the pixmap has a colour space, so the chunk
  // name in the encoded bytes is the assertion.
  const auto* bytes = static_cast<const uint8_t*>(png->data());
  bool tagged = false;
  for (size_t i = 8; i + 4 <= png->size(); i++) {
    if (bytes[i] == 'i' && bytes[i + 1] == 'C' && bytes[i + 2] == 'C' && bytes[i + 3] == 'P') {
      tagged = true;
      break;
    }
  }
  CHECK(tagged);
}

int main() {
  std::printf("snapshotRoundTripsThroughPng\n");
  snapshotRoundTripsThroughPng();
  std::printf("saveWritesADecodableFile\n");
  saveWritesADecodableFile();
  std::printf("badRequestsReportInsteadOfProducingABlankImage\n");
  badRequestsReportInsteadOfProducingABlankImage();
  std::printf("fractionalRectsRoundInsteadOfTruncating\n");
  fractionalRectsRoundInsteadOfTruncating();
  std::printf("cropAndScaleLandWhereTheyWereAsked\n");
  cropAndScaleLandWhereTheyWereAsked();
  std::printf("aScaledReadbackKeepsTheSurfaceProfile\n");
  aScaledReadbackKeepsTheSurfaceProfile();
  std::printf(failures == 0 ? "PASS\n" : "FAILED (%d)\n", failures);
  return failures == 0 ? 0 : 1;
}
