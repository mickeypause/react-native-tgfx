// Host tests for the compiled draw program, through a real headless Metal surface.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "BufferRegistry.hpp"
#include "FontRegistry.hpp"
#include "ImageRegistry.hpp"
#include "TextLayout.hpp"
#include "SvgImage.hpp"
#include "PathOps.hpp"
#include "SvgPath.hpp"
#include <fstream>

#include "OffscreenRender.hpp"
#include "PdfExport.hpp"
#include "OffscreenSurface.hpp"
#include "Snapshot.hpp"
#include "SvgExport.hpp"
#include "TgfxScene.hpp"
#include "Viewport.hpp"
#include "tgfx/core/Canvas.h"
#include "tgfx/core/Data.h"
#include "tgfx/core/EncodedFormat.h"
#include "tgfx/core/Image.h"
#include "tgfx/core/ImageCodec.h"
#include "tgfx/core/ImageInfo.h"
#include "core/codecs/jpeg/JpegCodec.h"
#include "tgfx/core/Orientation.h"
#include "tgfx/core/Pixmap.h"
#include "tgfx/core/ColorSpace.h"
#include "tgfx/core/Surface.h"
#include "tgfx/core/Typeface.h"
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

tgfxrn::TextRun run(std::string text) {
  tgfxrn::TextRun out;
  out.text = std::move(text);
  return out;
}

constexpr int kWidth = 64;
constexpr int kHeight = 64;

/// The example app's bundled fonts, set by cpp/tests/CMakeLists.txt.
constexpr const char* kFontDir = TGFXRN_TEST_FONT_DIR;

/// Mirrors src/compiler/writer.ts: header, node stream, then the two patch tables.
struct Writer {
  std::vector<double> words{0, 0, 0};
  std::vector<double> patches;
  std::vector<double> stringPatches;

  void push(double value) {
    words.push_back(value);
  }

  void pushAnim(double literal, int slot = -1) {
    if (slot >= 0) {
      patches.push_back(static_cast<double>(words.size()));
      patches.push_back(slot);
    }
    words.push_back(literal);
  }

  /// A string-table index that may follow a string slot, like pushStringAnim().
  void pushStringAnim(int tableIndex, int slot = -1) {
    if (slot >= 0) {
      stringPatches.push_back(static_cast<double>(words.size()));
      stringPatches.push_back(slot);
    }
    words.push_back(tableIndex);
  }

  void pushPaint(uint32_t color, bool stroke, double strokeWidth = 1,
                 double alpha = 1, int alphaSlot = -1,
                 int strokeWidthSlot = -1) {
    push(color);
    push((stroke ? 1 : 0) | 2);
    pushAnim(strokeWidth, strokeWidthSlot);
    push(0);
    push(0);
    pushAnim(alpha, alphaSlot);
    push(-1);
  }

  void pushLiveColorPaint(uint32_t color,
                          const std::array<std::pair<double, int>, 4>& rgba) {
    push(color);
    push(2 | 8);
    for (const auto& channel : rgba) {
      pushAnim(channel.first, channel.second);
    }
    pushAnim(1);  // strokeWidth
    push(0);
    push(0);
    pushAnim(1);  // alpha
    push(-1);     // no shader
  }

  struct Filters {
    int blend = 3;  // SrcOver
    std::vector<double> colorFilter{-1};
    std::vector<double> imageFilter{-1};
  };

  void pushFilteredPaint(uint32_t color, const Filters& filters,
                         int alphaSlot = -1, double strokeWidth = 1) {
    push(color);
    push(2 | 4);
    pushAnim(strokeWidth);
    push(0);
    push(0);
    pushAnim(1, alphaSlot);
    push(filters.blend);
    for (double word : filters.colorFilter) {
      push(word);
    }
    for (double word : filters.imageFilter) {
      push(word);
    }
    push(-1);  // no shader
  }

  struct Uniform {
    int name;
    std::vector<std::pair<double, int>> components;
  };

  /// A runtime-shader paint. Strings: 0 = body, then names; `children` is one payload per child.
  void pushRuntimePaint(uint32_t color, const std::vector<Uniform>& uniforms, int preamble,
                        const std::vector<int>& functions = {},
                        const std::vector<std::vector<double>>& children = {}) {
    push(color);
    push(2);
    pushAnim(1);
    push(0);
    push(0);
    pushAnim(1);
    push(6);  // runtime shader
    push(0);  // body
    push(preamble);
    push(static_cast<double>(functions.size()));
    for (int function : functions) {
      push(function);
    }
    push(static_cast<double>(uniforms.size()));
    for (const auto& uniform : uniforms) {
      push(uniform.name);
      push(static_cast<double>(uniform.components.size()));
      for (const auto& [value, slot] : uniform.components) {
        pushAnim(value, slot);
      }
    }
    push(static_cast<double>(children.size()));
    for (const auto& child : children) {
      for (double word : child) {
        push(word);
      }
    }
  }

  /// A vertices node, mirroring writeVertices() in compile.ts.
  /// Arguments in wire order: positions, texture coordinates, colours, then indices.
  void pushVertices(const std::vector<std::array<double, 2>>& positions,
                    const std::vector<std::array<double, 2>>& texCoords = {},
                    const std::vector<uint32_t>& colors = {},
                    const std::vector<int>& indices = {}, int topology = 0) {
    push(16);
    push(0);  // inline, no shared buffer
    push(topology);
    push(static_cast<double>(positions.size()));
    for (const auto& point : positions) {
      push(point[0]);
      push(point[1]);
    }
    push(texCoords.empty() ? 0 : 1);
    for (const auto& point : texCoords) {
      push(point[0]);
      push(point[1]);
    }
    push(colors.empty() ? 0 : 1);
    for (uint32_t color : colors) {
      push(static_cast<double>(color));
    }
    push(static_cast<double>(indices.size()));
    for (int index : indices) {
      push(static_cast<double>(index));
    }
    pushPaint(0xffffffff, false);
  }

  /// An atlas node, mirroring writeAtlas() in compile.ts. Sprites carry the
  /// v26 centre form: srcX, srcY, srcW, srcH, scale, centreX, centreY.
  void pushAtlas(int32_t handle, const std::vector<std::array<double, 7>>& sprites,
                 int sampling = 1) {
    push(15);
    pushAnim(handle);
    push(sampling);
    push(0);  // inline, no shared buffer
    push(static_cast<double>(sprites.size()));
    for (const auto& sprite : sprites) {
      for (double word : sprite) {
        push(word);
      }
    }
    pushPaint(0xffffffff, false);
  }

  /// An image node, mirroring writeDrawing() in compile.ts.
  void pushImage(int32_t handle, double x, double y, double w, double h,
                 int fit = 0, int sampling = 1, const double* srcRect = nullptr) {
    push(7);
    pushAnim(x);
    pushAnim(y);
    pushAnim(w);
    pushAnim(h);
    push(handle);
    push(fit);
    push(sampling);
    if (srcRect == nullptr) {
      push(0);
    } else {
      push(1);
      for (int i = 0; i < 4; i++) {
        push(srcRect[i]);
      }
    }
    pushPaint(0xffffffff, false);
  }

  /// A <CameraFrame>: paint, then `childCount` nodes the caller pushes after this.
  void pushCameraFrame(int32_t handle, int childCount, int fit = 2,
                       int quarterTurns = 0, bool mirror = false,
                       int sampling = 1) {
    push(11);
    push(handle);
    push(fit);
    push(sampling);
    push(quarterTurns);
    push(mirror ? 1 : 0);
    pushPaint(0xffffffff, false);
    push(childCount);
  }

  void pushImageShaderPaint(int32_t handle, int tileX, int tileY,
                            int sampling = 1) {
    push(0xffffffff);
    push(2);
    pushAnim(1);
    push(0);
    push(0);
    pushAnim(1);
    push(7);  // image shader
    push(handle);
    push(tileX);
    push(tileY);
    push(sampling);
    push(0);  // no fit rect
  }

  std::vector<double> finish() {
    auto patchStart = words.size();
    words.push_back(static_cast<double>(patches.size() / 2));
    for (double patch : patches) {
      words.push_back(patch);
    }
    words.push_back(static_cast<double>(stringPatches.size() / 2));
    for (double patch : stringPatches) {
      words.push_back(patch);
    }
    words[0] = tgfxrn::kSceneVersion;
    words[1] = static_cast<double>(words.size());
    words[2] = static_cast<double>(patchStart);
    return words;
  }
};

tgfx::Device* device() {
  static auto shared = tgfx::MetalDevice::Make();
  return shared.get();
}

using Pixels = std::vector<uint8_t>;

/// Draw one frame of an already-compiled program, so a test can prove what a *second* frame does.
Pixels drawFrame(tgfxrn::SceneRenderer& renderer, const std::vector<double>& values,
                 float pixelRatio = 1, int width = kWidth, int height = kHeight) {
  auto* context = device()->lockContext();
  auto surface = tgfx::Surface::Make(context, width, height);
  CHECK(surface != nullptr);
  auto* canvas = surface->getCanvas();
  canvas->clear();
  // What TgfxRenderer does: scene coordinates are logical pixels, the surface physical.
  if (pixelRatio != 1) {
    canvas->save();
    canvas->concat(tgfx::Matrix::MakeScale(pixelRatio, pixelRatio));
  }
  renderer.render(canvas, values);
  if (pixelRatio != 1) {
    canvas->restore();
  }
  context->flushAndSubmit(true);
  auto info = tgfx::ImageInfo::Make(width, height, tgfx::ColorType::RGBA_8888);
  Pixels pixels(static_cast<size_t>(info.byteSize()));
  surface->readPixels(info, pixels.data());
  device()->unlock();
  return pixels;
}

Pixels render(const std::vector<double>& words,
              const std::vector<std::string>& strings,
              const std::vector<double>& values,
              float pixelRatio = 1, int width = kWidth, int height = kHeight) {
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(words, strings);
  return drawFrame(renderer, values, pixelRatio, width, height);
}

bool anyPixelSet(const Pixels& pixels) {
  for (uint8_t byte : pixels) {
    if (byte != 0) {
      return true;
    }
  }
  return false;
}

std::array<uint8_t, 4> cellColor(int index) {
  return {static_cast<uint8_t>(10 + index * 6),
          static_cast<uint8_t>(200 - index * 4),
          static_cast<uint8_t>(60 + index * 3), 255};
}

/// Encode as PNG and register it the way the TgfxImage hybrid object does. Returns the handle.
tgfxrn::ImageHandle registerPixels(int width, int height,
                                   const std::vector<uint8_t>& pixels) {
  auto info = tgfx::ImageInfo::Make(width, height, tgfx::ColorType::RGBA_8888,
                                    tgfx::AlphaType::Premultiplied);
  tgfx::Pixmap pixmap(info, pixels.data());
  auto data = tgfx::ImageCodec::Encode(pixmap, tgfx::EncodedFormat::PNG, 100);
  if (data == nullptr) {
    return {};
  }
  return tgfxrn::ImageRegistry::instance().add(
      tgfx::ImageCodec::MakeFrom(std::move(data)));
}

tgfxrn::ImageHandle registerSolid(const std::array<uint8_t, 4>& rgba,
                                  int width = 2, int height = 2) {
  std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4);
  for (size_t i = 0; i < pixels.size(); i += 4) {
    std::copy(rgba.begin(), rgba.end(), pixels.begin() + static_cast<long>(i));
  }
  return registerPixels(width, height, pixels);
}

std::array<uint8_t, 3> pixelAt(const Pixels& pixels, int x, int y) {
  size_t at = (static_cast<size_t>(y) * kWidth + static_cast<size_t>(x)) * 4;
  return {pixels[at], pixels[at + 1], pixels[at + 2]};
}

size_t inkPixels(const Pixels& pixels) {
  size_t count = 0;
  for (size_t at = 3; at < pixels.size(); at += 4) {
    if (pixels[at] > 0) {
      count++;
    }
  }
  return count;
}

bool opaqueAt(const Pixels& pixels, int x, int y) {
  return pixels[(static_cast<size_t>(y) * kWidth + static_cast<size_t>(x)) * 4 + 3] > 0;
}

uint8_t alphaAt(const Pixels& pixels, int x, int y) {
  return pixels[(static_cast<size_t>(y) * kWidth + static_cast<size_t>(x)) * 4 + 3];
}

std::vector<double> oneRect() {
  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(8);
  w.pushAnim(8);
  w.pushAnim(40);
  w.pushAnim(40);
  w.pushPaint(0xff3366cc, false);
  return w.finish();
}

/// A corrupt program must be rejected without first trusting its counts: a regression here is an OOM.
void aCorruptProgramIsRejectedWithoutTrustingItsCounts() {
  auto valid = oneRect();
  CHECK(anyPixelSet(render(valid, {}, {})));

  const double kHuge = 2147483647.0;
  const double kNan = std::nan("");
  std::vector<std::vector<double>> corrupt;
  auto rootCount = valid;
  rootCount[3] = kHuge;
  corrupt.push_back(rootCount);
  Writer group;
  group.push(1);
  group.push(0);  // Group
  group.push(0);  // no transform ops
  group.push(kHuge);
  corrupt.push_back(group.finish());
  // The header's own words: a NaN or negative length is not a size_t.
  for (double bad : {kNan, -1.0, 1e300}) {
    auto header = valid;
    header[1] = bad;
    corrupt.push_back(header);
    auto patchStart = valid;
    patchStart[2] = bad;
    corrupt.push_back(patchStart);
  }

  for (const auto& words : corrupt) {
    // A fresh renderer, so "kept the previous program" cannot pass for "read it".
    tgfxrn::SceneRenderer renderer;
    renderer.setScene(words, {});
    CHECK(!anyPixelSet(drawFrame(renderer, {})));
  }
}

// tgfx batches different paths drawn back to back into one ShapeMultiDrawOp once their triangle
// meshes are resident — i.e. from the second frame on. The first frame therefore goes through
// one ShapeDrawOp per path and the later frames through the batch; both must draw the same
// pixels. Half the paths are stroked, every one has its own colour. The rotated pass is what an
// animated group produces and triangulates at any size; the axis-aligned pass exercises the
// identity split, so its paths must exceed tgfx's 162 px triangulation floor or they take the
// texture-mask route and never batch.
void batchedPathsDrawWhatPerShapeOpsDraw() {
  // Built once, like a scene keeps its paths: a Path's cache key is its PathRef's identity.
  auto squiggles = [](int cols, int rows, float w, float h) {
    std::vector<tgfx::Path> paths;
    for (int i = 0; i < cols * rows; i++) {
      float x = 8 + static_cast<float>(i % cols) * (w + 4);
      float y = 8 + static_cast<float>(i / cols) * (h + 22);
      tgfx::Path path;
      path.moveTo(x, y + h / 2);
      path.cubicTo(x + w / 4, y, x + w / 2, y + h, x + w, y + h / 2);
      path.cubicTo(x + w * 0.75f, y + h, x + w / 2, y, x, y + h / 2);
      path.close();
      paths.push_back(path);
    }
    return paths;
  };
  auto* context = device()->lockContext();
  auto frame = [&](const std::vector<tgfx::Path>& paths, int size, bool rotated) {
    auto surface = tgfx::Surface::Make(context, size, size);
    CHECK(surface != nullptr);
    auto* canvas = surface->getCanvas();
    canvas->clear();
    if (rotated) {
      canvas->rotate(17, size / 2.0f, size / 2.0f);
    }
    for (size_t i = 0; i < paths.size(); i++) {
      tgfx::Paint paint;
      paint.setColor(tgfx::Color::FromRGBA(static_cast<uint8_t>(40 + i * 17),
                                           static_cast<uint8_t>(200 - i * 13),
                                           static_cast<uint8_t>(90 + i * 11), 255));
      if (i % 2 == 1) {
        paint.setStyle(tgfx::PaintStyle::Stroke);
        paint.setStrokeWidth(2);
      }
      canvas->drawPath(paths[i], paint);
    }
    context->flushAndSubmit(true);
    auto info = tgfx::ImageInfo::Make(size, size, tgfx::ColorType::RGBA_8888);
    Pixels pixels(static_cast<size_t>(info.byteSize()));
    surface->readPixels(info, pixels.data());
    return pixels;
  };
  auto maxDiff = [](const Pixels& a, const Pixels& b) {
    int diff = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
      diff = std::max(diff, std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])));
    }
    return a.size() == b.size() ? diff : 255;
  };
  auto small = squiggles(4, 3, 42, 40);
  auto large = squiggles(2, 2, 200, 200);
  for (bool rotated : {true, false}) {
    const auto& paths = rotated ? small : large;
    int size = rotated ? 200 : 480;
    auto first = frame(paths, size, rotated);
    auto second = frame(paths, size, rotated);
    auto third = frame(paths, size, rotated);
    CHECK(anyPixelSet(first));
    // The batch adds the per-shape translation in the vertex shader instead of folding it into
    // the matrix, so allow last-bit coverage rounding but nothing a misplaced shape could pass.
    CHECK(maxDiff(first, second) <= 2);
    CHECK(second == third);
  }
  device()->unlock();
}

/// The glue strokes a static <Path> once and keeps the Shape; a cubic caches, and the rebuild
/// (an animated stroke width at the same value) takes drawPath, so the two must agree. A closed
/// rectangle must not cache: drawPath's fast path is what keeps it a <rect> in the export.
void aStrokedPathDrawsWhatTheRebuildDraws() {
  Writer literal;
  literal.push(1);
  literal.push(6);  // Path
  literal.push(0);
  literal.push(0);
  literal.pushPaint(0xffcc3366, true, 3);

  Writer animated;
  animated.push(1);
  animated.push(6);
  animated.push(0);
  animated.push(0);
  animated.pushPaint(0xffcc3366, true, 3, 1, -1, 0);

  const std::vector<std::string> cubic = {"M8 32 C20 4 44 60 56 32"};
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(literal.finish(), cubic);
  auto first = drawFrame(renderer, {});
  auto second = drawFrame(renderer, {});
  CHECK(anyPixelSet(first));
  CHECK(first == second);
  CHECK(first == render(animated.finish(), cubic, {3}));

  auto svg = tgfxrn::renderSceneToSvg(literal.finish(), {"M8 8 L56 8 L56 48 L8 48 Z"}, 64, 64,
                                      false);
  CHECK(svg.find("<rect") != std::string::npos);
  CHECK(svg.find("stroke-width") != std::string::npos);
}

void staticPaintDrawsWhatTheRebuildDraws() {
  Writer literal;
  literal.push(1);
  literal.push(1);  // Rect
  literal.pushAnim(8);
  literal.pushAnim(8);
  literal.pushAnim(40);
  literal.pushAnim(40);
  literal.pushPaint(0xff3366cc, false, 1, 0.5);

  Writer animated;
  animated.push(1);
  animated.push(1);
  animated.pushAnim(8);
  animated.pushAnim(8);
  animated.pushAnim(40);
  animated.pushAnim(40);
  animated.pushPaint(0xff3366cc, false, 1, 0, 0);

  auto cached = render(literal.finish(), {}, {});
  auto rebuilt = render(animated.finish(), {}, {0.5});
  CHECK(anyPixelSet(cached));
  CHECK(cached == rebuilt);
}

void animatedColorDrawsWhatTheLiteralDraws() {
  Writer literal;
  literal.push(1);
  literal.push(1);  // Rect
  literal.pushAnim(8);
  literal.pushAnim(8);
  literal.pushAnim(40);
  literal.pushAnim(40);
  literal.pushPaint(0xff3366cc, false);

  Writer animated;
  animated.push(1);
  animated.push(1);
  animated.pushAnim(8);
  animated.pushAnim(8);
  animated.pushAnim(40);
  animated.pushAnim(40);
  // The packed word is deliberately wrong: only the slots may reach the paint.
  animated.pushLiveColorPaint(
      0xff000000, {{{0, 0}, {0, 1}, {0, 2}, {1, -1}}});

  auto words = animated.finish();
  auto cached = render(literal.finish(), {}, {});
  CHECK(anyPixelSet(cached));
  CHECK(cached == render(words, {}, {0x33 / 255.0, 0x66 / 255.0, 0xcc / 255.0}));
  CHECK(cached != render(words, {}, {1.0, 0.0, 0.0}));
}

void staticLineDrawsWhatTheRebuildDraws() {
  Writer literal;
  literal.push(1);
  literal.push(5);  // Line
  literal.pushAnim(8);
  literal.pushAnim(8);
  literal.pushAnim(56);
  literal.pushAnim(48);
  literal.pushPaint(0xffcc3366, true, 3);

  Writer animated;
  animated.push(1);
  animated.push(5);
  animated.pushAnim(0, 0);
  animated.pushAnim(8);
  animated.pushAnim(56);
  animated.pushAnim(48);
  animated.pushPaint(0xffcc3366, true, 3);

  auto cached = render(literal.finish(), {}, {});
  auto rebuilt = render(animated.finish(), {}, {8});
  CHECK(anyPixelSet(cached));
  CHECK(cached == rebuilt);
}

void animatedStrokeWidthDrawsWhatTheRebuildDraws() {
  Writer literal;
  literal.push(1);
  literal.push(1);  // Rect
  literal.pushAnim(8);
  literal.pushAnim(8);
  literal.pushAnim(40);
  literal.pushAnim(40);
  literal.pushPaint(0xffcc3366, true, 6);

  Writer animated;
  animated.push(1);
  animated.push(1);
  animated.pushAnim(8);
  animated.pushAnim(8);
  animated.pushAnim(40);
  animated.pushAnim(40);
  animated.pushPaint(0xffcc3366, true, 0, 1, -1, 0);

  auto words = animated.finish();
  auto cached = render(literal.finish(), {}, {});
  CHECK(anyPixelSet(cached));
  CHECK(cached == render(words, {}, {6}));
  CHECK(cached != render(words, {}, {2}));
}

/// 0 the text, 1 the family, 2 an empty ellipsis, 3 a two-line text.
const std::vector<std::string> kTextStrings{"Wg", "Helvetica", "", "Wg\nWg"};

/// A <Text> node body, mirroring the Kind.Text case of writeDrawing(); zeroes mean "the paragraph's".
void writeRun(Writer& w, int textIndex, int familyIndex = 2, int weight = 0, int italic = 0,
              double size = 0, int hasColor = 0, double color = 0) {
  w.push(1);  // one run
  w.push(textIndex);
  w.push(familyIndex);
  w.push(weight);
  w.push(italic);
  w.push(0);  // letterSpacing
  w.push(size);
  w.push(hasColor);
  w.push(color);
}

void writeTextNode(Writer& w, double fontSize, int fontSizeSlot, int textIndex = 0,
                   double wrapWidth = 0, int align = 0, double lineHeight = 0,
                   int weight = 400, int style = 0, int runItalic = 0) {
  w.push(8);  // Text
  w.pushAnim(4);
  w.pushAnim(44);
  w.pushAnim(fontSize, fontSizeSlot);
  w.push(1);  // font family string index
  w.push(weight);
  w.push(style);  // 1 italic, 2 faux bold, 4 faux italic
  w.push(0);      // letterSpacing
  writeRun(w, textIndex, 2, 0, runItalic);
  w.push(wrapWidth);
  w.push(align);
  w.push(lineHeight);
  w.push(0);  // maxLines
  w.push(2);  // ellipsis string index (empty)
  w.pushPaint(0xff000000, false);
}

void writeText(Writer& w, double fontSize, int fontSizeSlot, int textIndex = 0,
               double wrapWidth = 0, int align = 0, double lineHeight = 0, int weight = 400,
               int style = 0, int runItalic = 0) {
  w.push(1);
  writeTextNode(w, fontSize, fontSizeSlot, textIndex, wrapWidth, align, lineHeight, weight, style,
                runItalic);
}

void staticTextDrawsWhatTheRebuildDraws() {
  Writer literal;
  writeText(literal, 20, -1);
  Writer animated;
  writeText(animated, 0, 0);

  auto cached = render(literal.finish(), kTextStrings, {});
  auto rebuilt = render(animated.finish(), kTextStrings, {20});
  CHECK(anyPixelSet(cached));
  CHECK(cached == rebuilt);
}

void animatedFieldsStillAnimate() {
  Writer rect;
  rect.push(1);
  rect.push(1);
  rect.pushAnim(0, 0);
  rect.pushAnim(8);
  rect.pushAnim(16);
  rect.pushAnim(16);
  rect.pushPaint(0xff3366cc, false);
  auto rectWords = rect.finish();
  CHECK(render(rectWords, {}, {4}) != render(rectWords, {}, {40}));

  Writer alpha;
  alpha.push(1);
  alpha.push(1);
  alpha.pushAnim(8);
  alpha.pushAnim(8);
  alpha.pushAnim(40);
  alpha.pushAnim(40);
  alpha.pushPaint(0xff3366cc, false, 1, 0, 0);
  auto alphaWords = alpha.finish();
  CHECK(render(alphaWords, {}, {1.0}) != render(alphaWords, {}, {0.25}));

  Writer line;
  line.push(1);
  line.push(5);
  line.pushAnim(8);
  line.pushAnim(8);
  line.pushAnim(0, 0);
  line.pushAnim(48);
  line.pushPaint(0xffcc3366, true, 3);
  auto lineWords = line.finish();
  CHECK(render(lineWords, {}, {20}) != render(lineWords, {}, {56}));

  Writer text;
  writeText(text, 0, 0);
  auto textWords = text.finish();
  auto small = render(textWords, kTextStrings, {10});
  auto large = render(textWords, kTextStrings, {30});
  CHECK(anyPixelSet(small));
  CHECK(small != large);
}

}  // namespace

/// A slotted uniform must draw what the literal drew and still move, from a flat cross-uniform run.
void animatedUniformDrawsWhatTheRebuildDraws() {
  const std::vector<std::string> strings = {
      "return vec4(uLevel, uTint) + vec4(uPad.x, uPad.y, 0.0, 0.0);", "uTint",
      "uLevel", "uPad", ""};
  const std::vector<Writer::Uniform> baseUniforms = {
      {1, {{0.1, -1}, {0.2, -1}, {0.3, -1}}},
      {2, {{0.6, -1}}},
      {3, {{0.05, -1}, {0.0, -1}}},
  };
  auto slotted = baseUniforms;
  // Slot 1, not 0: resolving every component against values[0] would survive a slot-0 test.
  slotted[1].components[0] = {0, 1};

  auto write = [](const std::vector<Writer::Uniform>& uniforms) {
    Writer w;
    w.push(1);
    w.push(1);  // Rect
    w.pushAnim(8);
    w.pushAnim(8);
    w.pushAnim(40);
    w.pushAnim(40);
    w.pushRuntimePaint(0xff000000, uniforms, 4);
    return w.finish();
  };

  auto words = write(slotted);
  auto baked = render(write(baseUniforms), strings, {});
  auto resolved = render(words, strings, {0.0, 0.6});
  CHECK(anyPixelSet(baked));
  CHECK(baked == resolved);

  CHECK(render(words, strings, {0.0, 0.2}) != resolved);

  // An empty-named uniform must take its slots with it, or the flat run desyncs.
  auto withDropped = slotted;
  withDropped.insert(withDropped.begin(), {4, {{0.9, 0}, {0.9, -1}}});
  CHECK(render(write(withDropped), strings, {0.0, 0.6}) == baked);
}

/// A file-scope helper must reach the shader, and the preamble must key the compiled program.
void aHelperFunctionCompilesAndDoesNotCollide() {
  const std::vector<std::string> strings = {
      "return vec4(tint(uLevel), 0.0, 0.0, 1.0);",  // 0: the only body
      "uLevel",                                     // 1
      "float tint(float v) { return v; }",          // 2: identity preamble
      "tint",                                       // 3
      "float tint(float v) { return 1.0 - v; }",    // 4: inverting preamble
      "",                                           // 5: no preamble
  };
  const std::vector<Writer::Uniform> uniforms = {{1, {{0.75, -1}}}};

  auto draw = [&](int preamble, const std::vector<int>& functions) {
    Writer w;
    w.push(1);
    w.push(1);  // Rect
    w.pushAnim(8);
    w.pushAnim(8);
    w.pushAnim(40);
    w.pushAnim(40);
    w.pushRuntimePaint(0xff000000, uniforms, preamble, functions);
    return render(w.finish(), strings, {});
  };

  // Without the file-scope helper the body calls an undeclared tint and nothing paints.
  auto identity = draw(2, {3});
  CHECK(anyPixelSet(identity));
  // uLevel 0.75 here, 0.25 next, through a helper of the same name: one body, two programs.
  auto inverted = draw(4, {3});
  CHECK(anyPixelSet(inverted));
  CHECK(identity != inverted);
}

void aChildShaderSamplesWhereItIsAsked() {
  // A 2x1 image repeated across 32x32, so one texel of drift is a whole colour change.
  auto left = cellColor(2);
  auto right = cellColor(20);
  std::vector<uint8_t> pixels(8);
  std::copy(left.begin(), left.end(), pixels.begin());
  std::copy(right.begin(), right.end(), pixels.begin() + 4);
  auto image = registerPixels(2, 1, pixels);

  // repeat/repeat, nearest: the same child both paths draw.
  const std::vector<double> child = {7, static_cast<double>(image.handle), 1, 1, 0, 0};
  const std::vector<std::string> strings = {
      "return TGFX_CHILD_0(fragCoord);",                   // 0
      "",                                                  // 1: no preamble
      "return TGFX_CHILD_0(fragCoord + vec2(1.0, 0.0));",  // 2: shifted a texel
  };

  auto draw = [&](int body) {
    Writer w;
    w.push(1);
    w.push(1);  // Rect
    w.pushAnim(0);
    w.pushAnim(0);
    w.pushAnim(32);
    w.pushAnim(32);
    w.pushRuntimePaint(0xffffffff, {}, 1, {}, {child});
    // pushRuntimePaint writes string 0 as the body; swap it per case.
    std::vector<std::string> table = strings;
    table[0] = strings[static_cast<size_t>(body)];
    return render(w.finish(), table, {});
  };

  auto sampled = draw(0);
  CHECK(anyPixelSet(sampled));
  auto at0 = pixelAt(sampled, 0, 0);
  auto at1 = pixelAt(sampled, 1, 0);
  auto at2 = pixelAt(sampled, 2, 0);
  CHECK(at0[0] == left[0] && at0[1] == left[1]);
  CHECK(at1[0] == right[0] && at1[1] == right[1]);
  CHECK(at2[0] == left[0] && at2[1] == left[1]);

  // The same child one texel right: a wrapper ignoring its argument redraws the block above.
  auto shifted = draw(2);
  auto shifted0 = pixelAt(shifted, 0, 0);
  CHECK(shifted0[0] == right[0] && shifted0[1] == right[1]);
}

void aGradientChildShades() {
  // Linear black -> white across the 32-wide rect.
  const std::vector<double> child = {0, 0, 0, 32, 0, 0, 2, 0xff000000, 0xffffffff, 0};
  const std::vector<std::string> strings = {"return TGFX_CHILD_0(fragCoord);", ""};

  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(32);
  w.pushAnim(32);
  w.pushRuntimePaint(0xffffffff, {}, 1, {}, {child});
  auto drawn = render(w.finish(), strings, {});
  CHECK(anyPixelSet(drawn));
  // Dark left, bright right: the child's own coordinate ramp reached it.
  CHECK(pixelAt(drawn, 1, 16)[0] < 40);
  CHECK(pixelAt(drawn, 30, 16)[0] > 200);
}

void twoChildrenOnOneShaderBothPaint() {
  // Child 0: a 2x1 image, red on the left texel and green on the right.
  auto left = cellColor(2);
  auto right = cellColor(20);
  std::vector<uint8_t> pixels(8);
  std::copy(left.begin(), left.end(), pixels.begin());
  std::copy(right.begin(), right.end(), pixels.begin() + 4);
  auto image = registerPixels(2, 1, pixels);

  const std::vector<std::vector<double>> children = {
      {7, static_cast<double>(image.handle), 1, 1, 0, 0},
      // Child 1: a black -> white ramp across the 32-wide rect.
      {0, 0, 0, 32, 0, 0, 2, 0xff000000, 0xffffffff, 0},
  };
  // Child 0 in red and green (one shifted), child 1 in blue: a wrong binding is a wrong channel.
  const std::vector<std::string> strings{
      "return vec4(TGFX_CHILD_0(fragCoord).r, "
      "TGFX_CHILD_0(fragCoord + vec2(1.0, 0.0)).g, TGFX_CHILD_1(fragCoord).b, 1.0);",
      ""};

  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(32);
  w.pushAnim(32);
  w.pushRuntimePaint(0xffffffff, {}, 1, {}, children);
  auto drawn = render(w.finish(), strings, {});
  CHECK(anyPixelSet(drawn));
  // Column 0: child 0 at texels 0 and 1, child 1 near the dark end of its ramp.
  auto at0 = pixelAt(drawn, 0, 16);
  CHECK(at0[0] == left[0]);
  CHECK(at0[1] == right[1]);
  CHECK(at0[2] < 40);
  // The far end only differs in blue, which is the one channel child 1 drives.
  auto at30 = pixelAt(drawn, 30, 16);
  CHECK(at30[0] == left[0]);
  CHECK(at30[2] > 200);
}

void aRuntimeShaderChildNestsInOneProgram() {
  auto only = cellColor(7);
  std::vector<uint8_t> onePixel(only.begin(), only.end());
  auto image = registerPixels(1, 1, onePixel);

  const std::vector<std::string> strings = {
      "return TGFX_CHILD_0(fragCoord);",  // 0: both bodies
      "",                                 // 1: no preamble
  };
  // The inner runtime shader as a child payload: body, preamble, no functions, one child.
  const std::vector<double> inner = {6, 0, 1, 0, 0, 1,
                                     7, static_cast<double>(image.handle), 0, 0, 0, 0};

  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(32);
  w.pushAnim(32);
  w.pushRuntimePaint(0xffffffff, {}, 1, {}, {inner});
  auto drawn = render(w.finish(), strings, {});
  auto middle = pixelAt(drawn, 16, 16);
  CHECK(middle[0] == only[0] && middle[1] == only[1]);
}

void twoShadersWithChildrenDoNotCollide() {
  auto first = cellColor(3);
  auto second = cellColor(30);
  std::vector<uint8_t> onePixel(4);
  std::copy(first.begin(), first.end(), onePixel.begin());
  auto imageA = registerPixels(1, 1, onePixel);
  std::copy(second.begin(), second.end(), onePixel.begin());
  auto imageB = registerPixels(1, 1, onePixel);

  const std::vector<std::string> strings = {"return TGFX_CHILD_0(fragCoord);", ""};
  Writer w;
  w.push(2);
  for (auto handle : {imageA.handle, imageB.handle}) {
    w.push(1);  // Rect
    w.pushAnim(handle == imageA.handle ? 0 : 32);
    w.pushAnim(0);
    w.pushAnim(32);
    w.pushAnim(32);
    w.pushRuntimePaint(0xffffffff, {}, 1, {},
                       {{7, static_cast<double>(handle), 0, 0, 0, 0}});
  }
  auto drawn = render(w.finish(), strings, {});
  auto onLeft = pixelAt(drawn, 16, 16);
  auto onRight = pixelAt(drawn, 48, 16);
  CHECK(onLeft[0] == first[0] && onLeft[1] == first[1]);
  CHECK(onRight[0] == second[0] && onRight[1] == second[1]);
}

/// A miter limit under the join's ratio cuts the spike back to a bevel; over it, the spike stands.
void aMiterLimitCutsBackASharpSpike() {
  auto spike = [](double limit) {
    Writer w;
    w.push(1);
    w.push(6);  // Path
    w.push(0);  // path string index
    w.push(0);  // no path effects
    w.push(0xff000000);
    w.push(1 | 2 | (limit > 0 ? 16 : 0));  // stroke, antiAlias, and a miter limit when asked
    w.pushAnim(6);                         // strokeWidth
    w.push(0);                             // butt cap
    w.push(0);                             // miter join
    if (limit > 0) {
      w.push(limit);
    }
    w.pushAnim(1);  // alpha
    w.push(-1);     // no shader
    return render(w.finish(), {"M4 60 L32 6 L60 60"}, {});
  };
  auto bevelled = spike(1);
  auto mitered = spike(20);
  CHECK(anyPixelSet(bevelled));
  CHECK(bevelled != mitered);
  // The join's ratio here is ~2.2, so the spike runs well past where a bevel would cut it off.
  CHECK(opaqueAt(mitered, 32, 2));
  CHECK(!opaqueAt(bevelled, 32, 2));
  // The default, with no limit word at all, is 4 — over that ratio, so it keeps the spike.
  CHECK(opaqueAt(spike(0), 32, 2));
}

/// A group clip confines its subtree; inverting it keeps exactly the complement.
void aGroupClipConfinesItsSubtree() {
  // kind 13 (GroupLayer): opCount, flags, clip block, [alpha], childCount, children.
  // clipKind 3 reads its path from the string table instead of four geometry words.
  auto scene = [](int clipKind, bool invert) {
    Writer w;
    w.push(1);
    w.push(13);
    w.push(0);  // no transform ops
    w.push(1 | (invert ? 2 : 0));
    w.push(clipKind);
    if (clipKind == 3) {
      w.push(0);
    } else {
      w.push(8);
      w.push(8);
      w.push(24);
      w.push(24);
      if (clipKind == 2) {
        w.push(6);
        w.push(6);
      }
    }
    w.push(1);  // one child
    w.push(9);  // Fill: covers the canvas, so only the clip decides what lands
    w.pushPaint(0xff3366cc, false);
    return render(w.finish(), {"M8 8 L32 8 L32 32 L8 32 Z"}, {});
  };

  auto clipped = scene(1, false);
  CHECK(opaqueAt(clipped, 16, 16));  // inside the clip
  CHECK(!opaqueAt(clipped, 2, 2));   // outside it
  CHECK(!opaqueAt(clipped, 40, 40));

  auto inverted = scene(1, true);
  CHECK(!opaqueAt(inverted, 16, 16));  // exactly the complement
  CHECK(opaqueAt(inverted, 2, 2));
  CHECK(opaqueAt(inverted, 40, 40));

  // A round clip cuts its corners off; the centre of the box is still covered.
  auto rounded = scene(2, false);
  CHECK(opaqueAt(rounded, 20, 20));
  CHECK(opaqueAt(clipped, 9, 9));
  CHECK(pixelAt(rounded, 9, 9) != pixelAt(clipped, 9, 9));

  // A path clip covers the same box, so it must land exactly where the rect clip did.
  CHECK(scene(3, false) == clipped);
  CHECK(scene(3, true) == inverted);
}

/// Two overlapping half-opaque rects: a layer composites once, so the overlap does not darken.
void aLayerCompositesTheSubtreeOnce() {
  auto scene = [](bool layer) {
    Writer w;
    w.push(1);
    w.push(layer ? 13 : 0);
    w.push(0);
    if (layer) {
      w.push(4);  // layer, no clip
      w.pushAnim(0.5);
    }
    w.push(2);
    for (int i = 0; i < 2; i++) {
      w.push(1);  // Rect
      w.pushAnim(4 + i * 12);
      w.pushAnim(4);
      w.pushAnim(28);
      w.pushAnim(28);
      // Without a layer the group's alpha has to ride each child instead.
      w.push(0xff3366cc);
      w.push(2);
      w.pushAnim(1);
      w.push(0);
      w.push(0);
      w.pushAnim(layer ? 1.0 : 0.5);
      w.push(-1);
    }
    return render(w.finish(), {}, {});
  };

  auto layered = scene(true);
  auto perChild = scene(false);
  // Off the overlap both agree; inside it the per-child version has blended twice.
  CHECK(pixelAt(layered, 8, 16) == pixelAt(perChild, 8, 16));
  CHECK(pixelAt(layered, 20, 16) != pixelAt(perChild, 20, 16));
  CHECK(pixelAt(layered, 20, 16) == pixelAt(layered, 8, 16));

  // Clip and layer together: the clip still bites, and the surviving ink is faded once.
  Writer both;
  both.push(1);
  both.push(13);
  both.push(0);
  both.push(1 | 4);  // clip + layer
  both.push(1);      // rect clip
  both.push(0);
  both.push(0);
  both.push(24);
  both.push(24);
  both.pushAnim(0.5);
  both.push(1);
  both.push(9);  // Fill
  both.pushPaint(0xff3366cc, false);
  auto clipLayer = render(both.finish(), {}, {});
  CHECK(opaqueAt(clipLayer, 12, 12));
  CHECK(!opaqueAt(clipLayer, 40, 40));
  CHECK(pixelAt(clipLayer, 12, 12) == pixelAt(layered, 8, 16));
}

/// Dash, corner and trim reshape the geometry; trim's slots move without a re-commit.
void pathEffectsReshapeTheGeometry() {
  // path | flags | [start end] | [count intervals phase] | [corner] then the paint.
  auto stroke = [](int flags, const std::vector<double>& block) {
    Writer w;
    w.push(1);
    w.push(6);
    w.push(0);
    w.push(flags);
    for (double word : block) {
      w.push(word);
    }
    w.push(0xff000000);
    w.push(1 | 2);
    w.pushAnim(4);
    w.push(0);
    w.push(0);
    w.pushAnim(1);
    w.push(-1);
    return render(w.finish(), {"M4 32 L60 32"}, {});
  };

  auto solid = stroke(0, {});
  CHECK(anyPixelSet(solid));

  // A dash removes ink between the on-intervals, so the same line covers fewer pixels.
  auto dashed = stroke(2, {2, 8, 8, 0});
  CHECK(anyPixelSet(dashed));
  CHECK(inkPixels(dashed) < inkPixels(solid));

  // Phase slides the pattern, so the same intervals land elsewhere.
  CHECK(stroke(2, {2, 8, 8, 4}) != dashed);

  // Trim keeps the first half only.
  auto half = stroke(1, {0, 0.5});
  CHECK(inkPixels(half) < inkPixels(solid));
  CHECK(opaqueAt(half, 10, 32));
  CHECK(!opaqueAt(half, 55, 32));

  // An even-odd fill has to survive corner and dash, which rebuild the path.
  auto donut = [](int flags, const std::vector<double>& block) {
    Writer w;
    w.push(1);
    w.push(6);
    w.push(0);
    w.push(flags);
    for (double word : block) {
      w.push(word);
    }
    w.pushPaint(0xff000000, false);
    return render(w.finish(),
                  {"M4 4 L60 4 L60 60 L4 60 Z M20 20 L44 20 L44 44 L20 44 Z"}, {});
  };
  CHECK(!opaqueAt(donut(8, {}), 32, 32));                // the hole is open
  CHECK(opaqueAt(donut(0, {}), 32, 32));                 // winding fills it
  CHECK(!opaqueAt(donut(8 | 4, {6}), 32, 32));           // still open through a corner effect
  CHECK(!opaqueAt(donut(8 | 2, {2, 6, 6, 0}), 32, 32));  // and through a dash

  // An animated trim re-applies per frame: the same program, two different values.
  tgfxrn::SceneRenderer renderer;
  Writer w;
  w.push(1);
  w.push(6);
  w.push(0);
  w.push(1);
  w.pushAnim(0, 0);
  w.push(1);
  w.push(0xff000000);
  w.push(1 | 2);
  w.pushAnim(4);
  w.push(0);
  w.push(0);
  w.pushAnim(1);
  w.push(-1);
  renderer.setScene(w.finish(), {"M4 32 L60 32"});
  auto early = drawFrame(renderer, {0.0});
  auto late = drawFrame(renderer, {0.6});
  CHECK(inkPixels(late) < inkPixels(early));
  CHECK(!opaqueAt(late, 10, 32));

  // A static trim in front of a moving phase is baked, so only the dash re-runs.
  Writer mixed;
  mixed.push(1);
  mixed.push(6);
  mixed.push(0);
  mixed.push(1 | 2);
  mixed.push(0);
  mixed.push(0.5);  // static trim: the first half only
  mixed.push(2);
  mixed.push(4);
  mixed.push(4);
  mixed.pushAnim(0, 0);
  mixed.push(0xff000000);
  mixed.push(1 | 2);
  mixed.pushAnim(4);
  mixed.push(0);
  mixed.push(0);
  mixed.pushAnim(1);
  mixed.push(-1);
  tgfxrn::SceneRenderer marching;
  marching.setScene(mixed.finish(), {"M4 32 L60 32"});
  auto atZero = drawFrame(marching, {0.0});
  auto atFour = drawFrame(marching, {4.0});
  CHECK(anyPixelSet(atZero));
  CHECK(atZero != atFour);           // the phase still moves
  CHECK(!opaqueAt(atZero, 55, 32));  // and the baked trim still bites
}

/// Luma and the inner shadow: the filter kinds #179 was missing.
void theLateFilterKindsAllDraw() {
  auto shape = [](const std::vector<double>& colorFilter,
                  const std::vector<double>& imageFilter, uint32_t fill = 0xff3366cc) {
    Writer w;
    w.push(1);
    w.push(1);  // Rect
    w.pushAnim(12);
    w.pushAnim(12);
    w.pushAnim(40);
    w.pushAnim(40);
    w.pushFilteredPaint(fill, {3, colorFilter, imageFilter});
    return render(w.finish(), {}, {});
  };

  auto plain = shape({-1}, {-1});
  CHECK(anyPixelSet(plain));

  // Luma turns the fill into its own luminance, so the drawn colour changes.
  auto before = tgfxrn::filterBuildCount();
  auto luma = shape({2}, {-1});
  CHECK(luma != plain);
  CHECK(tgfxrn::filterBuildCount() == before + 1);

  // An inner shadow paints inside the rect, a drop shadow outside it.
  auto dropped = shape({-1}, {1, 4, 4, 3, 3, 0xff000000, 0});
  auto innerLit = shape({-1}, {1, 4, 4, 3, 3, 0xff000000, 2});
  CHECK(dropped != innerLit);
  CHECK(opaqueAt(dropped, 54, 54));    // the drop shadow spills past the rect
  CHECK(!opaqueAt(innerLit, 54, 54));  // the inner one never leaves it
  CHECK(pixelAt(innerLit, 16, 16) != pixelAt(plain, 16, 16));

  // Shadow-only composes with inner rather than replacing it.
  CHECK(shape({-1}, {1, 4, 4, 3, 3, 0xff000000, 3}) != innerLit);
}

/// The last two filter kinds tgfx offers and #179 asked for: an alpha threshold,
/// and a colour filter placed in the *image* chain so it runs after what
/// precedes it rather than on the paint before the draw.
void aThresholdCutsAlphaAndAColorFilterCanFollowAnImageFilter() {
  auto shape = [](const std::vector<double>& colorFilter,
                  const std::vector<double>& imageFilter, uint32_t fill) {
    Writer w;
    w.push(1);
    w.push(1);  // Rect
    w.pushAnim(12);
    w.pushAnim(12);
    w.pushAnim(40);
    w.pushAnim(40);
    w.pushFilteredPaint(fill, {3, colorFilter, imageFilter});
    return render(w.finish(), {}, {});
  };

  // A half-transparent fill: below the threshold it disappears, above it goes solid.
  const uint32_t half = 0x80336699;
  auto plain = shape({-1}, {-1}, half);
  auto cut = shape({3, 0.9}, {-1}, half);    // 0.5 alpha < 0.9 => transparent
  auto kept = shape({3, 0.25}, {-1}, half);  // 0.5 alpha >= 0.25 => opaque
  CHECK(!opaqueAt(cut, 32, 32));
  CHECK(kept != plain);
  // The colour, not just "something is there". This filter shipped once before
  // and was reverted in 0f0b627 because tgfx's shader never re-premultiplied,
  // which comes back as {0x1a,0x33,0x4d} — every alpha-only check passes on it.
  CHECK(alphaAt(kept, 32, 32) == 255);
  CHECK(pixelAt(kept, 32, 32) == (std::array<uint8_t, 3>{0x33, 0x66, 0x99}));

  // The threshold is an anim word, so a slot drives it without a re-encode.
  // Written out rather than through pushFilteredPaint, which has no slot for it.
  Writer live;
  live.push(1);
  live.push(1);  // Rect
  live.pushAnim(12);
  live.pushAnim(12);
  live.pushAnim(40);
  live.pushAnim(40);
  live.push(half);
  live.push(2 | 4);  // antiAlias, and a filter block follows
  live.pushAnim(1);
  live.push(0);
  live.push(0);
  live.pushAnim(1);
  live.push(3);           // blend srcOver
  live.push(3);           // colour filter kind: alphaThreshold
  live.pushAnim(0.9, 0);  // threshold, slot 0
  live.push(-1);          // no image filter
  live.push(-1);          // no shader
  auto program = live.finish();
  CHECK(!opaqueAt(render(program, {}, {0.9}), 32, 32));
  CHECK(opaqueAt(render(program, {}, {0.25}), 32, 32));

  // The same program drawn twice from one setScene, which is what a handle
  // write does: the value moves, the scene does not.
  tgfxrn::SceneRenderer patched;
  patched.setScene(program, {});
  CHECK(opaqueAt(drawFrame(patched, {0.25}), 32, 32));
  CHECK(!opaqueAt(drawFrame(patched, {0.9}), 32, 32));

  // Ordering. The filter has to be non-linear for the order to be visible at
  // all — Luma is a weighted sum of channels and a blur is a weighted sum of
  // pixels, so those two commute. A threshold does not.
  const uint32_t solid = 0xff3366cc;
  // On the paint: the rect's edge is already hard, so the threshold changes
  // nothing and the blur then softens it.
  auto thresholdThenBlur = shape({3, 0.6}, {0, 6, 6, 3}, solid);
  // In the image chain: the blur softens, then the threshold cuts the falloff
  // back to a hard edge.
  auto blurThenThreshold = shape({-1}, {2, 0, 6, 6, 3, 3, 3, 0.6}, solid);
  CHECK(anyPixelSet(thresholdThenBlur));
  CHECK(anyPixelSet(blurThenThreshold));
  CHECK(thresholdThenBlur != blurThenThreshold);

  // Which one ran first, stated both ways. An opaque rect's alpha already
  // clears 0.6, so a threshold on the paint is a no-op and the blur alone has
  // to reproduce it exactly; in the image chain the threshold cuts the blur's
  // falloff back, so it cannot.
  auto blurred = shape({-1}, {0, 6, 6, 3}, solid);
  CHECK(thresholdThenBlur == blurred);
  CHECK(blurThenThreshold != blurred);
  // And the colour survives the image chain, the same regression as above.
  CHECK(pixelAt(blurThenThreshold, 32, 32) == (std::array<uint8_t, 3>{0x33, 0x66, 0xcc}));
}

/// <Blend> folds shader children; <ColorShader> is the leaf that makes it testable.
void blendFoldsItsShaderChildren() {
  auto shape = [](const std::vector<double>& shader) {
    Writer w;
    w.push(1);
    w.push(1);  // Rect
    w.pushAnim(0);
    w.pushAnim(0);
    w.pushAnim(64);
    w.pushAnim(64);
    w.push(0xffffffff);
    w.push(2);
    w.pushAnim(1);
    w.push(0);
    w.push(0);
    w.pushAnim(1);
    for (double word : shader) {
      w.push(word);
    }
    return render(w.finish(), {}, {});
  };

  // A lone colour shader paints its colour.
  auto red = shape({9, 0xffff0000});
  CHECK(pixelAt(red, 32, 32)[0] > 200);
  CHECK(pixelAt(red, 32, 32)[1] < 40);

  // Multiply of red over a half-grey: red survives, green and blue are cut.
  const int multiply = 24;
  auto folded = shape({8, multiply, 2, 9, 0xffff0000, 9, 0xff808080});
  auto pixel = pixelAt(folded, 32, 32);
  CHECK(pixel[0] > 100 && pixel[0] < 200);
  CHECK(pixel[1] < 40);

  // The mode matters, and a single child folds to just that child.
  CHECK(shape({8, 14, 2, 9, 0xffff0000, 9, 0xff808080}) != folded);  // screen
  CHECK(shape({8, multiply, 1, 9, 0xffff0000}) == red);
}

/// Two filters of a class compose rather than the last one winning.
void filtersOfOneClassCompose() {
  auto shape = [](const std::vector<double>& colorFilter) {
    Writer w;
    w.push(1);
    w.push(1);  // Rect
    w.pushAnim(12);
    w.pushAnim(12);
    w.pushAnim(40);
    w.pushAnim(40);
    w.pushFilteredPaint(0xff3366cc, {3, colorFilter, {-1}});
    return render(w.finish(), {}, {});
  };
  // Greyscale, then swap R and B: composing is not the same as either alone.
  const std::vector<double> grey = {
      0, 0.21, 0.72, 0.07, 0, 0, 0.21, 0.72, 0.07, 0, 0,
      0.21, 0.72, 0.07, 0, 0, 0, 0, 0, 1, 0};
  const std::vector<double> swap = {
      0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0};

  auto greyOnly = shape(grey);
  auto swapOnly = shape(swap);
  std::vector<double> composed = {4};
  composed.insert(composed.end(), grey.begin(), grey.end());
  composed.insert(composed.end(), swap.begin(), swap.end());
  auto both = shape(composed);

  CHECK(greyOnly != swapOnly);
  CHECK(both != swapOnly);  // the greyscale really did run first
  // Grey then swap is still grey, so the composed result matches the grey one.
  CHECK(both == greyOnly);

  // Two blurs compose into one image filter rather than the last one winning.
  auto blurred = [](const std::vector<double>& imageFilter) {
    Writer w;
    w.push(1);
    w.push(1);
    w.pushAnim(20);
    w.pushAnim(20);
    w.pushAnim(24);
    w.pushAnim(24);
    w.pushFilteredPaint(0xff3366cc, {3, {-1}, imageFilter});
    return render(w.finish(), {}, {});
  };
  auto once = blurred({0, 2, 2, 3});
  auto twice = blurred({2, 0, 2, 2, 3, 0, 2, 2, 3});
  CHECK(anyPixelSet(twice));
  CHECK(twice != once);
  CHECK(inkPixels(twice) > inkPixels(once));  // blurring twice spreads further
}

/// A <Blend> child that resolves per frame must keep doing so inside the fold.
void blendKeepsItsDynamicChildrenLive() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  auto first = registerSolid(cellColor(13), 32, 32);
  auto second = registerSolid(cellColor(31), 32, 32);
  auto external = registry.addExternal(registry.acquire(first.handle));
  CHECK(external.handle != 0);

  // mode `dst` keeps the first child only, so the image is what lands on screen.
  const int dst = 2;
  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(32);
  w.pushAnim(32);
  w.push(0xffffffff);
  w.push(2);
  w.pushAnim(1);
  w.push(0);
  w.push(0);
  w.pushAnim(1);
  w.push(8);  // Blend
  w.push(dst);
  w.push(2);
  w.push(7);  // an external image shader
  w.push(external.handle);
  w.push(0);
  w.push(0);
  w.push(1);
  w.push(0);
  w.push(9);  // and a colour, which `dst` discards
  w.push(0xff00ff00);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});

  auto expectedFirst = cellColor(13);
  auto before = pixelAt(drawFrame(renderer, {}), 16, 16);
  CHECK(before[0] == expectedFirst[0] && before[1] == expectedFirst[1]);

  // Swapping the buffer behind the handle must reach the next frame, with no re-commit.
  CHECK(registry.replaceExternal(external.handle, registry.acquire(second.handle)));
  auto expectedSecond = cellColor(31);
  auto after = pixelAt(drawFrame(renderer, {}), 16, 16);
  CHECK(after[0] == expectedSecond[0] && after[1] == expectedSecond[1]);
}

/// An animated runtime uniform has to survive being wrapped in a <Blend>.
void blendKeepsAnAnimatedUniformLive() {
  const std::vector<std::string> strings = {
      "return vec4(uLevel, 0.0, 0.0, 1.0);", "", "uLevel"};
  const int dst = 2;
  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(32);
  w.pushAnim(32);
  w.push(0xffffffff);
  w.push(2);
  w.pushAnim(1);
  w.push(0);
  w.push(0);
  w.pushAnim(1);
  w.push(8);  // Blend
  w.push(dst);
  w.push(2);
  w.push(6);  // runtime shader, one animated uniform
  w.push(0);
  w.push(1);
  w.push(0);
  w.push(1);
  w.push(2);
  w.push(1);
  w.pushAnim(0, 0);
  w.push(0);
  w.push(9);  // a colour `dst` discards
  w.push(0xff00ff00);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), strings);
  CHECK(pixelAt(drawFrame(renderer, {0.0}), 16, 16)[0] < 40);
  CHECK(pixelAt(drawFrame(renderer, {1.0}), 16, 16)[0] > 200);
}

/// A mask keeps the content only where its own drawing is opaque (or bright).
void aMaskCutsItsContentDown() {
  auto scene = [](int luminance, uint32_t maskColor) {
    Writer w;
    w.push(1);
    w.push(14);  // Mask
    w.push(luminance);
    w.push(1);  // one mask node: a rect over the left half
    w.push(1);
    w.pushAnim(0);
    w.pushAnim(0);
    w.pushAnim(32);
    w.pushAnim(64);
    w.pushPaint(maskColor, false);
    w.push(1);  // one content node: a rect over the whole canvas
    w.push(1);
    w.pushAnim(0);
    w.pushAnim(0);
    w.pushAnim(64);
    w.pushAnim(64);
    w.pushPaint(0xff3366cc, false);
    return render(w.finish(), {}, {});
  };

  auto alpha = scene(0, 0xffffffff);
  CHECK(opaqueAt(alpha, 16, 32));   // under the mask, the content survives
  CHECK(!opaqueAt(alpha, 48, 32));  // outside it, nothing is left

  // A transparent mask keeps nothing, whatever its colour.
  CHECK(!anyPixelSet(scene(0, 0x00ffffff)));

  // An empty side means nothing survives: a one-record layer would be unrolled, not composited.
  auto emptySide = [](int maskCount, int contentCount) {
    Writer w;
    w.push(1);
    w.push(14);
    w.push(0);
    w.push(maskCount);
    for (int i = 0; i < maskCount; i++) {
      w.push(1);
      w.pushAnim(0);
      w.pushAnim(0);
      w.pushAnim(32);
      w.pushAnim(64);
      w.pushPaint(0xffffffff, false);
    }
    w.push(contentCount);
    for (int i = 0; i < contentCount; i++) {
      w.push(1);
      w.pushAnim(0);
      w.pushAnim(0);
      w.pushAnim(64);
      w.pushAnim(64);
      w.pushPaint(0xff3366cc, false);
    }
    return render(w.finish(), {}, {});
  };
  CHECK(!anyPixelSet(emptySide(0, 1)));  // no mask hides everything
  CHECK(!anyPixelSet(emptySide(1, 0)));  // no content paints nothing, least of all the mask
  CHECK(!anyPixelSet(emptySide(0, 0)));

  // Luminance mode reads brightness, so black masks out where white masked in.
  auto whiteLuma = scene(1, 0xffffffff);
  CHECK(opaqueAt(whiteLuma, 16, 32));
  CHECK(!opaqueAt(scene(1, 0xff000000), 16, 32));

  // A mask cuts its own content and nothing else: a sibling drawn before it stands.
  Writer w;
  w.push(2);
  w.push(1);  // a rect over the right half, drawn under the mask node
  w.pushAnim(32);
  w.pushAnim(0);
  w.pushAnim(32);
  w.pushAnim(64);
  w.pushPaint(0xff112233, false);
  w.push(14);  // Mask
  w.push(0);
  w.push(1);
  w.push(1);
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(32);
  w.pushAnim(64);
  w.pushPaint(0xffffffff, false);
  w.push(1);
  w.push(1);
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(64);
  w.pushAnim(64);
  w.pushPaint(0xff3366cc, false);
  CHECK(opaqueAt(render(w.finish(), {}, {}), 48, 32));
}

void fillCoversTheWholeCanvas() {
  Writer fill;
  fill.push(1);
  fill.push(9);  // Fill: no geometry words
  fill.pushPaint(0xff3366cc, false);

  Writer rect;
  rect.push(1);
  rect.push(1);  // Rect
  rect.pushAnim(0);
  rect.pushAnim(0);
  rect.pushAnim(kWidth);
  rect.pushAnim(kHeight);
  rect.pushPaint(0xff3366cc, false);

  auto filled = render(fill.finish(), {}, {});
  CHECK(anyPixelSet(filled));
  CHECK(filled == render(rect.finish(), {}, {}));

  Writer half;
  half.push(1);
  half.push(1);
  half.pushAnim(0);
  half.pushAnim(0);
  half.pushAnim(kWidth);
  half.pushAnim(kHeight / 2);
  half.pushPaint(0xff3366cc, false);
  CHECK(filled != render(half.finish(), {}, {}));

  const std::vector<std::string> strings = {
      "return vec4(uLevel, 0.0, 0.0, 1.0);", "uLevel", ""};
  const std::vector<Writer::Uniform> uniforms = {{1, {{0, 0}}}};

  Writer shaded;
  shaded.push(1);
  shaded.push(9);
  shaded.pushRuntimePaint(0xff000000, uniforms, 2);
  auto shadedWords = shaded.finish();
  auto shadedPixels = render(shadedWords, strings, {0.75});
  CHECK(anyPixelSet(shadedPixels));
  CHECK(shadedPixels != filled);
  CHECK(shadedPixels != render(shadedWords, strings, {0.25}));

  Writer faded;
  faded.push(1);
  faded.push(9);
  faded.pushPaint(0xff3366cc, false, 1, 0, 0);
  CHECK(render(faded.finish(), {}, {0.5}) != filled);
}

void everyImageDrawsItsOwnHandle() {
  constexpr int kCount = 40;
  constexpr int kCell = 8;
  Writer w;
  w.push(kCount);
  for (int i = 0; i < kCount; i++) {
    auto image = registerSolid(cellColor(i));
    CHECK(image.handle != 0 && image.width == 2 && image.height == 2);
    w.pushImage(image.handle, (i % kCell) * kCell, (i / kCell) * kCell, kCell,
                kCell);
  }
  auto pixels = render(w.finish(), {}, {});
  for (int i = 0; i < kCount; i++) {
    auto color = cellColor(i);
    auto got = pixelAt(pixels, (i % kCell) * kCell + kCell / 2,
                       (i / kCell) * kCell + kCell / 2);
    CHECK(got[0] == color[0] && got[1] == color[1] && got[2] == color[2]);
  }
}

void reCommittingAnImageSceneDecodesNothing() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  auto image = registerSolid(cellColor(3));
  size_t before = registry.decodeCount();

  tgfxrn::SceneRenderer renderer;
  for (int i = 0; i < 5; i++) {
    Writer w;
    w.push(1);
    w.pushImage(image.handle, i, 0, 8, 8);
    renderer.setScene(w.finish(), {});
  }
  CHECK(registry.decodeCount() == before);
}

/// Bytes with a real LRU, not a count with FIFO. Invisible in the pixels, so it is counted.
void theByteBudgetEvictsLeastRecentlyUsed() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  size_t restore = registry.cacheLimit();
  auto a = registerSolid(cellColor(1), 32, 32);
  auto b = registerSolid(cellColor(2), 32, 32);
  auto c = registerSolid(cellColor(3), 32, 32);
  auto d = registerSolid(cellColor(4), 32, 32);
  CHECK(d.handle != 0);
  registry.setCacheLimit(3 * 32 * 32 * 4);
  CHECK(registry.cachedBytes() <= 3 * 32 * 32 * 4);
  size_t decodes = registry.decodeCount();

  // acquire() runs on the render thread, so an evicted entry comes back empty, never decoding.
  CHECK(registry.acquire(a.handle) == nullptr);
  CHECK(registry.decodeCount() == decodes);
  registry.waitForDecodes();
  CHECK(registry.decodeCount() == decodes + 1);
  CHECK(registry.acquire(a.handle) != nullptr);
  CHECK(registry.acquire(b.handle) == nullptr);

  CHECK(registry.acquire(c.handle) != nullptr);
  registry.waitForDecodes();
  CHECK(registry.decodeCount() == decodes + 2);
  CHECK(registry.acquire(c.handle) != nullptr);
  CHECK(registry.acquire(d.handle) == nullptr);

  registry.waitForDecodes();
  registry.setCacheLimit(restore);
}

/// A frame someone else recycles is no basis for a derived image.
void aLiveHandleRefusesToBeDerivedFrom() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  auto still = registerSolid({255, 0, 0, 255}, 8, 8);
  CHECK(registry.sourceImage(still.handle) != nullptr);

  auto frame = registry.addExternal(registry.sourceImage(still.handle), 0, true);
  CHECK(frame.handle != 0);
  CHECK(registry.sourceImage(frame.handle) == nullptr);
}

void fitModesLetterboxAndCrop() {
  auto image = registerSolid(cellColor(7), 4, 2);
  auto draw = [&](int fit) {
    Writer w;
    w.push(1);
    w.pushImage(image.handle, 0, 0, 32, 32, fit);
    return render(w.finish(), {}, {});
  };

  auto contain = draw(1);
  CHECK(opaqueAt(contain, 16, 16));  // the letterboxed band
  CHECK(!opaqueAt(contain, 16, 2));  // above it, untouched
  CHECK(!opaqueAt(contain, 16, 30));

  auto cover = draw(2);
  CHECK(opaqueAt(cover, 16, 2));  // fills the box
  CHECK(opaqueAt(cover, 16, 30));
  CHECK(!opaqueAt(cover, 40, 16));  // but never past it

  auto fill = draw(0);
  CHECK(opaqueAt(fill, 1, 1) && opaqueAt(fill, 30, 30));

  auto none = draw(5);
  CHECK(opaqueAt(none, 16, 16));
  CHECK(!opaqueAt(none, 16, 4));
}

void aSourceRectPicksOneSprite() {
  auto left = cellColor(1);
  auto right = cellColor(9);
  std::vector<uint8_t> pixels(8);
  std::copy(left.begin(), left.end(), pixels.begin());
  std::copy(right.begin(), right.end(), pixels.begin() + 4);
  auto image = registerPixels(2, 1, pixels);
  CHECK(image.handle != 0);

  const double rightHalf[4] = {1, 0, 1, 1};
  Writer w;
  w.push(1);
  w.pushImage(image.handle, 0, 0, 32, 32, 0, 0, rightHalf);
  auto got = pixelAt(render(w.finish(), {}, {}), 16, 16);
  CHECK(got[0] == right[0] && got[1] == right[1] && got[2] == right[2]);
}

void anImageShaderTilesAcrossAShape() {
  auto left = cellColor(2);
  auto right = cellColor(20);
  std::vector<uint8_t> pixels(8);
  std::copy(left.begin(), left.end(), pixels.begin());
  std::copy(right.begin(), right.end(), pixels.begin() + 4);
  auto image = registerPixels(2, 1, pixels);

  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(32);
  w.pushAnim(32);
  w.pushImageShaderPaint(image.handle, 1, 1, 0);  // repeat/repeat, nearest
  auto drawn = render(w.finish(), {}, {});

  auto at0 = pixelAt(drawn, 0, 0);
  auto at1 = pixelAt(drawn, 1, 0);
  auto at2 = pixelAt(drawn, 2, 0);
  CHECK(at0[0] == left[0] && at0[1] == left[1]);
  CHECK(at1[0] == right[0] && at1[1] == right[1]);
  CHECK(at2[0] == at0[0] && at2[1] == at0[1]);
  Writer m;
  m.push(1);
  m.push(1);
  m.pushAnim(0);
  m.pushAnim(0);
  m.pushAnim(32);
  m.pushAnim(32);
  m.pushImageShaderPaint(image.handle, 2, 1, 0);
  auto mirrored = render(m.finish(), {}, {});
  auto m2 = pixelAt(mirrored, 2, 0);
  CHECK(m2[0] == right[0] && m2[1] == right[1]);
}

void anExternalHandleIsReplacedInPlaceAndNotBudgeted() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  auto first = registerSolid(cellColor(11), 32, 32);
  auto second = registerSolid(cellColor(29), 32, 32);
  auto frameA = registry.acquire(first.handle);
  auto frameB = registry.acquire(second.handle);
  CHECK(frameA != nullptr && frameB != nullptr);

  size_t bytesBefore = registry.cachedBytes();
  auto external = registry.addExternal(frameA);
  CHECK(external.handle != 0 && external.width == 32);
  CHECK(registry.cachedBytes() == bytesBefore);

  Writer w;
  w.push(1);
  w.pushImage(external.handle, 0, 0, 32, 32);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});
  uint64_t epochBefore = registry.epoch();

  auto expectedFirst = cellColor(11);
  auto before = pixelAt(drawFrame(renderer, {}), 16, 16);
  CHECK(before[0] == expectedFirst[0] && before[1] == expectedFirst[1]);

  CHECK(registry.replaceExternal(external.handle, frameB));
  CHECK(registry.epoch() != epochBefore);
  // ...and the previous buffer is released there and then. Two refs left: local and entry.
  CHECK(frameA.use_count() == 2);

  auto expectedSecond = cellColor(29);
  auto after = pixelAt(drawFrame(renderer, {}), 16, 16);
  CHECK(after[0] == expectedSecond[0] && after[1] == expectedSecond[1]);

  CHECK(!registry.replaceExternal(first.handle, frameA));
  registry.remove(external.handle);
}

void aStreamHandleRefusesReplaceExternalAndBytesFollowTheSwap() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  auto source = registerSolid(cellColor(11), 32, 32);
  auto frame = registry.acquire(source.handle);
  CHECK(frame != nullptr);

  // A stream's frames come from produce(); a swapped-in image would never be drawn.
  auto stream = registry.addStream([frame]() { return frame; }, 32, 32);
  CHECK(stream.handle != 0);
  CHECK(!registry.replaceExternal(stream.handle, frame));

  auto external = registry.addExternal(frame, 1024, true);
  CHECK(registry.byteSize(external.handle) == 1024);
  CHECK(registry.replaceExternal(external.handle, frame, 4096));
  CHECK(registry.byteSize(external.handle) == 4096);

  registry.remove(stream.handle);
  registry.remove(external.handle);
  registry.remove(source.handle);
}

void anExternalHandleAlsoDrivesAnImageShaderPerFrame() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  auto first = registerSolid(cellColor(13), 32, 32);
  auto second = registerSolid(cellColor(31), 32, 32);
  auto external = registry.addExternal(registry.acquire(first.handle));
  CHECK(external.handle != 0);

  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(32);
  w.pushAnim(32);
  w.pushImageShaderPaint(external.handle, 0, 0);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});

  auto expectedFirst = cellColor(13);
  auto before = pixelAt(drawFrame(renderer, {}), 16, 16);
  CHECK(before[0] == expectedFirst[0] && before[1] == expectedFirst[1]);

  CHECK(registry.replaceExternal(external.handle, registry.acquire(second.handle)));
  auto expectedSecond = cellColor(31);
  auto after = pixelAt(drawFrame(renderer, {}), 16, 16);
  CHECK(after[0] == expectedSecond[0] && after[1] == expectedSecond[1]);
  registry.remove(external.handle);
}

void anExternalHandleDrivesAChildShaderPerFrame() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  auto first = registerSolid(cellColor(11), 32, 32);
  auto second = registerSolid(cellColor(27), 32, 32);
  auto external = registry.addExternal(registry.acquire(first.handle));
  CHECK(external.handle != 0);

  const std::vector<std::string> strings = {"return TGFX_CHILD_0(fragCoord);", ""};
  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(32);
  w.pushAnim(32);
  w.pushRuntimePaint(0xffffffff, {}, 1, {},
                     {{7, static_cast<double>(external.handle), 0, 0, 0, 0}});
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), strings);

  auto expectedFirst = cellColor(11);
  auto before = pixelAt(drawFrame(renderer, {}), 16, 16);
  CHECK(before[0] == expectedFirst[0] && before[1] == expectedFirst[1]);

  CHECK(registry.replaceExternal(external.handle, registry.acquire(second.handle)));
  auto expectedSecond = cellColor(27);
  auto after = pixelAt(drawFrame(renderer, {}), 16, 16);
  CHECK(after[0] == expectedSecond[0] && after[1] == expectedSecond[1]);
  registry.remove(external.handle);
}

/// The frame and its children share one matrix, so an overlay in frame pixels cannot drift.
void aCameraFrameCoversTheCanvasAndItsChildrenRideItsTransform() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  auto frame = registerSolid(cellColor(7), 32, 32);
  auto next = registerSolid(cellColor(23), 32, 32);
  auto external = registry.addExternal(registry.acquire(frame.handle));
  CHECK(external.handle != 0);

  Writer w;
  w.push(1);
  w.pushCameraFrame(external.handle, 1);
  w.push(1);  // Rect, in frame pixels
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(16);
  w.pushAnim(16);
  w.pushPaint(0xff0000ff, false);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});

  auto first = cellColor(7);
  auto pixels = drawFrame(renderer, {});
  auto overlay = pixelAt(pixels, 8, 8);
  CHECK(overlay[0] == 0 && overlay[2] == 255);
  CHECK(!(pixelAt(pixels, 40, 8)[2] == 255));
  auto uncovered = pixelAt(pixels, 48, 48);
  CHECK(uncovered[0] == first[0] && uncovered[1] == first[1]);

  // A new buffer behind the same handle, no re-commit: the overlay must not move.
  CHECK(registry.replaceExternal(external.handle, registry.acquire(next.handle)));
  auto second = cellColor(23);
  pixels = drawFrame(renderer, {});
  auto swapped = pixelAt(pixels, 48, 48);
  CHECK(swapped[0] == second[0] && swapped[1] == second[1]);
  auto stillThere = pixelAt(pixels, 8, 8);
  CHECK(stillThere[0] == 0 && stillThere[2] == 255);

  registry.remove(external.handle);
}

/// Which canvas corner a child at the frame's origin lands in, as {TL, TR, BL, BR}.
std::array<bool, 4> cameraFrameChildCorner(int32_t handle, int frameW, int frameH,
                                           int quarterTurns, bool mirror) {
  Writer w;
  w.push(1);
  w.pushCameraFrame(handle, 1, 2, quarterTurns, mirror);
  w.push(1);  // Rect, in frame pixels
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(frameW / 2);
  w.pushAnim(frameH / 2);
  w.pushPaint(0xff0000ff, false);
  auto pixels = render(w.finish(), {}, {});
  return {pixelAt(pixels, 8, 8)[2] == 255, pixelAt(pixels, 56, 8)[2] == 255,
          pixelAt(pixels, 8, 56)[2] == 255, pixelAt(pixels, 56, 56)[2] == 255};
}

constexpr std::array<bool, 4> kTopLeft{true, false, false, false};
constexpr std::array<bool, 4> kTopRight{false, true, false, false};
constexpr std::array<bool, 4> kBottomLeft{false, false, true, false};
constexpr std::array<bool, 4> kBottomRight{false, false, false, true};

void aCameraFrameMirrorsAndRotatesItsChildrenWithTheFrame() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  auto frame = registerSolid(cellColor(7), 32, 32);
  auto external = registry.addExternal(registry.acquire(frame.handle));

  auto quadrant = [&](int quarterTurns, bool mirror) {
    return cameraFrameChildCorner(external.handle, 32, 32, quarterTurns, mirror);
  };

  CHECK(quadrant(0, false) == kTopLeft);
  CHECK(quadrant(0, true) == kTopRight);
  CHECK(quadrant(1, false) == kTopRight);
  CHECK(quadrant(2, false) == kBottomRight);
  CHECK(quadrant(3, false) == kBottomLeft);
  registry.remove(external.handle);
}

/// Four turns on a non-square frame: with w == h the translate and the ow/oh swap both collapse.
void aCameraFrameRotatesANonSquareFrame() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  auto frame = registerSolid(cellColor(7), 32, 16);
  auto external = registry.addExternal(registry.acquire(frame.handle));

  auto quadrant = [&](int quarterTurns) {
    return cameraFrameChildCorner(external.handle, 32, 16, quarterTurns, false);
  };

  CHECK(quadrant(0) == kTopLeft);
  CHECK(quadrant(1) == kTopRight);
  CHECK(quadrant(2) == kBottomRight);
  CHECK(quadrant(3) == kBottomLeft);
  // Mirroring flips about the frame's own axis: the wrong width puts the child off-canvas.
  CHECK(cameraFrameChildCorner(external.handle, 32, 16, 0, true) == kTopRight);
  registry.remove(external.handle);
}

void aCameraFrameCoverCropsTheOverflowingAxis() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  auto wide = registerSolid(cellColor(7), 32, 16);
  auto external = registry.addExternal(registry.acquire(wide.handle));

  Writer w;
  w.push(1);
  w.pushCameraFrame(external.handle, 0);
  auto pixels = render(w.finish(), {}, {});
  auto expected = cellColor(7);
  for (auto [x, y] : {std::pair{1, 1}, {62, 1}, {1, 62}, {62, 62}}) {
    auto pixel = pixelAt(pixels, x, y);
    CHECK(pixel[0] == expected[0] && pixel[1] == expected[1]);
  }
  registry.remove(external.handle);
}

void aCameraFrameWithNoFrameDrawsNothing() {
  Writer w;
  w.push(1);
  w.pushCameraFrame(0, 1);
  w.push(1);  // Rect
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(16);
  w.pushAnim(16);
  w.pushPaint(0xff0000ff, false);
  auto pixels = render(w.finish(), {}, {});
  CHECK(!opaqueAt(pixels, 8, 8));
}

/// An asset evicted before the scene compiled must still appear when its re-decode lands.
void anImageEvictedBeforeCompileComesBackWhenItsDecodeLands() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  size_t restore = registry.cacheLimit();
  auto victim = registerSolid(cellColor(17), 32, 32);
  auto keep = registerSolid(cellColor(23), 32, 32);
  CHECK(keep.handle != 0);
  registry.setCacheLimit(32 * 32 * 4);
  size_t decodes = registry.decodeCount();

  Writer w;
  w.push(1);
  w.pushImage(victim.handle, 0, 0, 32, 32);
  tgfxrn::SceneRenderer renderer;
  // Nothing has acquired `victim` since the eviction, so this compile schedules the re-decode.
  renderer.setScene(w.finish(), {});

  registry.setCacheLimit(restore);
  registry.waitForDecodes();
  CHECK(registry.decodeCount() == decodes + 1);
  CHECK(registry.acquire(victim.handle) != nullptr);

  auto expected = cellColor(17);
  auto after = pixelAt(drawFrame(renderer, {}), 16, 16);
  CHECK(after[0] == expected[0] && after[1] == expected[1]);
}

/// Both render loops are push-driven, so a swap that only moves the epoch schedules no frame.
void aBufferSwapSchedulesADraw() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  int wakes = 0;
  auto token = registry.addWakeListener([&wakes] { wakes++; });

  auto first = registerSolid(cellColor(11), 32, 32);
  auto second = registerSolid(cellColor(29), 32, 32);
  auto external = registry.addExternal(registry.acquire(first.handle));
  int before = wakes;
  CHECK(registry.replaceExternal(external.handle, registry.acquire(second.handle)));
  CHECK(wakes == before + 1);

  registry.remove(external.handle);
  CHECK(wakes == before + 2);
  registry.removeWakeListener(token);
  registry.replaceExternal(external.handle, registry.acquire(first.handle));
  CHECK(wakes == before + 2);
}

void aBlendModeCompositesAgainstWhatIsBelow() {
  auto scene = [](int blend) {
    Writer w;
    w.push(2);
    w.push(1);  // Rect: the backdrop
    w.pushAnim(0);
    w.pushAnim(0);
    w.pushAnim(32);
    w.pushAnim(32);
    w.pushPaint(0xffffff00, false);
    w.push(1);  // Rect: drawn over it
    w.pushAnim(0);
    w.pushAnim(0);
    w.pushAnim(32);
    w.pushAnim(32);
    if (blend < 0) {
      w.pushPaint(0xffff00ff, false);
    } else {
      w.pushFilteredPaint(0xffff00ff, {blend, {-1}, {-1}});
    }
    return render(w.finish(), {}, {});
  };

  auto over = scene(-1);
  auto multiply = scene(24);  // BlendMode::Multiply
  CHECK(anyPixelSet(over));
  auto plain = pixelAt(over, 16, 16);
  CHECK(plain[0] == 255 && plain[1] == 0 && plain[2] == 255);
  auto blended = pixelAt(multiply, 16, 16);
  CHECK(blended[0] == 255 && blended[1] == 0 && blended[2] == 0);

  Writer srcOver;
  srcOver.push(2);
  srcOver.push(1);
  srcOver.pushAnim(0);
  srcOver.pushAnim(0);
  srcOver.pushAnim(32);
  srcOver.pushAnim(32);
  srcOver.pushPaint(0xffffff00, false);
  srcOver.push(1);
  srcOver.pushAnim(0);
  srcOver.pushAnim(0);
  srcOver.pushAnim(32);
  srcOver.pushAnim(32);
  srcOver.pushFilteredPaint(0xffff00ff, {3, {-1}, {-1}});
  CHECK(render(srcOver.finish(), {}, {}) == over);
}

/// A blend mode left on the inner draw composites against the empty layer an image filter forces.
void aBlendModeSurvivesAnImageFilter() {
  auto scene = [](double sigma) {
    Writer w;
    w.push(3);
    w.push(1);  // Rect: the backdrop
    w.pushAnim(0);
    w.pushAnim(0);
    w.pushAnim(32);
    w.pushAnim(32);
    w.pushPaint(0xffffff00, false);
    w.push(1);  // Rect: multiplied over it
    w.pushAnim(0);
    w.pushAnim(0);
    w.pushAnim(32);
    w.pushAnim(32);
    std::vector<double> blur{-1};
    if (sigma > 0) {
      blur = {0, sigma, sigma, 3};
    }
    w.pushFilteredPaint(0xffff00ff, {24, {-1}, blur});  // BlendMode::Multiply
    w.push(1);                                          // Rect: an ordinary green one, drawn after the layered node
    w.pushAnim(44);
    w.pushAnim(44);
    w.pushAnim(16);
    w.pushAnim(16);
    w.pushPaint(0xff00ff00, false);
    return render(w.finish(), {}, {});
  };

  auto sharp = pixelAt(scene(0), 16, 16);
  CHECK(sharp[0] == 255 && sharp[1] == 0 && sharp[2] == 0);
  auto blurred = pixelAt(scene(0.5), 16, 16);
  CHECK(blurred[0] == 255 && blurred[1] == 0 && blurred[2] == 0);
  CHECK(opaqueAt(scene(6), 34, 16));
  auto after = pixelAt(scene(0.5), 52, 52);
  CHECK(after[0] == 0 && after[1] == 255 && after[2] == 0);
}

void aColourMatrixTransformsTheDrawnColour() {
  auto draw = [](const std::vector<double>& matrix) {
    Writer w;
    w.push(1);
    w.push(1);  // Rect
    w.pushAnim(0);
    w.pushAnim(0);
    w.pushAnim(32);
    w.pushAnim(32);
    std::vector<double> filter{0};
    filter.insert(filter.end(), matrix.begin(), matrix.end());
    w.pushFilteredPaint(0xff2050c0, {3, filter, {-1}});
    return render(w.finish(), {}, {});
  };

  Writer plain;
  plain.push(1);
  plain.push(1);
  plain.pushAnim(0);
  plain.pushAnim(0);
  plain.pushAnim(32);
  plain.pushAnim(32);
  plain.pushPaint(0xff2050c0, false);
  auto unfiltered = render(plain.finish(), {}, {});
  CHECK(anyPixelSet(unfiltered));

  const std::vector<double> identity = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                        0, 0, 1, 0, 0, 0, 0, 0, 1, 0};
  CHECK(draw(identity) == unfiltered);

  const std::vector<double> swap = {0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                    1, 0, 0, 0, 0, 0, 0, 0, 1, 0};
  auto swapped = pixelAt(draw(swap), 16, 16);
  auto source = pixelAt(unfiltered, 16, 16);
  CHECK(swapped[0] == source[2] && swapped[2] == source[0]);
  CHECK(swapped[1] == source[1]);
}

void aBlendColourFilterTints() {
  auto draw = [](bool filtered) {
    Writer w;
    w.push(1);
    w.push(1);  // Rect
    w.pushAnim(0);
    w.pushAnim(0);
    w.pushAnim(32);
    w.pushAnim(32);
    if (filtered) {
      w.pushFilteredPaint(0xff2050c0, {3, {1, 0xffff0000, 1}, {-1}});
    } else {
      w.pushPaint(0xff2050c0, false);
    }
    return render(w.finish(), {}, {});
  };
  auto tinted = pixelAt(draw(true), 16, 16);
  CHECK(tinted[0] == 255 && tinted[1] == 0 && tinted[2] == 0);
  CHECK(draw(false) != draw(true));
}

void aBlurSpreadsPastTheGeometry() {
  auto draw = [](double sigma) {
    Writer w;
    w.push(1);
    w.push(1);  // Rect
    w.pushAnim(20);
    w.pushAnim(20);
    w.pushAnim(24);
    w.pushAnim(24);
    if (sigma <= 0) {
      w.pushPaint(0xff3366cc, false);
    } else {
      w.pushFilteredPaint(0xff3366cc, {3, {-1}, {0, sigma, sigma, 3}});
    }
    return render(w.finish(), {}, {});
  };

  auto sharp = draw(0);
  CHECK(anyPixelSet(sharp));
  CHECK(!opaqueAt(sharp, 14, 32));
  auto blurred = draw(6);
  CHECK(opaqueAt(blurred, 14, 32));
  CHECK(opaqueAt(blurred, 32, 32));
}

void aDropShadowPaintsUnderItsContent() {
  auto draw = [](int only) {
    Writer w;
    w.push(1);
    w.push(1);  // Rect
    w.pushAnim(24);
    w.pushAnim(24);
    w.pushAnim(16);
    w.pushAnim(16);
    if (only < 0) {
      w.pushPaint(0xffffffff, false);
    } else {
      w.pushFilteredPaint(0xffffffff,
                          {3,
                           {-1},
                           {1, 8, 8, 0, 0, 0xff000000,
                            static_cast<double>(only)}});
    }
    return render(w.finish(), {}, {});
  };

  auto plain = draw(-1);
  CHECK(!opaqueAt(plain, 44, 44));  // nothing past the rect without a filter

  auto shadowed = draw(0);
  auto shadow = pixelAt(shadowed, 44, 44);
  CHECK(opaqueAt(shadowed, 44, 44));
  CHECK(shadow[0] == 0 && shadow[1] == 0 && shadow[2] == 0);
  auto content = pixelAt(shadowed, 28, 28);
  CHECK(content[0] == 255 && content[1] == 255 && content[2] == 255);

  auto only = draw(1);
  CHECK(opaqueAt(only, 44, 44));
  CHECK(!opaqueAt(only, 28, 28));
}

/// A filter must be built once, by setScene, and never again. Counted, not reasoned about.
void filtersAreBuiltOnceAndNeverOnTheDrawPath() {
  constexpr int kNodes = 16;
  Writer w;
  w.push(kNodes);
  for (int i = 0; i < kNodes; i++) {
    w.push(1);  // Rect
    w.pushAnim(i);
    w.pushAnim(i);
    w.pushAnim(8);
    w.pushAnim(8);
    w.pushFilteredPaint(
        0xff3366cc,
        {24,
         {0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0},
         {0, 2, 2, 3}},
        0);
  }

  size_t before = tgfxrn::filterBuildCount();
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});
  size_t afterCompile = tgfxrn::filterBuildCount();
  CHECK(afterCompile == before + 2 * kNodes);

  for (int frame = 0; frame < 10; frame++) {
    CHECK(anyPixelSet(drawFrame(renderer, {1})));
  }
  CHECK(tgfxrn::filterBuildCount() == afterCompile);
}

/// Composition made a chain rebuildable as a unit; only the parts with a slot
/// should actually be rebuilt. Counted, not reasoned about.
void onlyTheAnimatedPartOfAFilterChainIsRebuilt() {
  // <Blur blur={slot}/><Luma/>: an animated blur composed with a static colour
  // filter, so the chain is compose(blur, colorFilter(luma)) — four nodes, two
  // of which can never change.
  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(12);
  w.pushAnim(12);
  w.pushAnim(40);
  w.pushAnim(40);
  w.push(0xff3366cc);
  w.push(2 | 4);  // antiAlias, and a filter block follows
  w.pushAnim(1);  // stroke width
  w.push(0);
  w.push(0);
  w.pushAnim(1);  // alpha
  w.push(3);      // blend srcOver
  w.push(-1);     // no colour filter on the paint
  w.push(2);      // image kind: compose
  w.push(0);      // blur
  w.pushAnim(4, 0);
  w.pushAnim(4, 0);
  w.push(3);   // decal
  w.push(3);   // colorFilter in the image chain
  w.push(2);   // luma
  w.push(-1);  // no shader
  auto words = w.finish();

  tgfxrn::SceneRenderer renderer;
  renderer.setScene(words, {});
  // The first frame builds all four; the luma and its wrapper are static, so
  // every frame after that rebuilds only the blur and the compose over it.
  CHECK(anyPixelSet(drawFrame(renderer, {4})));
  size_t afterFirst = tgfxrn::filterBuildCount();
  for (int frame = 1; frame < 5; frame++) {
    CHECK(anyPixelSet(drawFrame(renderer, {4.0 + frame})));
  }
  CHECK(tgfxrn::filterBuildCount() == afterFirst + 4 * 2);
}

int rightmostInk(const Pixels& pixels) {
  int rightmost = -1;
  for (int y = 0; y < kHeight; y++) {
    for (int x = 0; x < kWidth; x++) {
      if (pixels[static_cast<size_t>((y * kWidth + x) * 4 + 3)] != 0) {
        rightmost = std::max(rightmost, x);
      }
    }
  }
  return rightmost;
}

int inkRows(const Pixels& pixels) {
  int rows = 0;
  for (int y = 0; y < kHeight; y++) {
    for (int x = 0; x < kWidth; x++) {
      if (pixels[static_cast<size_t>((y * kWidth + x) * 4 + 3)] != 0) {
        rows++;
        break;
      }
    }
  }
  return rows;
}

/// The acceptance case: an Android image with none of the three paths the old code hardcoded.
void aFontIsFoundOnAnImageWithNoneOfTheOldHardcodedPaths() {
  auto best = [](const std::vector<std::string>& files, const std::string& family, int weight,
                 bool italic) {
    auto ranked = tgfxrn::rankFontFiles(files, family, weight, italic);
    return ranked.empty() ? -1 : ranked.front();
  };
  const std::vector<std::string> oemImage{
      "/system/fonts/NotoColorEmoji.ttf",
      "/system/fonts/SamsungOne-400.ttf",
      "/system/fonts/SamsungOne-700.ttf",
      "/system/fonts/SamsungSharpSans-Bold.ttf",
  };
  CHECK(best(oemImage, "", 400, false) >= 0);
  CHECK(best(oemImage, "", 400, false) != 0);
  CHECK(best(oemImage, "", 700, false) != best(oemImage, "", 400, false));

  const std::vector<std::string> withFamily{"/system/fonts/Roboto-Regular.ttf",
                                            "/system/fonts/PlayfairDisplay-Regular.ttf"};
  CHECK(best(withFamily, "Playfair Display", 400, false) == 1);
  CHECK(best(withFamily, "playfair-display", 400, false) == 1);
  CHECK(tgfxrn::rankFontFiles({"/system/fonts/NotoColorEmoji.ttf"}, "", 400, false).empty());

  // Every candidate, best first: NotoSansCham-Bold outranks and has no Latin, so walking matters.
  const std::vector<std::string> samsung{
      "/system/fonts/DroidSans-Bold.ttf", "/system/fonts/NotoColorEmoji.ttf",
      "/system/fonts/NotoSansCham-Bold.ttf", "/system/fonts/Roboto-Regular.ttf"};
  auto ranked = tgfxrn::rankFontFiles(samsung, "", 700, false);
  CHECK(ranked.size() >= 3);
  CHECK(std::find(ranked.begin(), ranked.end(), 0) != ranked.end());
  CHECK(std::find(ranked.begin(), ranked.end(), 1) == ranked.end());

  // A named family must not promote a special-purpose face: "Noto" prefix-matches NotoColorEmoji.
  const std::vector<std::string> noto{"/system/fonts/NotoColorEmoji.ttf",
                                      "/system/fonts/SamsungOne-400.ttf"};
  CHECK(best(noto, "Noto", 400, false) == 1);
  CHECK(tgfxrn::rankFontFiles(noto, "Noto", 400, false).size() == 1);
  CHECK(best(noto, "Noto Color Emoji", 400, false) == 0);
}

/// The path cache is shared with the fallback chain, so a hit must clear the same bar as an open.
void aCachedFaceStillHasToBeAbleToDrawText() {
  auto& registry = tgfxrn::FontRegistry::instance();
  registry.clear();
  auto fallbacks = registry.fallbacks();
  CHECK(!fallbacks.empty());
  for (const auto& typeface : fallbacks) {
    CHECK(typeface->getGlyphID('A') == 0);
  }
  for (const char* family : {"", "Noto", "Apple Color Emoji", "No Such Family"}) {
    auto match = registry.match(family, 400, false);
    CHECK(match.typeface != nullptr);
    if (match.typeface != nullptr) {
      CHECK(match.typeface->getGlyphID('A') != 0);
    }
  }
  registry.clear();
}

/// <Text> with no fontFamily must resolve to something that can draw Latin, which CoreText will not.
void theDefaultFamilyCanActuallyDrawText() {
  auto& registry = tgfxrn::FontRegistry::instance();
  registry.clear();
  auto fallback = registry.match("", 400, false);
  CHECK(fallback.typeface != nullptr);
  if (fallback.typeface != nullptr) {
    CHECK(fallback.typeface->getGlyphID('A') != 0);
    CHECK(fallback.typeface->getGlyphID('0') != 0);
  }
  auto unknown = registry.match("No Such Family At All", 400, false);
  CHECK(unknown.typeface != nullptr && unknown.typeface->getGlyphID('A') != 0);

  Writer w;
  writeText(w, 24, -1, 0);
  CHECK(anyPixelSet(render(w.finish(), {"Wg", "", ""}, {})));
}

/// loadFont() is a bare promise, so registering a face has to schedule the draw itself.
void aFontRegistrationSchedulesADraw() {
  auto& registry = tgfxrn::FontRegistry::instance();
  registry.clear();
  int wakes = 0;
  auto token = tgfxrn::ImageRegistry::instance().addWakeListener([&] { wakes++; });
  uint64_t before = registry.epoch();

  registry.add(tgfx::Typeface::MakeFromPath(std::string(kFontDir) + "/PT_Serif-Web-Regular.ttf"),
               "Wake Serif");
  CHECK(wakes == 1);
  CHECK(registry.epoch() > before);

  registry.add(nullptr, "Wake Serif");
  CHECK(wakes == 1);

  tgfxrn::ImageRegistry::instance().removeWakeListener(token);
  registry.add(tgfx::Typeface::MakeFromPath(std::string(kFontDir) + "/PT_Serif-Web-Bold.ttf"),
               "Wake Serif");
  CHECK(wakes == 1);
  registry.clear();
}

/// The layout is memoised per command, so a late face is only picked up by recompiling.
void recompilingAfterARegistrationPicksUpTheNewFace() {
  auto& registry = tgfxrn::FontRegistry::instance();
  registry.clear();
  const std::vector<std::string> strings{"Wg", "Late Serif", ""};
  Writer w;
  writeText(w, 28, -1, 0);
  auto words = w.finish();

  tgfxrn::SceneRenderer renderer;
  renderer.setScene(words, strings);
  auto fallback = drawFrame(renderer, {});
  CHECK(anyPixelSet(fallback));

  registry.add(tgfx::Typeface::MakeFromPath(std::string(kFontDir) + "/PT_Serif-Web-Bold.ttf"),
               "Late Serif");
  CHECK(drawFrame(renderer, {}) == fallback);
  renderer.setScene(words, strings);
  CHECK(drawFrame(renderer, {}) != fallback);
  registry.clear();
}

void aRegisteredFamilyMatchesByWeightAndStyle() {
  auto& registry = tgfxrn::FontRegistry::instance();
  registry.clear();
  auto regular = tgfx::Typeface::MakeFromPath(std::string(kFontDir) + "/PT_Serif-Web-Regular.ttf");
  auto bold = tgfx::Typeface::MakeFromPath(std::string(kFontDir) + "/PT_Serif-Web-Bold.ttf");
  auto italic = tgfx::Typeface::MakeFromPath(std::string(kFontDir) + "/PT_Serif-Web-Italic.ttf");
  CHECK(regular != nullptr && bold != nullptr && italic != nullptr);
  if (regular == nullptr || bold == nullptr || italic == nullptr) {
    return;
  }
  CHECK(tgfxrn::readFaceStyle(regular).weight == 400);
  CHECK(tgfxrn::readFaceStyle(bold).weight >= 700);
  CHECK(!tgfxrn::readFaceStyle(regular).italic);
  CHECK(tgfxrn::readFaceStyle(italic).italic);

  uint64_t before = registry.epoch();
  CHECK(registry.add(regular, "Demo Serif") == "Demo Serif");
  registry.add(bold, "Demo Serif");
  registry.add(italic, "Demo Serif");
  CHECK(registry.epoch() > before);

  CHECK(registry.match("Demo Serif", 400, false).typeface == regular);
  CHECK(registry.match("Demo Serif", 700, false).typeface == bold);
  CHECK(registry.match("Demo Serif", 500, false).typeface == regular);
  CHECK(registry.match("Demo Serif", 400, true).typeface == italic);
  CHECK(registry.match("demoserif", 700, false).typeface == bold);
  auto black = registry.match("Demo Serif", 900, false);
  CHECK(black.typeface == bold && black.fauxBold);
  CHECK(registry.match("Nothing By This Name", 400, false).typeface != nullptr);
  registry.clear();
}

void aRegisteredFontDrawsAndBoldDiffersFromRegular() {
  auto& registry = tgfxrn::FontRegistry::instance();
  registry.clear();
  registry.add(tgfx::Typeface::MakeFromPath(std::string(kFontDir) + "/PT_Serif-Web-Regular.ttf"),
               "Demo Serif");
  registry.add(tgfx::Typeface::MakeFromPath(std::string(kFontDir) + "/PT_Serif-Web-Bold.ttf"),
               "Demo Serif");
  const std::vector<std::string> strings{"Wg", "Demo Serif", ""};

  Writer regular;
  writeText(regular, 28, -1, 0, 0, 0, 0, 400, 0);
  Writer bold;
  writeText(bold, 28, -1, 0, 0, 0, 0, 700, 0);
  auto regularPixels = render(regular.finish(), strings, {});
  auto boldPixels = render(bold.finish(), strings, {});
  CHECK(anyPixelSet(regularPixels));
  CHECK(regularPixels != boldPixels);

  Writer system;
  writeText(system, 28, -1, 0, 0, 0, 0, 400, 0);
  CHECK(render(system.finish(), {"Wg", "Helvetica", ""}, {}) != regularPixels);
  registry.clear();
}

/// The faux bits force synthesis even where the family has a face good enough to skip it.
void theFauxBitsEmboldenAndShearAMatchedFace() {
  auto& registry = tgfxrn::FontRegistry::instance();
  registry.clear();
  registry.add(tgfx::Typeface::MakeFromPath(std::string(kFontDir) + "/PT_Serif-Web-Regular.ttf"),
               "Demo Serif");
  const std::vector<std::string> strings{"Wg", "Demo Serif", ""};

  auto draw = [&](int style) {
    Writer w;
    writeText(w, 28, -1, 0, 0, 0, 0, 400, style);
    return render(w.finish(), strings, {});
  };
  auto plain = draw(0);
  CHECK(anyPixelSet(plain));
  CHECK(draw(2) != plain);
  CHECK(draw(4) != plain);
  CHECK(draw(2) != draw(4));

  // A run that overrode the axis keeps its own face: 2 forces upright, 1 forces italic.
  auto span = [&](int style, int runItalic) {
    Writer w;
    writeText(w, 28, -1, 0, 0, 0, 0, 400, style, runItalic);
    return render(w.finish(), strings, {});
  };
  CHECK(span(4, 2) == span(0, 2));
  CHECK(span(4, 0) != span(0, 0));
  registry.clear();
}

void aParagraphWrapsAlignsAndStacks() {
  tgfxrn::FontRegistry::instance().clear();
  const std::vector<std::string> strings{"one two three four", "Helvetica", ""};

  Writer unwrapped;
  writeText(unwrapped, 10, -1, 0, 0, 0, 0);
  Writer wrapped;
  writeText(wrapped, 10, -1, 0, 40, 0, 0);
  auto flat = render(unwrapped.finish(), strings, {});
  auto stacked = render(wrapped.finish(), strings, {});
  CHECK(anyPixelSet(flat));
  CHECK(inkRows(stacked) > inkRows(flat));
  CHECK(rightmostInk(stacked) < rightmostInk(flat));

  Writer right;
  writeText(right, 10, -1, 0, 40, 2, 0);
  CHECK(render(right.finish(), strings, {}) != stacked);

  tgfxrn::TextSpec spec;
  spec.family = "Helvetica";
  spec.wrapWidth = 40;
  auto natural = tgfxrn::layoutText({run("one two three four")}, spec, 10);
  spec.lineHeight = 24;
  auto tall = tgfxrn::layoutText({run("one two three four")}, spec, 10);
  CHECK(natural.metrics.lineCount == tall.metrics.lineCount);
  CHECK(tall.metrics.height > natural.metrics.height);
}

void anExplicitNewlineBreaksTheLine() {
  tgfxrn::FontRegistry::instance().clear();
  const std::vector<std::string> strings{"Wg", "Helvetica", "", "Wg\nWg"};
  Writer single;
  writeText(single, 12, -1, 0);
  Writer two;
  writeText(two, 12, -1, 3);
  CHECK(inkRows(render(two.finish(), strings, {})) >
        inkRows(render(single.finish(), strings, {})));
}

/// Register a system face by name, so a script test uses a face that actually
/// covers it rather than whatever the Latin-first matcher settles on. Returns
/// false when the host has no such face, which is a skip and not a failure.
bool registerSystemFamily(const char* family) {
  auto typeface = tgfx::Typeface::MakeFromName(family, "");
  if (typeface == nullptr) {
    return false;
  }
  tgfxrn::FontRegistry::instance().add(typeface, family);
  return true;
}

/// #143: Arabic and Hebrew laid out in logical order with unshaped glyphs. The
/// caret API is the cleanest probe there is — on a right-to-left line the
/// leftmost glyph is the *last* character of the string.
void rightToLeftTextReadsRightToLeft() {
  tgfxrn::FontRegistry::instance().clear();
  if (!registerSystemFamily("Geeza Pro")) {
    std::printf("  skip: no Arabic face on this host\n");
    return;
  }
  tgfxrn::TextSpec spec;
  spec.family = "Geeza Pro";
  const std::string arabic = "مرحبا";
  const auto runs = std::vector<tgfxrn::TextRun>{run(arabic)};
  auto layout = tgfxrn::layoutText(runs, spec, 24);
  CHECK(layout.metrics.width > 0);

  // The left edge of the line is the end of the string and the right edge is
  // the start, which is the whole of what "reads right to left" means.
  auto left = tgfxrn::positionAtCoordinate(runs, spec, 24, 0, 0);
  auto right = tgfxrn::positionAtCoordinate(runs, spec, 24, layout.metrics.width, 0);
  CHECK(left.index > right.index);
  CHECK(left.index == static_cast<int32_t>(arabic.size()));
  CHECK(right.index == 0);

  // Latin in the same paragraph still reads left to right, so this is not just
  // "everything is reversed".
  tgfxrn::TextSpec latin;
  latin.family = "Helvetica";
  const auto plain = std::vector<tgfxrn::TextRun>{run("abcdef")};
  auto plainBox = tgfxrn::layoutText(plain, latin, 24);
  CHECK(tgfxrn::positionAtCoordinate(plain, latin, 24, 0, 0).index == 0);
  CHECK(tgfxrn::positionAtCoordinate(plain, latin, 24, plainBox.metrics.width, 0).index == 6);

  // Hebrew, which is right-to-left but does not join, so it exercises the
  // reordering without the contextual forms below.
  // Shaping collapses codepoints into clusters and splits them into several
  // glyphs, so a cluster owns every byte up to the next one — not just its own
  // first codepoint's. Every caret offset has to be a boundary of the string.
  const std::vector<int32_t> boundaries = {0, 2, 4, 6, 8, 10};
  for (int step = 0; step <= 24; step++) {
    auto here = tgfxrn::positionAtCoordinate(
        runs, spec, 24, layout.metrics.width * static_cast<float>(step) / 24.0f, 0);
    CHECK(std::find(boundaries.begin(), boundaries.end(), here.index) != boundaries.end());
  }
  // And a range over one Arabic letter is that letter, not part of a cluster.
  auto oneLetter = tgfxrn::boxesForRange(runs, spec, 24, 0, 2);
  CHECK(oneLetter.size() == 1);
  CHECK(oneLetter[0].startIndex == 0);
  CHECK(oneLetter[0].endIndex == 2);

  if (registerSystemFamily("Arial Hebrew")) {
    tgfxrn::TextSpec hebrew;
    hebrew.family = "Arial Hebrew";
    const std::string shalom = "שלום";
    const auto hebrewRuns = std::vector<tgfxrn::TextRun>{run(shalom)};
    auto hebrewBox = tgfxrn::layoutText(hebrewRuns, hebrew, 24);
    CHECK(hebrewBox.metrics.width > 0);
    auto hebrewLeft = tgfxrn::positionAtCoordinate(hebrewRuns, hebrew, 24, 0, 0);
    CHECK(hebrewLeft.index == static_cast<int32_t>(shalom.size()));
  }
}

/// The other half of #143: contextual forms. A naive per-codepoint lookup gets
/// the isolated shapes; a shaper gets initial, medial and final ones — which
/// are different glyphs, with different advances.
void arabicTakesItsContextualForms() {
  tgfxrn::FontRegistry::instance().clear();
  // Arial rather than Geeza Pro: Geeza Pro's cmap already returns the glyphs
  // the shaper picks, so it cannot tell a joined form from an isolated one.
  if (!registerSystemFamily("Arial")) {
    std::printf("  skip: no Arabic face on this host\n");
    return;
  }
  tgfxrn::TextSpec spec;
  spec.family = "Arial";
  const std::string arabic = "مرحبا";
  const auto runs = std::vector<tgfxrn::TextRun>{run(arabic)};
  auto layout = tgfxrn::layoutText(runs, spec, 40);
  CHECK(layout.metrics.width > 0);

  // Each letter measured twice: on its own, where it can only be the isolated
  // form, and in the word, where it joins. A joined form is a different glyph,
  // so its advance differs — and one letter differing is enough, because a
  // font may happen to give two forms the same width.
  int differing = 0;
  for (int32_t at = 0; at + 2 <= static_cast<int32_t>(arabic.size()); at += 2) {
    auto inWord = tgfxrn::boxesForRange(runs, spec, 40, at, at + 2);
    const auto alone = std::vector<tgfxrn::TextRun>{run(arabic.substr(
        static_cast<size_t>(at), 2))};
    auto isolated = tgfxrn::boxesForRange(alone, spec, 40, 0, 2);
    CHECK(inWord.size() == 1);
    CHECK(isolated.size() == 1);
    if (inWord.size() == 1 && isolated.size() == 1 &&
        std::abs(inWord[0].width - isolated[0].width) > 0.5f) {
      differing++;
    }
  }
  CHECK(differing > 0);

  // And the word is not simply the isolated letters laid end to end: joining
  // and the kerning that comes with it change the whole string's width.
  float isolatedTotal = 0;
  for (int32_t at = 0; at + 2 <= static_cast<int32_t>(arabic.size()); at += 2) {
    const auto alone = std::vector<tgfxrn::TextRun>{run(arabic.substr(
        static_cast<size_t>(at), 2))};
    isolatedTotal += tgfxrn::layoutText(alone, spec, 40).metrics.width;
  }
  CHECK(std::abs(isolatedTotal - layout.metrics.width) > 0.5f);
}

/// The three marks whose only job is to be strong, and the line and range
/// indices a reordered line must still report in logical order.
void bidiMarksPlaceNeutralsAndIndicesStayLogical() {
  tgfxrn::FontRegistry::instance().clear();
  if (!registerSystemFamily("Geeza Pro")) {
    std::printf("  skip: no Arabic face on this host\n");
    return;
  }
  tgfxrn::TextSpec spec;
  spec.family = "Geeza Pro";

  // "a عربي!" — the '!' is a neutral between the Arabic and eos, which in a
  // left-to-right paragraph is L, so N1 finds no agreement, N2 gives it the
  // paragraph and it sits at the right-hand end. A right-to-left mark *after*
  // it makes both sides R, so it joins the Arabic run and moves left. That is
  // the whole point of the mark, and it does nothing at all if it is read as a
  // formatting character.
  const std::string plain = "a عربي!";
  const std::string marked = plain + "\u200F";
  const auto plainRuns = std::vector<tgfxrn::TextRun>{run(plain)};
  const auto markedRuns = std::vector<tgfxrn::TextRun>{run(marked)};
  auto plainBox = tgfxrn::layoutText(plainRuns, spec, 24);
  CHECK(plainBox.metrics.width > 0);
  auto bang = static_cast<int32_t>(plain.size()) - 1;
  auto plainBang = tgfxrn::boxesForRange(plainRuns, spec, 24, bang, bang + 1);
  auto markedBang = tgfxrn::boxesForRange(markedRuns, spec, 24, bang, bang + 1);
  CHECK(plainBang.size() == 1);
  CHECK(markedBang.size() == 1);
  if (plainBang.size() == 1 && markedBang.size() == 1) {
    CHECK(markedBang[0].x < plainBang[0].x);
  }

  // A right-to-left line is stored visually once it is placed, so its reported
  // first and last bytes have to come from the logical order, not from the ends
  // of the array.
  const std::string arabic = "مرحبا";
  const auto runs = std::vector<tgfxrn::TextRun>{run(arabic)};
  auto layout = tgfxrn::layoutText(runs, spec, 24);
  CHECK(layout.metrics.lines.size() == 1);
  CHECK(layout.metrics.lines[0].startIndex == 0);
  CHECK(layout.metrics.lines[0].endIndex > layout.metrics.lines[0].startIndex);

  // And a range over the whole string reports the whole string, not the one
  // cluster that happens to be leftmost.
  auto whole = tgfxrn::boxesForRange(runs, spec, 24, 0,
                                     static_cast<int32_t>(arabic.size()));
  CHECK(whole.size() == 1);
  CHECK(whole[0].startIndex == 0);
  CHECK(whole[0].endIndex == static_cast<int32_t>(arabic.size()));
}

/// <TextPath> is one line and was never reordered, so right-to-left text ran
/// along the path backwards even with every glyph correctly joined.
void textOnAPathIsReorderedToo() {
  tgfxrn::FontRegistry::instance().clear();
  if (!registerSystemFamily("Geeza Pro")) {
    std::printf("  skip: no Arabic face on this host\n");
    return;
  }
  tgfxrn::TextSpec spec;
  spec.family = "Geeza Pro";
  tgfx::Path line;
  line.moveTo(4, 40);
  line.lineTo(200, 40);

  const std::string arabic = "مرحبا";
  const auto runs = std::vector<tgfxrn::TextRun>{run(arabic)};
  auto onPath = tgfxrn::layoutTextOnPath(runs, spec, 24, line);
  CHECK(!onPath.runs.empty());
  // Straight and horizontal, so the path layout has to put the glyphs exactly
  // where the flat one does, reordering included.
  auto flat = tgfxrn::layoutText(runs, spec, 24);
  CHECK(std::abs(onPath.metrics.width - flat.metrics.width) < 1.0f);
}

/// Mixed direction: the Latin runs keep their order while the Arabic between
/// them is reversed, which is the run resolution the issue said was missing.
void mixedDirectionResolvesIntoRuns() {
  tgfxrn::FontRegistry::instance().clear();
  if (!registerSystemFamily("Geeza Pro")) {
    std::printf("  skip: no Arabic face on this host\n");
    return;
  }
  tgfxrn::TextSpec spec;
  spec.family = "Geeza Pro";
  const std::string mixed = "ab مرحبا cd";
  const auto runs = std::vector<tgfxrn::TextRun>{run(mixed)};
  auto layout = tgfxrn::layoutText(runs, spec, 24);
  CHECK(layout.metrics.width > 0);
  // The paragraph starts with Latin, so its base direction is left to right:
  // the leftmost glyph is byte 0 and the rightmost is the last byte.
  auto left = tgfxrn::positionAtCoordinate(runs, spec, 24, 0, 0);
  CHECK(left.index == 0);
  auto right = tgfxrn::positionAtCoordinate(runs, spec, 24, layout.metrics.width, 0);
  CHECK(right.index == static_cast<int32_t>(mixed.size()));

  // Inside the Arabic, though, the offsets run backwards. Walk left to right
  // across the whole line and count how often the offset goes down: a purely
  // left-to-right layout never would.
  int32_t previous = -1;
  int backwards = 0;
  for (int step = 0; step <= 40; step++) {
    auto here = tgfxrn::positionAtCoordinate(runs, spec, 24,
                                             layout.metrics.width * static_cast<float>(step) / 40.0f,
                                             0);
    if (here.index < previous) {
      backwards++;
    }
    previous = here.index;
  }
  CHECK(backwards > 0);
}

/// The caret half of #145: which byte offset a point lands on, and the boxes a
/// byte range occupies. Both read the same layout the renderer draws.
void aCoordinateResolvesToACaretAndARangeToItsBoxes() {
  tgfxrn::FontRegistry::instance().clear();
  tgfxrn::TextSpec spec;
  spec.family = "Helvetica";
  const auto runs = std::vector<tgfxrn::TextRun>{run("abcdef")};
  auto measured = tgfxrn::layoutText(runs, spec, 20);
  CHECK(measured.metrics.width > 0);

  auto at = [&](float x, float y) {
    return tgfxrn::positionAtCoordinate(runs, spec, 20, x, y);
  };
  // Left of everything is the very start, on the leading half of 'a'.
  auto first = at(-50, 0);
  CHECK(first.index == 0);
  CHECK(first.leading);
  // Past the right edge is after the last byte, and no longer leading.
  auto past = at(measured.metrics.width + 50, 0);
  CHECK(past.index == 6);
  CHECK(!past.leading);
  // The midpoint of the string lands somewhere inside it, and moving right
  // never moves the caret left.
  int32_t previous = 0;
  for (int step = 0; step <= 12; step++) {
    float x = measured.metrics.width * static_cast<float>(step) / 12.0f;
    auto here = at(x, 0);
    CHECK(here.index >= previous);
    previous = here.index;
  }
  CHECK(previous == 6);

  // One glyph's bounds. 'a' is the first byte, and its box starts at the origin.
  auto one = tgfxrn::boxesForRange(runs, spec, 20, 0, 1);
  CHECK(one.size() == 1);
  CHECK(one[0].startIndex == 0);
  CHECK(one[0].endIndex == 1);
  CHECK(one[0].width > 0);
  CHECK(one[0].height > 0);
  CHECK(one[0].line == 0);
  // Above the baseline, as measureText reports: ascent is negative.
  CHECK(one[0].y < 0);

  // The whole string is one box on one line, as wide as the paragraph.
  auto all = tgfxrn::boxesForRange(runs, spec, 20, 0, 6);
  CHECK(all.size() == 1);
  CHECK(all[0].width > one[0].width);
  CHECK(std::abs(all[0].width - measured.metrics.width) < 0.5f);
  // An empty or reversed range has nothing to highlight.
  CHECK(tgfxrn::boxesForRange(runs, spec, 20, 3, 3).empty());
  CHECK(tgfxrn::boxesForRange(runs, spec, 20, 4, 2).empty());

  // Wrapped, a range crossing the break comes back as one box per line, and a
  // point below the first line resolves onto the second.
  tgfxrn::TextSpec wrapped = spec;
  wrapped.wrapWidth = 40;
  const auto para = std::vector<tgfxrn::TextRun>{run("one two three four")};
  auto layout = tgfxrn::layoutText(para, wrapped, 12);
  CHECK(layout.metrics.lineCount > 1);
  auto crossing = tgfxrn::boxesForRange(para, wrapped, 12, 0,
                                        static_cast<int32_t>(std::string("one two three four").size()));
  CHECK(crossing.size() == static_cast<size_t>(layout.metrics.lineCount));
  for (size_t i = 1; i < crossing.size(); i++) {
    CHECK(crossing[i].line == crossing[i - 1].line + 1);
    CHECK(crossing[i].y > crossing[i - 1].y);
  }
  auto second = tgfxrn::positionAtCoordinate(para, wrapped, 12, 0,
                                             layout.metrics.lines[1].baseline);
  CHECK(second.line == 1);
  CHECK(second.index >= layout.metrics.lines[1].startIndex);

  // A caret on a trailing half says so, and reports the offset after the glyph
  // rather than before it — one offset, two positions, which is what a caret at
  // a wrap needs to tell apart.
  auto firstBox = tgfxrn::boxesForRange(runs, spec, 20, 0, 1);
  auto secondBox = tgfxrn::boxesForRange(runs, spec, 20, 1, 2);
  CHECK(firstBox.size() == 1);
  CHECK(secondBox.size() == 1);
  auto trailing = at(firstBox[0].x + firstBox[0].width * 0.75f, 0);
  CHECK(!trailing.leading);
  CHECK(trailing.index == 1);
  auto leadingHalf = at(secondBox[0].x + secondBox[0].width * 0.25f, 0);
  CHECK(leadingHalf.leading);
  CHECK(leadingHalf.index == 1);

  // An explicit line height opens a gap between the lines, and the gap belongs
  // to the line above it. Measured from the second line's own top, not from the
  // first line's ink, which is where this used to hand the whole gap away.
  tgfxrn::TextSpec spaced = spec;
  spaced.wrapWidth = 40;
  spaced.lineHeight = 40;
  auto tall = tgfxrn::layoutText(para, spaced, 12);
  CHECK(tall.metrics.lineCount > 1);
  float gap = tall.metrics.lines[1].baseline + tall.metrics.lines[1].ascent - 2;
  CHECK(tgfxrn::positionAtCoordinate(para, spaced, 12, 0, gap).line == 0);
  CHECK(tgfxrn::positionAtCoordinate(para, spaced, 12, 0,
                                     tall.metrics.lines[1].baseline)
            .line == 1);

  // An ellipsis stands in for text that is no longer laid out, so it owns no
  // bytes. A caret past it lands exactly after the last glyph the line kept —
  // asserted as an equality, because "somewhere in the string" passes whatever
  // the ellipsis character's own byte length happens to be.
  tgfxrn::TextSpec clipped = wrapped;
  clipped.maxLines = 1;
  clipped.ellipsis = "…";
  auto cut = tgfxrn::layoutText(para, clipped, 12);
  CHECK(cut.metrics.lineCount == 1);
  auto tail = tgfxrn::positionAtCoordinate(para, clipped, 12, cut.metrics.width + 20, 0);
  auto kept = tgfxrn::boxesForRange(para, clipped, 12, 0, 18);
  CHECK(kept.size() == 1);
  CHECK(tail.index == kept[0].endIndex);
  // And the ellipsis never joins a range: it would otherwise straddle every one
  // around its offset, widening the box and dragging its end index backwards.
  auto around = tgfxrn::boxesForRange(para, clipped, 12, 1, tail.index - 1);
  CHECK(around.size() == 1);
  CHECK(around[0].endIndex == tail.index - 1);
  CHECK(around[0].width < kept[0].width);

  // Multi-byte text, which is the only thing the byte lengths exist for. Each
  // caret offset has to start a codepoint, and a range over one accented letter
  // has to be that letter's box rather than half of it.
  const auto accented = std::vector<tgfxrn::TextRun>{run("héllo")};
  auto accentedBox = tgfxrn::layoutText(accented, spec, 20);
  CHECK(accentedBox.metrics.width > 0);
  // h=1 byte, é=2, l l o=1 each, so the boundaries are 0 1 3 4 5 6.
  const std::vector<int32_t> boundaries = {0, 1, 3, 4, 5, 6};
  int32_t previousOffset = -1;
  for (int step = 0; step <= 20; step++) {
    float x = accentedBox.metrics.width * static_cast<float>(step) / 20.0f;
    auto here = tgfxrn::positionAtCoordinate(accented, spec, 20, x, 0);
    CHECK(std::find(boundaries.begin(), boundaries.end(), here.index) != boundaries.end());
    CHECK(here.index >= previousOffset);
    previousOffset = here.index;
  }
  auto accentBoxes = tgfxrn::boxesForRange(accented, spec, 20, 1, 2);
  CHECK(accentBoxes.size() == 1);
  CHECK(accentBoxes[0].startIndex == 1);
  CHECK(accentBoxes[0].endIndex == 3);
  CHECK(tgfxrn::boxesForRange(accented, spec, 20, 1, 3).size() == 1);
}

void theMeasuredWidthIsTheDrawnWidth() {
  tgfxrn::FontRegistry::instance().clear();
  tgfxrn::TextSpec spec;
  spec.family = "Helvetica";
  auto measured = tgfxrn::layoutText({run("Wg")}, spec, 24);
  CHECK(!measured.runs.empty());
  CHECK(measured.metrics.width > 0);
  CHECK(measured.metrics.lineCount == 1);

  Writer w;
  w.push(1);
  w.push(8);
  w.pushAnim(0);
  w.pushAnim(40);
  w.pushAnim(24);
  w.push(1);
  w.push(400);
  w.push(0);
  w.push(0);
  writeRun(w, 0);
  w.push(0);
  w.push(0);
  w.push(0);
  w.push(0);
  w.push(2);
  w.pushPaint(0xff000000, false);
  auto ink = rightmostInk(render(w.finish(), {"Wg", "Helvetica", ""}, {}));
  CHECK(ink >= 0);
  CHECK(static_cast<float>(ink) <= measured.metrics.width);
  CHECK(static_cast<float>(ink) > measured.metrics.width * 0.7f);

  spec.wrapWidth = 40;
  auto wrapped = tgfxrn::layoutText({run("one two three four")}, spec, 12);
  CHECK(wrapped.metrics.lineCount > 1);
  CHECK(wrapped.metrics.width <= 40);
  CHECK(wrapped.metrics.height > wrapped.metrics.descent - wrapped.metrics.ascent);

  spec.maxLines = 1;
  spec.ellipsis = "...";
  auto clipped = tgfxrn::layoutText({run("one two three four")}, spec, 12);
  CHECK(clipped.metrics.lineCount == 1);
  CHECK(clipped.metrics.width <= 40);
}

/// Cached by rounded size: without it an animated fontSize builds a ScalerContext every frame.
void theLayoutIsCachedByQuantisedSize() {
  tgfxrn::FontRegistry::instance().clear();
  tgfxrn::TextCmd cmd;
  cmd.runs = {run("Wg")};
  cmd.spec.family = "Helvetica";
  auto blobOf = [](const tgfxrn::TextLayout& layout) {
    return layout.runs.empty() ? nullptr : layout.runs[0].blob;
  };
  auto at20 = blobOf(cmd.resolve(20, {}, 0, {}));
  CHECK(at20 != nullptr);
  CHECK(blobOf(cmd.resolve(20.1f, {}, 0, {})) == at20);
  CHECK(blobOf(cmd.resolve(20, {}, 0, {})) == at20);
  CHECK(blobOf(cmd.resolve(24, {}, 0, {})) != at20);
}

/// A codepoint the primary face lacks must reach a fallback, not the blank gap glyph 0 leaves.
void anEmojiFallsBackInsteadOfVanishing() {
  tgfxrn::FontRegistry::instance().clear();
  const std::vector<std::string> strings{"\xF0\x9F\x98\x80", "Helvetica", ""};
  Writer w;
  writeText(w, 28, -1, 0);
  auto pixels = render(w.finish(), strings, {});
  CHECK(!tgfxrn::FontRegistry::instance().fallbacks().empty());
  CHECK(anyPixelSet(pixels));

  // A forced faux bit has to reach the fallback too, or one string draws at two weights.
  const std::vector<std::string> mixed{"a\xE4\xB8\xAD", "Helvetica", ""};
  Writer plainRun;
  writeText(plainRun, 28, -1, 0);
  Writer boldRun;
  writeText(boldRun, 28, -1, 0, 0, 0, 0, 400, 2);
  CHECK(render(boldRun.finish(), mixed, {}) != render(plainRun.finish(), mixed, {}));
}

void textOnAPathFollowsTheCurve() {
  tgfxrn::FontRegistry::instance().clear();
  const std::vector<std::string> strings{"M4 40 q28 -32 56 0", "curve", "Helvetica"};
  Writer w;
  w.push(1);
  w.push(10);  // TextPath
  w.push(0);   // path
  w.pushAnim(12);
  w.push(2);  // family
  w.push(400);
  w.push(0);
  w.push(0);
  writeRun(w, 1, 2);
  w.push(0);  // startOffset
  w.pushPaint(0xff000000, false);
  auto curved = render(w.finish(), strings, {});
  CHECK(anyPixelSet(curved));

  Writer flat;
  writeText(flat, 12, -1, 0);
  CHECK(render(flat.finish(), {"curve", "Helvetica", ""}, {}) != curved);
}

void aDisposedHandleDrawsNothing() {
  auto image = registerSolid(cellColor(5));
  tgfxrn::ImageRegistry::instance().remove(image.handle);
  Writer w;
  w.push(1);
  w.pushImage(image.handle, 0, 0, 32, 32);
  CHECK(!anyPixelSet(render(w.finish(), {}, {})));
}

/// A JPEG with the EXIF marker a phone writes for a portrait photo, built here so the bytes show.
std::shared_ptr<tgfx::Data> exifRotatedJpeg(int width, int height) {
  std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4, 255);
  auto info = tgfx::ImageInfo::Make(width, height, tgfx::ColorType::RGBA_8888,
                                    tgfx::AlphaType::Opaque);
  tgfx::Pixmap pixmap(info, pixels.data());
  auto jpeg = tgfx::ImageCodec::Encode(pixmap, tgfx::EncodedFormat::JPEG, 100);
  if (jpeg == nullptr) {
    return nullptr;
  }
  // APP1: length, "Exif\0\0", then a little-endian TIFF whose one entry is Orientation = 6.
  const std::array<uint8_t, 36> app1 = {
      0xFF, 0xE1, 0x00, 0x22,                          // marker, length (34)
      'E', 'x', 'i', 'f', 0x00, 0x00,                  // Exif\0\0
      'I', 'I', 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00,    // II, 42, IFD0 at 8
      0x01, 0x00,                                      // one entry
      0x12, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00,  // Orientation, SHORT, 1
      0x06, 0x00, 0x00, 0x00,                          // = 6
      0x00, 0x00, 0x00, 0x00};                         // no next IFD
  auto bytes = static_cast<const uint8_t*>(jpeg->data());
  std::vector<uint8_t> out;
  out.reserve(jpeg->size() + app1.size());
  out.insert(out.end(), bytes, bytes + 2);
  out.insert(out.end(), app1.begin(), app1.end());
  out.insert(out.end(), bytes + 2, bytes + jpeg->size());
  return tgfx::Data::MakeWithCopy(out.data(), out.size());
}

/// A portrait photo must draw upright, through JpegCodec by name rather than the host codec.
void anExifRotatedJpegDecodesUpright() {
  auto data = exifRotatedJpeg(64, 32);
  CHECK(data != nullptr);
  if (data == nullptr) {
    return;
  }
  CHECK(tgfx::JpegCodec::IsJpeg(data));
  auto codec = tgfx::JpegCodec::MakeFrom(data);
  CHECK(codec != nullptr);
  if (codec == nullptr) {
    return;
  }
  CHECK(codec->orientation() == tgfx::Orientation::RightTop);

  auto handle = tgfxrn::ImageRegistry::instance().add(codec);
  auto image = tgfxrn::ImageRegistry::instance().acquire(handle.handle);
  CHECK(image != nullptr);
  if (image != nullptr) {
    CHECK(image->width() == 32);
    CHECK(image->height() == 64);
  }
  tgfxrn::ImageRegistry::instance().remove(handle.handle);
}

/// A 24x24 document whose viewBox is half its declared size, so a scale that ignored either
/// one would land the square somewhere else.
constexpr const char* kSvgDoc =
    R"(<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 12 12">
         <rect x="0" y="0" width="12" height="12" fill="#3366cc"/>
       </svg>)";

void anSvgDocumentBecomesAnImageHandle() {
  auto data = tgfx::Data::MakeWithCopy(kSvgDoc, std::strlen(kSvgDoc));
  CHECK(tgfxrn::looksLikeSvg(*data));
  CHECK(!tgfxrn::looksLikeSvg(*tgfx::Data::MakeWithCopy("\x89PNG\r\n", 6)));

  // The declared 24x24, not the 12x12 viewBox: a scale that read either alone would disagree.
  auto image = tgfxrn::svgImage(data);
  CHECK(image != nullptr && image->width() == 24 && image->height() == 24);
  if (image == nullptr) {
    return;
  }

  auto handle = tgfxrn::ImageRegistry::instance().addExternal(std::move(image));
  CHECK(handle.handle != 0 && handle.width == 24 && handle.height == 24);
  Writer w;
  w.push(1);
  w.pushImage(handle.handle, 0, 0, kWidth, kHeight);
  auto got = pixelAt(render(w.finish(), {}, {}), kWidth / 2, kHeight / 2);
  CHECK(got[0] == 0x33 && got[1] == 0x66 && got[2] == 0xcc);
  tgfxrn::ImageRegistry::instance().remove(handle.handle);
}

/// The reason there is no explicit SVG raster size: a handle built at the document's own size
/// replays the picture at draw resolution, so drawing it 8x larger stays crisp. If tgfx ever
/// starts resampling the nominal raster instead, the upscaled edge widens and this fails.
void anSvgHandleStaysCrispWhenDrawnLargerThanItsSize() {
  // Two draws, so Image::MakeFrom cannot take its single-draw unwrap path — this has to be a
  // real PictureImage, which is the case the resampling worry was about.
  const char* doc =
      R"(<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8" viewBox="0 0 8 8">
           <path d="M0 0 L8 0 L0 8 Z" fill="#000000"/>
           <rect x="6" y="6" width="2" height="2" fill="#000000"/>
         </svg>)";
  auto draw = [](std::shared_ptr<tgfx::Image> image) {
    auto handle = tgfxrn::ImageRegistry::instance().addExternal(std::move(image));
    Writer w;
    w.push(1);
    w.pushImage(handle.handle, 0, 0, kWidth, kHeight);
    auto pixels = render(w.finish(), {}, {});
    tgfxrn::ImageRegistry::instance().remove(handle.handle);
    return pixels;
  };
  auto eightPx = tgfxrn::svgImage(tgfx::Data::MakeWithCopy(doc, std::strlen(doc)));
  CHECK(eightPx != nullptr && eightPx->width() == 8);
  if (eightPx == nullptr) {
    return;
  }
  auto drawn = draw(std::move(eightPx));

  // Partly-covered pixels along the diagonal, on the row through the middle. Antialiasing gives
  // one; resampling an 8px raster into a 64px box would smear it across about eight.
  int soft = 0;
  for (int x = 0; x < kWidth; x++) {
    auto alpha = drawn[static_cast<size_t>((kHeight / 2 * kWidth + x) * 4 + 3)];
    if (alpha > 24 && alpha < 231) {
      soft++;
    }
  }
  CHECK(soft <= 2);
}

void malformedSvgIsRejectedRatherThanDrawn() {
  CHECK(tgfxrn::svgImage(tgfx::Data::MakeWithCopy("<svg><rect", 10)) == nullptr);
  // Well-formed XML, but not an <svg> root.
  CHECK(tgfxrn::svgImage(tgfx::Data::MakeWithCopy("<html></html>", 13)) == nullptr);
  // Parses, but has no size to draw into.
  const char* sizeless = R"(<svg xmlns="http://www.w3.org/2000/svg"/>)";
  CHECK(tgfxrn::svgImage(tgfx::Data::MakeWithCopy(sizeless, std::strlen(sizeless))) == nullptr);
  CHECK(tgfxrn::svgImage(nullptr) == nullptr);
}

void aParagraphKeepsItsItalicThroughTheRuns() {
  tgfxrn::FontRegistry::instance().clear();
  tgfxrn::TextSpec spec;
  spec.family = "Helvetica";
  // Pixels, not widths: a synthesised italic is a shear at identical advances.
  const std::vector<std::string> strings{"italic", "Helvetica", ""};
  Writer upright;
  writeText(upright, 24, -1, 0, 0, 0, 0, 400, 0);
  Writer slanted;
  writeText(slanted, 24, -1, 0, 0, 0, 0, 400, 1);
  auto flat = render(upright.finish(), strings, {});
  auto leaning = render(slanted.finish(), strings, {});
  CHECK(anyPixelSet(flat));
  CHECK(flat != leaning);

  Writer overridden;
  writeText(overridden, 24, -1, 0, 0, 0, 0, 400, 1, 2);
  CHECK(render(overridden.finish(), strings, {}) == flat);
  Writer opted;
  writeText(opted, 24, -1, 0, 0, 0, 0, 400, 0, 1);
  CHECK(render(opted.finish(), strings, {}) == leaning);

  tgfxrn::TextSpec italicSpec = spec;
  italicSpec.italic = true;
  auto outline = [](const tgfxrn::TextRun& r, const tgfxrn::TextSpec& sp) {
    return tgfxrn::textToPath({r}, sp, 24).getBounds();
  };
  tgfxrn::TextRun inherits;
  inherits.text = "italic";
  tgfxrn::TextRun forced = inherits;
  forced.italic = 2;
  CHECK(outline(inherits, italicSpec) != outline(inherits, spec));
  CHECK(outline(forced, italicSpec) == outline(inherits, spec));
}

void theEllipsisBelongsToTheRunItFollows() {
  tgfxrn::FontRegistry::instance().clear();
  tgfxrn::TextSpec spec;
  spec.family = "Helvetica";
  spec.wrapWidth = 60;
  spec.maxLines = 1;
  spec.ellipsis = "...";
  tgfxrn::TextRun first;
  first.text = "one two ";
  tgfxrn::TextRun tail;
  tail.text = "three four five";
  tail.hasColor = true;
  tail.color = 0xFF00FF00;
  auto layout = tgfxrn::layoutText({first, tail}, spec, 12);
  CHECK(layout.metrics.lineCount == 1);
  CHECK(!layout.metrics.lines.empty());
  CHECK(layout.runs.size() == 1);
  CHECK(!layout.runs[0].hasColor);
  CHECK(layout.metrics.lines[0].endIndex > 0);

  tgfxrn::TextRun lead;
  lead.text = "one ";
  tgfxrn::TextRun rest;
  rest.text = "two three four five";
  rest.hasColor = true;
  rest.color = 0xFF00FF00;
  auto coloured = tgfxrn::layoutText({lead, rest}, spec, 12);
  CHECK(coloured.metrics.lineCount == 1);
  CHECK(coloured.runs.size() == 2);
  CHECK(coloured.runs.back().hasColor);
}

void aSpanCarriesItsOwnFace() {
  tgfxrn::FontRegistry::instance().clear();
  tgfxrn::TextSpec spec;
  spec.family = "Helvetica";
  auto plain = tgfxrn::layoutText({run("one two")}, spec, 20);

  tgfxrn::TextRun first;
  first.text = "one ";
  tgfxrn::TextRun bold;
  bold.text = "two";
  bold.weight = 700;
  auto mixed = tgfxrn::layoutText({first, bold}, spec, 20);
  CHECK(!mixed.runs.empty());
  CHECK(mixed.metrics.lineCount == 1);
  CHECK(mixed.metrics.width > plain.metrics.width);

  tgfxrn::TextRun big;
  big.text = "two";
  big.size = 40;
  auto tall = tgfxrn::layoutText({first, big}, spec, 20);
  CHECK(tall.metrics.lineCount == 1);
  CHECK(tall.metrics.height > plain.metrics.height);
}

void aColouredSpanSplitsTheBlobs() {
  tgfxrn::FontRegistry::instance().clear();
  tgfxrn::TextSpec spec;
  spec.family = "Helvetica";
  tgfxrn::TextRun plain;
  plain.text = "one ";
  tgfxrn::TextRun red;
  red.text = "two";
  red.hasColor = true;
  red.color = 0xFFFF0000;
  auto layout = tgfxrn::layoutText({plain, red}, spec, 20);
  CHECK(layout.runs.size() == 2);
  CHECK(!layout.runs[0].hasColor);
  CHECK(layout.runs[1].hasColor);
  CHECK(layout.runs[1].color == 0xFFFF0000);

  tgfxrn::TextRun more;
  more.text = "three";
  CHECK(tgfxrn::layoutText({plain, more}, spec, 20).runs.size() == 1);

  const std::vector<std::string> strings{"one ", "Helvetica", "", "two"};
  auto write = [&](bool coloured) {
    Writer w;
    w.push(1);
    w.push(8);  // Text
    w.pushAnim(2);
    w.pushAnim(40);
    w.pushAnim(20);
    w.push(1);
    w.push(400);
    w.push(0);
    w.push(0);
    w.push(2);  // two runs
    w.push(0);
    w.push(2);
    w.push(0);
    w.push(0);
    w.push(0);
    w.push(0);
    w.push(0);
    w.push(0);
    w.push(3);
    w.push(2);
    w.push(0);
    w.push(0);
    w.push(0);
    w.push(0);
    w.push(coloured ? 1 : 0);
    w.push(coloured ? 0xFFFF0000 : 0);
    w.push(0);
    w.push(0);
    w.push(0);
    w.push(0);
    w.push(2);
    w.pushPaint(0xff000000, false);
    return render(w.finish(), strings, {});
  };
  auto mono = write(false);
  auto tinted = write(true);
  CHECK(anyPixelSet(mono));
  CHECK(mono != tinted);
}

void everyLineIsMeasured() {
  tgfxrn::FontRegistry::instance().clear();
  tgfxrn::TextSpec spec;
  spec.family = "Helvetica";
  spec.wrapWidth = 40;
  spec.align = tgfxrn::TextAlign::Right;
  auto layout = tgfxrn::layoutText({run("one two three four")}, spec, 12);
  CHECK(layout.metrics.lines.size() == static_cast<size_t>(layout.metrics.lineCount));
  CHECK(layout.metrics.lines.size() > 1);
  CHECK(layout.metrics.lines[0].baseline == 0);

  tgfxrn::TextSpec plain;
  plain.family = "Helvetica";
  auto blanks = tgfxrn::layoutText({run("a\n\nb")}, plain, 12);
  CHECK(blanks.metrics.lines.size() == 3);
  CHECK(blanks.metrics.lines[1].startIndex == 2);
  auto trailing = tgfxrn::layoutText({run("a\n")}, plain, 12);
  CHECK(trailing.metrics.lines.size() == 2);
  CHECK(trailing.metrics.lines[1].startIndex == 2);
  for (size_t i = 1; i < layout.metrics.lines.size(); i++) {
    const auto& line = layout.metrics.lines[i];
    CHECK(line.baseline > layout.metrics.lines[i - 1].baseline);
    CHECK(std::abs(line.x + line.width - spec.wrapWidth) < 0.01f);
    CHECK(line.startIndex >= layout.metrics.lines[i - 1].endIndex);
  }
}

void textComesBackAsAPath() {
  tgfxrn::FontRegistry::instance().clear();
  tgfxrn::TextSpec spec;
  spec.family = "Helvetica";
  auto path = tgfxrn::textToPath({run("Wg")}, spec, 40);
  CHECK(!path.isEmpty());

  auto measured = tgfxrn::layoutText({run("Wg")}, spec, 40);
  auto bounds = path.getBounds();
  CHECK(bounds.width() <= measured.metrics.width + 1);
  CHECK(bounds.width() > measured.metrics.width * 0.7f);
  CHECK(bounds.top >= measured.metrics.ascent - 1);
  CHECK(bounds.top < 0);

  auto data = tgfxrn::pathToSvg(path);
  CHECK(!data.empty());
  auto reparsed = tgfxrn::parseSvgPath(data);
  auto again = reparsed.getBounds();
  CHECK(std::abs(again.width() - bounds.width()) < 0.5f);
  CHECK(std::abs(again.height() - bounds.height()) < 0.5f);

  spec.wrapWidth = 30;
  auto wrapped = tgfxrn::textToPath({run("Wg Wg Wg")}, spec, 40);
  CHECK(wrapped.getBounds().height() > bounds.height());
}

void pathUtilitiesAgreeWithTheGeometryTheyWrap() {
  const std::string big = "M0 0 L40 0 L40 40 L0 40 Z";
  const std::string small = "M10 10 L30 10 L30 30 L10 30 Z";

  auto boxOf = [](const std::string& d) { return tgfxrn::pathBounds(d, false); };
  CHECK(boxOf(big).width() == 40 && boxOf(big).height() == 40);
  CHECK(boxOf(big).left == 0 && boxOf(big).top == 0);

  // Difference punches the small square out, so the centre stops being inside.
  auto holed = tgfxrn::pathCombine(big, small, "difference");
  CHECK(tgfxrn::pathContains(big, 20, 20));
  CHECK(!tgfxrn::pathContains(holed, 20, 20));
  CHECK(tgfxrn::pathContains(holed, 5, 5));
  CHECK(boxOf(holed).width() == 40);

  // Curves take the same reversal path as lines, so a round hole has to survive it too.
  auto ring = tgfxrn::pathCombine(
      "M0 0 L60 0 L60 60 L0 60 Z",
      "M30 10 C41 10 50 19 50 30 C50 41 41 50 30 50 C19 50 10 41 10 30 C10 19 19 10 30 10 Z",
      "difference");
  CHECK(!tgfxrn::pathContains(ring, 30, 30));
  CHECK(tgfxrn::pathContains(ring, 3, 3));

  // A lens and an oval have almost no endpoint area, so nesting cannot be ranked by area.
  for (const char* hull : {"M0 50 Q50 0 100 50 Q50 100 0 50 Z",
                           "M0 30 C0 -10 80 -10 80 30 C80 70 0 70 0 30 Z"}) {
    for (const char* op : {"difference", "xor"}) {
      auto punched = tgfxrn::pathCombine(hull, "M35 25 L55 25 L55 45 L35 45 Z", op);
      CHECK(!tgfxrn::pathContains(punched, 45, 35));
      CHECK(tgfxrn::pathContains(punched, 50, 50));
    }
  }

  auto both = tgfxrn::pathCombine(big, small, "intersect");
  CHECK(std::abs(boxOf(both).width() - 20) < 0.01f);

  // An unknown op appends rather than guessing, so the union of the boxes stands.
  CHECK(std::abs(boxOf(tgfxrn::pathCombine(big, small, "wat")).width() - 40) < 0.01f);

  // [a, b, c, d, tx, ty]: scale 2 and shift 5, as SVG writes it.
  auto moved = tgfxrn::pathTransform(big, {2, 0, 0, 2, 5, 5});
  CHECK(std::abs(boxOf(moved).left - 5) < 0.01f);
  CHECK(std::abs(boxOf(moved).width() - 80) < 0.01f);
  CHECK(tgfxrn::pathTransform(big, {1, 0}) == tgfxrn::pathTransform(big, {1, 0, 0, 1, 0, 0}));

  const std::string line = "M0 0 L100 0";
  CHECK(std::abs(tgfxrn::pathLength(line) - 100) < 0.01f);
  auto at = tgfxrn::pathPointAtLength(line, 25);
  CHECK(at.ok && std::abs(at.x - 25) < 0.01f && std::abs(at.y) < 0.01f);
  CHECK(std::abs(at.angle) < 0.01f);  // straight along +x
  // Distance is pinned to the path, not extrapolated past its end.
  CHECK(std::abs(tgfxrn::pathPointAtLength(line, 500).x - 100) < 0.01f);

  auto middle = tgfxrn::pathSegment(line, 20, 60);
  CHECK(std::abs(tgfxrn::pathLength(middle) - 40) < 0.01f);
  CHECK(tgfxrn::pathSegment(line, 60, 20).empty());
  CHECK(tgfxrn::pathLength("") == 0);
  CHECK(!tgfxrn::pathPointAtLength("", 1).ok);

  // Every contour counts, not just the first: PathMeasure walks them one at a time.
  const std::string twoLines = "M0 0 L100 0 M0 10 L100 10";
  CHECK(std::abs(tgfxrn::pathLength(twoLines) - 200) < 0.01f);
  auto second = tgfxrn::pathPointAtLength(twoLines, 150);
  CHECK(second.ok && std::abs(second.x - 50) < 0.01f && std::abs(second.y - 10) < 0.01f);
  CHECK(std::abs(tgfxrn::pathLength(tgfxrn::pathSegment(twoLines, 0, 1000)) - 200) < 0.01f);
  CHECK(std::abs(tgfxrn::pathLength(holed) - (160 + 80)) < 0.01f);
}

/// <Glyphs> draws exactly the ids it is handed, where it is told: no shaping, no fallback.
void placedGlyphsDrawWhereTheyArePut() {
  tgfxrn::FontRegistry::instance().clear();
  tgfxrn::TextSpec spec;
  spec.family = "Helvetica";
  auto ids = tgfxrn::glyphIdsFor("Wg", spec, 20);
  CHECK(ids.size() == 2);
  CHECK(ids[0] != 0);

  auto write = [&](float secondX) {
    Writer w;
    w.push(1);
    w.push(12);  // Glyphs
    w.pushAnim(2);
    w.pushAnim(40);
    w.pushAnim(20);
    w.push(1);  // family
    w.push(400);
    w.push(0);
    w.push(0);
    w.push(2);  // two glyphs
    w.push(ids[0]);
    w.push(0);
    w.push(0);
    w.push(ids[1]);
    w.push(secondX);
    w.push(0);
    w.pushPaint(0xff000000, false);
    return render(w.finish(), {"", "Helvetica"}, {});
  };
  auto together = write(14);
  auto apart = write(34);
  CHECK(anyPixelSet(together));
  CHECK(anyPixelSet(apart));
  CHECK(together != apart);
  CHECK(rightmostInk(apart) > rightmostInk(together));

  Writer bogus;
  bogus.push(1);
  bogus.push(12);
  bogus.pushAnim(2);
  bogus.pushAnim(40);
  bogus.pushAnim(20);
  bogus.push(1);
  bogus.push(400);
  bogus.push(0);
  bogus.push(0);
  bogus.push(1);
  bogus.push(60000);
  bogus.push(0);
  bogus.push(0);
  bogus.pushPaint(0xff000000, false);
  render(bogus.finish(), {"", "Helvetica"}, {});
}

/// One op, N sprites: each lands where its own transform puts it.
void anAtlasPlacesEverySpriteItIsGiven() {
  auto red = registerSolid({255, 0, 0, 255});
  CHECK(red.handle != 0);

  // Four 2x2 sprites scaled to 8x8, centred on the quadrants of a 64x64 canvas.
  std::vector<std::array<double, 7>> sprites;
  const double centres[4][2] = {{16, 16}, {48, 16}, {16, 48}, {48, 48}};
  for (const auto& centre : centres) {
    // v26 carries the centre; the reader applies the half-sprite bias.
    sprites.push_back({0, 0, 2, 2, 4, centre[0], centre[1]});
  }
  Writer w;
  w.push(1);
  w.pushAtlas(red.handle, sprites);
  auto pixels = render(w.finish(), {}, {});
  for (const auto& centre : centres) {
    auto got = pixelAt(pixels, static_cast<int>(centre[0]), static_cast<int>(centre[1]));
    CHECK(got[0] == 255 && got[1] == 0 && got[2] == 0);
  }
  // Between the sprites nothing was drawn, and no sprite spilled past its scale.
  CHECK(pixelAt(pixels, 32, 32)[0] == 0);
  CHECK(inkPixels(pixels) == 4 * 8 * 8);

  // Each sprite samples its own sub-rect: left half red, right half blue.
  std::vector<uint8_t> half{255, 0, 0, 255, 0, 0, 255, 255};
  auto split = registerPixels(2, 1, half);
  Writer q;
  q.push(1);
  q.pushAtlas(split.handle, {{0, 0, 1, 1, 16, 32 - 8, 32 - 8}, {1, 0, 1, 1, 16, 32 + 8, 32 - 8}});
  auto halves = render(q.finish(), {}, {});
  CHECK(pixelAt(halves, 32 - 8, 32 - 8)[0] == 255);
  CHECK(pixelAt(halves, 32 + 8, 32 - 8)[2] == 255);
}

/// A mesh below the shape level: per-vertex colour, and indices reusing vertices.
void verticesDrawTheTrianglesTheyDescribe() {
  // One triangle covering the top-left half of a 64x64 canvas, paint-coloured.
  Writer plain;
  plain.push(1);
  plain.pushVertices({{0, 0}, {56, 0}, {0, 56}});
  auto flat = render(plain.finish(), {}, {});
  CHECK(pixelAt(flat, 8, 8)[0] == 255 && pixelAt(flat, 8, 8)[2] == 255);
  // Past the hypotenuse nothing was drawn.
  CHECK(pixelAt(flat, 56, 56)[0] == 0);

  // Per-vertex colour: red at the origin, blue at the far corner, blended between.
  Writer shaded;
  shaded.push(1);
  shaded.pushVertices({{0, 0}, {56, 0}, {0, 56}}, {}, {0xffff0000, 0xff0000ff, 0xff0000ff});
  auto blended = render(shaded.finish(), {}, {});
  auto near = pixelAt(blended, 4, 4);
  auto far = pixelAt(blended, 40, 4);
  CHECK(near[0] > far[0] && far[2] > near[2]);

  // Two triangles from four vertices: indices reuse the shared edge.
  Writer quad;
  quad.push(1);
  quad.pushVertices({{8, 8}, {56, 8}, {8, 56}, {56, 56}}, {}, {}, {0, 1, 2, 1, 2, 3});
  auto square = render(quad.finish(), {}, {});
  CHECK(pixelAt(square, 12, 12)[0] == 255);
  CHECK(pixelAt(square, 52, 52)[0] == 255);
  CHECK(pixelAt(square, 4, 4)[0] == 0);

  // The same four vertices as a strip need no indices at all.
  Writer strip;
  strip.push(1);
  strip.pushVertices({{8, 8}, {56, 8}, {8, 56}, {56, 56}}, {}, {}, {}, 1);
  auto stripped = render(strip.finish(), {}, {});
  CHECK(pixelAt(stripped, 32, 32)[0] == 255);

  // Fewer than three vertices describes no triangle and must draw nothing.
  Writer degenerate;
  degenerate.push(1);
  degenerate.pushVertices({{0, 0}, {56, 0}});
  CHECK(inkPixels(render(degenerate.finish(), {}, {})) == 0);

  // An index past the last vertex would read another draw's buffer: skip the mesh.
  Writer past;
  past.push(1);
  past.pushVertices({{8, 8}, {56, 8}, {8, 56}}, {}, {}, {0, 1, 3});
  CHECK(inkPixels(render(past.finish(), {}, {})) == 0);
  Writer negative;
  negative.push(1);
  negative.pushVertices({{8, 8}, {56, 8}, {8, 56}}, {}, {}, {0, 1, -1});
  CHECK(inkPixels(render(negative.finish(), {}, {})) == 0);
  // A node after a rejected mesh still parses, so the reader stayed in step.
  Writer after;
  after.push(2);
  after.pushVertices({{8, 8}, {56, 8}, {8, 56}}, {}, {}, {0, 1, 3});
  after.push(1);  // Rect
  after.pushAnim(8);
  after.pushAnim(8);
  after.pushAnim(16);
  after.pushAnim(16);
  after.pushPaint(0xff00ff00, false);
  CHECK(pixelAt(render(after.finish(), {}, {}), 16, 16)[1] == 255);
}

/// Texture coordinates against an image shader: the warp samples where it is told.
void verticesSampleTheirTextureCoordinates() {
  // Left half red, right half blue, sampled so the whole quad takes the blue half.
  std::vector<uint8_t> half{255, 0, 0, 255, 0, 0, 255, 255};
  auto split = registerPixels(2, 1, half);
  CHECK(split.handle != 0);

  Writer w;
  w.push(1);
  w.push(16);
  w.push(0);  // inline, no shared buffer
  w.push(0);  // triangles
  w.push(4);
  const double quad[4][2] = {{8, 8}, {56, 8}, {8, 56}, {56, 56}};
  for (const auto& point : quad) {
    w.push(point[0]);
    w.push(point[1]);
  }
  w.push(1);
  // All four corners land in the right-hand pixel, in image pixel space.
  for (int i = 0; i < 4; i++) {
    w.push(1.5);
    w.push(0.5);
  }
  w.push(0);
  w.push(6);
  for (int index : {0, 1, 2, 1, 2, 3}) {
    w.push(index);
  }
  w.pushImageShaderPaint(split.handle, 0, 0);
  auto warped = render(w.finish(), {}, {});
  auto got = pixelAt(warped, 32, 32);
  CHECK(got[2] == 255 && got[0] == 0);
}

/// No window, no view: renderToImage's surface belongs to nothing on screen.
void anOffscreenRenderProducesTheSamePixels() {
  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(32);
  w.pushAnim(32);
  w.pushPaint(0xff0000ff, false);
  auto program = w.finish();

  size_t bytes = 0;
  auto image = tgfxrn::renderSceneOffscreen(program, {}, 64, 64, 1, /*opaque=*/false, /*displayP3=*/false, bytes);
  CHECK(image != nullptr);
  CHECK(bytes == 64u * 64u * 4u);
  CHECK(image->width() == 64 && image->height() == 64);

  // The pixels came back on the CPU, so they draw in a context that never saw the surface.
  auto* context = device()->lockContext();
  auto surface = tgfx::Surface::Make(context, kWidth, kHeight);
  auto* canvas = surface->getCanvas();
  canvas->clear();
  canvas->drawImage(image);
  context->flushAndSubmit(true);
  auto info = tgfx::ImageInfo::Make(kWidth, kHeight, tgfx::ColorType::RGBA_8888);
  Pixels pixels(static_cast<size_t>(info.byteSize()));
  surface->readPixels(info, pixels.data());
  device()->unlock();

  CHECK(pixelAt(pixels, 16, 16)[2] == 255);
  CHECK(pixelAt(pixels, 48, 48)[2] == 0);

  // pixelRatio scales the surface without moving the geometry.
  auto dense = tgfxrn::renderSceneOffscreen(program, {}, 64, 64, 2, /*opaque=*/false, /*displayP3=*/false, bytes);
  CHECK(dense != nullptr && dense->width() == 128 && dense->height() == 128);

  // An opaque render leaves nothing transparent, even where the scene drew nothing.
  auto solid = tgfxrn::renderSceneOffscreen(program, {}, 64, 64, 1, /*opaque=*/true, /*displayP3=*/false, bytes);
  CHECK(solid != nullptr);
  auto* solidContext = device()->lockContext();
  auto solidSurface = tgfx::Surface::Make(solidContext, kWidth, kHeight);
  solidSurface->getCanvas()->clear();
  solidSurface->getCanvas()->drawImage(solid);
  solidContext->flushAndSubmit(true);
  Pixels solidPixels(static_cast<size_t>(info.byteSize()));
  solidSurface->readPixels(info, solidPixels.data());
  device()->unlock();
  // Black where the rect did not reach, not transparent.
  CHECK(solidPixels[(48 * kWidth + 48) * 4 + 3] == 255);

  // A decoded table, not the packed delta setScene takes: index 0 is the first string.
  Writer stringy;
  stringy.push(1);
  stringy.push(6);  // Path
  stringy.push(0);  // string index 0
  stringy.push(0);  // no path effects
  stringy.pushPaint(0xff00ff00, false);
  size_t pathBytes = 0;
  auto drawn = tgfxrn::renderSceneOffscreen(stringy.finish(), {"M0 0 L48 0 L48 48 Z"}, 64, 64, 1,
                                            /*opaque=*/false, /*displayP3=*/false, pathBytes);
  CHECK(drawn != nullptr);
  auto* pathContext = device()->lockContext();
  auto pathSurface = tgfx::Surface::Make(pathContext, kWidth, kHeight);
  pathSurface->getCanvas()->clear();
  pathSurface->getCanvas()->drawImage(drawn);
  pathContext->flushAndSubmit(true);
  Pixels pathPixels(static_cast<size_t>(info.byteSize()));
  pathSurface->readPixels(info, pathPixels.data());
  device()->unlock();
  CHECK(pixelAt(pathPixels, 32, 16)[1] == 255);

  bool threw = false;
  try {
    tgfxrn::renderSceneOffscreen(program, {}, 0, 64, 1, /*opaque=*/false, /*displayP3=*/false, bytes);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
}

/// An offscreen render is colour-managed on both ends. Untagged, tgfx skips the
/// conversion entirely: the same sRGB red would land byte-identical on a P3
/// surface, and the pixels would come back with nothing saying which space they
/// are in for whatever draws them next.
void anOffscreenRenderCarriesItsColorSpace() {
  Writer w;
  w.push(1);
  w.push(1);  // Rect covering the whole surface
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(64);
  w.pushAnim(64);
  w.pushPaint(0xffff0000, false);  // opaque sRGB red, 0xAARRGGBB
  auto program = w.finish();

  size_t bytes = 0;
  auto srgb = tgfxrn::renderSceneOffscreen(program, {}, 64, 64, 1, /*opaque=*/true, /*displayP3=*/false, bytes);
  auto p3 = tgfxrn::renderSceneOffscreen(program, {}, 64, 64, 1, /*opaque=*/true, /*displayP3=*/true, bytes);
  CHECK(srgb != nullptr && p3 != nullptr);
  // Tagged at all: an untagged result is what the destination happens to assume.
  CHECK(srgb->colorSpace() != nullptr && srgb->colorSpace()->isSRGB());
  CHECK(p3->colorSpace() != nullptr && !p3->colorSpace()->isSRGB());

  // The same red, encoded for a wider gamut, is a different triple.
  auto readInto = [&](const std::shared_ptr<tgfx::Image>& image,
                      const std::shared_ptr<tgfx::ColorSpace>& space) {
    auto* context = device()->lockContext();
    auto surface = tgfx::Surface::Make(context, kWidth, kHeight, false, 1, false, 0, space);
    CHECK(surface != nullptr);
    surface->getCanvas()->clear();
    surface->getCanvas()->drawImage(image);
    context->flushAndSubmit(true);
    auto info = tgfx::ImageInfo::Make(kWidth, kHeight, tgfx::ColorType::RGBA_8888,
                                      tgfx::AlphaType::Premultiplied, 0, space);
    Pixels out(static_cast<size_t>(info.byteSize()));
    surface->readPixels(info, out.data());
    device()->unlock();
    return out;
  };

  // Each read back in its own space, so neither draw converts: the encodings
  // themselves have to differ, which is the surface having done the conversion
  // on the way in. Untagged, both come back as the same raw sRGB red.
  auto asSrgb = readInto(srgb, tgfx::ColorSpace::SRGB());
  auto asP3 = readInto(p3, tgfx::ColorSpace::DisplayP3());
  CHECK(pixelAt(asSrgb, 32, 32)[0] == 255 && pixelAt(asSrgb, 32, 32)[1] == 0);
  // sRGB red gamma-encoded for P3's wider primaries is (234, 51, 35): less red,
  // and some green with it. Interpreting it as P3 instead would clamp back to
  // (255, 0, 0), so the figures are pinned rather than only bounded.
  CHECK(std::abs(static_cast<int>(pixelAt(asP3, 32, 32)[0]) - 234) <= 2);
  CHECK(std::abs(static_cast<int>(pixelAt(asP3, 32, 32)[1]) - 51) <= 2);
  CHECK(std::abs(static_cast<int>(pixelAt(asP3, 32, 32)[2]) - 35) <= 2);

  // And they are the same colour: drawn into one space, both land on the same pixel.
  auto viaSrgb = readInto(srgb, tgfx::ColorSpace::DisplayP3());
  auto viaP3 = readInto(p3, tgfx::ColorSpace::DisplayP3());
  for (int channel = 0; channel < 3; channel++) {
    int delta = static_cast<int>(pixelAt(viaSrgb, 32, 32)[channel]) -
                static_cast<int>(pixelAt(viaP3, 32, 32)[channel]);
    // Two tolerances' worth: these came through different conversions — a paint
    // colour on one side, a texel on the other — into an 8-bit target.
    CHECK(delta <= 2 && delta >= -2);
  }
}

/// An embedded ICC profile that is neither sRGB nor Display-P3 — Adobe RGB here —
/// has to survive the decode and be converted on draw. Dropped, the same bytes
/// render with sRGB's primaries, which is a different colour and not an obvious
/// one: a screenshot looks plausible, so this asserts the pixels instead.
void anEmbeddedIccProfileSurvivesTheDecode() {
  auto adobe = tgfx::ColorSpace::MakeRGB(tgfx::NamedTransferFunction::TwoDotTwo,
                                         tgfx::NamedGamut::AdobeRGB);
  CHECK(adobe != nullptr);
  CHECK(!adobe->isSRGB());

  // The same bytes twice, tagged two ways: only the profile differs, so any
  // difference downstream is the profile being read and honoured.
  // kWidth x kHeight: pixelAt strides by kWidth, so a smaller surface reads past the end.
  constexpr int kSide = kWidth;
  std::vector<uint8_t> rgba(static_cast<size_t>(kSide) * kSide * 4);
  for (size_t i = 0; i < rgba.size(); i += 4) {
    rgba[i] = 60;       // a green whose red clips on the way into sRGB, so the
    rgba[i + 1] = 180;  // conversion cannot come out as a no-op
    rgba[i + 2] = 75;
    rgba[i + 3] = 255;
  }
  auto encode = [&](const std::shared_ptr<tgfx::ColorSpace>& space,
                    tgfx::EncodedFormat format) {
    auto info = tgfx::ImageInfo::Make(kSide, kSide, tgfx::ColorType::RGBA_8888,
                                      tgfx::AlphaType::Unpremultiplied, 0, space);
    tgfx::Pixmap pixmap(info, rgba.data());
    return tgfx::ImageCodec::Encode(pixmap, format, 100);
  };
  auto wide = encode(adobe, tgfx::EncodedFormat::PNG);
  auto plain = encode(tgfx::ColorSpace::SRGB(), tgfx::EncodedFormat::PNG);
  CHECK(wide != nullptr && plain != nullptr);
  if (wide == nullptr || plain == nullptr) {
    return;
  }

  // Through the file and back: the decoder has to read iCCP, not guess.
  auto wideCodec = tgfx::ImageCodec::MakeFrom(wide);
  auto plainCodec = tgfx::ImageCodec::MakeFrom(plain);
  CHECK(wideCodec != nullptr && plainCodec != nullptr);
  if (wideCodec == nullptr || plainCodec == nullptr) {
    return;
  }
  CHECK(wideCodec->colorSpace() != nullptr && !wideCodec->colorSpace()->isSRGB());
  CHECK(plainCodec->colorSpace() != nullptr && plainCodec->colorSpace()->isSRGB());

  // And through the registry, which is what a scene actually draws.
  auto& registry = tgfxrn::ImageRegistry::instance();
  auto wideHandle = registry.add(wideCodec);
  auto plainHandle = registry.add(plainCodec);
  CHECK(wideHandle.handle != 0 && plainHandle.handle != 0);
  auto wideImage = registry.sourceImage(wideHandle.handle);
  auto plainImage = registry.sourceImage(plainHandle.handle);
  CHECK(wideImage != nullptr && plainImage != nullptr);
  if (wideImage == nullptr || plainImage == nullptr) {
    return;
  }

  auto drawn = [&](const std::shared_ptr<tgfx::Image>& image) {
    auto* context = device()->lockContext();
    auto surface = tgfx::Surface::Make(context, kSide, kSide, false, 1, false, 0,
                                       tgfx::ColorSpace::SRGB());
    CHECK(surface != nullptr);
    surface->getCanvas()->clear();
    surface->getCanvas()->drawImage(image);
    context->flushAndSubmit(true);
    auto info = tgfx::ImageInfo::Make(kSide, kSide, tgfx::ColorType::RGBA_8888,
                                      tgfx::AlphaType::Premultiplied, 0,
                                      tgfx::ColorSpace::SRGB());
    Pixels out(static_cast<size_t>(info.byteSize()));
    surface->readPixels(info, out.data());
    device()->unlock();
    return out;
  };

  auto fromWide = drawn(wideImage);
  auto fromPlain = drawn(plainImage);
  // The sRGB-tagged one lands on the bytes it started as; the Adobe RGB one does not.
  CHECK(pixelAt(fromPlain, 32, 32)[0] == 60);
  CHECK(pixelAt(fromPlain, 32, 32)[1] == 180);
  CHECK(pixelAt(fromPlain, 32, 32)[2] == 75);
  int spread = 0;
  for (int channel = 0; channel < 3; channel++) {
    spread += std::abs(static_cast<int>(pixelAt(fromWide, 32, 32)[channel]) -
                       static_cast<int>(pixelAt(fromPlain, 32, 32)[channel]));
  }
  // Adobe RGB's wider primaries and 2.2 gamma move this green a long way; a
  // dropped profile would leave the two byte-identical.
  CHECK(spread > 12);

  // The issue's own example is a tagged JPEG, and JPEG carries its profile in an
  // APP2 marker rather than a PNG chunk — a different reader on the same path.
  auto wideJpeg = tgfx::ImageCodec::MakeFrom(encode(adobe, tgfx::EncodedFormat::JPEG));
  CHECK(wideJpeg != nullptr);
  if (wideJpeg != nullptr) {
    CHECK(wideJpeg->colorSpace() != nullptr && !wideJpeg->colorSpace()->isSRGB());
    auto jpegHandle = registry.add(wideJpeg);
    CHECK(jpegHandle.handle != 0);
    auto jpegImage = registry.sourceImage(jpegHandle.handle);
    CHECK(jpegImage != nullptr);
    if (jpegImage != nullptr) {
      auto fromJpeg = drawn(jpegImage);
      // JPEG is lossy, so this is the conversion showing through the loss, not equality.
      int jpegSpread = 0;
      for (int channel = 0; channel < 3; channel++) {
        jpegSpread += std::abs(static_cast<int>(pixelAt(fromJpeg, 32, 32)[channel]) -
                               static_cast<int>(pixelAt(fromPlain, 32, 32)[channel]));
      }
      CHECK(jpegSpread > 12);
    }
    registry.remove(jpegHandle.handle);
  }

  // An image with no profile at all is read as sRGB, which is the web's default
  // and what tgfx substitutes for a null source space. This is the "no profile"
  // half of the same guarantee, and it is why the registry does not re-tag: the
  // conversion already treats untagged as sRGB, so tagging it would change no
  // pixel and would cost a second decode on every platform-decoded image.
  auto untagged = tgfx::ImageCodec::MakeFrom(
      tgfx::ImageInfo::Make(kSide, kSide, tgfx::ColorType::RGBA_8888,
                            tgfx::AlphaType::Unpremultiplied),
      tgfx::Data::MakeWithCopy(rgba.data(), rgba.size()));
  CHECK(untagged != nullptr);
  if (untagged != nullptr) {
    CHECK(untagged->colorSpace() == nullptr);
    auto untaggedHandle = registry.add(untagged);
    CHECK(untaggedHandle.handle != 0);
    auto untaggedImage = registry.sourceImage(untaggedHandle.handle);
    CHECK(untaggedImage != nullptr);
    if (untaggedImage != nullptr) {
      auto fromUntagged = drawn(untaggedImage);
      // Byte-identical to the sRGB-tagged draw: read as sRGB, not as Adobe RGB
      // and not as the destination's space.
      for (int channel = 0; channel < 3; channel++) {
        CHECK(pixelAt(fromUntagged, 32, 32)[channel] == pixelAt(fromPlain, 32, 32)[channel]);
      }
      registry.remove(untaggedHandle.handle);
    }
  }

  // A derived handle keeps the profile too: a thumbnail that lost it would be a
  // different colour from the image it came from.
  auto scaled = wideImage->makeScaled(kSide / 2, kSide / 2);
  CHECK(scaled != nullptr);
  if (scaled != nullptr) {
    CHECK(scaled->colorSpace() != nullptr && !scaled->colorSpace()->isSRGB());
  }
  auto oriented = wideImage->makeOriented(tgfx::Orientation::RightTop);
  CHECK(oriented != nullptr);
  if (oriented != nullptr) {
    CHECK(oriented->colorSpace() != nullptr && !oriented->colorSpace()->isSRGB());
  }

  // The registry is process-wide and the byte-budget test counts what is in it.
  registry.remove(wideHandle.handle);
  registry.remove(plainHandle.handle);
}

/// A derived image must outlive its parent's eviction, so it comes from the codec.
void aDerivedImageSurvivesTheParentsEviction() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  size_t restore = registry.cacheLimit();
  auto parent = registerSolid({0, 0, 255, 255}, 16, 16);
  CHECK(parent.handle != 0);

  auto source = registry.sourceImage(parent.handle);
  CHECK(source != nullptr && source->width() == 16);
  // One decode chain per parent: a fresh MakeFrom would mint a second GPU key and texture.
  CHECK(registry.sourceImage(parent.handle) == source);
  auto half = source->makeScaled(8, 8, {});
  CHECK(half != nullptr && half->width() == 8 && half->height() == 8);
  auto crop = source->makeSubset(tgfx::Rect::MakeXYWH(0, 0, 4, 4));
  CHECK(crop != nullptr && crop->width() == 4);

  // Registered as external, so it is charged nothing and never evicted itself.
  auto derived = registry.addExternal(half);
  CHECK(derived.handle != 0 && registry.byteSize(derived.handle) == 0);

  // Squeeze the budget until the parent's pixels go, then draw the derivative.
  registry.setCacheLimit(1);
  for (int i = 0; i < 4; i++) {
    registerSolid(cellColor(i), 16, 16);
  }
  auto* context = device()->lockContext();
  auto surface = tgfx::Surface::Make(context, kWidth, kHeight);
  surface->getCanvas()->clear();
  surface->getCanvas()->drawImage(registry.acquire(derived.handle));
  context->flushAndSubmit(true);
  auto info = tgfx::ImageInfo::Make(kWidth, kHeight, tgfx::ColorType::RGBA_8888);
  Pixels pixels(static_cast<size_t>(info.byteSize()));
  surface->readPixels(info, pixels.data());
  device()->unlock();
  CHECK(pixelAt(pixels, 4, 4)[2] == 255);

  registry.setCacheLimit(restore);
}

/// A recorded subtree replays; the flag is only ever set on one with nothing animated.
void aRecordedGroupReplaysWhatItDrew() {
  auto draw = [](bool record) {
    Writer w;
    w.push(1);
    w.push(13);  // GroupLayer
    w.push(0);   // no transform ops
    w.push(record ? 8 : 0);
    w.push(2);  // two children
    w.push(1);  // Rect
    w.pushAnim(8);
    w.pushAnim(8);
    w.pushAnim(24);
    w.pushAnim(24);
    w.pushPaint(0xff0000ff, false);
    w.push(4);  // Circle
    w.pushAnim(44);
    w.pushAnim(44);
    w.pushAnim(12);
    w.pushPaint(0xffff0000, false);
    return w.finish();
  };

  auto direct = render(draw(false), {}, {});
  auto recorded = render(draw(true), {}, {});
  CHECK(direct == recorded);

  // A slot the writer deduped still looks static in JS, so the reader has to catch it.
  {
    Writer w;
    w.push(1);
    w.push(13);
    w.push(0);
    w.push(8);
    w.push(1);
    w.push(1);  // Rect
    w.pushAnim(8, 0);
    w.pushAnim(8);
    w.pushAnim(24);
    w.pushAnim(24);
    w.pushPaint(0xff0000ff, false);
    auto program = w.finish();
    tgfxrn::SceneRenderer moving;
    moving.setScene(program, {});
    // Drawn directly, so the slot still moves the rect between frames.
    CHECK(drawFrame(moving, {8}) != drawFrame(moving, {40}));
  }

  // An image whose decode has not arrived is not the same next frame either.
  {
    auto& registry = tgfxrn::ImageRegistry::instance();
    size_t restore = registry.cacheLimit();
    auto image = registerSolid({0, 255, 0, 255}, 16, 16);
    registry.setCacheLimit(1);
    for (int i = 0; i < 4; i++) {
      registerSolid(cellColor(i), 16, 16);
    }
    Writer w;
    w.push(1);
    w.push(13);
    w.push(0);
    w.push(8);
    w.push(1);
    w.pushImage(image.handle, 8, 8, 32, 32);
    tgfxrn::SceneRenderer renderer;
    renderer.setScene(w.finish(), {});
    registry.waitForDecodes();
    // It draws once the decode lands, which a recording taken while it was missing would not.
    auto later = drawFrame(renderer, {});
    CHECK(inkPixels(later) > 0);
    registry.setCacheLimit(restore);
  }

  // And the replay is stable: a second frame off the same program draws the same thing.
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(draw(true), {});
  auto first = drawFrame(renderer, {});
  auto second = drawFrame(renderer, {});
  CHECK(first == second && first == recorded);
}

/// Front-to-back, through the group transform stack, in the node's own space.
void hitTestReportsWhatIsUnderThePoint() {
  Writer w;
  w.push(2);
  // A rect at 0,0 32x32 under a group translated by 20, holding a circle at 10,10 r8.
  w.push(1);
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(32);
  w.pushAnim(32);
  w.pushPaint(0xff0000ff, false);
  w.push(0);  // Group
  w.push(1);  // one transform op
  w.push(0);  // translate
  w.pushAnim(20);
  w.pushAnim(0);
  w.push(1);  // one child
  w.push(4);  // Circle
  w.pushAnim(10);
  w.pushAnim(10);
  w.pushAnim(8);
  w.pushPaint(0xffff0000, false);

  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});

  // Inside both: the circle is drawn later, so it comes first.
  auto both = renderer.hitTest(30, 10, false, {});
  CHECK(both.size() == 5);
  CHECK(both[0] == 2 && both[1] == 1 && both[2] == 0);  // group 1, child 0
  CHECK(both[3] == 1 && both[4] == 0);                  // then the rect

  // Inside the rect only.
  auto rectOnly = renderer.hitTest(4, 28, false, {});
  CHECK(rectOnly.size() == 2 && rectOnly[0] == 1 && rectOnly[1] == 0);

  // Outside everything.
  CHECK(renderer.hitTest(200, 200, false, {}).empty());

  // The group's translate has to be undone: without it 10,10 would land on the circle.
  auto shifted = renderer.hitTest(10, 10, false, {});
  CHECK(shifted.size() == 2 && shifted[1] == 0);

  // A clip bounds the tap target: nothing outside it is reported, subtree included.
  {
    Writer c;
    c.push(1);
    c.push(13);  // GroupLayer
    c.push(0);   // no transform ops
    c.push(1);   // clip, no invert/layer/picture
    c.push(1);   // rect clip
    c.push(0);
    c.push(0);
    c.push(10);
    c.push(10);
    c.push(1);  // one child
    c.push(1);  // Rect
    c.pushAnim(0);
    c.pushAnim(0);
    c.pushAnim(100);
    c.pushAnim(100);
    c.pushPaint(0xff0000ff, false);
    tgfxrn::SceneRenderer clipped;
    clipped.setScene(c.finish(), {});
    CHECK(!clipped.hitTest(5, 5, false, {}).empty());
    // Inside the rect but outside the clip, where nothing is drawn.
    CHECK(clipped.hitTest(50, 50, false, {}).empty());
  }

  // 23,3 sits in the circle's bounding box but outside the disc, and inside the rect.
  auto boxed = renderer.hitTest(23, 3, false, {});
  auto exact = renderer.hitTest(23, 3, true, {});
  CHECK(boxed.size() == 5);  // the circle, then the rect
  CHECK(exact.size() == 2);  // only the rect
  CHECK(exact[0] == 1 && exact[1] == 0);
}

/// The root viewport is the same matrix the renderer draws under, so the walk
/// has to invert it too — otherwise a zoomed scene reports hits from where the
/// nodes used to be.
/// The composition order itself, which no pixel test off a device can catch: get
/// preTranslate and preScale the wrong way round and the pan starts scaling with
/// the zoom, which looks plausible until a scene is both zoomed and panned.
void theRootMatrixPansInCanvasPointsAfterTheZoom() {
  // p -> pixelRatio * (offset + zoom * p)
  auto matrix = tgfxrn::rootMatrix(2, 3, 10, -4);
  auto mapped = matrix.mapXY(5, 6);
  CHECK(std::abs(mapped.x - 2 * (10 + 3 * 5)) < 1e-4f);
  CHECK(std::abs(mapped.y - 2 * (-4 + 3 * 6)) < 1e-4f);

  // The offset does not scale with the zoom: the origin lands on it exactly.
  auto atOrigin = tgfxrn::rootMatrix(1, 4, 30, 40).mapXY(0, 0);
  CHECK(std::abs(atOrigin.x - 30) < 1e-4f && std::abs(atOrigin.y - 40) < 1e-4f);

  // The identity triple is the identity, so an unset viewport changes nothing.
  auto identity = tgfxrn::rootMatrix(1, 1, 0, 0);
  CHECK(identity.isIdentity());

  // A focal-point pinch holds the scene point under the focus still.
  float zoom = 1.5f;
  float offset = 12;
  float focus = 80;
  float scene = (focus - offset) / zoom;
  float next = 4;
  auto pinched = tgfxrn::rootMatrix(1, next, focus - next * scene, 0);
  CHECK(std::abs(pinched.mapXY(scene, 0).x - focus) < 1e-3f);
}

void hitTestFollowsTheViewport() {
  Writer w;
  w.push(1);
  // One 32x32 rect at the origin.
  w.push(1);
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(32);
  w.pushAnim(32);
  w.pushPaint(0xff0000ff, false);

  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});

  // zoom 2 about the origin: the rect covers 0..64 on screen.
  auto zoom = tgfx::Matrix::MakeScale(2, 2);
  CHECK(!renderer.hitTest(60, 60, false, {}, {}, {}, zoom).empty());
  CHECK(renderer.hitTest(60, 60, false, {}).empty());
  // 20,20 is still inside at either scale, and 70,70 outside at both.
  CHECK(!renderer.hitTest(20, 20, false, {}, {}, {}, zoom).empty());
  CHECK(renderer.hitTest(70, 70, false, {}, {}, {}, zoom).empty());

  // translate(offset) * scale(zoom): the pan is in canvas points, after the zoom.
  auto panned = tgfx::Matrix::MakeTrans(100, 0);
  panned.preScale(2, 2);
  CHECK(!renderer.hitTest(110, 10, false, {}, {}, {}, panned).empty());
  CHECK(renderer.hitTest(10, 10, false, {}, {}, {}, panned).empty());

  // A degenerate root collapses the scene, so nothing is under any point.
  CHECK(renderer.hitTest(10, 10, false, {}, {}, {},
                         tgfx::Matrix::MakeScale(0, 0))
            .empty());
}

/// A PDF page is vectors, not pixels: the file exists, is a PDF, and holds every page.
void aPdfIsWrittenWithAPagePerScene() {
  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(10);
  w.pushAnim(10);
  w.pushAnim(80);
  w.pushAnim(60);
  w.pushPaint(0xff3366cc, false);
  auto program = w.finish();

  std::string path = std::string(TGFXRN_TEST_TMP_DIR) + "/scene.pdf";
  std::remove(path.c_str());
  tgfxrn::PdfMetadata info;
  info.title = "tgfx test";
  info.author = "SceneTest";
  tgfxrn::writePdf(path, {{program, {}, 200, 120}, {program, {}, 120, 200}}, info);

  std::ifstream file(path, std::ios::binary);
  CHECK(file.good());
  std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  CHECK(bytes.size() > 400);
  CHECK(bytes.rfind("%PDF-", 0) == 0);
  CHECK(bytes.find("%%EOF") != std::string::npos);
  // Two pages, and the metadata rode along.
  CHECK(bytes.find("/Count 2") != std::string::npos);
  CHECK(bytes.find("tgfx test") != std::string::npos);
  std::remove(path.c_str());

  // A failure must leave whatever was already there alone, not a truncated file.
  {
    std::ofstream seed(path, std::ios::binary);
    seed << "not a pdf, but somebody's";
    seed.close();
    bool threw = false;
    try {
      tgfxrn::writePdf(path, {{program, {}, 200, 120}, {program, {}, 0, 120}}, {});
    } catch (const std::runtime_error&) {
      threw = true;
    }
    CHECK(threw);
    std::ifstream after(path, std::ios::binary);
    std::string kept((std::istreambuf_iterator<char>(after)), std::istreambuf_iterator<char>());
    CHECK(kept == "not a pdf, but somebody's");
  }
  std::remove(path.c_str());

  // And with nothing there before, a failure leaves nothing behind.
  bool threw = false;
  try {
    tgfxrn::writePdf(path, {{program, {}, 0, 120}}, {});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
  std::ifstream missing(path, std::ios::binary);
  CHECK(!missing.good());
  std::remove((path + ".tgfx-partial").c_str());
}

/// A backdrop reads what is already on the surface, so order is what makes it one.
void aBackdropFrostsWhatIsBeneathIt() {
  auto program = [](double refraction) {
    Writer w;
    w.push(2);
    // A hard-edged blue/red split underneath, so a blur has something to smear.
    w.push(1);
    w.pushAnim(0);
    w.pushAnim(0);
    w.pushAnim(32);
    w.pushAnim(64);
    w.pushPaint(0xff0000ff, false);
    w.push(17);  // Backdrop
    w.pushAnim(8);
    w.pushAnim(8);
    w.pushAnim(48);
    w.pushAnim(48);
    w.push(0);               // corner radius
    w.pushAnim(10);          // blur
    w.pushAnim(refraction);  // 0 is a plain blur, above 0 is glass
    w.pushAnim(20);
    w.pushAnim(20);
    w.pushAnim(0);
    w.pushAnim(0);   // splay
    w.pushAnim(45);  // light angle
    w.pushAnim(50);  // light intensity
    w.pushPaint(0xffffffff, false);
    return w.finish();
  };

  auto plain = render(program(0), {}, {});
  CHECK(inkPixels(plain) > 0);

  // Without the backdrop the split stays hard; with it the panel samples across the edge.
  Writer bare;
  bare.push(1);
  bare.push(1);
  bare.pushAnim(0);
  bare.pushAnim(0);
  bare.pushAnim(32);
  bare.pushAnim(64);
  bare.pushPaint(0xff0000ff, false);
  auto unfrosted = render(bare.finish(), {}, {});
  CHECK(plain != unfrosted);

  // Just right of the rect's edge, inside the panel: the blur has pulled blue across.
  auto pulled = pixelAt(plain, 36, 30);
  CHECK(pulled[2] > 0);

  // Glass takes a different path through tgfx, so it must not draw the same thing.
  auto glassy = render(program(40), {}, {});
  CHECK(glassy != plain);
}

/// Film grain is a NoiseStyle over the group's own pixels, so it has to be
/// clipped to what the group drew and has to answer to every one of its words.
void filmGrainDustsOnlyWhatTheGroupDrew() {
  // mode: 0 mono, 1 duo, 2 multi. A 40x40 blue square at (8,8), grained.
  auto program = [](int mode, double density, double size, double seed) {
    Writer w;
    w.push(1);
    w.push(13);  // GroupLayer
    w.push(0);   // no transform ops
    w.push(16);  // flags: grain
    w.push(mode);
    w.pushAnim(size);
    w.pushAnim(density);
    w.pushAnim(seed);
    w.pushAnim(3);  // srcOver
    if (mode == 2) {
      w.pushAnim(1);  // opacity
    } else {
      w.push(0xffffffff);  // white grain
      w.push(0);           // static colour
      if (mode == 1) {
        w.push(0xff000000);  // second colour: black
        w.push(0);
      }
    }
    w.push(1);  // one child
    w.push(1);  // Rect
    w.pushAnim(8);
    w.pushAnim(8);
    w.pushAnim(40);
    w.pushAnim(40);
    w.pushPaint(0xff0000ff, false);
    return w.finish();
  };

  // The same group with no grain block at all.
  Writer bare;
  bare.push(1);
  bare.push(13);
  bare.push(0);
  bare.push(0);
  bare.push(1);
  bare.push(1);
  bare.pushAnim(8);
  bare.pushAnim(8);
  bare.pushAnim(40);
  bare.pushAnim(40);
  bare.pushPaint(0xff0000ff, false);
  auto plain = render(bare.finish(), {}, {});
  CHECK(inkPixels(plain) == 40 * 40);

  auto mono = render(program(0, 0.5, 4, 0), {}, {});
  CHECK(mono != plain);
  // Clipped to the content's alpha: outside the rect nothing is dusted, and the
  // grain adds no coverage of its own inside it either.
  CHECK(inkPixels(mono) == inkPixels(plain));
  CHECK(!opaqueAt(mono, 2, 2));

  // White grain over blue can only lighten, never darken.
  size_t lightened = 0;
  for (int y = 8; y < 48; y++) {
    for (int x = 8; x < 48; x++) {
      auto grained = pixelAt(mono, x, y);
      auto flat = pixelAt(plain, x, y);
      CHECK(grained[0] >= flat[0]);
      if (grained[0] > flat[0]) {
        lightened++;
      }
    }
  }
  CHECK(lightened > 0);

  // Density is the proportion of pixels the grain reaches, so zero is off.
  CHECK(render(program(0, 0, 4, 0), {}, {}) == plain);

  // Every remaining word has to reach NoiseStyle, or it is decoration.
  CHECK(render(program(0, 0.5, 16, 0), {}, {}) != mono);  // coarser grain
  CHECK(render(program(0, 0.5, 4, 7), {}, {}) != mono);   // a different pattern
  CHECK(render(program(0, 0.9, 4, 0), {}, {}) != mono);   // denser
  // Duo fills the complementary half too, and multi keeps the noise's own RGB.
  auto duo = render(program(1, 0.5, 4, 0), {}, {});
  auto multi = render(program(2, 0.5, 4, 0), {}, {});
  CHECK(duo != mono);
  CHECK(multi != mono);
  CHECK(multi != duo);
}

/// A recorder canvas has an identity matrix and no surface, so grain inside a
/// recorded group would be dusted at the wrong scale and then frozen. The grain
/// block counts as dynamic for exactly that reason; this is the proof.
void aRecordedGroupHoldingGrainIsNotRecorded() {
  Writer w;
  w.push(1);
  w.push(13);  // outer GroupLayer
  w.push(0);
  w.push(8);   // <Group picture>
  w.push(1);   // one child
  w.push(13);  // inner GroupLayer
  w.push(0);
  w.push(16);  // grain
  w.push(0);   // mono
  w.pushAnim(4);
  w.pushAnim(0.6);
  w.pushAnim(0, 0);  // seed, slot 0
  w.pushAnim(3);
  w.push(0xff000000);
  w.push(0);
  w.push(1);  // one child
  w.push(1);  // Rect
  w.pushAnim(8);
  w.pushAnim(8);
  w.pushAnim(40);
  w.pushAnim(40);
  w.pushPaint(0xffffffff, false);
  auto program = w.finish();

  tgfxrn::SceneRenderer renderer;
  renderer.setScene(program, {});
  auto first = drawFrame(renderer, {0});
  auto second = drawFrame(renderer, {11});
  // A recording would replay the first frame's grain for ever.
  CHECK(first != second);
}

/// The two grain words the mode/size/seed test cannot reach: the blend mode,
/// and the live-colour branch of readGrainColor, which is the one cursor path
/// in the block that changes how many words follow.
void filmGrainAnswersToItsBlendModeAndLiveColour() {
  // Mode 0 with the colour as four live channels behind the packed word.
  auto liveColor = [](int blendMode) {
    Writer w;
    w.push(1);
    w.push(13);  // GroupLayer
    w.push(0);
    w.push(16);  // grain
    w.push(0);   // mono
    w.pushAnim(4);
    w.pushAnim(0.6);
    w.pushAnim(0);
    w.pushAnim(blendMode);
    w.push(0xff000000);  // packed, ignored once the live flag is set
    w.push(1);           // live colour
    w.pushAnim(0, 0);    // red
    w.pushAnim(0, 1);    // green
    w.pushAnim(0, 2);    // blue
    w.pushAnim(1, 3);    // alpha
    w.push(1);           // one child
    w.push(1);           // Rect
    w.pushAnim(8);
    w.pushAnim(8);
    w.pushAnim(40);
    w.pushAnim(40);
    w.pushPaint(0xff808080, false);
    return w.finish();
  };

  auto program = liveColor(3);  // srcOver
  auto black = render(program, {}, {0, 0, 0, 1});
  auto white = render(program, {}, {1, 1, 1, 1});
  auto clear = render(program, {}, {0, 0, 0, 0});
  // The channels reach the style: black grain darkens, white lightens, and a
  // zero alpha leaves the grey rect exactly as it drew itself.
  CHECK(black != white);
  CHECK(pixelAt(black, 16, 16)[0] <= 0x80);
  CHECK(pixelAt(white, 16, 16)[0] >= 0x80);
  CHECK(inkPixels(clear) == 40 * 40);

  // The live branch reads four extra words; the reader has to consume exactly
  // those, or the child rect that follows would be read from the wrong offset.
  CHECK(opaqueAt(black, 16, 16));
  CHECK(!opaqueAt(black, 2, 2));

  // Plus-lighter cannot darken, so it has to differ from srcOver black grain.
  auto lighter = render(liveColor(12), {}, {0, 0, 0, 1});
  CHECK(lighter != black);
}

/// Grain is rebuilt per frame from the buffer, so a handle drives it without a
/// re-encode. Every word that shapes the pattern has to be slot-capable.
void filmGrainResolvesFromValueSlots() {
  Writer w;
  w.push(1);
  w.push(13);
  w.push(0);
  w.push(16);
  w.push(0);  // mono
  w.pushAnim(4);
  w.pushAnim(0, 0);  // density
  w.pushAnim(0, 1);  // seed
  w.pushAnim(3);
  w.push(0xffffffff);
  w.push(0);
  w.push(1);
  w.push(1);
  w.pushAnim(8);
  w.pushAnim(8);
  w.pushAnim(40);
  w.pushAnim(40);
  w.pushPaint(0xff0000ff, false);
  auto words = w.finish();

  auto off = render(words, {}, {0, 0});
  auto on = render(words, {}, {0.6, 0});
  auto reseeded = render(words, {}, {0.6, 9});
  CHECK(off != on);
  CHECK(reseeded != on);
  // Density zero leaves the group exactly as it drew itself.
  CHECK(inkPixels(off) == 40 * 40);
}

/// The three light words scene 28 added. Hardcoded, a <Glass> could only ever be
/// lit from one direction at one brightness; each has to reach GlassStyle::Make
/// and change what is drawn, or the props are decoration.
void glassTakesItsLightFromTheScene() {
  auto program = [](double splay, double lightAngle, double lightIntensity) {
    Writer w;
    w.push(2);
    // A blue rect over transparency, so the panel's edges have an edge to catch.
    w.push(1);
    w.pushAnim(0);
    w.pushAnim(0);
    w.pushAnim(32);
    w.pushAnim(64);
    w.pushPaint(0xff0000ff, false);
    w.push(17);  // Backdrop
    w.pushAnim(8);
    w.pushAnim(8);
    w.pushAnim(48);
    w.pushAnim(48);
    w.push(12);      // corner radius, so there is a curved edge to refract along
    w.pushAnim(10);  // blur, ignored on the glass path
    w.pushAnim(60);  // refraction: above 0 is glass
    w.pushAnim(30);  // depth
    w.pushAnim(20);  // frost
    w.pushAnim(0);   // dispersion
    w.pushAnim(splay);
    w.pushAnim(lightAngle);
    w.pushAnim(lightIntensity);
    w.pushPaint(0xffffffff, false);
    return w.finish();
  };

  // The same scene with no panel at all, so "the glass drew something" is a
  // comparison rather than a pixel count the rect underneath would satisfy.
  Writer bare;
  bare.push(1);
  bare.push(1);
  bare.pushAnim(0);
  bare.pushAnim(0);
  bare.pushAnim(32);
  bare.pushAnim(64);
  bare.pushPaint(0xff0000ff, false);
  auto unpanelled = render(bare.finish(), {}, {});

  // The defaults the compiler writes when a <Glass> names none of the three.
  auto base = render(program(0, 45, 50), {}, {});
  CHECK(base != unpanelled);

  // Each one alone has to move the frame.
  auto splayed = render(program(100, 45, 50), {}, {});
  auto turned = render(program(0, -135, 50), {}, {});
  auto unlit = render(program(0, 45, 0), {}, {});
  CHECK(splayed != base);  // splay
  CHECK(turned != base);   // light from the other side
  CHECK(unlit != base);    // highlight off entirely

  // And they are three separate words: transposing any two in the reader would
  // leave every check above green, because each variant still differs from base.
  CHECK(splayed != turned);
  CHECK(splayed != unlit);
  CHECK(turned != unlit);

  // An unlit panel is not an absent one: refraction and frost stay.
  CHECK(unlit != unpanelled);
}

/// Every material word is a slot, so a handle drives the glass without a
/// re-encode. Written as a literal in the program and patched by a value.
void theGlassMaterialResolvesFromValueSlots() {
  Writer w;
  w.push(2);
  w.push(1);
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(32);
  w.pushAnim(64);
  w.pushPaint(0xff0000ff, false);
  w.push(17);  // Backdrop
  w.pushAnim(8);
  w.pushAnim(8);
  w.pushAnim(48);
  w.pushAnim(48);
  w.push(12);         // corner radius, static
  w.pushAnim(10);     // blur
  w.pushAnim(60);     // refraction
  w.pushAnim(30);     // depth
  w.pushAnim(20);     // frost
  w.pushAnim(0);      // dispersion
  w.pushAnim(0);      // splay
  w.pushAnim(45, 0);  // light angle, slot 0
  w.pushAnim(50, 1);  // light intensity, slot 1
  w.pushPaint(0xffffffff, false);
  auto program = w.finish();

  auto asWritten = render(program, {}, {45, 50});
  // Only the slots move; the program is byte-identical.
  auto turned = render(program, {}, {-135, 50});
  auto unlit = render(program, {}, {45, 0});
  CHECK(turned != asWritten);
  CHECK(unlit != asWritten);
  CHECK(turned != unlit);
}

/// The example app's own geometry: a wide canvas at a phone's pixel ratio, with a panel big
/// enough that the blur radius is a real number of pixels. A 64x64 scratch surface hides both.
void aBackdropOnACanvasTheSizeOfARealOne() {
  const int kW = 1110;
  const int kH = 390;
  const float kRatio = 3;
  Writer w;
  w.push(3);
  // A hard black edge at x=80, which a 14pt blur must smear across inside the panel.
  w.push(1);
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(80);
  w.pushAnim(120);
  w.pushPaint(0xff000000, false);
  w.push(1);
  w.pushAnim(80);
  w.pushAnim(0);
  w.pushAnim(280);
  w.pushAnim(120);
  w.pushPaint(0xff808080, false);
  w.push(17);  // Backdrop
  w.pushAnim(12);
  w.pushAnim(24);
  w.pushAnim(150);
  w.pushAnim(72);
  w.push(18);      // corner radius
  w.pushAnim(14);  // blur
  w.pushAnim(0);   // refraction: a plain frosted panel
  w.pushAnim(20);
  w.pushAnim(20);
  w.pushAnim(0);
  w.pushAnim(0);   // splay
  w.pushAnim(45);  // light angle
  w.pushAnim(50);  // light intensity
  w.pushPaint(0xffffffff, false);
  auto drawn = render(w.finish(), {}, {}, kRatio, kW, kH);

  auto at = [&](int x, int y) {
    size_t i = (static_cast<size_t>(y) * kW + static_cast<size_t>(x)) * 4;
    return std::array<uint8_t, 3>{drawn[i], drawn[i + 1], drawn[i + 2]};
  };
  // Just right of the black edge: outside the panel it stays grey, inside the blur darkens it.
  auto above = at(250, 30);
  auto inside = at(250, 180);
  CHECK(above[0] > 100);
  CHECK(inside[0] < above[0] - 20);
  // Just left of it, the opposite: grey pulled into the black.
  CHECK(at(230, 180)[0] > at(230, 30)[0] + 20);
  // 40px right of the edge, which only a blur scaled by the pixel ratio still reaches: at this
  // ratio that reads 108, and a sigma left in scene points reads the grey it never darkened.
  CHECK(at(280, 180)[0] < 118);
}

/// A panel hanging off the surface gets a cropped backdrop, and glass maps its shape across
/// whatever it is given, so the crop has to move the shape rather than squash it.
void aBackdropHangingOffTheEdgeStillDrawsWhereItOverlaps() {
  // A hard blue/red split at x=8, which is inside the part of a left-hanging panel that still
  // lands on the surface, so the panel has something to smear.
  auto program = [](double refraction, double x, bool panel) {
    Writer w;
    w.push(panel ? 3 : 2);
    w.push(1);
    w.pushAnim(0);
    w.pushAnim(0);
    w.pushAnim(8);
    w.pushAnim(64);
    w.pushPaint(0xff0000ff, false);
    w.push(1);
    w.pushAnim(8);
    w.pushAnim(0);
    w.pushAnim(56);
    w.pushAnim(64);
    w.pushPaint(0xffff0000, false);
    if (panel) {
      w.push(17);  // Backdrop
      w.pushAnim(x);
      w.pushAnim(8);
      w.pushAnim(40);
      w.pushAnim(40);
      w.push(8);
      w.pushAnim(6);
      w.pushAnim(refraction);
      w.pushAnim(20);
      w.pushAnim(20);
      w.pushAnim(0);
      w.pushAnim(0);   // splay
      w.pushAnim(45);  // light angle
      w.pushAnim(50);  // light intensity
      w.pushPaint(0xffffffff, false);
    }
    return w.finish();
  };

  for (double refraction : {0.0, 40.0}) {
    auto bare = render(program(refraction, 0, false), {}, {});
    // Straddling the left edge: it still draws over the part that overlaps the surface.
    auto hanging = render(program(refraction, -20, true), {}, {});
    CHECK(hanging != bare);
    CHECK(opaqueAt(hanging, 4, 20));
    // Near the surface edge the panel's own corner is off-screen, so the shape is straight-edged
    // there. A shape cropped to the visible slice would round it off and leave this pixel bare.
    CHECK(pixelAt(hanging, 1, 10) != pixelAt(bare, 1, 10));
    // Entirely off the surface: nothing to sample, so the scene is left exactly as it was.
    CHECK(render(program(refraction, -60, true), {}, {}) == bare);
  }
}

/// The device differs from a scratch surface in two ways, and both broke this: it draws at a
/// pixel ratio, and its content is an image still on its way to the GPU when the panel samples.
void aBackdropAtAPixelRatioLeavesTheRestOfTheSceneAlone() {
  auto photo = registerSolid({0, 0, 255, 255});
  auto program = [&](bool panel, double refraction) {
    Writer w;
    w.push(panel ? 2 : 1);
    // Only the left half, so a panel straddling x=16 has an edge to pull colour across.
    w.pushImage(photo.handle, 0, 0, 16, 32);
    if (panel) {
      w.push(17);  // Backdrop
      w.pushAnim(10);
      w.pushAnim(8);
      w.pushAnim(16);
      w.pushAnim(16);
      w.push(3);               // corner radius
      w.pushAnim(6);           // blur
      w.pushAnim(refraction);  // 0 is a plain blur, above 0 is glass
      w.pushAnim(20);
      w.pushAnim(20);
      w.pushAnim(0);
      w.pushAnim(0);   // splay
      w.pushAnim(45);  // light angle
      w.pushAnim(50);  // light intensity
      w.pushPaint(0xffffffff, false);
    }
    return w.finish();
  };

  for (float ratio : {1.0f, 2.0f}) {
    auto scaled = [ratio](int point) { return static_cast<int>(static_cast<float>(point) * ratio); };
    auto without = render(program(false, 0), {}, {}, ratio);
    auto with = render(program(true, 0), {}, {}, ratio);

    // The photo is still there, both under the panel's own half and well clear of it.
    auto photoPixel = pixelAt(with, scaled(4), scaled(4));
    CHECK(photoPixel[2] > 200 && photoPixel[0] == 0);
    CHECK(pixelAt(without, scaled(4), scaled(4)) == photoPixel);

    // Inside the panel, 6 points right of the photo's edge: the blur pulled blue across.
    auto pulled = pixelAt(with, scaled(22), scaled(16));
    // Not merely non-zero: a sigma left in scene points still leaks a little blue this far in.
    CHECK(pulled[2] > 20 && pulled[0] == 0);
    CHECK(pixelAt(without, scaled(22), scaled(16))[2] == 0);

    // Outside the panel and right of the photo, nothing was painted over the empty canvas.
    CHECK(!opaqueAt(with, scaled(30), scaled(30)));

    // Glass reads the same backdrop through a different path, so it must draw a panel too.
    auto glassy = render(program(true, 40), {}, {}, ratio);
    CHECK(glassy != without);
    CHECK(glassy != with);
    CHECK(pixelAt(glassy, scaled(4), scaled(4)) == photoPixel);
    CHECK(opaqueAt(glassy, scaled(22), scaled(16)));
  }
}

/// SVG is text: the same scene comes out as markup a reader can parse, not pixels.
void anSvgIsWrittenAsMarkup() {
  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(10);
  w.pushAnim(20);
  w.pushAnim(80);
  w.pushAnim(60);
  w.pushPaint(0xff3366cc, false);
  auto program = w.finish();
  auto svg = tgfxrn::renderSceneToSvg(program, {}, 200, 120, false);

  CHECK(svg.rfind("<?xml", 0) == 0 || svg.find("<svg") != std::string::npos);
  CHECK(svg.find("</svg>") != std::string::npos);
  // The viewBox is what was asked for, and the rect survived as geometry.
  CHECK(svg.find("200") != std::string::npos && svg.find("120") != std::string::npos);
  CHECK(svg.find("rect") != std::string::npos || svg.find("path") != std::string::npos);

  // With a real font, the flag turns <text> into outlines.
  auto& fonts = tgfxrn::FontRegistry::instance();
  fonts.clear();
  fonts.add(tgfx::Typeface::MakeFromPath(std::string(kFontDir) + "/PT_Serif-Web-Regular.ttf"),
            "Svg Serif");
  Writer t;
  writeText(t, 28, -1);
  auto textProgram = t.finish();
  const std::vector<std::string> textStrings{"Wg", "Svg Serif", ""};
  auto asText = tgfxrn::renderSceneToSvg(textProgram, textStrings, 200, 120, false);
  auto asPaths = tgfxrn::renderSceneToSvg(textProgram, textStrings, 200, 120, true);
  CHECK(asText.find("<text") != std::string::npos);
  CHECK(asPaths.find("<text") == std::string::npos);
  CHECK(asPaths.find("<path") != std::string::npos);
  fonts.clear();

  // A program the reader rejects is an error, not a well-formed empty document.
  bool rejected = false;
  try {
    tgfxrn::renderSceneToSvg({program.data(), 4}, {}, 200, 120, false);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  CHECK(rejected);

  bool threw = false;
  try {
    tgfxrn::renderSceneToSvg(program, {}, 0, 120, false);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
}

// ---- v26: imperative slot coverage (#204) ----

/// One frame with string slots, against a caller-held renderer so the
/// generation can move between frames the way setStringValues moves it.
Pixels drawFrameWithStrings(tgfxrn::SceneRenderer& renderer,
                            const std::vector<double>& values,
                            const std::vector<std::string>& stringValues,
                            uint64_t gen,
                            const std::vector<uint8_t>* pushed = nullptr) {
  std::vector<uint8_t> allPushed(stringValues.size(), 1);
  auto* context = device()->lockContext();
  auto surface = tgfx::Surface::Make(context, kWidth, kHeight);
  auto* canvas = surface->getCanvas();
  canvas->clear();
  renderer.render(canvas, values, stringValues, gen,
                  pushed != nullptr ? *pushed : allPushed);
  context->flushAndSubmit(true);
  auto info = tgfx::ImageInfo::Make(kWidth, kHeight, tgfx::ColorType::RGBA_8888);
  Pixels pixels(static_cast<size_t>(info.byteSize()));
  surface->readPixels(info, pixels.data());
  device()->unlock();
  return pixels;
}

void aHiddenNodeDrawsNothingAndAVisibleSlotTogglesIt() {
  // Literal 0: skipped outright.
  Writer hidden;
  hidden.push(1);
  hidden.push(1 | 64);  // Rect + visible word
  hidden.pushAnim(0);   // invisible
  hidden.pushAnim(8);
  hidden.pushAnim(8);
  hidden.pushAnim(48);
  hidden.pushAnim(48);
  hidden.pushPaint(0xffff0000, false);
  CHECK(!anyPixelSet(render(hidden.finish(), {}, {})));

  // Slot-driven: the same program shows and hides through setValues.
  Writer pooled;
  pooled.push(1);
  pooled.push(1 | 64);
  pooled.pushAnim(1, 0);
  pooled.pushAnim(8);
  pooled.pushAnim(8);
  pooled.pushAnim(48);
  pooled.pushAnim(48);
  pooled.pushPaint(0xffff0000, false);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(pooled.finish(), {});
  CHECK(anyPixelSet(drawFrame(renderer, {1})));
  CHECK(!anyPixelSet(drawFrame(renderer, {0})));
  // Hidden nodes are not hit either.
  CHECK(renderer.hitTest(32, 32, false, std::vector<double>{0}).empty());
  CHECK(!renderer.hitTest(32, 32, false, std::vector<double>{1}).empty());
}

void liveTextReshapesOnAStringPush() {
  tgfxrn::FontRegistry::instance().clear();
  const std::vector<std::string> strings{"i", "Helvetica", "", "WWWW"};
  Writer w;
  w.push(1);
  w.push(8);  // Text
  w.pushAnim(4);
  w.pushAnim(44);
  w.pushAnim(28);
  w.push(1);
  w.push(400);
  w.push(0);
  w.push(0);
  w.push(1);               // one run
  w.pushStringAnim(0, 0);  // run text follows string slot 0
  w.push(2);               // run family ("")
  w.push(0);               // run weight
  w.push(0);               // run italic
  w.push(0);               // run letterSpacing
  w.push(0);               // run size
  w.push(0);               // run hasColor
  w.push(0);               // run color
  w.push(0);               // wrapWidth
  w.push(0);               // align
  w.push(0);               // lineHeight
  w.push(0);               // maxLines
  w.push(2);               // ellipsis ("")
  w.pushPaint(0xff000000, false);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), strings);
  auto narrow = drawFrameWithStrings(renderer, {}, {"i"}, 1);
  auto wide = drawFrameWithStrings(renderer, {}, {"WWWW"}, 2);
  CHECK(anyPixelSet(narrow));
  CHECK(inkPixels(wide) > inkPixels(narrow) * 2);
  // The same generation with the same content lays out nothing new.
  auto again = drawFrameWithStrings(renderer, {}, {"WWWW"}, 2);
  CHECK(again == wide);
}

void aLivePathReparsesOnAStringPush() {
  const std::vector<std::string> strings{"M8 8 L16 8 L16 16 L8 16 Z"};
  Writer w;
  w.push(1);
  w.push(6);  // Path
  w.pushStringAnim(0, /*slot=*/0);
  w.push(0);  // no effects
  w.pushPaint(0xffff0000, false);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), strings);
  auto small = drawFrameWithStrings(renderer, {}, {"M8 8 L16 8 L16 16 L8 16 Z"}, 1);
  auto big = drawFrameWithStrings(renderer, {}, {"M8 8 L56 8 L56 56 L8 56 Z"}, 2);
  CHECK(anyPixelSet(small));
  CHECK(inkPixels(big) > inkPixels(small) * 4);
}

void anImageHandleSlotSwapsThePixels() {
  auto red = registerSolid({255, 0, 0, 255});
  auto blue = registerSolid({0, 0, 255, 255});
  Writer w;
  w.push(1);
  w.push(7);  // Image
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(64);
  w.pushAnim(64);
  w.pushAnim(static_cast<double>(red.handle), /*slot=*/0);
  w.push(0);  // fill
  w.push(1);
  w.push(0);  // no source rect
  w.pushPaint(0xffffffff, false);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});
  auto first = drawFrame(renderer, {static_cast<double>(red.handle)});
  CHECK(pixelAt(first, 32, 32)[0] == 255);
  auto swapped = drawFrame(renderer, {static_cast<double>(blue.handle)});
  CHECK(pixelAt(swapped, 32, 32)[2] == 255);
}

void animatedGradientGeometryMoves() {
  // A linear ramp whose end.x is a slot: moving it moves the midpoint's shade.
  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(64);
  w.pushAnim(64);
  w.push(0xffffffff);
  w.push(2);
  w.pushAnim(1);
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(1);
  w.push(0);  // linear gradient
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(64, /*slot=*/0);
  w.pushAnim(0);
  w.push(0);  // static stops
  w.push(2);
  w.push(0xff000000);
  w.push(0xffffffff);
  w.push(0);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});
  auto wide = drawFrame(renderer, {64});
  auto tight = drawFrame(renderer, {16});
  // With the ramp ending at 16, x=32 is already fully white; at 64 it is mid-grey.
  CHECK(pixelAt(tight, 32, 32)[0] > 220);
  CHECK(pixelAt(wide, 32, 32)[0] < 180);
  CHECK(pixelAt(wide, 32, 32)[0] > 80);
}

void animatedStopColoursShift() {
  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(64);
  w.pushAnim(64);
  w.push(0xffffffff);
  w.push(2);
  w.pushAnim(1);
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(1);
  w.push(0);  // linear
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(64);
  w.pushAnim(0);
  w.push(1);  // animated colours
  w.push(2);
  // Stop 0: r slot, g 0, b 0, a 1. Stop 1: fixed blue.
  w.pushAnim(1, /*slot=*/0);
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(1);
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(1);
  w.pushAnim(1);
  w.push(0);  // no positions
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});
  auto redStart = drawFrame(renderer, {1});
  auto darkStart = drawFrame(renderer, {0});
  CHECK(pixelAt(redStart, 1, 32)[0] > 200);
  CHECK(pixelAt(darkStart, 1, 32)[0] < 40);
}

void animatedBlurRadiusChangesThePixels() {
  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(24);
  w.pushAnim(24);
  w.pushAnim(16);
  w.pushAnim(16);
  w.push(0xffff0000);
  w.push(2 | 4);  // antiAlias + filter block
  w.pushAnim(1);
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(1);
  w.pushAnim(3);  // blend srcOver
  w.push(-1);     // no colour filter
  w.push(0);      // blur image filter
  w.pushAnim(0, /*slot=*/0);
  w.pushAnim(0, /*slot=*/0);
  w.push(3);   // decal
  w.push(-1);  // no shader
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});
  auto sharp = drawFrame(renderer, {0.01});
  auto soft = drawFrame(renderer, {8});
  // Just outside the box: a blur spreads ink past the geometry, sharp does not.
  CHECK(pixelAt(sharp, 20, 32)[0] == 0);
  CHECK(pixelAt(soft, 20, 32)[0] > 0);
}

void animatedStyleWordFlipsFillToStroke() {
  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(16);
  w.pushAnim(16);
  w.pushAnim(32);
  w.pushAnim(32);
  w.push(0xffff0000);
  w.push(2 | 32);             // antiAlias + animated style/aa words
  w.pushAnim(0, /*slot=*/0);  // style word
  w.pushAnim(1);              // antiAlias word
  w.pushAnim(2);              // strokeWidth
  w.pushAnim(0);              // cap
  w.pushAnim(0);              // join
  w.pushAnim(1);              // alpha
  w.push(-1);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});
  auto filled = drawFrame(renderer, {0});
  auto stroked = drawFrame(renderer, {1});
  CHECK(pixelAt(filled, 32, 32)[0] == 255);
  CHECK(pixelAt(stroked, 32, 32)[0] == 0);
  CHECK(pixelAt(stroked, 16, 32)[0] == 255);
}

void atlasDrawsFromASharedBuffer() {
  auto red = registerSolid({255, 0, 0, 255});
  auto block = std::make_shared<tgfxrn::SharedFloats>(7 * 2);
  int32_t bufferId = tgfxrn::BufferRegistry::instance().add(block);
  // Two 2x2 sprites scaled 8x, centred at (16,16) and (48,48).
  const float sprites[14] = {0, 0, 2, 2, 8, 16, 16, 0, 0, 2, 2, 8, 48, 48};
  std::copy(sprites, sprites + 14, block->data.begin());
  Writer w;
  w.push(1);
  w.push(15);  // Atlas
  w.pushAnim(static_cast<double>(red.handle));
  w.push(1);
  w.push(bufferId);
  w.pushAnim(2, /*slot=*/0);
  w.pushPaint(0xffffffff, false);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});
  auto both = drawFrame(renderer, {2});
  CHECK(pixelAt(both, 16, 16)[0] == 255);
  CHECK(pixelAt(both, 48, 48)[0] == 255);
  // The count slot draws a prefix without touching the data.
  auto one = drawFrame(renderer, {1});
  CHECK(pixelAt(one, 16, 16)[0] == 255);
  CHECK(pixelAt(one, 48, 48)[0] == 0);
  tgfxrn::BufferRegistry::instance().remove(bufferId);
}

void verticesDrawFromASharedBuffer() {
  // Positions for 4 vertices + 6 indices: two triangles covering a square.
  auto block = std::make_shared<tgfxrn::SharedFloats>(4 * 2 + 6);
  int32_t bufferId = tgfxrn::BufferRegistry::instance().add(block);
  const float quad[8] = {8, 8, 56, 8, 8, 56, 56, 56};
  const float indices[6] = {0, 1, 2, 1, 2, 3};
  std::copy(quad, quad + 8, block->data.begin());
  std::copy(indices, indices + 6, block->data.begin() + 8);
  Writer w;
  w.push(1);
  w.push(16);  // Vertices
  w.push(bufferId);
  w.push(0);  // triangles
  w.push(4);  // maxVertices
  w.push(6);  // maxIndices
  w.push(0);  // no textures, no colours
  w.pushAnim(4, /*slot=*/0);
  w.pushAnim(6, /*slot=*/1);
  w.pushPaint(0xffff0000, false);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});
  auto drawn = drawFrame(renderer, {4, 6});
  CHECK(pixelAt(drawn, 32, 32)[0] == 255);
  // Half the indices: only the first triangle survives.
  auto half = drawFrame(renderer, {4, 3});
  CHECK(pixelAt(half, 12, 12)[0] == 255);
  CHECK(pixelAt(half, 55, 55)[0] == 0);
  tgfxrn::BufferRegistry::instance().remove(bufferId);
}

void srcRectSlotStepsSprites() {
  // A 2x1 sheet, red then blue; the slot steps which texel the image shows.
  std::vector<uint8_t> half{255, 0, 0, 255, 0, 0, 255, 255};
  auto split = registerPixels(2, 1, half);
  Writer w;
  w.push(1);
  w.push(7);  // Image
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(64);
  w.pushAnim(64);
  w.pushAnim(static_cast<double>(split.handle));
  w.push(0);  // fill
  w.push(0);  // nearest
  w.push(1);  // source rect follows
  w.pushAnim(0, /*slot=*/0);
  w.pushAnim(0);
  w.pushAnim(1);
  w.pushAnim(1);
  w.pushPaint(0xffffffff, false);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});
  auto redFrame = drawFrame(renderer, {0});
  auto blueFrame = drawFrame(renderer, {1});
  CHECK(pixelAt(redFrame, 32, 32)[0] == 255);
  CHECK(pixelAt(blueFrame, 32, 32)[2] == 255);
}

void aSparseStringPushNeverBlanksANeighbour() {
  // Two live-path nodes on slots 0 and 1; pushing only slot 1 must leave 0 drawn.
  const std::vector<std::string> strings{"M8 8 L28 8 L28 28 L8 28 Z",
                                         "M36 8 L56 8 L56 28 L36 28 Z"};
  Writer w;
  w.push(2);
  w.push(6);
  w.pushStringAnim(0, 0);
  w.push(0);
  w.pushPaint(0xffff0000, false);
  w.push(6);
  w.pushStringAnim(1, 1);
  w.push(0);
  w.pushPaint(0xff0000ff, false);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), strings);
  // Only slot 1 was pushed; slot 0's entry is the resize gap fill ("").
  std::vector<uint8_t> pushed{0, 1};
  auto frame = drawFrameWithStrings(
      renderer, {}, {"", "M36 36 L56 36 L56 56 L36 56 Z"}, 1, &pushed);
  CHECK(pixelAt(frame, 18, 18)[0] == 255);  // slot 0 keeps its compiled literal
  CHECK(pixelAt(frame, 46, 46)[2] == 255);  // slot 1 moved to the pushed value
  CHECK(pixelAt(frame, 46, 18)[2] == 0);    // and left its old spot
}

void hitTestFollowsALivePath() {
  const std::vector<std::string> strings{"M8 8 L24 8 L24 24 L8 24 Z"};
  Writer w;
  w.push(1);
  w.push(6);
  w.pushStringAnim(0, 0);
  w.push(0);
  w.pushPaint(0xffff0000, false);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), strings);
  std::vector<std::string> live{"M40 40 L56 40 L56 56 L40 56 Z"};
  std::vector<uint8_t> pushed{1};
  // The hit test sees the pushed geometry, not the compiled literal.
  CHECK(!renderer.hitTest(16, 16, true, {}, live, pushed).empty() == false);
  CHECK(!renderer.hitTest(48, 48, true, {}, live, pushed).empty());
  // Without the push it still answers from the literal.
  CHECK(!renderer.hitTest(16, 16, true, {}, {}, {}).empty());
}

void aBadBufferMeshIndexDropsTheFrameNotTheProcess() {
  auto block = std::make_shared<tgfxrn::SharedFloats>(3 * 2 + 3);
  int32_t bufferId = tgfxrn::BufferRegistry::instance().add(block);
  const float tri[6] = {8, 8, 56, 8, 8, 56};
  std::copy(tri, tri + 6, block->data.begin());
  block->data[6] = 0;
  block->data[7] = std::numeric_limits<float>::quiet_NaN();
  block->data[8] = 2;
  Writer w;
  w.push(1);
  w.push(16);
  w.push(bufferId);
  w.push(0);
  w.push(3);
  w.push(3);
  w.push(0);
  w.pushAnim(3);
  w.pushAnim(3);
  w.pushPaint(0xffff0000, false);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});
  CHECK(!anyPixelSet(drawFrame(renderer, {})));
  tgfxrn::BufferRegistry::instance().remove(bufferId);
}

void anImageShaderSlotSeededZeroStillGoesLive() {
  auto red = registerSolid({255, 0, 0, 255});
  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(64);
  w.pushAnim(64);
  w.push(0xffffffff);
  w.push(2);
  w.pushAnim(1);
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(1);
  w.push(7);         // image shader
  w.pushAnim(0, 0);  // handle slot seeded 0: nothing loaded at commit time
  w.push(1);
  w.push(1);
  w.push(1);
  w.push(0);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});
  // Slot 0: no image resolves, so the paint colour shows (white, green=255).
  auto unlit = drawFrame(renderer, {0});
  CHECK(pixelAt(unlit, 32, 32)[1] == 255);
  // The slot write re-points the shader without a re-commit: red, green=0.
  auto lit = drawFrame(renderer, {static_cast<double>(red.handle)});
  CHECK(pixelAt(lit, 32, 32)[0] == 255);
  CHECK(pixelAt(lit, 32, 32)[1] == 0);
}

/// encode/save's rasterize path: any registry entry becomes CPU pixels finishSnapshot can eat.
void aRegisteredImageRasterizesForEncoding() {
  auto& registry = tgfxrn::ImageRegistry::instance();
  auto solid = registerSolid({250, 60, 90, 255}, 4, 4);
  auto raster = tgfxrn::rasterizeRegisteredImage(solid.handle);
  CHECK(raster.pixels != nullptr);
  CHECK(raster.info.width() == 4);
  CHECK(raster.info.height() == 4);

  tgfxrn::SnapshotRequest request;
  request.format = "raw";
  bool called = false;
  tgfxrn::finishSnapshot(
      raster.pixels, raster.info, request,
      [&](std::shared_ptr<tgfx::Data> data, int w, int h, const std::string& error) {
        called = true;
        CHECK(error.empty());
        CHECK(w == 4 && h == 4);
        CHECK(data != nullptr && data->size() == 4u * 4u * 4u);
        const auto* px = static_cast<const uint8_t*>(data->data());
        CHECK(px[0] == 250 && px[1] == 60 && px[2] == 90 && px[3] == 255);
      });
  CHECK(called);

  registry.remove(solid.handle);
  bool threw = false;
  try {
    tgfxrn::rasterizeRegisteredImage(solid.handle);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

namespace {
/// Bounded spin for the offscreen worker's callback; SceneTest has no runloop.
bool waitFor(const std::atomic<bool>& flag) {
  for (int i = 0; i < 500 && !flag.load(); i++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return flag.load();
}

std::vector<uint8_t> surfaceRaw(const std::shared_ptr<tgfxrn::OffscreenSurface>& surface) {
  std::atomic<bool> done{false};
  std::vector<uint8_t> out;
  std::string error;
  tgfxrn::SnapshotRequest request;
  request.format = "raw";
  surface->requestSave(request, [&](std::shared_ptr<tgfx::Data> data, int, int,
                                    const std::string& err) {
    error = err;
    if (data != nullptr) {
      const auto* bytes = static_cast<const uint8_t*>(data->data());
      out.assign(bytes, bytes + data->size());
    }
    done.store(true);
  });
  CHECK(waitFor(done));
  CHECK(error.empty());
  return out;
}
}  // namespace

/// The retained surface: staged state renders on the worker, readbacks serve on demand.
void anOffscreenSurfaceRendersSlotsAndViewport() {
  auto surface = std::make_shared<tgfxrn::OffscreenSurface>(8, 8, 1.0f, true, false);
  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(0, 0);
  w.pushAnim(0);
  w.pushAnim(4);
  w.pushAnim(4);
  w.pushPaint(0xffff2d55, false);
  auto words = w.finish();
  surface->setScene(words, {"0"});
  surface->setValues({0});

  auto first = surfaceRaw(surface);
  CHECK(first.size() == 8u * 8u * 4u);
  CHECK(first[0] == 255 && first[1] == 45 && first[2] == 85);  // (0,0) inside the rect
  size_t right = (0 * 8 + 6) * 4;
  CHECK(first[right + 0] == 0 && first[right + 1] == 0);  // (6,0) is background

  // A slot write moves the rect with no re-encode; the retained program re-renders.
  surface->setValues({4});
  auto moved = surfaceRaw(surface);
  CHECK(moved[0] == 0);                                      // left edge vacated
  CHECK(moved[right + 0] == 255 && moved[right + 2] == 85);  // landed at x=4..8

  // The viewport rides above the program, exactly as on a view.
  surface->setValues({0});
  surface->setViewport(2.0f, 0.0f, 0.0f);
  auto zoomed = surfaceRaw(surface);
  size_t past = (0 * 8 + 7) * 4;
  CHECK(zoomed[past + 0] == 255);  // 4pt rect fills 8px at zoom 2

  // Headless hit test answers from the staged scene in canvas points.
  surface->setViewport(1.0f, 0.0f, 0.0f);
  auto hit = surface->hitTest(2.0f, 2.0f, true);
  CHECK(hit.size() >= 2 && hit[0] == 1 && hit[1] == 0);
  auto miss = surface->hitTest(7.5f, 7.5f, true);
  CHECK(miss.empty());

  // Dispose rejects late readbacks instead of hanging them.
  surface->dispose();
  std::atomic<bool> rejected{false};
  tgfxrn::SnapshotRequest request;
  surface->requestSave(request, [&](std::shared_ptr<tgfx::Data> data, int, int,
                                    const std::string& err) {
    CHECK(data == nullptr);
    CHECK(!err.empty());
    rejected.store(true);
  });
  CHECK(waitFor(rejected));
}

/// The unpremultiply in the encode tail, at an alpha where it is not a no-op,
/// and the two rejection branches encode must take.
void encodingUnpremultipliesAndRejectsLiveFrames() {
  // 50%-alpha red, premultiplied: channels store 128, straight alpha wants 255.
  auto info = tgfx::ImageInfo::Make(2, 2, tgfx::ColorType::RGBA_8888,
                                    tgfx::AlphaType::Premultiplied);
  std::vector<uint8_t> premul(info.byteSize());
  for (size_t i = 0; i < premul.size(); i += 4) {
    premul[i] = 128;
    premul[i + 1] = 0;
    premul[i + 2] = 0;
    premul[i + 3] = 128;
  }
  auto pixels = tgfx::Data::MakeWithCopy(premul.data(), premul.size());
  tgfxrn::SnapshotRequest request;
  request.format = "raw";
  bool called = false;
  tgfxrn::finishSnapshot(
      pixels, info, request,
      [&](std::shared_ptr<tgfx::Data> data, int w, int h, const std::string& error) {
        called = true;
        CHECK(error.empty());
        CHECK(w == 2 && h == 2);
        const auto* px = static_cast<const uint8_t*>(data->data());
        CHECK(px[0] == 255 && px[1] == 0 && px[2] == 0 && px[3] == 128);
      });
  CHECK(called);

  // A live producer entry has no still to encode.
  auto& registry = tgfxrn::ImageRegistry::instance();
  auto solid = registerSolid({10, 20, 30, 255});
  auto live = registry.addExternal(registry.sourceImage(solid.handle), 0, true);
  CHECK(live.handle != 0);
  bool threw = false;
  try {
    tgfxrn::rasterizeRegisteredImage(live.handle);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
  registry.remove(live.handle);
  registry.remove(solid.handle);
}

/// The commit-time batch clustering may regroup disjoint siblings by op class,
/// but an overlap must keep its paint order. Pixels prove both: the regrouped
/// interleave draws exactly what a hand-grouped list draws, and flipping the
/// overlapping pair still changes the picture — order survived where it shows.
void batchClusteringKeepsOverlapOrderAndPixels() {
  auto pushFillerRect = [](Writer& w, int i) {
    w.push(1);  // Rect
    w.pushAnim(i * 7 + 1);
    w.pushAnim(50);
    w.pushAnim(4);
    w.pushAnim(4);
    w.pushPaint(0xff00cc44, false);
  };
  auto pushFillerCircle = [](Writer& w, int i) {
    w.push(4);  // Circle
    w.pushAnim(i * 7 + 3);
    w.pushAnim(59);
    w.pushAnim(2);
    w.pushPaint(0xff8844cc, false);
  };
  auto pushOverlap = [](Writer& w, bool circleLast) {
    auto rect = [&] {
      w.push(1);
      w.pushAnim(10);
      w.pushAnim(10);
      w.pushAnim(30);
      w.pushAnim(30);
      w.pushPaint(0xffff2d55, false);
    };
    auto circle = [&] {
      w.push(4);
      w.pushAnim(25);
      w.pushAnim(25);
      w.pushAnim(10);
      w.pushPaint(0xff007aff, false);
    };
    if (circleLast) {
      rect();
      circle();
    } else {
      circle();
      rect();
    }
  };

  Writer interleaved;
  interleaved.push(10);
  for (int i = 0; i < 8; i++) {
    i % 2 == 0 ? pushFillerRect(interleaved, i) : pushFillerCircle(interleaved, i);
  }
  pushOverlap(interleaved, true);

  // The same nodes with the fillers already grouped by kind: the reference the
  // clustering pass is supposed to be equivalent to.
  Writer grouped;
  grouped.push(10);
  for (int i = 0; i < 8; i += 2) {
    pushFillerRect(grouped, i);
  }
  for (int i = 1; i < 8; i += 2) {
    pushFillerCircle(grouped, i);
  }
  pushOverlap(grouped, true);

  Writer flipped;
  flipped.push(10);
  for (int i = 0; i < 8; i++) {
    i % 2 == 0 ? pushFillerRect(flipped, i) : pushFillerCircle(flipped, i);
  }
  pushOverlap(flipped, false);

  auto a = render(interleaved.finish(), {}, {});
  CHECK(anyPixelSet(a));
  CHECK(a == render(grouped.finish(), {}, {}));
  CHECK(a != render(flipped.finish(), {}, {}));
}

/// setScene-then-hitTest must answer before the worker's first drain.
void anOffscreenSurfaceHitTestsBeforeTheFirstDrain() {
  auto surface = std::make_shared<tgfxrn::OffscreenSurface>(8, 8, 1.0f, true, false);
  Writer w;
  w.push(1);
  w.push(1);  // Rect
  w.pushAnim(0);
  w.pushAnim(0);
  w.pushAnim(4);
  w.pushAnim(4);
  w.pushPaint(0xffff2d55, false);
  auto words = w.finish();
  surface->setScene(words, {"0"});
  auto hit = surface->hitTest(2.0f, 2.0f, true);
  CHECK(hit.size() >= 2 && hit[0] == 1 && hit[1] == 0);
  surface->dispose();
}

/// Transform ops: translate(32,32) then perspective 200, so a 3D group centres on the canvas.
void push3DOps(Writer& w, std::initializer_list<std::array<double, 3>> extra, int rotateSlot = -1) {
  w.push(2 + static_cast<double>(extra.size()));
  w.push(0);
  w.pushAnim(32);
  w.pushAnim(32);
  w.push(7);
  w.pushAnim(200);
  w.pushAnim(0);
  for (const auto& op : extra) {
    w.push(op[0]);
    w.pushAnim(op[1], op[0] == 6 ? rotateSlot : -1);
    w.pushAnim(op[2]);
  }
}

void pushPlane(Writer& w, std::initializer_list<std::array<double, 3>> ops, uint32_t color,
               double half = 16) {
  w.push(0);
  w.push(static_cast<double>(ops.size()));
  for (const auto& op : ops) {
    w.push(op[0]);
    w.pushAnim(op[1]);
    w.pushAnim(op[2]);
  }
  w.push(1);
  w.push(1);
  w.pushAnim(-half);
  w.pushAnim(-half);
  w.pushAnim(half * 2);
  w.pushAnim(half * 2);
  w.pushPaint(color, false);
}

/// Clip, group opacity and the rest flatten preserve3d instead of being dropped by it.
void preserve3dKeepsTheGroupsOwnClipAndLayer() {
  auto scene = [](int flags) {
    Writer w;
    w.push(1);
    w.push(13);
    push3DOps(w, {});
    w.push(32 | flags);
    if ((flags & 1) != 0) {
      w.push(1);
      w.push(-16);
      w.push(-16);
      w.push(16);
      w.push(16);
    }
    if ((flags & 4) != 0) {
      w.pushAnim(0.5);
    }
    w.push(1);
    pushPlane(w, {{6, 0, 0}}, 0xffff0000);
    return render(w.finish(), {}, {});
  };
  auto plain = scene(0);
  CHECK(alphaAt(plain, 32, 32) == 255);
  auto faded = scene(4);
  CHECK(alphaAt(faded, 32, 32) > 100 && alphaAt(faded, 32, 32) < 160);
  auto clipped = scene(1);
  CHECK(opaqueAt(clipped, 24, 24));
  CHECK(!opaqueAt(clipped, 40, 40));
}

/// A plane of a 3D context keeps its own clip and layer opacity.
void aPlaneOfA3DContextKeepsItsClipAndLayer() {
  Writer w;
  w.push(1);
  w.push(13);
  push3DOps(w, {});
  w.push(32);
  w.push(1);
  w.push(13);
  w.push(1);
  w.push(6);
  w.pushAnim(0);
  w.pushAnim(0);
  w.push(1 | 4);
  w.push(1);
  w.push(-16);
  w.push(-16);
  w.push(16);
  w.push(16);
  w.pushAnim(0.5);
  w.push(1);
  w.push(1);
  w.pushAnim(-16);
  w.pushAnim(-16);
  w.pushAnim(32);
  w.pushAnim(32);
  w.pushPaint(0xffff0000, false);
  auto pixels = render(w.finish(), {}, {});
  CHECK(alphaAt(pixels, 24, 24) > 100 && alphaAt(pixels, 24, 24) < 160);
  CHECK(!opaqueAt(pixels, 40, 40));
}

/// A <Fill> has no bounds of its own; on a plane it still draws.
void aFillOnA3DPlaneDraws() {
  Writer w;
  w.push(1);
  w.push(13);
  push3DOps(w, {});
  w.push(32);
  w.push(1);
  w.push(0);
  w.push(1);
  w.push(6);
  w.pushAnim(0.3);
  w.pushAnim(0);
  w.push(1);
  w.push(9);
  w.pushPaint(0xff3366cc, false);
  CHECK(opaqueAt(render(w.finish(), {}, {}), 32, 32));
}

/// perspective 0 is no perspective, not a division by zero that blanks the group.
void aZeroPerspectiveDrawsFlat() {
  Writer w;
  w.push(1);
  w.push(0);
  w.push(2);
  w.push(7);
  w.pushAnim(0);
  w.pushAnim(0);
  w.push(6);
  w.pushAnim(0);
  w.pushAnim(0);
  w.push(1);
  w.push(1);
  w.pushAnim(8);
  w.pushAnim(8);
  w.pushAnim(32);
  w.pushAnim(32);
  w.pushPaint(0xffff0000, false);
  CHECK(opaqueAt(render(w.finish(), {}, {}), 20, 20));
}

/// Planes turned edge-on alone face the viewer once composed; the nearer one is hit first.
void hitTestComposesA3DContextNearestFirst() {
  constexpr double kQuarter = 1.5707963267948966;
  Writer w;
  w.push(1);
  w.push(13);
  push3DOps(w, {{6, -kQuarter, 0}});
  w.push(32);
  w.push(2);
  pushPlane(w, {{6, kQuarter, 0}, {8, 10, 0}}, 0xffff0000);
  pushPlane(w, {{6, kQuarter, 0}, {8, -10, 0}}, 0xff0000ff);
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});
  auto hits = renderer.hitTest(32, 32, false, {});
  CHECK(hits.size() == 8);
  CHECK(hits.size() >= 4 && hits[0] == 3 && hits[1] == 0 && hits[2] == 0 && hits[3] == 0);
  CHECK(hits.size() == 8 && hits[4] == 3 && hits[5] == 0 && hits[6] == 1 && hits[7] == 0);
  auto pixels = drawFrame(renderer, {});
  CHECK(pixelAt(pixels, 32, 32)[0] > 200 && pixelAt(pixels, 32, 32)[2] < 50);
}

/// Static plane content survives the context turning; live content still changes per frame.
void a3DPlaneRedrawsLiveContentAndKeepsStaticContent() {
  Writer w;
  w.push(1);
  w.push(13);
  push3DOps(w, {{6, 0, 0}}, 0);
  w.push(32);
  w.push(2);
  pushPlane(w, {{8, 0, 0}}, 0xff00ff00, 8);
  w.push(0);
  w.push(1);
  w.push(8);
  w.pushAnim(1);
  w.pushAnim(0);
  w.push(1);
  w.push(1);
  w.pushAnim(-16);
  w.pushAnim(-16);
  w.pushAnim(32);
  w.pushAnim(6);
  w.pushLiveColorPaint(0xffff0000, {{{1, 1}, {0, 2}, {0, 3}, {1, -1}}});
  tgfxrn::SceneRenderer renderer;
  renderer.setScene(w.finish(), {});
  auto first = drawFrame(renderer, {0, 1, 0, 0});
  CHECK(opaqueAt(first, 32, 32));
  CHECK(pixelAt(first, 32, 18)[0] > 200);
  auto second = drawFrame(renderer, {0.6, 0, 0, 1});
  CHECK(opaqueAt(second, 32, 32));
  CHECK(inkPixels(second) < inkPixels(first));
  CHECK(pixelAt(second, 32, 18)[2] > 200 && pixelAt(second, 32, 18)[0] < 50);
}

int main() {
  if (device() == nullptr) {
    std::printf("no GPU device available\n");
    return 1;
  }
  aHiddenNodeDrawsNothingAndAVisibleSlotTogglesIt();
  aSparseStringPushNeverBlanksANeighbour();
  hitTestFollowsALivePath();
  aBadBufferMeshIndexDropsTheFrameNotTheProcess();
  anImageShaderSlotSeededZeroStillGoesLive();
  liveTextReshapesOnAStringPush();
  aLivePathReparsesOnAStringPush();
  anImageHandleSlotSwapsThePixels();
  animatedGradientGeometryMoves();
  animatedStopColoursShift();
  animatedBlurRadiusChangesThePixels();
  animatedStyleWordFlipsFillToStroke();
  atlasDrawsFromASharedBuffer();
  verticesDrawFromASharedBuffer();
  srcRectSlotStepsSprites();
  aCorruptProgramIsRejectedWithoutTrustingItsCounts();
  batchedPathsDrawWhatPerShapeOpsDraw();
  aStrokedPathDrawsWhatTheRebuildDraws();
  staticPaintDrawsWhatTheRebuildDraws();
  animatedColorDrawsWhatTheLiteralDraws();
  staticLineDrawsWhatTheRebuildDraws();
  animatedStrokeWidthDrawsWhatTheRebuildDraws();
  staticTextDrawsWhatTheRebuildDraws();
  animatedFieldsStillAnimate();
  animatedUniformDrawsWhatTheRebuildDraws();
  aHelperFunctionCompilesAndDoesNotCollide();
  aChildShaderSamplesWhereItIsAsked();
  aGradientChildShades();
  twoChildrenOnOneShaderBothPaint();
  aRuntimeShaderChildNestsInOneProgram();
  twoShadersWithChildrenDoNotCollide();
  aMiterLimitCutsBackASharpSpike();
  aGroupClipConfinesItsSubtree();
  aLayerCompositesTheSubtreeOnce();
  pathEffectsReshapeTheGeometry();
  theLateFilterKindsAllDraw();
  aThresholdCutsAlphaAndAColorFilterCanFollowAnImageFilter();
  blendFoldsItsShaderChildren();
  filtersOfOneClassCompose();
  blendKeepsItsDynamicChildrenLive();
  blendKeepsAnAnimatedUniformLive();
  aMaskCutsItsContentDown();
  fillCoversTheWholeCanvas();
  anAtlasPlacesEverySpriteItIsGiven();
  verticesDrawTheTrianglesTheyDescribe();
  verticesSampleTheirTextureCoordinates();
  anOffscreenRenderProducesTheSamePixels();
  anOffscreenRenderCarriesItsColorSpace();
  anEmbeddedIccProfileSurvivesTheDecode();
  aDerivedImageSurvivesTheParentsEviction();
  aLiveHandleRefusesToBeDerivedFrom();
  aRecordedGroupReplaysWhatItDrew();
  hitTestReportsWhatIsUnderThePoint();
  theRootMatrixPansInCanvasPointsAfterTheZoom();
  hitTestFollowsTheViewport();
  aPdfIsWrittenWithAPagePerScene();
  aBackdropFrostsWhatIsBeneathIt();
  glassTakesItsLightFromTheScene();
  theGlassMaterialResolvesFromValueSlots();
  filmGrainDustsOnlyWhatTheGroupDrew();
  filmGrainResolvesFromValueSlots();
  filmGrainAnswersToItsBlendModeAndLiveColour();
  aRecordedGroupHoldingGrainIsNotRecorded();
  aBackdropAtAPixelRatioLeavesTheRestOfTheSceneAlone();
  aBackdropOnACanvasTheSizeOfARealOne();
  aBackdropHangingOffTheEdgeStillDrawsWhereItOverlaps();
  anSvgIsWrittenAsMarkup();
  everyImageDrawsItsOwnHandle();
  reCommittingAnImageSceneDecodesNothing();
  theByteBudgetEvictsLeastRecentlyUsed();
  fitModesLetterboxAndCrop();
  aSourceRectPicksOneSprite();
  anImageShaderTilesAcrossAShape();
  anExternalHandleIsReplacedInPlaceAndNotBudgeted();
  aStreamHandleRefusesReplaceExternalAndBytesFollowTheSwap();
  anExternalHandleAlsoDrivesAnImageShaderPerFrame();
  anExternalHandleDrivesAChildShaderPerFrame();
  aCameraFrameCoversTheCanvasAndItsChildrenRideItsTransform();
  aCameraFrameMirrorsAndRotatesItsChildrenWithTheFrame();
  aCameraFrameRotatesANonSquareFrame();
  aCameraFrameCoverCropsTheOverflowingAxis();
  aCameraFrameWithNoFrameDrawsNothing();
  anImageEvictedBeforeCompileComesBackWhenItsDecodeLands();
  aBufferSwapSchedulesADraw();
  aDisposedHandleDrawsNothing();
  anExifRotatedJpegDecodesUpright();
  anSvgDocumentBecomesAnImageHandle();
  anSvgHandleStaysCrispWhenDrawnLargerThanItsSize();
  malformedSvgIsRejectedRatherThanDrawn();
  aParagraphKeepsItsItalicThroughTheRuns();
  theEllipsisBelongsToTheRunItFollows();
  aSpanCarriesItsOwnFace();
  aColouredSpanSplitsTheBlobs();
  everyLineIsMeasured();
  textComesBackAsAPath();
  pathUtilitiesAgreeWithTheGeometryTheyWrap();
  placedGlyphsDrawWhereTheyArePut();
  aBlendModeCompositesAgainstWhatIsBelow();
  aBlendModeSurvivesAnImageFilter();
  aColourMatrixTransformsTheDrawnColour();
  aBlendColourFilterTints();
  aBlurSpreadsPastTheGeometry();
  aDropShadowPaintsUnderItsContent();
  filtersAreBuiltOnceAndNeverOnTheDrawPath();
  onlyTheAnimatedPartOfAFilterChainIsRebuilt();
  aFontIsFoundOnAnImageWithNoneOfTheOldHardcodedPaths();
  theDefaultFamilyCanActuallyDrawText();
  aCachedFaceStillHasToBeAbleToDrawText();
  aFontRegistrationSchedulesADraw();
  recompilingAfterARegistrationPicksUpTheNewFace();
  aRegisteredFamilyMatchesByWeightAndStyle();
  aRegisteredFontDrawsAndBoldDiffersFromRegular();
  theFauxBitsEmboldenAndShearAMatchedFace();
  aParagraphWrapsAlignsAndStacks();
  anExplicitNewlineBreaksTheLine();
  theMeasuredWidthIsTheDrawnWidth();
  aCoordinateResolvesToACaretAndARangeToItsBoxes();
  rightToLeftTextReadsRightToLeft();
  arabicTakesItsContextualForms();
  mixedDirectionResolvesIntoRuns();
  bidiMarksPlaceNeutralsAndIndicesStayLogical();
  textOnAPathIsReorderedToo();
  theLayoutIsCachedByQuantisedSize();
  anEmojiFallsBackInsteadOfVanishing();
  textOnAPathFollowsTheCurve();
  aRegisteredImageRasterizesForEncoding();
  anOffscreenSurfaceRendersSlotsAndViewport();
  encodingUnpremultipliesAndRejectsLiveFrames();
  anOffscreenSurfaceHitTestsBeforeTheFirstDrain();
  batchClusteringKeepsOverlapOrderAndPixels();
  preserve3dKeepsTheGroupsOwnClipAndLayer();
  aPlaneOfA3DContextKeepsItsClipAndLayer();
  aFillOnA3DPlaneDraws();
  aZeroPerspectiveDrawsFlat();
  hitTestComposesA3DContextNearestFirst();
  a3DPlaneRedrawsLiveContentAndKeepsStaticContent();
  std::printf(failures == 0 ? "SceneTest: ok\n" : "SceneTest: %d failure(s)\n",
              failures);
  return failures == 0 ? 0 : 1;
}
