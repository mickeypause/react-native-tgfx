#include "TgfxScene.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <iterator>
#include <limits>
#include <numbers>

#include "BufferRegistry.hpp"
#include "FontRegistry.hpp"
#include "ImageRegistry.hpp"
#include "Log.hpp"
#include "SvgPath.hpp"
#include "TextLayout.hpp"
#include "tgfx/core/Matrix3D.h"
#include "tgfx/core/PictureRecorder.h"
#include "tgfx/layers/ImageLayer.h"
#include "tgfx/layers/Layer.h"
#include "tgfx/layers/layerstyles/GlassStyle.h"
#include "tgfx/layers/layerstyles/NoiseStyle.h"
#include "tgfx/core/Canvas.h"
#include "tgfx/core/Color.h"
#include "tgfx/core/ColorFilter.h"
#include "tgfx/core/Image.h"
#include "tgfx/core/ImageFilter.h"
#include "tgfx/core/Paint.h"
#include "tgfx/core/PathEffect.h"
#include "tgfx/core/Point.h"
#include "tgfx/core/RRect.h"
#include "tgfx/core/Rect.h"
#include "tgfx/core/SamplingOptions.h"
#include "tgfx/core/Shader.h"
#include "tgfx/core/Stroke.h"
#include "tgfx/core/Surface.h"
#include "tgfx/core/TileMode.h"
#include "tgfx/core/TextBlob.h"

namespace tgfxrn {

/// Cursor over the binary draw program from src/compiler/writer.ts; fields are read in write order.
struct SceneReader {
  SceneReader(std::span<const double> words, std::span<const double> patches,
              std::span<const double> stringPatches,
              const std::vector<std::string>& strings)
      : words(words),
        patches(patches),
        stringPatches(stringPatches),
        strings(strings) {}

  std::span<const double> words;
  std::span<const double> patches;
  std::span<const double> stringPatches;
  const std::vector<std::string>& strings;
  size_t at = 3;  // past the header
  size_t patchAt = 0;
  size_t stringPatchAt = 0;
  bool ok = true;

  size_t remaining() const {
    return at < words.size() ? words.size() - at : 0;
  }

  double next() {
    if (at >= words.size()) {
      ok = false;
      return 0;
    }
    return words[at++];
  }

  int nextInt() {
    double v = next();
    // Casting a NaN or out-of-range double to int is undefined behaviour.
    if (!(v >= std::numeric_limits<int>::min() &&
          v <= std::numeric_limits<int>::max())) {
      ok = false;
      return 0;
    }
    return static_cast<int>(v);
  }

  /// How much read so far resolves per frame; a group records only if it adds none.
  int dynamic = 0;

  AnimF nextAnim() {
    AnimF f;
    // Compared as a double: casting a NaN or negative wire value to size_t is UB.
    if (patchAt + 1 < patches.size() &&
        patches[patchAt] == static_cast<double>(at)) {
      double slot = patches[patchAt + 1];
      f.slot = slot >= 0 && slot <= std::numeric_limits<int>::max()
                   ? static_cast<int>(slot)
                   : -1;
      patchAt += 2;
      if (f.slot >= 0) {
        // The one choke point for value slots: the writer dedupes, so counting them in JS misses reuse.
        dynamic++;
      }
    }
    f.literal = static_cast<float>(next());
    return f;
  }

  const std::string& nextString() {
    static const std::string kEmpty;
    int index = nextInt();
    if (index < 0 || static_cast<size_t>(index) >= strings.size()) {
      ok = false;
      return kEmpty;
    }
    return strings[static_cast<size_t>(index)];
  }

  /// A string word that may follow a string slot (live text or path data).
  struct StringRef {
    const std::string* value;
    int slot = -1;
  };

  StringRef nextStringRef() {
    StringRef ref;
    if (stringPatchAt + 1 < stringPatches.size() &&
        stringPatches[stringPatchAt] == static_cast<double>(at)) {
      double slot = stringPatches[stringPatchAt + 1];
      ref.slot = slot >= 0 && slot <= std::numeric_limits<int>::max()
                     ? static_cast<int>(slot)
                     : -1;
      stringPatchAt += 2;
      if (ref.slot >= 0) {
        // Live content resolves per frame, which a picture recording would freeze.
        dynamic++;
      }
    }
    ref.value = &nextString();
    return ref;
  }
};

namespace {

/// Bounds readNode and so drawCmd too; its frame is ~1104 B, so 256 deep is ~276 KiB of stack.
constexpr int kMaxNodeDepth = 256;

/// A wire count clamped to the buffer; reserving on the raw value asks for terabytes.
size_t boundedCount(const SceneReader& r, int count) {
  return std::min(static_cast<size_t>(std::max(0, count)), r.remaining());
}

std::atomic<size_t> gFilterBuilds{0};

/// Node kinds, mirrored from src/compiler/writer.ts.
enum Kind {
  KGroup = 0,
  KRect,
  KRRect,
  KOval,
  KCircle,
  KLine,
  KPath,
  KImage,
  KText,
  KFill,
  KTextPath,
  KCameraFrame,
  KGlyphs,
  KGroupLayer,
  KMask,
  KAtlas,
  KVertices,
  KBackdrop
};

/// Sampling codes, mirrored from src/compiler/writer.ts.
tgfx::SamplingOptions samplingOf(int code) {
  switch (code) {
    case 0:
      return tgfx::SamplingOptions(tgfx::FilterMode::Nearest, tgfx::MipmapMode::None);
    case 2:
      // Mipmaps also need Image::makeMipmapped(); the filter alone does nothing.
      return tgfx::SamplingOptions(tgfx::FilterMode::Linear, tgfx::MipmapMode::Linear);
    default:
      return tgfx::SamplingOptions(tgfx::FilterMode::Linear, tgfx::MipmapMode::None);
  }
}

/// Blend modes travel as the index into tgfx's enum; BLEND_MODES mirrors that list.
tgfx::BlendMode blendModeOf(int code) {
  constexpr int kLast = static_cast<int>(tgfx::BlendMode::PlusDarker);
  if (code < 0 || code > kLast) {
    return tgfx::BlendMode::SrcOver;
  }
  return static_cast<tgfx::BlendMode>(code);
}

tgfx::TileMode tileModeOf(int code) {
  switch (code) {
    case 1: return tgfx::TileMode::Repeat;
    case 2: return tgfx::TileMode::Mirror;
    case 3: return tgfx::TileMode::Decal;
    default: return tgfx::TileMode::Clamp;
  }
}

/// Flutter's applyBoxFit + inscribe: `cover` crops the source, so no clip is needed.
struct FitRects {
  tgfx::Rect src;
  tgfx::Rect dst;
};

tgfx::Rect inscribe(float w, float h, const tgfx::Rect& bounds) {
  return tgfx::Rect::MakeXYWH(bounds.left + (bounds.width() - w) / 2,
                              bounds.top + (bounds.height() - h) / 2, w, h);
}

FitRects fitRects(Fit fit, const tgfx::Rect& source, const tgfx::Rect& dest) {
  float iw = source.width(), ih = source.height();
  float ow = dest.width(), oh = dest.height();
  if (iw <= 0 || ih <= 0 || ow <= 0 || oh <= 0) {
    return {source, dest};
  }
  // "Wider" in the Flutter sense: output aspect exceeds input aspect.
  bool wider = ow / oh > iw / ih;
  float sw = iw, sh = ih, dw = ow, dh = oh;
  switch (fit) {
    case Fit::Fill:
      break;
    case Fit::Contain:
      if (wider) {
        dw = iw * oh / ih;
      } else {
        dh = ih * ow / iw;
      }
      break;
    case Fit::Cover:
      if (wider) {
        sh = iw * oh / ow;
      } else {
        sw = ih * ow / oh;
      }
      break;
    case Fit::FitWidth:
      if (wider) {
        sh = iw * oh / ow;  // like cover
      } else {
        dh = ih * ow / iw;  // like contain
      }
      break;
    case Fit::FitHeight:
      if (wider) {
        dw = iw * oh / ih;  // like contain
      } else {
        sw = ih * ow / oh;  // like cover
      }
      break;
    case Fit::None:
      sw = dw = std::min(iw, ow);
      sh = dh = std::min(ih, oh);
      break;
    case Fit::ScaleDown:
      dw = iw;
      dh = ih;
      if (dh > oh) {
        dw = oh * (iw / ih);
        dh = oh;
      }
      if (dw > ow) {
        dh = ow / (iw / ih);
        dw = ow;
      }
      break;
  }
  return {inscribe(sw, sh, source), inscribe(dw, dh, dest)};
}

/// Frame pixels to node coordinates, so an overlay in frame pixels cannot drift.
tgfx::Matrix cameraMatrix(const DrawCmd& cmd, tgfx::Canvas* canvas, float w, float h) {
  tgfx::Matrix m = tgfx::Matrix::I();
  if (cmd.mirror) {
    m.postScale(-1, 1);
    m.postTranslate(w, 0);
  }
  int turns = cmd.quarterTurns & 3;
  if (turns != 0) {
    m.postRotate(static_cast<float>(turns * 90));
    // Rotation alone puts the frame in a negative quadrant; slide it back.
    switch (turns) {
      case 1: m.postTranslate(h, 0); break;
      case 2: m.postTranslate(w, h); break;
      default: m.postTranslate(0, w); break;
    }
  }
  float ow = (turns & 1) != 0 ? h : w;
  float oh = (turns & 1) != 0 ? w : h;

  auto* surface = canvas->getSurface();
  auto dest = surface != nullptr
                  ? tgfx::Rect::MakeWH(static_cast<float>(surface->width()),
                                       static_cast<float>(surface->height()))
                  : tgfx::Rect::MakeWH(ow, oh);
  tgfx::Matrix inverse;
  if (canvas->getMatrix().invert(&inverse)) {
    inverse.mapRect(&dest);
  }
  auto fitted = fitRects(cmd.fit, tgfx::Rect::MakeWH(ow, oh), dest);
  if (fitted.src.width() <= 0 || fitted.src.height() <= 0) {
    return m;
  }
  // Draw the whole frame through the fit matrix, so cover overflows instead of scaling.
  m.postTranslate(-fitted.src.left, -fitted.src.top);
  m.postScale(fitted.dst.width() / fitted.src.width(),
              fitted.dst.height() / fitted.src.height());
  m.postTranslate(fitted.dst.left, fitted.dst.top);
  return m;
}

/// The registry image for `handle`. Never decodes, so a render thread may call it.
std::shared_ptr<tgfx::Image> resolveImage(int handle,
                                          const tgfx::SamplingOptions& sampling,
                                          bool* external = nullptr) {
  auto image = ImageRegistry::instance().acquire(handle, external);
  if (image == nullptr) {
    if (handle != 0) {
      TGFXRN_LOGW("image handle %d has no pixels (disposed, or re-decoding)", handle);
    }
    return nullptr;
  }
  if (sampling.mipmapMode != tgfx::MipmapMode::None) {
    // The sampling options alone are ignored unless the image carries mipmaps.
    auto mipmapped = image->makeMipmapped(true);
    if (mipmapped != nullptr) {
      return mipmapped;
    }
  }
  return image;
}

tgfx::Color colorFromPacked(uint32_t c) {
  uint8_t a = (c >> 24) & 0xff;
  uint8_t r = (c >> 16) & 0xff;
  uint8_t g = (c >> 8) & 0xff;
  uint8_t b = c & 0xff;
  return tgfx::Color::FromRGBA(r, g, b, a);
}

/**
 * The v26 stops block into a GradientShaderCmd: a flags word (bit 1 = colours
 * travel as four live channels each), the colours, then the positions — each
 * position an anim word.
 */
void readStops(SceneReader& r, GradientShaderCmd& g) {
  g.animColors = (r.nextInt() & 1) != 0;
  int colorCount = r.nextInt();
  if (g.animColors) {
    size_t bounded = std::min(static_cast<size_t>(std::max(0, colorCount)),
                              r.remaining() / 4);
    g.colorChannels.reserve(bounded * 4);
    for (size_t i = 0; i < bounded * 4 && r.ok; i++) {
      g.colorChannels.push_back(r.nextAnim());
    }
  } else {
    g.colors.reserve(boundedCount(r, colorCount));
    for (int i = 0; i < colorCount && r.ok; i++) {
      g.colors.push_back(static_cast<uint32_t>(r.next()));
    }
  }
  int positionCount = r.nextInt();
  g.positions.reserve(boundedCount(r, positionCount));
  for (int i = 0; i < positionCount && r.ok; i++) {
    g.positions.push_back(r.nextAnim());
  }
}
tgfx::Rect rectOf(const DrawCmd& c, std::span<const double> v) {
  return tgfx::Rect::MakeXYWH(c.x.resolve(v), c.y.resolve(v), c.w.resolve(v),
                              c.h.resolve(v));
}

struct Affine {
  float a = 1, c = 0, e = 0, b = 0, d = 1, f = 0;
};

Affine multiply(const Affine& m1, const Affine& m2) {
  return Affine{m1.a * m2.a + m1.c * m2.b, m1.a * m2.c + m1.c * m2.d,
                m1.a * m2.e + m1.c * m2.f + m1.e, m1.b * m2.a + m1.d * m2.b,
                m1.b * m2.c + m1.d * m2.d, m1.b * m2.e + m1.d * m2.f + m1.f};
}

tgfx::Matrix matrixFromOps(const std::vector<TransformOpCmd>& ops,
                           std::span<const double> values) {
  Affine m;
  for (const auto& op : ops) {
    float a = op.a.resolve(values);
    float b = op.b.resolve(values);
    Affine o;
    switch (op.type) {
      case 0: o = Affine{1, 0, a, 0, 1, b}; break;
      case 1: o = Affine{a, 0, 0, 0, b, 0}; break;
      case 2: {
        float cs = std::cos(a), sn = std::sin(a);
        o = Affine{cs, -sn, 0, sn, cs, 0};
        break;
      }
      case 3: o = Affine{1, std::tan(a), 0, 0, 1, 0}; break;
      case 4: o = Affine{1, 0, 0, std::tan(a), 1, 0}; break;
      default: break;
    }
    m = multiply(m, o);
  }
  return tgfx::Matrix::MakeAll(m.a, m.c, m.e, m.b, m.d, m.f);
}

float degrees(float radians) {
  return radians * (180.0f / std::numbers::pi_v<float>);
}

/// Composed like matrixFromOps, from the Matrix3D builders tgfx's own LayerTest uses.
tgfx::Matrix3D matrix3DFromOps(const std::vector<TransformOpCmd>& ops,
                               std::span<const double> values) {
  auto m = tgfx::Matrix3D::I();
  for (const auto& op : ops) {
    float a = op.a.resolve(values);
    float b = op.b.resolve(values);
    auto o = tgfx::Matrix3D::I();
    switch (op.type) {
      case 0: o = tgfx::Matrix3D::MakeTranslate(a, b, 0); break;
      case 1: o = tgfx::Matrix3D::MakeScale(a, b, 1); break;
      case 2: o = tgfx::Matrix3D::MakeRotate({0, 0, 1}, degrees(a)); break;
      case 3: o = tgfx::Matrix3D(tgfx::Matrix::MakeSkew(std::tan(a), 0)); break;
      case 4: o = tgfx::Matrix3D(tgfx::Matrix::MakeSkew(0, std::tan(a))); break;
      case 5: o = tgfx::Matrix3D::MakeRotate({1, 0, 0}, degrees(a)); break;
      case 6: o = tgfx::Matrix3D::MakeRotate({0, 1, 0}, degrees(a)); break;
      case 7:
        if (a != 0) {
          o.setRowColumn(3, 2, -1 / a);
        }
        break;
      case 8: o = tgfx::Matrix3D::MakeTranslate(0, 0, a); break;
      default: break;
    }
    m = m * o;
  }
  return m;
}

/// Clip, layer, picture and grain flatten preserve3d, like CSS grouping properties.
bool opensContext(const DrawCmd& cmd) {
  return cmd.kind == DrawCmd::Group && cmd.preserve3D && cmd.clipKind == 0 && !cmd.layer &&
         !cmd.cachePicture && cmd.grain == nullptr;
}

bool hidden(const DrawCmd& cmd, std::span<const double> v) {
  return (cmd.visible.slot >= 0 || cmd.visible.literal < 0.5f) && cmd.visible.resolve(v) < 0.5f;
}

/// The group's transform as the canvas takes it: a 3D one flattened to its projective 3x3.
tgfx::Matrix groupMatrix(const DrawCmd& cmd, std::span<const double> values) {
  if (cmd.hasMatrix) {
    return cmd.matrix;
  }
  return cmd.threeD ? matrix3DFromOps(cmd.ops, values).asMatrix()
                    : matrixFromOps(cmd.ops, values);
}

uint8_t channelOf(const AnimF& f, std::span<const double> v) {
  return static_cast<uint8_t>(std::clamp(f.resolve(v), 0.0f, 1.0f) * 255.0f +
                              0.5f);
}

void applyPaint(const PaintCmd& pc, std::span<const double> v,
                tgfx::Paint& paint) {
  paint.reset();
  tgfx::Color color =
      pc.animColor ? tgfx::Color::FromRGBA(channelOf(pc.rgba[0], v),
                                           channelOf(pc.rgba[1], v),
                                           channelOf(pc.rgba[2], v),
                                           channelOf(pc.rgba[3], v))
                   : colorFromPacked(pc.color);
  paint.setColor(color);
  if (pc.runtime != nullptr) {
    paint.setShader(pc.runtime->build(v));
  } else if (pc.blendShader != nullptr) {
    paint.setShader(pc.blendShader->build(v));
  } else if (pc.gradient != nullptr) {
    paint.setShader(pc.gradient->build(v));
  } else if (pc.imageShader != nullptr) {
    paint.setShader(pc.imageShader->build(resolveImage(
        pc.imageShader->handle.resolveInt(v), pc.imageShader->sampling)));
  } else if (pc.shader != nullptr) {
    paint.setShader(pc.shader);
  }
  bool stroke = pc.animStyle ? pc.styleWord.resolveInt(v) == 1 : pc.stroke;
  if (stroke) {
    paint.setStyle(tgfx::PaintStyle::Stroke);
    paint.setStrokeWidth(pc.strokeWidth.resolve(v));
    switch (pc.cap.resolveInt(v)) {
      case 1: paint.setLineCap(tgfx::LineCap::Round); break;
      case 2: paint.setLineCap(tgfx::LineCap::Square); break;
      default: paint.setLineCap(tgfx::LineCap::Butt); break;
    }
    switch (pc.join.resolveInt(v)) {
      case 1: paint.setLineJoin(tgfx::LineJoin::Round); break;
      case 2: paint.setLineJoin(tgfx::LineJoin::Bevel); break;
      default: paint.setLineJoin(tgfx::LineJoin::Miter); break;
    }
    paint.setMiterLimit(pc.miter.resolve(v));
  } else {
    paint.setStyle(tgfx::PaintStyle::Fill);
  }
  paint.setAntiAlias(pc.animStyle ? pc.aaWord.resolve(v) >= 0.5f
                                  : pc.antiAlias);
  paint.setAlpha(color.alpha * pc.alpha.resolve(v));
  paint.setBlendMode(pc.blendWord.slot >= 0
                         ? blendModeOf(pc.blendWord.resolveInt(v))
                         : pc.blend);
  paint.setColorFilter(pc.colorFilterCmd != nullptr ? pc.colorFilterCmd->build(v)
                                                    : pc.colorFilter);
  paint.setImageFilter(pc.imageFilterCmd != nullptr ? pc.imageFilterCmd->build(v)
                                                    : pc.imageFilter);
}

}  // namespace

void PathEffectCmd::apply(tgfx::Path* path, std::span<const double> v) const {
  // Corner and dash rebuild the path and drop its fill type, so carry it across them.
  auto fill = path->getFillType();
  if (trim) {
    if (auto effect = tgfx::PathEffect::MakeTrim(start.resolve(v), end.resolve(v))) {
      effect->filterPath(path);
    }
  }
  if (corner > 0) {
    if (auto effect = tgfx::PathEffect::MakeCorner(corner)) {
      effect->filterPath(path);
    }
  }
  if (dash) {
    if (auto effect = tgfx::PathEffect::MakeDash(intervals.data(),
                                                 static_cast<int>(intervals.size()),
                                                 phase.resolve(v))) {
      effect->filterPath(path);
    }
  }
  path->setFillType(fill);
}

/// A missing child drops out of the fold rather than blanking everything after it.
std::shared_ptr<tgfx::Shader> blendOnto(tgfx::BlendMode mode,
                                        std::shared_ptr<tgfx::Shader> folded,
                                        std::shared_ptr<tgfx::Shader> next) {
  if (next == nullptr) {
    return folded;
  }
  if (folded == nullptr) {
    return next;
  }
  return tgfx::Shader::MakeBlend(mode, std::move(folded), std::move(next));
}

std::shared_ptr<tgfx::Shader> ChildShaderCmd::resolve(
    std::span<const double> values) const {
  if (blend != nullptr) {
    return blend->build(values);
  }
  if (image != nullptr) {
    return image->build(
        resolveImage(image->handle.resolveInt(values), image->sampling));
  }
  if (runtime != nullptr) {
    return runtime->build(values);
  }
  if (gradient != nullptr) {
    return gradient->build(values);
  }
  return shader;
}

std::shared_ptr<tgfx::Shader> BlendShaderCmd::build(
    std::span<const double> values) const {
  std::shared_ptr<tgfx::Shader> folded;
  for (const auto& child : children) {
    folded = blendOnto(mode, std::move(folded), child.resolve(values));
  }
  return folded;
}

std::shared_ptr<tgfx::Shader> RuntimeShaderCmd::build(
    std::span<const double> values) const {
  auto resolved = uniforms;
  size_t at = 0;
  for (auto& uniform : resolved) {
    for (auto& value : uniform.values) {
      value = slots[at++].resolve(values);
    }
  }
  std::vector<std::shared_ptr<tgfx::Shader>> resolvedChildren;
  resolvedChildren.reserve(children.size());
  for (const auto& child : children) {
    auto shader = child.resolve(values);
    if (shader == nullptr) {
      return nullptr;
    }
    resolvedChildren.push_back(std::move(shader));
  }
  return tgfx::Shader::MakeRuntimeShader(body, std::move(resolved), preamble,
                                         functionNames,
                                         std::move(resolvedChildren));
}

bool GradientShaderCmd::animated() const {
  for (const auto& word : geometry) {
    if (word.slot >= 0) {
      return true;
    }
  }
  for (const auto& word : colorChannels) {
    if (word.slot >= 0) {
      return true;
    }
  }
  for (const auto& word : positions) {
    if (word.slot >= 0) {
      return true;
    }
  }
  return false;
}

std::shared_ptr<tgfx::Shader> GradientShaderCmd::build(
    std::span<const double> v) const {
  std::vector<tgfx::Color> resolvedColors;
  if (animColors) {
    resolvedColors.reserve(colorChannels.size() / 4);
    for (size_t i = 0; i + 3 < colorChannels.size(); i += 4) {
      resolvedColors.push_back(tgfx::Color::FromRGBA(
          channelOf(colorChannels[i], v), channelOf(colorChannels[i + 1], v),
          channelOf(colorChannels[i + 2], v),
          channelOf(colorChannels[i + 3], v)));
    }
  } else {
    resolvedColors.reserve(colors.size());
    for (uint32_t packed : colors) {
      resolvedColors.push_back(colorFromPacked(packed));
    }
  }
  if (resolvedColors.size() < 2) {
    return nullptr;
  }
  std::vector<float> resolvedPositions;
  resolvedPositions.reserve(positions.size());
  for (const auto& word : positions) {
    resolvedPositions.push_back(word.resolve(v));
  }
  float a = geometry[0].resolve(v);
  float b = geometry[1].resolve(v);
  float c = geometry[2].resolve(v);
  float d = geometry[3].resolve(v);
  switch (kind) {
    case 2:
      return tgfx::Shader::MakeConicGradient({a, b}, c, d, resolvedColors,
                                             resolvedPositions);
    case 1:
      return tgfx::Shader::MakeRadialGradient({a, b}, c, resolvedColors,
                                              resolvedPositions);
    case 3:
      return tgfx::Shader::MakeDiamondGradient({a, b}, c, resolvedColors,
                                               resolvedPositions);
    default:
      return tgfx::Shader::MakeLinearGradient({a, b}, {c, d}, resolvedColors,
                                              resolvedPositions);
  }
}

std::shared_ptr<tgfx::Shader> ImageShaderCmd::build(
    const std::shared_ptr<tgfx::Image>& image) const {
  if (image == nullptr) {
    return nullptr;
  }
  auto shader = tgfx::Shader::MakeImageShader(image, tileX, tileY, sampling);
  if (shader == nullptr || !hasRect) {
    return shader;
  }
  // The shader samples in image space, so `fit`/`rect` become its matrix.
  auto bounds = tgfx::Rect::MakeWH(static_cast<float>(image->width()),
                                   static_cast<float>(image->height()));
  auto fitted = fitRects(fit, bounds, rect);
  if (fitted.src.width() <= 0 || fitted.src.height() <= 0) {
    return shader;
  }
  auto matrix = tgfx::Matrix::MakeTrans(-fitted.src.left, -fitted.src.top);
  matrix.postScale(fitted.dst.width() / fitted.src.width(),
                   fitted.dst.height() / fitted.src.height());
  matrix.postTranslate(fitted.dst.left, fitted.dst.top);
  return shader->makeWithMatrix(matrix);
}

void SceneRenderer::readTextSpec(SceneReader& r, TextSpec& spec) {
  spec.family = r.nextString();
  spec.weight = r.nextInt();
  int style = r.nextInt();
  spec.italic = (style & 1) != 0;
  spec.fauxBold = (style & 2) != 0;
  spec.fauxItalic = (style & 4) != 0;
  spec.letterSpacing = static_cast<float>(r.next());
}

void SceneRenderer::readTextRuns(SceneReader& r, TextCmd& text) {
  int count = r.nextInt();
  if (count < 1 || count > kMaxTextRuns) {
    TGFXRN_LOGW("text node has %d runs, which is outside 1..%d", count, kMaxTextRuns);
    r.ok = false;
    text.runs.assign(1, TextRun{});
    return;
  }
  text.runs.resize(static_cast<size_t>(count));
  for (auto& run : text.runs) {
    auto content = r.nextStringRef();
    run.text = *content.value;
    run.slot = content.slot;
    text.liveRuns = text.liveRuns || run.slot >= 0;
    run.family = r.nextString();
    run.weight = r.nextInt();
    // Not `!= 0`: 2 means "upright", which a bool would fold into "italic".
    run.italic = r.nextInt();
    run.letterSpacing = static_cast<float>(r.next());
    run.size = static_cast<float>(r.next());
    run.hasColor = r.nextInt() != 0;
    run.color = static_cast<uint32_t>(r.next());
  }
}

const TextLayout& TextCmd::resolve(float size,
                                   std::span<const std::string> strings,
                                   uint64_t stringGen,
                                   std::span<const uint8_t> pushed) const {
  // Quantised: getScalerContext keys its cache on the raw float and never erases. See #145.
  size = std::round(size * 4) / 4;
  bool contentChanged = false;
  if (liveRuns && stringGen != layoutGen) {
    // Only on a push, and only for slotted runs: an idle frame compares nothing.
    layoutGen = stringGen;
    for (auto& run : runs) {
      if (run.slot < 0 || static_cast<size_t>(run.slot) >= strings.size()) {
        continue;
      }
      // A slot no push has written keeps its compiled literal: a sparse push
      // must never blank a neighbour ("" is a legitimate pushed value).
      if (static_cast<size_t>(run.slot) >= pushed.size() ||
          pushed[static_cast<size_t>(run.slot)] == 0) {
        continue;
      }
      const auto& now = strings[static_cast<size_t>(run.slot)];
      if (run.text != now) {
        // Mutable by the same rule as the layout: only the render thread is here.
        const_cast<TextRun&>(run).text = now;
        contentChanged = true;
      }
    }
  }
  if (size == layoutSize && !contentChanged) {
    return layout;
  }
  layoutSize = size;
  layout = !glyphs.empty() ? layoutGlyphs(glyphs, spec, size)
           : onPath        ? layoutTextOnPath(runs, spec, size, path)
                           : layoutText(runs, spec, size);
  return layout;
}

void SceneRenderer::clearCaches() {
  std::lock_guard<std::mutex> lock(programMutex_);
  program_ = nullptr;
}

std::shared_ptr<tgfx::Shader> SceneRenderer::readShader(SceneReader& r, PaintCmd* out,
                                                        int depth) {
  if (depth > kMaxNodeDepth) {
    r.ok = false;
    return nullptr;
  }
  int kind = r.nextInt();
  if (kind < 0) {
    return nullptr;
  }
  if (kind == 6) {
    std::string body = r.nextString();
    // File-scope helpers: GLSL ES has no nested functions.
    std::string preamble = r.nextString();
    int functionCount = r.nextInt();
    std::vector<std::string> functionNames;
    functionNames.reserve(boundedCount(r, functionCount));
    for (int i = 0; i < functionCount && r.ok; i++) {
      functionNames.push_back(r.nextString());
    }
    int uniformCount = r.nextInt();
    std::vector<tgfx::RuntimeShaderUniform> uniforms;
    uniforms.reserve(boundedCount(r, uniformCount));
    std::vector<AnimF> slots;
    bool animated = false;
    for (int i = 0; i < uniformCount && r.ok; i++) {
      tgfx::RuntimeShaderUniform uniform;
      uniform.name = r.nextString();
      int size = r.nextInt();
      std::vector<AnimF> uniformSlots;
      for (int v = 0; v < size && r.ok; v++) {
        AnimF f = r.nextAnim();
        uniform.values.push_back(f.literal);
        uniformSlots.push_back(f);
      }
      // Dropped uniforms must drop their slots too, or the flat run desyncs.
      if (!uniform.name.empty() && !uniform.values.empty()) {
        uniforms.push_back(std::move(uniform));
        for (const auto& f : uniformSlots) {
          animated = animated || f.slot >= 0;
          slots.push_back(f);
        }
      }
    }
    int childCount = r.nextInt();
    std::vector<ChildShaderCmd> children;
    children.reserve(boundedCount(r, childCount));
    bool dynamicChild = false;
    for (int i = 0; i < childCount && r.ok; i++) {
      ChildShaderCmd child;
      PaintCmd childPaint;
      child.shader = readShader(r, &childPaint, depth + 1);
      child.image = std::move(childPaint.imageShader);
      child.runtime = std::move(childPaint.runtime);
      child.blend = std::move(childPaint.blendShader);
      child.gradient = std::move(childPaint.gradient);
      dynamicChild = dynamicChild || child.dynamic();
      children.push_back(std::move(child));
    }
    if ((animated || dynamicChild) && out != nullptr) {
      auto cmd = std::make_shared<RuntimeShaderCmd>();
      cmd->body = body;
      cmd->preamble = preamble;
      cmd->functionNames = functionNames;
      cmd->uniforms = uniforms;
      cmd->slots = std::move(slots);
      cmd->children = std::move(children);
      out->runtime = std::move(cmd);
      return nullptr;
    }
    std::vector<std::shared_ptr<tgfx::Shader>> staticChildren;
    staticChildren.reserve(children.size());
    for (auto& child : children) {
      staticChildren.push_back(std::move(child.shader));
    }
    return tgfx::Shader::MakeRuntimeShader(std::move(body), std::move(uniforms),
                                           std::move(preamble),
                                           std::move(functionNames),
                                           std::move(staticChildren));
  }
  if (kind == 7) {
    ImageShaderCmd cmd;
    cmd.handle = r.nextAnim();
    cmd.tileX = tileModeOf(r.nextInt());
    cmd.tileY = tileModeOf(r.nextInt());
    cmd.sampling = samplingOf(r.nextInt());
    cmd.hasRect = r.nextInt() != 0;
    if (cmd.hasRect) {
      float x = static_cast<float>(r.next());
      float y = static_cast<float>(r.next());
      float w = static_cast<float>(r.next());
      float h = static_cast<float>(r.next());
      cmd.rect = tgfx::Rect::MakeXYWH(x, y, w, h);
      cmd.fit = static_cast<Fit>(r.nextInt());
    }
    int handle = cmd.handle.resolveInt({});
    bool external = false;
    auto image = resolveImage(handle, cmd.sampling, &external);
    if ((external || image == nullptr || cmd.handle.slot >= 0) &&
        (handle != 0 || cmd.handle.slot >= 0) && out != nullptr) {
      // Baking here would freeze whatever the registry held at compile time.
      out->imageShader = std::make_shared<ImageShaderCmd>(cmd);
      return nullptr;
    }
    return cmd.build(image);
  }
  if (kind == 9) {
    return tgfx::Shader::MakeColorShader(colorFromPacked(static_cast<uint32_t>(r.next())));
  }
  if (kind == 8) {
    auto mode = blendModeOf(r.nextInt());
    int count = r.nextInt();
    // Fold as we go; the child vector is only built once a dynamic child turns up.
    std::shared_ptr<tgfx::Shader> folded;
    std::vector<ChildShaderCmd> children;
    for (int i = 0; i < count && r.ok; i++) {
      ChildShaderCmd child;
      PaintCmd childPaint;
      child.shader = readShader(r, &childPaint, depth + 1);
      child.image = std::move(childPaint.imageShader);
      child.runtime = std::move(childPaint.runtime);
      child.blend = std::move(childPaint.blendShader);
      child.gradient = std::move(childPaint.gradient);
      if (children.empty() && !child.dynamic()) {
        folded = blendOnto(mode, std::move(folded), std::move(child.shader));
        continue;
      }
      if (children.empty()) {
        children.reserve(boundedCount(r, count));
        if (folded != nullptr) {
          children.push_back(
              {std::move(folded), nullptr, nullptr, nullptr, nullptr});
        }
      }
      children.push_back(std::move(child));
    }
    if (children.empty()) {
      return folded;
    }
    auto cmd = std::make_shared<BlendShaderCmd>();
    cmd->mode = mode;
    cmd->children = std::move(children);
    if (out == nullptr) {
      return cmd->build({});
    }
    out->blendShader = std::move(cmd);
    return nullptr;
  }
  if (kind == 4 || kind == 5) {
    float fx = static_cast<float>(r.next());
    float fy = static_cast<float>(r.next());
    int octaves = r.nextInt();
    float seed = static_cast<float>(r.next());
    return kind == 4 ? tgfx::Shader::MakeFractalNoise(fx, fy, octaves, seed)
                     : tgfx::Shader::MakeTurbulence(fx, fy, octaves, seed);
  }

  // Gradients: geometry words then the stops block, all animatable. A static
  // one still builds exactly once, here; only a live one is rebuilt per frame.
  auto gradient = std::make_shared<GradientShaderCmd>();
  gradient->kind = kind;
  int geometryWords = kind == 0 || kind == 2 ? 4 : 3;
  for (int i = 0; i < geometryWords; i++) {
    gradient->geometry[static_cast<size_t>(i)] = r.nextAnim();
  }
  readStops(r, *gradient);
  if (!gradient->animated()) {
    return gradient->build({});
  }
  if (out != nullptr) {
    out->gradient = std::move(gradient);
    return nullptr;
  }
  return gradient->build({});
}

size_t filterBuildCount() {
  return gFilterBuilds.load(std::memory_order_relaxed);
}

bool ColorFilterCmd::animated() const {
  if (animColor || mode.slot >= 0 || threshold.slot >= 0) {
    return true;
  }
  for (const auto& word : matrix) {
    if (word.slot >= 0) {
      return true;
    }
  }
  return (inner != nullptr && inner->animated()) ||
         (outer != nullptr && outer->animated());
}

std::shared_ptr<tgfx::ColorFilter> ColorFilterCmd::build(
    std::span<const double> v) const {
  if (cached != nullptr) {
    return cached;
  }
  std::shared_ptr<tgfx::ColorFilter> filter;
  if (kind == 0) {
    std::array<float, 20> resolved{};
    for (size_t i = 0; i < resolved.size(); i++) {
      resolved[i] = matrix[i].resolve(v);
    }
    filter = tgfx::ColorFilter::Matrix(resolved);
  } else if (kind == 1) {
    auto c = animColor
                 ? tgfx::Color::FromRGBA(channelOf(rgba[0], v), channelOf(rgba[1], v),
                                         channelOf(rgba[2], v), channelOf(rgba[3], v))
                 : colorFromPacked(color);
    filter = tgfx::ColorFilter::Blend(c, blendModeOf(mode.resolveInt(v)));
  } else if (kind == 2) {
    filter = tgfx::ColorFilter::Luma();
  } else if (kind == 3) {
    filter = tgfx::ColorFilter::AlphaThreshold(threshold.resolve(v));
  } else if (kind == 4) {
    auto a = inner != nullptr ? inner->build(v) : nullptr;
    auto b = outer != nullptr ? outer->build(v) : nullptr;
    if (a == nullptr || b == nullptr) {
      return a != nullptr ? a : b;
    }
    filter = tgfx::ColorFilter::Compose(std::move(a), std::move(b));
  } else {
    return nullptr;
  }
  gFilterBuilds++;
  // Only what can change is rebuilt: a static leaf under an animated compose is
  // built on the first frame and read back on every one after.
  if (!animated()) {
    cached = filter;
  }
  return filter;
}

bool ImageFilterCmd::animated() const {
  return animColor || blurX.slot >= 0 || blurY.slot >= 0 || dx.slot >= 0 ||
         dy.slot >= 0 || (first != nullptr && first->animated()) ||
         (then != nullptr && then->animated()) ||
         (colorFilter != nullptr && colorFilter->animated());
}

std::shared_ptr<tgfx::ImageFilter> ImageFilterCmd::build(
    std::span<const double> v) const {
  if (cached != nullptr) {
    return cached;
  }
  std::shared_ptr<tgfx::ImageFilter> filter;
  if (kind == 0) {
    filter = tgfx::ImageFilter::Blur(blurX.resolve(v), blurY.resolve(v), tile);
  } else if (kind == 1) {
    auto c = animColor
                 ? tgfx::Color::FromRGBA(channelOf(rgba[0], v), channelOf(rgba[1], v),
                                         channelOf(rgba[2], v), channelOf(rgba[3], v))
                 : colorFromPacked(color);
    float x = dx.resolve(v);
    float y = dy.resolve(v);
    float bx = blurX.resolve(v);
    float by = blurY.resolve(v);
    bool only = (flags & 1) != 0;
    filter = (flags & 2) != 0
                 ? (only ? tgfx::ImageFilter::InnerShadowOnly(x, y, bx, by, c)
                         : tgfx::ImageFilter::InnerShadow(x, y, bx, by, c))
                 : (only ? tgfx::ImageFilter::DropShadowOnly(x, y, bx, by, c)
                         : tgfx::ImageFilter::DropShadow(x, y, bx, by, c));
  } else if (kind == 2) {
    auto a = first != nullptr ? first->build(v) : nullptr;
    auto b = then != nullptr ? then->build(v) : nullptr;
    if (a == nullptr || b == nullptr) {
      return a != nullptr ? a : b;
    }
    filter = tgfx::ImageFilter::Compose(std::move(a), std::move(b));
  } else if (kind == 3) {
    auto inner = colorFilter != nullptr ? colorFilter->build(v) : nullptr;
    if (inner == nullptr) {
      return nullptr;
    }
    filter = tgfx::ImageFilter::ColorFilter(std::move(inner));
  } else {
    return nullptr;
  }
  gFilterBuilds++;
  if (!animated()) {
    cached = filter;
  }
  return filter;
}

/// A grain colour: the packed word, a live flag, then the channels when live.
static void readGrainColor(SceneReader& r, uint32_t& packed, std::array<AnimF, 4>& rgba,
                           bool& live) {
  packed = static_cast<uint32_t>(r.next());
  live = r.nextInt() != 0;
  if (live) {
    for (auto& channel : rgba) {
      channel = r.nextAnim();
    }
  }
}

std::shared_ptr<ColorFilterCmd> SceneRenderer::readColorFilterCmd(SceneReader& r, int depth) {
  if (depth > kMaxNodeDepth) {
    r.ok = false;
    return nullptr;
  }
  int kind = r.nextInt();
  auto cmd = std::make_shared<ColorFilterCmd>();
  if (kind == 0) {
    cmd->kind = 0;
    for (auto& word : cmd->matrix) {
      word = r.nextAnim();
    }
  } else if (kind == 1 || kind == 5) {
    // Kind 5 is kind 1 with the colour as four live channels after the word.
    cmd->kind = 1;
    cmd->color = static_cast<uint32_t>(r.next());
    if (kind == 5) {
      cmd->animColor = true;
      for (auto& channel : cmd->rgba) {
        channel = r.nextAnim();
      }
    }
    cmd->mode = r.nextAnim();
  } else if (kind == 2) {
    cmd->kind = 2;
  } else if (kind == 3) {
    cmd->kind = 3;
    cmd->threshold = r.nextAnim();
  } else if (kind == 4) {
    cmd->kind = 4;
    // Read both sides whatever happens, or a null child would strand the cursor.
    cmd->inner = readColorFilterCmd(r, depth + 1);
    cmd->outer = readColorFilterCmd(r, depth + 1);
    if (cmd->inner == nullptr || cmd->outer == nullptr) {
      return cmd->inner != nullptr ? cmd->inner : cmd->outer;
    }
  } else {
    return nullptr;
  }
  return cmd;
}

std::shared_ptr<ImageFilterCmd> SceneRenderer::readImageFilterCmd(SceneReader& r, int depth) {
  if (depth > kMaxNodeDepth) {
    r.ok = false;
    return nullptr;
  }
  int kind = r.nextInt();
  auto cmd = std::make_shared<ImageFilterCmd>();
  if (kind == 0) {
    cmd->kind = 0;
    cmd->blurX = r.nextAnim();
    cmd->blurY = r.nextAnim();
    cmd->tile = tileModeOf(r.nextInt());
  } else if (kind == 1) {
    cmd->kind = 1;
    cmd->dx = r.nextAnim();
    cmd->dy = r.nextAnim();
    cmd->blurX = r.nextAnim();
    cmd->blurY = r.nextAnim();
    cmd->color = static_cast<uint32_t>(r.next());
    int flags = r.nextInt();
    cmd->flags = flags & 3;
    // Bit 4: the colour follows as four live channels.
    if ((flags & 4) != 0) {
      cmd->animColor = true;
      for (auto& channel : cmd->rgba) {
        channel = r.nextAnim();
      }
    }
  } else if (kind == 2) {
    cmd->kind = 2;
    cmd->first = readImageFilterCmd(r, depth + 1);
    cmd->then = readImageFilterCmd(r, depth + 1);
    if (cmd->first == nullptr || cmd->then == nullptr) {
      return cmd->first != nullptr ? cmd->first : cmd->then;
    }
  } else if (kind == 3) {
    cmd->kind = 3;
    cmd->colorFilter = readColorFilterCmd(r, depth + 1);
    if (cmd->colorFilter == nullptr) {
      return nullptr;
    }
  } else {
    return nullptr;
  }
  return cmd;
}

void SceneRenderer::readFilters(SceneReader& r, PaintCmd& paint) {
  paint.blendWord = r.nextAnim();
  paint.blend = blendModeOf(paint.blendWord.resolveInt({}));
  auto colorCmd = readColorFilterCmd(r, 0);
  if (colorCmd != nullptr) {
    if (colorCmd->animated()) {
      paint.colorFilterCmd = std::move(colorCmd);
    } else {
      paint.colorFilter = colorCmd->build({});
    }
  }
  auto imageCmd = readImageFilterCmd(r, 0);
  if (imageCmd != nullptr) {
    if (imageCmd->animated()) {
      paint.imageFilterCmd = std::move(imageCmd);
    } else {
      paint.imageFilter = imageCmd->build({});
    }
  }
  if (paint.imageFilter != nullptr && paint.blend != tgfx::BlendMode::SrcOver &&
      paint.blendWord.slot < 0) {
    // Taking the layer here keeps the blend mode tgfx would otherwise composite away.
    paint.layer = std::make_shared<tgfx::Paint>();
    paint.layer->setBlendMode(paint.blend);
    paint.layer->setImageFilter(std::move(paint.imageFilter));
    paint.blend = tgfx::BlendMode::SrcOver;
    paint.imageFilter = nullptr;
  }
}

void SceneRenderer::readPaint(SceneReader& r, PaintCmd& paint) {
  paint.color = static_cast<uint32_t>(r.next());
  int flags = r.nextInt();
  paint.stroke = (flags & 1) != 0;
  paint.antiAlias = (flags & 2) != 0;
  // Bit 32: style and antiAlias travel as their own (possibly live) words.
  paint.animStyle = (flags & 32) != 0;
  if (paint.animStyle) {
    paint.styleWord = r.nextAnim();
    paint.aaWord = r.nextAnim();
  }
  // Bit 8: the colour is animated and travels as four live channels.
  paint.animColor = (flags & 8) != 0;
  if (paint.animColor) {
    for (auto& channel : paint.rgba) {
      channel = r.nextAnim();
    }
  }
  paint.strokeWidth = r.nextAnim();
  paint.cap = r.nextAnim();
  paint.join = r.nextAnim();
  // Bit 16: a miter limit other than tgfx's default of 4.
  paint.miter = (flags & 16) != 0 ? r.nextAnim() : AnimF{4, -1};
  paint.alpha = r.nextAnim();
  // Bit 4: the paint carries a filter block; unset costs no words at all.
  if ((flags & 4) != 0) {
    readFilters(r, paint);
  }
  paint.shader = readShader(r, &paint);
  applyPaint(paint, {}, paint.cached);
}

/// One GPU batch per draw-op class run: tgfx's OpsCompositor merges only
/// *adjacent* compatible draws, so a sibling list that interleaves classes
/// (rect, circle, rect, ...) flushes a batch at nearly every node — at 20k
/// nodes that is ~13k pipeline binds a frame, and the render thread dies on
/// per-batch bookkeeping. Draws whose padded bounds are disjoint commute under
/// any blend mode, so static siblings may be clustered by batch class at
/// commit time — once, zero per-frame cost, identical pixels.
struct BatchCluster {
  uint64_t key = 0;
  /// Anything unprovable: never hopped over, never merged into.
  bool barrier = false;
  /// Per member, not a union: a cluster's bounding box quickly spans the whole
  /// canvas on a grid, which would read as "overlaps everything" and stop all
  /// clustering right when it matters most.
  std::vector<tgfx::Rect> memberBounds;
  std::vector<size_t> members;
};

/// 0 = tgfx's rect op, 1 = its round-rect op (oval and circle land there too),
/// 2 = its image op, which additionally batches per texture.
int batchClassOf(const DrawCmd& cmd) {
  switch (cmd.kind) {
    case DrawCmd::Rect: return 0;
    case DrawCmd::RRect:
    case DrawCmd::Oval:
    case DrawCmd::Circle: return 1;
    case DrawCmd::Image: return 2;
    default: return -1;
  }
}

/// The node's literal painted bounds, padded for the AA fringe; false when the
/// geometry can change per frame, which makes the node an immovable barrier.
bool literalShapeBounds(const DrawCmd& cmd, tgfx::Rect* out) {
  if (cmd.kind == DrawCmd::Circle) {
    if (cmd.cx.slot >= 0 || cmd.cy.slot >= 0 || cmd.r.slot >= 0) {
      return false;
    }
    float r = cmd.r.literal;
    *out = tgfx::Rect::MakeXYWH(cmd.cx.literal - r, cmd.cy.literal - r, r * 2, r * 2);
  } else {
    if (cmd.x.slot >= 0 || cmd.y.slot >= 0 || cmd.w.slot >= 0 || cmd.h.slot >= 0) {
      return false;
    }
    *out = tgfx::Rect::MakeXYWH(cmd.x.literal, cmd.y.literal, cmd.w.literal, cmd.h.literal);
  }
  // A draw touches exactly the pixels whose centers sit within half a pixel of
  // its geometry (the AA fringe), so half a pixel is the disjointness margin —
  // a full pixel would glue every 1px-gap grid into one immovable blob.
  out->outset(0.5f, 0.5f);
  return true;
}

bool clusterable(const DrawCmd& cmd, uint64_t* key, tgfx::Rect* bounds) {
  int opClass = batchClassOf(cmd);
  if (opClass < 0) {
    return false;
  }
  if (cmd.visible.slot >= 0 || cmd.visible.literal < 0.5f) {
    return false;
  }
  // Solid literal fills only. A stroke, shader, or filter changes both the
  // painted bounds and the batch identity; colour differences batch fine.
  const PaintCmd& p = cmd.paint;
  if (p.animated() || p.stroke || p.shader != nullptr || p.colorFilter != nullptr ||
      p.imageFilter != nullptr || p.layer != nullptr) {
    return false;
  }
  if (!literalShapeBounds(cmd, bounds)) {
    return false;
  }
  // Key layout: bits 0-1 op class, bit 2 antiAlias, bits 8-15 blend mode,
  // bits 16-47 image handle, bits 48-53 sampling (two 2-bit filter modes and a
  // 2-bit mipmap mode). Colour stays out: tgfx batches it per vertex.
  *key = static_cast<uint64_t>(opClass) | (p.antiAlias ? 4u : 0u) |
         (static_cast<uint64_t>(p.blend) << 8);
  if (cmd.kind == DrawCmd::Image) {
    // An image batch is per texture and per sampling; interleaving two handles
    // splits batches exactly the way interleaving rects and circles does.
    if (cmd.imageWord.slot >= 0 || cmd.animSrcRect) {
      return false;
    }
    *key |= static_cast<uint64_t>(static_cast<uint32_t>(cmd.imageWord.resolveInt({}))) << 16 |
            static_cast<uint64_t>(cmd.sampling.minFilterMode) << 48 |
            static_cast<uint64_t>(cmd.sampling.magFilterMode) << 50 |
            static_cast<uint64_t>(cmd.sampling.mipmapMode) << 52;
  }
  return true;
}

/// Reorders `children` so same-class disjoint draws sit together. A node may
/// only hop over clusters whose united bounds it provably does not touch, so
/// every move is between draws that commute; hit-test results cannot change
/// either, since a point can hit at most one of any pair this separates.
void clusterBatchableChildren(std::vector<DrawCmd>& children) {
  // Too few nodes to fragment into batches worth saving.
  if (children.size() < 8) {
    return;
  }
  std::vector<BatchCluster> clusters;
  clusters.reserve(children.size() / 2);
  // Bounds the exact pairwise checks per node, so a pathological list stops
  // paying instead of going quadratic. At 20k nodes this still cuts ~13k
  // batches to a few hundred.
  constexpr int kTestBudget = 1024;
  bool moved = false;
  for (size_t i = 0; i < children.size(); i++) {
    uint64_t key = 0;
    tgfx::Rect bounds = tgfx::Rect::MakeEmpty();
    if (!clusterable(children[i], &key, &bounds)) {
      clusters.push_back({0, true, {}, {i}});
      continue;
    }
    int target = -1;
    int budget = kTestBudget;
    for (int j = static_cast<int>(clusters.size()) - 1; j >= 0 && budget > 0; j--) {
      auto& cluster = clusters[static_cast<size_t>(j)];
      if (!cluster.barrier && cluster.key == key) {
        target = j;
        break;
      }
      if (cluster.barrier) {
        break;
      }
      bool blocked = false;
      for (const auto& member : cluster.memberBounds) {
        if (--budget < 0) {
          blocked = true;
          break;
        }
        if (tgfx::Rect::Intersects(member, bounds)) {
          blocked = true;
          break;
        }
      }
      if (blocked) {
        break;
      }
    }
    if (target >= 0) {
      auto& cluster = clusters[static_cast<size_t>(target)];
      moved = moved || target != static_cast<int>(clusters.size()) - 1;
      cluster.memberBounds.push_back(bounds);
      cluster.members.push_back(i);
    } else {
      clusters.push_back({key, false, {bounds}, {i}});
    }
  }
  if (!moved) {
    return;
  }
  std::vector<DrawCmd> ordered;
  ordered.reserve(children.size());
  for (auto& cluster : clusters) {
    for (size_t index : cluster.members) {
      ordered.push_back(std::move(children[index]));
    }
  }
  children = std::move(ordered);
}

/// Out of line: its four vectors would otherwise sit in every readNode frame, 256 deep.
[[gnu::noinline]] void readMesh(SceneReader& r, DrawCmd& cmd) {
  auto topology = r.nextInt() == 1 ? tgfx::MeshTopology::TriangleStrip
                                   : tgfx::MeshTopology::Triangles;
  int count = r.nextInt();
  // Two words per position, so the buffer bounds the count before any reserve.
  size_t vertexCount = std::min(static_cast<size_t>(std::max(0, count)), r.remaining() / 2);
  std::vector<tgfx::Point> positions;
  positions.reserve(vertexCount);
  for (size_t i = 0; i < vertexCount && r.ok; i++) {
    float x = static_cast<float>(r.next());
    float y = static_cast<float>(r.next());
    positions.push_back({x, y});
  }
  std::vector<tgfx::Point> texCoords;
  if (r.nextInt() != 0) {
    texCoords.reserve(positions.size());
    for (size_t i = 0; i < positions.size() && r.ok; i++) {
      float x = static_cast<float>(r.next());
      float y = static_cast<float>(r.next());
      texCoords.push_back({x, y});
    }
  }
  std::vector<tgfx::Color> colors;
  if (r.nextInt() != 0) {
    colors.reserve(positions.size());
    for (size_t i = 0; i < positions.size() && r.ok; i++) {
      colors.push_back(colorFromPacked(static_cast<uint32_t>(r.next())));
    }
  }
  std::vector<uint16_t> indices;
  int indexCount = r.nextInt();
  size_t bounded = boundedCount(r, indexCount);
  indices.reserve(bounded);
  bool outOfRange = false;
  for (size_t i = 0; i < bounded && r.ok; i++) {
    // Through nextInt, not a raw cast: an out-of-range double is undefined behaviour.
    int index = r.nextInt();
    outOfRange |= index < 0 || static_cast<size_t>(index) >= positions.size();
    indices.push_back(static_cast<uint16_t>(index & 0xffff));
  }
  if (outOfRange) {
    // Nothing rejects these downstream, so an index past the end reads another draw's buffer.
    TGFXRN_LOGW("vertices: an index is outside the %zu given, so the mesh is skipped",
                positions.size());
    return;
  }
  if (r.ok && !positions.empty()) {
    cmd.mesh = tgfx::Mesh::MakeCopy(
        topology, static_cast<int>(positions.size()), positions.data(),
        texCoords.empty() ? nullptr : texCoords.data(),
        colors.empty() ? nullptr : colors.data(), static_cast<int>(indices.size()),
        indices.empty() ? nullptr : indices.data());
    if (cmd.mesh == nullptr) {
      TGFXRN_LOGW("vertices: %zu of them could not build a mesh; an indexed one caps at 65536",
                  positions.size());
    }
  }
}

bool SceneRenderer::readNode(SceneReader& r, DrawCmd& cmd, int depth) {
  int dynamicBefore = r.dynamic;
  bool ok = readNodeFields(r, cmd, depth);
  cmd.dynamic = r.dynamic != dynamicBefore;
  cmd.fills = cmd.kind == DrawCmd::Fill ||
              std::any_of(cmd.children.begin(), cmd.children.end(),
                          [](const DrawCmd& child) { return child.fills; });
  return ok;
}

bool SceneRenderer::readNodeFields(SceneReader& r, DrawCmd& cmd, int depth) {
  if (depth > kMaxNodeDepth) {
    r.ok = false;
    return false;
  }
  int kind = r.nextInt();
  if (!r.ok) {
    return false;
  }
  // Bit 64: a visibility word follows the kind. See pushKind() in compile.ts.
  if ((kind & 64) != 0) {
    kind &= ~64;
    cmd.visible = r.nextAnim();
  }
  switch (kind) {
    case KGroup:
    case KGroupLayer: {
      cmd.kind = DrawCmd::Group;
      int opCount = r.nextInt();
      cmd.ops.reserve(boundedCount(r, opCount));
      bool animated = false;
      for (int i = 0; i < opCount && r.ok; i++) {
        TransformOpCmd op;
        op.type = r.nextInt();
        op.a = r.nextAnim();
        op.b = r.nextAnim();
        animated = animated || op.a.slot >= 0 || op.b.slot >= 0;
        cmd.threeD = cmd.threeD || op.type >= kFirst3DOp;
        cmd.ops.push_back(op);
      }
      if (!animated && !cmd.ops.empty()) {
        cmd.hasMatrix = true;
        cmd.matrix = cmd.threeD ? matrix3DFromOps(cmd.ops, {}).asMatrix()
                                : matrixFromOps(cmd.ops, {});
      }
      // Only KGroupLayer carries the clip/layer block; a plain group is unchanged.
      if (kind == KGroupLayer) {
        int flags = r.nextInt();
        cmd.invertClip = (flags & 2) != 0;
        cmd.layer = (flags & 4) != 0;
        cmd.cachePicture = (flags & 8) != 0;
        cmd.preserve3D = (flags & 32) != 0;
        if ((flags & 1) != 0) {
          cmd.clipKind = r.nextInt();
          if (cmd.clipKind == 3) {
            cmd.clipPath = std::make_shared<tgfx::Path>(parseSvgPath(r.nextString()));
          } else {
            float x = static_cast<float>(r.next());
            float y = static_cast<float>(r.next());
            cmd.clipRect = tgfx::Rect::MakeXYWH(x, y, static_cast<float>(r.next()),
                                                static_cast<float>(r.next()));
            if (cmd.clipKind == 2) {
              cmd.clipRadiusX = static_cast<float>(r.next());
              cmd.clipRadiusY = static_cast<float>(r.next());
            }
          }
          if (cmd.invertClip && cmd.clipKind != 3) {
            // Only a path can carry an inverse fill, so a rect clip becomes one.
            tgfx::Path box;
            if (cmd.clipKind == 2) {
              box.addRRect(tgfx::RRect::MakeRectXY(cmd.clipRect, cmd.clipRadiusX,
                                                   cmd.clipRadiusY));
            } else {
              box.addRect(cmd.clipRect);
            }
            cmd.clipPath = std::make_shared<tgfx::Path>(std::move(box));
            cmd.clipKind = 3;
          }
          if (cmd.invertClip && cmd.clipPath != nullptr) {
            cmd.clipPath->toggleInverseFillType();
          }
        }
        if (cmd.layer) {
          cmd.layerAlpha = r.nextAnim();
        }
        if ((flags & 16) != 0) {
          auto grain = std::make_shared<GrainCmd>();
          grain->mode = r.nextInt();
          grain->size = r.nextAnim();
          grain->density = r.nextAnim();
          grain->seed = r.nextAnim();
          grain->blendMode = r.nextAnim();
          if (grain->mode == 2) {
            grain->opacity = r.nextAnim();
          } else {
            readGrainColor(r, grain->color, grain->rgba, grain->animColor);
            if (grain->mode == 1) {
              readGrainColor(r, grain->secondColor, grain->secondRgba,
                             grain->animSecondColor);
            }
          }
          // A recorder canvas has an identity matrix and no surface, so a
          // grained group inside a recorded one would be dusted at scale 1 and
          // replayed under the real matrix. Counting the block as dynamic takes
          // the ancestor off the recording path, the way a video handle does.
          r.dynamic++;
          cmd.grain = std::move(grain);
        }
      }
      int childCount = r.nextInt();
      cmd.children.reserve(boundedCount(r, childCount));
      int dynamicBefore = r.dynamic;
      for (int i = 0; i < childCount && r.ok; i++) {
        DrawCmd child;
        if (!readNode(r, child, depth + 1)) {
          return false;
        }
        cmd.children.push_back(std::move(child));
      }
      if (cmd.cachePicture && r.dynamic != dynamicBefore) {
        TGFXRN_LOGW(
            "<Group picture> holds something that resolves per frame, so it is drawn "
            "directly; a recording would freeze it");
        cmd.cachePicture = false;
      }
      if (r.ok) {
        clusterBatchableChildren(cmd.children);
      }
      return r.ok;
    }
    case KMask: {
      cmd.kind = DrawCmd::Mask;
      cmd.maskLuminance = r.nextInt() != 0;
      int maskCount = r.nextInt();
      cmd.maskChildren.reserve(boundedCount(r, maskCount));
      for (int i = 0; i < maskCount && r.ok; i++) {
        DrawCmd child;
        if (!readNode(r, child, depth + 1)) {
          return false;
        }
        cmd.maskChildren.push_back(std::move(child));
      }
      int childCount = r.nextInt();
      cmd.children.reserve(boundedCount(r, childCount));
      int dynamicBefore = r.dynamic;
      for (int i = 0; i < childCount && r.ok; i++) {
        DrawCmd child;
        if (!readNode(r, child, depth + 1)) {
          return false;
        }
        cmd.children.push_back(std::move(child));
      }
      if (cmd.cachePicture && r.dynamic != dynamicBefore) {
        TGFXRN_LOGW(
            "<Group picture> holds something that resolves per frame, so it is drawn "
            "directly; a recording would freeze it");
        cmd.cachePicture = false;
      }
      return r.ok;
    }
    case KFill:
      cmd.kind = DrawCmd::Fill;
      break;
    case KCameraFrame: {
      // Like a group, but the transform comes from the live frame, with no re-commit.
      cmd.kind = DrawCmd::CameraFrame;
      // Always a new frame, and its matrix needs a surface that a recording canvas lacks.
      r.dynamic++;
      cmd.imageHandle = r.nextInt();
      cmd.fit = static_cast<Fit>(r.nextInt());
      cmd.sampling = samplingOf(r.nextInt());
      cmd.quarterTurns = r.nextInt();
      cmd.mirror = r.nextInt() != 0;
      readPaint(r, cmd.paint);
      int childCount = r.nextInt();
      cmd.children.reserve(boundedCount(r, childCount));
      int dynamicBefore = r.dynamic;
      for (int i = 0; i < childCount && r.ok; i++) {
        DrawCmd child;
        if (!readNode(r, child, depth + 1)) {
          return false;
        }
        cmd.children.push_back(std::move(child));
      }
      if (cmd.cachePicture && r.dynamic != dynamicBefore) {
        TGFXRN_LOGW(
            "<Group picture> holds something that resolves per frame, so it is drawn "
            "directly; a recording would freeze it");
        cmd.cachePicture = false;
      }
      return r.ok;
    }
    case KRect:
      cmd.kind = DrawCmd::Rect;
      cmd.x = r.nextAnim();
      cmd.y = r.nextAnim();
      cmd.w = r.nextAnim();
      cmd.h = r.nextAnim();
      break;
    case KRRect:
      cmd.kind = DrawCmd::RRect;
      cmd.x = r.nextAnim();
      cmd.y = r.nextAnim();
      cmd.w = r.nextAnim();
      cmd.h = r.nextAnim();
      cmd.rx = r.nextAnim();
      cmd.ry = r.nextAnim();
      break;
    case KOval:
      cmd.kind = DrawCmd::Oval;
      cmd.x = r.nextAnim();
      cmd.y = r.nextAnim();
      cmd.w = r.nextAnim();
      cmd.h = r.nextAnim();
      break;
    case KCircle:
      cmd.kind = DrawCmd::Circle;
      cmd.cx = r.nextAnim();
      cmd.cy = r.nextAnim();
      cmd.r = r.nextAnim();
      break;
    case KLine:
      cmd.kind = DrawCmd::Line;
      cmd.x1 = r.nextAnim();
      cmd.y1 = r.nextAnim();
      cmd.x2 = r.nextAnim();
      cmd.y2 = r.nextAnim();
      cmd.lineAnimated = cmd.x1.slot >= 0 || cmd.y1.slot >= 0 ||
                         cmd.x2.slot >= 0 || cmd.y2.slot >= 0;
      if (!cmd.lineAnimated) {
        cmd.path.moveTo(cmd.x1.literal, cmd.y1.literal);
        cmd.path.lineTo(cmd.x2.literal, cmd.y2.literal);
      }
      break;
    case KPath: {
      cmd.kind = DrawCmd::Path;
      auto data = r.nextStringRef();
      cmd.pathSlot = data.slot;
      cmd.path = parseSvgPath(*data.value);
      // Through contains, not getBounds: it is SkPathRef's own lazy bounds that a later
      // hitTest would otherwise compute while the render thread is reading them.
      (void)cmd.path.contains(0, 0);
      int flags = r.nextInt();
      cmd.pathFlags = flags;
      if ((flags & 8) != 0) {
        cmd.path.setFillType(tgfx::PathFillType::EvenOdd);
      }
      if ((flags & 7) != 0) {
        auto effects = std::make_shared<PathEffectCmd>();
        effects->trim = (flags & 1) != 0;
        if (effects->trim) {
          effects->start = r.nextAnim();
          effects->end = r.nextAnim();
        }
        if ((flags & 2) != 0) {
          int count = r.nextInt();
          effects->intervals.reserve(boundedCount(r, count));
          for (int i = 0; i < count && r.ok; i++) {
            effects->intervals.push_back(static_cast<float>(r.next()));
          }
          // MakeDash also rejects an odd count, so match it rather than build a dead effect.
          effects->dash = effects->intervals.size() >= 2 && effects->intervals.size() % 2 == 0;
          effects->phase = r.nextAnim();
        }
        if ((flags & 4) != 0) {
          effects->corner = static_cast<float>(r.next());
        }
        // Bake the longest static prefix now; only what actually moves stays per frame.
        // A slotted path cannot bake at all: every re-parse needs the full effect chain.
        bool movingTrim = effects->trim && (effects->start.slot >= 0 || effects->end.slot >= 0);
        bool movingDash = effects->dash && effects->phase.slot >= 0;
        if (movingTrim || cmd.pathSlot >= 0) {
          cmd.effects = std::move(effects);
        } else {
          PathEffectCmd baked = *effects;
          baked.dash = effects->dash && !movingDash;
          baked.apply(&cmd.path, {});
          if (movingDash) {
            auto rest = std::make_shared<PathEffectCmd>();
            rest->dash = true;
            rest->intervals = effects->intervals;
            rest->phase = effects->phase;
            cmd.effects = std::move(rest);
          }
        }
      }
      break;
    }
    case KImage: {
      cmd.kind = DrawCmd::Image;
      cmd.x = r.nextAnim();
      cmd.y = r.nextAnim();
      cmd.w = r.nextAnim();
      cmd.h = r.nextAnim();
      cmd.imageWord = r.nextAnim();
      int handle = cmd.imageWord.resolveInt({});
      cmd.fit = static_cast<Fit>(r.nextInt());
      cmd.sampling = samplingOf(r.nextInt());
      if (r.nextInt() != 0) {
        for (auto& word : cmd.srcRectWords) {
          word = r.nextAnim();
        }
        cmd.animSrcRect = cmd.srcRectWords[0].slot >= 0 ||
                          cmd.srcRectWords[1].slot >= 0 ||
                          cmd.srcRectWords[2].slot >= 0 ||
                          cmd.srcRectWords[3].slot >= 0;
        cmd.srcRect = tgfx::Rect::MakeXYWH(
            cmd.srcRectWords[0].literal, cmd.srcRectWords[1].literal,
            cmd.srcRectWords[2].literal, cmd.srcRectWords[3].literal);
      }
      // `cmd.image` is only a compile-time cache, empty for a buffer, an evicted
      // asset, or a live handle the draw resolves per frame.
      cmd.imageHandle = handle;
      if (cmd.imageWord.slot < 0) {
        bool external = false;
        cmd.image = resolveImage(handle, cmd.sampling, &external);
        if (external) {
          cmd.image = nullptr;
        }
      }
      break;
    }
    case KAtlas: {
      cmd.kind = DrawCmd::Atlas;
      cmd.imageWord = r.nextAnim();
      int handle = cmd.imageWord.resolveInt({});
      cmd.sampling = samplingOf(r.nextInt());
      cmd.atlas = std::make_shared<AtlasCmd>();
      int bufferId = r.nextInt();
      if (bufferId > 0) {
        // Shared-buffer sprites: seven floats each, read straight from the
        // registry block at draw. Always per-frame by design.
        cmd.atlas->bufferId = bufferId;
        cmd.atlas->count = r.nextAnim();
        r.dynamic++;
      } else {
        int count = r.nextInt();
        // Seven words per sprite: the buffer bounds the loop, so bound the reserve to match.
        size_t bounded = std::min(static_cast<size_t>(std::max(0, count)), r.remaining() / 7);
        auto& atlas = *cmd.atlas;
        atlas.words.reserve(bounded * 7);
        bool animated = false;
        for (size_t i = 0; i < bounded * 7 && r.ok; i++) {
          AnimF word = r.nextAnim();
          animated = animated || word.slot >= 0;
          atlas.words.push_back(word);
        }
        atlas.animated = animated;
        if (!animated) {
          // Every field literal: bake the rects and matrices once, as before.
          atlas.sprites.reserve(bounded);
          atlas.matrices.reserve(bounded);
          for (size_t i = 0; i + 6 < atlas.words.size(); i += 7) {
            float sw = atlas.words[i + 2].literal;
            float sh = atlas.words[i + 3].literal;
            atlas.sprites.push_back(tgfx::Rect::MakeXYWH(
                atlas.words[i].literal, atlas.words[i + 1].literal, sw, sh));
            float scale = atlas.words[i + 4].literal;
            atlas.matrices.push_back(tgfx::Matrix::MakeAll(
                scale, 0, atlas.words[i + 5].literal - scale * sw / 2, 0, scale,
                atlas.words[i + 6].literal - scale * sh / 2));
          }
          atlas.words.clear();
          atlas.words.shrink_to_fit();
        }
      }
      cmd.imageHandle = handle;
      if (cmd.imageWord.slot < 0) {
        bool external = false;
        cmd.image = resolveImage(handle, cmd.sampling, &external);
        if (external) {
          cmd.image = nullptr;
        }
        if (cmd.image == nullptr && handle != 0) {
          // A stream, or a decode still to arrive: either way it is not the same next frame.
          r.dynamic++;
        }
      }
      break;
    }
    case KVertices: {
      cmd.kind = DrawCmd::Vertices;
      int bufferId = r.nextInt();
      if (bufferId > 0) {
        auto mesh = std::make_shared<MeshBufferCmd>();
        mesh->bufferId = bufferId;
        mesh->strip = r.nextInt() == 1;
        mesh->maxVertices = std::clamp(r.nextInt(), 0, 65536);
        mesh->maxIndices = std::clamp(r.nextInt(), 0, 1 << 20);
        int flags = r.nextInt();
        mesh->hasTextures = (flags & 1) != 0;
        mesh->hasColors = (flags & 2) != 0;
        mesh->vertexCount = r.nextAnim();
        mesh->indexCount = r.nextAnim();
        cmd.meshBuffer = std::move(mesh);
        r.dynamic++;
      } else {
        readMesh(r, cmd);
      }
      break;
    }
    case KBackdrop: {
      cmd.kind = DrawCmd::Backdrop;
      cmd.x = r.nextAnim();
      cmd.y = r.nextAnim();
      cmd.w = r.nextAnim();
      cmd.h = r.nextAnim();
      cmd.cornerRadius = static_cast<float>(r.next());
      cmd.blur = r.nextAnim();
      cmd.refraction = r.nextAnim();
      cmd.depth = r.nextAnim();
      cmd.frost = r.nextAnim();
      cmd.dispersion = r.nextAnim();
      cmd.splay = r.nextAnim();
      cmd.lightAngle = r.nextAnim();
      cmd.lightIntensity = r.nextAnim();
      // It samples what is already on the surface, so it can never be recorded.
      r.dynamic++;
      break;
    }
    case KText: {
      cmd.kind = DrawCmd::Text;
      cmd.x = r.nextAnim();
      cmd.y = r.nextAnim();
      cmd.text = std::make_shared<TextCmd>();
      cmd.fontSize = r.nextAnim();
      readTextSpec(r, cmd.text->spec);
      readTextRuns(r, *cmd.text);
      auto& spec = cmd.text->spec;
      spec.wrapWidth = static_cast<float>(r.next());
      spec.align = static_cast<TextAlign>(std::clamp(r.nextInt(), 0, 2));
      spec.lineHeight = static_cast<float>(r.next());
      spec.maxLines = r.nextInt();
      spec.ellipsis = r.nextString();
      break;
    }
    case KTextPath: {
      // Same command as Text; the path is the baseline, so there is no x/y.
      cmd.kind = DrawCmd::Text;
      cmd.text = std::make_shared<TextCmd>();
      cmd.text->onPath = true;
      cmd.text->path = parseSvgPath(r.nextString());
      cmd.fontSize = r.nextAnim();
      readTextSpec(r, cmd.text->spec);
      readTextRuns(r, *cmd.text);
      cmd.text->spec.startOffset = static_cast<float>(r.next());
      break;
    }
    case KGlyphs: {
      cmd.kind = DrawCmd::Text;
      cmd.x = r.nextAnim();
      cmd.y = r.nextAnim();
      cmd.text = std::make_shared<TextCmd>();
      cmd.fontSize = r.nextAnim();
      readTextSpec(r, cmd.text->spec);
      int count = r.nextInt();
      if (count < 0 || count > kMaxGlyphs) {
        TGFXRN_LOGW("glyph node has %d glyphs, which is outside 0..%d", count, kMaxGlyphs);
        r.ok = false;
        break;
      }
      cmd.text->glyphs.resize(static_cast<size_t>(count));
      bool warnedId = false;
      for (auto& glyph : cmd.text->glyphs) {
        int id = r.nextInt();
        if (id < 0 || id > 0xFFFF) {
          if (!warnedId) {
            warnedId = true;
            TGFXRN_LOGW("glyph id %d is outside 0..65535; drawing .notdef", id);
          }
          id = 0;
        }
        glyph.id = static_cast<tgfx::GlyphID>(id);
        auto x = static_cast<float>(r.next());
        auto y = static_cast<float>(r.next());
        // A non-finite position poisons the whole blob's bounds.
        glyph.x = std::isfinite(x) ? x : 0;
        glyph.y = std::isfinite(y) ? y : 0;
      }
      break;
    }
    default:
      TGFXRN_LOGW("unknown node kind %d in draw program", kind);
      r.ok = false;
      return false;
  }
  readPaint(r, cmd.paint);
  if (cmd.paint.animated()) {
    r.dynamic++;
  }
  return r.ok;
}

namespace {

/**
 * Whether the program resolves anything from the image or font registry.
 *
 * Both epochs are process-wide, so without this every canvas in the app
 * redraws on any decode, buffer swap or face load — a video frame would cost
 * one full redraw per mounted canvas. Deliberately over-inclusive: a false
 * positive only costs the redraw that used to happen anyway, a false negative
 * drops a frame that should have been drawn.
 */
bool childShadersReadImages(const std::vector<ChildShaderCmd>& children) {
  for (const auto& c : children) {
    if (c.image != nullptr ||
        (c.runtime != nullptr && childShadersReadImages(c.runtime->children)) ||
        (c.blend != nullptr && childShadersReadImages(c.blend->children))) {
      return true;
    }
  }
  return false;
}

void scanDependencies(const Program& program, bool& images, bool& text,
                      bool& backdrop) {
  for (const auto& cmd : program) {
    if (images && text && backdrop) {
      return;
    }
    if (cmd.kind == DrawCmd::Backdrop) {
      backdrop = true;
    }
    if (cmd.kind == DrawCmd::Image || cmd.kind == DrawCmd::CameraFrame ||
        cmd.kind == DrawCmd::Atlas || cmd.imageHandle != 0 ||
        cmd.imageWord.slot >= 0 || cmd.paint.imageShader != nullptr ||
        (cmd.paint.runtime != nullptr &&
         childShadersReadImages(cmd.paint.runtime->children)) ||
        (cmd.paint.blendShader != nullptr &&
         childShadersReadImages(cmd.paint.blendShader->children))) {
      images = true;
    }
    if (cmd.text != nullptr) {
      text = true;
    }
    scanDependencies(cmd.children, images, text, backdrop);
    scanDependencies(cmd.maskChildren, images, text, backdrop);
  }
}

}  // namespace

void SceneRenderer::setScene(std::span<const double> words,
                             const std::vector<std::string>& strings) {
  if (words.size() < 4 || words[0] != kSceneVersion) {
    TGFXRN_LOGW("draw program has an unexpected header");
    return;
  }
  // A NaN, negative or out-of-range JS double is not a size_t; bound by the buffer.
  auto index = [limit = static_cast<double>(words.size())](double v,
                                                           size_t& out) {
    if (!std::isfinite(v) || v < 0 || v > limit) {
      return false;
    }
    out = static_cast<size_t>(v);
    return true;
  };
  size_t total = 0;
  size_t patchStart = 0;
  if (!index(words[1], total) || !index(words[2], patchStart)) {
    TGFXRN_LOGW("draw program has an unexpected header");
    return;
  }
  if (total > words.size() || patchStart + 1 > total) {
    TGFXRN_LOGW("draw program is truncated");
    return;
  }
  size_t patchCount = 0;
  if (!index(words[patchStart], patchCount) ||
      patchStart + 1 + patchCount * 2 > total) {
    TGFXRN_LOGW("draw program patch table is truncated");
    return;
  }
  size_t stringPatchStart = patchStart + 1 + patchCount * 2;
  size_t stringPatchCount = 0;
  if (stringPatchStart >= total ||
      !index(words[stringPatchStart], stringPatchCount) ||
      stringPatchStart + 1 + stringPatchCount * 2 > total) {
    TGFXRN_LOGW("draw program string-patch table is truncated");
    return;
  }
  SceneReader reader(words.first(patchStart),
                     words.subspan(patchStart + 1, patchCount * 2),
                     words.subspan(stringPatchStart + 1, stringPatchCount * 2),
                     strings);

  auto program = std::make_shared<Program>();
  int rootCount = reader.nextInt();
  program->reserve(boundedCount(reader, rootCount));
  for (int i = 0; i < rootCount && reader.ok; i++) {
    DrawCmd cmd;
    if (!readNode(reader, cmd, 0)) {
      break;
    }
    program->push_back(std::move(cmd));
  }
  if (!reader.ok) {
    // A half-read program would draw a truncated scene; keep the previous one.
    TGFXRN_LOGW("draw program could not be read");
    return;
  }
  if (reader.at != reader.words.size() ||
      reader.patchAt != reader.patches.size() ||
      reader.stringPatchAt != reader.stringPatches.size()) {
    TGFXRN_LOGW(
        "draw program not fully consumed: %zu/%zu words, %zu/%zu patch words, "
        "%zu/%zu string-patch words",
        reader.at, reader.words.size(), reader.patchAt, reader.patches.size(),
        reader.stringPatchAt, reader.stringPatches.size());
  }

  // Top-level siblings batch the same way a group's children do.
  clusterBatchableChildren(*program);

  bool images = false;
  bool text = false;
  bool backdrop = false;
  scanDependencies(*program, images, text, backdrop);

  std::lock_guard<std::mutex> lock(programMutex_);
  program_ = std::move(program);
  usesImages_ = images;
  usesText_ = text;
  usesBackdrop_ = backdrop;
}

namespace {

/**
 * The path a hit test should judge a `<Path>` by. A string-slotted path is
 * re-parsed here from the pushed value — `livePath` belongs to the render
 * thread, and hit tests run on whichever thread asked.
 */
const tgfx::Path* hitPathOf(const DrawCmd& cmd,
                            std::span<const std::string> strings,
                            std::span<const uint8_t> pushed,
                            tgfx::Path& scratch) {
  if (cmd.pathSlot < 0 || static_cast<size_t>(cmd.pathSlot) >= strings.size() ||
      static_cast<size_t>(cmd.pathSlot) >= pushed.size() ||
      pushed[static_cast<size_t>(cmd.pathSlot)] == 0) {
    return &cmd.path;
  }
  scratch = parseSvgPath(strings[static_cast<size_t>(cmd.pathSlot)]);
  if ((cmd.pathFlags & 8) != 0) {
    scratch.setFillType(tgfx::PathFillType::EvenOdd);
  }
  return &scratch;
}

/// The geometry a node covers, in its own space. Empty means it draws nothing to hit.
tgfx::Rect hitBounds(const DrawCmd& cmd, std::span<const double> v) {
  switch (cmd.kind) {
    case DrawCmd::Rect:
    case DrawCmd::RRect:
    case DrawCmd::Oval:
    case DrawCmd::Image:
      return rectOf(cmd, v);
    case DrawCmd::Circle: {
      float r = cmd.r.resolve(v);
      return tgfx::Rect::MakeXYWH(cmd.cx.resolve(v) - r, cmd.cy.resolve(v) - r, r * 2, r * 2);
    }
    case DrawCmd::Path:
      return cmd.path.getBounds();
    case DrawCmd::Vertices:
      return cmd.mesh != nullptr ? cmd.mesh->bounds() : tgfx::Rect::MakeEmpty();
    default:
      // Text, fills, groups and the rest have no geometry of their own to test.
      return tgfx::Rect::MakeEmpty();
  }
}

/// Does the node cover `local`? `precise` asks the fill region rather than the box.
bool hitNode(const DrawCmd& cmd, const tgfx::Point& local, bool precise,
             std::span<const double> v,
             std::span<const std::string> strings,
             std::span<const uint8_t> pushed) {
  if (cmd.kind == DrawCmd::Path) {
    tgfx::Path scratch;
    const tgfx::Path* path = hitPathOf(cmd, strings, pushed, scratch);
    // The cached bounds first: they are thread-safe and skip `contains` for most nodes.
    if (!path->getBounds().contains(local.x, local.y)) {
      return false;
    }
    return !precise || path->contains(local.x, local.y);
  }
  if (precise && cmd.kind == DrawCmd::Circle) {
    float dx = local.x - cmd.cx.resolve(v);
    float dy = local.y - cmd.cy.resolve(v);
    float r = cmd.r.resolve(v);
    return dx * dx + dy * dy <= r * r;
  }
  if (precise && cmd.kind == DrawCmd::Oval) {
    auto box = rectOf(cmd, v);
    float rx = box.width() / 2;
    float ry = box.height() / 2;
    if (rx <= 0 || ry <= 0) {
      return false;
    }
    float dx = (local.x - (box.left + rx)) / rx;
    float dy = (local.y - (box.top + ry)) / ry;
    return dx * dx + dy * dy <= 1;
  }
  auto bounds = hitBounds(cmd, v);
  return !bounds.isEmpty() && bounds.contains(local.x, local.y);
}

void hitContext(const DrawCmd& root, const tgfx::Matrix& parent, const tgfx::Point& point,
                bool precise, std::span<const double> v, std::span<const std::string> strings,
                std::span<const uint8_t> pushed, std::vector<int32_t>& path,
                std::vector<double>& out);

void reportHit(const std::vector<int32_t>& path, std::vector<double>& out) {
  out.push_back(static_cast<double>(path.size()));
  for (int32_t step : path) {
    out.push_back(static_cast<double>(step));
  }
}

/// Back to front, so the first hit is the one drawn last; `planesOnly` skips a 3D context's groups.
void hitWalk(const std::vector<DrawCmd>& nodes, const tgfx::Matrix& parent,
             const tgfx::Point& point, bool precise, std::span<const double> v,
             std::span<const std::string> strings,
             std::span<const uint8_t> pushed, std::vector<int32_t>& path,
             std::vector<double>& out, bool planesOnly = false) {
  for (size_t i = nodes.size(); i-- > 0;) {
    const auto& cmd = nodes[i];
    if (planesOnly && cmd.kind == DrawCmd::Group) {
      continue;
    }
    if (cmd.kind == DrawCmd::CameraFrame) {
      // Its matrix needs the surface, which a hit test has not got: no answer beats a wrong one.
      continue;
    }
    // Hidden nodes are not drawn, so they are not hit either — subtree included.
    if ((cmd.visible.slot >= 0 || cmd.visible.literal < 0.5f) &&
        cmd.visible.resolve(v) < 0.5f) {
      continue;
    }
    if (opensContext(cmd)) {
      path.push_back(static_cast<int32_t>(i));
      hitContext(cmd, parent, point, precise, v, strings, pushed, path, out);
      path.pop_back();
      continue;
    }
    auto matrix = parent;
    if (cmd.hasMatrix || !cmd.ops.empty()) {
      matrix.preConcat(groupMatrix(cmd, v));
    }
    tgfx::Matrix inverted = tgfx::Matrix::I();
    if (!matrix.invert(&inverted)) {
      // A degenerate transform collapses the subtree to nothing, so nothing is under the point.
      continue;
    }
    auto local = inverted.mapXY(point.x, point.y);
    if (cmd.clipKind != 0 && !cmd.invertClip) {
      // Prunes the subtree too, which is most of what makes a deep scene affordable.
      auto clip = cmd.clipKind == 3
                      ? (cmd.clipPath != nullptr ? cmd.clipPath->getBounds() : tgfx::Rect::MakeEmpty())
                      : cmd.clipRect;
      if (!clip.contains(local.x, local.y)) {
        continue;
      }
    }
    path.push_back(static_cast<int32_t>(i));
    if (!cmd.children.empty()) {
      hitWalk(cmd.children, matrix, point, precise, v, strings, pushed, path, out);
    }
    if (hitNode(cmd, local, precise, v, strings, pushed)) {
      reportHit(path, out);
    }
    path.pop_back();
  }
}

struct HitPlane {
  const DrawCmd* cmd;
  tgfx::Matrix3D placed;
  std::vector<int32_t> path;
  tgfx::Point local = {};
  float depth = 0;
};

void collectHitPlanes(const DrawCmd& cmd, const tgfx::Matrix3D& parent,
                      std::span<const double> v, std::vector<int32_t>& path,
                      std::vector<HitPlane>& out) {
  auto placed = parent * matrix3DFromOps(cmd.ops, v);
  out.push_back({&cmd, placed, path});
  if (!opensContext(cmd)) {
    return;
  }
  for (size_t i = 0; i < cmd.children.size(); i++) {
    const auto& child = cmd.children[i];
    if (child.kind != DrawCmd::Group || hidden(child, v)) {
      continue;
    }
    path.push_back(static_cast<int32_t>(i));
    collectHitPlanes(child, placed, v, path, out);
    path.pop_back();
  }
}

/// Composes a 3D context's planes in 4x4 as the draw does, nearest plane under the point first.
void hitContext(const DrawCmd& root, const tgfx::Matrix& parent, const tgfx::Point& point,
                bool precise, std::span<const double> v, std::span<const std::string> strings,
                std::span<const uint8_t> pushed, std::vector<int32_t>& path,
                std::vector<double>& out) {
  std::vector<HitPlane> planes;
  std::vector<int32_t> relative;
  collectHitPlanes(root, tgfx::Matrix3D(parent), v, relative, planes);
  constexpr float kBehind = -std::numeric_limits<float>::infinity();
  for (auto& plane : planes) {
    tgfx::Matrix inverted = tgfx::Matrix::I();
    if (!plane.placed.asMatrix().invert(&inverted)) {
      plane.depth = kBehind;
      continue;
    }
    plane.local = inverted.mapXY(point.x, point.y);
    auto clip = plane.placed.mapHomogeneous(plane.local.x, plane.local.y, 0, 1);
    plane.depth = clip.w > 0 ? clip.z / clip.w : kBehind;
  }
  // Reversed first, so planes at one depth keep the drawn-last-on-top order.
  std::reverse(planes.begin(), planes.end());
  std::stable_sort(planes.begin(), planes.end(),
                   [](const HitPlane& a, const HitPlane& b) { return a.depth > b.depth; });
  for (const auto& plane : planes) {
    const auto& cmd = *plane.cmd;
    if (plane.depth == kBehind) {
      continue;
    }
    if (cmd.clipKind != 0 && !cmd.invertClip) {
      auto clip = cmd.clipKind == 3 ? (cmd.clipPath != nullptr ? cmd.clipPath->getBounds()
                                                               : tgfx::Rect::MakeEmpty())
                                    : cmd.clipRect;
      if (!clip.contains(plane.local.x, plane.local.y)) {
        continue;
      }
    }
    path.insert(path.end(), plane.path.begin(), plane.path.end());
    hitWalk(cmd.children, plane.placed.asMatrix(), point, precise, v, strings, pushed, path, out,
            opensContext(cmd));
    if (hitNode(cmd, plane.local, precise, v, strings, pushed)) {
      reportHit(path, out);
    }
    path.resize(path.size() - plane.path.size());
  }
}

}  // namespace

size_t SceneRenderer::rootCount() const {
  std::lock_guard<std::mutex> lock(programMutex_);
  return program_ == nullptr ? 0 : program_->size();
}

bool SceneRenderer::usesImages() const {
  std::lock_guard<std::mutex> lock(programMutex_);
  return usesImages_;
}

bool SceneRenderer::usesText() const {
  std::lock_guard<std::mutex> lock(programMutex_);
  return usesText_;
}

bool SceneRenderer::usesBackdrop() const {
  std::lock_guard<std::mutex> lock(programMutex_);
  return usesBackdrop_;
}

std::vector<double> SceneRenderer::hitTest(float x, float y, bool precise,
                                           std::span<const double> values,
                                           std::span<const std::string> stringValues,
                                           std::span<const uint8_t> stringPushed,
                                           const tgfx::Matrix& root) {
  std::shared_ptr<const Program> program;
  {
    std::lock_guard<std::mutex> lock(programMutex_);
    program = program_;
  }
  std::vector<double> out;
  if (program == nullptr) {
    return out;
  }
  std::vector<int32_t> path;
  hitWalk(*program, root, {x, y}, precise, values, stringValues,
          stringPushed, path, out);
  return out;
}

/// A ceiling on the offscreen a grained group can ask for, so a group hanging
/// far off the surface cannot turn into an unbounded render target.
constexpr int64_t kMaxGrainPixels = 4096 * 4096;

/// The grain the group's pixels are dusted with. Rebuilt per frame, as tgfx's
/// own layer does: a NoiseStyle is a shader, not a render target.
static std::shared_ptr<tgfx::NoiseStyle> buildNoiseStyle(const GrainCmd& g,
                                                         std::span<const double> v) {
  float size = g.size.resolve(v);
  if (!(size > 0)) {
    return nullptr;
  }
  float density = g.density.resolve(v);
  float seed = g.seed.resolve(v);
  if (g.mode == 2) {
    return tgfx::NoiseStyle::MakeMulti(size, density, g.opacity.resolve(v), seed);
  }
  auto first = g.animColor ? tgfx::Color::FromRGBA(channelOf(g.rgba[0], v), channelOf(g.rgba[1], v),
                                                   channelOf(g.rgba[2], v), channelOf(g.rgba[3], v))
                           : colorFromPacked(g.color);
  if (g.mode == 0) {
    return tgfx::NoiseStyle::MakeMono(size, density, first, seed);
  }
  auto second = g.animSecondColor
                    ? tgfx::Color::FromRGBA(channelOf(g.secondRgba[0], v),
                                            channelOf(g.secondRgba[1], v),
                                            channelOf(g.secondRgba[2], v),
                                            channelOf(g.secondRgba[3], v))
                    : colorFromPacked(g.secondColor);
  return tgfx::NoiseStyle::MakeDuo(size, density, first, second, seed);
}

/// Dusts an already-drawn group. The picture is what the group drew, so the
/// style clips the grain to that alpha rather than to any bounding box. The
/// noise covers the whole content rect, so this is a second full replay of the
/// subtree into an offscreen every frame, on top of the direct draw — which is
/// why the docs say to keep a grained subtree to what needs the grain.
static void drawGrain(tgfx::Canvas* canvas, const GrainCmd& g,
                      const std::shared_ptr<tgfx::Picture>& picture,
                      std::span<const double> v) {
  auto* surface = canvas->getSurface();
  if (picture == nullptr || surface == nullptr) {
    // No surface means a recording or an exporter canvas, where the matrix is
    // the identity and the grain would be baked at the wrong scale.
    return;
  }
  auto local = picture->getBounds();
  if (local.isEmpty()) {
    return;
  }
  // The style works in surface pixels, so the content image and everything
  // handed to it are in pixels too — the same convention the backdrop uses.
  const auto& matrix = canvas->getMatrix();
  float scale = matrix.getMaxScale();
  if (!(scale > 0)) {
    scale = 1;
  }
  auto device = matrix.mapRect(local);
  device.roundOut();
  // Sized from everything the group drew, not from what survives the surface or
  // the clip: the style anchors its pattern to half the content image's size,
  // so cropping would slide the grain half a pixel per pixel of crop and make
  // it crawl as the group scrolls off an edge. The canvas clip drops what falls
  // outside anyway, and the ceiling below bounds the offscreen.
  int width = static_cast<int>(device.width());
  int height = static_cast<int>(device.height());
  if (width <= 0 || height <= 0 ||
      static_cast<int64_t>(width) * height > kMaxGrainPixels) {
    return;
  }
  auto style = buildNoiseStyle(g, v);
  if (style == nullptr) {
    return;
  }
  auto contentMatrix = matrix;
  contentMatrix.postTranslate(-device.left, -device.top);
  auto content = tgfx::Image::MakeFrom(picture, width, height, &contentMatrix);
  if (content == nullptr) {
    return;
  }
  style->setBlendMode(blendModeOf(g.blendMode.resolveInt(v)));
  tgfx::LayerStyleInput input;
  input.content = std::move(content);
  // In pixels, so the pattern stays put while the group moves under it.
  input.contentOffset = {device.left, device.top};
  input.contentScale = scale;
  canvas->save();
  canvas->setMatrix(tgfx::Matrix::MakeTrans(device.left, device.top));
  style->draw(canvas, input, 1.0f);
  canvas->restore();
}

// Canvas::drawPath sends line/rect/oval/rrect strokes to dedicated draws (and the SVG exporter to
// <rect>/<line>), so only a generic path is worth caching as a pre-stroked Shape. Probed once.
static bool cacheStrokedShape(const DrawCmd& cmd) {
  if (cmd.strokeCache == 0) {
    const auto& path = cmd.path;
    cmd.strokeCache =
        path.isLine() || path.isRect() || path.isOval() || path.isRRect(nullptr) ? -1 : 1;
  }
  return cmd.strokeCache > 0;
}

void SceneRenderer::drawGroupBody(tgfx::Canvas* canvas, const DrawCmd& cmd,
                                  std::span<const double> v,
                                  std::span<const std::string> strings, uint64_t stringGen,
                                  std::span<const uint8_t> stringPushed,
                                  tgfx::Paint& paint) const {
  bool saved = cmd.clipKind != 0 || cmd.layer;
  if (saved) {
    canvas->save();
  }
  switch (cmd.clipKind) {
    case 1: canvas->clipRect(cmd.clipRect); break;
    case 2:
      canvas->clipRRect(
          tgfx::RRect::MakeRectXY(cmd.clipRect, cmd.clipRadiusX, cmd.clipRadiusY));
      break;
    case 3:
      if (cmd.clipPath != nullptr) {
        canvas->clipPath(*cmd.clipPath);
      }
      break;
    default: break;
  }
  if (cmd.layer) {
    canvas->saveLayerAlpha(cmd.layerAlpha.resolve(v));
  }
  if (cmd.cachePicture || cmd.grain != nullptr) {
    // Recorded under the identity, so the transform above stays live and the
    // replay is not baked to whatever matrix the first frame happened to have.
    // Grain needs the recording anyway, so it costs one walk, not two.
    auto picture = cmd.picture;
    if (picture == nullptr) {
      tgfx::PictureRecorder recorder;
      auto* into = recorder.beginRecording();
      for (const auto& child : cmd.children) {
        drawCmd(into, child, v, strings, stringGen, stringPushed, paint);
      }
      picture = recorder.finishRecordingAsPicture();
      if (cmd.cachePicture) {
        cmd.picture = picture;
      }
    }
    if (picture != nullptr) {
      canvas->drawPicture(picture);
      if (cmd.grain != nullptr) {
        drawGrain(canvas, *cmd.grain, picture, v);
      }
    }
  } else {
    for (const auto& child : cmd.children) {
      drawCmd(canvas, child, v, strings, stringGen, stringPushed, paint);
    }
  }
  if (cmd.layer) {
    canvas->restore();
  }
  if (saved) {
    canvas->restore();
  }
}

void SceneRenderer::drawCmd(tgfx::Canvas* canvas, const DrawCmd& cmd,
                            std::span<const double> v,
                            std::span<const std::string> strings,
                            uint64_t stringGen,
                            std::span<const uint8_t> stringPushed,
                            tgfx::Paint& paint) const {
  // The pool gate: a hidden node costs one resolve and nothing else.
  if (cmd.visible.slot >= 0 || cmd.visible.literal < 0.5f) {
    if (cmd.visible.resolve(v) < 0.5f) {
      return;
    }
  }
  if (cmd.kind == DrawCmd::Group) {
    auto* surface = canvas->getSurface();
    if (surface != nullptr && opensContext(cmd)) {
      Layer3DArgs args{canvas->getMatrix().getMaxScale(), surface->colorSpace(),
                       tgfx::Rect::MakeEmpty(), v, strings, stringGen, stringPushed, &paint};
      auto device = canvas->getTotalClipBounds().value_or(
          tgfx::Rect::MakeWH(surface->width(), surface->height()));
      tgfx::Matrix toLocal = tgfx::Matrix::I();
      args.canvasRect = canvas->getMatrix().invert(&toLocal) ? toLocal.mapRect(device) : device;
      // Into the root's own plane, where an unturned plane's fill covers the canvas.
      if (groupMatrix(cmd, v).invert(&toLocal)) {
        auto local = toLocal.mapRect(args.canvasRect);
        if (std::isfinite(local.left) && std::isfinite(local.top) &&
            std::isfinite(local.right) && std::isfinite(local.bottom) && !local.isEmpty()) {
          args.canvasRect = local;
        }
      }
      // Drawn directly, a preserve3D layer renders nothing.
      auto root = tgfx::Layer::Make();
      root->addChild(buildLayer3D(cmd, tgfx::Matrix3D::I(), args));
      root->draw(canvas);
      return;
    }
    bool transformed = cmd.hasMatrix || !cmd.ops.empty();
    if (transformed) {
      canvas->save();
      canvas->concat(groupMatrix(cmd, v));
    }
    drawGroupBody(canvas, cmd, v, strings, stringGen, stringPushed, paint);
    if (transformed) {
      canvas->restore();
    }
    return;
  }

  if (cmd.kind == DrawCmd::Mask) {
    // Either side empty means nothing survives the intersection, and a one-record
    // layer would be unrolled rather than composited, inverting that.
    if (cmd.children.empty() || cmd.maskChildren.empty()) {
      return;
    }
    // Content into a layer, then the mask over it with DstIn: what survives is the overlap.
    canvas->saveLayer(nullptr);
    for (const auto& child : cmd.children) {
      drawCmd(canvas, child, v, strings, stringGen, stringPushed, paint);
    }
    tgfx::Paint maskPaint;
    maskPaint.setBlendMode(tgfx::BlendMode::DstIn);
    if (cmd.maskLuminance) {
      // Built once: Luma is stateless, and this runs per masked node per frame.
      static const auto luma = tgfx::ColorFilter::Luma();
      maskPaint.setColorFilter(luma);
    }
    canvas->saveLayer(&maskPaint);
    // Clear, not a transparent colour: a zero-alpha fill is dropped and widens nothing.
    tgfx::Paint clear;
    clear.setColor(tgfx::Color::Black());
    clear.setBlendMode(tgfx::BlendMode::Clear);
    // As a rect, not drawPaint: under a clip that degrades to an inverse-fill path.
    if (auto bounds = canvas->getTotalClipBounds()) {
      canvas->drawRect(*bounds, clear);
    } else {
      canvas->drawPaint(clear);
    }
    for (const auto& child : cmd.maskChildren) {
      drawCmd(canvas, child, v, strings, stringGen, stringPushed, paint);
    }
    canvas->restore();
    canvas->restore();
    return;
  }

  if (cmd.kind == DrawCmd::CameraFrame) {
    // Never cached: the producer swaps the buffer behind the handle between frames.
    auto image = resolveImage(cmd.imageHandle, cmd.sampling);
    if (image == nullptr || image->width() <= 0 || image->height() <= 0) {
      return;
    }
    canvas->save();
    canvas->concat(cameraMatrix(cmd, canvas, static_cast<float>(image->width()),
                                static_cast<float>(image->height())));
    const tgfx::Paint* framePaint = &cmd.paint.cached;
    if (cmd.paint.animated()) {
      applyPaint(cmd.paint, v, paint);
      framePaint = &paint;
    }
    canvas->drawImage(std::move(image), cmd.sampling, framePaint);
    for (const auto& child : cmd.children) {
      drawCmd(canvas, child, v, strings, stringGen, stringPushed, paint);
    }
    canvas->restore();
    return;
  }

  // A static paint is handed over untouched; an animated one is rebuilt here.
  const tgfx::Paint* p = &cmd.paint.cached;
  if (cmd.paint.animated()) {
    applyPaint(cmd.paint, v, paint);
    p = &paint;
  }
  // Replaces the layer tgfx would take for the image filter — the same one, not an extra.
  bool tookLayer = cmd.paint.layer != nullptr;
  if (tookLayer) {
    canvas->saveLayer(cmd.paint.layer.get());
  } else if (p == &paint && paint.getImageFilter() != nullptr &&
             paint.getBlendMode() != tgfx::BlendMode::SrcOver) {
    // The animated form of the same fix: blend + image filter cannot share a paint.
    tgfx::Paint layerPaint;
    layerPaint.setBlendMode(paint.getBlendMode());
    layerPaint.setImageFilter(paint.getImageFilter());
    paint.setBlendMode(tgfx::BlendMode::SrcOver);
    paint.setImageFilter(nullptr);
    canvas->saveLayer(&layerPaint);
    tookLayer = true;
  }

  switch (cmd.kind) {
    case DrawCmd::Fill:
      canvas->drawPaint(*p);
      break;
    case DrawCmd::Rect:
      canvas->drawRect(rectOf(cmd, v), *p);
      break;
    case DrawCmd::RRect:
      canvas->drawRRect(
          tgfx::RRect::MakeRectXY(rectOf(cmd, v), cmd.rx.resolve(v),
                                  cmd.ry.resolve(v)),
          *p);
      break;
    case DrawCmd::Oval:
      canvas->drawOval(rectOf(cmd, v), *p);
      break;
    case DrawCmd::Circle:
      canvas->drawCircle(cmd.cx.resolve(v), cmd.cy.resolve(v), cmd.r.resolve(v),
                         *p);
      break;
    case DrawCmd::Line:
      if (cmd.lineAnimated) {
        tgfx::Path line;
        line.moveTo(cmd.x1.resolve(v), cmd.y1.resolve(v));
        line.lineTo(cmd.x2.resolve(v), cmd.y2.resolve(v));
        canvas->drawPath(line, *p);
      } else {
        canvas->drawPath(cmd.path, *p);
      }
      break;
    case DrawCmd::Path: {
      const tgfx::Path* path = &cmd.path;
      if (cmd.pathSlot >= 0 &&
          static_cast<size_t>(cmd.pathSlot) < strings.size() &&
          static_cast<size_t>(cmd.pathSlot) < stringPushed.size() &&
          stringPushed[static_cast<size_t>(cmd.pathSlot)] != 0) {
        // Re-parse only when a push changed this slot; render thread only.
        if (stringGen != cmd.pathGen) {
          cmd.pathGen = stringGen;
          tgfx::Path parsed =
              parseSvgPath(strings[static_cast<size_t>(cmd.pathSlot)]);
          if ((cmd.pathFlags & 8) != 0) {
            parsed.setFillType(tgfx::PathFillType::EvenOdd);
          }
          cmd.livePath = std::move(parsed);
        }
        path = &cmd.livePath;
      }
      if (cmd.effects != nullptr) {
        tgfx::Path trimmed = *path;
        cmd.effects->apply(&trimmed, v);
        canvas->drawPath(trimmed, *p);
      } else if (path == &cmd.path && p == &cmd.paint.cached && p->getStroke() != nullptr &&
                 cacheStrokedShape(cmd)) {
        if (cmd.strokedShape == nullptr) {
          cmd.strokedShape = tgfx::Shape::ApplyStroke(tgfx::Shape::MakeFrom(*path), p->getStroke());
        }
        tgfx::Paint fill = *p;
        fill.setStyle(tgfx::PaintStyle::Fill);
        canvas->drawShape(cmd.strokedShape, fill);
      } else {
        canvas->drawPath(*path, *p);
      }
      break;
    }
    case DrawCmd::Image: {
      std::shared_ptr<tgfx::Image> live;
      int32_t handle = cmd.imageWord.slot >= 0 ? cmd.imageWord.resolveInt(v)
                                               : cmd.imageHandle;
      if (cmd.imageWord.slot >= 0) {
        // One compare per frame while the handle is stable; a swap re-resolves.
        if (handle != cmd.liveHandle || cmd.liveImage == nullptr) {
          cmd.liveHandle = handle;
          bool external = false;
          cmd.liveImage = resolveImage(handle, cmd.sampling, &external);
          if (external) {
            // A producer-owned buffer must be re-acquired every frame.
            live = std::move(cmd.liveImage);
          }
        }
        if (live == nullptr) {
          live = cmd.liveImage;
        }
      } else if (cmd.image == nullptr && handle != 0) {
        // Same one-compare cache as the slot path: without it an image that
        // finished decoding after parse re-acquired (and re-mipmapped) per frame.
        if (handle != cmd.liveHandle || cmd.liveImage == nullptr) {
          cmd.liveHandle = handle;
          bool external = false;
          cmd.liveImage = resolveImage(handle, cmd.sampling, &external);
          if (external) {
            live = std::move(cmd.liveImage);
          }
        }
        if (live == nullptr) {
          live = cmd.liveImage;
        }
      }
      const auto& image = cmd.imageWord.slot >= 0
                              ? live
                              : (cmd.image != nullptr ? cmd.image : live);
      if (image == nullptr || image->width() <= 0) {
        break;
      }
      auto srcRect = cmd.srcRect;
      if (cmd.animSrcRect) {
        srcRect = tgfx::Rect::MakeXYWH(
            cmd.srcRectWords[0].resolve(v), cmd.srcRectWords[1].resolve(v),
            cmd.srcRectWords[2].resolve(v), cmd.srcRectWords[3].resolve(v));
      }
      // `fit` crops the source, so cover needs no clip and no save/restore.
      auto source = srcRect.isEmpty()
                        ? tgfx::Rect::MakeWH(static_cast<float>(image->width()),
                                             static_cast<float>(image->height()))
                        : srcRect;
      auto fitted = fitRects(cmd.fit, source, rectOf(cmd, v));
      canvas->drawImageRect(image, fitted.src, fitted.dst, cmd.sampling, p);
      break;
    }
    case DrawCmd::Atlas: {
      if (cmd.atlas == nullptr) {
        break;
      }
      std::shared_ptr<tgfx::Image> live;
      int32_t handle = cmd.imageWord.slot >= 0 ? cmd.imageWord.resolveInt(v)
                                               : cmd.imageHandle;
      if (cmd.image == nullptr && handle != 0) {
        live = resolveImage(handle, cmd.sampling);
      }
      const auto& sheet = cmd.image != nullptr ? cmd.image : live;
      if (sheet == nullptr || sheet->width() <= 0) {
        break;
      }
      auto& atlas = *cmd.atlas;
      const tgfx::Matrix* matrices = atlas.matrices.data();
      const tgfx::Rect* rects = atlas.sprites.data();
      size_t count = atlas.sprites.size();
      if (atlas.bufferId > 0) {
        // Shared buffer: seven floats per sprite, resolved into the scratch
        // vectors — sized once, refilled in place, no per-frame allocation.
        auto block = BufferRegistry::instance().acquire(atlas.bufferId);
        if (block == nullptr) {
          break;
        }
        size_t capacity = block->data.size() / 7;
        size_t wanted = static_cast<size_t>(
            std::max(0, atlas.count.resolveInt(v)));
        count = std::min(wanted, capacity);
        atlas.scratchMatrices.resize(count);
        atlas.scratchSprites.resize(count);
        const float* f = block->data.data();
        for (size_t i = 0; i < count; i++) {
          const float* s = f + i * 7;
          atlas.scratchSprites[i] = tgfx::Rect::MakeXYWH(s[0], s[1], s[2], s[3]);
          atlas.scratchMatrices[i] = tgfx::Matrix::MakeAll(
              s[4], 0, s[5] - s[4] * s[2] / 2, 0, s[4], s[6] - s[4] * s[3] / 2);
        }
        matrices = atlas.scratchMatrices.data();
        rects = atlas.scratchSprites.data();
      } else if (atlas.animated) {
        size_t spriteCount = atlas.words.size() / 7;
        atlas.scratchMatrices.resize(spriteCount);
        atlas.scratchSprites.resize(spriteCount);
        for (size_t i = 0; i < spriteCount; i++) {
          const AnimF* s = atlas.words.data() + i * 7;
          float sw = s[2].resolve(v);
          float sh = s[3].resolve(v);
          float scale = s[4].resolve(v);
          atlas.scratchSprites[i] =
              tgfx::Rect::MakeXYWH(s[0].resolve(v), s[1].resolve(v), sw, sh);
          atlas.scratchMatrices[i] = tgfx::Matrix::MakeAll(
              scale, 0, s[5].resolve(v) - scale * sw / 2, 0, scale,
              s[6].resolve(v) - scale * sh / 2);
        }
        matrices = atlas.scratchMatrices.data();
        rects = atlas.scratchSprites.data();
        count = spriteCount;
      }
      if (count == 0) {
        break;
      }
      // No per-sprite colours: tgfx only applies them to an alpha-only atlas.
      canvas->drawAtlas(sheet, matrices, rects, nullptr, count, cmd.sampling, p);
      break;
    }
    case DrawCmd::Backdrop: {
      auto rect = rectOf(cmd, v);
      if (rect.isEmpty()) {
        break;
      }
      auto* surface = canvas->getSurface();
      if (surface == nullptr) {
        // No surface to sample: an offscreen export or a recording has no backdrop.
        break;
      }
      // makeImageSnapshot() flushes the surface itself and orders the copy after the ops beneath.
      auto below = surface->makeImageSnapshot();
      if (below == nullptr) {
        break;
      }
      // The snapshot is in surface pixels while the rect is in scene points, so the panel, the
      // region it reads and everything handed to tgfx below are all in pixels.
      const auto& matrix = canvas->getMatrix();
      float scale = matrix.getMaxScale();
      if (!(scale > 0)) {
        scale = 1;
      }
      auto panel = matrix.mapRect(rect);
      panel.roundOut();
      auto surfaceBounds = tgfx::Rect::MakeWH(surface->width(), surface->height());
      float refraction = cmd.refraction.resolve(v);
      auto glass = refraction > 0
                       ? tgfx::GlassStyle::Make(refraction, cmd.depth.resolve(v),
                                                cmd.frost.resolve(v), cmd.dispersion.resolve(v),
                                                cmd.splay.resolve(v), cmd.lightAngle.resolve(v),
                                                cmd.lightIntensity.resolve(v))
                       : nullptr;
      // Mirrored at the edges, as tgfx's own frost is: a panel flush with the surface edge would
      // otherwise blur against transparency and let the sharp content show through underneath.
      float blurAmount = cmd.blur.resolve(v);
      auto blur = glass == nullptr && blurAmount > 0
                      ? tgfx::ImageFilter::Blur(blurAmount * scale, blurAmount * scale,
                                                tgfx::TileMode::Mirror)
                      : nullptr;
      if (glass == nullptr && blur == nullptr) {
        break;
      }
      // Glass maps its shape across the whole background it is given, so that has to be the panel
      // exactly, the way tgfx's own capturer sizes it. A blur instead reads past the panel.
      auto background = glass != nullptr ? panel : blur->filterBounds(panel);
      background.roundOut();
      if (!background.intersect(surfaceBounds)) {
        break;
      }
      // A panel hanging off the surface leaves the glass a cropped background. Its shape has to be
      // cropped with it, or the mask stretches over the smaller image and the refraction squashes.
      auto shape = panel;
      if (glass != nullptr) {
        panel = background;
      }
      auto source = below->makeSubset(background);
      if (source == nullptr) {
        break;
      }
      canvas->save();
      if (cmd.cornerRadius > 0) {
        canvas->clipRRect(tgfx::RRect::MakeRectXY(rect, cmd.cornerRadius, cmd.cornerRadius));
      } else {
        canvas->clipRect(rect);
      }
      if (glass == nullptr) {
        // The clip is already the panel's shape, so a plain blur is just the blurred backdrop
        // drawn into it. No layer style, and no per-frame mask to go with one.
        tgfx::Point offset = {};
        auto blurred = source->makeWithFilter(blur, &offset);
        if (blurred != nullptr) {
          tgfx::Paint panelPaint;
          panelPaint.setAlpha(p->getAlpha());
          canvas->setMatrix(tgfx::Matrix::I());
          canvas->drawImage(blurred, background.left + offset.x, background.top + offset.y,
                            &panelPaint);
        }
        canvas->restore();
        break;
      }
      // Glass refracts against the panel's own shape, which it reads as the content's alpha.
      tgfx::PictureRecorder recorder;
      auto* maskCanvas = recorder.beginRecording();
      auto local = tgfx::Rect::MakeWH(panel.width(), panel.height());
      // The shape sits where it falls inside the image, so a cropped panel keeps the corners it
      // still has rather than growing new ones at the crop.
      auto shapeLocal = tgfx::Rect::MakeXYWH(shape.left - panel.left, shape.top - panel.top,
                                             shape.width(), shape.height());
      float radius = cmd.cornerRadius * scale;
      tgfx::Paint maskPaint;
      if (radius > 0) {
        maskCanvas->drawRRect(tgfx::RRect::MakeRectXY(shapeLocal, radius, radius), maskPaint);
      } else {
        maskCanvas->drawRect(shapeLocal, maskPaint);
      }
      auto content = tgfx::Image::MakeFrom(recorder.finishRecordingAsPicture(),
                                           static_cast<int>(local.width()),
                                           static_cast<int>(local.height()));
      if (content == nullptr) {
        canvas->restore();
        break;
      }
      tgfx::LayerStyleInput input;
      input.content = content;
      input.contentOffset = {panel.left, panel.top};
      input.contentScale = scale;
      input.extraSource = std::make_shared<tgfx::StyleInputSource>(
          source, tgfx::Point{background.left - panel.left, background.top - panel.top});
      // The style draws in pixels with the panel's top-left at the origin.
      canvas->setMatrix(tgfx::Matrix::MakeTrans(panel.left, panel.top));
      glass->draw(canvas, input, p->getAlpha());
      canvas->restore();
      break;
    }
    case DrawCmd::Vertices: {
      if (cmd.meshBuffer != nullptr) {
        auto& mb = *cmd.meshBuffer;
        auto block = BufferRegistry::instance().acquire(mb.bufferId);
        if (block == nullptr) {
          break;
        }
        // Section offsets are fixed by the declared capacities; counts are live.
        size_t maxV = static_cast<size_t>(mb.maxVertices);
        size_t need = maxV * 2 + (mb.hasTextures ? maxV * 2 : 0) +
                      (mb.hasColors ? maxV * 4 : 0) +
                      static_cast<size_t>(mb.maxIndices);
        if (block->data.size() < need) {
          break;
        }
        size_t vCount = std::min(
            static_cast<size_t>(std::max(0, mb.vertexCount.resolveInt(v))),
            maxV);
        size_t iCount = std::min(
            static_cast<size_t>(std::max(0, mb.indexCount.resolveInt(v))),
            static_cast<size_t>(mb.maxIndices));
        if (vCount == 0) {
          break;
        }
        const float* f = block->data.data();
        const float* tex = f + maxV * 2;
        const float* col = tex + (mb.hasTextures ? maxV * 2 : 0);
        const float* idx = col + (mb.hasColors ? maxV * 4 : 0);
        mb.scratchPositions.resize(vCount);
        for (size_t i = 0; i < vCount; i++) {
          mb.scratchPositions[i] = {f[i * 2], f[i * 2 + 1]};
        }
        if (mb.hasTextures) {
          mb.scratchTexCoords.resize(vCount);
          for (size_t i = 0; i < vCount; i++) {
            mb.scratchTexCoords[i] = {tex[i * 2], tex[i * 2 + 1]};
          }
        }
        if (mb.hasColors) {
          mb.scratchColors.resize(vCount);
          for (size_t i = 0; i < vCount; i++) {
            mb.scratchColors[i] = tgfx::Color{
                std::clamp(col[i * 4], 0.0f, 1.0f),
                std::clamp(col[i * 4 + 1], 0.0f, 1.0f),
                std::clamp(col[i * 4 + 2], 0.0f, 1.0f),
                std::clamp(col[i * 4 + 3], 0.0f, 1.0f)};
          }
        }
        bool badIndex = false;
        mb.scratchIndices.resize(iCount);
        for (size_t i = 0; i < iCount; i++) {
          float raw = idx[i];
          // Validate before the cast: an out-of-range float to int is UB.
          if (!(raw >= 0 && raw < static_cast<float>(vCount))) {
            badIndex = true;
            break;
          }
          mb.scratchIndices[i] = static_cast<uint16_t>(raw);
        }
        if (badIndex) {
          break;
        }
        auto mesh = tgfx::Mesh::MakeCopy(
            mb.strip ? tgfx::MeshTopology::TriangleStrip
                     : tgfx::MeshTopology::Triangles,
            static_cast<int>(vCount), mb.scratchPositions.data(),
            mb.hasTextures ? mb.scratchTexCoords.data() : nullptr,
            mb.hasColors ? mb.scratchColors.data() : nullptr,
            static_cast<int>(iCount),
            iCount > 0 ? mb.scratchIndices.data() : nullptr);
        if (mesh != nullptr) {
          canvas->drawMesh(std::move(mesh), *p);
        }
        break;
      }
      if (cmd.mesh != nullptr) {
        canvas->drawMesh(cmd.mesh, *p);
      }
      break;
    }
    case DrawCmd::Text: {
      if (cmd.text == nullptr) {
        break;
      }
      const auto& layout = cmd.text->resolve(cmd.fontSize.resolve(v), strings,
                                             stringGen, stringPushed);
      float x = cmd.x.resolve(v);
      float y = cmd.y.resolve(v);
      for (const auto& run : layout.runs) {
        if (run.blob == nullptr) {
          continue;
        }
        if (!run.hasColor) {
          canvas->drawTextBlob(run.blob, x, y, *p);
          continue;
        }
        tgfx::Paint spanPaint = *p;
        spanPaint.setColor(tgfx::Color::FromRGBA(
            static_cast<uint8_t>((run.color >> 16) & 0xFF),
            static_cast<uint8_t>((run.color >> 8) & 0xFF), static_cast<uint8_t>(run.color & 0xFF),
            static_cast<uint8_t>((run.color >> 24) & 0xFF)));
        // The node's opacity, not its resolved alpha: the span colour replaces that.
        spanPaint.setAlpha(spanPaint.getAlpha() * cmd.paint.alpha.resolve(v));
        canvas->drawTextBlob(run.blob, x, y, spanPaint);
      }
      break;
    }
    default:
      break;
  }

  if (tookLayer) {
    canvas->restore();
  }
}

constexpr float kMaxLayerSide = 4096;

/// The placement's largest edge magnification of `bounds`, in sqrt(2) steps to limit re-rasters.
float layerMagnification(const tgfx::Matrix3D& placed, const tgfx::Rect& bounds) {
  tgfx::Point quad[4] = {{bounds.left, bounds.top},
                         {bounds.right, bounds.top},
                         {bounds.right, bounds.bottom},
                         {bounds.left, bounds.bottom}};
  for (const auto& p : quad) {
    if (placed.mapHomogeneous(p.x, p.y, 0, 1).w <= 0) {
      return 1;
    }
  }
  tgfx::Point mapped[4];
  placed.asMatrix().mapPoints(mapped, quad, 4);
  float best = 0;
  for (int i = 0; i < 4; i++) {
    float local = i % 2 == 0 ? bounds.width() : bounds.height();
    if (local > 0) {
      best = std::max(best, tgfx::Point::Distance(mapped[i], mapped[(i + 1) % 4]) / local);
    }
  }
  if (!(best > 0) || !std::isfinite(best)) {
    return 1;
  }
  return std::exp2(std::ceil(std::log2(std::clamp(best, 0.25f, 8.0f)) * 2) / 2);
}

/// A child group of a 3D context is a layer of its own; any other group is one flat plane.
std::shared_ptr<tgfx::Layer> SceneRenderer::buildLayer3D(const DrawCmd& cmd,
                                                         const tgfx::Matrix3D& parent,
                                                         const Layer3DArgs& args) const {
  auto layer = tgfx::Layer::Make();
  auto own = matrix3DFromOps(cmd.ops, args.values);
  layer->setMatrix3D(own);
  auto placed = parent * own;
  if (!opensContext(cmd)) {
    addLayerContent(layer.get(), cmd, 0, &cmd, {}, placed, args);
    return layer;
  }
  layer->setPreserve3D(true);
  std::vector<const DrawCmd*> run;
  size_t slot = 0;
  auto flush = [&] {
    if (!run.empty()) {
      addLayerContent(layer.get(), cmd, slot, nullptr, run, placed, args);
      run.clear();
    }
    slot++;
  };
  for (const auto& child : cmd.children) {
    if (child.kind != DrawCmd::Group) {
      run.push_back(&child);
      continue;
    }
    // Flushed even for a hidden group, so each run keeps its cache slot.
    flush();
    if (!hidden(child, args.values)) {
      layer->addChild(buildLayer3D(child, placed, args));
    }
  }
  flush();
  return layer;
}

/// `body` (a plane: clip, layer, grain and subtree) or `run`, rasterised into one image layer.
void SceneRenderer::addLayerContent(tgfx::Layer* layer, const DrawCmd& owner, size_t slot,
                                    const DrawCmd* body, std::span<const DrawCmd* const> run,
                                    const tgfx::Matrix3D& placed, const Layer3DArgs& args) const {
  if (owner.layerContents.size() <= slot) {
    owner.layerContents.resize(slot + 1);
  }
  auto& content = owner.layerContents[slot];
  auto& paint = *args.paint;
  bool still = true;
  bool fills = false;
  if (body != nullptr) {
    still = body->layerAlpha.slot < 0 && body->grain == nullptr &&
            std::none_of(body->children.begin(), body->children.end(), [](const DrawCmd& child) { return child.dynamic; });
    fills = body->fills;
  } else {
    for (const auto* cmd : run) {
      still = still && !cmd->dynamic;
      fills = fills || cmd->fills;
    }
  }
  std::shared_ptr<tgfx::Picture> picture;
  auto record = [&] {
    tgfx::PictureRecorder recorder;
    auto* into = recorder.beginRecording();
    if (body != nullptr) {
      drawGroupBody(into, *body, args.values, args.strings, args.stringGen, args.stringPushed,
                    paint);
    } else {
      for (const auto* cmd : run) {
        drawCmd(into, *cmd, args.values, args.strings, args.stringGen, args.stringPushed, paint);
      }
    }
    picture = recorder.finishRecordingAsPicture();
  };
  tgfx::Rect bounds = content.bounds;
  if (!still || content.image == nullptr) {
    record();
    if (picture == nullptr) {
      return;
    }
    bounds = picture->getBounds();
    if (fills) {
      // A fill draws to the clip: the plane's own, else the canvas the context sits on.
      bool clipped = body != nullptr && body->clipKind != 0 && !body->invertClip;
      bounds.join(clipped && body->clipKind == 3 && body->clipPath != nullptr
                      ? body->clipPath->getBounds()
                      : (clipped ? body->clipRect : args.canvasRect));
    }
    bounds.roundOut();
  }
  if (bounds.isEmpty()) {
    return;
  }
  float scale = args.canvasScale * layerMagnification(placed, bounds);
  scale = std::min({scale, kMaxLayerSide / bounds.width(), kMaxLayerSide / bounds.height()});
  if (!(scale > 0)) {
    return;
  }
  if (!still || content.image == nullptr || content.scale != scale) {
    if (picture == nullptr) {
      record();
      if (picture == nullptr) {
        return;
      }
    }
    int width = std::max(1, static_cast<int>(std::ceil(bounds.width() * scale)));
    int height = std::max(1, static_cast<int>(std::ceil(bounds.height() * scale)));
    auto toPixels = tgfx::Matrix::MakeTrans(-bounds.left, -bounds.top);
    toPixels.postScale(scale, scale);
    content = {tgfx::Image::MakeFrom(picture, width, height, &toPixels, args.colorSpace), bounds,
               scale};
  }
  auto image = tgfx::ImageLayer::Make();
  image->setImage(content.image);
  auto toLocal = tgfx::Matrix::MakeScale(1 / scale);
  toLocal.postTranslate(bounds.left, bounds.top);
  image->setMatrix(toLocal);
  layer->addChild(std::move(image));
  if (!still) {
    content = {};
  }
}

void SceneRenderer::render(tgfx::Canvas* canvas,
                           std::span<const double> values,
                           std::span<const std::string> stringValues,
                           uint64_t stringGen,
                           std::span<const uint8_t> stringPushed) {
  std::shared_ptr<const Program> program;
  {
    std::lock_guard<std::mutex> lock(programMutex_);
    program = program_;
  }
  if (program == nullptr) {
    return;
  }
  tgfx::Paint paint;
  for (const auto& cmd : *program) {
    drawCmd(canvas, cmd, values, stringValues, stringGen, stringPushed, paint);
  }
}

}  // namespace tgfxrn
