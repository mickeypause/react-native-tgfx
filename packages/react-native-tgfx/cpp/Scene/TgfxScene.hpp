// SceneRenderer compiles the draw program once per change; render() resolves only the value slots.
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "TextLayout.hpp"
#include "tgfx/core/BlendMode.h"
#include "tgfx/core/Matrix.h"
#include "tgfx/core/Paint.h"
#include "tgfx/core/Mesh.h"
#include "tgfx/core/Path.h"
#include "tgfx/core/Picture.h"
#include "tgfx/core/Rect.h"
#include "tgfx/core/SamplingOptions.h"
#include "tgfx/core/Shape.h"
#include "tgfx/core/TileMode.h"

namespace tgfx {
class Canvas;
class ColorFilter;
class ColorSpace;
class Image;
class ImageFilter;
class Layer;
class Matrix3D;
class Shader;
class Typeface;
}  // namespace tgfx

namespace tgfxrn {

/// Wire-format version, mirrored by SCENE_VERSION in src/compiler/writer.ts.
inline constexpr double kSceneVersion = 31;

/// How an image is scaled into its destination rect; see fitRects() in TgfxScene.cpp.
enum class Fit { Fill = 0,
                 Contain,
                 Cover,
                 FitWidth,
                 FitHeight,
                 None,
                 ScaleDown };

struct SceneReader;

/// Filters built since process start. Built at setScene only, which SceneTest.cpp asserts.
size_t filterBuildCount();

/// A float that is either a literal or an animated slot (slot < 0 => literal).
struct AnimF {
  float literal = 0;
  int slot = -1;

  float resolve(std::span<const double> values) const {
    if (slot >= 0 && static_cast<size_t>(slot) < values.size()) {
      return static_cast<float>(values[static_cast<size_t>(slot)]);
    }
    return literal;
  }

  /// The same word as an integer code — enum slots and registry handles.
  int resolveInt(std::span<const double> values) const {
    float v = resolve(values);
    return v >= static_cast<float>(std::numeric_limits<int>::min()) &&
                   v <= static_cast<float>(std::numeric_limits<int>::max())
               ? static_cast<int>(v)
               : 0;
  }
};

struct ImageShaderCmd {
  AnimF handle{};
  tgfx::TileMode tileX = tgfx::TileMode::Clamp;
  tgfx::TileMode tileY = tgfx::TileMode::Clamp;
  tgfx::SamplingOptions sampling{tgfx::FilterMode::Linear, tgfx::MipmapMode::None};
  bool hasRect = false;
  tgfx::Rect rect = tgfx::Rect::MakeEmpty();
  Fit fit = Fit::Contain;

  std::shared_ptr<tgfx::Shader> build(const std::shared_ptr<tgfx::Image>& image) const;
};

/// One `uniform shader` slot; `image` is set instead when the child resolves per frame.
struct RuntimeShaderCmd;

struct ChildShaderCmd {
  std::shared_ptr<tgfx::Shader> shader;
  std::shared_ptr<ImageShaderCmd> image;
  std::shared_ptr<RuntimeShaderCmd> runtime;
  std::shared_ptr<struct BlendShaderCmd> blend;
  std::shared_ptr<struct GradientShaderCmd> gradient;

  bool dynamic() const {
    return image != nullptr || runtime != nullptr || blend != nullptr ||
           gradient != nullptr;
  }

  std::shared_ptr<tgfx::Shader> resolve(std::span<const double> values) const;
};

struct RuntimeShaderCmd {
  std::string body;
  std::string preamble;
  std::vector<std::string> functionNames;
  std::vector<tgfx::RuntimeShaderUniform> uniforms;
  std::vector<AnimF> slots;
  std::vector<ChildShaderCmd> children;

  std::shared_ptr<tgfx::Shader> build(std::span<const double> values) const;
};

/// A gradient whose geometry, stops or stop colours carry value slots.
struct GradientShaderCmd {
  /// 0 linear, 1 radial, 2 sweep, 3 diamond — the wire's shader kinds.
  int kind = 0;
  /// linear: x0 y0 x1 y1; radial/diamond: cx cy r; sweep: cx cy a0 a1.
  std::array<AnimF, 4> geometry{};
  /// One packed word per stop, or four channels per stop when animColors.
  bool animColors = false;
  std::vector<uint32_t> colors;
  std::vector<AnimF> colorChannels;
  std::vector<AnimF> positions;

  bool animated() const;
  std::shared_ptr<tgfx::Shader> build(std::span<const double> values) const;
};

/// `<FilmGrain>` on a group: tgfx's NoiseStyle over the group's own pixels.
struct GrainCmd {
  /// 0 mono, 1 duo, 2 multi.
  int mode = 0;
  AnimF size{4, -1};
  AnimF density{0.5f, -1};
  AnimF seed{0, -1};
  AnimF blendMode{3, -1};
  AnimF opacity{1, -1};
  uint32_t color = 0xffffffff;
  uint32_t secondColor = 0xff000000;
  std::array<AnimF, 4> rgba{};
  std::array<AnimF, 4> secondRgba{};
  bool animColor = false;
  bool animSecondColor = false;
};

/// A colour filter whose parameters may carry value slots; static ones never get here.
struct ColorFilterCmd {
  /// 0 matrix, 1 blend, 2 luma, 3 alphaThreshold, 4 compose
  /// (5 on the wire folds into 1 + animColor).
  int kind = 0;
  std::array<AnimF, 20> matrix{};
  AnimF threshold{};
  uint32_t color = 0;
  std::array<AnimF, 4> rgba{};
  bool animColor = false;
  AnimF mode{3, -1};
  std::shared_ptr<ColorFilterCmd> inner;
  std::shared_ptr<ColorFilterCmd> outer;
  /// A subtree with no slot of its own is built once, even inside a chain that
  /// has one somewhere else. Filled on the first build.
  mutable std::shared_ptr<tgfx::ColorFilter> cached;

  bool animated() const;
  std::shared_ptr<tgfx::ColorFilter> build(std::span<const double> values) const;
};

/// An image filter (blur / drop shadow) whose parameters may carry value slots.
struct ImageFilterCmd {
  /// 0 blur, 1 dropShadow, 2 compose, 3 colorFilter.
  int kind = 0;
  AnimF blurX{};
  AnimF blurY{};
  tgfx::TileMode tile = tgfx::TileMode::Decal;
  AnimF dx{};
  AnimF dy{};
  uint32_t color = 0xff000000;
  std::array<AnimF, 4> rgba{};
  bool animColor = false;
  /// Bit 1 shadow only, bit 2 inner.
  int flags = 0;
  std::shared_ptr<ImageFilterCmd> first;
  std::shared_ptr<ImageFilterCmd> then;
  /// Kind 3: a colour filter placed in the image chain, so it runs *after*
  /// whatever precedes it rather than on the paint before the draw.
  std::shared_ptr<ColorFilterCmd> colorFilter;
  /// As on ColorFilterCmd: a static subtree of an animated chain is built once.
  mutable std::shared_ptr<tgfx::ImageFilter> cached;

  bool animated() const;
  std::shared_ptr<tgfx::ImageFilter> build(std::span<const double> values) const;
};

/// A <Blend> whose children cannot all be folded at commit — a video frame or an animated uniform.
struct BlendShaderCmd {
  tgfx::BlendMode mode = tgfx::BlendMode::SrcOver;
  std::vector<ChildShaderCmd> children;

  std::shared_ptr<tgfx::Shader> build(std::span<const double> values) const;
};

struct PaintCmd {
  uint32_t color = 0xff000000;
  /// Set when the colour is animated; `color` above holds the last committed value.
  std::array<AnimF, 4> rgba{};
  bool animColor = false;
  bool stroke = false;
  /// Set when style/antiAlias travel as their own animated words (flags bit 32).
  bool animStyle = false;
  AnimF styleWord{};
  AnimF aaWord{1, -1};
  AnimF strokeWidth{1, -1};
  AnimF cap{};
  AnimF join{};
  AnimF miter{4, -1};
  bool antiAlias = true;
  AnimF alpha{1, -1};
  std::shared_ptr<tgfx::Shader> shader;  // built once for static gradients
  std::shared_ptr<RuntimeShaderCmd> runtime;
  /// Set when the gradient's geometry or stops animate; `shader` is null then.
  std::shared_ptr<GradientShaderCmd> gradient;
  /// Set only for an <ImageShader> on a hardware-buffer handle; otherwise the shader stands.
  std::shared_ptr<ImageShaderCmd> imageShader;
  /// Set when a <Blend> child is dynamic, so the fold has to happen per frame.
  std::shared_ptr<BlendShaderCmd> blendShader;
  /// Filters and blend mode, built once by setScene when their inputs are literals.
  AnimF blendWord{3, -1};
  tgfx::BlendMode blend = tgfx::BlendMode::SrcOver;
  std::shared_ptr<tgfx::ColorFilter> colorFilter;
  std::shared_ptr<tgfx::ImageFilter> imageFilter;
  /// Set instead of the two above when the filter's parameters animate.
  std::shared_ptr<ColorFilterCmd> colorFilterCmd;
  std::shared_ptr<ImageFilterCmd> imageFilterCmd;
  /// Set when a blend mode meets an image filter, which tgfx cannot honour on one paint.
  std::shared_ptr<tgfx::Paint> layer;
  tgfx::Paint cached;

  bool animated() const {
    return animColor || alpha.slot >= 0 || runtime != nullptr ||
           gradient != nullptr || imageShader != nullptr ||
           blendShader != nullptr || colorFilterCmd != nullptr ||
           imageFilterCmd != nullptr || animStyle || cap.slot >= 0 ||
           join.slot >= 0 || miter.slot >= 0 || blendWord.slot >= 0 ||
           ((stroke || animStyle) && strokeWidth.slot >= 0);
  }
};

/// type: 0 translate(a=x,b=y), 1 scale(a=x,b=y), 2 rotate(a), 3 skewX(a), 4 skewY(a),
/// 5 rotateX(a), 6 rotateY(a), 7 perspective(a), 8 translateZ(a). 5 and up make the group 3D.
inline constexpr int kFirst3DOp = 5;

struct TransformOpCmd {
  int type = 0;
  AnimF a;
  AnimF b;
};

/// A ceiling on <Span> children, so a corrupt run count cannot loop without bound.
inline constexpr int kMaxTextRuns = 4096;

/// The same ceiling for <Glyphs>, whose count comes straight off the wire.
inline constexpr int kMaxGlyphs = 65536;

/// A <Text> or <TextPath>. Behind a shared_ptr so DrawCmd, which readNode recurses on, stays small.
struct TextCmd {
  std::vector<TextRun> runs;
  TextSpec spec;
  tgfx::Path path;
  bool onPath = false;
  std::vector<PlacedGlyph> glyphs;
  /// True when any run's content follows a string slot.
  bool liveRuns = false;

  /**
   * The layout for the size (and, for slotted runs, string values) last drawn.
   * Slotted run content is pulled from `strings` and re-shaped on change — on
   * the render thread, with no scene re-encode. Mutable: render thread only.
   */
  const TextLayout& resolve(float size, std::span<const std::string> strings,
                            uint64_t stringGen,
                            std::span<const uint8_t> pushed) const;

 private:
  mutable float layoutSize = -1;
  mutable uint64_t layoutGen = 0;
  mutable TextLayout layout;
};

/**
 * An <Atlas>'s sprites: baked matrices when every field is a literal, the
 * animated words (or a shared buffer id) when not — resolved into the scratch
 * vectors at draw, which are mutable because only the render thread fills them.
 */
struct AtlasCmd {
  std::vector<tgfx::Matrix> matrices;
  std::vector<tgfx::Rect> sprites;
  /// Seven AnimF per sprite (srcX srcY srcW srcH scale cx cy) when any moves.
  std::vector<AnimF> words;
  bool animated = false;
  /// A BufferRegistry id; > 0 replaces both inline forms above.
  int32_t bufferId = 0;
  AnimF count{};
  mutable std::vector<tgfx::Matrix> scratchMatrices;
  mutable std::vector<tgfx::Rect> scratchSprites;
};

/// A <Vertices> fed from a shared buffer, rebuilt into a mesh at draw.
struct MeshBufferCmd {
  int32_t bufferId = 0;
  bool strip = false;
  int maxVertices = 0;
  int maxIndices = 0;
  bool hasTextures = false;
  bool hasColors = false;
  AnimF vertexCount{};
  AnimF indexCount{};
  mutable std::vector<tgfx::Point> scratchPositions;
  mutable std::vector<tgfx::Point> scratchTexCoords;
  mutable std::vector<tgfx::Color> scratchColors;
  mutable std::vector<uint16_t> scratchIndices;
};

struct PathEffectCmd {
  bool trim = false;
  AnimF start{0, -1};
  AnimF end{1, -1};
  std::vector<float> intervals;
  /// Only ever set for an even count of at least two, which is all MakeDash accepts.
  bool dash = false;
  AnimF phase{0, -1};
  float corner = 0;

  /// Trim first, as react-native-skia bakes it into the path, then corner and dash.
  void apply(tgfx::Path* path, std::span<const double> v) const;
};

/// One rasterised run of a 3D layer, kept while what it draws is static.
struct LayerContent {
  std::shared_ptr<tgfx::Image> image;
  tgfx::Rect bounds = tgfx::Rect::MakeEmpty();
  float scale = 0;
};

struct DrawCmd {
  enum Kind {
    Group,
    Rect,
    RRect,
    Oval,
    Circle,
    Line,
    Path,
    Image,
    Text,
    Fill,
    CameraFrame,
    Mask,
    Atlas,
    Vertices,
    Backdrop
  };

  /// Backdrop only: the panel's shape and the style's parameters.
  /// Static: it is a clip, rebuilt only when the program is.
  float cornerRadius = 0;
  /// The material, all animatable: each is read per frame into the style the
  /// draw builds anyway, so a slot write costs nothing a literal does not.
  AnimF blur;
  AnimF refraction;
  AnimF depth;
  AnimF frost;
  AnimF dispersion;
  AnimF splay;
  AnimF lightAngle;
  AnimF lightIntensity;

  /// Mask only: the subtree whose alpha or luminance cuts `children` down.
  std::vector<DrawCmd> maskChildren;
  bool maskLuminance = false;

  /// Group extras: 0 no clip, 1 rect, 2 round rect, 3 path.
  int clipKind = 0;
  bool invertClip = false;
  bool layer = false;
  /// Group only: record this subtree once and replay it, set only for a static one.
  bool cachePicture = false;
  /// Filled on the first draw and replayed after, so the recording costs one frame.
  mutable std::shared_ptr<tgfx::Picture> picture;
  /// Group only: behind a pointer so an ungrained group costs one word.
  std::shared_ptr<GrainCmd> grain;
  AnimF layerAlpha{1, -1};
  tgfx::Rect clipRect = tgfx::Rect::MakeEmpty();
  float clipRadiusX = 0;
  float clipRadiusY = 0;
  /// Behind a pointer so a DrawCmd that does not clip costs no Path allocation.
  std::shared_ptr<tgfx::Path> clipPath;
  Kind kind = Group;

  /// Kind-word bit 64: skip draw and hit-test when it resolves below 0.5.
  AnimF visible{1, -1};
  AnimF x, y, w, h, rx, ry, cx, cy, r, x1, y1, x2, y2, fontSize;
  PaintCmd paint;

  bool hasMatrix = false;
  tgfx::Matrix matrix = tgfx::Matrix::I();
  std::vector<TransformOpCmd> ops;
  bool threeD = false;
  bool preserve3D = false;
  /// Read-time: something in this subtree resolves per frame.
  bool dynamic = false;
  /// Read-time: a <Fill> in this subtree, which has no bounds of its own.
  bool fills = false;
  mutable std::vector<LayerContent> layerContents;
  std::vector<DrawCmd> children;

  tgfx::Path path;
  bool lineAnimated = false;
  /// String slot the path data follows; re-parsed on change, render thread only.
  int pathSlot = -1;
  /// Wire flags for a slotted path, re-applied after each re-parse (bit 8 even-odd).
  int pathFlags = 0;
  mutable uint64_t pathGen = 0;
  mutable tgfx::Path livePath;
  /// Stroked form of a static path + static stroke paint, built once on the render thread so a
  /// frame does not re-stroke and re-probe (rect/oval) every path it draws.
  mutable std::shared_ptr<tgfx::Shape> strokedShape;
  /// 0 unprobed, 1 cache as strokedShape, -1 line/rect/oval/rrect: keeps drawPath's fast path.
  mutable int8_t strokeCache = 0;
  /// Path effects. Held rather than baked when trim or dash phase can move per frame.
  std::shared_ptr<PathEffectCmd> effects;
  // Null for a hardware-buffer handle: caching the resolved image would pin the producer's buffer.
  std::shared_ptr<tgfx::Image> image;
  int32_t imageHandle = 0;
  /// The handle word's slot form; when live, the drawn image follows it per frame.
  AnimF imageWord{};
  /// The image last resolved for a live handle, so a stable frame is one compare.
  mutable int32_t liveHandle = 0;
  mutable std::shared_ptr<tgfx::Image> liveImage;
  Fit fit = Fit::Contain;
  tgfx::SamplingOptions sampling{tgfx::FilterMode::Linear, tgfx::MipmapMode::None};
  tgfx::Rect srcRect = tgfx::Rect::MakeEmpty();
  /// Set when the source rect animates; `srcRect` holds the literals then.
  bool animSrcRect = false;
  std::array<AnimF, 4> srcRectWords{};
  // CameraFrame: quarter turns clockwise, 0-3, applied before the fit.
  int quarterTurns = 0;
  bool mirror = false;
  std::shared_ptr<TextCmd> text;
  /// Behind a pointer so every non-atlas DrawCmd costs one word, not two vectors.
  std::shared_ptr<AtlasCmd> atlas;
  /// Built once when the program is read: MakeCopy owns its own vertex data.
  std::shared_ptr<tgfx::Mesh> mesh;
  /// Set instead of `mesh` when the vertices come from a shared buffer.
  std::shared_ptr<MeshBufferCmd> meshBuffer;
};

using Program = std::vector<DrawCmd>;

class SceneRenderer {
 public:
  void setScene(std::span<const double> words,
                const std::vector<std::string>& strings);

  /**
   * `stringValues`/`stringGen` back the program's string slots (live text and
   * path data). `stringPushed` marks which slots a push has actually written —
   * an unpushed slot keeps its compiled literal, so a sparse push can never
   * blank a neighbour. The generation gates change detection, so an idle frame
   * never compares strings. Callers with no string slots may omit all three.
   */
  void render(tgfx::Canvas* canvas, std::span<const double> values,
              std::span<const std::string> stringValues = {},
              uint64_t stringGen = 0,
              std::span<const uint8_t> stringPushed = {});

  /**
   * Index paths of the nodes whose geometry contains the point, front-to-back.
   * Flattened as [depth, i0, i1, ..., depth, j0, ...] so it crosses the bridge
   * as one array. `precise` tests the filled region rather than the bounds.
   * `stringValues`/`stringPushed` let a string-slotted path be tested against
   * what is drawn (re-parsed locally — `livePath` belongs to the render thread).
   * `root` is the transform the renderer draws under (the viewport), which the
   * walk inverts along with each group's — without it a zoomed scene reports
   * hits from where the nodes used to be.
   */
  std::vector<double> hitTest(float x, float y, bool precise,
                              std::span<const double> values,
                              std::span<const std::string> stringValues = {},
                              std::span<const uint8_t> stringPushed = {},
                              const tgfx::Matrix& root = tgfx::Matrix::I());

  void clearCaches();

  /// How many root nodes the last accepted program holds; 0 if it was rejected.
  size_t rootCount() const;

  /// Whether the program reads the image / font registry, whose epochs are
  /// process-wide: a scene that reads neither must not redraw when they move.
  bool usesImages() const;
  bool usesText() const;
  /// Whether the program samples its own surface (Backdrop); see TgfxRenderer::draw.
  bool usesBackdrop() const;

 private:
  std::shared_ptr<const Program> program_;
  bool usesImages_ = false;
  bool usesText_ = false;
  bool usesBackdrop_ = false;
  mutable std::mutex programMutex_;

  bool readNode(SceneReader& reader, DrawCmd& cmd, int depth);
  bool readNodeFields(SceneReader& reader, DrawCmd& cmd, int depth);
  void readTextSpec(SceneReader& reader, TextSpec& spec);
  void readTextRuns(SceneReader& reader, TextCmd& text);
  void readPaint(SceneReader& reader, PaintCmd& paint);
  void readFilters(SceneReader& reader, PaintCmd& paint);
  std::shared_ptr<ColorFilterCmd> readColorFilterCmd(SceneReader& reader, int depth);
  std::shared_ptr<ImageFilterCmd> readImageFilterCmd(SceneReader& reader, int depth);
  std::shared_ptr<tgfx::Shader> readShader(SceneReader& reader, PaintCmd* out,
                                           int depth = 0);

  void drawCmd(tgfx::Canvas* canvas, const DrawCmd& cmd, std::span<const double> values,
               std::span<const std::string> strings, uint64_t stringGen,
               std::span<const uint8_t> stringPushed, tgfx::Paint& paint) const;

  void drawGroupBody(tgfx::Canvas* canvas, const DrawCmd& cmd, std::span<const double> values,
                     std::span<const std::string> strings, uint64_t stringGen,
                     std::span<const uint8_t> stringPushed, tgfx::Paint& paint) const;

  struct Layer3DArgs {
    float canvasScale = 1;
    std::shared_ptr<tgfx::ColorSpace> colorSpace;
    tgfx::Rect canvasRect = tgfx::Rect::MakeEmpty();
    std::span<const double> values;
    std::span<const std::string> strings;
    uint64_t stringGen = 0;
    std::span<const uint8_t> stringPushed;
    tgfx::Paint* paint = nullptr;
  };

  std::shared_ptr<tgfx::Layer> buildLayer3D(const DrawCmd& cmd, const tgfx::Matrix3D& parent,
                                            const Layer3DArgs& args) const;

  void addLayerContent(tgfx::Layer* layer, const DrawCmd& owner, size_t slot,
                       const DrawCmd* body, std::span<const DrawCmd* const> run,
                       const tgfx::Matrix3D& placed, const Layer3DArgs& args) const;
};

}  // namespace tgfxrn
