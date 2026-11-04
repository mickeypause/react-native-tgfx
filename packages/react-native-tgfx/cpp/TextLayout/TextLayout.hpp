// Paragraph layout for <Text> and <TextPath>; the memoisation trade-off is in docs/text-layout.md.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tgfx/core/Font.h"
#include "tgfx/core/Path.h"

namespace tgfx {
class TextBlob;
}

namespace tgfxrn {

enum class TextAlign { Left = 0,
                       Center = 1,
                       Right = 2 };

struct TextRun {
  std::string text;
  std::string family;
  int weight = 0;
  /// 0 inherits the paragraph's, 1 forces italic, 2 forces upright. A bool cannot say "unset".
  int italic = 0;
  float letterSpacing = 0;
  /// 0 means the paragraph's size, which is the one that can animate; a span's own is static.
  float size = 0;
  bool hasColor = false;
  uint32_t color = 0;
  /// String slot the content follows (setStringValues), or -1 for static text.
  int slot = -1;
};

struct TextSpec {
  std::string family;
  int weight = 400;
  bool italic = false;
  /// Forced on top of whatever FontRegistry::match already synthesises for a far match.
  bool fauxBold = false;
  bool fauxItalic = false;
  float letterSpacing = 0;
  /// Wrap width in px. 0 means no wrapping: only explicit '\n' breaks a line.
  float wrapWidth = 0;
  TextAlign align = TextAlign::Left;
  /// Baseline-to-baseline distance in px. 0 means the font's own line spacing.
  float lineHeight = 0;
  int maxLines = 0;
  /// Appended to the last line when maxLines truncates. Empty means a hard cut.
  std::string ellipsis;
  /// <TextPath> only: distance along the path where the first glyph starts.
  float startOffset = 0;
};

struct PlacedGlyph {
  uint16_t id = 0;
  float x = 0;
  float y = 0;
};

struct LineMetrics {
  float width = 0;
  float x = 0;
  float baseline = 0;
  /// Negative, as in tgfx and Skia: the extent above this line's baseline.
  float ascent = 0;
  float descent = 0;
  /// Byte offsets into the concatenated run text, for hit testing.
  int32_t startIndex = 0;
  int32_t endIndex = 0;
};

struct TextMetrics {
  float width = 0;
  float height = 0;
  /// Negative, as in tgfx and Skia: the extent above the baseline.
  float ascent = 0;
  float descent = 0;
  float lineCount = 0;
  std::vector<LineMetrics> lines;
};

/// Where a point in paragraph coordinates falls, as an insertion point.
struct TextPosition {
  /// The insertion point, as a UTF-8 byte offset into the concatenated run
  /// text. Always the offset a caret goes at — never a glyph index to advance
  /// past.
  int32_t index = 0;
  /// Which half produced it: true when the point fell on the leading half of
  /// the glyph that starts at `index`, false when it fell on the trailing half
  /// of the glyph that ends there, or past the end of the line. That is the
  /// affinity a caret needs at a wrap, where one offset has two positions.
  bool leading = true;
  int32_t line = 0;
};

/// One run of a range on a single line, in paragraph coordinates.
struct TextBox {
  float x = 0;
  float y = 0;
  float width = 0;
  float height = 0;
  int32_t line = 0;
  /// The UTF-8 byte range of the glyphs this box covers: the requested range
  /// clipped to the line and rounded out to whole codepoints. `endIndex` is
  /// exclusive, unlike LineMetrics::endIndex, which names its last glyph.
  int32_t startIndex = 0;
  int32_t endIndex = 0;
};

struct TextBlobRun {
  std::shared_ptr<tgfx::TextBlob> blob;
  bool hasColor = false;
  uint32_t color = 0;
};

struct TextLayout {
  std::vector<TextBlobRun> runs;
  TextMetrics metrics;
};

TextLayout layoutText(const std::vector<TextRun>& runs, const TextSpec& spec, float size);

/// One id per codepoint in the face `spec` names, with no fallback: 0 where it has no glyph.
std::vector<double> glyphIdsFor(const std::string& text, const TextSpec& spec, float size);

TextLayout layoutGlyphs(const std::vector<PlacedGlyph>& glyphs, const TextSpec& spec, float size);

/// The insertion point nearest (x, y), for a caret or a hit test. The y picks a
/// line and is clamped to the paragraph, so a point above or below still lands.
TextPosition positionAtCoordinate(const std::vector<TextRun>& runs, const TextSpec& spec,
                                  float size, float x, float y);

/// The boxes a UTF-8 byte range occupies, one per line it crosses — what a
/// selection highlight draws, and what a single-glyph range reports as bounds.
std::vector<TextBox> boxesForRange(const std::vector<TextRun>& runs, const TextSpec& spec,
                                   float size, int32_t start, int32_t end);

/// The same paragraph as one outline path. Empty for a face with no outlines, e.g. colour emoji.
tgfx::Path textToPath(const std::vector<TextRun>& runs, const TextSpec& spec, float size);

TextLayout layoutTextOnPath(const std::vector<TextRun>& runs, const TextSpec& spec,
                            float size, const tgfx::Path& path);

}  // namespace tgfxrn
