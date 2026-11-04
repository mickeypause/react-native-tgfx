#include "TextLayout.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <unordered_set>

#include "FontRegistry.hpp"
#include "TextShaping.hpp"
#include "Log.hpp"
#include "tgfx/core/FontMetrics.h"
#include "tgfx/core/PathMeasure.h"
#include "tgfx/core/Point.h"
#include "tgfx/core/RSXform.h"
#include "tgfx/core/TextBlob.h"
#include "tgfx/core/TextBlobBuilder.h"
#include "tgfx/core/UTF.h"

namespace tgfxrn {
namespace {

struct Glyph {
  tgfx::GlyphID id = 0;
  uint8_t font = 0;
  float advance = 0;
  int32_t code = 0;
  uint16_t run = 0;
  int32_t index = 0;
  /// UTF-8 bytes this glyph's cluster took, so a caret can land after it.
  int32_t length = 0;
  /// Mark positioning from the shaper, in px, relative to the pen.
  float offsetX = 0;
  float offsetY = 0;
  /// The bidi embedding level. Odd is right-to-left; a line is reordered by it.
  uint8_t level = 0;
  /// Whitespace UAX#9 L1 resets to the paragraph level at the end of a line.
  bool whitespace = false;
};

struct Line {
  std::vector<Glyph> glyphs;
  float width = 0;
  int32_t startIndex = 0;
};

using Stacks = std::vector<std::unique_ptr<struct FontStack>>;

struct FontStack {
  std::vector<tgfx::Font> fonts;
};

std::unique_ptr<FontStack> buildStack(const std::string& family, int weight, bool italic,
                                      float size, bool forceBold, bool forceItalic) {
  auto& registry = FontRegistry::instance();
  auto match = registry.match(family, weight, italic);
  if (match.typeface == nullptr || size <= 0) {
    return nullptr;
  }
  auto stack = std::make_unique<FontStack>();
  tgfx::Font primary(match.typeface, size);
  primary.setFauxBold(match.fauxBold || forceBold);
  primary.setFauxItalic(match.fauxItalic || forceItalic);
  stack->fonts.push_back(std::move(primary));
  for (const auto& typeface : registry.fallbacks()) {
    // Only what the caller forced: `match` describes the primary face, not this one.
    auto& fallback = stack->fonts.emplace_back(typeface, size);
    fallback.setFauxBold(forceBold);
    fallback.setFauxItalic(forceItalic);
  }
  return stack;
}

/// Shape one run of the paragraph and append it in logical order. Newlines and
/// carriage returns keep their place as zero-width breaks, exactly as the
/// codepoint walk this replaced did, so line breaking is unchanged.
void shapeInto(std::vector<Glyph>& glyphs, const std::string& paragraph, int32_t begin,
               int32_t end, const FontStack& stack, float letterSpacing, uint16_t run,
               const BidiLevels& levels) {
  auto shaped = shapeRun(paragraph, begin, end, stack.fonts, letterSpacing, levels);
  for (const auto& s : shaped) {
    Glyph glyph;
    glyph.id = s.id;
    glyph.font = s.font;
    glyph.advance = s.advance;
    glyph.offsetX = s.offsetX;
    glyph.offsetY = s.offsetY;
    glyph.level = s.level;
    glyph.code = s.code;
    glyph.whitespace = s.whitespace;
    glyph.run = run;
    glyph.index = s.index;
    glyph.length = s.length;
    if (glyph.code == '\n' || glyph.code == '\r') {
      // A break carries no ink and no width, whatever the face gave it.
      glyph.id = 0;
      glyph.advance = 0;
      glyph.offsetX = 0;
      glyph.offsetY = 0;
    }
    glyphs.push_back(glyph);
  }
}

std::vector<Line> breakLines(const std::vector<Glyph>& glyphs, float wrapWidth,
                             int32_t textLength) {
  std::vector<Line> lines;
  auto emit = [&](size_t from, size_t to, float width) {
    Line line;
    line.glyphs.assign(glyphs.begin() + static_cast<long>(from),
                       glyphs.begin() + static_cast<long>(to));
    line.width = width;
    line.startIndex = from < glyphs.size() ? glyphs[from].index : textLength;
    lines.push_back(std::move(line));
  };
  size_t start = 0;
  size_t breakStart = std::string::npos;  // first glyph of the next line
  size_t breakEnd = 0;                    // one past the last glyph of this one
  float breakWidth = 0;
  float x = 0;
  for (size_t i = 0; i < glyphs.size(); i++) {
    const auto& glyph = glyphs[i];
    if (glyph.code == '\n') {
      emit(start, i, x);
      start = i + 1;
      x = 0;
      breakStart = std::string::npos;
      continue;
    }
    if (wrapWidth > 0 && i > start && x + glyph.advance > wrapWidth) {
      if (breakStart != std::string::npos) {
        emit(start, breakEnd, breakWidth);
        start = breakStart;
        i = start - 1;  // re-measure the tail of the word on the new line
        x = 0;
        breakStart = std::string::npos;
        continue;
      }
      // A cluster is one unit of text however many glyphs it shaped to, so a
      // hard break walks back off one: an Indic matra reorders ahead of its own
      // consonant, and splitting the two puts them on different lines.
      size_t at = i;
      while (at > start && glyphs[at].index == glyphs[at - 1].index) {
        at--;
      }
      if (at == start) {
        at = i;  // one cluster wider than the wrap: it has to break somewhere.
      }
      float kept = 0;
      for (size_t j = start; j < at; j++) {
        kept += glyphs[j].advance;
      }
      emit(start, at, kept);
      start = at;
      i = at - 1;
      x = 0;
      continue;
    }
    if (glyph.code == ' ') {
      breakEnd = i;
      breakWidth = x;
      breakStart = i + 1;
    }
    x += glyph.advance;
  }
  emit(start, glyphs.size(), x);
  return lines;
}

float widthOf(const std::vector<Glyph>& glyphs) {
  float width = 0;
  for (const auto& glyph : glyphs) {
    width += glyph.advance;
  }
  return width;
}

/// UAX#9 L1 then L2 over one line: reset the trailing whitespace to the
/// paragraph level, then reverse each maximal run of level >= n for n from the
/// highest level down to the lowest odd one.
void reorderLine(std::vector<Glyph>& glyphs, uint8_t paragraphLevel) {
  if (glyphs.empty()) {
    return;
  }
  // L1: whitespace and separators at the end of the line belong to the
  // paragraph, not to whatever run they happen to follow. The bit comes from
  // the resolved class rather than a codepoint list, which would disagree with
  // it — U+00A0 looks like a space and is a common separator, not whitespace.
  for (size_t i = glyphs.size(); i-- > 0;) {
    if (!glyphs[i].whitespace) {
      break;
    }
    glyphs[i].level = paragraphLevel;
  }
  uint8_t highest = 0;
  uint8_t lowestOdd = 255;
  for (const auto& glyph : glyphs) {
    highest = std::max(highest, glyph.level);
    if ((glyph.level & 1) != 0) {
      lowestOdd = std::min(lowestOdd, glyph.level);
    }
  }
  if (lowestOdd == 255) {
    return;
  }
  for (uint8_t level = highest; level >= lowestOdd; level--) {
    size_t i = 0;
    while (i < glyphs.size()) {
      if (glyphs[i].level < level) {
        i++;
        continue;
      }
      size_t end = i;
      while (end < glyphs.size() && glyphs[end].level >= level) {
        end++;
      }
      std::reverse(glyphs.begin() + static_cast<long>(i), glyphs.begin() + static_cast<long>(end));
      i = end;
    }
    if (level == 0) {
      break;
    }
  }
}

void truncate(std::vector<Line>& lines, const TextSpec& spec, const FontStack& stack,
              uint16_t run) {
  if (spec.maxLines <= 0 || lines.size() <= static_cast<size_t>(spec.maxLines)) {
    return;
  }
  lines.resize(static_cast<size_t>(spec.maxLines));
  if (spec.ellipsis.empty()) {
    return;
  }
  std::vector<Glyph> tail;
  auto& last = lines.back();
  auto tailLevels = resolveParagraph(spec.ellipsis);
  shapeInto(tail, spec.ellipsis, 0, static_cast<int32_t>(spec.ellipsis.size()), stack,
            spec.letterSpacing, run, tailLevels);
  float tailWidth = widthOf(tail);
  while (!last.glyphs.empty() && spec.wrapWidth > 0 &&
         last.width + tailWidth > spec.wrapWidth) {
    last.width -= last.glyphs.back().advance;
    last.glyphs.pop_back();
  }
  // After the trim, not before it: the ellipsis stands in for text that is no
  // longer laid out, so it takes the offset just past the last glyph actually
  // kept. Length zero keeps a caret on it from landing in the middle of a
  // codepoint the ellipsis character happens to be as long as, and keeps it out
  // of a range's boxes, which measure real text.
  int32_t at = last.glyphs.empty() ? last.startIndex
                                   : last.glyphs.back().index + last.glyphs.back().length;
  for (auto& glyph : tail) {
    glyph.index = at;
    glyph.length = 0;
  }
  last.glyphs.insert(last.glyphs.end(), tail.begin(), tail.end());
  last.width += tailWidth;
}

std::vector<TextBlobRun> buildBlobs(const std::vector<Glyph>& glyphs,
                                    const std::vector<tgfx::Point>& positions,
                                    const std::vector<TextRun>& runs, const Stacks& stacks) {
  std::vector<TextBlobRun> out;
  auto paintOf = [&](const Glyph& glyph) {
    const auto& run = runs[glyph.run];
    return std::pair<bool, uint32_t>{run.hasColor, run.color};
  };
  size_t group = 0;
  while (group < glyphs.size()) {
    size_t groupEnd = group;
    while (groupEnd < glyphs.size() && paintOf(glyphs[groupEnd]) == paintOf(glyphs[group])) {
      groupEnd++;
    }
    tgfx::TextBlobBuilder builder;
    size_t run = group;
    while (run < groupEnd) {
      size_t end = run;
      while (end < groupEnd && glyphs[end].font == glyphs[run].font &&
             glyphs[end].run == glyphs[run].run) {
        end++;
      }
      size_t count = end - run;
      const auto& stack = *stacks[glyphs[run].run];
      const auto& buffer = builder.allocRunPos(stack.fonts[glyphs[run].font], count);
      if (buffer.glyphs == nullptr || buffer.positions == nullptr) {
        return {};
      }
      auto* points = reinterpret_cast<tgfx::Point*>(buffer.positions);
      for (size_t i = 0; i < count; i++) {
        buffer.glyphs[i] = glyphs[run + i].id;
        points[i] = positions[run + i];
      }
      run = end;
    }
    auto blob = builder.build();
    if (blob != nullptr) {
      auto paint = paintOf(glyphs[group]);
      out.push_back({std::move(blob), paint.first, paint.second});
    }
    group = groupEnd;
  }
  return out;
}

struct Paragraph {
  Stacks stacks;
  std::vector<tgfx::FontMetrics> metrics;
  bool empty = true;
};

size_t lastUsableRun(const struct Paragraph& paragraph);

Paragraph buildParagraph(const std::vector<TextRun>& runs, const TextSpec& spec, float size) {
  Paragraph out;
  out.stacks.reserve(runs.size());
  out.metrics.resize(runs.size());
  for (size_t i = 0; i < runs.size(); i++) {
    const auto& run = runs[i];
    // A faux bit reaches a run only on the axis it inherits: a span that set its own opts out.
    auto stack = buildStack(run.family.empty() ? spec.family : run.family,
                            run.weight != 0 ? run.weight : spec.weight,
                            run.italic == 0 ? spec.italic : run.italic == 1,
                            run.size > 0 ? run.size : size,
                            run.weight == 0 && spec.fauxBold,
                            run.italic == 0 && spec.fauxItalic);
    if (stack != nullptr) {
      out.metrics[i] = stack->fonts[0].getMetrics();
      out.empty = false;
    }
    out.stacks.push_back(std::move(stack));
  }
  return out;
}

size_t lastUsableRun(const Paragraph& paragraph) {
  for (size_t i = paragraph.stacks.size(); i > 0; i--) {
    if (paragraph.stacks[i - 1] != nullptr) {
      return i - 1;
    }
  }
  return 0;
}

struct Placement {
  std::vector<Glyph> glyphs;
  std::vector<tgfx::Point> positions;
  /// Which line each glyph landed on, parallel to the two above.
  std::vector<int32_t> lines;
  TextMetrics metrics;
};

/// Every run's text as one string, which is what a bidi paragraph is. Resolving
/// per run would let a `<Span>` that changes only a colour change the layout,
/// because sos and eos would reset at its seam.
std::string paragraphText(const std::vector<TextRun>& runs) {
  std::string all;
  for (const auto& run : runs) {
    all += run.text;
  }
  return all;
}

Placement place(const std::vector<TextRun>& runs, const TextSpec& spec,
                const Paragraph& paragraph) {
  Placement out;
  std::vector<Glyph> glyphs;
  auto text = paragraphText(runs);
  auto levels = resolveParagraph(text);
  bool rtl = levels.rtl;
  int32_t index = 0;
  for (size_t i = 0; i < runs.size(); i++) {
    const auto& run = runs[i];
    auto next = index + static_cast<int32_t>(run.text.size());
    if (paragraph.stacks[i] != nullptr) {
      shapeInto(glyphs, text, index, next, *paragraph.stacks[i],
                run.letterSpacing != 0 ? run.letterSpacing : spec.letterSpacing,
                static_cast<uint16_t>(i), levels);
    }
    index = next;
  }
  if (glyphs.empty()) {
    return out;
  }

  auto lines = breakLines(glyphs, spec.wrapWidth, index);
  // The last run still on screen after the trim, so a span on a dropped line cannot colour it.
  size_t kept = spec.maxLines > 0 ? std::min(lines.size(), static_cast<size_t>(spec.maxLines))
                                  : lines.size();
  size_t tailRun = lastUsableRun(paragraph);
  for (size_t i = kept; i > 0; i--) {
    if (!lines[i - 1].glyphs.empty()) {
      tailRun = lines[i - 1].glyphs.back().run;
      break;
    }
  }
  if (paragraph.stacks[tailRun] == nullptr) {
    tailRun = lastUsableRun(paragraph);
  }
  truncate(lines, spec, *paragraph.stacks[tailRun], static_cast<uint16_t>(tailRun));
  // UAX#9 L1 and L2, per line: trailing whitespace goes back to the paragraph's
  // own direction, and then the line is reordered from logical into visual.
  // Levels mean nothing before the line is known, which is why it happens here.
  for (auto& line : lines) {
    reorderLine(line.glyphs, rtl ? 1 : 0);
  }

  float widest = 0;
  for (const auto& line : lines) {
    widest = std::max(widest, line.width);
  }
  float box = spec.wrapWidth > 0 ? spec.wrapWidth : widest;
  out.metrics.width = widest;
  out.metrics.lineCount = static_cast<float>(lines.size());
  out.metrics.lines.reserve(lines.size());

  float baseline = 0;
  for (size_t i = 0; i < lines.size(); i++) {
    const auto& line = lines[i];
    // Per line, over the runs on it: mixed sizes lead with the tallest face, not the first.
    float ascent = 0;
    float descent = 0;
    float leading = 0;
    if (line.glyphs.empty()) {
      ascent = paragraph.metrics[0].ascent;
      descent = paragraph.metrics[0].descent;
      leading = paragraph.metrics[0].leading;
    }
    for (const auto& glyph : line.glyphs) {
      const auto& m = paragraph.metrics[glyph.run];
      ascent = std::min(ascent, m.ascent);
      descent = std::max(descent, m.descent);
      leading = std::max(leading, m.leading);
    }
    float step = spec.lineHeight > 0 ? spec.lineHeight : descent - ascent + leading;
    if (i > 0) {
      baseline += step;
    }
    float x = spec.align == TextAlign::Center  ? (box - line.width) / 2
              : spec.align == TextAlign::Right ? box - line.width
                                               : 0;
    LineMetrics metrics;
    metrics.width = line.width;
    metrics.x = x;
    metrics.baseline = baseline;
    metrics.ascent = ascent;
    metrics.descent = descent;
    // Min and max, not front and back: the line is in visual order by now, so
    // on a right-to-left line those two are the logically last and first.
    metrics.startIndex = line.startIndex;
    metrics.endIndex = line.startIndex;
    for (const auto& glyph : line.glyphs) {
      metrics.startIndex = std::min(metrics.startIndex, glyph.index);
      metrics.endIndex = std::max(metrics.endIndex, glyph.index);
    }
    out.metrics.lines.push_back(metrics);
    if (i == 0) {
      out.metrics.ascent = ascent;
    }
    out.metrics.descent = descent;

    for (const auto& glyph : line.glyphs) {
      if (glyph.code != '\n') {
        out.glyphs.push_back(glyph);
        out.positions.push_back({x + glyph.offsetX, baseline + glyph.offsetY});
        out.lines.push_back(static_cast<int32_t>(i));
      }
      x += glyph.advance;
    }
  }
  out.metrics.height = baseline + (out.metrics.descent - out.metrics.ascent);
  return out;
}

}  // namespace

TextLayout layoutText(const std::vector<TextRun>& runs, const TextSpec& spec, float size) {
  TextLayout layout;
  auto paragraph = buildParagraph(runs, spec, size);
  if (paragraph.empty) {
    return layout;
  }
  auto placement = place(runs, spec, paragraph);
  layout.metrics = std::move(placement.metrics);
  if (!placement.glyphs.empty()) {
    layout.runs = buildBlobs(placement.glyphs, placement.positions, runs, paragraph.stacks);
  }
  return layout;
}

TextPosition positionAtCoordinate(const std::vector<TextRun>& runs, const TextSpec& spec,
                                  float size, float x, float y) {
  TextPosition out;
  auto paragraph = buildParagraph(runs, spec, size);
  if (paragraph.empty) {
    return out;
  }
  auto placement = place(runs, spec, paragraph);
  const auto& lines = placement.metrics.lines;
  if (lines.empty()) {
    return out;
  }
  // The line whose band holds y, clamped: a point above the first line or below
  // the last one still resolves rather than reporting nothing. The band runs to
  // where the next line begins, not to where this one's ink ends — an explicit
  // lineHeight opens a gap between the two, and the gap belongs to the line
  // above it, not below.
  size_t line = lines.size() - 1;
  for (size_t i = 0; i + 1 < lines.size(); i++) {
    if (y < lines[i + 1].baseline + lines[i + 1].ascent) {
      line = i;
      break;
    }
  }
  out.line = static_cast<int32_t>(line);
  // Every glyph on that line, in order. Empty lines keep the caret at their own
  // start offset, which is the only offset they have.
  const Glyph* last = nullptr;
  bool found = false;
  for (size_t i = 0; i < placement.glyphs.size(); i++) {
    if (placement.lines[i] != out.line) {
      continue;
    }
    const auto& glyph = placement.glyphs[i];
    float left = placement.positions[i].x;
    // Which side of a glyph is its *leading* edge depends on the direction it
    // reads in: for right-to-left text the caret goes before the character on
    // its right, not on its left.
    bool rtl = (glyph.level & 1) != 0;
    if (x < left + glyph.advance / 2) {
      out.index = rtl ? glyph.index + glyph.length : glyph.index;
      out.leading = !rtl;
      found = true;
      break;
    }
    if (x < left + glyph.advance) {
      out.index = rtl ? glyph.index : glyph.index + glyph.length;
      out.leading = rtl;
      found = true;
      break;
    }
    last = &glyph;
  }
  if (!found) {
    // Past the rightmost glyph on the line, so the caret goes beyond it — which
    // in right-to-left text is the *start* of what that glyph covers. An
    // ellipsis carries no bytes, so this lands past the last real glyph.
    if (last == nullptr) {
      out.index = lines[line].startIndex;
      out.leading = true;
    } else {
      out.index = (last->level & 1) != 0 ? last->index : last->index + last->length;
      out.leading = (last->level & 1) != 0;
    }
  }
  return out;
}

std::vector<TextBox> boxesForRange(const std::vector<TextRun>& runs, const TextSpec& spec,
                                   float size, int32_t start, int32_t end) {
  std::vector<TextBox> boxes;
  if (start >= end) {
    return boxes;
  }
  auto paragraph = buildParagraph(runs, spec, size);
  if (paragraph.empty) {
    return boxes;
  }
  auto placement = place(runs, spec, paragraph);
  const auto& lines = placement.metrics.lines;
  // One box per contiguous stretch of the range on one line and in one
  // direction. A direction flip ends a box as surely as a line does: the
  // selected bytes are contiguous but what they cover on screen is not, and one
  // box across the flip would highlight the text in between.
  const Glyph* last = nullptr;
  for (size_t i = 0; i < placement.glyphs.size(); i++) {
    const auto& glyph = placement.glyphs[i];
    // An ellipsis owns no bytes and stands for text that was dropped, so it is
    // never part of a range: without this it straddles every range around its
    // offset and both widens the box and drags its end index backwards.
    if (glyph.length <= 0) {
      continue;
    }
    if (glyph.index >= end || glyph.index + glyph.length <= start) {
      continue;
    }
    int32_t line = placement.lines[i];
    float left = placement.positions[i].x;
    float right = left + glyph.advance;
    bool joins = !boxes.empty() && boxes.back().line == line && last != nullptr &&
                 ((last->level ^ glyph.level) & 1) == 0;
    last = &glyph;
    if (joins) {
      auto& box = boxes.back();
      // Min and max, not first and last: a right-to-left stretch runs backwards
      // through the bytes while its rectangle still grows to the right.
      box.width = std::max(box.width, right - box.x);
      box.startIndex = std::min(box.startIndex, glyph.index);
      box.endIndex = std::max(box.endIndex, glyph.index + glyph.length);
      continue;
    }
    const auto& metrics = lines[static_cast<size_t>(line)];
    TextBox box;
    box.x = left;
    box.y = metrics.baseline + metrics.ascent;
    box.width = right - left;
    box.height = metrics.descent - metrics.ascent;
    box.line = line;
    box.startIndex = glyph.index;
    box.endIndex = glyph.index + glyph.length;
    boxes.push_back(box);
  }
  return boxes;
}

std::vector<double> glyphIdsFor(const std::string& text, const TextSpec& spec, float size) {
  std::vector<double> ids;
  auto stack = buildStack(spec.family, spec.weight, spec.italic, size, spec.fauxBold,
                          spec.fauxItalic);
  if (stack == nullptr) {
    return ids;
  }
  const char* cursor = text.data();
  const char* end = cursor + text.size();
  while (cursor < end) {
    auto code = tgfx::UTF::NextUTF8(&cursor, end);
    if (code < 0) {
      break;
    }
    ids.push_back(stack->fonts[0].getGlyphID(code));
  }
  return ids;
}

TextLayout layoutGlyphs(const std::vector<PlacedGlyph>& glyphs, const TextSpec& spec,
                        float size) {
  TextLayout layout;
  auto stack = buildStack(spec.family, spec.weight, spec.italic, size, spec.fauxBold,
                          spec.fauxItalic);
  if (stack == nullptr || glyphs.empty()) {
    return layout;
  }
  const auto& font = stack->fonts[0];
  auto metrics = font.getMetrics();
  layout.metrics.ascent = metrics.ascent;
  layout.metrics.descent = metrics.descent;
  layout.metrics.lineCount = 1;
  layout.metrics.height = metrics.descent - metrics.ascent;

  tgfx::TextBlobBuilder builder;
  const auto& buffer = builder.allocRunPos(font, glyphs.size());
  if (buffer.glyphs == nullptr || buffer.positions == nullptr) {
    return layout;
  }
  auto* points = reinterpret_cast<tgfx::Point*>(buffer.positions);
  float right = 0;
  for (size_t i = 0; i < glyphs.size(); i++) {
    buffer.glyphs[i] = glyphs[i].id;
    points[i] = {glyphs[i].x, glyphs[i].y};
    right = std::max(right, glyphs[i].x + font.getAdvance(glyphs[i].id));
  }
  auto blob = builder.build();
  if (blob == nullptr) {
    return layout;
  }
  layout.metrics.width = right;
  LineMetrics line;
  line.width = right;
  line.ascent = metrics.ascent;
  line.descent = metrics.descent;
  layout.metrics.lines.push_back(line);
  layout.runs.push_back({std::move(blob), false, 0});
  return layout;
}

tgfx::Path textToPath(const std::vector<TextRun>& runs, const TextSpec& spec, float size) {
  tgfx::Path path;
  auto paragraph = buildParagraph(runs, spec, size);
  if (paragraph.empty) {
    return path;
  }
  auto placement = place(runs, spec, paragraph);
  for (size_t i = 0; i < placement.glyphs.size(); i++) {
    const auto& glyph = placement.glyphs[i];
    tgfx::Path outline;
    // False for a bitmap-only face — a colour emoji has no outline to give.
    if (!paragraph.stacks[glyph.run]->fonts[glyph.font].getPath(glyph.id, &outline)) {
      continue;
    }
    outline.transform(tgfx::Matrix::MakeTrans(placement.positions[i].x, placement.positions[i].y));
    path.addPath(outline);
  }
  return path;
}

TextLayout layoutTextOnPath(const std::vector<TextRun>& runs, const TextSpec& spec,
                            float size, const tgfx::Path& path) {
  TextLayout layout;
  auto paragraph = buildParagraph(runs, spec, size);
  if (paragraph.empty || path.isEmpty()) {
    return layout;
  }
  auto measure = tgfx::PathMeasure::MakeFrom(path);
  if (measure == nullptr) {
    return layout;
  }
  float length = measure->getLength();

  std::vector<Glyph> glyphs;
  auto text = paragraphText(runs);
  auto levels = resolveParagraph(text);
  bool rtl = levels.rtl;
  int32_t index = 0;
  for (size_t i = 0; i < runs.size(); i++) {
    const auto& run = runs[i];
    auto next = index + static_cast<int32_t>(run.text.size());
    if (paragraph.stacks[i] != nullptr) {
      shapeInto(glyphs, text, index, next, *paragraph.stacks[i],
                run.letterSpacing != 0 ? run.letterSpacing : spec.letterSpacing,
                static_cast<uint16_t>(i), levels);
    }
    index = next;
  }
  if (glyphs.empty()) {
    return layout;
  }
  // Text on a path is one line, so it is reordered here for the same reason
  // place() does it: without this, right-to-left text runs along the path
  // backwards even though every glyph is correctly joined.
  reorderLine(glyphs, rtl ? 1 : 0);

  float ascent = 0;
  float descent = 0;
  for (const auto& glyph : glyphs) {
    ascent = std::min(ascent, paragraph.metrics[glyph.run].ascent);
    descent = std::max(descent, paragraph.metrics[glyph.run].descent);
  }
  layout.metrics.ascent = ascent;
  layout.metrics.descent = descent;
  layout.metrics.lineCount = 1;
  layout.metrics.height = descent - ascent;

  auto paintOf = [&](const Glyph& glyph) {
    const auto& run = runs[glyph.run];
    return std::pair<bool, uint32_t>{run.hasColor, run.color};
  };
  float distance = spec.startOffset;
  size_t group = 0;
  while (group < glyphs.size()) {
    size_t groupEnd = group;
    while (groupEnd < glyphs.size() && paintOf(glyphs[groupEnd]) == paintOf(glyphs[group])) {
      groupEnd++;
    }
    tgfx::TextBlobBuilder builder;
    size_t run = group;
    while (run < groupEnd) {
      size_t end = run;
      while (end < groupEnd && glyphs[end].font == glyphs[run].font &&
             glyphs[end].run == glyphs[run].run) {
        end++;
      }
      size_t count = end - run;
      const auto& stack = *paragraph.stacks[glyphs[run].run];
      const auto& buffer = builder.allocRunRSXform(stack.fonts[glyphs[run].font], count);
      if (buffer.glyphs == nullptr || buffer.positions == nullptr) {
        return layout;
      }
      auto* xforms = reinterpret_cast<tgfx::RSXform*>(buffer.positions);
      for (size_t i = 0; i < count; i++) {
        const auto& glyph = glyphs[run + i];
        buffer.glyphs[i] = glyph.id;
        // Anchored at its midpoint, so a glyph sits centred on a tight bend rather than pivoting.
        float half = glyph.advance / 2;
        tgfx::Point position = {};
        tgfx::Point tangent = {1, 0};
        // Along the path by the shaper's own offset, so a combining mark sits
        // where GPOS put it rather than at its base's origin.
        measure->getPosTan(std::clamp(distance + half + glyph.offsetX, 0.0f, length), &position,
                           &tangent);
        xforms[i] = tgfx::RSXform::MakeFromRadians(1, std::atan2(tangent.y, tangent.x), position.x,
                                                   position.y, half, -glyph.offsetY);
        distance += glyph.advance;
      }
      run = end;
    }
    auto blob = builder.build();
    if (blob != nullptr) {
      auto paint = paintOf(glyphs[group]);
      layout.runs.push_back({std::move(blob), paint.first, paint.second});
    }
    group = groupEnd;
  }
  layout.metrics.width = distance - spec.startOffset;
  LineMetrics line;
  line.width = layout.metrics.width;
  line.ascent = ascent;
  line.descent = descent;
  line.endIndex = index;
  layout.metrics.lines.push_back(line);
  return layout;
}

}  // namespace tgfxrn
