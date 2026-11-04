#include "TextShaping.hpp"

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "Log.hpp"
#include "hb.h"
#include "tgfx/core/Data.h"
#include "tgfx/core/Typeface.h"
#include "tgfx/core/UTF.h"

namespace tgfxrn {
namespace {

/// The bidi classes the resolver needs. UAX#9 has more; the ones left out are
/// the explicit embedding, override and isolate controls, which are treated as
/// BN — see the note on shapeText().
enum class Bidi : uint8_t { L,
                            R,
                            AL,
                            EN,
                            AN,
                            ES,
                            ET,
                            CS,
                            NSM,
                            BN,
                            B,
                            S,
                            WS,
                            ON };

bool isStrong(Bidi c) {
  return c == Bidi::L || c == Bidi::R || c == Bidi::AL;
}

/// A neutral or isolate for rules N1/N2. The isolate classes are not produced.
bool isNeutral(Bidi c) {
  return c == Bidi::B || c == Bidi::S || c == Bidi::WS || c == Bidi::ON;
}

/// The right-to-left scripts that join, which UAX#9 calls AL rather than R.
/// The distinction matters for W2 and W3, so European digits after Arabic come
/// out as Arabic numbers.
bool isArabicLike(hb_script_t script) {
  switch (script) {
    case HB_SCRIPT_ARABIC:
    case HB_SCRIPT_SYRIAC:
    case HB_SCRIPT_THAANA:
    case HB_SCRIPT_NKO:
    case HB_SCRIPT_SAMARITAN:
    case HB_SCRIPT_MANDAIC:
    case HB_SCRIPT_HANIFI_ROHINGYA:
    case HB_SCRIPT_SOGDIAN:
    case HB_SCRIPT_OLD_SOGDIAN:
    case HB_SCRIPT_ADLAM:
      return true;
    default:
      return false;
  }
}

Bidi classOf(hb_codepoint_t code, hb_unicode_funcs_t* unicode) {
  // The three marks whose whole job is to be strong. They are Cf, so the
  // category switch below would call them formatting and they would do nothing
  // — and they are the only tool an app has to place a neutral itself.
  if (code == 0x200E) {
    return Bidi::L;
  }
  if (code == 0x200F) {
    return Bidi::R;
  }
  if (code == 0x061C) {
    return Bidi::AL;
  }
  // Paragraph and segment separators next: they are Cc, which would otherwise
  // fall through to BN and stop resolving lines.
  if (code == 0x000A || code == 0x000D || code == 0x001C || code == 0x001D ||
      code == 0x001E || code == 0x0085 || code == 0x2029) {
    return Bidi::B;
  }
  if (code == 0x0009 || code == 0x000B || code == 0x001F) {
    return Bidi::S;
  }
  if (code == 0x000C || code == 0x0020 || code == 0x1680 || code == 0x2028 ||
      (code >= 0x2000 && code <= 0x200A) || code == 0x205F || code == 0x3000) {
    return Bidi::WS;
  }
  // Arabic numbers, which are Nd or Cf and would otherwise read as EN or BN.
  if ((code >= 0x0600 && code <= 0x0605) || (code >= 0x0660 && code <= 0x0669) ||
      (code >= 0x066B && code <= 0x066C) || code == 0x06DD || code == 0x08E2 ||
      (code >= 0x10D30 && code <= 0x10D39)) {
    return Bidi::AN;
  }
  // The separators and terminators W4-W6 work on. Kept to the set that actually
  // appears between digits, rather than every character Unicode gives the class.
  if (code == '+' || code == '-' || code == 0x2212) {
    return Bidi::ES;
  }
  if (code == '#' || code == '$' || code == '%' || code == 0x00A2 || code == 0x00A3 ||
      code == 0x00A4 || code == 0x00A5 || code == 0x00B0 || code == 0x2030 ||
      (code >= 0x20A0 && code <= 0x20CF)) {
    return Bidi::ET;
  }
  if (code == ',' || code == '.' || code == ':' || code == '/' || code == 0x00A0 ||
      code == 0x060C || code == 0x202F || code == 0x2044 || code == 0xFE50) {
    return Bidi::CS;
  }

  auto category = hb_unicode_general_category(unicode, code);
  switch (category) {
    case HB_UNICODE_GENERAL_CATEGORY_NON_SPACING_MARK:
    case HB_UNICODE_GENERAL_CATEGORY_ENCLOSING_MARK:
      return Bidi::NSM;
    case HB_UNICODE_GENERAL_CATEGORY_FORMAT:
    case HB_UNICODE_GENERAL_CATEGORY_CONTROL:
    case HB_UNICODE_GENERAL_CATEGORY_UNASSIGNED:
    case HB_UNICODE_GENERAL_CATEGORY_SURROGATE:
      return Bidi::BN;
    case HB_UNICODE_GENERAL_CATEGORY_SPACE_SEPARATOR:
      return Bidi::WS;
    case HB_UNICODE_GENERAL_CATEGORY_DECIMAL_NUMBER:
      // Only these are European numbers. Devanagari, Bengali and Thai digits
      // are plain L, and calling them EN would drag them into a neighbouring
      // Arabic run as Arabic numbers.
      if (code <= 0x0039 || (code >= 0x06F0 && code <= 0x06F9) ||
          (code >= 0xFF10 && code <= 0xFF19)) {
        return Bidi::EN;
      }
      break;
    default:
      break;
  }

  auto script = hb_unicode_script(unicode, code);
  if (script == HB_SCRIPT_COMMON || script == HB_SCRIPT_INHERITED ||
      script == HB_SCRIPT_UNKNOWN) {
    // Script says nothing here, so the category does: letters are strong and
    // everything else is a neutral the resolver places from its surroundings.
    switch (category) {
      case HB_UNICODE_GENERAL_CATEGORY_LOWERCASE_LETTER:
      case HB_UNICODE_GENERAL_CATEGORY_MODIFIER_LETTER:
      case HB_UNICODE_GENERAL_CATEGORY_OTHER_LETTER:
      case HB_UNICODE_GENERAL_CATEGORY_TITLECASE_LETTER:
      case HB_UNICODE_GENERAL_CATEGORY_UPPERCASE_LETTER:
        return Bidi::L;
      default:
        return Bidi::ON;
    }
  }
  if (hb_script_get_horizontal_direction(script) == HB_DIRECTION_RTL) {
    return isArabicLike(script) ? Bidi::AL : Bidi::R;
  }
  return Bidi::L;
}

struct Char {
  hb_codepoint_t code = 0;
  int32_t index = 0;
  int32_t length = 0;
  Bidi original = Bidi::L;
  Bidi resolved = Bidi::L;
  uint8_t level = 0;
  hb_script_t script = HB_SCRIPT_COMMON;
};

std::vector<Char> decode(const std::string& text, hb_unicode_funcs_t* unicode) {
  std::vector<Char> out;
  const char* start = text.data();
  const char* cursor = start;
  const char* end = cursor + text.size();
  while (cursor < end) {
    auto at = static_cast<int32_t>(cursor - start);
    auto code = tgfx::UTF::NextUTF8(&cursor, end);
    if (code < 0) {
      break;
    }
    Char c;
    c.code = static_cast<hb_codepoint_t>(code);
    c.index = at;
    c.length = static_cast<int32_t>(cursor - start) - at;
    c.original = classOf(c.code, unicode);
    c.resolved = c.original;
    c.script = hb_unicode_script(unicode, c.code);
    out.push_back(c);
  }
  // A script of Common or Inherited belongs to whatever run it sits in, so it
  // never splits one — "abc." shapes as one run, not two. The forward pass
  // gives each one the script before it, which leaves only the ones at the very
  // start still Common; the reverse pass then gives those the script after
  // them. A string with no real script anywhere stays Common throughout.
  hb_script_t last = HB_SCRIPT_COMMON;
  for (auto& c : out) {
    if (c.script == HB_SCRIPT_COMMON || c.script == HB_SCRIPT_INHERITED ||
        c.script == HB_SCRIPT_UNKNOWN) {
      c.script = last;
    } else {
      last = c.script;
    }
  }
  for (auto it = out.rbegin(); it != out.rend(); ++it) {
    if (it->script == HB_SCRIPT_COMMON || it->script == HB_SCRIPT_INHERITED ||
        it->script == HB_SCRIPT_UNKNOWN) {
      it->script = last;
    } else {
      last = it->script;
    }
  }
  return out;
}

/// UAX#9 W1-W7, N1-N2 and I1-I2, over the whole paragraph as one level run.
/// Without the explicit codes there is exactly one run, so sos and eos are both
/// the paragraph direction.
void resolveLevels(std::vector<Char>& chars, uint8_t paragraphLevel) {
  if (chars.empty()) {
    return;
  }
  Bidi sos = paragraphLevel == 0 ? Bidi::L : Bidi::R;

  // W1: a non-spacing mark takes the class of what it attaches to. X9 makes a
  // format character invisible, so a mark behind one attaches to whatever came
  // before *it*, not to the format character.
  Bidi previous = sos;
  for (auto& c : chars) {
    if (c.resolved == Bidi::BN) {
      continue;
    }
    if (c.resolved == Bidi::NSM) {
      c.resolved = previous;
    }
    previous = c.resolved;
  }
  // W2: a European digit after an Arabic letter is an Arabic number.
  Bidi strong = sos;
  for (auto& c : chars) {
    if (isStrong(c.resolved)) {
      strong = c.resolved;
    } else if (c.resolved == Bidi::EN && strong == Bidi::AL) {
      c.resolved = Bidi::AN;
    }
  }
  // W3: Arabic letters are just right-to-left from here on.
  for (auto& c : chars) {
    if (c.resolved == Bidi::AL) {
      c.resolved = Bidi::R;
    }
  }
  // The rules below read "the character before" and "the character after",
  // which X9 makes mean the nearest one that is not a format character.
  std::vector<size_t> visible;
  visible.reserve(chars.size());
  for (size_t i = 0; i < chars.size(); i++) {
    if (chars[i].resolved != Bidi::BN) {
      visible.push_back(i);
    }
  }
  auto at = [&](size_t k) -> Bidi { return chars[visible[k]].resolved; };

  // W4: one separator between two numbers of the same kind joins them.
  for (size_t k = 1; k + 1 < visible.size(); k++) {
    auto here = at(k);
    auto before = at(k - 1);
    auto after = at(k + 1);
    if (here == Bidi::ES && before == Bidi::EN && after == Bidi::EN) {
      chars[visible[k]].resolved = Bidi::EN;
    } else if (here == Bidi::CS && before == after &&
               (before == Bidi::EN || before == Bidi::AN)) {
      chars[visible[k]].resolved = before;
    }
  }
  // W5: a run of terminators next to a European number joins it.
  for (size_t k = 0; k < visible.size(); k++) {
    if (at(k) != Bidi::ET) {
      continue;
    }
    size_t end = k;
    while (end < visible.size() && at(end) == Bidi::ET) {
      end++;
    }
    bool adjacent = (k > 0 && at(k - 1) == Bidi::EN) ||
                    (end < visible.size() && at(end) == Bidi::EN);
    if (adjacent) {
      for (size_t j = k; j < end; j++) {
        chars[visible[j]].resolved = Bidi::EN;
      }
    }
    k = end - 1;
  }
  // W6: whatever separators and terminators are left are ordinary neutrals.
  for (auto& c : chars) {
    if (c.resolved == Bidi::ES || c.resolved == Bidi::ET || c.resolved == Bidi::CS) {
      c.resolved = Bidi::ON;
    }
  }
  // W7: a European digit after a left-to-right letter is left-to-right.
  strong = sos;
  for (auto& c : chars) {
    if (c.resolved == Bidi::BN) {
      continue;
    }
    if (c.resolved == Bidi::L || c.resolved == Bidi::R) {
      strong = c.resolved;
    } else if (c.resolved == Bidi::EN && strong == Bidi::L) {
      c.resolved = Bidi::L;
    }
  }

  // N1/N2: a run of neutrals between two of the same direction takes it, and
  // otherwise takes the paragraph's. Numbers count as right-to-left here.
  auto directionOf = [](Bidi c) {
    return (c == Bidi::R || c == Bidi::EN || c == Bidi::AN) ? Bidi::R
           : c == Bidi::L                                   ? Bidi::L
                                                            : Bidi::ON;
  };
  Bidi embedding = paragraphLevel == 0 ? Bidi::L : Bidi::R;
  for (size_t k = 0; k < visible.size(); k++) {
    if (!isNeutral(at(k))) {
      continue;
    }
    size_t end = k;
    while (end < visible.size() && isNeutral(at(end))) {
      end++;
    }
    Bidi before = k > 0 ? directionOf(at(k - 1)) : sos;
    Bidi after = end < visible.size() ? directionOf(at(end)) : embedding;
    Bidi take = before == after ? before : embedding;
    for (size_t j = k; j < end; j++) {
      chars[visible[j]].resolved = take;
    }
    k = end - 1;
  }

  // I1/I2: levels. Inside a left-to-right run, right-to-left text goes up one
  // and numbers two; inside a right-to-left run everything else goes up one.
  for (auto& c : chars) {
    if (paragraphLevel == 0) {
      c.level = c.resolved == Bidi::R                                ? 1
                : (c.resolved == Bidi::EN || c.resolved == Bidi::AN) ? 2
                                                                     : 0;
    } else {
      c.level = c.resolved == Bidi::R ? 1 : 2;
    }
  }
  // X9 again: a format character has no direction of its own, so it takes the
  // level of what precedes it rather than a level of its own that could split
  // a number run in two and reverse the digits.
  uint8_t carried = paragraphLevel;
  for (auto& c : chars) {
    if (c.resolved == Bidi::BN) {
      c.level = carried;
    } else {
      carried = c.level;
    }
  }
}

// --- HarfBuzz faces ------------------------------------------------------

hb_blob_t* tableOf(hb_face_t*, hb_tag_t tag, void* user) {
  // The user data is the shared_ptr the face holds, not the typeface itself:
  // the face reads tables lazily and has to keep it alive to do so.
  const auto& typeface = *static_cast<std::shared_ptr<tgfx::Typeface>*>(user);
  if (typeface == nullptr) {
    return nullptr;
  }
  auto data = typeface->copyTableData(tag);
  if (data == nullptr || data->size() == 0) {
    return nullptr;
  }
  // The blob owns a strong reference, so the table outlives this call whatever
  // the caller does with the typeface.
  auto* held = new std::shared_ptr<tgfx::Data>(std::move(data));
  return hb_blob_create(static_cast<const char*>((*held)->data()),
                        static_cast<unsigned int>((*held)->size()), HB_MEMORY_MODE_READONLY, held,
                        [](void* p) { delete static_cast<std::shared_ptr<tgfx::Data>*>(p); });
}

/// One hb_face_t per typeface, kept for the process: building one re-reads and
/// re-parses every OpenType table, which is far too much to do per layout.
class FaceCache {
 public:
  static FaceCache& instance() {
    static FaceCache cache;
    return cache;
  }

  hb_face_t* get(const std::shared_ptr<tgfx::Typeface>& typeface) {
    if (typeface == nullptr) {
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex);
    auto id = typeface->uniqueID();
    auto found = faces.find(id);
    if (found != faces.end()) {
      return found->second;
    }
    // The face keeps the typeface alive, since it reads tables from it lazily.
    auto* held = new std::shared_ptr<tgfx::Typeface>(typeface);
    auto* face = hb_face_create_for_tables(
        tableOf, held, [](void* p) { delete static_cast<std::shared_ptr<tgfx::Typeface>*>(p); });
    // Sealed here, under the lock: hb_font_create would otherwise be the one to
    // seal it, and two threads reaching a shared face first would both write.
    hb_face_make_immutable(face);
    faces.emplace(id, face);
    return face;
  }

 private:
  std::mutex mutex;
  std::unordered_map<uint32_t, hb_face_t*> faces;
};

/// HarfBuzz reports in 1/64 px once the scale is the size in 64ths.
constexpr float kHbScale = 64.0f;

/// Resolved once. hb_language_get_default() calls setlocale, which HarfBuzz's
/// own source warns is not thread safe in many implementations — and layout
/// runs on the render thread while measureText runs on the JS one.
hb_language_t defaultLanguage() {
  static const hb_language_t language = hb_language_get_default();
  return language;
}

/// Once per codepoint per process, or an uncovered script logs a line per
/// character per size. Shared with TextLayout's own warning for the same thing.
void warnMissingGlyph(int32_t code) {
  static std::mutex mutex;
  static std::unordered_set<int32_t> seen;
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!seen.insert(code).second) {
      return;
    }
  }
  TGFXRN_LOGW(
      "no glyph for U+%04X in the matched face or any fallback; it draws as .notdef, which most "
      "fonts leave blank. Register a font that covers this script.",
      code);
}

}  // namespace

/// The headers are vendored while the code lives in the prebuilt tgfx archive,
/// so the two can drift apart on a tgfx bump. Checked once, and only warned
/// about: a mismatched minor is usually harmless, and refusing to draw text
/// over it would be worse than the risk.
void checkVersionOnce() {
  static const bool once = [] {
    if (std::string(hb_version_string()) != HB_VERSION_STRING) {
      TGFXRN_LOGW(
          "HarfBuzz headers say %s but the linked library is %s. Refresh "
          "cpp/third_party/harfbuzz — see its README.",
          HB_VERSION_STRING, hb_version_string());
    }
    return true;
  }();
  (void)once;
}

BidiLevels resolveParagraph(const std::string& text) {
  BidiLevels out;
  checkVersionOnce();
  auto* unicode = hb_unicode_funcs_get_default();
  auto chars = decode(text, unicode);
  // P2/P3: the first strong character sets the base direction.
  for (const auto& c : chars) {
    if (c.original == Bidi::L) {
      break;
    }
    if (c.original == Bidi::R || c.original == Bidi::AL) {
      out.rtl = true;
      break;
    }
  }
  resolveLevels(chars, out.rtl ? 1 : 0);
  out.byByte.assign(text.size() + 1, out.rtl ? 1 : 0);
  out.whitespaceByByte.assign(text.size() + 1, 0);
  for (const auto& c : chars) {
    for (int32_t i = 0; i < c.length; i++) {
      auto at = static_cast<size_t>(c.index + i);
      if (at < out.byByte.size()) {
        out.byByte[at] = c.level;
        out.whitespaceByByte[at] =
            (c.original == Bidi::WS || c.original == Bidi::S || c.original == Bidi::B) ? 1 : 0;
      }
    }
  }
  return out;
}

std::vector<ShapedGlyph> shapeRun(const std::string& paragraph, int32_t begin, int32_t end,
                                  const std::vector<tgfx::Font>& fonts, float letterSpacing,
                                  const BidiLevels& levels) {
  std::vector<ShapedGlyph> out;
  if (fonts.empty() || begin >= end || paragraph.empty()) {
    return out;
  }
  auto* unicode = hb_unicode_funcs_get_default();
  // Only the run's own characters, but with the levels the whole paragraph
  // resolved: a <Span> seam must not change what its neighbours read as.
  std::vector<Char> chars;
  {
    const char* start = paragraph.data();
    const char* cursor = start + begin;
    const char* stop = start + std::min(static_cast<size_t>(end), paragraph.size());
    while (cursor < stop) {
      auto at = static_cast<int32_t>(cursor - start);
      auto code = tgfx::UTF::NextUTF8(&cursor, stop);
      if (code < 0) {
        break;
      }
      Char c;
      c.code = static_cast<hb_codepoint_t>(code);
      c.index = at;
      c.length = static_cast<int32_t>(cursor - start) - at;
      c.level = levels.levelAt(at);
      c.script = hb_unicode_script(unicode, c.code);
      chars.push_back(c);
    }
  }
  if (chars.empty()) {
    return out;
  }
  // Script inheritance, as decode() does it: a Common or Inherited character
  // joins the run around it rather than splitting one.
  hb_script_t last = HB_SCRIPT_COMMON;
  for (auto& c : chars) {
    if (c.script == HB_SCRIPT_COMMON || c.script == HB_SCRIPT_INHERITED ||
        c.script == HB_SCRIPT_UNKNOWN) {
      c.script = last;
    } else {
      last = c.script;
    }
  }
  for (auto it = chars.rbegin(); it != chars.rend(); ++it) {
    if (it->script == HB_SCRIPT_COMMON || it->script == HB_SCRIPT_INHERITED ||
        it->script == HB_SCRIPT_UNKNOWN) {
      it->script = last;
    } else {
      last = it->script;
    }
  }

  // Which font covers each character, by the same first-hit rule the unshaped
  // path used. A default-ignorable takes its neighbour's, so a zero-width
  // joiner or a mark cannot split a run away from the base it belongs to.
  std::vector<uint8_t> fontOf(chars.size(), 0);
  std::vector<uint8_t> covered(chars.size(), 0);
  for (size_t i = 0; i < chars.size(); i++) {
    auto category = hb_unicode_general_category(unicode, chars[i].code);
    bool ignorable = category == HB_UNICODE_GENERAL_CATEGORY_FORMAT ||
                     category == HB_UNICODE_GENERAL_CATEGORY_NON_SPACING_MARK ||
                     category == HB_UNICODE_GENERAL_CATEGORY_ENCLOSING_MARK;
    for (size_t f = 0; f < fonts.size(); f++) {
      if (fonts[f].getGlyphID(static_cast<tgfx::Unichar>(chars[i].code)) != 0) {
        fontOf[i] = static_cast<uint8_t>(f);
        covered[i] = 1;
        break;
      }
    }
    if (covered[i] == 0) {
      if (ignorable && i > 0) {
        fontOf[i] = fontOf[i - 1];
      } else if (!ignorable) {
        warnMissingGlyph(static_cast<int32_t>(chars[i].code));
      }
    }
  }

  // Byte offset to the character that starts there, so a shaped cluster finds
  // its source in one step rather than scanning every character it might be.
  std::vector<int32_t> charAt(paragraph.size() + 1, -1);
  for (size_t i = 0; i < chars.size(); i++) {
    charAt[static_cast<size_t>(chars[i].index)] = static_cast<int32_t>(i);
  }

  auto* buffer = hb_buffer_create();
  // One hb_font_t per face for the whole run: a script boundary is not a reason
  // to rebuild one, and a run of alternating scripts has many.
  std::unordered_map<uint8_t, hb_font_t*> hbFonts;
  size_t start = 0;
  while (start < chars.size()) {
    size_t stop = start + 1;
    while (stop < chars.size() && chars[stop].level == chars[start].level &&
           chars[stop].script == chars[start].script && fontOf[stop] == fontOf[start]) {
      stop++;
    }
    const auto& font = fonts[fontOf[start]];
    auto* face = FaceCache::instance().get(font.getTypeface());
    bool rtl = (chars[start].level & 1) != 0;
    // A face with no glyphs is one whose tables could not be read — a bitmap
    // typeface, or a user one that has none. Shaping against it silently draws
    // nothing, so the codepoint walk takes over instead.
    if (face == nullptr || hb_face_get_glyph_count(face) == 0) {
      for (size_t i = start; i < stop; i++) {
        ShapedGlyph glyph;
        glyph.id = font.getGlyphID(static_cast<tgfx::Unichar>(chars[i].code));
        glyph.font = fontOf[start];
        glyph.advance = font.getAdvance(glyph.id) + letterSpacing;
        glyph.index = chars[i].index;
        glyph.length = chars[i].length;
        glyph.level = chars[i].level;
        glyph.code = static_cast<int32_t>(chars[i].code);
        glyph.whitespace = levels.whitespaceAt(chars[i].index);
        out.push_back(glyph);
      }
      start = stop;
      continue;
    }
    auto found = hbFonts.find(fontOf[start]);
    if (found == hbFonts.end()) {
      auto* created = hb_font_create(face);
      auto scale = static_cast<int>(font.getSize() * kHbScale);
      hb_font_set_scale(created, scale, scale);
      found = hbFonts.emplace(fontOf[start], created).first;
    }
    auto* hbFont = found->second;
    hb_buffer_clear_contents(buffer);
    // The whole paragraph goes in with the run marked by item_offset, so
    // HarfBuzz can see the context on both sides — which is what decides an
    // Arabic letter's initial, medial or final form at a run boundary.
    hb_buffer_add_utf8(buffer, paragraph.data(), static_cast<int>(paragraph.size()),
                       static_cast<unsigned int>(chars[start].index),
                       static_cast<int>(chars[stop - 1].index + chars[stop - 1].length -
                                        chars[start].index));
    // All three set, not guessed: the run was itemised by level and script
    // above, and guessing would overwrite both from the buffer's contents.
    hb_buffer_set_direction(buffer, rtl ? HB_DIRECTION_RTL : HB_DIRECTION_LTR);
    hb_buffer_set_script(buffer, chars[start].script);
    hb_buffer_set_language(buffer, defaultLanguage());
    hb_shape(hbFont, buffer, nullptr, 0);

    unsigned int count = 0;
    auto* infos = hb_buffer_get_glyph_infos(buffer, &count);
    auto* positions = hb_buffer_get_glyph_positions(buffer, &count);
    // Back to logical order: a right-to-left run comes out visually, and the
    // line breaker and the reorder pass both want logical.
    for (unsigned int k = 0; k < count; k++) {
      unsigned int i = rtl ? count - 1 - k : k;
      ShapedGlyph glyph;
      glyph.id = static_cast<tgfx::GlyphID>(infos[i].codepoint);
      glyph.font = fontOf[start];
      glyph.advance = static_cast<float>(positions[i].x_advance) / kHbScale;
      glyph.offsetX = static_cast<float>(positions[i].x_offset) / kHbScale;
      glyph.offsetY = -static_cast<float>(positions[i].y_offset) / kHbScale;
      glyph.index = static_cast<int32_t>(infos[i].cluster);
      glyph.level = chars[start].level;
      glyph.whitespace = levels.whitespaceAt(glyph.index);
      auto at = static_cast<size_t>(glyph.index);
      if (at < charAt.size() && charAt[at] >= 0) {
        glyph.code = static_cast<int32_t>(chars[static_cast<size_t>(charAt[at])].code);
      }
      out.push_back(glyph);
    }
    // Cluster lengths, from the boundaries this run's own clusters make: each
    // ends where the next begins, and the last at the end of the run. A cluster
    // is not one codepoint — a ligature is one over several.
    {
      std::vector<int32_t> starts;
      starts.reserve(count);
      for (unsigned int i = 0; i < count; i++) {
        starts.push_back(static_cast<int32_t>(infos[i].cluster));
      }
      std::sort(starts.begin(), starts.end());
      starts.erase(std::unique(starts.begin(), starts.end()), starts.end());
      int32_t runEnd = chars[stop - 1].index + chars[stop - 1].length;
      for (size_t g = out.size() - count; g < out.size(); g++) {
        auto next = std::upper_bound(starts.begin(), starts.end(), out[g].index);
        out[g].length = (next == starts.end() ? runEnd : *next) - out[g].index;
      }
      // Letter spacing is a gap between clusters, not between glyphs: a mark
      // has no advance of its own and must not gain one.
      for (size_t g = out.size() - count; g < out.size(); g++) {
        bool lastOfCluster = g + 1 == out.size() || out[g + 1].index != out[g].index;
        if (lastOfCluster) {
          out[g].advance += letterSpacing;
        }
      }
    }
    start = stop;
  }
  hb_buffer_destroy(buffer);
  for (auto& entry : hbFonts) {
    hb_font_destroy(entry.second);
  }
  return out;
}

}  // namespace tgfxrn
