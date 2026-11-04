#include "FontRegistry.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>

#include <dirent.h>

#include "ImageRegistry.hpp"
#include "Log.hpp"
#include "tgfx/core/Data.h"
#include "tgfx/core/FontStyle.h"
#include "tgfx/core/Typeface.h"

namespace tgfxrn {
namespace {

/// Lowercased and stripped, so "Playfair Display" and "playfair-display" are one key.
std::string fold(const std::string& name) {
  std::string out;
  out.reserve(name.size());
  for (unsigned char c : name) {
    if (std::isalnum(c) != 0) {
      out.push_back(static_cast<char>(std::tolower(c)));
    }
  }
  return out;
}

std::string foldedStem(const std::string& path) {
  auto slash = path.find_last_of('/');
  auto name = slash == std::string::npos ? path : path.substr(slash + 1);
  auto dot = name.find_last_of('.');
  return fold(dot == std::string::npos ? name : name.substr(0, dot));
}

bool contains(const std::string& haystack, const char* needle) {
  return haystack.find(needle) != std::string::npos;
}

constexpr uint32_t tableTag(const char (&tag)[5]) {
  return (static_cast<uint32_t>(tag[0]) << 24) | (static_cast<uint32_t>(tag[1]) << 16) |
         (static_cast<uint32_t>(tag[2]) << 8) | static_cast<uint32_t>(tag[3]);
}

uint16_t beU16(const uint8_t* bytes) {
  return static_cast<uint16_t>((bytes[0] << 8) | bytes[1]);
}

/// Where a font file can live. Android first, then the desktop hosts. No file names by design.
constexpr std::array<const char*, 6> kFontDirs = {
    "/system/fonts",
    "/product/fonts",
    "/system/font",
    "/data/fonts",
    "/System/Library/Fonts",
    "/Library/Fonts",
};

bool isFontFile(const std::string& name) {
  auto dot = name.find_last_of('.');
  if (dot == std::string::npos) {
    return false;
  }
  std::string ext = fold(name.substr(dot));
  return ext == "ttf" || ext == "otf" || ext == "ttc" || ext == "otc";
}

/// Walked once per process: readdir is cheap, but not per <Text>.
const std::vector<std::string>& systemFontFiles();

std::vector<std::string> scanFontDirs() {
  std::vector<std::string> files;
  for (const char* dir : kFontDirs) {
    DIR* handle = opendir(dir);
    if (handle == nullptr) {
      continue;
    }
    while (auto* entry = readdir(handle)) {
      std::string name = entry->d_name;
      if (isFontFile(name)) {
        files.push_back(std::string(dir) + "/" + name);
      }
    }
    closedir(handle);
  }
  std::sort(files.begin(), files.end());
  return files;
}

const std::vector<std::string>& systemFontFiles() {
  static const std::vector<std::string> files = scanFontDirs();
  return files;
}

/// Families to try when the app asked for none or for one that is not installed. Folded substrings.
constexpr std::array<const char*, 5> kDefaultFamilies = {"roboto", "notosans", "droidsans",
                                                         "dejavusans", "sans"};

/// Fonts that must never answer a plain text request, or a paragraph renders in emoji.
bool isSpecialPurpose(const std::string& stem) {
  return contains(stem, "emoji") || contains(stem, "symbol") || contains(stem, "icon") ||
         contains(stem, "material") || contains(stem, "dingbat") || contains(stem, "awesome");
}

/// "Resolved" has to mean "can draw text": CoreText answers an empty descriptor with a symbol face.
bool canDrawText(const std::shared_ptr<tgfx::Typeface>& typeface) {
  return typeface != nullptr && typeface->getGlyphID('A') != 0 &&
         typeface->getGlyphID('0') != 0;
}

/// Asked for by name: Apple answers the first of these, FreeType platforms fall through to the scan.
constexpr std::array<const char*, 4> kSystemDefaults = {"Helvetica", "Arial", "sans-serif",
                                                        "Roboto"};

}  // namespace

FaceStyle readFaceStyle(const std::shared_ptr<tgfx::Typeface>& typeface) {
  FaceStyle style;
  if (typeface == nullptr) {
    return style;
  }
  // OS/2: usWeightClass at byte 4, fsSelection at byte 62 (bit 0 italic, 5 bold, 9 oblique).
  auto os2 = typeface->copyTableData(tableTag("OS/2"));
  if (os2 != nullptr && os2->size() >= 64) {
    const auto* bytes = static_cast<const uint8_t*>(os2->data());
    auto weight = beU16(bytes + 4);
    // Some old fonts store 1-9 instead of 100-900.
    if (weight > 0 && weight <= 9) {
      weight = static_cast<uint16_t>(weight * 100);
    }
    if (weight >= 100 && weight <= 1000) {
      style.weight = weight;
    }
    auto selection = beU16(bytes + 62);
    style.italic = (selection & 0x01) != 0 || (selection & 0x200) != 0;
    if ((selection & 0x20) != 0 && style.weight < kWeightBold) {
      style.weight = kWeightBold;
    }
    return style;
  }
  // head.macStyle at byte 44: bit 0 bold, bit 1 italic.
  auto head = typeface->copyTableData(tableTag("head"));
  if (head != nullptr && head->size() >= 46) {
    auto macStyle = beU16(static_cast<const uint8_t*>(head->data()) + 44);
    style.italic = (macStyle & 0x02) != 0;
    if ((macStyle & 0x01) != 0) {
      style.weight = kWeightBold;
    }
  }
  return style;
}

std::vector<int> rankFontFiles(const std::vector<std::string>& files, const std::string& family,
                               int weight, bool italic) {
  auto wanted = fold(family);
  bool wantBold = weight >= 600;
  // Only answers a request that named one: "Noto" prefix-matches NotoColorEmoji otherwise.
  bool wantsSpecial = !wanted.empty() && isSpecialPurpose(wanted);
  std::vector<std::pair<int, int>> scored;  // (score, index)
  for (size_t i = 0; i < files.size(); i++) {
    auto stem = foldedStem(files[i]);
    if (isSpecialPurpose(stem) && !wantsSpecial) {
      continue;
    }
    int score = 0;
    if (!wanted.empty()) {
      if (stem.rfind(wanted, 0) == 0) {
        score = 1000;
      } else if (contains(stem, wanted.c_str())) {
        score = 500;
      }
    }
    if (score == 0) {
      for (size_t d = 0; d < kDefaultFamilies.size(); d++) {
        if (contains(stem, kDefaultFamilies[d])) {
          score = static_cast<int>(kDefaultFamilies.size() - d) * 10;
          break;
        }
      }
      if (score == 0) {
        score = 1;
      }
    }
    bool fileBold = contains(stem, "bold") || contains(stem, "black") || contains(stem, "heavy");
    bool fileItalic = contains(stem, "italic") || contains(stem, "oblique");
    if (fileBold == wantBold) {
      score += 60;
    }
    if (fileItalic == italic) {
      score += 8;
    }
    if (contains(stem, "regular")) {
      score += 4;
    }
    scored.emplace_back(score, static_cast<int>(i));
  }
  std::stable_sort(scored.begin(), scored.end(),
                   [](const auto& a, const auto& b) { return a.first > b.first; });
  std::vector<int> ranked;
  ranked.reserve(scored.size());
  for (const auto& [score, index] : scored) {
    ranked.push_back(index);
  }
  return ranked;
}

FontRegistry& FontRegistry::instance() {
  static FontRegistry registry;
  return registry;
}

std::string FontRegistry::add(std::shared_ptr<tgfx::Typeface> typeface,
                              const std::string& family) {
  if (typeface == nullptr) {
    return {};
  }
  auto name = family.empty() ? typeface->fontFamily() : family;
  if (name.empty()) {
    return {};
  }
  Face face{std::move(typeface), FaceStyle{}};
  face.style = readFaceStyle(face.typeface);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto key = fold(name);
    auto& faces = families_[key];
    if (faces.empty()) {
      familyOrder_.push_back(name);
    }
    auto same = std::find_if(faces.begin(), faces.end(), [&](const Face& f) {
      return f.style.weight == face.style.weight && f.style.italic == face.style.italic;
    });
    if (same != faces.end()) {
      *same = std::move(face);
    } else {
      faces.push_back(std::move(face));
    }
    // Every cached match may now be stale: this family may be the one a scene fell back from.
    matches_.clear();
  }
  epoch_.fetch_add(1, std::memory_order_release);
  // The epoch only changes what a draw decides; loadFont() has no commit behind it to schedule one.
  ImageRegistry::instance().wakeAll();
  return name;
}

FontMatch FontRegistry::match(const std::string& family, int weight, bool italic) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto key = fold(family) + "|" + std::to_string(weight) + "|" + (italic ? "1" : "0");
  auto it = matches_.find(key);
  if (it != matches_.end()) {
    return it->second;
  }
  auto found = resolveLocked(family, weight, italic);
  // ponytail: a flat cap with a full clear, not an LRU — re-resolving costs a name lookup.
  if (matches_.size() >= 64) {
    matches_.clear();
  }
  matches_[key] = found;
  return found;
}

FontMatch FontRegistry::resolveLocked(const std::string& family, int weight, bool italic) {
  FontMatch match;
  auto it = families_.find(fold(family));
  if (it != families_.end() && !it->second.empty()) {
    const Face* best = nullptr;
    int bestCost = 0;
    for (const auto& face : it->second) {
      int cost = std::abs(face.style.weight - weight) + (face.style.italic == italic ? 0 : 10000);
      if (best == nullptr || cost < bestCost) {
        best = &face;
        bestCost = cost;
      }
    }
    match.typeface = best->typeface;
    // Synthesise what the family does not ship. 150 is half a step: 500-as-400 is not worth faking.
    match.fauxBold = weight - best->style.weight >= 150;
    match.fauxItalic = italic && !best->style.italic;
    return match;
  }
  match.typeface = systemTypefaceLocked(family, weight, italic);
  if (match.typeface == nullptr) {
    return match;
  }
  auto style = readFaceStyle(match.typeface);
  match.fauxBold = weight - style.weight >= 150;
  match.fauxItalic = italic && !style.italic;
  return match;
}

std::shared_ptr<tgfx::Typeface> FontRegistry::systemTypefaceLocked(const std::string& family,
                                                                   int weight, bool italic) {
  // Apple resolves here and never reaches the scan; every FreeType platform returns nullptr.
  auto style = tgfx::FontStyle(static_cast<tgfx::FontWeight>(std::clamp(weight, 100, 1000) / 100),
                               tgfx::FontWidth::Normal,
                               italic ? tgfx::FontSlant::Italic : tgfx::FontSlant::Upright);
  if (!family.empty()) {
    if (auto typeface = tgfx::Typeface::MakeFromName(family, style); canDrawText(typeface)) {
      return typeface;
    }
  }
  for (const char* name : kSystemDefaults) {
    if (auto typeface = tgfx::Typeface::MakeFromName(name, style); canDrawText(typeface)) {
      return typeface;
    }
  }
  // Down the ranking until one opens and can draw: NotoSansCham-Bold outranks and has no Latin.
  const auto& files = systemFontFiles();
  for (int index : rankFontFiles(files, family, weight, italic)) {
    const auto& path = files[static_cast<size_t>(index)];
    auto cached = files_.find(path);
    if (cached != files_.end()) {
      if (canDrawText(cached->second)) {
        return cached->second;
      }
      continue;
    }
    auto typeface = tgfx::Typeface::MakeFromPath(path);
    if (!canDrawText(typeface)) {
      continue;
    }
    TGFXRN_LOGI("family '%s' weight %d resolved to %s", family.c_str(), weight, path.c_str());
    files_[path] = typeface;
    return typeface;
  }
  TGFXRN_LOGW(
      "no typeface for family '%s': nothing registered, no system match, and no font file that "
      "can draw Latin in any of the platform font directories",
      family.c_str());
  return nullptr;
}

std::vector<std::shared_ptr<tgfx::Typeface>> FontRegistry::fallbacks() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (fallbacksResolved_) {
    return fallbacks_;
  }
  fallbacksResolved_ = true;
  // Apple names it; Android ships a file whose name contains "emoji", which the scan finds.
  for (const char* name : {"Apple Color Emoji", "Segoe UI Emoji"}) {
    if (auto typeface = tgfx::Typeface::MakeFromName(name, "")) {
      fallbacks_.push_back(std::move(typeface));
      return fallbacks_;
    }
  }
  for (const auto& path : systemFontFiles()) {
    if (!contains(foldedStem(path), "emoji")) {
      continue;
    }
    auto cached = files_.find(path);
    auto typeface = cached != files_.end() ? cached->second : tgfx::Typeface::MakeFromPath(path);
    if (typeface != nullptr) {
      files_[path] = typeface;
      fallbacks_.push_back(std::move(typeface));
      break;
    }
  }
  return fallbacks_;
}

std::vector<std::string> FontRegistry::familyNames() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return familyOrder_;
}

void FontRegistry::clear() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    families_.clear();
    familyOrder_.clear();
    matches_.clear();
    files_.clear();
    fallbacks_.clear();
    fallbacksResolved_ = false;
  }
  epoch_.fetch_add(1, std::memory_order_release);
}

}  // namespace tgfxrn
