#include "SvgPath.hpp"

#include <algorithm>
#include <cstdio>
#include <string_view>

#include "Log.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>

namespace tgfxrn {

namespace {

struct Cursor {
  const char* p;
  const char* end;

  void skipSep() {
    while (p < end && (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n' ||
                       *p == '\r')) {
      p++;
    }
  }

  float number() {
    skipSep();
    char* next = nullptr;
    float v = std::strtof(p, &next);
    if (next == p) {
      p = end;
      return 0.0f;
    }
    p = next;
    return v;
  }

  // SVG 2 grammar: `flag ::= "0" | "1"`, one character. Reading a number desynchronises the string.
  int flag() {
    skipSep();
    if (p < end && (*p == '0' || *p == '1')) {
      return *p++ - '0';
    }
    p = end;
    return -1;
  }
};

}  // namespace

tgfx::Path parseSvgPath(const std::string& data) {
  tgfx::Path path;
  Cursor cur{data.c_str(), data.c_str() + data.size()};

  float curX = 0, curY = 0;      // current point
  float startX = 0, startY = 0;  // subpath start
  float ctrlX = 0, ctrlY = 0;    // last control point (for S/T reflection)
  char prevCmd = 0;
  char cmd = 0;
  bool tookArgs = false;

  while (true) {
    cur.skipSep();
    if (cur.p >= cur.end) {
      break;
    }
    char c = *cur.p;
    if (std::isalpha(static_cast<unsigned char>(c))) {
      cmd = c;
      cur.p++;
    } else {
      if (!tookArgs) {
        break;
      }
      if (cmd == 'M') {
        cmd = 'L';
      } else if (cmd == 'm') {
        cmd = 'l';
      }
    }

    const char* argsBegin = cur.p;
    bool relative = std::islower(static_cast<unsigned char>(cmd)) != 0;
    switch (std::toupper(static_cast<unsigned char>(cmd))) {
      case 'M': {
        float x = cur.number();
        float y = cur.number();
        if (relative) {
          x += curX;
          y += curY;
        }
        curX = x;
        curY = y;
        startX = x;
        startY = y;
        path.moveTo(x, y);
        break;
      }
      case 'L': {
        float x = cur.number();
        float y = cur.number();
        if (relative) {
          x += curX;
          y += curY;
        }
        curX = x;
        curY = y;
        path.lineTo(x, y);
        break;
      }
      case 'H': {
        float x = cur.number();
        if (relative) {
          x += curX;
        }
        curX = x;
        path.lineTo(curX, curY);
        break;
      }
      case 'V': {
        float y = cur.number();
        if (relative) {
          y += curY;
        }
        curY = y;
        path.lineTo(curX, curY);
        break;
      }
      case 'C': {
        float x1 = cur.number(), y1 = cur.number();
        float x2 = cur.number(), y2 = cur.number();
        float x = cur.number(), y = cur.number();
        if (relative) {
          x1 += curX;
          y1 += curY;
          x2 += curX;
          y2 += curY;
          x += curX;
          y += curY;
        }
        path.cubicTo(x1, y1, x2, y2, x, y);
        ctrlX = x2;
        ctrlY = y2;
        curX = x;
        curY = y;
        break;
      }
      case 'S': {
        float x2 = cur.number(), y2 = cur.number();
        float x = cur.number(), y = cur.number();
        if (relative) {
          x2 += curX;
          y2 += curY;
          x += curX;
          y += curY;
        }
        float x1 = curX, y1 = curY;
        char up = static_cast<char>(std::toupper(static_cast<unsigned char>(prevCmd)));
        if (up == 'C' || up == 'S') {
          x1 = 2 * curX - ctrlX;
          y1 = 2 * curY - ctrlY;
        }
        path.cubicTo(x1, y1, x2, y2, x, y);
        ctrlX = x2;
        ctrlY = y2;
        curX = x;
        curY = y;
        break;
      }
      case 'Q': {
        float x1 = cur.number(), y1 = cur.number();
        float x = cur.number(), y = cur.number();
        if (relative) {
          x1 += curX;
          y1 += curY;
          x += curX;
          y += curY;
        }
        path.quadTo(x1, y1, x, y);
        ctrlX = x1;
        ctrlY = y1;
        curX = x;
        curY = y;
        break;
      }
      case 'T': {
        float x = cur.number(), y = cur.number();
        if (relative) {
          x += curX;
          y += curY;
        }
        float x1 = curX, y1 = curY;
        char up = static_cast<char>(std::toupper(static_cast<unsigned char>(prevCmd)));
        if (up == 'Q' || up == 'T') {
          x1 = 2 * curX - ctrlX;
          y1 = 2 * curY - ctrlY;
        }
        path.quadTo(x1, y1, x, y);
        ctrlX = x1;
        ctrlY = y1;
        curX = x;
        curY = y;
        break;
      }
      case 'A': {
        float rx = cur.number(), ry = cur.number(), rotation = cur.number();
        int largeArc = cur.flag(), sweep = cur.flag();
        float x = cur.number(), y = cur.number();
        if (largeArc < 0 || sweep < 0) {
          TGFXRN_LOGW("svg path: arc flag is not 0 or 1, dropping the rest of the path");
          break;
        }
        if (relative) {
          x += curX;
          y += curY;
        }
        if (path.isEmpty()) {
          path.moveTo(curX, curY);
        }
        // tgfx applies F.6.5, but its `reversed` is the inverse of SVG's sweep.
        path.arcTo(rx, ry, rotation,
                   largeArc != 0 ? tgfx::PathArcSize::Large : tgfx::PathArcSize::Small,
                   sweep == 0, tgfx::Point::Make(x, y));
        curX = x;
        curY = y;
        break;
      }
      case 'Z': {
        path.close();
        curX = startX;
        curY = startY;
        break;
      }
      default:
        cur.p = cur.end;
        break;
    }
    tookArgs = cur.p != argsBegin;
    prevCmd = cmd;
  }
  return path;
}

namespace {

void appendNumber(std::string& out, float value) {
  char buffer[32];
  int written = std::snprintf(buffer, sizeof(buffer), "%.3f", static_cast<double>(value));
  if (written <= 0) {
    return;
  }
  // snprintf returns what it would have written, which is longer than the buffer on truncation.
  auto length = std::min(static_cast<size_t>(written), sizeof(buffer) - 1);
  std::string_view text(buffer, length);
  if (text.find('.') != std::string_view::npos) {
    while (!text.empty() && text.back() == '0') {
      text.remove_suffix(1);
    }
    if (!text.empty() && text.back() == '.') {
      text.remove_suffix(1);
    }
  }
  out += text.empty() ? "0" : text;
}

void appendPoint(std::string& out, const tgfx::Point& point) {
  appendNumber(out, point.x);
  out += ' ';
  appendNumber(out, point.y);
}

}  // namespace

std::string pathToSvg(const tgfx::Path& path) {
  std::string out;
  for (const auto& segment : path) {
    switch (segment.verb) {
      case tgfx::PathVerb::Move:
        out += 'M';
        appendPoint(out, segment.points[0]);
        break;
      case tgfx::PathVerb::Line:
        out += 'L';
        appendPoint(out, segment.points[1]);
        break;
      case tgfx::PathVerb::Quad:
        out += 'Q';
        appendPoint(out, segment.points[1]);
        out += ' ';
        appendPoint(out, segment.points[2]);
        break;
      case tgfx::PathVerb::Conic: {
        // SVG has no conic. Two quads are exact at weight sqrt(2)/2, the 90° arc fonts emit.
        auto quads = tgfx::Path::ConvertConicToQuads(segment.points[0], segment.points[1],
                                                     segment.points[2], segment.conicWeight, 1);
        for (size_t i = 1; i + 1 < quads.size(); i += 2) {
          out += 'Q';
          appendPoint(out, quads[i]);
          out += ' ';
          appendPoint(out, quads[i + 1]);
        }
        break;
      }
      case tgfx::PathVerb::Cubic:
        out += 'C';
        appendPoint(out, segment.points[1]);
        out += ' ';
        appendPoint(out, segment.points[2]);
        out += ' ';
        appendPoint(out, segment.points[3]);
        break;
      case tgfx::PathVerb::Close:
        out += 'Z';
        break;
      case tgfx::PathVerb::Done:
        break;
    }
  }
  return out;
}

}  // namespace tgfxrn
