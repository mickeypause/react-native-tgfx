#include "PathOps.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "SvgPath.hpp"
#include "tgfx/core/Matrix.h"
#include "tgfx/core/PathMeasure.h"
#include "tgfx/core/PathTypes.h"
#include "tgfx/core/Point.h"
#include "tgfx/core/Rect.h"

namespace tgfxrn {
namespace {

/// One contour, kept as its own path so tgfx can answer the nesting question below.
struct Contour {
  tgfx::Path path;
  std::vector<tgfx::Path::Segment> segments;
  std::vector<tgfx::Point> outline;
  tgfx::Rect box = tgfx::Rect::MakeEmpty();
  float area = 0;
};

/// Twice the signed area of the contour's polygon hull: sign is the winding direction.
float shoelace(const std::vector<tgfx::Path::Segment>& segments) {
  float sum = 0;
  tgfx::Point previous = {};
  tgfx::Point start = {};
  bool first = true;
  auto edge = [&](const tgfx::Point& point) {
    if (!first) {
      sum += previous.x * point.y - point.x * previous.y;
    } else {
      start = point;
    }
    first = false;
    previous = point;
  };
  for (const auto& segment : segments) {
    switch (segment.verb) {
      case tgfx::PathVerb::Move: edge(segment.points[0]); break;
      case tgfx::PathVerb::Line: edge(segment.points[1]); break;
      case tgfx::PathVerb::Quad: edge(segment.points[2]); break;
      case tgfx::PathVerb::Conic: edge(segment.points[2]); break;
      case tgfx::PathVerb::Cubic: edge(segment.points[3]); break;
      default: break;
    }
  }
  // The closing edge, so the magnitude does not depend on where the contour sits.
  sum += previous.x * start.y - start.x * previous.y;
  return sum;
}

void appendSegment(tgfx::Path& out, const tgfx::Path::Segment& segment) {
  switch (segment.verb) {
    case tgfx::PathVerb::Move: out.moveTo(segment.points[0]); break;
    case tgfx::PathVerb::Line: out.lineTo(segment.points[1]); break;
    case tgfx::PathVerb::Quad:
      out.quadTo(segment.points[1], segment.points[2]);
      break;
    case tgfx::PathVerb::Conic: {
      auto quads = tgfx::Path::ConvertConicToQuads(segment.points[0], segment.points[1],
                                                   segment.points[2], segment.conicWeight, 1);
      for (size_t i = 1; i + 1 < quads.size(); i += 2) {
        out.quadTo(quads[i], quads[i + 1]);
      }
      break;
    }
    case tgfx::PathVerb::Cubic:
      out.cubicTo(segment.points[1], segment.points[2], segment.points[3]);
      break;
    case tgfx::PathVerb::Close: out.close(); break;
    default: break;
  }
}

/// Points along the contour's own outline: a contour enclosed by another has all of
/// them inside it, and shared vertices can only ever be a minority of the sample.
std::vector<tgfx::Point> outlinePoints(const tgfx::Path& contour) {
  std::vector<tgfx::Point> points;
  points.reserve(5);
  auto measure = tgfx::PathMeasure::MakeFrom(contour);
  if (measure == nullptr) {
    return points;
  }
  float length = measure->getLength();
  if (length <= 0) {
    return points;
  }
  for (int i = 1; i <= 9; i += 2) {
    tgfx::Point position = {};
    tgfx::Point tangent = {};
    if (measure->getPosTan(length * static_cast<float>(i) / 10.0f, &position, &tangent)) {
      points.push_back(position);
    }
  }
  return points;
}

/// Whether `outer` encloses the contour those outline points came from.
bool encloses(const tgfx::Path& outer, const std::vector<tgfx::Point>& points) {
  size_t inside = 0;
  for (const auto& point : points) {
    if (outer.contains(point.x, point.y)) {
      inside++;
    }
  }
  return inside * 2 > points.size();
}

std::vector<Contour> splitContours(const tgfx::Path& path) {
  std::vector<Contour> contours;
  for (const auto& segment : path) {
    if (segment.verb == tgfx::PathVerb::Move) {
      contours.push_back({});
    }
    if (contours.empty()) {
      continue;
    }
    contours.back().segments.push_back(segment);
  }
  if (contours.size() < 2) {
    return contours;
  }
  for (auto& contour : contours) {
    for (const auto& segment : contour.segments) {
      appendSegment(contour.path, segment);
    }
    contour.path.close();
    contour.box = contour.path.getBounds();
    contour.area = shoelace(contour.segments);
    contour.outline = outlinePoints(contour.path);
  }
  return contours;
}

/// Walk a contour backwards, which flips its winding direction without moving any point.
void appendReversed(tgfx::Path& out, const std::vector<tgfx::Path::Segment>& segments) {
  std::vector<tgfx::Point> ends;
  std::vector<tgfx::Path::Segment> curves;
  for (const auto& segment : segments) {
    switch (segment.verb) {
      case tgfx::PathVerb::Move: ends.push_back(segment.points[0]); break;
      case tgfx::PathVerb::Line:
        ends.push_back(segment.points[1]);
        curves.push_back(segment);
        break;
      case tgfx::PathVerb::Quad:
        ends.push_back(segment.points[2]);
        curves.push_back(segment);
        break;
      case tgfx::PathVerb::Conic:
        ends.push_back(segment.points[2]);
        curves.push_back(segment);
        break;
      case tgfx::PathVerb::Cubic:
        ends.push_back(segment.points[3]);
        curves.push_back(segment);
        break;
      default: break;
    }
  }
  if (ends.empty()) {
    return;
  }
  out.moveTo(ends.back());
  for (size_t i = curves.size(); i > 0; i--) {
    const auto& segment = curves[i - 1];
    auto target = ends[i - 1];
    switch (segment.verb) {
      case tgfx::PathVerb::Quad: out.quadTo(segment.points[1], target); break;
      case tgfx::PathVerb::Conic: {
        auto quads = tgfx::Path::ConvertConicToQuads(segment.points[0], segment.points[1],
                                                     segment.points[2], segment.conicWeight, 1);
        for (size_t q = quads.size(); q >= 3; q -= 2) {
          out.quadTo(quads[q - 2], quads[q - 3]);
        }
        break;
      }
      case tgfx::PathVerb::Cubic:
        out.cubicTo(segment.points[2], segment.points[1], target);
        break;
      default: out.lineTo(target); break;
    }
  }
  out.close();
}

/// Flip every contour nested at odd depth, so winding fills what even-odd filled.
tgfx::Path toWinding(const tgfx::Path& path) {
  if (path.getFillType() != tgfx::PathFillType::EvenOdd || path.isEmpty()) {
    return path;
  }
  auto contours = splitContours(path);
  if (contours.size() < 2) {
    return path;
  }
  tgfx::Path out;
  for (size_t i = 0; i < contours.size(); i++) {
    int depth = 0;
    for (size_t j = 0; j < contours.size(); j++) {
      if (j != i && contours[j].box.contains(contours[i].box) &&
          encloses(contours[j].path, contours[i].outline)) {
        depth++;
      }
    }
    bool wantPositive = (depth % 2 == 0) == (contours[0].area > 0);
    if ((contours[i].area > 0) == wantPositive) {
      for (const auto& segment : contours[i].segments) {
        appendSegment(out, segment);
      }
      out.close();
    } else {
      appendReversed(out, contours[i].segments);
    }
  }
  return out;
}

tgfx::PathOp opOf(const std::string& name) {
  if (name == "difference") {
    return tgfx::PathOp::Difference;
  }
  if (name == "intersect") {
    return tgfx::PathOp::Intersect;
  }
  if (name == "union") {
    return tgfx::PathOp::Union;
  }
  if (name == "xor") {
    return tgfx::PathOp::XOR;
  }
  return tgfx::PathOp::Append;
}

}  // namespace

std::string pathCombine(const std::string& a, const std::string& b, const std::string& op) {
  auto path = parseSvgPath(a);
  path.addPath(parseSvgPath(b), opOf(op));
  return pathToSvg(toWinding(path));
}

tgfx::Rect pathBounds(const std::string& path, bool tight) {
  auto parsed = parseSvgPath(path);
  return tight ? parsed.computeTightBounds() : parsed.getBounds();
}

bool pathContains(const std::string& path, float x, float y) {
  return parseSvgPath(path).contains(x, y);
}

std::string pathTransform(const std::string& path, const std::vector<double>& matrix) {
  auto parsed = parseSvgPath(path);
  if (matrix.size() < 6) {
    return pathToSvg(parsed);
  }
  auto at = [&](size_t i) { return static_cast<float>(matrix[i]); };
  parsed.transform(tgfx::Matrix::MakeAll(at(0), at(2), at(4), at(1), at(3), at(5)));
  return pathToSvg(parsed);
}

float pathLength(const std::string& path) {
  auto measure = tgfx::PathMeasure::MakeFrom(parseSvgPath(path));
  if (measure == nullptr) {
    return 0;
  }
  float total = 0;
  do {
    total += measure->getLength();
  } while (measure->nextContour());
  return total;
}

PathPointResult pathPointAtLength(const std::string& path, float distance) {
  auto measure = tgfx::PathMeasure::MakeFrom(parseSvgPath(path));
  if (measure == nullptr) {
    return {};
  }
  // PathMeasure walks one contour at a time, so spend the distance contour by contour.
  float remaining = std::max(distance, 0.0f);
  PathPointResult last;
  do {
    float length = measure->getLength();
    tgfx::Point position = {};
    tgfx::Point tangent = {};
    if (measure->getPosTan(std::min(remaining, length), &position, &tangent)) {
      last = {position.x, position.y, std::atan2(tangent.y, tangent.x), true};
    }
    if (remaining <= length) {
      return last;
    }
    remaining -= length;
  } while (measure->nextContour());
  return last;
}

std::string pathSegment(const std::string& path, float start, float stop) {
  auto measure = tgfx::PathMeasure::MakeFrom(parseSvgPath(path));
  if (measure == nullptr || start >= stop) {
    return "";
  }
  tgfx::Path out;
  float walked = 0;
  bool any = false;
  do {
    float length = measure->getLength();
    tgfx::Path piece;
    if (measure->getSegment(start - walked, stop - walked, &piece)) {
      out.addPath(piece);
      any = true;
    }
    walked += length;
  } while (walked < stop && measure->nextContour());
  return any ? pathToSvg(out) : "";
}

}  // namespace tgfxrn
