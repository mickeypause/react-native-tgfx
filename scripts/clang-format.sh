#!/bin/bash
# Formats the hand-written C++, Objective-C++ and C in place with the repo's
# .clang-format. CI runs this, then fails on any diff it leaves.
#
#   bun lint-cpp
#
# Generated code (nitrogen/) and the vendored tgfx submodule are not ours to format.
set -euo pipefail

# CI installs exactly this version; another major version formats differently.
CLANG_FORMAT_VERSION=23.1.3

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CPP_DIRS=(
  packages/react-native-tgfx/cpp
  packages/react-native-tgfx/ios
  packages/react-native-tgfx/android/src/main/cpp
  packages/react-native-tgfx-vision-camera/cpp
  packages/react-native-tgfx-vision-camera/android/src/main/cpp
)

if ! command -v clang-format >/dev/null; then
  echo "error: clang-format not installed. Install the version CI uses with" >&2
  echo "  pipx install clang-format==$CLANG_FORMAT_VERSION" >&2
  exit 1
fi

installed="$(clang-format --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
if [ "${installed%%.*}" != "${CLANG_FORMAT_VERSION%%.*}" ]; then
  # On CI a stray runner clang-format would pass or fail the check by accident.
  if [ -n "${CI:-}" ]; then
    echo "error: clang-format $installed on PATH, expected $CLANG_FORMAT_VERSION" >&2
    exit 1
  fi
  echo "warning: clang-format $installed, CI uses $CLANG_FORMAT_VERSION, so the result may differ from CI's" >&2
fi

find "${CPP_DIRS[@]}" -path '*/third_party' -prune -o -type f \
  \( -name '*.h' -o -name '*.hpp' -o -name '*.c' -o -name '*.cpp' -o -name '*.m' -o -name '*.mm' \) \
  -print0 | xargs -0 clang-format -style=file -i

echo "C++ format done!"
