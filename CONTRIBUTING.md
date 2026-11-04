# Contributing

## Setup without the tgfx submodule

`cpp/third_party/tgfx` is a private repo. Without access to it, fetch the
prebuilt static libs and tgfx's headers from R2 instead of cloning and building
the submodule:

```sh
bun run bootstrap
```

That writes `packages/react-native-tgfx/{ios,android}/libs` and
`cpp/third_party/tgfx/include`, which is everything the podspec and
`android/CMakeLists.txt` need — both link the prebuilt archive and never compile
tgfx. The object key is the submodule commit the checkout pins, so a bump has no
cached build until someone uploads one (`bash scripts/tgfx-prebuilt.sh upload`).

## Running the example app

```sh
git clone https://github.com/riteshshukla04/react-native-tgfx.git
cd react-native-tgfx
bun install
bun run bootstrap        # prebuilt tgfx, see above — no submodule needed

cd apps/example
bun run android          # ANDROID_HOME must be set; boot an emulator first

bundle install && (cd ios && bundle exec pod install) && bun run ios
```

`pod install` needs a UTF-8 locale — under `LANG=C` CocoaPods dies with
"Unicode Normalization not appropriate for ASCII-8BIT". Prefix it with
`LANG=en_US.UTF-8` if your shell does not set one.

The tab bar switches between the feature gallery, the camera effects and the
other demos. The camera screen asks for the camera permission the first time it
is opened, and falls back to a message when the device has no camera - an
emulator without a webcam included.

## Running the tests

`apps/example/src/*.harness.tsx` runs on a device through
[React Native Harness](https://github.com/margelo/react-native-harness): the
tests render into the real app and screenshot the GPU output, so a rendering
regression shows up as an image diff rather than a passing unit test.

```sh
cd apps/example
bun run harness                     # every suite
bun run harness src/paths.harness.tsx   # one of them
```

The default runner, `android`, names a specific phone in
`rn-harness.config.mjs`. Point it at yours (`adb shell getprop
ro.product.manufacturer` and `ro.product.model`) and regenerate — the pixels
are the GPU's, so baselines do not carry between devices. Install a **debug**
build first (`bun run android`); Harness drives the app through `run-as`, which
a release build refuses.

`ios-sim` runs a suite on an iOS simulator. Point it at a simulator you have
booted, install a debug build (`bun run ios`), and select it with
`--harnessRunner=ios-sim`. Only run suites there that check pixels, not image
suites.

```sh
bun run harness --harnessRunner=ios-sim src/<suite>.harness.tsx
```

`patches/` carries a patch on `@react-native-harness/runtime`: the stock
runtime defines a `globalThis.jest` trap inside the app, which trips
Reanimated's Jest detection and swaps in its mocks — every mapper dies, so
shared values and `handle.set` slot writes stop repainting *under the harness
only* (#230). Do not drop the patch on a harness upgrade without re-running
the imperative suite.

Baselines live in `src/__image_snapshots__/<runner>/`, and the directory is
named after the runner — a second runner forks the whole set, which is why
image suites run only on `android`. A missing baseline is **written** on the first run rather
than failing, so open the new PNG before committing it: whatever lands there is
what every later run is compared to.

The failure that costs the most time is a baseline of a blank canvas. A canvas
that has not presented screenshots as one flat colour, and two flat frames in a
row look as settled as a finished one — so the blank is written, and every run
after it passes. Anything a scene waits for (an image decode, a font, a face
detector) belongs in `beforeAll`, not inside the rendered tree. Note also that
image decodes are cached per source: `dispose()`ing a handle from `loadImage`
kills the handle every other test drawing that asset is using.

Run the image suites on a real phone. On the `Pixel_8_API_35` AVD's software
renderer every image-backed scene comes back blank, so those tests fail there
by design rather than writing an empty baseline.

## CI

Each workflow runs on the pull requests whose paths it covers, and on `main`.

- `lint-cpp.yml` - clang-format over the C++ and Objective-C++ in `packages/`.
  Fix with `bun lint-cpp`, using the version `scripts/clang-format.sh` pins
  (`pipx install clang-format==23.1.3`).
- `lint-typescript.yml` - `tsc`, eslint and prettier in each package, eslint in
  `apps/example` and `apps/benchmark`.
- `harness-android.yml` - every harness suite on a hosted emulator (`ci-emu`).
- `harness-ios.yml` - every harness suite on a simulator (`ios-sim`), x86_64
  build since ML Kit has no arm64 simulator slice.

The CI runners write their baselines fresh on each run and upload them as an
artifact. Committed baselines are for the `android` phone only.

## The memory-leak gate

`scripts/leak-check.mjs` drives `apps/example/src/leakSoak.harness.tsx` — ~100s
of image, scene, and canvas-surface churn on the connected device — while
sampling `adb shell dumpsys meminfo` from the host every two seconds. The
verdict compares the median `TOTAL PSS` of the soak's last third against its
first third (after a warmup cut); growth past `--threshold-mb` (default 30)
fails the run with exit code 1 (2 means the run itself broke, so a broken soak
can never pass as "no leak"). All measurement is OS accounting read over adb —
nothing in-app reports its own memory, deliberately: in-process numbers are the
flaky kind.

```sh
node scripts/leak-check.mjs        # device from rn-harness.config.mjs, debug build installed
```

The soak is excluded from the default `bun run harness` run
(`testPathIgnorePatterns` in `jest.harness.config.mjs`); the script re-enables
it. It does not run in CI. `--runner ci-emu` runs it on the CI emulator
instead of the phone, which catches allocation-path leaks (the registries, the
scene, the JSI boundary) but not GPU-side growth.

## Imperative APIs first

A lesson learned the hard way (see
[#204](https://github.com/riteshshukla04/react-native-tgfx/issues/204)): every
API that shipped declarative-only had its imperative path retrofitted later, or
never. When adding a new drawable, paint, or effect:

1. **Design the imperative path first** — which props are value slots, what the
   wire format carries, what a write from a worklet costs.
2. **Then wrap the declarative component over it.**

A PR that adds a declarative API with no slot story for its animatable props is
incomplete.

## Performance — no compromise

The imperative path is the hot path. Non-negotiables:

- A slot patch happens **in native**, on the render thread, resolved at draw.
  No JSON, no per-write parse, no scene re-encode.
- Slot writes are **worklet-safe** — they must work while the JS thread is
  blocked.
- **Zero per-frame JS allocations.** Bulk data travels via a pre-allocated
  shared `ArrayBuffer` JS writes into, not a fresh array per frame.
- Re-encode is the fallback, never the design.

## Docs

Every PR that changes the public surface updates `docs/content/docs` (and
`packages/react-native-tgfx/docs/wire-format.md` when the wire format moves).
Code-only PRs are incomplete. `docs/content/api` is generated by typedoc from
the package's doc comments — edit the source, not the output.
