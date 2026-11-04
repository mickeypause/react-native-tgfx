# VisionCamera — drawing frames through tgfx

The camera is a **headless frame source only**. Every visible pixel is drawn by
tgfx: `useCamera()` runs the session with no camera view mounted, and a
full-screen `<Canvas>` renders the frame plus whatever you compose over it.

```bash
npm install react-native-vision-camera react-native-vision-camera-worklets
```

Requires `react-native-reanimated` + `react-native-worklets` (the handle has to
outlive one callback on the producer's worklet runtime).

## The minimal preview

```tsx
import { StyleSheet } from 'react-native'
import { useCamera, useFrameOutput } from 'react-native-vision-camera'
import { Canvas, CameraFrame, useCameraFrame } from 'react-native-tgfx'

export function Preview() {
  const { image, posture, onFrame } = useCameraFrame()

  const frameOutput = useFrameOutput({
    pixelFormat: 'yuv',                  // never 'rgb'
    dropFramesWhileBusy: true,
    enablePhysicalBufferRotation: true,
    allowDeferredStart: false,           // headless: nothing to defer behind
    onFrame,
  })

  useCamera({
    isActive: true,
    device: 'back',                      // a CameraPosition or a CameraDevice
    outputs: [frameOutput],
  })

  return (
    <Canvas style={StyleSheet.absoluteFill} opaque={true}>
      <CameraFrame image={image} {...posture} />
    </Canvas>
  )
}
```

Gate on `useCameraPermission()` before mounting — `apps/example` does.

## `useCameraFrame`

```ts
function useCameraFrame<F extends FrameLike = FrameLike>(
  onEachFrame?: (frame: F) => void,      // your own worklet, e.g. a detector
  options?: CameraFrameOptions<F>
): CameraFrameSource<F>

interface CameraFrameSource<F> {
  image: TgfxImage | null                // pass to <CameraFrame image={...}>
  posture: CameraFramePosture
  onFrame: (frame: F) => void            // hand to useFrameOutput
}

interface CameraFramePosture {
  rotation: number                       // clockwise degrees that cancel the buffer's
  mirror: boolean
  width: number                          // frame pixels — the space children draw in
  height: number
}

interface CameraFrameOptions<F> {
  format?: (frame: F) => PixelFormat     // worklet; runs on the frame thread
  rotation?: (frame: F) => number        // worklet
}
```

Frames are typed **structurally** (`FrameLike`: `getNativeBuffer()`,
`pixelFormat`, `orientation`, `isMirrored`, `width`, `height`, `dispose()`), so
the core package has no VisionCamera dependency and any producer works.

Defaults:

- `format` reads the producer's string — `rgb`/`rgba`/`bgra`/`argb` → `'rgba'`;
  `709` / `2020` pick that matrix; anything else is BT.601; a `-full` suffix (or
  `full-range`) selects the full-range variant. Getting this wrong crushes the
  blacks, so override it only for a producer that names formats differently.
- `rotation` understands `'up' | 'right' | 'down' | 'left'` → `0 | 270 | 180 | 90`;
  anything else is 0.

`posture` reaches React state only when it actually changes. The handle is
disposed on unmount — an external registry entry is never evicted, so nothing
else would ever free it.

## `<CameraFrame>` — `CameraFrameProps`

| Prop | Notes |
| --- | --- |
| `image` | the handle from `useCameraFrame`. `null` draws nothing **at all, children included** — there is no frame to place them against. |
| `fit` | defaults to `'cover'` (fills the canvas, the overflowing axis runs off it) |
| `sampling` | defaults to `'linear'` |
| `rotation` | clockwise degrees that *counter* `frame.orientation`. A frame reporting `'right'` needs `270`. Rounded to the nearest quarter turn and wrapped, so `-90` and `270` are the same — a detector reporting signed degrees passes straight through. |
| `mirror` | pass `frame.isMirrored` straight through |

Spreading `{...posture}` supplies `rotation` and `mirror` (plus `width`/`height`,
which `<CameraFrame>` ignores and you use for shader uniforms).

**Children of `<CameraFrame>` draw in frame-pixel space** — the same coordinates
a detector reports, so detector bounds go in unscaled.

## Zero copy — what actually happens

The frame's platform buffer (`AHardwareBuffer*` on Android, `CVPixelBufferRef`
on iOS) is imported straight to a texture. No decode, no pixel copy, no promise:
`imageFromNativeBuffer` is **synchronous**, and YUV planes are converted in the
fragment shader rather than on the CPU. Such a handle is not charged against the
decoded-pixel budget — the memory belongs to whoever produced the buffer.

```ts
handle.value == null
  ? imageFromNativeBuffer(buffer.pointer, format)          // first frame
  : handle.value.setNativeBuffer(buffer.pointer, format)   // every frame after
```

`setNativeBuffer` retains the new buffer and drops the previous one before
returning, so a producer recycling a fixed pool gets its slot back on the swap.
A frame the renderer has not drawn yet is **superseded** by the next, not queued
behind it.

### Lifetime rules

- The buffer's **contents** must stay unchanged while a handle points at it —
  tgfx samples it directly. Write into a pool slot only after it has been
  swapped out.
- **The frame your `onEachFrame` receives dies when it returns.** Do not retain
  it. Do not call `getNativeBuffer()` on it again — that is a second retain
  nothing releases, and one unreleased buffer stalls the producer's bounded
  pool. Never store `getNativeBuffer().pointer`; a stale pointer does not throw,
  it imports whatever memory now sits at that address.
- `useCameraFrame` owns all of this: the raw pointer never leaves its worklet,
  and a `try`/`finally` releases the buffer and disposes the frame on every path
  including a throw. Your callback runs inside that same `try`.

## Face blur

The scene compiles once, so allocate a
**fixed** number of slots; an unused slot has zero size and draws nothing.

```tsx
import { useCallback } from 'react'
import { StyleSheet, View } from 'react-native'
import type { Frame } from 'react-native-vision-camera'
import { useCamera, useCameraDevice, useCameraPermission, useFrameOutput } from 'react-native-vision-camera'
import { useFaceDetector } from 'react-native-vision-camera-face-detector'
import {
  Blur, CameraFrame, Canvas, ImageShader, Oval, Paint,
  useCameraFrame, useNode, type NodeHandle,
} from 'react-native-tgfx'

const BLUR_RADIUS = 25
const SLOTS = [0, 1, 2, 3, 4, 5, 6, 7]

export function TgfxFaceBlur() {
  const { hasPermission, requestPermission } = useCameraPermission()
  const device = useCameraDevice('front')

  // Fixed count: hooks cannot be called in a loop, and the scene must not resize.
  const ovals: NodeHandle[] = [
    useNode(), useNode(), useNode(), useNode(),
    useNode(), useNode(), useNode(), useNode(),
  ]

  const faceDetector = useFaceDetector({
    performanceMode: 'fast',
    runContours: false,
    runLandmarks: false,
    runClassifications: false,
  })

  const detect = useCallback(
    (frame: Frame) => {
      'worklet'
      // Already in frame pixels — the space <CameraFrame>'s children draw in — so unscaled.
      const faces = faceDetector.detectFaces(frame)
      for (let i = 0; i < SLOTS.length; i++) {
        const bounds = i < faces.length ? faces[i]!.bounds : null
        if (bounds == null) {
          ovals[i]!.set({ width: 0, height: 0 })
          continue
        }
        ovals[i]!.set({
          x: bounds.x, y: bounds.y,
          width: bounds.width, height: bounds.height,
        })
      }
    },
    [faceDetector, ovals]
  )

  const { image, posture, onFrame } = useCameraFrame(detect)

  const frameOutput = useFrameOutput({
    // 'yuv', not 'rgb': tgfx imports the planes straight to a texture and converts in the shader.
    pixelFormat: 'yuv',
    dropFramesWhileBusy: true,
    // The pipeline rotates the buffers, so `orientation` arrives 'up' and `isMirrored` false.
    enablePhysicalBufferRotation: true,
    onFrame,
  })

  useCamera({
    isActive: hasPermission && device != null,
    device: device ?? 'front',
    outputs: [frameOutput],
  })

  if (!hasPermission) return <TapToGrant onPress={requestPermission} />
  if (device == null) return null

  return (
    <View style={styles.fill}>
      <Canvas style={StyleSheet.absoluteFill} opaque={true}>
        <CameraFrame image={image} {...posture}>
          {SLOTS.map((slot) => (
            <Oval key={slot} ref={ovals[slot]} x={0} y={0} width={0} height={0}>
              <Paint>
                {/* The same frame, sampled 1:1 in this space, so the blurred pixels
                    are the pixels underneath. Tile modes left at 'clamp'. */}
                <ImageShader image={image} />
                <Blur blur={BLUR_RADIUS} />
              </Paint>
            </Oval>
          ))}
        </CameraFrame>
      </Canvas>
    </View>
  )
}
```

Why it is fast: the detector worklet writes only **slot-backed** props
(`x`, `y`, `width`, `height`) on pooled handles. No re-encode, no React render,
no reconciler commit — per frame the JS side does a detection and eight slot
writes.

`useCallback` the detector, or the frame output re-registers every render.

## No JS per frame — `react-native-tgfx-vision-camera`

```bash
npm i react-native-tgfx-vision-camera
cd ios && pod install
```

```
camera thread ──▶ HybridTgfxCameraSink ──▶ tgfxrn::setFrame ──▶ ImageRegistry ──▶ renderer
```

Moves the same zero-copy import into a native frame-processor plugin, so after
attachment **no frame reaches a JS runtime at all**.

```tsx
import { useTgfxCameraSink } from 'react-native-tgfx-vision-camera'

const { image, posture, onFrame } = useCameraFrame()
const sink = useTgfxCameraSink()

const frameOutput = useFrameOutput({
  pixelFormat: 'yuv',
  dropFramesWhileBusy: true,
  enablePhysicalBufferRotation: true,
  onFrame,
})

useEffect(() => {
  if (image != null) sink.attach(frameOutput, image.handle)
}, [sink, frameOutput, image])
```

- `createTgfxCameraSink()` is the non-hook form. `useTgfxCameraSink()` scopes it
  to the component and detaches on unmount.
- **One sink per camera output.** The sink holds the output it is attached to;
  sharing one between two cameras silently moves the second camera's frames onto
  the first camera's handle.
- The first frame is still JS: `imageFromNativeBuffer` needs a buffer, and there
  is no buffer before the first frame. The hook's worklet mints the handle;
  `attach()` then replaces that worklet with a native callback.
- `attach` is **asynchronous** — the worklet keeps receiving frames until the
  native callback is installed.

### When you can and cannot use it

`setOnFrameCallback` holds exactly **one** callback (a single `private var
onFrame` on Android, a single `delegate.onFrame` on iOS — no chaining, no
getter). Installing the native callback *removes* the JS one.

| the frame processor… | use |
| --- | --- |
| runs a detector, classifier, tracker — anything in JS per frame | `useCameraFrame()` alone |
| only needs the frame on the canvas | add the native sink |

Both are the same zero-copy import; the difference is only who calls it. When JS
is already on the path for a detector, the sink buys nothing — the import is one
Nitro call inside a worklet that had to run anyway.

### Posture freezes after attach

Because frames stop reaching JS, so does `frame.orientation`. Keep
`enablePhysicalBufferRotation: true` so the pipeline rotates the buffers itself
and `orientation` stays `'up'` for the session — otherwise orientation changes
mid-session and the sink will not tell you. If orientation must keep tracking,
stay on `useCameraFrame()` alone.

## Shading a frame

```tsx
<CameraFrame image={image} {...posture}>
  <Fill>
    <Shader source={NEON_EDGE} uniforms={{ uSize: [posture.width, posture.height], uTime: clock }}>
      <ImageShader image={image} />
    </Shader>
  </Fill>
</CameraFrame>
```

Guard on `posture.width > 0` — the first frame has not landed yet and a zero
`uSize` divides by zero.

A plain effect needs no shader at all — a `<Blur>` child of `<CameraFrame>`
blurs the whole frame:

```tsx
<CameraFrame image={image} {...posture}>
  <Blur blur={28} />
</CameraFrame>
```

Full shader sources and the rules of the SkSL subset are in
[api-effects-shaders.md](api-effects-shaders.md).

## Camera checklist

- [ ] `pixelFormat: 'yuv'` — never `'rgb'` (a full-frame CPU copy per frame).
- [ ] `enablePhysicalBufferRotation: true`, especially with the native sink.
- [ ] `dropFramesWhileBusy: true` unless every frame must be processed.
- [ ] `useCamera()` drives the session headlessly — no `<Camera>` view; the `<Canvas>` is the preview.
- [ ] `allowDeferredStart: false` — the default defers the frame output behind
      a preview output that a headless screen never mounts.
- [ ] Every options object handed to VisionCamera hooks (`useFrameOutput`,
      detector options) is a module constant or memoised — a fresh identity
      re-registers the output every render.
- [ ] `opaque={true}` on a full-screen camera canvas.
- [ ] No `tx`/`ty` on an `<ImageShader>` fed a YUV frame — it renders fully
      transparent.
- [ ] Detector callback memoised with `useCallback`.
- [ ] Fixed slot count, collapsed to zero size when unused — never map over a
      variable-length detection array.
- [ ] Detector bounds used unscaled: `<CameraFrame>` children are in frame pixels.
- [ ] Never retain the frame or its buffer pointer past the callback.
