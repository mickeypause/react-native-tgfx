import { useEffect, useState } from 'react'
import { Pressable, StyleSheet, Text as RNText, View } from 'react-native'
import type { CameraDevice } from 'react-native-vision-camera'
import {
  useCamera,
  useCameraDevice,
  useCameraPermission,
  useFrameOutput,
  usePreviewOutput,
} from 'react-native-vision-camera'
import {
  Blur,
  CameraFrame,
  Canvas,
  Fill,
  ImageShader,
  Shader,
  useCameraFrame,
  useClock,
  type TgfxImage,
} from 'react-native-tgfx'
import { useTgfxCameraSink } from 'react-native-tgfx-vision-camera'

const BLUR_RADIUS = 28

const NEON_EDGE = `
uniform shader cam;
uniform float2 uSize;
uniform float uTime;

float luma(half4 c) {
  return dot(c.rgb, half3(0.2126, 0.7152, 0.0722));
}

half4 main(float2 pos) {
  float t = 1.7;

  float lo = luma(cam.eval(pos - float2(t, t)));
  float hi = luma(cam.eval(pos + float2(t, t)));
  float dirEdge = hi - lo;

  float gx = luma(cam.eval(pos + float2(t, 0.0))) -
             luma(cam.eval(pos - float2(t, 0.0)));
  float gy = luma(cam.eval(pos + float2(0.0, t))) -
             luma(cam.eval(pos - float2(0.0, t)));
  float mag = sqrt(gx * gx + gy * gy);

  half3 cyan = half3(0.25, 1.00, 0.95);
  half3 magenta = half3(1.00, 0.15, 0.90);
  half3 tint = mix(cyan, magenta, clamp(dirEdge * 7.0 + 0.5, 0.0, 1.0));

  float edge = pow(clamp(mag * 6.5, 0.0, 1.0), 0.70);

  half4 base = cam.eval(pos);
  half3 col = base.rgb * half3(0.14, 0.09, 0.28) + tint * edge * 1.45;

  col *= 0.93 + 0.07 * sin(uTime * 1.6);

  float2 uv = pos / uSize;
  // Rising edges only: smoothstep with edge0 > edge1 is undefined in GLSL ES.
  float vig = 1.0 - smoothstep(0.38, 1.10, length(uv - 0.5));
  col *= 0.40 + 0.60 * vig;

  return half4(clamp(col, 0.0, 1.0), base.a);
}
`

type Mode = 'preview' | 'none' | 'blur' | 'cyberpunk'

/**
 * The gate is its own component: the frame hooks below reach for a camera as
 * soon as they run, so on a device without one they must never run at all —
 * and a hook cannot be called conditionally.
 */
export function TgfxCameraEffects() {
  const { hasPermission, requestPermission } = useCameraPermission()
  const device = useCameraDevice('front')

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <RNText
          testID="effects-permission-prompt"
          style={styles.text}
          onPress={() => {
            requestPermission()
          }}
        >
          Tap to grant camera permission.
        </RNText>
      </View>
    )
  }
  if (device == null) {
    return (
      <View style={styles.center}>
        <RNText testID="effects-no-camera" style={styles.text}>
          No camera available.
        </RNText>
      </View>
    )
  }
  return <EffectsCamera device={device} />
}

/** Own component: useClock ticks every frame, so it must only mount with the effect. */
function CyberpunkLayer({
  image,
  width,
  height,
}: {
  image: TgfxImage
  width: number
  height: number
}) {
  const clock = useClock()
  return (
    <Fill>
      <Shader
        source={NEON_EDGE}
        uniforms={{
          uSize: [width, height],
          uTime: clock,
        }}
      >
        {/* Tile modes left at 'clamp': a YUV frame sampled with any
            other one comes back transparent. */}
        <ImageShader image={image} />
      </Shader>
    </Fill>
  )
}

function EffectsCamera({ device }: { device: CameraDevice }) {
  const [mode, setMode] = useState<Mode>('blur')

  const { image, posture, onFrame } = useCameraFrame()
  const sink = useTgfxCameraSink()

  // HAL-direct preview, no tgfx: the latency floor to compare the canvas against.
  const previewOutput = usePreviewOutput()
  const frameOutput = useFrameOutput({
    pixelFormat: 'yuv',
    dropFramesWhileBusy: true,
    // Headless: this output is the only one, so deferring it behind a preview starts it late.
    allowDeferredStart: false,
    onFrame,
  })

  // No per-frame JS here, so the native sink takes over once the hook has a handle.
  useEffect(() => {
    if (image == null) return
    sink.attach(frameOutput, image.handle)
  }, [sink, frameOutput, image])

  useCamera({
    isActive: true,
    device: device,
    outputs: [frameOutput, previewOutput],
    // Without a floor the AE drops to ~15fps in dim light, which doubles glass-to-glass latency.
    constraints: [{ fps: 30 }],
  })

  return (
    <View testID="effects-live" style={styles.fill}>
      {/* Both surfaces stay mounted: unbinding the preview surface stalls the
          whole capture session in the HAL, and a remounted Canvas misses its
          registry wake. The inactive one parks at 1x1. */}
      {/* <NativePreviewView
        style={mode === 'preview' ? StyleSheet.absoluteFill : styles.parked}
        previewOutput={previewOutput}
      /> */}
      <Canvas
        style={mode === 'preview' ? styles.parked : StyleSheet.absoluteFill}
        opaque={true}
      >
        <CameraFrame image={image} {...posture}>
          {mode === 'blur' ? <Blur blur={BLUR_RADIUS} /> : null}
          {mode === 'cyberpunk' && image != null && posture.width > 0 ? (
            <CyberpunkLayer
              image={image}
              width={posture.width}
              height={posture.height}
            />
          ) : null}
        </CameraFrame>
      </Canvas>
      <View style={styles.bar}>
        {(['preview', 'none', 'blur', 'cyberpunk'] as Mode[]).map(m => (
          <Pressable
            key={m}
            testID={`effects-mode-${m}`}
            accessibilityRole="button"
            accessibilityState={{ selected: mode === m }}
            style={[styles.button, mode === m && styles.buttonOn]}
            onPress={() => {
              setMode(m)
            }}
          >
            <RNText style={styles.buttonText}>{m}</RNText>
          </Pressable>
        ))}
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: '#000' },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#000',
  },
  text: { color: '#fff', fontSize: 16 },
  parked: { position: 'absolute', top: 0, left: 0, width: 1, height: 1 },
  bar: {
    position: 'absolute',
    bottom: 48,
    alignSelf: 'center',
    flexDirection: 'row',
    gap: 12,
  },
  button: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 24,
    backgroundColor: '#0008',
    borderWidth: 1,
    borderColor: '#0ff6',
  },
  buttonOn: { backgroundColor: '#0ff3', borderColor: '#0ff' },
  buttonText: { color: '#fff', fontSize: 15, fontWeight: '600' },
})
