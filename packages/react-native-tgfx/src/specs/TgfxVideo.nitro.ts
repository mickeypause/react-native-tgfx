import type { HybridObject } from 'react-native-nitro-modules'

/**
 * A decoding video, driven by the platform player - AVPlayer on iOS,
 * MediaPlayer on Android - so audio, looping and seeking cost this package no
 * decoder of its own.
 *
 * A frame never reaches JS at all, as pixels or as a pointer: the decoder writes
 * into a registry entry natively and JS only ever sees its id.
 */
export interface TgfxVideo extends HybridObject<{
  ios: 'swift'
  android: 'kotlin'
}> {
  /** Total length in milliseconds, 0 when the source does not report one. */
  readonly duration: number
  readonly framerate: number
  /** Clockwise degrees the buffer must be rotated by to be upright. */
  readonly rotation: number
  readonly width: number
  readonly height: number
  /** Playback position in milliseconds. */
  readonly currentTime: number

  paused: boolean
  looping: boolean
  /** 0 to 1. */
  volume: number

  seek(millis: number): void

  /**
   * Point this video's image at its newest decoded frame and return that image's
   * registry id, or 0 when nothing new has been decoded since the last call.
   *
   * The id is stable for the life of the video - a new frame replaces what the
   * id points at, so a scene referencing it costs no re-commit - and the frame
   * is whatever the decoder produced: a `CVPixelBuffer` on iOS, a
   * `SurfaceTexture` stream on Android, neither of them copied.
   */
  updateFrame(): number

  /**
   * Stop playback and free the decoder and its buffers now, rather than at the
   * next GC. A player left to a finalizer holds a hardware decoder - a device
   * has a handful of those, so a screen that opens videos runs out.
   */
  release(): void
}

export interface TgfxVideoFactory extends HybridObject<{
  ios: 'swift'
  android: 'kotlin'
}> {
  /** A `file://`, `http(s)://`, `content://` or `asset://` URI, or a bare path. */
  load(uri: string): Promise<TgfxVideo>
}
