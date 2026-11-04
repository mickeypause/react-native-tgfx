import type { HybridObject } from 'react-native-nitro-modules'
import type { TgfxImage } from './TgfxImage.nitro'

/**
 * A retained GPU surface no view owns - the render target for headless work.
 *
 * `renderToImage` is one-shot: every call allocates a surface, compiles the
 * program, renders, reads back, and throws it all away. This object keeps the
 * surface and the compiled program alive, so an iterate-and-export loop pays
 * per frame only what an on-screen canvas does: slot resolution and a render
 * pass, all on a dedicated offscreen worker thread. Readback happens only when
 * {@link toImage} or {@link save} asks.
 *
 * `setScene` / `setValues` / `setStringValues` are byte-for-byte the
 * `TgfxViewMethods` contract, so a surface and a view are interchangeable
 * scene targets - `Scene.attach(surface)` and the Reanimated drivers work
 * unchanged. As a Nitro object it boxes with `NitroModules.box()`, so a
 * worklet runtime can own the whole mutate-flush loop.
 *
 * The colour space is a `createOffscreenSurface` option rather than a method:
 * a surface's space is fixed at creation, exactly like a view's.
 */
export interface TgfxOffscreenSurface extends HybridObject<{
  ios: 'c++'
  android: 'c++'
}> {
  /** Logical points, as given at creation. */
  readonly width: number
  readonly height: number

  /**
   * Stage a draw program. Same contract as the view: the buffer is only valid
   * for the duration of the call, and `strings` is a `packStringTable` delta
   * against the table this surface already holds.
   */
  setScene(scene: ArrayBuffer, strings: string[]): void

  /** Patch the animated value slots. Worklet-safe through a boxed surface. */
  setValues(values: number[]): void

  /** Sparse writes into the string-slot array (live text and path data). */
  setStringValues(indices: number[], values: string[]): void

  /** `[zoom, x, y]`, the view's viewport contract - a zoomed crop export. */
  setViewport(viewport: number[]): void

  /**
   * Index paths under a point in canvas points, flattened `[depth, i0, i1, …]`,
   * topmost first - the view's `hitTest`, answered headless from the same
   * staged scene and viewport. The program compiles on the worker's first
   * drain, so a hit test racing the very first `flush()` can answer empty.
   */
  hitTest(x: number, y: number, precise: boolean): number[]

  /**
   * Render the staged state now. Stages and wakes the offscreen worker - it
   * never renders on the calling thread, so a UI-runtime caller pays a method
   * call, not a GPU pass. Consecutive flushes coalesce: the worker draws the
   * latest state, and readbacks only run for pending {@link toImage}/{@link save}.
   */
  flush(): void

  /**
   * The current frame as a registered image handle, read back to CPU pixels
   * (a texture cannot cross out of the offscreen context). Renders the latest
   * staged state first, so `flush()` beforehand is not required.
   */
  toImage(): Promise<TgfxImage>

  /**
   * Encode the current frame straight to `path` - png (default) / jpeg / webp,
   * `quality` 1-100. The bytes never enter JS, and no image handle is minted.
   */
  save(path: string, format?: string, quality?: number): Promise<void>

  // `dispose()` comes from HybridObject and is overridden natively: it frees
  // the GPU surface now and rejects whatever readbacks were still pending.
}

/**
 * Creates {@link TgfxOffscreenSurface}s. Use the exported
 * `createOffscreenSurface` wrapper, which validates and applies defaults.
 */
export interface TgfxOffscreenSurfaceFactory extends HybridObject<{
  ios: 'c++'
  android: 'c++'
}> {
  create(
    width: number,
    height: number,
    pixelRatio: number,
    opaque: boolean,
    colorSpace: string
  ): TgfxOffscreenSurface
}
