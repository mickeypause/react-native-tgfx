'worklet'

// The React-free scene container: holds factory-built nodes, compiles them
// through the same TgfxContainer the reconciler uses, and fans the program out
// to any number of render targets with a per-target delta string table.
import type { MaybeSharedValue } from './compiler/values'
import { packStringTable } from './compiler/writer'
import type { NodeHandle } from './imperative'
import {
  isBlockSlot,
  readLiveNumber,
  readLiveString,
} from './reanimated/liveValue'
import { reanimated } from './reanimated/peer'
import { flagsOf, valuesOf } from './reanimated/slotFlags'
import { TgfxContainer } from './reconciler/Container'
import type { SceneNode } from './elements'
import type { TgfxImage as TgfxImageHandle } from './specs/TgfxImage.nitro'
import type { ColorSpace } from './types'

/** A default render size, so the export sugar needs no per-call one. */
export interface SceneOptions {
  width?: number
  height?: number
}

export interface SceneStillOptions extends SceneOptions {
  /** Pixels per point. Defaults to 1. */
  pixelRatio?: number
  opaque?: boolean
  colorSpace?: ColorSpace
  /** For `save`: defaults from the file extension, else png. */
  format?: 'png' | 'jpeg' | 'webp'
  /** 1-100, for the lossy formats. */
  quality?: number
}

export interface SceneVectorOptions extends SceneOptions {
  /** For `toSvg`: write glyphs as path outlines. */
  textAsPaths?: boolean
}

/**
 * Anything a {@link Scene} can drive: a mounted canvas (`TgfxViewRef`) or an
 * offscreen surface. The buffer passed to `setScene` is only valid for the
 * duration of the call; the target must copy what it keeps.
 */
export interface SceneTarget {
  setScene(scene: ArrayBuffer, strings: string[]): void
  setValues(values: number[]): void
  setStringValues?(indices: number[], values: string[]): void
}

interface Attachment {
  target: SceneTarget
  /// The string table this target holds, for delta encoding. Per target: a
  /// mispacked delta misindexes every string after entry 0.
  sent: string[]
}

/**
 * A retained scene built from {@link SceneNode}s - the imperative counterpart
 * of `<Canvas>`, with no React anywhere.
 *
 * Structural mutations (`add`, `remove`, ...) batch per microtask into one
 * compile; `flush()` commits now. Value mutation goes through node handles
 * (`createHandle`), which patch slots without re-encoding - do not rebuild the
 * tree per frame.
 */
export class Scene {
  private readonly roots: SceneNode[] = []
  private readonly targets: Attachment[] = []
  private dirty = false
  private disposed = false
  private compiled = false
  private last: { buffer: ArrayBuffer; strings: string[] } | null = null
  /// Bumped by flush/dispose so a stale queued microtask commits nothing.
  private generation = 0
  private mapperId: number | null = null
  private options: SceneOptions | null = null
  /// True during an export-only recompile: targets already show this state, so
  /// re-sending the program and restarting the mapper would be pure churn.
  private quiet = false
  private lastValues: MaybeSharedValue<unknown>[] = []
  private lastStringValues: MaybeSharedValue<string>[] = []

  private readonly container = new TgfxContainer(
    (buffer, strings, values, stringValues) => {
      this.compiled = true
      this.last = { buffer, strings }
      this.lastValues = values
      this.lastStringValues = stringValues
      if (this.quiet) {
        return
      }
      for (const t of this.targets) {
        t.target.setScene(buffer, packStringTable(t.sent, strings))
        t.sent = strings.slice()
      }
      this.syncMapper()
    }
  )

  private constructor() {
    this.container.onValues = (values) => {
      for (const t of this.targets) {
        t.target.setValues(values)
      }
    }
    this.container.onStrings = (indices, values) => {
      for (const t of this.targets) {
        t.target.setStringValues?.(indices, values)
      }
    }
  }

  static create(options?: SceneOptions): Scene {
    const scene = new Scene()
    scene.options = options ?? null
    return scene
  }

  static from(roots: SceneNode[], options?: SceneOptions): Scene {
    const scene = Scene.create(options)
    scene.replaceChildren(roots)
    return scene
  }

  /** Append a root node. */
  add(node: SceneNode): void {
    this.roots.push(node)
    this.markDirty()
  }

  /** Insert a root node before the given index (z-order). */
  insert(node: SceneNode, before: number): void {
    const at = Math.max(0, Math.min(before, this.roots.length))
    this.roots.splice(at, 0, node)
    this.markDirty()
  }

  /** Detach a root node by identity. Returns whether it was present. */
  remove(node: SceneNode): boolean {
    const at = this.roots.indexOf(node)
    if (at < 0) {
      return false
    }
    this.roots.splice(at, 1)
    this.markDirty()
    return true
  }

  /** Swap the whole root list. */
  replaceChildren(roots: SceneNode[]): void {
    this.roots.length = 0
    this.roots.push(...roots)
    this.markDirty()
  }

  clear(): void {
    this.replaceChildren([])
  }

  /** Commit pending structural changes now instead of at the microtask. */
  flush(): void {
    if (!this.dirty || this.disposed) {
      return
    }
    this.dirty = false
    this.generation++
    // Reseed=false, unlike a JSX commit: factory props are construction-time
    // values, not a re-asserted truth, so a structural commit must not snap
    // written handles back to them. A handle's first compile still seeds its
    // slots from the props - apply() treats an unbound handle as seedable.
    this.container.root = this.roots
    this.container.recompile()
  }

  /**
   * Drive a target from this scene. The new target receives the full current
   * program; every later commit and slot push fans out to all attached targets.
   * Returns a detach function.
   */
  attach(target: SceneTarget): () => void {
    if (this.disposed) {
      throw new Error('[tgfx] Scene.attach: this scene is disposed')
    }
    const entry: Attachment = { target, sent: [] }
    this.targets.push(entry)
    if (this.dirty) {
      this.flush()
    } else if (this.compiled) {
      // Reseed=false: current handle writes are baked as literals, so the new
      // target starts from what is on screen, not from the JSX-time props.
      this.container.recompile()
    }
    return () => {
      const at = this.targets.indexOf(entry)
      if (at >= 0) {
        this.targets.splice(at, 1)
        this.syncMapper()
      }
    }
  }

  /**
   * The current program, for the offscreen renderers. Recompiles without
   * reseeding, so live handle writes are baked in as literals rather than
   * reset to their construction-time props. Internal to the package.
   */
  compileForExport(what: string): { buffer: ArrayBuffer; strings: string[] } {
    if (this.disposed) {
      throw new Error(`[tgfx] ${what}: this scene is disposed`)
    }
    if (this.dirty) {
      this.flush()
    } else if (this.compiled) {
      this.quiet = true
      try {
        this.container.recompile()
      } finally {
        this.quiet = false
      }
    }
    if (this.last == null || this.container.root.length === 0) {
      throw new Error(`[tgfx] ${what}: the scene compiled to nothing`)
    }
    return this.last
  }

  /**
   * Name the nodes behind a flat hit-test answer - what `surface.hitTest` and
   * `view.hitTest` return raw. Topmost first, `null` for a node with no
   * handle; feed the result to `dispatchPress`.
   */
  handlesAt(flat: number[]): (NodeHandle | null)[] {
    const named = this.container.byPath
    const out: (NodeHandle | null)[] = []
    for (let i = 0; i < flat.length; ) {
      const depth = flat[i]!
      const path = flat.slice(i + 1, i + 1 + depth)
      i += 1 + depth
      out.push(named.get(path.join(',')) ?? null)
    }
    return out
  }

  /**
   * Render this scene and write the still to `path` - the full headless loop
   * in one call. Size comes from `options`, then `Scene.create`'s options,
   * then an attached offscreen surface. Format from the file extension unless
   * given. Resolves once the file is on disk; the bytes never enter JS.
   */
  async save(path: string, options?: SceneStillOptions): Promise<void> {
    this.assertJsThread('scene.save')
    const image = await this.toImage(options)
    try {
      const extension = path.slice(path.lastIndexOf('.') + 1).toLowerCase()
      const format =
        options?.format ??
        (extension === 'jpg' || extension === 'jpeg' || extension === 'webp'
          ? extension
          : 'png')
      await image.save(path, format, options?.quality)
    } finally {
      image.dispose()
    }
  }

  /** Render this scene to a registered image handle. Caller disposes it. */
  toImage(options?: SceneStillOptions): Promise<TgfxImageHandle> {
    const { width, height } = this.sizeFor('scene.toImage', options)
    const { renderToImage } =
      require('./offscreen') as typeof import('./offscreen')
    return renderToImage(this, {
      width,
      height,
      pixelRatio: options?.pixelRatio,
      opaque: options?.opaque,
      colorSpace: options?.colorSpace,
    })
  }

  /** Render this scene to SVG markup. Vectors stay vectors. */
  toSvg(options?: SceneVectorOptions): Promise<string> {
    const { width, height } = this.sizeFor('scene.toSvg', options)
    const { renderToSvg } =
      require('./offscreen') as typeof import('./offscreen')
    return renderToSvg(this, {
      width,
      height,
      textAsPaths: options?.textAsPaths,
    })
  }

  /** Write this scene as a one-page vector PDF. Size in points; A4 by naming it. */
  toPdf(path: string, options?: SceneVectorOptions): Promise<void> {
    const { width, height } = this.sizeFor('scene.toPdf', options)
    const { renderToPdf } =
      require('./offscreen') as typeof import('./offscreen')
    return renderToPdf(path, [{ width, height, scene: this }])
  }

  private assertJsThread(what: string): void {
    if ((globalThis as { _WORKLET?: boolean })._WORKLET === true) {
      throw new Error(
        `[tgfx] ${what}: exports run on the JS thread only. From a worklet, attach an offscreen surface and use surface.save/toImage.`
      )
    }
  }

  private sizeFor(
    what: string,
    options?: { width?: number; height?: number }
  ): { width: number; height: number } {
    this.assertJsThread(what)
    const width = options?.width ?? this.options?.width
    const height = options?.height ?? this.options?.height
    if (width != null && height != null) {
      return { width, height }
    }
    // A surface knows its size; a lone one can stand in for the scene's.
    if (this.targets.length === 1) {
      const target = this.targets[0]!.target as {
        width?: unknown
        height?: unknown
      }
      if (
        typeof target.width === 'number' &&
        typeof target.height === 'number'
      ) {
        return { width: target.width, height: target.height }
      }
    }
    throw new Error(
      `[tgfx] ${what}: no size to render at — pass { width, height }, create the scene with one (Scene.create({ width, height })), or attach an offscreen surface.`
    )
  }

  /** Detach every target and stop committing. Targets keep their last frame. */
  dispose(): void {
    this.disposed = true
    this.dirty = false
    this.generation++
    this.targets.length = 0
    this.roots.length = 0
    this.container.muted = true
    this.syncMapper()
  }

  /**
   * With Reanimated installed, handle writes and SharedValue props change on
   * the UI thread and `handle.set` defers its push to a driver - `<Canvas>`
   * mounts one, so a scene must run its own. The mapper is AnimationDriver's:
   * fill the slot array from the live values, push to every target, diff the
   * string slots. Without Reanimated the JS-thread push in `pushValues` covers
   * everything and no mapper exists.
   */
  // ponytail: restarted on every commit and attach/detach rather than diffed;
  // per-commit mapper churn only matters if structural commits are per-frame,
  // which the slot path exists to avoid.
  private syncMapper(): void {
    // On a worklet runtime there is no Reanimated module to start a mapper on,
    // and none is owed: a worklet scene is structural nodes and literal props -
    // value slots land at flush(), and per-frame motion drives the surface's
    // setValues directly.
    if ((globalThis as { _WORKLET?: boolean })._WORKLET === true) {
      return
    }
    const { startMapper, stopMapper } = reanimated()
    if (startMapper == null || stopMapper == null) {
      return
    }
    if (this.mapperId != null) {
      stopMapper(this.mapperId)
      this.mapperId = null
    }
    // Copies, never the writer's own arrays: Reanimated freezes what a worklet
    // captures, and a frozen writer array kills the next recompile's push.
    const values = this.lastValues.slice()
    const stringValues = this.lastStringValues.slice()
    if (
      this.disposed ||
      this.targets.length === 0 ||
      (values.length === 0 && stringValues.length === 0)
    ) {
      return
    }
    // A worklet clones captured arrays at creation, so membership changes
    // restart the mapper instead of mutating these.
    const views = this.targets.map((t) => t.target)
    const mirror = this.container.mirror?.block.boxed ?? null
    const slots = new Array<number>(values.length)
    const sent = new Array<string | null>(stringValues.length).fill(null)
    // Block slots have no listeners, so they cannot be mapper inputs; the
    // handles' gen values - bumped once per worklet-side set() - stand in.
    const listenable = [...values, ...stringValues].filter(
      (v) => !isBlockSlot(v)
    )
    for (const gen of this.container.gens) {
      listenable.push(gen)
    }
    if (listenable.length === 0) {
      return
    }
    this.mapperId = startMapper(() => {
      'worklet'
      const mirrored = mirror != null ? valuesOf(mirror) : null
      for (let i = 0; i < values.length; i++) {
        const v = readLiveNumber(values[i]!)
        slots[i] = v
        if (mirrored != null) {
          mirrored[i] = v
        }
      }
      if (mirror != null) {
        flagsOf(mirror)[0] = 1
      }
      for (const view of views) {
        view.setValues(slots)
      }
      if (stringValues.length > 0) {
        let indices: number[] | null = null
        let out: string[] | null = null
        for (let i = 0; i < stringValues.length; i++) {
          const now = readLiveString(stringValues[i]!)
          if (now === sent[i]) {
            continue
          }
          sent[i] = now
          ;(indices ??= []).push(i)
          ;(out ??= []).push(now)
        }
        if (indices != null && out != null) {
          for (const view of views) {
            view.setStringValues?.(indices, out)
          }
        }
      }
    }, listenable)
  }

  private markDirty(): void {
    if (this.disposed || this.dirty) {
      return
    }
    this.dirty = true
    const expected = ++this.generation
    void Promise.resolve().then(() => {
      if (this.generation === expected && this.dirty) {
        this.flush()
      }
    })
  }
}

/**
 * `Scene.create`, as a plain function. Inside a worklet this is the only form
 * that works: a class captured across modules crosses as a host reference, so
 * `Scene.create` is not callable there - `createScene()` is.
 */
export function createScene(options?: SceneOptions): Scene {
  // The casts step around `private`: this factory IS the constructor's public
  // face, split out of the class because statics cannot cross into a worklet.
  const scene = new (Scene as unknown as new () => Scene)()
  ;(scene as unknown as { options: SceneOptions | null }).options =
    options ?? null
  return scene
}

/** `Scene.from`, as a plain function - the worklet-callable form. */
export function sceneFrom(roots: SceneNode[], options?: SceneOptions): Scene {
  const scene = createScene(options)
  scene.replaceChildren(roots)
  return scene
}
