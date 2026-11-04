'worklet'

import { colorChannels, resolveColor, type Color } from './compiler/colors'
import { BLEND_MODES, blendCode } from './compiler/filters'
import { imageHandle } from './compiler/image'
import { CAPS, JOINS, STYLES } from './compiler/paint'
import type { Transform, TransformOp } from './compiler/transform'
import { isSharedValue, type MaybeSharedValue } from './compiler/values'
import type { SceneWriter } from './compiler/writer'
import {
  getRunOnJS,
  liveFromBlock,
  liveStringFromBlock,
  makeLive,
  makeLiveString,
  makeShared,
  writeLiveNumber,
  writeLiveString,
  type LiveString,
  type LiveValue,
} from './reanimated/liveValue'
import { flagsOf, makeFlagBlock } from './reanimated/slotFlags'
import { NodeType, type Node } from './reconciler/types'
import type { UniformValue } from './types'

const SLOT_PROPS = [
  'x',
  'y',
  'width',
  'height',
  'cx',
  'cy',
  'r',
  'rx',
  'ry',
  'x1',
  'y1',
  'x2',
  'y2',
  'strokeWidth',
  'opacity',
  'fontSize',
  'visible',
  // Filter parameters, for a handle on <Blur> / <DropShadow>.
  'blur',
  'blurX',
  'blurY',
  'dx',
  'dy',
  'threshold',
  // Gradient angles, for a handle on <SweepGradient>.
  'startAngle',
  'endAngle',
  'strokeMiter',
  // The backdrop material, for a handle on <BackdropBlur> / <Glass>.
  'refraction',
  'depth',
  'frost',
  'dispersion',
  'splay',
  'lightAngle',
  'lightIntensity',
  // The grain, for a handle on <FilmGrain>. `opacity` and `blendMode` are
  // already slots above; `size` and `seed` are the grain's own.
  'size',
  'density',
  'seed',
] as const

/** Enum props: the slot carries the wire code, seeded from the written name. */
const ENUM_SLOTS = [
  'style',
  'strokeCap',
  'strokeJoin',
  'blendMode',
  'mode',
  'antiAlias',
] as const

/** Vector props split across two slots each. */
const VECTOR_SLOTS = ['start', 'end', 'center'] as const

/** How many gradient stops a handle keeps live slots for. */
const MAX_STOPS = 16

// Appended in this order when the transform lacks them: perspective before the rotations.
const TRANSFORM_SLOTS = [
  'translateX',
  'translateY',
  'perspective',
  'scale',
  'scaleX',
  'scaleY',
  'rotate',
  'skewX',
  'skewY',
  'rotateX',
  'rotateY',
  'translateZ',
] as const

type TransformSlot = (typeof TRANSFORM_SLOTS)[number]

const KEY_INDEX: Record<string, number> = {}
for (const name of SLOT_PROPS) {
  KEY_INDEX[name] = Object.keys(KEY_INDEX).length
}
for (const name of TRANSFORM_SLOTS) {
  KEY_INDEX[name] = Object.keys(KEY_INDEX).length
}
for (const name of ENUM_SLOTS) {
  KEY_INDEX[name] = Object.keys(KEY_INDEX).length
}
for (const name of VECTOR_SLOTS) {
  KEY_INDEX[name] = Object.keys(KEY_INDEX).length
}
for (const name of [
  'color',
  'image',
  'rect',
  'matrix',
  'positions',
  'colors',
  'text',
  'path',
]) {
  KEY_INDEX[name] = Object.keys(KEY_INDEX).length
}
const FLAG_BYTES = Object.keys(KEY_INDEX).length

const UNIFORM_SLOTS = 32

// Fixed value-block layout, so a lazily created pool still lands on the same
// indices every runtime's view agrees on.
const OFF_EXTRA = SLOT_PROPS.length + TRANSFORM_SLOTS.length
const N_EXTRA = ENUM_SLOTS.length + VECTOR_SLOTS.length * 2 + 5
const OFF_COLOR = OFF_EXTRA + N_EXTRA
const OFF_UNIFORM = OFF_COLOR + 4
const OFF_MATRIX = OFF_UNIFORM + UNIFORM_SLOTS
const OFF_POSITIONS = OFF_MATRIX + 20
const OFF_STOP_COLORS = OFF_POSITIONS + MAX_STOPS
const VALUE_COUNT = OFF_STOP_COLORS + MAX_STOPS * 4
const STRING_COUNT = 2

interface UniformRun {
  at: number
  size: number
}

/**
 * What the container gives a bound handle: the two ways a mounted scene can be
 * changed without React. `pushValues` patches the slots the program already
 * has; `recompile` re-encodes it, which is what a prop with no slot costs.
 */
export interface NodeOwner {
  pushValues(): void
  recompile(): void
}

/**
 * A ref on a drawing node: changes a mounted node without a React render.
 *
 * ```tsx
 * const box = useNode()
 * <Rect ref={box} x={0} y={0} width={40} height={40} />
 * box.set({ x: 120 })
 * ```
 *
 * A prop the program keeps a value slot for is patched in place: one
 * `setValues`, no re-encode, and it works **inside a worklet**, so a gesture can
 * drive it without ever waking the JS thread. That covers the geometry props
 * (`x`, `y`, `width`, `height`, `cx`, `cy`, `r`, `rx`, `ry`, `x1`, `y1`, `x2`,
 * `y2`), `strokeWidth`, `strokeMiter`, `opacity`, `fontSize`, `visible`,
 * `color`, the paint enums (`style`, `strokeCap`, `strokeJoin`, `blendMode`,
 * `antiAlias`), `image` and `rect`, a `<Group>`'s transform operands, a
 * `<Shader>`'s `uniforms`, a gradient's geometry / `positions` / `colors`, a
 * filter's parameters, and `text` / `path` content (string slots, re-shaped or
 * re-parsed on the render thread). A prop with no slot re-encodes the scene -
 * still with no React render, no reconciler commit and no Fabric commit - and
 * from a worklet it hops to the JS thread to do it.
 *
 * Most of the non-geometry props are made live by the first write, not by the
 * `ref`: each costs the render thread work every frame, so that write
 * re-encodes once and every one after it is a slot patch. A `ref` on a
 * `<Shader>` has no other job, so its `uniforms` are live from the first
 * compile. A handle keeps at most 16 gradient stops (longer imperative writes
 * are truncated) and 32 uniform floats (uniforms outside that pool re-encode).
 * Docs: docs/content/docs/imperative.mdx.
 *
 * **The next React commit wins.** An imperative write survives until the canvas
 * re-renders, and then the JSX props are the truth again. Drive a value from one
 * side or the other, not both.
 */
export interface NodeHandle {
  set(props: Record<string, unknown>): void
}

interface Internals {
  /** Bumped by a worklet-side set(): the mapper listens to it, block slots have no listeners. */
  gen: MaybeSharedValue<number>
  apply(node: Node, reseed: boolean, w: SceneWriter): Node
  applyUniforms(
    uniforms: Record<string, UniformValue> | undefined,
    reseed: boolean,
    w: SceneWriter
  ): Record<string, UniformValue>
  bind(owner: NodeOwner, w: SceneWriter): void
}

// Off the handle itself: a worklet clones every own enumerable property, and
// these hold the graph. Lazily and behind a holder: a WeakMap crosses a worklet
// capture as `undefined`, so each runtime builds its own on first use.
const internalsCache = { map: null as WeakMap<NodeHandle, Internals> | null }
const internalsMap = (): WeakMap<NodeHandle, Internals> => {
  'worklet'
  return (internalsCache.map ??= new WeakMap<NodeHandle, Internals>())
}

// Properties, not module `let`s: a worklet body assigning a captured `let`
// compiles to an assignment to a `const` and fails to parse on the UI runtime.
const warned = {
  opOnNonGroup: false,
  fullPool: false,
  nonUniformOnShader: false,
  uniformsOnNode: false,
  uniformNeverLands: false,
}

function warnOpOnNonGroup(key: string): void {
  if (warned.opOnNonGroup) {
    return
  }
  warned.opOnNonGroup = true
  console.warn(
    `[tgfx] '${key}' was written to a handle on a node that is not a <Group>: only a group carries a transform, so the write is dropped. Wrap the node in a <Group> and put the handle on that.`
  )
}

function warnNonUniformOnShader(key: string): void {
  if (warned.nonUniformOnShader) {
    return
  }
  warned.nonUniformOnShader = true
  console.warn(
    `[tgfx] '${key}' was written to a handle on a <Shader>, which is reached only through its uniforms — the write is dropped. Put the handle on the drawing node if you meant to change the node.`
  )
}

function warnUniformsOnNode(): void {
  if (warned.uniformsOnNode) {
    return
  }
  warned.uniformsOnNode = true
  console.warn(
    '[tgfx] `uniforms` was written to a handle on a drawing node, which carries none — the write is dropped. Put the handle on the <Shader> itself.'
  )
}

function warnUniformNeverLands(name: string): void {
  if (warned.uniformNeverLands) {
    return
  }
  warned.uniformNeverLands = true
  console.warn(
    `[tgfx] uniform '${name}' is not one the compiled program carries — the shader source does not declare it, or nothing draws the paint holding it. Writes to it are dropped.`
  )
}

function warnFullPool(name: string): void {
  if (warned.fullPool) {
    return
  }
  warned.fullPool = true
  console.warn(
    `[tgfx] uniform '${name}' is past the ${UNIFORM_SLOTS}-float pool one node handle keeps, so it gets no value slot and every write to it re-encodes the scene. Split the shader across two handles, or pack fewer uniforms into this one.`
  )
}

export function isNodeHandle(value: unknown): value is NodeHandle {
  return (
    typeof value === 'object' &&
    value !== null &&
    (value as { isTgfxNode?: unknown }).isTgfxNode === true
  )
}

export function nodeInternals(handle: NodeHandle): Internals {
  return internalsMap().get(handle)!
}

function slotOfOp(op: Record<string, unknown>): TransformSlot | null {
  if ('rotateZ' in op || 'translate' in op) {
    return 'rotateZ' in op ? 'rotate' : 'translateX'
  }
  for (const name of TRANSFORM_SLOTS) {
    if (name in op) {
      return name
    }
  }
  return null
}

function mergeUniform(
  written: number[] | undefined,
  raw: UniformValue | undefined
): UniformValue {
  if (written === undefined) {
    return raw as UniformValue
  }
  if (raw === undefined) {
    return written as UniformValue
  }
  const base = Array.isArray(raw) ? raw : [raw]
  const width = Math.max(base.length, written.length)
  const out: UniformValue & unknown[] = []
  for (let i = 0; i < width; i++) {
    out.push(written[i] ?? base[i] ?? 0)
  }
  return out
}

function isOpSlot(key: string): key is TransformSlot {
  return (TRANSFORM_SLOTS as readonly string[]).includes(key)
}

export function createHandle(): NodeHandle {
  // Resolved now, or the worklet would capture a remote handle instead of the function.
  const runOnJS = getRunOnJS()
  // One native block per handle: flag bytes, every numeric slot as shared
  // Float64 bytes, and the two string slots. Slot writes are plain stores into
  // memory all runtimes share - a SharedValue written from the JS thread
  // serializes a fresh worklet per write, and at set()-per-frame rates that
  // garbage grows the native heap faster than any GC reclaims it (issue #254).
  const flagBlock = makeFlagBlock(FLAG_BYTES, VALUE_COUNT, STRING_COUNT)
  const boxedFlags = flagBlock?.boxed ?? null
  const flagBytes = flagBlock?.bytes ?? null
  const live = (at: number, initial: number): LiveValue =>
    flagBlock != null
      ? liveFromBlock(flagBlock, at, initial)
      : makeLive(initial)
  const liveString = (at: number, initial: string): LiveString =>
    flagBlock != null
      ? liveStringFromBlock(flagBlock, at, initial)
      : makeLiveString(initial)
  const slots: Record<string, LiveValue> = {}
  let slotAt = 0
  for (const name of SLOT_PROPS) {
    slots[name] = live(slotAt++, name === 'visible' ? 1 : 0)
  }
  for (const name of TRANSFORM_SLOTS) {
    slots[name] = live(slotAt++, name.startsWith('scale') ? 1 : 0)
  }
  // Enum codes, vector axes, the image handle and the source rect are lazy,
  // like the matrix and stop pools: most handles never touch them, and a
  // worklet that captured them as null defers to the JS thread, which still
  // lands the write as a slot patch. Their block indices are fixed by layout,
  // so laziness only defers the descriptor objects.
  let extra: Record<string, LiveValue> | null = null
  const ensureExtras = (): Record<string, LiveValue> => {
    if (extra == null) {
      extra = {}
      let at = OFF_EXTRA
      for (const name of ENUM_SLOTS) {
        extra[name] = live(at++, 0)
      }
      for (const name of VECTOR_SLOTS) {
        extra[`${name}X`] = live(at++, 0)
        extra[`${name}Y`] = live(at++, 0)
      }
      extra.image = live(at++, 0)
      extra.rectX = live(at++, 0)
      extra.rectY = live(at++, 0)
      extra.rectW = live(at++, 0)
      extra.rectH = live(at++, 0)
    }
    return extra
  }
  const colorSlots: LiveValue[] = []
  for (let i = 0; i < 4; i++) {
    colorSlots.push(live(OFF_COLOR + i, i === 3 ? 1 : 0))
  }
  const uniformSlots: LiveValue[] = []
  for (let i = 0; i < UNIFORM_SLOTS; i++) {
    uniformSlots.push(live(OFF_UNIFORM + i, 0))
  }
  // Live string content, for `text` and `path`: a write re-shapes or re-parses
  // on the render thread with no re-encode. Live from the first write.
  const textShared: LiveString = liveString(0, '')
  const pathShared: LiveString = liveString(1, '')
  // Big pools stay lazy: 20 matrix values and 16 stops x 5 would double every
  // handle's cost for props most nodes never carry. Created when first needed
  // on the JS thread; a worklet that captured them as null defers, which still
  // lands as a slot patch pushed from JS.
  let matrixSlots: LiveValue[] | null = null
  let positionSlots: LiveValue[] | null = null
  let stopColorSlots: LiveValue[] | null = null
  // How many stops the mounted program compiled; a count change must re-encode.
  const stopColorCount = makeShared(0)
  const stopPositionCount = makeShared(0)
  // Whether this handle sits on a gradient, so 'colors' means stops there only.
  let onGradient = false
  const ensureMatrix = (): LiveValue[] => {
    if (matrixSlots == null) {
      matrixSlots = []
      for (let i = 0; i < 20; i++) {
        matrixSlots.push(live(OFF_MATRIX + i, i % 6 === 0 ? 1 : 0))
      }
    }
    return matrixSlots
  }
  const ensureStops = (): { pos: LiveValue[]; col: LiveValue[] } => {
    if (positionSlots == null) {
      positionSlots = []
      for (let i = 0; i < MAX_STOPS; i++) {
        positionSlots.push(live(OFF_POSITIONS + i, 0))
      }
    }
    if (stopColorSlots == null) {
      stopColorSlots = []
      for (let i = 0; i < MAX_STOPS * 4; i++) {
        stopColorSlots.push(live(OFF_STOP_COLORS + i, i % 4 === 3 ? 1 : 0))
      }
    }
    return { pos: positionSlots, col: stopColorSlots }
  }
  const patchable: MaybeSharedValue<Record<string, boolean>> = makeShared({})
  // The one SharedValue a worklet-side set() still writes: block writes fire no
  // listener, so this is what wakes the scene's mapper. On the UI runtime the
  // write is host-side and free; a worker runtime pays one scheduled message
  // per set() call, the same wake its writes had to buy before.
  const gen = makeShared(0)
  const uniformBase: MaybeSharedValue<Record<string, UniformRun>> = makeShared(
    {}
  )

  const overrides: Record<string, unknown> = {}
  const ops: Record<string, boolean> = {}
  const pendingUniforms: Record<string, number[]> = {}
  let pendingBases: Record<string, UniformRun> = {}
  const unslottable: Record<string, boolean> = {}
  let poolNext = 0
  let owner: NodeOwner | null = null
  // The owner's writer, so a compile by anyone else is recognisably foreign.
  let boundWriter: SceneWriter | null = null
  let dirty = false
  let boundVia: 'node' | 'shader' | null = null
  let nodeTakesOps: boolean | null = null

  const seedColor = (value: unknown): void => {
    const channels = colorChannels(resolveColor(value as Color))
    for (let i = 0; i < 4; i++) {
      colorSlots[i]!.value = channels[i]!
    }
  }

  const num = (v: unknown): number => {
    'worklet'
    if (typeof v === 'number') {
      return v
    }
    if (v != null && typeof v === 'object' && 'value' in v) {
      return Number((v as { value: unknown }).value)
    }
    return Number(v)
  }

  const components = (raw: unknown): number[] => {
    'worklet'
    if (Array.isArray(raw)) {
      const out: number[] = []
      for (let i = 0; i < raw.length; i++) {
        out.push(num(raw[i]))
      }
      return out
    }
    return [num(raw)]
  }

  /** The wire code for an enum prop's written value, or null when unknown. */
  const enumCodeOnJS = (key: string, value: unknown): number | null => {
    if (typeof value === 'number') {
      return value
    }
    if (key === 'antiAlias') {
      return value === false ? 0 : 1
    }
    if (key === 'blendMode' || key === 'mode') {
      return blendCode(value)
    }
    const table = key === 'style' ? STYLES : key === 'strokeCap' ? CAPS : JOINS
    const code = (table as Record<string, number>)[String(value)]
    return code === undefined ? null : code
  }

  const applyOnJS = (props: Record<string, unknown>): void => {
    let recompile = false
    let touched = false
    for (const key in props) {
      const value = props[key]
      const slot = slots[key]
      if (boundVia === 'shader' && key !== 'uniforms') {
        warnNonUniformOnShader(key)
        continue
      }
      if (key === 'color') {
        seedColor(value)
        if (patchable.value.color === true) {
          touched = true
          continue
        }
        overrides.color = colorSlots
      } else if (key === 'text' || key === 'path') {
        const shared = key === 'text' ? textShared : pathShared
        shared.value = String(value ?? '')
        if (patchable.value[key] === true) {
          touched = true
          continue
        }
        overrides[key] = shared
      } else if (key === 'image') {
        const pool = ensureExtras()
        const handle = typeof value === 'number' ? value : imageHandle(value)
        pool.image!.value = handle
        if (patchable.value.image === true) {
          touched = true
          continue
        }
        overrides.image = pool.image
      } else if (key === 'rect') {
        const pool = ensureExtras()
        const r = (value ?? {}) as Record<string, unknown>
        pool.rectX!.value = Number(r.x) || 0
        pool.rectY!.value = Number(r.y) || 0
        pool.rectW!.value = Number(r.width) || 0
        pool.rectH!.value = Number(r.height) || 0
        if (patchable.value.rect === true) {
          touched = true
          continue
        }
        overrides.rect = {
          x: pool.rectX,
          y: pool.rectY,
          width: pool.rectW,
          height: pool.rectH,
        }
      } else if (key === 'matrix') {
        const pool = ensureMatrix()
        const written = Array.isArray(value) ? value : []
        for (let i = 0; i < 20; i++) {
          pool[i]!.value = Number(written[i]) || 0
        }
        if (patchable.value.matrix === true) {
          touched = true
          continue
        }
        overrides.matrix = pool
      } else if (key === 'positions' && onGradient) {
        const { pos } = ensureStops()
        const written = Array.isArray(value) ? value : []
        const n = Math.min(written.length, MAX_STOPS)
        for (let i = 0; i < n; i++) {
          pos[i]!.value = Number(written[i]) || 0
        }
        // A different stop count than compiled cannot patch: re-encode with it.
        if (
          patchable.value.positions === true &&
          n === stopPositionCount.value
        ) {
          touched = true
          continue
        }
        stopPositionCount.value = n
        overrides.positions = pos.slice(0, n)
      } else if (key === 'colors' && onGradient) {
        const { col } = ensureStops()
        const written = Array.isArray(value) ? value : []
        const n = Math.min(written.length, MAX_STOPS)
        const live: LiveValue[][] = []
        for (let i = 0; i < n; i++) {
          const channels = colorChannels(resolveColor(written[i] as Color))
          for (let c = 0; c < 4; c++) {
            col[i * 4 + c]!.value = channels[c]!
          }
          live.push(col.slice(i * 4, i * 4 + 4))
        }
        if (patchable.value.colors === true && n === stopColorCount.value) {
          touched = true
          continue
        }
        stopColorCount.value = n
        overrides.colors = live
      } else if (
        (VECTOR_SLOTS as readonly string[]).includes(key) &&
        value != null &&
        typeof value === 'object' &&
        !isSharedValue(value)
      ) {
        const v = value as { x?: unknown; y?: unknown }
        const pool = ensureExtras()
        const sx = pool[`${key}X`]!
        const sy = pool[`${key}Y`]!
        sx.value = Number(v.x) || 0
        sy.value = Number(v.y) || 0
        if (patchable.value[key] === true) {
          touched = true
          continue
        }
        overrides[key] = { x: sx, y: sy }
      } else if (key === 'start' || key === 'end') {
        // A numeric start/end is a sweep gradient's angle.
        const angle = key === 'start' ? slots.startAngle! : slots.endAngle!
        angle.value = Number(value) || 0
        if (
          patchable.value[key === 'start' ? 'startAngle' : 'endAngle'] === true
        ) {
          touched = true
          continue
        }
        overrides[key] = angle
      } else if ((ENUM_SLOTS as readonly string[]).includes(key)) {
        const code = enumCodeOnJS(key, value)
        if (code == null) {
          continue
        }
        const pool = ensureExtras()
        pool[key]!.value = code
        if (patchable.value[key] === true) {
          touched = true
          continue
        }
        overrides[key] = pool[key]
      } else if (key === 'visible' && typeof value === 'boolean') {
        slots.visible!.value = value ? 1 : 0
        if (patchable.value.visible === true) {
          touched = true
          continue
        }
        overrides.visible = slots.visible
      } else if (key === 'uniforms') {
        if (boundVia === 'node') {
          warnUniformsOnNode()
          continue
        }
        const written = value as Record<string, UniformValue> | undefined
        let regrow = false
        for (const name in written) {
          const wrote = components(written[name])
          const run = uniformBase.value[name]
          const fits = Math.min(wrote.length, run?.size ?? 0)
          for (let i = 0; i < fits; i++) {
            uniformSlots[run!.at + i]!.value = wrote[i]!
          }
          if (fits > 0) {
            touched = true
          }
          if (wrote.length > (run?.size ?? 0)) {
            if (unslottable[name] === true) {
              pendingUniforms[name] = wrote
              warnUniformNeverLands(name)
              continue
            }
            const prev = pendingUniforms[name]
            if (prev == null || prev.length !== wrote.length) {
              pendingUniforms[name] = wrote
              regrow = true
            } else {
              for (let i = 0; i < wrote.length; i++) {
                if (prev[i] !== wrote[i]) {
                  prev[i] = wrote[i]!
                  regrow = true
                }
              }
            }
          }
        }
        if (!regrow) {
          continue
        }
      } else if (typeof value === 'number' && slot !== undefined) {
        if (isOpSlot(key) && nodeTakesOps === false) {
          warnOpOnNonGroup(key)
          continue
        }
        slot.value = value
        if (patchable.value[key] === true) {
          touched = true
          continue
        }
        if (isOpSlot(key)) {
          ops[key] = true
        } else {
          overrides[key] = slot
        }
      } else if (overrides[key] !== value) {
        overrides[key] = value
      } else {
        continue
      }
      dirty = true
      recompile = true
    }
    if (owner == null) {
      return
    }
    if (recompile) {
      owner.recompile()
    } else if (touched) {
      // A block write fires no SharedValue listener, so a JS-thread write pushes
      // itself - one direct Nitro call, nothing serialized across runtimes.
      owner.pushValues()
    }
  }

  const clearOverrides = (): void => {
    if (!dirty) {
      return
    }
    for (const key in overrides) {
      delete overrides[key]
    }
    for (const key in ops) {
      delete ops[key]
    }
    for (const name in pendingUniforms) {
      delete pendingUniforms[name]
    }
    for (const name in unslottable) {
      delete unslottable[name]
    }
    dirty = false
  }

  const set = (props: Record<string, unknown>): void => {
    'worklet'
    const onUI = (globalThis as { _WORKLET?: boolean })._WORKLET === true
    if (!onUI) {
      // The JS thread never consults the shared records: a read back this tick is the last compile's.
      applyOnJS(props)
      return
    }
    const flags = boxedFlags != null ? flagsOf(boxedFlags) : null
    let live: Record<string, boolean> | null = null
    let bases: Record<string, UniformRun> | null = null
    let deferred = false
    let servedAny = false
    const served: Record<string, boolean> = {}
    for (const key in props) {
      const value = props[key]
      if (key === 'color') {
        if (flags === null && live === null) {
          live = patchable.value
        }
        if (
          flags !== null ? flags[KEY_INDEX.color!] !== 1 : live!.color !== true
        ) {
          deferred = true
        } else if (typeof value === 'number') {
          const packed = value >>> 0
          writeLiveNumber(colorSlots[0]!, ((packed >>> 16) & 0xff) / 255)
          writeLiveNumber(colorSlots[1]!, ((packed >>> 8) & 0xff) / 255)
          writeLiveNumber(colorSlots[2]!, (packed & 0xff) / 255)
          writeLiveNumber(colorSlots[3]!, ((packed >>> 24) & 0xff) / 255)
          served.color = true
        } else if (Array.isArray(value) && value.length === 4) {
          for (let i = 0; i < 4; i++) {
            writeLiveNumber(colorSlots[i]!, num(value[i]))
          }
          served.color = true
        } else {
          deferred = true
        }
      } else if (key === 'uniforms') {
        if (bases === null) {
          bases = uniformBase.value
        }
        const written = value as Record<string, UniformValue>
        for (const name in written) {
          const run = bases[name]
          const wrote = components(written[name])
          if (run === undefined || wrote.length > run.size) {
            deferred = true
            continue
          }
          for (let i = 0; i < wrote.length; i++) {
            writeLiveNumber(uniformSlots[run.at + i]!, wrote[i]!)
          }
        }
        if (!deferred) {
          served.uniforms = true
        }
      } else {
        if (flags === null && live === null) {
          live = patchable.value
        }
        const flagged = (name: string): boolean => {
          const at = KEY_INDEX[name]
          return flags !== null
            ? at !== undefined && flags[at] === 1
            : live![name] === true
        }
        if (key === 'text' || key === 'path') {
          if (flagged(key) && typeof value === 'string') {
            const shared = key === 'text' ? textShared : pathShared
            writeLiveString(shared, value)
            served[key] = true
          } else {
            deferred = true
          }
        } else if (key === 'image') {
          if (flagged('image') && extra != null && typeof value === 'number') {
            writeLiveNumber(extra.image!, value)
            served[key] = true
          } else {
            deferred = true
          }
        } else if (key === 'rect') {
          const r = value as Record<string, unknown> | null
          if (
            flagged('rect') &&
            extra != null &&
            r != null &&
            typeof r === 'object'
          ) {
            writeLiveNumber(extra.rectX!, num(r.x))
            writeLiveNumber(extra.rectY!, num(r.y))
            writeLiveNumber(extra.rectW!, num(r.width))
            writeLiveNumber(extra.rectH!, num(r.height))
            served[key] = true
          } else {
            deferred = true
          }
        } else if (key === 'matrix') {
          if (
            flagged('matrix') &&
            matrixSlots != null &&
            Array.isArray(value)
          ) {
            for (let i = 0; i < 20; i++) {
              writeLiveNumber(matrixSlots[i]!, num(value[i]))
            }
            served[key] = true
          } else {
            deferred = true
          }
        } else if (key === 'positions') {
          if (
            flagged('positions') &&
            positionSlots != null &&
            Array.isArray(value) &&
            Math.min(value.length, MAX_STOPS) === stopPositionCount.value
          ) {
            const n = Math.min(value.length, positionSlots.length)
            for (let i = 0; i < n; i++) {
              writeLiveNumber(positionSlots[i]!, num(value[i]))
            }
            served[key] = true
          } else {
            deferred = true
          }
        } else if (key === 'colors') {
          // Packed 0xAARRGGBB numbers or [r, g, b, a] arrays; strings need the JS parser.
          let ok =
            flagged('colors') &&
            stopColorSlots != null &&
            Array.isArray(value) &&
            Math.min(value.length, MAX_STOPS) === stopColorCount.value
          if (ok) {
            const list = value as unknown[]
            for (let i = 0; i < list.length && ok; i++) {
              const entry = list[i]
              ok = typeof entry === 'number' || Array.isArray(entry)
            }
            if (ok) {
              const n = Math.min(list.length, MAX_STOPS)
              for (let i = 0; i < n; i++) {
                const entry = list[i]
                if (typeof entry === 'number') {
                  const packed = entry >>> 0
                  writeLiveNumber(
                    stopColorSlots![i * 4]!,
                    ((packed >>> 16) & 0xff) / 255
                  )
                  writeLiveNumber(
                    stopColorSlots![i * 4 + 1]!,
                    ((packed >>> 8) & 0xff) / 255
                  )
                  writeLiveNumber(
                    stopColorSlots![i * 4 + 2]!,
                    (packed & 0xff) / 255
                  )
                  writeLiveNumber(
                    stopColorSlots![i * 4 + 3]!,
                    ((packed >>> 24) & 0xff) / 255
                  )
                } else {
                  for (let c = 0; c < 4; c++) {
                    writeLiveNumber(
                      stopColorSlots![i * 4 + c]!,
                      num((entry as unknown[])[c])
                    )
                  }
                }
              }
              served[key] = true
            }
          }
          if (!ok) {
            deferred = true
          }
        } else if (
          (key === 'start' || key === 'end' || key === 'center') &&
          value != null &&
          typeof value === 'object' &&
          !Array.isArray(value)
        ) {
          const v = value as { x?: unknown; y?: unknown }
          if (flagged(key) && extra != null) {
            writeLiveNumber(extra[`${key}X`]!, num(v.x))
            writeLiveNumber(extra[`${key}Y`]!, num(v.y))
            served[key] = true
          } else {
            deferred = true
          }
        } else if (key === 'start' || key === 'end') {
          const name = key === 'start' ? 'startAngle' : 'endAngle'
          if (flagged(name) && typeof value === 'number') {
            writeLiveNumber(slots[name]!, value)
            served[key] = true
          } else {
            deferred = true
          }
        } else if (
          key === 'style' ||
          key === 'strokeCap' ||
          key === 'strokeJoin' ||
          key === 'blendMode' ||
          key === 'mode' ||
          key === 'antiAlias'
        ) {
          let code: number | null = typeof value === 'number' ? value : null
          if (code == null) {
            if (key === 'antiAlias') {
              code = value === false ? 0 : 1
            } else if (key === 'blendMode' || key === 'mode') {
              const at = BLEND_MODES.indexOf(value as never)
              code = at < 0 ? null : at
            } else if (key === 'style') {
              code = value === 'stroke' ? 1 : value === 'fill' ? 0 : null
            } else {
              const table: Record<string, number> =
                key === 'strokeCap'
                  ? { butt: 0, round: 1, square: 2 }
                  : { miter: 0, round: 1, bevel: 2 }
              const at = table[String(value)]
              code = at === undefined ? null : at
            }
          }
          if (code != null && flagged(key) && extra != null) {
            writeLiveNumber(extra[key]!, code)
            served[key] = true
          } else {
            deferred = true
          }
        } else if (key === 'visible' && typeof value === 'boolean') {
          if (flagged('visible')) {
            writeLiveNumber(slots.visible!, value ? 1 : 0)
            served[key] = true
          } else {
            deferred = true
          }
        } else if (typeof value === 'number' && flagged(key)) {
          writeLiveNumber(slots[key]!, value)
          served[key] = true
        } else {
          deferred = true
        }
      }
      if (served[key] === true) {
        servedAny = true
      }
    }
    if (servedAny) {
      // Block writes fire no listener; one gen write is what wakes the mapper.
      const scope = globalThis as { __tgfxGenTick?: number }
      scope.__tgfxGenTick = ((scope.__tgfxGenTick ?? 0) + 1) % 0x40000000
      gen.value = scope.__tgfxGenTick
    }
    if (deferred) {
      // Only the unservable keys travel: the whole snapshot would undo what this runtime moved past.
      const rest: Record<string, unknown> = {}
      for (const key in props) {
        if (served[key] !== true) {
          rest[key] = props[key]
        }
      }
      if (runOnJS != null) {
        runOnJS(applyOnJS)(rest)
      } else {
        applyOnJS(rest)
      }
    }
  }

  const hasOps = (): boolean => {
    for (const key in ops) {
      if (ops[key] === true) {
        return true
      }
    }
    return false
  }

  const applyOps = (node: Node): Transform | undefined => {
    if (!hasOps()) {
      return undefined
    }
    const taken: Record<string, boolean> = {}
    const out: TransformOp[] = []
    for (const op of (node.props.transform as Transform | undefined) ?? []) {
      const record = op as Record<string, unknown>
      const name = slotOfOp(record)
      if (
        name === 'translateX' &&
        'translate' in record &&
        taken.translateX !== true &&
        taken.translateY !== true
      ) {
        const pair = record.translate as [unknown, unknown]
        taken.translateX = true
        taken.translateY = true
        out.push({
          translate: [
            ops.translateX === true ? slots.translateX! : pair[0],
            ops.translateY === true ? slots.translateY! : pair[1],
          ],
        } as TransformOp)
        continue
      }
      if (name == null || ops[name] !== true || taken[name] === true) {
        out.push(op)
        continue
      }
      taken[name] = true
      out.push({ [name]: slots[name]! } as unknown as TransformOp)
    }
    for (const name of TRANSFORM_SLOTS) {
      if (ops[name] === true && taken[name] !== true) {
        out.push({ [name]: slots[name]! } as unknown as TransformOp)
      }
    }
    return out
  }

  const handle: NodeHandle = { set }
  Object.defineProperty(handle, 'isTgfxNode', { value: true })
  // React assigns the host instance here; swallowed, or a worklet would clone a node tree.
  Object.defineProperty(handle, 'current', { get: () => null, set: () => {} })

  internalsMap().set(handle, {
    gen,
    apply(node: Node, reseed: boolean, w: SceneWriter): Node {
      // A compile by anyone but the owner - an export writer rendering nodes
      // that are also mounted somewhere - must read the live state, never
      // reset it: clearing overrides or reseeding slots would visibly damage
      // the scene that owns this handle.
      const foreign = owner != null && w !== boundWriter
      // Here, not in the container: an unmounted node is in no compile, and its writes must not outlive the commit that brings it back.
      if (reseed && owner != null && !foreign) {
        clearOverrides()
      }
      const props: Record<string, unknown> = { ...node.props }
      // An unbound handle's slots hold nothing meaningful yet, so its first
      // compile must seed them from the props whatever `reseed` says - that is
      // what lets a Scene's structural commits use reseed=false without
      // snapping already-written handles back to their factory-time props.
      const seed = (reseed && !foreign) || owner == null
      for (const key of SLOT_PROPS) {
        const value = props[key]
        if (typeof value !== 'number') {
          continue
        }
        if (key === 'opacity' && overrides[key] === undefined) {
          continue
        }
        // A write made before this handle had a program survived via `overrides`; do not reseed it.
        if (seed && overrides[key] === undefined) {
          slots[key]!.value = value
        }
        props[key] = slots[key]
      }
      boundVia = 'node'
      nodeTakesOps = node.type === NodeType.Group
      onGradient =
        node.type === NodeType.LinearGradient ||
        node.type === NodeType.RadialGradient ||
        node.type === NodeType.SweepGradient ||
        node.type === NodeType.DiamondGradient
      if (nodeTakesOps) {
        const transform = applyOps(node)
        if (transform !== undefined) {
          props.transform = transform
        }
      }
      Object.assign(props, overrides)
      return { type: node.type, props, children: node.children }
    },
    applyUniforms(
      uniforms: Record<string, UniformValue> | undefined,
      reseed: boolean,
      w: SceneWriter
    ): Record<string, UniformValue> {
      const foreign = owner != null && w !== boundWriter
      if (foreign) {
        // Read-only: existing runs stay live, everything else exports as its
        // merged literal; the owner's pool layout is not touched.
        const out: Record<string, UniformValue> = {}
        const names = new Set([
          ...Object.keys(uniforms ?? {}),
          ...Object.keys(pendingUniforms),
          ...Object.keys(pendingBases),
        ])
        for (const name of names) {
          const existing = pendingBases[name]
          if (existing !== undefined) {
            const live: (number | MaybeSharedValue<number>)[] = []
            for (let i = 0; i < existing.size; i++) {
              live.push(uniformSlots[existing.at + i]!)
            }
            out[name] = live as UniformValue
            continue
          }
          const merged = mergeUniform(pendingUniforms[name], uniforms?.[name])
          if (merged !== undefined) {
            out[name] = merged
          }
        }
        return out
      }
      boundVia = 'shader'
      nodeTakesOps = false
      if (reseed && owner != null) {
        clearOverrides()
      }
      // A commit reclaims the pool: the JSX is the truth again, so the layout is rebuilt from it.
      if (reseed) {
        pendingBases = {}
        poolNext = 0
      }
      const out: Record<string, UniformValue> = {}
      const bases: Record<string, UniformRun> = {}
      const names = new Set([
        ...Object.keys(uniforms ?? {}),
        ...Object.keys(pendingUniforms),
        ...Object.keys(pendingBases),
      ])
      for (const name of names) {
        const raw = uniforms?.[name]
        const written = pendingUniforms[name]
        const declared = Array.isArray(raw) ? raw.length : raw != null ? 1 : 0
        const existing = pendingBases[name]
        const size = Math.max(
          declared,
          written?.length ?? 0,
          existing?.size ?? 0
        )
        // A caller's own SharedValue stays theirs, or their animation dies against a slot nothing drives.
        const caller =
          (Array.isArray(raw) && raw.some(isSharedValue)) || isSharedValue(raw)
        const keep =
          existing !== undefined && existing.size >= size ? existing : null
        const room = keep != null || poolNext + size <= UNIFORM_SLOTS
        if (size === 0 || caller || !room) {
          out[name] = mergeUniform(written, raw)
          if (out[name] === undefined) {
            delete out[name]
          }
          if (!caller && size > 0 && !room) {
            warnFullPool(name)
          }
          continue
        }
        const run = keep ?? { at: poolNext, size }
        if (keep == null) {
          poolNext += size
        }
        bases[name] = run
        const live: (number | MaybeSharedValue<number>)[] = []
        for (let i = 0; i < run.size; i++) {
          if (keep == null) {
            const seed =
              written?.[i] ?? (Array.isArray(raw) ? raw[i] : i === 0 ? raw : 0)
            if (typeof seed === 'number') {
              uniformSlots[run.at + i]!.value = seed
            }
          }
          live.push(uniformSlots[run.at + i]!)
        }
        out[name] = live as UniformValue
      }
      pendingBases = bases
      return out
    },
    bind(nextOwner: NodeOwner, w: SceneWriter): void {
      owner = nextOwner
      boundWriter = w
      const next: Record<string, boolean> = {}
      for (const key of SLOT_PROPS) {
        if (w.hasSlot(slots[key]!)) {
          next[key] = true
        }
      }
      for (const key of TRANSFORM_SLOTS) {
        if (w.hasSlot(slots[key]!)) {
          next[key] = true
        }
      }
      if (extra != null) {
        for (const key of ENUM_SLOTS) {
          if (w.hasSlot(extra[key]!)) {
            next[key] = true
          }
        }
        for (const key of VECTOR_SLOTS) {
          if (w.hasSlot(extra[`${key}X`]!)) {
            next[key] = true
          }
        }
        if (w.hasSlot(extra.image!)) {
          next.image = true
        }
        if (w.hasSlot(extra.rectX!)) {
          next.rect = true
        }
      }
      if (matrixSlots != null && w.hasSlot(matrixSlots[0]!)) {
        next.matrix = true
      }
      if (positionSlots != null && w.hasSlot(positionSlots[0]!)) {
        next.positions = true
      }
      if (stopColorSlots != null && w.hasSlot(stopColorSlots[0]!)) {
        next.colors = true
      }
      if (w.hasStringSlot(textShared)) {
        next.text = true
      }
      if (w.hasStringSlot(pathShared)) {
        next.path = true
      }
      if (w.hasSlot(colorSlots[0]!)) {
        next.color = true
      }
      if (flagBytes !== null) {
        for (const key in KEY_INDEX) {
          flagBytes[KEY_INDEX[key]!] = next[key] === true ? 1 : 0
        }
      }
      patchable.value = next
      const bases: Record<string, UniformRun> = {}
      for (const name in pendingBases) {
        const run = pendingBases[name]!
        if (!w.hasSlot(uniformSlots[run.at]!)) {
          unslottable[name] = true
          continue
        }
        delete unslottable[name]
        bases[name] = run
        // The seed has done its job; left in place it would outrank the slot on a later compile.
        delete pendingUniforms[name]
      }
      uniformBase.value = bases
    },
  })
  return handle
}
