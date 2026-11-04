import type { MaybeSharedValue } from '../compiler/values'
import { reanimated, type Reanimated } from './peer'
import { slotsOf, valuesOf, type BoxedSlots, type FlagBlock } from './slotFlags'

/** A value slot this package created, rather than one the caller animates. */
export type LiveValue = MaybeSharedValue<number>

// A set, not a property: `makeMutable` returns Reanimated's object, and branding it mutates a foreign one.
const LIVE = new WeakSet<object>()

/**
 * A live slot backed by the handle's native block instead of a Reanimated
 * SharedValue. A SharedValue written from the JS thread serializes a fresh
 * worklet per write; at the write rates an imperative handle exists for
 * (`set()` on a timer or per gesture frame) that garbage outruns the GC -
 * measured ~170 KB/s of native heap on an idle demo screen - because its
 * native weight is invisible to the JS heaps. A block slot write is one
 * aligned 8-byte store into memory every runtime shares: no serialization, no
 * allocation, nothing for a GC to chase.
 *
 * On the runtime that created it, `value` is a real accessor over the block,
 * so every existing `.value` read and write - the compiler snapshot, seeding,
 * `applyOnJS` - works unchanged. A worklet capture clones it, which freezes
 * `value` into a stale plain number; worklet-side code must go through
 * {@link readLiveNumber}/{@link writeLiveNumber}, which reach the live bytes
 * through the boxed hybrid that survives the capture.
 */
interface BlockSlot extends MaybeSharedValue<number> {
  __tgfxBoxed: BoxedSlots
  __tgfxAt: number
}

interface BlockStringSlot extends MaybeSharedValue<string> {
  __tgfxBoxed: BoxedSlots
  __tgfxStrAt: number
}

/**
 * A slot object whose keys cannot be redefined. In dev, react-native-worklets
 * redefines every configurable key of an object a worklet captures - on the
 * original, not only the clone - as a getter frozen at capture time and a
 * setter that only warns. A captured block slot then dropped every JS-thread
 * write. Non-configurable keys are skipped, and stay enumerable so the clone
 * still carries `__tgfxBoxed` and the slot index.
 */
function pinnedSlot<T extends object>(keys: PropertyDescriptorMap): T {
  const slot = {}
  for (const key in keys) {
    Object.defineProperty(slot, key, {
      ...keys[key],
      enumerable: true,
      configurable: false,
    })
  }
  return slot as T
}

export function liveFromBlock(
  block: FlagBlock,
  at: number,
  initial: number
): LiveValue {
  block.values[at] = initial
  const slot = pinnedSlot<BlockSlot>({
    _isReanimatedSharedValue: { value: true },
    __tgfxBoxed: { value: block.boxed },
    __tgfxAt: { value: at },
    value: {
      get: (): number => block.values[at]!,
      set: (v: number) => {
        block.values[at] = v
      },
    },
  })
  LIVE.add(slot)
  return slot
}

export function liveStringFromBlock(
  block: FlagBlock,
  at: number,
  initial: string
): LiveString {
  if (initial !== '') {
    block.slots.setString(at, initial)
  }
  const slot = pinnedSlot<BlockStringSlot>({
    _isReanimatedSharedValue: { value: true },
    __tgfxBoxed: { value: block.boxed },
    __tgfxStrAt: { value: at },
    value: {
      get: (): string => block.slots.getString(at),
      set: (v: string) => {
        block.slots.setString(at, v)
      },
    },
  })
  LIVE.add(slot)
  return slot
}

/** Read a live number wherever it is - block slot (cloned or not) or SharedValue. */
export function readLiveNumber(v: MaybeSharedValue<unknown>): number {
  'worklet'
  const at = (v as Partial<BlockSlot>).__tgfxAt
  if (at !== undefined) {
    return valuesOf((v as BlockSlot).__tgfxBoxed)[at]!
  }
  return Number(v.value)
}

/** Write a live number wherever it is; worklet-side block writes land in the shared bytes. */
export function writeLiveNumber(v: MaybeSharedValue<unknown>, x: number): void {
  'worklet'
  const at = (v as Partial<BlockSlot>).__tgfxAt
  if (at !== undefined) {
    valuesOf((v as BlockSlot).__tgfxBoxed)[at] = x
    return
  }
  ;(v as MaybeSharedValue<number>).value = x
}

/** Read a live string - block slot (cloned or not) or SharedValue. */
export function readLiveString(v: MaybeSharedValue<unknown>): string {
  'worklet'
  const at = (v as Partial<BlockStringSlot>).__tgfxStrAt
  if (at !== undefined) {
    return slotsOf((v as BlockStringSlot).__tgfxBoxed).getString(at)
  }
  return String(v.value ?? '')
}

/** Write a live string; worklet-side block writes go through the boxed hybrid. */
export function writeLiveString(v: MaybeSharedValue<unknown>, s: string): void {
  'worklet'
  const at = (v as Partial<BlockStringSlot>).__tgfxStrAt
  if (at !== undefined) {
    slotsOf((v as BlockStringSlot).__tgfxBoxed).setString(at, s)
    return
  }
  ;(v as MaybeSharedValue<string>).value = s
}

/** True for a block-backed slot, which a Reanimated mapper cannot listen to. */
export function isBlockSlot(v: unknown): boolean {
  'worklet'
  return (
    typeof v === 'object' &&
    v !== null &&
    ((v as Partial<BlockSlot>).__tgfxAt !== undefined ||
      (v as Partial<BlockStringSlot>).__tgfxStrAt !== undefined)
  )
}

/**
 * A value the compiled program can patch in place: it is written as a value slot
 * exactly like an animated prop, so writing it again costs one `setValues`
 * instead of a re-encode.
 *
 * A real SharedValue whenever Reanimated is installed, because it has to be
 * writable from the UI thread - both by the animation driver's mapper, which
 * copies a plain object over once and would then push the stale copy back every
 * frame, and by an imperative write inside a worklet.
 */
export function makeShared<T>(value: T): MaybeSharedValue<T> {
  const makeMutable = reanimated().makeMutable
  if (makeMutable != null) {
    return makeMutable(value)
  }
  return {
    value,
    _isReanimatedSharedValue: true,
  } as MaybeSharedValue<T>
}

export function makeLive(value: number): LiveValue {
  const live = makeShared(value)
  LIVE.add(live)
  return live
}

/** A live string slot: `text` and `path` content the engine patches in place. */
export type LiveString = MaybeSharedValue<string>

export function makeLiveString(value: string): LiveString {
  const live = makeShared(value)
  LIVE.add(live)
  return live
}

/** True for a value only this package writes, so no driver is needed for it. */
export function isLive(value: MaybeSharedValue<unknown>): boolean {
  return LIVE.has(value)
}

/**
 * Reanimated's own `runOnJS`, which is itself a worklet - so it can be captured
 * by one of ours and called on the UI thread. A wrapper around it could not: a
 * plain function reached from a worklet is a remote handle, and calling one on
 * the UI runtime throws.
 *
 * Undefined when Reanimated is absent, and never needed then: without it there
 * is no UI runtime to be on in the first place.
 */
export function getRunOnJS(): Reanimated['runOnJS'] {
  return reanimated().runOnJS
}
