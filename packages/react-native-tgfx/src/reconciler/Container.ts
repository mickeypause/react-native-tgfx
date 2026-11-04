'worklet'

import type { NodeHandle } from '../imperative'
import { compileScene } from '../compiler/compile'
import type { MaybeSharedValue } from '../compiler/values'
import { SceneWriter } from '../compiler/writer'
import { nodeInternals, type NodeOwner } from '../imperative'
import { makeFlagBlock, type FlagBlock } from '../reanimated/slotFlags'
import type { Node } from './types'

const warned = { shared: false }

/** A Reanimated mutable, whose JS-thread read may hop: a plain slot object cannot. */
function isMutable(v: unknown): boolean {
  return typeof (v as { modify?: unknown }).modify === 'function'
}

function sameSet(
  a: readonly MaybeSharedValue<unknown>[],
  b: readonly MaybeSharedValue<unknown>[]
): boolean {
  if (a.length !== b.length) {
    return false
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return false
    }
  }
  return true
}

/** Called on each committed change with the draw program and its animated values. */
export type CommitListener = (
  buffer: ArrayBuffer,
  strings: string[],
  values: MaybeSharedValue<unknown>[],
  stringValues: MaybeSharedValue<string>[]
) => void

/**
 * The reconciler container. Holds the immutable root node list produced by the
 * persistence-mode host config and, on each committed change, compiles it into
 * the binary draw program plus the ordered animated SharedValues.
 */
export class TgfxContainer implements NodeOwner {
  root: Node[] = []
  muted = false
  /**
   * Whether commits bind this container as the refs' owner. A throwaway export
   * container turns it off: stealing ownership would strand the live scene's
   * handles on a muted container and misclassify its next commit as foreign.
   */
  claimRefs = true
  onValues: ((values: number[]) => void) | null = null
  onStrings: ((indices: number[], values: string[]) => void) | null = null
  /**
   * The slots as the UI-thread driver last pushed them, in shared memory.
   *
   * A commit snapshots every SharedValue prop into the program, and on
   * Reanimated 4 a JS-thread read of a value the UI thread has written since
   * is a synchronous hop to that runtime, queued behind whatever frame it is
   * in - N hops per commit, minutes at a few thousand animating slots. The
   * driver mirrors what it pushes here, so the snapshot is a plain load. Flag
   * byte 0 marks the mirror written. Block slots are never taken from it: a
   * JS-thread write to one fires no driver, so the mirror can trail it.
   */
  mirror: { values: MaybeSharedValue<unknown>[]; block: FlagBlock } | null =
    null
  /// What the last program wrote per SharedValue: exactly what native draws
  /// until the driver's first push, so a commit before that push reuses it.
  private lastLiterals: Map<object, number> | null = null
  /**
   * Set by an owner whose UI-thread driver pushes every slot right after each
   * commit (a Canvas with Reanimated). A mutable this container has no value
   * for is then left at its fallback and the program marked `unresolved`, so
   * the view holds it until that push, instead of the commit reading N values
   * across runtimes.
   */
  deferReads = false
  /// The last program carries placeholder literals a push has to complete.
  unresolved = false
  private readonly writer = new SceneWriter()
  /// Rebuilt each commit: hit results come back as index paths and need naming.
  refPaths: { path: number[]; ref: NodeHandle }[] = []
  /// One per bound handle: the SharedValues a worklet-side set() bumps, which is
  /// all a mapper can listen to now that slot values live in shared block bytes.
  gens: MaybeSharedValue<unknown>[] = []
  /// Keyed for hit results, which arrive as paths and would otherwise rescan the list.
  readonly byPath = new Map<string, NodeHandle>()
  private readonly onCommit: CommitListener
  private values: MaybeSharedValue<unknown>[] = []
  private stringValues: MaybeSharedValue<string>[] = []
  private readonly sentStrings: (string | null)[] = []
  private readonly scratch: number[] = []

  constructor(onCommit: CommitListener) {
    this.onCommit = onCommit
  }

  setRootDirect(roots: Node[]): void {
    this.root = roots
    this.redraw()
  }

  redraw(fromReact = true): void {
    if (this.muted) {
      return
    }
    this.writer.known = this.knownValues()
    // A worklet runtime reads mutables in place, so nothing is deferred there.
    this.writer.deferUnknown =
      this.deferReads &&
      (globalThis as { _WORKLET?: boolean })._WORKLET !== true
    const { buffer, strings, values, stringValues, refs, refPaths } =
      compileScene(this.root, this.writer, fromReact)
    this.writer.known = null
    this.unresolved = this.writer.unresolved
    this.syncMirror(values)
    // Mutables only: a plain slot is written synchronously on this thread and
    // must be read fresh, and a block slot lives in shared memory anyway.
    this.lastLiterals = new Map()
    for (let i = 0; i < values.length; i++) {
      const literal = this.writer.literals[i]
      if (literal !== undefined && isMutable(values[i])) {
        this.lastLiterals.set(values[i] as object, literal)
      }
    }
    this.values = values
    this.stringValues = stringValues
    // The program carries current literals, so nothing is pending for the new table.
    this.sentStrings.length = stringValues.length
    for (let i = 0; i < stringValues.length; i++) {
      this.sentStrings[i] = String(stringValues[i]!.value ?? '')
    }
    this.refPaths = refPaths
    this.byPath.clear()
    for (const entry of refPaths) {
      this.byPath.set(entry.path.join(','), entry.ref)
    }
    if (this.claimRefs) {
      this.gens = []
      for (let i = 0; i < refs.length; i++) {
        const ref = refs[i]!
        if (!warned.shared && refs.indexOf(ref) !== i) {
          warned.shared = true
          console.warn(
            '[tgfx] one node handle is on more than one node: they share its value slots, so a write moves all of them together. Use one handle per node.'
          )
        }
        const internals = nodeInternals(ref)
        internals.bind(this, this.writer)
        if (refs.indexOf(ref) === i) {
          this.gens.push(internals.gen)
        }
      }
    }
    this.onCommit(buffer, strings, values, stringValues)
  }

  recompile(): void {
    this.redraw(false)
  }

  pushValues(): void {
    this.pushStrings()
    if (this.values.length === 0) {
      // Nothing to patch, and an empty push still costs the render thread a redraw on Android.
      return
    }
    if (this.unresolved && this.mirror?.block.bytes[0] !== 1) {
      // The driver's first push carries every slot, block slots included.
      return
    }
    const out = this.scratch
    const known = this.knownValues()
    out.length = this.values.length
    for (let i = 0; i < this.values.length; i++) {
      const sv = this.values[i]!
      const k = known?.get(sv as object)
      out[i] = k !== undefined ? k : Number(sv.value)
    }
    this.onValues?.(out)
  }

  private knownValues(): Map<object, number> | null {
    const mirror = this.mirror
    if (mirror == null || mirror.block.bytes[0] !== 1) {
      return this.lastLiterals
    }
    const known = new Map<object, number>()
    for (let i = 0; i < mirror.values.length; i++) {
      const sv = mirror.values[i]!
      const v = mirror.block.values[i]!
      if (isMutable(sv) && Number.isFinite(v)) {
        known.set(sv as object, v)
      }
    }
    return known
  }

  /// Kept across a commit with the same slots: the running driver keeps writing it.
  private syncMirror(values: MaybeSharedValue<unknown>[]): void {
    if (this.mirror != null && sameSet(this.mirror.values, values)) {
      return
    }
    const block = values.length > 0 ? makeFlagBlock(1, values.length) : null
    this.mirror = block == null ? null : { values: values.slice(), block }
  }

  /// The changed string slots only; an unchanged frame sends nothing.
  private pushStrings(): void {
    if (this.stringValues.length === 0 || this.onStrings == null) {
      return
    }
    let indices: number[] | null = null
    let out: string[] | null = null
    for (let i = 0; i < this.stringValues.length; i++) {
      const now = String(this.stringValues[i]!.value ?? '')
      if (now === this.sentStrings[i]) {
        continue
      }
      this.sentStrings[i] = now
      ;(indices ??= []).push(i)
      ;(out ??= []).push(now)
    }
    if (indices != null && out != null) {
      this.onStrings(indices, out)
    }
  }
}
