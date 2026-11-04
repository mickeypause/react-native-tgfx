import { expect, mock, test } from 'bun:test'

mock.module('react-native-reanimated', () => ({}))
import { TgfxContainer } from '../reconciler/Container'
import { NodeType, type Node } from '../reconciler/types'
import type { FlagBlock } from '../reanimated/slotFlags'

/** A SharedValue whose `.value` read is a cross-runtime hop on Reanimated 4: count them. */
function shared(value: number): {
  sv: { value: number; _isReanimatedSharedValue: true }
  reads: () => number
} {
  let reads = 0
  const sv = {
    _isReanimatedSharedValue: true as const,
    modify() {},
    get value(): number {
      reads++
      return value
    },
  }
  return { sv, reads: () => reads }
}

const block = (values: number[], written: boolean): FlagBlock =>
  ({
    bytes: new Uint8Array([written ? 1 : 0]),
    values: new Float64Array(values),
    slots: {},
    boxed: {},
  }) as unknown as FlagBlock

test('a commit snapshots an animating slot from the mirror, not from .value', () => {
  const x = shared(5)
  const y = shared(6)
  const words: Float64Array[] = []
  const container = new TgfxContainer((buffer) => {
    words.push(new Float64Array(buffer))
  })
  const roots = [
    {
      type: NodeType.Circle,
      props: { cx: x.sv, cy: y.sv, r: 1, color: '#fff' },
      children: [],
    },
  ] as Node[]
  container.setRootDirect(roots)
  // First commit: nothing mirrored yet, so the literals come from .value.
  expect(Array.from(words[0]!.subarray(5, 7))).toEqual([5, 6])
  expect(x.reads()).toBe(1)
  // The driver pushed since; a mirror stands in for the UI thread.
  container.mirror = { values: [x.sv, y.sv], block: block([50, 60], true) }
  container.redraw()
  expect(Array.from(words[1]!.subarray(5, 7))).toEqual([50, 60])
  expect(x.reads()).toBe(1)
  // An unwritten mirror is not trusted; the last program's literals stand in.
  container.mirror = { values: [x.sv, y.sv], block: block([0, 0], false) }
  container.redraw()
  expect(Array.from(words[2]!.subarray(5, 7))).toEqual([50, 60])
  expect(x.reads()).toBe(1)
  // A value this container has never seen is read once.
  const z = shared(9)
  container.setRootDirect([
    {
      type: NodeType.Circle,
      props: { cx: x.sv, cy: z.sv, r: 1, color: '#fff' },
      children: [],
    },
  ] as Node[])
  expect(Array.from(words[3]!.subarray(5, 7))).toEqual([50, 9])
  expect(x.reads()).toBe(1)
  expect(z.reads()).toBe(1)
})

test('a JS-thread push takes mirrored values the same way', () => {
  const x = shared(5)
  const pushed: number[][] = []
  const container = new TgfxContainer(() => {})
  container.onValues = (values) => pushed.push(values.slice())
  container.setRootDirect([
    {
      type: NodeType.Rect,
      props: { x: x.sv, y: 0, width: 1, height: 1 },
      children: [],
    },
  ] as Node[])
  container.mirror = { values: [x.sv], block: block([7], true) }
  const before = x.reads()
  container.pushValues()
  expect(pushed).toEqual([[7]])
  expect(x.reads()).toBe(before)
})

test('with a driver to follow, an unseen mutable is deferred and the program flagged', () => {
  const x = shared(5)
  const words: Float64Array[] = []
  const container = new TgfxContainer((buffer) => {
    words.push(new Float64Array(buffer))
  })
  container.deferReads = true
  const pushed: number[][] = []
  container.onValues = (values) => pushed.push(values.slice())
  container.setRootDirect([
    {
      type: NodeType.Rect,
      props: { x: x.sv, y: 0, width: 1, height: 1 },
      children: [],
    },
  ] as Node[])
  expect(x.reads()).toBe(0)
  expect(container.unresolved).toBe(true)
  expect(words[0]![5]).toBe(0)
  // The push that would carry a placeholder is left to the driver.
  container.pushValues()
  expect(pushed).toEqual([])
  // The driver pushed since: the mirror resolves it and the hold is over.
  container.mirror = { values: [x.sv], block: block([7], true) }
  container.redraw()
  expect(container.unresolved).toBe(false)
  expect(words[1]![5]).toBe(7)
  container.pushValues()
  expect(pushed).toEqual([[7]])
  expect(x.reads()).toBe(0)
})
