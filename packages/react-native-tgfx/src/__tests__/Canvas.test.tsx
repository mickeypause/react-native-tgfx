import { beforeEach, expect, mock, test } from 'bun:test'
import React, { StrictMode, useEffect, useLayoutEffect, useState } from 'react'
import TestRenderer, { act } from 'react-test-renderer'
import { nativeMocks } from '../testing/nativeMocks'
;(
  globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true

// Scene word layout is [VERSION, wordCount, patchTableStart, rootNodeCount, ...].
let scenes: number[][] = []
let pushed: number[][] = []
let viewports: [number, number, number][] = []
let invalidated = 0

function FakeView({ hybridRef }: { hybridRef: (ref: unknown) => void }): null {
  useEffect(() => {
    hybridRef({
      setScene: (buffer: ArrayBuffer) => {
        const words = new Float64Array(buffer)
        scenes.push(Array.from(words.subarray(0, words[1])))
      },
      setValues: (values: number[]) => pushed.push(values.slice()),
      setViewport: (viewport: number[]) =>
        viewports.push([viewport[0]!, viewport[1]!, viewport[2]!]),
      invalidate: () => invalidated++,
      snapshot: async (options: { scale: number }) => ({
        data: new ArrayBuffer(0),
        width: options.scale,
        height: options.scale,
        format: 'png',
      }),
    })
  }, [hybridRef])
  return null
}

mock.module('react-native-reanimated', () => ({
  startMapper: () => 0,
  stopMapper: () => {},
  makeMutable: (value: number) => ({
    value,
    _isReanimatedSharedValue: true,
  }),
}))
nativeMocks.hostComponent = FakeView as never

// Synchronous: a top-level await lets another test file's mock.module land first.
const { Canvas } =
  require('../components/Canvas') as typeof import('../components/Canvas')
const { Rect, Path, Text } =
  require('../components/drawings') as typeof import('../components/drawings')

function Star(): React.ReactElement {
  return <Path path="M0 0L1 1" color="gold" />
}

function Toggle({ custom }: { custom: boolean }): React.ReactElement {
  return (
    <Canvas>
      <Rect x={0} y={0} width={10} height={10} color="red" />
      {custom ? <Star /> : null}
    </Canvas>
  )
}

const rootCounts = () => scenes.map((words) => words[3])

const SCENE = (
  <Canvas>
    <Rect x={0} y={0} width={10} height={10} color="red" />
  </Canvas>
)

async function render(
  element: React.ReactElement
): Promise<TestRenderer.ReactTestRenderer> {
  let tree!: TestRenderer.ReactTestRenderer
  await act(async () => {
    tree = TestRenderer.create(element)
  })
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 20))
  })
  return tree
}

beforeEach(() => {
  scenes = []
  pushed = []
  viewports = []
  invalidated = 0
})

// StrictMode re-runs effects on remount but not the render body that builds the root.
test('a static scene reaches the native view after a StrictMode remount', async () => {
  await render(SCENE)
  const plain = scenes.at(-1)
  expect(plain?.[3]).toBe(1)

  scenes = []
  await render(<StrictMode>{SCENE}</StrictMode>)

  expect(scenes.at(-1)).toEqual(plain!)
})

test('switching from the fallback path back to the direct one keeps the scene', async () => {
  const tree = await render(<Toggle custom />)
  expect(scenes.at(-1)![3]).toBe(2)

  scenes = []
  await act(async () => {
    tree.update(<Toggle custom={false} />)
    await new Promise((resolve) => setTimeout(resolve, 50))
  })

  expect(scenes.at(-1)![3]).toBe(1)
  expect(rootCounts()).not.toContain(0)
})

test('the paths keep swapping without losing the scene', async () => {
  const tree = await render(<Toggle custom />)
  for (const custom of [false, true, false]) {
    await act(async () => {
      tree.update(<Toggle custom={custom} />)
      await new Promise((resolve) => setTimeout(resolve, 50))
    })
    expect(scenes.at(-1)![3]).toBe(custom ? 2 : 1)
  }
  expect(rootCounts()).not.toContain(0)
})

// A SurfaceView punches a hole through the window, so the style's colour has to be drawn.
test("the style's backgroundColor becomes a Fill under the scene", async () => {
  await render(
    <Canvas style={{ backgroundColor: '#ff0000' }}>
      <Rect x={0} y={0} width={10} height={10} color="blue" />
    </Canvas>
  )
  const words = scenes.at(-1)!
  expect(words[3]).toBe(2) // the fill, then the rect
  expect(words[4]).toBe(9) // Kind.Fill first
  expect(words[5]).toBe(0xffff0000)

  scenes = []
  await render(
    <Canvas style={[{ padding: 1 }, { backgroundColor: '#00ff00' }]}>
      <Rect x={0} y={0} width={10} height={10} color="blue" />
    </Canvas>
  )
  expect(scenes.at(-1)![5]).toBe(0xff00ff00) // flattened out of an array style

  scenes = []
  await render(SCENE)
  expect(scenes.at(-1)![3]).toBe(1) // no backgroundColor, no extra node
})

test('a background the scene cannot encode is left to the platform view', async () => {
  for (const background of [
    { resource_paths: ['label'] }, // PlatformColor()
    0xff0000ff, // 0xRRGGBBAA to React Native, 0xAARRGGBB to <Fill>
    'transparent',
  ]) {
    scenes = []
    await render(
      <Canvas style={{ backgroundColor: background as string }}>
        <Rect x={0} y={0} width={10} height={10} color="blue" />
      </Canvas>
    )
    expect(scenes.at(-1)![3]).toBe(1)
  }
})

test('a later style entry unsets an earlier backgroundColor', async () => {
  await render(
    <Canvas
      style={[{ backgroundColor: '#ff0000' }, { backgroundColor: undefined }]}
    >
      <Rect x={0} y={0} width={10} height={10} color="blue" />
    </Canvas>
  )
  expect(scenes.at(-1)![3]).toBe(1)
})

test('a background does not recompile the scene on every render', async () => {
  const tree = await render(
    <Canvas style={{ backgroundColor: '#ff0000' }}>
      <Rect x={0} y={0} width={10} height={10} color="blue" />
    </Canvas>
  )
  scenes = []
  for (let i = 0; i < 5; i++) {
    await act(async () => {
      tree.update(
        <Canvas style={{ backgroundColor: '#ff0000' }}>
          <Rect x={0} y={0} width={10} height={10} color="blue" />
        </Canvas>
      )
      await new Promise((resolve) => setTimeout(resolve, 20))
    })
  }
  const withBackground = scenes.length

  const plain = await render(
    <Canvas>
      <Rect x={0} y={0} width={10} height={10} color="blue" />
    </Canvas>
  )
  scenes = []
  for (let i = 0; i < 5; i++) {
    await act(async () => {
      plain.update(
        <Canvas>
          <Rect x={0} y={0} width={10} height={10} color="blue" />
        </Canvas>
      )
      await new Promise((resolve) => setTimeout(resolve, 20))
    })
  }
  // A background must cost no more commits than the same scene without one.
  expect(withBackground).toBe(scenes.length)

  const stable = <Rect x={0} y={0} width={10} height={10} color="blue" />
  const held = await render(
    <Canvas style={{ backgroundColor: '#ff0000' }}>{stable}</Canvas>
  )
  scenes = []
  for (let i = 0; i < 5; i++) {
    await act(async () => {
      held.update(
        <Canvas style={{ backgroundColor: '#ff0000' }}>{stable}</Canvas>
      )
      await new Promise((resolve) => setTimeout(resolve, 20))
    })
  }
  expect(scenes.length).toBe(0)
})

// Fabric can recycle the view into the next canvas before this one's teardown
// runs, so the teardown stays off the view or it blanks that canvas.
test('unmounting tears the root down without writing to the view', async () => {
  let cleaned = 0
  function Child(): React.ReactElement {
    useEffect(
      () => () => {
        cleaned++
      },
      []
    )
    return <Rect x={0} y={0} width={10} height={10} color="red" />
  }
  const direct = await render(SCENE)
  const reconciled = await render(
    <Canvas>
      <Child />
    </Canvas>
  )
  expect(scenes.at(-1)![3]).toBe(1)

  scenes = []
  await act(async () => {
    direct.unmount()
    reconciled.unmount()
    await new Promise((resolve) => setTimeout(resolve, 20))
  })

  expect(cleaned).toBe(1)
  expect(scenes).toEqual([])
})

// A single <Rect> starts at word 4: [kind, x, y, width, height, ...paint].
const X = 5
const { useNode } =
  require('../hooks/useNode') as typeof import('../hooks/useNode')

type NodeHandleLike = ReturnType<typeof useNode>
let box!: NodeHandleLike
let rerender!: () => void

function Handled(): React.ReactElement {
  const [, setTick] = useState(0)
  box = useNode()
  rerender = () => setTick((t) => t + 1)
  return (
    <Canvas>
      <Rect ref={box} x={0} y={0} width={10} height={10} color="red" />
    </Canvas>
  )
}

test('a written prop reaches the engine without a React render or a re-encode', async () => {
  await render(<Handled />)
  expect(scenes.at(-1)![X]).toBe(0)
  scenes = []

  await act(async () => box.set({ x: 120 }))

  // Slots exist for every patchable prop the element carries, so even the first write patches.
  expect(scenes).toEqual([])
  expect(pushed).toEqual([[120, 0, 10, 10]])
})

test('writing a slotted prop again still patches values', async () => {
  await render(<Handled />)
  scenes = []
  await act(async () => box.set({ x: 200 }))
  await act(async () => box.set({ x: 240 }))

  expect(scenes).toEqual([])
  expect(pushed.map((values) => values[0])).toEqual([200, 240])
})

test('a prop with no slot re-encodes every write', async () => {
  await render(<Handled />)
  scenes = []

  await act(async () => box.set({ path: 'M0 0L1 1' }))
  await act(async () => box.set({ path: 'M0 0L2 2' }))

  expect(scenes.length).toBe(2)
  expect(pushed).toEqual([])
})

// Colour costs four slots and a per-frame rebuild, so the first write buys them — the opacity bargain.
test('colour re-encodes once, then patches', async () => {
  await render(<Handled />)
  scenes = []

  await act(async () => box.set({ color: '#ff00ff' }))
  expect(scenes.length).toBe(1)
  pushed = []

  await act(async () => box.set({ color: '#00ff00' }))

  expect(scenes.length).toBe(1)
  expect(pushed).toEqual([[0, 0, 10, 10, 0, 1, 0, 1]])
})

test('a packed colour is patchable from a worklet', async () => {
  await render(<Handled />)
  await act(async () => box.set({ color: '#ff00ff' }))
  scenes = []
  const globals = globalThis as { _WORKLET?: boolean }
  globals._WORKLET = true
  try {
    await act(async () => box.set({ color: 0xff0000ff }))
  } finally {
    delete globals._WORKLET
  }

  await act(async () => box.set({ x: 1 }))
  expect(scenes).toEqual([])
  expect(pushed.at(-1)).toEqual([1, 0, 10, 10, 0, 0, 1, 1])
})

test('a write from a worklet patches the slot and never touches JS', async () => {
  await render(<Handled />)
  scenes = []
  const globals = globalThis as { _WORKLET?: boolean }
  globals._WORKLET = true
  try {
    await act(async () => box.set({ x: 180 }))
  } finally {
    delete globals._WORKLET
  }

  // On the UI thread the write wakes the driver's mapper, so nothing is pushed from here.
  expect(scenes).toEqual([])
  expect(pushed).toEqual([])
  await act(async () => box.set({ y: 5 }))
  expect(pushed).toEqual([[180, 5, 10, 10]])
})

test('the next React commit takes the node back', async () => {
  await render(<Handled />)
  await act(async () => box.set({ x: 120 }))
  expect(pushed.at(-1)![0]).toBe(120)

  await act(async () => {
    rerender()
    await new Promise((resolve) => setTimeout(resolve, 20))
  })

  expect(scenes.at(-1)![X]).toBe(0)
})

test('redraw() forces a frame without a commit', async () => {
  const canvas = React.createRef<import('../types').CanvasHandle>()
  await render(<Canvas ref={canvas}>{SCENE.props.children}</Canvas>)
  scenes = []

  canvas.current!.redraw()

  expect(invalidated).toBe(1)
  expect(scenes).toEqual([])
})

function Boxed({ handle }: { handle: NodeHandleLike }): React.ReactElement {
  return <Rect ref={handle} x={0} y={0} width={10} height={10} color="red" />
}

function BoxedScene(): React.ReactElement {
  box = useNode()
  return (
    <Canvas>
      <Boxed handle={box} />
    </Canvas>
  )
}

test('a handle works through the reconciler path too', async () => {
  await render(<BoxedScene />)
  scenes = []

  await act(async () => box.set({ x: 90 }))

  expect(scenes).toEqual([])
  expect(pushed).toEqual([[90, 0, 10, 10]])
})

// `Text`'s wrap width looks patchable but is written as a literal, so every write re-encodes.
function Labelled(): React.ReactElement {
  box = useNode()
  return (
    <Canvas>
      <Text ref={box} x={0} y={10} text="hi" width={100} />
    </Canvas>
  )
}

test('a slot-shaped prop the compiler writes as a literal re-encodes every time', async () => {
  await render(<Labelled />)
  scenes = []

  await act(async () => box.set({ width: 200 }))
  await act(async () => box.set({ width: 300 }))

  expect(scenes.length).toBe(2)
  expect(scenes.at(-1)!).toContain(300)
})

function Toggled({ show }: { show: boolean }): React.ReactElement {
  box = useNode()
  return (
    <Canvas>
      {show ? (
        <Rect ref={box} x={0} y={0} width={10} height={10} color="red" />
      ) : null}
    </Canvas>
  )
}

test('a write made while the node is unmounted does not outlive its remount', async () => {
  const tree = await render(<Toggled show />)
  await act(async () => {
    tree.update(<Toggled show={false} />)
    await new Promise((resolve) => setTimeout(resolve, 20))
  })

  await act(async () => box.set({ color: '#00ff00' }))
  scenes = []
  await act(async () => {
    tree.update(<Toggled show />)
    await new Promise((resolve) => setTimeout(resolve, 20))
  })

  expect(scenes.at(-1)!).toContain(4294901760)
})

// A layout effect runs before the passive effect the canvas compiles in, so there is no program yet.
function EarlyWrite(): React.ReactElement {
  box = useNode()
  useLayoutEffect(() => {
    box.set({ color: '#00ff00' })
  }, [])
  return (
    <Canvas>
      <Rect ref={box} x={0} y={0} width={10} height={10} color="red" />
    </Canvas>
  )
}

test('a write made before the first compile survives it', async () => {
  await render(<EarlyWrite />)

  expect(scenes.at(-1)!).toContain(4278255360)
})

const { Group } =
  require('../components/Group') as typeof import('../components/Group')
const { Fill } =
  require('../components/drawings') as typeof import('../components/drawings')
const { Shader } =
  require('../components/Paint') as typeof import('../components/Paint')

function Spun(): React.ReactElement {
  box = useNode()
  return (
    <Canvas>
      <Group ref={box} transform={[{ rotate: 0 }]}>
        <Rect x={0} y={0} width={10} height={10} color="red" />
      </Group>
    </Canvas>
  )
}

test('a transform operand re-encodes once, then patches', async () => {
  await render(<Spun />)
  scenes = []

  await act(async () => box.set({ rotate: 1 }))
  expect(scenes.length).toBe(1)
  pushed = []

  await act(async () => box.set({ rotate: 2 }))

  expect(scenes.length).toBe(1)
  expect(pushed).toEqual([[2]])
})

// A `ref` alone must not make the matrix live: one slot costs a matrix rebuild every frame.
test('an unwritten transform stays folded into a literal matrix', async () => {
  await render(<Spun />)

  await act(async () => box.set({ x: 5 }))

  expect(pushed).toEqual([])
})

function Tilted(): React.ReactElement {
  box = useNode()
  return (
    <Canvas>
      <Group
        ref={box}
        preserve3d
        transform={[{ perspective: 900 }, { rotateY: 0 }]}
      >
        <Rect x={0} y={0} width={10} height={10} color="red" />
      </Group>
    </Canvas>
  )
}

test('a 3D transform operand re-encodes once, then patches', async () => {
  await render(<Tilted />)
  scenes = []

  await act(async () => box.set({ rotateY: 1 }))
  expect(scenes.length).toBe(1)
  pushed = []

  await act(async () => box.set({ rotateY: 2 }))

  expect(scenes.length).toBe(1)
  expect(pushed).toEqual([[2]])
})

function Flat(): React.ReactElement {
  box = useNode()
  return (
    <Canvas>
      <Group ref={box} transform={[{ translate: [10, 10] }]}>
        <Rect x={0} y={0} width={10} height={10} color="red" />
      </Group>
    </Canvas>
  )
}

test('a perspective written through a handle lands before the rotations', async () => {
  await render(<Flat />)

  await act(async () => box.set({ rotateY: 1, perspective: 900 }))

  // kind | opCount | (type a b) x3: translate, perspective, rotateY
  const words = scenes.at(-1)!
  expect(words[5]).toBe(3)
  expect([words[6], words[9], words[12]]).toEqual([0, 7, 6])
})

function Scaled(): React.ReactElement {
  box = useNode()
  return (
    <Canvas>
      <Group ref={box}>
        <Rect x={0} y={0} width={10} height={10} color="red" />
      </Group>
    </Canvas>
  )
}

test('an operand the group does not declare is appended', async () => {
  await render(<Scaled />)
  scenes = []

  await act(async () => box.set({ scale: 3 }))
  await act(async () => box.set({ scale: 4 }))

  expect(scenes.length).toBe(1)
  expect(pushed).toEqual([[4]])
})

const SOURCE = `
uniform float2 uCenter;
uniform float uRadius;
half4 main(float2 pos) {
  return half4(half(length(pos - uCenter) / uRadius), 0.0, 0.0, 1.0);
}
`

function Shaded(): React.ReactElement {
  box = useNode()
  return (
    <Canvas>
      <Fill>
        <Shader
          ref={box}
          source={SOURCE}
          uniforms={{ uCenter: [0, 0], uRadius: 1 }}
        />
      </Fill>
    </Canvas>
  )
}

// A `ref` on a <Shader> has one job, so its uniforms are live from the first compile.
test('shader uniforms are slot-patched, never re-encoded', async () => {
  await render(<Shaded />)
  scenes = []

  await act(async () => box.set({ uniforms: { uCenter: [4, 5] } }))
  await act(async () => box.set({ uniforms: { uCenter: [6, 7], uRadius: 8 } }))

  expect(scenes).toEqual([])
  expect(pushed).toEqual([
    [4, 5, 1],
    [6, 7, 8],
  ])
})

test('a uniform write from a worklet never touches JS', async () => {
  await render(<Shaded />)
  scenes = []
  const globals = globalThis as { _WORKLET?: boolean }
  globals._WORKLET = true
  try {
    await act(async () => box.set({ uniforms: { uRadius: 9 } }))
  } finally {
    delete globals._WORKLET
  }

  await act(async () => box.set({ uniforms: { uCenter: [1, 2] } }))
  expect(scenes).toEqual([])
  expect(pushed.at(-1)).toEqual([1, 2, 9])
})

const sv = (value: number) => ({
  value,
  _isReanimatedSharedValue: true as const,
})

const TWO = `
uniform float uA;
uniform float uB;
half4 main(float2 pos) { return half4(half(uA), half(uB), 0.0, 1.0); }
`

let caller!: { value: number }

function CallerUniform(): React.ReactElement {
  box = useNode()
  return (
    <Canvas>
      <Fill>
        <Shader ref={box} source={TWO} uniforms={{ uA: caller, uB: 5 }} />
      </Fill>
    </Canvas>
  )
}

// A ref must not take over a uniform the caller is already animating, leaving it driven by nothing.
test("a ref does not evict a caller's own SharedValue uniform", async () => {
  caller = sv(0.9)
  await render(<CallerUniform />)

  await act(async () => box.set({ uniforms: { uB: 7 } }))
  expect(pushed.at(-1)).toEqual([0.9, 7])

  caller.value = 0.5
  await act(async () => box.set({ uniforms: { uB: 8 } }))

  expect(pushed.at(-1)).toEqual([0.5, 8])
})

const VEC = `
uniform float2 uA;
uniform float uB;
half4 main(float2 pos) { return half4(half(uA.x), half(uA.y), half(uB), 1.0); }
`

function WidenedUniform(): React.ReactElement {
  box = useNode()
  return (
    <Canvas>
      <Fill>
        <Shader ref={box} source={VEC} uniforms={{ uA: 1, uB: 3 }} />
      </Fill>
    </Canvas>
  )
}

// Widening shifts every later base, and the recompile does not reseed from the JSX.
test('widening a uniform keeps its own write and its neighbour', async () => {
  await render(<WidenedUniform />)
  scenes = []

  await act(async () => box.set({ uniforms: { uA: [7, 8] } }))

  const words = scenes.at(-1)!
  expect(words).toContain(7)
  expect(words).toContain(8)
  expect(words).toContain(3)
})

test('an animated colour channel is unwrapped, not stored as NaN', async () => {
  await render(<Handled />)
  const red = sv(1)

  // The first write seeds through the parser; the second takes the worklet fast path.
  await act(async () => box.set({ color: [red, 0, 0, 1] }))
  await act(async () => box.set({ color: [red, 0, 0, 1] }))

  expect(pushed.at(-1)!.every(Number.isFinite)).toBe(true)
  expect(pushed.at(-1)!.slice(4)).toEqual([1, 0, 0, 1])
})

function Untransformable(): React.ReactElement {
  box = useNode()
  return (
    <Canvas>
      <Rect ref={box} x={0} y={0} width={10} height={10} color="red" />
    </Canvas>
  )
}

// Only a <Group> gets a matrix, so asking for a recompile would re-encode on every write, forever.
test('a transform write to a non-Group is dropped without re-encoding', async () => {
  await render(<Untransformable />)
  scenes = []

  await act(async () => box.set({ translateX: 5 }))
  await act(async () => box.set({ translateX: 9 }))
  await act(async () => box.set({ translateX: 13 }))

  expect(scenes).toEqual([])
  expect(pushed).toEqual([])
})

const THREE = `
uniform float uA;
uniform float uB;
uniform float uC;
half4 main(float2 pos) { return half4(half(uA), half(uB), half(uC), 1.0); }
`

let rerenderGrown!: () => void

function GrowingUniforms(): React.ReactElement {
  const [, setTick] = useState(0)
  box = useNode()
  rerenderGrown = () => setTick((t) => t + 1)
  return (
    <Canvas>
      <Fill>
        {/* uA and uC are declared by the shader but not by the JSX, so each
            needs a run handed over before it can be patched. */}
        <Shader ref={box} source={THREE} uniforms={{ uB: 3 }} />
      </Fill>
    </Canvas>
  )
}

// The carried value must be consumed by the compile that hands over the run, or it outranks the slot.
test('a uniform does not snap back to the value that won it its slot', async () => {
  await render(<GrowingUniforms />)

  await act(async () => box.set({ uniforms: { uA: 1 } }))
  await act(async () => box.set({ uniforms: { uA: 9 } }))
  await act(async () => box.set({ uniforms: { uC: 5 } }))
  pushed = []
  await act(async () => box.set({ uniforms: { uB: 11 } }))

  expect(pushed.at(-1)).toEqual([9, 11, 5])
})

// Growth appends rather than relocating, so nothing reads back a slot the UI thread may own.
test('growing one uniform leaves its neighbours where they are', async () => {
  await render(<GrowingUniforms />)

  await act(async () => box.set({ uniforms: { uA: 4 } }))
  await act(async () => box.set({ uniforms: { uB: 6 } }))
  await act(async () => box.set({ uniforms: { uC: 8 } }))
  pushed = []
  await act(async () => box.set({ uniforms: { uB: 6 } }))

  expect(pushed.at(-1)).toEqual([4, 6, 8])
})

function ShaderOnly(): React.ReactElement {
  box = useNode()
  return (
    <Canvas>
      <Fill>
        <Shader ref={box} source={TWO} uniforms={{ uA: 1, uB: 2 }} />
      </Fill>
    </Canvas>
  )
}

// A <Shader> is reached through applyUniforms alone, so anything else re-encodes for nothing.
test('a non-uniform write to a shader handle is dropped, not re-encoded', async () => {
  await render(<ShaderOnly />)
  scenes = []

  await act(async () => box.set({ x: 5 }))
  await act(async () => box.set({ translateX: 7 }))
  await act(async () => box.set({ color: '#ff0000' }))

  expect(scenes).toEqual([])
  expect(pushed).toEqual([])
})

// The commit is the source of truth for uniforms too, including names the JSX does not declare.
test('a React commit drops an imperatively grown uniform', async () => {
  await render(<GrowingUniforms />)

  await act(async () => box.set({ uniforms: { uA: 9 } }))
  pushed = []
  await act(async () => box.set({ uniforms: { uB: 3 } }))
  expect(pushed.at(-1)).toEqual([9, 3])

  scenes = []
  await act(async () => {
    rerenderGrown()
    await new Promise((resolve) => setTimeout(resolve, 20))
  })

  pushed = []
  await act(async () => box.set({ uniforms: { uB: 3 } }))
  expect(pushed.at(-1)).toEqual([3])
})

const ONE = `
uniform float uA;
half4 main(float2 pos) { return half4(half(uA), 0.0, 0.0, 1.0); }
`

function ShaderInGroup(): React.ReactElement {
  box = useNode()
  return (
    <Canvas>
      <Group>
        {/* No drawing descendant, so this shader's paint is never written and
            the run it is granted never reaches the program. */}
        <Shader ref={box} source={ONE} uniforms={{ uA: 0 }} />
      </Group>
    </Canvas>
  )
}

// A run granted from bookkeeping alone may never reach the program, and the value must survive.
test('a uniform whose run never reaches the program stops re-encoding', async () => {
  await render(<ShaderInGroup />)
  scenes = []

  for (let i = 0; i < 5; i++) {
    await act(async () => box.set({ uniforms: { uA: 3 } }))
  }

  expect(scenes.length).toBeLessThanOrEqual(1)
})

function UniformsOnNode(): React.ReactElement {
  box = useNode()
  return (
    <Canvas>
      {/* The handle is on the drawing node, not on the <Shader>. */}
      <Fill ref={box}>
        <Shader source={ONE} uniforms={{ uA: 0 }} />
      </Fill>
    </Canvas>
  )
}

// apply() never reads uniforms, so the re-encode is byte-identical and the write lands nowhere.
test('uniforms written to a drawing-node handle are dropped, not re-encoded', async () => {
  await render(<UniformsOnNode />)
  scenes = []

  await act(async () => box.set({ uniforms: { uA: 1 } }))
  await act(async () => box.set({ uniforms: { uA: 2 } }))
  await act(async () => box.set({ uniforms: { uA: 3 } }))

  expect(scenes).toEqual([])
  expect(pushed).toEqual([])
})

// A bad scale reaches the GPU as "could not allocate a scaled target", blaming the device.
test('a non-positive snapshot scale is rejected at the boundary', async () => {
  const handle = React.createRef<import('../types').CanvasHandle>()
  await render(<Canvas ref={handle}>{null}</Canvas>)
  for (const scale of [0, -1, NaN]) {
    await expect(handle.current!.snapshot({ scale })).rejects.toThrow(
      /scale must be a positive finite number/
    )
  }
  await expect(
    handle.current!.snapshot({ rect: { x: NaN, y: 0, width: 1, height: 1 } })
  ).rejects.toThrow(/rect must have four finite numbers/)
  const snap = await handle.current!.snapshot({ scale: 2 })
  expect(snap.width).toBe(2)
})

// Unconditional on attach, identity included: a recycled native view resets its
// viewport in clearScene(), which both platforms defer to detach, so this side
// cannot know whether that has landed before the next canvas mounts.
test('an unset viewport still claims the identity once on attach', async () => {
  await render(SCENE)
  expect(viewports).toEqual([[1, 0, 0]])
})

test('a commit that does not move the viewport leaves an imperative write alone', async () => {
  const ref = React.createRef<import('../types').CanvasHandle>()
  const scene = (width: number) => (
    <Canvas ref={ref} viewport={{ zoom: 2, x: 4, y: 6 }}>
      <Rect x={0} y={0} width={width} height={10} color="red" />
    </Canvas>
  )
  const tree = await render(scene(10))
  await act(async () => {
    ref.current!.setViewport({ zoom: 3 })
  })
  // Re-rendered for a reason that has nothing to do with the viewport: folding
  // the prop back in here would silently drop the zoom to 2 again.
  await act(async () => {
    tree.update(scene(20))
  })
  viewports = []
  await act(async () => {
    ref.current!.setViewport({ x: 10 })
  })
  expect(viewports).toEqual([[3, 10, 6]])
})

test('a static viewport is pushed once, and again only when it moves', async () => {
  const scene = (zoom: number) => (
    <Canvas viewport={{ zoom, x: -10, y: -20 }}>
      <Rect x={0} y={0} width={10} height={10} color="red" />
    </Canvas>
  )
  const tree = await render(scene(2))
  expect(viewports).toEqual([[2, -10, -20]])
  viewports = []
  // A commit that leaves the viewport alone must not re-push it.
  await act(async () => {
    tree.update(scene(2))
  })
  expect(viewports).toEqual([])
  await act(async () => {
    tree.update(scene(3))
  })
  expect(viewports).toEqual([[3, -10, -20]])
})

// The UI-thread driver owns a SharedValue viewport; the prop path must stay at
// the identity and not push the shared value's fields from the JS side.
test('a shared value viewport keeps the prop path at the identity', async () => {
  const vp = {
    value: { zoom: 3, x: 9, y: 9 },
    _isReanimatedSharedValue: true as const,
  }
  await render(
    <Canvas viewport={vp}>
      <Rect x={0} y={0} width={10} height={10} color="red" />
    </Canvas>
  )
  expect(viewports).toEqual([[1, 0, 0]])
})

test('a non-positive or non-finite field falls back rather than reaching native', async () => {
  await render(
    <Canvas viewport={{ zoom: 0, x: Number.NaN, y: 5 }}>
      <Rect x={0} y={0} width={10} height={10} color="red" />
    </Canvas>
  )
  // Native rejects a bad triple whole, which would strand y at 0 as well.
  expect(viewports).toEqual([[1, 0, 5]])
})

test('setViewport merges with the fields it omits', async () => {
  const ref = React.createRef<import('../types').CanvasHandle>()
  await render(
    <Canvas ref={ref} viewport={{ zoom: 2, x: 4, y: 6 }}>
      <Rect x={0} y={0} width={10} height={10} color="red" />
    </Canvas>
  )
  await act(async () => {
    ref.current!.setViewport({ zoom: 3 })
  })
  expect(viewports).toEqual([
    [2, 4, 6],
    [3, 4, 6],
  ])
  // Writing the same triple again is not a bridge call.
  viewports = []
  await act(async () => {
    ref.current!.setViewport({ zoom: 3 })
  })
  expect(viewports).toEqual([])
})

test('the fonts prop withholds children until the faces register', async () => {
  const previous = nativeMocks.hybridObject
  let resolveLoad!: (family: string) => void
  const loading = new Promise<string>((r) => {
    resolveLoad = r
  })
  nativeMocks.hybridObject = () => ({
    registerFromFile: () => loading,
  })
  try {
    await render(
      <Canvas fonts={{ Playfair: ['file:///Playfair-Regular.ttf'] }}>
        <Rect x={0} y={0} width={10} height={10} color="red" />
      </Canvas>
    )
    // Gated: the committed program has no root nodes yet.
    expect(scenes[scenes.length - 1]![3]).toBe(0)

    await act(async () => {
      resolveLoad('Playfair')
      await loading
    })
    expect(scenes[scenes.length - 1]![3]).toBe(1)
  } finally {
    nativeMocks.hybridObject = previous
  }
})

test('a failed fonts load warns and renders the children anyway', async () => {
  const previous = nativeMocks.hybridObject
  nativeMocks.hybridObject = () => ({
    registerFromFile: () => Promise.reject(new Error('no such font')),
  })
  try {
    await render(
      <Canvas fonts={{ Broken: ['file:///missing.ttf'] }}>
        <Rect x={0} y={0} width={10} height={10} color="red" />
      </Canvas>
    )
    // Fallback text beats a canvas that stays blank forever.
    expect(scenes[scenes.length - 1]![3]).toBe(1)
  } finally {
    nativeMocks.hybridObject = previous
  }
})

test('a canvas without the fonts prop commits its children on the first frame', async () => {
  await render(
    <Canvas>
      <Rect x={0} y={0} width={10} height={10} color="red" />
    </Canvas>
  )
  // Every committed program carried the child: no empty gating frame.
  expect(scenes.length).toBeGreaterThan(0)
  for (const words of scenes) {
    expect(words[3]).toBe(1)
  }
})

test('a remount with already-settled fonts is ready synchronously', async () => {
  const previous = nativeMocks.hybridObject
  nativeMocks.hybridObject = () => ({
    registerFromFile: () => Promise.resolve('Settled'),
  })
  try {
    const fonts = { Settled: ['file:///Settled-Regular.ttf'] }
    await render(
      <Canvas fonts={fonts}>
        <Rect x={0} y={0} width={10} height={10} color="red" />
      </Canvas>
    )
    expect(scenes[scenes.length - 1]![3]).toBe(1)

    // Second canvas, same key: every commit carries the child — no gate frame.
    scenes = []
    await render(
      <Canvas fonts={fonts}>
        <Rect x={0} y={0} width={10} height={10} color="red" />
      </Canvas>
    )
    for (const words of scenes) {
      expect(words[3]).toBe(1)
    }
  } finally {
    nativeMocks.hybridObject = previous
  }
})

test('Image loads its source and draws once the decode lands', async () => {
  const previous = nativeMocks.hybridObject
  let resolveDecode!: (image: unknown) => void
  const decoding = new Promise((r) => {
    resolveDecode = r
  })
  nativeMocks.hybridObject = () => ({
    fromFile: () => decoding,
  })
  try {
    const { Image } =
      require('../components/Image') as typeof import('../components/Image')
    await render(
      <Canvas>
        <Image source="file:///cat.png" x={0} y={0} width={10} height={10} />
      </Canvas>
    )
    // The node mounts at once with handle 0 — draws nothing, keeps refs bound.
    let words = scenes[scenes.length - 1]!
    expect(words[3]).toBe(1)
    expect(words[9]).toBe(0)

    await act(async () => {
      resolveDecode({ handle: 42, width: 10, height: 10, dispose: () => {} })
      await decoding
    })
    words = scenes[scenes.length - 1]!
    expect(words[3]).toBe(1)
    expect(words[9]).toBe(42) // kind x y w h, then the loaded handle
  } finally {
    nativeMocks.hybridObject = previous
  }
})

test('Image with a failing source warns and stays empty', async () => {
  const previous = nativeMocks.hybridObject
  nativeMocks.hybridObject = () => ({
    fromFile: () => Promise.reject(new Error('404')),
  })
  try {
    const { Image } =
      require('../components/Image') as typeof import('../components/Image')
    await render(
      <Canvas>
        <Image
          source="file:///missing.png"
          x={0}
          y={0}
          width={10}
          height={10}
        />
      </Canvas>
    )
    // The node stays mounted with handle 0: present, drawing nothing.
    const words = scenes[scenes.length - 1]!
    expect(words[3]).toBe(1)
    expect(words[9]).toBe(0)
  } finally {
    nativeMocks.hybridObject = previous
  }
})
