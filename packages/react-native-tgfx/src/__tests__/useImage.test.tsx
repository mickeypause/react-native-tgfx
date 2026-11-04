import { beforeEach, expect, test } from 'bun:test'
import React from 'react'
import TestRenderer, { act } from 'react-test-renderer'
import { nativeMocks } from '../testing/nativeMocks'

const DEFAULT_ASSET = { uri: 'https://example.test/a.png' }

interface Made {
  via: 'fromFile' | 'fromBytes' | 'fromBase64' | 'fromSvg'
  arg: string
  handle: FakeImage
}
let made: Made[] = []
let nextHandle = 1

class FakeImage {
  handle = nextHandle++
  width = 16
  height = 9
  disposed = false
  dispose(): void {
    this.disposed = true
  }
}

function record(via: Made['via'], arg: string): FakeImage {
  const handle = new FakeImage()
  made.push({ via, arg, handle })
  return handle
}

nativeMocks.hybridObject = () => ({
  fromFile: async (path: string) => record('fromFile', path),
  fromBytes: async (bytes: ArrayBuffer) =>
    record('fromBytes', new TextDecoder().decode(bytes)),
  fromBase64: async (data: string) => record('fromBase64', data),
  fromSvg: async (markup: string) => record('fromSvg', markup),
  setCacheLimit: () => {},
  cachedBytes: 0,
})

const { useImageState, loadImage } =
  require('../hooks/useImage') as typeof import('../hooks/useImage')

beforeEach(() => {
  made = []
  nativeMocks.assetSource = DEFAULT_ASSET
})

function reply(body: string, ok = true, status = 200) {
  return {
    ok,
    status,
    arrayBuffer: async () => new TextEncoder().encode(body).buffer,
  }
}

async function mount(source: string) {
  let seen!: ReturnType<typeof useImageState>
  function Probe({ src }: { src: string }): null {
    seen = useImageState(src)
    return null
  }
  let tree!: TestRenderer.ReactTestRenderer
  await act(async () => {
    tree = TestRenderer.create(<Probe src={source} />)
  })
  return {
    get state() {
      return seen
    },
    async setSource(next: string) {
      await act(async () => {
        tree.update(<Probe src={next} />)
      })
    },
    async unmount() {
      await act(async () => {
        tree.unmount()
      })
    },
  }
}

test('a successful load reports a handle and stops loading', async () => {
  globalThis.fetch = (async () => reply('hi')) as unknown as typeof fetch
  const probe = await mount('https://example.test/a.png')
  expect(probe.state.data).toBe(made[0]!.handle as never)
  expect(probe.state.error).toBeNull()
  expect(probe.state.loading).toBe(false)
})

test('a remote source hands the raw bytes to the factory, never base64', async () => {
  globalThis.fetch = (async () => reply('hi')) as unknown as typeof fetch
  await mount('https://example.test/a.png')
  expect(made).toHaveLength(1)
  expect(made[0]!.via).toBe('fromBytes')
  expect(made[0]!.arg).toBe('hi')
})

test('a local file is opened natively, so its bytes never enter JS', async () => {
  let fetched = false
  globalThis.fetch = (async () => {
    fetched = true
    return reply('hi')
  }) as unknown as typeof fetch
  await mount('file:///tmp/pic.png')
  expect(fetched).toBe(false)
  expect(made[0]!.via).toBe('fromFile')
  expect(made[0]!.arg).toBe('file:///tmp/pic.png')
})

test('a bundled asset resolving to a bare path is opened as a file', async () => {
  nativeMocks.assetSource = { uri: '/var/app/assets/pic.png' }
  await mount(1 as unknown as string)
  expect(made[0]!.via).toBe('fromFile')
  expect(made[0]!.arg).toBe('/var/app/assets/pic.png')
})

test('an absolute path, as save() writes, is opened as a file', async () => {
  await mount('/data/user/0/app/cache/pic.png')
  expect(made[0]!.via).toBe('fromFile')
  expect(made[0]!.arg).toBe('/data/user/0/app/cache/pic.png')
})

// A JPEG payload starts "/9j/", so a leading slash alone cannot mean "path".
test('a raw base64 source goes to the base64 path even when it starts with /', async () => {
  await mount('/9j/4AAQSkZJRg==')
  expect(made[0]!.via).toBe('fromBase64')
  expect(made[0]!.arg).toBe('/9j/4AAQSkZJRg==')
})

test('line-wrapped base64 that starts with / is still base64', async () => {
  await mount('/9j/4AAQ\nSkZJRg==')
  expect(made[0]!.via).toBe('fromBase64')
})

test('a data: URI is decoded from its payload, not its whole URI', async () => {
  await mount('data:image/png;base64,aGk=')
  expect(made[0]!.via).toBe('fromBase64')
  expect(made[0]!.arg).toBe('aGk=')
})

test('a 404 is a failure, and its body is never decoded as an image', async () => {
  globalThis.fetch = (async () =>
    reply('<html>not found</html>', false, 404)) as unknown as typeof fetch
  const probe = await mount('https://example.test/missing.png')
  expect(probe.state.data).toBeNull()
  expect(probe.state.error?.message).toMatch(/HTTP 404/)
  expect(probe.state.loading).toBe(false)
  expect(made).toHaveLength(0)
})

test('a failing new source clears the previous image rather than keeping it', async () => {
  globalThis.fetch = (async () => reply('hi')) as unknown as typeof fetch
  const probe = await mount('https://example.test/a.png')
  expect(probe.state.data).not.toBeNull()

  globalThis.fetch = (async () =>
    reply('nope', false, 500)) as unknown as typeof fetch
  await probe.setSource('https://example.test/b.png')
  expect(probe.state.data).toBeNull()
  expect(probe.state.error?.message).toMatch(/HTTP 500/)
})

test('a slow new source clears the old image immediately, not when it settles', async () => {
  globalThis.fetch = (async () => reply('hi')) as unknown as typeof fetch
  const probe = await mount('https://example.test/a.png')
  expect(probe.state.data).not.toBeNull()

  globalThis.fetch = (() => new Promise(() => {})) as unknown as typeof fetch
  await probe.setSource('https://example.test/slow.png')
  expect(probe.state.data).toBeNull()
  expect(probe.state.loading).toBe(true)
})

// Decoded pixels are native memory; waiting for GC is how a long list holds every image it passed.
test('a replaced source disposes the handle it replaces', async () => {
  globalThis.fetch = (async () => reply('hi')) as unknown as typeof fetch
  const probe = await mount('https://example.test/a.png')
  const first = made[0]!.handle
  expect(first.disposed).toBe(false)

  await probe.setSource('https://example.test/b.png')
  expect(first.disposed).toBe(true)
  expect(made[1]!.handle.disposed).toBe(false)
})

test('unmounting disposes the handle', async () => {
  globalThis.fetch = (async () => reply('hi')) as unknown as typeof fetch
  const probe = await mount('https://example.test/a.png')
  await probe.unmount()
  expect(made[0]!.handle.disposed).toBe(true)
})

test('an unresolvable asset is an error, not a silent freeze', async () => {
  nativeMocks.assetSource = null
  const probe = await mount(1 as unknown as string)
  expect(probe.state.data).toBeNull()
  expect(probe.state.error?.message).toMatch(/could not resolve asset source/)
  expect(probe.state.loading).toBe(false)
})

test('a slow old source cannot overwrite the result of a newer one', async () => {
  let releaseOld!: (v: unknown) => void
  globalThis.fetch = ((url: string) =>
    url.includes('old')
      ? new Promise((r) => {
          releaseOld = r
        })
      : Promise.resolve(reply('new'))) as unknown as typeof fetch

  const probe = await mount('https://example.test/old.png')
  expect(probe.state.data).toBeNull()

  await probe.setSource('https://example.test/new.png')
  const current = probe.state.data
  expect(current).not.toBeNull()

  // The old fetch lands last: its cleanup already ran, so it must not write, and must release.
  await act(async () => {
    releaseOld(reply('old'))
  })
  expect(probe.state.data).toBe(current as never)
  const stale = made.find((m) => m.arg === 'old')
  expect(stale?.handle.disposed).toBe(true)
})

test('a failed source recovers when a working one replaces it', async () => {
  globalThis.fetch = (async () =>
    reply('nope', false, 404)) as unknown as typeof fetch
  const probe = await mount('https://example.test/missing.png')
  expect(probe.state.error).not.toBeNull()

  globalThis.fetch = (async () => reply('hi')) as unknown as typeof fetch
  await probe.setSource('https://example.test/a.png')
  expect(probe.state.data).not.toBeNull()
  expect(probe.state.error).toBeNull()
})

// An SVG is not a new source kind, and needs no new entry point: native sniffs the bytes.
test('an svg source goes through the same entry points as a raster', async () => {
  await mount('file:///tmp/logo.svg')
  expect(made[0]!.via).toBe('fromFile')
  expect(made[0]!.arg).toBe('file:///tmp/logo.svg')

  globalThis.fetch = (async () => reply('<svg/>')) as unknown as typeof fetch
  await mount('https://example.test/logo.svg')
  expect(made[1]!.via).toBe('fromBytes')
  expect(made[1]!.arg).toBe('<svg/>')
})

test('loadImage resolves a handle outside React', async () => {
  const image = await loadImage('file:///tmp/logo.svg')
  expect(image).toBe(made[0]!.handle as never)
  expect(made[0]!.via).toBe('fromFile')
})

test('the pending state cannot be poisoned by a caller', async () => {
  globalThis.fetch = (() => new Promise(() => {})) as unknown as typeof fetch
  const probe = await mount('https://example.test/slow.png')
  expect(() => {
    ;(probe.state as { data: unknown }).data = 'POISON'
  }).toThrow()
  const second = await mount('https://example.test/other.png')
  expect(second.state.data).toBeNull()
})

const SVG = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/></svg>'

test('SVG markup loads as a document rather than as base64', async () => {
  await loadImage(SVG)
  expect(made.map((m) => m.via)).toEqual(['fromSvg'])
  expect(made[0]!.arg).toBe(SVG)
})

test('markup is recognised through a prolog, leading space and a BOM', async () => {
  await loadImage('\uFEFF\n  <?xml version="1.0"?><svg viewBox="0 0 1 1"/>')
  expect(made.map((m) => m.via)).toEqual(['fromSvg'])
})

test('a base64 payload is still base64, not markup', async () => {
  // A JPEG payload starts "/9j/", which must not be mistaken for a document.
  await loadImage('/9j/4AAQSkZJRg==')
  expect(made.map((m) => m.via)).toEqual(['fromBase64'])
})

test('a data: URI holding an SVG still goes through base64', async () => {
  await loadImage('data:image/svg+xml;base64,PHN2Zy8+')
  expect(made.map((m) => m.via)).toEqual(['fromBase64'])
})

test('the hook routes markup the same way loadImage does', async () => {
  await mount(SVG)
  expect(made.map((m) => m.via)).toEqual(['fromSvg'])
})

test('a viewBox-only document needs no width or height attribute', async () => {
  await loadImage(
    '<svg viewBox="0 0 24 24"><rect width="24" height="24"/></svg>'
  )
  expect(made.map((m) => m.via)).toEqual(['fromSvg'])
})

test('an empty or blank string is not markup', async () => {
  // Neither is a document, and both must keep reporting as an unusable source
  // rather than being handed to the SVG parser.
  await expect(loadImage('')).rejects.toThrow('no image source')
  await loadImage('   ').catch(() => {})
  expect(made.map((m) => m.via)).toEqual(['fromBase64'])
})

test('a path is still a path', async () => {
  await loadImage('file:///tmp/icon.svg')
  expect(made.map((m) => m.via)).toEqual(['fromFile'])
})
