import { Image as RNImage } from 'react-native'

/**
 * The three ways bytes reach native. Images and fonts both have exactly these,
 * so the routing below - which of the three a given source needs - is written
 * once rather than once per asset kind.
 */
export interface SourceReader<T> {
  fromFile(path: string): Promise<T>
  fromBytes(bytes: ArrayBuffer): Promise<T>
  fromBase64(data: string): Promise<T>
}

/**
 * True when the string opens like XML, so it is an SVG document rather than a
 * path, a URI or base64. The same test native applies to bytes: no encoded
 * raster format and no base64 payload starts with `<`.
 */
export function looksLikeSvgMarkup(source: string): boolean {
  // A UTF-8 BOM ahead of the prolog is legal and common out of editors.
  const start = source.charCodeAt(0) === 0xfeff ? 1 : 0
  for (let i = start; i < source.length; i++) {
    const c = source[i]!
    if (c === ' ' || c === '\t' || c === '\r' || c === '\n') {
      continue
    }
    return c === '<'
  }
  return false
}

/**
 * True for an absolute path such as `save()` writes to. Base64 can open with
 * "/" too (a JPEG payload starts "/9j/"), but never holds a character outside
 * its alphabet, and a real path's name or extension nearly always does.
 */
function looksLikeAbsolutePath(source: string): boolean {
  return source.startsWith('/') && /[^A-Za-z0-9+/=\s]/.test(source)
}

function dataPayload(uri: string): string | null {
  if (!uri.startsWith('data:')) {
    return null
  }
  const comma = uri.indexOf(',')
  return comma >= 0 ? uri.slice(comma + 1) : ''
}

/**
 * Load a `require()`d asset, a URI or a raw base64 string through `api`.
 *
 * A local file is opened natively, so its bytes never enter JS at all. Only a
 * remote or Metro-served URI still goes through `fetch`, and then as an
 * `ArrayBuffer` - nothing is ever base64-encoded on the JS thread.
 */
export async function loadSource<T>(
  source: number | string,
  api: SourceReader<T>,
  signal?: AbortSignal
): Promise<T> {
  if (!source) {
    throw new Error(
      `tgfx: no image source was given (received ${JSON.stringify(source)})`
    )
  }
  let uri: string
  // A bundled asset may be a bare path; a caller's own string never is (JPEG base64 starts "/9j/").
  let bareIsPath = false
  if (typeof source === 'number') {
    const resolved = RNImage.resolveAssetSource(source)
    if (resolved?.uri == null) {
      throw new Error(`could not resolve asset source: ${String(source)}`)
    }
    uri = resolved.uri
    bareIsPath = true
  } else {
    uri = source
  }

  const payload = dataPayload(uri)
  if (payload != null) {
    return api.fromBase64(payload)
  }
  // A release Android build compiles a bundled asset into a drawable/raw
  // resource, so this is a bare resource name - no scheme, no path. Native
  // reads it by name; base64 would only see the name's own characters.
  if (
    uri.startsWith('file:') ||
    (bareIsPath && !/^[a-z][a-z\d+.-]*:/i.test(uri)) ||
    looksLikeAbsolutePath(uri)
  ) {
    return api.fromFile(uri)
  }
  if (/^(https?|content|asset):/.test(uri)) {
    // Without the signal a hung source holds its socket long after the caller gave up.
    const res = await fetch(uri, { signal })
    if (!res.ok) {
      throw new Error(`${uri} returned HTTP ${res.status}`)
    }
    return api.fromBytes(await res.arrayBuffer())
  }
  return api.fromBase64(uri)
}
