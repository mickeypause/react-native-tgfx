// <TgfxImage>, <Atlas>, <Vertices>, <ImageShader>, the decoders, and the native handle ops.
import React from 'react';
import {
  afterAll,
  beforeAll,
  describe,
  expect,
  render,
  test,
} from 'react-native-harness';
import {
  Atlas,
  Circle,
  Fill,
  Group,
  TgfxImage,
  ImageShader,
  RoundedRect,
  Vertices,
  loadImage,
  renderToImage,
  type Fit,
} from 'react-native-tgfx';
import {
  BACKGROUND,
  Stage,
  drawnSnapshot,
  sameFrame,
  stageBytes,
} from './testSupport';

const SOURCES = {
  sample: require('./assets/sample.png'),
  jpeg: require('./assets/sample-jpeg.jpg'),
  webp: require('./assets/sample-webp.webp'),
  sprites: require('./assets/sprites.png'),
  tile: require('./assets/tile.png'),
  logo: require('./assets/logo.svg'),
  // Stored 240x120 landscape, with an EXIF marker saying "rotate 90 CW".
  portrait: require('./assets/portrait-exif.jpg'),
};

type Key = keyof typeof SOURCES;

/**
 * Decoded once, up front. A `useImage` inside the scene lands after the
 * screenshot has already settled on the empty frame, and that blank becomes
 * the baseline.
 */
const decoded = {} as Record<Key, TgfxImage>;
let derived: Record<'thumb' | 'crop' | 'turned' | 'soft', TgfxImage>;
let offscreen: TgfxImage;

const FITS: Fit[] = [
  'contain',
  'cover',
  'fill',
  'fitWidth',
  'fitHeight',
  'none',
  'scaleDown',
];

beforeAll(async () => {
  await Promise.all(
    (Object.keys(SOURCES) as Key[]).map(async key => {
      decoded[key] = await loadImage(SOURCES[key]);
    }),
  );

  const photo = decoded.sample;
  derived = {
    thumb: photo.scaled(32, 32),
    crop: photo.subset(0, 0, 64, 64),
    turned: photo.oriented('rightTop'),
    soft: photo.blurred(6, 6),
  };

  offscreen = await renderToImage(
    <>
      <Fill color="#101725" />
      {[0, 1, 2, 3, 4].map(i => (
        <Circle
          key={i}
          cx={24 + i * 28}
          cy={50 - i * 6}
          r={16}
          color={`hsl(${i * 48}, 80%, 60%)`}
        />
      ))}
    </>,
    { width: 160, height: 100, pixelRatio: 2 },
  );
});

afterAll(() => {
  // Derived handles first: disposing the source they came from invalidates
  // them, and touching a disposed handle at all — even to read `handle` —
  // throws. Teardown is not what these tests assert, so it swallows that.
  for (const handle of [...Object.values(derived ?? {}), offscreen]) {
    try {
      handle?.dispose();
    } catch {}
  }
  for (const handle of Object.values(decoded)) {
    try {
      handle?.dispose();
    } catch {}
  }
});

describe('tgfx image decoding', () => {
  test('decodes a png to a handle with a size', () => {
    expect(decoded.sample.width).toBeGreaterThan(0);
    expect(decoded.sample.height).toBeGreaterThan(0);
    expect(typeof decoded.sample.handle).toBe('number');
  });

  test('decodes jpeg and webp as well as png', () => {
    for (const key of ['sample', 'jpeg', 'webp'] as const) {
      expect(decoded[key].width).toBeGreaterThan(0);
      expect(decoded[key].height).toBeGreaterThan(0);
    }
  });

  test('applies the EXIF orientation marker', () => {
    // Stored landscape; the marker turns it, so the handle reports portrait.
    expect(decoded.portrait.height).toBeGreaterThan(decoded.portrait.width);
  });

  test('reports an SVG at its declared size', () => {
    expect(decoded.logo.width).toBe(24);
    expect(decoded.logo.height).toBe(24);
  });

  test('derives handles without decoding again', () => {
    expect(derived.thumb.width).toBe(32);
    expect(derived.thumb.height).toBe(32);
    expect(derived.crop.width).toBe(64);
    expect(derived.crop.height).toBe(64);
    // A quarter turn swaps the axes.
    expect(derived.turned.width).toBe(decoded.sample.height);
    expect(derived.turned.height).toBe(decoded.sample.width);
    // A blur grows the image by its margin rather than filtering in place.
    expect(derived.soft.width).toBeGreaterThan(decoded.sample.width);
  });

  test('renders a scene offscreen with no <Canvas> mounted', () => {
    expect(offscreen.width).toBe(320);
    expect(offscreen.height).toBe(200);
  });

  test('a disposed handle refuses further native calls', () => {
    // Derived, not `loadImage`d: decodes are cached per source, so disposing a
    // second handle to the same asset would kill the one every other test draws.
    const throwaway = decoded.sample.scaled(16, 16);
    expect(throwaway.width).toBe(16);
    throwaway.dispose();
    // The native object is gone, so a second call throws rather than crashing.
    expect(() => throwaway.dispose()).toThrow();
  });

  for (const [label, source, received] of [
    ['undefined', undefined, 'undefined'],
    ['null', null, 'null'],
    ['an empty string', '', '""'],
  ] as const) {
    test(`rejects ${label} by name`, async () => {
      let message = '';
      try {
        await loadImage(source as string);
        throw new Error('loadImage resolved, expected it to reject');
      } catch (e: unknown) {
        message = e instanceof Error ? e.message : String(e);
      }
      expect(message).toContain('no image source was given');
      expect(message).toContain(`(received ${received})`);
    });
  }
});

describe('tgfx image rendering', () => {
  test('draws every fit mode', async () => {
    await drawnSnapshot(
      'image-fit-modes',
      <>
        {FITS.map((fit, i) => (
          <TgfxImage
            key={fit}
            x={(i % 3) * 66 + 2}
            y={Math.floor(i / 3) * 66 + 2}
            width={62}
            height={44}
            image={decoded.sample}
            fit={fit}
          />
        ))}
      </>,
    );
  });

  /** The settled frame of the 128x128 sample drawn into one box with `fit`. */
  async function fitFrame(fit: Fit, width: number, height: number) {
    await render(
      <Stage>
        <Fill color={BACKGROUND} />
        <TgfxImage
          x={20}
          y={20}
          width={width}
          height={height}
          image={decoded.sample}
          fit={fit}
        />
      </Stage>,
    );
    return stageBytes();
  }

  test('scaleDown shrinks an image larger than its box like contain', async () => {
    const scaleDown = await fitFrame('scaleDown', 62, 44);
    expect(sameFrame(scaleDown, await fitFrame('contain', 62, 44))).toBe(true);
    // Also proves the image drew: two blank frames would match too.
    expect(sameFrame(scaleDown, await fitFrame('fill', 62, 44))).toBe(false);
  });

  test('scaleDown leaves an image smaller than its box at its own size', async () => {
    const scaleDown = await fitFrame('scaleDown', 160, 150);
    expect(sameFrame(scaleDown, await fitFrame('none', 160, 150))).toBe(true);
    expect(sameFrame(scaleDown, await fitFrame('fill', 160, 150))).toBe(false);
  });

  test('draws every sampling mode', async () => {
    const modes = ['nearest', 'linear', 'mipmap'] as const;
    await drawnSnapshot(
      'image-sampling-modes',
      <>
        {modes.map((sampling, i) => (
          <TgfxImage
            key={sampling}
            x={10}
            y={10 + i * 62}
            width={54}
            height={54}
            image={decoded.sample}
            fit="contain"
            sampling={sampling}
          />
        ))}
      </>,
    );
  });

  test('cuts sprites out of one sheet with rect', async () => {
    await drawnSnapshot(
      'image-sprite-rect',
      <>
        {[0, 1, 2, 3].map(i => (
          <TgfxImage
            key={i}
            x={8 + (i % 2) * 96}
            y={8 + Math.floor(i / 2) * 96}
            width={88}
            height={88}
            image={decoded.sprites}
            fit="contain"
            sampling="nearest"
            rect={{ x: i * 32, y: 0, width: 32, height: 32 }}
          />
        ))}
      </>,
    );
  });

  test('draws many sprites from one <Atlas> node', async () => {
    const field = Array.from({ length: 60 }, (_, i) => {
      const angle = i * 0.7;
      const radius = 4 + i * 1.4;
      return {
        rect: { x: (i % 4) * 32, y: 0, width: 32, height: 32 },
        x: 100 + Math.cos(angle) * radius,
        y: 100 + Math.sin(angle) * radius * 0.6,
        scale: 0.3 + (i % 5) * 0.1,
      };
    });
    await drawnSnapshot(
      'image-atlas',
      <>
        <Atlas image={decoded.sprites} sprites={field} sampling="nearest" />
      </>,
    );
  });

  test('draws a colour mesh and the same mesh warping an image', async () => {
    const size = 3;
    const step = 84 / size;
    const points: { x: number; y: number }[] = [];
    const coords: { x: number; y: number }[] = [];
    const hues = ['#f2a33c', '#4f8ef7', '#34c759', '#ff2d55'];
    const colors: string[] = [];
    for (let row = 0; row <= size; row++) {
      for (let col = 0; col <= size; col++) {
        const wobble = Math.sin((col / size) * Math.PI) * 10;
        points.push({ x: 8 + col * step, y: 8 + row * step + wobble });
        coords.push({ x: (col / size) * 128, y: (row / size) * 128 });
        colors.push(hues[(row + col) % hues.length]!);
      }
    }
    const indices: number[] = [];
    for (let row = 0; row < size; row++) {
      for (let col = 0; col < size; col++) {
        const at = row * (size + 1) + col;
        indices.push(
          at,
          at + 1,
          at + size + 1,
          at + 1,
          at + size + 1,
          at + size + 2,
        );
      }
    }

    await drawnSnapshot(
      'image-vertices',
      <>
        <Vertices vertices={points} colors={colors} indices={indices} />
        <Group transform={[{ translateY: 100 }]}>
          <Vertices vertices={points} textures={coords} indices={indices}>
            <ImageShader image={decoded.sample} />
          </Vertices>
        </Group>
      </>,
    );
  });

  test('tiles an <ImageShader> in every tile mode', async () => {
    const modes = ['clamp', 'repeat', 'mirror'] as const;
    await drawnSnapshot(
      'image-shader-tile-modes',
      <>
        {modes.map((mode, i) => (
          <RoundedRect
            key={mode}
            x={8}
            y={8 + i * 64}
            width={184}
            height={56}
            r={10}
          >
            <ImageShader image={decoded.tile} tx={mode} ty={mode} />
          </RoundedRect>
        ))}
      </>,
    );
  });

  test('fills an arbitrary shape with an image', async () => {
    await drawnSnapshot(
      'image-shader-shape-fill',
      <>
        <Circle cx={100} cy={100} r={90}>
          <ImageShader
            image={decoded.sample}
            rect={{ x: 10, y: 10, width: 180, height: 180 }}
            fit="cover"
          />
        </Circle>
      </>,
    );
  });

  test('draws png, jpeg and webp side by side', async () => {
    await drawnSnapshot(
      'image-codec-formats',
      <>
        {(['sample', 'jpeg', 'webp'] as const).map((key, i) => (
          <TgfxImage
            key={key}
            x={8}
            y={8 + i * 64}
            width={184}
            height={56}
            image={decoded[key]}
            fit="contain"
          />
        ))}
      </>,
    );
  });

  test('replays an SVG at draw size rather than resampling it', async () => {
    await drawnSnapshot(
      'image-svg-source',
      <>
        <TgfxImage x={8} y={8} width={160} height={160} image={decoded.logo} />
        <TgfxImage x={170} y={8} width={24} height={24} image={decoded.logo} />
      </>,
    );
  });

  test('draws a photo whose EXIF marker turned it upright', async () => {
    await drawnSnapshot(
      'image-exif-orientation',
      <>
        <TgfxImage
          x={8}
          y={8}
          width={184}
          height={184}
          image={decoded.portrait}
          fit="contain"
        />
      </>,
    );
  });

  test('draws the derived handles', async () => {
    await drawnSnapshot(
      'image-handle-ops',
      <>
        <TgfxImage image={derived.thumb} x={8} y={8} width={88} height={88} />
        <TgfxImage image={derived.crop} x={104} y={8} width={88} height={88} />
        <TgfxImage
          image={derived.turned}
          x={8}
          y={104}
          width={88}
          height={88}
        />
        <TgfxImage
          image={derived.soft}
          x={104}
          y={104}
          width={88}
          height={88}
        />
      </>,
    );
  });

  test('draws an offscreen render like any other image', async () => {
    await drawnSnapshot(
      'image-offscreen-render',
      <>
        <TgfxImage image={offscreen} x={8} y={20} width={184} height={80} />
        {/* The same handle again: a one-shot would come back blank. */}
        <TgfxImage image={offscreen} x={8} y={110} width={92} height={58} />
      </>,
    );
  });
});
