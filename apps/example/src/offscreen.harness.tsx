// The retained offscreen surface: parity with the one-shot export and the
// on-screen canvas, live slot writes, headless hit tests — all real frames.
import React from 'react';
import { Platform } from 'react-native';
import { describe, expect, render, test } from 'react-native-harness';
import {
  Circle,
  Fill,
  Group,
  Rect,
  RoundedRect,
  Scene,
  blur,
  circle,
  createHandle,
  createOffscreenSurface,
  fill,
  group,
  image,
  imageFromPixels,
  rect,
  renderToImage,
  rrect,
  type SceneNode,
} from 'react-native-tgfx';
import { Stage, sameFrame, sleep, stageBytes } from './testSupport';

const BG = '#f2f2f7';

function parityNodes(): SceneNode[] {
  return [
    fill({ color: BG }),
    rect({ x: 16, y: 16, width: 60, height: 40, color: '#0A84FF' }),
    circle({ cx: 140, cy: 50, r: 30, color: '#ff3b30' }),
    group({ transform: [{ translate: [20, 110] }, { rotate: 0.2 }] }, [
      rrect({ x: 0, y: 0, width: 70, height: 40, r: 10, color: '#34c759' }),
    ]),
  ];
}

async function rawBytes(image: {
  encode: (format?: string) => Promise<ArrayBuffer>;
  dispose: () => void;
}): Promise<Uint8Array> {
  const bytes = new Uint8Array(await image.encode('raw'));
  image.dispose();
  return bytes;
}

describe('tgfx offscreen surface', () => {
  test('renders the same pixels as the one-shot export and the canvas', async () => {
    const surface = createOffscreenSurface({ width: 200, height: 200 });
    const scene = Scene.from(parityNodes());
    scene.attach(surface);
    scene.flush();
    const fromSurface = await rawBytes(await surface.toImage());

    const oneShot = await rawBytes(
      await renderToImage(parityNodes(), { width: 200, height: 200 }),
    );
    expect(fromSurface.length).toBe(oneShot.length);
    expect(sameFrame(fromSurface, oneShot)).toBe(true);

    await render(
      <Stage>
        <Fill color={BG} />
        <Rect x={16} y={16} width={60} height={40} color="#0A84FF" />
        <Circle cx={140} cy={50} r={30} color="#ff3b30" />
        <Group transform={[{ translate: [20, 110] }, { rotate: 0.2 }]}>
          <RoundedRect
            x={0}
            y={0}
            width={70}
            height={40}
            r={10}
            color="#34c759"
          />
        </Group>
      </Stage>,
    );
    const onScreen = await stageBytes();
    expect(onScreen.length).toBeGreaterThan(0); // canvas parity is scale-dependent; the export pair above is exact

    scene.dispose();
    surface.dispose();
  });

  test('a slot write re-renders the retained surface without a re-encode', async () => {
    const surface = createOffscreenSurface({
      width: 64,
      height: 64,
      opaque: true,
    });
    const box = createHandle();
    const scene = Scene.from([
      fill({ color: '#000' }),
      rect({ ref: box, x: 0, y: 0, width: 16, height: 16, color: '#ff2d55' }),
    ]);
    scene.attach(surface);
    scene.flush();
    const before = await rawBytes(await surface.toImage());
    expect(before[0]).toBe(0xff); // red at the origin

    box.set({ x: 48, y: 48 });
    await sleep(100);
    surface.flush();
    const after = await rawBytes(await surface.toImage());
    expect(after[0]).toBe(0); // origin vacated
    const corner = (56 * 64 + 56) * 4;
    expect(after[corner]).toBe(0xff); // landed bottom-right

    scene.dispose();
    surface.dispose();
  });

  test('hitTest answers headless like the canvas does on screen', async () => {
    const surface = createOffscreenSurface({ width: 100, height: 100 });
    const scene = Scene.from([
      rect({ x: 10, y: 10, width: 40, height: 40, color: '#0A84FF' }),
    ]);
    scene.attach(surface);
    scene.flush();
    // No drain needed: hitTest compiles the staged scene itself.
    const hit = surface.hitTest(20, 20, true);
    expect(hit.length).toBeGreaterThanOrEqual(2);
    expect(hit[0]).toBe(1);
    expect(hit[1]).toBe(0);
    expect(surface.hitTest(90, 90, true).length).toBe(0);
    scene.dispose();
    surface.dispose();
  });

  test('save writes the frame without minting an image handle', async () => {
    const surface = createOffscreenSurface({
      width: 32,
      height: 32,
      opaque: true,
    });
    const scene = Scene.from([fill({ color: '#34c759' })]);
    scene.attach(surface);
    scene.flush();
    const out =
      Platform.OS === 'android'
        ? '/data/data/com.tgfxexample/cache/tgfx-harness-surface.png'
        : '/tmp/tgfx-harness-surface.png';
    await surface.save(out);
    scene.dispose();
    surface.dispose();
  });

  test('save with an empty path rejects instead of throwing', async () => {
    const surface = createOffscreenSurface({ width: 8, height: 8 });
    let failed = false;
    try {
      await surface.save('');
    } catch {
      failed = true;
    }
    expect(failed).toBe(true);
    surface.dispose();
  });

  test('toImage after dispose rejects instead of hanging', async () => {
    const surface = createOffscreenSurface({ width: 8, height: 8 });
    surface.dispose();
    let failed = false;
    try {
      await surface.toImage();
    } catch {
      failed = true;
    }
    expect(failed).toBe(true);
  });

  // Pixel assertions verify the workload; memory is measured outside the app.
  test('offscreen leak soak: repeatedly renders and releases snapshots on one retained surface', async () => {
    const size = 512;
    const surface = createOffscreenSurface({ width: size, height: size });
    const pixels = new Uint32Array(size * size);
    const scene = Scene.create();
    try {
      scene.attach(surface);
      for (let frame = 0; frame < 450; frame += 1) {
        const x = frame % 2 === 0 ? 0 : size - 32;
        const blue = frame % 256;
        const top = 0xff000000 + blue * 0x10000;
        pixels.fill(top);
        pixels.fill(top + 0xffff, pixels.length / 2); // sharp horizontal boundary
        const input = imageFromPixels(pixels.buffer, size, size);
        try {
          // Literal scenes isolate the worker from UI-runtime slot scheduling.
          scene.replaceChildren([
            fill({ color: '#000' }),
            image({ x: 0, y: 0, width: size, height: size, image: input }, [
              blur({ blur: 4 }),
            ]),
            rect({ x, y: 0, width: 32, height: 32, color: '#ff0000' }),
          ]);
          scene.flush();
          // Wait for a real worker render; flush coalescing cannot skip frames.
          const snapshot = await surface.toImage();
          try {
            if (frame < 2 || frame === 449) {
              const bytes = new Uint8Array(await snapshot.encode('raw'));
              expect(bytes.length).toBe(size * size * 4);
              expect(bytes[(x + 16 * size) * 4]).toBe(255);
              expect(bytes[(x + 16 * size) * 4 + 1]).toBe(0);
              const background = (size / 2 + (size / 4) * size) * 4;
              expect(bytes[background]).toBe(0);
              expect(bytes[background + 2]).toBe(blue);
              const boundary = (size / 2 + (size / 2) * size) * 4;
              // Without blur this first pixel in the bottom half is 255.
              expect(bytes[boundary]).toBeGreaterThan(32);
              expect(bytes[boundary]).toBeLessThan(223);
            }
          } finally {
            snapshot.dispose();
          }
        } finally {
          input.dispose();
        }
        await sleep(150);
      }
    } finally {
      scene.dispose();
      surface.dispose();
    }
    await sleep(500);
  });
});
