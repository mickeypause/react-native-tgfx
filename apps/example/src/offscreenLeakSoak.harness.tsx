// Run with scripts/leak-check.mjs; memory is measured outside the app.
import { describe, expect, test } from 'react-native-harness';
import {
  Scene,
  blur,
  createOffscreenSurface,
  fill,
  image,
  imageFromPixels,
  rect,
} from 'react-native-tgfx';
import { sleep } from './testSupport';

describe('offscreen leak soak', () => {
  test('repeatedly renders and releases snapshots on one retained surface', async () => {
    const size = 512;
    const surface = createOffscreenSurface({ width: size, height: size });
    const pixels = new Uint32Array(size * size);
    const scene = Scene.create();
    try {
      scene.attach(surface);
      for (let frame = 0; frame < 450; frame += 1) {
        const x = frame % 2 === 0 ? 0 : size - 32;
        pixels.fill(0xff000000 + (frame % 256) * 0x10000); // opaque blue RGBA
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
              const background = (size / 2 + (size / 2) * size) * 4;
              expect(bytes[background]).toBe(0);
              expect(bytes[background + 2]).toBe(frame % 256);
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
