// UI test of the shell the screens live in: tabs, switching, and what each screen mounts.
import React from 'react';
import { describe, expect, render, test, waitFor } from 'react-native-harness';
import { screen, userEvent } from '@react-native-harness/ui';
import App from '../App';

const TABS = ['gallery', 'effects'] as const;
type Tab = (typeof TABS)[number];

// waitFor defaults to 1s; a camera screen's first mount on real hardware —
// native camera init, then a full-screen GPU surface — takes longer than that.
const MOUNT = { timeout: 20_000, interval: 250 };

// The gallery is thirty-odd live canvases; mounting the app per assertion times
// the device out. One mount per test, and the walk does its asserting in place.
function mounted(): Tab[] {
  return TABS.filter(tab => screen.queryByTestId(`screen-${tab}`) != null);
}

describe('example app shell', () => {
  test('opens on the gallery with a tab per screen', async () => {
    await render(<App />);

    for (const name of TABS) {
      expect(await screen.findByTestId(`tab-${name}`)).toBeTruthy();
    }
    expect(await screen.findByTestId('screen-gallery')).toBeTruthy();
    expect(mounted()).toEqual(['gallery']);
  });

  test('a camera tab takes the gallery down', async () => {
    await render(<App />);
    expect(await screen.findByTestId('screen-gallery')).toBeTruthy();

    await userEvent.press(await screen.findByTestId('tab-effects'));
    await waitFor(() => {
      expect(screen.queryByTestId('screen-gallery')).toBeFalsy();
    }, MOUNT);

    // Deliberately not asserting that `screen-effects` mounts. VisionCamera
    // does not come up inside a Harness overlay render — on an emulator or on
    // a device — so the camera screens are covered by running the app, not
    // from here. What this does protect is the gallery being torn down, which
    // is what keeps thirty canvases from drawing behind a camera preview.
  });
});
