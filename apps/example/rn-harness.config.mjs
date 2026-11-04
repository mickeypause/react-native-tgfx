import {
  androidEmulator,
  androidPlatform,
  physicalAndroidDevice,
} from '@react-native-harness/platform-android';
import {
  applePlatform,
  appleSimulator,
} from '@react-native-harness/platform-apple';

const CI = process.env.CI === 'true';

export default {
  entryPoint: './index.js',
  appRegistryComponentName: 'TgfxExample',

  // The tgfx canvas is a full-screen GPU surface; keep the views it wraps.
  disableViewFlattening: true,

  // The bridge waits this long for a whole test file. The default 60s is
  // shorter than the ~90s leak soak (leakSoak.harness.tsx), which would be
  // reported as "the device did not respond".
  bridgeTimeout: 180000,
  // A cold hosted emulator or simulator boots and bundles far slower than a device.
  platformReadyTimeout: CI ? 420000 : 300000,
  bundleStartTimeout: CI ? 120000 : 60000,

  runners: [
    // The default runner, so the one baseline directory: __image_snapshots__/android.
    // A runner is a specific device, and the directory is named after the
    // runner — running an image suite on another runner would fork the baselines.
    androidPlatform({
      name: 'android',
      device: physicalAndroidDevice('samsung', 'SM-E146B'),
      bundleId: 'com.tgfxexample',
    }),
    // CI's emulator (harness-android.yml), selected with --harnessRunner=ci-emu.
    // Harness creates the AVD when it is missing. Its baselines are written on
    // every run and never committed: a hosted emulator's software renderer is
    // not the reference device, so on CI the frame-comparison and drew-something
    // assertions gate, and the snapshots are uploaded for a look.
    androidPlatform({
      name: 'ci-emu',
      device: androidEmulator(process.env.HARNESS_AVD_NAME ?? 'ci-avd', {
        apiLevel: Number(process.env.HARNESS_AVD_API_LEVEL ?? 35),
        profile: process.env.HARNESS_AVD_PROFILE ?? 'pixel_7',
        diskSize: '4G',
        heapSize: '1G',
        snapshot: { enabled: CI },
      }),
      bundleId: 'com.tgfxexample',
    }),
    // An iOS simulator (harness-ios.yml), selected with --harnessRunner=ios-sim.
    // Same baseline rule as ci-emu.
    applePlatform({
      name: 'ios-sim',
      device: appleSimulator(
        process.env.HARNESS_SIM_DEVICE ?? 'iPhone 17 Pro',
        process.env.HARNESS_SIM_OS ?? '26.5',
      ),
      bundleId: 'com.pikachu.ritesh.dev',
    }),
  ],
  defaultRunner: 'android',
};
