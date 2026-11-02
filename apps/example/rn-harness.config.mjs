import {
  androidEmulator,
  androidPlatform,
  physicalAndroidDevice,
} from '@react-native-harness/platform-android';
import {
  applePlatform,
  appleSimulator,
} from '@react-native-harness/platform-apple';

export default {
  entryPoint: './index.js',
  appRegistryComponentName: 'TgfxExample',

  // The tgfx canvas is a full-screen GPU surface; keep the views it wraps.
  disableViewFlattening: true,

  // The bridge waits this long for a whole test file. The default 60s is
  // shorter than the ~90s leak soak (leakSoak.harness.tsx), which would be
  // reported as "the device did not respond".
  bridgeTimeout: 180000,

  runners: [
    // The default runner, so the one baseline directory: __image_snapshots__/android.
    // A runner is a specific device, and the directory is named after the
    // runner — running an image suite on another runner would fork the baselines.
    androidPlatform({
      name: 'android',
      device: physicalAndroidDevice('samsung', 'SM-E146B'),
      bundleId: 'com.tgfxexample',
    }),
    // CI's emulator, selected with --harnessRunner=ci-emu. Only the leak soak
    // runs there (ci.yml `leak` job), and it snapshots nothing.
    androidPlatform({
      name: 'ci-emu',
      device: androidEmulator('ci-avd'),
      bundleId: 'com.tgfxexample',
    }),
    // An iOS simulator, selected with --harnessRunner=ios-sim.
    applePlatform({
      name: 'ios-sim',
      device: appleSimulator(
        process.env.TGFX_IOS_SIMULATOR || 'iPhone 17 Pro',
        process.env.TGFX_IOS_VERSION || '26.5'
      ),
      bundleId: 'com.pikachu.ritesh.dev',
    }),
  ],
  defaultRunner: 'android',
};
