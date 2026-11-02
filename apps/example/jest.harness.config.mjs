export default {
  preset: 'react-native-harness',
  testMatch: ['<rootDir>/src/**/*.harness.[jt]s?(x)'],
  // The ~90s memory soak is scripts/leak-check.mjs's business, not the default
  // suite's; the script re-enables it with --testPathIgnorePatterns='$^'.
  // The ~35s adb-watched purge scenario is scripts/cache-purge-check.mjs's
  // business, not the default suite's; the script re-enables it with
  // --testPathIgnorePatterns='$^'.
  testPathIgnorePatterns: ['leakSoak', 'offscreenLeakSoak', 'cachePurgeAdb'],
};
