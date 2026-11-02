#!/usr/bin/env node
// Memory-leak gate: runs the leak soak (apps/example/src/leakSoak.harness.tsx)
// while sampling Android `adb shell dumpsys meminfo` or iOS Simulator `ps`
// from the host. All measurement lives out here in OS accounting — the app
// deliberately reports nothing about its own memory.
//
//   node scripts/leak-check.mjs [--package com.tgfxexample] [--threshold-mb 30]
//                               [--test src/leakSoak.harness.tsx] [--interval-ms 2000]
//                               [--runner ci-emu]
// iOS Simulator: --ios-simulator <UDID> --runner ios-sim --metro-port 8094
// Select that same simulator with TGFX_IOS_SIMULATOR / TGFX_IOS_VERSION.
//
// Exit codes: 0 no leak, 1 leak detected, 2 the run itself failed.
import { execFile, spawn } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2);
function arg(name, fallback) {
  const at = args.indexOf(`--${name}`);
  return at >= 0 && args[at + 1] !== undefined ? args[at + 1] : fallback;
}

const PACKAGE = arg('package', 'com.tgfxexample');
const THRESHOLD_MB = Number(arg('threshold-mb', '30'));
const TEST = arg('test', 'src/leakSoak.harness.tsx');
const INTERVAL_MS = Number(arg('interval-ms', '2000'));
const IOS_SIMULATOR = arg('ios-simulator', null);
const METRO_PORT = arg('metro-port', null);
const METRIC = IOS_SIMULATOR ? 'RSS' : 'TOTAL PSS';

const exampleDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'apps', 'example');

function adb(...adbArgs) {
  return new Promise(resolve => {
    execFile('adb', adbArgs, { maxBuffer: 4 * 1024 * 1024 }, (error, stdout) =>
      resolve(error ? null : stdout)
    );
  });
}

/** The App Summary PSS rows, in KB, or null while the process is not up. */
function parseMeminfo(text) {
  const grab = label => {
    const match = text.match(new RegExp(`${label}:\\s+(\\d+)`));
    return match ? Number(match[1]) : null;
  };
  const total = grab('TOTAL PSS');
  if (total === null) return null;
  return { total, native: grab('Native Heap'), graphics: grab('Graphics') };
}

const median = values => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};
const mb = kb => (kb / 1024).toFixed(1);

const samples = [];
let sampling = false;
async function sample() {
  if (sampling) return; // a slow dumpsys must not stack behind the timer
  sampling = true;
  const out = IOS_SIMULATOR
    ? await new Promise(resolve => {
        execFile(
          'ps', ['-axo', 'rss=,command='], { maxBuffer: 4 * 1024 * 1024 },
          (error, stdout) => resolve(error ? null : stdout)
        );
      })
    : await adb('shell', 'dumpsys', 'meminfo', PACKAGE);
  sampling = false;
  // Match this clone's executable path, never another booted simulator's app.
  let parsed = null;
  if (IOS_SIMULATOR) {
    const rows = out?.split('\n').filter(row =>
        row.includes(`/Devices/${IOS_SIMULATOR}/`) &&
        row.includes('/TgfxExample.app/TgfxExample')
    );
    if (rows?.length === 1) parsed = { total: Number.parseInt(rows[0], 10) };
  } else {
    parsed = out && parseMeminfo(out);
  }
  if (!parsed || !Number.isFinite(parsed.total)) return;
  samples.push(parsed);
  console.log(
    `[leak-check] #${samples.length} ${METRIC} ${mb(parsed.total)}MB` +
      (IOS_SIMULATOR
        ? ''
        : ` native ${mb(parsed.native ?? 0)}MB graphics ${mb(parsed.graphics ?? 0)}MB`)
  );
}

function verdict() {
  // The first samples straddle install/launch/first-frames; judge steady state.
  const warmup = Math.max(5, Math.ceil(samples.length * 0.25));
  const settled = samples.slice(warmup);
  if (settled.length < 12) {
    console.error(`[leak-check] only ${settled.length} settled samples — soak too short to judge`);
    return 2;
  }
  const third = Math.floor(settled.length / 3);
  const growth = key =>
    median(settled.slice(-third).map(s => s[key] ?? 0)) -
    median(settled.slice(0, third).map(s => s[key] ?? 0));
  const totalGrowth = growth('total');
  console.log(
    `[leak-check] settled growth (median of last third vs first third, ${settled.length} samples):`
  );
  console.log(`[leak-check]   ${METRIC}   ${mb(totalGrowth)}MB  (threshold ${THRESHOLD_MB}MB)`);
  if (!IOS_SIMULATOR) {
    console.log(`[leak-check]   Native Heap ${mb(growth('native'))}MB`);
    console.log(`[leak-check]   Graphics    ${mb(growth('graphics'))}MB`);
  }
  if (totalGrowth > THRESHOLD_MB * 1024) {
    console.error(`::error::memory leak: ${METRIC} grew ${mb(totalGrowth)}MB during the soak`);
    return 1;
  }
  console.log('[leak-check] OK — no leak detected');
  return 0;
}

// A leftover instance would poison the early samples with its own memory.
if (!IOS_SIMULATOR) await adb('shell', 'am', 'force-stop', PACKAGE);

const runner = arg('runner', null);
const child = spawn(
  'bun',
  [
    'harness',
    // The test path must come before the ignore override: yargs array options
    // greedily swallow every following positional, `=` form included.
    TEST,
    '--testPathIgnorePatterns=$^',
    ...(runner ? [`--harnessRunner=${runner}`] : []),
    ...(METRO_PORT ? [`--metroPort=${METRO_PORT}`] : []),
  ],
  { cwd: exampleDir, stdio: 'inherit' }
);
const timer = setInterval(sample, INTERVAL_MS);
child.on('exit', code => {
  clearInterval(timer);
  if (code !== 0) {
    console.error(`[leak-check] harness exited with ${code} — no verdict`);
    process.exit(2);
  }
  process.exit(verdict());
});
