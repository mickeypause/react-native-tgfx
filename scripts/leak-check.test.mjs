import assert from 'node:assert/strict';
import test from 'node:test';
import { recordSample, verdict } from './leak-check.mjs';

function trace(pid, count, startMiB, growthMiB = 0) {
  for (let i = 0; i < count; i++) {
    recordSample({ pid, total: (startMiB + i * growthMiB) * 1024 });
  }
}

test('an older large process cannot hide growth in the restarted app', () => {
  trace(101, 30, 900); // 60 seconds of preparation at a two-second interval.
  trace(102, 40, 200, 2);
  assert.equal(verdict(), 1);
});

test('a new process must supply enough of its own settled samples', () => {
  trace(201, 40, 250);
  trace(202, 10, 250);
  assert.equal(verdict(), 2);
});

test('a stable process with enough samples passes', () => {
  trace(301, 40, 250);
  assert.equal(verdict(), 0);
});
