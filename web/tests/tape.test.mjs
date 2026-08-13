/**
 * Command-tape conformance harness (dija R10).
 *
 * Runs headless under `node --test` — no browser, no WebGPU: the point
 * is the WIRE, which the wasm module produces on its own.
 *
 * Two gates:
 *
 *   1. Zero imports. The module must declare NO imports at all and
 *      instantiate with `{}`. Any `env:*` import creeping back in fails
 *      here, loudly, before any browser is involved.
 *   2. Byte-exact tape. Driving the `web_add_one` entry and feeding the
 *      async results back through `quanta_resolve` / `quanta_reject`
 *      produces the goldens below, byte for byte. dija asserts this
 *      same format, so a change here is a change to a published
 *      contract: bump `TAPE_VERSION` on both sides rather than editing
 *      a golden.
 *
 * The wasm comes from the existing smoke build (`quanta build web`,
 * which stages it next to the example's page); a plain
 * `cargo build --target wasm32-unknown-unknown -p web-add-one` is
 * enough for a local run.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const REPO_ROOT = resolve(HERE, "..", "..");

const WASM_CANDIDATES = [
  join(REPO_ROOT, "examples", "web_add_one", "web_add_one.wasm"),
  join(REPO_ROOT, "target", "wasm32-unknown-unknown", "debug", "web_add_one.wasm"),
  join(REPO_ROOT, "target", "wasm32-unknown-unknown", "release", "web_add_one.wasm"),
];

function wasmBytes() {
  for (const path of WASM_CANDIDATES) {
    if (existsSync(path)) return readFileSync(path);
  }
  throw new Error(
    `web_add_one.wasm not found; run \`cargo run -p quanta-cli -- build web\`.\n` +
      `Looked in:\n  ${WASM_CANDIDATES.join("\n  ")}`,
  );
}

const MODULE = new WebAssembly.Module(wasmBytes());

/** A fresh instance with NO imports — the zero-import contract itself. */
function fresh() {
  const exports = new WebAssembly.Instance(MODULE, {}).exports;
  // What the glue pushes right after instantiation: WebGPU present,
  // bgra8unorm preferred.
  exports.__quanta_env_init(1, 1);
  return exports;
}

/** Read the tape out and clear it, exactly as `drainTape` does. */
function drain(exports) {
  const ptr = exports.__quanta_tape_ptr();
  const len = exports.__quanta_tape_len();
  const bytes = new Uint8Array(exports.memory.buffer, ptr, len).slice();
  exports.__quanta_tape_clear();
  return bytes;
}

/** Whitespace-separated hex → bytes. Goldens are written this way. */
function hexBytes(text) {
  return Uint8Array.from(text.trim().split(/\s+/), (b) => parseInt(b, 16));
}

function hex(bytes) {
  return [...bytes].map((b) => b.toString(16).padStart(2, "0")).join(" ");
}

function assertBytes(actual, expected, what) {
  assert.equal(hex(actual), hex(expected), what);
}

test("the module declares no imports and instantiates with {}", () => {
  assert.deepEqual(WebAssembly.Module.imports(MODULE), []);
  const exports = new WebAssembly.Instance(MODULE, {}).exports;
  assert.equal(typeof exports.web_add_one_run, "function");
  assert.equal(typeof exports.__quanta_tape_ptr, "function");
  assert.equal(typeof exports.__quanta_tape_len, "function");
  assert.equal(typeof exports.__quanta_tape_clear, "function");
  assert.equal(typeof exports.__quanta_env_init, "function");
  assert.equal(typeof exports.__quanta_canvas_dims, "function");
});

test("an entry that has appended nothing leaves the tape empty", () => {
  const exports = fresh();
  assert.equal(exports.__quanta_tape_len(), 0);
});

test("device acquisition rides the tape word for word", () => {
  const exports = fresh();

  // The entry runs to its first await: one RequestAdapter op naming the
  // Rust-side promise id.
  exports.web_add_one_run(7);
  assertBytes(
    drain(exports),
    hexBytes(`
      51 54 41 50   01 00 00 00
      01 00 00 00   01 00 00 00
    `),
    "[magic QTAP][version 1][op 01 RequestAdapter][task 1]",
  );

  // The glue answers through the exported setter; the continuation asks
  // for a device off that adapter handle.
  exports.quanta_resolve(1, 0x11);
  assertBytes(
    drain(exports),
    hexBytes(`
      51 54 41 50   01 00 00 00
      02 00 00 00   11 00 00 00   02 00 00 00
    `),
    "[magic][version][op 02 RequestDevice][adapter 0x11][task 2]",
  );
});

test("creators mint high-bit ids and payloads ride inline", () => {
  const exports = fresh();
  exports.web_add_one_run(7);
  drain(exports);
  exports.quanta_resolve(1, 0x11);
  drain(exports);

  // With a device in hand the entry allocates its field and writes the
  // input: the release of the spent adapter handle, a create whose
  // destination id the DRIVER minted (high bit set), and a write whose
  // 64 u32s are copied into the tape itself.
  exports.quanta_resolve(2, 0x22);
  const tape = drain(exports);
  assertBytes(
    tape.subarray(0, 64),
    hexBytes(`
      51 54 41 50   01 00 00 00
      f0 00 00 00   11 00 00 00
      10 00 00 00   01 00 00 80   22 00 00 00   00 00 00 00   00 00 70 40   fc 03 00 00
      12 00 00 00   22 00 00 00   01 00 00 80   00 00 00 00   00 00 00 00   00 01 00 00
    `),
    "[magic][version]" +
      "[op f0 Release][adapter 0x11]" +
      "[op 10 CreateBuffer][dst 0x80000001][device 0x22][size f64 256][usage 0x3fc]" +
      "[op 12 WriteBuffer][device 0x22][buffer 0x80000001][offset f64 0][payload len 256]",
  );

  // The payload is the buffer contents themselves — [0, 1, …, 63] as LE
  // u32s — proving inline framing, not a pointer.
  const payload = new DataView(tape.buffer, tape.byteOffset + 64, 256);
  for (let i = 0; i < 64; i++) {
    assert.equal(payload.getUint32(i * 4, true), i, `payload word ${i}`);
  }
});

test("a failed acquisition completes the top-level task on the tape", () => {
  const exports = fresh();
  exports.web_add_one_run(7);
  drain(exports);
  exports.quanta_resolve(1, 0x11);
  drain(exports);

  // No import left to call: the error reaches the host as an op naming
  // the top-level task id the harness minted.
  exports.quanta_reject(2);
  const tape = drain(exports);
  assertBytes(
    tape.subarray(0, 20),
    hexBytes(`
      51 54 41 50   01 00 00 00
      e1 00 00 00   07 00 00 00   56 00 00 00
    `),
    "[magic][version][op e1 CompleteErr][task 7][payload len 86]",
  );
  const message = new TextDecoder().decode(tape.subarray(20, 20 + 0x56));
  assert.match(message, /requestDevice rejected/);
  // Payload padded to a word: 86 bytes + 2.
  assert.equal(tape.length, 20 + 88);
});
