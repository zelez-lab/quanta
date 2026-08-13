# The web command tape

The WebGPU driver crosses the wasm↔JS boundary with **zero imports**:
instead of calling out through `env:*` functions, it appends commands
to a tape in linear memory, and the glue **drains** the tape — reads
it, performs the WebGPU calls, clears it — after every wasm entry
returns. Async results (adapter/device acquisition, mapAsync,
submitted-work-done) come back through the exported setters
(`quanta_resolve` / `quanta_reject`), the same machinery `init_poll`
has always used. Frames leave through memory; nothing is imported.

This makes the GPU lane headless-testable: a runtime with no browser
(node, wasmtime, jsc) can instantiate the module with `{}` imports,
drive entries, and assert the command stream byte-for-byte — the
conformance suite in `web/tests/tape.test.mjs` does exactly that and
runs in CI.

## Wire format (version 1)

Little-endian `u32` words. A non-empty tape begins with
`[0x50415451 "QTAP"][version]`, followed by ops:

```
[opcode u32] [fixed args, one word each] [payloads]
```

- `f64` args are TWO words (the IEEE-754 bit pattern, low word first).
- `f32` args are ONE word (their bit pattern).
- Byte payloads (shader source, buffer/texture data, entry-point
  names, bundle lists) are inline: `[len u32][bytes… zero-padded to a
  word boundary]`. Payloads follow the fixed words, in argument order.
- Ops that create glue-side objects carry a **driver-minted
  destination id as their first word**, with the high bit set
  (`0x8000_0000 | n`). Glue-minted ids (`registerCanvas`) stay below
  the high bit; one handle table serves both.

The opcode table is `Op` in
`crates/gpu/quanta-core/src/driver/webgpu/tape.rs`; the interpreter is
`web/src/tape.ts`. Opcodes are never renumbered or reused — additions
append, and an incompatible change bumps the version word.

## The drain contract

The glue drains after **every** wasm entry returns: frame entries,
`runReturningBytes` invocations, and the async setters (a completion
that runs wasm code may append — the drain after the setter call picks
it up). JS is single-threaded, so ops execute in exactly emission
order and async completions never interleave a drain.

Embedders driving raw exports (a frame loop calling its own export)
must route the call through `QuantaModule.enter(...)` so the drain
runs; the module's own helpers do this internally.

## Push-state (no sync queries)

The driver never asks the environment anything synchronously:

- `__quanta_env_init(available, preferred_format)` — pushed once
  right after instantiation.
- `__quanta_canvas_dims(canvas, w, h)` — pushed at `registerCanvas`
  and whenever the glue (which owns the backing store — R1) changes a
  canvas's size. The driver also records dims it sets itself
  (`configure`, offscreen creation) at emit time, so a
  configure→acquire sequence inside one wasm entry sees its own
  size without waiting for a drain.
- mapAsync readbacks are **push copies**: the `MapAsyncRead` op
  carries a destination pointer; at promise resolution the glue copies
  the mapped bytes into wasm memory (re-viewing it — memory may have
  grown), unmaps, and only then resolves the task.

## Error surfacing

A tape op that fails executes glue-side, after the emitting wasm entry
has already returned — so failures surface as glue-side exceptions
(rejected promises / console errors), not as Rust `Result`s. The one
user-reachable case: creating a WebGPU context on a canvas that
already handed out a 2d/webgl context throws at drain rather than
returning `NotSupported` from the Rust call. This is the deferred
boundary's honest cost; validation that must be a Rust error belongs
before emission.

## Top-level completion

`quanta_complete_bytes` / `quanta_complete_err` (the
`runReturningBytes` result channel) are tape ops too
(`CompleteBytes` / `CompleteErr`) — the module keeps zero imports even
for its own completion plumbing.
