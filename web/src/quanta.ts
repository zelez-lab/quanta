// Quanta WebGPU glue — entry point for browser-side smoke tests.
//
// Public API:
//
//   const mod = await instantiate("./web_add_one.wasm");
//   const bytes = await mod.runReturningBytes("web_add_one_run");
//
// Internally:
//
//   1. `instantiate` fetches and instantiates the wasm module with NO
//      imports at all (dija R10): the module talks to the browser
//      through the command tape, never through `env:*`.
//   2. The environment the driver used to query synchronously —
//      `navigator.gpu` and the preferred canvas format — is PUSHED into
//      wasm right after instantiation via `__quanta_env_init`.
//   3. Every call into wasm goes through `state.enter`, which drains the
//      tape (`tape.ts`) the moment the call returns: that is where the
//      WebGPU work actually happens.
//   4. Smoke tests export a function like `web_add_one_run(task: u32)`
//      that runs the test and eventually appends a `CompleteBytes` /
//      `CompleteErr` op naming the task.
//   5. `runReturningBytes` allocates a fresh top-level task id, calls
//      the export, and returns a Promise those ops settle at drain time.

import { HandleTable } from "./handles.js";
import { drainTape, type GlueState, type TapeExports } from "./tape.js";

/**
 * The wasm exports used by the glue. Smoke tests add their own exports
 * (e.g. `web_add_one_run`) on top of this base.
 */
interface BaseExports extends TapeExports {
  memory: WebAssembly.Memory;
}

export interface QuantaModule {
  /**
   * Raw access for callers that need to invoke other exports directly —
   * a host-driven frame loop calling its own entry, say. Wrap every such
   * call in [`enter`]: an export that returns without a drain leaves its
   * WebGPU work queued on the tape.
   */
  exports: WebAssembly.Exports;
  /**
   * Invoke a wasm entry and perform the commands it appended. Every call
   * this module makes into wasm goes through here; an embedder driving
   * `exports` itself must do the same.
   */
  enter(call: () => void): void;
  /** Diagnostic accessor to inspect handle pressure. */
  liveHandles(): number;
  /**
   * Diagnostic accessor: how many drains actually carried commands —
   * the wasm↔JS crossing count a frame loop wants to keep flat.
   */
  drains(): number;
  /**
   * Register a canvas for presentation (step 096). The returned id is
   * what `SurfaceTarget::Canvas { canvas }` names on the Rust side —
   * pass it into the wasm entry point. The registration stays live
   * until the page drops the module; quanta never releases it (the
   * embedder owns the canvas, quanta drives only its backing size).
   */
  registerCanvas(canvas: HTMLCanvasElement | OffscreenCanvas): number;
  /**
   * Invoke a wasm export of the form
   * `extern "C" fn run(task: u32, ...)` and resolve with the bytes the
   * Rust side hands back through the tape's `CompleteBytes` op. Reject
   * on `CompleteErr`. Trailing `args` are passed after the task id
   * (e.g. a registered canvas handle).
   */
  runReturningBytes(exportName: string, ...args: number[]): Promise<Uint8Array>;
}

/**
 * `navigator.gpu` presence and its preferred canvas format, as the
 * `(available, format_code)` pair `__quanta_env_init` takes. Format
 * codes mirror ffi.rs `format`: rgba8unorm = 0, bgra8unorm = 1.
 */
function environment(): [number, number] {
  const gpu = typeof navigator === "undefined" ? undefined : navigator.gpu;
  if (gpu === undefined) return [0, 0];
  return [1, gpu.getPreferredCanvasFormat() === "rgba8unorm" ? 0 : 1];
}

export async function instantiate(wasmUrl: string): Promise<QuantaModule> {
  const handles = new HandleTable();
  const state: GlueState = {
    // `memory` is filled in below — we need to instantiate first to read
    // the exported memory. Default to a zero-page placeholder; never
    // dereferenced before the post-instantiate fixup.
    memory: new WebAssembly.Memory({ initial: 0 }),
    exports: null,
    handles,
    drains: 0,
    topLevelTasks: new Map(),
    enter(call: () => void): void {
      try {
        call();
      } finally {
        drainTape(state);
      }
    },
  };

  let nextTopLevelTask = 1;

  // No imports: the module is self-contained, and everything it wants
  // from the host rides the tape it fills in.
  const imports: WebAssembly.Imports = {};

  const response = fetch(wasmUrl);
  let result: WebAssembly.WebAssemblyInstantiatedSource;
  if (typeof WebAssembly.instantiateStreaming === "function") {
    result = await WebAssembly.instantiateStreaming(response, imports);
  } else {
    const buf = await (await response).arrayBuffer();
    result = await WebAssembly.instantiate(buf, imports);
  }
  const instance = result.instance;
  const exports = instance.exports as unknown as BaseExports;

  state.memory = exports.memory;
  state.exports = exports;

  // Push what the driver can no longer ask for synchronously.
  const [available, preferredFormat] = environment();
  exports.__quanta_env_init(available, preferredFormat);

  return {
    exports: instance.exports,
    enter: (call) => state.enter(call),
    liveHandles: () => handles.size(),
    drains: () => state.drains,
    registerCanvas: (canvas) => {
      const id = handles.alloc(canvas);
      // The embedder owns this canvas' backing size until a surface
      // configures it, so the driver's first read of the extent has to
      // come from here.
      exports.__quanta_canvas_dims(id, canvas.width, canvas.height);
      return id;
    },
    runReturningBytes(exportName: string, ...args: number[]): Promise<Uint8Array> {
      const fn = (instance.exports as Record<string, unknown>)[exportName];
      if (typeof fn !== "function") {
        return Promise.reject(
          new Error(`quanta glue: export ${exportName} not a function`),
        );
      }
      return new Promise<Uint8Array>((resolve, reject) => {
        const task = nextTopLevelTask++;
        state.topLevelTasks.set(task, { resolve, reject });
        try {
          state.enter(() => (fn as (t: number, ...rest: number[]) => void)(task, ...args));
        } catch (e) {
          state.topLevelTasks.delete(task);
          reject(e instanceof Error ? e : new Error(String(e)));
        }
      });
    },
  };
}
