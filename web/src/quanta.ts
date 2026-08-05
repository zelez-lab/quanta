// Quanta WebGPU glue — entry point for browser-side smoke tests.
//
// Public API:
//
//   const mod = await instantiate("./web_add_one.wasm");
//   const bytes = await mod.runReturningBytes("web_add_one_run");
//
// Internally:
//
//   1. `instantiate` fetches and instantiates the wasm module, providing
//      `makeImports` from `webgpu.ts` as the `env` namespace.
//   2. The wasm module's `quanta_resolve` / `quanta_reject` exports are
//      stitched into `state.exports` so async imports can wake the Rust
//      executor.
//   3. Smoke tests export a function like `web_add_one_run(task: u32)`
//      that runs the test and eventually calls back into JS via
//      `quanta_complete_bytes(task, ptr, len)` or
//      `quanta_complete_err(task, msg_ptr, msg_len)`.
//   4. `runReturningBytes` allocates a fresh top-level task id, calls
//      the export, and returns a Promise the imports above resolve.

import { HandleTable } from "./handles.js";
import { readBytes, readUtf8 } from "./strings.js";
import { makeImports, type GlueState } from "./webgpu.js";
import type { WasmExports } from "./tasks.js";

/**
 * The wasm exports used by the glue. Smoke tests add their own exports
 * (e.g. `web_add_one_run`) on top of this base.
 */
interface BaseExports extends WasmExports {
  memory: WebAssembly.Memory;
}

/**
 * Top-level (JS-initiated) tasks waiting on a wasm-driven Promise.
 *
 * Distinct from the `pending_promises` table in the Rust executor: this
 * table tracks the *outermost* JS Promise the host returned from
 * `runReturningBytes`; the Rust table tracks intermediate `await`
 * points. The two namespaces never interleave — both sides only mint
 * ids in their own namespace.
 */
interface TopLevelTask {
  resolve: (bytes: Uint8Array) => void;
  reject: (err: Error) => void;
}

export interface QuantaModule {
  /** Raw access for callers that need to invoke other exports directly. */
  exports: WebAssembly.Exports;
  /** Diagnostic accessor to inspect handle pressure. */
  liveHandles(): number;
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
   * Rust side hands back via `quanta_complete_bytes`. Reject if the
   * Rust side calls `quanta_complete_err`. Trailing `args` are passed
   * after the task id (e.g. a registered canvas handle).
   */
  runReturningBytes(exportName: string, ...args: number[]): Promise<Uint8Array>;
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
    syncCalls: 0,
  };

  const topLevelTasks = new Map<number, TopLevelTask>();
  let nextTopLevelTask = 1;

  const baseImports = makeImports(state);

  const completionImports: WebAssembly.ModuleImports = {
    quanta_complete_bytes(task: number, ptr: number, len: number): void {
      const t = topLevelTasks.get(task);
      if (t === undefined) {
        console.error(`quanta glue: unknown top-level task ${task}`);
        return;
      }
      topLevelTasks.delete(task);
      t.resolve(readBytes(state.memory, ptr, len));
    },
    quanta_complete_err(task: number, ptr: number, len: number): void {
      const t = topLevelTasks.get(task);
      if (t === undefined) {
        console.error(`quanta glue: unknown top-level task ${task}`);
        return;
      }
      topLevelTasks.delete(task);
      t.reject(new Error(readUtf8(state.memory, ptr, len)));
    },
  };

  const imports: WebAssembly.Imports = {
    env: { ...baseImports, ...completionImports },
  };

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

  return {
    exports: instance.exports,
    liveHandles: () => handles.size(),
    registerCanvas: (canvas) => handles.alloc(canvas),
    runReturningBytes(exportName: string, ...args: number[]): Promise<Uint8Array> {
      const fn = (instance.exports as Record<string, unknown>)[exportName];
      if (typeof fn !== "function") {
        return Promise.reject(
          new Error(`quanta glue: export ${exportName} not a function`),
        );
      }
      return new Promise<Uint8Array>((resolve, reject) => {
        const task = nextTopLevelTask++;
        topLevelTasks.set(task, { resolve, reject });
        try {
          (fn as (t: number, ...rest: number[]) => void)(task, ...args);
        } catch (e) {
          topLevelTasks.delete(task);
          reject(e instanceof Error ? e : new Error(String(e)));
        }
      });
    },
  };
}
