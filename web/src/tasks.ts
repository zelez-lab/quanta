// Async task plumbing.
//
// WebGPU exposes async work through Promises (`requestAdapter`,
// `requestDevice`, `mapAsync`, `onSubmittedWorkDone`). Wasm cannot
// `await` directly. The contract:
//
//   1. Rust-side `Promise` future allocates a `task_id` and passes it
//      to the FFI import.
//   2. JS-side `runAsync(taskId, promise)` chains a `.then` /
//      `.catch` onto the WebGPU promise.
//   3. When the promise resolves, JS calls back into the wasm export
//      `quanta_resolve(task_id, handle)`. On rejection it calls
//      `quanta_reject(task_id)`.
//   4. The Rust executor wakes the future associated with that task id;
//      the future's `poll` returns `Ready`.
//
// Only the wasm side allocates task ids. The JS side never picks a
// task id — it just forwards the one wasm gave it. This avoids the
// JS-and-wasm-both-handing-out-ids hazard.

export interface WasmExports {
  /** Wake a Rust Promise with a successful result handle (0 for unit). */
  quanta_resolve(task: number, handle: number): void;
  /** Wake a Rust Promise with a rejection. */
  quanta_reject(task: number): void;
}

/**
 * Bind a JS Promise to a wasm-side task id. Resolution + rejection are
 * forwarded to the wasm executor via its exported callback functions.
 *
 * `mapHandle` extracts the integer handle to hand back on success.
 * Most async ops resolve to a JS object; we allocate a handle for the
 * object inside `mapHandle` (using the closure's captured handle table).
 * Some resolve to `undefined` (e.g. `mapAsync`); those use `() => 0`.
 */
export function bindTask<T>(
  exports: WasmExports,
  task: number,
  promise: Promise<T>,
  mapHandle: (value: T) => number,
): void {
  promise.then(
    (value) => {
      let handle: number;
      try {
        handle = mapHandle(value);
      } catch (e) {
        console.error("quanta glue: mapHandle threw", e);
        exports.quanta_reject(task);
        return;
      }
      exports.quanta_resolve(task, handle);
    },
    (err) => {
      console.error("quanta glue: task rejected", err);
      exports.quanta_reject(task);
    },
  );
}
