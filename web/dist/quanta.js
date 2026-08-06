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
import { makeImports } from "./webgpu.js";
export async function instantiate(wasmUrl) {
    const handles = new HandleTable();
    const state = {
        // `memory` is filled in below — we need to instantiate first to read
        // the exported memory. Default to a zero-page placeholder; never
        // dereferenced before the post-instantiate fixup.
        memory: new WebAssembly.Memory({ initial: 0 }),
        exports: null,
        handles,
        syncCalls: 0,
    };
    const topLevelTasks = new Map();
    let nextTopLevelTask = 1;
    const baseImports = makeImports(state);
    const completionImports = {
        quanta_complete_bytes(task, ptr, len) {
            const t = topLevelTasks.get(task);
            if (t === undefined) {
                console.error(`quanta glue: unknown top-level task ${task}`);
                return;
            }
            topLevelTasks.delete(task);
            t.resolve(readBytes(state.memory, ptr, len));
        },
        quanta_complete_err(task, ptr, len) {
            const t = topLevelTasks.get(task);
            if (t === undefined) {
                console.error(`quanta glue: unknown top-level task ${task}`);
                return;
            }
            topLevelTasks.delete(task);
            t.reject(new Error(readUtf8(state.memory, ptr, len)));
        },
    };
    const imports = {
        env: { ...baseImports, ...completionImports },
    };
    const response = fetch(wasmUrl);
    let result;
    if (typeof WebAssembly.instantiateStreaming === "function") {
        result = await WebAssembly.instantiateStreaming(response, imports);
    }
    else {
        const buf = await (await response).arrayBuffer();
        result = await WebAssembly.instantiate(buf, imports);
    }
    const instance = result.instance;
    const exports = instance.exports;
    state.memory = exports.memory;
    state.exports = exports;
    return {
        exports: instance.exports,
        liveHandles: () => handles.size(),
        registerCanvas: (canvas) => handles.alloc(canvas),
        runReturningBytes(exportName, ...args) {
            const fn = instance.exports[exportName];
            if (typeof fn !== "function") {
                return Promise.reject(new Error(`quanta glue: export ${exportName} not a function`));
            }
            return new Promise((resolve, reject) => {
                const task = nextTopLevelTask++;
                topLevelTasks.set(task, { resolve, reject });
                try {
                    fn(task, ...args);
                }
                catch (e) {
                    topLevelTasks.delete(task);
                    reject(e instanceof Error ? e : new Error(String(e)));
                }
            });
        },
    };
}
