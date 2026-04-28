// Handle table — single source of truth for wasm ↔ JS object identity.
//
// Wasm-side code only ever sees `u32` handles. JS-side code resolves a
// handle to the underlying JS object (a `GPUDevice`, a `GPUBuffer`,
// a render-pipeline descriptor under construction, …) via this table.
//
// Handle 0 is reserved as the null handle; live handles start at 1 and
// monotonically increase. There is no recycling — a u32 gives 4 billion
// handles, more than any session is plausibly going to use, and skipping
// recycling means a stale handle never aliases a fresh resource. This
// mirrors how a libc fd table would behave with `O_CLOEXEC` + a fresh
// process.

const NULL_HANDLE = 0;

export class HandleTable {
  private slots: Map<number, unknown> = new Map();
  private next: number = 1;

  /**
   * Allocate a fresh handle for `value`. Returns the handle (always > 0).
   */
  alloc(value: unknown): number {
    const id = this.next++;
    this.slots.set(id, value);
    return id;
  }

  /**
   * Resolve `handle` to its underlying object. Throws if the handle is
   * null or unknown — the wasm side is supposed to track liveness and a
   * lookup miss indicates a real bug worth surfacing.
   */
  get<T>(handle: number): T {
    if (handle === NULL_HANDLE) {
      throw new Error("quanta glue: null handle");
    }
    const v = this.slots.get(handle);
    if (v === undefined) {
      throw new Error(`quanta glue: unknown handle ${handle}`);
    }
    return v as T;
  }

  /**
   * Release a handle, dropping the JS-side reference so the GC can
   * reclaim the object. No-op for the null handle.
   */
  release(handle: number): void {
    if (handle === NULL_HANDLE) return;
    this.slots.delete(handle);
  }

  /**
   * Diagnostic: number of live handles. Used by the smoke tests to
   * detect leaks.
   */
  size(): number {
    return this.slots.size;
  }
}

export { NULL_HANDLE };
