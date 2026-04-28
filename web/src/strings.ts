// String marshalling across the wasm boundary.
//
// Wasm passes UTF-8 byte ranges as `(ptr: u32, len: u32)`. JS reads the
// range out of the wasm linear memory using TextDecoder. Going the other
// direction (JS → wasm) is rarer in this driver — error messages cross
// pre-formatted from Rust. When we do need it, the wasm side allocates
// a buffer first and JS writes into it.

const decoder = new TextDecoder("utf-8");

/**
 * Read a UTF-8 string from wasm memory. The returned string is a fresh
 * JS string — safe to hold across subsequent wasm calls that may
 * relocate the memory buffer.
 */
export function readUtf8(memory: WebAssembly.Memory, ptr: number, len: number): string {
  // `subarray` would create a view; we want a copy because TextDecoder.decode
  // on a view-into-the-WebAssembly-memory may break across memory growth.
  const bytes = new Uint8Array(memory.buffer, ptr, len).slice();
  return decoder.decode(bytes);
}

/**
 * Read a Uint8Array copy from wasm memory. Used for buffer write data
 * paths where we need a fresh array independent of the wasm memory.
 */
export function readBytes(memory: WebAssembly.Memory, ptr: number, len: number): Uint8Array {
  return new Uint8Array(memory.buffer, ptr, len).slice();
}

/**
 * Read a borrowed Uint8Array view (no copy). Only safe to use until the
 * next wasm call that might grow memory. WebGPU `writeBuffer` /
 * `writeTexture` copy synchronously into the GPU queue, so a borrowed
 * view is fine for that path.
 */
export function viewBytes(memory: WebAssembly.Memory, ptr: number, len: number): Uint8Array {
  return new Uint8Array(memory.buffer, ptr, len);
}
