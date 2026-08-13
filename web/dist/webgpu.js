// WebGPU descriptor shapes — the objects the tape builds up piecemeal.
//
// The Rust side has no JS objects: it opens a descriptor, adds vertex
// buffers / color targets / bind-group entries one op at a time, and
// finally names it in a create op. Each shape below is what lives in the
// handle table between the open and the create; `tape.ts` fills them in
// and hands the finished object to WebGPU.
//
// All long-lived JS objects (devices, buffers, pipelines, …) live in the
// shared `HandleTable` and cross the wasm boundary as `u32`.
import "./webgpu-types.js";
/**
 * The "compare not configured" sentinel for samplers — mirrors
 * `compare::UNSET` in `ffi.rs`, where the real compare ops start at 1 to
 * keep the sentinel out of band.
 */
export const COMPARE_UNSET = 0;
