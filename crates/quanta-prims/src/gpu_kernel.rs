//! GPU `#[quanta::kernel]` entry points + device-callable
//! cooperative primitives.
//!
//! Each device function in this module is meant to be called from
//! inside a user `#[quanta::kernel]`. They consume per-thread
//! values and produce per-thread results that depend on every
//! thread in the workgroup — the "block-cooperative" part.
//!
//! ## Host vs wasm resolution
//!
//! Device functions compile in two places:
//!
//! 1. **Wasm shell** (per-kernel) — the kernel macro splices the
//!    source into a wasm32 module that already declares the
//!    `quanta` wasm-import intrinsics (`reduce_add_u32`,
//!    `shuffle_u32`, …). Calls resolve to the real subgroup ops.
//! 2. **Host** (this crate's build) — the device function still
//!    exists as ordinary Rust, so referenced intrinsics need a
//!    host-side declaration too. We provide identity stubs below.
//!
//! This dual-resolution pattern is the standard trick for any
//! quanta-prims-style crate that lifts subgroup intrinsics into
//! reusable cooperative primitives.

// `quanta::*` brings in `quark_id`, `nucleus_id`, `proton_id`,
// and the `#[quanta::kernel]` / `#[quanta::device]` machinery.
// Suppress unused-import warning on the host build — these names
// are referenced in the kernel and device fn bodies, which the
// macro lifts into wasm at compile time.
#[allow(unused_imports)]
use quanta::*;

// ── Subgroup intrinsic shims ──────────────────────────────────────

/// Wasm-side declaration of the `reduce_add_u32` subgroup intrinsic.
/// When the device function source is spliced into the kernel
/// macro's wasm shell, the shell's own extern block resolves the
/// call. This declaration covers the path where rustc compiles
/// this file directly for wasm32 (rare but supported).
#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "quanta")]
unsafe extern "C" {
    fn reduce_add_u32(value: u32) -> u32;
}

/// Host-side stub for `reduce_add_u32`. Returns the input
/// unchanged — single-lane reduce semantics, exactly what the CPU
/// driver also does for `SubgroupReduceAdd`.
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn reduce_add_u32(value: u32) -> u32 {
    value
}

// ── Device functions (cooperative primitives) ─────────────────────

/// Block-wide sum reduction. Every thread in the workgroup
/// contributes its `value`; the function returns the workgroup-
/// wide sum (replicated in every lane that participated).
///
/// Implementation: a single warp-level reduce. With the current
/// `workgroup_size = 32` default, one subgroup covers the whole
/// workgroup — the warp result IS the block result. Larger
/// workgroups need a cross-warp stage (planned).
///
/// Constraints today (will relax):
/// - `workgroup_size <= subgroup_size`. On Apple / NVIDIA the
///   subgroup is 32 lanes; on AMD it's 64.
#[allow(dead_code, unused_unsafe)]
#[quanta::device]
fn block_reduce_add_u32_kernel(value: u32) -> u32 {
    // `unsafe` is required on the wasm side (FFI call into the
    // `quanta` import module) and elided to a safe call on the
    // host build via the stub above. The `unused_unsafe` allow
    // covers the host path.
    unsafe { reduce_add_u32(value) }
}

// ── Top-level kernels (convenience wrappers) ──────────────────────

/// Convenience kernel: reads `N` inputs, computes one per-block
/// sum, writes one output per block.
///
/// Caller must:
/// - Allocate `data` with `N = workgroup_size * num_blocks`.
/// - Allocate `out` with `num_blocks` elements.
/// - Dispatch with `quark_count = N`.
#[quanta::kernel(workgroup_size = [32, 1, 1])]
pub fn block_reduce_add_u32_buffer(data: &[u32], out: &mut [u32]) {
    let i = quark_id();
    let block = nucleus_id();
    let lane = proton_id();

    let value = data[i as usize];
    let block_sum = block_reduce_add_u32_kernel(value);

    if lane == 0 {
        out[block as usize] = block_sum;
    }
}
