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
//!    `shuffle_u32`, `shared_load_u32`, `shared_store_u32`,
//!    `barrier`, `subgroup_size`, `proton_id`, …). Calls resolve
//!    to the real subgroup / shared-memory ops.
//! 2. **Host** (this crate's build) — the device function still
//!    exists as ordinary Rust, so referenced intrinsics need a
//!    host-side declaration too. We provide identity stubs below
//!    that match the single-thread semantics the CPU driver uses.
//!
//! This dual-resolution pattern is the standard trick for any
//! quanta-prims-style crate that lifts subgroup intrinsics into
//! reusable cooperative primitives.
//!
//! ## Cooperative storage convention
//!
//! Multi-warp primitives (block reduce, scan, sort) need a
//! workgroup-shared scratch area. Quanta's `#[quanta::shared]`
//! decls are harvested at the kernel level, not the device-fn
//! level — so each primitive documents which **slot** it expects
//! the caller's kernel to have declared, and the caller is
//! responsible for the `#[quanta::shared] let SCRATCH: [u32; N];`
//! statement.

// `quanta::*` brings in `quark_id`, `nucleus_id`, `proton_id`,
// and the `#[quanta::kernel]` / `#[quanta::device]` machinery.
// Suppress unused-import warning on the host build — these names
// are referenced in the kernel and device fn bodies, which the
// macro lifts into wasm at compile time.
#[allow(unused_imports)]
use quanta::*;

// ── Subgroup + shared-memory intrinsic shims ──────────────────────

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "quanta")]
unsafe extern "C" {
    fn reduce_add_u32(value: u32) -> u32;
    fn subgroup_size() -> u32;
    fn proton_id() -> u32;
    fn barrier();
    fn shared_load_u32(slot: u32, index: u32) -> u32;
    fn shared_store_u32(slot: u32, index: u32, val: u32);
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn reduce_add_u32(value: u32) -> u32 {
    value
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn subgroup_size() -> u32 {
    1
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn proton_id() -> u32 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn barrier() {}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn shared_load_u32(_slot: u32, _index: u32) -> u32 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn shared_store_u32(_slot: u32, _index: u32, _val: u32) {}

// ── Device functions (cooperative primitives) ─────────────────────

/// Reserved shared-memory slot for the block-reduce scratch
/// area. The caller's kernel must declare
/// `#[quanta::shared] let block_reduce_scratch: [u32; 32];`
/// (or equivalent at slot 0) before invoking
/// `block_reduce_add_u32_kernel`.
///
/// 32 entries is the max number of warps a typical workgroup
/// has (workgroup_size = 1024 / subgroup_size = 32 on Apple/NVIDIA).
pub const BLOCK_REDUCE_SCRATCH_SLOT: u32 = 0;

/// Block-wide sum reduction across the threads of a workgroup.
///
/// Every thread in the workgroup contributes its `value`; the
/// function returns the block-wide sum in **lane 0 of the
/// workgroup** (proton_id == 0). Other lanes receive an
/// unspecified partial value — callers that need the result in
/// every lane should write it through shared memory after this
/// call.
///
/// **Caller contract:** the surrounding kernel must declare a
/// `[u32; 32]` shared array at slot
/// [`BLOCK_REDUCE_SCRATCH_SLOT`]. The function uses it as
/// scratch for the cross-warp re-reduction stage and assumes it
/// is otherwise unused at the call site (the contents are
/// overwritten).
///
/// # Algorithm
///
/// 1. **Warp-level reduce** — each subgroup uses the
///    `reduce_add_u32` intrinsic; every lane in the subgroup
///    ends with the warp sum.
/// 2. **Cross-warp aggregation** — lane 0 of each warp writes
///    the warp sum to `scratch[warp_id]`; a workgroup barrier
///    publishes all partials.
/// 3. **Re-reduce** — the first warp reads `scratch[lane]`, runs
///    `reduce_add_u32` again, and lane 0 holds the total.
///
/// # Constraints
///
/// - `workgroup_size <= subgroup_size * subgroup_size`. On
///   Apple/NVIDIA (subgroup_size = 32) this caps at 1024 lanes;
///   on AMD (64) at 4096 lanes. Comfortably above the typical
///   GEMM/sort workgroup_size of 256 or 512.
/// - `workgroup_size` must be a multiple of `subgroup_size`.
///   Mixed partial-warp workgroups aren't supported.
#[allow(dead_code, unused_unsafe)]
#[quanta::device]
fn block_reduce_add_u32_kernel(value: u32) -> u32 {
    // Stage 1: warp-level reduce.
    let warp_sum = unsafe { reduce_add_u32(value) };

    // Stage 2: lane 0 of each warp writes its partial sum.
    let sub_size = unsafe { subgroup_size() };
    let lane_in_block = unsafe { proton_id() };
    let lane_in_warp = lane_in_block % sub_size;
    let warp_id = lane_in_block / sub_size;

    if lane_in_warp == 0 {
        unsafe { shared_store_u32(0, warp_id, warp_sum) };
    }
    unsafe { barrier() };

    // Stage 3: the first warp re-reduces the partials. We unify
    // the path with an unconditional shared_load: every lane in
    // the workgroup reads `scratch[lane_in_warp]`, but only the
    // first warp's contribution feeds into the final reduce
    // (other warps' reduce-result is discarded — the caller
    // reads from lane 0). This avoids the conditionally-
    // initialised local pattern that the WASM lowerer struggles
    // with.
    let partial = unsafe { shared_load_u32(0, lane_in_warp) };
    unsafe { reduce_add_u32(partial) }
}

// ── Top-level kernels (convenience wrappers) ──────────────────────

/// Convenience kernel: reads `N` inputs, computes one per-block
/// sum, writes one output per block.
///
/// Workgroup size: 256 threads. Each block sums its 256 inputs;
/// the cross-warp reduce inside the kernel handles the 8 warps
/// of partial sums (on Apple/NVIDIA subgroup_size = 32).
///
/// Caller must:
/// - Allocate `data` with `N = 256 * num_blocks`.
/// - Allocate `out` with `num_blocks` elements.
/// - Dispatch with `quark_count = N`.
#[quanta::kernel(workgroup_size = [256, 1, 1])]
pub fn block_reduce_add_u32_buffer(data: &[u32], out: &mut [u32]) {
    #[quanta::shared]
    let block_reduce_scratch: [u32; 32];

    let i = quark_id();
    let block = nucleus_id();
    let lane = proton_id();

    // Zero the scratch slot for warps that won't write it. With
    // workgroup_size = 256 and subgroup_size = 32, we have 8
    // warps; entries 8..32 must be zero before the re-reduce.
    if lane < 32u32 {
        block_reduce_scratch[lane] = 0u32;
    }
    barrier();

    let value = data[i as usize];
    let block_sum = block_reduce_add_u32_kernel(value);

    // Lane 0 of the block holds the final sum.
    if lane == 0u32 {
        out[block as usize] = block_sum;
    }
}
