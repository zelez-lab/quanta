//! GPU `#[quanta::kernel]` entry points + device-callable
//! cooperative primitives.
//!
//! Each device function in this module is meant to be called from
//! inside a user `#[quanta::kernel]`. They consume per-thread
//! values and produce per-thread results that depend on every
//! thread in the workgroup — the "block-cooperative" part.
//!
//! ## Block reduce family
//!
//! For each `(op, ty)` in `{add, min, max} × {u32, i32, f32}`,
//! quanta-prims ships a `block_reduce_<op>_<ty>_kernel` device
//! function with the same signature: takes a per-thread value,
//! returns the workgroup-wide reduction in **lane 0**. The nine
//! variants share the two-stage warp/cross-warp algorithm; only
//! the per-warp intrinsic and the cross-warp identity element
//! differ.
//!
//! Each device function expects the caller's kernel to declare a
//! `[<ty>; 32]` shared array at slot [`BLOCK_REDUCE_SCRATCH_SLOT`]
//! (= 0). The convenience top-level kernels in this file do that
//! initialization and dispatch.
//!
//! ## Host vs wasm resolution
//!
//! Device functions compile in two places: the kernel macro's
//! wasm shell (where the `quanta` import block resolves
//! `reduce_*`, `shared_*`, `barrier`, …), and the host build of
//! this crate. We provide host stubs that match the single-thread
//! semantics the CPU driver uses.

// `quanta::*` brings in `quark_id`, `nucleus_id`, `proton_id`,
// and the `#[quanta::kernel]` / `#[quanta::device]` machinery.
#[allow(unused_imports)]
use quanta::*;

// ── Subgroup + shared-memory intrinsic shims ──────────────────────

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "quanta")]
unsafe extern "C" {
    fn reduce_add_u32(value: u32) -> u32;
    fn reduce_add_i32(value: i32) -> i32;
    fn reduce_add_f32(value: f32) -> f32;
    fn reduce_min_u32(value: u32) -> u32;
    fn reduce_min_i32(value: i32) -> i32;
    fn reduce_min_f32(value: f32) -> f32;
    fn reduce_max_u32(value: u32) -> u32;
    fn reduce_max_i32(value: i32) -> i32;
    fn reduce_max_f32(value: f32) -> f32;
    fn subgroup_size() -> u32;
    fn proton_id() -> u32;
    fn barrier();
    fn shared_load_u32(slot: u32, index: u32) -> u32;
    fn shared_load_i32(slot: u32, index: u32) -> i32;
    fn shared_load_f32(slot: u32, index: u32) -> f32;
    fn shared_store_u32(slot: u32, index: u32, val: u32);
    fn shared_store_i32(slot: u32, index: u32, val: i32);
    fn shared_store_f32(slot: u32, index: u32, val: f32);
}

// Host stubs. Single-lane semantics: reduce/scan/min/max return
// the input; shared mem is a no-op; barrier is a no-op.
#[cfg(not(target_arch = "wasm32"))]
mod host_stubs {
    #![allow(dead_code)]
    pub fn reduce_add_u32(v: u32) -> u32 {
        v
    }
    pub fn reduce_add_i32(v: i32) -> i32 {
        v
    }
    pub fn reduce_add_f32(v: f32) -> f32 {
        v
    }
    pub fn reduce_min_u32(v: u32) -> u32 {
        v
    }
    pub fn reduce_min_i32(v: i32) -> i32 {
        v
    }
    pub fn reduce_min_f32(v: f32) -> f32 {
        v
    }
    pub fn reduce_max_u32(v: u32) -> u32 {
        v
    }
    pub fn reduce_max_i32(v: i32) -> i32 {
        v
    }
    pub fn reduce_max_f32(v: f32) -> f32 {
        v
    }
    pub fn subgroup_size() -> u32 {
        1
    }
    pub fn proton_id() -> u32 {
        0
    }
    pub fn barrier() {}
    pub fn shared_load_u32(_: u32, _: u32) -> u32 {
        0
    }
    pub fn shared_load_i32(_: u32, _: u32) -> i32 {
        0
    }
    pub fn shared_load_f32(_: u32, _: u32) -> f32 {
        0.0
    }
    pub fn shared_store_u32(_: u32, _: u32, _: u32) {}
    pub fn shared_store_i32(_: u32, _: u32, _: i32) {}
    pub fn shared_store_f32(_: u32, _: u32, _: f32) {}
}
#[cfg(not(target_arch = "wasm32"))]
use host_stubs::*;

// ── Public constants ──────────────────────────────────────────────

/// Reserved shared-memory slot for the block-reduce scratch
/// area. Every block-reduce device function in this crate uses
/// this slot. The caller's kernel must declare an array of the
/// appropriate type, sized to hold one entry per warp — for the
/// typical workgroup_size ≤ 1024 case, 32 entries suffice on
/// Apple/NVIDIA (subgroup_size=32) and Apple/AMD (subgroup_size=64
/// → workgroup_size ≤ 4096 → 64 entries).
pub const BLOCK_REDUCE_SCRATCH_SLOT: u32 = 0;

// ── Device functions: block reduce family ─────────────────────────
//
// Nine variants, all built on the same recipe:
//
//   1. Warp-level reduce via the matching subgroup intrinsic.
//   2. Lane 0 of each warp writes its partial to scratch[warp_id].
//   3. Barrier.
//   4. Every lane in the workgroup reads scratch[lane_in_warp] and
//      runs another warp-level reduce. Warp 0 lane 0 holds the
//      workgroup total; other lanes hold a same-or-redundant
//      value.
//
// The caller's kernel is responsible for initializing the scratch
// slots with the **identity element** for the chosen operation:
// 0 for add, type::MAX for min, type::MIN for max. The
// convenience top-level kernels below do this initialization
// before calling the device fn.

/// Block-wide u32 sum reduction. Result in lane 0 of the
/// workgroup. See module-level docs for caller contract.
#[allow(dead_code, unused_unsafe)]
#[quanta::device]
fn block_reduce_add_u32_kernel(value: u32) -> u32 {
    let warp_sum = unsafe { reduce_add_u32(value) };
    let sub_size = unsafe { subgroup_size() };
    let lane_in_block = unsafe { proton_id() };
    let lane_in_warp = lane_in_block % sub_size;
    let warp_id = lane_in_block / sub_size;
    if lane_in_warp == 0 {
        unsafe { shared_store_u32(0, warp_id, warp_sum) };
    }
    unsafe { barrier() };
    let partial = unsafe { shared_load_u32(0, lane_in_warp) };
    unsafe { reduce_add_u32(partial) }
}

/// Block-wide i32 sum reduction. Result in lane 0 of the
/// workgroup. See module-level docs for caller contract.
#[allow(dead_code, unused_unsafe)]
#[quanta::device]
fn block_reduce_add_i32_kernel(value: i32) -> i32 {
    let warp_sum = unsafe { reduce_add_i32(value) };
    let sub_size = unsafe { subgroup_size() };
    let lane_in_block = unsafe { proton_id() };
    let lane_in_warp = lane_in_block % sub_size;
    let warp_id = lane_in_block / sub_size;
    if lane_in_warp == 0 {
        unsafe { shared_store_i32(0, warp_id, warp_sum) };
    }
    unsafe { barrier() };
    let partial = unsafe { shared_load_i32(0, lane_in_warp) };
    unsafe { reduce_add_i32(partial) }
}

/// Block-wide f32 sum reduction. Result in lane 0 of the
/// workgroup. See module-level docs for caller contract.
#[allow(dead_code, unused_unsafe)]
#[quanta::device]
fn block_reduce_add_f32_kernel(value: f32) -> f32 {
    let warp_sum = unsafe { reduce_add_f32(value) };
    let sub_size = unsafe { subgroup_size() };
    let lane_in_block = unsafe { proton_id() };
    let lane_in_warp = lane_in_block % sub_size;
    let warp_id = lane_in_block / sub_size;
    if lane_in_warp == 0 {
        unsafe { shared_store_f32(0, warp_id, warp_sum) };
    }
    unsafe { barrier() };
    let partial = unsafe { shared_load_f32(0, lane_in_warp) };
    unsafe { reduce_add_f32(partial) }
}

/// Block-wide u32 min reduction. Result in lane 0 of the
/// workgroup. Identity element: u32::MAX. See module-level docs
/// for caller contract.
#[allow(dead_code, unused_unsafe)]
#[quanta::device]
fn block_reduce_min_u32_kernel(value: u32) -> u32 {
    let warp_min = unsafe { reduce_min_u32(value) };
    let sub_size = unsafe { subgroup_size() };
    let lane_in_block = unsafe { proton_id() };
    let lane_in_warp = lane_in_block % sub_size;
    let warp_id = lane_in_block / sub_size;
    if lane_in_warp == 0 {
        unsafe { shared_store_u32(0, warp_id, warp_min) };
    }
    unsafe { barrier() };
    let partial = unsafe { shared_load_u32(0, lane_in_warp) };
    unsafe { reduce_min_u32(partial) }
}

/// Block-wide i32 min reduction. Result in lane 0 of the
/// workgroup. Identity element: i32::MAX. See module-level docs
/// for caller contract.
#[allow(dead_code, unused_unsafe)]
#[quanta::device]
fn block_reduce_min_i32_kernel(value: i32) -> i32 {
    let warp_min = unsafe { reduce_min_i32(value) };
    let sub_size = unsafe { subgroup_size() };
    let lane_in_block = unsafe { proton_id() };
    let lane_in_warp = lane_in_block % sub_size;
    let warp_id = lane_in_block / sub_size;
    if lane_in_warp == 0 {
        unsafe { shared_store_i32(0, warp_id, warp_min) };
    }
    unsafe { barrier() };
    let partial = unsafe { shared_load_i32(0, lane_in_warp) };
    unsafe { reduce_min_i32(partial) }
}

/// Block-wide f32 min reduction. Result in lane 0 of the
/// workgroup. Identity element: f32::INFINITY. See module-level
/// docs for caller contract.
#[allow(dead_code, unused_unsafe)]
#[quanta::device]
fn block_reduce_min_f32_kernel(value: f32) -> f32 {
    let warp_min = unsafe { reduce_min_f32(value) };
    let sub_size = unsafe { subgroup_size() };
    let lane_in_block = unsafe { proton_id() };
    let lane_in_warp = lane_in_block % sub_size;
    let warp_id = lane_in_block / sub_size;
    if lane_in_warp == 0 {
        unsafe { shared_store_f32(0, warp_id, warp_min) };
    }
    unsafe { barrier() };
    let partial = unsafe { shared_load_f32(0, lane_in_warp) };
    unsafe { reduce_min_f32(partial) }
}

/// Block-wide u32 max reduction. Result in lane 0 of the
/// workgroup. Identity element: 0. See module-level docs for
/// caller contract.
#[allow(dead_code, unused_unsafe)]
#[quanta::device]
fn block_reduce_max_u32_kernel(value: u32) -> u32 {
    let warp_max = unsafe { reduce_max_u32(value) };
    let sub_size = unsafe { subgroup_size() };
    let lane_in_block = unsafe { proton_id() };
    let lane_in_warp = lane_in_block % sub_size;
    let warp_id = lane_in_block / sub_size;
    if lane_in_warp == 0 {
        unsafe { shared_store_u32(0, warp_id, warp_max) };
    }
    unsafe { barrier() };
    let partial = unsafe { shared_load_u32(0, lane_in_warp) };
    unsafe { reduce_max_u32(partial) }
}

/// Block-wide i32 max reduction. Result in lane 0 of the
/// workgroup. Identity element: i32::MIN. See module-level docs
/// for caller contract.
#[allow(dead_code, unused_unsafe)]
#[quanta::device]
fn block_reduce_max_i32_kernel(value: i32) -> i32 {
    let warp_max = unsafe { reduce_max_i32(value) };
    let sub_size = unsafe { subgroup_size() };
    let lane_in_block = unsafe { proton_id() };
    let lane_in_warp = lane_in_block % sub_size;
    let warp_id = lane_in_block / sub_size;
    if lane_in_warp == 0 {
        unsafe { shared_store_i32(0, warp_id, warp_max) };
    }
    unsafe { barrier() };
    let partial = unsafe { shared_load_i32(0, lane_in_warp) };
    unsafe { reduce_max_i32(partial) }
}

/// Block-wide f32 max reduction. Result in lane 0 of the
/// workgroup. Identity element: f32::NEG_INFINITY. See
/// module-level docs for caller contract.
#[allow(dead_code, unused_unsafe)]
#[quanta::device]
fn block_reduce_max_f32_kernel(value: f32) -> f32 {
    let warp_max = unsafe { reduce_max_f32(value) };
    let sub_size = unsafe { subgroup_size() };
    let lane_in_block = unsafe { proton_id() };
    let lane_in_warp = lane_in_block % sub_size;
    let warp_id = lane_in_block / sub_size;
    if lane_in_warp == 0 {
        unsafe { shared_store_f32(0, warp_id, warp_max) };
    }
    unsafe { barrier() };
    let partial = unsafe { shared_load_f32(0, lane_in_warp) };
    unsafe { reduce_max_f32(partial) }
}

// ── Top-level kernels (convenience wrappers) ──────────────────────
//
// One per (op, type) combination, with workgroup_size = 256 and
// identity-initialised scratch. Caller dispatches with
// quark_count = num_blocks * 256.

/// Convenience kernel: u32 sum reduce, one output per block.
/// Workgroup size 256 → up to 8 warps (Apple/NVIDIA).
#[quanta::kernel(workgroup_size = [256, 1, 1])]
pub fn block_reduce_add_u32_buffer(data: &[u32], out: &mut [u32]) {
    #[quanta::shared]
    let scratch: [u32; 32];

    let i = quark_id();
    let block = nucleus_id();
    let lane = proton_id();

    if lane < 32u32 {
        scratch[lane] = 0u32; // identity: 0
    }
    barrier();

    let value = data[i as usize];
    let block_sum = block_reduce_add_u32_kernel(value);

    if lane == 0u32 {
        out[block as usize] = block_sum;
    }
}

/// Convenience kernel: i32 sum reduce, one output per block.
#[quanta::kernel(workgroup_size = [256, 1, 1])]
pub fn block_reduce_add_i32_buffer(data: &[i32], out: &mut [i32]) {
    #[quanta::shared]
    let scratch: [i32; 32];

    let i = quark_id();
    let block = nucleus_id();
    let lane = proton_id();

    if lane < 32u32 {
        scratch[lane] = 0i32; // identity: 0
    }
    barrier();

    let value = data[i as usize];
    let block_sum = block_reduce_add_i32_kernel(value);

    if lane == 0u32 {
        out[block as usize] = block_sum;
    }
}

/// Convenience kernel: f32 sum reduce, one output per block.
#[quanta::kernel(workgroup_size = [256, 1, 1])]
pub fn block_reduce_add_f32_buffer(data: &[f32], out: &mut [f32]) {
    #[quanta::shared]
    let scratch: [f32; 32];

    let i = quark_id();
    let block = nucleus_id();
    let lane = proton_id();

    if lane < 32u32 {
        scratch[lane] = 0.0f32; // identity: 0
    }
    barrier();

    let value = data[i as usize];
    let block_sum = block_reduce_add_f32_kernel(value);

    if lane == 0u32 {
        out[block as usize] = block_sum;
    }
}

/// Convenience kernel: u32 min reduce, one output per block.
/// Scratch identity = u32::MAX.
#[quanta::kernel(workgroup_size = [256, 1, 1])]
pub fn block_reduce_min_u32_buffer(data: &[u32], out: &mut [u32]) {
    #[quanta::shared]
    let scratch: [u32; 32];

    let i = quark_id();
    let block = nucleus_id();
    let lane = proton_id();

    if lane < 32u32 {
        scratch[lane] = 4294967295u32; // u32::MAX
    }
    barrier();

    let value = data[i as usize];
    let block_min = block_reduce_min_u32_kernel(value);

    if lane == 0u32 {
        out[block as usize] = block_min;
    }
}

/// Convenience kernel: i32 min reduce, one output per block.
/// Scratch identity = i32::MAX.
#[quanta::kernel(workgroup_size = [256, 1, 1])]
pub fn block_reduce_min_i32_buffer(data: &[i32], out: &mut [i32]) {
    #[quanta::shared]
    let scratch: [i32; 32];

    let i = quark_id();
    let block = nucleus_id();
    let lane = proton_id();

    if lane < 32u32 {
        scratch[lane] = 2147483647i32; // i32::MAX
    }
    barrier();

    let value = data[i as usize];
    let block_min = block_reduce_min_i32_kernel(value);

    if lane == 0u32 {
        out[block as usize] = block_min;
    }
}

/// Convenience kernel: f32 min reduce, one output per block.
/// Scratch identity = f32::INFINITY.
#[quanta::kernel(workgroup_size = [256, 1, 1])]
pub fn block_reduce_min_f32_buffer(data: &[f32], out: &mut [f32]) {
    #[quanta::shared]
    let scratch: [f32; 32];

    let i = quark_id();
    let block = nucleus_id();
    let lane = proton_id();

    if lane < 32u32 {
        // Large finite sentinel as the min identity. True
        // f32::INFINITY would be ideal but the kernel macro's
        // body parser doesn't recognise `f32::INFINITY` /
        // `f32::from_bits()` paths today. 1e38 is well past any
        // realistic numerical workload's value range while still
        // being a normal float (avoids subnormal / inf edge
        // cases in the min reducer).
        scratch[lane] = 1.0e38f32;
    }
    barrier();

    let value = data[i as usize];
    let block_min = block_reduce_min_f32_kernel(value);

    if lane == 0u32 {
        out[block as usize] = block_min;
    }
}

/// Convenience kernel: u32 max reduce, one output per block.
/// Scratch identity = 0.
#[quanta::kernel(workgroup_size = [256, 1, 1])]
pub fn block_reduce_max_u32_buffer(data: &[u32], out: &mut [u32]) {
    #[quanta::shared]
    let scratch: [u32; 32];

    let i = quark_id();
    let block = nucleus_id();
    let lane = proton_id();

    if lane < 32u32 {
        scratch[lane] = 0u32; // u32::MIN
    }
    barrier();

    let value = data[i as usize];
    let block_max = block_reduce_max_u32_kernel(value);

    if lane == 0u32 {
        out[block as usize] = block_max;
    }
}

/// Convenience kernel: i32 max reduce, one output per block.
/// Scratch identity = i32::MIN.
#[quanta::kernel(workgroup_size = [256, 1, 1])]
pub fn block_reduce_max_i32_buffer(data: &[i32], out: &mut [i32]) {
    #[quanta::shared]
    let scratch: [i32; 32];

    let i = quark_id();
    let block = nucleus_id();
    let lane = proton_id();

    if lane < 32u32 {
        scratch[lane] = -2147483648i32; // i32::MIN
    }
    barrier();

    let value = data[i as usize];
    let block_max = block_reduce_max_i32_kernel(value);

    if lane == 0u32 {
        out[block as usize] = block_max;
    }
}

/// Convenience kernel: f32 max reduce, one output per block.
/// Scratch identity = f32::NEG_INFINITY.
#[quanta::kernel(workgroup_size = [256, 1, 1])]
pub fn block_reduce_max_f32_buffer(data: &[f32], out: &mut [f32]) {
    #[quanta::shared]
    let scratch: [f32; 32];

    let i = quark_id();
    let block = nucleus_id();
    let lane = proton_id();

    if lane < 32u32 {
        // Large negative finite sentinel. See min_f32 above for
        // why we avoid f32::NEG_INFINITY.
        scratch[lane] = -1.0e38f32;
    }
    barrier();

    let value = data[i as usize];
    let block_max = block_reduce_max_f32_kernel(value);

    if lane == 0u32 {
        out[block as usize] = block_max;
    }
}
