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
    fn scan_add_u32(value: u32) -> u32;
    fn scan_add_i32(value: i32) -> i32;
    fn scan_add_f32(value: f32) -> f32;
    fn scan_add_exclusive_u32(value: u32) -> u32;
    fn scan_add_exclusive_i32(value: i32) -> i32;
    fn scan_add_exclusive_f32(value: f32) -> f32;
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
    pub fn scan_add_u32(v: u32) -> u32 {
        v
    }
    pub fn scan_add_i32(v: i32) -> i32 {
        v
    }
    pub fn scan_add_f32(v: f32) -> f32 {
        v
    }
    pub fn scan_add_exclusive_u32(_: u32) -> u32 {
        0
    }
    pub fn scan_add_exclusive_i32(_: i32) -> i32 {
        0
    }
    pub fn scan_add_exclusive_f32(_: f32) -> f32 {
        0.0
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
#[quanta::kernel(workgroup = [256])]
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
#[quanta::kernel(workgroup = [256])]
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
#[quanta::kernel(workgroup = [256])]
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
#[quanta::kernel(workgroup = [256])]
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
#[quanta::kernel(workgroup = [256])]
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
#[quanta::kernel(workgroup = [256])]
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
#[quanta::kernel(workgroup = [256])]
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
#[quanta::kernel(workgroup = [256])]
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
#[quanta::kernel(workgroup = [256])]
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

// ── Device functions: block scan family ──────────────────────────
//
// Block-wide **inclusive** prefix-sum scan. For each lane k in the
// workgroup, returns `sum(value over lanes 0..=k)`. Three-stage
// algorithm:
//
//   1. Warp scan: scan_add_X gives each lane the inclusive sum
//      across its subgroup. Lane (sub_size - 1) of each warp now
//      holds the warp's total.
//   2. Warp totals: lane (sub_size - 1) writes its total to
//      scratch[warp_id]. Barrier. Warp 0 then runs an exclusive
//      scan across the first num_warps scratch slots to get
//      per-warp prefix offsets, writing them back.
//   3. Apply prefix: each lane reads scratch[warp_id] (its warp's
//      prefix offset) and adds it to its warp-local scan result.
//
// Caller contract: the kernel must declare
// `[<ty>; 32]` at slot BLOCK_REDUCE_SCRATCH_SLOT and **zero-init**
// it before calling. Slots beyond num_warps stay 0; the exclusive
// scan in stage 2 ignores them correctly because 0 is the additive
// identity.

/// Block-wide u32 inclusive prefix-sum scan. Returns this lane's
/// running sum (lanes 0..=self inclusive).
#[allow(dead_code, unused_unsafe)]
#[quanta::device]
fn block_scan_add_u32_kernel(value: u32) -> u32 {
    let warp_inc = unsafe { scan_add_u32(value) };
    let sub_size = unsafe { subgroup_size() };
    let lane_in_block = unsafe { proton_id() };
    let lane_in_warp = lane_in_block % sub_size;
    let warp_id = lane_in_block / sub_size;

    // Stage 2a: lane (sub_size - 1) of each warp publishes its
    // warp total (= the last value of the warp-local scan).
    if lane_in_warp == sub_size - 1u32 {
        unsafe { shared_store_u32(0, warp_id, warp_inc) };
    }
    unsafe { barrier() };

    // Stage 2b: warp 0 scans the warp-totals array. All lanes
    // participate in lockstep; lanes beyond num_warps contribute
    // 0 (the additive identity) since the caller zero-inits.
    if warp_id == 0u32 {
        let total = unsafe { shared_load_u32(0, lane_in_warp) };
        let prefix = unsafe { scan_add_exclusive_u32(total) };
        unsafe { shared_store_u32(0, lane_in_warp, prefix) };
    }
    unsafe { barrier() };

    // Stage 3: every lane reads its warp's prefix offset.
    let warp_offset = unsafe { shared_load_u32(0, warp_id) };
    warp_inc + warp_offset
}

/// Block-wide i32 inclusive prefix-sum scan.
#[allow(dead_code, unused_unsafe)]
#[quanta::device]
fn block_scan_add_i32_kernel(value: i32) -> i32 {
    let warp_inc = unsafe { scan_add_i32(value) };
    let sub_size = unsafe { subgroup_size() };
    let lane_in_block = unsafe { proton_id() };
    let lane_in_warp = lane_in_block % sub_size;
    let warp_id = lane_in_block / sub_size;

    if lane_in_warp == sub_size - 1u32 {
        unsafe { shared_store_i32(0, warp_id, warp_inc) };
    }
    unsafe { barrier() };

    if warp_id == 0u32 {
        let total = unsafe { shared_load_i32(0, lane_in_warp) };
        let prefix = unsafe { scan_add_exclusive_i32(total) };
        unsafe { shared_store_i32(0, lane_in_warp, prefix) };
    }
    unsafe { barrier() };

    let warp_offset = unsafe { shared_load_i32(0, warp_id) };
    warp_inc + warp_offset
}

/// Block-wide f32 inclusive prefix-sum scan.
#[allow(dead_code, unused_unsafe)]
#[quanta::device]
fn block_scan_add_f32_kernel(value: f32) -> f32 {
    let warp_inc = unsafe { scan_add_f32(value) };
    let sub_size = unsafe { subgroup_size() };
    let lane_in_block = unsafe { proton_id() };
    let lane_in_warp = lane_in_block % sub_size;
    let warp_id = lane_in_block / sub_size;

    if lane_in_warp == sub_size - 1u32 {
        unsafe { shared_store_f32(0, warp_id, warp_inc) };
    }
    unsafe { barrier() };

    if warp_id == 0u32 {
        let total = unsafe { shared_load_f32(0, lane_in_warp) };
        let prefix = unsafe { scan_add_exclusive_f32(total) };
        unsafe { shared_store_f32(0, lane_in_warp, prefix) };
    }
    unsafe { barrier() };

    let warp_offset = unsafe { shared_load_f32(0, warp_id) };
    warp_inc + warp_offset
}

// ── Top-level kernels: block scan (per-lane output) ──────────────
//
// Output shape mirrors input: one prefix-sum value per input
// element. Caller dispatches with quark_count = num_blocks * 256.

/// Convenience kernel: u32 inclusive prefix-sum scan.
/// `out[i]` = sum of `data[block_start..=i]` where `block_start`
/// is the first lane of the block containing lane `i`.
#[quanta::kernel(workgroup = [256])]
pub fn block_scan_add_u32_buffer(data: &[u32], out: &mut [u32]) {
    #[quanta::shared]
    let scratch: [u32; 32];

    let i = quark_id();
    let lane = proton_id();

    if lane < 32u32 {
        scratch[lane] = 0u32;
    }
    barrier();

    let value = data[i as usize];
    let scan_result = block_scan_add_u32_kernel(value);
    out[i as usize] = scan_result;
}

/// Convenience kernel: i32 inclusive prefix-sum scan.
#[quanta::kernel(workgroup = [256])]
pub fn block_scan_add_i32_buffer(data: &[i32], out: &mut [i32]) {
    #[quanta::shared]
    let scratch: [i32; 32];

    let i = quark_id();
    let lane = proton_id();

    if lane < 32u32 {
        scratch[lane] = 0i32;
    }
    barrier();

    let value = data[i as usize];
    let scan_result = block_scan_add_i32_kernel(value);
    out[i as usize] = scan_result;
}

/// Convenience kernel: f32 inclusive prefix-sum scan.
#[quanta::kernel(workgroup = [256])]
pub fn block_scan_add_f32_buffer(data: &[f32], out: &mut [f32]) {
    #[quanta::shared]
    let scratch: [f32; 32];

    let i = quark_id();
    let lane = proton_id();

    if lane < 32u32 {
        scratch[lane] = 0.0f32;
    }
    barrier();

    let value = data[i as usize];
    let scan_result = block_scan_add_f32_kernel(value);
    out[i as usize] = scan_result;
}

// ── Block sort (bitonic) ────────────────────────────────────────
//
// Sorts a 256-element block of u32 keys per workgroup using
// **bitonic sort**, not radix. Bitonic was chosen for v0.1 over
// the more theoretically efficient radix LSD sort because:
//
//   - Bitonic's access pattern is data-independent: at each
//     stage every lane swaps with lane `self ^ k` for some `k`.
//     No prefix-sum dependency, no ping-pong device fn calls.
//   - Single shared-memory buffer + arithmetic, no auxiliary
//     scratch beyond `barrier` + `shared_load/store`. The kernel
//     macro's WASM-route lowerer handles this cleanly.
//   - For BLOCK = 256, bitonic runs 36 compare-exchange stages
//     (8 outer steps × up to 8 inner steps each); LSD radix runs
//     32 passes × 1 block_scan_add each. Comparable wall-clock
//     work; bitonic wins on simplicity.
//
// Radix-sort variants (multi-bit, segmented, key-value) will
// land in Tier 2 once the device-fn inliner handles nested
// control flow in the kernel lowerer. The function is named
// `block_radix_sort_*` for API forward-compatibility — callers
// don't see the algorithm choice.

/// Convenience kernel: sort each 256-element block of u32 keys
/// in ascending order. Workgroup size 256, one workgroup per
/// block. Caller dispatches with `quark_count = 256 * num_blocks`
/// and the same-sized output buffer.
///
/// **Block-local sort.** Each 256-element block is sorted
/// independently of the others. Producing a globally-sorted
/// output requires chaining a multi-block merge or device-wide
/// sort — out of scope for v0.1.
#[quanta::kernel(workgroup = [256])]
pub fn block_radix_sort_u32_buffer(data: &[u32], out: &mut [u32]) {
    #[quanta::shared]
    let buf: [u32; 256];

    let i = quark_id();
    let lane = proton_id();

    buf[lane] = data[i as usize];
    barrier();

    // Bitonic sort outer step `k` doubles each iteration: k = 2,
    // 4, 8, …, 256. For each `k` the inner step `j` halves from
    // `k/2` down to 1; each lane swaps with `lane ^ j` if the
    // direction-determined comparator says so.
    //
    // Direction: ascending if (lane & k) == 0, descending else.
    // Comparator: if direction is ascending and partner_key < my_key
    // (or both reversed for descending), swap.
    let mut k: u32 = 2u32;
    while k <= 256u32 {
        let mut j: u32 = k / 2u32;
        while j > 0u32 {
            let partner = lane ^ j;
            let my_key = buf[lane];
            let partner_key = buf[partner];

            // Ascending direction: keep smaller at lower index.
            let want_ascending = (lane & k) == 0u32;
            // Pairs cooperate: only the lower-index lane decides
            // and writes both slots — but that needs a barrier
            // between read and write. Simpler: both lanes do the
            // same read, decide whether to take their partner's
            // value, write back. The XOR partnering guarantees
            // each pair sees consistent comparisons because both
            // lanes use the same `(lane & k)` direction.
            let take_partner = if want_ascending {
                partner_key < my_key
            } else {
                partner_key > my_key
            };
            let new_key = if take_partner { partner_key } else { my_key };
            barrier();
            buf[lane] = new_key;
            barrier();

            j = j / 2u32;
        }
        k = k * 2u32;
    }

    out[i as usize] = buf[lane];
}
