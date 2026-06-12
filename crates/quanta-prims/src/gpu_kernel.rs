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
    fn atomic_add_shared_u32(slot: u32, index: u32, val: u32, order: u32) -> u32;
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
    pub fn atomic_add_shared_u32(_: u32, _: u32, _: u32, _: u32) -> u32 {
        0
    }
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
pub fn block_reduce_add_u32_kernel(value: u32) -> u32 {
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
pub fn block_reduce_add_i32_kernel(value: i32) -> i32 {
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
pub fn block_reduce_add_f32_kernel(value: f32) -> f32 {
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
pub fn block_reduce_min_u32_kernel(value: u32) -> u32 {
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
pub fn block_reduce_min_i32_kernel(value: i32) -> i32 {
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
pub fn block_reduce_min_f32_kernel(value: f32) -> f32 {
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
pub fn block_reduce_max_u32_kernel(value: u32) -> u32 {
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
pub fn block_reduce_max_i32_kernel(value: i32) -> i32 {
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
pub fn block_reduce_max_f32_kernel(value: f32) -> f32 {
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
pub fn block_scan_add_u32_kernel(value: u32) -> u32 {
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
pub fn block_scan_add_i32_kernel(value: i32) -> i32 {
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
pub fn block_scan_add_f32_kernel(value: f32) -> f32 {
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
/// independently of the others. For a globally-sorted output
/// use [`crate::device_sort_u32`], which chains
/// [`global_bitonic_pass_u32`] launches across blocks.
#[quanta::kernel(workgroup = [256])]
pub fn block_radix_sort_u32_buffer(data: &[u32], out: &mut [u32]) {
    #[quanta::shared]
    let buf: [u32; 256];

    let i = quark_id();
    let lane = proton_id();

    buf[lane] = data[i as usize];
    barrier();

    // Bitonic sort outer step `k` doubles each iteration:
    // k = 2, 4, 8, …, 256. For each `k` the inner step `j`
    // halves from `k/2` down to 1; each lane swaps with
    // `lane ^ j` if the direction-determined comparator says so.
    //
    // The lanes in a XOR-pair share the same `(lane & k)` bit
    // (since they differ only in bit `j < k`), so both lanes
    // agree on the pair's *direction*. They differ in `(lane &
    // j)`, which tells each lane whether it is the lower-indexed
    // partner. Combining: a lane should take the partner's value
    // exactly when its role (min-keeper vs max-keeper) AND the
    // partner's value disagree with what should be at this slot.
    //
    //   ascending = (lane & k) == 0      // direction
    //   i_am_lower = (lane & j) == 0     // role within pair
    //
    // For ascending direction:
    //   lower lane keeps min  -> takes partner if partner < me
    //   upper lane keeps max  -> takes partner if partner > me
    //
    // The "take partner" condition is therefore controlled by
    // `ascending == i_am_lower`: when both are true (ascending +
    // I'm the lower lane) or both are false (descending + I'm
    // the upper lane), the smaller-keeping rule applies. The
    // opposite case uses the larger-keeping rule.
    let mut k: u32 = 2u32;
    while k <= 256u32 {
        let mut j: u32 = k / 2u32;
        while j > 0u32 {
            let partner = lane ^ j;
            let my_key = buf[lane];
            let partner_key = buf[partner];

            // Compute `want_smaller` as a u32 bit expression
            // rather than a `bool == bool` to dodge an LLVM
            // constant-folding edge case observed on this path
            // (the bool-equality compiled to `r35 = true` which
            // killed the inner body).
            //
            // ascending_bit = ((lane >> log2(k)) & 1) == 0
            // lower_bit     = ((lane & j) == 0)
            // want_smaller  = ascending_bit == lower_bit
            //               = (lane & k) == 0   iff   (lane & j) == 0
            //               = ((lane & k) XOR (lane & j)) is "both same"
            //               = !( ((lane & k) != 0) XOR ((lane & j) != 0) )
            //
            // Equivalently, packing the bits: bit_k = (lane & k) != 0,
            // bit_j = (lane & j) != 0, want_smaller = (bit_k == bit_j).
            // Encode as integer compare to keep LLVM honest.
            let bit_k = if (lane & k) == 0u32 { 0u32 } else { 1u32 };
            let bit_j = if (lane & j) == 0u32 { 0u32 } else { 1u32 };
            let take_partner = if bit_k == bit_j {
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

// ── Tier 2 — Block compact ───────────────────────────────────────
//
// Per-block stream compaction: each 256-element block selects the
// lanes whose predicate is non-zero and writes their data values
// contiguously to the start of the block's output region. The
// per-block kept count goes to `counts[block]`.
//
// Algorithm:
//   1. exclusive scan of predicates → per-lane output offset
//   2. if predicate set, write data[i] to out[block_start + offset]
//   3. lane (BLOCK-1) writes the inclusive prefix sum (= kept
//      count for the block) to counts[block]
//
// Exclusive-scan = inclusive_scan - own_value. We reuse
// `block_scan_add_u32_kernel` (inclusive) and subtract the lane's
// own predicate.

/// Convenience kernel: per-block stream compaction with explicit
/// `predicates` array (non-zero = keep). Output buffer must be at
/// least `data.len()`; `counts.len()` must equal `data.len() / 256`.
/// Within each 256-element block, kept entries are written
/// contiguously starting at `block * 256`.
#[quanta::kernel(workgroup = [256])]
pub fn block_compact_u32_buffer(
    predicates: &[u32],
    data: &[u32],
    out: &mut [u32],
    counts: &mut [u32],
) {
    #[quanta::shared]
    let scratch: [u32; 32];

    let i = quark_id();
    let lane = proton_id();
    let block = nucleus_id();

    if lane < 32u32 {
        scratch[lane] = 0u32;
    }
    barrier();

    let pred = predicates[i as usize];
    let inclusive = block_scan_add_u32_kernel(pred);
    let exclusive = inclusive - pred;
    let block_start = block * 256u32;

    if pred != 0u32 {
        out[(block_start + exclusive) as usize] = data[i as usize];
    }

    if lane == 255u32 {
        counts[block as usize] = inclusive;
    }
}

// ── Tier 2 — Block histogram ─────────────────────────────────────
//
// Per-block bucket histogram via shared-memory atomic increment.
// Fixed at 256 buckets (= workgroup_size). The caller pre-computes
// bucket indices (each value in `buckets_in` is the lane's bucket
// in 0..256). The output has one count per (block, bucket) and is
// stored block-major: out[block * 256 + bucket].
//
// Algorithm:
//   1. Every lane zero-inits its own slot of `local_counts`.
//   2. Read `buckets_in[i]`, atomically increment
//      `local_counts[bucket]` via shared-mem atomic_add.
//   3. Every lane copies one bucket count to global output.
//
// Shared-memory atomics emit on Metal today (substrate gap 3 fix
// from 2026-05-18). WGSL / SPIR-V / LLVM paths return NotSupported.

/// Convenience kernel: per-block histogram with 256 buckets. The
/// caller pre-computes bucket indices (each value in `buckets_in`
/// must be in 0..256). Output: one count per (block, bucket),
/// block-major. `counts_out.len()` must equal
/// `(buckets_in.len() / 256) * 256` = `buckets_in.len()`.
#[quanta::kernel(workgroup = [256])]
pub fn block_histogram_u32_buffer(buckets_in: &[u32], counts_out: &mut [u32]) {
    #[quanta::shared]
    let local_counts: [u32; 256];

    let i = quark_id();
    let lane = proton_id();
    let block = nucleus_id();

    local_counts[lane] = 0u32;
    barrier();

    let bucket = buckets_in[i as usize];
    unsafe {
        atomic_add_shared_u32(0u32, bucket, 1u32, 0u32);
    }
    barrier();

    counts_out[(block * 256u32 + lane) as usize] = local_counts[lane];
}

// ── Tier 2 — Block top-k ─────────────────────────────────────────
//
// Per-block selection of the K largest u32 values. Built on the
// bitonic sort body inlined here — sorting the block ascending
// gives the K largest at indices BLOCK-K..BLOCK-1. Lanes 0..K
// emit them in descending order to the per-block output region.
//
// Workgroup size is fixed at 256 (same as the underlying sort).
// K is a runtime push-constant; the caller must ensure `K <= 256`.
// Output layout: top_k_out[block * K + i] = i-th-largest value
// from block, with i=0 the largest.

/// Convenience kernel: per-block top-K selection. Sorts each
/// 256-element block of u32 keys ascending (bitonic) and emits
/// the K largest in descending order to the per-block output
/// region (`top_k_out[block*k + i]`). `k <= 256`.
///
/// Each workgroup processes one block. Caller dispatches with
/// `quark_count = 256 * num_blocks` and `top_k_out.len() >=
/// num_blocks * k`. When `k = 256` this is just a per-block
/// descending sort (same work as `block_radix_sort_u32_buffer`
/// with the order inverted).
#[quanta::kernel(workgroup = [256])]
pub fn block_top_k_u32_buffer(data: &[u32], top_k_out: &mut [u32], k: u32) {
    #[quanta::shared]
    let buf: [u32; 256];

    let i = quark_id();
    let lane = proton_id();
    let block = nucleus_id();

    buf[lane] = data[i as usize];
    barrier();

    // Bitonic sort body — identical to block_radix_sort_u32_buffer.
    // See its comments for the want_smaller derivation. Inlined
    // because factoring it into a device fn would force the inliner
    // to re-thread the nested-while-loop body and that path isn't
    // exercised by anything else today.
    let mut outer: u32 = 2u32;
    while outer <= 256u32 {
        let mut inner: u32 = outer / 2u32;
        while inner > 0u32 {
            let partner = lane ^ inner;
            let my_key = buf[lane];
            let partner_key = buf[partner];

            let bit_k = if (lane & outer) == 0u32 { 0u32 } else { 1u32 };
            let bit_j = if (lane & inner) == 0u32 { 0u32 } else { 1u32 };
            let take_partner = if bit_k == bit_j {
                partner_key < my_key
            } else {
                partner_key > my_key
            };
            let new_key = if take_partner { partner_key } else { my_key };
            barrier();
            buf[lane] = new_key;
            barrier();

            inner = inner / 2u32;
        }
        outer = outer * 2u32;
    }

    // Sorted ascending: buf[0] is smallest, buf[255] is largest.
    // Lane `i` writes the (i+1)-th largest, which lives at buf[255-i].
    if lane < k {
        top_k_out[(block * k + lane) as usize] = buf[(255u32 - lane) as usize];
    }
}

// ── Tier 3 — Device-wide bitonic pass ────────────────────────────
//
// One compare-exchange pass of a *device-wide* bitonic sorting
// network. The block sort above is tile-local (each workgroup
// sorts its own 256 keys through shared memory); sorting a whole
// buffer needs compare-exchange at strides that cross workgroup
// boundaries, which means one kernel launch per (k, j) pass with
// a device-memory barrier (the dispatch boundary) in between.
//
// The host driver lives in `device_wide::device_sort_u32`: it
// pads the buffer to a power of two with u32::MAX and loops
// k = 2, 4, …, n; j = k/2, …, 1 — log²(n) launches total.

/// One global bitonic compare-exchange pass: every element pairs
/// with `index ^ j` and the pair is ordered according to the
/// direction bit `(index & k)`. The lower-indexed thread of each
/// pair performs the swap; the upper-indexed thread does nothing.
///
/// Building block for [`crate::device_sort_u32`] — callers
/// dispatch with `quark_count = data.len()`, which must be a
/// power of two and a multiple of the workgroup size.
#[quanta::kernel(workgroup = [256])]
pub fn global_bitonic_pass_u32(data: &mut [u32], k: u32, j: u32) {
    let i = quark_id();
    let partner = i ^ j;
    // Only the lower-indexed thread of each XOR-pair acts, and it
    // writes both slots — pairs are disjoint, so no two threads
    // ever write the same element in one pass.
    if partner > i {
        let a = data[i as usize];
        let b = data[partner as usize];
        // Direction for this pair: ascending when (i & k) == 0.
        // Same integer-compare encoding as the block sort above
        // (see its comment block for the bool-equality LLVM edge
        // case this dodges).
        let bit_k = if (i & k) == 0u32 { 0u32 } else { 1u32 };
        let out_of_order = if bit_k == 0u32 { b < a } else { b > a };
        if out_of_order {
            data[i as usize] = b;
            data[partner as usize] = a;
        }
    }
}
