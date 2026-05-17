//! GPU intrinsics declared as `extern "C"` imports.
//!
//! When `#[quanta::kernel]` emits its WASM-compilable twin
//! (the `extern "C" fn` rustc lowers to wasm32 and the lowering pass
//! consumes), it injects `use quanta::intrinsics::*` so kernel bodies
//! can call these functions naturally. Each appears in the output
//! WASM module as `import "quanta" "<name>"`. The lowering pass
//! resolves them to `KernelOp::Intrinsic` nodes; the existing
//! per-backend emitters lower those to PTX / GCN / MSL / SPIR-V /
//! WGSL equivalents.
//!
//! On wasm32 the imports are stubs the host (Quanta's lowering pass)
//! never actually calls — they exist only so rustc's typechecker is
//! happy and so the symbols appear in the WASM module's import
//! section, which the lowering pass then walks.
//!
//! On native targets these functions are unused (kernel code only
//! runs on GPU, never on host). The module is `cfg`-gated to
//! `wasm32` so we don't accidentally make them callable from host
//! Rust.

#![cfg(target_arch = "wasm32")]
#![allow(unused, dead_code)]

// ── Identity ───────────────────────────────────────────────────────────

#[link(wasm_import_module = "quanta")]
unsafe extern "C" {
    /// Global thread index. `0..total_quarks_dispatched`.
    pub fn quark_id() -> u32;

    /// Thread index within the workgroup. `0..workgroup_size`.
    pub fn local_id() -> u32;

    /// Workgroup index within the dispatch grid.
    pub fn group_id() -> u32;

    /// Configured workgroup size (set by the `#[quanta::kernel(workgroup = ...)]` attribute).
    pub fn workgroup_size() -> u32;
}

// ── Synchronization ────────────────────────────────────────────────────

#[link(wasm_import_module = "quanta")]
unsafe extern "C" {
    /// Workgroup-scope barrier. All quarks in the workgroup wait until
    /// every quark has reached this point.
    pub fn barrier();

    /// Memory fence with the given ordering. `order` matches
    /// `quanta::MemoryOrder` discriminants:
    /// 0 = Relaxed, 1 = Acquire, 2 = Release, 3 = AcqRel, 4 = SeqCst.
    pub fn memory_fence(order: u32);
}

// ── Atomics ────────────────────────────────────────────────────────────

#[link(wasm_import_module = "quanta")]
unsafe extern "C" {
    pub fn atomic_add_u32(addr: *mut u32, val: u32, order: u32) -> u32;
    pub fn atomic_sub_u32(addr: *mut u32, val: u32, order: u32) -> u32;
    pub fn atomic_min_u32(addr: *mut u32, val: u32, order: u32) -> u32;
    pub fn atomic_max_u32(addr: *mut u32, val: u32, order: u32) -> u32;
    pub fn atomic_and_u32(addr: *mut u32, val: u32, order: u32) -> u32;
    pub fn atomic_or_u32(addr: *mut u32, val: u32, order: u32) -> u32;
    pub fn atomic_xor_u32(addr: *mut u32, val: u32, order: u32) -> u32;
    pub fn atomic_exchange_u32(addr: *mut u32, val: u32, order: u32) -> u32;

    /// Compare-and-swap. Returns the value found at `*addr` before
    /// the operation. Updates `*addr` to `desired` only if the
    /// previous value equalled `expected`.
    pub fn atomic_cas_u32(
        addr: *mut u32,
        expected: u32,
        desired: u32,
        success_order: u32,
        failure_order: u32,
    ) -> u32;

    pub fn atomic_add_i32(addr: *mut i32, val: i32, order: u32) -> i32;
    pub fn atomic_sub_i32(addr: *mut i32, val: i32, order: u32) -> i32;
}

// ── Math ───────────────────────────────────────────────────────────────

#[link(wasm_import_module = "quanta")]
unsafe extern "C" {
    pub fn sqrt_f32(x: f32) -> f32;
    pub fn rsqrt_f32(x: f32) -> f32;
    pub fn sin_f32(x: f32) -> f32;
    pub fn cos_f32(x: f32) -> f32;
    pub fn tan_f32(x: f32) -> f32;
    pub fn exp_f32(x: f32) -> f32;
    pub fn log_f32(x: f32) -> f32;
    pub fn pow_f32(base: f32, exp: f32) -> f32;
    pub fn abs_f32(x: f32) -> f32;
    pub fn floor_f32(x: f32) -> f32;
    pub fn ceil_f32(x: f32) -> f32;
    pub fn round_f32(x: f32) -> f32;
    pub fn min_f32(a: f32, b: f32) -> f32;
    pub fn max_f32(a: f32, b: f32) -> f32;
    pub fn clamp_f32(x: f32, lo: f32, hi: f32) -> f32;
    pub fn fma_f32(a: f32, b: f32, c: f32) -> f32;
}

// ── Math intrinsics (f64) ──────────────────────────────────────────────
//
// f64 variants of the math intrinsics above. Used by f64-precision
// distribution kernels (`fill_normal_f64`, `fill_exponential_f64`,
// `fill_lognormal_f64`) and by any user kernel that needs double-
// precision math.
//
// Per-backend support:
//   - LLVM / CPU JIT: `llvm.sqrt.f64`, `llvm.sin.f64`, ... all native.
//   - Metal MSL: `metal::sqrt(double)` etc. — requires Metal 2.4+.
//   - SPIR-V / Vulkan: OpExtInst with f64 result type; needs the
//     `Float64` capability declared on the module.
//   - WGSL / WebGPU: `f64` is not a WGSL primitive type. f64 math
//     in a kernel returns `NotSupported` from the WGSL emitter.

#[link(wasm_import_module = "quanta")]
unsafe extern "C" {
    pub fn sqrt_f64(x: f64) -> f64;
    pub fn rsqrt_f64(x: f64) -> f64;
    pub fn sin_f64(x: f64) -> f64;
    pub fn cos_f64(x: f64) -> f64;
    pub fn tan_f64(x: f64) -> f64;
    pub fn exp_f64(x: f64) -> f64;
    pub fn log_f64(x: f64) -> f64;
    pub fn pow_f64(base: f64, exp: f64) -> f64;
    pub fn abs_f64(x: f64) -> f64;
    pub fn floor_f64(x: f64) -> f64;
    pub fn ceil_f64(x: f64) -> f64;
    pub fn round_f64(x: f64) -> f64;
    pub fn min_f64(a: f64, b: f64) -> f64;
    pub fn max_f64(a: f64, b: f64) -> f64;
    pub fn clamp_f64(x: f64, lo: f64, hi: f64) -> f64;
    pub fn fma_f64(a: f64, b: f64, c: f64) -> f64;
}

// ── Subgroup / wave ────────────────────────────────────────────────────
//
// Type coverage today: u32, i32, f32 (the "portable Tier-1" set
// — every Quanta backend supports subgroup operations on these
// natively). u64 / i64 subgroup ops are intentionally absent
// because Metal's simdgroup instructions don't include 64-bit
// arithmetic and WGSL's `subgroupAdd` family is defined only for
// 32-bit and 16-bit element types. Downstream code that needs
// 64-bit reductions can fall back to a shared-memory-only tree
// reduce (slower but works everywhere) — see
// `quanta-prims::block_reduce_u64` (planned) for a worked
// example.

#[link(wasm_import_module = "quanta")]
unsafe extern "C" {
    pub fn subgroup_size() -> u32;
    pub fn subgroup_id() -> u32;

    // Ballot / any / all take a predicate (any non-zero u32 == true);
    // a single u32 variant suffices regardless of the value type
    // being voted on.
    pub fn ballot_u32(predicate: u32) -> u32;
    pub fn any_u32(predicate: u32) -> u32;
    pub fn all_u32(predicate: u32) -> u32;

    // Shuffle: read `value` from lane `self_lane ^ lane_delta`.
    // The second argument is an XOR mask, not a source-lane
    // index — `lane_delta = 1` swaps adjacent pairs, `2` swaps
    // pairs of pairs, etc. (the standard butterfly pattern used
    // by tree reductions). Mirrors Metal's `simd_shuffle_xor`
    // and WGSL's `subgroupShuffleXor`.
    pub fn shuffle_u32(value: u32, lane_delta: u32) -> u32;
    pub fn shuffle_i32(value: i32, lane_delta: u32) -> i32;
    pub fn shuffle_f32(value: f32, lane_delta: u32) -> f32;

    // Reduce: every lane gets the warp-wide reduction.
    pub fn reduce_add_u32(value: u32) -> u32;
    pub fn reduce_add_i32(value: i32) -> i32;
    pub fn reduce_add_f32(value: f32) -> f32;
    pub fn reduce_min_u32(value: u32) -> u32;
    pub fn reduce_min_i32(value: i32) -> i32;
    pub fn reduce_min_f32(value: f32) -> f32;
    pub fn reduce_max_u32(value: u32) -> u32;
    pub fn reduce_max_i32(value: i32) -> i32;
    pub fn reduce_max_f32(value: f32) -> f32;

    // Inclusive prefix scan: every lane gets the running sum of
    // lanes 0..=self.
    pub fn scan_add_u32(value: u32) -> u32;
    pub fn scan_add_i32(value: i32) -> i32;
    pub fn scan_add_f32(value: f32) -> f32;

    // Exclusive prefix scan: every lane gets the running sum of
    // lanes 0..self (lane 0 receives 0). Pairs with the
    // inclusive form: `inclusive[k] = exclusive[k] + value[k]`.
    pub fn scan_add_exclusive_u32(value: u32) -> u32;
    pub fn scan_add_exclusive_i32(value: i32) -> i32;
    pub fn scan_add_exclusive_f32(value: f32) -> f32;
}

// ── Workgroup-shared memory ────────────────────────────────────────────

#[link(wasm_import_module = "quanta")]
unsafe extern "C" {
    /// Load from workgroup-shared memory at `(slot, index)`.
    pub fn shared_load_f32(slot: u32, index: u32) -> f32;
    pub fn shared_load_u32(slot: u32, index: u32) -> u32;
    pub fn shared_load_i32(slot: u32, index: u32) -> i32;

    /// Store to workgroup-shared memory at `(slot, index)`.
    pub fn shared_store_f32(slot: u32, index: u32, val: f32);
    pub fn shared_store_u32(slot: u32, index: u32, val: u32);
    pub fn shared_store_i32(slot: u32, index: u32, val: i32);
}

// ── Textures ───────────────────────────────────────────────────────────

#[link(wasm_import_module = "quanta")]
unsafe extern "C" {
    /// Sampled texture read. Slot is bound to a Texture2D<T> at
    /// dispatch time via `wave.bind_texture(slot, tex)`.
    pub fn texture_sample_2d_f32(slot: u32, x: u32, y: u32) -> f32;

    /// Unsampled (raw integer-coord) texture read.
    pub fn texture_load_2d_f32(slot: u32, x: u32, y: u32) -> f32;
    pub fn texture_load_3d_f32(slot: u32, x: u32, y: u32, z: u32) -> f32;

    /// Texture write (storage texture). Slot must be bound to a
    /// `Texture2D<T>` declared as writable.
    pub fn texture_write_2d_f32(slot: u32, x: u32, y: u32, val: f32);
}

// ── Memory-order discriminants ─────────────────────────────────────────

/// Memory ordering values used by `*_order` parameters above.
/// Mirrors `quanta::MemoryOrder` so kernel code can name them
/// symbolically:
///
/// ```ignore
/// atomic_add_u32(p, 1, ORDER_RELAXED);
/// ```
pub const ORDER_RELAXED: u32 = 0;
pub const ORDER_ACQUIRE: u32 = 1;
pub const ORDER_RELEASE: u32 = 2;
pub const ORDER_ACQ_REL: u32 = 3;
pub const ORDER_SEQ_CST: u32 = 4;
