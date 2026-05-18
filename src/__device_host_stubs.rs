//! Host-side stubs for every GPU intrinsic.
//!
//! These exist to make `#[quanta::device]` function bodies parse and
//! name-resolve on the **host** (non-wasm32) target as well as on
//! wasm32. The `_src!()` macro a device function generates re-injects
//! the function body into downstream crates inside a
//! `const _: () = { ... }` block; that block must compile end-to-end,
//! and the body calls intrinsic names by bare identifier. Without
//! these stubs the downstream compile fails with `E0425: cannot find
//! function reduce_add_u32 in this scope`, etc.
//!
//! On wasm32 (the actual GPU compilation path) the matching
//! `extern "C"` declarations in [`crate::intrinsics`] take over —
//! they appear as `import "quanta" "<name>"` in the WASM module that
//! the lowering pass walks. These host stubs are never called on the
//! GPU side.
//!
//! On the host, these are degenerate single-thread fallbacks:
//! reduce / scan return the input unchanged, shared-memory load
//! returns zero, store / barrier / fence are no-ops, atomics return
//! zero without touching the address, math intrinsics delegate to
//! `libm`-style methods. They are NOT a CPU implementation of the
//! kernel programming model — they are placeholders so the host-side
//! Rust toolchain can typecheck the body. Callers that actually want
//! to run a kernel on CPU go through `init_cpu()`, which dispatches
//! the wasm-lowering pipeline.

#![allow(unused, dead_code, clippy::too_many_arguments)]

// ── Memory-order discriminants ────────────────────────────────────────
//
// Match `crate::intrinsics::ORDER_*` so device-fn bodies that name
// these constants compile on host as well. Placed first so the
// public-API alias module below can reference them.

pub const ORDER_RELAXED: u32 = 0;
pub const ORDER_ACQUIRE: u32 = 1;
pub const ORDER_RELEASE: u32 = 2;
pub const ORDER_ACQ_REL: u32 = 3;
pub const ORDER_SEQ_CST: u32 = 4;

// ── Identity ──────────────────────────────────────────────────────────

pub fn quark_id() -> u32 {
    0
}
pub fn local_id() -> u32 {
    0
}
pub fn group_id() -> u32 {
    0
}
pub fn workgroup_size() -> u32 {
    1
}

// Quanta-API aliases. The wasm shell exposes `proton_id`,
// `nucleus_id`, `proton_size` as inline wrappers around
// `local_id`/`group_id`/`workgroup_size`; mirror them on host so
// device fn bodies written against the public-API names compile.

pub fn proton_id() -> u32 {
    0
}
pub fn nucleus_id() -> u32 {
    0
}
pub fn proton_size() -> u32 {
    1
}

// Public-API math aliases (suffix-free), mirroring the wasm-shell
// `quanta::intrinsics::*` wrappers. Real implementations so host-side
// reference uses produce useful values.

pub fn sqrt(x: f32) -> f32 {
    x.sqrt()
}
pub fn rsqrt(x: f32) -> f32 {
    1.0 / x.sqrt()
}
pub fn sin(x: f32) -> f32 {
    x.sin()
}
pub fn cos(x: f32) -> f32 {
    x.cos()
}
pub fn tan(x: f32) -> f32 {
    x.tan()
}
pub fn exp(x: f32) -> f32 {
    x.exp()
}
pub fn ln(x: f32) -> f32 {
    x.ln()
}
pub fn fabs(x: f32) -> f32 {
    x.abs()
}
pub fn floor(x: f32) -> f32 {
    x.floor()
}
pub fn ceil(x: f32) -> f32 {
    x.ceil()
}
pub fn round(x: f32) -> f32 {
    x.round()
}
pub fn fmin(a: f32, b: f32) -> f32 {
    a.min(b)
}
pub fn fmax(a: f32, b: f32) -> f32 {
    a.max(b)
}
pub fn powf(b: f32, e: f32) -> f32 {
    b.powf(e)
}
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}
pub fn clamp_f(x: f32, lo: f32, hi: f32) -> f32 {
    x.clamp(lo, hi)
}

// Atomic-API wrappers — `atomic_add(&mut buf[i], v)` etc. Single-
// thread host stubs that just perform the operation in place.
// They never enter shared state, so atomic semantics don't matter
// here; the GPU path emits real atomics.

/// # Safety
/// Host stub: dereferences `addr` exactly once with no concurrency.
pub unsafe fn atomic_add(addr: &mut u32, val: u32) -> u32 {
    let old = *addr;
    *addr = old.wrapping_add(val);
    old
}
/// # Safety
/// Host stub: see `atomic_add`.
pub unsafe fn atomic_sub(addr: &mut u32, val: u32) -> u32 {
    let old = *addr;
    *addr = old.wrapping_sub(val);
    old
}
/// # Safety
/// Host stub: see `atomic_add`.
pub unsafe fn atomic_min(addr: &mut u32, val: u32) -> u32 {
    let old = *addr;
    *addr = old.min(val);
    old
}
/// # Safety
/// Host stub: see `atomic_add`.
pub unsafe fn atomic_max(addr: &mut u32, val: u32) -> u32 {
    let old = *addr;
    *addr = old.max(val);
    old
}
/// # Safety
/// Host stub: see `atomic_add`.
pub unsafe fn atomic_and(addr: &mut u32, val: u32) -> u32 {
    let old = *addr;
    *addr = old & val;
    old
}
/// # Safety
/// Host stub: see `atomic_add`.
pub unsafe fn atomic_or(addr: &mut u32, val: u32) -> u32 {
    let old = *addr;
    *addr = old | val;
    old
}
/// # Safety
/// Host stub: see `atomic_add`.
pub unsafe fn atomic_xor(addr: &mut u32, val: u32) -> u32 {
    let old = *addr;
    *addr = old ^ val;
    old
}
/// # Safety
/// Host stub: see `atomic_add`.
pub unsafe fn atomic_exchange(addr: &mut u32, val: u32) -> u32 {
    let old = *addr;
    *addr = val;
    old
}
/// # Safety
/// Host stub: see `atomic_add`.
pub unsafe fn fence(_order: u32) {}

// MemoryOrder enum-style identifier constants. Kernels write
// `fence(Release)` with bare names; expose them as `pub const`
// aliases of the integer-tagged constants.
#[allow(non_upper_case_globals)]
pub const Relaxed: u32 = ORDER_RELAXED;
#[allow(non_upper_case_globals)]
pub const Acquire: u32 = ORDER_ACQUIRE;
#[allow(non_upper_case_globals)]
pub const Release: u32 = ORDER_RELEASE;
#[allow(non_upper_case_globals)]
pub const AcqRel: u32 = ORDER_ACQ_REL;
#[allow(non_upper_case_globals)]
pub const SeqCst: u32 = ORDER_SEQ_CST;

// ── Synchronization ───────────────────────────────────────────────────

pub fn barrier() {}
pub fn memory_fence(_order: u32) {}

// ── Atomics ───────────────────────────────────────────────────────────

/// # Safety
/// Host stub never dereferences `addr`. Provided for name-resolution
/// only — the GPU path emits real atomic ops.
pub unsafe fn atomic_add_u32(_addr: *mut u32, _val: u32, _order: u32) -> u32 {
    0
}
/// # Safety
/// Host stub: see `atomic_add_u32`.
pub unsafe fn atomic_sub_u32(_addr: *mut u32, _val: u32, _order: u32) -> u32 {
    0
}
/// # Safety
/// Host stub: see `atomic_add_u32`.
pub unsafe fn atomic_min_u32(_addr: *mut u32, _val: u32, _order: u32) -> u32 {
    0
}
/// # Safety
/// Host stub: see `atomic_add_u32`.
pub unsafe fn atomic_max_u32(_addr: *mut u32, _val: u32, _order: u32) -> u32 {
    0
}
/// # Safety
/// Host stub: see `atomic_add_u32`.
pub unsafe fn atomic_and_u32(_addr: *mut u32, _val: u32, _order: u32) -> u32 {
    0
}
/// # Safety
/// Host stub: see `atomic_add_u32`.
pub unsafe fn atomic_or_u32(_addr: *mut u32, _val: u32, _order: u32) -> u32 {
    0
}
/// # Safety
/// Host stub: see `atomic_add_u32`.
pub unsafe fn atomic_xor_u32(_addr: *mut u32, _val: u32, _order: u32) -> u32 {
    0
}
/// # Safety
/// Host stub: see `atomic_add_u32`.
pub unsafe fn atomic_exchange_u32(_addr: *mut u32, _val: u32, _order: u32) -> u32 {
    0
}
/// # Safety
/// Host stub: see `atomic_add_u32`.
pub unsafe fn atomic_cas_u32(
    _addr: *mut u32,
    _expected: u32,
    _desired: u32,
    _success_order: u32,
    _failure_order: u32,
) -> u32 {
    0
}
/// # Safety
/// Host stub: see `atomic_add_u32`.
pub unsafe fn atomic_add_i32(_addr: *mut i32, _val: i32, _order: u32) -> i32 {
    0
}
/// # Safety
/// Host stub: see `atomic_add_u32`.
pub unsafe fn atomic_sub_i32(_addr: *mut i32, _val: i32, _order: u32) -> i32 {
    0
}

// ── Math (f32) ────────────────────────────────────────────────────────
//
// Real implementations so device-fn bodies that compute on host
// (tests, doctests, host-side reference) produce sensible values.

pub fn sqrt_f32(x: f32) -> f32 {
    x.sqrt()
}
pub fn rsqrt_f32(x: f32) -> f32 {
    1.0 / x.sqrt()
}
pub fn sin_f32(x: f32) -> f32 {
    x.sin()
}
pub fn cos_f32(x: f32) -> f32 {
    x.cos()
}
pub fn tan_f32(x: f32) -> f32 {
    x.tan()
}
pub fn exp_f32(x: f32) -> f32 {
    x.exp()
}
pub fn log_f32(x: f32) -> f32 {
    x.ln()
}
pub fn pow_f32(base: f32, exp: f32) -> f32 {
    base.powf(exp)
}
pub fn abs_f32(x: f32) -> f32 {
    x.abs()
}
pub fn floor_f32(x: f32) -> f32 {
    x.floor()
}
pub fn ceil_f32(x: f32) -> f32 {
    x.ceil()
}
pub fn round_f32(x: f32) -> f32 {
    x.round()
}
pub fn min_f32(a: f32, b: f32) -> f32 {
    a.min(b)
}
pub fn max_f32(a: f32, b: f32) -> f32 {
    a.max(b)
}
pub fn clamp_f32(x: f32, lo: f32, hi: f32) -> f32 {
    x.clamp(lo, hi)
}
pub fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

// ── Math (f64) ────────────────────────────────────────────────────────

pub fn sqrt_f64(x: f64) -> f64 {
    x.sqrt()
}
pub fn rsqrt_f64(x: f64) -> f64 {
    1.0 / x.sqrt()
}
pub fn sin_f64(x: f64) -> f64 {
    x.sin()
}
pub fn cos_f64(x: f64) -> f64 {
    x.cos()
}
pub fn tan_f64(x: f64) -> f64 {
    x.tan()
}
pub fn exp_f64(x: f64) -> f64 {
    x.exp()
}
pub fn log_f64(x: f64) -> f64 {
    x.ln()
}
pub fn pow_f64(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}
pub fn abs_f64(x: f64) -> f64 {
    x.abs()
}
pub fn floor_f64(x: f64) -> f64 {
    x.floor()
}
pub fn ceil_f64(x: f64) -> f64 {
    x.ceil()
}
pub fn round_f64(x: f64) -> f64 {
    x.round()
}
pub fn min_f64(a: f64, b: f64) -> f64 {
    a.min(b)
}
pub fn max_f64(a: f64, b: f64) -> f64 {
    a.max(b)
}
pub fn clamp_f64(x: f64, lo: f64, hi: f64) -> f64 {
    x.clamp(lo, hi)
}
pub fn fma_f64(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

// ── Subgroup / wave (single-lane) ─────────────────────────────────────
//
// On a single host thread the subgroup is a degenerate 1-lane warp.
// Reduce and scan return the input; ballot returns the predicate bit
// in position 0; any/all return 1 iff the predicate is true.

pub fn subgroup_size() -> u32 {
    1
}
pub fn subgroup_id() -> u32 {
    0
}
pub fn ballot_u32(predicate: u32) -> u32 {
    if predicate != 0 { 1 } else { 0 }
}
pub fn any_u32(predicate: u32) -> u32 {
    if predicate != 0 { 1 } else { 0 }
}
pub fn all_u32(predicate: u32) -> u32 {
    if predicate != 0 { 1 } else { 0 }
}
pub fn shuffle_u32(value: u32, _lane_delta: u32) -> u32 {
    value
}
pub fn shuffle_i32(value: i32, _lane_delta: u32) -> i32 {
    value
}
pub fn shuffle_f32(value: f32, _lane_delta: u32) -> f32 {
    value
}

pub fn reduce_add_u32(value: u32) -> u32 {
    value
}
pub fn reduce_add_i32(value: i32) -> i32 {
    value
}
pub fn reduce_add_f32(value: f32) -> f32 {
    value
}
pub fn reduce_min_u32(value: u32) -> u32 {
    value
}
pub fn reduce_min_i32(value: i32) -> i32 {
    value
}
pub fn reduce_min_f32(value: f32) -> f32 {
    value
}
pub fn reduce_max_u32(value: u32) -> u32 {
    value
}
pub fn reduce_max_i32(value: i32) -> i32 {
    value
}
pub fn reduce_max_f32(value: f32) -> f32 {
    value
}

pub fn scan_add_u32(value: u32) -> u32 {
    value
}
pub fn scan_add_i32(value: i32) -> i32 {
    value
}
pub fn scan_add_f32(value: f32) -> f32 {
    value
}
pub fn scan_add_exclusive_u32(_value: u32) -> u32 {
    0
}
pub fn scan_add_exclusive_i32(_value: i32) -> i32 {
    0
}
pub fn scan_add_exclusive_f32(_value: f32) -> f32 {
    0.0
}

// ── Workgroup-shared memory (no-op) ───────────────────────────────────

pub fn shared_load_f32(_slot: u32, _index: u32) -> f32 {
    0.0
}
pub fn shared_load_u32(_slot: u32, _index: u32) -> u32 {
    0
}
pub fn shared_load_i32(_slot: u32, _index: u32) -> i32 {
    0
}
pub fn shared_store_f32(_slot: u32, _index: u32, _val: f32) {}
pub fn shared_store_u32(_slot: u32, _index: u32, _val: u32) {}
pub fn shared_store_i32(_slot: u32, _index: u32, _val: i32) {}

// Shared-memory atomic stubs. Single-thread fallbacks: always
// return zero (no real concurrent state to read), no-op on the
// store side.
pub fn atomic_add_shared_u32(_slot: u32, _index: u32, _val: u32, _order: u32) -> u32 {
    0
}
pub fn atomic_sub_shared_u32(_slot: u32, _index: u32, _val: u32, _order: u32) -> u32 {
    0
}
pub fn atomic_min_shared_u32(_slot: u32, _index: u32, _val: u32, _order: u32) -> u32 {
    0
}
pub fn atomic_max_shared_u32(_slot: u32, _index: u32, _val: u32, _order: u32) -> u32 {
    0
}
pub fn atomic_and_shared_u32(_slot: u32, _index: u32, _val: u32, _order: u32) -> u32 {
    0
}
pub fn atomic_or_shared_u32(_slot: u32, _index: u32, _val: u32, _order: u32) -> u32 {
    0
}
pub fn atomic_xor_shared_u32(_slot: u32, _index: u32, _val: u32, _order: u32) -> u32 {
    0
}
pub fn atomic_exchange_shared_u32(_slot: u32, _index: u32, _val: u32, _order: u32) -> u32 {
    0
}
pub fn atomic_add_shared_i32(_slot: u32, _index: u32, _val: i32, _order: u32) -> i32 {
    0
}
pub fn atomic_sub_shared_i32(_slot: u32, _index: u32, _val: i32, _order: u32) -> i32 {
    0
}

// ── Textures (no-op) ──────────────────────────────────────────────────

pub fn texture_sample_2d_f32(_slot: u32, _x: u32, _y: u32) -> f32 {
    0.0
}
pub fn texture_load_2d_f32(_slot: u32, _x: u32, _y: u32) -> f32 {
    0.0
}
pub fn texture_load_3d_f32(_slot: u32, _x: u32, _y: u32, _z: u32) -> f32 {
    0.0
}
pub fn texture_write_2d_f32(_slot: u32, _x: u32, _y: u32, _val: f32) {}
