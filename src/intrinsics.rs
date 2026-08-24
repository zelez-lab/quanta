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

    /// Record `value` (with the calling quark's id) into the kernel's
    /// debug buffer; the host prints `[quanta gpu_print] quark=… = …`
    /// to stderr after the dispatch completes. JIT-only — a debug
    /// tool, not an output path.
    pub fn gpu_print_u32(value: u32);
    /// Signed twin of `gpu_print_u32`.
    pub fn gpu_print_i32(value: i32);
    /// Float twin of `gpu_print_u32`.
    pub fn gpu_print_f32(value: f32);

    /// Memory fence with the given ordering. `order` matches
    /// `quanta::MemoryOrder` discriminants:
    /// 0 = Relaxed, 1 = Acquire, 2 = Release, 3 = AcqRel, 4 = SeqCst.
    pub fn memory_fence(order: u32);
}

// ── Atomics ────────────────────────────────────────────────────────────

#[link(wasm_import_module = "quanta")]
unsafe extern "C" {
    /// Atomically add `val` into `*addr`, returning the value found there
    /// before the operation. Every atomic here returns that previous value,
    /// and every `order` is a `MemoryOrder` discriminant — see the `ORDER_*`
    /// constants at the bottom of this module.
    pub fn atomic_add_u32(addr: *mut u32, val: u32, order: u32) -> u32;
    /// Atomically subtract `val` from `*addr`, returning the previous value.
    pub fn atomic_sub_u32(addr: *mut u32, val: u32, order: u32) -> u32;
    /// Atomically store `min(*addr, val)`, returning the previous value.
    pub fn atomic_min_u32(addr: *mut u32, val: u32, order: u32) -> u32;
    /// Atomically store `max(*addr, val)`, returning the previous value.
    pub fn atomic_max_u32(addr: *mut u32, val: u32, order: u32) -> u32;
    /// Atomically store `*addr & val`, returning the previous value.
    pub fn atomic_and_u32(addr: *mut u32, val: u32, order: u32) -> u32;
    /// Atomically store `*addr | val`, returning the previous value.
    pub fn atomic_or_u32(addr: *mut u32, val: u32, order: u32) -> u32;
    /// Atomically store `*addr ^ val`, returning the previous value.
    pub fn atomic_xor_u32(addr: *mut u32, val: u32, order: u32) -> u32;
    /// Atomically overwrite `*addr` with `val`, returning the previous value.
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

    /// Signed twin of `atomic_add_u32`: atomically add `val` into `*addr`,
    /// returning the previous value.
    pub fn atomic_add_i32(addr: *mut i32, val: i32, order: u32) -> i32;
    /// Signed twin of `atomic_sub_u32`: atomically subtract `val` from
    /// `*addr`, returning the previous value.
    pub fn atomic_sub_i32(addr: *mut i32, val: i32, order: u32) -> i32;
}

// ── Math ───────────────────────────────────────────────────────────────

#[link(wasm_import_module = "quanta")]
unsafe extern "C" {
    /// Square root of `x`.
    pub fn sqrt_f32(x: f32) -> f32;
    /// Reciprocal square root, `1 / sqrt(x)`.
    pub fn rsqrt_f32(x: f32) -> f32;
    /// Sine of `x`, in radians.
    pub fn sin_f32(x: f32) -> f32;
    /// Cosine of `x`, in radians.
    pub fn cos_f32(x: f32) -> f32;
    /// Tangent of `x`, in radians.
    pub fn tan_f32(x: f32) -> f32;
    /// Base-e exponential, `e^x`.
    pub fn exp_f32(x: f32) -> f32;
    /// Natural logarithm of `x`.
    pub fn log_f32(x: f32) -> f32;
    /// `base` raised to the power `exp`.
    pub fn pow_f32(base: f32, exp: f32) -> f32;
    /// Absolute value of `x`.
    pub fn abs_f32(x: f32) -> f32;
    /// Largest integer not greater than `x`.
    pub fn floor_f32(x: f32) -> f32;
    /// Smallest integer not less than `x`.
    pub fn ceil_f32(x: f32) -> f32;
    /// `x` rounded to the nearest integer. The tie rule is the backend's
    /// (MSL rounds halves away from zero, WGSL to even) — kernels that
    /// care must not feed it exact halves.
    pub fn round_f32(x: f32) -> f32;
    /// The smaller of `a` and `b`.
    pub fn min_f32(a: f32, b: f32) -> f32;
    /// The larger of `a` and `b`.
    pub fn max_f32(a: f32, b: f32) -> f32;
    /// `x` clamped to the closed range `[lo, hi]`.
    pub fn clamp_f32(x: f32, lo: f32, hi: f32) -> f32;
    /// Fused multiply-add, `a * b + c` with a single rounding.
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
    /// Square root of `x`, in double precision.
    pub fn sqrt_f64(x: f64) -> f64;
    /// Reciprocal square root, `1 / sqrt(x)`, in double precision.
    pub fn rsqrt_f64(x: f64) -> f64;
    /// Sine of `x`, in radians, in double precision.
    pub fn sin_f64(x: f64) -> f64;
    /// Cosine of `x`, in radians, in double precision.
    pub fn cos_f64(x: f64) -> f64;
    /// Tangent of `x`, in radians, in double precision.
    pub fn tan_f64(x: f64) -> f64;
    /// Base-e exponential, `e^x`, in double precision.
    pub fn exp_f64(x: f64) -> f64;
    /// Natural logarithm of `x`, in double precision.
    pub fn log_f64(x: f64) -> f64;
    /// `base` raised to the power `exp`, in double precision.
    pub fn pow_f64(base: f64, exp: f64) -> f64;
    /// Absolute value of `x`, in double precision.
    pub fn abs_f64(x: f64) -> f64;
    /// Largest integer not greater than `x`, in double precision.
    pub fn floor_f64(x: f64) -> f64;
    /// Smallest integer not less than `x`, in double precision.
    pub fn ceil_f64(x: f64) -> f64;
    /// `x` rounded to the nearest integer, in double precision. The tie
    /// rule is the backend's, exactly as for `round_f32`.
    pub fn round_f64(x: f64) -> f64;
    /// The smaller of `a` and `b`, in double precision.
    pub fn min_f64(a: f64, b: f64) -> f64;
    /// The larger of `a` and `b`, in double precision.
    pub fn max_f64(a: f64, b: f64) -> f64;
    /// `x` clamped to the closed range `[lo, hi]`, in double precision.
    pub fn clamp_f64(x: f64, lo: f64, hi: f64) -> f64;
    /// Fused multiply-add, `a * b + c` with a single rounding, in double
    /// precision.
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
    /// The device's real subgroup width (4/8 on lavapipe, 8–32 on
    /// Intel, 32 on NVIDIA and Apple, 32 or 64 on AMD). Read it at
    /// runtime — it is a builtin on every backend, never a constant.
    pub fn subgroup_size() -> u32;
    /// This thread's lane index within its subgroup, `0..subgroup_size()`
    /// — a real builtin on every backend (`SubgroupLocalInvocationId`,
    /// `thread_index_in_simdgroup`, `subgroup_invocation_id`; the CPU
    /// executor's warp cohorts are consecutive `subgroup_size()`-wide
    /// slices of the workgroup, so `proton_id() % subgroup_size()` there).
    pub fn subgroup_id() -> u32;

    // Ballot / any / all take a predicate (any non-zero u32 == true);
    // a single u32 variant suffices regardless of the value type
    // being voted on.

    /// Bitmask of the subgroup lanes whose `predicate` holds, low bit =
    /// lane 0, truncated to 32 bits.
    pub fn ballot_u32(predicate: u32) -> u32;
    /// `1` if `predicate` holds in at least one lane of the subgroup,
    /// else `0`.
    pub fn any_u32(predicate: u32) -> u32;
    /// `1` if `predicate` holds in every lane of the subgroup, else `0`.
    pub fn all_u32(predicate: u32) -> u32;

    // The second argument is an XOR mask, not a source-lane index —
    // `lane_delta = 1` swaps adjacent pairs, `2` swaps pairs of pairs,
    // etc. (the standard butterfly pattern used by tree reductions).
    // Mirrors Metal's `simd_shuffle_xor` and WGSL's `subgroupShuffleXor`.

    /// Butterfly exchange: this lane receives the `value` held by lane
    /// `subgroup_id() ^ lane_delta`.
    pub fn shuffle_u32(value: u32, lane_delta: u32) -> u32;
    /// Butterfly exchange of a signed value — see `shuffle_u32`.
    pub fn shuffle_i32(value: i32, lane_delta: u32) -> i32;
    /// Butterfly exchange of a float value — see `shuffle_u32`.
    pub fn shuffle_f32(value: f32, lane_delta: u32) -> f32;

    /// Sum of `value` across the subgroup, broadcast to every lane.
    pub fn reduce_add_u32(value: u32) -> u32;
    /// Sum of `value` across the subgroup, broadcast to every lane.
    pub fn reduce_add_i32(value: i32) -> i32;
    /// Sum of `value` across the subgroup, broadcast to every lane.
    pub fn reduce_add_f32(value: f32) -> f32;
    /// Minimum of `value` across the subgroup, broadcast to every lane.
    pub fn reduce_min_u32(value: u32) -> u32;
    /// Minimum of `value` across the subgroup, broadcast to every lane.
    pub fn reduce_min_i32(value: i32) -> i32;
    /// Minimum of `value` across the subgroup, broadcast to every lane.
    pub fn reduce_min_f32(value: f32) -> f32;
    /// Maximum of `value` across the subgroup, broadcast to every lane.
    pub fn reduce_max_u32(value: u32) -> u32;
    /// Maximum of `value` across the subgroup, broadcast to every lane.
    pub fn reduce_max_i32(value: i32) -> i32;
    /// Maximum of `value` across the subgroup, broadcast to every lane.
    pub fn reduce_max_f32(value: f32) -> f32;

    /// Inclusive prefix sum: this lane receives the sum of `value` over
    /// lanes `0..=self`.
    pub fn scan_add_u32(value: u32) -> u32;
    /// Inclusive prefix sum: this lane receives the sum of `value` over
    /// lanes `0..=self`.
    pub fn scan_add_i32(value: i32) -> i32;
    /// Inclusive prefix sum: this lane receives the sum of `value` over
    /// lanes `0..=self`.
    pub fn scan_add_f32(value: f32) -> f32;

    // Pairs with the inclusive form above:
    // `inclusive[k] = exclusive[k] + value[k]`.

    /// Exclusive prefix sum: this lane receives the sum of `value` over
    /// lanes `0..self`, so lane 0 receives 0.
    pub fn scan_add_exclusive_u32(value: u32) -> u32;
    /// Exclusive prefix sum: this lane receives the sum of `value` over
    /// lanes `0..self`, so lane 0 receives 0.
    pub fn scan_add_exclusive_i32(value: i32) -> i32;
    /// Exclusive prefix sum: this lane receives the sum of `value` over
    /// lanes `0..self`, so lane 0 receives 0.
    pub fn scan_add_exclusive_f32(value: f32) -> f32;
}

// ── Workgroup-shared memory ────────────────────────────────────────────

#[link(wasm_import_module = "quanta")]
unsafe extern "C" {
    /// Load from workgroup-shared memory at `(slot, index)`.
    pub fn shared_load_f32(slot: u32, index: u32) -> f32;
    /// Load from workgroup-shared memory at `(slot, index)`.
    pub fn shared_load_u32(slot: u32, index: u32) -> u32;
    /// Load from workgroup-shared memory at `(slot, index)`.
    pub fn shared_load_i32(slot: u32, index: u32) -> i32;

    /// Store to workgroup-shared memory at `(slot, index)`.
    pub fn shared_store_f32(slot: u32, index: u32, val: f32);
    /// Store to workgroup-shared memory at `(slot, index)`.
    pub fn shared_store_u32(slot: u32, index: u32, val: u32);
    /// Store to workgroup-shared memory at `(slot, index)`.
    pub fn shared_store_i32(slot: u32, index: u32, val: i32);

    /// Atomic read-modify-write on workgroup-shared memory at
    /// `(slot, index)`. Returns the value at the slot **before** the
    /// operation, mirroring `atomic_add_u32` for buffers. `order`
    /// matches `quanta::MemoryOrder` discriminants — see the
    /// `ORDER_*` constants below.
    ///
    /// Use these for per-bucket histogram increments, in-block
    /// counters, etc., that would otherwise need a buffer-backed
    /// counter + a global-memory round-trip.
    pub fn atomic_add_shared_u32(slot: u32, index: u32, val: u32, order: u32) -> u32;
    /// Atomically subtract `val` from shared `(slot, index)`, returning
    /// the previous value.
    pub fn atomic_sub_shared_u32(slot: u32, index: u32, val: u32, order: u32) -> u32;
    /// Atomically store the smaller of `val` and shared `(slot, index)`,
    /// returning the previous value.
    pub fn atomic_min_shared_u32(slot: u32, index: u32, val: u32, order: u32) -> u32;
    /// Atomically store the larger of `val` and shared `(slot, index)`,
    /// returning the previous value.
    pub fn atomic_max_shared_u32(slot: u32, index: u32, val: u32, order: u32) -> u32;
    /// Atomically AND `val` into shared `(slot, index)`, returning the
    /// previous value.
    pub fn atomic_and_shared_u32(slot: u32, index: u32, val: u32, order: u32) -> u32;
    /// Atomically OR `val` into shared `(slot, index)`, returning the
    /// previous value.
    pub fn atomic_or_shared_u32(slot: u32, index: u32, val: u32, order: u32) -> u32;
    /// Atomically XOR `val` into shared `(slot, index)`, returning the
    /// previous value.
    pub fn atomic_xor_shared_u32(slot: u32, index: u32, val: u32, order: u32) -> u32;
    /// Atomically overwrite shared `(slot, index)` with `val`, returning
    /// the previous value.
    pub fn atomic_exchange_shared_u32(slot: u32, index: u32, val: u32, order: u32) -> u32;
    /// Signed twin of `atomic_add_shared_u32`.
    pub fn atomic_add_shared_i32(slot: u32, index: u32, val: i32, order: u32) -> i32;
    /// Signed twin of `atomic_sub_shared_u32`.
    pub fn atomic_sub_shared_i32(slot: u32, index: u32, val: i32, order: u32) -> i32;
}

// ── Textures ───────────────────────────────────────────────────────────

#[link(wasm_import_module = "quanta")]
unsafe extern "C" {
    /// Sampled texture read through the fixed NEAREST/CLAMP_TO_EDGE sampler.
    /// Slot must be a `&Sampled2D<T>` param, bound at dispatch time via
    /// `wave.bind_texture(slot, tex)`.
    pub fn texture_sample_2d_f32(slot: u32, x: u32, y: u32) -> f32;

    /// Unsampled (raw integer-coord) texture read. Legal against every 2D
    /// texture kind: texel slots (`&Texture2D` / `&mut Texture2D`) read the
    /// storage image directly; a sampled slot (`&Sampled2D`) reads via
    /// texelFetch — the texel-read path for textures without storage usage.
    pub fn texture_load_2d_f32(slot: u32, x: u32, y: u32) -> f32;
    /// Unsampled texel read from a 3-D texture slot — the volumetric
    /// twin of `texture_load_2d_f32`.
    pub fn texture_load_3d_f32(slot: u32, x: u32, y: u32, z: u32) -> f32;

    /// Packed-u32 texel read. Slot must be a `Texture2D<u32>` texel image
    /// (`&` read-only or `&mut` read-write), which is an RGBA8-unorm texture:
    /// the four unorm channels are packed into one `0xAABBGGRR` u32
    /// (little-endian byte order R,G,B,A). Unpack in the kernel with bit math
    /// — `let r = v & 0xFF; let g = (v >> 8) & 0xFF; ...`. (Sampled
    /// `&Sampled2D<u32>` is a different, unwired meaning and is rejected at
    /// emit; the read-only `&Texture2D<u32>` form works on every Metal tier.)
    pub fn texture_load_2d_u32(slot: u32, x: u32, y: u32) -> u32;

    /// Texture write (texel image). Slot must be bound to a
    /// `&mut Texture2D<T>` param — writes against the read-only `&Texture2D`
    /// form are rejected at emit.
    pub fn texture_write_2d_f32(slot: u32, x: u32, y: u32, val: f32);

    /// Packed-u32 storage-image write — the RGBA8-unorm twin of
    /// `texture_write_2d_f32`. The `val` is a `0xAABBGGRR` packed u32 (the same
    /// little-endian R,G,B,A channel order as `texture_load_2d_u32`); build it
    /// with `pack_unorm4x8` (preferred) or bit math — `let v = r | (g << 8) |
    /// (b << 16) | (a << 24)`.
    pub fn texture_write_2d_u32(slot: u32, x: u32, y: u32, val: u32);
}

// ── Packed RGBA8 pack / unpack ─────────────────────────────────────────
//
// Convenience intrinsics for the packed-u32 RGBA8 texel contract shared with
// `texture_load_2d_u32` / `texture_write_2d_u32` (byte 0 = R, little-endian
// `0xAABBGGRR`). They are the preferred, self-documenting form of the channel
// bit math; the lowering expands each to a short KernelOp sequence (no new IR
// op), so they run on every backend the composed ops already support.
//
// `pack_unorm4x8` clamps each channel to [0,1], rounds `x * 255`, and packs
// the four bytes R,G,B,A. The four `unpack_unorm4x8_*` invert it, returning
// `channel_byte as f32 / 255.0`. The round-trip is exact for byte-valued
// channels: `pack_unorm4x8(unpack_r(v), unpack_g(v), unpack_b(v),
// unpack_a(v)) == v` for every `v`.
//
// Example — scale the green channel of a packed texel in place:
//
// ```ignore
// let v = texture_load_2d_u32(0, x, y);
// let g = unpack_unorm4x8_g(v) * 0.5;
// let out = pack_unorm4x8(
//     unpack_unorm4x8_r(v), g, unpack_unorm4x8_b(v), unpack_unorm4x8_a(v),
// );
// texture_write_2d_u32(0, x, y, out);
// ```
#[link(wasm_import_module = "quanta")]
unsafe extern "C" {
    /// Pack four unorm channels into a `0xAABBGGRR` u32. Each channel is
    /// clamped to `[0, 1]`, scaled by 255, rounded to nearest, and stored
    /// in its byte (R = byte 0). Byte-identical to the `Texture2D<u32>`
    /// texel `texture_write_2d_u32` stores.
    pub fn pack_unorm4x8(r: f32, g: f32, b: f32, a: f32) -> u32;

    /// Unpack the R channel (byte 0) of a packed RGBA8 u32 as `byte / 255`.
    pub fn unpack_unorm4x8_r(v: u32) -> f32;
    /// Unpack the G channel (byte 1) of a packed RGBA8 u32 as `byte / 255`.
    pub fn unpack_unorm4x8_g(v: u32) -> f32;
    /// Unpack the B channel (byte 2) of a packed RGBA8 u32 as `byte / 255`.
    pub fn unpack_unorm4x8_b(v: u32) -> f32;
    /// Unpack the A channel (byte 3) of a packed RGBA8 u32 as `byte / 255`.
    pub fn unpack_unorm4x8_a(v: u32) -> f32;
}

// ── Memory-order discriminants ─────────────────────────────────────────

/// Memory ordering values used by `*_order` parameters above.
/// Mirrors `quanta::MemoryOrder` so kernel code can name them
/// symbolically:
///
/// ```ignore
/// atomic_add_u32(p, 1, ORDER_RELAXED);
/// ```
///
/// `ORDER_RELAXED` asks for atomicity and nothing more: no ordering is
/// imposed on the reads and writes around the operation.
pub const ORDER_RELAXED: u32 = 0;
/// Acquire: no read or write placed after this operation may be observed
/// to happen before it.
pub const ORDER_ACQUIRE: u32 = 1;
/// Release: no read or write placed before this operation may be observed
/// to happen after it.
pub const ORDER_RELEASE: u32 = 2;
/// Acquire and release together — both restrictions on one operation.
pub const ORDER_ACQ_REL: u32 = 3;
/// Sequential consistency: every `ORDER_SEQ_CST` operation in the
/// dispatch is observed in one total order by every quark.
pub const ORDER_SEQ_CST: u32 = 4;
