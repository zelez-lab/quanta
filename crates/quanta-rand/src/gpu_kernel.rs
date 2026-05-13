//! GPU `#[quanta::kernel]` entry points for filling a buffer with
//! per-quark pseudo-random values, using Philox4×32-10 as the
//! counter-based generator.
//!
//! Each kernel takes the per-quark `quark_id` as its counter (with
//! the other three counter words held at zero) and a shared
//! `(seed_lo, seed_hi)` key, runs the full 10-round Philox
//! bijection in-kernel, and writes the result through the host-
//! side `u32`/`u64`/`f32`/`f64` conversion path.
//!
//! Determinism: running the same kernel with the same `seed` and
//! `quark_count` produces bit-identical output across host and
//! GPU. The integration tests in `tests/uniform_fill_correctness`
//! validate this end-to-end.
//!
//! ## Why a local copy of `philox4x32_10_first_u32`?
//!
//! `#[quanta::device]` registers a fn's source in a *per-crate*
//! registry that the kernel macro reads from when expanding. The
//! registry is process-wide but only sees fns that were attribute-
//! expanded in the same crate compilation, so a `#[quanta::device]`
//! fn in `philox4x32.rs` IS visible from kernels in `gpu_kernel.rs`
//! — same crate.

use quanta::*;

/// 32×32 → 64-bit multiply, returning hi half. Used inside the
/// in-kernel Philox round. `#[quanta::device]` exposes the source
/// to the wasm shell at macro expansion time; the LLVM optimiser
/// at -O3 typically folds these into the caller.
#[allow(dead_code)]
#[quanta::device]
fn philox_mulhi32(a: u32, b: u32) -> u32 {
    let prod = (a as u64).wrapping_mul(b as u64);
    (prod >> 32u32) as u32
}

/// In-kernel Philox4×32-10, returning the first output word. Same
/// algorithm as `philox4x32::philox4x32_10_first_u32` in the host
/// API; transcribed here so the kernel macro can splice it into
/// the wasm shell.
#[allow(dead_code)]
#[quanta::device]
fn philox4x32_10_first_u32_kernel(
    c0: u32,
    c1: u32,
    c2: u32,
    c3: u32,
    k0: u32,
    k1: u32,
) -> u32 {
    const M0_K: u32 = 0xD251_1F53;
    const M1_K: u32 = 0xCD9E_8D57;
    const W0_K: u32 = 0x9E37_79B9;
    const W1_K: u32 = 0xBB67_AE85;

    let mut x0 = c0;
    let mut x1 = c1;
    let mut x2 = c2;
    let mut x3 = c3;
    let mut key0 = k0;
    let mut key1 = k1;

    let mut i: u32 = 0;
    while i < 10u32 {
        if i > 0 {
            key0 = key0.wrapping_add(W0_K);
            key1 = key1.wrapping_add(W1_K);
        }
        let p0 = (M0_K as u64).wrapping_mul(x0 as u64);
        let hi0 = (p0 >> 32u32) as u32;
        let lo0 = p0 as u32;
        let p1 = (M1_K as u64).wrapping_mul(x2 as u64);
        let hi1 = (p1 >> 32u32) as u32;
        let lo1 = p1 as u32;
        let new_x0 = hi1 ^ x1 ^ key0;
        let new_x1 = lo1;
        let new_x2 = hi0 ^ x3 ^ key1;
        let new_x3 = lo0;
        x0 = new_x0;
        x1 = new_x1;
        x2 = new_x2;
        x3 = new_x3;
        i += 1;
    }
    x0
}

// ── u32 fill ─────────────────────────────────────────────────────────

#[derive(quanta::Fields)]
pub struct FillUniformU32Data {
    pub out: Vec<u32>,
    pub seed_lo: u32,
    pub seed_hi: u32,
}

/// Per-quark fill: `out[id] = philox4x32_10(counter=id, key=seed).x0`.
#[quanta::kernel]
pub fn fill_uniform_u32(d: &FillUniformU32Data) {
    let id = quark_id();
    let r: u32 = philox4x32_10_first_u32_kernel(id, 0u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    d.out[id as usize] = r;
}

/// Host-side dispatch for `fill_uniform_u32`. Returns a fresh
/// `Vec<u32>` of length `len` filled with bit-exact Philox4×32-10
/// output (counter = quark_id).
pub fn fill_uniform_u32_gpu(
    gpu: &Gpu,
    len: usize,
    seed: u64,
) -> Result<Vec<u32>, QuantaError> {
    let mut data = FillUniformU32Data {
        out: vec![0u32; len],
        seed_lo: seed as u32,
        seed_hi: (seed >> 32) as u32,
    };
    fill_uniform_u32(gpu, &mut data, len as u32)?.wait()?;
    Ok(data.out)
}

// ── u64 fill ─────────────────────────────────────────────────────────

#[derive(quanta::Fields)]
pub struct FillUniformU64Data {
    pub out: Vec<u64>,
    pub seed_lo: u32,
    pub seed_hi: u32,
}

/// Per-quark u64 fill. Two Philox draws (different counter words)
/// packed `(hi << 32) | lo` — same packing convention as
/// `Rng::next_u64` in the CPU API.
#[quanta::kernel]
pub fn fill_uniform_u64(d: &FillUniformU64Data) {
    let id = quark_id();
    // First draw: counter = (id, 0, 0, 0).
    let hi: u32 = philox4x32_10_first_u32_kernel(id, 0u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    // Second draw: counter = (id, 1, 0, 0). Bumping a different
    // counter slot is the standard counter-based-RNG idiom to get
    // an independent output from the same key.
    let lo: u32 = philox4x32_10_first_u32_kernel(id, 1u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    let packed: u64 = ((hi as u64) << 32u32) | (lo as u64);
    d.out[id as usize] = packed;
}

/// Host-side dispatch for `fill_uniform_u64`.
pub fn fill_uniform_u64_gpu(
    gpu: &Gpu,
    len: usize,
    seed: u64,
) -> Result<Vec<u64>, QuantaError> {
    let mut data = FillUniformU64Data {
        out: vec![0u64; len],
        seed_lo: seed as u32,
        seed_hi: (seed >> 32) as u32,
    };
    fill_uniform_u64(gpu, &mut data, len as u32)?.wait()?;
    Ok(data.out)
}

// ── f32 fill (uniform [0, 1)) ────────────────────────────────────────

#[derive(quanta::Fields)]
pub struct FillUniformF32Data {
    pub out: Vec<f32>,
    pub seed_lo: u32,
    pub seed_hi: u32,
}

/// Per-quark f32 fill in `[0, 1)`. Same `u32_to_unit_f32`
/// conversion as `Rng::next_f32` — bit-exact between host and
/// GPU.
#[quanta::kernel]
pub fn fill_uniform_f32(d: &FillUniformF32Data) {
    let id = quark_id();
    let r: u32 = philox4x32_10_first_u32_kernel(id, 0u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    // Top 24 bits as mantissa, scaled by 2^-24 → uniform [0, 1).
    let bits: u32 = r >> 8u32;
    let v: f32 = (bits as f32) * (1.0f32 / 16_777_216.0f32);
    d.out[id as usize] = v;
}

/// Host-side dispatch for `fill_uniform_f32`.
pub fn fill_uniform_f32_gpu(
    gpu: &Gpu,
    len: usize,
    seed: u64,
) -> Result<Vec<f32>, QuantaError> {
    let mut data = FillUniformF32Data {
        out: vec![0.0f32; len],
        seed_lo: seed as u32,
        seed_hi: (seed >> 32) as u32,
    };
    fill_uniform_f32(gpu, &mut data, len as u32)?.wait()?;
    Ok(data.out)
}

// ── f64 fill (uniform [0, 1)) ────────────────────────────────────────

#[derive(quanta::Fields)]
pub struct FillUniformF64Data {
    pub out: Vec<f64>,
    pub seed_lo: u32,
    pub seed_hi: u32,
}

/// Per-quark f64 fill in `[0, 1)`. Two Philox draws packed into a
/// u64, then top 53 bits as mantissa scaled by 2^-53 — same path
/// as `Rng::next_f64`.
#[quanta::kernel]
pub fn fill_uniform_f64(d: &FillUniformF64Data) {
    let id = quark_id();
    let hi: u32 = philox4x32_10_first_u32_kernel(id, 0u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    let lo: u32 = philox4x32_10_first_u32_kernel(id, 1u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    let packed: u64 = ((hi as u64) << 32u32) | (lo as u64);
    let bits: u64 = packed >> 11u32;
    let v: f64 = (bits as f64) * (1.0f64 / 9_007_199_254_740_992.0f64);
    d.out[id as usize] = v;
}

/// Host-side dispatch for `fill_uniform_f64`.
pub fn fill_uniform_f64_gpu(
    gpu: &Gpu,
    len: usize,
    seed: u64,
) -> Result<Vec<f64>, QuantaError> {
    let mut data = FillUniformF64Data {
        out: vec![0.0f64; len],
        seed_lo: seed as u32,
        seed_hi: (seed >> 32) as u32,
    };
    fill_uniform_f64(gpu, &mut data, len as u32)?.wait()?;
    Ok(data.out)
}

// ── Backwards-compat aliases ─────────────────────────────────────────
//
// The original v0 API was `fill_buffer` / `fill_buffer_gpu` (always
// u32, splitmix-based). Keep the names alive as thin Philox-backed
// aliases so existing callers don't break — they get the upgraded
// algorithm for free.

/// Auto-dispatch struct alias matching the original `FillBufferData`
/// shape; identical to `FillUniformU32Data`.
pub use FillUniformU32Data as FillBufferData;

/// Legacy alias for `fill_uniform_u32_gpu`. Output now comes from
/// Philox4×32-10 (a BigCrush-clean generator) instead of the v0
/// splitmix-based xoshiro128++ shape — same `(seed, quark_id)`
/// determinism contract, different bits.
pub fn fill_buffer_gpu(gpu: &Gpu, len: usize, seed: u64) -> Result<Vec<u32>, QuantaError> {
    fill_uniform_u32_gpu(gpu, len, seed)
}
