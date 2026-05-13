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
fn philox4x32_10_first_u32_kernel(c0: u32, c1: u32, c2: u32, c3: u32, k0: u32, k1: u32) -> u32 {
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
pub fn fill_uniform_u32_gpu(gpu: &Gpu, len: usize, seed: u64) -> Result<Vec<u32>, QuantaError> {
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
pub fn fill_uniform_u64_gpu(gpu: &Gpu, len: usize, seed: u64) -> Result<Vec<u64>, QuantaError> {
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
pub fn fill_uniform_f32_gpu(gpu: &Gpu, len: usize, seed: u64) -> Result<Vec<f32>, QuantaError> {
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
pub fn fill_uniform_f64_gpu(gpu: &Gpu, len: usize, seed: u64) -> Result<Vec<f64>, QuantaError> {
    let mut data = FillUniformF64Data {
        out: vec![0.0f64; len],
        seed_lo: seed as u32,
        seed_hi: (seed >> 32) as u32,
    };
    fill_uniform_f64(gpu, &mut data, len as u32)?.wait()?;
    Ok(data.out)
}

// ── Normal (Box-Muller) ──────────────────────────────────────────────
//
// Box-Muller transforms two independent uniforms in (0, 1] into two
// independent normals with mean 0 and variance 1:
//   r     = sqrt(-2 * ln(u1))
//   theta = 2π * u2
//   n1    = r * cos(theta)
//   n2    = r * sin(theta)
//
// One Philox4×32 draw gives us four u32s, which is *almost* enough
// for two normals (we need 2 uniforms ≈ 2 u32s) — but we want
// independent counter slots for clarity, so each quark does two
// independent Philox draws (counter words 0 and 1), then produces
// the pair n1/n2 from u1/u2.

#[derive(quanta::Fields)]
pub struct FillNormalF32Data {
    pub out: Vec<f32>,
    pub seed_lo: u32,
    pub seed_hi: u32,
}

/// Per-quark Box-Muller. Each quark produces *two* f32 normals
/// from two Philox4×32 draws and writes them at positions
/// `id*2` and `id*2 + 1`. Host dispatches `len / 2` quarks
/// (rounded up for odd lengths; the host trims).
#[quanta::kernel]
pub fn fill_normal_f32(d: &FillNormalF32Data) {
    let id = quark_id();

    // Two independent uniforms in (0, 1] — using the
    // open-on-zero conversion so `ln(u1)` is finite. The
    // formula `u * 2^-32 + 2^-33` from `uniform::u32_to_open_unit_f32`.
    let r0: u32 = philox4x32_10_first_u32_kernel(id, 0u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    let r1: u32 = philox4x32_10_first_u32_kernel(id, 1u32, 0u32, 0u32, d.seed_lo, d.seed_hi);

    let bits0: u32 = r0 >> 8u32;
    let bits1: u32 = r1 >> 8u32;
    let u1: f32 = (bits0 as f32) * (1.0f32 / 16_777_216.0f32) + (1.0f32 / 33_554_432.0f32);
    let u2: f32 = (bits1 as f32) * (1.0f32 / 16_777_216.0f32) + (1.0f32 / 33_554_432.0f32);

    // Box-Muller. `ln_u1` is the only place a non-finite could leak
    // in, and the open-unit conversion above guarantees `u1 > 0`.
    let ln_u1: f32 = ln(u1);
    let r: f32 = sqrt(-2.0f32 * ln_u1);
    let two_pi: f32 = 6.2831_8530_7179_586f32; // 2π
    let theta: f32 = two_pi * u2;
    let n1: f32 = r * cos(theta);
    let n2: f32 = r * sin(theta);

    // Write the pair at id*2 and id*2 + 1. Compute the two indices
    // independently so rustc doesn't fold them into one byte-offset
    // store-with-immediate-offset (which the WASM-route lowering
    // doesn't yet handle for f32). The redundant `id*2` work folds
    // out in LLVM by the time we reach the lowering.
    let idx0: u32 = id.wrapping_mul(2u32);
    let idx1: u32 = id.wrapping_mul(2u32).wrapping_add(1u32);
    d.out[idx0 as usize] = n1;
    d.out[idx1 as usize] = n2;
}

/// Host-side dispatch for `fill_normal_f32`. Produces `len` f32
/// values drawn from N(0, 1). Internally dispatches `(len + 1) / 2`
/// quarks (each produces a pair) and trims the result.
pub fn fill_normal_f32_gpu(gpu: &Gpu, len: usize, seed: u64) -> Result<Vec<f32>, QuantaError> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let quarks = len.div_ceil(2);
    let padded = quarks * 2;
    let mut data = FillNormalF32Data {
        out: vec![0.0f32; padded],
        seed_lo: seed as u32,
        seed_hi: (seed >> 32) as u32,
    };
    fill_normal_f32(gpu, &mut data, quarks as u32)?.wait()?;
    data.out.truncate(len);
    Ok(data.out)
}

// ── Normal f64 ───────────────────────────────────────────────────────

#[derive(quanta::Fields)]
pub struct FillNormalF64Data {
    pub out: Vec<f64>,
    pub seed_lo: u32,
    pub seed_hi: u32,
}

/// f64 Box-Muller. Same algorithm as the f32 form but draws four
/// Philox words per quark to build TWO u64 → two f64 uniforms in
/// `(0, 1]`, then `(r*cos, r*sin)` with f64 math.
#[quanta::kernel]
pub fn fill_normal_f64(d: &FillNormalF64Data) {
    let id = quark_id();

    // Two independent u64 uniforms. Each needs two Philox draws.
    let r0a: u32 = philox4x32_10_first_u32_kernel(id, 0u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    let r0b: u32 = philox4x32_10_first_u32_kernel(id, 1u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    let r1a: u32 = philox4x32_10_first_u32_kernel(id, 2u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    let r1b: u32 = philox4x32_10_first_u32_kernel(id, 3u32, 0u32, 0u32, d.seed_lo, d.seed_hi);

    let packed0: u64 = ((r0a as u64) << 32u32) | (r0b as u64);
    let packed1: u64 = ((r1a as u64) << 32u32) | (r1b as u64);

    // Open-on-zero f64 in (0, 1]: top 53 bits + half-ULP.
    let bits0: u64 = packed0 >> 11u32;
    let bits1: u64 = packed1 >> 11u32;
    let u1: f64 = (bits0 as f64) * (1.0f64 / 9_007_199_254_740_992.0f64)
        + (1.0f64 / 18_014_398_509_481_984.0f64);
    let u2: f64 = (bits1 as f64) * (1.0f64 / 9_007_199_254_740_992.0f64)
        + (1.0f64 / 18_014_398_509_481_984.0f64);

    let ln_u1: f64 = log_f64(u1);
    let r: f64 = sqrt_f64(-2.0f64 * ln_u1);
    let two_pi: f64 = 6.283_185_307_179_586f64;
    let theta: f64 = two_pi * u2;
    let n1: f64 = r * cos_f64(theta);
    let n2: f64 = r * sin_f64(theta);

    let idx0: u32 = id.wrapping_mul(2u32);
    let idx1: u32 = id.wrapping_mul(2u32).wrapping_add(1u32);
    d.out[idx0 as usize] = n1;
    d.out[idx1 as usize] = n2;
}

pub fn fill_normal_f64_gpu(gpu: &Gpu, len: usize, seed: u64) -> Result<Vec<f64>, QuantaError> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let quarks = len.div_ceil(2);
    let padded = quarks * 2;
    let mut data = FillNormalF64Data {
        out: vec![0.0f64; padded],
        seed_lo: seed as u32,
        seed_hi: (seed >> 32) as u32,
    };
    fill_normal_f64(gpu, &mut data, quarks as u32)?.wait()?;
    data.out.truncate(len);
    Ok(data.out)
}

// ── Exponential ──────────────────────────────────────────────────────
//
// Exponential distribution via inverse-CDF: `X = -ln(1 - U) / lambda`.
// With `U` uniform in `(0, 1]`, `1 - U` is uniform in `[0, 1)` and
// `ln(0)` would be `-inf`. We use the open-on-zero conversion so
// `U > 0`, then sample `-ln(U) / lambda` — equivalent in
// distribution because U and 1-U have the same uniform law.

#[derive(quanta::Fields)]
pub struct FillExponentialF32Data {
    pub out: Vec<f32>,
    pub seed_lo: u32,
    pub seed_hi: u32,
    /// Rate parameter `lambda` (mean of the distribution is `1/lambda`).
    pub lambda: f32,
}

/// Per-quark Exponential(lambda) draw, inverse-CDF.
#[quanta::kernel]
pub fn fill_exponential_f32(d: &FillExponentialF32Data) {
    let id = quark_id();
    let r: u32 = philox4x32_10_first_u32_kernel(id, 0u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    let bits: u32 = r >> 8u32;
    let u: f32 = (bits as f32) * (1.0f32 / 16_777_216.0f32) + (1.0f32 / 33_554_432.0f32);
    let v: f32 = -ln(u) / d.lambda;
    d.out[id as usize] = v;
}

/// Host-side dispatch for `fill_exponential_f32`.
pub fn fill_exponential_f32_gpu(
    gpu: &Gpu,
    len: usize,
    seed: u64,
    lambda: f32,
) -> Result<Vec<f32>, QuantaError> {
    let mut data = FillExponentialF32Data {
        out: vec![0.0f32; len],
        seed_lo: seed as u32,
        seed_hi: (seed >> 32) as u32,
        lambda,
    };
    fill_exponential_f32(gpu, &mut data, len as u32)?.wait()?;
    Ok(data.out)
}

// ── Exponential f64 ─────────────────────────────────────────────────

#[derive(quanta::Fields)]
pub struct FillExponentialF64Data {
    pub out: Vec<f64>,
    pub seed_lo: u32,
    pub seed_hi: u32,
    pub lambda: f64,
}

#[quanta::kernel]
pub fn fill_exponential_f64(d: &FillExponentialF64Data) {
    let id = quark_id();
    let ra: u32 = philox4x32_10_first_u32_kernel(id, 0u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    let rb: u32 = philox4x32_10_first_u32_kernel(id, 1u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    let packed: u64 = ((ra as u64) << 32u32) | (rb as u64);
    let bits: u64 = packed >> 11u32;
    let u: f64 = (bits as f64) * (1.0f64 / 9_007_199_254_740_992.0f64)
        + (1.0f64 / 18_014_398_509_481_984.0f64);
    let lam: f64 = d.lambda;
    let v: f64 = -log_f64(u) / lam;
    d.out[id as usize] = v;
}

pub fn fill_exponential_f64_gpu(
    gpu: &Gpu,
    len: usize,
    seed: u64,
    lambda: f64,
) -> Result<Vec<f64>, QuantaError> {
    let mut data = FillExponentialF64Data {
        out: vec![0.0f64; len],
        seed_lo: seed as u32,
        seed_hi: (seed >> 32) as u32,
        lambda,
    };
    fill_exponential_f64(gpu, &mut data, len as u32)?.wait()?;
    Ok(data.out)
}

// ── LogNormal f64 ────────────────────────────────────────────────────

#[derive(quanta::Fields)]
pub struct FillLogNormalF64Data {
    pub out: Vec<f64>,
    pub seed_lo: u32,
    pub seed_hi: u32,
    pub mu: f64,
    pub sigma: f64,
}

#[quanta::kernel]
pub fn fill_lognormal_f64(d: &FillLogNormalF64Data) {
    let id = quark_id();
    let r0a: u32 = philox4x32_10_first_u32_kernel(id, 0u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    let r0b: u32 = philox4x32_10_first_u32_kernel(id, 1u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    let r1a: u32 = philox4x32_10_first_u32_kernel(id, 2u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    let r1b: u32 = philox4x32_10_first_u32_kernel(id, 3u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    let packed0: u64 = ((r0a as u64) << 32u32) | (r0b as u64);
    let packed1: u64 = ((r1a as u64) << 32u32) | (r1b as u64);
    let bits0: u64 = packed0 >> 11u32;
    let bits1: u64 = packed1 >> 11u32;
    let u1: f64 = (bits0 as f64) * (1.0f64 / 9_007_199_254_740_992.0f64)
        + (1.0f64 / 18_014_398_509_481_984.0f64);
    let u2: f64 = (bits1 as f64) * (1.0f64 / 9_007_199_254_740_992.0f64)
        + (1.0f64 / 18_014_398_509_481_984.0f64);
    let r: f64 = sqrt_f64(-2.0f64 * log_f64(u1));
    let two_pi: f64 = 6.283_185_307_179_586f64;
    let theta: f64 = two_pi * u2;
    let n1: f64 = r * cos_f64(theta);
    let n2: f64 = r * sin_f64(theta);
    let mu: f64 = d.mu;
    let sigma: f64 = d.sigma;
    let v1: f64 = exp_f64(mu + sigma * n1);
    let v2: f64 = exp_f64(mu + sigma * n2);
    let idx0: u32 = id.wrapping_mul(2u32);
    let idx1: u32 = id.wrapping_mul(2u32).wrapping_add(1u32);
    d.out[idx0 as usize] = v1;
    d.out[idx1 as usize] = v2;
}

pub fn fill_lognormal_f64_gpu(
    gpu: &Gpu,
    len: usize,
    seed: u64,
    mu: f64,
    sigma: f64,
) -> Result<Vec<f64>, QuantaError> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let quarks = len.div_ceil(2);
    let padded = quarks * 2;
    let mut data = FillLogNormalF64Data {
        out: vec![0.0f64; padded],
        seed_lo: seed as u32,
        seed_hi: (seed >> 32) as u32,
        mu,
        sigma,
    };
    fill_lognormal_f64(gpu, &mut data, quarks as u32)?.wait()?;
    data.out.truncate(len);
    Ok(data.out)
}

// ── LogNormal ────────────────────────────────────────────────────────
//
// LogNormal(mu, sigma): `X = exp(mu + sigma * N)` where `N ~ N(0, 1)`.
// Uses Box-Muller for the normal, same shape as `fill_normal_f32`.

#[derive(quanta::Fields)]
pub struct FillLogNormalF32Data {
    pub out: Vec<f32>,
    pub seed_lo: u32,
    pub seed_hi: u32,
    pub mu: f32,
    pub sigma: f32,
}

/// Per-quark LogNormal(mu, sigma). Each quark produces two outputs
/// (same Box-Muller pair structure as `fill_normal_f32`), exp'd
/// through the (mu, sigma) shift+scale.
#[quanta::kernel]
pub fn fill_lognormal_f32(d: &FillLogNormalF32Data) {
    let id = quark_id();
    let r0: u32 = philox4x32_10_first_u32_kernel(id, 0u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    let r1: u32 = philox4x32_10_first_u32_kernel(id, 1u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    let bits0: u32 = r0 >> 8u32;
    let bits1: u32 = r1 >> 8u32;
    let u1: f32 = (bits0 as f32) * (1.0f32 / 16_777_216.0f32) + (1.0f32 / 33_554_432.0f32);
    let u2: f32 = (bits1 as f32) * (1.0f32 / 16_777_216.0f32) + (1.0f32 / 33_554_432.0f32);
    let r: f32 = sqrt(-2.0f32 * ln(u1));
    let two_pi: f32 = 6.2831_8530_7179_586f32;
    let theta: f32 = two_pi * u2;
    let n1: f32 = r * cos(theta);
    let n2: f32 = r * sin(theta);
    let v1: f32 = exp(d.mu + d.sigma * n1);
    let v2: f32 = exp(d.mu + d.sigma * n2);
    let idx0: u32 = id.wrapping_mul(2u32);
    let idx1: u32 = id.wrapping_mul(2u32).wrapping_add(1u32);
    d.out[idx0 as usize] = v1;
    d.out[idx1 as usize] = v2;
}

/// Host-side dispatch for `fill_lognormal_f32`.
pub fn fill_lognormal_f32_gpu(
    gpu: &Gpu,
    len: usize,
    seed: u64,
    mu: f32,
    sigma: f32,
) -> Result<Vec<f32>, QuantaError> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let quarks = len.div_ceil(2);
    let padded = quarks * 2;
    let mut data = FillLogNormalF32Data {
        out: vec![0.0f32; padded],
        seed_lo: seed as u32,
        seed_hi: (seed >> 32) as u32,
        mu,
        sigma,
    };
    fill_lognormal_f32(gpu, &mut data, quarks as u32)?.wait()?;
    data.out.truncate(len);
    Ok(data.out)
}

// ── Bernoulli ────────────────────────────────────────────────────────
//
// Bernoulli(p): output 1 with probability p, 0 otherwise. Implemented
// as `u < p` where `u` is uniform in `[0, 1)`. Output stored as u32
// (1 or 0) for compactness; users who want bool can cast on host.

#[derive(quanta::Fields)]
pub struct FillBernoulliU32Data {
    pub out: Vec<u32>,
    pub seed_lo: u32,
    pub seed_hi: u32,
    /// Success probability in `[0, 1]`. Out-of-range values still
    /// produce defined behaviour: p ≤ 0 → all zeros, p ≥ 1 → all ones.
    pub p: f32,
}

/// Per-quark Bernoulli(p) draw.
#[quanta::kernel]
pub fn fill_bernoulli_u32(d: &FillBernoulliU32Data) {
    let id = quark_id();
    let r: u32 = philox4x32_10_first_u32_kernel(id, 0u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    let bits: u32 = r >> 8u32;
    let u: f32 = (bits as f32) * (1.0f32 / 16_777_216.0f32);
    // Branchless cast: `(u < p) as u32`. Using arithmetic so the
    // WASM lowering doesn't depend on a bool-store path.
    let v: u32 = if u < d.p { 1u32 } else { 0u32 };
    d.out[id as usize] = v;
}

/// Host-side dispatch for `fill_bernoulli_u32`. Returns a `Vec<u32>`
/// of length `len` containing 1s and 0s (1 with probability `p`).
pub fn fill_bernoulli_u32_gpu(
    gpu: &Gpu,
    len: usize,
    seed: u64,
    p: f32,
) -> Result<Vec<u32>, QuantaError> {
    let mut data = FillBernoulliU32Data {
        out: vec![0u32; len],
        seed_lo: seed as u32,
        seed_hi: (seed >> 32) as u32,
        p,
    };
    fill_bernoulli_u32(gpu, &mut data, len as u32)?.wait()?;
    Ok(data.out)
}

// ── Poisson (Knuth, small lambda only) ───────────────────────────────
//
// Knuth's algorithm: draw uniforms until their product drops below
// `exp(-lambda)`. The expected iteration count is `lambda + 1`, so
// this is efficient for small lambda. v0.1 caps iterations at
// `POISSON_MAX_K = 64` — adequate for `lambda <= ~30` with vanishing
// truncation probability. Large-lambda (transformed-rejection /
// PTRD) variants are queued for a future release.
//
// Each quark draws an independent stream of uniforms from
// Philox4×32 by bumping the second counter slot — every iteration
// uses `(quark_id, iter, 0, 0)` as its counter.

const POISSON_MAX_K_U32: u32 = 64u32;

#[derive(quanta::Fields)]
pub struct FillPoissonU32Data {
    pub out: Vec<u32>,
    pub seed_lo: u32,
    pub seed_hi: u32,
    /// Mean of the Poisson distribution. v0.1 supports lambda up to
    /// ~30 with the iteration cap at 64.
    pub lambda: f32,
}

/// Per-quark Poisson(lambda) draw via Knuth's algorithm. Bounded at
/// 64 iterations; for the small-lambda regime this is effectively
/// the same as the unbounded algorithm.
#[quanta::kernel]
pub fn fill_poisson_u32(d: &FillPoissonU32Data) {
    let id = quark_id();
    let lam: f32 = d.lambda;
    let l_threshold: f32 = exp(0.0f32 - lam);
    let mut p: f32 = 1.0f32;
    let mut k: u32 = 0u32;
    let mut iter: u32 = 0u32;
    while iter < 64u32 {
        let r: u32 = philox4x32_10_first_u32_kernel(id, iter, 0u32, 0u32, d.seed_lo, d.seed_hi);
        let bits: u32 = r >> 8u32;
        let u: f32 = (bits as f32) * (1.0f32 / 16_777_216.0f32);
        p = p * u;
        if p <= l_threshold {
            // Found the stopping iteration — Knuth returns k.
            break;
        }
        k = k + 1u32;
        iter = iter + 1u32;
    }
    d.out[id as usize] = k;
}

/// Host-side dispatch for `fill_poisson_u32`.
///
/// Caveat: v0.1 caps the inner iteration at 64, so `lambda` above
/// ~30 will under-sample the tail. For larger means, downstream
/// users should use the host-side `Rng` API and a proper rejection
/// sampler until a transformed-rejection kernel ships.
pub fn fill_poisson_u32_gpu(
    gpu: &Gpu,
    len: usize,
    seed: u64,
    lambda: f32,
) -> Result<Vec<u32>, QuantaError> {
    let mut data = FillPoissonU32Data {
        out: vec![0u32; len],
        seed_lo: seed as u32,
        seed_hi: (seed >> 32) as u32,
        lambda,
    };
    fill_poisson_u32(gpu, &mut data, len as u32)?.wait()?;
    Ok(data.out)
}

// Unused but kept for doc consistency with the iteration cap.
#[allow(dead_code)]
const POISSON_MAX_K_HOST: u32 = POISSON_MAX_K_U32;

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
