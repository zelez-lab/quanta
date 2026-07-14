//! Tour of the quanta-rand fill API — one call per distribution.
//!
//! This is the "what does the library look like in practice" example.
//! Each section is a complete, copy-pasteable snippet for one
//! distribution. The output is deterministic — re-running with the
//! same seed produces identical bytes.
//!
//! Backend: software CPU JIT. Replace `quanta::init_cpu()` with
//! `quanta::init()?` on a machine with a real GPU to hit the
//! Metal / Vulkan / WebGPU path; the API surface is identical.
//!
//! Run: cargo run -p quanta-rand --example fill_buffer --features gpu

use quanta_rand::{
    fill_bernoulli_u32_gpu, fill_exponential_f32_gpu, fill_lognormal_f32_gpu, fill_normal_f32_gpu,
    fill_poisson_u32_gpu, fill_uniform_f32_gpu, fill_uniform_f64_gpu, fill_uniform_u32_gpu,
    fill_uniform_u64_gpu,
};

fn main() {
    let gpu = quanta::init_cpu();
    let len = 16; // small enough to print every value
    let seed = 0xCAFE_BABE_DEAD_BEEFu64;

    println!("== quanta-rand fill API tour ==");
    println!("backend: {}", gpu.name());
    println!("len    : {len}");
    println!("seed   : 0x{seed:016X}");
    println!();

    // ── Uniform ──────────────────────────────────────────────────────
    //
    // The four uniform variants share the same shape: take a `&Gpu`,
    // length, and u64 seed, return a `Vec<T>` of pseudo-random
    // values. Output is bit-exact with the host-side
    // `philox4x32_10_first_u32` reference.

    let u32_out = fill_uniform_u32_gpu(&gpu, len, seed).unwrap();
    println!("uniform_u32: {u32_out:?}");

    let u64_out = fill_uniform_u64_gpu(&gpu, len, seed).unwrap();
    println!("uniform_u64: {u64_out:?}");

    let f32_out = fill_uniform_f32_gpu(&gpu, len, seed).unwrap();
    println!("uniform_f32 ([0, 1)): {f32_out:.4?}");

    let f64_out = fill_uniform_f64_gpu(&gpu, len, seed).unwrap();
    println!("uniform_f64 ([0, 1)): {f64_out:.6?}");

    // ── Normal ───────────────────────────────────────────────────────
    //
    // Standard normal N(0, 1) via Box-Muller. To get N(mu, sigma²),
    // map the output: `mu + sigma * z` host-side.

    let normal = fill_normal_f32_gpu(&gpu, len, seed).unwrap();
    println!("normal_f32 (mu=0, sigma=1): {normal:.4?}");

    // ── Exponential ─────────────────────────────────────────────────
    //
    // Inverse-CDF: `-ln(u) / lambda`. Mean of the distribution is
    // `1 / lambda`.

    let lambda = 2.0;
    let exp = fill_exponential_f32_gpu(&gpu, len, seed, lambda).unwrap();
    println!("exponential_f32 (lambda={lambda}): {exp:.4?}");

    // ── LogNormal ───────────────────────────────────────────────────

    let (mu, sigma) = (0.0, 1.0);
    let ln = fill_lognormal_f32_gpu(&gpu, len, seed, mu, sigma).unwrap();
    println!("lognormal_f32 (mu={mu}, sigma={sigma}): {ln:.4?}");

    // ── Bernoulli ───────────────────────────────────────────────────
    //
    // Returns u32 (1 with probability p, 0 otherwise). Use the
    // pattern below to map directly to bool / a mask.

    let p = 0.7;
    let bern = fill_bernoulli_u32_gpu(&gpu, len, seed, p).unwrap();
    let mask: Vec<bool> = bern.iter().map(|&v| v == 1).collect();
    println!("bernoulli_u32 (p={p}): {bern:?}");
    println!("  as bool mask           : {mask:?}");

    // ── Poisson ─────────────────────────────────────────────────────
    //
    // Knuth's algorithm caps at 64 inner iterations. v0.1 supports
    // lambda up to ~30 with effectively-zero truncation probability.

    let pois_lambda = 4.0;
    let pois = fill_poisson_u32_gpu(&gpu, len, seed, pois_lambda).unwrap();
    println!("poisson_u32 (lambda={pois_lambda}): {pois:?}");

    println!();
    println!("== same seed → same bytes ==");
    let again = fill_uniform_u32_gpu(&gpu, len, seed).unwrap();
    assert_eq!(u32_out, again, "determinism check");
    println!("re-ran uniform_u32 — bit-identical, as expected.");
}
