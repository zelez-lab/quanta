//! Bit-exact end-to-end correctness for the `fill_uniform_*`
//! kernels: GPU output (CPU backend in this run; native GPUs
//! when available) must match the host-side Philox4×32-10
//! reference for every quark, every variant, every seed.
//!
//! Validates the full chain: kernel macro expansion, the
//! `#[quanta::device]` source-splice, WASM-route lowering, CPU
//! eval (push-const path + i64.mul path), and the host-side
//! `u32`/`u64`/`f32`/`f64` conversion contract.
//!
//! Runs only with the `gpu` feature.

#![cfg(feature = "gpu")]

use quanta_rand::philox4x32::philox4x32_10_first_u32;
use quanta_rand::uniform::{
    u32_to_open_unit_f32, u32_to_unit_f32, u64_to_open_unit_f64, u64_to_unit_f64,
};
use quanta_rand::{
    fill_bernoulli_u32_gpu, fill_exponential_f32_gpu, fill_exponential_f64_gpu,
    fill_lognormal_f32_gpu, fill_lognormal_f64_gpu, fill_normal_f32_gpu, fill_normal_f64_gpu,
    fill_poisson_u32_gpu, fill_uniform_f32_gpu, fill_uniform_f64_gpu, fill_uniform_u32_gpu,
    fill_uniform_u64_gpu,
};

const SEED: u64 = 0xCAFE_BABE_DEAD_BEEFu64;

fn seed_words() -> (u32, u32) {
    (SEED as u32, (SEED >> 32) as u32)
}

#[test]
fn fill_uniform_u32_matches_host_philox() {
    let gpu = quanta::init_cpu();
    let len = 64;
    let out = fill_uniform_u32_gpu(&gpu, len, SEED).expect("dispatch");

    let (lo, hi) = seed_words();
    let expected: Vec<u32> = (0..len as u32)
        .map(|id| philox4x32_10_first_u32(id, 0, 0, 0, lo, hi))
        .collect();

    assert_eq!(out, expected, "u32 fill must match host Philox4x32-10");
}

#[test]
fn fill_uniform_u64_matches_host_philox() {
    let gpu = quanta::init_cpu();
    let len = 64;
    let out = fill_uniform_u64_gpu(&gpu, len, SEED).expect("dispatch");

    let (lo, hi) = seed_words();
    let expected: Vec<u64> = (0..len as u32)
        .map(|id| {
            let h = philox4x32_10_first_u32(id, 0, 0, 0, lo, hi);
            let l = philox4x32_10_first_u32(id, 1, 0, 0, lo, hi);
            ((h as u64) << 32) | (l as u64)
        })
        .collect();

    assert_eq!(out, expected, "u64 fill must match host Philox packing");
}

#[test]
fn fill_uniform_f32_matches_host_philox() {
    let gpu = quanta::init_cpu();
    let len = 64;
    let out = fill_uniform_f32_gpu(&gpu, len, SEED).expect("dispatch");

    let (lo, hi) = seed_words();
    let expected: Vec<f32> = (0..len as u32)
        .map(|id| u32_to_unit_f32(philox4x32_10_first_u32(id, 0, 0, 0, lo, hi)))
        .collect();

    assert_eq!(
        out, expected,
        "f32 fill must match u32→unit-f32 of host Philox"
    );
    // Every value must be in [0, 1).
    for &v in &out {
        assert!((0.0..1.0).contains(&v), "f32 out of [0, 1): {v}");
    }
}

#[test]
fn fill_uniform_f64_matches_host_philox() {
    let gpu = quanta::init_cpu();
    let len = 64;
    let out = fill_uniform_f64_gpu(&gpu, len, SEED).expect("dispatch");

    let (lo, hi) = seed_words();
    let expected: Vec<f64> = (0..len as u32)
        .map(|id| {
            let h = philox4x32_10_first_u32(id, 0, 0, 0, lo, hi);
            let l = philox4x32_10_first_u32(id, 1, 0, 0, lo, hi);
            let packed = ((h as u64) << 32) | (l as u64);
            u64_to_unit_f64(packed)
        })
        .collect();

    assert_eq!(
        out, expected,
        "f64 fill must match u64→unit-f64 of host Philox"
    );
    for &v in &out {
        assert!((0.0..1.0).contains(&v), "f64 out of [0, 1): {v}");
    }
}

#[test]
fn distinct_seeds_produce_distinct_streams() {
    let gpu = quanta::init_cpu();
    let a = fill_uniform_u32_gpu(&gpu, 256, 1).expect("seed 1");
    let b = fill_uniform_u32_gpu(&gpu, 256, 2).expect("seed 2");
    // Two unrelated Philox streams should not coincide on every
    // element — at most a handful of incidental matches over 256
    // u32 samples.
    let matches = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    assert!(
        matches < 4,
        "seeds 1 and 2 produced suspiciously many matching outputs: {matches} / 256",
    );
}

#[test]
fn empty_fill_returns_empty() {
    let gpu = quanta::init_cpu();
    let out = fill_uniform_u32_gpu(&gpu, 0, SEED).expect("zero-length");
    assert!(out.is_empty());
}

/// Host-side reference for `fill_normal_f32` — mirrors the kernel
/// body byte-for-byte. Each quark id produces a pair (n1, n2)
/// from two Philox draws.
fn host_normal_pair(id: u32, lo: u32, hi: u32) -> (f32, f32) {
    let r0 = philox4x32_10_first_u32(id, 0, 0, 0, lo, hi);
    let r1 = philox4x32_10_first_u32(id, 1, 0, 0, lo, hi);
    let u1 = u32_to_open_unit_f32(r0);
    let u2 = u32_to_open_unit_f32(r1);
    let ln_u1 = u1.ln();
    let r = (-2.0f32 * ln_u1).sqrt();
    let two_pi = 6.2831_8530_7179_586f32;
    let theta = two_pi * u2;
    (r * theta.cos(), r * theta.sin())
}

#[test]
fn fill_normal_f32_matches_host_box_muller() {
    let gpu = quanta::init_cpu();
    let len = 64;
    let out = fill_normal_f32_gpu(&gpu, len, SEED).expect("dispatch");

    let (lo, hi) = seed_words();
    let mut expected = Vec::with_capacity(len);
    for id in 0..(len.div_ceil(2)) as u32 {
        let (n1, n2) = host_normal_pair(id, lo, hi);
        expected.push(n1);
        if expected.len() < len {
            expected.push(n2);
        }
    }
    assert_eq!(out.len(), expected.len());
    for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "normal[{i}] bit-exact mismatch: got {got} (0x{:08x}) want {want} (0x{:08x})",
            got.to_bits(),
            want.to_bits()
        );
    }
}

#[test]
fn fill_normal_f32_distribution_is_approximately_standard() {
    // Light statistical sanity (not a proper K-S test — that's M10).
    let gpu = quanta::init_cpu();
    let n = 10_000;
    let out = fill_normal_f32_gpu(&gpu, n, SEED).expect("dispatch");

    // Mean should be ~0, variance ~1.
    let mean: f32 = out.iter().sum::<f32>() / (n as f32);
    let var: f32 = out.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (n as f32);
    assert!(
        mean.abs() < 0.05,
        "sample mean {mean} too far from 0 (expected ~0 for N(0, 1) at n={n})"
    );
    assert!(
        (var - 1.0).abs() < 0.1,
        "sample variance {var} too far from 1 (expected ~1 for N(0, 1) at n={n})"
    );

    // ~99.7% should fall in [-3, 3] (three-sigma).
    let within_3sigma = out.iter().filter(|&&x| x.abs() < 3.0).count();
    let frac = within_3sigma as f32 / n as f32;
    assert!(
        frac > 0.99,
        "{:.3}% within ±3σ — expected ~99.7%",
        frac * 100.0
    );
}

#[test]
fn fill_normal_f32_handles_odd_length() {
    // Box-Muller produces two normals per quark; odd `len` requires
    // the host to dispatch ceil(len/2) quarks and trim.
    let gpu = quanta::init_cpu();
    let out = fill_normal_f32_gpu(&gpu, 5, SEED).expect("dispatch");
    assert_eq!(out.len(), 5);
    // First 4 must match a 4-output call (same quarks 0,1 → pairs).
    let out_even = fill_normal_f32_gpu(&gpu, 4, SEED).expect("dispatch");
    assert_eq!(&out[..4], &out_even[..]);
}

// ── M8 — Exponential / LogNormal / Bernoulli ────────────────────────

#[test]
fn fill_exponential_f32_matches_host_inverse_cdf() {
    let gpu = quanta::init_cpu();
    let len = 64;
    let lambda: f32 = 2.0;
    let out = fill_exponential_f32_gpu(&gpu, len, SEED, lambda).expect("dispatch");

    let (lo, hi) = seed_words();
    let expected: Vec<f32> = (0..len as u32)
        .map(|id| {
            let r = philox4x32_10_first_u32(id, 0, 0, 0, lo, hi);
            let u = u32_to_open_unit_f32(r);
            -u.ln() / lambda
        })
        .collect();
    for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "exp[{i}] bit-exact mismatch: got {got} want {want}"
        );
    }
}

#[test]
fn fill_exponential_f32_distribution_is_approximately_exponential() {
    let gpu = quanta::init_cpu();
    let n = 10_000;
    let lambda: f32 = 1.5;
    let out = fill_exponential_f32_gpu(&gpu, n, SEED, lambda).expect("dispatch");
    // E[X] = 1/lambda, Var[X] = 1/lambda^2.
    let mean: f32 = out.iter().sum::<f32>() / n as f32;
    let expected_mean = 1.0 / lambda;
    assert!(
        (mean - expected_mean).abs() < 0.05,
        "sample mean {mean} too far from {expected_mean} for Exp({lambda})"
    );
    // All draws non-negative.
    for &v in &out {
        assert!(v >= 0.0, "negative exponential sample: {v}");
    }
}

#[test]
fn fill_lognormal_f32_bit_exact_pair_with_host() {
    let gpu = quanta::init_cpu();
    let len = 16;
    let mu: f32 = 0.5;
    let sigma: f32 = 1.0;
    let out = fill_lognormal_f32_gpu(&gpu, len, SEED, mu, sigma).expect("dispatch");

    let (lo, hi) = seed_words();
    let mut expected = Vec::with_capacity(len);
    for id in 0..(len.div_ceil(2)) as u32 {
        let r0 = philox4x32_10_first_u32(id, 0, 0, 0, lo, hi);
        let r1 = philox4x32_10_first_u32(id, 1, 0, 0, lo, hi);
        let u1 = u32_to_open_unit_f32(r0);
        let u2 = u32_to_open_unit_f32(r1);
        let r = (-2.0f32 * u1.ln()).sqrt();
        let theta = 6.2831_8530_7179_586f32 * u2;
        let n1 = r * theta.cos();
        let n2 = r * theta.sin();
        expected.push((mu + sigma * n1).exp());
        if expected.len() < len {
            expected.push((mu + sigma * n2).exp());
        }
    }
    for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "lognormal[{i}] mismatch: got {got} want {want}"
        );
    }
}

#[test]
fn fill_lognormal_f32_is_positive() {
    let gpu = quanta::init_cpu();
    let out = fill_lognormal_f32_gpu(&gpu, 1024, SEED, 0.0, 1.0).expect("dispatch");
    for &v in &out {
        assert!(v > 0.0, "non-positive lognormal sample: {v}");
    }
}

#[test]
fn fill_bernoulli_u32_matches_host() {
    let gpu = quanta::init_cpu();
    let len = 64;
    let p: f32 = 0.3;
    let out = fill_bernoulli_u32_gpu(&gpu, len, SEED, p).expect("dispatch");

    let (lo, hi) = seed_words();
    let expected: Vec<u32> = (0..len as u32)
        .map(|id| {
            let r = philox4x32_10_first_u32(id, 0, 0, 0, lo, hi);
            let u = u32_to_unit_f32(r);
            if u < p { 1 } else { 0 }
        })
        .collect();
    assert_eq!(out, expected);
}

#[test]
fn fill_bernoulli_u32_proportion_is_close_to_p() {
    let gpu = quanta::init_cpu();
    let n = 10_000;
    let p: f32 = 0.2;
    let out = fill_bernoulli_u32_gpu(&gpu, n, SEED, p).expect("dispatch");
    let ones = out.iter().filter(|&&v| v == 1).count();
    let frac = ones as f32 / n as f32;
    // Standard error ~ sqrt(p(1-p)/n) ≈ 0.004 at n=10k, p=0.2.
    // Allow 3 standard errors of slack.
    assert!(
        (frac - p).abs() < 0.02,
        "Bernoulli proportion {frac} too far from p={p}"
    );
    // Edge cases: p=0 → all zeros, p=1 → all ones.
    let out0 = fill_bernoulli_u32_gpu(&gpu, 32, SEED, 0.0).expect("p=0");
    assert!(out0.iter().all(|&v| v == 0));
    let out1 = fill_bernoulli_u32_gpu(&gpu, 32, SEED, 1.0).expect("p=1");
    assert!(out1.iter().all(|&v| v == 1));
}

// ── M9 — Poisson (Knuth) ────────────────────────────────────────────

/// Host-side reference for `fill_poisson_u32`. Mirrors the kernel
/// body exactly: 64-iter cap, same Philox draws, same uniform
/// conversion, same comparison.
fn host_poisson_knuth(id: u32, lo: u32, hi: u32, lambda: f32) -> u32 {
    let l = (-lambda).exp();
    let mut p: f32 = 1.0;
    let mut k: u32 = 0;
    for iter in 0..64u32 {
        // Per-iteration Philox draw with counter=(id, iter, 0, 0)
        // — matches the kernel body exactly.
        let r = philox4x32_10_first_u32(id, iter, 0, 0, lo, hi);
        let u = u32_to_unit_f32(r);
        p *= u;
        if p <= l {
            return k;
        }
        k += 1;
    }
    k
}

#[test]
fn fill_poisson_u32_matches_host_knuth() {
    let gpu = quanta::init_cpu();
    let len = 64;
    let lambda: f32 = 4.0;
    let out = fill_poisson_u32_gpu(&gpu, len, SEED, lambda).expect("dispatch");

    let (lo, hi) = seed_words();
    let expected: Vec<u32> = (0..len as u32)
        .map(|id| host_poisson_knuth(id, lo, hi, lambda))
        .collect();
    assert_eq!(out, expected);
}

#[test]
fn fill_poisson_u32_mean_close_to_lambda() {
    let gpu = quanta::init_cpu();
    let n = 10_000;
    let lambda: f32 = 5.0;
    let out = fill_poisson_u32_gpu(&gpu, n, SEED, lambda).expect("dispatch");
    let sum: u64 = out.iter().map(|&v| v as u64).sum();
    let mean = sum as f32 / n as f32;
    // Std error ~ sqrt(lambda/n) ≈ 0.022 at n=10k, lambda=5.
    assert!(
        (mean - lambda).abs() < 0.1,
        "Poisson({lambda}) sample mean {mean} too far from {lambda}",
    );
    // All values non-negative; with lambda=5, exceeding 64 is
    // essentially impossible (we cap at 64 anyway).
    for &v in &out {
        assert!(v <= 64, "Poisson sample {v} exceeded the iteration cap");
    }
}

#[test]
fn fill_poisson_u32_lambda_zero_returns_zeros() {
    let gpu = quanta::init_cpu();
    let out = fill_poisson_u32_gpu(&gpu, 64, SEED, 0.0).expect("lambda=0");
    // exp(-0) = 1, so the very first uniform u<=1 always satisfies
    // p*u <= 1 — k stays at 0 for every quark.
    for &v in &out {
        assert_eq!(v, 0);
    }
}

// ── f64 distributions ────────────────────────────────────────────────

fn host_normal_pair_f64(id: u32, lo: u32, hi: u32) -> (f64, f64) {
    let r0a = philox4x32_10_first_u32(id, 0, 0, 0, lo, hi);
    let r0b = philox4x32_10_first_u32(id, 1, 0, 0, lo, hi);
    let r1a = philox4x32_10_first_u32(id, 2, 0, 0, lo, hi);
    let r1b = philox4x32_10_first_u32(id, 3, 0, 0, lo, hi);
    let packed0 = ((r0a as u64) << 32) | (r0b as u64);
    let packed1 = ((r1a as u64) << 32) | (r1b as u64);
    let u1 = u64_to_open_unit_f64(packed0);
    let u2 = u64_to_open_unit_f64(packed1);
    let r = (-2.0f64 * u1.ln()).sqrt();
    let two_pi = 6.283_185_307_179_586f64;
    let theta = two_pi * u2;
    (r * theta.cos(), r * theta.sin())
}

#[test]
fn fill_normal_f64_matches_host_box_muller() {
    let gpu = quanta::init_cpu();
    let len = 64;
    let out = fill_normal_f64_gpu(&gpu, len, SEED).expect("dispatch");

    let (lo, hi) = seed_words();
    let mut expected = Vec::with_capacity(len);
    for id in 0..(len.div_ceil(2)) as u32 {
        let (n1, n2) = host_normal_pair_f64(id, lo, hi);
        expected.push(n1);
        if expected.len() < len {
            expected.push(n2);
        }
    }
    for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "normal_f64[{i}] bit-exact mismatch: got {got} want {want}"
        );
    }
}

#[test]
fn fill_normal_f64_distribution_is_approximately_standard() {
    let gpu = quanta::init_cpu();
    let n = 10_000;
    let out = fill_normal_f64_gpu(&gpu, n, SEED).expect("dispatch");
    let mean: f64 = out.iter().sum::<f64>() / (n as f64);
    let var: f64 = out.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n as f64);
    assert!(mean.abs() < 0.05, "f64 sample mean {mean} too far from 0");
    assert!(
        (var - 1.0).abs() < 0.1,
        "f64 sample variance {var} too far from 1"
    );
}

#[test]
fn fill_exponential_f64_matches_host_inverse_cdf() {
    let gpu = quanta::init_cpu();
    let len = 64;
    let lambda: f64 = 2.0;
    let out = fill_exponential_f64_gpu(&gpu, len, SEED, lambda).expect("dispatch");

    let (lo, hi) = seed_words();
    let expected: Vec<f64> = (0..len as u32)
        .map(|id| {
            let ra = philox4x32_10_first_u32(id, 0, 0, 0, lo, hi);
            let rb = philox4x32_10_first_u32(id, 1, 0, 0, lo, hi);
            let packed = ((ra as u64) << 32) | (rb as u64);
            let u = u64_to_open_unit_f64(packed);
            -u.ln() / lambda
        })
        .collect();
    for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "exp_f64[{i}] bit-exact mismatch"
        );
    }
    for &v in &out {
        assert!(v >= 0.0, "negative f64 exponential sample: {v}");
    }
}

#[test]
fn fill_lognormal_f64_is_positive() {
    let gpu = quanta::init_cpu();
    let out = fill_lognormal_f64_gpu(&gpu, 1024, SEED, 0.0, 1.0).expect("dispatch");
    for &v in &out {
        assert!(v > 0.0, "non-positive f64 lognormal sample: {v}");
    }
}
