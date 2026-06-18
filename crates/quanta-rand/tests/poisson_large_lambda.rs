//! Statistical sanity for the large-lambda Poisson kernel (PTRD).
//!
//! Bit-exact host parity isn't feasible because PTRD's acceptance
//! test is sensitive to floating-point evaluation order (different
//! GPU codegen paths can reorder `log` operands), so this file
//! instead validates the **distributional** properties:
//!
//! 1. Sample mean ≈ λ (Poisson E[X] = λ).
//! 2. Sample variance ≈ λ (Poisson Var[X] = λ).
//! 3. Tail behavior: no draws clamped to 0 (which would indicate the
//!    early `k < 0` early-reject was firing too often).
//!
//! Tolerances are set per λ using the CLT — for N samples, the
//! sample mean has std-dev √(λ/N). We use 6σ bounds, so the
//! per-test false-failure probability is ~1e-9.

#![cfg(feature = "gpu")]

use quanta_rand::gpu_kernel::{
    fill_poisson_u32_auto_gpu, fill_poisson_u32_gpu, fill_poisson_u32_large_gpu,
};

const SEED: u64 = 0xCAFE_BABE_DEAD_BEEFu64;

fn sample_stats(samples: &[u32]) -> (f64, f64) {
    let n = samples.len() as f64;
    let sum: f64 = samples.iter().map(|&x| x as f64).sum();
    let mean = sum / n;
    let var = samples
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (mean, var)
}

/// 6σ tolerance for the sample mean of a Poisson(λ) over N samples.
fn mean_tol(lambda: f64, n: usize) -> f64 {
    6.0 * (lambda / n as f64).sqrt()
}

/// Generous tolerance for the sample variance. For a Poisson, the
/// sample variance has approximate std-dev √(2λ²/N) — we use 8σ.
fn var_tol(lambda: f64, n: usize) -> f64 {
    8.0 * (2.0 * lambda * lambda / n as f64).sqrt()
}

// Activated 2026-06-12: the natural nested-shape kernel only pays
// the Stirling chain until acceptance (~1.2 iterations expected), so
// the full N=32k λ-sweep runs in ~75 s debug / ~4 s release on the
// parallel CPU dispatch — fast enough for the default quanta-rand
// suite. (The flat-shape kernel needed ~11 min release, which is why
// these were `#[ignore]`d historically.)
#[test]
fn fill_poisson_u32_large_mean_variance_lambda_10() {
    let gpu = quanta::init_cpu();
    let lambda: f32 = 10.0;
    let n = 32_768;
    let samples = fill_poisson_u32_large_gpu(&gpu, n, SEED, lambda).expect("dispatch");

    let (mean, var) = sample_stats(&samples);
    let exp = lambda as f64;
    let m_tol = mean_tol(exp, n);
    let v_tol = var_tol(exp, n);

    assert!(
        (mean - exp).abs() < m_tol,
        "lambda=10: sample mean {mean} vs expected {exp}, tol {m_tol}"
    );
    assert!(
        (var - exp).abs() < v_tol,
        "lambda=10: sample variance {var} vs expected {exp}, tol {v_tol}"
    );
}

#[test]
fn fill_poisson_u32_large_mean_variance_lambda_50() {
    let gpu = quanta::init_cpu();
    let lambda: f32 = 50.0;
    let n = 32_768;
    let samples = fill_poisson_u32_large_gpu(&gpu, n, SEED, lambda).expect("dispatch");

    let (mean, var) = sample_stats(&samples);
    let exp = lambda as f64;
    let m_tol = mean_tol(exp, n);
    let v_tol = var_tol(exp, n);

    assert!(
        (mean - exp).abs() < m_tol,
        "lambda=50: sample mean {mean} vs expected {exp}, tol {m_tol}"
    );
    assert!(
        (var - exp).abs() < v_tol,
        "lambda=50: sample variance {var} vs expected {exp}, tol {v_tol}"
    );
}

#[test]
fn fill_poisson_u32_large_mean_variance_lambda_200() {
    let gpu = quanta::init_cpu();
    let lambda: f32 = 200.0;
    let n = 16_384;
    let samples = fill_poisson_u32_large_gpu(&gpu, n, SEED, lambda).expect("dispatch");

    let (mean, var) = sample_stats(&samples);
    let exp = lambda as f64;
    let m_tol = mean_tol(exp, n);
    let v_tol = var_tol(exp, n);

    assert!(
        (mean - exp).abs() < m_tol,
        "lambda=200: sample mean {mean} vs expected {exp}, tol {m_tol}"
    );
    assert!(
        (var - exp).abs() < v_tol,
        "lambda=200: sample variance {var} vs expected {exp}, tol {v_tol}"
    );
}

#[test]
fn fill_poisson_u32_large_no_default_zeros() {
    // PTRD's safety-cap (32 rejections per quark) defaults the
    // output to 0. If the algorithm misfires (e.g. wrong constants,
    // bad acceptance test), many quarks would hit the cap and
    // output 0. For Poisson(50) the true Pr(X = 0) is e^-50 ≈ 2e-22.
    // So we should see effectively zero zeros across thousands of
    // samples — any positive fraction signals algorithm breakage.
    let gpu = quanta::init_cpu();
    let n = 8192;
    let samples = fill_poisson_u32_large_gpu(&gpu, n, SEED, 50.0f32).expect("dispatch");
    let zero_count = samples.iter().filter(|&&x| x == 0).count();
    assert_eq!(
        zero_count, 0,
        "lambda=50 should have ~0 zero draws (Pr(X=0) = e^-50); got {zero_count}/{n}"
    );
}

#[test]
fn fill_poisson_u32_auto_uses_knuth_below_threshold() {
    // λ = 5 should dispatch to Knuth and match the existing
    // small-lambda code path bit-for-bit.
    let gpu = quanta::init_cpu();
    let lambda: f32 = 5.0;
    let n = 1024;
    let knuth = fill_poisson_u32_gpu(&gpu, n, SEED, lambda).expect("dispatch knuth");
    let auto = fill_poisson_u32_auto_gpu(&gpu, n, SEED, lambda).expect("dispatch auto");
    assert_eq!(
        knuth, auto,
        "auto-dispatch at λ=5 must match the Knuth kernel bit-for-bit"
    );
}

// Bit-exact auto-vs-PTRD comparison at N=1024 — verifies the
// auto-dispatch picks the right kernel above the λ ≥ 10 threshold.
#[test]
fn fill_poisson_u32_auto_uses_ptrd_above_threshold() {
    let gpu = quanta::init_cpu();
    let lambda: f32 = 20.0;
    let n = 1024;
    let ptrd = fill_poisson_u32_large_gpu(&gpu, n, SEED, lambda).expect("dispatch ptrd");
    let auto = fill_poisson_u32_auto_gpu(&gpu, n, SEED, lambda).expect("dispatch auto");
    assert_eq!(
        ptrd, auto,
        "auto-dispatch at λ=20 must match the PTRD kernel bit-for-bit"
    );
}
