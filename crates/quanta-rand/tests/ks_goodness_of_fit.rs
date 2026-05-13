//! Kolmogorov–Smirnov goodness-of-fit checks for each fill_*
//! distribution.
//!
//! For each distribution we draw a large sample on the GPU (CPU
//! backend in this run; native GPUs when available), compute the
//! empirical CDF F_n, and compare against the analytic CDF F:
//!
//!   D_n = sup_x |F_n(x) - F(x)|
//!
//! The asymptotic critical value for K_alpha at confidence 1 - alpha
//! is approximately c / sqrt(n) where c depends on alpha. We use
//! n = 50_000 and a generous c = 1.95 (~0.001 confidence) — failures
//! at this threshold flag a real distributional issue, not just
//! sampling noise.
//!
//! Runs only with the `gpu` feature.

#![cfg(feature = "gpu")]

use quanta_rand::{
    fill_bernoulli_u32_gpu, fill_exponential_f32_gpu, fill_exponential_f64_gpu,
    fill_lognormal_f32_gpu, fill_lognormal_f64_gpu, fill_normal_f32_gpu, fill_normal_f64_gpu,
    fill_poisson_u32_gpu, fill_uniform_f32_gpu, fill_uniform_f64_gpu,
};

const SEED: u64 = 0x1234_5678_9ABC_DEF0u64;
const SAMPLE_N: usize = 50_000;
const KS_CRITICAL_C: f64 = 1.95; // ~ 0.001 confidence threshold.

fn ks_threshold(n: usize) -> f64 {
    KS_CRITICAL_C / (n as f64).sqrt()
}

/// Generic K-S test against a continuous CDF `cdf`. Takes f32
/// samples, returns D_n. Sorts in place.
fn ks_statistic_continuous(samples: &mut [f32], cdf: impl Fn(f32) -> f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len() as f64;
    let mut d_max: f64 = 0.0;
    for (i, &x) in samples.iter().enumerate() {
        let fx = cdf(x);
        let fn_below = i as f64 / n; // empirical CDF just below x
        let fn_at = (i as f64 + 1.0) / n; // just at x
        let d1 = (fx - fn_below).abs();
        let d2 = (fn_at - fx).abs();
        if d1 > d_max {
            d_max = d1;
        }
        if d2 > d_max {
            d_max = d2;
        }
    }
    d_max
}

/// f64 variant of `ks_statistic_continuous`.
fn ks_statistic_continuous_f64(samples: &mut [f64], cdf: impl Fn(f64) -> f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len() as f64;
    let mut d_max: f64 = 0.0;
    for (i, &x) in samples.iter().enumerate() {
        let fx = cdf(x);
        let fn_below = i as f64 / n;
        let fn_at = (i as f64 + 1.0) / n;
        let d1 = (fx - fn_below).abs();
        let d2 = (fn_at - fx).abs();
        if d1 > d_max {
            d_max = d1;
        }
        if d2 > d_max {
            d_max = d2;
        }
    }
    d_max
}

// ── Standard CDFs ───────────────────────────────────────────────────

fn cdf_uniform_01(x: f32) -> f64 {
    let x = x as f64;
    x.clamp(0.0, 1.0)
}

fn cdf_normal_standard(x: f32) -> f64 {
    let x = x as f64;
    // Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn cdf_uniform_01_f64(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

fn cdf_normal_standard_f64(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn cdf_exponential_f64(x: f64, lambda: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        1.0 - (-lambda * x).exp()
    }
}

fn cdf_lognormal_f64(x: f64, mu: f64, sigma: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let z = (x.ln() - mu) / sigma;
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Abramowitz & Stegun rational approximation for erf, accurate to
/// ~1.5e-7 — comfortably below our K-S threshold for n=50_000.
fn erf(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn cdf_exponential(x: f32, lambda: f32) -> f64 {
    let x = x as f64;
    let l = lambda as f64;
    if x < 0.0 { 0.0 } else { 1.0 - (-l * x).exp() }
}

fn cdf_lognormal(x: f32, mu: f32, sigma: f32) -> f64 {
    let x = x as f64;
    if x <= 0.0 {
        return 0.0;
    }
    // F(x) = Φ((ln x - mu) / sigma)
    let z = (x.ln() - mu as f64) / sigma as f64;
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

// ── Tests ───────────────────────────────────────────────────────────

#[test]
fn ks_uniform_f32() {
    let gpu = quanta::init_cpu();
    let mut samples = fill_uniform_f32_gpu(&gpu, SAMPLE_N, SEED).expect("dispatch");
    let d = ks_statistic_continuous(&mut samples, cdf_uniform_01);
    let threshold = ks_threshold(SAMPLE_N);
    assert!(
        d < threshold,
        "uniform K-S failed: D = {d:.6}, threshold = {threshold:.6}"
    );
}

#[test]
fn ks_normal_f32() {
    let gpu = quanta::init_cpu();
    let mut samples = fill_normal_f32_gpu(&gpu, SAMPLE_N, SEED).expect("dispatch");
    let d = ks_statistic_continuous(&mut samples, cdf_normal_standard);
    let threshold = ks_threshold(SAMPLE_N);
    assert!(
        d < threshold,
        "normal K-S failed: D = {d:.6}, threshold = {threshold:.6}"
    );
}

#[test]
fn ks_exponential_f32() {
    let gpu = quanta::init_cpu();
    let lambda: f32 = 1.5;
    let mut samples = fill_exponential_f32_gpu(&gpu, SAMPLE_N, SEED, lambda).expect("dispatch");
    let d = ks_statistic_continuous(&mut samples, |x| cdf_exponential(x, lambda));
    let threshold = ks_threshold(SAMPLE_N);
    assert!(
        d < threshold,
        "exponential K-S failed: D = {d:.6}, threshold = {threshold:.6}"
    );
}

#[test]
fn ks_lognormal_f32() {
    let gpu = quanta::init_cpu();
    let mu: f32 = 0.0;
    let sigma: f32 = 0.5;
    let mut samples = fill_lognormal_f32_gpu(&gpu, SAMPLE_N, SEED, mu, sigma).expect("dispatch");
    let d = ks_statistic_continuous(&mut samples, |x| cdf_lognormal(x, mu, sigma));
    let threshold = ks_threshold(SAMPLE_N);
    assert!(
        d < threshold,
        "lognormal K-S failed: D = {d:.6}, threshold = {threshold:.6}"
    );
}

#[test]
fn chi_square_bernoulli() {
    // K-S doesn't apply to discrete distributions. Use a simple
    // proportion test instead — sample proportion within 3 standard
    // errors of p.
    let gpu = quanta::init_cpu();
    let p: f32 = 0.4;
    let out = fill_bernoulli_u32_gpu(&gpu, SAMPLE_N, SEED, p).expect("dispatch");
    let ones = out.iter().filter(|&&v| v == 1).count();
    let p_hat = ones as f64 / SAMPLE_N as f64;
    let stderr = (p as f64 * (1.0 - p as f64) / SAMPLE_N as f64).sqrt();
    assert!(
        (p_hat - p as f64).abs() < 3.0 * stderr,
        "Bernoulli proportion test failed: p_hat = {p_hat:.5}, p = {p}, stderr = {stderr:.5}",
    );
}

#[test]
fn chi_square_poisson() {
    // Discrete: compute observed vs expected counts per k and check
    // chi-square. Probability mass `P(k) = exp(-lam) * lam^k / k!`.
    let gpu = quanta::init_cpu();
    let lambda: f32 = 3.0;
    let out = fill_poisson_u32_gpu(&gpu, SAMPLE_N, SEED, lambda).expect("dispatch");

    // Histogram up to k = 20 (Poisson(3) tail is negligible past 15).
    let max_k = 20usize;
    let mut observed = vec![0u32; max_k + 1];
    let mut over = 0u32;
    for &v in &out {
        if (v as usize) <= max_k {
            observed[v as usize] += 1;
        } else {
            over += 1;
        }
    }
    let _ = over; // tail count, ignored for the test

    // Expected counts.
    let lam = lambda as f64;
    let mut chi2: f64 = 0.0;
    let mut p_k = (-lam).exp();
    for k in 0..=max_k {
        let expected = SAMPLE_N as f64 * p_k;
        if expected >= 5.0 {
            let diff = observed[k] as f64 - expected;
            chi2 += diff * diff / expected;
        }
        p_k *= lam / (k as f64 + 1.0);
    }
    // Degrees of freedom ≈ number of bins counted (~10), critical
    // value at p=0.001 ≈ 30. Use a generous threshold; main goal
    // is to catch egregious distributional bugs.
    assert!(
        chi2 < 40.0,
        "Poisson chi-square failed: chi2 = {chi2:.3}, threshold = 40"
    );
}

// ── f64 distributions ───────────────────────────────────────────────

#[test]
fn ks_uniform_f64() {
    let gpu = quanta::init_cpu();
    let mut samples = fill_uniform_f64_gpu(&gpu, SAMPLE_N, SEED).expect("dispatch");
    let d = ks_statistic_continuous_f64(&mut samples, cdf_uniform_01_f64);
    let threshold = ks_threshold(SAMPLE_N);
    assert!(
        d < threshold,
        "uniform_f64 K-S failed: D = {d:.6}, threshold = {threshold:.6}"
    );
}

#[test]
fn ks_normal_f64() {
    let gpu = quanta::init_cpu();
    let mut samples = fill_normal_f64_gpu(&gpu, SAMPLE_N, SEED).expect("dispatch");
    let d = ks_statistic_continuous_f64(&mut samples, cdf_normal_standard_f64);
    let threshold = ks_threshold(SAMPLE_N);
    assert!(
        d < threshold,
        "normal_f64 K-S failed: D = {d:.6}, threshold = {threshold:.6}"
    );
}

#[test]
fn ks_exponential_f64() {
    let gpu = quanta::init_cpu();
    let lambda: f64 = 1.5;
    let mut samples = fill_exponential_f64_gpu(&gpu, SAMPLE_N, SEED, lambda).expect("dispatch");
    let d = ks_statistic_continuous_f64(&mut samples, |x| cdf_exponential_f64(x, lambda));
    let threshold = ks_threshold(SAMPLE_N);
    assert!(
        d < threshold,
        "exp_f64 K-S failed: D = {d:.6}, threshold = {threshold:.6}"
    );
}

#[test]
fn ks_lognormal_f64() {
    let gpu = quanta::init_cpu();
    let mu: f64 = 0.0;
    let sigma: f64 = 0.5;
    let mut samples = fill_lognormal_f64_gpu(&gpu, SAMPLE_N, SEED, mu, sigma).expect("dispatch");
    let d = ks_statistic_continuous_f64(&mut samples, |x| cdf_lognormal_f64(x, mu, sigma));
    let threshold = ks_threshold(SAMPLE_N);
    assert!(
        d < threshold,
        "lognormal_f64 K-S failed: D = {d:.6}, threshold = {threshold:.6}"
    );
}
