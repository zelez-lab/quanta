//! Fused LayerNorm / RMSNorm — differential and property tests.
//!
//! Every gradient check runs against BOTH independent references: the
//! composed tape path (`Var::layer_norm` / `Var::rms_norm`, whose per-op
//! VJPs are proven in `Quanta/Autograd`) and a host f64 implementation of
//! the T9210/T9211 three-term formulas. Stability exercises the `√ε` bound
//! (T9213/T9214): huge-magnitude rows and a constant row (variance → 0).

use quanta_array::Array;
use quanta_autograd::Tape;
use quanta_nn::norm::{layer_norm_var, rms_norm_var};

fn gpu() -> quanta::Gpu {
    #[cfg(any(feature = "metal", feature = "vulkan"))]
    {
        quanta::init().expect("a GPU device (metal/vulkan feature is on)")
    }
    #[cfg(not(any(feature = "metal", feature = "vulkan")))]
    {
        quanta::init_cpu()
    }
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Mixed relative/absolute error: relative for O(1)+ quantities, an
/// absolute floor of 1e-4 (via the 0.1 denominator clamp at the 1e-3
/// thresholds) below that. The floor is load-bearing: degenerate cases
/// (C = 1, where x̂ ≡ 0 and dx is analytically exactly zero) leave a
/// ~1-ulp residue in `h − mean(h)` that `rstd ≈ 1/√ε` amplifies to
/// ~2e-5 under Metal's FMA reassociation — a pure-relative metric
/// against a true zero would read that as infinite error.
fn rel_err(a: f32, b: f64) -> f64 {
    let d = (a as f64 - b).abs();
    d / b.abs().max(0.1)
}

fn max_rel_err(got: &[f32], want: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(&a, &b)| rel_err(a, b))
        .fold(0.0, f64::max)
}

/// Host f64 LayerNorm forward + the T9210 backward.
#[allow(clippy::type_complexity)]
fn host_ln(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    g: &[f32],
    n: usize,
    c: usize,
    eps: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut out = vec![0.0f64; n * c];
    let mut dx = vec![0.0f64; n * c];
    let mut dgamma = vec![0.0f64; c];
    let mut dbeta = vec![0.0f64; c];
    for i in 0..n {
        let row = &x[i * c..(i + 1) * c];
        let mu = row.iter().map(|&v| v as f64).sum::<f64>() / c as f64;
        let var = row.iter().map(|&v| (v as f64 - mu).powi(2)).sum::<f64>() / c as f64;
        let rstd = 1.0 / (var + eps).sqrt();
        let xh: Vec<f64> = row.iter().map(|&v| (v as f64 - mu) * rstd).collect();
        let h: Vec<f64> = (0..c)
            .map(|j| g[i * c + j] as f64 * gamma[j] as f64)
            .collect();
        let m1 = h.iter().sum::<f64>() / c as f64;
        let m2 = h.iter().zip(&xh).map(|(a, b)| a * b).sum::<f64>() / c as f64;
        for j in 0..c {
            out[i * c + j] = xh[j] * gamma[j] as f64 + beta[j] as f64;
            dx[i * c + j] = rstd * (h[j] - m1 - xh[j] * m2);
            dgamma[j] += g[i * c + j] as f64 * xh[j];
            dbeta[j] += g[i * c + j] as f64;
        }
    }
    (out, dx, dgamma, dbeta)
}

/// Host f64 RMSNorm forward + the T9211 backward.
#[allow(clippy::type_complexity)]
fn host_rms(
    x: &[f32],
    gamma: &[f32],
    g: &[f32],
    n: usize,
    c: usize,
    eps: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut out = vec![0.0f64; n * c];
    let mut dx = vec![0.0f64; n * c];
    let mut dgamma = vec![0.0f64; c];
    for i in 0..n {
        let row = &x[i * c..(i + 1) * c];
        let ms = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / c as f64;
        let rrms = 1.0 / (ms + eps).sqrt();
        let xh: Vec<f64> = row.iter().map(|&v| v as f64 * rrms).collect();
        let h: Vec<f64> = (0..c)
            .map(|j| g[i * c + j] as f64 * gamma[j] as f64)
            .collect();
        let m2 = h.iter().zip(&xh).map(|(a, b)| a * b).sum::<f64>() / c as f64;
        for j in 0..c {
            out[i * c + j] = xh[j] * gamma[j] as f64;
            dx[i * c + j] = rrms * (h[j] - xh[j] * m2);
            dgamma[j] += g[i * c + j] as f64 * xh[j];
        }
    }
    (out, dx, dgamma)
}

/// Run the fused LN through the tape with a weighted-sum loss (weights `g`)
/// so the upstream gradient is exactly `g`; return (out, dx, dgamma, dbeta).
#[allow(clippy::type_complexity)]
fn fused_ln(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    g: &[f32],
    n: usize,
    c: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let gpu = gpu();
    let tape: Tape<f32> = Tape::new();
    let xv = tape.var(Array::from_slice(&gpu, x, &[n, c]).unwrap());
    let gv = tape.var(Array::from_slice(&gpu, gamma, &[c]).unwrap());
    let bv = tape.var(Array::from_slice(&gpu, beta, &[c]).unwrap());
    let out = layer_norm_var(&tape, &xv, &gv, &bv, eps).unwrap();
    let out_host = out.value().to_vec().unwrap();
    let w = tape.var(Array::from_slice(&gpu, g, &[n, c]).unwrap());
    let loss = out.mul(&w).unwrap().sum().unwrap();
    let dx = loss.grad(&xv).unwrap().to_vec().unwrap();
    let dgamma = loss.grad(&gv).unwrap().to_vec().unwrap();
    let dbeta = loss.grad(&bv).unwrap().to_vec().unwrap();
    (out_host, dx, dgamma, dbeta)
}

/// Same shape for the composed oracle.
#[allow(clippy::type_complexity)]
fn composed_ln(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    g: &[f32],
    n: usize,
    c: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let gpu = gpu();
    let tape: Tape<f32> = Tape::new();
    let xv = tape.var(Array::from_slice(&gpu, x, &[n, c]).unwrap());
    let gv = tape.var(Array::from_slice(&gpu, gamma, &[c]).unwrap());
    let bv = tape.var(Array::from_slice(&gpu, beta, &[c]).unwrap());
    let out = xv.layer_norm(&gv, &bv, eps as f64).unwrap();
    let out_host = out.value().to_vec().unwrap();
    let w = tape.var(Array::from_slice(&gpu, g, &[n, c]).unwrap());
    let loss = out.mul(&w).unwrap().sum().unwrap();
    let dx = loss.grad(&xv).unwrap().to_vec().unwrap();
    let dgamma = loss.grad(&gv).unwrap().to_vec().unwrap();
    let dbeta = loss.grad(&bv).unwrap().to_vec().unwrap();
    (out_host, dx, dgamma, dbeta)
}

#[test]
fn layer_norm_matches_f64_host_and_composed_oracle() {
    for &(n, c, seed) in &[(4usize, 8usize, 1u64), (3, 7, 2), (1, 5, 3), (6, 1, 4)] {
        let x = fill(n * c, seed);
        let gamma = fill(c, seed + 100);
        let beta = fill(c, seed + 200);
        let g = fill(n * c, seed + 300);
        let eps = 1e-5f32;

        let (f_out, f_dx, f_dg, f_db) = fused_ln(&x, &gamma, &beta, &g, n, c, eps);
        let (h_out, h_dx, h_dg, h_db) = host_ln(&x, &gamma, &beta, &g, n, c, eps as f64);
        assert!(
            max_rel_err(&f_out, &h_out) < 1e-4,
            "LN fwd vs f64 ({n}x{c})"
        );
        assert!(max_rel_err(&f_dx, &h_dx) < 1e-3, "LN dx vs f64 ({n}x{c})");
        assert!(
            max_rel_err(&f_dg, &h_dg) < 1e-3,
            "LN dgamma vs f64 ({n}x{c})"
        );
        assert!(
            max_rel_err(&f_db, &h_db) < 1e-3,
            "LN dbeta vs f64 ({n}x{c})"
        );

        let (c_out, c_dx, c_dg, c_db) = composed_ln(&x, &gamma, &beta, &g, n, c, eps);
        let pair = |a: &[f32], b: &[f32]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| rel_err(x, y as f64))
                .fold(0.0, f64::max)
        };
        assert!(pair(&f_out, &c_out) < 1e-4, "LN fwd vs composed ({n}x{c})");
        assert!(pair(&f_dx, &c_dx) < 1e-3, "LN dx vs composed ({n}x{c})");
        assert!(pair(&f_dg, &c_dg) < 1e-3, "LN dgamma vs composed ({n}x{c})");
        assert!(pair(&f_db, &c_db) < 1e-3, "LN dbeta vs composed ({n}x{c})");
    }
}

#[test]
fn rms_norm_matches_f64_host_and_composed_oracle() {
    for &(n, c, seed) in &[(4usize, 8usize, 11u64), (3, 7, 12), (1, 5, 13)] {
        let x = fill(n * c, seed);
        let gamma = fill(c, seed + 100);
        let g = fill(n * c, seed + 300);
        let eps = 1e-5f32;

        let gpu_h = gpu();
        let tape: Tape<f32> = Tape::new();
        let xv = tape.var(Array::from_slice(&gpu_h, &x, &[n, c]).unwrap());
        let gv = tape.var(Array::from_slice(&gpu_h, &gamma, &[c]).unwrap());
        let out = rms_norm_var(&tape, &xv, &gv, eps).unwrap();
        let f_out = out.value().to_vec().unwrap();
        let w = tape.var(Array::from_slice(&gpu_h, &g, &[n, c]).unwrap());
        let loss = out.mul(&w).unwrap().sum().unwrap();
        let f_dx = loss.grad(&xv).unwrap().to_vec().unwrap();
        let f_dg = loss.grad(&gv).unwrap().to_vec().unwrap();

        let (h_out, h_dx, h_dg) = host_rms(&x, &gamma, &g, n, c, eps as f64);
        assert!(
            max_rel_err(&f_out, &h_out) < 1e-4,
            "RMS fwd vs f64 ({n}x{c})"
        );
        assert!(max_rel_err(&f_dx, &h_dx) < 1e-3, "RMS dx vs f64 ({n}x{c})");
        assert!(
            max_rel_err(&f_dg, &h_dg) < 1e-3,
            "RMS dgamma vs f64 ({n}x{c})"
        );

        // Composed oracle.
        let tape2: Tape<f32> = Tape::new();
        let xv2 = tape2.var(Array::from_slice(&gpu_h, &x, &[n, c]).unwrap());
        let gv2 = tape2.var(Array::from_slice(&gpu_h, &gamma, &[c]).unwrap());
        let out2 = xv2.rms_norm(&gv2, eps as f64).unwrap();
        let c_out = out2.value().to_vec().unwrap();
        let w2 = tape2.var(Array::from_slice(&gpu_h, &g, &[n, c]).unwrap());
        let loss2 = out2.mul(&w2).unwrap().sum().unwrap();
        let c_dx = loss2.grad(&xv2).unwrap().to_vec().unwrap();
        let c_dg = loss2.grad(&gv2).unwrap().to_vec().unwrap();
        let pair = |a: &[f32], b: &[f32]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| rel_err(x, y as f64))
                .fold(0.0, f64::max)
        };
        assert!(pair(&f_out, &c_out) < 1e-4, "RMS fwd vs composed");
        assert!(pair(&f_dx, &c_dx) < 1e-3, "RMS dx vs composed");
        assert!(pair(&f_dg, &c_dg) < 1e-3, "RMS dgamma vs composed");
    }
}

#[test]
fn gradcheck_layer_norm_central_difference() {
    let (n, c) = (2usize, 4usize);
    let x = fill(n * c, 21);
    let gamma = fill(c, 22);
    let beta = fill(c, 23);
    let g = fill(n * c, 24);
    let eps = 1e-5f32;
    let (_, dx, _, _) = fused_ln(&x, &gamma, &beta, &g, n, c, eps);

    // Central differences on the weighted-sum loss w.r.t. each x entry.
    let h = 1e-3f32;
    for idx in 0..n * c {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[idx] += h;
        xm[idx] -= h;
        let loss = |xv: &[f32]| -> f64 {
            let (out, _, _, _) = fused_ln(xv, &gamma, &beta, &g, n, c, eps);
            out.iter()
                .zip(&g)
                .map(|(&o, &w)| o as f64 * w as f64)
                .sum::<f64>()
        };
        let num = (loss(&xp) - loss(&xm)) / (2.0 * h as f64);
        let ana = dx[idx] as f64;
        assert!(
            (num - ana).abs() < 2e-2_f64.max(2e-2 * ana.abs()),
            "gradcheck x[{idx}]: numeric {num} vs analytic {ana}"
        );
    }
}

#[test]
fn stability_extreme_rows() {
    // Row 0: huge magnitudes; row 1: constant (variance -> 0, the eps case);
    // row 2: tiny values. T9213 says nothing blows up.
    let (n, c) = (3usize, 6usize);
    let mut x = vec![0.0f32; n * c];
    for j in 0..c {
        x[j] = ((j as f32) - 2.5) * 1.0e3;
        x[c + j] = 7.25; // constant row
        x[2 * c + j] = ((j as f32) - 2.5) * 1.0e-6;
    }
    let gamma = fill(c, 31);
    let beta = fill(c, 32);
    let g = fill(n * c, 33);
    let eps = 1e-5f32;

    let (out, dx, dgamma, dbeta) = fused_ln(&x, &gamma, &beta, &g, n, c, eps);
    for (name, v) in [
        ("out", &out),
        ("dx", &dx),
        ("dgamma", &dgamma),
        ("dbeta", &dbeta),
    ] {
        assert!(
            v.iter().all(|x| x.is_finite()),
            "LN {name} must stay finite on extreme rows"
        );
    }
    // The constant row normalizes to ~zero x-hat, so out ~= beta there.
    let (h_out, ..) = host_ln(&x, &gamma, &beta, &g, n, c, eps as f64);
    assert!(max_rel_err(&out, &h_out) < 1e-3, "extreme rows vs f64 host");
}
