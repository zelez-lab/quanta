//! Bluestein's chirp-z algorithm — arbitrary-N DFT as a power-of-2
//! convolution.
//!
//! [`fft`](crate::fft::fft)/[`ifft`](crate::fft::ifft) route non-power-of-2
//! sizes here. The identity `nk = (n² + k² − (k−n)²) / 2` rewrites the DFT
//!
//! ```text
//! X[k] = Σₙ x[n]·exp(∓2πi·nk/N)
//!      = exp(∓πi·k²/N) · Σₙ [x[n]·exp(∓πi·n²/N)] · exp(±πi·(k−n)²/N)
//! ```
//!
//! — an N-point **linear convolution** of the chirped input
//! `a[n] = x[n]·exp(∓πi·n²/N)` with the chirp kernel `b[m] = exp(±πi·m²/N)`,
//! followed by the output chirp `exp(∓πi·k²/N)` (upper signs forward, lower
//! inverse; the inverse additionally scales by `1/N`). The convolution runs
//! as a circular one at `M = next_pow2(2N−1)` on the existing radix-2
//! [`FftPlan`](crate::FftPlan): forward-FFT both operands (kernel laid out
//! wrap-around, `b[M−m] = b[m]`), pointwise-multiply, inverse-FFT, keep the
//! first N bins. `M ≥ 2N−1` guarantees the circular wrap never aliases the
//! linear result.
//!
//! ## Phase reduction (why `n² mod 2N` and not `n²`)
//!
//! The chirp phase is `θₙ = π·n²/N`. Evaluated naively, `n²` grows to ~10⁶
//! by N = 1000, and `cos`/`sin` of a large argument lose exactly the low-order
//! phase bits that matter — even in f64 the reduction of a large θ against π
//! amplifies the representation error of `π·n²/N` catastrophically. But
//! `exp(iπ·q/N)` has period `q → q + 2N`, so `n²` is reduced **exactly in
//! integer arithmetic** first: `q = n² mod 2N` keeps `θ = π·q/N` inside
//! `[0, 2π)` where f64 trig is accurate to the ulp. All three chirp arrays
//! are computed host-side in f64 with this reduction, then downcast to f32
//! for the device convolution.

use core::f64::consts::PI;

use quanta_core::{Gpu, QuantaError};

use crate::plan::FftPlan;

/// Arbitrary-N DFT (forward or inverse) via the chirp-z convolution.
///
/// Called by `fft`/`ifft` for non-power-of-2 `n ≥ 1`; power-of-2 sizes take
/// the plain radix-2 path instead (it is faster and this module would just
/// wrap it in three extra transforms).
pub(crate) fn transform(
    gpu: &Gpu,
    re: &[f32],
    im: &[f32],
    inverse: bool,
) -> Result<(Vec<f32>, Vec<f32>), QuantaError> {
    let n = re.len();
    debug_assert!(n > 0, "empty input is handled by the caller");
    let m = (2 * n - 1).next_power_of_two();
    // Upper (forward) sign of exp(s·πi·n²/N): s = −1 forward, +1 inverse.
    let s: f64 = if inverse { 1.0 } else { -1.0 };

    // Host chirp table in f64: chirp[j] = exp(s·πi·j²/N), with j² reduced
    // mod 2N in exact integer arithmetic BEFORE the trig (see module doc —
    // this is what keeps the phase accurate at large N). The input and
    // output chirps are `chirp` itself; the convolution kernel is its
    // conjugate.
    let two_n = 2 * n;
    let mut chirp_re = Vec::with_capacity(n);
    let mut chirp_im = Vec::with_capacity(n);
    for j in 0..n {
        let q = (j * j) % two_n; // exact: j < N < 2³², j² fits in usize
        let theta = s * PI * (q as f64) / (n as f64);
        chirp_re.push(theta.cos());
        chirp_im.push(theta.sin());
    }

    // a = x·chirp, zero-padded to M.
    let mut a_re = vec![0.0f32; m];
    let mut a_im = vec![0.0f32; m];
    for j in 0..n {
        let (xr, xi) = (re[j] as f64, im[j] as f64);
        a_re[j] = (xr * chirp_re[j] - xi * chirp_im[j]) as f32;
        a_im[j] = (xr * chirp_im[j] + xi * chirp_re[j]) as f32;
    }

    // b[j] = conj(chirp[j]) laid out for circular convolution: index j and
    // its wrap-around mirror M−j both hold b[j] (the kernel depends on j², so
    // b[−j] = b[j]); M ≥ 2N−1 keeps the two ranges disjoint.
    let mut b_re = vec![0.0f32; m];
    let mut b_im = vec![0.0f32; m];
    b_re[0] = 1.0; // chirp[0] = exp(0)
    for j in 1..n {
        let br = chirp_re[j] as f32;
        let bi = (-chirp_im[j]) as f32;
        b_re[j] = br;
        b_im[j] = bi;
        b_re[m - j] = br;
        b_im[m - j] = bi;
    }

    // Circular convolution at M on the radix-2 plans: IFFT(FFT(a)·FFT(b)).
    // One forward plan serves both operands; the inverse plan's built-in 1/M
    // scale is exactly the convolution normalisation.
    let mut fwd = FftPlan::new(gpu, m, false)?;
    let (fa_re, fa_im) = fwd.execute(&a_re, &a_im)?;
    let (fb_re, fb_im) = fwd.execute(&b_re, &b_im)?;

    let mut prod_re = vec![0.0f32; m];
    let mut prod_im = vec![0.0f32; m];
    for k in 0..m {
        let (ar, ai) = (fa_re[k] as f64, fa_im[k] as f64);
        let (br, bi) = (fb_re[k] as f64, fb_im[k] as f64);
        prod_re[k] = (ar * br - ai * bi) as f32;
        prod_im[k] = (ar * bi + ai * br) as f32;
    }

    let (conv_re, conv_im) = FftPlan::new(gpu, m, true)?.execute(&prod_re, &prod_im)?;

    // First N bins, output chirp, and the inverse DFT's 1/N.
    let scale = if inverse { 1.0 / n as f64 } else { 1.0 };
    let mut out_re = Vec::with_capacity(n);
    let mut out_im = Vec::with_capacity(n);
    for k in 0..n {
        let (cr, ci) = (conv_re[k] as f64, conv_im[k] as f64);
        out_re.push(((cr * chirp_re[k] - ci * chirp_im[k]) * scale) as f32);
        out_im.push(((cr * chirp_im[k] + ci * chirp_re[k]) * scale) as f32);
    }
    Ok((out_re, out_im))
}
