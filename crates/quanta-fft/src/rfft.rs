//! Real-input FFT ([`rfft`]) and its inverse ([`irfft`]) — the packed
//! half-size method.
//!
//! A real signal's spectrum is conjugate-symmetric (`X[N−k] = conj(X[k])`),
//! so the first `N/2 + 1` bins determine all `N`. [`rfft`] returns exactly
//! those bins; [`irfft`] takes them back to the real signal.
//!
//! ## How: the packed real-FFT (~2× over the full complex transform)
//!
//! Instead of running an N-point complex FFT with a zero imaginary part,
//! the N real samples are **packed** as N/2 complex pairs
//! `z[k] = x[2k] + i·x[2k+1]`, one N/2-point complex [`FftPlan`] runs on the
//! device — half the butterflies, half the device memory — and an O(N)
//! **split** post-pass separates the interleaved spectra:
//!
//! ```text
//! Fe[k] = (Z[k] + conj(Z[(N/2−k) mod N/2])) / 2      (DFT of even samples)
//! Fo[k] = −i·(Z[k] − conj(Z[(N/2−k) mod N/2])) / 2   (DFT of odd samples)
//! X[k]  = Fe[k] + e^(−2πik/N)·Fo[k],   k = 0 .. N/2
//! ```
//!
//! [`irfft`] applies the exact algebraic inverse of the split (recovering
//! `Z[k] = Fe[k] + i·Fo[k]`), runs the half-size inverse plan, and
//! de-interleaves. The split/merge passes run on the host in `f64` (the
//! transform I/O is host vectors already; the O(N) pass is negligible next
//! to the O(N log N) device work and tightens rounding).
//!
//! Sizes must be a power of 2 (`NotSupported` otherwise), matching the
//! complex path. Ground truth is [`reference::rdft`](crate::reference::rdft) /
//! [`irdft`](crate::reference::irdft); the packed method reorders the f32
//! summations relative to the full complex FFT, so results match the oracle
//! (and the full transform's first bins) to the crate's differential
//! tolerance, not bit-for-bit.

use core::f64::consts::PI;

use quanta_core::{Gpu, QuantaError};

use crate::plan::FftPlan;

/// Forward real-input FFT: real signal of length `n` (a power of 2) → the
/// first `n/2 + 1` complex bins `(re, im)` of its spectrum. The remaining
/// bins are their conjugates (`X[n−k] = conj(X[k])`) and carry no extra
/// information.
///
/// Runs one half-size (`n/2`) complex [`FftPlan`] on the device plus an
/// O(n) host split pass — about half the work and device memory of the
/// full complex transform. Errors with `NotSupported` if `n` is not a
/// power of 2.
///
/// `im[0]` and `im[n/2]` of the result are exactly `0.0` (DC and Nyquist
/// bins of a real signal are real).
pub fn rfft(gpu: &Gpu, real: &[f32]) -> Result<(Vec<f32>, Vec<f32>), QuantaError> {
    let n = real.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    if !n.is_power_of_two() {
        return Err(QuantaError::not_supported(
            "rfft: length must be a power of 2",
        ));
    }
    if n == 1 {
        return Ok((vec![real[0]], vec![0.0]));
    }
    let h = n / 2;

    // Pack: z[k] = x[2k] + i·x[2k+1], then one half-size complex FFT.
    let mut pack_re = Vec::with_capacity(h);
    let mut pack_im = Vec::with_capacity(h);
    for k in 0..h {
        pack_re.push(real[2 * k]);
        pack_im.push(real[2 * k + 1]);
    }
    let (zr, zi) = FftPlan::new(gpu, h, false)?.execute(&pack_re, &pack_im)?;

    // Split (host, f64): X[k] = Fe[k] + e^(−2πik/n)·Fo[k], k = 0..=n/2.
    // Z is n/2-periodic, so Z[n/2] reads Z[0].
    let mut out_re = Vec::with_capacity(h + 1);
    let mut out_im = Vec::with_capacity(h + 1);
    for k in 0..=h {
        let zk = k % h; // k == h wraps to 0
        let zc = (h - k) % h; // index of the conjugate partner
        let (ar, ai) = (zr[zk] as f64, zi[zk] as f64);
        let (br, bi) = (zr[zc] as f64, -(zi[zc] as f64)); // conj(Z[(h−k) mod h])
        // Fe = (Z[k] + conj)/2 ; Fo = −i·(Z[k] − conj)/2.
        let fe_re = 0.5 * (ar + br);
        let fe_im = 0.5 * (ai + bi);
        let fo_re = 0.5 * (ai - bi); // −i·(dr + i·di)/2 = (di − i·dr)/2
        let fo_im = -0.5 * (ar - br);
        let theta = -2.0 * PI * (k as f64) / (n as f64);
        let (s, c) = theta.sin_cos();
        out_re.push((fe_re + c * fo_re - s * fo_im) as f32);
        out_im.push((fe_im + c * fo_im + s * fo_re) as f32);
    }
    // DC and Nyquist of a real signal are exactly real; the split arithmetic
    // gets there only to rounding, so pin them.
    out_im[0] = 0.0;
    out_im[h] = 0.0;
    Ok((out_re, out_im))
}

/// Inverse real FFT: half-spectrum `(re, im)` of length `n/2 + 1` → the real
/// signal of length `n` (a power of 2). Inverse of [`rfft`]:
/// `irfft(rfft(x), x.len()) ≈ x`.
///
/// The half-spectrum is assumed conjugate-symmetric-consistent — in
/// particular `im[0]` and `im[n/2]` (DC and Nyquist) should be 0 for the
/// result to be a genuine real signal; the merge symmetrizes, so small
/// deviations are averaged away rather than amplified.
///
/// Errors: `NotSupported` if `n` is not a power of 2; `invalid_param` if
/// `re`/`im` lengths disagree or are not `n/2 + 1`.
pub fn irfft(gpu: &Gpu, re: &[f32], im: &[f32], n: usize) -> Result<Vec<f32>, QuantaError> {
    if re.len() != im.len() {
        return Err(QuantaError::invalid_param("irfft: re/im length mismatch"));
    }
    if n == 0 {
        return if re.is_empty() {
            Ok(vec![])
        } else {
            Err(QuantaError::invalid_param(
                "irfft: non-empty spectrum for n = 0",
            ))
        };
    }
    if !n.is_power_of_two() {
        return Err(QuantaError::not_supported(
            "irfft: length must be a power of 2",
        ));
    }
    if re.len() != n / 2 + 1 {
        return Err(QuantaError::invalid_param(
            "irfft: half-spectrum must hold n/2 + 1 bins",
        ));
    }
    if n == 1 {
        return Ok(vec![re[0]]);
    }
    let h = n / 2;

    // Merge (host, f64) — the exact algebraic inverse of the rfft split:
    //   Fe[k] = (X[k] + conj(X[h−k])) / 2
    //   Fo[k] = e^(+2πik/n) · (X[k] − conj(X[h−k])) / 2
    //   Z[k]  = Fe[k] + i·Fo[k],   k = 0..h
    let mut zre = Vec::with_capacity(h);
    let mut zim = Vec::with_capacity(h);
    for k in 0..h {
        let (ar, ai) = (re[k] as f64, im[k] as f64);
        let (br, bi) = (re[h - k] as f64, -(im[h - k] as f64)); // conj(X[h−k])
        let fe_re = 0.5 * (ar + br);
        let fe_im = 0.5 * (ai + bi);
        let tr = 0.5 * (ar - br);
        let ti = 0.5 * (ai - bi);
        let theta = 2.0 * PI * (k as f64) / (n as f64);
        let (s, c) = theta.sin_cos();
        let fo_re = c * tr - s * ti;
        let fo_im = c * ti + s * tr;
        zre.push((fe_re - fo_im) as f32); // Z = Fe + i·Fo
        zim.push((fe_im + fo_re) as f32);
    }

    // Half-size inverse plan (its 1/(n/2) scale is exactly the packed
    // normalisation), then de-interleave: x[2k] + i·x[2k+1] = z[k].
    let (or_, oi_) = FftPlan::new(gpu, h, true)?.execute(&zre, &zim)?;
    let mut out = Vec::with_capacity(n);
    for k in 0..h {
        out.push(or_[k]);
        out.push(oi_[k]);
    }
    Ok(out)
}
