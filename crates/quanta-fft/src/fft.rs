//! Radix-2 Cooley-Tukey FFT on the GPU (split re/im, power-of-2) — the
//! one-shot entry points.
//!
//! [`fft`] and [`ifft`] are one-shot plans: each call builds an
//! [`FftPlan`](crate::FftPlan) for the input size and executes it once. The
//! transform itself — bit-reversal, `log₂N` butterfly stages loading
//! precomputed twiddles, the inverse `1/N` scale — lives in [`crate::plan`];
//! repeated same-size transforms should hold a plan and call
//! [`execute`](crate::FftPlan::execute) to skip the per-call kernel JIT and
//! twiddle upload.
//!
//! Sizes must be a power of 2; others return `NotSupported`.

use quanta::{Gpu, QuantaError};

use crate::plan::FftPlan;

/// Forward FFT. `re`/`im` are split complex inputs of length `n` (a power of
/// 2); returns the transformed `(re, im)` as host vectors. Errors if `n` is
/// not a power of 2 or the lengths disagree.
///
/// One-shot: builds and runs an [`FftPlan`] for this size. Transforming many
/// same-size signals? Build the plan once and reuse it.
pub fn fft(gpu: &Gpu, re: &[f32], im: &[f32]) -> Result<(Vec<f32>, Vec<f32>), QuantaError> {
    run(gpu, re, im, false)
}

/// Inverse FFT (`+` twiddle sign, `1/n` scale): `ifft(fft(x)) == x`.
///
/// One-shot: builds and runs an [`FftPlan`] for this size. Transforming many
/// same-size signals? Build the plan once and reuse it.
pub fn ifft(gpu: &Gpu, re: &[f32], im: &[f32]) -> Result<(Vec<f32>, Vec<f32>), QuantaError> {
    run(gpu, re, im, true)
}

fn run(
    gpu: &Gpu,
    re: &[f32],
    im: &[f32],
    inverse: bool,
) -> Result<(Vec<f32>, Vec<f32>), QuantaError> {
    if re.len() != im.len() {
        return Err(QuantaError::invalid_param("fft: re/im length mismatch"));
    }
    if re.is_empty() {
        return Ok((vec![], vec![]));
    }
    FftPlan::new(gpu, re.len(), inverse)?.execute(re, im)
}
