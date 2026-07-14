//! FFT on the GPU (split re/im, any N ≥ 1) — the one-shot entry points.
//!
//! [`fft`] and [`ifft`] accept any length. Power-of-2 sizes take the radix-2
//! Cooley-Tukey path: each call builds an [`FftPlan`](crate::FftPlan) for the
//! input size and executes it once (bit-reversal, `log₂N` butterfly stages
//! loading precomputed twiddles, the inverse `1/N` scale — all in
//! [`crate::plan`]). Repeated same-size power-of-2 transforms should hold a
//! plan and call [`execute`](crate::FftPlan::execute) to skip the per-call
//! kernel JIT and twiddle upload.
//!
//! Non-power-of-2 sizes route through [`crate::bluestein`] — the chirp-z
//! reformulation of the DFT as a power-of-2 convolution, run on the same
//! radix-2 plans at `M = next_pow2(2N−1)`. It costs three length-M
//! transforms, so a power-of-2 N is always the faster shape.

use quanta_core::{Gpu, QuantaError};

use crate::bluestein;
use crate::plan::FftPlan;

/// Forward FFT. `re`/`im` are split complex inputs of length `n` (any size);
/// returns the transformed `(re, im)` as host vectors. Errors if the lengths
/// disagree.
///
/// Power-of-2 `n` runs the radix-2 kernels directly (one-shot [`FftPlan`] —
/// transforming many same-size signals? Build the plan once and reuse it).
/// Other sizes use Bluestein's chirp-z algorithm: the same radix-2 plans at
/// `next_pow2(2n−1)`, so they cost roughly three power-of-2 transforms.
pub fn fft(gpu: &Gpu, re: &[f32], im: &[f32]) -> Result<(Vec<f32>, Vec<f32>), QuantaError> {
    run(gpu, re, im, false)
}

/// Inverse FFT (`+` twiddle sign, `1/n` scale): `ifft(fft(x)) == x`, any `n`.
///
/// Power-of-2 `n` runs the radix-2 kernels directly (one-shot [`FftPlan`] —
/// transforming many same-size signals? Build the plan once and reuse it).
/// Other sizes use Bluestein's chirp-z algorithm with conjugated chirps.
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
    if re.len().is_power_of_two() {
        // Radix-2 path, unchanged: power-of-2 inputs never pay the chirp-z
        // convolution overhead.
        FftPlan::new(gpu, re.len(), inverse)?.execute(re, im)
    } else {
        bluestein::transform(gpu, re, im, inverse)
    }
}
