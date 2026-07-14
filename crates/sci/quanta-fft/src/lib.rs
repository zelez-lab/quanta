//! # quanta-fft — GPU FFT for Quanta
//!
//! Fast Fourier transform: forward + inverse, complex data split into
//! real/imag `f32` arrays, **any length N ≥ 1**. Power-of-2 sizes run the
//! radix-2 Cooley-Tukey kernels; other sizes run Bluestein's chirp-z
//! algorithm on top of the same radix-2 plans. Built on the Quanta JIT —
//! one kernel, every backend.
//!
//! ## This release
//!
//! - [`fft`](fft::fft) / [`ifft`](fft::ifft) — any N (one-shot: plan +
//!   execute in a single call). Power-of-2 N takes the radix-2 path;
//!   non-power-of-2 N takes the Bluestein chirp-z convolution at
//!   `M = next_pow2(2N−1)` (≈ three power-of-2 transforms, so 2^k input
//!   sizes are the faster shape).
//! - [`fft2`](fft2::fft2) / [`ifft2`](fft2::ifft2) — 2-D transform of a
//!   row-major H×W grid (both dims powers of 2) by row-column decomposition:
//!   row pass → transpose → column-as-row pass → transpose back, each pass
//!   reusing one [`FftPlan`](plan::FftPlan).
//! - [`FftPlan`](plan::FftPlan) — plan-based dispatch (the VkFFT pattern):
//!   fixes size + direction once, JIT-compiles the kernels once, precomputes
//!   the twiddle table into a device buffer, then
//!   [`execute`](plan::FftPlan::execute)s any number of same-size transforms
//!   without rebuilds or per-butterfly `sin`/`cos`.
//! - [`rfft`](rfft::rfft) / [`irfft`](rfft::irfft) — real-input FFT: real
//!   signal of length N → the `N/2 + 1` half-spectrum (and back), via the
//!   packed method — one half-size complex plan on the device plus an O(N)
//!   split pass, ~2× the throughput and half the memory of transforming the
//!   real signal as complex-with-zero-imag.
//! - [`reference`] — the pure-Rust direct DFT + real DFT (always available,
//!   no `gpu` feature needed); the differential-test oracles.
//!
//! Off by default the crate is the reference library; enable `gpu` (+ a backend
//! like `gpu-metal`) for the device FFT. Correctness is established by the
//! differential test (GPU FFT vs the direct DFT, power-of-2 and Bluestein
//! sizes alike) and the round trip `ifft(fft(x)) == x`; the
//! Cooley-Tukey-equals-DFT Lean proof is a parallel track
//! (`specs/verify/lean/Quanta/Fft/`) and models the radix-2 recursion only —
//! the Bluestein path's correctness bar is the differential oracle.
//!
//! [`FftPlan`](plan::FftPlan) itself remains power-of-2 (it IS the radix-2
//! engine — Bluestein builds on it); the `fft`/`ifft` entry points are the
//! arbitrary-N surface.

pub mod reference;

#[cfg(feature = "gpu")]
mod bluestein;
#[cfg(feature = "gpu")]
pub mod fft;
#[cfg(feature = "gpu")]
pub mod fft2;
#[cfg(feature = "gpu")]
pub mod plan;
#[cfg(feature = "gpu")]
pub mod rfft;

#[cfg(feature = "gpu")]
pub use fft::{fft, ifft};
#[cfg(feature = "gpu")]
pub use fft2::{fft2, ifft2};
#[cfg(feature = "gpu")]
pub use plan::FftPlan;
#[cfg(feature = "gpu")]
pub use rfft::{irfft, rfft};
