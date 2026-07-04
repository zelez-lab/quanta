//! # quanta-fft — GPU FFT for Quanta
//!
//! Radix-2 Cooley-Tukey fast Fourier transform: forward + inverse, complex
//! data split into real/imag `f32` arrays, sizes a power of 2. Built on the
//! Quanta JIT — one kernel, every backend.
//!
//! ## This release
//!
//! - [`fft`](fft::fft) / [`ifft`](fft::ifft) — radix-2, N a power of 2
//!   (one-shot: plan + execute in a single call).
//! - [`FftPlan`](plan::FftPlan) — plan-based dispatch (the VkFFT pattern):
//!   fixes size + direction once, JIT-compiles the kernels once, precomputes
//!   the twiddle table into a device buffer, then
//!   [`execute`](plan::FftPlan::execute)s any number of same-size transforms
//!   without rebuilds or per-butterfly `sin`/`cos`.
//! - [`reference`] — the pure-Rust direct DFT (always available, no `gpu`
//!   feature needed); the differential-test oracle.
//!
//! Off by default the crate is the reference library; enable `gpu` (+ a backend
//! like `gpu-metal`) for the device FFT. Correctness is established by the
//! differential test (GPU FFT vs the direct DFT) and the round trip
//! `ifft(fft(x)) == x`; the Cooley-Tukey-equals-DFT Lean proof is a parallel
//! track (`specs/verify/lean/Quanta/Fft/`).
//!
//! Non-power-of-2 sizes return `NotSupported` (mixed-radix is a later
//! increment).

pub mod reference;

#[cfg(feature = "gpu")]
pub mod fft;
#[cfg(feature = "gpu")]
pub mod plan;

#[cfg(feature = "gpu")]
pub use fft::{fft, ifft};
#[cfg(feature = "gpu")]
pub use plan::FftPlan;
