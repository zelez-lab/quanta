//! # quanta-blas — verified-numerics BLAS for Quanta
//!
//! The linear-algebra companion crate. The headline claim: **every op
//! ships a mechanically-proven forward-error bound** (Higham-style),
//! formalised in `specs/verify/lean/Quanta/Blas/Reference.lean`. It builds
//! on `quanta-tensor` (shape proofs), `quanta-prims` (device-resident
//! reductions), and the Quanta JIT.
//!
//! ## This release: Level-1 + GEMV + tiled GEMM (f32) + mixed-precision (bf16/f16/fp8/int8/int4)
//!
//! - [`scal`](level1::scal) — `x ← α·x` (in place)
//! - [`axpy`](level1::axpy) — `y ← α·x + y` (in place)
//! - [`dot`](level1::dot) — `Σ xᵢ·yᵢ` (device-resident reduction)
//! - [`nrm2`](level1::nrm2) — `‖x‖₂ = √(Σ xᵢ²)`
//! - [`gemv`](level2::gemv) — `y ← α·A·x + β·y` (Level-2, via GEMM N=1)
//! - [`gemm`](gemm::gemm) — `C ← α·A·B + β·C` (Level-3, tiled kernel)
//! - [`gemm_mixed`](mixed::gemm_mixed) / [`gemv_mixed`](mixed::gemv_mixed) —
//!   narrow float inputs (bf16 / f16 via `gemm_mixed`; fp8 via `gemm_mixed8`),
//!   f32 accumulate
//! - [`gemm_quant`](mixed_quant::gemm_quant) /
//!   [`gemm_quant4`](mixed_quant::gemm_quant4) (+ `gemv_*`) — int8 (Q8) and
//!   int4 (Q4) symmetric codes + per-tensor scales, f32 accumulate
//!
//! `scal`/`axpy` mutate their target buffer in place (these ops are
//! memory-bandwidth-bound, so avoiding a second buffer is the win); `dot`/
//! `nrm2` multiply into a temp field on the device and reduce there, so the
//! data never leaves the GPU. `gemv` is a GEMM with one output column
//! (`gemm(m, 1, n, …)`) — a gemv entry *is* a gemm entry, so it reuses the
//! gemm kernel and the same proven bound. `gemm` uses the **tiled
//! shared-memory** kernel — correct on every backend and matching the proven
//! Higham §3.5 contract; the cooperative-matrix path that closes the vendor
//! perf gap is a later increment.
//!
//! Off by default, the crate is a pure-Rust reference library (the
//! differential-test oracle in [`reference`]). Enable `gpu` (plus a backend
//! feature like `gpu-metal`) for the JIT ops in [`level1`].
//!
//! ## Performance framing (honest)
//!
//! quanta-blas v0.1 targets ~50% of vendor BLAS on tier-1 datacentre GPUs,
//! ~80% on tier-2 consumer/Apple-Silicon GPUs, and is the *only* option on
//! surfaces where vendor BLAS doesn't exist (WebGPU, mobile). Level-1 ops
//! are bandwidth-bound, so the generic cross-backend kernel is already
//! near memory roofline; the GEMM tensor-core work is where the tuned
//! per-backend paths land.

#![cfg_attr(not(feature = "gpu"), allow(dead_code))]

pub mod reference;

#[cfg(feature = "gpu")]
pub mod level1;

#[cfg(feature = "gpu")]
pub mod level2;

#[cfg(feature = "gpu")]
pub mod gemm;

#[cfg(feature = "gpu")]
pub mod mixed;

#[cfg(feature = "gpu")]
mod mixed_kernel;

#[cfg(feature = "gpu")]
pub mod mixed_quant;

#[cfg(feature = "gpu")]
pub use gemm::gemm;
#[cfg(feature = "gpu")]
pub use level1::{axpy, dot, nrm2, scal};
#[cfg(feature = "gpu")]
pub use level2::gemv;
#[cfg(feature = "gpu")]
pub use mixed::{GemmInputType, gemm_mixed, gemm_mixed8, gemv_mixed, gemv_mixed8};
#[cfg(feature = "gpu")]
pub use mixed_quant::{GemmQuantType, gemm_quant, gemm_quant4, gemv_quant, gemv_quant4};
