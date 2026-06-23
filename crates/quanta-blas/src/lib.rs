//! # quanta-blas — verified-numerics BLAS for Quanta
//!
//! The linear-algebra companion crate. The headline claim: **every op
//! ships a mechanically-proven forward-error bound** (Higham-style),
//! formalised in `specs/verify/lean/Quanta/Blas/Reference.lean`. It builds
//! on `quanta-tensor` (shape proofs), `quanta-prims` (device-resident
//! reductions), and the Quanta JIT.
//!
//! ## This release: Level-1 (f32)
//!
//! - [`scal`](level1::scal) — `x ← α·x` (in place)
//! - [`axpy`](level1::axpy) — `y ← α·x + y` (in place)
//! - [`dot`](level1::dot) — `Σ xᵢ·yᵢ` (device-resident reduction)
//! - [`nrm2`](level1::nrm2) — `‖x‖₂ = √(Σ xᵢ²)`
//!
//! `scal`/`axpy` mutate their target buffer in place (these ops are
//! memory-bandwidth-bound, so avoiding a second buffer is the win); `dot`/
//! `nrm2` multiply into a temp field on the device and reduce there, so the
//! data never leaves the GPU.
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
pub use level1::{axpy, dot, nrm2, scal};
