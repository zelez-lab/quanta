//! # quanta-tensor
//!
//! Tensor layout algebra — the substrate every downstream Quanta
//! math crate (sort, blas, fft, rand, …) plugs into for shape
//! correctness.
//!
//! ## What this is
//!
//! A pure-Rust types-and-functions library. No GPU runtime, no
//! proc-macro, no kernels. Just:
//!
//! - [`Shape`] — a multi-dimensional shape.
//! - [`Layout`] — a function-style index from a coordinate tuple to
//!   a flat-buffer offset, parameterised by strides.
//! - Composable ops on `Layout`: `transpose`, `permute`, `slice`,
//!   `broadcast`. Each returns a new layout without touching data.
//!
//! ## Why
//!
//! CUTLASS CuTe treats layouts as algebraic objects with associative
//! composition, bijective permutations, and provable tile-offset
//! bounds. Lifting that into Rust + Lean amortises shape proofs
//! across the whole companion-crate catalogue: GEMM's `M×K @ K×N →
//! M×N`, sort's length-preserving permutation, FFT's bijection on
//! power-of-two sizes — they all inherit lemmas from this crate.
//!
//! ## What's coming
//!
//! Phase 1 (this commit) ships the runtime substrate. A second
//! commit lands the Lean + Verus proof artifacts. Downstream
//! companion crates land afterwards in their own roadmap entries:
//!
//! - `quanta-sort` — block radix sort + scan + reduce.
//! - `quanta-blas` — GEMM, GEMV, axpy.
//! - `quanta-fft` — Stockham FFT.
//!
//! See `roadmap/081_companion_crates/README.md` for the full plan.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod layout;
pub mod shape;

pub use layout::Layout;
pub use shape::Shape;
