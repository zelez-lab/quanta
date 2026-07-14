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
//! - **Local ops** on `Layout` (`transpose`, `permute`, `slice`,
//!   `broadcast`) — transformations of a single layout.
//! - **Global algebra** on `Layout` (`compose`, `complement`,
//!   `logical_divide`, `tiled_divide`) — combinators of two
//!   layouts. These are the load-bearing ops for downstream
//!   tiling: GEMM, FFT, sort all express their work as
//!   `layout.logical_divide(&tile)`.
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
//! ## Design notes
//!
//! - **Dynamic rank only.** `Shape` and `Layout` carry their
//!   extents and strides at runtime (`Vec<usize>`, `Vec<isize>`),
//!   not in the type system. CuTe uses compile-time integer
//!   tuples (`Shape<Int<M>, Int<K>>`) to enable kernel-time
//!   specialisation. Quanta deliberately doesn't follow: we'd lose
//!   interop with the dynamic-shape paths every downstream math
//!   crate eventually needs. Divisibility checks happen at runtime
//!   and return [`layout::LayoutError::DivisibilityFailed`] on
//!   violation.
//! - **Downstream proc-macros should consume accessors, not
//!   fields.** Use [`Layout::shape`] and [`Layout::strides`] (and
//!   [`Shape::dims`]) rather than the private struct fields. The
//!   internal representation may shift (e.g. small-vector
//!   optimisation, packed strides) without breaking accessor
//!   callers.
//! - **The algebra is the public surface.** When in doubt about
//!   how to tile a layout, reach for `logical_divide` / `compose`
//!   first. The local ops (`transpose`, `slice`, …) handle
//!   per-axis transformations; they don't compose into GEMM-style
//!   tiling on their own.
//!
//! ## What's coming
//!
//! Downstream math crates depend on this substrate:
//!
//! - `quanta-sort` — block radix sort + scan + reduce.
//! - `quanta-blas` — GEMM, GEMV, axpy.
//! - `quanta-fft` — Stockham FFT.
//!
//! Algebraic theorems beyond the structural ones already shipped
//! (composition associativity, permutation bijectivity, tile-offset
//! bounds, op-by-op preservation lemmas) land in follow-up
//! commits.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod layout;
pub mod shape;

pub use layout::{Layout, LayoutError};
pub use shape::{Shape, ShapeError};

#[cfg(test)]
mod layout_test;

#[cfg(test)]
#[path = "layout/algebra_test.rs"]
mod algebra_test;
