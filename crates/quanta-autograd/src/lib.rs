//! # quanta-autograd — reverse-mode automatic differentiation for Quanta
//!
//! The tier-2 **differentiation primitive**: a tape-based reverse-mode autodiff
//! engine over [`quanta_array::Array`], the substrate ML consumers (tier-5,
//! e.g. `ai_project`) build training on. It is *not* an ML framework — it adds
//! gradients to Quanta's GPU array ops, nothing more.
//!
//! Each elementwise op's VJP (vector-Jacobian product) multiplier is the
//! analytic derivative, proven in
//! `specs/verify/lean/Quanta/Autograd/Vjp.lean` via Mathlib's `HasDerivAt`.
//! The Rust [`vjp`] module mirrors those rules as pure functions; [`tape`] owns
//! the graph + reverse sweep.
//!
//! ## Example
//!
//! ```no_run
//! use quanta_autograd::Tape;
//! use quanta_array::Array;
//!
//! let gpu = quanta::init_cpu();
//! let tape = Tape::<f32>::new();
//! let x = tape.var(Array::from_slice(&gpu, &[1.0, 2.0, 3.0], &[3]).unwrap());
//! // loss = sum(x * x)  ⇒  d loss / d x = 2x
//! let sq = x.mul(&x).unwrap();
//! let loss = sq.sum().unwrap();
//! let gx = loss.grad(&x).unwrap();
//! assert_eq!(gx.to_vec().unwrap(), vec![2.0, 4.0, 6.0]);
//! ```
//!
//! ## This release
//!
//! Tape-based reverse mode over elementwise ops (neg, add, sub, mul, div, exp,
//! log, sqrt), activations (relu, sigmoid, tanh), the `sum`/`sum_axis`/
//! `mean_axis` reductions, `matmul`, `conv2d` (NCHW, via im2col → matmul →
//! reshape, with the backward reusing the matmul VJP and the im2col/col2im
//! adjoint pair), `avgpool2d`/`maxpool2d`, and `reshape`/`flatten`. The VJP
//! rules are factored as standalone functions ([`vjp`]) so a future
//! graph/fusion layer can reuse them.

#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod conv;
pub mod error;
pub mod ops;
pub mod pool;
pub mod scalar;
pub mod tape;
pub mod vjp;

pub use error::AutogradError;
pub use scalar::DiffScalar;
pub use tape::{Tape, Var};
