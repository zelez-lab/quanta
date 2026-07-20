//! Neural network stack for Quanta — the ML face of the `quanta::`
//! namespace (`quanta::nn`, feature `nn`).
//!
//! Layers, fused kernels (attention, norms, rotary), losses, optimizers,
//! initialization, and the training loop — built on [`quanta_autograd`]
//! (the tape) and [`quanta_array`] (the data plane). Everything here is
//! single-node pure compute; the distributed, atom-aware mirrors live in
//! dija-nn and wrap this crate.
//!
//! # Completeness contract
//!
//! The crate's definition of done is `PARITY.md` at the crate root: the
//! declared torch.nn/burn-core surface, where every row either ships or
//! carries a documented deferral. "Complete" is a claim checkable against
//! that table, not a promise.
//!
//! # Functional ops & fused kernels
//!
//! [`functional`] holds the stateless ops. First up: fused scaled
//! dot-product attention — [`functional::scaled_dot_product_attention`]
//! (forward, returns the context + softmax stats) and the tape-differentiable
//! [`functional::sdpa_var`]. The online-softmax kernel behind them
//! ([`kernel`]) is the FlashAttention-style streaming form proven correct in
//! `specs/verify/lean/Quanta/Nn/OnlineSoftmax.lean` (T9200–T9209): it never
//! materialises the `seq_q × seq_k` score matrix.
//!
//! # Substrate re-exports
//!
//! The tape and array types are part of this crate's public vocabulary:

pub mod activation;
pub mod attention;
pub mod batchnorm;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod functional;
pub mod kernel;
pub mod layer;
pub mod loss;
pub mod norm;
pub mod optim;
pub mod rope;
pub mod state;

pub use quanta_array::Array;
pub use quanta_autograd::{AutogradError, DiffScalar, Tape, Var};
