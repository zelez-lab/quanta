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
//! # Substrate re-exports
//!
//! The tape and array types are part of this crate's public vocabulary:

pub use quanta_array::Array;
pub use quanta_autograd::{Tape, Var};
