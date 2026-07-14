//! Differential CI harness — Step D / Step 077 / Step 082 Layer 1.
//!
//! Runs each kernel in `kernels::` on every available backend lane,
//! then compares lane outputs against a pure-Rust reference oracle
//! using the tolerance rules in `compare`:
//!
//! - integer kernels: bit-exact across lanes
//! - float kernels:   ≤ 1 ULP versus the reference
//! - atomic kernels:  trace membership in the 055 / 056 memory model
//! - op-matrix:       bit-exact per (op, type, edge-input) — see
//!   `op_matrix` module (step 082 Layer 1).
//!
//! Each top-level test binary (`tests/differential.rs`,
//! `tests/op_matrix.rs`) imports the parts of this module it
//! actually uses. `#[allow(dead_code)]` keeps the per-binary
//! dead-code analyzer quiet for items used by sibling binaries.

#![allow(dead_code)]

pub mod compare;
pub mod histogram;
pub mod kernels;
pub mod lane;
pub mod op_matrix;
pub mod output;
