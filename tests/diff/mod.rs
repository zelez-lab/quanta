//! Differential CI harness — Step D / Step 077.
//!
//! Runs each kernel in `kernels::` on every available backend lane,
//! then compares lane outputs against a pure-Rust reference oracle
//! using the tolerance rules in `compare`:
//!
//! - integer kernels: bit-exact across lanes
//! - float kernels:   ≤ 1 ULP versus the reference
//! - atomic kernels:  trace membership in the 055 / 056 memory model
//!   (added when atomic kernels land)
//!
//! This commit only enables the `software` lane. Metal, Vulkan, and
//! WGSL lanes land in follow-ups.

pub mod compare;
pub mod kernels;
pub mod lane;
pub mod output;
