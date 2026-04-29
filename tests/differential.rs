//! Step D / Step 077 — differential CI entry point.
//!
//! Runs each kernel in `diff::kernels` on every available backend
//! lane and asserts agreement with the pure-Rust reference under
//! the kernel's tolerance policy:
//!
//! - integer kernels: bit-exact
//! - float kernels:   ≤ 1 ULP
//!
//! This commit only enables the `software` lane. Metal, Vulkan, and
//! WGSL lanes plus more kernels (reduce_sum, atomic counter, fence
//! pair) land in subsequent commits.
//!
//! Run: `cargo test --test differential --features software`

#![cfg(feature = "software")]

#[path = "diff/mod.rs"]
mod diff;

use diff::compare::compare_f32;
use diff::kernels::saxpy;

#[test]
fn saxpy_reference_self_consistent() {
    let oracle = saxpy::run_reference();
    let again = saxpy::run_reference();
    compare_f32(&oracle, &again, 0).expect("reference SAXPY must be deterministic");
}

#[test]
fn saxpy_software_within_one_ulp_of_reference() {
    let oracle = saxpy::run_reference();
    let candidate = saxpy::run_software();
    if let Err(div) = compare_f32(&oracle, &candidate, 1) {
        panic!("SAXPY divergence: {}", div);
    }
}
