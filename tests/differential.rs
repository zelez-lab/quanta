//! Step D / Step 077 — differential CI entry point.
//!
//! Runs each kernel in `diff::kernels` on every available backend
//! lane and asserts agreement with the pure-Rust reference under
//! the kernel's tolerance policy:
//!
//! - integer kernels: bit-exact
//! - float kernels:   ≤ 1 ULP
//!
//! Per-PR (host-tests in ci.yml) enables the `software` lane.
//! Nightly + `run-full-diff`-labelled PRs (diff-full.yml) additionally
//! enable the `metal` lane on macOS runners and the `vulkan` lane on
//! ubuntu via Mesa lavapipe. WGSL lane runs through `web-smoke.yml`
//! against examples/web_diff/.
//!
//! Run locally:
//!   cargo test --test differential --features software --no-default-features
//!   cargo test --test differential --features software,metal           # macOS
//!   cargo test --test differential --no-default-features --features vulkan  # Linux + libvulkan-dev

#![cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]

#[path = "diff/mod.rs"]
mod diff;

use diff::compare::{compare_f32, compare_u32};
use diff::kernels::{counter, reduce_sum, saxpy};

// ── Reference oracle self-consistency ────────────────────────────────

#[test]
fn saxpy_reference_self_consistent() {
    let oracle = saxpy::run_reference();
    let again = saxpy::run_reference();
    compare_f32(&oracle, &again, 0).expect("reference SAXPY must be deterministic");
}

#[test]
fn reduce_sum_reference_self_consistent() {
    let oracle = reduce_sum::run_reference();
    let again = reduce_sum::run_reference();
    compare_u32(&oracle, &again).expect("reference reduce_sum must be deterministic");
}

// ── Software lane ────────────────────────────────────────────────────

#[cfg(feature = "software")]
#[test]
fn saxpy_software_within_one_ulp_of_reference() {
    let oracle = saxpy::run_reference();
    let candidate = saxpy::run_software();
    if let Err(div) = compare_f32(&oracle, &candidate, 1) {
        panic!("SAXPY divergence: {}", div);
    }
}

#[cfg(feature = "software")]
#[test]
fn reduce_sum_software_bit_exact_versus_reference() {
    let oracle = reduce_sum::run_reference();
    let candidate = reduce_sum::run_software();
    if let Err(div) = compare_u32(&oracle, &candidate) {
        panic!("reduce_sum divergence: {}", div);
    }
}

#[cfg(feature = "software")]
#[test]
fn counter_software_bit_exact_versus_reference() {
    let oracle = counter::run_reference();
    let candidate = counter::run_software();
    if let Err(div) = compare_u32(&oracle, &candidate) {
        panic!("counter divergence: {}", div);
    }
}

// ── Metal lane (nightly + label-gated; macOS only) ───────────────────

#[cfg(feature = "metal")]
#[test]
fn saxpy_metal_within_one_ulp_of_reference() {
    let oracle = saxpy::run_reference();
    let candidate = saxpy::run_metal();
    if let Err(div) = compare_f32(&oracle, &candidate, 1) {
        panic!("SAXPY divergence: {}", div);
    }
}

#[cfg(feature = "metal")]
#[test]
fn reduce_sum_metal_bit_exact_versus_reference() {
    let oracle = reduce_sum::run_reference();
    let candidate = reduce_sum::run_metal();
    if let Err(div) = compare_u32(&oracle, &candidate) {
        panic!("reduce_sum divergence: {}", div);
    }
}

#[cfg(feature = "metal")]
#[test]
fn counter_metal_bit_exact_versus_reference() {
    let oracle = counter::run_reference();
    let candidate = counter::run_metal();
    if let Err(div) = compare_u32(&oracle, &candidate) {
        panic!("counter divergence: {}", div);
    }
}

// ── Vulkan lane (nightly + label-gated; ubuntu via lavapipe) ─────────

#[cfg(feature = "vulkan")]
#[test]
fn saxpy_vulkan_within_one_ulp_of_reference() {
    let oracle = saxpy::run_reference();
    let candidate = saxpy::run_vulkan();
    if let Err(div) = compare_f32(&oracle, &candidate, 1) {
        panic!("SAXPY divergence: {}", div);
    }
}

#[cfg(feature = "vulkan")]
#[test]
fn reduce_sum_vulkan_bit_exact_versus_reference() {
    let oracle = reduce_sum::run_reference();
    let candidate = reduce_sum::run_vulkan();
    if let Err(div) = compare_u32(&oracle, &candidate) {
        panic!("reduce_sum divergence: {}", div);
    }
}

#[cfg(feature = "vulkan")]
#[test]
fn counter_vulkan_bit_exact_versus_reference() {
    let oracle = counter::run_reference();
    let candidate = counter::run_vulkan();
    if let Err(div) = compare_u32(&oracle, &candidate) {
        panic!("counter divergence: {}", div);
    }
}
