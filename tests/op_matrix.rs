//! Step 082 Layer 1 — per-op differential matrix.
//!
//! See `tests/diff/op_matrix.rs` for the case generator and the
//! per-lane dispatcher. This top-level driver runs every case on
//! every available backend lane and asserts bit-exact agreement
//! with the CPU reference.
//!
//! Initial coverage: one U32 `Shr` case exercising the
//! sign-extension path of the bug fixed in `06e764c`. Later tasks
//! in step 082 expand to cover all BinOp / UnaryOp / CmpOp / Cast
//! variants × ScalarType pairs.
//!
//! Run locally:
//!   cargo test --test op_matrix --features software --no-default-features
//!   cargo test --test op_matrix --features software,metal    # macOS
//!   cargo test --test op_matrix --no-default-features --features vulkan
//!     # Linux + libvulkan-dev

#![cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]

#[path = "diff/mod.rs"]
mod diff;

use diff::compare::compare_bit_exact;
use diff::lane::Lane;
use diff::op_matrix::{cases, dispatch_on};

/// Software lane: every case must agree with the (host-computed)
/// CPU reference bit-exact. The software lane *is* the CPU
/// interpreter, so this checks that our expected-value computation
/// matches the IR interpreter — a sanity check, not a backend
/// check.
#[cfg(feature = "software")]
#[test]
fn op_matrix_software_matches_reference() {
    let gpu = quanta::init_cpu();
    let mut failures = Vec::new();
    for case in cases() {
        let oracle = case.oracle();
        let candidate = dispatch_on(&gpu, &case, Lane::Software);
        if let Err(div) = compare_bit_exact(&oracle, &candidate) {
            failures.push(format!("{}: {}", case.name, div));
        }
    }
    if !failures.is_empty() {
        panic!(
            "op_matrix software-vs-reference divergences:\n  {}",
            failures.join("\n  ")
        );
    }
}

/// Metal lane: every case must agree bit-exact with the CPU
/// reference. This is the lane that would have caught the
/// `06e764c` shift sign-extension bug — the U32 Shr case on
/// input `0x80000000` produces `0xFF800000` on the buggy emitter
/// and `0x00800000` on the fixed one.
#[cfg(feature = "metal")]
#[test]
fn op_matrix_metal_matches_reference() {
    let gpu = quanta::init().expect("metal lane requires a metal-capable device");
    let mut failures = Vec::new();
    for case in cases() {
        let oracle = case.oracle();
        let candidate = dispatch_on(&gpu, &case, Lane::Metal);
        if let Err(div) = compare_bit_exact(&oracle, &candidate) {
            failures.push(format!("{}: {}", case.name, div));
        }
    }
    if !failures.is_empty() {
        panic!(
            "op_matrix metal-vs-reference divergences:\n  {}",
            failures.join("\n  ")
        );
    }
}

/// Vulkan lane: same shape as Metal. SPIR-V emitter exercises a
/// different code path entirely; same matrix catches the same
/// class of bug there.
#[cfg(feature = "vulkan")]
#[test]
fn op_matrix_vulkan_matches_reference() {
    let gpu = quanta::init().expect("vulkan lane requires a vulkan-capable device");
    let mut failures = Vec::new();
    for case in cases() {
        let oracle = case.oracle();
        let candidate = dispatch_on(&gpu, &case, Lane::Vulkan);
        if let Err(div) = compare_bit_exact(&oracle, &candidate) {
            failures.push(format!("{}: {}", case.name, div));
        }
    }
    if !failures.is_empty() {
        panic!(
            "op_matrix vulkan-vs-reference divergences:\n  {}",
            failures.join("\n  ")
        );
    }
}
