//! Step 082 Layer 1 — per-op differential matrix.
//!
//! See `tests/diff/op_matrix.rs` for the case generator and the
//! per-lane dispatcher. This top-level driver runs every case on
//! every available backend lane and asserts bit-exact (integer) or
//! ULP-bounded (float) agreement with the CPU reference.
//!
//! Coverage: all 10 integer BinOps × {U32, U64, I32, I64} ×
//! handpicked edge inputs, plus all 4 float BinOps × F32 × float
//! edge inputs (zeros, denormals, infinities, the small-magnitude
//! constants that exposed the `85551fa` const-format bug). F64
//! cases run on the software lane only — Metal has no `double`
//! type and the structural fix is queued for step 082 Layer 4.
//!
//! Run locally:
//!   cargo test --test op_matrix --features software --no-default-features
//!   cargo test --test op_matrix --features software,metal    # macOS
//!   cargo test --test op_matrix --no-default-features --features vulkan
//!     # Linux + libvulkan-dev

#![cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]

#[path = "diff/mod.rs"]
mod diff;

use diff::compare::{Divergence, compare_bit_exact, compare_f32};
use diff::lane::Lane;
use diff::op_matrix::{OpCase, cases, dispatch_on, oracle};
use diff::output::{RawOutput, RawValues};

/// Pick the right comparator for a case's output type and ULP
/// tolerance. Float outputs with `max_ulps > 0` use `compare_f32`;
/// everything else (including bit-exact F32 ops like Add/Sub/Mul)
/// goes through `compare_bit_exact`, which itself dispatches on
/// the variant.
fn compare_case(
    case: &OpCase,
    oracle: &RawOutput,
    candidate: &RawOutput,
) -> Result<(), Divergence> {
    let is_float = matches!(case.expected, RawValues::F32(_));
    if is_float && case.max_ulps > 0 {
        compare_f32(oracle, candidate, case.max_ulps)
    } else {
        compare_bit_exact(oracle, candidate)
    }
}

/// Software lane: every case must agree with the (host-computed)
/// CPU reference. The software lane *is* the CPU interpreter, so
/// this checks that our expected-value computation matches the IR
/// interpreter — a sanity check, not a backend check.
#[cfg(feature = "software")]
#[test]
fn op_matrix_software_matches_reference() {
    let gpu = quanta::init_cpu();
    let mut failures = Vec::new();
    for case in cases() {
        let oracle = oracle(&case);
        let candidate = dispatch_on(&gpu, &case, Lane::Software);
        if let Err(div) = compare_case(&case, &oracle, &candidate) {
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

/// Metal lane: every case must agree (bit-exact or ≤max_ulps) with
/// the CPU reference. This is the lane that would have caught the
/// `06e764c` shift sign-extension bug, the `85551fa` float-const
/// bug, and the SatAdd silent-wrap bug fixed in `0515a8f`. Cases
/// flagged `skip_on_metal` are deferred to Layer 4 (capability
/// table) — currently the F64 cases, since MSL has no `double`.
#[cfg(feature = "metal")]
#[test]
fn op_matrix_metal_matches_reference() {
    let gpu = quanta::init().expect("metal lane requires a metal-capable device");
    let mut failures = Vec::new();
    for case in cases() {
        if case.skip_on_metal {
            continue;
        }
        let oracle = oracle(&case);
        let candidate = dispatch_on(&gpu, &case, Lane::Metal);
        if let Err(div) = compare_case(&case, &oracle, &candidate) {
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
    let f64_ok = gpu.supports_f64();
    let i64_ok = gpu.supports_i64();
    let mut failures = Vec::new();
    for case in cases() {
        // F64 / Int64 kernels emit the matching SPIR-V capability, which
        // only builds a pipeline on a device that enables the feature
        // (llvmpipe yes, Broadcom V3D no). Skip them where the device
        // can't run them rather than failing pipeline creation.
        if !f64_ok && matches!(case.expected, RawValues::F64(_)) {
            continue;
        }
        if !i64_ok && matches!(case.input_a, RawValues::U64(_) | RawValues::I64(_)) {
            continue;
        }
        let oracle = oracle(&case);
        let candidate = dispatch_on(&gpu, &case, Lane::Vulkan);
        if let Err(div) = compare_case(&case, &oracle, &candidate) {
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
