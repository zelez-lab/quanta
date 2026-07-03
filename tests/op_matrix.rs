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

/// Whether a device with the given 64-bit capabilities can run a
/// case. F64 / 64-bit-int kernels emit the matching capability
/// (SPIR-V `Float64`/`Int64`, MSL `double`/`long`), which only
/// builds a pipeline on a device that enables the feature —
/// llvmpipe yes, Broadcom V3D no. A device that can't doesn't just
/// fail pipeline creation: V3D's Mesa driver aborts the whole
/// process on `pack_64_2x32_split`, so the guard must run before
/// dispatch.
///
/// Every lane goes through this one predicate so the skip logic
/// can't drift between lanes again (the metal lane once lacked it
/// and SIGABRTed the vulkan-only Pi when the default `metal`
/// feature leaked into a `--features vulkan` run). Each lane
/// supplies its own `f64_ok` / `i64_ok` because the Metal lane
/// can't take them from `supports_f64()` / `supports_i64()` — see
/// the comment there.
fn device_supports_case(case: &OpCase, f64_ok: bool, i64_ok: bool) -> bool {
    let needs_f64 =
        matches!(case.input_a, RawValues::F64(_)) || matches!(case.expected, RawValues::F64(_));
    let needs_i64 = matches!(case.input_a, RawValues::U64(_) | RawValues::I64(_))
        || matches!(case.expected, RawValues::U64(_) | RawValues::I64(_));
    (!needs_f64 || f64_ok) && (!needs_i64 || i64_ok)
}

/// Software lane: every case must agree with the (host-computed)
/// CPU reference. The software lane *is* the CPU interpreter, so
/// this checks that our expected-value computation matches the IR
/// interpreter — a sanity check, not a backend check.
#[cfg(feature = "software")]
#[test]
fn op_matrix_software_matches_reference() {
    let gpu = quanta::init_cpu();
    // No-op today (the CPU interpreter supports f64 and i64) but
    // kept so all three lanes share the exact same skip logic.
    let (f64_ok, i64_ok) = (gpu.supports_f64(), gpu.supports_i64());
    let mut failures = Vec::new();
    for case in cases() {
        if !device_supports_case(&case, f64_ok, i64_ok) {
            continue;
        }
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
    // `default = ["metal", "render"]` means this test compiles on any
    // host that doesn't pass `--no-default-features` — including a
    // Linux box running the vulkan lane, where `init()` hands back a
    // Vulkan device (or nothing). The metal lane only means something
    // on an actual Metal backend; non-Metal devices are owned by the
    // vulkan lane. Skip, don't fail: compiling the test in is a
    // feature-resolution artifact, not a device error.
    let gpu = match quanta::init() {
        Ok(gpu) if gpu.caps().vendor == quanta::Vendor::Apple => gpu,
        Ok(gpu) => {
            eprintln!(
                "op_matrix metal lane: init() returned non-Metal device '{}' — skipping",
                gpu.name()
            );
            return;
        }
        Err(e) => {
            if cfg!(target_os = "macos") {
                panic!("metal lane: no device found on macOS: {e}");
            }
            eprintln!("op_matrix metal lane: no device on this host — skipping");
            return;
        }
    };
    // Real Metal from here on. MSL has native `long`/`ulong` and this
    // suite's u64/i64 cases have always run on this backend, but
    // `supports_i64()` still reports false on Metal (it gates heavier
    // u64 consumers like quanta-rand, whose kernels don't yet produce
    // correct bits here). Trusting it would silently drop the lane's
    // 64-bit-int coverage, so only f64 comes from the device query.
    let (f64_ok, i64_ok) = (gpu.supports_f64(), true);
    let mut failures = Vec::new();
    for case in cases() {
        if case.skip_on_metal {
            continue;
        }
        if !device_supports_case(&case, f64_ok, i64_ok) {
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
    let (f64_ok, i64_ok) = (gpu.supports_f64(), gpu.supports_i64());
    let mut failures = Vec::new();
    for case in cases() {
        if !device_supports_case(&case, f64_ok, i64_ok) {
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
