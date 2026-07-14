//! GPU litmus kernel suite — race-freedom L2, Phase 1.
//!
//! Empirical companion to the herd7 tests in `specs/verify/herd7/`.
//! Runs the message-passing (MP) and store-buffer (SB) litmus shapes as
//! real Quanta kernels, packing 10^5+ independent instances into a
//! single dispatch and building an outcome histogram (see
//! `diff::histogram` for the instance-packing layout).
//!
//! # Epistemics — these are falsifiers, not proofs
//!
//! Same standing as `specs/verify/herd7/README.md`: observing the
//! forbidden MP outcome even once fails the test and falsifies the
//! claim. A clean run over 10^5 instances is corroboration, never a
//! proof — a given driver / GPU / scheduler may simply never exercise
//! the offending interleaving. The SB anomaly under rel/acq is *allowed*
//! by the model but may or may not appear on a device (an in-order
//! software executor never shows it), so it is asserted permitted, not
//! required.
//!
//! # Gating
//!
//! Requires `jit` (kernels are built as `KernelDef` IR and fed through
//! `wave_jit`). The GPU path runs whenever `quanta::init()` yields a
//! device; when no device is available the GPU tests skip (print +
//! return) rather than fail. The software lane always runs and is the
//! in-order reference: it must never show a forbidden or an anomalous
//! outcome.
//!
//! Run locally:
//!   cargo test --test litmus --features software,jit
//!   cargo test --test litmus --features software,metal,jit      # macOS
//!   cargo test --test litmus --no-default-features --features vulkan,jit,compute

#![cfg(all(
    any(feature = "software", feature = "metal", feature = "vulkan"),
    feature = "jit"
))]

#[path = "diff/mod.rs"]
mod diff;

use diff::histogram::{Histogram, assert_outcomes};
use diff::kernels::{mp, sb};

// ── Software lane (always available; in-order reference) ─────────────

#[cfg(feature = "software")]
#[test]
fn mp_software_no_forbidden_outcome() {
    let hist = mp::run_software();
    check_mp(&hist, "software", false);
}

#[cfg(feature = "software")]
#[test]
fn sb_relacq_software_within_allowed() {
    let hist = sb::run_software(sb::Variant::RelAcq);
    check_sb_relacq(&hist, "software", false);
}

#[cfg(feature = "software")]
#[test]
fn sb_seqcst_software_forbids_anomaly() {
    let hist = sb::run_software(sb::Variant::SeqCst);
    check_sb_seqcst(&hist, "software");
}

// ── Metal lane (skips when no device) ────────────────────────────────

#[cfg(feature = "metal")]
#[test]
fn mp_metal_no_forbidden_outcome() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping mp_metal: no GPU available");
        return;
    };
    let hist = mp::run_metal(&gpu);
    // On real Metal, require the synchronized outcome `[1,42]` to appear
    // (consumer observed the release-published flag AND the data), which
    // proves the test is non-vacuous. The forbidden `[1,0]` must never
    // appear.
    check_mp(&hist, "metal", true);
}

#[cfg(feature = "metal")]
#[test]
fn sb_relacq_metal_within_allowed() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping sb_relacq_metal: no GPU available");
        return;
    };
    let hist = sb::run_on(&gpu, sb::Variant::RelAcq);
    check_sb_relacq(&hist, "metal", true);
}

#[cfg(feature = "metal")]
#[test]
fn sb_seqcst_metal_forbids_anomaly() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping sb_seqcst_metal: no GPU available");
        return;
    };
    let hist = sb::run_on(&gpu, sb::Variant::SeqCst);
    check_sb_seqcst(&hist, "metal");
}

// ── Vulkan lane (skips when no device) ───────────────────────────────

#[cfg(feature = "vulkan")]
#[test]
fn mp_vulkan_no_forbidden_outcome() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping mp_vulkan: no GPU available");
        return;
    };
    let hist = mp::run_vulkan(&gpu);
    check_mp(&hist, "vulkan", false);
}

#[cfg(feature = "vulkan")]
#[test]
fn sb_relacq_vulkan_within_allowed() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping sb_relacq_vulkan: no GPU available");
        return;
    };
    let hist = sb::run_on(&gpu, sb::Variant::RelAcq);
    check_sb_relacq(&hist, "vulkan", false);
}

#[cfg(feature = "vulkan")]
#[test]
fn sb_seqcst_vulkan_forbids_anomaly() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping sb_seqcst_vulkan: no GPU available");
        return;
    };
    let hist = sb::run_on(&gpu, sb::Variant::SeqCst);
    check_sb_seqcst(&hist, "vulkan");
}

// ── Shared assertions ────────────────────────────────────────────────

#[cfg(feature = "metal")]
fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[cfg(all(feature = "vulkan", not(feature = "metal")))]
fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

/// MP: the forbidden `[1,0]` must never appear; every outcome must be in
/// the allowed set. `require_synced`: on a real GPU we require the good
/// synchronized outcome `[1,42]` (consumer saw the flag AND the data) to
/// appear at least once, which proves the test is non-vacuous — the
/// consumer really did observe the release-published flag. We do NOT
/// require the "flag not yet seen" states (`[0,0]` / `[0,42]`): a device
/// that co-schedules each pair tightly may always synchronize, and that
/// is a perfectly valid (indeed ideal) outcome.
fn check_mp(hist: &Histogram, lane: &str, require_synced: bool) {
    let must = if require_synced {
        vec![vec![1, mp::DATA_VALUE]]
    } else {
        vec![]
    };
    let report = assert_outcomes(hist, &mp::allowed(), &mp::forbidden(), &must);
    println!("── MP [{lane}] ──\n{}", report.message);
    assert!(report.ok, "MP litmus failed on {lane}:\n{}", report.message);
}

/// SB rel/acq: the anomaly `[0,0]` is allowed (it is in the allowed set),
/// nothing is forbidden. We never hard-require any specific outcome —
/// the anomaly may legitimately never show, and a tightly co-scheduled
/// device may drive every instance to the same non-anomalous outcome.
/// The only assertion is membership: every outcome is model-permitted.
fn check_sb_relacq(hist: &Histogram, lane: &str, _require_good: bool) {
    let report = assert_outcomes(hist, &sb::allowed(sb::Variant::RelAcq), &[], &[]);
    println!("── SB rel/acq [{lane}] ──\n{}", report.message);
    assert!(
        report.ok,
        "SB rel/acq litmus failed on {lane}:\n{}",
        report.message
    );
}

/// SB SeqCst: the anomaly `[0,0]` is forbidden (count 0).
fn check_sb_seqcst(hist: &Histogram, lane: &str) {
    let report = assert_outcomes(
        hist,
        &sb::allowed(sb::Variant::SeqCst),
        &[sb::anomaly()],
        &[],
    );
    println!("── SB seqcst [{lane}] ──\n{}", report.message);
    assert!(
        report.ok,
        "SB seqcst litmus failed on {lane}:\n{}",
        report.message
    );
}
