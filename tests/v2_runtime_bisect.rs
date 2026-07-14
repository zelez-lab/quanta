//! Bisecting the v2 runtime bug surfaced 2026-06-12.
//!
//! Background: under default v2 (HEAD 987288f), the natural-shape
//! PTRD kernel compiles cleanly but produces all-zero output at
//! runtime on CPU. This file isolates the trigger.
//!
//! ## Findings (session 7, 2026-06-12)
//!
//! - L0 (full PTRD-shape, stubs for both philox + log_gamma): PASS
//! - L1 (add real philox device fn, stub log_gamma): PASS
//! - L2.1 (real philox + stub gamma): PASS
//! - L2.2 (stub philox + real log_gamma): **FAIL**
//! - L3 (real log_gamma BUT with the inner `if z<1 {1} else {z}`
//!   clamp removed): PASS
//!
//! Conclusion: **the trigger is the inner `if cond { A } else { B }`
//! inside the device fn `log_gamma`**, combined with the surrounding
//! arithmetic context that consumes the device-fn result via lhs/rhs
//! comparison in a deeply-nested else-if arm with shared mutables.
//!
//! Narrower attempts to minimize the kernel body around the call site
//! (L4, L5, L6 — not committed) all passed, so the failure requires
//! more context than just "device fn with if/else called from deep
//! nested arm". The L2.2 ↔ L3 diff is the surgical isolation.
//!
//! ## Status of tests
//!
//! The bug is FIXED — and so is its sibling (the hoisted cond slice
//! reading a zero-init for a reg whose real defining ops LLVM had
//! sunk into an intermediate frame, which skewed PTRD's sample mean
//! by +0.44σ while passing every non-exact assertion). Both fell to
//! the same change: v2 no longer hoists the cond's backward slice
//! at all. `materialize_cond_for_v2` declares a zero-init register
//! at the target frame and `Copy`s the cond into it at the source
//! position, so no computation ever moves across control flow. All
//! levels pass; L2.2 stays as the regression witness, and
//! `crates/sci/quanta-rand/tests/ptrd_host_oracle.rs` pins the full
//! production kernel to a bit-exact host reference.
//!
//! The loop-condition sibling bug (`while iter < 32 && done == 0`
//! reading a shared mutable written in nested arms) is ALSO closed:
//! LLVM merges `iter`/`result` into one local and encodes the two
//! loop exits as brs to different labels; the lowering's label-lossy
//! `Break` ran the exhaustion continuation (`result = 0`) on the
//! accept path too. Fixed by the exit-flag record
//! (`emit_loop_crossing_exit` in lower.rs). L4 below is the
//! lowering-level witness; the production PTRD kernel now uses the
//! early-exit loop condition and is pinned bit-exact by
//! `crates/sci/quanta-rand/tests/ptrd_host_oracle.rs`.
//!
//! Run only:
//!   cargo test -p quanta --features software --test v2_runtime_bisect
//!   cargo test -p quanta --features software --test v2_runtime_bisect -- --ignored

#![cfg(feature = "software")]

// ---------------------------------------------------------------------------
// L0 — PTRD-shape with arithmetic stubs for both device fns. PASS.
// ---------------------------------------------------------------------------

#[quanta::kernel(workgroup = [1])]
fn bisect_l0_full_ptrd(input: &[f32], out: &mut [u32]) {
    let id = quark_id();
    let lam: f32 = input[0];
    let smu: f32 = sqrt(lam);
    let b: f32 = 0.931f32 + 2.53f32 * smu;
    let a: f32 = -0.059f32 + 0.02483f32 * b;
    let inv_alpha: f32 = 1.1239f32 + 1.1328f32 / (b - 3.4f32);
    let v_r: f32 = 0.9277f32 - 3.6224f32 / (b - 2.0f32);
    let log_lam: f32 = ln(lam);
    let log_inv_alpha: f32 = ln(inv_alpha);
    let mut iter: u32 = 0u32;
    let mut result: u32 = 0u32;
    let mut done: u32 = 0u32;
    while iter < 32u32 {
        let scale: f32 = 1.0f32 / 16_777_216.0f32;
        let u: f32 = ((iter * 1_000_003u32) as f32) * scale - 0.5f32;
        let v: f32 = ((iter * 2_000_011u32) as f32) * scale;
        let us: f32 = 0.5f32 - fabs(u);
        let k_f: f32 = floor((2.0f32 * a / us + b) * u + lam + 0.43f32);
        if k_f >= 0.0f32 && done == 0u32 {
            if us >= 0.07f32 && v <= v_r {
                result = k_f as u32;
                done = 1u32;
            } else if !(us < 0.013f32 && v > us) {
                let lhs: f32 = ln(v) + log_inv_alpha - ln(a / (us * us) + b);
                let rhs: f32 = (0.0f32 - lam) + (k_f * log_lam) - (k_f * 0.5f32);
                if lhs <= rhs {
                    result = k_f as u32;
                    done = 1u32;
                }
            }
        }
        iter = iter + 1u32;
    }
    out[id as usize] = result;
}

#[test]
fn bisect_l0_full_ptrd_passes() {
    let gpu = quanta::init_cpu();
    let input = gpu.field::<f32>(1).unwrap();
    let out = gpu.field::<u32>(1).unwrap();
    input.write(&[10.0f32]).unwrap();
    out.write(&[0xDEAD_BEEFu32]).unwrap();
    let mut wave = bisect_l0_full_ptrd(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &out);
    gpu.dispatch(&wave, 1).unwrap().wait().unwrap();
    let got = out.read().unwrap();
    assert_ne!(got[0], 0xDEAD_BEEFu32, "kernel did not write");
    assert_ne!(got[0], 0u32, "L0: result stuck at 0");
}

// ---------------------------------------------------------------------------
// L2.2 — Stub philox + REAL log_gamma (with inner if-else clamp). FAIL.
// ---------------------------------------------------------------------------

#[quanta::kernel(workgroup = [1])]
fn bisect_l2_2_stub_philox_real_gamma(input: &[f32], out: &mut [u32]) {
    fn bisect_log_gamma(z_in: f32) -> f32 {
        // THIS LINE IS THE TRIGGER:
        let z: f32 = if z_in < 1.0f32 { 1.0f32 } else { z_in };
        let half_log_2pi: f32 = 0.918938533f32;
        let log_z: f32 = z.ln();
        let inv_z: f32 = 1.0f32 / z;
        let inv_z3: f32 = inv_z * inv_z * inv_z;
        (z - 0.5f32) * log_z - z + half_log_2pi + inv_z * (1.0f32 / 12.0f32)
            - inv_z3 * (1.0f32 / 360.0f32)
    }
    let id = quark_id();
    let lam: f32 = input[0];
    let smu: f32 = sqrt(lam);
    let b: f32 = 0.931f32 + 2.53f32 * smu;
    let a: f32 = -0.059f32 + 0.02483f32 * b;
    let inv_alpha: f32 = 1.1239f32 + 1.1328f32 / (b - 3.4f32);
    let v_r: f32 = 0.9277f32 - 3.6224f32 / (b - 2.0f32);
    let log_lam: f32 = ln(lam);
    let log_inv_alpha: f32 = ln(inv_alpha);
    let mut iter: u32 = 0u32;
    let mut result: u32 = 0u32;
    let mut done: u32 = 0u32;
    while iter < 32u32 {
        let scale: f32 = 1.0f32 / 16_777_216.0f32;
        let u: f32 = ((iter * 1_000_003u32) as f32) * scale - 0.5f32;
        let v: f32 = ((iter * 2_000_011u32) as f32) * scale;
        let us: f32 = 0.5f32 - fabs(u);
        let k_f: f32 = floor((2.0f32 * a / us + b) * u + lam + 0.43f32);
        if k_f >= 0.0f32 && done == 0u32 {
            if us >= 0.07f32 && v <= v_r {
                result = k_f as u32;
                done = 1u32;
            } else if !(us < 0.013f32 && v > us) {
                let lhs: f32 = ln(v) + log_inv_alpha - ln(a / (us * us) + b);
                let rhs: f32 = (0.0f32 - lam) + (k_f * log_lam) - bisect_log_gamma(k_f + 1.0f32);
                if lhs <= rhs {
                    result = k_f as u32;
                    done = 1u32;
                }
            }
        }
        iter = iter + 1u32;
    }
    out[id as usize] = result;
}

#[test]
fn bisect_l2_2_stub_philox_real_gamma_runs() {
    let gpu = quanta::init_cpu();
    let input = gpu.field::<f32>(1).unwrap();
    let out = gpu.field::<u32>(1).unwrap();
    input.write(&[10.0f32]).unwrap();
    out.write(&[0xDEAD_BEEFu32]).unwrap();
    let mut wave = bisect_l2_2_stub_philox_real_gamma(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &out);
    gpu.dispatch(&wave, 1).unwrap().wait().unwrap();
    let got = out.read().unwrap();
    assert_ne!(got[0], 0xDEAD_BEEFu32, "kernel did not write");
    assert_ne!(got[0], 0u32, "L2.2: result stuck at 0 (the v2 runtime bug)");
}

// ---------------------------------------------------------------------------
// L4 — L2.2 body with the EARLY-EXIT loop condition
// (`while iter < 32 && done == 0`). Witness for the label-lossy
// loop-exit bug: LLVM encodes the accept path as a br past the
// natural-exhaustion continuation; before `emit_loop_crossing_exit`,
// the lowering's plain `Break` ran that continuation on the accept
// path too. Asserted against an exact host replica of the body, so
// any wrong-value lowering (not just all-zeros) trips it.
// ---------------------------------------------------------------------------

#[quanta::kernel(workgroup = [1])]
fn bisect_l4_early_exit_loop_cond(input: &[f32], out: &mut [u32]) {
    fn bisect_log_gamma_l4(z_in: f32) -> f32 {
        let z: f32 = if z_in < 1.0f32 { 1.0f32 } else { z_in };
        let half_log_2pi: f32 = 0.918938533f32;
        let log_z: f32 = z.ln();
        let inv_z: f32 = 1.0f32 / z;
        let inv_z3: f32 = inv_z * inv_z * inv_z;
        (z - 0.5f32) * log_z - z + half_log_2pi + inv_z * (1.0f32 / 12.0f32)
            - inv_z3 * (1.0f32 / 360.0f32)
    }
    let id = quark_id();
    let lam: f32 = input[0];
    let smu: f32 = sqrt(lam);
    let b: f32 = 0.931f32 + 2.53f32 * smu;
    let a: f32 = -0.059f32 + 0.02483f32 * b;
    let inv_alpha: f32 = 1.1239f32 + 1.1328f32 / (b - 3.4f32);
    let v_r: f32 = 0.9277f32 - 3.6224f32 / (b - 2.0f32);
    let log_lam: f32 = ln(lam);
    let log_inv_alpha: f32 = ln(inv_alpha);
    let mut iter: u32 = 0u32;
    let mut result: u32 = 0u32;
    let mut done: u32 = 0u32;
    while iter < 32u32 && done == 0u32 {
        let scale: f32 = 1.0f32 / 16_777_216.0f32;
        let u: f32 = ((iter * 1_000_003u32) as f32) * scale - 0.5f32;
        let v: f32 = ((iter * 2_000_011u32) as f32) * scale;
        let us: f32 = 0.5f32 - fabs(u);
        let k_f: f32 = floor((2.0f32 * a / us + b) * u + lam + 0.43f32);
        if k_f >= 0.0f32 {
            if us >= 0.07f32 && v <= v_r {
                result = k_f as u32;
                done = 1u32;
            } else if !(us < 0.013f32 && v > us) {
                let lhs: f32 = ln(v) + log_inv_alpha - ln(a / (us * us) + b);
                let rhs: f32 = (0.0f32 - lam) + (k_f * log_lam) - bisect_log_gamma_l4(k_f + 1.0f32);
                if lhs <= rhs {
                    result = k_f as u32;
                    done = 1u32;
                }
            }
        }
        iter = iter + 1u32;
    }
    out[id as usize] = result;
}

/// Host replica of `bisect_l4_early_exit_loop_cond` for one quark.
fn l4_host_reference(lam: f32) -> u32 {
    let log_gamma = |z_in: f32| -> f32 {
        let z = if z_in < 1.0 { 1.0f32 } else { z_in };
        let half_log_2pi = 0.918_938_5_f32;
        let log_z = z.ln();
        let inv_z = 1.0 / z;
        let inv_z3 = inv_z * inv_z * inv_z;
        (z - 0.5) * log_z - z + half_log_2pi + inv_z * (1.0 / 12.0) - inv_z3 * (1.0 / 360.0)
    };
    let smu = lam.sqrt();
    let b = 0.931f32 + 2.53f32 * smu;
    let a = -0.059f32 + 0.02483f32 * b;
    let inv_alpha = 1.1239f32 + 1.1328f32 / (b - 3.4f32);
    let v_r = 0.9277f32 - 3.6224f32 / (b - 2.0f32);
    let log_lam = lam.ln();
    let log_inv_alpha = inv_alpha.ln();
    let mut iter = 0u32;
    let mut result = 0u32;
    let mut done = 0u32;
    while iter < 32 && done == 0 {
        let scale = 1.0f32 / 16_777_216.0f32;
        let u = ((iter.wrapping_mul(1_000_003)) as f32) * scale - 0.5;
        let v = ((iter.wrapping_mul(2_000_011)) as f32) * scale;
        let us = 0.5 - u.abs();
        let k_f = ((2.0 * a / us + b) * u + lam + 0.43).floor();
        if k_f >= 0.0 {
            if us >= 0.07 && v <= v_r {
                result = k_f as u32;
                done = 1;
            } else if !(us < 0.013 && v > us) {
                let lhs = v.ln() + log_inv_alpha - (a / (us * us) + b).ln();
                let rhs = (0.0 - lam) + (k_f * log_lam) - log_gamma(k_f + 1.0);
                if lhs <= rhs {
                    result = k_f as u32;
                    done = 1;
                }
            }
        }
        iter += 1;
    }
    result
}

#[test]
fn bisect_l4_early_exit_loop_cond_matches_host() {
    let gpu = quanta::init_cpu();
    let input = gpu.field::<f32>(1).unwrap();
    let out = gpu.field::<u32>(1).unwrap();
    input.write(&[10.0f32]).unwrap();
    out.write(&[0xDEAD_BEEFu32]).unwrap();
    let mut wave = bisect_l4_early_exit_loop_cond(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &out);
    gpu.dispatch(&wave, 1).unwrap().wait().unwrap();
    let got = out.read().unwrap();
    let want = l4_host_reference(10.0);
    assert_ne!(want, 0, "host reference should accept a draw for λ=10");
    assert_eq!(
        got[0], want,
        "L4: early-exit loop produced {} but host reference says {want}",
        got[0]
    );
}

// ---------------------------------------------------------------------------
// L3 — L2.2 with the inner `if z_in < 1.0` clamp REMOVED from gamma. PASS.
// This is the surgical isolation: the difference between L2.2 and L3 is
// a single line in the device fn.
// ---------------------------------------------------------------------------

#[quanta::kernel(workgroup = [1])]
fn bisect_l3_gamma_no_clamp(input: &[f32], out: &mut [u32]) {
    fn bisect_log_gamma_noclamp(z_in: f32) -> f32 {
        // No `if z_in < 1.0 { 1.0 } else { z_in }` — use z_in directly.
        let z: f32 = z_in;
        let half_log_2pi: f32 = 0.918938533f32;
        let log_z: f32 = z.ln();
        let inv_z: f32 = 1.0f32 / z;
        let inv_z3: f32 = inv_z * inv_z * inv_z;
        (z - 0.5f32) * log_z - z + half_log_2pi + inv_z * (1.0f32 / 12.0f32)
            - inv_z3 * (1.0f32 / 360.0f32)
    }
    let id = quark_id();
    let lam: f32 = input[0];
    let smu: f32 = sqrt(lam);
    let b: f32 = 0.931f32 + 2.53f32 * smu;
    let a: f32 = -0.059f32 + 0.02483f32 * b;
    let inv_alpha: f32 = 1.1239f32 + 1.1328f32 / (b - 3.4f32);
    let v_r: f32 = 0.9277f32 - 3.6224f32 / (b - 2.0f32);
    let log_lam: f32 = ln(lam);
    let log_inv_alpha: f32 = ln(inv_alpha);
    let mut iter: u32 = 0u32;
    let mut result: u32 = 0u32;
    let mut done: u32 = 0u32;
    while iter < 32u32 {
        let scale: f32 = 1.0f32 / 16_777_216.0f32;
        let u: f32 = ((iter * 1_000_003u32) as f32) * scale - 0.5f32;
        let v: f32 = ((iter * 2_000_011u32) as f32) * scale;
        let us: f32 = 0.5f32 - fabs(u);
        let k_f: f32 = floor((2.0f32 * a / us + b) * u + lam + 0.43f32);
        if k_f >= 0.0f32 && done == 0u32 {
            if us >= 0.07f32 && v <= v_r {
                result = k_f as u32;
                done = 1u32;
            } else if !(us < 0.013f32 && v > us) {
                let lhs: f32 = ln(v) + log_inv_alpha - ln(a / (us * us) + b);
                let rhs: f32 =
                    (0.0f32 - lam) + (k_f * log_lam) - bisect_log_gamma_noclamp(k_f + 1.0f32);
                if lhs <= rhs {
                    result = k_f as u32;
                    done = 1u32;
                }
            }
        }
        iter = iter + 1u32;
    }
    out[id as usize] = result;
}

#[test]
fn bisect_l3_gamma_no_clamp_passes() {
    let gpu = quanta::init_cpu();
    let input = gpu.field::<f32>(1).unwrap();
    let out = gpu.field::<u32>(1).unwrap();
    input.write(&[10.0f32]).unwrap();
    out.write(&[0xDEAD_BEEFu32]).unwrap();
    let mut wave = bisect_l3_gamma_no_clamp(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &out);
    gpu.dispatch(&wave, 1).unwrap().wait().unwrap();
    let got = out.read().unwrap();
    assert_ne!(got[0], 0xDEAD_BEEFu32, "kernel did not write");
    assert_ne!(got[0], 0u32, "L3: result stuck at 0");
}
