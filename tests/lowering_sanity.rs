//! Lowering regression tests for the Option C v2 redirect mechanism.
//!
//! Originally written as diagnostic witnesses for bugs in v1; under
//! default v2 they all pass and serve as regression coverage.
//!
//! | # | shape                                          | runs in CI |
//! |---|------------------------------------------------|------------|
//! | 1 | while + early `break` over shared mutable      | yes        |
//! | 2 | nested if/else over shared mutable, device fn  | yes        |
//! | 3 | block_compact cross-warp scan                  | `crates/quanta-prims/tests/block_compact.rs` (Metal) |
//!
//! Sanity 3 lives in the prims crate because it needs a Metal device.

#![cfg(feature = "software")]

// ===========================================================================
// Sanity 1 — loop + early break + shared mutable
// ===========================================================================
//
// The bug-#1 shape: a `while` with a shared mutable written on the
// `break` arm. Closed by `hoist_cond_defining_ops` + the 2026-06-03
// Block-merge fix. Regression net for both.

#[quanta::kernel(workgroup = [1])]
fn sanity1_loop_early_break(out: &mut [u32]) {
    let mut result: u32 = 0u32;
    let mut iter: u32 = 0u32;
    while iter < 32u32 {
        if iter == 7u32 {
            result = 42u32;
            iter = 99u32; // exit on next check
        } else {
            iter = iter + 1u32;
        }
    }
    out[0] = result;
}

#[test]
fn sanity1_loop_early_break_runs() {
    let gpu = quanta::init_cpu();
    let out = gpu.field::<u32>(1).unwrap();
    out.write(&[0u32]).unwrap();

    let mut wave = sanity1_loop_early_break(&gpu).unwrap();
    wave.bind(0, &out);
    gpu.dispatch(&wave, 1).unwrap().wait().unwrap();

    assert_eq!(
        out.read().unwrap(),
        vec![42u32],
        "early-break write of 42 must reach the post-loop read"
    );
}

// ===========================================================================
// Sanity 2 — PTRD-shape nested if/else with device-fn call
// ===========================================================================
//
// Under v1 this errored at proc-macro time with `r? used before def
// at BinOp.a`: the redirect-chain splice from the device-fn inline
// landed ops in a sibling Branch arm of the outer scope. v2's
// position-aware mechanism (Option C, sessions 1-4) encodes the
// control-flow graph correctly; the IR is now scope-valid and the
// kernel runs.
//
// Trigger: nested `if/else` over **shared mutables** inside a
// `while`, **with a `#[quanta::device]` call** in the deepest arm.
// Shorter shapes (no device fn, single-level if) compile fine under
// either path.

#[quanta::device]
fn sanity2_envelope(z_in: f32) -> f32 {
    let z: f32 = if z_in < 1.0f32 { 1.0f32 } else { z_in };
    let log_z: f32 = z.ln();
    let inv_z: f32 = 1.0f32 / z;
    (z - 0.5f32) * log_z - z + 0.9189385f32 + inv_z * 0.0833333f32
}

#[quanta::kernel(workgroup = [1])]
fn sanity2_ptrd_shape_nested_if(input: &[f32], out: &mut [u32]) {
    fn sanity2_envelope(z_in: f32) -> f32 {
        let z: f32 = if z_in < 1.0f32 { 1.0f32 } else { z_in };
        let log_z: f32 = z.ln();
        let inv_z: f32 = 1.0f32 / z;
        (z - 0.5f32) * log_z - z + 0.9189385f32 + inv_z * 0.0833333f32
    }
    let id = quark_id();
    let lam: f32 = input[0];
    let a: f32 = input[1];
    let b: f32 = input[2];
    let inv_alpha: f32 = input[3];
    let v_r: f32 = input[4];
    let log_lam: f32 = ln(lam);
    let log_inv_alpha: f32 = ln(inv_alpha);
    let mut iter: u32 = 0u32;
    let mut result: u32 = 0u32;
    let mut done: u32 = 0u32;
    while iter < 32u32 {
        let u: f32 = (iter as f32) * 0.03125f32 - 0.5f32;
        let v: f32 = (iter as f32) * 0.03125f32;
        let us: f32 = 0.5f32 - fabs(u);
        let k_f: f32 = floor((2.0f32 * a / us + b) * u + lam + 0.43f32);
        if k_f >= 0.0f32 && done == 0u32 {
            if us >= 0.07f32 && v <= v_r {
                result = k_f as u32;
                done = 1u32;
            } else if !(us < 0.013f32 && v > us) {
                let lhs: f32 = ln(v) + log_inv_alpha - ln(a / (us * us) + b);
                let rhs: f32 = (0.0f32 - lam) + (k_f * log_lam) - sanity2_envelope(k_f + 1.0f32);
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

// The PROC-MACRO-PHASE assertion: the kernel must compile (no
// scope_check violation). v1 errored here; v2 doesn't. This part of
// sanity 2 IS a passing regression test.
//
// The RUNTIME assertion below regressed under early v2 (writes in
// deeply-nested if/else arms didn't reach the post-loop read); fixed
// by replacing the transitive cond-slice hoist with cond
// materialization (`materialize_cond_for_v2` in lower.rs). See
// tests/v2_runtime_bisect.rs for the surgical isolation of that bug.
#[test]
fn sanity2_ptrd_shape_nested_if_compiles() {
    // Successful proc-macro expansion of the kernel below is the
    // assertion: if the kernel item's macro expansion fails, this
    // test won't even compile. Body unused.
    let _ = &SANITY2_PTRD_SHAPE_NESTED_IF_BINARY;
}

#[test]
fn sanity2_ptrd_shape_nested_if_runs() {
    let gpu = quanta::init_cpu();
    let input = gpu.field::<f32>(5).unwrap();
    let out = gpu.field::<u32>(1).unwrap();
    input
        .write(&[10.0f32, 0.119f32, 8.929f32, 1.328f32, 0.286f32])
        .unwrap();
    // Sentinel — distinguishes "kernel never wrote" from "kernel
    // wrote 0".
    out.write(&[0xDEAD_BEEFu32]).unwrap();

    let mut wave = sanity2_ptrd_shape_nested_if(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &out);
    gpu.dispatch(&wave, 1).unwrap().wait().unwrap();

    let got = out.read().unwrap();
    assert_ne!(got[0], 0xDEAD_BEEFu32, "kernel did not write to out");
    assert_ne!(
        got[0], 0u32,
        "v2 runtime bug: result stuck at 0 despite the inner-if write \
         being present in the IR. See memory \
         redirect-chain-v2-closed-2026-06-12."
    );
}
