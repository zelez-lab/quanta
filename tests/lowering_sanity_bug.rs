//! Sanity 2 — the redirect-chain encoding bug captured as a kernel.
//!
//! This file is **build-gated**: by default it compiles to nothing.
//! Enable it by passing the cfg flag at compile time:
//!
//!   RUSTFLAGS='--cfg redirect_chain_bug' cargo test -p quanta \
//!     --features software --test lowering_sanity_bug
//!
//! Why a cfg gate? The `#[quanta::kernel]` proc macro rejects this
//! kernel at expansion time with a scope_check violation
//! (`r? used before def at BinOp.a`). Without a gate, just *building*
//! this file fails — which would break unrelated `cargo build --tests`
//! / `cargo check` flows. With the gate, the file is normally dead and
//! the bug witness can still be exercised on demand.
//!
//! ## What this kernel pins down
//!
//! - Trigger: nested `if/else` over **shared mutables** inside a
//!   `while`, **with a `#[quanta::device]` call** in the deepest arm.
//!   Shorter shapes (no device fn, or single-level if) compile fine.
//! - Failure mode: scope_check rejects at proc-macro time. IR dump
//!   under `QUANTA_SCOPE_DUMP=1` shows the
//!   `rejects_nested_branch_sibling_sub_scope_def` pattern from
//!   `crates/quanta-ir/src/scope_check.rs` — a BinOp in `then_ops`
//!   uses a register defined in a sibling `else_ops`.
//! - Diagnosis: the redirect-chain splice from the device-fn inline
//!   lands ops in a sibling Branch arm.
//!
//! See memory notes `redirect-chain-substrate-redesign` and
//! `lowering-bug-nested-if-2026-06-01` for the full architectural
//! account and the Option C fix plan.

#![allow(unexpected_cfgs)]
#![cfg(all(feature = "software", redirect_chain_bug))]

// Local device fn — exercises the device-fn-inlining path that the
// memory note flags as part of the redirect-chain trigger.
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

#[test]
fn sanity2_ptrd_shape_nested_if_runs() {
    let gpu = quanta::init_cpu();
    let input = gpu.field::<f32>(5).unwrap();
    let out = gpu.field::<u32>(1).unwrap();
    input
        .write(&[10.0f32, 0.119f32, 8.929f32, 1.328f32, 0.286f32])
        .unwrap();
    out.write(&[0u32]).unwrap();

    let mut wave = sanity2_ptrd_shape_nested_if(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &out);
    gpu.dispatch(&wave, 1).unwrap().wait().unwrap();

    let _ = out.read().unwrap();
}
