//! Bit-exact host oracle for the PTRD large-lambda Poisson kernel.
//!
//! `host_ptrd` mirrors `fill_poisson_u32_large` operation-for-
//! operation in host f32 arithmetic over the same Philox stream, so
//! on the CPU backend the kernel must reproduce it exactly. This is
//! a far stronger net than the distributional checks in
//! `poisson_large_lambda.rs`: any lowering bug that reorders,
//! re-gates, or stales a value (the 2026-06-12 redirect-chain hoist
//! bug skewed the sample mean by +0.44σ while still "looking like a
//! Poisson") shows up as a per-quark mismatch here.
//!
//! Scope: CPU backend only. Native GPU backends may legally diverge
//! (fast-math reassociation of the log operands) — that's what the
//! distributional tests are for.

#![cfg(feature = "gpu")]

use quanta_rand::gpu_kernel::fill_poisson_u32_large_gpu;
use quanta_rand::philox4x32::philox4x32_10_first_u32;

const SEED: u64 = 0xCAFE_BABE_DEAD_BEEFu64;

/// Host-side reference of the PTRD kernel body, one quark.
fn host_ptrd(id: u32, seed: u64, lam: f32) -> u32 {
    let (seed_lo, seed_hi) = (seed as u32, (seed >> 32) as u32);
    let smu = lam.sqrt();
    let b = 0.931f32 + 2.53f32 * smu;
    let a = -0.059f32 + 0.02483f32 * b;
    let inv_alpha = 1.1239f32 + 1.1328f32 / (b - 3.4f32);
    let v_r = 0.9277f32 - 3.6224f32 / (b - 2.0f32);
    let log_lam = lam.ln();
    let log_inv_alpha = inv_alpha.ln();
    let log_gamma = |z_in: f32| -> f32 {
        let z = if z_in < 1.0 { 1.0 } else { z_in };
        let half_log_2pi = 0.918938533f32;
        let log_z = z.ln();
        let inv_z = 1.0 / z;
        let inv_z3 = inv_z * inv_z * inv_z;
        (z - 0.5) * log_z - z + half_log_2pi + inv_z * (1.0 / 12.0) - inv_z3 * (1.0 / 360.0)
    };
    let mut result = 0u32;
    let mut done = 0u32;
    for iter in 0..32u32 {
        let r1 = philox4x32_10_first_u32(id, iter, 0, 0, seed_lo, seed_hi);
        let r2 = philox4x32_10_first_u32(id, iter, 1, 0, seed_lo, seed_hi);
        let u = ((r1 >> 8) as f32) * (1.0 / 16_777_216.0f32) - 0.5;
        let v = ((r2 >> 8) as f32) * (1.0 / 16_777_216.0f32);
        let us = 0.5 - u.abs();
        let k_f = ((2.0 * a / us + b) * u + lam + 0.43).floor();
        if k_f >= 0.0 && done == 0 {
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
    }
    result
}

fn assert_kernel_matches_host(lam: f32) {
    let gpu = quanta::init_cpu();
    let n = 256usize;
    let samples = fill_poisson_u32_large_gpu(&gpu, n, SEED, lam).expect("dispatch");
    for (id, &got) in samples.iter().enumerate() {
        let want = host_ptrd(id as u32, SEED, lam);
        assert_eq!(
            got, want,
            "lambda={lam} quark {id}: kernel produced {got}, host reference {want}"
        );
    }
}

#[test]
fn ptrd_kernel_matches_host_reference_lambda_10() {
    assert_kernel_matches_host(10.0);
}

#[test]
fn ptrd_kernel_matches_host_reference_lambda_50() {
    assert_kernel_matches_host(50.0);
}

// ── Generated-oracle parity ──────────────────────────────────────────
//
// The `<kernel>_host_oracle` twins below are auto-emitted by
// `#[quanta::kernel]` (the same rewritten body compiled natively and
// looped over quark ids) — no hand-written replica to keep in sync.
// The hand-written `host_ptrd` above stays as an independent
// cross-check of the twin GENERATOR itself for one kernel.

use quanta_rand::gpu_kernel::{
    FillPoissonLargeU32Data, FillUniformF32Data, FillUniformU32Data, FillUniformU64Data,
    fill_poisson_u32_large_host_oracle, fill_uniform_f32_gpu, fill_uniform_f32_host_oracle,
    fill_uniform_u32_gpu, fill_uniform_u32_host_oracle, fill_uniform_u64_gpu,
    fill_uniform_u64_host_oracle,
};

#[test]
fn ptrd_kernel_matches_generated_oracle() {
    let gpu = quanta::init_cpu();
    let n = 256usize;
    let lam = 10.0f32;
    let got = fill_poisson_u32_large_gpu(&gpu, n, SEED, lam).expect("dispatch");
    let mut want = FillPoissonLargeU32Data {
        out: vec![0u32; n],
        seed_lo: SEED as u32,
        seed_hi: (SEED >> 32) as u32,
        lambda: lam,
    };
    unsafe { fill_poisson_u32_large_host_oracle(n as u32, &mut want) };
    assert_eq!(got, want.out, "PTRD kernel diverged from generated oracle");
}

#[test]
fn uniform_fills_match_generated_oracles() {
    let gpu = quanta::init_cpu();
    let n = 512usize;
    let (lo, hi) = (SEED as u32, (SEED >> 32) as u32);

    let got_u32 = fill_uniform_u32_gpu(&gpu, n, SEED).expect("dispatch u32");
    let mut want_u32 = FillUniformU32Data { out: vec![0u32; n], seed_lo: lo, seed_hi: hi };
    unsafe { fill_uniform_u32_host_oracle(n as u32, &mut want_u32) };
    assert_eq!(got_u32, want_u32.out, "fill_uniform_u32 diverged from oracle");

    let got_u64 = fill_uniform_u64_gpu(&gpu, n, SEED).expect("dispatch u64");
    let mut want_u64 = FillUniformU64Data { out: vec![0u64; n], seed_lo: lo, seed_hi: hi };
    unsafe { fill_uniform_u64_host_oracle(n as u32, &mut want_u64) };
    assert_eq!(got_u64, want_u64.out, "fill_uniform_u64 diverged from oracle");

    let got_f32 = fill_uniform_f32_gpu(&gpu, n, SEED).expect("dispatch f32");
    let mut want_f32 = FillUniformF32Data { out: vec![0.0f32; n], seed_lo: lo, seed_hi: hi };
    unsafe { fill_uniform_f32_host_oracle(n as u32, &mut want_f32) };
    assert_eq!(got_f32, want_f32.out, "fill_uniform_f32 diverged from oracle");
}
