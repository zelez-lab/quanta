//! Differential-oracle parity: every `#[quanta::kernel]` in the
//! single-quark-pure subset gets an auto-generated
//! `<name>_host_oracle` twin — the same rewritten body compiled
//! natively by rustc and looped over quark ids. Running the kernel
//! on the CPU backend must reproduce the oracle bit-exactly; any
//! divergence is a lowering / IR-execution bug.
//!
//! This is the systematized form of the hand-written replicas that
//! caught the 2026-06 redirect-chain miscompiles (distribution-level
//! wrongness that passed every conventional assertion). The kernels
//! below deliberately exercise the shapes those bugs lived in:
//! nested if/else over shared mutables inside a while loop, device-fn
//! calls in deep arms, and early-exit loop conditions.

#![cfg(feature = "software")]

// ── saxpy: straight-line arithmetic baseline ─────────────────────────

#[quanta::kernel(workgroup = [4])]
fn oracle_demo_saxpy(x: &[f32], y: &mut [f32], a: f32) {
    let i = quark_id() as usize;
    y[i] = a * x[i] + y[i];
}

#[test]
fn saxpy_matches_generated_oracle() {
    let gpu = quanta::init_cpu();
    let n = 64usize;
    let xs: Vec<f32> = (0..n).map(|i| (i as f32) * 0.37 - 3.0).collect();
    let ys: Vec<f32> = (0..n).map(|i| (i as f32) * -1.61 + 0.5).collect();
    let a = 2.25f32;

    let x = gpu.field::<f32>(n).unwrap();
    let y = gpu.field::<f32>(n).unwrap();
    x.write(&xs).unwrap();
    y.write(&ys).unwrap();
    let mut wave = oracle_demo_saxpy(&gpu).unwrap();
    wave.bind(0, &x);
    wave.bind(1, &y);
    wave.set_value(2, a);
    gpu.dispatch(&wave, n as u32).unwrap().wait().unwrap();
    let got = y.read().unwrap();

    let mut want = ys.clone();
    unsafe { oracle_demo_saxpy_host_oracle(n as u32, &xs, &mut want, a) };
    assert_eq!(got, want, "saxpy kernel diverged from host oracle");
}

// ── branchy: the redirect-chain bug shapes ───────────────────────────
//
// Nested if/else-if over shared mutables inside a while loop, a
// device-fn with an inner value-select called from the deepest arm,
// and an early-exit `&&` loop condition — the exact surfaces of the
// φ-select hoist bug, the intermediate-frame stale-read bug, and the
// label-lossy loop-exit bug.

#[quanta::kernel(workgroup = [1])]
fn oracle_demo_branchy(input: &[f32], out: &mut [u32]) {
    fn clampy(z_in: f32) -> f32 {
        let z: f32 = if z_in < 1.0f32 { 1.0f32 } else { z_in };
        let log_z: f32 = z.ln();
        (z - 0.5f32) * log_z + 1.0f32 / z
    }
    let id = quark_id();
    let base: f32 = input[id as usize];
    let mut iter: u32 = 0u32;
    let mut result: u32 = 0u32;
    let mut done: u32 = 0u32;
    while iter < 16u32 && done == 0u32 {
        let u: f32 = base + (iter as f32) * 0.125f32 - 1.0f32;
        let us: f32 = 0.5f32 - fabs(u) * 0.25f32;
        let k_f: f32 = floor(u * 3.0f32 + 4.0f32);
        if k_f >= 0.0f32 {
            if us >= 0.45f32 && u <= 0.0f32 {
                result = k_f as u32 + 100u32;
                done = 1u32;
            } else if !(us < 0.2f32 && u > us) {
                let lhs: f32 = ln(us + 1.5f32);
                let rhs: f32 = clampy(k_f + 0.5f32) - 2.0f32;
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
fn branchy_matches_generated_oracle() {
    let gpu = quanta::init_cpu();
    let n = 96usize;
    let inputs: Vec<f32> = (0..n).map(|i| (i as f32) * 0.041 - 1.7).collect();

    let input = gpu.field::<f32>(n).unwrap();
    let out = gpu.field::<u32>(n).unwrap();
    input.write(&inputs).unwrap();
    let mut wave = oracle_demo_branchy(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &out);
    gpu.dispatch(&wave, n as u32).unwrap().wait().unwrap();
    let got = out.read().unwrap();

    let mut want = vec![0u32; n];
    unsafe { oracle_demo_branchy_host_oracle(n as u32, &inputs, &mut want) };
    let diffs: Vec<String> = got
        .iter()
        .zip(want.iter())
        .enumerate()
        .filter(|(_, (g, w))| g != w)
        .map(|(i, (g, w))| format!("quark {i}: kernel={g} oracle={w} input={}", inputs[i]))
        .collect();
    assert!(
        diffs.is_empty(),
        "branchy kernel diverged from host oracle at {} of {n} quarks:\n{}",
        diffs.len(),
        diffs.join("\n")
    );
    // Guard against the all-zero degenerate case silently passing.
    assert!(
        want.iter().any(|&v| v != 0),
        "expected some quarks to accept a value"
    );
}
