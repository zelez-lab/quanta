//! PTRD large-lambda Poisson CPU smoke test.
//!
//! Previously the WASM-route lowering had two structural bugs that
//! together produced all-zero output for PTRD on the CPU backend:
//! (1) Block frames didn't merge locals on close, so reads after
//! `block @N` saw the inner-frame fresh reg instead of the merge
//! anchor; (2) `as` casts emitted the multi-instruction saturation
//! block because `-nontrapping-fptoint` was on, and the lowering
//! couldn't represent the post-redirect ops correctly. Both are
//! fixed (Block merge_locals_post_frame; recognise i32.trunc_sat_*_*
//! variants and let rustc use the single-opcode form).
//!
//! This is the smallest assertion that catches a regression of
//! either fix: 32 quarks, λ=10, full PTRD acceptance loop. Runs in
//! about 8 seconds on Apple M-series with the parallel CPU executor;
//! the slower full statistical tests stay `#[ignore]`d.

#![cfg(feature = "gpu")]

use quanta_rand::gpu_kernel::fill_poisson_u32_large_gpu;

#[test]
fn ptrd_cpu_smoke_lambda10() {
    let gpu = quanta::init_cpu();
    let samples = fill_poisson_u32_large_gpu(&gpu, 32, 0xCAFE_BABE, 10.0)
        .expect("dispatch");
    let nonzero = samples.iter().filter(|&&x| x != 0).count();
    let mean: f64 = samples.iter().map(|&x| x as f64).sum::<f64>() / 32.0;
    // For Poisson(10), Pr(X=0) ≈ 4.5e-5; essentially all samples are non-zero.
    assert!(
        nonzero >= 30,
        "expected ≥30 non-zero samples for λ=10, got {nonzero}/32: {samples:?}"
    );
    // Sample mean has σ ≈ √(10/32) ≈ 0.56; tolerate 3σ.
    assert!(
        (mean - 10.0).abs() < 2.0,
        "sample mean {mean} too far from λ=10"
    );
}
