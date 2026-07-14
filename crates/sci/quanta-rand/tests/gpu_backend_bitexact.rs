//! Hardware-backend bit-exactness for the RNG fills.
//!
//! `tests/uniform_fill_correctness.rs` proves every fill against the
//! host Philox reference on the CPU software backend. This suite
//! runs the same differential comparison against a REAL hardware
//! backend — build with `--features gpu-metal` (Mac) or
//! `--features gpu-vulkan` (e.g. the Raspberry Pi's V3D) — and is
//! the proof that the pure-32-bit Philox mulhi produces identical
//! bits on devices without `shaderInt64`.
//!
//! With the plain `gpu` feature (software backend only, no hardware
//! discovery) `quanta::init()` finds no device and every test skips
//! gracefully — same convention as the quanta-prims suites.
//!
//! Coverage:
//! - u32 / f32 uniform fills: bit-for-bit equal to `init_cpu()`
//!   output AND the host Philox reference (the CPU path is
//!   KAT-validated, so equality here is transitively KAT-exact).
//! - u64 fill: bit-exact where `supports_i64()`, a clean
//!   `NotSupported` where not (Metal, V3D).
//! - f64 fills: uniform bit-exact where `supports_f64()`; ALL f64
//!   variants must refuse with `NotSupported` where not — never
//!   garbage bits.
//! - f32 distributions (normal/exponential/lognormal/bernoulli/
//!   poisson): must dispatch and return sane values on the hardware
//!   backend. Not compared bit-for-bit — they use transcendentals
//!   (ln/cos/sin/exp) whose precision legitimately differs across
//!   backends; distribution shape is covered by the K-S suite.

#![cfg(feature = "gpu")]

use quanta::QuantaErrorKind;
use quanta_rand::philox4x32::philox4x32_10_first_u32;
use quanta_rand::{
    fill_bernoulli_u32_gpu, fill_exponential_f32_gpu, fill_exponential_f64_gpu,
    fill_lognormal_f32_gpu, fill_lognormal_f64_gpu, fill_normal_f32_gpu, fill_normal_f64_gpu,
    fill_poisson_u32_gpu, fill_uniform_f32_gpu, fill_uniform_f64_gpu, fill_uniform_u32_gpu,
    fill_uniform_u64_gpu,
};

const SEEDS: [u64; 3] = [0, 0xCAFE_BABE_DEAD_BEEF, u64::MAX];
const LEN: usize = 4096;

/// First discovered hardware device, or `None` (skip) when the
/// build has no hardware backend feature enabled.
fn try_hw_gpu() -> Option<quanta::Gpu> {
    let gpu = quanta::init().ok()?;
    eprintln!(
        "device: {} (supports_i64={}, supports_f64={})",
        gpu.name(),
        gpu.supports_i64(),
        gpu.supports_f64()
    );
    Some(gpu)
}

#[test]
fn uniform_u32_bitexact_vs_cpu_and_host() {
    let Some(gpu) = try_hw_gpu() else { return };
    let cpu = quanta::init_cpu();
    for seed in SEEDS {
        let hw = fill_uniform_u32_gpu(&gpu, LEN, seed).expect("hw u32 fill");
        let sw = fill_uniform_u32_gpu(&cpu, LEN, seed).expect("cpu u32 fill");
        assert_eq!(hw, sw, "u32 fill diverges from CPU backend, seed={seed:#x}");

        let (lo, hi) = (seed as u32, (seed >> 32) as u32);
        let host: Vec<u32> = (0..LEN as u32)
            .map(|id| philox4x32_10_first_u32(id, 0, 0, 0, lo, hi))
            .collect();
        assert_eq!(
            hw, host,
            "u32 fill diverges from host Philox, seed={seed:#x}"
        );
    }
}

#[test]
fn uniform_f32_bitexact_vs_cpu() {
    let Some(gpu) = try_hw_gpu() else { return };
    let cpu = quanta::init_cpu();
    for seed in SEEDS {
        let hw = fill_uniform_f32_gpu(&gpu, LEN, seed).expect("hw f32 fill");
        let sw = fill_uniform_f32_gpu(&cpu, LEN, seed).expect("cpu f32 fill");
        // Compare raw bit patterns — stricter than float equality.
        let hw_bits: Vec<u32> = hw.iter().map(|v| v.to_bits()).collect();
        let sw_bits: Vec<u32> = sw.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            hw_bits, sw_bits,
            "f32 fill bits diverge from CPU backend, seed={seed:#x}"
        );
        for &v in &hw {
            assert!((0.0..1.0).contains(&v), "f32 out of [0,1): {v}");
        }
    }
}

#[test]
fn uniform_u64_bitexact_or_cleanly_refused() {
    let Some(gpu) = try_hw_gpu() else { return };
    eprintln!(
        "device: supports_i64={} — u64 arm: {}",
        gpu.supports_i64(),
        if gpu.supports_i64() {
            "bit-exact"
        } else {
            "refusal"
        }
    );
    if gpu.supports_i64() {
        let cpu = quanta::init_cpu();
        for seed in SEEDS {
            let hw = fill_uniform_u64_gpu(&gpu, LEN, seed).expect("hw u64 fill");
            let sw = fill_uniform_u64_gpu(&cpu, LEN, seed).expect("cpu u64 fill");
            assert_eq!(hw, sw, "u64 fill diverges from CPU backend, seed={seed:#x}");
        }
    } else {
        let err = fill_uniform_u64_gpu(&gpu, LEN, SEEDS[1])
            .expect_err("u64 fill must refuse without shaderInt64");
        assert!(
            matches!(err.kind, QuantaErrorKind::NotSupported(_)),
            "u64 refusal must be NotSupported, got {:?}",
            err.kind
        );
    }
}

#[test]
fn f64_fills_bitexact_or_cleanly_refused() {
    let Some(gpu) = try_hw_gpu() else { return };
    let f64_ok = gpu.supports_f64() && gpu.supports_i64();
    eprintln!(
        "device: supports_f64={}, supports_i64={} — f64 arm: {}",
        gpu.supports_f64(),
        gpu.supports_i64(),
        if f64_ok { "bit-exact" } else { "refusal" }
    );
    if f64_ok {
        // The uniform f64 path is pure arithmetic (u64 pack, shift,
        // convert, one mul + add) — bit-exact is required.
        let cpu = quanta::init_cpu();
        for seed in SEEDS {
            let hw = fill_uniform_f64_gpu(&gpu, LEN, seed).expect("hw f64 fill");
            let sw = fill_uniform_f64_gpu(&cpu, LEN, seed).expect("cpu f64 fill");
            let hw_bits: Vec<u64> = hw.iter().map(|v| v.to_bits()).collect();
            let sw_bits: Vec<u64> = sw.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                hw_bits, sw_bits,
                "f64 fill bits diverge from CPU backend, seed={seed:#x}"
            );
        }
    } else {
        // Every f64 variant must refuse — NotSupported, not garbage.
        let seed = SEEDS[1];
        let errs = [
            fill_uniform_f64_gpu(&gpu, LEN, seed).err(),
            fill_normal_f64_gpu(&gpu, LEN, seed).err(),
            fill_exponential_f64_gpu(&gpu, LEN, seed, 1.5).err(),
            fill_lognormal_f64_gpu(&gpu, LEN, seed, 0.0, 1.0).err(),
        ];
        for (i, err) in errs.into_iter().enumerate() {
            let err = err.unwrap_or_else(|| panic!("f64 variant #{i} must refuse on this device"));
            assert!(
                matches!(err.kind, QuantaErrorKind::NotSupported(_)),
                "f64 variant #{i} refusal must be NotSupported, got {:?}",
                err.kind
            );
        }
    }
}

#[test]
fn f32_distributions_dispatch_on_hardware() {
    let Some(gpu) = try_hw_gpu() else { return };
    let seed = SEEDS[1];

    let normal = fill_normal_f32_gpu(&gpu, LEN, seed).expect("normal f32");
    assert_eq!(normal.len(), LEN);
    assert!(
        normal.iter().all(|v| v.is_finite()),
        "normal must be finite"
    );
    // N(0,1) over 4096 draws: mean well inside ±0.2.
    let mean: f32 = normal.iter().sum::<f32>() / LEN as f32;
    assert!(mean.abs() < 0.2, "normal mean off: {mean}");

    let exp = fill_exponential_f32_gpu(&gpu, LEN, seed, 1.5).expect("exponential f32");
    assert!(
        exp.iter().all(|v| v.is_finite() && *v >= 0.0),
        "exponential must be finite and non-negative"
    );

    let lognorm = fill_lognormal_f32_gpu(&gpu, LEN, seed, 0.0, 0.5).expect("lognormal f32");
    assert!(
        lognorm.iter().all(|v| v.is_finite() && *v > 0.0),
        "lognormal must be finite and positive"
    );

    let bern = fill_bernoulli_u32_gpu(&gpu, LEN, seed, 0.5).expect("bernoulli u32");
    assert!(bern.iter().all(|&v| v <= 1), "bernoulli must be 0/1");
    let ones = bern.iter().sum::<u32>() as f32 / LEN as f32;
    assert!(
        (0.4..0.6).contains(&ones),
        "bernoulli p=0.5 rate off: {ones}"
    );

    let pois = fill_poisson_u32_gpu(&gpu, LEN, seed, 4.0).expect("poisson u32");
    let pmean = pois.iter().sum::<u32>() as f32 / LEN as f32;
    assert!((3.0..5.0).contains(&pmean), "poisson λ=4 mean off: {pmean}");
}
