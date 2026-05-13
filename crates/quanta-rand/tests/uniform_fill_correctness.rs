//! Bit-exact end-to-end correctness for the `fill_uniform_*`
//! kernels: GPU output (CPU backend in this run; native GPUs
//! when available) must match the host-side Philox4×32-10
//! reference for every quark, every variant, every seed.
//!
//! Validates the full chain: kernel macro expansion, the
//! `#[quanta::device]` source-splice, WASM-route lowering, CPU
//! eval (push-const path + i64.mul path), and the host-side
//! `u32`/`u64`/`f32`/`f64` conversion contract.
//!
//! Runs only with the `gpu` feature.

#![cfg(feature = "gpu")]

use quanta_rand::philox4x32::philox4x32_10_first_u32;
use quanta_rand::uniform::{u32_to_open_unit_f32, u32_to_unit_f32, u64_to_unit_f64};
use quanta_rand::{
    fill_normal_f32_gpu, fill_uniform_f32_gpu, fill_uniform_f64_gpu, fill_uniform_u32_gpu,
    fill_uniform_u64_gpu,
};

const SEED: u64 = 0xCAFE_BABE_DEAD_BEEFu64;

fn seed_words() -> (u32, u32) {
    (SEED as u32, (SEED >> 32) as u32)
}

#[test]
fn fill_uniform_u32_matches_host_philox() {
    let gpu = quanta::init_cpu();
    let len = 64;
    let out = fill_uniform_u32_gpu(&gpu, len, SEED).expect("dispatch");

    let (lo, hi) = seed_words();
    let expected: Vec<u32> = (0..len as u32)
        .map(|id| philox4x32_10_first_u32(id, 0, 0, 0, lo, hi))
        .collect();

    assert_eq!(out, expected, "u32 fill must match host Philox4x32-10");
}

#[test]
fn fill_uniform_u64_matches_host_philox() {
    let gpu = quanta::init_cpu();
    let len = 64;
    let out = fill_uniform_u64_gpu(&gpu, len, SEED).expect("dispatch");

    let (lo, hi) = seed_words();
    let expected: Vec<u64> = (0..len as u32)
        .map(|id| {
            let h = philox4x32_10_first_u32(id, 0, 0, 0, lo, hi);
            let l = philox4x32_10_first_u32(id, 1, 0, 0, lo, hi);
            ((h as u64) << 32) | (l as u64)
        })
        .collect();

    assert_eq!(out, expected, "u64 fill must match host Philox packing");
}

#[test]
fn fill_uniform_f32_matches_host_philox() {
    let gpu = quanta::init_cpu();
    let len = 64;
    let out = fill_uniform_f32_gpu(&gpu, len, SEED).expect("dispatch");

    let (lo, hi) = seed_words();
    let expected: Vec<f32> = (0..len as u32)
        .map(|id| u32_to_unit_f32(philox4x32_10_first_u32(id, 0, 0, 0, lo, hi)))
        .collect();

    assert_eq!(
        out, expected,
        "f32 fill must match u32→unit-f32 of host Philox"
    );
    // Every value must be in [0, 1).
    for &v in &out {
        assert!((0.0..1.0).contains(&v), "f32 out of [0, 1): {v}");
    }
}

#[test]
fn fill_uniform_f64_matches_host_philox() {
    let gpu = quanta::init_cpu();
    let len = 64;
    let out = fill_uniform_f64_gpu(&gpu, len, SEED).expect("dispatch");

    let (lo, hi) = seed_words();
    let expected: Vec<f64> = (0..len as u32)
        .map(|id| {
            let h = philox4x32_10_first_u32(id, 0, 0, 0, lo, hi);
            let l = philox4x32_10_first_u32(id, 1, 0, 0, lo, hi);
            let packed = ((h as u64) << 32) | (l as u64);
            u64_to_unit_f64(packed)
        })
        .collect();

    assert_eq!(
        out, expected,
        "f64 fill must match u64→unit-f64 of host Philox"
    );
    for &v in &out {
        assert!((0.0..1.0).contains(&v), "f64 out of [0, 1): {v}");
    }
}

#[test]
fn distinct_seeds_produce_distinct_streams() {
    let gpu = quanta::init_cpu();
    let a = fill_uniform_u32_gpu(&gpu, 256, 1).expect("seed 1");
    let b = fill_uniform_u32_gpu(&gpu, 256, 2).expect("seed 2");
    // Two unrelated Philox streams should not coincide on every
    // element — at most a handful of incidental matches over 256
    // u32 samples.
    let matches = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    assert!(
        matches < 4,
        "seeds 1 and 2 produced suspiciously many matching outputs: {matches} / 256",
    );
}

#[test]
fn empty_fill_returns_empty() {
    let gpu = quanta::init_cpu();
    let out = fill_uniform_u32_gpu(&gpu, 0, SEED).expect("zero-length");
    assert!(out.is_empty());
}

/// Host-side reference for `fill_normal_f32` — mirrors the kernel
/// body byte-for-byte. Each quark id produces a pair (n1, n2)
/// from two Philox draws.
fn host_normal_pair(id: u32, lo: u32, hi: u32) -> (f32, f32) {
    let r0 = philox4x32_10_first_u32(id, 0, 0, 0, lo, hi);
    let r1 = philox4x32_10_first_u32(id, 1, 0, 0, lo, hi);
    let u1 = u32_to_open_unit_f32(r0);
    let u2 = u32_to_open_unit_f32(r1);
    let ln_u1 = u1.ln();
    let r = (-2.0f32 * ln_u1).sqrt();
    let two_pi = 6.2831_8530_7179_586f32;
    let theta = two_pi * u2;
    (r * theta.cos(), r * theta.sin())
}

#[test]
fn fill_normal_f32_matches_host_box_muller() {
    let gpu = quanta::init_cpu();
    let len = 64;
    let out = fill_normal_f32_gpu(&gpu, len, SEED).expect("dispatch");

    let (lo, hi) = seed_words();
    let mut expected = Vec::with_capacity(len);
    for id in 0..(len.div_ceil(2)) as u32 {
        let (n1, n2) = host_normal_pair(id, lo, hi);
        expected.push(n1);
        if expected.len() < len {
            expected.push(n2);
        }
    }
    assert_eq!(out.len(), expected.len());
    for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "normal[{i}] bit-exact mismatch: got {got} (0x{:08x}) want {want} (0x{:08x})",
            got.to_bits(),
            want.to_bits()
        );
    }
}

#[test]
fn fill_normal_f32_distribution_is_approximately_standard() {
    // Light statistical sanity (not a proper K-S test — that's M10).
    let gpu = quanta::init_cpu();
    let n = 10_000;
    let out = fill_normal_f32_gpu(&gpu, n, SEED).expect("dispatch");

    // Mean should be ~0, variance ~1.
    let mean: f32 = out.iter().sum::<f32>() / (n as f32);
    let var: f32 = out.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (n as f32);
    assert!(
        mean.abs() < 0.05,
        "sample mean {mean} too far from 0 (expected ~0 for N(0, 1) at n={n})"
    );
    assert!(
        (var - 1.0).abs() < 0.1,
        "sample variance {var} too far from 1 (expected ~1 for N(0, 1) at n={n})"
    );

    // ~99.7% should fall in [-3, 3] (three-sigma).
    let within_3sigma = out.iter().filter(|&&x| x.abs() < 3.0).count();
    let frac = within_3sigma as f32 / n as f32;
    assert!(
        frac > 0.99,
        "{:.3}% within ±3σ — expected ~99.7%",
        frac * 100.0
    );
}

#[test]
fn fill_normal_f32_handles_odd_length() {
    // Box-Muller produces two normals per quark; odd `len` requires
    // the host to dispatch ceil(len/2) quarks and trim.
    let gpu = quanta::init_cpu();
    let out = fill_normal_f32_gpu(&gpu, 5, SEED).expect("dispatch");
    assert_eq!(out.len(), 5);
    // First 4 must match a 4-output call (same quarks 0,1 → pairs).
    let out_even = fill_normal_f32_gpu(&gpu, 4, SEED).expect("dispatch");
    assert_eq!(&out[..4], &out_even[..]);
}
