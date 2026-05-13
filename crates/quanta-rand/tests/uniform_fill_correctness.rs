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
use quanta_rand::uniform::{u32_to_unit_f32, u64_to_unit_f64};
use quanta_rand::{
    fill_uniform_f32_gpu, fill_uniform_f64_gpu, fill_uniform_u32_gpu, fill_uniform_u64_gpu,
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

    assert_eq!(out, expected, "f32 fill must match u32→unit-f32 of host Philox");
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

    assert_eq!(out, expected, "f64 fill must match u64→unit-f64 of host Philox");
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
