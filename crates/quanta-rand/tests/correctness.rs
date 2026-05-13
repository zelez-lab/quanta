//! Bit-exact match between the CPU reference RNG and the
//! `#[quanta::kernel]` GPU implementation.
//!
//! Runs only with the `gpu` feature. The CPU reference is the
//! `Rng` API in `lib.rs`; the GPU kernel inlines the same algorithm
//! and the test asserts they produce identical u32 outputs for
//! every quark.

#![cfg(feature = "gpu")]

use quanta_rand::{fill_buffer_gpu, quark_next_u32};

/// CPU reference: build the same stream the GPU kernel produces.
/// `quark_next_u32` matches the GPU kernel byte-for-byte.
fn cpu_reference(len: usize, seed: u64) -> Vec<u32> {
    let seed_lo = seed as u32;
    let seed_hi = (seed >> 32) as u32;
    (0..len as u32)
        .map(|id| quark_next_u32(seed_lo, seed_hi, id))
        .collect()
}

#[test]
fn gpu_matches_cpu_reference() {
    let gpu = quanta::init().expect("gpu init");
    let len = 1024;
    let seed = 0xCAFE_BABE_DEAD_BEEFu64;

    let cpu_out = cpu_reference(len, seed);
    let gpu_out = fill_buffer_gpu(&gpu, len, seed).expect("dispatch");

    assert_eq!(gpu_out.len(), cpu_out.len(), "length mismatch");
    let mismatches: Vec<_> = (0..len)
        .filter(|&i| gpu_out[i] != cpu_out[i])
        .take(8)
        .collect();
    assert!(
        mismatches.is_empty(),
        "bit-exact mismatch at indices {mismatches:?}: gpu[0..4] = {:?}, cpu[0..4] = {:?}",
        &gpu_out[..4],
        &cpu_out[..4],
    );
}

#[test]
fn distinct_seeds_produce_distinct_outputs() {
    let gpu = quanta::init().expect("gpu init");
    let len = 256;

    let a = fill_buffer_gpu(&gpu, len, 1).expect("seed 1");
    let b = fill_buffer_gpu(&gpu, len, 2).expect("seed 2");

    // Two completely unrelated seed streams should not coincide on
    // every element. (We expect at most a handful of incidental
    // matches over 256 u32 samples.)
    let matches = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    assert!(
        matches < 4,
        "seeds 1 and 2 produced suspiciously many matching outputs: {matches} / {len}",
    );
}
