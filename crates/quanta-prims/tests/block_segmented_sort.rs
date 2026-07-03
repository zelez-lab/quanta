//! Differential tests for `block_segmented_sort_u32_buffer`.
//!
//! Head-flag convention (same as segmented scan/reduce): a non-zero
//! flag opens a segment; every 256-element block boundary opens one
//! too. The kernel sorts ascending WITHIN each segment, leaving
//! segments in input order. Stable, so every test asserts the exact
//! output sequence against the reference.
//!
//! Skips gracefully when no GPU backend is available.

#![cfg(feature = "gpu")]

use quanta_prims::{block_segmented_sort_u32_buffer, reference};

const BLOCK: usize = 256;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

fn xorshift(seed: u32, n: usize) -> Vec<u32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        })
        .collect()
}

fn run_sort(gpu: &quanta::Gpu, data: &[u32], flags: &[u32]) -> Vec<u32> {
    let n = data.len();
    let data_field = gpu.field::<u32>(n).unwrap();
    let flags_field = gpu.field::<u32>(n).unwrap();
    let out_field = gpu.field::<u32>(n).unwrap();
    data_field.write(data).unwrap();
    flags_field.write(flags).unwrap();
    out_field.write(&vec![0u32; n]).unwrap();

    let mut wave = block_segmented_sort_u32_buffer(gpu).unwrap();
    wave.bind(0, &data_field);
    wave.bind(1, &flags_field);
    wave.bind(2, &out_field);
    gpu.dispatch(&wave, n as u32).unwrap().wait().unwrap();
    out_field.read().unwrap()
}

fn check(gpu: &quanta::Gpu, data: &[u32], flags: &[u32]) {
    let got = run_sort(gpu, data, flags);
    let expected = reference::segmented_sort_u32_blocks(data, flags, BLOCK);
    assert_eq!(got, expected, "segmented sort mismatch");
}

#[test]
fn no_flags_is_whole_block_sort() {
    let Some(gpu) = try_gpu() else { return };
    // Zero flags → one segment per block → an ordinary block sort.
    let data: Vec<u32> = (0..BLOCK as u32).rev().collect();
    let flags = vec![0u32; BLOCK];
    check(&gpu, &data, &flags);
}

#[test]
fn every_element_is_a_singleton_segment() {
    let Some(gpu) = try_gpu() else { return };
    // All flags set → 256 singleton segments → output equals input
    // (each one-element segment is already sorted, order preserved).
    let data = xorshift(0x51D, BLOCK);
    let flags = vec![1u32; BLOCK];
    let got = run_sort(&gpu, &data, &flags);
    assert_eq!(got, data, "singletons must pass through unchanged");
    check(&gpu, &data, &flags);
}

#[test]
fn two_segments_split_mid_block() {
    let Some(gpu) = try_gpu() else { return };
    // One head at 100: each side sorts independently, neither
    // bleeds across the boundary.
    let data: Vec<u32> = (0..BLOCK as u32).rev().collect();
    let mut flags = vec![0u32; BLOCK];
    flags[100] = 1;
    check(&gpu, &data, &flags);
    // Boundary invariant: the first segment occupies [0,100), the
    // second [100,256), and each is ascending.
    let got = run_sort(&gpu, &data, &flags);
    assert!(got[..100].windows(2).all(|w| w[0] <= w[1]));
    assert!(got[100..].windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn fixed_width_segments() {
    let Some(gpu) = try_gpu() else { return };
    let data = xorshift(0xF1, BLOCK);
    let flags: Vec<u32> = (0..BLOCK).map(|i| u32::from(i % 16 == 0)).collect();
    check(&gpu, &data, &flags);
}

#[test]
fn segments_with_duplicate_values() {
    let Some(gpu) = try_gpu() else { return };
    // Values in 0..8 inside several segments — exercises the
    // within-segment ordering of equal-ish keys across the radix
    // passes and the segment-id dominance.
    let data: Vec<u32> = xorshift(0xD0B, BLOCK).into_iter().map(|x| x % 8).collect();
    let flags: Vec<u32> = (0..BLOCK).map(|i| u32::from(i % 37 == 0)).collect();
    check(&gpu, &data, &flags);
}

#[test]
fn random_flags_and_values() {
    let Some(gpu) = try_gpu() else { return };
    let data = xorshift(0xDA7A, BLOCK);
    let flags: Vec<u32> = xorshift(0xF1A6, BLOCK)
        .into_iter()
        .map(|x| u32::from(x % 16 == 0))
        .collect();
    check(&gpu, &data, &flags);
}

#[test]
fn extreme_values_within_segments() {
    let Some(gpu) = try_gpu() else { return };
    let mut data = xorshift(0xE7, BLOCK);
    data[5] = u32::MAX;
    data[6] = 0;
    data[200] = u32::MAX;
    data[201] = 0;
    let mut flags = vec![0u32; BLOCK];
    flags[128] = 1;
    check(&gpu, &data, &flags);
}

#[test]
fn multiple_blocks_independent() {
    let Some(gpu) = try_gpu() else { return };
    let n = 4 * BLOCK;
    let data = xorshift(0xB10C, n);
    // Distinct flag pattern per block, plus block-2 flag-free.
    let flags: Vec<u32> = (0..n)
        .map(|i| {
            let within = i % BLOCK;
            let blk = i / BLOCK;
            if blk == 2 {
                0
            } else {
                u32::from(within.is_multiple_of(11 + blk))
            }
        })
        .collect();
    check(&gpu, &data, &flags);
}
