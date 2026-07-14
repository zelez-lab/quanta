//! Differential tests for the segmented scan / reduce pair.
//!
//! Head-flag semantics: a non-zero flag starts a new segment; every
//! 256-element block boundary also starts one. The GPU kernels must
//! agree with the CPU references on segment layouts that stress the
//! flag propagation: no flags at all, every-element flags, runs of
//! mixed length, and heads at block boundaries.
//!
//! Skips gracefully when no GPU backend is available.

#![cfg(feature = "gpu")]

use quanta_prims::{
    block_segmented_reduce_add_u32_buffer, block_segmented_scan_add_u32_buffer, reference,
};

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

fn run_scan(gpu: &quanta::Gpu, data: &[u32], flags: &[u32]) -> Vec<u32> {
    let n = data.len();
    let data_field = gpu.field::<u32>(n).unwrap();
    let flags_field = gpu.field::<u32>(n).unwrap();
    let out_field = gpu.field::<u32>(n).unwrap();
    data_field.write(data).unwrap();
    flags_field.write(flags).unwrap();
    out_field.write(&vec![0u32; n]).unwrap();

    let mut wave = block_segmented_scan_add_u32_buffer(gpu).unwrap();
    wave.bind(0, &data_field);
    wave.bind(1, &flags_field);
    wave.bind(2, &out_field);
    gpu.dispatch(&wave, n as u32).unwrap().wait().unwrap();
    out_field.read().unwrap()
}

fn run_reduce(gpu: &quanta::Gpu, data: &[u32], flags: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let n = data.len();
    let num_blocks = n / BLOCK;
    let data_field = gpu.field::<u32>(n).unwrap();
    let flags_field = gpu.field::<u32>(n).unwrap();
    let totals_field = gpu.field::<u32>(n).unwrap();
    let counts_field = gpu.field::<u32>(num_blocks).unwrap();
    data_field.write(data).unwrap();
    flags_field.write(flags).unwrap();
    totals_field.write(&vec![0u32; n]).unwrap();
    counts_field.write(&vec![0u32; num_blocks]).unwrap();

    let mut wave = block_segmented_reduce_add_u32_buffer(gpu).unwrap();
    wave.bind(0, &data_field);
    wave.bind(1, &flags_field);
    wave.bind(2, &totals_field);
    wave.bind(3, &counts_field);
    gpu.dispatch(&wave, n as u32).unwrap().wait().unwrap();
    (totals_field.read().unwrap(), counts_field.read().unwrap())
}

fn check_both(gpu: &quanta::Gpu, data: &[u32], flags: &[u32]) {
    let n = data.len();
    let num_blocks = n / BLOCK;

    let mut expected_scan = vec![0u32; n];
    reference::segmented_scan_add_u32_blocks(data, flags, &mut expected_scan, BLOCK);
    assert_eq!(run_scan(gpu, data, flags), expected_scan, "scan mismatch");

    let mut expected_totals = vec![0u32; n];
    let mut expected_counts = vec![0u32; num_blocks];
    reference::segmented_reduce_add_u32_blocks(
        data,
        flags,
        &mut expected_totals,
        &mut expected_counts,
        BLOCK,
    );
    let (totals, counts) = run_reduce(gpu, data, flags);
    assert_eq!(counts, expected_counts, "segment counts mismatch");
    assert_eq!(totals, expected_totals, "segment totals mismatch");
}

#[test]
fn no_flags_is_plain_scan() {
    let Some(gpu) = try_gpu() else { return };
    // Zero flags → one segment per block; scan must equal the
    // ordinary inclusive prefix sum, reduce must equal block sum.
    let data: Vec<u32> = (1..=BLOCK as u32).collect();
    let flags = vec![0u32; BLOCK];
    check_both(&gpu, &data, &flags);
}

#[test]
fn every_element_is_a_head() {
    let Some(gpu) = try_gpu() else { return };
    // All flags set → 256 singleton segments; scan = identity,
    // reduce = the input itself.
    let data = xorshift(0x51D, BLOCK);
    let flags = vec![1u32; BLOCK];
    check_both(&gpu, &data, &flags);
}

#[test]
fn fixed_width_segments() {
    let Some(gpu) = try_gpu() else { return };
    // Heads every 8 elements: exercises propagation across
    // several doubling strides but not all.
    let data: Vec<u32> = (0..BLOCK as u32).collect();
    let flags: Vec<u32> = (0..BLOCK).map(|i| u32::from(i % 8 == 0)).collect();
    check_both(&gpu, &data, &flags);
}

#[test]
fn two_segments_split_mid_block() {
    let Some(gpu) = try_gpu() else { return };
    // One head at 100: the second segment's sums must not absorb
    // anything from the first across any of the 8 rounds.
    let data = vec![1u32; BLOCK];
    let mut flags = vec![0u32; BLOCK];
    flags[100] = 1;
    check_both(&gpu, &data, &flags);
}

#[test]
fn head_at_block_boundary_and_multi_block() {
    let Some(gpu) = try_gpu() else { return };
    // 4 blocks. Block 1 has an explicit head at its first lane
    // (redundant with the implicit restart — must not double-count),
    // block 2 has none (single implicit segment), block 3 mixes.
    let n = 4 * BLOCK;
    let data = xorshift(0xB10C, n)
        .into_iter()
        .map(|x| x % 100)
        .collect::<Vec<_>>();
    let mut flags = vec![0u32; n];
    flags[BLOCK] = 1; // explicit head exactly at block 1's start
    flags[BLOCK + 7] = 1;
    for i in 0..BLOCK {
        if i % 31 == 0 {
            flags[3 * BLOCK + i] = 1;
        }
    }
    check_both(&gpu, &data, &flags);
}

#[test]
fn random_flags_random_data() {
    let Some(gpu) = try_gpu() else { return };
    // 8 blocks of random data with ~1/16 flag density — irregular
    // segment lengths across every stride size.
    let n = 8 * BLOCK;
    let data = xorshift(0xDA7A, n);
    let flags: Vec<u32> = xorshift(0xF1A6, n)
        .into_iter()
        .map(|x| u32::from(x % 16 == 0))
        .collect();
    check_both(&gpu, &data, &flags);
}

#[test]
fn wrapping_sums_match_reference() {
    let Some(gpu) = try_gpu() else { return };
    // Large values force u32 wrap-around inside a segment; the
    // GPU's wrapping add must match the reference's wrapping_add.
    let data = vec![0x9000_0000u32; BLOCK];
    let mut flags = vec![0u32; BLOCK];
    flags[128] = 1;
    check_both(&gpu, &data, &flags);
}
