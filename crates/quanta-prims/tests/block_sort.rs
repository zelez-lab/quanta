//! Differential test for block_radix_sort_u32_buffer.
//!
//! The GPU sort produces a block-local sorted output (each
//! 256-element block sorted independently). We compare each
//! block against a CPU reference sort of the same block.
//!
//! Skips gracefully when no GPU backend is available.

#![cfg(feature = "gpu")]

use quanta_prims::{block_radix_sort_u32_buffer, reference};

const BLOCK: usize = 256;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

fn run_sort(gpu: &quanta::Gpu, data: &[u32]) -> Vec<u32> {
    assert_eq!(data.len() % BLOCK, 0, "data must be a multiple of BLOCK");
    let data_field = gpu.field::<u32>(data.len()).unwrap();
    let out_field = gpu.field::<u32>(data.len()).unwrap();
    data_field.write(data).unwrap();
    out_field.write(&vec![0u32; data.len()]).unwrap();
    let mut wave = block_radix_sort_u32_buffer(gpu).unwrap();
    wave.bind(0, &data_field);
    wave.bind(1, &out_field);
    let mut pulse = gpu.dispatch(&wave, data.len() as u32).unwrap();
    pulse.wait().unwrap();
    out_field.read().unwrap()
}

#[test]
fn sorts_descending_input_to_ascending() {
    let Some(gpu) = try_gpu() else { return };
    // Strictly descending: every compare-swap fires.
    let data: Vec<u32> = (0..BLOCK as u32).rev().collect();
    let expected = reference::radix_sort_u32(&data);
    let result = run_sort(&gpu, &data);
    assert_eq!(result, expected);
}

#[test]
fn sorts_already_sorted_input_unchanged() {
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<u32> = (0..BLOCK as u32).collect();
    let expected = data.clone();
    let result = run_sort(&gpu, &data);
    assert_eq!(result, expected);
}

#[test]
fn sorts_uniform_input_unchanged() {
    let Some(gpu) = try_gpu() else { return };
    let data = vec![42u32; BLOCK];
    let result = run_sort(&gpu, &data);
    assert_eq!(result, data);
}

#[test]
fn sorts_pseudo_random_input() {
    let Some(gpu) = try_gpu() else { return };
    // Reproducible pseudo-random input via a simple LCG. Tests
    // the general case where neither the ones nor zeros bucket
    // has a fixed-position pattern.
    let mut state: u32 = 0xCAFE_BABE;
    let data: Vec<u32> = (0..BLOCK)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        })
        .collect();
    let expected = reference::radix_sort_u32(&data);
    let result = run_sort(&gpu, &data);
    assert_eq!(result, expected);
}

#[test]
fn sorts_input_with_ties() {
    let Some(gpu) = try_gpu() else { return };
    // Many duplicates: tests the comparator's handling of
    // equal-key compare-exchanges. The sort doesn't promise
    // stability but the multiset of values must be preserved.
    let data: Vec<u32> = (0..BLOCK as u32).map(|i| i % 8u32).collect();
    let expected = reference::radix_sort_u32(&data);
    let result = run_sort(&gpu, &data);
    assert_eq!(result, expected);
}

#[test]
fn sorts_extreme_values() {
    let Some(gpu) = try_gpu() else { return };
    // Mix of 0, u32::MAX, and powers of 2 -- exercises the high
    // bit and edge cases for the comparator.
    let mut data = Vec::with_capacity(BLOCK);
    data.push(0u32);
    data.push(u32::MAX);
    for k in 0..30 {
        data.push(1u32 << k);
    }
    while data.len() < BLOCK {
        data.push((data.len() as u32).wrapping_mul(2654435761));
    }
    let expected = reference::radix_sort_u32(&data);
    let result = run_sort(&gpu, &data);
    assert_eq!(result, expected);
}

#[test]
fn sorts_multiple_blocks_independently() {
    let Some(gpu) = try_gpu() else { return };
    // Two blocks, each with its own descending range.
    let mut data = Vec::with_capacity(2 * BLOCK);
    data.extend((0..BLOCK as u32).rev()); // block 0: 255, 254, …, 0
    data.extend((1000..(1000 + BLOCK as u32)).rev()); // block 1: 1255, …, 1000
    let result = run_sort(&gpu, &data);

    // Each block should be sorted independently.
    let block0 = &result[..BLOCK];
    let block1 = &result[BLOCK..];

    let expected_block0 = reference::radix_sort_u32(&data[..BLOCK]);
    let expected_block1 = reference::radix_sort_u32(&data[BLOCK..]);

    assert_eq!(block0, expected_block0);
    assert_eq!(block1, expected_block1);
}
