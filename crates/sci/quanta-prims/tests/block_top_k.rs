//! Differential tests for `block_top_k_u32_buffer`. The GPU
//! kernel sorts each 256-element block ascending then emits the
//! K largest in descending order; the CPU reference does the
//! same. We compare block by block.

#![cfg(feature = "gpu")]

use quanta_prims::{block_top_k_u32_buffer, reference};

const BLOCK: usize = 256;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

fn run_top_k(gpu: &quanta::Gpu, data: &[u32], k: u32) -> Vec<u32> {
    let num_blocks = data.len() / BLOCK;
    let out_len = num_blocks * k as usize;

    let in_field = gpu.field::<u32>(data.len()).unwrap();
    let out_field = gpu.field::<u32>(out_len).unwrap();
    in_field.write(data).unwrap();
    out_field.write(&vec![0u32; out_len]).unwrap();

    let mut wave = block_top_k_u32_buffer(gpu).unwrap();
    wave.bind(0, &in_field);
    wave.bind(1, &out_field);
    wave.set_value(2, k);
    let mut pulse = gpu.dispatch(&wave, data.len() as u32).unwrap();
    pulse.wait().unwrap();

    out_field.read().unwrap()
}

fn check_top_k(data: &[u32], got: &[u32], k: u32) {
    let num_blocks = data.len() / BLOCK;
    let mut expected = vec![0u32; num_blocks * k as usize];
    reference::top_k_u32_blocks(data, &mut expected, BLOCK, k as usize);
    assert_eq!(got, &expected[..]);
}

#[test]
fn top_k_k_equals_one() {
    let Some(gpu) = try_gpu() else { return };
    // 256 values 0..256 → top-1 = 255.
    let data: Vec<u32> = (0..BLOCK as u32).collect();
    let out = run_top_k(&gpu, &data, 1);
    assert_eq!(out, vec![255]);
    check_top_k(&data, &out, 1);
}

#[test]
fn top_k_k_equals_eight() {
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<u32> = (0..BLOCK as u32).collect();
    let out = run_top_k(&gpu, &data, 8);
    // Top 8 of 0..256 = [255, 254, ..., 248].
    let expected: Vec<u32> = (248..256u32).rev().collect();
    assert_eq!(out, expected);
    check_top_k(&data, &out, 8);
}

#[test]
fn top_k_full_block() {
    // k == BLOCK_SIZE: this is just per-block descending sort.
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<u32> = (0..BLOCK as u32).collect();
    let out = run_top_k(&gpu, &data, BLOCK as u32);
    let expected: Vec<u32> = (0..BLOCK as u32).rev().collect();
    assert_eq!(out, expected);
    check_top_k(&data, &out, BLOCK as u32);
}

#[test]
fn top_k_random_distribution() {
    let Some(gpu) = try_gpu() else { return };
    // Pseudo-random u32 values via a simple xorshift seeded with
    // 0xC0FFEE. Deterministic, exercises every byte of the
    // bitonic sort body.
    let mut state: u32 = 0xC0FFEEu32;
    let data: Vec<u32> = (0..BLOCK)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        })
        .collect();
    let out = run_top_k(&gpu, &data, 16);
    check_top_k(&data, &out, 16);
}

#[test]
fn top_k_multi_block() {
    let Some(gpu) = try_gpu() else { return };
    // Two blocks, different patterns.
    let mut data = vec![0u32; 2 * BLOCK];
    for i in 0..BLOCK {
        data[i] = (i * 3) as u32; // 0, 3, 6, ..., 765
        data[BLOCK + i] = (1000 - i as i32) as u32; // 1000, 999, ..., 745
    }
    let k = 4u32;
    let out = run_top_k(&gpu, &data, k);
    check_top_k(&data, &out, k);
    // Spot-check: block-0 max = 765; block-1 max = 1000.
    assert_eq!(out[0], 765);
    assert_eq!(out[k as usize], 1000);
}
