//! Differential test: GPU block_reduce_add_u32 against the
//! reference CPU implementation.
//!
//! Workgroup size is 256, so each block sums 256 inputs. The
//! tests cover one-block and multi-block dispatches, plus edge
//! cases (zeros, uniform, ramp).
//!
//! Skips gracefully when no GPU backend is available.

#![cfg(feature = "gpu")]

use quanta_prims::{block_reduce_add_u32_buffer, reference};

const BLOCK: usize = 256;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

fn run_block_reduce(gpu: &quanta::Gpu, data: &[u32]) -> Vec<u32> {
    assert_eq!(data.len() % BLOCK, 0, "data must be a multiple of {BLOCK}");
    let num_blocks = data.len() / BLOCK;

    let data_field = gpu.field::<u32>(data.len()).unwrap();
    let out_field = gpu.field::<u32>(num_blocks).unwrap();
    data_field.write(data).unwrap();
    out_field.write(&vec![0u32; num_blocks]).unwrap();

    let mut wave = block_reduce_add_u32_buffer(gpu).unwrap();
    wave.bind(0, &data_field);
    wave.bind(1, &out_field);

    let mut pulse = gpu.dispatch(&wave, data.len() as u32).unwrap();
    pulse.wait().unwrap();

    out_field.read().unwrap()
}

#[test]
fn matches_reference_on_one_block() {
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<u32> = (1..=BLOCK as u32).collect();
    let expected = reference::reduce_add_u32(&data);

    let result = run_block_reduce(&gpu, &data);
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0], expected,
        "got {}, expected {} on ramp [1..={}]",
        result[0], expected, BLOCK
    );
}

#[test]
fn handles_zero_input() {
    let Some(gpu) = try_gpu() else { return };
    let data = vec![0u32; BLOCK];
    let result = run_block_reduce(&gpu, &data);
    assert_eq!(result, vec![0]);
}

#[test]
fn handles_uniform_input() {
    let Some(gpu) = try_gpu() else { return };
    let data = vec![7u32; BLOCK];
    let expected = 7u32 * BLOCK as u32; // 1792
    let result = run_block_reduce(&gpu, &data);
    assert_eq!(result, vec![expected]);
}

#[test]
fn matches_reference_on_multiple_blocks() {
    let Some(gpu) = try_gpu() else { return };
    // 4 blocks. Each block gets a different range of the ramp;
    // the per-block sum equals the sum of that range.
    let num_blocks = 4;
    let data: Vec<u32> = (1..=(BLOCK * num_blocks) as u32).collect();
    let result = run_block_reduce(&gpu, &data);

    assert_eq!(result.len(), num_blocks);
    for (b, &sum) in result.iter().enumerate() {
        let start = b * BLOCK;
        let end = start + BLOCK;
        let expected = reference::reduce_add_u32(&data[start..end]);
        assert_eq!(sum, expected, "block {b}: got {sum}, expected {expected}");
    }
}

#[test]
fn handles_max_values() {
    // Stress overflow semantics: BLOCK values of u32::MAX wrap
    // around to BLOCK lots of `u32::MAX`, total = BLOCK * u32::MAX
    // (mod 2^32). With BLOCK = 256, that's 256 * (2^32 - 1) mod
    // 2^32 = 2^32 * 256 - 256 ≡ -256 mod 2^32 = u32::MAX - 255.
    let Some(gpu) = try_gpu() else { return };
    let data = vec![u32::MAX; BLOCK];
    let expected = reference::reduce_add_u32(&data);
    let result = run_block_reduce(&gpu, &data);
    assert_eq!(result, vec![expected]);
}
