//! Differential test: GPU block_reduce_add_u32 against the
//! reference CPU implementation.
//!
//! Runs on whatever GPU backend is available; skips gracefully
//! when no GPU is present.

#![cfg(feature = "gpu")]

use quanta_prims::{block_reduce_add_u32_buffer, reference};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[test]
fn block_reduce_matches_reference_on_one_block() {
    let Some(gpu) = try_gpu() else { return };
    // Single 32-lane workgroup. The reduce result lands in out[0].
    const N: usize = 32;
    let data: Vec<u32> = (1..=N as u32).collect();
    let expected = reference::reduce_add_u32(&data);

    let data_field = gpu.field::<u32>(N).unwrap();
    let out_field = gpu.field::<u32>(1).unwrap();
    data_field.write(&data).unwrap();
    out_field.write(&[0u32]).unwrap();

    let mut wave = block_reduce_add_u32_buffer(&gpu).unwrap();
    wave.bind(0, &data_field);
    wave.bind(1, &out_field);

    let mut pulse = gpu.dispatch(&wave, N as u32).unwrap();
    pulse.wait().unwrap();

    let result = out_field.read().unwrap();
    assert_eq!(
        result[0], expected,
        "block reduce mismatch: got {}, expected {} on input {:?}",
        result[0], expected, data
    );
}

#[test]
fn block_reduce_handles_zero_input() {
    let Some(gpu) = try_gpu() else { return };
    const N: usize = 32;
    let data = vec![0u32; N];

    let data_field = gpu.field::<u32>(N).unwrap();
    let out_field = gpu.field::<u32>(1).unwrap();
    data_field.write(&data).unwrap();
    out_field.write(&[99u32]).unwrap();

    let mut wave = block_reduce_add_u32_buffer(&gpu).unwrap();
    wave.bind(0, &data_field);
    wave.bind(1, &out_field);

    let mut pulse = gpu.dispatch(&wave, N as u32).unwrap();
    pulse.wait().unwrap();

    let result = out_field.read().unwrap();
    assert_eq!(result[0], 0);
}

#[test]
fn block_reduce_handles_uniform_input() {
    let Some(gpu) = try_gpu() else { return };
    const N: usize = 32;
    // Every lane contributes 7 -> sum should be 32 * 7 = 224.
    let data = vec![7u32; N];
    let expected = 7u32 * N as u32;

    let data_field = gpu.field::<u32>(N).unwrap();
    let out_field = gpu.field::<u32>(1).unwrap();
    data_field.write(&data).unwrap();
    out_field.write(&[0u32]).unwrap();

    let mut wave = block_reduce_add_u32_buffer(&gpu).unwrap();
    wave.bind(0, &data_field);
    wave.bind(1, &out_field);

    let mut pulse = gpu.dispatch(&wave, N as u32).unwrap();
    pulse.wait().unwrap();

    let result = out_field.read().unwrap();
    assert_eq!(result[0], expected);
}
