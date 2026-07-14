//! Differential tests for the full block_reduce family:
//! {add, min, max} × {u32, i32, f32}.
//!
//! One test per variant. Each constructs an input known to
//! exercise the operation (negative values for signed mins,
//! mixed-sign for sums, etc.), runs the GPU kernel + reference,
//! and asserts equality (with a tolerance for f32).
//!
//! Skips gracefully when no GPU backend is available.

#![cfg(feature = "gpu")]

use quanta_prims::{
    block_reduce_add_f32_buffer, block_reduce_add_i32_buffer, block_reduce_add_u32_buffer,
    block_reduce_max_f32_buffer, block_reduce_max_i32_buffer, block_reduce_max_u32_buffer,
    block_reduce_min_f32_buffer, block_reduce_min_i32_buffer, block_reduce_min_u32_buffer,
    reference,
};

const BLOCK: usize = 256;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// ── helpers ─────────────────────────────────────────────────────

fn run_u32(
    gpu: &quanta::Gpu,
    builder: impl FnOnce(&quanta::Gpu) -> Result<quanta::Wave, quanta::QuantaError>,
    data: &[u32],
) -> Vec<u32> {
    let num_blocks = data.len() / BLOCK;
    let data_field = gpu.field::<u32>(data.len()).unwrap();
    let out_field = gpu.field::<u32>(num_blocks).unwrap();
    data_field.write(data).unwrap();
    out_field.write(&vec![0u32; num_blocks]).unwrap();
    let mut wave = builder(gpu).unwrap();
    wave.bind(0, &data_field);
    wave.bind(1, &out_field);
    let mut pulse = gpu.dispatch(&wave, data.len() as u32).unwrap();
    pulse.wait().unwrap();
    out_field.read().unwrap()
}

fn run_i32(
    gpu: &quanta::Gpu,
    builder: impl FnOnce(&quanta::Gpu) -> Result<quanta::Wave, quanta::QuantaError>,
    data: &[i32],
) -> Vec<i32> {
    let num_blocks = data.len() / BLOCK;
    let data_field = gpu.field::<i32>(data.len()).unwrap();
    let out_field = gpu.field::<i32>(num_blocks).unwrap();
    data_field.write(data).unwrap();
    out_field.write(&vec![0i32; num_blocks]).unwrap();
    let mut wave = builder(gpu).unwrap();
    wave.bind(0, &data_field);
    wave.bind(1, &out_field);
    let mut pulse = gpu.dispatch(&wave, data.len() as u32).unwrap();
    pulse.wait().unwrap();
    out_field.read().unwrap()
}

fn run_f32(
    gpu: &quanta::Gpu,
    builder: impl FnOnce(&quanta::Gpu) -> Result<quanta::Wave, quanta::QuantaError>,
    data: &[f32],
) -> Vec<f32> {
    let num_blocks = data.len() / BLOCK;
    let data_field = gpu.field::<f32>(data.len()).unwrap();
    let out_field = gpu.field::<f32>(num_blocks).unwrap();
    data_field.write(data).unwrap();
    out_field.write(&vec![0f32; num_blocks]).unwrap();
    let mut wave = builder(gpu).unwrap();
    wave.bind(0, &data_field);
    wave.bind(1, &out_field);
    let mut pulse = gpu.dispatch(&wave, data.len() as u32).unwrap();
    pulse.wait().unwrap();
    out_field.read().unwrap()
}

// ── add ─────────────────────────────────────────────────────────

#[test]
fn add_u32_matches_reference() {
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<u32> = (1..=BLOCK as u32).collect();
    let expected = reference::reduce_add_u32(&data);
    let result = run_u32(&gpu, block_reduce_add_u32_buffer, &data);
    assert_eq!(result, vec![expected]);
}

#[test]
fn add_i32_matches_reference_with_negatives() {
    let Some(gpu) = try_gpu() else { return };
    // Alternating sign: sums to a small number, exercises i32 path.
    let data: Vec<i32> = (0..BLOCK as i32)
        .map(|i| if i % 2 == 0 { i + 1 } else { -(i + 1) })
        .collect();
    let expected = reference::reduce_add_i32(&data);
    let result = run_i32(&gpu, block_reduce_add_i32_buffer, &data);
    assert_eq!(result, vec![expected]);
}

#[test]
fn add_f32_matches_reference_within_tolerance() {
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<f32> = (1..=BLOCK).map(|x| x as f32).collect();
    let expected = reference::reduce_add_f32(&data);
    let result = run_f32(&gpu, block_reduce_add_f32_buffer, &data);
    let got = result[0];
    // GPU tree-reduce produces a slightly different IEEE-754
    // rounding than the sequential reference; allow ~1 ULP per
    // BLOCK additions.
    let tol = expected.abs() * 1e-5;
    assert!(
        (got - expected).abs() < tol,
        "got {got}, expected {expected}, tol {tol}"
    );
}

// ── min ─────────────────────────────────────────────────────────

#[test]
fn min_u32_matches_reference() {
    let Some(gpu) = try_gpu() else { return };
    // Put the true minimum in the middle of the block, not at
    // lane 0, so the test confirms cross-warp aggregation.
    let mut data: Vec<u32> = vec![1000u32; BLOCK];
    data[123] = 42;
    let expected = reference::reduce_min_u32(&data);
    let result = run_u32(&gpu, block_reduce_min_u32_buffer, &data);
    assert_eq!(result, vec![expected]);
}

#[test]
fn min_i32_matches_reference_with_negatives() {
    let Some(gpu) = try_gpu() else { return };
    let mut data: Vec<i32> = vec![1000i32; BLOCK];
    data[200] = -987654;
    let expected = reference::reduce_min_i32(&data);
    let result = run_i32(&gpu, block_reduce_min_i32_buffer, &data);
    assert_eq!(result, vec![expected]);
}

#[test]
fn min_f32_matches_reference() {
    let Some(gpu) = try_gpu() else { return };
    let mut data: Vec<f32> = vec![1000.0f32; BLOCK];
    data[77] = -3.25;
    let expected = reference::reduce_min_f32(&data);
    let result = run_f32(&gpu, block_reduce_min_f32_buffer, &data);
    let got = result[0];
    assert!(
        (got - expected).abs() < 1e-6,
        "got {got}, expected {expected}"
    );
}

// ── max ─────────────────────────────────────────────────────────

#[test]
fn max_u32_matches_reference() {
    let Some(gpu) = try_gpu() else { return };
    let mut data: Vec<u32> = vec![10u32; BLOCK];
    data[55] = 99_999;
    let expected = reference::reduce_max_u32(&data);
    let result = run_u32(&gpu, block_reduce_max_u32_buffer, &data);
    assert_eq!(result, vec![expected]);
}

#[test]
fn max_i32_matches_reference_with_negatives() {
    let Some(gpu) = try_gpu() else { return };
    // All negatives so the i32 identity element (i32::MIN) is
    // exercised: every real partial dominates the identity.
    let data: Vec<i32> = (0..BLOCK as i32).map(|i| -(i + 1)).collect();
    let expected = reference::reduce_max_i32(&data);
    let result = run_i32(&gpu, block_reduce_max_i32_buffer, &data);
    assert_eq!(result, vec![expected]);
}

#[test]
fn max_f32_matches_reference() {
    let Some(gpu) = try_gpu() else { return };
    let mut data: Vec<f32> = vec![-1000.0f32; BLOCK];
    data[200] = 42.5;
    let expected = reference::reduce_max_f32(&data);
    let result = run_f32(&gpu, block_reduce_max_f32_buffer, &data);
    let got = result[0];
    assert!(
        (got - expected).abs() < 1e-6,
        "got {got}, expected {expected}"
    );
}
