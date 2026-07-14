//! Differential tests for the block_scan family: inclusive
//! prefix-sum scan over {u32, i32, f32}.
//!
//! For each (op, ty) variant, one full-block scan is computed on
//! the GPU and compared with the CPU reference. The assertion is
//! element-wise — scan output has the same length as the input.
//!
//! Skips gracefully when no GPU backend is available.

#![cfg(feature = "gpu")]

use quanta_prims::{
    block_scan_add_f32_buffer, block_scan_add_i32_buffer, block_scan_add_u32_buffer, reference,
};

const BLOCK: usize = 256;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

fn run_u32(
    gpu: &quanta::Gpu,
    builder: impl FnOnce(&quanta::Gpu) -> Result<quanta::Wave, quanta::QuantaError>,
    data: &[u32],
) -> Vec<u32> {
    let data_field = gpu.field::<u32>(data.len()).unwrap();
    let out_field = gpu.field::<u32>(data.len()).unwrap();
    data_field.write(data).unwrap();
    out_field.write(&vec![0u32; data.len()]).unwrap();
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
    let data_field = gpu.field::<i32>(data.len()).unwrap();
    let out_field = gpu.field::<i32>(data.len()).unwrap();
    data_field.write(data).unwrap();
    out_field.write(&vec![0i32; data.len()]).unwrap();
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
    let data_field = gpu.field::<f32>(data.len()).unwrap();
    let out_field = gpu.field::<f32>(data.len()).unwrap();
    data_field.write(data).unwrap();
    out_field.write(&vec![0f32; data.len()]).unwrap();
    let mut wave = builder(gpu).unwrap();
    wave.bind(0, &data_field);
    wave.bind(1, &out_field);
    let mut pulse = gpu.dispatch(&wave, data.len() as u32).unwrap();
    pulse.wait().unwrap();
    out_field.read().unwrap()
}

#[test]
fn scan_add_u32_matches_reference_on_ramp() {
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<u32> = (1..=BLOCK as u32).collect();
    let expected = reference::scan_add_u32(&data);
    let result = run_u32(&gpu, block_scan_add_u32_buffer, &data);
    assert_eq!(result, expected);
}

#[test]
fn scan_add_u32_matches_reference_on_uniform_input() {
    let Some(gpu) = try_gpu() else { return };
    // All ones: result[k] should equal k+1.
    let data = vec![1u32; BLOCK];
    let expected = reference::scan_add_u32(&data);
    let result = run_u32(&gpu, block_scan_add_u32_buffer, &data);
    assert_eq!(result, expected);
    // Spot-check structure: out[0] = 1, out[N-1] = N.
    assert_eq!(result[0], 1);
    assert_eq!(result[BLOCK - 1], BLOCK as u32);
}

#[test]
fn scan_add_i32_matches_reference_with_negatives() {
    let Some(gpu) = try_gpu() else { return };
    // Alternating sign: oscillating prefix sum exercises signed
    // arithmetic in the warp boundary stage.
    let data: Vec<i32> = (0..BLOCK as i32)
        .map(|i| if i % 2 == 0 { i + 1 } else { -(i + 1) })
        .collect();
    let expected = reference::scan_add_i32(&data);
    let result = run_i32(&gpu, block_scan_add_i32_buffer, &data);
    assert_eq!(result, expected);
}

#[test]
fn scan_add_f32_matches_reference_within_tolerance() {
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<f32> = (1..=BLOCK).map(|x| x as f32).collect();
    let expected = reference::scan_add_f32(&data);
    let result = run_f32(&gpu, block_scan_add_f32_buffer, &data);
    assert_eq!(result.len(), expected.len());
    for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
        // GPU and CPU may use different evaluation orders for the
        // floating-point scan; allow a small relative tolerance
        // that scales with the magnitude of the prefix sum.
        let tol = want.abs() * 1e-5 + 1e-6;
        assert!(
            (got - want).abs() < tol,
            "lane {i}: got {got}, expected {want}, tol {tol}"
        );
    }
}

#[test]
fn scan_add_u32_first_output_equals_first_input() {
    // Sanity check: inclusive scan never alters the first
    // element. Exercises the warp-0-lane-0 fast path.
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<u32> = (100..(100 + BLOCK as u32)).collect();
    let result = run_u32(&gpu, block_scan_add_u32_buffer, &data);
    assert_eq!(result[0], 100);
}

#[test]
fn scan_add_u32_last_output_equals_total_sum() {
    // Sanity check: inclusive scan's last element equals the
    // total reduction. Exercises the cross-warp prefix path.
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<u32> = (1..=BLOCK as u32).collect();
    let expected_total = reference::reduce_add_u32(&data);
    let result = run_u32(&gpu, block_scan_add_u32_buffer, &data);
    assert_eq!(result[BLOCK - 1], expected_total);
}
