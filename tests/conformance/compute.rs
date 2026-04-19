//! Compute conformance tests — kernel dispatch, results verification.

use quanta::Gpu;

/// MSL kernel for vector addition (Metal).
const VECTOR_ADD_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result  [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    result[idx] = a[idx] + b[idx];
}
"#;

/// MSL kernel for scalar multiply.
const SCALAR_MUL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void scalar_mul(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant float& factor    [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    output[idx] = input[idx] * factor;
}
"#;

/// MSL kernel that writes thread index.
const THREAD_ID_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void write_thread_id(
    device uint* output [[buffer(0)]],
    uint idx [[thread_position_in_grid]]
) {
    output[idx] = idx;
}
"#;

/// Basic vector addition — verify all elements correct.
pub fn vector_add(gpu: &Gpu) {
    let count = 10_000;
    let a_data: Vec<f32> = (0..count).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..count).map(|i| (i * 3) as f32).collect();

    let a = gpu.compute_field::<f32>(count).unwrap();
    let b = gpu.compute_field::<f32>(count).unwrap();
    let result = gpu.compute_field::<f32>(count).unwrap();

    gpu.write_field(&a, &a_data).unwrap();
    gpu.write_field(&b, &b_data).unwrap();

    let mut wave = gpu.wave(VECTOR_ADD_MSL.as_bytes()).unwrap();
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &result);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let output = gpu.read_field(&result).unwrap();
    assert_eq!(output.len(), count);
    for i in 0..count {
        let expected = a_data[i] + b_data[i];
        assert!(
            (output[i] - expected).abs() < 0.001,
            "vector_add mismatch at {}: expected {}, got {}",
            i,
            expected,
            output[i]
        );
    }
}

/// Push constant — multiply by a scalar factor.
pub fn push_constant(gpu: &Gpu) {
    let count = 1000;
    let input_data: Vec<f32> = (0..count).map(|i| i as f32).collect();
    let factor = 3.14f32;

    let input = gpu.compute_field::<f32>(count).unwrap();
    let output = gpu.compute_field::<f32>(count).unwrap();

    gpu.write_field(&input, &input_data).unwrap();

    let mut wave = gpu.wave(SCALAR_MUL_MSL.as_bytes()).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &output);
    wave.set_value(2, factor);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let result = gpu.read_field(&output).unwrap();
    for i in 0..count {
        let expected = input_data[i] * factor;
        assert!(
            (result[i] - expected).abs() < 0.01,
            "push_constant mismatch at {}: expected {}, got {}",
            i,
            expected,
            result[i]
        );
    }
}

/// Thread indexing — each thread writes its own global index.
pub fn thread_id(gpu: &Gpu) {
    let count = 4096;
    let output = gpu.compute_field::<u32>(count).unwrap();

    let mut wave = gpu.wave(THREAD_ID_MSL.as_bytes()).unwrap();
    wave.bind(0, &output);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let result = gpu.read_field(&output).unwrap();
    for i in 0..count {
        assert_eq!(
            result[i], i as u32,
            "thread_id mismatch at {}: expected {}, got {}",
            i, i, result[i]
        );
    }
}

/// Large dispatch — 1M elements.
pub fn large_dispatch(gpu: &Gpu) {
    let count = 1_000_000;
    let a_data: Vec<f32> = (0..count).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..count).map(|i| (i * 2) as f32).collect();

    let a = gpu.compute_field::<f32>(count).unwrap();
    let b = gpu.compute_field::<f32>(count).unwrap();
    let result = gpu.compute_field::<f32>(count).unwrap();

    gpu.write_field(&a, &a_data).unwrap();
    gpu.write_field(&b, &b_data).unwrap();

    let mut wave = gpu.wave(VECTOR_ADD_MSL.as_bytes()).unwrap();
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &result);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let output = gpu.read_field(&result).unwrap();
    // Spot-check first, middle, last
    assert!((output[0] - 0.0).abs() < 0.001);
    assert!((output[count / 2] - (count / 2 + count) as f32).abs() < 1.0);
    assert!((output[count - 1] - (count - 1 + (count - 1) * 2) as f32).abs() < 1.0);
}

/// Rebind fields and re-dispatch same wave.
pub fn wave_rebind(gpu: &Gpu) {
    let count = 256;
    let a1: Vec<f32> = vec![1.0; count];
    let a2: Vec<f32> = vec![10.0; count];
    let b: Vec<f32> = vec![5.0; count];

    let fa = gpu.compute_field::<f32>(count).unwrap();
    let fb = gpu.compute_field::<f32>(count).unwrap();
    let fr = gpu.compute_field::<f32>(count).unwrap();

    // First dispatch
    gpu.write_field(&fa, &a1).unwrap();
    gpu.write_field(&fb, &b).unwrap();

    let mut wave = gpu.wave(VECTOR_ADD_MSL.as_bytes()).unwrap();
    wave.bind(0, &fa);
    wave.bind(1, &fb);
    wave.bind(2, &fr);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let r1 = gpu.read_field(&fr).unwrap();
    assert!(
        (r1[0] - 6.0).abs() < 0.001,
        "first dispatch: expected 6.0, got {}",
        r1[0]
    );

    // Rebind with different data, re-dispatch
    gpu.write_field(&fa, &a2).unwrap();
    wave.bind(0, &fa); // rebind slot 0

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let r2 = gpu.read_field(&fr).unwrap();
    assert!(
        (r2[0] - 15.0).abs() < 0.001,
        "second dispatch: expected 15.0, got {}",
        r2[0]
    );
}

/// Run all compute tests.
pub fn run_all(gpu: &Gpu) {
    vector_add(gpu);
    push_constant(gpu);
    thread_id(gpu);
    large_dispatch(gpu);
    wave_rebind(gpu);
}
