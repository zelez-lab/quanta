//! Tier 2 — Compute dispatch correctness.
//!
//! Tests actual GPU kernel dispatch and readback.
//! Requires a GPU; skips gracefully if none available.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// --- Kernel definitions (proc macro compiles at build time) ---

#[quanta::kernel]
fn add_one(data: &[f32], result: &mut [f32]) {
    let i = quark_id();
    result[i] = data[i] + 1.0;
}

#[quanta::kernel]
fn vector_add(a: &[f32], b: &[f32], result: &mut [f32]) {
    let i = quark_id();
    result[i] = a[i] + b[i];
}

#[quanta::kernel]
fn scalar_multiply(data: &[f32], result: &mut [f32], factor: f32) {
    let i = quark_id();
    result[i] = data[i] * factor;
}

#[quanta::kernel]
fn identity_copy(input: &[f32], output: &mut [f32]) {
    let i = quark_id();
    output[i] = input[i];
}

// --- Tests ---

#[test]
fn dispatch_add_one() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 256;
    let data = vec![1.0f32; count];

    let input = gpu.compute_field::<f32>(count).unwrap();
    let output = gpu.compute_field::<f32>(count).unwrap();
    gpu.write_field(&input, &data).unwrap();

    let mut wave = add_one(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &output);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let result = gpu.read_field::<f32>(&output).unwrap();
    for (i, v) in result.iter().enumerate() {
        assert!(
            (*v - 2.0).abs() < 0.001,
            "add_one mismatch at {}: expected 2.0, got {}",
            i,
            v
        );
    }
}

#[test]
fn dispatch_vector_add() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 512;
    let a_data: Vec<f32> = (0..count).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..count).map(|i| (i * 2) as f32).collect();

    let a = gpu.compute_field::<f32>(count).unwrap();
    let b = gpu.compute_field::<f32>(count).unwrap();
    let result_field = gpu.compute_field::<f32>(count).unwrap();

    gpu.write_field(&a, &a_data).unwrap();
    gpu.write_field(&b, &b_data).unwrap();

    let mut wave = vector_add(&gpu).unwrap();
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &result_field);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let result = gpu.read_field::<f32>(&result_field).unwrap();
    for i in 0..count {
        let expected = a_data[i] + b_data[i];
        assert!(
            (result[i] - expected).abs() < 0.001,
            "vector_add mismatch at {}: expected {}, got {}",
            i,
            expected,
            result[i]
        );
    }
}

#[test]
fn dispatch_scalar_multiply() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 256;
    let factor = 3.5f32;
    let data: Vec<f32> = (0..count).map(|i| i as f32).collect();

    let input = gpu.compute_field::<f32>(count).unwrap();
    let output = gpu.compute_field::<f32>(count).unwrap();
    gpu.write_field(&input, &data).unwrap();

    let mut wave = scalar_multiply(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &output);
    wave.set_value(2, factor);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let result = gpu.read_field::<f32>(&output).unwrap();
    for i in 0..count {
        let expected = data[i] * factor;
        assert!(
            (result[i] - expected).abs() < 0.01,
            "scalar_multiply mismatch at {}: expected {}, got {}",
            i,
            expected,
            result[i]
        );
    }
}

#[test]
fn dispatch_large() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 1_000_000;
    let data: Vec<f32> = (0..count).map(|i| (i % 1000) as f32).collect();

    let input = gpu.compute_field::<f32>(count).unwrap();
    let output = gpu.compute_field::<f32>(count).unwrap();
    gpu.write_field(&input, &data).unwrap();

    let mut wave = add_one(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &output);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let result = gpu.read_field::<f32>(&output).unwrap();
    assert_eq!(result.len(), count);
    // Spot-check: first, middle, last
    assert!((result[0] - 1.0).abs() < 0.001);
    assert!((result[count / 2] - (((count / 2) % 1000) as f32 + 1.0)).abs() < 0.001);
    assert!((result[count - 1] - (((count - 1) % 1000) as f32 + 1.0)).abs() < 0.001);
}

#[test]
fn dispatch_multiple_same_wave() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 256;

    let input = gpu.compute_field::<f32>(count).unwrap();
    let output = gpu.compute_field::<f32>(count).unwrap();

    // First dispatch: start with 1.0, expect 2.0
    let data1 = vec![1.0f32; count];
    gpu.write_field(&input, &data1).unwrap();

    let mut wave = add_one(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &output);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let result1 = gpu.read_field::<f32>(&output).unwrap();
    for v in &result1 {
        assert!((*v - 2.0).abs() < 0.001);
    }

    // Second dispatch: start with 10.0, expect 11.0
    let data2 = vec![10.0f32; count];
    gpu.write_field(&input, &data2).unwrap();

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let result2 = gpu.read_field::<f32>(&output).unwrap();
    for v in &result2 {
        assert!((*v - 11.0).abs() < 0.001);
    }
}

#[test]
fn dispatch_identity() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 1024;
    let data: Vec<f32> = (0..count).map(|i| i as f32 * 0.25).collect();

    let input = gpu.compute_field::<f32>(count).unwrap();
    let output = gpu.compute_field::<f32>(count).unwrap();
    gpu.write_field(&input, &data).unwrap();

    let mut wave = identity_copy(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &output);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let result = gpu.read_field::<f32>(&output).unwrap();
    for i in 0..count {
        assert!(
            (result[i] - data[i]).abs() < 0.001,
            "identity mismatch at {}: expected {}, got {}",
            i,
            data[i],
            result[i]
        );
    }
}
