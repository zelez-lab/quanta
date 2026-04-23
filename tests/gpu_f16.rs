//! f16 arithmetic validation (step 042).
//!
//! Verifies that half-precision float operations produce correct results
//! within the expected precision limits (10-bit mantissa → ~3 decimal digits).

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[quanta::kernel]
fn f16_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    let i = quark_id();
    // Cast to f16, operate, cast back to f32 for readback
    let ah = a[i] as f16;
    let bh = b[i] as f16;
    let rh = ah + bh;
    out[i] = rh as f32;
}

#[quanta::kernel]
fn f16_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    let i = quark_id();
    let ah = a[i] as f16;
    let bh = b[i] as f16;
    let rh = ah * bh;
    out[i] = rh as f32;
}

#[test]
fn f16_addition_precision() {
    let Some(gpu) = try_gpu() else {
        return;
    };

    let a_data: Vec<f32> = vec![1.0, 0.5, 100.0, -1.5];
    let b_data: Vec<f32> = vec![2.0, 0.25, 200.0, 3.5];
    let n = a_data.len();

    let fa = gpu.compute_field::<f32>(n).unwrap();
    let fb = gpu.compute_field::<f32>(n).unwrap();
    let fo = gpu.compute_field::<f32>(n).unwrap();
    gpu.write_field(&fa, &a_data).unwrap();
    gpu.write_field(&fb, &b_data).unwrap();

    let mut wave = f16_add(&gpu).unwrap();
    wave.bind(0, &fa);
    wave.bind(1, &fb);
    wave.bind(2, &fo);

    let mut p = gpu.dispatch(&wave, n as u32).unwrap();
    gpu.wait(&mut p).unwrap();

    let result = gpu.read_field(&fo).unwrap();

    // f16 has ~3 digits of precision. Allow 0.1% relative error for small values,
    // 1% for larger values (100+200=300 has quantization).
    let expected = [3.0, 0.75, 300.0, 2.0];
    for i in 0..n {
        let err = (result[i] - expected[i]).abs();
        let rel = err / expected[i].abs().max(1.0);
        eprintln!(
            "  f16_add[{i}] = {:.4} (expected {:.4}, err {:.6})",
            result[i], expected[i], rel
        );
        assert!(
            rel < 0.01,
            "f16 add precision: expected {}, got {} (rel err {})",
            expected[i],
            result[i],
            rel
        );
    }
}

#[test]
fn f16_multiplication_precision() {
    let Some(gpu) = try_gpu() else {
        return;
    };

    let a_data: Vec<f32> = vec![2.0, 0.5, 10.0, -3.0];
    let b_data: Vec<f32> = vec![3.0, 0.5, 10.0, 4.0];
    let n = a_data.len();

    let fa = gpu.compute_field::<f32>(n).unwrap();
    let fb = gpu.compute_field::<f32>(n).unwrap();
    let fo = gpu.compute_field::<f32>(n).unwrap();
    gpu.write_field(&fa, &a_data).unwrap();
    gpu.write_field(&fb, &b_data).unwrap();

    let mut wave = f16_mul(&gpu).unwrap();
    wave.bind(0, &fa);
    wave.bind(1, &fb);
    wave.bind(2, &fo);

    let mut p = gpu.dispatch(&wave, n as u32).unwrap();
    gpu.wait(&mut p).unwrap();

    let result = gpu.read_field(&fo).unwrap();

    let expected = [6.0, 0.25, 100.0, -12.0];
    for i in 0..n {
        let err = (result[i] - expected[i]).abs();
        let rel = err / expected[i].abs().max(1.0);
        eprintln!(
            "  f16_mul[{i}] = {:.4} (expected {:.4}, err {:.6})",
            result[i], expected[i], rel
        );
        assert!(
            rel < 0.01,
            "f16 mul precision: expected {}, got {} (rel err {})",
            expected[i],
            result[i],
            rel
        );
    }
}
