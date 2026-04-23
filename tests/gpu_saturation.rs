//! Saturation arithmetic tests (step 041).
//!
//! Verifies saturating_add and saturating_sub clamp instead of wrapping.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[quanta::kernel]
fn sat_add_u32(a: &[u32], b: &[u32], out: &mut [u32]) {
    let i = quark_id();
    out[i] = a[i].saturating_add(b[i]);
}

#[quanta::kernel]
fn sat_sub_u32(a: &[u32], b: &[u32], out: &mut [u32]) {
    let i = quark_id();
    out[i] = a[i].saturating_sub(b[i]);
}

#[test]
fn saturating_add_clamps_to_max() {
    let Some(gpu) = try_gpu() else {
        return;
    };

    let a_data: Vec<u32> = vec![u32::MAX - 10, u32::MAX, 100, 0];
    let b_data: Vec<u32> = vec![20, 1, 50, 0];
    let n = a_data.len();

    let fa = gpu.compute_field::<u32>(n).unwrap();
    let fb = gpu.compute_field::<u32>(n).unwrap();
    let fo = gpu.compute_field::<u32>(n).unwrap();
    gpu.write_field(&fa, &a_data).unwrap();
    gpu.write_field(&fb, &b_data).unwrap();

    let mut wave = sat_add_u32(&gpu).unwrap();
    wave.bind(0, &fa);
    wave.bind(1, &fb);
    wave.bind(2, &fo);

    let mut p = gpu.dispatch(&wave, n as u32).unwrap();
    gpu.wait(&mut p).unwrap();

    let result = gpu.read_field(&fo).unwrap();
    // (MAX-10) + 20 would wrap → should clamp to MAX
    assert_eq!(result[0], u32::MAX, "sat_add should clamp to MAX");
    // MAX + 1 would wrap → should clamp to MAX
    assert_eq!(result[1], u32::MAX, "sat_add MAX+1 should clamp");
    // 100 + 50 = 150 (no overflow)
    assert_eq!(result[2], 150, "sat_add no overflow");
    // 0 + 0 = 0
    assert_eq!(result[3], 0, "sat_add zero");
}

#[test]
fn saturating_sub_clamps_to_zero() {
    let Some(gpu) = try_gpu() else {
        return;
    };

    let a_data: Vec<u32> = vec![10, 0, 100, 50];
    let b_data: Vec<u32> = vec![20, 1, 50, 50];
    let n = a_data.len();

    let fa = gpu.compute_field::<u32>(n).unwrap();
    let fb = gpu.compute_field::<u32>(n).unwrap();
    let fo = gpu.compute_field::<u32>(n).unwrap();
    gpu.write_field(&fa, &a_data).unwrap();
    gpu.write_field(&fb, &b_data).unwrap();

    let mut wave = sat_sub_u32(&gpu).unwrap();
    wave.bind(0, &fa);
    wave.bind(1, &fb);
    wave.bind(2, &fo);

    let mut p = gpu.dispatch(&wave, n as u32).unwrap();
    gpu.wait(&mut p).unwrap();

    let result = gpu.read_field(&fo).unwrap();
    // 10 - 20 would underflow → should clamp to 0
    assert_eq!(result[0], 0, "sat_sub should clamp to 0");
    // 0 - 1 would underflow → should clamp to 0
    assert_eq!(result[1], 0, "sat_sub 0-1 should clamp");
    // 100 - 50 = 50 (no underflow)
    assert_eq!(result[2], 50, "sat_sub no underflow");
    // 50 - 50 = 0
    assert_eq!(result[3], 0, "sat_sub equal");
}
