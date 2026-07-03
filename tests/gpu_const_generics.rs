//! Const generics in kernels (step 047).
//!
//! Verifies that `fn kernel<const TILE: u32>(...)` works as a compile-time
//! constant accessible in the kernel body.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[quanta::kernel]
fn fill_with_const<const VAL: u32>(output: &mut [u32]) {
    let i = quark_id();
    output[i] = VAL;
}

#[test]
fn const_generic_fills_buffer() {
    let Some(gpu) = try_gpu() else {
        return;
    };

    let n = 16usize;
    let output = gpu.field::<u32>(n).unwrap();

    // Dispatch with const generic = 42
    let mut wave = fill_with_const::<42>(&gpu).unwrap();
    wave.bind(0, &output);

    let mut p = gpu.dispatch(&wave, n as u32).unwrap();
    p.wait().unwrap();

    let result = output.read().unwrap();
    for &got in result.iter().take(n) {
        assert_eq!(got, 42, "const generic: expected 42, got {}", got);
    }
}

#[test]
fn const_generic_different_values() {
    let Some(gpu) = try_gpu() else {
        return;
    };

    let n = 16usize;
    let output = gpu.field::<u32>(n).unwrap();

    // Dispatch with const generic = 99
    let mut wave = fill_with_const::<99>(&gpu).unwrap();
    wave.bind(0, &output);

    let mut p = gpu.dispatch(&wave, n as u32).unwrap();
    p.wait().unwrap();

    let result = output.read().unwrap();
    assert_eq!(
        result[0], 99,
        "const generic: expected 99, got {}",
        result[0]
    );
}
