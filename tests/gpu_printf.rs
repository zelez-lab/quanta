//! In-kernel gpu_print (step 049) — end to end on the GPU.
//!
//! The kernel records (quark, tag, bits) triples into a debug buffer
//! the driver binds at buffer(30); after the dispatch completes, the
//! driver drains it to stderr as `[quanta gpu_print] quark=… = …`,
//! matching the CPU executor's inline print. JIT-only (the macro
//! enforces the flag); a printing dispatch completes synchronously.

#[quanta::kernel(jit)]
fn print_squares(input: &[f32], output: &mut [f32]) {
    let i = quark_id();
    let v = input[i] * input[i];
    if i < 2u32 {
        gpu_print_f32(v);
        gpu_print_u32(i);
    }
    output[i] = v;
}

#[test]
fn gpu_print_records_and_computes() {
    let Ok(gpu) = quanta::init() else {
        eprintln!("skipping: no GPU");
        return;
    };
    let wave = print_squares(&gpu).unwrap();
    let input = gpu.field::<f32>(8).unwrap();
    input
        .write(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .unwrap();
    let output = gpu.field::<f32>(8).unwrap();
    output.write(&[0.0; 8]).unwrap();
    let mut wave = wave;
    wave.bind(0, &input);
    wave.bind(1, &output);
    let mut pulse = gpu.wave_dispatch(&wave, [1, 1, 1]).unwrap();
    pulse.wait().unwrap();
    // The print is a side channel; the computation must be untouched.
    let out = output.read().unwrap();
    assert_eq!(out[2], 9.0);
    assert_eq!(out[7], 64.0);
}
