//! Bitwise operation tests — validates OpBitwiseAnd on all platforms.
//!
//! Broadcom V3D was returning the second operand instead of a & b.
//! This test catches the regression.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[quanta::kernel]
fn bitwise_and_const(input: &[u32], output: &mut [u32]) {
    let i = quark_id();
    output[i] = input[i] & 255u32;
}

#[quanta::kernel]
fn bitwise_and_runtime(a: &[u32], b: &[u32], output: &mut [u32]) {
    let i = quark_id();
    output[i] = a[i] & b[i];
}

#[quanta::kernel]
fn bitwise_or_xor(a: &[u32], b: &[u32], out_or: &mut [u32], out_xor: &mut [u32]) {
    let i = quark_id();
    out_or[i] = a[i] | b[i];
    out_xor[i] = a[i] ^ b[i];
}

#[test]
fn and_with_constant() {
    let Some(gpu) = try_gpu() else {
        return;
    };

    let input_data: Vec<u32> = vec![0, 1, 256, 0xFFFF, 0xDEAD, 0xFF];
    let n = input_data.len();
    let input = gpu.compute_field::<u32>(n).unwrap();
    let output = gpu.compute_field::<u32>(n).unwrap();
    gpu.write_field(&input, &input_data).unwrap();

    let mut wave = bitwise_and_const(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &output);
    let mut p = gpu.dispatch(&wave, n as u32).unwrap();
    gpu.wait(&mut p).unwrap();

    let result = gpu.read_field(&output).unwrap();
    let expected: Vec<u32> = input_data.iter().map(|x| x & 255).collect();
    for i in 0..n {
        assert_eq!(
            result[i], expected[i],
            "bitwise AND const: input {} & 255 = {}, got {}",
            input_data[i], expected[i], result[i]
        );
    }
}

#[test]
fn and_runtime_operands() {
    let Some(gpu) = try_gpu() else {
        return;
    };

    let a_data: Vec<u32> = vec![0xFF, 0x0F, 0xFF00, 0xABCD];
    let b_data: Vec<u32> = vec![0x0F, 0xFF, 0x00FF, 0xFF00];
    let n = a_data.len();

    let a = gpu.compute_field::<u32>(n).unwrap();
    let b = gpu.compute_field::<u32>(n).unwrap();
    let out = gpu.compute_field::<u32>(n).unwrap();
    gpu.write_field(&a, &a_data).unwrap();
    gpu.write_field(&b, &b_data).unwrap();

    let mut wave = bitwise_and_runtime(&gpu).unwrap();
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &out);
    let mut p = gpu.dispatch(&wave, n as u32).unwrap();
    gpu.wait(&mut p).unwrap();

    let result = gpu.read_field(&out).unwrap();
    for i in 0..n {
        let expected = a_data[i] & b_data[i];
        assert_eq!(
            result[i], expected,
            "bitwise AND runtime: {} & {} = {}, got {}",
            a_data[i], b_data[i], expected, result[i]
        );
    }
}

#[test]
fn or_and_xor() {
    let Some(gpu) = try_gpu() else {
        return;
    };

    let a_data: Vec<u32> = vec![0xF0, 0x0F, 0xAA, 0x55];
    let b_data: Vec<u32> = vec![0x0F, 0xF0, 0x55, 0xAA];
    let n = a_data.len();

    let a = gpu.compute_field::<u32>(n).unwrap();
    let b = gpu.compute_field::<u32>(n).unwrap();
    let out_or = gpu.compute_field::<u32>(n).unwrap();
    let out_xor = gpu.compute_field::<u32>(n).unwrap();
    gpu.write_field(&a, &a_data).unwrap();
    gpu.write_field(&b, &b_data).unwrap();

    let mut wave = bitwise_or_xor(&gpu).unwrap();
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &out_or);
    wave.bind(3, &out_xor);
    let mut p = gpu.dispatch(&wave, n as u32).unwrap();
    gpu.wait(&mut p).unwrap();

    let r_or = gpu.read_field(&out_or).unwrap();
    let r_xor = gpu.read_field(&out_xor).unwrap();
    for i in 0..n {
        assert_eq!(r_or[i], a_data[i] | b_data[i], "OR failed at {i}");
        assert_eq!(r_xor[i], a_data[i] ^ b_data[i], "XOR failed at {i}");
    }
}
