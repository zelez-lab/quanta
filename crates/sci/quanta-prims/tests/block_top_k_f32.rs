//! `block_top_k_f32_buffer` — the monotone-bijection composition over
//! the u32 bitonic network (totalOrder descending). Every comparison
//! against the host oracle is BITWISE (`to_bits`), so NaN payloads and
//! signed zeros are pinned exactly, not approximately.

use quanta_prims::reference;

const BLOCK: usize = 256;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

fn run_top_k_f32(gpu: &quanta::Gpu, data: &[f32], k: u32) -> Vec<f32> {
    let num_blocks = data.len() / BLOCK;
    let out_len = num_blocks * k as usize;

    let in_field = gpu.field::<f32>(data.len()).unwrap();
    let out_field = gpu.field::<f32>(out_len).unwrap();
    in_field.write(data).unwrap();
    out_field.write(&vec![0.0f32; out_len]).unwrap();

    let mut wave = quanta_prims::block_top_k_f32_buffer(gpu).unwrap();
    wave.bind(0, &in_field);
    wave.bind(1, &out_field);
    wave.set_value(2, k);
    let mut pulse = gpu.dispatch(&wave, data.len() as u32).unwrap();
    pulse.wait().unwrap();

    out_field.read().unwrap()
}

fn check_bitwise(data: &[f32], got: &[f32], k: u32) {
    let num_blocks = data.len() / BLOCK;
    let mut expected = vec![0.0f32; num_blocks * k as usize];
    reference::top_k_f32_blocks(data, &mut expected, BLOCK, k as usize);
    let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
    let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got_bits, want_bits);
}

#[test]
fn f32_top_k_mixed_signs() {
    let Some(gpu) = try_gpu() else { return };
    // Ramp crossing zero: −128.5 … +127.5 in one block.
    let data: Vec<f32> = (0..BLOCK).map(|i| i as f32 - 128.5).collect();
    let got = run_top_k_f32(&gpu, &data, 8);
    check_bitwise(&data, &got, 8);
    assert_eq!(got[0], 126.5); // largest
}

#[test]
fn f32_top_k_signed_zeros_order_totally() {
    let Some(gpu) = try_gpu() else { return };
    // A block full of ±0.0: totalOrder says −0.0 < +0.0, so the
    // descending top half is all +0.0 bit patterns.
    let mut data = vec![0.0f32; BLOCK];
    for (i, v) in data.iter_mut().enumerate() {
        if i % 2 == 0 {
            *v = -0.0;
        }
    }
    let got = run_top_k_f32(&gpu, &data, 16);
    check_bitwise(&data, &got, 16);
    assert!(
        got.iter().all(|v| v.to_bits() == 0),
        "+0.0 ranks above −0.0"
    );
}

#[test]
fn f32_top_k_nan_policy_is_total_order() {
    let Some(gpu) = try_gpu() else { return };
    // One quiet positive NaN, one negative NaN, ±inf, and finites:
    // descending totalOrder = +NaN, +inf, finites…, −inf, −NaN.
    let mut data: Vec<f32> = (0..BLOCK).map(|i| i as f32).collect();
    data[10] = f32::NAN;
    data[20] = f32::INFINITY;
    data[30] = f32::NEG_INFINITY;
    data[40] = -f32::NAN;
    let got = run_top_k_f32(&gpu, &data, 4);
    check_bitwise(&data, &got, 4);
    assert!(
        got[0].is_nan() && got[0].to_bits() >> 31 == 0,
        "positive NaN first"
    );
    assert_eq!(got[1], f32::INFINITY);
}

#[test]
fn f32_top_k_multi_block() {
    let Some(gpu) = try_gpu() else { return };
    // 4 blocks, pseudo-random values incl. negatives (LCG, no deps).
    let mut x = 0x1234_5678u32;
    let data: Vec<f32> = (0..BLOCK * 4)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            ((x >> 8) as f32 / 1_000_000.0) - 8.0
        })
        .collect();
    let got = run_top_k_f32(&gpu, &data, 32);
    check_bitwise(&data, &got, 32);
}
