//! Differential tests for `block_segmented_top_k_f32_buffer` — the
//! batch-kNN shape: B fixed-width segments of `seg_len` f32 keys, the k
//! largest of each segment under totalOrder.
//!
//! Every comparison against the host oracle is BITWISE (`to_bits`), so
//! NaN payloads and signed zeros are pinned exactly, not approximately.
//!
//! Skips gracefully when no GPU backend is available.

#![cfg(feature = "gpu")]

use quanta_prims::{block_segmented_top_k_f32_buffer, reference};

const BLOCK: usize = 256;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

fn run_segmented_top_k(gpu: &quanta::Gpu, data: &[f32], seg_len: u32, k: u32) -> Vec<f32> {
    let out_len = (data.len() / seg_len as usize) * k as usize;

    let in_field = gpu.field::<f32>(data.len()).unwrap();
    let out_field = gpu.field::<f32>(out_len).unwrap();
    in_field.write(data).unwrap();
    out_field.write(&vec![0.0f32; out_len]).unwrap();

    let mut wave = block_segmented_top_k_f32_buffer(gpu).unwrap();
    wave.bind(0, &in_field);
    wave.bind(1, &out_field);
    wave.set_value(2, seg_len);
    wave.set_value(3, k);
    let mut pulse = gpu.dispatch(&wave, data.len() as u32).unwrap();
    pulse.wait().unwrap();

    out_field.read().unwrap()
}

fn check_bitwise(data: &[f32], got: &[f32], seg_len: u32, k: u32) {
    let out_len = (data.len() / seg_len as usize) * k as usize;
    let mut expected = vec![0.0f32; out_len];
    reference::segmented_top_k_f32_blocks(data, &mut expected, seg_len as usize, k as usize);
    let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
    let want_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got_bits, want_bits);
}

/// Deterministic pseudo-random f32 stream spanning both signs.
fn ramp(seed: u32, n: usize) -> Vec<f32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            ((x >> 8) as f32 / 1_000_000.0) - 8.0
        })
        .collect()
}

#[test]
fn segments_of_sixteen_mixed_signs() {
    let Some(gpu) = try_gpu() else { return };
    // 16 segments per block — the batch shape with a small candidate set.
    let data = ramp(0x1234_5678, BLOCK * 2);
    let got = run_segmented_top_k(&gpu, &data, 16, 4);
    check_bitwise(&data, &got, 16, 4);
}

#[test]
fn segments_are_independent() {
    let Some(gpu) = try_gpu() else { return };
    // Segment s holds the ramp s*100 + t: the top-1 of each segment is
    // its own last element, and nothing bleeds across a boundary.
    let seg_len = 64usize;
    let data: Vec<f32> = (0..BLOCK)
        .map(|i| ((i / seg_len) * 100 + i % seg_len) as f32)
        .collect();
    let got = run_segmented_top_k(&gpu, &data, seg_len as u32, 1);
    check_bitwise(&data, &got, seg_len as u32, 1);
    assert_eq!(got, vec![63.0, 163.0, 263.0, 363.0]);
}

#[test]
fn nan_in_one_segment_only() {
    let Some(gpu) = try_gpu() else { return };
    // A positive NaN ranks above +inf, a negative NaN below −inf — and
    // only inside the segment that carries it.
    let seg_len = 32usize;
    let mut data = ramp(0xBEEF, BLOCK);
    data[seg_len + 3] = f32::NAN;
    data[seg_len + 7] = f32::INFINITY;
    data[seg_len + 9] = -f32::NAN;
    let got = run_segmented_top_k(&gpu, &data, seg_len as u32, 3);
    check_bitwise(&data, &got, seg_len as u32, 3);
    assert!(
        got[3].is_nan() && got[3].to_bits() >> 31 == 0,
        "positive NaN first in the segment that has one"
    );
    assert_eq!(got[4], f32::INFINITY);
    assert!(!got[0].is_nan(), "the neighbouring segment is untouched");
}

#[test]
fn signed_zeros_order_totally() {
    let Some(gpu) = try_gpu() else { return };
    // A block of ±0.0 in 8-wide segments: totalOrder says −0.0 < +0.0,
    // so every emitted top-4 is the +0.0 bit pattern.
    let mut data = vec![0.0f32; BLOCK];
    for (i, v) in data.iter_mut().enumerate() {
        if i % 2 == 0 {
            *v = -0.0;
        }
    }
    let got = run_segmented_top_k(&gpu, &data, 8, 4);
    check_bitwise(&data, &got, 8, 4);
    assert!(
        got.iter().all(|v| v.to_bits() == 0),
        "+0.0 ranks above −0.0"
    );
}

#[test]
fn k_of_one_is_the_segment_maximum() {
    let Some(gpu) = try_gpu() else { return };
    let data = ramp(0xF00D, BLOCK);
    let got = run_segmented_top_k(&gpu, &data, 32, 1);
    check_bitwise(&data, &got, 32, 1);
    for (s, chunk) in data.chunks(32).enumerate() {
        let want = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(got[s], want, "segment {s} maximum");
    }
}

#[test]
fn k_equals_segment_length_is_a_descending_sort() {
    let Some(gpu) = try_gpu() else { return };
    // k = seg_len: the whole segment comes back, descending.
    let data = ramp(0xC0FFEE, BLOCK);
    let got = run_segmented_top_k(&gpu, &data, 16, 16);
    check_bitwise(&data, &got, 16, 16);
    for chunk in got.chunks(16) {
        assert!(chunk.windows(2).all(|w| w[0] >= w[1]), "descending");
    }
}

#[test]
fn one_segment_per_block_matches_plain_top_k() {
    let Some(gpu) = try_gpu() else { return };
    // seg_len = 256 degenerates to the per-block top-k.
    let data = ramp(0xD15EA5E, BLOCK * 3);
    let got = run_segmented_top_k(&gpu, &data, BLOCK as u32, 8);
    check_bitwise(&data, &got, BLOCK as u32, 8);
}

#[test]
fn singleton_segments_pass_through() {
    let Some(gpu) = try_gpu() else { return };
    // seg_len = 1, k = 1: every element is its own segment maximum.
    let data = ramp(0x51D, BLOCK);
    let got = run_segmented_top_k(&gpu, &data, 1, 1);
    check_bitwise(&data, &got, 1, 1);
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        data.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn multiple_blocks_number_segments_globally() {
    let Some(gpu) = try_gpu() else { return };
    // 4 blocks × 4 segments: out[seg * k] must follow the GLOBAL segment
    // index, not restart per block.
    let seg_len = 64usize;
    let data: Vec<f32> = (0..BLOCK * 4)
        .map(|i| ((i / seg_len) as f32) * 1000.0 + (i % seg_len) as f32)
        .collect();
    let got = run_segmented_top_k(&gpu, &data, seg_len as u32, 2);
    check_bitwise(&data, &got, seg_len as u32, 2);
    for s in 0..16 {
        assert_eq!(got[s * 2], s as f32 * 1000.0 + 63.0, "segment {s} top-1");
    }
}
