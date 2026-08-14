//! Differential tests for the Tier-3 f32 wrappers.
//!
//! `device_sort_f32` / `device_top_k_f32` are the monotone bijection at
//! the host boundary over `device_sort_u32`, so these tests pin the two
//! things the composition can break: the totalOrder placement of ±0.0,
//! ±inf and NaNs (asserted BITWISE), and the `u32::MAX` padding, which
//! is the keyed totalOrder maximum and must stay out of the result.
//!
//! Skips gracefully when no GPU backend is available.

#![cfg(feature = "gpu")]

use quanta_prims::{device_sort_f32, device_top_k_f32};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
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

fn oracle_sorted(data: &[f32]) -> Vec<f32> {
    let mut want = data.to_vec();
    want.sort_by(f32::total_cmp);
    want
}

fn assert_bits(got: &[f32], want: &[f32]) {
    let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
    let want_bits: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got_bits, want_bits);
}

// ── sort ────────────────────────────────────────────────────────

#[test]
fn sort_empty_and_single() {
    let Some(gpu) = try_gpu() else { return };
    assert_eq!(device_sort_f32(&gpu, &[]).unwrap(), Vec::<f32>::new());
    assert_eq!(device_sort_f32(&gpu, &[7.5]).unwrap(), vec![7.5]);
}

#[test]
fn sort_single_tile_path() {
    let Some(gpu) = try_gpu() else { return };
    // n ≤ 256 takes the block-sort fast path; 100 is also not a power of
    // two, exercising the keyed-MAX padding + truncation.
    let data = ramp(0xBEEF, 100);
    assert_bits(
        &device_sort_f32(&gpu, &data).unwrap(),
        &oracle_sorted(&data),
    );
}

#[test]
fn sort_global_network_odd_length() {
    let Some(gpu) = try_gpu() else { return };
    let data = ramp(0xFACE, 1000);
    assert_bits(
        &device_sort_f32(&gpu, &data).unwrap(),
        &oracle_sorted(&data),
    );
}

#[test]
fn sort_signed_zeros_and_nans() {
    let Some(gpu) = try_gpu() else { return };
    // totalOrder ascending: −NaN < −inf < finites < −0.0 < +0.0 <
    // finites < +inf < +NaN. Bitwise, so ±0.0 can't pass by comparing
    // equal.
    let mut data = ramp(0xA11CE, 300);
    data[3] = -0.0;
    data[4] = 0.0;
    data[5] = f32::INFINITY;
    data[6] = f32::NEG_INFINITY;
    data[7] = f32::NAN;
    data[8] = -f32::NAN;
    let got = device_sort_f32(&gpu, &data).unwrap();
    assert_bits(&got, &oracle_sorted(&data));
    assert_eq!(
        got[0].to_bits(),
        (-f32::NAN).to_bits(),
        "negative NaN first"
    );
    assert_eq!(got[1], f32::NEG_INFINITY);
    assert_eq!(got[299].to_bits(), f32::NAN.to_bits(), "positive NaN last");
}

#[test]
fn sort_padding_maximum_survives_in_the_data() {
    let Some(gpu) = try_gpu() else { return };
    // The pad value IS the keyed totalOrder maximum (u32::MAX unkeys to
    // this NaN): real copies of it must survive the truncation.
    let pad_twin = f32::from_bits(0x7FFF_FFFF);
    let mut data = ramp(0xD00D, 500);
    data[3] = pad_twin;
    data[400] = pad_twin;
    let got = device_sort_f32(&gpu, &data).unwrap();
    assert_eq!(got.len(), 500);
    assert_bits(&got, &oracle_sorted(&data));
}

#[test]
fn sort_already_sorted_and_reversed() {
    let Some(gpu) = try_gpu() else { return };
    let sorted: Vec<f32> = (0..512).map(|i| i as f32 - 256.0).collect();
    assert_bits(&device_sort_f32(&gpu, &sorted).unwrap(), &sorted);
    let reversed: Vec<f32> = sorted.iter().rev().copied().collect();
    assert_bits(&device_sort_f32(&gpu, &reversed).unwrap(), &sorted);
}

#[test]
fn sort_large_random() {
    let Some(gpu) = try_gpu() else { return };
    let data = ramp(0xCAFE, 100_000);
    assert_bits(
        &device_sort_f32(&gpu, &data).unwrap(),
        &oracle_sorted(&data),
    );
}

// ── top-k ───────────────────────────────────────────────────────

#[test]
fn top_k_is_the_descending_tail_of_the_sort() {
    let Some(gpu) = try_gpu() else { return };
    let data = ramp(0x5EED, 1000);
    let want: Vec<f32> = oracle_sorted(&data).into_iter().rev().take(10).collect();
    assert_bits(&device_top_k_f32(&gpu, &data, 10).unwrap(), &want);
}

#[test]
fn top_k_edges() {
    let Some(gpu) = try_gpu() else { return };
    let data = ramp(0x1234, 300);
    let sorted = oracle_sorted(&data);
    assert!(device_top_k_f32(&gpu, &data, 0).unwrap().is_empty());
    assert_bits(&device_top_k_f32(&gpu, &data, 1).unwrap(), &sorted[299..]);
    // k = n is a full descending sort.
    let all: Vec<f32> = sorted.iter().rev().copied().collect();
    assert_bits(&device_top_k_f32(&gpu, &data, 300).unwrap(), &all);
}

#[test]
fn top_k_beyond_the_input_is_an_error() {
    let Some(gpu) = try_gpu() else { return };
    assert!(device_top_k_f32(&gpu, &[1.0, 2.0], 3).is_err());
}

#[test]
fn top_k_surfaces_positive_nan_first() {
    let Some(gpu) = try_gpu() else { return };
    let mut data = ramp(0x7A15, 400);
    data[42] = f32::NAN;
    data[43] = f32::INFINITY;
    let got = device_top_k_f32(&gpu, &data, 3).unwrap();
    assert!(got[0].is_nan() && got[0].to_bits() >> 31 == 0);
    assert_eq!(got[1], f32::INFINITY);
}
