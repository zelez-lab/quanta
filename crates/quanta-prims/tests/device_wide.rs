//! Differential tests for the Tier-3 device-wide wrappers.
//!
//! The wrappers must agree with the CPU reference oracles on
//! arbitrary input lengths — in particular lengths that are not
//! multiples of 256 (reduce padding) and not powers of two (sort
//! padding), which the block-level tests never exercise.
//!
//! Skips gracefully when no GPU backend is available.

#![cfg(feature = "gpu")]

use quanta_prims::{
    device_reduce_add_f32, device_reduce_add_i32, device_reduce_add_u32, device_reduce_max_f32,
    device_reduce_max_i32, device_reduce_max_u32, device_reduce_min_f32, device_reduce_min_i32,
    device_reduce_min_u32, device_sort_u32, reference,
};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

/// Deterministic pseudo-random u32 stream (xorshift32), same
/// scheme as the block-level tests.
fn xorshift(seed: u32, n: usize) -> Vec<u32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        })
        .collect()
}

// ── reduce: padding edges ───────────────────────────────────────

#[test]
fn reduce_add_u32_single_element() {
    let Some(gpu) = try_gpu() else { return };
    assert_eq!(device_reduce_add_u32(&gpu, &[42]).unwrap(), 42);
}

#[test]
fn reduce_add_u32_non_multiple_of_block() {
    let Some(gpu) = try_gpu() else { return };
    // 1000 = 3 blocks + 232 padded lanes; identity padding must
    // not perturb the sum.
    let data: Vec<u32> = (1..=1000u32).collect();
    let expected = reference::reduce_add_u32(&data);
    assert_eq!(device_reduce_add_u32(&gpu, &data).unwrap(), expected);
}

#[test]
fn reduce_add_u32_multi_pass() {
    let Some(gpu) = try_gpu() else { return };
    // 70_000 elements → pass 1 leaves 274 partials → pass 2
    // leaves 2 → pass 3 leaves 1. Exercises ≥ 3 GPU passes.
    let data: Vec<u32> = xorshift(0xC0FFEE, 70_000)
        .into_iter()
        .map(|x| x % 1000) // keep the sum far from u32 overflow
        .collect();
    let expected = reference::reduce_add_u32(&data);
    assert_eq!(device_reduce_add_u32(&gpu, &data).unwrap(), expected);
}

#[test]
fn reduce_empty_input_is_an_error() {
    let Some(gpu) = try_gpu() else { return };
    assert!(device_reduce_add_u32(&gpu, &[]).is_err());
}

// ── reduce: min/max identity padding ────────────────────────────
//
// The padding lanes carry the op identity; a wrong identity is
// invisible on multiples of 256 and shows up exactly here.

#[test]
fn reduce_min_u32_padding_does_not_win() {
    let Some(gpu) = try_gpu() else { return };
    // All values large; identity must be u32::MAX, not 0, or the
    // padded lanes would "win" the min.
    let data: Vec<u32> = (0..300u32).map(|i| 1_000_000 + i).collect();
    let expected = reference::reduce_min_u32(&data);
    assert_eq!(device_reduce_min_u32(&gpu, &data).unwrap(), expected);
}

#[test]
fn reduce_max_i32_all_negative() {
    let Some(gpu) = try_gpu() else { return };
    // All negatives: identity must be i32::MIN, not 0.
    let data: Vec<i32> = (1..=300i32).map(|i| -i).collect();
    let expected = reference::reduce_max_i32(&data);
    assert_eq!(device_reduce_max_i32(&gpu, &data).unwrap(), expected);
}

#[test]
fn reduce_min_i32_with_negatives() {
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<i32> = (0..777i32)
        .map(|i| if i == 400 { -9999 } else { i })
        .collect();
    let expected = reference::reduce_min_i32(&data);
    assert_eq!(device_reduce_min_i32(&gpu, &data).unwrap(), expected);
}

#[test]
fn reduce_add_i32_mixed_signs() {
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<i32> = (0..1234i32)
        .map(|i| if i % 2 == 0 { i + 1 } else { -(i + 1) })
        .collect();
    let expected = reference::reduce_add_i32(&data);
    assert_eq!(device_reduce_add_i32(&gpu, &data).unwrap(), expected);
}

#[test]
fn reduce_min_max_f32_padding_identities() {
    let Some(gpu) = try_gpu() else { return };
    // 300 elements: min is negative, max well below the +INF /
    // -INF padding identities.
    let mut data: Vec<f32> = (0..300).map(|i| (i as f32) * 0.5 + 1.0).collect();
    data[123] = -3.25;
    data[200] = 4096.0;
    assert_eq!(device_reduce_min_f32(&gpu, &data).unwrap(), -3.25);
    assert_eq!(device_reduce_max_f32(&gpu, &data).unwrap(), 4096.0);
}

#[test]
fn reduce_add_f32_within_tolerance() {
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<f32> = (1..=10_000).map(|x| x as f32).collect();
    // Ground truth in f64: the sequential f32 reference fold
    // itself drifts here (the accumulator passes 2^24), and the
    // GPU tree order is *closer* to the true sum — so compare
    // against f64, not against the fold.
    let expected: f64 = data.iter().map(|&x| x as f64).sum();
    let got = device_reduce_add_f32(&gpu, &data).unwrap() as f64;
    let tol = expected.abs() * 1e-4;
    assert!(
        (got - expected).abs() < tol,
        "got {got}, expected {expected}, tol {tol}"
    );
}

#[test]
fn reduce_max_u32_position_independent() {
    let Some(gpu) = try_gpu() else { return };
    // Maximum sits in the final partial block, after the last
    // full-block boundary.
    let mut data = vec![10u32; 600];
    data[599] = 99_999;
    assert_eq!(device_reduce_max_u32(&gpu, &data).unwrap(), 99_999);
}

// ── sort ────────────────────────────────────────────────────────

#[test]
fn sort_empty_and_single() {
    let Some(gpu) = try_gpu() else { return };
    assert_eq!(device_sort_u32(&gpu, &[]).unwrap(), Vec::<u32>::new());
    assert_eq!(device_sort_u32(&gpu, &[7]).unwrap(), vec![7]);
}

#[test]
fn sort_single_tile_path() {
    let Some(gpu) = try_gpu() else { return };
    // n ≤ 256 takes the block-sort fast path; 100 is also not a
    // power of two, exercising the MAX padding + truncation.
    let data = xorshift(0xBEEF, 100);
    let expected = reference::radix_sort_u32(&data);
    assert_eq!(device_sort_u32(&gpu, &data).unwrap(), expected);
}

#[test]
fn sort_exactly_one_tile() {
    let Some(gpu) = try_gpu() else { return };
    let data = xorshift(0x5EED, 256);
    let expected = reference::radix_sort_u32(&data);
    assert_eq!(device_sort_u32(&gpu, &data).unwrap(), expected);
}

#[test]
fn sort_global_network_power_of_two() {
    let Some(gpu) = try_gpu() else { return };
    let data = xorshift(0xDEAD, 1024);
    let expected = reference::radix_sort_u32(&data);
    assert_eq!(device_sort_u32(&gpu, &data).unwrap(), expected);
}

#[test]
fn sort_global_network_odd_length() {
    let Some(gpu) = try_gpu() else { return };
    // 1000 pads to 1024 with u32::MAX; the padding must sort to
    // the (truncated) tail.
    let data = xorshift(0xFACE, 1000);
    let expected = reference::radix_sort_u32(&data);
    assert_eq!(device_sort_u32(&gpu, &data).unwrap(), expected);
}

#[test]
fn sort_with_duplicates_and_max_values() {
    let Some(gpu) = try_gpu() else { return };
    // Real u32::MAX values in the data must survive next to the
    // u32::MAX padding — count preservation is the test.
    let mut data = xorshift(0xA11CE, 500)
        .into_iter()
        .map(|x| x % 16)
        .collect::<Vec<_>>();
    data[3] = u32::MAX;
    data[400] = u32::MAX;
    let expected = reference::radix_sort_u32(&data);
    assert_eq!(device_sort_u32(&gpu, &data).unwrap(), expected);
}

#[test]
fn sort_already_sorted_and_reversed() {
    let Some(gpu) = try_gpu() else { return };
    let sorted: Vec<u32> = (0..512u32).collect();
    assert_eq!(device_sort_u32(&gpu, &sorted).unwrap(), sorted);
    let reversed: Vec<u32> = (0..512u32).rev().collect();
    assert_eq!(device_sort_u32(&gpu, &reversed).unwrap(), sorted);
}

#[test]
fn sort_large_random() {
    let Some(gpu) = try_gpu() else { return };
    // 100k elements → pads to 131072, 17 stages, 153 passes.
    let data = xorshift(0xCAFE, 100_000);
    let expected = reference::radix_sort_u32(&data);
    assert_eq!(device_sort_u32(&gpu, &data).unwrap(), expected);
}
