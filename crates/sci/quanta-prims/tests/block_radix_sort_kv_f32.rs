//! Differential tests for `block_radix_sort_kv_f32u32_buffer`, the
//! STABLE f32-key / u32-payload LSD-radix sort.
//!
//! The monotone bijection is injective on bit patterns, so the
//! composition is exactly as stable as the u32 radix underneath it:
//! every test asserts the EXACT `(keys, vals)` sequence against the
//! stable reference, keys compared BITWISE (`to_bits`) so ±0.0 and NaN
//! payloads are pinned exactly, not approximately.
//!
//! Skips gracefully when no GPU backend is available.

#![cfg(feature = "gpu")]

use quanta_prims::{block_radix_sort_kv_f32u32_buffer, reference};

const BLOCK: usize = 256;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

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

fn run_kv(gpu: &quanta::Gpu, keys: &[f32], vals: &[u32]) -> (Vec<f32>, Vec<u32>) {
    let n = keys.len();
    let keys_field = gpu.field::<f32>(n).unwrap();
    let vals_field = gpu.field::<u32>(n).unwrap();
    let keys_out = gpu.field::<f32>(n).unwrap();
    let vals_out = gpu.field::<u32>(n).unwrap();
    keys_field.write(keys).unwrap();
    vals_field.write(vals).unwrap();
    keys_out.write(&vec![0.0f32; n]).unwrap();
    vals_out.write(&vec![0u32; n]).unwrap();

    let mut wave = block_radix_sort_kv_f32u32_buffer(gpu).unwrap();
    wave.bind(0, &keys_field);
    wave.bind(1, &vals_field);
    wave.bind(2, &keys_out);
    wave.bind(3, &vals_out);
    gpu.dispatch(&wave, n as u32).unwrap().wait().unwrap();
    (keys_out.read().unwrap(), vals_out.read().unwrap())
}

/// Exact `(keys, vals)` equality against the stable reference, keys
/// compared bit-for-bit — valid for any input because the kernel is
/// stable and the bijection is a bit-pattern bijection.
fn check_exact(gpu: &quanta::Gpu, keys: &[f32], vals: &[u32]) {
    let (gk, gv) = run_kv(gpu, keys, vals);
    let (rk, rv) = reference::radix_sort_kv_f32u32_blocks(keys, vals, BLOCK);
    let got_bits: Vec<u32> = gk.iter().map(|v| v.to_bits()).collect();
    let want_bits: Vec<u32> = rk.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got_bits, want_bits, "keys mismatch");
    assert_eq!(gv, rv, "vals mismatch (stability or permutation broken)");
}

#[test]
fn ramp_crossing_zero() {
    let Some(gpu) = try_gpu() else { return };
    // −128.5 … +127.5 in one block: the sign-bit half of the bijection
    // is what puts the negatives below the positives.
    let keys: Vec<f32> = (0..BLOCK).map(|i| i as f32 - 128.5).collect();
    let vals: Vec<u32> = (1000..1000 + BLOCK as u32).collect();
    check_exact(&gpu, &keys, &vals);
}

#[test]
fn reversed_keys_carry_vals() {
    let Some(gpu) = try_gpu() else { return };
    let keys: Vec<f32> = (0..BLOCK).map(|i| (BLOCK - i) as f32 * 0.25).collect();
    let vals: Vec<u32> = (0..BLOCK as u32).collect();
    check_exact(&gpu, &keys, &vals);
    // Independent invariant: the payload still names its key's input slot.
    let (gk, gv) = run_kv(&gpu, &keys, &vals);
    for (k, v) in gk.iter().zip(gv.iter()) {
        assert_eq!(*k, keys[*v as usize], "value detached from its key");
    }
}

#[test]
fn random_mixed_sign_keys() {
    let Some(gpu) = try_gpu() else { return };
    let keys: Vec<f32> = xorshift(0x5A17, BLOCK)
        .into_iter()
        .map(|x| ((x >> 8) as f32 / 1_000_000.0) - 8.0)
        .collect();
    let vals = xorshift(0x7A15, BLOCK);
    check_exact(&gpu, &keys, &vals);
}

#[test]
fn duplicate_keys_are_stable() {
    let Some(gpu) = try_gpu() else { return };
    // Keys in a tiny set (heavy duplication); vals are the input index.
    // Stability ⇒ within each key group the indices come back strictly
    // increasing. Exact comparison catches any reorder.
    let keys: Vec<f32> = xorshift(0xD0B, BLOCK)
        .into_iter()
        .map(|x| (x % 8) as f32 - 4.0)
        .collect();
    let vals: Vec<u32> = (0..BLOCK as u32).collect();
    check_exact(&gpu, &keys, &vals);

    let (gk, gv) = run_kv(&gpu, &keys, &vals);
    let mut idx = 0usize;
    while idx < BLOCK {
        let key = gk[idx].to_bits();
        let mut prev: Option<u32> = None;
        while idx < BLOCK && gk[idx].to_bits() == key {
            if let Some(p) = prev {
                assert!(
                    gv[idx] > p,
                    "unstable: key {key:#x} values out of input order"
                );
            }
            prev = Some(gv[idx]);
            idx += 1;
        }
    }
}

#[test]
fn signed_zeros_and_nans_order_totally() {
    let Some(gpu) = try_gpu() else { return };
    // totalOrder ascending: −NaN < −inf < finites < +inf < +NaN, and
    // −0.0 < +0.0. Payload rides along through every one of them.
    let mut keys: Vec<f32> = (0..BLOCK).map(|i| i as f32 - 128.0).collect();
    keys[10] = -0.0;
    keys[11] = 0.0;
    keys[20] = f32::INFINITY;
    keys[30] = f32::NEG_INFINITY;
    keys[40] = f32::NAN;
    keys[50] = -f32::NAN;
    let vals: Vec<u32> = (0..BLOCK as u32).collect();
    check_exact(&gpu, &keys, &vals);

    let (gk, gv) = run_kv(&gpu, &keys, &vals);
    assert_eq!(gv[0], 50, "negative NaN sorts below everything");
    assert_eq!(gk[1], f32::NEG_INFINITY);
    assert_eq!(gk[BLOCK - 1].to_bits(), f32::NAN.to_bits());
    assert_eq!(gk[BLOCK - 2], f32::INFINITY);
}

#[test]
fn all_equal_keys_preserve_value_order() {
    let Some(gpu) = try_gpu() else { return };
    // Every key identical ⇒ a stable sort is the identity on values.
    let keys = vec![-2.5f32; BLOCK];
    let vals: Vec<u32> = (0..BLOCK as u32).rev().collect();
    let (gk, gv) = run_kv(&gpu, &keys, &vals);
    assert_eq!(gk, keys);
    assert_eq!(gv, vals, "all-equal keys must leave values untouched");
}

#[test]
fn multiple_blocks_sort_independently() {
    let Some(gpu) = try_gpu() else { return };
    let n = 4 * BLOCK;
    let keys: Vec<f32> = xorshift(0xB10C, n)
        .into_iter()
        .map(|x| (x % 50) as f32 * 0.5 - 12.0)
        .collect();
    let vals: Vec<u32> = (0..n as u32).collect();
    check_exact(&gpu, &keys, &vals);
}
