//! Differential tests for `block_radix_sort_kv_u32_buffer`, the
//! STABLE key-value LSD-radix sort.
//!
//! Because the kernel is stable, every test asserts the EXACT
//! `(keys, vals)` sequence against the stable reference — including
//! the duplicate-key cases, where the bitonic kv sort could only
//! check pair multisets. The stability-specific test seeds equal
//! keys with strictly increasing payloads and confirms the payloads
//! come back in input order.
//!
//! Skips gracefully when no GPU backend is available.

#![cfg(feature = "gpu")]

use quanta_prims::{block_radix_sort_kv_u32_buffer, reference};

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

fn run_kv(gpu: &quanta::Gpu, keys: &[u32], vals: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let n = keys.len();
    let keys_field = gpu.field::<u32>(n).unwrap();
    let vals_field = gpu.field::<u32>(n).unwrap();
    let keys_out = gpu.field::<u32>(n).unwrap();
    let vals_out = gpu.field::<u32>(n).unwrap();
    keys_field.write(keys).unwrap();
    vals_field.write(vals).unwrap();
    keys_out.write(&vec![0u32; n]).unwrap();
    vals_out.write(&vec![0u32; n]).unwrap();

    let mut wave = block_radix_sort_kv_u32_buffer(gpu).unwrap();
    wave.bind(0, &keys_field);
    wave.bind(1, &vals_field);
    wave.bind(2, &keys_out);
    wave.bind(3, &vals_out);
    gpu.dispatch(&wave, n as u32).unwrap().wait().unwrap();
    (keys_out.read().unwrap(), vals_out.read().unwrap())
}

/// Exact `(keys, vals)` equality against the stable reference —
/// valid for any input because the kernel is stable.
fn check_exact(gpu: &quanta::Gpu, keys: &[u32], vals: &[u32]) {
    let (gk, gv) = run_kv(gpu, keys, vals);
    let (rk, rv) = reference::sort_kv_stable_u32_blocks(keys, vals, BLOCK);
    assert_eq!(gk, rk, "keys mismatch");
    assert_eq!(gv, rv, "vals mismatch (stability or permutation broken)");
}

#[test]
fn identity_permutation() {
    let Some(gpu) = try_gpu() else { return };
    let keys: Vec<u32> = (0..BLOCK as u32).collect();
    let vals: Vec<u32> = (1000..1000 + BLOCK as u32).collect();
    check_exact(&gpu, &keys, &vals);
}

#[test]
fn reversed_keys_carry_vals() {
    let Some(gpu) = try_gpu() else { return };
    let keys: Vec<u32> = (0..BLOCK as u32).rev().collect();
    let vals: Vec<u32> = keys.iter().map(|k| k * 7 + 1).collect();
    check_exact(&gpu, &keys, &vals);
    // Independent invariant: each value still equals key*7+1.
    let (gk, gv) = run_kv(&gpu, &keys, &vals);
    for (k, v) in gk.iter().zip(gv.iter()) {
        assert_eq!(*v, k * 7 + 1, "value detached from its key");
    }
}

#[test]
fn random_keys_and_vals() {
    let Some(gpu) = try_gpu() else { return };
    let keys = xorshift(0x5A17, BLOCK);
    let vals = xorshift(0x7A15, BLOCK);
    check_exact(&gpu, &keys, &vals);
}

#[test]
fn duplicate_keys_are_stable() {
    let Some(gpu) = try_gpu() else { return };
    // Keys in 0..8 (heavy duplication); vals are the input index.
    // Stability ⇒ within each key group the indices come back
    // strictly increasing. Exact comparison catches any reorder.
    let keys: Vec<u32> = xorshift(0xD0B, BLOCK).into_iter().map(|x| x % 8).collect();
    let vals: Vec<u32> = (0..BLOCK as u32).collect();
    check_exact(&gpu, &keys, &vals);

    // Spell out the stability invariant directly: for each key
    // group, the emitted vals (= input indices) must be ascending.
    let (gk, gv) = run_kv(&gpu, &keys, &vals);
    let mut idx = 0usize;
    while idx < BLOCK {
        let key = gk[idx];
        let mut prev: Option<u32> = None;
        while idx < BLOCK && gk[idx] == key {
            if let Some(p) = prev {
                assert!(gv[idx] > p, "unstable: key {key} values out of input order");
            }
            prev = Some(gv[idx]);
            idx += 1;
        }
    }
}

#[test]
fn all_equal_keys_preserve_value_order() {
    let Some(gpu) = try_gpu() else { return };
    // Every key identical ⇒ a stable sort is the identity on values.
    let keys = vec![42u32; BLOCK];
    let vals: Vec<u32> = (0..BLOCK as u32).rev().collect();
    let (gk, gv) = run_kv(&gpu, &keys, &vals);
    assert_eq!(gk, keys);
    assert_eq!(gv, vals, "all-equal keys must leave values untouched");
}

#[test]
fn extreme_key_values() {
    let Some(gpu) = try_gpu() else { return };
    // 0, u32::MAX, and high-bit-only keys exercise every digit pass.
    let mut keys = vec![0u32; BLOCK];
    keys[10] = u32::MAX;
    keys[20] = 0x8000_0000;
    keys[30] = 0x7FFF_FFFF;
    keys[40] = u32::MAX;
    let vals: Vec<u32> = (0..BLOCK as u32).collect();
    check_exact(&gpu, &keys, &vals);
}

#[test]
fn multiple_blocks_sort_independently() {
    let Some(gpu) = try_gpu() else { return };
    let n = 4 * BLOCK;
    let keys: Vec<u32> = xorshift(0xB10C, n).into_iter().map(|x| x % 50).collect();
    let vals: Vec<u32> = (0..n as u32).collect();
    check_exact(&gpu, &keys, &vals);
}
