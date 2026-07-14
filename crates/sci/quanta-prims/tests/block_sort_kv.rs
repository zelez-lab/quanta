//! Differential tests for `block_sort_kv_u32_buffer`.
//!
//! The bitonic network is unstable, so the comparison strategy
//! splits by key uniqueness: unique keys → exact (keys, vals)
//! equality against the stable reference; duplicate keys → keys
//! exact, (key, value) pair multiset preserved.
//!
//! Skips gracefully when no GPU backend is available.

#![cfg(feature = "gpu")]

use quanta_prims::{block_sort_kv_u32_buffer, reference};

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

    let mut wave = block_sort_kv_u32_buffer(gpu).unwrap();
    wave.bind(0, &keys_field);
    wave.bind(1, &vals_field);
    wave.bind(2, &keys_out);
    wave.bind(3, &vals_out);
    gpu.dispatch(&wave, n as u32).unwrap().wait().unwrap();
    (keys_out.read().unwrap(), vals_out.read().unwrap())
}

/// Exact comparison — valid only when keys are unique per block.
fn check_unique(gpu: &quanta::Gpu, keys: &[u32], vals: &[u32]) {
    let (gk, gv) = run_kv(gpu, keys, vals);
    let (rk, rv) = reference::sort_kv_u32_blocks(keys, vals, BLOCK);
    assert_eq!(gk, rk, "keys mismatch");
    assert_eq!(gv, rv, "vals mismatch");
}

#[test]
fn identity_permutation_unique_keys() {
    let Some(gpu) = try_gpu() else { return };
    // Already sorted: vals must come back untouched.
    let keys: Vec<u32> = (0..BLOCK as u32).collect();
    let vals: Vec<u32> = (1000..1000 + BLOCK as u32).collect();
    check_unique(&gpu, &keys, &vals);
}

#[test]
fn reversed_keys_carry_vals() {
    let Some(gpu) = try_gpu() else { return };
    let keys: Vec<u32> = (0..BLOCK as u32).rev().collect();
    // val = key * 7 + 1: after sorting, vals[i] must equal keys[i]*7+1.
    let vals: Vec<u32> = keys.iter().map(|k| k * 7 + 1).collect();
    let (gk, gv) = run_kv(&gpu, &keys, &vals);
    for (k, v) in gk.iter().zip(gv.iter()) {
        assert_eq!(*v, k * 7 + 1, "value detached from its key");
    }
    let expected: Vec<u32> = (0..BLOCK as u32).collect();
    assert_eq!(gk, expected);
}

#[test]
fn random_unique_keys_multi_block() {
    let Some(gpu) = try_gpu() else { return };
    // 4 blocks. Random-looking but unique keys per block: take a
    // random base permutation of 0..256 per block, offset by block.
    let n = 4 * BLOCK;
    let mut keys = Vec::with_capacity(n);
    for b in 0..4u32 {
        let mut perm: Vec<u32> = (0..BLOCK as u32).map(|i| i + b * 10_000).collect();
        // Fisher-Yates with the xorshift stream.
        let rnd = xorshift(0x4B5E_u32 ^ b, BLOCK);
        for i in (1..BLOCK).rev() {
            let j = (rnd[i] as usize) % (i + 1);
            perm.swap(i, j);
        }
        keys.extend(perm);
    }
    let vals: Vec<u32> = keys
        .iter()
        .map(|k| k.wrapping_mul(31).wrapping_add(5))
        .collect();
    check_unique(&gpu, &keys, &vals);
}

#[test]
fn duplicate_keys_preserve_pair_multiset() {
    let Some(gpu) = try_gpu() else { return };
    // Heavy duplication: keys in 0..8. Keys must sort exactly;
    // (key, val) pairs must be the same multiset (no value lost,
    // duplicated, or attached to the wrong key).
    let keys: Vec<u32> = xorshift(0xD0B, BLOCK).into_iter().map(|x| x % 8).collect();
    let vals: Vec<u32> = (0..BLOCK as u32).collect();
    let (gk, gv) = run_kv(&gpu, &keys, &vals);

    let (rk, _) = reference::sort_kv_u32_blocks(&keys, &vals, BLOCK);
    assert_eq!(gk, rk, "keys mismatch");

    let mut got_pairs: Vec<(u32, u32)> = gk.into_iter().zip(gv).collect();
    let mut want_pairs: Vec<(u32, u32)> = keys.into_iter().zip(vals).collect();
    got_pairs.sort_unstable();
    want_pairs.sort_unstable();
    assert_eq!(got_pairs, want_pairs, "pair multiset not preserved");
}

#[test]
fn all_equal_keys_keep_value_multiset() {
    let Some(gpu) = try_gpu() else { return };
    let keys = vec![42u32; BLOCK];
    let vals: Vec<u32> = (0..BLOCK as u32).rev().collect();
    let (gk, mut gv) = run_kv(&gpu, &keys, &vals);
    assert_eq!(gk, keys);
    gv.sort_unstable();
    let expected: Vec<u32> = (0..BLOCK as u32).collect();
    assert_eq!(gv, expected);
}
