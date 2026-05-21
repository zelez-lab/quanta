//! Differential tests for `block_compact_u32_buffer`. The GPU
//! kernel writes kept entries contiguously inside each 256-element
//! block; the CPU reference does the same. We compare both the
//! per-block `counts` and the prefix of `out` up to each block's
//! kept count.
//!
//! Skips gracefully when no GPU backend is available.

#![cfg(feature = "gpu")]

use quanta_prims::{block_compact_u32_buffer, reference};

const BLOCK: usize = 256;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

fn run_compact(gpu: &quanta::Gpu, predicates: &[u32], data: &[u32]) -> (Vec<u32>, Vec<u32>) {
    assert_eq!(predicates.len(), data.len());
    let num_blocks = data.len() / BLOCK;

    let preds_field = gpu.field::<u32>(predicates.len()).unwrap();
    let data_field = gpu.field::<u32>(data.len()).unwrap();
    let out_field = gpu.field::<u32>(data.len()).unwrap();
    let counts_field = gpu.field::<u32>(num_blocks).unwrap();

    preds_field.write(predicates).unwrap();
    data_field.write(data).unwrap();
    out_field.write(&vec![0u32; data.len()]).unwrap();
    counts_field.write(&vec![0u32; num_blocks]).unwrap();

    let mut wave = block_compact_u32_buffer(gpu).unwrap();
    wave.bind(0, &preds_field);
    wave.bind(1, &data_field);
    wave.bind(2, &out_field);
    wave.bind(3, &counts_field);
    let mut pulse = gpu.dispatch(&wave, data.len() as u32).unwrap();
    pulse.wait().unwrap();

    (out_field.read().unwrap(), counts_field.read().unwrap())
}

fn check_compact(predicates: &[u32], data: &[u32], got_out: &[u32], got_counts: &[u32]) {
    let num_blocks = data.len() / BLOCK;
    let mut expected_out = vec![0u32; data.len()];
    let mut expected_counts = vec![0u32; num_blocks];
    reference::compact_u32_blocks(
        predicates,
        data,
        &mut expected_out,
        &mut expected_counts,
        BLOCK,
    );
    assert_eq!(
        got_counts,
        &expected_counts[..],
        "per-block kept counts disagree"
    );
    for b in 0..num_blocks {
        let start = b * BLOCK;
        let kept = expected_counts[b] as usize;
        assert_eq!(
            &got_out[start..start + kept],
            &expected_out[start..start + kept],
            "block {b}: kept values disagree"
        );
    }
}

#[test]
fn compact_alternating_kept() {
    // Block 0: keep every even index → 128 kept.
    let Some(gpu) = try_gpu() else { return };
    let predicates: Vec<u32> = (0..BLOCK as u32).map(|i| (i + 1) % 2).collect(); // 1,0,1,0,...
    let data: Vec<u32> = (0..BLOCK as u32).collect();
    let (out, counts) = run_compact(&gpu, &predicates, &data);
    assert_eq!(counts, vec![128]);
    check_compact(&predicates, &data, &out, &counts);
}

#[test]
fn compact_all_kept() {
    let Some(gpu) = try_gpu() else { return };
    let predicates = vec![1u32; BLOCK];
    let data: Vec<u32> = (0..BLOCK as u32).collect();
    let (out, counts) = run_compact(&gpu, &predicates, &data);
    assert_eq!(counts, vec![BLOCK as u32]);
    check_compact(&predicates, &data, &out, &counts);
    // Output should be identical to input when nothing is dropped.
    assert_eq!(out, data);
}

#[test]
fn compact_all_dropped() {
    let Some(gpu) = try_gpu() else { return };
    let predicates = vec![0u32; BLOCK];
    let data: Vec<u32> = (0..BLOCK as u32).collect();
    let (out, counts) = run_compact(&gpu, &predicates, &data);
    assert_eq!(counts, vec![0]);
    // Output untouched (all initial zeros).
    assert_eq!(out, vec![0u32; BLOCK]);
}

#[test]
fn compact_sparse() {
    // Keep only every 16th index — exercises the inclusive-scan
    // sparse-1 path; offsets should be 0,1,2,...,15.
    let Some(gpu) = try_gpu() else { return };
    let predicates: Vec<u32> = (0..BLOCK as u32)
        .map(|i| if i % 16 == 0 { 1 } else { 0 })
        .collect();
    let data: Vec<u32> = (0..BLOCK as u32).map(|i| i * 10).collect();
    let (out, counts) = run_compact(&gpu, &predicates, &data);
    assert_eq!(counts, vec![16]);
    check_compact(&predicates, &data, &out, &counts);
    // Explicit values: kept = [0, 160, 320, ..., 16*15*10 = 2400].
    let expected: Vec<u32> = (0..16u32).map(|k| k * 16 * 10).collect();
    assert_eq!(&out[0..16], &expected[..]);
}

#[test]
fn compact_multi_block() {
    // Two blocks with different keep patterns to verify per-block
    // independence: block 0 keeps even, block 1 keeps odd.
    let Some(gpu) = try_gpu() else { return };
    let mut predicates = vec![0u32; 2 * BLOCK];
    let mut data = vec![0u32; 2 * BLOCK];
    for i in 0..BLOCK {
        predicates[i] = ((i + 1) % 2) as u32; // even
        data[i] = i as u32;
        predicates[BLOCK + i] = (i % 2) as u32; // odd
        data[BLOCK + i] = (BLOCK + i) as u32;
    }
    let (out, counts) = run_compact(&gpu, &predicates, &data);
    assert_eq!(counts, vec![128, 128]);
    check_compact(&predicates, &data, &out, &counts);
}
