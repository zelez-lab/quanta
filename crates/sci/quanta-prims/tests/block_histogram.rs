//! Differential tests for `block_histogram_u32_buffer`. The GPU
//! kernel atomically increments a shared `local_counts[256]` per
//! workgroup and copies the per-bucket totals to the output; the
//! CPU reference does the same per-block tally. We compare block
//! by block.
//!
//! Shared-memory atomics emit on Metal, SPIR-V (Vulkan), and
//! WGSL (WebGPU); the software CPU-JIT still refuses (its shared
//! memory is per-thread scratch, so atomics alone can't make the
//! kernel cooperative). The test treats NotSupported as a skip
//! rather than a failure so the suite stays green there.

#![cfg(feature = "gpu")]

use quanta_prims::{block_histogram_u32_buffer, reference};

const BLOCK: usize = 256;
const NUM_BUCKETS: usize = 256;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

fn run_histogram(gpu: &quanta::Gpu, buckets_in: &[u32]) -> Option<Vec<u32>> {
    let num_blocks = buckets_in.len() / BLOCK;
    let out_len = num_blocks * NUM_BUCKETS;

    let in_field = gpu.field::<u32>(buckets_in.len()).unwrap();
    let out_field = gpu.field::<u32>(out_len).unwrap();
    in_field.write(buckets_in).unwrap();
    out_field.write(&vec![0u32; out_len]).unwrap();

    let mut wave = block_histogram_u32_buffer(gpu).unwrap();
    wave.bind(0, &in_field);
    wave.bind(1, &out_field);

    match gpu.dispatch(&wave, buckets_in.len() as u32) {
        Ok(mut pulse) => {
            pulse.wait().unwrap();
            Some(out_field.read().unwrap())
        }
        Err(e) => {
            let msg = format!("{e:?}");
            if msg.contains("NotSupported") || msg.contains("not yet supported") {
                eprintln!("backend doesn't support shared atomics: {msg}");
                None
            } else {
                panic!("unexpected dispatch error: {msg}");
            }
        }
    }
}

fn check_histogram(buckets_in: &[u32], got: &[u32]) {
    let num_blocks = buckets_in.len() / BLOCK;
    let mut expected = vec![0u32; num_blocks * NUM_BUCKETS];
    reference::histogram_u32_blocks(buckets_in, &mut expected, BLOCK, NUM_BUCKETS);
    assert_eq!(got, &expected[..], "bucket counts disagree");
}

#[test]
fn histogram_uniform_low_buckets() {
    // 256 inputs all into buckets 0..4 (round-robin) — every
    // bucket gets 64 hits.
    let Some(gpu) = try_gpu() else { return };
    let buckets_in: Vec<u32> = (0..BLOCK as u32).map(|i| i % 4).collect();
    let Some(out) = run_histogram(&gpu, &buckets_in) else {
        return;
    };
    assert_eq!(out[0], 64);
    assert_eq!(out[1], 64);
    assert_eq!(out[2], 64);
    assert_eq!(out[3], 64);
    for o in out.iter().take(NUM_BUCKETS).skip(4) {
        assert_eq!(*o, 0);
    }
    check_histogram(&buckets_in, &out);
}

#[test]
fn histogram_all_same_bucket() {
    let Some(gpu) = try_gpu() else { return };
    let buckets_in = vec![42u32; BLOCK];
    let Some(out) = run_histogram(&gpu, &buckets_in) else {
        return;
    };
    assert_eq!(out[42], BLOCK as u32);
    for (i, o) in out.iter().take(NUM_BUCKETS).enumerate() {
        if i != 42 {
            assert_eq!(*o, 0, "bucket {i} should be 0");
        }
    }
    check_histogram(&buckets_in, &out);
}

#[test]
fn histogram_full_spread() {
    // 256 inputs, one per bucket. Every bucket gets exactly 1.
    let Some(gpu) = try_gpu() else { return };
    let buckets_in: Vec<u32> = (0..BLOCK as u32).collect();
    let Some(out) = run_histogram(&gpu, &buckets_in) else {
        return;
    };
    for o in out.iter().take(NUM_BUCKETS) {
        assert_eq!(*o, 1);
    }
    check_histogram(&buckets_in, &out);
}

#[test]
fn histogram_multi_block() {
    let Some(gpu) = try_gpu() else { return };
    // Block 0: bucket = i % 8. Block 1: bucket = i % 16.
    let mut buckets_in = vec![0u32; 2 * BLOCK];
    for i in 0..BLOCK {
        buckets_in[i] = (i % 8) as u32;
        buckets_in[BLOCK + i] = (i % 16) as u32;
    }
    let Some(out) = run_histogram(&gpu, &buckets_in) else {
        return;
    };
    check_histogram(&buckets_in, &out);
    // Block 0: each of 0..8 gets BLOCK/8 = 32 hits.
    for o in out.iter().take(8) {
        assert_eq!(*o, 32);
    }
    // Block 1: each of 0..16 gets BLOCK/16 = 16 hits.
    for o in out.iter().skip(NUM_BUCKETS).take(16) {
        assert_eq!(*o, 16);
    }
}
