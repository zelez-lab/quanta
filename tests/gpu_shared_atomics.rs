//! Shared-memory atomic operations.
//!
//! Exercises `atomic_add_shared_u32(slot, idx, val, order)` — a
//! per-bucket histogram increment that would otherwise require a
//! buffer-backed counter + a global-memory round-trip.
//!
//! The CPU executor path always runs; the GPU path runs when a real
//! backend is available. The shared-atomic family currently emits on
//! Metal only (WGSL needs `atomic<T>` decoration on the workgroup
//! declaration; SPIR-V needs the Workgroup-storage-class atomic ops
//! wired in — both are tracked separately).

#![cfg(feature = "software")]

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// Histogram into 4 buckets via shared-mem atomic increment.
// Each quark picks a bucket from its input value % 4 and increments
// the matching slot. After a barrier, every lane reads its
// corresponding bucket count (only lanes 0..3 produce useful output,
// but all 64 do the read so we avoid an `if lid < 4` block — the
// wasm-route lowerer currently mis-structures `if`-blocks that
// contain the whole-kernel epilogue, see gpu_shared_atomics
// commit notes).
#[quanta::kernel(workgroup = [64])]
fn shared_atomic_histogram(input: &[u32], counts: &mut [u32]) {
    #[quanta::shared]
    let buckets: [u32; 64];

    let lid = proton_id();
    let gid = quark_id();

    // Every lane zero-inits its own slot. Indices 0..3 are the
    // real bucket counters; 4..63 are unused but writing them is
    // cheap and keeps the IR shape uniform.
    buckets[lid] = 0u32;
    barrier();

    let v = input[gid];
    let bucket = v % 4;
    // 4th arg = MemoryOrder::Relaxed (== 0). Spelled out as a
    // literal to avoid a host-side `unused_const` warning, since
    // `quanta::intrinsics::ORDER_RELAXED` is wasm32-gated.
    unsafe {
        atomic_add_shared_u32(0u32, bucket, 1u32, 0u32);
    }
    barrier();

    // Every lane writes; lanes 0..3 deliver the four bucket
    // counts. Lanes 4..63 write `buckets[lid]` (which is 0).
    counts[lid as usize] = buckets[lid];
}

#[test]
fn shared_atomic_histogram_cpu() {
    let gpu = quanta::init_cpu();

    let count = 64usize;
    // Inputs: 0..64 → buckets [16, 16, 16, 16].
    let input: Vec<u32> = (0..count as u32).collect();

    let input_field = gpu.field::<u32>(count).unwrap();
    let counts_field = gpu.field::<u32>(count).unwrap();

    input_field.write(&input).unwrap();
    counts_field.write(&vec![0u32; count]).unwrap();

    let mut wave = shared_atomic_histogram(&gpu).unwrap();
    wave.bind(0, &input_field);
    wave.bind(1, &counts_field);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    pulse.wait().unwrap();

    let counts = counts_field.read().unwrap();
    // Lanes 0..3 hold the bucket counts (16 each); lanes 4..63
    // hold the zero-init value.
    assert_eq!(&counts[0..4], &[16, 16, 16, 16]);
    assert!(
        counts[4..].iter().all(|&c| c == 0),
        "unused lanes must be zero"
    );
}

#[test]
fn shared_atomic_histogram_gpu() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 64usize;
    let input: Vec<u32> = (0..count as u32).collect();

    let input_field = gpu.field::<u32>(count).unwrap();
    let counts_field = gpu.field::<u32>(count).unwrap();

    input_field.write(&input).unwrap();
    counts_field.write(&vec![0u32; count]).unwrap();

    let mut wave = shared_atomic_histogram(&gpu).unwrap();
    wave.bind(0, &input_field);
    wave.bind(1, &counts_field);

    let dispatch = gpu.dispatch(&wave, count as u32);
    match dispatch {
        Ok(mut pulse) => {
            pulse.wait().unwrap();
            let counts = counts_field.read().unwrap();
            assert_eq!(
                &counts[0..4],
                &[16, 16, 16, 16],
                "GPU histogram bucket counts must match the CPU oracle"
            );
        }
        Err(e) => {
            // On WGSL / SPIR-V the shared-atomic emit isn't wired
            // yet; accept a NotSupported error there. Metal must
            // succeed.
            let msg = format!("{e:?}");
            if msg.contains("NotSupported") || msg.contains("not yet supported") {
                eprintln!("backend doesn't support shared atomics yet: {msg}");
            } else {
                panic!("unexpected dispatch error: {msg}");
            }
        }
    }
}
