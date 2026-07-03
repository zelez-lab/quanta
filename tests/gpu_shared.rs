//! Tier 2 — Shared memory tests.
//!
//! Verifies workgroup-local shared memory and barrier synchronization.
//! Requires a GPU; skips gracefully if none available.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// --- Kernel definitions ---

/// Each workgroup of 64 quarks loads data into shared memory,
/// then the first quark sums all values and writes to output.
#[quanta::kernel]
fn shared_sum(data: &[f32], result: &mut [f32]) {
    #[quanta::shared]
    let local: [f32; 64];

    let lid = proton_id();
    let gid = quark_id();

    local[lid] = data[gid];
    barrier();

    if lid == 0 {
        let mut sum = 0.0f32;
        let mut j = 0u32;
        while j < 64 {
            sum = sum + local[j];
            j = j + 1;
        }
        result[nucleus_id()] = sum;
    }
}

/// Each workgroup reverses its 64 elements using shared memory.
#[quanta::kernel]
fn shared_reverse(data: &[f32], result: &mut [f32]) {
    #[quanta::shared]
    let local: [f32; 64];

    let lid = proton_id();
    let gid = quark_id();

    local[lid] = data[gid];
    barrier();

    result[gid] = local[63 - lid];
}

// --- Tests ---

#[test]
fn shared_memory_sum() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // 4 groups of 64 quarks = 256 elements
    let group_size = 64;
    let num_groups = 4;
    let count = group_size * num_groups;

    // Each element is 1.0 => each group sum = 64.0
    let data = vec![1.0f32; count];

    let input = gpu.field::<f32>(count).unwrap();
    let output = gpu.field::<f32>(num_groups).unwrap();
    input.write(&data).unwrap();

    let mut wave = shared_sum(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &output);

    let mut pulse = gpu.wave_dispatch(&wave, [num_groups as u32, 1, 1]).unwrap();
    pulse.wait().unwrap();

    let result = output.read().unwrap();
    for (g, v) in result.iter().enumerate() {
        assert!(
            (*v - 64.0).abs() < 0.01,
            "shared_sum group {}: expected 64.0, got {}",
            g,
            v
        );
    }
}

#[test]
fn shared_memory_sum_varying_data() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let group_size = 64;
    let num_groups = 2;
    let count = group_size * num_groups;

    // Group 0: values 0..64, sum = 63*64/2 = 2016
    // Group 1: values 64..128, sum = (64+127)*64/2 = 6112
    let data: Vec<f32> = (0..count).map(|i| i as f32).collect();

    let input = gpu.field::<f32>(count).unwrap();
    let output = gpu.field::<f32>(num_groups).unwrap();
    input.write(&data).unwrap();

    let mut wave = shared_sum(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &output);

    let mut pulse = gpu.wave_dispatch(&wave, [num_groups as u32, 1, 1]).unwrap();
    pulse.wait().unwrap();

    let result = output.read().unwrap();
    let expected_0: f32 = (0..64).map(|i| i as f32).sum();
    let expected_1: f32 = (64..128).map(|i| i as f32).sum();

    assert!(
        (result[0] - expected_0).abs() < 1.0,
        "group 0: expected {}, got {}",
        expected_0,
        result[0]
    );
    assert!(
        (result[1] - expected_1).abs() < 1.0,
        "group 1: expected {}, got {}",
        expected_1,
        result[1]
    );
}

#[test]
fn shared_memory_reverse() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let group_size = 64;
    let num_groups = 2;
    let count = group_size * num_groups;

    let data: Vec<f32> = (0..count).map(|i| i as f32).collect();

    let input = gpu.field::<f32>(count).unwrap();
    let output = gpu.field::<f32>(count).unwrap();
    input.write(&data).unwrap();

    let mut wave = shared_reverse(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &output);

    let mut pulse = gpu.wave_dispatch(&wave, [num_groups as u32, 1, 1]).unwrap();
    pulse.wait().unwrap();

    let result = output.read().unwrap();

    // Group 0: indices 0..64 reversed => [63, 62, ..., 0]
    #[allow(clippy::needless_range_loop)] // i is the reversed-index arithmetic, not just a cursor
    for i in 0..group_size {
        let expected = (group_size - 1 - i) as f32;
        assert!(
            (result[i] - expected).abs() < 0.001,
            "reverse group 0 at {}: expected {}, got {}",
            i,
            expected,
            result[i]
        );
    }

    // Group 1: indices 64..128 reversed => [127, 126, ..., 64]
    for i in 0..group_size {
        let expected = (group_size + group_size - 1 - i) as f32;
        assert!(
            (result[group_size + i] - expected).abs() < 0.001,
            "reverse group 1 at {}: expected {}, got {}",
            i,
            expected,
            result[group_size + i]
        );
    }
}
