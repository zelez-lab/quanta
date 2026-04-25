//! Tier 2 — Atomic operations on GPU.
//!
//! Verifies that atomic_add, atomic_max, and atomic_min produce correct results
//! when many quarks race on the same memory location.
//! Requires a GPU; skips gracefully if none available.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// --- Kernel definitions ---

#[quanta::kernel]
fn atomic_count(flags: &[u32], counter: &mut [u32]) {
    let i = quark_id();
    if flags[i] != 0 {
        atomic_add(&mut counter[0], 1u32);
    }
}

#[quanta::kernel]
fn atomic_find_max(values: &[u32], result: &mut [u32]) {
    let i = quark_id();
    atomic_max(&mut result[0], values[i]);
}

#[quanta::kernel]
fn atomic_find_min(values: &[u32], result: &mut [u32]) {
    let i = quark_id();
    atomic_min(&mut result[0], values[i]);
}

// --- Tests ---

#[test]
fn atomic_add_count_nonzero() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 1024;
    // Set every other element to 1 (512 nonzero)
    let flags: Vec<u32> = (0..count).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
    let expected_count = flags.iter().filter(|&&f| f != 0).count() as u32;

    let flags_field = gpu.field::<u32>(count).unwrap();
    let counter_field = gpu.field::<u32>(1).unwrap();

    flags_field.write(&flags).unwrap();
    counter_field.write(&[0u32]).unwrap();

    let mut wave = atomic_count(&gpu).unwrap();
    wave.bind(0, &flags_field);
    wave.bind(1, &counter_field);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    pulse.wait().unwrap();

    let result = counter_field.read().unwrap();
    assert_eq!(
        result[0], expected_count,
        "atomic_add count: expected {}, got {}",
        expected_count, result[0]
    );
}

#[test]
fn atomic_add_all_ones() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // All flags set: count should equal total elements
    let count = 512;
    let flags = vec![1u32; count];

    let flags_field = gpu.field::<u32>(count).unwrap();
    let counter_field = gpu.field::<u32>(1).unwrap();

    flags_field.write(&flags).unwrap();
    counter_field.write(&[0u32]).unwrap();

    let mut wave = atomic_count(&gpu).unwrap();
    wave.bind(0, &flags_field);
    wave.bind(1, &counter_field);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    pulse.wait().unwrap();

    let result = counter_field.read().unwrap();
    assert_eq!(
        result[0], count as u32,
        "atomic_add all-ones: expected {}, got {}",
        count, result[0]
    );
}

#[test]
fn atomic_max_finds_maximum() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 256;
    let values: Vec<u32> = (0..count as u32).collect();
    let expected_max = (count - 1) as u32;

    let values_field = gpu.field::<u32>(count).unwrap();
    let result_field = gpu.field::<u32>(1).unwrap();

    values_field.write(&values).unwrap();
    result_field.write(&[0u32]).unwrap();

    let mut wave = atomic_find_max(&gpu).unwrap();
    wave.bind(0, &values_field);
    wave.bind(1, &result_field);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    pulse.wait().unwrap();

    let result = result_field.read().unwrap();
    assert_eq!(
        result[0], expected_max,
        "atomic_max: expected {}, got {}",
        expected_max, result[0]
    );
}

#[test]
fn atomic_min_finds_minimum() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 256;
    // Values from 10 to 265 (min = 10)
    let values: Vec<u32> = (10..10 + count as u32).collect();
    let expected_min = 10u32;

    let values_field = gpu.field::<u32>(count).unwrap();
    let result_field = gpu.field::<u32>(1).unwrap();

    values_field.write(&values).unwrap();
    // Initialize to max u32 so any value is smaller
    result_field.write(&[u32::MAX]).unwrap();

    let mut wave = atomic_find_min(&gpu).unwrap();
    wave.bind(0, &values_field);
    wave.bind(1, &result_field);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    pulse.wait().unwrap();

    let result = result_field.read().unwrap();
    assert_eq!(
        result[0], expected_min,
        "atomic_min: expected {}, got {}",
        expected_min, result[0]
    );
}
