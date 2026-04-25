//! Tier 2 — Timestamp query operations.
//!
//! Verifies GPU timestamp recording and read-back.
//! Requires a GPU; skips gracefully if none available.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// --- Kernel for generating some GPU work ---

#[quanta::kernel]
fn busy_work(data: &mut [f32]) {
    let i = quark_id();
    let mut v = (i + 1) as f32;
    let mut j = 0u32;
    while j < 100 {
        v = v * 1.001;
        j = j + 1;
    }
    data[i] = v;
}

// --- Tests ---

#[test]
fn timestamp_basic_ordering() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let query = gpu.timestamp_query(2).unwrap();

    // Write first timestamp
    gpu.write_timestamp(&query, 0).unwrap();

    // Do some GPU work in between
    let count = 10_000;
    let field = gpu.field::<f32>(count).unwrap();
    let mut wave = busy_work(&gpu).unwrap();
    wave.bind(0, &field);
    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    pulse.wait().unwrap();

    // Write second timestamp
    gpu.write_timestamp(&query, 1).unwrap();

    let stamps = gpu.read_timestamps(&query).unwrap();
    assert_eq!(
        stamps.len(),
        2,
        "expected 2 timestamps, got {}",
        stamps.len()
    );
    assert!(
        stamps[1] >= stamps[0],
        "timestamps not monotonic: t0={}, t1={}",
        stamps[0],
        stamps[1]
    );
}

#[test]
fn timestamp_multiple_slots() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let slot_count = 4;
    let query = gpu.timestamp_query(slot_count).unwrap();

    // Write timestamps with work between each
    for slot in 0..slot_count {
        gpu.write_timestamp(&query, slot).unwrap();

        // Small dispatch to advance GPU time
        let field = gpu.field::<f32>(256).unwrap();
        let mut wave = busy_work(&gpu).unwrap();
        wave.bind(0, &field);
        let mut pulse = gpu.dispatch(&wave, 256).unwrap();
        pulse.wait().unwrap();
    }

    let stamps = gpu.read_timestamps(&query).unwrap();
    assert_eq!(stamps.len(), slot_count as usize);

    // Verify monotonic ordering
    for i in 1..stamps.len() {
        assert!(
            stamps[i] >= stamps[i - 1],
            "timestamps not monotonic at slot {}: t[{}]={}, t[{}]={}",
            i,
            i - 1,
            stamps[i - 1],
            i,
            stamps[i]
        );
    }
}

#[test]
fn timestamp_to_nanoseconds() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let query = gpu.timestamp_query(2).unwrap();

    gpu.write_timestamp(&query, 0).unwrap();

    // Generate measurable work
    let count = 100_000;
    let field = gpu.field::<f32>(count).unwrap();
    let mut wave = busy_work(&gpu).unwrap();
    wave.bind(0, &field);
    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    pulse.wait().unwrap();

    gpu.write_timestamp(&query, 1).unwrap();

    let stamps = gpu.read_timestamps(&query).unwrap();
    let ns_start = gpu.timestamp_to_ns(stamps[0]);
    let ns_end = gpu.timestamp_to_ns(stamps[1]);

    // End should be at or after start in nanoseconds
    assert!(
        ns_end >= ns_start,
        "ns timestamps not monotonic: start={}, end={}",
        ns_start,
        ns_end
    );
}

#[test]
fn timestamp_query_count() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let query = gpu.timestamp_query(8).unwrap();
    assert_eq!(query.count(), 8);
}
