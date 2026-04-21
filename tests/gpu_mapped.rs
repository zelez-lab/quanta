//! Tier 2 — Mapped buffer operations.
//!
//! Verifies CPU-mapped GPU memory for zero-copy access.
//! Requires a GPU; skips gracefully if none available.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// --- Tests ---

#[test]
fn mapped_write_read_single() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let mut mapped = gpu.field_mapped::<f32>(256).unwrap();

    // Write individual elements
    mapped.write(0, 1.0f32);
    mapped.write(1, 2.0f32);
    mapped.write(2, 3.0f32);
    mapped.write(3, 4.0f32);

    // Read back without GPU roundtrip
    assert_eq!(mapped.read(0), 1.0f32);
    assert_eq!(mapped.read(1), 2.0f32);
    assert_eq!(mapped.read(2), 3.0f32);
    assert_eq!(mapped.read(3), 4.0f32);
}

#[test]
fn mapped_as_slice() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let mut mapped = gpu.field_mapped::<f32>(64).unwrap();

    // Write via mutable slice
    let slice = mapped.as_mut_slice();
    for i in 0..64 {
        slice[i] = i as f32;
    }

    // Read via immutable slice
    let view = mapped.as_slice();
    for i in 0..64 {
        assert!(
            (view[i] - i as f32).abs() < 0.001,
            "mapped slice at {}: expected {}, got {}",
            i,
            i,
            view[i]
        );
    }
}

#[test]
fn mapped_overwrite() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let mut mapped = gpu.field_mapped::<u32>(128).unwrap();

    // Write pattern
    for i in 0..128 {
        mapped.write(i, i as u32);
    }

    // Overwrite
    for i in 0..128 {
        mapped.write(i, (i * 10) as u32);
    }

    // Verify overwritten values
    for i in 0..128 {
        assert_eq!(
            mapped.read(i),
            (i * 10) as u32,
            "overwrite at {}: expected {}, got {}",
            i,
            i * 10,
            mapped.read(i)
        );
    }
}

#[test]
fn mapped_len_and_byte_size() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let mapped = gpu.field_mapped::<f32>(512).unwrap();
    assert_eq!(mapped.len(), 512);
    assert_eq!(mapped.byte_size(), 512 * 4);
    assert!(!mapped.is_empty());
}

#[test]
fn mapped_u32_round_trip() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let mut mapped = gpu.field_mapped::<u32>(32).unwrap();

    let values: Vec<u32> = (0..32).map(|i| i * 7 + 3).collect();
    for (i, v) in values.iter().enumerate() {
        mapped.write(i, *v);
    }

    for (i, expected) in values.iter().enumerate() {
        assert_eq!(
            mapped.read(i),
            *expected,
            "u32 mapped at {}: expected {}, got {}",
            i,
            expected,
            mapped.read(i)
        );
    }
}
