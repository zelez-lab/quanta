//! Tier 2 -- Field allocation, copy, and lifecycle.
//!
//! Verifies field_alloc, field_free, field_copy, resize, and stress allocation.
//! Requires a GPU; skips gracefully if none available.

use quanta::FieldUsage;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[test]
fn field_alloc_write_read_cycle() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 128;
    let data: Vec<f32> = (0..count).map(|i| i as f32 * 0.5).collect();

    let field = gpu.field::<f32>(count).unwrap();
    field.write(&data).unwrap();

    let result = field.read().unwrap();
    assert_eq!(result.len(), count);
    for i in 0..count {
        assert!(
            (result[i] - data[i]).abs() < 0.001,
            "mismatch at {}: expected {}, got {}",
            i,
            data[i],
            result[i]
        );
    }
}

#[test]
fn field_copy_between_fields() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 256;
    let data: Vec<u32> = (0..count as u32).collect();

    let src = gpu.field::<u32>(count).unwrap();
    let dst = gpu.field::<u32>(count).unwrap();

    src.write(&data).unwrap();
    dst.copy_from(&src).unwrap();

    let result = dst.read().unwrap();
    assert_eq!(result, data, "copy_field should produce identical data");
}

#[test]
fn field_free_on_drop() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Allocate and immediately drop -- should not leak or crash.
    for _ in 0..100 {
        let field = gpu.field::<f32>(64).unwrap();
        drop(field);
    }
}

#[test]
fn field_resize_preserves_data() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 64;
    let data: Vec<f32> = (0..count).map(|i| (i + 1) as f32).collect();

    let old = gpu.field::<f32>(count).unwrap();
    old.write(&data).unwrap();

    // Resize to double the capacity: allocate new field and copy.
    let new_count = count * 2;
    let resized = gpu.field::<f32>(new_count).unwrap();
    resized.copy_from(&old).unwrap();

    let result = resized.read().unwrap();
    assert_eq!(result.len(), new_count);

    // First `count` elements should be preserved.
    for i in 0..count {
        assert!(
            (result[i] - data[i]).abs() < 0.001,
            "resize lost data at {}: expected {}, got {}",
            i,
            data[i],
            result[i]
        );
    }
}

#[test]
fn field_stress_alloc() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Allocate many small fields, write and read them all.
    let n = 50;
    let size = 32;
    let mut fields = Vec::new();

    for i in 0..n {
        let f = gpu.field::<u32>(size).unwrap();
        let data: Vec<u32> = (0..size as u32).map(|j| i as u32 * 1000 + j).collect();
        f.write(&data).unwrap();
        fields.push((f, data));
    }

    // Verify all fields still contain correct data.
    for (field, expected) in &fields {
        let result = field.read().unwrap();
        assert_eq!(&result, expected, "stress alloc: field data mismatch");
    }
}

#[test]
fn field_render_usage() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Verify render field allocation works.
    let count = 64;
    let field = gpu
        .field_with_usage::<f32>(count, FieldUsage::default_render())
        .unwrap();
    assert_eq!(field.len(), count);
    assert_eq!(field.byte_size(), count * 4);
    assert!(!field.is_empty());
}

#[test]
fn field_uniform_usage() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Verify uniform field allocation works.
    let count = 16;
    let field = gpu
        .field_with_usage::<[f32; 4]>(count, FieldUsage::default_uniform())
        .unwrap();
    assert_eq!(field.len(), count);
    assert_eq!(field.byte_size(), count * 16);
}

#[test]
fn field_copy_different_sizes() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Copy from a smaller field to a larger one -- only min(src, dst) bytes copied.
    let small_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let small = gpu.field::<f32>(4).unwrap();
    let large = gpu.field::<f32>(8).unwrap();

    small.write(&small_data).unwrap();
    large.write(&[0.0f32; 8]).unwrap();

    large.copy_from(&small).unwrap();

    let result = large.read().unwrap();
    // First 4 elements should be from small.
    for i in 0..4 {
        assert!(
            (result[i] - small_data[i]).abs() < 0.001,
            "copy small->large at {}: expected {}, got {}",
            i,
            small_data[i],
            result[i]
        );
    }
}
