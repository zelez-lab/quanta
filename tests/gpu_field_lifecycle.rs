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

    let field = gpu.compute_field::<f32>(count).unwrap();
    gpu.write_field(&field, &data).unwrap();

    let result = gpu.read_field::<f32>(&field).unwrap();
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

    let src = gpu.compute_field::<u32>(count).unwrap();
    let dst = gpu.compute_field::<u32>(count).unwrap();

    gpu.write_field(&src, &data).unwrap();
    gpu.copy_field(&dst, &src).unwrap();

    let result = gpu.read_field::<u32>(&dst).unwrap();
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
        let field = gpu.compute_field::<f32>(64).unwrap();
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

    let old = gpu.compute_field::<f32>(count).unwrap();
    gpu.write_field(&old, &data).unwrap();

    // Resize to double the capacity.
    let new_count = count * 2;
    let resized = gpu
        .resize_field(&old, new_count, FieldUsage::default_compute())
        .unwrap();

    let result = gpu.read_field::<f32>(&resized).unwrap();
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
        let f = gpu.compute_field::<u32>(size).unwrap();
        let data: Vec<u32> = (0..size as u32).map(|j| i as u32 * 1000 + j).collect();
        gpu.write_field(&f, &data).unwrap();
        fields.push((f, data));
    }

    // Verify all fields still contain correct data.
    for (field, expected) in &fields {
        let result = gpu.read_field::<u32>(field).unwrap();
        assert_eq!(&result, expected, "stress alloc: field data mismatch");
    }
}

#[test]
fn field_render_usage() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Verify render_field allocation works.
    let count = 64;
    let field = gpu.render_field::<f32>(count).unwrap();
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

    // Verify uniform_field allocation works.
    let count = 16;
    let field = gpu.uniform_field::<[f32; 4]>(count).unwrap();
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
    let small = gpu.compute_field::<f32>(4).unwrap();
    let large = gpu.compute_field::<f32>(8).unwrap();

    gpu.write_field(&small, &small_data).unwrap();
    gpu.write_field(&large, &[0.0f32; 8]).unwrap();

    gpu.copy_field(&large, &small).unwrap();

    let result = gpu.read_field::<f32>(&large).unwrap();
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
