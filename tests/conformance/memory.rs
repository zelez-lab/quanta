//! Memory conformance tests — field alloc, write, read, copy, resize.

use quanta::Gpu;

/// Write data to a field, read it back, verify exact match.
pub fn field_write_read(gpu: &Gpu) {
    let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
    let field = gpu.compute_field::<f32>(1024).unwrap();
    gpu.write_field(&field, &data).unwrap();
    let result = gpu.read_field(&field).unwrap();
    assert_eq!(data.len(), result.len(), "length mismatch");
    for i in 0..data.len() {
        assert!(
            (data[i] - result[i]).abs() < 0.001,
            "mismatch at {}: expected {}, got {}",
            i,
            data[i],
            result[i]
        );
    }
}

/// Write integers, read back, verify exact match.
pub fn field_write_read_u32(gpu: &Gpu) {
    let data: Vec<u32> = (0..512).map(|i| i * 7 + 3).collect();
    let field = gpu.compute_field::<u32>(512).unwrap();
    gpu.write_field(&field, &data).unwrap();
    let result = gpu.read_field(&field).unwrap();
    assert_eq!(data, result);
}

/// Copy data between two fields, verify destination matches source.
pub fn field_copy(gpu: &Gpu) {
    let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let src = gpu.compute_field::<f32>(256).unwrap();
    let dst = gpu.compute_field::<f32>(256).unwrap();
    gpu.write_field(&src, &data).unwrap();
    gpu.copy_field(&dst, &src).unwrap();
    let result = gpu.read_field(&dst).unwrap();
    assert_eq!(data.len(), result.len());
    for i in 0..data.len() {
        assert!(
            (data[i] - result[i]).abs() < 0.001,
            "copy mismatch at {}: expected {}, got {}",
            i,
            data[i],
            result[i]
        );
    }
}

/// Resize a field, verify old data is preserved.
pub fn field_resize(gpu: &Gpu) {
    let data: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let field = gpu.compute_field::<f32>(128).unwrap();
    gpu.write_field(&field, &data).unwrap();

    let bigger = gpu
        .resize_field(&field, 256, quanta::FieldUsage::default_compute())
        .unwrap();
    let result = gpu.read_field(&bigger).unwrap();
    assert_eq!(result.len(), 256);
    // First 128 elements should match original data
    for i in 0..128 {
        assert!(
            (data[i] - result[i]).abs() < 0.001,
            "resize data lost at {}: expected {}, got {}",
            i,
            data[i],
            result[i]
        );
    }
}

/// Allocate a large field, verify no error.
pub fn field_large_alloc(gpu: &Gpu) {
    let count = 1_000_000;
    let field = gpu.compute_field::<f32>(count).unwrap();
    let data: Vec<f32> = (0..count).map(|i| i as f32).collect();
    gpu.write_field(&field, &data).unwrap();
    let result = gpu.read_field(&field).unwrap();
    assert_eq!(result.len(), count);
    assert!((result[0] - 0.0).abs() < 0.001);
    assert!((result[count - 1] - (count - 1) as f32).abs() < 0.001);
}

/// Uniform buffer round-trip.
pub fn uniform_field(gpu: &Gpu) {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let field = gpu.uniform_field::<f32>(4).unwrap();
    gpu.write_field(&field, &data).unwrap();
    let result = gpu.read_field(&field).unwrap();
    assert_eq!(&data[..], &result[..]);
}

/// Run all memory tests.
pub fn run_all(gpu: &Gpu) {
    field_write_read(gpu);
    field_write_read_u32(gpu);
    field_copy(gpu);
    field_resize(gpu);
    field_large_alloc(gpu);
    uniform_field(gpu);
}
