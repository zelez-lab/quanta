//! Tier 2 — Resource transition barriers.
//!
//! Verifies that barriers correctly synchronize GPU resource accesses.
//! Requires a GPU; skips gracefully if none available.

use quanta::ResourceState;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// --- Kernel definitions ---

#[quanta::kernel]
fn write_pattern(output: &mut [f32]) {
    let i = quark_id();
    output[i] = (i + 1) as f32;
}

#[quanta::kernel]
fn double_values(data: &[f32], result: &mut [f32]) {
    let i = quark_id();
    result[i] = data[i] * 2.0;
}

// --- Tests ---

#[test]
fn barrier_full_pipeline() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 256;
    let field = gpu.compute_field::<f32>(count).unwrap();

    // Write via compute kernel
    let mut wave = write_pattern(&gpu).unwrap();
    wave.bind(0, &field);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    // Full barrier
    gpu.barrier().unwrap();

    // Read back should see updated data
    let result = gpu.read_field::<f32>(&field).unwrap();
    for i in 0..count {
        let expected = (i + 1) as f32;
        assert!(
            (result[i] - expected).abs() < 0.001,
            "barrier_full at {}: expected {}, got {}",
            i,
            expected,
            result[i]
        );
    }
}

#[test]
fn barrier_compute_write_then_read() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 256;
    let field_a = gpu.compute_field::<f32>(count).unwrap();
    let field_b = gpu.compute_field::<f32>(count).unwrap();

    // First kernel writes to field_a
    let mut wave1 = write_pattern(&gpu).unwrap();
    wave1.bind(0, &field_a);

    let mut pulse1 = gpu.dispatch(&wave1, count as u32).unwrap();
    gpu.wait(&mut pulse1).unwrap();

    // Transition field_a from compute-write to compute-read
    gpu.barrier_buffer(
        &field_a,
        ResourceState::ComputeWrite,
        ResourceState::ComputeRead,
    )
    .unwrap();

    // Second kernel reads field_a and writes to field_b
    let mut wave2 = double_values(&gpu).unwrap();
    wave2.bind(0, &field_a);
    wave2.bind(1, &field_b);

    let mut pulse2 = gpu.dispatch(&wave2, count as u32).unwrap();
    gpu.wait(&mut pulse2).unwrap();

    let result = gpu.read_field::<f32>(&field_b).unwrap();
    for i in 0..count {
        let expected = (i + 1) as f32 * 2.0;
        assert!(
            (result[i] - expected).abs() < 0.01,
            "barrier_compute at {}: expected {}, got {}",
            i,
            expected,
            result[i]
        );
    }
}

#[test]
fn barrier_buffer_multiple_transitions() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 128;
    let data = vec![5.0f32; count];
    let field = gpu.compute_field::<f32>(count).unwrap();
    gpu.write_field(&field, &data).unwrap();

    // Transition: General -> ComputeRead
    gpu.barrier_buffer(&field, ResourceState::General, ResourceState::ComputeRead)
        .unwrap();

    // Transition: ComputeRead -> ComputeWrite
    gpu.barrier_buffer(
        &field,
        ResourceState::ComputeRead,
        ResourceState::ComputeWrite,
    )
    .unwrap();

    // Transition: ComputeWrite -> General
    gpu.barrier_buffer(&field, ResourceState::ComputeWrite, ResourceState::General)
        .unwrap();

    // Verify data is still intact
    let result = gpu.read_field::<f32>(&field).unwrap();
    for (i, v) in result.iter().enumerate() {
        assert!(
            (*v - 5.0).abs() < 0.001,
            "barrier_multiple at {}: expected 5.0, got {}",
            i,
            v
        );
    }
}

#[test]
fn barrier_texture_transition() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let tex = gpu
        .create_texture(&quanta::TextureDesc {
            width: 16,
            height: 16,
            format: quanta::Format::RGBA8,
            usage: quanta::TextureUsage::SHADER_READ
                .union(quanta::TextureUsage::SHADER_WRITE)
                .union(quanta::TextureUsage::RENDER_TARGET),
            ..quanta::TextureDesc::default()
        })
        .unwrap();

    // Write pixel data
    let pixels = vec![128u8; 16 * 16 * 4];
    gpu.texture_write(&tex, &pixels).unwrap();

    // Transition texture state
    gpu.barrier_texture(&tex, ResourceState::TransferDst, ResourceState::ShaderRead)
        .unwrap();

    gpu.barrier_texture(&tex, ResourceState::ShaderRead, ResourceState::General)
        .unwrap();

    // Verify data persists through transitions
    let result = gpu.texture_read(&tex).unwrap();
    assert_eq!(
        pixels, result,
        "texture data lost after barrier transitions"
    );
}
