//! Ray tracing cookbook example.
//!
//! Builds a BLAS over four triangles. Hardware-gated: prints a notice and
//! exits cleanly when ray tracing isn't supported (Vulkan with
//! VK_KHR_ray_tracing_pipeline required).
//!
//! Run: cargo run --example cookbook_ray_tracing

use quanta::*;

fn main() -> Result<(), QuantaError> {
    let gpu = quanta::init()?;
    println!("GPU: {}", gpu.name());

    if !gpu.supports_ray_tracing() {
        println!("ray tracing not supported on this device — skipping");
        return Ok(());
    }

    // 12 vertices = 4 triangles, 3 floats per vertex.
    let vertices = gpu.field::<f32>(36)?;
    vertices.write(&[
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.5, 1.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.5, 1.0, 1.0,
    ])?;

    let _blas = gpu.acceleration_structure_blas(&[GeometryDesc {
        vertices: vertices.handle(),
        indices: None,
        vertex_count: 12,
        index_count: 0,
        vertex_stride: 12,
    }])?;

    println!("BLAS built: 4 triangles, 12 vertices");
    Ok(())
}
