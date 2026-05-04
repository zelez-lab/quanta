//! Mesh shaders cookbook example.
//!
//! Creates a mesh pipeline with custom meshlet limits and dispatches it.
//! Hardware-gated: prints a notice and exits cleanly if mesh shaders
//! aren't supported (Metal 3+ or VK_EXT_mesh_shader required).
//!
//! Run: cargo run --example cookbook_mesh_shaders

use quanta::*;

fn main() -> Result<(), QuantaError> {
    let gpu = quanta::init()?;
    println!("GPU: {}", gpu.name());

    if !gpu.supports_mesh_shaders() {
        println!("mesh shaders not supported on this device — skipping");
        return Ok(());
    }

    let pipe = gpu.mesh_pipeline(MeshPipelineDesc {
        max_vertices_per_meshlet: 64,
        max_primitives_per_meshlet: 124,
        task_threads_per_group: 32,
    })?;

    pipe.dispatch([1024, 1, 1])?;

    println!("mesh pipeline dispatched: 1024 task workgroups");
    println!("  meshlet cap: 64 verts / 124 prims");
    Ok(())
}
