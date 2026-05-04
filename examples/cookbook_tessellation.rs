//! Tessellation cookbook example.
//!
//! Builds a tessellation pipeline, sets outer + inner factors, and prints
//! the configured topology. Hardware-gated: prints a notice and exits
//! cleanly if the device doesn't support tessellation.
//!
//! Run: cargo run --example cookbook_tessellation

use quanta::*;

fn main() -> Result<(), QuantaError> {
    let gpu = quanta::init()?;
    println!("GPU: {}", gpu.name());

    if !gpu.supports_tessellation() {
        println!("tessellation not supported on this device — skipping");
        return Ok(());
    }

    // Triangle patches with 3 control points each.
    let pipe = gpu.tessellation_pipeline(TessTopology::Triangle, 3)?;

    // Subdivide every edge into 8 segments and the interior into 8 layers.
    for edge in 0..3 {
        pipe.set_outer(edge, 8)?;
    }
    pipe.set_inner(0, 8)?;

    println!("tessellation pipeline configured: triangle, 3 control points");
    println!("  outer factors: 8, 8, 8");
    println!("  inner factor:  8");
    Ok(())
}
