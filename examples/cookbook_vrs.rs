//! Variable rate shading cookbook example.
//!
//! Queries supported shading rates and walks through the rate transitions
//! a scene might use. Hardware-gated: prints a notice and exits cleanly
//! when VRS isn't supported.
//!
//! Run: cargo run --example cookbook_vrs

use quanta::*;

fn main() -> Result<(), QuantaError> {
    let gpu = quanta::init()?;
    println!("GPU: {}", gpu.name());

    if !gpu.supports_vrs() {
        println!("VRS not supported on this device — skipping");
        return Ok(());
    }

    let rates: Vec<(u32, u32)> = gpu.supported_shading_rates();
    println!("supported shading rates: {:?}", rates);

    let mut vrs = gpu.vrs_state()?;
    println!("initial rate: {:?}", vrs.current());

    vrs.set_rate(ShadingRate::R2x2)?;
    println!("after set_rate(R2x2): {:?}", vrs.current());

    vrs.set_rate(ShadingRate::R1x1)?;
    println!("after set_rate(R1x1): {:?}", vrs.current());
    Ok(())
}
