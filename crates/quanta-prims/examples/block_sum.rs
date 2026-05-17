//! Demo: cooperative block-wide sum reduction.
//!
//! Loads 256 u32 values into the GPU, sums them with
//! `block_reduce_add_u32_buffer`, prints the result alongside the
//! CPU reference. Demonstrates the most basic shape of every
//! reduce primitive in this crate.
//!
//! Backend: real GPU (Metal / Vulkan / WebGPU). Cooperative
//! primitives need multiple threads per workgroup; the
//! single-thread software JIT backend (`quanta::init_cpu()`)
//! returns degenerate results because subgroup intrinsics
//! degrade to identity functions when there's only one lane.
//! If `init()` fails this example prints a "no GPU" message
//! and exits 0.
//!
//! Run: cargo run -p quanta-prims --example block_sum --features gpu

use quanta_prims::{block_reduce_add_u32_buffer, reference};

fn main() -> Result<(), quanta::QuantaError> {
    let gpu = match quanta::init() {
        Ok(g) => g,
        Err(_) => {
            println!("no GPU backend available — skipping demo");
            return Ok(());
        }
    };
    println!("== quanta-prims block_sum demo ==");
    println!("backend: {}", gpu.name());
    println!();

    // Input: ramp 1..=256. Block size = 256, so one block.
    let n = 256;
    let data: Vec<u32> = (1..=n as u32).collect();

    let input = gpu.field::<u32>(n)?;
    let output = gpu.field::<u32>(1)?;
    input.write(&data)?;
    output.write(&[0u32])?;

    let mut wave = block_reduce_add_u32_buffer(&gpu)?;
    wave.bind(0, &input);
    wave.bind(1, &output);
    let mut pulse = gpu.dispatch(&wave, n as u32)?;
    pulse.wait()?;

    let gpu_sum = output.read()?[0];
    let cpu_sum = reference::reduce_add_u32(&data);

    println!("input:    1, 2, 3, ..., {}", n);
    println!("GPU sum:  {gpu_sum}");
    println!("CPU sum:  {cpu_sum}");
    println!(
        "match:    {}",
        if gpu_sum == cpu_sum { "OK" } else { "MISMATCH" }
    );

    Ok(())
}
