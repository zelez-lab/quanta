//! Fill a 1024-element u32 buffer with deterministic pseudo-random
//! values on the GPU. Demonstrates the v0 API surface end-to-end.
//!
//! Run: `cargo run -p quanta-rand --example fill_buffer --features gpu`

use quanta_rand::fill_buffer_gpu;

fn main() -> Result<(), quanta::QuantaError> {
    let gpu = quanta::init()?;
    println!("GPU: {}", gpu.name());

    let len = 1024;
    let seed = 0x1234_5678_9ABC_DEF0u64;
    let out = fill_buffer_gpu(&gpu, len, seed)?;

    println!("Generated {len} u32 values from seed 0x{seed:016X}.");
    println!("First 8: {:?}", &out[..8]);

    // Quick sanity: count the lower-bit distribution. For a good RNG,
    // ~half the samples should have bit 0 set.
    let ones = out.iter().filter(|x| **x & 1 == 1).count();
    println!(
        "Samples with low bit set: {ones} / {len} ({:.1}%)",
        100.0 * ones as f64 / len as f64
    );

    Ok(())
}
