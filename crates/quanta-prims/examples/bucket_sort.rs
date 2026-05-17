//! Demo: cooperative block-local sort.
//!
//! Generates pseudo-random u32 keys with an LCG, sorts each
//! 256-element block independently with
//! `block_radix_sort_u32_buffer`, and prints a snapshot of one
//! block before/after sorting plus a correctness check against
//! the CPU reference.
//!
//! "Bucket sort" in the name nods to the classic radix flavour;
//! the v0.1 algorithm is bitonic underneath. The API name keeps
//! the radix terminology for forward compatibility.
//!
//! Backend: real GPU. See `block_sum.rs` for why this example
//! needs a real GPU rather than `init_cpu()`.
//!
//! Run: cargo run -p quanta-prims --example bucket_sort --features gpu

use quanta_prims::{block_radix_sort_u32_buffer, reference};

fn lcg_seq(seed: u32, n: usize) -> Vec<u32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        })
        .collect()
}

fn main() -> Result<(), quanta::QuantaError> {
    let gpu = match quanta::init() {
        Ok(g) => g,
        Err(_) => {
            println!("no GPU backend available — skipping demo");
            return Ok(());
        }
    };
    println!("== quanta-prims bucket_sort demo ==");
    println!("backend: {}", gpu.name());
    println!();

    // Two blocks, each filled with its own pseudo-random sequence.
    let block = 256;
    let num_blocks = 2;
    let n = block * num_blocks;
    let mut data = Vec::with_capacity(n);
    data.extend(lcg_seq(0xCAFE_BABEu32, block));
    data.extend(lcg_seq(0xDEAD_BEEFu32, block));

    let input = gpu.field::<u32>(n)?;
    let output = gpu.field::<u32>(n)?;
    input.write(&data)?;
    output.write(&vec![0u32; n])?;

    let mut wave = block_radix_sort_u32_buffer(&gpu)?;
    wave.bind(0, &input);
    wave.bind(1, &output);
    let mut pulse = gpu.dispatch(&wave, n as u32)?;
    pulse.wait()?;

    let gpu_sorted = output.read()?;

    println!("Showing the first 8 keys of block 0:");
    println!("  before: {:?}", &data[..8]);
    println!("  after:  {:?}", &gpu_sorted[..8]);
    println!();

    // Correctness: each block matches the CPU reference sort of
    // its own block.
    let mut all_ok = true;
    for b in 0..num_blocks {
        let start = b * block;
        let end = start + block;
        let expected = reference::radix_sort_u32(&data[start..end]);
        let got = &gpu_sorted[start..end];
        let ok = got == expected.as_slice();
        all_ok &= ok;
        println!("block {b}: {}", if ok { "OK" } else { "MISMATCH" });
    }
    println!();
    println!("overall: {}", if all_ok { "OK" } else { "MISMATCH" });

    Ok(())
}
