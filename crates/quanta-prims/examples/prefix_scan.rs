//! Demo: cooperative inclusive prefix-sum scan.
//!
//! Runs `block_scan_add_u32_buffer` on a small input and prints
//! every prefix value next to the CPU reference. Useful for
//! understanding what the primitive does: each output position k
//! holds `sum(data[0..=k])`.
//!
//! Backend: real GPU. See `block_sum.rs` for why this example
//! needs a real GPU rather than `init_cpu()`.
//!
//! Run: cargo run -p quanta-prims --example prefix_scan --features gpu

use quanta_prims::{block_scan_add_u32_buffer, reference};

fn main() -> Result<(), quanta::QuantaError> {
    let gpu = match quanta::init() {
        Ok(g) => g,
        Err(_) => {
            println!("no GPU backend available — skipping demo");
            return Ok(());
        }
    };
    println!("== quanta-prims prefix_scan demo ==");
    println!("backend: {}", gpu.name());
    println!();

    // Workgroup size is fixed at 256. The input length must be a
    // multiple of 256, but the interesting output is the first
    // few lanes — every lane's result is "sum of inputs up to
    // (and including) self."
    let n = 256;
    let mut data = vec![0u32; n];
    // First 8 entries: 1, 2, 4, 8, 16, 32, 64, 128. Geometric
    // so prefix sums are easy to recognise: 1, 3, 7, 15, 31,
    // 63, 127, 255.
    for k in 0..8 {
        data[k] = 1u32 << k;
    }

    let input = gpu.field::<u32>(n)?;
    let output = gpu.field::<u32>(n)?;
    input.write(&data)?;
    output.write(&vec![0u32; n])?;

    let mut wave = block_scan_add_u32_buffer(&gpu)?;
    wave.bind(0, &input);
    wave.bind(1, &output);
    let mut pulse = gpu.dispatch(&wave, n as u32)?;
    pulse.wait()?;

    let gpu_scan = output.read()?;
    let cpu_scan = reference::scan_add_u32(&data);

    println!("input  (first 8): {:?}", &data[..8]);
    println!("GPU scan (first 8): {:?}", &gpu_scan[..8]);
    println!("CPU scan (first 8): {:?}", &cpu_scan[..8]);
    println!(
        "match: {}",
        if gpu_scan == cpu_scan {
            "OK"
        } else {
            "MISMATCH"
        }
    );
    println!();
    println!("Inclusive scan identity: out[N-1] == sum(in[0..N])");
    println!("  out[{}] = {}", n - 1, gpu_scan[n - 1]);
    println!("  sum    = {}", reference::reduce_add_u32(&data));

    Ok(())
}
