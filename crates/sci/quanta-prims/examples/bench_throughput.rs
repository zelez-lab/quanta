//! Throughput sweep for the Tier-1 primitives.
//!
//! For each primitive (block_reduce_add_u32, block_scan_add_u32,
//! block_radix_sort_u32) and a range of block counts, measure
//! the wall-clock time and print M-elements/sec.
//!
//! Includes dispatch overhead — total elapsed time is for one
//! end-to-end kernel launch (write inputs, dispatch, wait,
//! read outputs). For small N the dispatch fixed cost
//! dominates; for large N raw throughput dominates.
//!
//! Run:
//!   cargo run -p quanta-prims --features gpu-metal --release \
//!     --example bench_throughput

use std::time::Instant;

use quanta_prims::{
    block_compact_u32_buffer, block_histogram_u32_buffer, block_radix_sort_u32_buffer,
    block_reduce_add_u32_buffer, block_scan_add_u32_buffer, block_top_k_u32_buffer,
};

const BLOCK: usize = 256;
const WARMUP: usize = 2;
const ITERS: usize = 10;

fn bench<F>(name: &str, n: usize, mut f: F)
where
    F: FnMut() -> Vec<u32>,
{
    // Warmup. Throws away the first few results to amortize
    // shader-cache misses on the first dispatch.
    for _ in 0..WARMUP {
        let _ = f();
    }

    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let _result = f();
        let elapsed = t0.elapsed();
        samples.push(elapsed.as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = samples[ITERS / 2];
    let throughput_msps = (n as f64) / (median_ms * 1e3); // M-elements/sec
    println!(
        "  {name:<14}  n = {n:>8}  median = {median_ms:>7.3} ms   {throughput_msps:>7.1} M-elem/s"
    );
}

fn run_reduce(gpu: &quanta::Gpu, data_field: &quanta::Field<u32>, n: usize) -> Vec<u32> {
    let out_field = gpu.field::<u32>(n / BLOCK).unwrap();
    out_field.write(&vec![0u32; n / BLOCK]).unwrap();
    let mut wave = block_reduce_add_u32_buffer(gpu).unwrap();
    wave.bind(0, data_field);
    wave.bind(1, &out_field);
    let mut pulse = gpu.dispatch(&wave, n as u32).unwrap();
    pulse.wait().unwrap();
    out_field.read().unwrap()
}

fn run_scan(gpu: &quanta::Gpu, data_field: &quanta::Field<u32>, n: usize) -> Vec<u32> {
    let out_field = gpu.field::<u32>(n).unwrap();
    out_field.write(&vec![0u32; n]).unwrap();
    let mut wave = block_scan_add_u32_buffer(gpu).unwrap();
    wave.bind(0, data_field);
    wave.bind(1, &out_field);
    let mut pulse = gpu.dispatch(&wave, n as u32).unwrap();
    pulse.wait().unwrap();
    out_field.read().unwrap()
}

fn run_sort(gpu: &quanta::Gpu, data_field: &quanta::Field<u32>, n: usize) -> Vec<u32> {
    let out_field = gpu.field::<u32>(n).unwrap();
    out_field.write(&vec![0u32; n]).unwrap();
    let mut wave = block_radix_sort_u32_buffer(gpu).unwrap();
    wave.bind(0, data_field);
    wave.bind(1, &out_field);
    let mut pulse = gpu.dispatch(&wave, n as u32).unwrap();
    pulse.wait().unwrap();
    out_field.read().unwrap()
}

fn run_compact(gpu: &quanta::Gpu, data_field: &quanta::Field<u32>, n: usize) -> Vec<u32> {
    // Build a "keep every other element" predicate buffer once
    // per call. The cost of building it is included in the bench
    // wall time — same as a user would observe.
    let preds: Vec<u32> = (0..n as u32).map(|i| (i + 1) % 2).collect();
    let preds_field = gpu.field::<u32>(n).unwrap();
    preds_field.write(&preds).unwrap();

    let out_field = gpu.field::<u32>(n).unwrap();
    out_field.write(&vec![0u32; n]).unwrap();
    let counts_field = gpu.field::<u32>(n / BLOCK).unwrap();
    counts_field.write(&vec![0u32; n / BLOCK]).unwrap();

    let mut wave = block_compact_u32_buffer(gpu).unwrap();
    wave.bind(0, &preds_field);
    wave.bind(1, data_field);
    wave.bind(2, &out_field);
    wave.bind(3, &counts_field);
    let mut pulse = gpu.dispatch(&wave, n as u32).unwrap();
    pulse.wait().unwrap();
    out_field.read().unwrap()
}

fn run_histogram(gpu: &quanta::Gpu, data_field: &quanta::Field<u32>, n: usize) -> Vec<u32> {
    // The histogram kernel expects pre-computed bucket indices in
    // 0..256. We bucket by `value % 256`, which we precompute and
    // upload — same cost the user pays.
    let buckets: Vec<u32> = (0..n as u32).map(|i| i % 256u32).collect();
    let bucket_field = gpu.field::<u32>(n).unwrap();
    bucket_field.write(&buckets).unwrap();
    let _ = data_field; // unused — kept for signature parity with other run_*.

    let out_field = gpu.field::<u32>(n).unwrap();
    out_field.write(&vec![0u32; n]).unwrap();
    let mut wave = block_histogram_u32_buffer(gpu).unwrap();
    wave.bind(0, &bucket_field);
    wave.bind(1, &out_field);
    let mut pulse = gpu.dispatch(&wave, n as u32).unwrap();
    pulse.wait().unwrap();
    out_field.read().unwrap()
}

fn run_top_k(gpu: &quanta::Gpu, data_field: &quanta::Field<u32>, n: usize) -> Vec<u32> {
    let k: u32 = 16;
    let num_blocks = n / BLOCK;
    let out_field = gpu.field::<u32>(num_blocks * k as usize).unwrap();
    out_field
        .write(&vec![0u32; num_blocks * k as usize])
        .unwrap();
    let mut wave = block_top_k_u32_buffer(gpu).unwrap();
    wave.bind(0, data_field);
    wave.bind(1, &out_field);
    wave.set_value(2, k);
    let mut pulse = gpu.dispatch(&wave, n as u32).unwrap();
    pulse.wait().unwrap();
    out_field.read().unwrap()
}

fn main() -> Result<(), quanta::QuantaError> {
    let gpu = match quanta::init() {
        Ok(g) => g,
        Err(_) => {
            println!("no GPU backend available — skipping bench");
            return Ok(());
        }
    };
    println!("== quanta-prims throughput sweep ==");
    println!("backend: {}", gpu.name());
    println!("warmup  : {WARMUP} iter");
    println!("samples : {ITERS} iter (reporting median)");
    println!();

    // Sweep over different block counts.
    let block_counts = [1usize, 8, 64, 256, 1024];

    for &num_blocks in &block_counts {
        let n = num_blocks * BLOCK;
        println!("── num_blocks = {num_blocks}  (n = {n}) ──────────────────");

        let data_field = gpu.field::<u32>(n)?;
        let data: Vec<u32> = (0..n as u32).collect();
        data_field.write(&data)?;

        bench("reduce_add_u32", n, || run_reduce(&gpu, &data_field, n));
        bench("scan_add_u32  ", n, || run_scan(&gpu, &data_field, n));
        bench("radix_sort_u32", n, || run_sort(&gpu, &data_field, n));
        bench("compact_u32   ", n, || run_compact(&gpu, &data_field, n));
        bench("histogram_u32 ", n, || run_histogram(&gpu, &data_field, n));
        bench("top_k_u32 k=16", n, || run_top_k(&gpu, &data_field, n));
        println!();
    }

    Ok(())
}
