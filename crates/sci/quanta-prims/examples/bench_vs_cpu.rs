//! Head-to-head: GPU primitive vs single-thread CPU reference.
//!
//! Measures wall-clock time for each primitive against the
//! reference implementation in `quanta_prims::reference`. Both
//! are timed end-to-end (write inputs, run, read outputs for
//! GPU; just run for CPU).
//!
//! Honest framing: the reference is single-thread. A fair
//! "GPU vs CPU at peak" comparison would multiply the CPU time
//! by `1 / num_cores`. We print both raw and the per-core
//! adjustment so the reader can decide.
//!
//! Run:
//!   cargo run -p quanta-prims --features gpu-metal --release \
//!     --example bench_vs_cpu

use std::time::Instant;

use quanta_prims::{
    block_radix_sort_u32_buffer, block_reduce_add_u32_buffer, block_scan_add_u32_buffer, reference,
};

const BLOCK: usize = 256;
const N: usize = 64 * BLOCK; // 16384 elements -- one workgroup per block, 64 blocks
const WARMUP: usize = 2;
const ITERS: usize = 10;

fn time<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[ITERS / 2]
}

fn print_row(name: &str, gpu_ms: f64, cpu_ms: f64) {
    let speedup = cpu_ms / gpu_ms;
    let cpu_par_est = cpu_ms / num_cpu_cores() as f64;
    let par_speedup = cpu_par_est / gpu_ms;
    println!(
        "  {name:<14}  GPU = {gpu_ms:>7.3} ms   1-core CPU = {cpu_ms:>7.3} ms   ratio = {speedup:>5.2}x   vs N-core est = {par_speedup:>5.2}x"
    );
}

fn num_cpu_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn main() -> Result<(), quanta::QuantaError> {
    let gpu = match quanta::init() {
        Ok(g) => g,
        Err(_) => {
            println!("no GPU backend available -- skipping bench");
            return Ok(());
        }
    };
    println!("== quanta-prims GPU vs CPU head-to-head ==");
    println!("backend       : {}", gpu.name());
    println!("element count : {N} ({} blocks of {BLOCK})", N / BLOCK);
    println!("warmup        : {WARMUP} iter");
    println!("samples       : {ITERS} iter (reporting median)");
    println!("cpu cores     : {} (parallelism estimate)", num_cpu_cores());
    println!();
    println!("  N-core est = single-thread CPU time / cpu_cores");
    println!("    (rough upper bound for embarrassingly-parallel CPU code)");
    println!();

    let data: Vec<u32> = (0..N as u32).collect();
    let data_field = gpu.field::<u32>(N)?;
    data_field.write(&data)?;

    // ── reduce ──────────────────────────────────────────────────
    let out_reduce = gpu.field::<u32>(N / BLOCK)?;
    let mut wave_reduce = block_reduce_add_u32_buffer(&gpu)?;
    wave_reduce.bind(0, &data_field);
    wave_reduce.bind(1, &out_reduce);
    let gpu_reduce_ms = time(|| {
        out_reduce.write(&vec![0u32; N / BLOCK]).unwrap();
        let mut pulse = gpu.dispatch(&wave_reduce, N as u32).unwrap();
        pulse.wait().unwrap();
        let _ = out_reduce.read().unwrap();
    });
    let cpu_reduce_ms = time(|| {
        let _ = reference::reduce_add_u32(&data);
    });
    print_row("reduce_add_u32", gpu_reduce_ms, cpu_reduce_ms);

    // ── scan ────────────────────────────────────────────────────
    let out_scan = gpu.field::<u32>(N)?;
    let mut wave_scan = block_scan_add_u32_buffer(&gpu)?;
    wave_scan.bind(0, &data_field);
    wave_scan.bind(1, &out_scan);
    let gpu_scan_ms = time(|| {
        out_scan.write(&vec![0u32; N]).unwrap();
        let mut pulse = gpu.dispatch(&wave_scan, N as u32).unwrap();
        pulse.wait().unwrap();
        let _ = out_scan.read().unwrap();
    });
    let cpu_scan_ms = time(|| {
        let _ = reference::scan_add_u32(&data);
    });
    print_row("scan_add_u32  ", gpu_scan_ms, cpu_scan_ms);

    // ── sort ────────────────────────────────────────────────────
    // For sort, use a non-sorted input. The CPU reference sorts
    // a freshly-cloned copy each call so the timer doesn't see
    // an already-sorted input.
    let mut state: u32 = 0xCAFE_BABE;
    let scramble: Vec<u32> = (0..N)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        })
        .collect();
    data_field.write(&scramble)?;
    let out_sort = gpu.field::<u32>(N)?;
    let mut wave_sort = block_radix_sort_u32_buffer(&gpu)?;
    wave_sort.bind(0, &data_field);
    wave_sort.bind(1, &out_sort);
    let gpu_sort_ms = time(|| {
        out_sort.write(&vec![0u32; N]).unwrap();
        let mut pulse = gpu.dispatch(&wave_sort, N as u32).unwrap();
        pulse.wait().unwrap();
        let _ = out_sort.read().unwrap();
    });
    let cpu_sort_ms = time(|| {
        let _ = reference::radix_sort_u32(&scramble);
    });
    print_row("radix_sort_u32", gpu_sort_ms, cpu_sort_ms);

    println!();
    println!("note: GPU times include host<->device memcpy. For");
    println!("kernels chained inside a larger pipeline that keeps");
    println!("data resident on the GPU, the practical speedup is");
    println!("higher than reported here.");

    Ok(())
}
