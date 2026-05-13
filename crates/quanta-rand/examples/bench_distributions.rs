//! Quick benchmark sweep over the v0.1 distribution surface.
//!
//! Times each `fill_*` kernel on the CPU backend (which is what's
//! reachable on a Mac without Metal Toolchain). Real GPU numbers
//! will be substantially higher throughput; this gives a baseline.
//!
//! Output is a table of (distribution, len, ms, M-samples/sec).
//!
//! Run: cargo run --example bench_distributions --features gpu --release

use std::time::Instant;

use quanta_rand::{
    fill_bernoulli_u32_gpu, fill_exponential_f32_gpu, fill_lognormal_f32_gpu, fill_normal_f32_gpu,
    fill_poisson_u32_gpu, fill_uniform_f32_gpu, fill_uniform_f64_gpu, fill_uniform_u32_gpu,
    fill_uniform_u64_gpu,
};

const SEED: u64 = 0xCAFE_BABE_DEAD_BEEFu64;
// 64K samples per call — large enough for stable timing, small
// enough that the CPU-JIT backend completes the full sweep in
// reasonable time. Native GPU backends will handle 1M+ trivially.
const N: usize = 1 << 16;
const WARMUP: usize = 1;
const ITERS: usize = 3;

fn time_run<F>(name: &str, mut f: F)
where
    F: FnMut() -> usize,
{
    // Warm up so the wasm compile / cache hit doesn't pollute timing.
    for _ in 0..WARMUP {
        let _ = f();
    }
    let mut best_ms = f64::INFINITY;
    let mut total_n: usize = 0;
    for _ in 0..ITERS {
        let t = Instant::now();
        let n = f();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if ms < best_ms {
            best_ms = ms;
            total_n = n;
        }
    }
    let m_per_sec = total_n as f64 / 1e6 / (best_ms / 1e3);
    println!("  {name:<28} n={N:>8} best={best_ms:>7.2}ms  {m_per_sec:>7.1} M/s");
}

fn main() {
    println!("Quanta-rand v0.1 distribution benchmark");
    println!("  Backend: software (CPU JIT)");
    println!("  N per call: {N}");
    println!("  Warmup: {WARMUP} runs, Best of: {ITERS} runs");
    println!();

    let gpu = quanta::init_cpu();

    time_run("uniform_u32", || {
        fill_uniform_u32_gpu(&gpu, N, SEED).unwrap().len()
    });
    time_run("uniform_u64", || {
        fill_uniform_u64_gpu(&gpu, N, SEED).unwrap().len()
    });
    time_run("uniform_f32", || {
        fill_uniform_f32_gpu(&gpu, N, SEED).unwrap().len()
    });
    time_run("uniform_f64", || {
        fill_uniform_f64_gpu(&gpu, N, SEED).unwrap().len()
    });
    time_run("normal_f32", || {
        fill_normal_f32_gpu(&gpu, N, SEED).unwrap().len()
    });
    time_run("exponential_f32", || {
        fill_exponential_f32_gpu(&gpu, N, SEED, 1.0).unwrap().len()
    });
    time_run("lognormal_f32", || {
        fill_lognormal_f32_gpu(&gpu, N, SEED, 0.0, 1.0)
            .unwrap()
            .len()
    });
    time_run("bernoulli_u32(p=0.5)", || {
        fill_bernoulli_u32_gpu(&gpu, N, SEED, 0.5).unwrap().len()
    });
    time_run("poisson_u32(lambda=5)", || {
        fill_poisson_u32_gpu(&gpu, N, SEED, 5.0).unwrap().len()
    });

    println!();
    println!("note: numbers above are CPU JIT throughput; native GPU");
    println!("(Metal/Vulkan) will run substantially faster.");
}
