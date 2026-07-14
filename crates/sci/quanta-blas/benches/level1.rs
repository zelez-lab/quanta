//! Level-1 BLAS micro-benchmarks: reference CPU vs GPU JIT.
//!
//! Bandwidth-bound ops, so the headline number is GB/s vs memory roofline.
//! Run with a real backend: `cargo bench -p quanta-blas --features gpu-metal`.
//! Results land in `PERFORMANCE.md`. Vendor (Accelerate/cuBLAS) links are a
//! later increment.

use std::hint::black_box;
use std::time::Instant;

fn main() {
    let Ok(gpu) = quanta::init() else {
        eprintln!("no GPU backend available — skipping quanta-blas benches");
        return;
    };

    const N: usize = 1 << 20; // 1M elements
    let x_host: Vec<f32> = (0..N).map(|i| (i % 97) as f32 * 0.5).collect();
    let y_host: Vec<f32> = (0..N).map(|i| (i % 89) as f32 * 0.25).collect();

    let x = gpu.field::<f32>(N).unwrap();
    let y = gpu.field::<f32>(N).unwrap();
    x.write(&x_host).unwrap();
    y.write(&y_host).unwrap();

    // warm up the JIT + pipeline cache
    quanta_blas::scal(&gpu, 1.0, &x).unwrap();
    let _ = quanta_blas::dot(&gpu, &x, &y).unwrap();

    let iters = 50;

    let t = Instant::now();
    for _ in 0..iters {
        quanta_blas::axpy(&gpu, black_box(2.0), &x, &y).unwrap();
    }
    let axpy_s = t.elapsed().as_secs_f64() / iters as f64;
    // axpy touches 2 reads + 1 write of N f32 = 12N bytes
    let axpy_gbs = (12.0 * N as f64) / axpy_s / 1e9;

    let t = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..iters {
        acc += quanta_blas::dot(&gpu, &x, &y).unwrap();
    }
    let dot_s = t.elapsed().as_secs_f64() / iters as f64;
    let dot_gbs = (8.0 * N as f64) / dot_s / 1e9; // 2 reads of N f32

    println!("quanta-blas Level-1 @ N={N}");
    println!("  axpy: {:.3} ms  ({:.1} GB/s)", axpy_s * 1e3, axpy_gbs);
    println!(
        "  dot : {:.3} ms  ({:.1} GB/s)  [acc={}]",
        dot_s * 1e3,
        dot_gbs,
        black_box(acc)
    );
}
