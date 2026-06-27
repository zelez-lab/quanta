//! GEMM micro-benchmark: naive vs tiled, same shape, GFLOP/s.
//!
//! GEMM is compute-bound (2·M·N·K flops), so the headline number is GFLOP/s
//! and the tiled/naive speedup. Run with a real backend:
//! `cargo bench -p quanta-blas --features gpu-metal`. Results land in
//! `PERFORMANCE.md`. Vendor (Accelerate/cuBLAS) links are a later increment.
//!
//! The bench also cross-checks that naive and tiled agree on each shape, so a
//! perf run doubles as a correctness smoke test.

use std::hint::black_box;
use std::time::Instant;

/// Square sizes to sweep. 16 is one tile; 512 is 32×32 tiles.
const SIZES: [usize; 4] = [64, 128, 256, 512];

fn main() {
    let Ok(gpu) = quanta::init() else {
        eprintln!("no GPU backend available — skipping quanta-blas gemm bench");
        return;
    };

    let tc = gpu.supports_cooperative_matrix();
    println!(
        "quanta-blas GEMM (f32, square M=N=K), naive vs tiled{}",
        if tc { " vs tensor-core" } else { "" }
    );
    if tc {
        println!(
            "{:>5} | {:>10} {:>9} | {:>10} {:>9} | {:>10} {:>9} | {:>9}",
            "N", "naive ms", "GFLOP/s", "tiled ms", "GFLOP/s", "tc ms", "GFLOP/s", "tc/tiled"
        );
    } else {
        println!(
            "{:>5} | {:>10} {:>9} | {:>10} {:>9} | {:>7}",
            "N", "naive ms", "GFLOP/s", "tiled ms", "GFLOP/s", "speedup"
        );
    }

    for &n in &SIZES {
        let flops = 2.0 * (n as f64) * (n as f64) * (n as f64);

        // Deterministic data; A·B over varied values exercises the real path.
        let a_host: Vec<f32> = (0..n * n).map(|i| ((i % 17) as f32) * 0.25 - 2.0).collect();
        let b_host: Vec<f32> = (0..n * n).map(|i| ((i % 13) as f32) * 0.5 - 3.0).collect();

        let a = gpu.field::<f32>(n * n).unwrap();
        let b = gpu.field::<f32>(n * n).unwrap();
        let c_naive = gpu.field::<f32>(n * n).unwrap();
        let c_tiled = gpu.field::<f32>(n * n).unwrap();
        a.write(&a_host).unwrap();
        b.write(&b_host).unwrap();

        let nu = n as u32;
        let run_naive = |c: &quanta::Field<f32>| {
            c.write(&vec![0.0f32; n * n]).unwrap();
            quanta_blas::gemm::gemm_naive(&gpu, nu, nu, nu, 1.0, &a, &b, 0.0, c).unwrap();
        };
        let run_tiled = |c: &quanta::Field<f32>| {
            c.write(&vec![0.0f32; n * n]).unwrap();
            quanta_blas::gemm::gemm_tiled(&gpu, nu, nu, nu, 1.0, &a, &b, 0.0, c).unwrap();
        };

        // Warm the JIT + pipeline cache for both kernels.
        run_naive(&c_naive);
        run_tiled(&c_tiled);

        // Cross-check correctness on this shape.
        let vn = c_naive.read().unwrap();
        let vt = c_tiled.read().unwrap();
        let max_rel = vn
            .iter()
            .zip(vt.iter())
            .map(|(&x, &y)| (x - y).abs() / (1.0 + x.abs()))
            .fold(0.0f32, f32::max);
        assert!(
            max_rel <= 1e-3,
            "naive vs tiled disagree at N={n}: rel {max_rel}"
        );

        let iters = if n <= 128 { 50 } else { 10 };

        let t = Instant::now();
        for _ in 0..iters {
            run_naive(black_box(&c_naive));
        }
        let naive_s = t.elapsed().as_secs_f64() / iters as f64;

        let t = Instant::now();
        for _ in 0..iters {
            run_tiled(black_box(&c_tiled));
        }
        let tiled_s = t.elapsed().as_secs_f64() / iters as f64;

        let naive_gflops = flops / naive_s / 1e9;
        let tiled_gflops = flops / tiled_s / 1e9;

        // Tensor-core path (C += A·B): only when supported and N is a multiple
        // of 8 (the v1 fragment-tile constraint).
        if tc && n % 8 == 0 {
            let c_tc = gpu.field::<f32>(n * n).unwrap();
            let run_tc = |c: &quanta::Field<f32>| {
                c.write(&vec![0.0f32; n * n]).unwrap();
                quanta_blas::gemm_f32_tc(&gpu, nu, nu, nu, &a, &b, c).unwrap();
            };
            run_tc(&c_tc); // warm
            let t = Instant::now();
            for _ in 0..iters {
                run_tc(black_box(&c_tc));
            }
            let tc_s = t.elapsed().as_secs_f64() / iters as f64;
            println!(
                "{:>5} | {:>10.3} {:>9.2} | {:>10.3} {:>9.2} | {:>10.3} {:>9.2} | {:>8.2}x",
                n,
                naive_s * 1e3,
                naive_gflops,
                tiled_s * 1e3,
                tiled_gflops,
                tc_s * 1e3,
                flops / tc_s / 1e9,
                tiled_s / tc_s,
            );
        } else {
            println!(
                "{:>5} | {:>10.3} {:>9.2} | {:>10.3} {:>9.2} | {:>6.2}x",
                n,
                naive_s * 1e3,
                naive_gflops,
                tiled_s * 1e3,
                tiled_gflops,
                naive_s / tiled_s,
            );
        }
    }
}
