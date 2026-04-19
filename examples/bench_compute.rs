//! Benchmark: GPU vs CPU on compute-heavy work.
//!
//! Each element does 1000 iterations of math.
//!
//! Run: cargo run --example bench_compute --release

use std::hint::black_box;
use std::time::Instant;

#[quanta::kernel]
fn heavy_compute(input: &[f32], output: &mut [f32]) {
    let i = quark_id();
    let mut x = input[i];
    for _ in 0..1000 {
        x = (x.sin() * x.cos()) + (x.abs() + 1.0f32).sqrt();
    }
    output[i] = x;
}

fn main() {
    let gpu = quanta::init().expect("no GPU found");
    println!("GPU: {}\n", gpu.name());

    for &count in &[1_000, 10_000, 100_000, 1_000_000] {
        let input: Vec<f32> = (0..count).map(|i| i as f32 * 0.001).collect();

        let fi = gpu.compute_field::<f32>(count).unwrap();
        let fo = gpu.compute_field::<f32>(count).unwrap();
        gpu.write_field(&fi, &input).unwrap();

        let mut wave = heavy_compute(&gpu).expect("create wave");
        wave.bind(0, &fi);
        wave.bind(1, &fo);

        // Warm up
        let mut p = gpu.dispatch(&wave, count as u32).unwrap();
        gpu.wait(&mut p).unwrap();

        let start = Instant::now();
        let mut p = gpu.dispatch(&wave, count as u32).unwrap();
        gpu.wait(&mut p).unwrap();
        let _result = gpu.read_field(&fo).unwrap();
        let gpu_time = start.elapsed();

        // CPU
        let start = Instant::now();
        let mut cpu_result = vec![0.0f32; count];
        for i in 0..count {
            let mut x = input[i];
            for _ in 0..1000 {
                x = x.sin() * x.cos() + (x.abs() + 1.0).sqrt();
            }
            cpu_result[i] = x;
        }
        black_box(&cpu_result);
        let cpu_time = start.elapsed();

        let speedup = cpu_time.as_nanos() as f64 / gpu_time.as_nanos() as f64;
        println!(
            "{:>10} elements:  CPU {:>10.2}ms  GPU {:>10.2}ms  → {:.0}x GPU",
            count,
            cpu_time.as_secs_f64() * 1000.0,
            gpu_time.as_secs_f64() * 1000.0,
            speedup
        );
    }
}
