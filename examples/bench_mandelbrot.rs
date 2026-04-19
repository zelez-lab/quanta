//! Benchmark: 4K Mandelbrot fractal — GPU vs CPU.
//!
//! Pure compute stress test. Each pixel iterates up to 1000 times.
//!
//! Run: cargo run --example bench_mandelbrot --release

use std::hint::black_box;
use std::time::Instant;

const MANDELBROT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void mandelbrot(
    device uint* output [[buffer(0)]],
    constant uint& width [[buffer(1)]],
    constant uint& height [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    uint px = idx % width;
    uint py = idx / width;
    float x0 = ((float)px / (float)width) * 3.5 - 2.5;
    float y0 = ((float)py / (float)height) * 2.0 - 1.0;
    float x = 0.0, y = 0.0;
    uint iter = 0;
    while (x*x + y*y <= 4.0 && iter < 1000) {
        float tmp = x*x - y*y + x0;
        y = 2.0*x*y + y0;
        x = tmp;
        iter++;
    }
    output[idx] = iter;
}
"#;

fn main() {
    let gpu = quanta::init().expect("no GPU found");
    println!("GPU: {}\n", gpu.name());

    let width: u32 = 3840;
    let height: u32 = 2160;
    let count = (width * height) as usize;

    let fo = gpu.compute_field::<u32>(count).unwrap();
    let mut wave = gpu.wave(MANDELBROT_MSL.as_bytes()).unwrap();
    wave.bind(0, &fo);
    wave.set_value(1, width);
    wave.set_value(2, height);

    let start = Instant::now();
    let p = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(p).unwrap();
    let gpu_time = start.elapsed();

    // CPU
    let start = Instant::now();
    let mut cpu_out = vec![0u32; count];
    for idx in 0..count {
        let px = idx % width as usize;
        let py = idx / width as usize;
        let x0 = (px as f32 / width as f32) * 3.5 - 2.5;
        let y0 = (py as f32 / height as f32) * 2.0 - 1.0;
        let (mut x, mut y) = (0.0f32, 0.0f32);
        let mut iter = 0u32;
        while x * x + y * y <= 4.0 && iter < 1000 {
            let tmp = x * x - y * y + x0;
            y = 2.0 * x * y + y0;
            x = tmp;
            iter += 1;
        }
        cpu_out[idx] = iter;
    }
    black_box(&cpu_out);
    let cpu_time = start.elapsed();

    let speedup = cpu_time.as_nanos() as f64 / gpu_time.as_nanos() as f64;
    println!("Mandelbrot 4K ({}x{}, {} pixels):", width, height, count);
    println!(
        "  CPU: {:.2}ms  GPU: {:.2}ms  → {:.0}x GPU",
        cpu_time.as_secs_f64() * 1000.0,
        gpu_time.as_secs_f64() * 1000.0,
        speedup
    );
}
