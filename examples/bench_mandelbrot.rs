//! Benchmark: 4K Mandelbrot fractal — GPU vs CPU.
//!
//! Pure compute stress test. Each pixel iterates up to 1000 times.
//!
//! Run: cargo run --example bench_mandelbrot --release

use std::hint::black_box;
use std::time::Instant;

#[quanta::kernel]
fn mandelbrot(output: &mut [u32], width: u32, height: u32) {
    let idx = quark_id();
    let px = idx % width;
    let py = idx / width;
    let x0 = (px as f32 / width as f32) * 3.5f32 - 2.5f32;
    let y0 = (py as f32 / height as f32) * 2.0f32 - 1.0f32;
    let (mut x, mut y) = (0.0f32, 0.0f32);
    let mut iter = 0u32;
    while x * x + y * y <= 4.0f32 && iter < 1000u32 {
        let tmp = x * x - y * y + x0;
        y = 2.0f32 * x * y + y0;
        x = tmp;
        iter += 1u32;
    }
    output[idx] = iter;
}

fn main() {
    let gpu = quanta::init().expect("no GPU found");
    println!("GPU: {}\n", gpu.name());

    let width: u32 = 3840;
    let height: u32 = 2160;
    let count = (width * height) as usize;

    let fo = gpu.field::<u32>(count).unwrap();
    let mut wave = mandelbrot(&gpu).expect("create wave");
    wave.bind(0, &fo);
    wave.set_value(1, width);
    wave.set_value(2, height);

    let start = Instant::now();
    gpu.dispatch(&wave, count as u32).unwrap().wait().unwrap();
    let gpu_time = start.elapsed();

    // CPU
    let start = Instant::now();
    let mut cpu_out = vec![0u32; count];
    for (idx, out) in cpu_out.iter_mut().enumerate().take(count) {
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
        *out = iter;
    }
    black_box(&cpu_out);
    let cpu_time = start.elapsed();

    let speedup = cpu_time.as_nanos() as f64 / gpu_time.as_nanos() as f64;
    println!("Mandelbrot 4K ({}x{}, {} pixels):", width, height, count);
    println!(
        "  CPU: {:.2}ms  GPU: {:.2}ms  -> {:.0}x GPU",
        cpu_time.as_secs_f64() * 1000.0,
        gpu_time.as_secs_f64() * 1000.0,
        speedup
    );
}
