//! The four benchmarks. Each emits one or more `BenchResult` entries.

use std::time::Instant;

use crate::result::{BenchResult, Report};

#[quanta::kernel(jit)]
fn heavy_compute(input: &[f32], output: &mut [f32]) {
    let i = quark_id();
    let mut x = input[i];
    for _ in 0..1000 {
        x = (x.sin() * x.cos()) + (x.abs() + 1.0f32).sqrt();
    }
    output[i] = x;
}

#[quanta::kernel(jit)]
fn add_one(data: &mut [f32]) {
    let i = quark_id();
    data[i] = data[i] + 1.0;
}

#[quanta::kernel(jit)]
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

pub fn run_all(smoke: bool) -> Result<Report, String> {
    let gpu = quanta::init().map_err(|e| format!("init: {:?}", e))?;
    let gpu_name = gpu.name().to_string();
    let _ = smoke;
    let mut report = Report::new(platform_id(), gpu_name);

    bench_compute(&gpu, smoke, &mut report)?;
    bench_addone(&gpu, smoke, &mut report)?;
    bench_mandelbrot(&gpu, smoke, &mut report)?;

    Ok(report)
}

fn bench_compute(gpu: &quanta::Gpu, smoke: bool, report: &mut Report) -> Result<(), String> {
    let sizes: &[usize] = if smoke {
        &[1_000]
    } else {
        &[1_000, 10_000, 100_000, 1_000_000]
    };
    for &count in sizes {
        let input: Vec<f32> = (0..count).map(|i| i as f32 * 0.001).collect();
        let fi = gpu.field::<f32>(count).map_err(|e| format!("{:?}", e))?;
        let fo = gpu.field::<f32>(count).map_err(|e| format!("{:?}", e))?;
        fi.write(&input).map_err(|e| format!("{:?}", e))?;
        let mut wave = heavy_compute(gpu).map_err(|e| format!("{:?}", e))?;
        wave.bind(0, &fi);
        wave.bind(1, &fo);
        // Warm
        gpu.dispatch(&wave, count as u32)
            .map_err(|e| format!("{:?}", e))?
            .wait()
            .map_err(|e| format!("{:?}", e))?;
        let t = Instant::now();
        gpu.dispatch(&wave, count as u32)
            .map_err(|e| format!("{:?}", e))?
            .wait()
            .map_err(|e| format!("{:?}", e))?;
        let _ = fo.read().map_err(|e| format!("{:?}", e))?;
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        let cpu_ms = if smoke {
            None
        } else {
            let t2 = Instant::now();
            let mut out = vec![0.0f32; count];
            for i in 0..count {
                let mut x = input[i];
                for _ in 0..1000 {
                    x = x.sin() * x.cos() + (x.abs() + 1.0).sqrt();
                }
                out[i] = x;
            }
            std::hint::black_box(&out);
            Some(t2.elapsed().as_secs_f64() * 1000.0)
        };

        report.results.push(BenchResult {
            name: "heavy_compute".into(),
            workload: format!("{}_elements", count),
            elements: count as u64,
            gpu_ms,
            cpu_ms,
        });
    }
    Ok(())
}

fn bench_addone(gpu: &quanta::Gpu, smoke: bool, report: &mut Report) -> Result<(), String> {
    let count = if smoke { 1024 } else { 1024 * 1024 };
    let field = gpu.field::<f32>(count).map_err(|e| format!("{:?}", e))?;
    field
        .write(&vec![0.0f32; count])
        .map_err(|e| format!("{:?}", e))?;
    let mut wave = add_one(gpu).map_err(|e| format!("{:?}", e))?;
    wave.bind(0, &field);
    // Warm
    gpu.dispatch(&wave, count as u32)
        .map_err(|e| format!("{:?}", e))?
        .wait()
        .map_err(|e| format!("{:?}", e))?;
    let t = Instant::now();
    let iters = if smoke { 1 } else { 64 };
    for _ in 0..iters {
        gpu.dispatch(&wave, count as u32)
            .map_err(|e| format!("{:?}", e))?
            .wait()
            .map_err(|e| format!("{:?}", e))?;
    }
    let gpu_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    report.results.push(BenchResult {
        name: "add_one_dispatch".into(),
        workload: format!("{}x_dispatch_{}_elements", iters, count),
        elements: count as u64,
        gpu_ms,
        cpu_ms: None,
    });
    Ok(())
}

fn bench_mandelbrot(gpu: &quanta::Gpu, smoke: bool, report: &mut Report) -> Result<(), String> {
    let (width, height) = if smoke {
        (256u32, 256u32)
    } else {
        (3840u32, 2160u32)
    };
    let count = (width as usize) * (height as usize);
    let out = gpu.field::<u32>(count).map_err(|e| format!("{:?}", e))?;
    let mut wave = mandelbrot(gpu).map_err(|e| format!("{:?}", e))?;
    wave.bind(0, &out);
    wave.set_value(1, width);
    wave.set_value(2, height);
    // Warm
    gpu.dispatch(&wave, count as u32)
        .map_err(|e| format!("{:?}", e))?
        .wait()
        .map_err(|e| format!("{:?}", e))?;
    let t = Instant::now();
    gpu.dispatch(&wave, count as u32)
        .map_err(|e| format!("{:?}", e))?
        .wait()
        .map_err(|e| format!("{:?}", e))?;
    let _ = out.read().map_err(|e| format!("{:?}", e))?;
    let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;
    report.results.push(BenchResult {
        name: "mandelbrot".into(),
        workload: format!("{}x{}", width, height),
        elements: count as u64,
        gpu_ms,
        cpu_ms: None,
    });
    Ok(())
}

fn platform_id() -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    format!("{}-{}", os, arch)
}
