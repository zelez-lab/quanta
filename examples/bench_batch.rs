//! Benchmark: batched dispatch vs sequential dispatch.
//!
//! Dispatches N kernels sequentially (N command buffers, N commits, N fences)
//! vs batched (1 command buffer, 1 commit, 1 fence).
//!
//! Run: cargo run --example bench_batch --release

use std::time::Instant;

#[quanta::kernel]
fn add_one(data: &mut [f32]) {
    let i = quark_id();
    data[i] = data[i] + 1.0;
}

fn main() {
    let gpu = quanta::init().expect("no GPU found");
    println!("GPU: {}\n", gpu.name());

    let count = 1024usize;
    let field = gpu.field::<f32>(count).unwrap();
    field.write(&vec![0.0f32; count]).unwrap();

    let mut wave = add_one(&gpu).expect("create wave");
    wave.bind(0, &field);

    // Warmup
    gpu.dispatch(&wave, count as u32).unwrap().wait().unwrap();

    for &num_dispatches in &[4, 8, 16, 32, 64] {
        // Reset
        field.write(&vec![0.0f32; count]).unwrap();

        // Sequential: N separate dispatches, each with its own commit + fence
        let start = Instant::now();
        for _ in 0..num_dispatches {
            gpu.dispatch(&wave, count as u32).unwrap().wait().unwrap();
        }
        let seq_time = start.elapsed();

        // Reset
        field.write(&vec![0.0f32; count]).unwrap();

        // Batched: 1 command buffer, N dispatches, 1 commit + 1 fence
        let start = Instant::now();
        let mut batch = gpu.batch().unwrap();
        for _ in 0..num_dispatches {
            batch.dispatch(&wave, count as u32).unwrap();
        }
        batch.pulse().unwrap().wait().unwrap();
        let batch_time = start.elapsed();

        let speedup = seq_time.as_nanos() as f64 / batch_time.as_nanos() as f64;
        println!(
            "{:>3} dispatches:  sequential {:>8.2}ms  batched {:>8.2}ms  -> {:.1}x",
            num_dispatches,
            seq_time.as_secs_f64() * 1000.0,
            batch_time.as_secs_f64() * 1000.0,
            speedup
        );
    }

    // Verify batched dispatch produces correct results
    field.write(&vec![0.0f32; count]).unwrap();
    let mut batch = gpu.batch().unwrap();
    for _ in 0..10 {
        batch.dispatch(&wave, count as u32).unwrap();
    }
    batch.pulse().unwrap().wait().unwrap();
    let result = field.read().unwrap();
    assert!(
        (result[0] - 10.0).abs() < 0.01,
        "10 dispatches of add_one should give 10.0, got {}",
        result[0]
    );
    println!("\nCorrectness: 10 batched add_one -> {:.1}", result[0]);
}
