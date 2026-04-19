//! Benchmark: N-body gravity simulation — GPU vs CPU.
//!
//! O(N^2) interactions. The classic GPU workload.
//! Data stored as flat f32 arrays: [x0, y0, z0, mass0, x1, y1, z1, mass1, ...]
//!
//! Run: cargo run --example bench_nbody --release

use std::hint::black_box;
use std::time::Instant;

#[quanta::kernel]
fn nbody(positions: &[f32], velocities: &mut [f32], count: u32) {
    let idx = quark_id();
    let base = idx * 4u32;
    let px = positions[base];
    let py = positions[base + 1u32];
    let pz = positions[base + 2u32];
    let mut ax = 0.0f32;
    let mut ay = 0.0f32;
    let mut az = 0.0f32;
    let eps = 0.01f32;
    for j in 0..count {
        let jbase = j * 4u32;
        let dx = positions[jbase] - px;
        let dy = positions[jbase + 1u32] - py;
        let dz = positions[jbase + 2u32] - pz;
        let mass = positions[jbase + 3u32];
        let dist_sq = dx * dx + dy * dy + dz * dz + eps;
        let inv_dist = rsqrt(dist_sq);
        let inv_dist3 = inv_dist * inv_dist * inv_dist;
        ax += dx * inv_dist3 * mass;
        ay += dy * inv_dist3 * mass;
        az += dz * inv_dist3 * mass;
    }
    velocities[base] = velocities[base] + ax * 0.001f32;
    velocities[base + 1u32] = velocities[base + 1u32] + ay * 0.001f32;
    velocities[base + 2u32] = velocities[base + 2u32] + az * 0.001f32;
}

fn main() {
    let gpu = quanta::init().expect("no GPU found");
    println!("GPU: {}\n", gpu.name());

    let count = 16384usize;

    // Flatten positions: [x, y, z, mass] per particle
    let mut pos_flat = Vec::with_capacity(count * 4);
    for i in 0..count {
        let angle = i as f32 * 0.01;
        pos_flat.push(angle.cos() * (i as f32 * 0.01)); // x
        pos_flat.push(angle.sin() * (i as f32 * 0.01)); // y
        pos_flat.push(0.0f32); // z
        pos_flat.push(1.0f32); // mass
    }
    let vel_flat = vec![0.0f32; count * 4];

    let fp = gpu.compute_field::<f32>(count * 4).unwrap();
    let fv = gpu.compute_field::<f32>(count * 4).unwrap();
    gpu.write_field(&fp, &pos_flat).unwrap();
    gpu.write_field(&fv, &vel_flat).unwrap();

    let mut wave = nbody(&gpu).expect("create wave");
    wave.bind(0, &fp);
    wave.bind(1, &fv);
    wave.set_value(2, count as u32);

    let start = Instant::now();
    let mut p = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut p).unwrap();
    let gpu_time = start.elapsed();

    // CPU
    let start = Instant::now();
    let mut cpu_vel = vel_flat.clone();
    let eps = 0.01f32;
    for i in 0..count {
        let (mut ax, mut ay, mut az) = (0.0f32, 0.0f32, 0.0f32);
        let ib = i * 4;
        for j in 0..count {
            let jb = j * 4;
            let dx = pos_flat[jb] - pos_flat[ib];
            let dy = pos_flat[jb + 1] - pos_flat[ib + 1];
            let dz = pos_flat[jb + 2] - pos_flat[ib + 2];
            let mass = pos_flat[jb + 3];
            let dist_sq = dx * dx + dy * dy + dz * dz + eps;
            let inv_dist = 1.0 / dist_sq.sqrt();
            let inv_dist3 = inv_dist * inv_dist * inv_dist;
            ax += dx * inv_dist3 * mass;
            ay += dy * inv_dist3 * mass;
            az += dz * inv_dist3 * mass;
        }
        cpu_vel[ib] += ax * 0.001;
        cpu_vel[ib + 1] += ay * 0.001;
        cpu_vel[ib + 2] += az * 0.001;
    }
    black_box(&cpu_vel);
    let cpu_time = start.elapsed();

    let speedup = cpu_time.as_nanos() as f64 / gpu_time.as_nanos() as f64;
    println!(
        "N-body ({} particles, {}M interactions):",
        count,
        (count as u64 * count as u64) / 1_000_000
    );
    println!(
        "  CPU: {:.2}ms  GPU: {:.2}ms  → {:.0}x GPU",
        cpu_time.as_secs_f64() * 1000.0,
        gpu_time.as_secs_f64() * 1000.0,
        speedup
    );
}
