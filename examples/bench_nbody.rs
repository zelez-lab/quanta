//! Benchmark: N-body gravity simulation — GPU vs CPU.
//!
//! O(N²) interactions. The classic GPU workload.
//!
//! Run: cargo run --example bench_nbody --release

use std::hint::black_box;
use std::time::Instant;

// N-body uses complex index expressions (idx*4+1, compound assignment on indexed LHS).
// Use MSL kernel directly until the AST parser supports these patterns.
const NBODY_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void nbody(
    device const float4* positions [[buffer(0)]],
    device float4* velocities      [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    float4 pos = positions[idx];
    float3 acc = float3(0.0);
    float eps = 0.01;
    for (uint j = 0; j < count; j++) {
        float3 diff = positions[j].xyz - pos.xyz;
        float dist_sq = dot(diff, diff) + eps;
        float inv_dist = rsqrt(dist_sq);
        float inv_dist3 = inv_dist * inv_dist * inv_dist;
        acc += diff * inv_dist3 * positions[j].w;
    }
    velocities[idx].xyz += acc * 0.001;
}
"#;

fn main() {
    let gpu = quanta::init().expect("no GPU found");
    println!("GPU: {}\n", gpu.name());

    let count = 16384usize;
    let positions: Vec<[f32; 4]> = (0..count)
        .map(|i| {
            let angle = i as f32 * 0.01;
            [
                angle.cos() * (i as f32 * 0.01),
                angle.sin() * (i as f32 * 0.01),
                0.0,
                1.0,
            ]
        })
        .collect();
    let velocities: Vec<[f32; 4]> = vec![[0.0f32; 4]; count];

    let fp = gpu.compute_field::<[f32; 4]>(count).unwrap();
    let fv = gpu.compute_field::<[f32; 4]>(count).unwrap();
    gpu.write_field(&fp, &positions).unwrap();
    gpu.write_field(&fv, &velocities).unwrap();

    let mut wave = gpu.wave(NBODY_MSL.as_bytes()).unwrap();
    wave.bind(0, &fp);
    wave.bind(1, &fv);
    wave.set_value(2, count as u32);

    let start = Instant::now();
    let p = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(p).unwrap();
    let gpu_time = start.elapsed();

    // CPU
    let start = Instant::now();
    let mut cpu_vel = velocities.clone();
    let eps = 0.01f32;
    for i in 0..count {
        let (mut ax, mut ay, mut az) = (0.0f32, 0.0f32, 0.0f32);
        for j in 0..count {
            let dx = positions[j][0] - positions[i][0];
            let dy = positions[j][1] - positions[i][1];
            let dz = positions[j][2] - positions[i][2];
            let dist_sq = dx * dx + dy * dy + dz * dz + eps;
            let inv_dist = 1.0 / dist_sq.sqrt();
            let inv_dist3 = inv_dist * inv_dist * inv_dist;
            ax += dx * inv_dist3 * positions[j][3];
            ay += dy * inv_dist3 * positions[j][3];
            az += dz * inv_dist3 * positions[j][3];
        }
        cpu_vel[i][0] += ax * 0.001;
        cpu_vel[i][1] += ay * 0.001;
        cpu_vel[i][2] += az * 0.001;
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
