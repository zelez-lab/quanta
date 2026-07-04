//! Benchmark: N-body gravity simulation — GPU vs CPU.
//!
//! Optimized tiled kernel:
//! - SoA data layout for coalesced global reads
//! - Tile size 256 with shared memory (256 is the V3D max workgroup size,
//!   so the same kernel dispatches on Raspberry Pi as well as Metal/desktop)
//! - Inner loop unrolled 4x for better ILP
//! - addCompletedHandler for async GPU notification
//!
//! Run: cargo run --example bench_nbody --release

use std::hint::black_box;
use std::time::Instant;

const TILE: usize = 256;

/// Tiled N-body with SoA layout and 4x unrolled inner loop.
#[quanta::kernel(workgroup = [256, 1, 1])]
fn nbody_soa(
    px: &[f32],
    py: &[f32],
    pz: &[f32],
    pm: &[f32],
    vx: &mut [f32],
    vy: &mut [f32],
    vz: &mut [f32],
    count: u32,
) {
    #[quanta::shared]
    let sx: [f32; 256];
    #[quanta::shared]
    let sy: [f32; 256];
    #[quanta::shared]
    let sz: [f32; 256];
    #[quanta::shared]
    let sm: [f32; 256];

    let idx = quark_id();
    let lid = proton_id();

    let my_x = px[idx];
    let my_y = py[idx];
    let my_z = pz[idx];

    let mut ax = 0.0f32;
    let mut ay = 0.0f32;
    let mut az = 0.0f32;
    let eps = 0.01f32;

    let num_tiles = (count + 255u32) / 256u32;
    for t in 0..num_tiles {
        let src = t * 256u32 + lid;
        sx[lid] = px[src];
        sy[lid] = py[src];
        sz[lid] = pz[src];
        sm[lid] = pm[src];
        barrier();

        // Unrolled 4x: process 4 particles per iteration
        let iters = 256u32 / 4u32;
        for j in 0..iters {
            let j0 = j * 4u32;

            let dx0 = sx[j0] - my_x;
            let dy0 = sy[j0] - my_y;
            let dz0 = sz[j0] - my_z;
            let m0 = sm[j0];
            let d0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps;
            let i0 = rsqrt(d0);
            let i03 = i0 * i0 * i0;
            ax += dx0 * i03 * m0;
            ay += dy0 * i03 * m0;
            az += dz0 * i03 * m0;

            let dx1 = sx[j0 + 1u32] - my_x;
            let dy1 = sy[j0 + 1u32] - my_y;
            let dz1 = sz[j0 + 1u32] - my_z;
            let m1 = sm[j0 + 1u32];
            let d1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + eps;
            let i1 = rsqrt(d1);
            let i13 = i1 * i1 * i1;
            ax += dx1 * i13 * m1;
            ay += dy1 * i13 * m1;
            az += dz1 * i13 * m1;

            let dx2 = sx[j0 + 2u32] - my_x;
            let dy2 = sy[j0 + 2u32] - my_y;
            let dz2 = sz[j0 + 2u32] - my_z;
            let m2 = sm[j0 + 2u32];
            let d2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2 + eps;
            let i2 = rsqrt(d2);
            let i23 = i2 * i2 * i2;
            ax += dx2 * i23 * m2;
            ay += dy2 * i23 * m2;
            az += dz2 * i23 * m2;

            let dx3 = sx[j0 + 3u32] - my_x;
            let dy3 = sy[j0 + 3u32] - my_y;
            let dz3 = sz[j0 + 3u32] - my_z;
            let m3 = sm[j0 + 3u32];
            let d3 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3 + eps;
            let i3 = rsqrt(d3);
            let i33 = i3 * i3 * i3;
            ax += dx3 * i33 * m3;
            ay += dy3 * i33 * m3;
            az += dz3 * i33 * m3;
        }
        barrier();
    }

    let dt = 0.001f32;
    vx[idx] = vx[idx] + ax * dt;
    vy[idx] = vy[idx] + ay * dt;
    vz[idx] = vz[idx] + az * dt;
}

fn main() {
    let gpu = quanta::init().expect("no GPU found");
    println!("GPU: {}\n", gpu.name());

    for &count in &[1024usize, 4096, 16384, 65536] {
        let padded = count.div_ceil(TILE) * TILE;

        let mut pos_x = vec![0.0f32; padded];
        let mut pos_y = vec![0.0f32; padded];
        let pos_z = vec![0.0f32; padded];
        let mut pos_m = vec![0.0f32; padded];
        for i in 0..count {
            let angle = i as f32 * 0.01;
            pos_x[i] = angle.cos() * (i as f32 * 0.01);
            pos_y[i] = angle.sin() * (i as f32 * 0.01);
            pos_m[i] = 1.0;
        }
        let vel_x = vec![0.0f32; padded];
        let vel_y = vec![0.0f32; padded];
        let vel_z = vec![0.0f32; padded];

        let fpx = gpu.field::<f32>(padded).unwrap();
        let fpy = gpu.field::<f32>(padded).unwrap();
        let fpz = gpu.field::<f32>(padded).unwrap();
        let fpm = gpu.field::<f32>(padded).unwrap();
        let fvx = gpu.field::<f32>(padded).unwrap();
        let fvy = gpu.field::<f32>(padded).unwrap();
        let fvz = gpu.field::<f32>(padded).unwrap();
        fpx.write(&pos_x).unwrap();
        fpy.write(&pos_y).unwrap();
        fpz.write(&pos_z).unwrap();
        fpm.write(&pos_m).unwrap();
        fvx.write(&vel_x).unwrap();
        fvy.write(&vel_y).unwrap();
        fvz.write(&vel_z).unwrap();

        let mut wave = nbody_soa(&gpu).expect("create wave");
        wave.bind(0, &fpx);
        wave.bind(1, &fpy);
        wave.bind(2, &fpz);
        wave.bind(3, &fpm);
        wave.bind(4, &fvx);
        wave.bind(5, &fvy);
        wave.bind(6, &fvz);
        wave.set_value(7, padded as u32);

        // Warmup
        gpu.dispatch(&wave, padded as u32).unwrap().wait().unwrap();
        fvx.write(&vel_x).unwrap();
        fvy.write(&vel_y).unwrap();
        fvz.write(&vel_z).unwrap();

        // Timed GPU
        let start = Instant::now();
        gpu.dispatch(&wave, padded as u32).unwrap().wait().unwrap();
        let gpu_time = start.elapsed();

        // CPU (single-threaded)
        let start = Instant::now();
        let mut cvx = vel_x.clone();
        let mut cvy = vel_y.clone();
        let mut cvz = vel_z.clone();
        let eps = 0.01f32;
        let dt = 0.001f32;
        for i in 0..count {
            let (mut cax, mut cay, mut caz) = (0.0f32, 0.0f32, 0.0f32);
            for j in 0..count {
                let dx = pos_x[j] - pos_x[i];
                let dy = pos_y[j] - pos_y[i];
                let dz = pos_z[j] - pos_z[i];
                let mass = pos_m[j];
                let dist_sq = dx * dx + dy * dy + dz * dz + eps;
                let inv_dist = 1.0 / dist_sq.sqrt();
                let inv_dist3 = inv_dist * inv_dist * inv_dist;
                cax += dx * inv_dist3 * mass;
                cay += dy * inv_dist3 * mass;
                caz += dz * inv_dist3 * mass;
            }
            cvx[i] += cax * dt;
            cvy[i] += cay * dt;
            cvz[i] += caz * dt;
        }
        black_box((&cvx, &cvy, &cvz));
        let cpu_time = start.elapsed();

        let speedup = cpu_time.as_nanos() as f64 / gpu_time.as_nanos() as f64;
        println!(
            "{:>6} particles ({:>5}M):  CPU {:>9.2}ms  GPU {:>9.2}ms  -> {:.1}x",
            count,
            (count as u64 * count as u64) / 1_000_000,
            cpu_time.as_secs_f64() * 1000.0,
            gpu_time.as_secs_f64() * 1000.0,
            speedup
        );
    }
}
