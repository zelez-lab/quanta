//! Benchmark: N-body gravity simulation — GPU vs CPU.
//!
//! Tiled kernel with SoA (Structure of Arrays) data layout.
//! Positions stored as separate x[], y[], z[], mass[] arrays for coalesced
//! GPU memory access. Tile size 512 for maximum shared memory utilization.
//!
//! Run: cargo run --example bench_nbody --release

use std::hint::black_box;
use std::time::Instant;

const TILE: usize = 512;

/// Tiled N-body with SoA layout.
/// Separate arrays: px[], py[], pz[], pm[] for coalesced reads.
/// Each workgroup loads a tile of 512 particles into shared memory.
#[quanta::kernel(workgroup = [512, 1, 1])]
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
    let sx: [f32; 512];
    #[quanta::shared]
    let sy: [f32; 512];
    #[quanta::shared]
    let sz: [f32; 512];
    #[quanta::shared]
    let sm: [f32; 512];

    let idx = quark_id();
    let lid = local_id();

    let my_x = px[idx];
    let my_y = py[idx];
    let my_z = pz[idx];

    let mut ax = 0.0f32;
    let mut ay = 0.0f32;
    let mut az = 0.0f32;
    let eps = 0.01f32;

    let num_tiles = (count + 511u32) / 512u32;
    for t in 0..num_tiles {
        // Cooperative load: each thread loads one particle (coalesced!)
        let src = t * 512u32 + lid;
        sx[lid] = px[src];
        sy[lid] = py[src];
        sz[lid] = pz[src];
        sm[lid] = pm[src];
        barrier();

        for j in 0..512u32 {
            let dx = sx[j] - my_x;
            let dy = sy[j] - my_y;
            let dz = sz[j] - my_z;
            let mass = sm[j];
            let dist_sq = dx * dx + dy * dy + dz * dz + eps;
            let inv_dist = rsqrt(dist_sq);
            let inv_dist3 = inv_dist * inv_dist * inv_dist;
            ax += dx * inv_dist3 * mass;
            ay += dy * inv_dist3 * mass;
            az += dz * inv_dist3 * mass;
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

    for &count in &[1024, 4096, 16384, 65536] {
        // Pad to multiple of TILE
        let padded = ((count + TILE - 1) / TILE) * TILE;

        let mut pos_x = vec![0.0f32; padded];
        let mut pos_y = vec![0.0f32; padded];
        let mut pos_z = vec![0.0f32; padded];
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

        let fpx = gpu.compute_field::<f32>(padded).unwrap();
        let fpy = gpu.compute_field::<f32>(padded).unwrap();
        let fpz = gpu.compute_field::<f32>(padded).unwrap();
        let fpm = gpu.compute_field::<f32>(padded).unwrap();
        let fvx = gpu.compute_field::<f32>(padded).unwrap();
        let fvy = gpu.compute_field::<f32>(padded).unwrap();
        let fvz = gpu.compute_field::<f32>(padded).unwrap();
        gpu.write_field(&fpx, &pos_x).unwrap();
        gpu.write_field(&fpy, &pos_y).unwrap();
        gpu.write_field(&fpz, &pos_z).unwrap();
        gpu.write_field(&fpm, &pos_m).unwrap();
        gpu.write_field(&fvx, &vel_x).unwrap();
        gpu.write_field(&fvy, &vel_y).unwrap();
        gpu.write_field(&fvz, &vel_z).unwrap();

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
        let mut pw = gpu.dispatch(&wave, padded as u32).unwrap();
        gpu.wait(&mut pw).unwrap();
        gpu.write_field(&fvx, &vel_x).unwrap();
        gpu.write_field(&fvy, &vel_y).unwrap();
        gpu.write_field(&fvz, &vel_z).unwrap();

        // Timed GPU
        let start = Instant::now();
        let mut p = gpu.dispatch(&wave, padded as u32).unwrap();
        gpu.wait(&mut p).unwrap();
        let gpu_time = start.elapsed();

        // CPU (single-threaded, SoA layout)
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
