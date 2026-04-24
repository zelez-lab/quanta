//! API Design Playground
//!
//! This file shows a COMPLETE GPU computation using today's API.
//! Read it, then rewrite it in examples/api_design_dream.rs with
//! the API you WISH you had. We'll make it real.
//!
//! Task: N-body gravity with tiled shared memory.
//! - Upload particle positions (x, y, z, mass as separate arrays)
//! - Dispatch a compute kernel that accumulates forces
//! - Read back updated velocities
//! - Print a result to verify correctness

// ============================================================================
// Step 1: Define the kernel
//
// The macro compiles this to SPIR-V + metallib + WGSL + PTX + GCN.
// Parameters become buffer bindings (slot 0, 1, 2...) or push constants.
// The user must remember the slot order when binding later.
// ============================================================================

#[quanta::kernel(workgroup = [256, 1, 1])]
fn gravity(
    pos_x: &[f32],     // slot 0: read-only buffer
    pos_y: &[f32],     // slot 1: read-only buffer
    pos_z: &[f32],     // slot 2: read-only buffer
    mass: &[f32],      // slot 3: read-only buffer
    vel_x: &mut [f32], // slot 4: read-write buffer
    vel_y: &mut [f32], // slot 5: read-write buffer
    vel_z: &mut [f32], // slot 6: read-write buffer
    count: u32,        // slot 7: push constant
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

    let my_x = pos_x[idx];
    let my_y = pos_y[idx];
    let my_z = pos_z[idx];

    let mut ax = 0.0f32;
    let mut ay = 0.0f32;
    let mut az = 0.0f32;

    let num_tiles = (count + 255u32) / 256u32;
    for t in 0..num_tiles {
        let src = t * 256u32 + lid;
        sx[lid] = pos_x[src];
        sy[lid] = pos_y[src];
        sz[lid] = pos_z[src];
        sm[lid] = mass[src];
        barrier();

        for j in 0..256u32 {
            let dx = sx[j] - my_x;
            let dy = sy[j] - my_y;
            let dz = sz[j] - my_z;
            let m = sm[j];
            let dist_sq = dx * dx + dy * dy + dz * dz + 0.01f32;
            let inv = rsqrt(dist_sq);
            let inv3 = inv * inv * inv;
            ax += dx * inv3 * m;
            ay += dy * inv3 * m;
            az += dz * inv3 * m;
        }
        barrier();
    }

    vel_x[idx] = vel_x[idx] + ax * 0.001f32;
    vel_y[idx] = vel_y[idx] + ay * 0.001f32;
    vel_z[idx] = vel_z[idx] + az * 0.001f32;
}

// ============================================================================
// Step 2: The main function — allocate, bind, dispatch, read back
//
// Notice:
// - 7 separate gpu.compute_field() calls
// - 7 separate gpu.write_field() calls
// - 7 wave.bind() calls with manual slot numbers (0-6)
// - 1 wave.set_value() for the push constant (slot 7)
// - Easy to swap slot 4 and 5 with no compiler error
// ============================================================================

fn main() {
    let gpu = quanta::init().expect("no GPU");
    println!("GPU: {}\n", gpu.name());

    let n = 1024usize;
    let padded = ((n + 255) / 256) * 256;

    // --- Generate particle data ---
    let mut px = vec![0.0f32; padded];
    let mut py = vec![0.0f32; padded];
    let mut pz = vec![0.0f32; padded];
    let mut pm = vec![0.0f32; padded];
    for i in 0..n {
        let angle = i as f32 * 0.01;
        px[i] = angle.cos() * (i as f32 * 0.01);
        py[i] = angle.sin() * (i as f32 * 0.01);
        pm[i] = 1.0;
    }
    let vx = vec![0.0f32; padded];
    let vy = vec![0.0f32; padded];
    let vz = vec![0.0f32; padded];

    // --- Allocate GPU buffers (7 calls) ---
    let fpx = gpu.compute_field::<f32>(padded).unwrap();
    let fpy = gpu.compute_field::<f32>(padded).unwrap();
    let fpz = gpu.compute_field::<f32>(padded).unwrap();
    let fpm = gpu.compute_field::<f32>(padded).unwrap();
    let fvx = gpu.compute_field::<f32>(padded).unwrap();
    let fvy = gpu.compute_field::<f32>(padded).unwrap();
    let fvz = gpu.compute_field::<f32>(padded).unwrap();

    // --- Upload data (7 calls) ---
    gpu.write_field(&fpx, &px).unwrap();
    gpu.write_field(&fpy, &py).unwrap();
    gpu.write_field(&fpz, &pz).unwrap();
    gpu.write_field(&fpm, &pm).unwrap();
    gpu.write_field(&fvx, &vx).unwrap();
    gpu.write_field(&fvy, &vy).unwrap();
    gpu.write_field(&fvz, &vz).unwrap();

    // --- Create wave and bind (8 calls, manual slot numbers) ---
    let mut wave = gravity(&gpu).expect("create wave");
    wave.bind(0, &fpx); // pos_x  — must match kernel param order
    wave.bind(1, &fpy); // pos_y
    wave.bind(2, &fpz); // pos_z
    wave.bind(3, &fpm); // mass
    wave.bind(4, &fvx); // vel_x  — swap with slot 5 = silent bug
    wave.bind(5, &fvy); // vel_y
    wave.bind(6, &fvz); // vel_z
    wave.set_value(7, padded as u32); // count

    // --- Dispatch ---
    let mut pulse = gpu.dispatch(&wave, padded as u32).unwrap();
    pulse.wait().unwrap();

    // --- Read back ---
    let result_vx = gpu.read_field(&fvx).unwrap();
    let result_vy = gpu.read_field(&fvy).unwrap();

    // --- Verify ---
    println!("particle 0: vx={:.6}, vy={:.6}", result_vx[0], result_vy[0]);
    println!("particle 1: vx={:.6}, vy={:.6}", result_vx[1], result_vy[1]);
    println!(
        "non-zero velocities: {}",
        result_vx.iter().filter(|v| v.abs() > 1e-10).count()
    );
}
