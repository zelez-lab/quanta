//! API Design Dream — final version
//!
//! Same N-body gravity as api_design_playground.rs.
//! User defines the struct. Framework does the rest.

/// User-defined data layout. The derive macro generates GPU bindings.
/// Vec<T> → GPU buffer. Scalars → push constants. Names match kernel params.
#[derive(quanta::Fields)]
struct Particles {
    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    pos_z: Vec<f32>,
    mass: Vec<f32>,
    vel_x: Vec<f32>,
    vel_y: Vec<f32>,
    vel_z: Vec<f32>,
    count: u32,
}

/// The kernel. Takes a reference to the Fields struct.
/// Each struct field is accessible by name inside the kernel body.
#[quanta::kernel(workgroup = [256, 1, 1])]
fn gravity(p: &Particles) {
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

    let my_x = p.pos_x[idx];
    let my_y = p.pos_y[idx];
    let my_z = p.pos_z[idx];

    let mut ax = 0.0f32;
    let mut ay = 0.0f32;
    let mut az = 0.0f32;

    let num_tiles = (p.count + 255u32) / 256u32;
    for t in 0..num_tiles {
        let src = t * 256u32 + lid;
        sx[lid] = p.pos_x[src];
        sy[lid] = p.pos_y[src];
        sz[lid] = p.pos_z[src];
        sm[lid] = p.mass[src];
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

    p.vel_x[idx] = p.vel_x[idx] + ax * 0.001f32;
    p.vel_y[idx] = p.vel_y[idx] + ay * 0.001f32;
    p.vel_z[idx] = p.vel_z[idx] + az * 0.001f32;
}

fn main() -> Result<(), quanta::QuantaError> {
    let device = quanta::init()?;
    println!("GPU: {}\n", device.name());

    let n = 1024usize;
    let pad = ((n + 255) / 256) * 256;

    // Build particle data
    let mut px = vec![0.0f32; pad];
    let mut py = vec![0.0f32; pad];
    let pz = vec![0.0f32; pad];
    let mut pm = vec![0.0f32; pad];
    for i in 0..n {
        let angle = i as f32 * 0.01;
        px[i] = angle.cos() * (i as f32 * 0.01);
        py[i] = angle.sin() * (i as f32 * 0.01);
        pm[i] = 1.0;
    }

    // One struct, all data
    let mut particles = Particles {
        pos_x: px,
        pos_y: py,
        pos_z: pz,
        mass: pm,
        vel_x: vec![0.0; pad],
        vel_y: vec![0.0; pad],
        vel_z: vec![0.0; pad],
        count: pad as u32,
    };

    // One line: dispatch and wait
    gravity(&device, &mut particles, pad as u32)?.wait()?;

    // Read back — it's just a Vec
    println!(
        "particle 0: vx={:.6}, vy={:.6}",
        particles.vel_x[0], particles.vel_y[0]
    );
    println!(
        "particle 1: vx={:.6}, vy={:.6}",
        particles.vel_x[1], particles.vel_y[1]
    );
    println!(
        "non-zero velocities: {}",
        particles.vel_x.iter().filter(|v| v.abs() > 1e-10).count()
    );

    Ok(())
}
