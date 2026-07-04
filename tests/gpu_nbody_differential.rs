//! Differential test for the tiled n-body kernel shape (bench_nbody).
//!
//! The kernel is a verbatim copy of `examples/bench_nbody.rs`'s
//! `nbody_soa`: four `#[quanta::shared]` tiles filled from computed
//! global indices, a 4x-unrolled inner loop reading the tiles at
//! `j0 + 0..4`, and a `v[idx] += a * dt` read-modify-write epilogue.
//! rustc CSEs the epilogue's `&mut v[idx]` byte offset (`idx << 2`)
//! into a wasm local that survives the tile loops — the lowering must
//! keep that local's `ScaledIdx` structure symbolic across the loops
//! so the post-loop `BufferPtr + offset` adds still fold into
//! Load/Store pairs (this file is the regression net for that fix).
//!
//! Shared-memory kernels get no auto-generated `_host_oracle` twin
//! (cross-quark semantics), so the oracle here is a hand-written
//! scalar reference that accumulates in the same j-order.

/// Tiled N-body with SoA layout and 4x unrolled inner loop —
/// verbatim from examples/bench_nbody.rs.
#[quanta::kernel(workgroup = [512, 1, 1])]
fn nbody_soa_diff(
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
    let lid = proton_id();

    let my_x = px[idx];
    let my_y = py[idx];
    let my_z = pz[idx];

    let mut ax = 0.0f32;
    let mut ay = 0.0f32;
    let mut az = 0.0f32;
    let eps = 0.01f32;

    let num_tiles = (count + 511u32) / 512u32;
    for t in 0..num_tiles {
        let src = t * 512u32 + lid;
        sx[lid] = px[src];
        sy[lid] = py[src];
        sz[lid] = pz[src];
        sm[lid] = pm[src];
        barrier();

        let iters = 512u32 / 4u32;
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

struct Inputs {
    px: Vec<f32>,
    py: Vec<f32>,
    pz: Vec<f32>,
    pm: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
    vz: Vec<f32>,
}

fn make_inputs(n: usize) -> Inputs {
    let mut px = vec![0.0f32; n];
    let mut py = vec![0.0f32; n];
    let mut pz = vec![0.0f32; n];
    let mut pm = vec![0.0f32; n];
    for i in 0..n {
        let a = i as f32 * 0.01;
        px[i] = a.cos() * (i as f32 * 0.01);
        py[i] = a.sin() * (i as f32 * 0.01);
        pz[i] = (i as f32 * 0.003).sin();
        pm[i] = 1.0 + (i % 3) as f32 * 0.25;
    }
    let vx: Vec<f32> = (0..n).map(|i| i as f32 * 1e-4).collect();
    let vy: Vec<f32> = (0..n).map(|i| i as f32 * -2e-4).collect();
    let vz: Vec<f32> = (0..n).map(|i| 0.5 - i as f32 * 1e-4).collect();
    Inputs {
        px,
        py,
        pz,
        pm,
        vx,
        vy,
        vz,
    }
}

/// Scalar reference: same math, same per-particle accumulation order
/// (j = 0..n ascending, exactly the tile-by-tile unrolled order).
fn reference(inp: &Inputs) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = inp.px.len();
    let eps = 0.01f32;
    let dt = 0.001f32;
    let mut rvx = inp.vx.clone();
    let mut rvy = inp.vy.clone();
    let mut rvz = inp.vz.clone();
    for i in 0..n {
        let (mut ax, mut ay, mut az) = (0.0f32, 0.0f32, 0.0f32);
        for j in 0..n {
            let dx = inp.px[j] - inp.px[i];
            let dy = inp.py[j] - inp.py[i];
            let dz = inp.pz[j] - inp.pz[i];
            let m = inp.pm[j];
            let d = dx * dx + dy * dy + dz * dz + eps;
            let inv = 1.0 / d.sqrt();
            let inv3 = inv * inv * inv;
            ax += dx * inv3 * m;
            ay += dy * inv3 * m;
            az += dz * inv3 * m;
        }
        rvx[i] += ax * dt;
        rvy[i] += ay * dt;
        rvz[i] += az * dt;
    }
    (rvx, rvy, rvz)
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = 1e-3f32.max(w.abs() * 1e-3);
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}] diverged: got {g}, want {w} (tol {tol})"
        );
    }
}

fn run_differential(gpu: &quanta::Gpu) {
    const N: usize = 1024; // two full 512-tiles

    let inp = make_inputs(N);
    let (rvx, rvy, rvz) = reference(&inp);

    let fpx = gpu.field::<f32>(N).unwrap();
    let fpy = gpu.field::<f32>(N).unwrap();
    let fpz = gpu.field::<f32>(N).unwrap();
    let fpm = gpu.field::<f32>(N).unwrap();
    let fvx = gpu.field::<f32>(N).unwrap();
    let fvy = gpu.field::<f32>(N).unwrap();
    let fvz = gpu.field::<f32>(N).unwrap();
    fpx.write(&inp.px).unwrap();
    fpy.write(&inp.py).unwrap();
    fpz.write(&inp.pz).unwrap();
    fpm.write(&inp.pm).unwrap();
    fvx.write(&inp.vx).unwrap();
    fvy.write(&inp.vy).unwrap();
    fvz.write(&inp.vz).unwrap();

    let mut wave = nbody_soa_diff(gpu).expect("create wave");
    wave.bind(0, &fpx);
    wave.bind(1, &fpy);
    wave.bind(2, &fpz);
    wave.bind(3, &fpm);
    wave.bind(4, &fvx);
    wave.bind(5, &fvy);
    wave.bind(6, &fvz);
    wave.set_value(7, N as u32);

    gpu.dispatch(&wave, N as u32).unwrap().wait().unwrap();

    assert_close(&fvx.read().unwrap(), &rvx, "vx");
    assert_close(&fvy.read().unwrap(), &rvy, "vy");
    assert_close(&fvz.read().unwrap(), &rvz, "vz");
}

#[test]
fn nbody_matches_reference_on_gpu() {
    let Ok(gpu) = quanta::init() else {
        eprintln!("no GPU available — skipping nbody differential");
        return;
    };
    run_differential(&gpu);
}

#[cfg(feature = "software")]
#[test]
fn nbody_matches_reference_on_cpu() {
    let gpu = quanta::init_cpu();
    run_differential(&gpu);
}
