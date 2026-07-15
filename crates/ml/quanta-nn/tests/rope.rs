//! Fused RoPE — differential and property tests: the composed `Var::rope`
//! (matmul-based rotate-half, per-op-proven VJPs) and a host f64 rotation
//! are the references; T9217's isometry is checked empirically per pair.

use quanta_array::Array;
use quanta_autograd::{RopeCache, Tape};
use quanta_nn::rope::rope_var;

fn gpu() -> quanta::Gpu {
    #[cfg(any(feature = "metal", feature = "vulkan"))]
    {
        quanta::init().expect("a GPU device (metal/vulkan feature is on)")
    }
    #[cfg(not(any(feature = "metal", feature = "vulkan")))]
    {
        quanta::init_cpu()
    }
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Host f64 reference: forward rotation + the adjoint applied to `g`.
fn host_rope(x: &[f32], g: &[f32], n: usize, d: usize, base: f64) -> (Vec<f32>, Vec<f32>) {
    let half = d / 2;
    let mut out = vec![0.0f32; n * d];
    let mut dx = vec![0.0f32; n * d];
    for p in 0..n {
        for i in 0..half {
            let theta = (p as f64) * base.powf(-2.0 * (i as f64) / (d as f64));
            let (c, s) = (theta.cos(), theta.sin());
            let (a, b) = (x[p * d + i] as f64, x[p * d + i + half] as f64);
            out[p * d + i] = (a * c - b * s) as f32;
            out[p * d + i + half] = (a * s + b * c) as f32;
            let (ge, go) = (g[p * d + i] as f64, g[p * d + i + half] as f64);
            // T9216: the adjoint is the rotation by −θ.
            dx[p * d + i] = (ge * c + go * s) as f32;
            dx[p * d + i + half] = (go * c - ge * s) as f32;
        }
    }
    (out, dx)
}

#[test]
fn rope_matches_f64_host_and_composed_oracle() {
    let gpu = gpu();
    for &(n, d, seed) in &[(5usize, 8usize, 41u64), (3, 4, 42), (7, 2, 43)] {
        let base = 10000.0f64;
        let cache = RopeCache::<f32>::new(&gpu, n + 2, d, base).unwrap();
        let x = fill(n * d, seed);
        let g = fill(n * d, seed + 7);

        // Fused path through the tape.
        let tape: Tape<f32> = Tape::new();
        let xv = tape.var(Array::from_slice(&gpu, &x, &[n, d]).unwrap());
        let out = rope_var(&tape, &xv, &cache).unwrap();
        let f_out = out.value().to_vec().unwrap();
        let w = tape.var(Array::from_slice(&gpu, &g, &[n, d]).unwrap());
        let loss = out.mul(&w).unwrap().sum().unwrap();
        let f_dx = loss.grad(&xv).unwrap().to_vec().unwrap();

        // Host f64 reference.
        let (h_out, h_dx) = host_rope(&x, &g, n, d, base);
        assert!(
            max_abs_diff(&f_out, &h_out) < 1e-5,
            "rope fwd vs f64 ({n}x{d})"
        );
        assert!(
            max_abs_diff(&f_dx, &h_dx) < 1e-5,
            "rope dx vs f64 ({n}x{d})"
        );

        // Composed oracle.
        let tape2: Tape<f32> = Tape::new();
        let xv2 = tape2.var(Array::from_slice(&gpu, &x, &[n, d]).unwrap());
        let out2 = xv2.rope(&cache).unwrap();
        let c_out = out2.value().to_vec().unwrap();
        let w2 = tape2.var(Array::from_slice(&gpu, &g, &[n, d]).unwrap());
        let loss2 = out2.mul(&w2).unwrap().sum().unwrap();
        let c_dx = loss2.grad(&xv2).unwrap().to_vec().unwrap();
        assert!(
            max_abs_diff(&f_out, &c_out) < 1e-5,
            "rope fwd vs composed ({n}x{d})"
        );
        assert!(
            max_abs_diff(&f_dx, &c_dx) < 1e-5,
            "rope dx vs composed ({n}x{d})"
        );
    }
}

#[test]
fn rope_is_an_isometry_per_pair() {
    // T9217 made empirical: |out pair|² == |in pair|² per frequency pair.
    let gpu = gpu();
    let (n, d) = (6usize, 8usize);
    let cache = RopeCache::<f32>::new(&gpu, n, d, 10000.0).unwrap();
    let x = fill(n * d, 51);

    let tape: Tape<f32> = Tape::new();
    let xv = tape.var(Array::from_slice(&gpu, &x, &[n, d]).unwrap());
    let out = rope_var(&tape, &xv, &cache).unwrap();
    let y = out.value().to_vec().unwrap();

    let half = d / 2;
    for p in 0..n {
        for i in 0..half {
            let nin = x[p * d + i].powi(2) + x[p * d + i + half].powi(2);
            let nout = y[p * d + i].powi(2) + y[p * d + i + half].powi(2);
            assert!(
                (nin - nout).abs() < 1e-5,
                "pair ({p},{i}): |in|²={nin} vs |out|²={nout}"
            );
        }
    }
}

#[test]
fn gradcheck_rope_central_difference() {
    let gpu = gpu();
    let (n, d) = (2usize, 4usize);
    let cache = RopeCache::<f32>::new(&gpu, n, d, 10000.0).unwrap();
    let x = fill(n * d, 61);
    let g = fill(n * d, 62);

    let loss_of = |xs: &[f32]| -> f64 {
        let tape: Tape<f32> = Tape::new();
        let xv = tape.var(Array::from_slice(&gpu, xs, &[n, d]).unwrap());
        let out = rope_var(&tape, &xv, &cache).unwrap();
        out.value()
            .to_vec()
            .unwrap()
            .iter()
            .zip(&g)
            .map(|(&o, &w)| o as f64 * w as f64)
            .sum()
    };

    let tape: Tape<f32> = Tape::new();
    let xv = tape.var(Array::from_slice(&gpu, &x, &[n, d]).unwrap());
    let out = rope_var(&tape, &xv, &cache).unwrap();
    let w = tape.var(Array::from_slice(&gpu, &g, &[n, d]).unwrap());
    let loss = out.mul(&w).unwrap().sum().unwrap();
    let dx = loss.grad(&xv).unwrap().to_vec().unwrap();

    let h = 1e-3f32;
    for idx in 0..n * d {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[idx] += h;
        xm[idx] -= h;
        let num = (loss_of(&xp) - loss_of(&xm)) / (2.0 * h as f64);
        let ana = dx[idx] as f64;
        assert!(
            (num - ana).abs() < 2e-2_f64.max(2e-2 * ana.abs()),
            "gradcheck x[{idx}]: numeric {num} vs analytic {ana}"
        );
    }
}
