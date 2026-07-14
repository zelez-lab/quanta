//! GEMV differential tests: the GPU kernel vs the pure-Rust reference
//! oracle, on the software lane.

#![cfg(feature = "gpu")]

use quanta_blas::reference;

/// The device these tests run on: the real GPU under a hardware backend
/// feature (gpu-metal / gpu-vulkan), else the CPU JIT (portable, no GPU needed).
fn gpu() -> quanta::Gpu {
    #[cfg(any(feature = "gpu-metal", feature = "gpu-vulkan"))]
    {
        quanta::init().expect("a GPU device")
    }
    #[cfg(not(any(feature = "gpu-metal", feature = "gpu-vulkan")))]
    {
        quanta::init_cpu()
    }
}

/// Deterministic f32 vector/matrix of `len` entries.
fn vals(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 17) as f32 - 8.0)
        .collect()
}

/// Run the GPU gemv and return y.
#[allow(clippy::too_many_arguments)]
fn run_gemv(
    g: &quanta::Gpu,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    x: &[f32],
    beta: f32,
    y0: &[f32],
) -> Vec<f32> {
    let af = g.field::<f32>(m * n).unwrap();
    let xf = g.field::<f32>(n).unwrap();
    let yf = g.field::<f32>(m).unwrap();
    af.write(a).unwrap();
    xf.write(x).unwrap();
    yf.write(y0).unwrap();
    quanta_blas::gemv(g, m as u32, n as u32, alpha, &af, &xf, beta, &yf).unwrap();
    yf.read().unwrap()
}

fn check(m: usize, n: usize, alpha: f32, beta: f32) {
    let g = gpu();
    let a = vals(m * n, 1);
    let x = vals(n, 2);
    let y0 = vals(m, 3);

    let got = run_gemv(&g, m, n, alpha, &a, &x, beta, &y0);

    let mut want = y0.clone();
    reference::gemv(m, n, alpha, &a, &x, beta, &mut want);

    for (idx, (&gv, &wv)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (gv - wv).abs() <= 1e-3 * (1.0 + wv.abs()),
            "gemv {m}x{n} a={alpha} b={beta}: entry {idx}: {gv} vs {wv}"
        );
    }
}

#[test]
fn gemv_square() {
    check(4, 4, 1.0, 0.0);
}

#[test]
fn gemv_rectangular() {
    // wide (m < n) and tall (m > n)
    check(3, 7, 1.0, 0.0);
    check(7, 3, 1.0, 0.0);
}

#[test]
fn gemv_alpha_beta() {
    check(6, 5, 2.5, -1.5);
}

#[test]
fn gemv_row_vector() {
    // m=1 → single dot product
    check(1, 9, 1.0, 0.0);
}

#[test]
fn gemv_single_column() {
    // n=1 → y[i] = alpha·A[i,0]·x[0] + beta·y[i]
    check(8, 1, 1.0, 0.0);
}

#[test]
fn gemv_larger() {
    // m crosses the workgroup boundary (256) → exercises over-dispatch clamp
    check(300, 48, 1.25, 0.5);
}

#[test]
fn gemv_identity() {
    // I · x = x  (alpha = 1, beta = 0)
    let g = gpu();
    let m = 5;
    let mut id = vec![0.0f32; m * m];
    for d in 0..m {
        id[d * m + d] = 1.0;
    }
    let x = vals(m, 9);
    let y0 = vec![0.0f32; m];
    let got = run_gemv(&g, m, m, 1.0, &id, &x, 0.0, &y0);
    for (idx, (&gv, &xv)) in got.iter().zip(x.iter()).enumerate() {
        assert!((gv - xv).abs() <= 1e-4, "I·x entry {idx}: {gv} vs {xv}");
    }
}

#[test]
fn gemv_ones() {
    // All-ones A·(all-ones x) → every y entry equals n.
    let g = gpu();
    let (m, n) = (40usize, 24);
    let a = vec![1.0f32; m * n];
    let x = vec![1.0f32; n];
    let y0 = vec![0.0f32; m];
    let got = run_gemv(&g, m, n, 1.0, &a, &x, 0.0, &y0);
    assert!(
        got.iter().all(|&v| (v - (n as f32)).abs() < 1e-3),
        "all entries should equal n={n}; y[0]={}",
        got[0]
    );
}

#[test]
fn gemv_shape_mismatch_errors() {
    let g = gpu();
    let a = g.field::<f32>(6).unwrap(); // claims 2x3
    let x = g.field::<f32>(3).unwrap();
    let y = g.field::<f32>(2).unwrap();
    // declare wrong A length: m=2,n=4 → needs 8, have 6
    assert!(quanta_blas::gemv(&g, 2, 4, 1.0, &a, &x, 0.0, &y).is_err());
}
