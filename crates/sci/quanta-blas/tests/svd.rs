//! Singular value decomposition differential tests: the GPU one-sided Jacobi
//! `svd` vs the pure-Rust f64 reference, plus intrinsic checks
//! (U·diag(s)·Vᵀ ≈ A, UᵀU ≈ I, VᵀV ≈ I) that hold regardless of the reference.

#![cfg(feature = "gpu")]

use quanta_blas::reference;
use quanta_blas::svd;

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

/// Deterministic pseudo-random f32 in roughly [-1, 1).
fn val(i: usize, seed: u32) -> f32 {
    ((((i as u32).wrapping_mul(2654435761) ^ seed) % 1000) as f32) / 500.0 - 1.0
}

/// A deterministic `m×n` row-major matrix.
fn rand_mat(m: usize, n: usize, seed: u32) -> Vec<f32> {
    (0..m * n).map(|i| val(i, seed)).collect()
}

/// Run GPU svd and return (U, s, V).
fn run_svd(g: &quanta::Gpu, m: usize, n: usize, a: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let af = g.field::<f32>(m * n).unwrap();
    af.write(a).unwrap();
    let uf = g.field::<f32>(m * n).unwrap();
    let sf = g.field::<f32>(n).unwrap();
    let vf = g.field::<f32>(n * n).unwrap();
    svd(g, m as u32, n as u32, &af, &uf, &sf, &vf).unwrap();
    (uf.read().unwrap(), sf.read().unwrap(), vf.read().unwrap())
}

/// Reconstruct U·diag(s)·Vᵀ (m×n) and compare against A.
fn assert_reconstructs(m: usize, n: usize, a: &[f32], u: &[f32], s: &[f32], v: &[f32], tol: f32) {
    for i in 0..m {
        for j in 0..n {
            // (U·diag(s)·Vᵀ)[i,j] = Σ_k U[i,k]·s[k]·V[j,k]
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += u[i * n + k] * s[k] * v[j * n + k];
            }
            let got = acc;
            let want = a[i * n + j];
            assert!(
                (got - want).abs() <= tol,
                "reconstruction[{i},{j}]: got {got}, want {want} (m={m}, n={n})"
            );
        }
    }
}

/// Check that the `n` columns of an `r×n` row-major matrix are orthonormal
/// (columnsᵀ·columns ≈ I).
fn assert_orthonormal_cols(r: usize, n: usize, mat: &[f32], tol: f32) {
    for p in 0..n {
        for q in 0..n {
            let mut dot = 0.0f32;
            for i in 0..r {
                dot += mat[i * n + p] * mat[i * n + q];
            }
            let want = if p == q { 1.0 } else { 0.0 };
            assert!(
                (dot - want).abs() <= tol,
                "orthonormality[{p},{q}]: got {dot}, want {want} (r={r}, n={n})"
            );
        }
    }
}

fn check(g: &quanta::Gpu, m: usize, n: usize, seed: u32) {
    let a = rand_mat(m, n, seed);
    let (u, s, v) = run_svd(g, m, n, &a);
    let (_ru, rs, _rv) = reference::gesvd(m, n, &a);

    // Singular values match the reference (descending), relative tolerance.
    for k in 0..n {
        let denom = rs[k].abs().max(1e-3);
        assert!(
            (s[k] - rs[k]).abs() / denom <= 1e-3,
            "sigma[{k}]: got {}, want {} (m={m}, n={n})",
            s[k],
            rs[k]
        );
    }
    // Singular values are descending and non-negative.
    for k in 0..n {
        assert!(s[k] >= -1e-4, "sigma[{k}] negative: {}", s[k]);
        if k > 0 {
            assert!(s[k] <= s[k - 1] + 1e-4, "sigma not descending at {k}");
        }
    }
    // Intrinsic checks.
    assert_reconstructs(m, n, &a, &u, &s, &v, 1e-2);
    assert_orthonormal_cols(m, n, &u, 1e-2); // U columns orthonormal
    assert_orthonormal_cols(n, n, &v, 1e-2); // V orthonormal
}

#[test]
fn svd_square_small() {
    let g = gpu();
    for n in [2usize, 3, 4] {
        check(&g, n, n, 0x51 + n as u32);
    }
}

#[test]
fn svd_square_medium() {
    let g = gpu();
    check(&g, 8, 8, 0xA1);
    check(&g, 16, 16, 0xB2);
}

#[test]
fn svd_tall() {
    let g = gpu();
    check(&g, 8, 4, 0xC3);
    check(&g, 16, 8, 0xD4);
    check(&g, 33, 16, 0xE5);
}

#[test]
fn svd_known_spectrum() {
    // Build A = U0 · diag(σ0) · V0ᵀ with a known, well-separated spectrum
    // and verify the recovered singular values (n×n, U0 = V0 = I gives a
    // diagonal A, the cleanest known case).
    let g = gpu();
    let n = 4;
    let sigma0 = [5.0f32, 3.0, 2.0, 0.5];
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        a[i * n + i] = sigma0[i];
    }
    let (_u, s, _v) = run_svd(&g, n, n, &a);
    for k in 0..n {
        assert!(
            (s[k] - sigma0[k]).abs() <= 1e-3,
            "known sigma[{k}]: got {}, want {}",
            s[k],
            sigma0[k]
        );
    }
}

#[test]
fn svd_single_column() {
    let g = gpu();
    // m×1: σ = ‖a‖, U = a/σ, V = [1].
    let a = vec![3.0f32, 4.0, 0.0]; // norm 5
    let (u, s, v) = run_svd(&g, 3, 1, &a);
    assert!((s[0] - 5.0).abs() <= 1e-4, "sigma: {}", s[0]);
    assert!((u[0] - 0.6).abs() <= 1e-4 && (u[1] - 0.8).abs() <= 1e-4);
    assert!((v[0] - 1.0).abs() <= 1e-4);
}

#[test]
fn svd_rejects_wide() {
    let g = gpu();
    // m < n must error (NotSupported).
    let af = g.field::<f32>(2 * 3).unwrap();
    af.write(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let uf = g.field::<f32>(2 * 3).unwrap();
    let sf = g.field::<f32>(3).unwrap();
    let vf = g.field::<f32>(3 * 3).unwrap();
    assert!(svd(&g, 2, 3, &af, &uf, &sf, &vf).is_err());
}
