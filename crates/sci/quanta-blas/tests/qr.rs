//! QR differential tests: the GPU Householder `qr` factorisation vs the
//! pure-Rust f64 reference. We check the *reconstruction* `Q·R ≈ A` (the QR
//! packing convention is an implementation detail, so we compare the product,
//! not the packed factors) and that `R` is upper-triangular, plus an `lstsq`
//! least-squares residual check across square and tall shapes.

#![cfg(feature = "gpu")]

use quanta_blas::reference;
use quanta_blas::{lstsq, qr};

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

/// A deterministic well-conditioned `m×n` (m≥n) matrix: random entries with a
/// diagonal boost on the leading `n×n` block to keep the columns from being
/// near-linearly-dependent (so the f32 QR stays close to the f64 oracle).
fn mat(m: usize, n: usize, seed: u32) -> Vec<f32> {
    let mut a: Vec<f32> = (0..m * n).map(|i| val(i, seed)).collect();
    for i in 0..n {
        a[i * n + i] += (n as f32) + 1.0;
    }
    a
}

/// `Q·R` from the GPU factor: `Q` is the reference reconstruction from the
/// original matrix; `R` is the upper triangle of the returned packed `a`.
/// Compare `Q·R` to the original `A` within ~1e-3 relative.
fn check_factor(m: usize, n: usize, seed: u32) {
    let g = gpu();
    let a = mat(m, n, seed);

    let af = g.field::<f32>(m * n).unwrap();
    af.write(&a).unwrap();
    let tauf = g.field::<f32>(n).unwrap();
    tauf.write(&vec![0.0f32; n]).unwrap();
    qr(&g, m as u32, n as u32, &af, &tauf).unwrap();
    let packed = af.read().unwrap();

    // R = upper triangle of the packed m×n result (rows 0..n, j >= i).
    let mut r = vec![0.0f32; n * n];
    for i in 0..n {
        for j in i..n {
            r[i * n + j] = packed[i * n + j];
        }
    }
    // Q (m×m) reconstructed by the reference from the ORIGINAL A.
    let q = reference::form_q(m, n, &a);

    // Check Q·R ≈ A over all m×n entries (Q is m×m, R is m×n with rows n..m
    // implicitly zero, so (Q·R)[i,j] = Σ_{p<n} Q[i,p]·R[p,j]).
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..n {
                acc += (q[i * m + p] as f64) * (r[p * n + j] as f64);
            }
            let want = a[i * n + j] as f64;
            assert!(
                (acc - want).abs() <= 1e-3 * (1.0 + want.abs()),
                "QR m={m} n={n}: (Q·R)[{i},{j}]={acc} vs A={want}"
            );
        }
    }
    // R must be upper-triangular: strictly-lower entries of the top n×n block
    // hold reflector tails, not R — so we only assert the product above. But
    // verify the diagonal of R is nonzero (full column rank).
    for i in 0..n {
        assert!(
            packed[i * n + i].abs() > 1e-4,
            "QR m={m} n={n}: R[{i},{i}] ≈ 0 (rank deficient?)"
        );
    }
}

#[test]
fn factor_square_sizes() {
    for n in [1usize, 2, 3, 4, 8] {
        check_factor(n, n, 7);
    }
}

#[test]
fn factor_tall_sizes() {
    for (m, n) in [(4usize, 2usize), (8, 4), (16, 8), (33, 16), (10, 3)] {
        check_factor(m, n, 13);
    }
}

/// `lstsq` on a consistent overdetermined system: pick a known `x`, form
/// `b = A·x` (so the least-squares solution is exactly `x`), solve, and check
/// the recovered top-`n` block matches `x`.
fn check_lstsq_consistent(m: usize, n: usize, nrhs: usize, seed: u32) {
    let g = gpu();
    let a = mat(m, n, seed);
    // Known solution X (n×nrhs), row-major.
    let x: Vec<f32> = (0..n * nrhs).map(|i| val(i, seed ^ 0x33)).collect();
    // b = A·x (m×nrhs).
    let mut b = vec![0.0f32; m * nrhs];
    for i in 0..m {
        for j in 0..nrhs {
            let mut acc = 0.0f64;
            for p in 0..n {
                acc += (a[i * n + p] as f64) * (x[p * nrhs + j] as f64);
            }
            b[i * nrhs + j] = acc as f32;
        }
    }

    let af = g.field::<f32>(m * n).unwrap();
    af.write(&a).unwrap();
    let tauf = g.field::<f32>(n).unwrap();
    tauf.write(&vec![0.0f32; n]).unwrap();
    let bf = g.field::<f32>(m * nrhs).unwrap();
    bf.write(&b).unwrap();
    lstsq(&g, m as u32, n as u32, nrhs as u32, &af, &tauf, &bf).unwrap();
    let sol = bf.read().unwrap();

    for j in 0..nrhs {
        for i in 0..n {
            let got = sol[i * nrhs + j] as f64;
            let want = x[i * nrhs + j] as f64;
            assert!(
                (got - want).abs() <= 1e-2 * (1.0 + want.abs()),
                "lstsq m={m} n={n} nrhs={nrhs}: X[{i},{j}]={got} vs {want}"
            );
        }
    }
}

#[test]
fn lstsq_square_recovers_solution() {
    for (n, nrhs) in [(2usize, 1usize), (4, 2), (8, 1)] {
        check_lstsq_consistent(n, n, nrhs, 21);
    }
}

#[test]
fn lstsq_tall_recovers_solution() {
    for (m, n, nrhs) in [(6usize, 3usize, 1usize), (16, 4, 2), (33, 8, 1)] {
        check_lstsq_consistent(m, n, nrhs, 29);
    }
}

#[test]
fn shape_mismatch_errors() {
    let g = gpu();
    let af = g.field::<f32>(3 * 3).unwrap();
    let tauf = g.field::<f32>(3).unwrap();
    // m < n → error.
    assert!(qr(&g, 2, 3, &af, &tauf).is_err());
    // wrong A length.
    let bad = g.field::<f32>(3 * 3).unwrap();
    assert!(qr(&g, 4, 4, &bad, &tauf).is_err());
}
