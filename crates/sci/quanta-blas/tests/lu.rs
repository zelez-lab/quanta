//! LU differential tests: the GPU `getrf` factorisation with partial
//! pivoting vs the pure-Rust f64 reference oracle, across a range of sizes,
//! plus `lu_solve` and `lu_inv` round-trips (solve then multiply back ≈ B;
//! A·A⁻¹ ≈ I).

#![cfg(feature = "gpu")]

use quanta_blas::reference;
use quanta_blas::{lu, lu_inv, lu_solve};

/// The device these tests run on: the real GPU under a hardware backend
/// feature (gpu-metal / gpu-vulkan), else the CPU JIT.
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

/// A deterministic well-conditioned `n×n` row-major matrix: a random matrix
/// with a `+ n·I` diagonal shift, keeping it far from singular so the f32
/// factorisation stays close to the f64 oracle.
fn gen_mat(n: usize, seed: u32) -> Vec<f32> {
    let mut a: Vec<f32> = (0..n * n).map(|i| val(i, seed)).collect();
    for i in 0..n {
        a[i * n + i] += n as f32;
    }
    a
}

/// Multiply two row-major `n×n` matrices (f64 accumulate).
fn matmul(n: usize, x: &[f32], y: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for p in 0..n {
                s += x[i * n + p] as f64 * y[p * n + j] as f64;
            }
            out[i * n + j] = s as f32;
        }
    }
    out
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs();
        let rel = d / (w.abs().max(1.0));
        assert!(
            rel <= tol,
            "{what}: entry {idx} got {g} want {w} (rel {rel} > {tol})"
        );
    }
}

/// GPU LU factor must match the f64 reference bit-for-bit (up to f32 rounding)
/// in the packed L\U layout AND in the recorded pivots.
fn check_factor(n: usize, seed: u32) {
    let g = gpu();
    let a = gen_mat(n, seed);

    let af = g.field::<f32>(n * n).unwrap();
    af.write(&a).unwrap();
    let ipivf = g.field::<u32>(n).unwrap();
    lu(&g, n as u32, &af, &ipivf).unwrap();
    let got = af.read().unwrap();
    let got_piv = ipivf.read().unwrap();

    let mut want = a.clone();
    let mut want_piv = vec![0usize; n];
    reference::getrf(n, &mut want, &mut want_piv);

    assert_close(&got, &want, 1e-3, &format!("getrf packed L\\U n={n}"));
    for k in 0..n {
        assert_eq!(
            got_piv[k] as usize, want_piv[k],
            "getrf pivot {k} n={n}: got {} want {}",
            got_piv[k], want_piv[k]
        );
    }
}

/// Reconstruct P·A from L, U (packed) and pivots, and check it equals the
/// permuted original — the exact factorisation identity end to end.
fn check_reconstruct(n: usize, seed: u32) {
    let g = gpu();
    let a = gen_mat(n, seed);
    let af = g.field::<f32>(n * n).unwrap();
    af.write(&a).unwrap();
    let ipivf = g.field::<u32>(n).unwrap();
    lu(&g, n as u32, &af, &ipivf).unwrap();
    let lu_packed = af.read().unwrap();
    let piv = ipivf.read().unwrap();

    // L (unit lower) and U (upper) from the packed factor.
    let mut lmat = vec![0.0f32; n * n];
    let mut umat = vec![0.0f32; n * n];
    for i in 0..n {
        lmat[i * n + i] = 1.0;
        for j in 0..n {
            if j < i {
                lmat[i * n + j] = lu_packed[i * n + j];
            } else {
                umat[i * n + j] = lu_packed[i * n + j];
            }
        }
    }
    let recon = matmul(n, &lmat, &umat); // = L·U

    // P·A: replay the pivot swaps on the original A.
    let mut pa = a.clone();
    for k in 0..n {
        let r = piv[k] as usize;
        if r != k {
            for j in 0..n {
                pa.swap(k * n + j, r * n + j);
            }
        }
    }
    assert_close(&recon, &pa, 2e-3, &format!("L·U = P·A n={n}"));
}

#[test]
fn factor_sizes() {
    for n in [1usize, 2, 3, 4, 8, 16, 33] {
        check_factor(n, 0x51);
    }
}

#[test]
fn factor_reconstructs() {
    for n in [2usize, 3, 5, 8, 16] {
        check_reconstruct(n, 0xC0FFEE);
    }
}

/// `lu_solve` A·X = B round-trip: solve, then A·X ≈ B (using a fresh copy of A
/// since lu_solve overwrites its input with the factors).
fn check_solve(n: usize, nrhs: usize, seed: u32) {
    let g = gpu();
    let a = gen_mat(n, seed);
    let b: Vec<f32> = (0..n * nrhs).map(|i| val(i, seed ^ 0x9e)).collect();

    let af = g.field::<f32>(n * n).unwrap();
    af.write(&a).unwrap();
    let bf = g.field::<f32>(n * nrhs).unwrap();
    bf.write(&b).unwrap();
    let ipivf = g.field::<u32>(n).unwrap();
    lu_solve(&g, n as u32, nrhs as u32, &af, &ipivf, &bf).unwrap();
    let x = bf.read().unwrap();

    // Reconstruct A·X (original A, row-major; X is n×nrhs).
    let mut ax = vec![0.0f32; n * nrhs];
    for i in 0..n {
        for j in 0..nrhs {
            let mut s = 0.0f64;
            for p in 0..n {
                s += a[i * n + p] as f64 * x[p * nrhs + j] as f64;
            }
            ax[i * nrhs + j] = s as f32;
        }
    }
    assert_close(&ax, &b, 1e-2, &format!("lu_solve A·X≈B n={n} nrhs={nrhs}"));
}

#[test]
fn solve_round_trip() {
    for n in [2usize, 3, 4, 8, 16] {
        for nrhs in [1usize, 2, 3] {
            check_solve(n, nrhs, 0x1234);
        }
    }
}

/// `lu_inv`: A·A⁻¹ ≈ I.
fn check_inv(n: usize, seed: u32) {
    let g = gpu();
    let a = gen_mat(n, seed);
    let af = g.field::<f32>(n * n).unwrap();
    af.write(&a).unwrap();
    let ipivf = g.field::<u32>(n).unwrap();
    let invf = g.field::<f32>(n * n).unwrap();
    lu_inv(&g, n as u32, &af, &ipivf, &invf).unwrap();
    let inv = invf.read().unwrap();

    let prod = matmul(n, &a, &inv); // A·A⁻¹
    let mut ident = vec![0.0f32; n * n];
    for i in 0..n {
        ident[i * n + i] = 1.0;
    }
    assert_close(&prod, &ident, 1e-2, &format!("lu_inv A·A⁻¹≈I n={n}"));
}

#[test]
fn inverse_round_trip() {
    for n in [2usize, 3, 4, 8, 16] {
        check_inv(n, 0x77);
    }
}

#[test]
fn shape_mismatch_errors() {
    let g = gpu();
    let af = g.field::<f32>(3 * 3).unwrap();
    let ipivf = g.field::<u32>(2).unwrap(); // wrong length
    assert!(lu(&g, 3, &af, &ipivf).is_err());
}
