//! Cholesky differential tests: the GPU `potrf` factorisation vs the
//! pure-Rust f64 reference oracle, both uplo forms across a range of sizes,
//! plus a `chol_solve` round-trip (solve then multiply back ≈ B).

#![cfg(feature = "gpu")]

use quanta_blas::reference;
use quanta_blas::{Uplo, chol_solve, cholesky};

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

/// A deterministic symmetric positive-definite `n×n` row-major matrix:
/// `A = MᵀM + n·I`. The `+ n·I` shift keeps the smallest eigenvalue well
/// away from zero (well-conditioned), so the f32 factorisation stays close
/// to the f64 oracle.
fn spd_mat(n: usize, seed: u32) -> Vec<f32> {
    let m: Vec<f64> = (0..n * n).map(|i| val(i, seed) as f64).collect();
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            // (MᵀM)[i,j] = Σ_p M[p,i]·M[p,j]
            let mut s = 0.0f64;
            for p in 0..n {
                s += m[p * n + i] * m[p * n + j];
            }
            if i == j {
                s += n as f64;
            }
            a[i * n + j] = s as f32;
        }
    }
    a
}

/// Compare within the crate's ~1e-3 relative tolerance over the referenced
/// triangle only (the opposite triangle is left untouched by both paths).
fn assert_tri_close(got: &[f32], want: &[f32], n: usize, uplo: Uplo, what: &str) {
    for i in 0..n {
        for j in 0..n {
            let referenced = match uplo {
                Uplo::Lower => j <= i,
                Uplo::Upper => j >= i,
            };
            if !referenced {
                continue;
            }
            let idx = i * n + j;
            let (gv, wv) = (got[idx], want[idx]);
            assert!(
                (gv - wv).abs() <= 1e-3 * (1.0 + wv.abs()),
                "{what}: entry ({i},{j})={idx}: {gv} vs {wv}"
            );
        }
    }
}

fn check_factor(n: usize, uplo: Uplo, seed: u32) {
    let g = gpu();
    let a = spd_mat(n, seed);

    let af = g.field::<f32>(n * n).unwrap();
    af.write(&a).unwrap();
    cholesky(&g, uplo, n as u32, &af).unwrap();
    let got = af.read().unwrap();

    let mut want = a.clone();
    reference::potrf(uplo, n, &mut want);

    assert_tri_close(&got, &want, n, uplo, &format!("potrf n={n} {uplo:?}"));
}

#[test]
fn factor_lower_sizes() {
    for n in [1usize, 2, 3, 4, 8, 16, 33] {
        check_factor(n, Uplo::Lower, 7);
    }
}

#[test]
fn factor_upper_sizes() {
    for n in [1usize, 2, 3, 4, 8, 16, 33] {
        check_factor(n, Uplo::Upper, 13);
    }
}

#[test]
fn factor_identity_is_identity() {
    // Cholesky of I is I (L = I), a clean fixed point.
    let g = gpu();
    let n = 6usize;
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        a[i * n + i] = 1.0;
    }
    let af = g.field::<f32>(n * n).unwrap();
    af.write(&a).unwrap();
    cholesky(&g, Uplo::Lower, n as u32, &af).unwrap();
    let got = af.read().unwrap();
    for i in 0..n {
        assert!(
            (got[i * n + i] - 1.0).abs() < 1e-5,
            "diag {i} = {}",
            got[i * n + i]
        );
    }
}

/// `chol_solve` round-trip: solve `A·X = B`, then check `A·X ≈ B`.
fn check_solve(n: usize, nrhs: usize, uplo: Uplo, seed: u32) {
    let g = gpu();
    let a = spd_mat(n, seed);
    let b: Vec<f32> = (0..n * nrhs).map(|i| val(i, seed ^ 0x55)).collect();

    let af = g.field::<f32>(n * n).unwrap();
    af.write(&a).unwrap();
    let bf = g.field::<f32>(n * nrhs).unwrap();
    bf.write(&b).unwrap();
    chol_solve(&g, uplo, n as u32, nrhs as u32, &af, &bf).unwrap();
    let x = bf.read().unwrap();

    // Reconstruct A·X in f64 from the ORIGINAL A and compare to B.
    for j in 0..nrhs {
        for i in 0..n {
            let mut acc = 0.0f64;
            for p in 0..n {
                acc += (a[i * n + p] as f64) * (x[p * nrhs + j] as f64);
            }
            let want = b[i * nrhs + j] as f64;
            assert!(
                (acc - want).abs() <= 1e-2 * (1.0 + want.abs()),
                "chol_solve n={n} nrhs={nrhs} {uplo:?}: (A·X)[{i},{j}]={acc} vs B={want}"
            );
        }
    }
}

#[test]
fn solve_lower_roundtrip() {
    for (n, nrhs) in [(2usize, 1usize), (4, 2), (8, 3), (16, 1)] {
        check_solve(n, nrhs, Uplo::Lower, 21);
    }
}

#[test]
fn solve_upper_roundtrip() {
    for (n, nrhs) in [(2usize, 1usize), (4, 2), (8, 3), (16, 1)] {
        check_solve(n, nrhs, Uplo::Upper, 29);
    }
}

#[test]
fn shape_mismatch_errors() {
    let g = gpu();
    let af = g.field::<f32>(3 * 3).unwrap();
    // n=4 but A is 3×3 → error.
    assert!(cholesky(&g, Uplo::Lower, 4, &af).is_err());
}
