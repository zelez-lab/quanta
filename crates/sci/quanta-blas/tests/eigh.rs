//! Symmetric eigendecomposition differential tests: the GPU Jacobi `eigh`
//! vs the pure-Rust f64 reference, plus intrinsic checks (A·V ≈ V·Λ, V
//! orthogonal) that hold regardless of the reference.

#![cfg(feature = "gpu")]

use quanta_blas::reference;
use quanta_blas::{Uplo, eigh};

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

/// A deterministic symmetric `n×n` row-major matrix: `A = (M + Mᵀ)/2`.
fn sym_mat(n: usize, seed: u32) -> Vec<f32> {
    let m: Vec<f32> = (0..n * n).map(|i| val(i, seed)).collect();
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = 0.5 * (m[i * n + j] + m[j * n + i]);
        }
    }
    a
}

/// Run `eigh` on the GPU; return `(w, v)`.
fn run_eigh(g: &quanta::Gpu, n: usize, uplo: Uplo, a: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let af = g.field::<f32>(n * n).unwrap();
    af.write(a).unwrap();
    let wf = g.field::<f32>(n).unwrap();
    let vf = g.field::<f32>(n * n).unwrap();
    eigh(g, uplo, n as u32, &af, &wf, &vf).unwrap();
    (wf.read().unwrap(), vf.read().unwrap())
}

/// Reconstruct the full symmetric matrix (f64) from the referenced triangle.
fn full_sym(n: usize, uplo: Uplo, a: &[f32]) -> Vec<f64> {
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let src = match uplo {
                Uplo::Lower => {
                    if j <= i {
                        a[i * n + j]
                    } else {
                        a[j * n + i]
                    }
                }
                Uplo::Upper => {
                    if j >= i {
                        a[i * n + j]
                    } else {
                        a[j * n + i]
                    }
                }
            };
            m[i * n + j] = src as f64;
        }
    }
    m
}

/// Check the three intrinsic properties: eigenvalues match the f64 reference
/// (sorted), `A·vⱼ ≈ wⱼ·vⱼ` for each column, and `VᵀV ≈ I`.
fn check(n: usize, uplo: Uplo, seed: u32) {
    let g = gpu();
    let a = sym_mat(n, seed);
    let (w, v) = run_eigh(&g, n, uplo, &a);

    // Reference eigenvalues.
    let mut wref = vec![0.0f32; n];
    let _vref = reference::syev(uplo, n, &a, &mut wref);
    for i in 0..n {
        assert!(
            (w[i] - wref[i]).abs() <= 1e-3 * (1.0 + wref[i].abs()),
            "eig n={n} {uplo:?}: w[{i}]={} vs ref {}",
            w[i],
            wref[i]
        );
    }

    let m = full_sym(n, uplo, &a);

    // A·vⱼ ≈ wⱼ·vⱼ (eigenpair residual).
    for j in 0..n {
        for i in 0..n {
            let mut av = 0.0f64;
            for k in 0..n {
                av += m[i * n + k] * (v[k * n + j] as f64);
            }
            let want = (w[j] as f64) * (v[i * n + j] as f64);
            assert!(
                (av - want).abs() <= 1e-2 * (1.0 + want.abs()),
                "eigpair n={n} {uplo:?}: (A·v)[{i},{j}]={av} vs w·v={want}"
            );
        }
    }

    // VᵀV ≈ I (orthonormal columns).
    for i in 0..n {
        for j in 0..n {
            let mut d = 0.0f64;
            for k in 0..n {
                d += (v[k * n + i] as f64) * (v[k * n + j] as f64);
            }
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (d - want).abs() <= 1e-2,
                "orthonormal n={n} {uplo:?}: (VᵀV)[{i},{j}]={d} vs {want}"
            );
        }
    }
}

#[test]
fn eig_lower_sizes() {
    for n in [2usize, 3, 4, 8, 16, 33] {
        check(n, Uplo::Lower, 7);
    }
}

#[test]
fn eig_upper_sizes() {
    for n in [2usize, 3, 4, 8] {
        check(n, Uplo::Upper, 13);
    }
}

/// Diagonal matrix: eigenvalues are the diagonal, eigenvectors are ±eₖ.
#[test]
fn eig_diagonal() {
    let g = gpu();
    let n = 5usize;
    let mut a = vec![0.0f32; n * n];
    let diag = [3.0f32, -1.0, 2.5, 0.0, 7.0];
    for i in 0..n {
        a[i * n + i] = diag[i];
    }
    let (w, _v) = run_eigh(&g, n, Uplo::Lower, &a);
    let mut sorted = diag;
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    for i in 0..n {
        assert!(
            (w[i] - sorted[i]).abs() < 1e-4,
            "diagonal: w[{i}]={} vs {}",
            w[i],
            sorted[i]
        );
    }
}

/// Known-spectrum construction: build `A = V·Λ·Vᵀ` from a fixed orthonormal
/// V (a 2×2 rotation embedded) and recover Λ.
#[test]
fn eig_known_spectrum() {
    let g = gpu();
    let n = 2usize;
    // A = [[2, 1], [1, 2]] → eigenvalues 1 and 3, eigenvectors (1,∓1)/√2.
    let a = vec![2.0f32, 1.0, 1.0, 2.0];
    let (w, _v) = run_eigh(&g, n, Uplo::Lower, &a);
    assert!((w[0] - 1.0).abs() < 1e-4, "w[0]={}", w[0]);
    assert!((w[1] - 3.0).abs() < 1e-4, "w[1]={}", w[1]);
}

#[test]
fn shape_mismatch_errors() {
    let g = gpu();
    let af = g.field::<f32>(3 * 3).unwrap();
    let wf = g.field::<f32>(3).unwrap();
    let vf = g.field::<f32>(3 * 3).unwrap();
    // n=4 but buffers are 3×3 → error.
    assert!(eigh(&g, Uplo::Lower, 4, &af, &wf, &vf).is_err());
}
