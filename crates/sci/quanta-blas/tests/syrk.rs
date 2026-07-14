//! syrk differential tests: the GPU kernel vs the pure-Rust f64 reference
//! oracle, both NoTrans/Trans forms, both triangles, plus the
//! opposite-triangle-untouched contract.

#![cfg(feature = "gpu")]

use quanta_blas::reference;
use quanta_blas::{Trans, Uplo};

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

/// Deterministic f32 matrix.
fn mat(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 17) as f32 - 8.0)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_syrk(
    g: &quanta::Gpu,
    uplo: Uplo,
    trans: Trans,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    beta: f32,
    c0: &[f32],
) -> Vec<f32> {
    let af = g.field::<f32>(n * k).unwrap();
    af.write(a).unwrap();
    let cf = g.field::<f32>(n * n).unwrap();
    cf.write(c0).unwrap();
    quanta_blas::syrk(g, uplo, trans, n as u32, k as u32, alpha, &af, beta, &cf).unwrap();
    cf.read().unwrap()
}

fn check(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: f32, beta: f32) {
    let g = gpu();
    let a = mat(n * k, 1);
    let c0 = mat(n * n, 2);

    let got = run_syrk(&g, uplo, trans, n, k, alpha, &a, beta, &c0);

    let mut want = c0.clone();
    reference::syrk(uplo, trans, n, k, alpha, &a, beta, &mut want);

    for (idx, (&gv, &wv)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (gv - wv).abs() <= 1e-3 * (1.0 + wv.abs()),
            "syrk {uplo:?} {trans:?} n={n} k={k} a={alpha} b={beta}: entry {idx}: {gv} vs {wv}"
        );
    }

    // The opposite triangle must be bit-identical to the input — never
    // written (the reference oracle already leaves it untouched; pin the
    // GPU path to the same contract).
    for i in 0..n {
        for j in 0..n {
            let off_tri = match uplo {
                Uplo::Lower => j > i,
                Uplo::Upper => j < i,
            };
            if off_tri {
                assert_eq!(
                    got[i * n + j],
                    c0[i * n + j],
                    "syrk {uplo:?} {trans:?}: opposite-triangle entry ({i},{j}) was written"
                );
            }
        }
    }
}

#[test]
fn syrk_all_forms_small() {
    for &uplo in &[Uplo::Lower, Uplo::Upper] {
        for &trans in &[Trans::NoTrans, Trans::Trans] {
            check(uplo, trans, 4, 4, 1.0, 0.0);
        }
    }
}

#[test]
fn syrk_all_forms_rectangular() {
    // n ≠ k exercises both stride mappings properly.
    for &uplo in &[Uplo::Lower, Uplo::Upper] {
        for &trans in &[Trans::NoTrans, Trans::Trans] {
            check(uplo, trans, 17, 9, 1.0, 0.0);
            check(uplo, trans, 9, 17, 1.0, 0.0);
        }
    }
}

#[test]
fn syrk_alpha_beta() {
    check(Uplo::Lower, Trans::NoTrans, 12, 7, 2.5, -1.5);
    check(Uplo::Upper, Trans::Trans, 7, 12, -0.5, 0.75);
}

#[test]
fn syrk_larger() {
    check(Uplo::Lower, Trans::NoTrans, 32, 40, 1.25, 0.5);
    check(Uplo::Upper, Trans::NoTrans, 33, 20, 1.0, 1.0);
}

#[test]
fn syrk_n_one() {
    check(Uplo::Lower, Trans::NoTrans, 1, 6, 1.0, 2.0);
}

#[test]
fn syrk_diagonal_is_gram() {
    // C[i,i] = α·‖row i‖² with β = 0 — a quick semantic pin independent
    // of the oracle.
    let g = gpu();
    let (n, k) = (5usize, 8usize);
    let a = mat(n * k, 3);
    let c0 = vec![0.0f32; n * n];
    let got = run_syrk(&g, Uplo::Lower, Trans::NoTrans, n, k, 1.0, &a, 0.0, &c0);
    for i in 0..n {
        let want: f32 = a[i * k..(i + 1) * k].iter().map(|&x| x * x).sum();
        let gv = got[i * n + i];
        assert!(
            (gv - want).abs() <= 1e-3 * (1.0 + want.abs()),
            "diag {i}: {gv} vs {want}"
        );
    }
}

#[test]
fn syrk_shape_mismatch_errors() {
    let g = gpu();
    let a = g.field::<f32>(12).unwrap(); // 3×4
    let c = g.field::<f32>(8).unwrap(); // wrong: needs 9
    assert!(quanta_blas::syrk(&g, Uplo::Lower, Trans::NoTrans, 3, 4, 1.0, &a, 0.0, &c).is_err());
}
