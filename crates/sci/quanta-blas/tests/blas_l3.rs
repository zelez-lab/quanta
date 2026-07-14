//! Differential tests for the remaining BLAS-3 ops — symm, syr2k, trmm —
//! the GPU kernels vs the pure-Rust f64 reference oracles, across the
//! standard side/uplo/trans/diag variants.

#![cfg(feature = "gpu")]

use quanta_blas::reference;
use quanta_blas::{Diag, Side, Trans, Uplo};

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

fn mat(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 17) as f32 - 8.0)
        .collect()
}

fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (idx, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs();
        let rel = d / (w.abs().max(1.0));
        assert!(
            rel <= tol,
            "{what}: entry {idx} got {g} want {w} (rel {rel} > {tol})"
        );
    }
}

// ─── symm ───────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn check_symm(side: Side, uplo: Uplo, m: usize, n: usize, alpha: f32, beta: f32) {
    let g = gpu();
    let d = match side {
        Side::Left => m,
        Side::Right => n,
    };
    let a = mat(d * d, 3);
    let b = mat(m * n, 5);
    let c0 = mat(m * n, 7);

    let af = g.field::<f32>(d * d).unwrap();
    af.write(&a).unwrap();
    let bf = g.field::<f32>(m * n).unwrap();
    bf.write(&b).unwrap();
    let cf = g.field::<f32>(m * n).unwrap();
    cf.write(&c0).unwrap();
    quanta_blas::symm(
        &g, side, uplo, m as u32, n as u32, alpha, &af, &bf, beta, &cf,
    )
    .unwrap();
    let got = cf.read().unwrap();

    let mut want = c0.clone();
    reference::symm(side, uplo, m, n, alpha, &a, &b, beta, &mut want);
    close(&got, &want, 2e-3, "symm");
}

#[test]
fn symm_all_variants() {
    for &side in &[Side::Left, Side::Right] {
        for &uplo in &[Uplo::Lower, Uplo::Upper] {
            for &(m, n) in &[(1, 1), (2, 3), (4, 4), (8, 5), (16, 9)] {
                check_symm(side, uplo, m, n, 1.0, 0.0);
            }
        }
    }
}

#[test]
fn symm_alpha_beta() {
    check_symm(Side::Left, Uplo::Lower, 8, 6, 1.5, -0.75);
    check_symm(Side::Right, Uplo::Upper, 6, 8, -2.0, 0.5);
}

// ─── syr2k ──────────────────────────────────────────────────────────────

fn check_syr2k(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: f32, beta: f32) {
    let g = gpu();
    let a = mat(n * k, 11);
    let b = mat(n * k, 13);
    let c0 = mat(n * n, 17);

    let af = g.field::<f32>(n * k).unwrap();
    af.write(&a).unwrap();
    let bf = g.field::<f32>(n * k).unwrap();
    bf.write(&b).unwrap();
    let cf = g.field::<f32>(n * n).unwrap();
    cf.write(&c0).unwrap();
    quanta_blas::syr2k(
        &g, uplo, trans, n as u32, k as u32, alpha, &af, &bf, beta, &cf,
    )
    .unwrap();
    let got = cf.read().unwrap();

    let mut want = c0.clone();
    reference::syr2k(uplo, trans, n, k, alpha, &a, &b, beta, &mut want);
    // only compare the stored triangle (the other triangle is never written)
    for i in 0..n {
        for j in 0..n {
            let in_tri = match uplo {
                Uplo::Lower => j <= i,
                Uplo::Upper => j >= i,
            };
            if !in_tri {
                continue;
            }
            let idx = i * n + j;
            close(&got[idx..=idx], &want[idx..=idx], 2e-3, "syr2k");
        }
    }
}

#[test]
fn syr2k_all_variants() {
    for &uplo in &[Uplo::Lower, Uplo::Upper] {
        for &trans in &[Trans::NoTrans, Trans::Trans] {
            for &(n, k) in &[(1, 1), (3, 2), (4, 4), (8, 5), (16, 3)] {
                check_syr2k(uplo, trans, n, k, 1.0, 0.0);
            }
        }
    }
}

#[test]
fn syr2k_alpha_beta() {
    check_syr2k(Uplo::Lower, Trans::NoTrans, 8, 6, 1.25, -0.5);
    check_syr2k(Uplo::Upper, Trans::Trans, 6, 4, -1.0, 2.0);
}

#[test]
fn syr2k_k_zero_is_beta_scale() {
    // k = 0 => C ← β·C on the triangle.
    check_syr2k(Uplo::Lower, Trans::NoTrans, 4, 0, 1.0, 0.5);
}

// ─── trmm ───────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn check_trmm(side: Side, uplo: Uplo, trans: Trans, diag: Diag, m: usize, n: usize, alpha: f32) {
    let g = gpu();
    let na = match side {
        Side::Left => m,
        Side::Right => n,
    };
    let a = mat(na * na, 19);
    let b0 = mat(m * n, 23);

    let af = g.field::<f32>(na * na).unwrap();
    af.write(&a).unwrap();
    let bf = g.field::<f32>(m * n).unwrap();
    bf.write(&b0).unwrap();
    quanta_blas::trmm(
        &g, side, uplo, trans, diag, m as u32, n as u32, alpha, &af, &bf,
    )
    .unwrap();
    let got = bf.read().unwrap();

    let mut want = b0.clone();
    reference::trmm(side, uplo, trans, diag, m, n, alpha, &a, &mut want);
    close(&got, &want, 2e-3, "trmm");
}

#[test]
fn trmm_all_variants() {
    for &side in &[Side::Left, Side::Right] {
        for &uplo in &[Uplo::Lower, Uplo::Upper] {
            for &trans in &[Trans::NoTrans, Trans::Trans] {
                for &diag in &[Diag::NonUnit, Diag::Unit] {
                    for &(m, n) in &[(1, 1), (2, 3), (4, 4), (8, 5)] {
                        check_trmm(side, uplo, trans, diag, m, n, 1.0);
                    }
                }
            }
        }
    }
}

#[test]
fn trmm_alpha() {
    check_trmm(
        Side::Left,
        Uplo::Lower,
        Trans::NoTrans,
        Diag::NonUnit,
        8,
        6,
        1.5,
    );
    check_trmm(
        Side::Right,
        Uplo::Upper,
        Trans::Trans,
        Diag::Unit,
        6,
        8,
        -2.0,
    );
    check_trmm(
        Side::Left,
        Uplo::Upper,
        Trans::NoTrans,
        Diag::NonUnit,
        16,
        4,
        0.5,
    );
}
