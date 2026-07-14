//! trsv/trsm differential tests: the GPU substitution kernels vs the
//! pure-Rust f64 reference oracle, across every side/uplo/trans/diag
//! variant, plus round-trip (solve then multiply back) sanity.

#![cfg(feature = "gpu")]

use quanta_blas::reference;
use quanta_blas::{Diag, Side, Trans, Uplo};

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

/// Deterministic pseudo-random f32 in roughly [-1, 1).
fn val(i: usize, seed: u32) -> f32 {
    ((((i as u32).wrapping_mul(2654435761) ^ seed) % 1000) as f32) / 500.0 - 1.0
}

/// Deterministic well-conditioned `na×na` triangular matrix: small
/// off-diagonal entries in the `uplo` triangle, a dominant diagonal, and
/// **garbage in the opposite triangle** (and on the diagonal for
/// `Diag::Unit`) — the op must never read those slots.
fn tri_mat(na: usize, uplo: Uplo, diag: Diag, seed: u32) -> Vec<f32> {
    let mut a = vec![0.0f32; na * na];
    for i in 0..na {
        for j in 0..na {
            let idx = i * na + j;
            let stored = match uplo {
                Uplo::Lower => j < i,
                Uplo::Upper => j > i,
            };
            a[idx] = if i == j {
                match diag {
                    // Dominant diagonal keeps the solve well-conditioned.
                    Diag::NonUnit => 4.0 + val(idx, seed).abs(),
                    // Unit diag: the stored slot must never be read.
                    Diag::Unit => 1.0e30,
                }
            } else if stored {
                val(idx, seed)
            } else {
                // Opposite triangle: poison — never referenced.
                1.0e30
            };
        }
    }
    a
}

/// Deterministic dense RHS.
fn rhs(len: usize, seed: u32) -> Vec<f32> {
    (0..len).map(|i| val(i, seed) * 4.0).collect()
}

const VARIANTS: [(Uplo, Trans, Diag); 8] = [
    (Uplo::Lower, Trans::NoTrans, Diag::NonUnit),
    (Uplo::Lower, Trans::NoTrans, Diag::Unit),
    (Uplo::Lower, Trans::Trans, Diag::NonUnit),
    (Uplo::Lower, Trans::Trans, Diag::Unit),
    (Uplo::Upper, Trans::NoTrans, Diag::NonUnit),
    (Uplo::Upper, Trans::NoTrans, Diag::Unit),
    (Uplo::Upper, Trans::Trans, Diag::NonUnit),
    (Uplo::Upper, Trans::Trans, Diag::Unit),
];

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    for (idx, (&gv, &wv)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (gv - wv).abs() <= 1e-3 * (1.0 + wv.abs()),
            "{what}: entry {idx}: {gv} vs {wv}"
        );
    }
}

// ── trsv ─────────────────────────────────────────────────────────────

fn check_trsv(n: usize, uplo: Uplo, trans: Trans, diag: Diag) {
    let g = gpu();
    let a = tri_mat(n, uplo, diag, 11);
    let b = rhs(n, 22);

    let af = g.field::<f32>(n * n).unwrap();
    af.write(&a).unwrap();
    let xf = g.field::<f32>(n).unwrap();
    xf.write(&b).unwrap();
    quanta_blas::trsv(&g, uplo, trans, diag, n as u32, &af, &xf).unwrap();
    let got = xf.read().unwrap();

    let mut want = b.clone();
    reference::trsv(uplo, trans, diag, n, &a, &mut want);

    assert_close(
        &got,
        &want,
        &format!("trsv n={n} {uplo:?} {trans:?} {diag:?}"),
    );
}

#[test]
fn trsv_all_variants_small() {
    for &(uplo, trans, diag) in VARIANTS.iter() {
        check_trsv(5, uplo, trans, diag);
    }
}

#[test]
fn trsv_all_variants_medium() {
    for &(uplo, trans, diag) in VARIANTS.iter() {
        check_trsv(33, uplo, trans, diag);
    }
}

#[test]
fn trsv_n_one() {
    check_trsv(1, Uplo::Lower, Trans::NoTrans, Diag::NonUnit);
    check_trsv(1, Uplo::Upper, Trans::Trans, Diag::Unit);
}

#[test]
fn trsv_shape_mismatch_errors() {
    let g = gpu();
    let a = g.field::<f32>(9).unwrap(); // 3×3
    let x = g.field::<f32>(4).unwrap(); // wrong: n=3 needs 3
    assert!(quanta_blas::trsv(&g, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, 3, &a, &x).is_err());
}

// ── trsm ─────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn check_trsm(side: Side, uplo: Uplo, trans: Trans, diag: Diag, m: usize, n: usize, alpha: f32) {
    let g = gpu();
    let na = match side {
        Side::Left => m,
        Side::Right => n,
    };
    let a = tri_mat(na, uplo, diag, 7);
    let b0 = rhs(m * n, 13);

    let af = g.field::<f32>(na * na).unwrap();
    af.write(&a).unwrap();
    let bf = g.field::<f32>(m * n).unwrap();
    bf.write(&b0).unwrap();
    quanta_blas::trsm(
        &g, side, uplo, trans, diag, m as u32, n as u32, alpha, &af, &bf,
    )
    .unwrap();
    let got = bf.read().unwrap();

    let mut want = b0.clone();
    reference::trsm(side, uplo, trans, diag, m, n, alpha, &a, &mut want);

    assert_close(
        &got,
        &want,
        &format!("trsm {side:?} {uplo:?} {trans:?} {diag:?} {m}x{n} a={alpha}"),
    );
}

#[test]
fn trsm_left_all_variants() {
    for &(uplo, trans, diag) in VARIANTS.iter() {
        check_trsm(Side::Left, uplo, trans, diag, 8, 5, 1.0);
    }
}

#[test]
fn trsm_right_all_variants() {
    for &(uplo, trans, diag) in VARIANTS.iter() {
        check_trsm(Side::Right, uplo, trans, diag, 5, 8, 1.0);
    }
}

#[test]
fn trsm_left_lower_larger() {
    // The LU/Cholesky workhorse variant, above one workgroup of lanes
    // is not reachable cheaply (lanes = n), but partial-workgroup + a
    // longer dependency chain is the interesting shape.
    check_trsm(
        Side::Left,
        Uplo::Lower,
        Trans::NoTrans,
        Diag::NonUnit,
        48,
        33,
        1.0,
    );
}

#[test]
fn trsm_right_upper_larger() {
    check_trsm(
        Side::Right,
        Uplo::Upper,
        Trans::NoTrans,
        Diag::NonUnit,
        33,
        48,
        1.0,
    );
}

#[test]
fn trsm_alpha() {
    check_trsm(
        Side::Left,
        Uplo::Lower,
        Trans::NoTrans,
        Diag::NonUnit,
        9,
        6,
        2.5,
    );
    check_trsm(
        Side::Right,
        Uplo::Upper,
        Trans::Trans,
        Diag::Unit,
        6,
        9,
        -0.5,
    );
}

#[test]
fn trsm_single_rhs_column() {
    // n = 1: the trsv shape through the trsm surface.
    check_trsm(
        Side::Left,
        Uplo::Upper,
        Trans::NoTrans,
        Diag::NonUnit,
        17,
        1,
        1.0,
    );
}

#[test]
fn trsm_shape_mismatch_errors() {
    let g = gpu();
    let a = g.field::<f32>(9).unwrap(); // 3×3
    let b = g.field::<f32>(10).unwrap(); // wrong: 3×4 needs 12
    assert!(
        quanta_blas::trsm(
            &g,
            Side::Left,
            Uplo::Lower,
            Trans::NoTrans,
            Diag::NonUnit,
            3,
            4,
            1.0,
            &a,
            &b
        )
        .is_err()
    );
}

// ── round-trip: solve then multiply back ≈ α·B ──────────────────────

/// Densify the referenced triangle of `a` (zeros elsewhere, 1s on the
/// diagonal for Unit) so the reference GEMM can multiply by the *actual*
/// matrix the solve used.
fn densify(na: usize, a: &[f32], uplo: Uplo, diag: Diag) -> Vec<f32> {
    let mut d = vec![0.0f32; na * na];
    for i in 0..na {
        for j in 0..na {
            let stored = match uplo {
                Uplo::Lower => j < i,
                Uplo::Upper => j > i,
            };
            if i == j {
                d[i * na + j] = match diag {
                    Diag::NonUnit => a[i * na + j],
                    Diag::Unit => 1.0,
                };
            } else if stored {
                d[i * na + j] = a[i * na + j];
            }
        }
    }
    d
}

#[test]
fn trsm_round_trip_left() {
    // Solve A·X = α·B on the GPU, then check A·X ≈ α·B with the f64
    // reference GEMM.
    let g = gpu();
    let (m, n) = (16usize, 11usize);
    let alpha = 1.5f32;
    let a = tri_mat(m, Uplo::Lower, Diag::NonUnit, 3);
    let b0 = rhs(m * n, 4);

    let af = g.field::<f32>(m * m).unwrap();
    af.write(&a).unwrap();
    let bf = g.field::<f32>(m * n).unwrap();
    bf.write(&b0).unwrap();
    quanta_blas::trsm(
        &g,
        Side::Left,
        Uplo::Lower,
        Trans::NoTrans,
        Diag::NonUnit,
        m as u32,
        n as u32,
        alpha,
        &af,
        &bf,
    )
    .unwrap();
    let x = bf.read().unwrap();

    let dense = densify(m, &a, Uplo::Lower, Diag::NonUnit);
    let mut back = vec![0.0f32; m * n];
    reference::gemm(m, n, m, 1.0, &dense, &x, 0.0, &mut back);

    for (idx, (&bv, &b0v)) in back.iter().zip(b0.iter()).enumerate() {
        let want = alpha * b0v;
        assert!(
            (bv - want).abs() <= 1e-2 * (1.0 + want.abs()),
            "round-trip entry {idx}: A·X = {bv} vs α·B = {want}"
        );
    }
}

#[test]
fn trsm_round_trip_right() {
    // Solve X·A = α·B, then X·A ≈ α·B.
    let g = gpu();
    let (m, n) = (11usize, 16usize);
    let alpha = 0.75f32;
    let a = tri_mat(n, Uplo::Upper, Diag::NonUnit, 5);
    let b0 = rhs(m * n, 6);

    let af = g.field::<f32>(n * n).unwrap();
    af.write(&a).unwrap();
    let bf = g.field::<f32>(m * n).unwrap();
    bf.write(&b0).unwrap();
    quanta_blas::trsm(
        &g,
        Side::Right,
        Uplo::Upper,
        Trans::NoTrans,
        Diag::NonUnit,
        m as u32,
        n as u32,
        alpha,
        &af,
        &bf,
    )
    .unwrap();
    let x = bf.read().unwrap();

    let dense = densify(n, &a, Uplo::Upper, Diag::NonUnit);
    let mut back = vec![0.0f32; m * n];
    reference::gemm(m, n, n, 1.0, &x, &dense, 0.0, &mut back);

    for (idx, (&bv, &b0v)) in back.iter().zip(b0.iter()).enumerate() {
        let want = alpha * b0v;
        assert!(
            (bv - want).abs() <= 1e-2 * (1.0 + want.abs()),
            "round-trip entry {idx}: X·A = {bv} vs α·B = {want}"
        );
    }
}
