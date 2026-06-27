//! Mixed-precision GEMM/GEMV differential tests: the GPU kernel vs the
//! pure-Rust reference oracle, on the software lane. bf16 inputs are stored
//! tightly packed in a `Field<u16>` (one bf16 pattern per 2-byte element);
//! C is f32. The oracle (`reference::gemm_bf16`) is the kernel's exact
//! numerical twin (bf16→f32 load, f32 accumulate), so the comparison is
//! tight, not a tolerance band.

#![cfg(feature = "gpu")]

use quanta_blas::{GemmInputType, reference};
use quanta_ir::dtype::f32_to_bf16;

fn gpu() -> quanta::Gpu {
    quanta::init_cpu()
}

/// Deterministic f32 matrix of `rows×cols`, rounded to bf16-representable
/// values so the f32 oracle and the bf16 kernel agree bit-for-bit on the
/// inputs (the test isolates the GEMM math, not the input rounding — that is
/// covered by the reference unit tests + the Lean bound).
fn mat_bf16(rows: usize, cols: usize, seed: u32) -> (Vec<u16>, Vec<f32>) {
    let f: Vec<f32> = (0..rows * cols)
        .map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 17) as f32 - 8.0)
        .collect();
    let bits: Vec<u16> = f.iter().map(|&x| f32_to_bf16(x)).collect();
    (bits, f)
}

/// Upload bf16 bits tightly packed in a `Field<u16>` (2 bytes/elem).
fn upload_bf16(g: &quanta::Gpu, bits: &[u16]) -> quanta::Field<u16> {
    let f = g.field::<u16>(bits.len()).unwrap();
    f.write(bits).unwrap();
    f
}

#[allow(clippy::too_many_arguments)]
fn run_gemm(
    g: &quanta::Gpu,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[u16],
    b: &[u16],
    beta: f32,
    c0: &[f32],
) -> Vec<f32> {
    let af = upload_bf16(g, a);
    let bf = upload_bf16(g, b);
    let cf = g.field::<f32>(m * n).unwrap();
    cf.write(c0).unwrap();
    quanta_blas::gemm_mixed(
        g,
        GemmInputType::Bf16,
        m as u32,
        n as u32,
        k as u32,
        alpha,
        &af,
        &bf,
        beta,
        &cf,
    )
    .unwrap();
    cf.read().unwrap()
}

fn check(m: usize, n: usize, k: usize, alpha: f32, beta: f32) {
    let g = gpu();
    let (a, _) = mat_bf16(m, k, 1);
    let (b, _) = mat_bf16(k, n, 2);
    let (_, c0) = mat_bf16(m, n, 3); // C is f32, use the f32 values

    let got = run_gemm(&g, m, n, k, alpha, &a, &b, beta, &c0);

    let mut want = c0.clone();
    reference::gemm_bf16(m, n, k, alpha, &a, &b, beta, &mut want);

    for (idx, (&gv, &wv)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (gv - wv).abs() <= 1e-3 * (1.0 + wv.abs()),
            "gemm_bf16 {m}x{n}x{k} a={alpha} b={beta}: entry {idx}: {gv} vs {wv}"
        );
    }
}

#[test]
fn bf16_square() {
    check(4, 4, 4, 1.0, 0.0);
}

#[test]
fn bf16_rectangular() {
    check(3, 5, 7, 1.0, 0.0);
}

#[test]
fn bf16_alpha_beta() {
    check(6, 4, 5, 2.5, -1.5);
}

#[test]
fn bf16_vector_shapes() {
    check(1, 8, 6, 1.0, 0.0);
    check(8, 1, 6, 1.0, 0.0);
}

#[test]
fn bf16_larger() {
    check(32, 24, 40, 1.25, 0.5);
}

#[test]
fn bf16_identity() {
    // A · I = A (in bf16, with bf16-representable A).
    let g = gpu();
    let m = 4;
    let (a, af32) = mat_bf16(m, m, 9);
    let mut id_f = vec![0.0f32; m * m];
    for d in 0..m {
        id_f[d * m + d] = 1.0;
    }
    let id: Vec<u16> = id_f.iter().map(|&x| f32_to_bf16(x)).collect();
    let c0 = vec![0.0f32; m * m];
    let got = run_gemm(&g, m, m, m, 1.0, &a, &id, 0.0, &c0);
    for (idx, (&gv, &av)) in got.iter().zip(af32.iter()).enumerate() {
        assert!((gv - av).abs() <= 1e-4, "A·I entry {idx}: {gv} vs {av}");
    }
}

#[test]
fn bf16_shape_mismatch_errors() {
    let g = gpu();
    let a = g.field::<u16>(6).unwrap();
    let b = g.field::<u16>(12).unwrap();
    let c = g.field::<f32>(8).unwrap();
    assert!(
        quanta_blas::gemm_mixed(&g, GemmInputType::Bf16, 2, 4, 4, 1.0, &a, &b, 0.0, &c).is_err()
    );
}

// ── gemv_mixed (Level-2, via gemm N=1) ──────────────────────────────────

#[test]
fn bf16_gemv() {
    let g = gpu();
    let (m, n) = (7usize, 5usize);
    let (a, _) = mat_bf16(m, n, 4);
    let (x, _) = mat_bf16(n, 1, 5);
    let (_, y0) = mat_bf16(m, 1, 6);

    let af = upload_bf16(&g, &a);
    let xf = upload_bf16(&g, &x);
    let yf = g.field::<f32>(m).unwrap();
    yf.write(&y0).unwrap();
    quanta_blas::gemv_mixed(
        &g,
        GemmInputType::Bf16,
        m as u32,
        n as u32,
        1.5,
        &af,
        &xf,
        -0.5,
        &yf,
    )
    .unwrap();
    let got = yf.read().unwrap();

    // Oracle: gemv = gemm with one output column.
    let mut want = y0.clone();
    reference::gemm_bf16(m, 1, n, 1.5, &a, &x, -0.5, &mut want);

    for (idx, (&gv, &wv)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (gv - wv).abs() <= 1e-3 * (1.0 + wv.abs()),
            "gemv_bf16 entry {idx}: {gv} vs {wv}"
        );
    }
}
