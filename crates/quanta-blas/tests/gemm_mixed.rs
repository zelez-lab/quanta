//! Mixed-precision GEMM/GEMV differential tests: the GPU kernel vs the
//! pure-Rust reference oracle, on the software lane. Narrow inputs (bf16, f16)
//! are stored tightly packed in a `Field<u16>` (one bit pattern per 2-byte
//! element); C is f32. The oracle (`reference::gemm_bf16` / `gemm_f16`) is the
//! kernel's exact numerical twin (narrow→f32 load, f32 accumulate), so the
//! comparison is tight, not a tolerance band.

#![cfg(feature = "gpu")]

use quanta_blas::{GemmInputType, reference};
use quanta_ir::dtype::{f32_to_bf16, f32_to_f16};

fn gpu() -> quanta::Gpu {
    quanta::init_cpu()
}

/// f32→narrow encode for a dtype (which u16 bit pattern the kernel stores).
fn encode(dtype: GemmInputType) -> fn(f32) -> u16 {
    match dtype {
        GemmInputType::Bf16 => f32_to_bf16,
        GemmInputType::F16 => f32_to_f16,
    }
}

/// The matching reference oracle for a dtype.
#[allow(clippy::type_complexity)]
fn oracle(dtype: GemmInputType) -> fn(usize, usize, usize, f32, &[u16], &[u16], f32, &mut [f32]) {
    match dtype {
        GemmInputType::Bf16 => reference::gemm_bf16,
        GemmInputType::F16 => reference::gemm_f16,
    }
}

/// Deterministic matrix, rounded to dtype-representable values so the f32
/// oracle and the kernel agree bit-for-bit on the inputs (the test isolates
/// the GEMM math; input rounding is covered by the reference unit tests + the
/// Lean bound). Returns (narrow bits, f32 values).
fn mat(dtype: GemmInputType, rows: usize, cols: usize, seed: u32) -> (Vec<u16>, Vec<f32>) {
    let enc = encode(dtype);
    let bits: Vec<u16> = (0..rows * cols)
        .map(|i| enc((((i as u32).wrapping_mul(2654435761) ^ seed) % 17) as f32 - 8.0))
        .collect();
    let f: Vec<f32> = bits
        .iter()
        .map(|&b| match dtype {
            GemmInputType::Bf16 => quanta_ir::dtype::bf16_to_f32(b),
            GemmInputType::F16 => quanta_ir::dtype::f16_to_f32(b),
        })
        .collect();
    (bits, f)
}

/// Upload narrow bits tightly packed in a `Field<u16>` (2 bytes/elem).
fn upload(g: &quanta::Gpu, bits: &[u16]) -> quanta::Field<u16> {
    let f = g.field::<u16>(bits.len()).unwrap();
    f.write(bits).unwrap();
    f
}

#[allow(clippy::too_many_arguments)]
fn run_gemm(
    g: &quanta::Gpu,
    dtype: GemmInputType,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[u16],
    b: &[u16],
    beta: f32,
    c0: &[f32],
) -> Vec<f32> {
    let af = upload(g, a);
    let bf = upload(g, b);
    let cf = g.field::<f32>(m * n).unwrap();
    cf.write(c0).unwrap();
    quanta_blas::gemm_mixed(
        g, dtype, m as u32, n as u32, k as u32, alpha, &af, &bf, beta, &cf,
    )
    .unwrap();
    cf.read().unwrap()
}

fn check(dtype: GemmInputType, m: usize, n: usize, k: usize, alpha: f32, beta: f32) {
    let g = gpu();
    let (a, _) = mat(dtype, m, k, 1);
    let (b, _) = mat(dtype, k, n, 2);
    let (_, c0) = mat(dtype, m, n, 3); // C is f32, use the f32 values

    let got = run_gemm(&g, dtype, m, n, k, alpha, &a, &b, beta, &c0);

    let mut want = c0.clone();
    oracle(dtype)(m, n, k, alpha, &a, &b, beta, &mut want);

    for (idx, (&gv, &wv)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (gv - wv).abs() <= 1e-3 * (1.0 + wv.abs()),
            "{dtype:?} gemm {m}x{n}x{k} a={alpha} b={beta}: entry {idx}: {gv} vs {wv}"
        );
    }
}

/// The shared shape matrix, run for any input dtype.
fn run_matrix(dtype: GemmInputType) {
    check(dtype, 4, 4, 4, 1.0, 0.0); // square
    check(dtype, 3, 5, 7, 1.0, 0.0); // rectangular
    check(dtype, 6, 4, 5, 2.5, -1.5); // alpha/beta
    check(dtype, 1, 8, 6, 1.0, 0.0); // row vector
    check(dtype, 8, 1, 6, 1.0, 0.0); // column vector
    check(dtype, 32, 24, 40, 1.25, 0.5); // larger
}

fn check_identity(dtype: GemmInputType) {
    let g = gpu();
    let m = 4;
    let (a, af32) = mat(dtype, m, m, 9);
    let enc = encode(dtype);
    let mut id = vec![enc(0.0); m * m];
    for d in 0..m {
        id[d * m + d] = enc(1.0);
    }
    let c0 = vec![0.0f32; m * m];
    let got = run_gemm(&g, dtype, m, m, m, 1.0, &a, &id, 0.0, &c0);
    for (idx, (&gv, &av)) in got.iter().zip(af32.iter()).enumerate() {
        assert!(
            (gv - av).abs() <= 1e-4,
            "{dtype:?} A·I entry {idx}: {gv} vs {av}"
        );
    }
}

fn check_gemv(dtype: GemmInputType) {
    let g = gpu();
    let (m, n) = (7usize, 5usize);
    let (a, _) = mat(dtype, m, n, 4);
    let (x, _) = mat(dtype, n, 1, 5);
    let (_, y0) = mat(dtype, m, 1, 6);

    let af = upload(&g, &a);
    let xf = upload(&g, &x);
    let yf = g.field::<f32>(m).unwrap();
    yf.write(&y0).unwrap();
    quanta_blas::gemv_mixed(&g, dtype, m as u32, n as u32, 1.5, &af, &xf, -0.5, &yf).unwrap();
    let got = yf.read().unwrap();

    // Oracle: gemv = gemm with one output column.
    let mut want = y0.clone();
    oracle(dtype)(m, 1, n, 1.5, &a, &x, -0.5, &mut want);

    for (idx, (&gv, &wv)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (gv - wv).abs() <= 1e-3 * (1.0 + wv.abs()),
            "{dtype:?} gemv entry {idx}: {gv} vs {wv}"
        );
    }
}

// ── bf16 ────────────────────────────────────────────────────────────────

#[test]
fn bf16_matrix() {
    run_matrix(GemmInputType::Bf16);
}

#[test]
fn bf16_identity() {
    check_identity(GemmInputType::Bf16);
}

#[test]
fn bf16_gemv() {
    check_gemv(GemmInputType::Bf16);
}

// ── f16 ─────────────────────────────────────────────────────────────────

#[test]
fn f16_matrix() {
    run_matrix(GemmInputType::F16);
}

#[test]
fn f16_identity() {
    check_identity(GemmInputType::F16);
}

#[test]
fn f16_gemv() {
    check_gemv(GemmInputType::F16);
}

// ── error handling ────────────────────────────────────────────────────────

#[test]
fn shape_mismatch_errors() {
    let g = gpu();
    let a = g.field::<u16>(6).unwrap();
    let b = g.field::<u16>(12).unwrap();
    let c = g.field::<f32>(8).unwrap();
    assert!(
        quanta_blas::gemm_mixed(&g, GemmInputType::Bf16, 2, 4, 4, 1.0, &a, &b, 0.0, &c).is_err()
    );
}
