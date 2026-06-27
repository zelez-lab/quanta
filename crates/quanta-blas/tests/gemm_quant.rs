//! Quantized GEMM/GEMV differential tests: the GPU kernel vs the pure-Rust
//! reference oracle, software lane. int8 (Q8 symmetric) codes ride a
//! `Field<i32>`; C is f32. Per-tensor scales fold into the effective alpha, so
//! the oracle (`reference::gemm_q8_sym`) is the kernel's exact twin.

#![cfg(feature = "gpu")]

use quanta_blas::{GemmQuantType, reference};
use quanta_ir::dtype::{dequantize_sym, quantize_sym};

fn gpu() -> quanta::Gpu {
    quanta::init_cpu()
}

/// Deterministic f32 matrix, quantized to int8 with the given scale. Returns
/// the int8 codes (as i32) and the dequantised f32 values the kernel "sees".
fn mat_q8(rows: usize, cols: usize, scale: f32, seed: u32) -> (Vec<i32>, Vec<f32>) {
    let codes: Vec<i32> = (0..rows * cols)
        .map(|i| {
            let x = (((i as u32).wrapping_mul(2654435761) ^ seed) % 64) as f32 - 32.0;
            quantize_sym(x * scale, scale, 8)
        })
        .collect();
    let deq: Vec<f32> = codes.iter().map(|&q| dequantize_sym(q, scale)).collect();
    (codes, deq)
}

fn upload_i32(g: &quanta::Gpu, codes: &[i32]) -> quanta::Field<i32> {
    let f = g.field::<i32>(codes.len()).unwrap();
    f.write(codes).unwrap();
    f
}

#[allow(clippy::too_many_arguments)]
fn check(m: usize, n: usize, k: usize, alpha: f32, beta: f32, sa: f32, sb: f32) {
    let g = gpu();
    let (a, _) = mat_q8(m, k, sa, 1);
    let (b, _) = mat_q8(k, n, sb, 2);
    let c0: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.25 - 1.0).collect();

    let af = upload_i32(&g, &a);
    let bf = upload_i32(&g, &b);
    let cf = g.field::<f32>(m * n).unwrap();
    cf.write(&c0).unwrap();
    quanta_blas::gemm_quant(
        &g,
        GemmQuantType::Q8Symmetric,
        m as u32,
        n as u32,
        k as u32,
        alpha,
        sa,
        sb,
        &af,
        &bf,
        beta,
        &cf,
    )
    .unwrap();
    let got = cf.read().unwrap();

    let mut want = c0.clone();
    reference::gemm_q8_sym(m, n, k, alpha, sa, sb, &a, &b, beta, &mut want);

    for (idx, (&gv, &wv)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (gv - wv).abs() <= 1e-3 * (1.0 + wv.abs()),
            "q8 gemm {m}x{n}x{k} a={alpha} b={beta}: entry {idx}: {gv} vs {wv}"
        );
    }
}

#[test]
fn q8_square() {
    check(4, 4, 4, 1.0, 0.0, 0.05, 0.05);
}

#[test]
fn q8_rectangular() {
    check(3, 5, 7, 1.0, 0.0, 0.1, 0.02);
}

#[test]
fn q8_alpha_beta() {
    check(6, 4, 5, 2.0, -1.0, 0.05, 0.05);
}

#[test]
fn q8_vector_shapes() {
    check(1, 8, 6, 1.0, 0.0, 0.05, 0.05);
    check(8, 1, 6, 1.0, 0.0, 0.05, 0.05);
}

#[test]
fn q8_larger() {
    check(32, 24, 40, 1.0, 0.5, 0.02, 0.03);
}

#[test]
fn q8_gemv() {
    let g = gpu();
    let (m, n) = (7usize, 5usize);
    let (sa, sx) = (0.05f32, 0.04f32);
    let (a, _) = mat_q8(m, n, sa, 4);
    let (x, _) = mat_q8(n, 1, sx, 5);
    let y0: Vec<f32> = (0..m).map(|i| i as f32 * 0.1).collect();

    let af = upload_i32(&g, &a);
    let xf = upload_i32(&g, &x);
    let yf = g.field::<f32>(m).unwrap();
    yf.write(&y0).unwrap();
    quanta_blas::gemv_quant(
        &g,
        GemmQuantType::Q8Symmetric,
        m as u32,
        n as u32,
        1.0,
        sa,
        sx,
        &af,
        &xf,
        0.0,
        &yf,
    )
    .unwrap();
    let got = yf.read().unwrap();

    let mut want = y0.clone();
    reference::gemm_q8_sym(m, 1, n, 1.0, sa, sx, &a, &x, 0.0, &mut want);
    for (idx, (&gv, &wv)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (gv - wv).abs() <= 1e-3 * (1.0 + wv.abs()),
            "q8 gemv entry {idx}: {gv} vs {wv}"
        );
    }
}
