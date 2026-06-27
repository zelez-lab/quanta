//! Tensor-core (cooperative-matrix) GEMM tests.
//!
//! `gemm_f32_tc` computes `C ← A·B + C` via Metal `simdgroup_matrix`. On
//! backends without cooperative-matrix support (the software lane) it returns
//! `NotSupported`; on Metal (Apple family 7+) the result is checked against the
//! pure-Rust `reference::gemm` oracle (α=1, β=1).

#![cfg(feature = "gpu")]

use quanta_blas::reference;

fn mat(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 13) as f32 - 6.0)
        .collect()
}

#[test]
fn tc_unsupported_on_software() {
    // The CPU lane reports supports_cooperative_matrix() == false.
    let g = quanta::init_cpu();
    let a = g.field::<f32>(64).unwrap();
    let b = g.field::<f32>(64).unwrap();
    let c = g.field::<f32>(64).unwrap();
    assert!(quanta_blas::gemm_f32_tc(&g, 8, 8, 8, &a, &b, &c).is_err());
}

#[cfg(feature = "gpu-metal")]
fn check_metal(m: usize, n: usize, k: usize) {
    let g = quanta::init().expect("metal device");
    if !g.supports_cooperative_matrix() {
        eprintln!("skip: device lacks cooperative-matrix support");
        return;
    }
    let a = mat(m, k, 1);
    let b = mat(k, n, 2);
    let c0 = mat(m, n, 3);

    let af = g.field::<f32>(m * k).unwrap();
    let bf = g.field::<f32>(k * n).unwrap();
    let cf = g.field::<f32>(m * n).unwrap();
    af.write(&a).unwrap();
    bf.write(&b).unwrap();
    cf.write(&c0).unwrap();
    quanta_blas::gemm_f32_tc(&g, m as u32, n as u32, k as u32, &af, &bf, &cf).unwrap();
    let got = cf.read().unwrap();

    // Oracle: C ← A·B + C  (α = 1, β = 1).
    let mut want = c0.clone();
    reference::gemm(m, n, k, 1.0, &a, &b, 1.0, &mut want);

    for (idx, (&gv, &wv)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (gv - wv).abs() <= 1e-3 * (1.0 + wv.abs()),
            "tc gemm {m}x{n}x{k}: entry {idx}: {gv} vs {wv}"
        );
    }
}

#[cfg(feature = "gpu-metal")]
#[test]
fn tc_single_tile() {
    // One 16×16 output tile (one subgroup, 2×2 fragments), k = one fragment.
    check_metal(16, 16, 8);
}

#[cfg(feature = "gpu-metal")]
#[test]
fn tc_multi_tile_square() {
    check_metal(32, 32, 32);
}

#[cfg(feature = "gpu-metal")]
#[test]
fn tc_k_sweep() {
    // one 16×16 tile; k spans 5 fragments (40 = 8·5).
    check_metal(16, 16, 40);
}

#[cfg(feature = "gpu-metal")]
#[test]
fn tc_rectangular() {
    check_metal(48, 32, 16);
}

#[cfg(feature = "gpu-metal")]
#[test]
fn tc_larger() {
    check_metal(128, 128, 128);
}
