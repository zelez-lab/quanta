//! GEMM differential tests: the GPU kernel vs the pure-Rust reference
//! oracle, on the software lane.

#![cfg(feature = "gpu")]

use quanta_blas::reference;

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

/// Deterministic f32 matrix of `rows×cols`.
fn mat(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 17) as f32 - 8.0)
        .collect()
}

/// Run the GPU gemm and return C.
#[allow(clippy::too_many_arguments)]
fn run_gemm(
    g: &quanta::Gpu,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c0: &[f32],
) -> Vec<f32> {
    let af = g.field::<f32>(m * k).unwrap();
    let bf = g.field::<f32>(k * n).unwrap();
    let cf = g.field::<f32>(m * n).unwrap();
    af.write(a).unwrap();
    bf.write(b).unwrap();
    cf.write(c0).unwrap();
    quanta_blas::gemm(g, m as u32, n as u32, k as u32, alpha, &af, &bf, beta, &cf).unwrap();
    cf.read().unwrap()
}

fn check(m: usize, n: usize, k: usize, alpha: f32, beta: f32) {
    let g = gpu();
    let a = mat(m, k, 1);
    let b = mat(k, n, 2);
    let c0 = mat(m, n, 3);

    let got = run_gemm(&g, m, n, k, alpha, &a, &b, beta, &c0);

    let mut want = c0.clone();
    reference::gemm(m, n, k, alpha, &a, &b, beta, &mut want);

    for (idx, (&gv, &wv)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (gv - wv).abs() <= 1e-3 * (1.0 + wv.abs()),
            "gemm {m}x{n}x{k} a={alpha} b={beta}: entry {idx}: {gv} vs {wv}"
        );
    }
}

#[test]
fn gemm_square() {
    check(4, 4, 4, 1.0, 0.0);
}

#[test]
fn gemm_rectangular() {
    check(3, 5, 7, 1.0, 0.0);
}

#[test]
fn gemm_alpha_beta() {
    check(6, 4, 5, 2.5, -1.5);
}

#[test]
fn gemm_vector_shapes() {
    // m=1 (row vector × matrix) and n=1 (matrix × column vector)
    check(1, 8, 6, 1.0, 0.0);
    check(8, 1, 6, 1.0, 0.0);
}

#[test]
fn gemm_k_one() {
    // outer product: A is m×1, B is 1×n
    check(5, 5, 1, 1.0, 0.0);
}

#[test]
fn gemm_larger() {
    check(32, 24, 40, 1.25, 0.5);
}

#[test]
fn gemm_identity() {
    // A · I = A  (beta = 0, alpha = 1)
    let g = gpu();
    let m = 4;
    let a = mat(m, m, 9);
    let mut id = vec![0.0f32; m * m];
    for d in 0..m {
        id[d * m + d] = 1.0;
    }
    let c0 = vec![0.0f32; m * m];
    let got = run_gemm(&g, m, m, m, 1.0, &a, &id, 0.0, &c0);
    for (idx, (&gv, &av)) in got.iter().zip(a.iter()).enumerate() {
        assert!((gv - av).abs() <= 1e-4, "A·I entry {idx}: {gv} vs {av}");
    }
}

#[test]
fn gemm_shape_mismatch_errors() {
    let g = gpu();
    let a = g.field::<f32>(6).unwrap(); // claims 2x3
    let b = g.field::<f32>(12).unwrap(); // 3x4
    let c = g.field::<f32>(8).unwrap(); // 2x4
    // declare wrong A length: m=2,k=4 → needs 8, have 6
    assert!(quanta_blas::gemm(&g, 2, 4, 4, 1.0, &a, &b, 0.0, &c).is_err());
}

#[test]
fn gemm_tiled_exact_32() {
    check(32, 32, 32, 1.0, 0.0);
}

#[test]
fn gemm_tiled_one_block_32k() {
    check(17, 17, 17, 1.0, 0.0);
}

#[test]
fn gemm_tiled_ones() {
    // All-ones A·B over the tiled path → every C entry equals k.
    let g = gpu();
    let (m, n, k) = (32usize, 32, 32);
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let af = g.field::<f32>(m * k).unwrap();
    af.write(&a).unwrap();
    let bf = g.field::<f32>(k * n).unwrap();
    bf.write(&b).unwrap();
    let cf = g.field::<f32>(m * n).unwrap();
    cf.write(&vec![0.0; m * n]).unwrap();
    quanta_blas::gemm(&g, m as u32, n as u32, k as u32, 1.0, &af, &bf, 0.0, &cf).unwrap();
    let got = cf.read().unwrap();
    assert!(
        got.iter().all(|&x| (x - (k as f32)).abs() < 1e-3),
        "all entries should equal k={k}; C[0]={}",
        got[0]
    );
}

#[test]
fn gemm_tiled_n_partial() {
    // m,k multiple of 16; n NOT (24 = 16+8). Isolates N-tail.
    check(32, 24, 32, 1.0, 0.0);
}

#[test]
fn gemm_tiled_k_partial() {
    // m,n multiple of 16; k NOT (40 = 16+16+8). Isolates K-tail.
    check(32, 32, 40, 1.0, 0.0);
}

#[test]
fn gemm_tiled_m_partial() {
    // n,k multiple of 16; m NOT (24). Isolates M-tail.
    check(24, 32, 32, 1.0, 0.0);
}

#[test]
fn gemm_tiled_n24_simple() {
    // smallest N-partial: 16 rows (1 M-tile), 24 cols (2 N-tiles, 2nd partial), 16 k
    check(16, 24, 16, 1.0, 0.0);
}
