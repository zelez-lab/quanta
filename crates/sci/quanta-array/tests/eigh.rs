//! `eigh_symmetric` — cyclic Jacobi over GPU matmuls, checked against a
//! plain-Rust host Jacobi oracle plus known-answer cases (diagonal, 2×2,
//! reconstruction `V·diag(λ)·Vᵀ = A`, orthonormality `VᵀV = I`).

use quanta_array::Array;

/// The device these tests run on: the real GPU under a hardware backend
/// feature (metal / vulkan), else the CPU JIT (portable, no GPU needed).
fn gpu() -> quanta::Gpu {
    #[cfg(any(feature = "metal", feature = "vulkan"))]
    {
        quanta::init().expect("a GPU device")
    }
    #[cfg(not(any(feature = "metal", feature = "vulkan")))]
    {
        quanta::init_cpu()
    }
}

/// Deterministic pseudo-random symmetric matrix: B (Knuth-hash entries in
/// [−3, 3]) symmetrized as (B + Bᵀ)/2.
fn sym_mat(d: usize, seed: u32) -> Vec<f32> {
    let b: Vec<f32> = (0..d * d)
        .map(|i| ((((i as u32).wrapping_mul(2654435761) ^ seed) % 601) as f32 - 300.0) / 100.0)
        .collect();
    let mut a = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            a[i * d + j] = 0.5 * (b[i * d + j] + b[j * d + i]);
        }
    }
    a
}

// ---------------------------------------------------------------------------
// Host oracle: sequential classical cyclic Jacobi over Vec<f32>, independent
// of the GPU path (no matmul composition — rotations applied one by one).
// ---------------------------------------------------------------------------

/// Returns (eigenvalues descending, eigenvectors row-major [d, d] with
/// column j ↔ eigenvalue j), same sign convention as the library
/// (largest-magnitude entry of each eigenvector non-negative).
fn host_eigh(a_in: &[f32], d: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a: Vec<f64> = a_in.iter().map(|&x| x as f64).collect();
    let mut v = vec![0.0f64; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }
    for _sweep in 0..50 {
        let off: f64 = (0..d)
            .flat_map(|i| (0..d).map(move |j| (i, j)))
            .filter(|&(i, j)| i != j)
            .map(|(i, j)| a[i * d + j] * a[i * d + j])
            .sum::<f64>()
            .sqrt();
        if off < 1e-12 {
            break;
        }
        for p in 0..d {
            for q in (p + 1)..d {
                let apq = a[p * d + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * d + q] - a[p * d + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..d {
                    let (akp, akq) = (a[k * d + p], a[k * d + q]);
                    a[k * d + p] = c * akp - s * akq;
                    a[k * d + q] = s * akp + c * akq;
                }
                for k in 0..d {
                    let (apk, aqk) = (a[p * d + k], a[q * d + k]);
                    a[p * d + k] = c * apk - s * aqk;
                    a[q * d + k] = s * apk + c * aqk;
                }
                a[p * d + q] = 0.0;
                a[q * d + p] = 0.0;
                for k in 0..d {
                    let (vkp, vkq) = (v[k * d + p], v[k * d + q]);
                    v[k * d + p] = c * vkp - s * vkq;
                    v[k * d + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    // Sort descending + sign convention (mirrors the library contract).
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&i, &j| a[j * d + j].total_cmp(&a[i * d + i]));
    let mut evals = vec![0.0f32; d];
    let mut evecs = vec![0.0f32; d * d];
    for (nj, &oj) in order.iter().enumerate() {
        evals[nj] = a[oj * d + oj] as f32;
        let mut arg = 0usize;
        let mut best = -1.0f64;
        for i in 0..d {
            if v[i * d + oj].abs() > best {
                best = v[i * d + oj].abs();
                arg = i;
            }
        }
        let sgn = if v[arg * d + oj] < 0.0 { -1.0 } else { 1.0 };
        for i in 0..d {
            evecs[i * d + nj] = (sgn * v[i * d + oj]) as f32;
        }
    }
    (evals, evecs)
}

// ---------------------------------------------------------------------------
// Shared checks (run on the software lane by default, on Metal when the
// `metal` feature test below calls them with a hardware device).
// ---------------------------------------------------------------------------

fn check_against_oracle(g: &quanta::Gpu, d: usize, seed: u32) {
    let a = sym_mat(d, seed);
    let arr = Array::from_slice(g, &a, &[d, d]).unwrap();
    let (evals, evecs) = arr.eigh_symmetric().unwrap();
    assert_eq!(evals.shape(), &[d]);
    assert_eq!(evecs.shape(), &[d, d]);
    let ev = evals.to_vec().unwrap();
    let vv = evecs.to_vec().unwrap();
    let (hev, hv) = host_eigh(&a, d);

    let scale: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    // Eigenvalues match the oracle (sorted descending on both sides).
    for j in 0..d {
        assert!(
            (ev[j] - hev[j]).abs() <= 1e-3 * (1.0 + scale),
            "d={d} seed={seed}: eigenvalue {j}: gpu {} vs host {}",
            ev[j],
            hev[j]
        );
        if j + 1 < d {
            assert!(ev[j] >= ev[j + 1], "eigenvalues not descending at {j}");
        }
    }
    // Eigenvectors match up to sign: |⟨v_gpu, v_host⟩| ≈ 1 per column.
    for j in 0..d {
        let dot: f32 = (0..d).map(|i| vv[i * d + j] * hv[i * d + j]).sum();
        assert!(
            dot.abs() > 0.999,
            "d={d} seed={seed}: eigenvector {j}: |dot| = {} (≠ 1)",
            dot.abs()
        );
    }
    // Sign convention holds on the GPU output itself.
    for j in 0..d {
        let mut arg = 0usize;
        let mut best = -1.0f32;
        for i in 0..d {
            if vv[i * d + j].abs() > best {
                best = vv[i * d + j].abs();
                arg = i;
            }
        }
        assert!(
            vv[arg * d + j] >= 0.0,
            "column {j}: largest-magnitude entry is negative"
        );
    }
}

fn check_reconstruction_and_orthonormality(g: &quanta::Gpu, d: usize, seed: u32) {
    let a = sym_mat(d, seed);
    let arr = Array::from_slice(g, &a, &[d, d]).unwrap();
    let (evals, evecs) = arr.eigh_symmetric().unwrap();
    let ev = evals.to_vec().unwrap();
    let vv = evecs.to_vec().unwrap();
    let scale: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();

    // V·diag(λ)·Vᵀ == A.
    for i in 0..d {
        for k in 0..d {
            let recon: f32 = (0..d).map(|j| ev[j] * vv[i * d + j] * vv[k * d + j]).sum();
            assert!(
                (recon - a[i * d + k]).abs() <= 1e-3 * (1.0 + scale),
                "reconstruction ({i},{k}): {recon} vs {}",
                a[i * d + k]
            );
        }
    }
    // VᵀV == I.
    for j0 in 0..d {
        for j1 in 0..d {
            let dot: f32 = (0..d).map(|i| vv[i * d + j0] * vv[i * d + j1]).sum();
            let want = if j0 == j1 { 1.0 } else { 0.0 };
            assert!(
                (dot - want).abs() <= 1e-3,
                "VᵀV ({j0},{j1}): {dot} vs {want}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Software-lane tests
// ---------------------------------------------------------------------------

#[test]
fn diagonal_matrix_known_answer() {
    let g = gpu();
    // diag(3, 1, 2) → eigenvalues [3, 2, 1], eigenvectors = axis vectors.
    let a = Array::from_slice(
        &g,
        &[3.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0],
        &[3, 3],
    )
    .unwrap();
    let (evals, evecs) = a.eigh_symmetric().unwrap();
    let ev = evals.to_vec().unwrap();
    let vv = evecs.to_vec().unwrap();
    assert!((ev[0] - 3.0).abs() < 1e-5 && (ev[1] - 2.0).abs() < 1e-5 && (ev[2] - 1.0).abs() < 1e-5);
    // Columns: e0 (λ=3), e2 (λ=2), e1 (λ=1).
    let want = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    for (i, (&got, &w)) in vv.iter().zip(want.iter()).enumerate() {
        assert!((got - w).abs() < 1e-5, "evec entry {i}: {got} vs {w}");
    }
}

#[test]
fn two_by_two_hand_computed() {
    let g = gpu();
    // [[2, 1], [1, 2]] → λ = 3 with v = [1, 1]/√2, λ = 1 with v = ±[1, −1]/√2.
    let a = Array::from_slice(&g, &[2.0f32, 1.0, 1.0, 2.0], &[2, 2]).unwrap();
    let (evals, evecs) = a.eigh_symmetric().unwrap();
    let ev = evals.to_vec().unwrap();
    let vv = evecs.to_vec().unwrap();
    assert!((ev[0] - 3.0).abs() < 1e-5, "λ0 = {}", ev[0]);
    assert!((ev[1] - 1.0).abs() < 1e-5, "λ1 = {}", ev[1]);
    let s = std::f32::consts::FRAC_1_SQRT_2;
    // Column 0 = [s, s] (sign convention makes both entries positive).
    assert!((vv[0] - s).abs() < 1e-4 && (vv[2] - s).abs() < 1e-4);
    // Column 1 = ±[s, −s]: entries opposite in sign, magnitude 1/√2.
    assert!((vv[1].abs() - s).abs() < 1e-4 && (vv[3].abs() - s).abs() < 1e-4);
    assert!(vv[1] * vv[3] < 0.0, "λ=1 eigenvector entries must oppose");
}

#[test]
fn one_by_one_trivial() {
    let g = gpu();
    let a = Array::from_slice(&g, &[-4.0f32], &[1, 1]).unwrap();
    let (evals, evecs) = a.eigh_symmetric().unwrap();
    assert_eq!(evals.to_vec().unwrap(), vec![-4.0]);
    assert_eq!(evecs.to_vec().unwrap(), vec![1.0]);
}

#[test]
fn matches_host_oracle() {
    let g = gpu();
    check_against_oracle(&g, 4, 7);
    check_against_oracle(&g, 6, 99);
    check_against_oracle(&g, 8, 12345);
}

#[test]
fn reconstruction_and_orthonormality() {
    let g = gpu();
    check_reconstruction_and_orthonormality(&g, 8, 4242);
}

#[test]
fn nearly_symmetric_input_is_symmetrized() {
    let g = gpu();
    // A GEMM-computed covariance can be off-symmetric by round-off; the
    // solver must symmetrize rather than diverge.
    let a = Array::from_slice(&g, &[2.0f32, 1.0 + 1e-6, 1.0 - 1e-6, 2.0], &[2, 2]).unwrap();
    let (evals, _) = a.eigh_symmetric().unwrap();
    let ev = evals.to_vec().unwrap();
    assert!((ev[0] - 3.0).abs() < 1e-4 && (ev[1] - 1.0).abs() < 1e-4);
}

#[test]
fn rejects_non_square() {
    let g = gpu();
    let a = Array::<f32>::zeros(&g, &[2, 3]).unwrap();
    assert!(a.eigh_symmetric().is_err());
    let b = Array::<f32>::zeros(&g, &[4]).unwrap();
    assert!(b.eigh_symmetric().is_err());
}

// ---------------------------------------------------------------------------
// Metal lane — same checks on the hardware device.
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
#[test]
fn metal_matches_host_oracle() {
    let g = quanta::init().expect("metal device");
    check_against_oracle(&g, 6, 99);
    check_against_oracle(&g, 8, 12345);
    check_reconstruction_and_orthonormality(&g, 8, 4242);
}
