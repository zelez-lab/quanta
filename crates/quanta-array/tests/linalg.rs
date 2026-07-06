//! Linear-algebra tests for Array<f32> (matmul/dot/norm), software lane.

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

/// Host matmul reference (row-major), f64 accumulate.
fn host_matmul(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += (a[row * k + p] as f64) * (b[p * n + col] as f64);
            }
            c[row * n + col] = acc as f32;
        }
    }
    c
}

fn approx(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-3 * (1.0 + w.abs()),
            "{ctx}: entry {i}: {g} vs {w}"
        );
    }
}

#[test]
fn matmul_square() {
    let g = gpu();
    // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Array::from_slice(&g, &[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    approx(
        &c.to_vec().unwrap(),
        &[19.0, 22.0, 43.0, 50.0],
        "matmul_square",
    );
}

#[test]
fn matmul_rectangular() {
    let g = gpu();
    let (m, k, n) = (3usize, 4usize, 2usize);
    let ah: Vec<f32> = (0..m * k).map(|i| (i % 5) as f32 - 2.0).collect();
    let bh: Vec<f32> = (0..k * n).map(|i| (i % 3) as f32 - 1.0).collect();
    let a = Array::from_slice(&g, &ah, &[m, k]).unwrap();
    let b = Array::from_slice(&g, &bh, &[k, n]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[m, n]);
    approx(
        &c.to_vec().unwrap(),
        &host_matmul(m, n, k, &ah, &bh),
        "matmul_rect",
    );
}

#[test]
fn matmul_identity() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let id = Array::<f32>::eye(&g, 3).unwrap();
    let c = a.matmul(&id).unwrap();
    approx(
        &c.to_vec().unwrap(),
        &a.to_vec().unwrap(),
        "matmul_identity",
    );
}

#[test]
fn matmul_on_transposed_view() {
    let g = gpu();
    // A is 2×3; Aᵀ is 3×2. (Aᵀ)·A is 3×3. Exercises the device-gather of a
    // strided operand before the gemm.
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let at = a.transpose(0, 1).unwrap(); // 3×2 strided
    let c = at.matmul(&a).unwrap();
    assert_eq!(c.shape(), &[3, 3]);
    // reference: gather Aᵀ to contiguous, multiply
    let at_host = at.to_vec().unwrap();
    let a_host = a.to_vec().unwrap();
    approx(
        &c.to_vec().unwrap(),
        &host_matmul(3, 3, 2, &at_host, &a_host),
        "matmul_transposed",
    );
}

#[test]
fn matmul_inner_dim_mismatch_errors() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert!(a.matmul(&b).is_err(), "3 != 2 inner dim");
}

#[test]
fn matmul_rank_error() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0], &[3]).unwrap(); // 1-D
    let b = Array::from_slice(&g, &[1.0f32, 2.0, 3.0], &[3]).unwrap();
    assert!(a.matmul(&b).is_err(), "1-D matmul must error");
}

#[test]
fn dot_vectors() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0], &[3]).unwrap();
    let b = Array::from_slice(&g, &[4.0f32, 5.0, 6.0], &[3]).unwrap();
    let d = a.dot(&b).unwrap(); // 4+10+18 = 32
    assert!((d - 32.0).abs() <= 1e-4, "dot {d}");
}

#[test]
fn dot_rank_and_length_errors() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0], &[3]).unwrap();
    let short = Array::from_slice(&g, &[1.0f32, 2.0], &[2]).unwrap();
    assert!(a.dot(&short).is_err(), "length mismatch");
    let mat = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert!(a.dot(&mat).is_err(), "2-D dot must error");
}

#[test]
fn norm_l2() {
    let g = gpu();
    let a = Array::from_slice(&g, &[3.0f32, 4.0], &[2]).unwrap();
    assert!((a.norm().unwrap() - 5.0).abs() <= 1e-4);
    // shape doesn't matter — flattens
    let m = Array::from_slice(&g, &[1.0f32, 2.0, 2.0, 4.0], &[2, 2]).unwrap();
    assert!((m.norm().unwrap() - 5.0).abs() <= 1e-4); // sqrt(1+4+4+16)=5
}

// ── Factorization-backed solves ─────────────────────────────────────────

/// Host matmul helper reused below (row-major, f64 accumulate) is `host_matmul`.

#[test]
fn solve_square() {
    let g = gpu();
    // A·x = b with A = [[3,1],[1,2]], b = [9,8] → x = [2,3].
    let a = Array::from_slice(&g, &[3.0f32, 1.0, 1.0, 2.0], &[2, 2]).unwrap();
    let b = Array::from_slice(&g, &[9.0f32, 8.0], &[2]).unwrap();
    let x = a.solve(&b).unwrap();
    assert_eq!(x.shape(), &[2, 1]);
    approx(&x.to_vec().unwrap(), &[2.0, 3.0], "solve_square");
    // round-trip: A·x ≈ b
    let bb = a.matmul(&x).unwrap();
    approx(&bb.to_vec().unwrap(), &[9.0, 8.0], "solve_roundtrip");
}

#[test]
fn solve_multi_rhs() {
    let g = gpu();
    let n = 4usize;
    // Diagonally dominant A for conditioning.
    let mut ah = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            ah[i * n + j] = if i == j { (n + 2) as f32 } else { 0.5 };
        }
    }
    let a = Array::from_slice(&g, &ah, &[n, n]).unwrap();
    let nrhs = 3usize;
    let bh: Vec<f32> = (0..n * nrhs).map(|i| (i % 7) as f32 - 3.0).collect();
    let b = Array::from_slice(&g, &bh, &[n, nrhs]).unwrap();
    let x = a.solve(&b).unwrap();
    assert_eq!(x.shape(), &[n, nrhs]);
    // round-trip A·X ≈ B
    let bb = a.matmul(&x).unwrap();
    approx(&bb.to_vec().unwrap(), &bh, "solve_multi_rhs");
}

#[test]
fn lstsq_overdetermined_recovers_solution() {
    let g = gpu();
    // Overdetermined m×n, m>n; consistent system b = A·x with known x.
    let (m, n, nrhs) = (5usize, 3usize, 2usize);
    let ah: Vec<f32> = (0..m * n)
        .map(|i| ((i * 7 + 3) % 11) as f32 - 5.0)
        .collect();
    let a = Array::from_slice(&g, &ah, &[m, n]).unwrap();
    let xh: Vec<f32> = (0..n * nrhs).map(|i| (i % 5) as f32 - 2.0).collect();
    let x_known = Array::from_slice(&g, &xh, &[n, nrhs]).unwrap();
    // b = A·x  (m×nrhs), so the least-squares solution is exactly x.
    let b = a.matmul(&x_known).unwrap();
    let x = a.lstsq(&b).unwrap();
    assert_eq!(x.shape(), &[n, nrhs]);
    approx(&x.to_vec().unwrap(), &xh, "lstsq_recovers");
    // and the residual is minimized: A·x ≈ b
    let bb = a.matmul(&x).unwrap();
    approx(
        &bb.to_vec().unwrap(),
        &b.to_vec().unwrap(),
        "lstsq_residual",
    );
}

#[test]
fn lstsq_square_matches_solve() {
    let g = gpu();
    // Square case: lstsq should agree with solve.
    let a = Array::from_slice(&g, &[3.0f32, 1.0, 1.0, 2.0], &[2, 2]).unwrap();
    let b = Array::from_slice(&g, &[9.0f32, 8.0], &[2, 1]).unwrap();
    let x = a.lstsq(&b).unwrap();
    approx(&x.to_vec().unwrap(), &[2.0, 3.0], "lstsq_square");
}

#[test]
fn inv_roundtrip() {
    let g = gpu();
    let n = 3usize;
    let ah = [4.0f32, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    let a = Array::from_slice(&g, &ah, &[n, n]).unwrap();
    let ainv = a.inv().unwrap();
    assert_eq!(ainv.shape(), &[n, n]);
    // A·A⁻¹ ≈ I
    let prod = a.matmul(&ainv).unwrap();
    let mut ident = vec![0.0f32; n * n];
    for i in 0..n {
        ident[i * n + i] = 1.0;
    }
    approx(&prod.to_vec().unwrap(), &ident, "inv_roundtrip");
}

#[test]
fn cholesky_spd() {
    let g = gpu();
    let n = 3usize;
    // SPD A = MᵀM + n·I, with M arbitrary.
    let mh = [1.0f32, 2.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0, 1.0];
    let m_arr = Array::from_slice(&g, &mh, &[n, n]).unwrap();
    let mt = m_arr.transpose(0, 1).unwrap();
    let mut a = mt.matmul(&m_arr).unwrap().to_vec().unwrap();
    for i in 0..n {
        a[i * n + i] += n as f32;
    }
    let a_arr = Array::from_slice(&g, &a, &[n, n]).unwrap();
    let l = a_arr.cholesky().unwrap();
    assert_eq!(l.shape(), &[n, n]);
    // Upper triangle of L must be zero.
    let lh = l.to_vec().unwrap();
    for i in 0..n {
        for j in (i + 1)..n {
            assert!(lh[i * n + j].abs() <= 1e-5, "L not lower-triangular");
        }
    }
    // L·Lᵀ ≈ A
    let lt = l.transpose(0, 1).unwrap();
    let recon = l.matmul(&lt).unwrap();
    approx(&recon.to_vec().unwrap(), &a, "cholesky_spd");
}

#[test]
fn qr_reconstruction() {
    let g = gpu();
    let (m, n) = (4usize, 3usize);
    let ah: Vec<f32> = (0..m * n)
        .map(|i| ((i * 7 + 1) % 11) as f32 - 5.0)
        .collect();
    let a = Array::from_slice(&g, &ah, &[m, n]).unwrap();
    let (q, r) = a.qr().unwrap();
    assert_eq!(q.shape(), &[m, n]);
    assert_eq!(r.shape(), &[n, n]);
    // Q·R ≈ A
    let recon = q.matmul(&r).unwrap();
    approx(&recon.to_vec().unwrap(), &ah, "qr_reconstruction");
    // QᵀQ ≈ I (orthonormal columns)
    let qt = q.transpose(0, 1).unwrap();
    let qtq = qt.matmul(&q).unwrap();
    let mut ident = vec![0.0f32; n * n];
    for i in 0..n {
        ident[i * n + i] = 1.0;
    }
    approx(&qtq.to_vec().unwrap(), &ident, "qr_orthonormal");
    // R upper-triangular
    let rh = r.to_vec().unwrap();
    for i in 0..n {
        for j in 0..i {
            assert!(rh[i * n + j].abs() <= 1e-4, "R not upper-triangular");
        }
    }
}

#[test]
fn eigh_symmetric_gpu() {
    let g = gpu();
    let n = 3usize;
    // Symmetric A.
    let ah = [2.0f32, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    let a = Array::from_slice(&g, &ah, &[n, n]).unwrap();
    let (w, v) = a.eigh().unwrap();
    assert_eq!(w.shape(), &[n]);
    assert_eq!(v.shape(), &[n, n]);
    // eigenvalues ascending
    let wh = w.to_vec().unwrap();
    for i in 1..n {
        assert!(wh[i] >= wh[i - 1] - 1e-4, "eigenvalues not ascending");
    }
    // A·V ≈ V·diag(w): compare column by column.
    let av = a.matmul(&v).unwrap().to_vec().unwrap();
    let vh = v.to_vec().unwrap();
    let mut vw = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            vw[i * n + j] = vh[i * n + j] * wh[j];
        }
    }
    approx(&av, &vw, "eigh_gpu");
}

#[test]
fn svd_reconstruction() {
    let g = gpu();
    let (m, n) = (4usize, 3usize);
    let ah: Vec<f32> = (0..m * n).map(|i| ((i * 5 + 2) % 9) as f32 - 4.0).collect();
    let a = Array::from_slice(&g, &ah, &[m, n]).unwrap();
    let (u, s, v) = a.svd().unwrap();
    assert_eq!(u.shape(), &[m, n]);
    assert_eq!(s.shape(), &[n]);
    assert_eq!(v.shape(), &[n, n]);
    // singular values descending, non-negative
    let sh = s.to_vec().unwrap();
    for i in 0..n {
        assert!(sh[i] >= -1e-5, "negative singular value");
        if i > 0 {
            assert!(sh[i] <= sh[i - 1] + 1e-4, "singular values not descending");
        }
    }
    // U·diag(s)·Vᵀ ≈ A
    let uh = u.to_vec().unwrap();
    let mut us = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            us[i * n + j] = uh[i * n + j] * sh[j];
        }
    }
    let us_arr = Array::from_slice(&g, &us, &[m, n]).unwrap();
    let vt = v.transpose(0, 1).unwrap();
    let recon = us_arr.matmul(&vt).unwrap();
    approx(&recon.to_vec().unwrap(), &ah, "svd_reconstruction");
}
