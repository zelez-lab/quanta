//! Pure-Rust Level-1 BLAS reference — the differential-test oracle.
//!
//! Slow but provably-correct (see the Lean forward-error bounds in
//! `specs/verify/lean/Quanta/Blas/Reference.lean`). Every GPU op is
//! validated against these on a fixed corpus. `dot`/`nrm2` accumulate in
//! `f64` so the oracle is tighter than any `f32` summation order — the
//! GPU result is compared to it within a relative tolerance, matching the
//! Higham bound stated against the exact real sum.

/// `scal`: scale each element by `alpha`, in place (`x[i] ← alpha·x[i]`).
pub fn scal(alpha: f32, x: &mut [f32]) {
    for xi in x.iter_mut() {
        *xi *= alpha;
    }
}

/// `axpy`: `y[i] ← alpha·x[i] + y[i]`, in place into `y`. `x` and `y`
/// must have the same length.
pub fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len(), "axpy: length mismatch");
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// `dot`: inner product `Σ x[i]·y[i]`, accumulated in `f64` for a tight
/// reference, returned as `f32`. `x` and `y` must have the same length.
pub fn dot(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len(), "dot: length mismatch");
    let mut acc = 0.0f64;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        acc += (xi as f64) * (yi as f64);
    }
    acc as f32
}

/// `nrm2`: Euclidean norm `sqrt(Σ x[i]²)`, accumulated in `f64`.
pub fn nrm2(x: &[f32]) -> f32 {
    let mut acc = 0.0f64;
    for &xi in x.iter() {
        acc += (xi as f64) * (xi as f64);
    }
    acc.sqrt() as f32
}

/// `gemm`: `C ← alpha·A·B + beta·C`, all row-major.
///
/// `a` is `m×k`, `b` is `k×n`, `c` is `m×n` (read for the `beta·C` term and
/// overwritten with the result). The inner product accumulates in `f64` for
/// a tight reference. Mirrors `quanta_blas::level1`-style contract; this is
/// the differential oracle for `gemm_f32`.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    assert_eq!(a.len(), m * k, "gemm: A must be m×k");
    assert_eq!(b.len(), k * n, "gemm: B must be k×n");
    assert_eq!(c.len(), m * n, "gemm: C must be m×n");
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += (a[row * k + p] as f64) * (b[p * n + col] as f64);
            }
            let cval = c[row * n + col];
            c[row * n + col] = (alpha as f64 * acc + beta as f64 * cval as f64) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scal_basic() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        scal(2.0, &mut x);
        assert_eq!(x, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn axpy_basic() {
        let x = vec![1.0f32, 2.0, 3.0];
        let mut y = vec![10.0f32, 20.0, 30.0];
        axpy(2.0, &x, &mut y);
        assert_eq!(y, vec![12.0, 24.0, 36.0]);
    }

    #[test]
    fn dot_basic() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn nrm2_basic() {
        assert_eq!(nrm2(&[3.0, 4.0]), 5.0);
    }

    #[test]
    fn gemm_basic() {
        // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];
        gemm(2, 2, 2, 1.0, &a, &b, 0.0, &mut c);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn gemm_alpha_beta() {
        // C starts at 1; alpha=2, beta=3 → 2·(A·B) + 3·C
        let a = vec![1.0f32, 0.0, 0.0, 1.0]; // identity
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c = vec![1.0f32; 4];
        gemm(2, 2, 2, 2.0, &a, &b, 3.0, &mut c);
        // 2·B + 3·1 = [13,15,17,19]
        assert_eq!(c, vec![13.0, 15.0, 17.0, 19.0]);
    }
}
