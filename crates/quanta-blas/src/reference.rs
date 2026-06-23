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
}
