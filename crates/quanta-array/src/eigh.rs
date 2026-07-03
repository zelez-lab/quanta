//! Symmetric eigendecomposition — cyclic Jacobi over GPU matmuls.
//!
//! `eigh_symmetric` diagonalizes a symmetric `D×D` matrix with the classical
//! cyclic Jacobi method: each sweep visits every off-diagonal pivot `(p, q)`
//! once and applies a Givens rotation that zeroes it. Jacobi converges
//! quadratically and is numerically robust — the right shape for the small
//! `D×D` matrices PCA produces (`D` = feature count).
//!
//! ## Host/GPU split
//!
//! The rotation *angles* are cheap scalar work; the matrix *updates* are
//! matmuls. Per sweep:
//!
//! 1. Read `A` back to the host (one `to_vec` per sweep) and test the
//!    off-diagonal Frobenius norm for convergence.
//! 2. Simulate the sweep's rotations on the host copy (each angle depends on
//!    the partially-updated matrix), composing them into a single orthogonal
//!    sweep matrix `Q = J₁·J₂·…·Jₘ`.
//! 3. On the GPU: `A ← Qᵀ·(A·Q)` and `V ← V·Q` — three `D×D`
//!    [`Array::matmul`]s through the proven quanta-blas GEMM path.
//!
//! This keeps the device arrays authoritative (the host mirror is re-synced
//! from the GPU each sweep, so accumulated fp drift self-corrects through the
//! next sweep's angles) while dispatching O(sweeps) GPU ops instead of
//! O(D²·sweeps).
//!
//! ## Conventions
//!
//! - Eigenvalues are returned **descending**; `eigenvectors` column `j` (i.e.
//!   `v[:, j]`, numpy-style) pairs with `eigenvalues[j]`.
//! - Sign ambiguity is fixed deterministically: in each eigenvector, the
//!   entry of largest magnitude (first index on ties) is made non-negative.

use crate::array::Array;
use crate::error::ArrayError;

/// Maximum number of cyclic Jacobi sweeps. Quadratic convergence means small
/// matrices settle in ~5-8 sweeps; 30 is a generous safety cap.
const MAX_SWEEPS: usize = 30;

/// Convergence: stop when `off(A) ≤ TOL·‖A‖_F`. Comfortably above the f32
/// GEMM round-off floor, comfortably below any tolerance PCA cares about.
const TOL: f32 = 1e-5;

/// Frobenius norm of the full matrix (f64 accumulate).
fn frob_norm(a: &[f32], d: usize) -> f64 {
    a[..d * d]
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt()
}

/// Frobenius norm of the off-diagonal part (f64 accumulate).
fn off_norm(a: &[f32], d: usize) -> f64 {
    let mut s = 0.0f64;
    for i in 0..d {
        for j in 0..d {
            if i != j {
                let x = a[i * d + j] as f64;
                s += x * x;
            }
        }
    }
    s.sqrt()
}

/// Average `a` with its transpose in place, so tiny GEMM asymmetries never
/// leak into the rotation angles.
fn symmetrize(a: &mut [f32], d: usize) {
    for i in 0..d {
        for j in (i + 1)..d {
            let m = 0.5 * (a[i * d + j] as f64 + a[j * d + i] as f64);
            a[i * d + j] = m as f32;
            a[j * d + i] = m as f32;
        }
    }
}

/// One cyclic Jacobi sweep, simulated on the host mirror `a` (f64 rotation
/// math, f32 storage). The rotations are composed into `q` (which must come
/// in as the identity), so the caller can replay the whole sweep on the GPU
/// as `A ← Qᵀ·A·Q`, `V ← V·Q`.
fn jacobi_sweep_host(a: &mut [f32], q: &mut [f32], d: usize) {
    for p in 0..d {
        for r in (p + 1)..d {
            let apq = a[p * d + r] as f64;
            if apq == 0.0 {
                continue;
            }
            let app = a[p * d + p] as f64;
            let aqq = a[r * d + r] as f64;
            // Stable rotation (Numerical Recipes): t is the smaller-magnitude
            // root of t² + 2·t·θ − 1 = 0 with θ = (a_qq − a_pp)/(2·a_pq).
            let theta = (aqq - app) / (2.0 * apq);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;

            // A ← JᵀAJ where J is identity except J[p][p]=J[q][q]=c,
            // J[p][q]=s, J[q][p]=−s. Update columns p,r then rows p,r.
            for k in 0..d {
                let akp = a[k * d + p] as f64;
                let akq = a[k * d + r] as f64;
                a[k * d + p] = (c * akp - s * akq) as f32;
                a[k * d + r] = (s * akp + c * akq) as f32;
            }
            for k in 0..d {
                let apk = a[p * d + k] as f64;
                let aqk = a[r * d + k] as f64;
                a[p * d + k] = (c * apk - s * aqk) as f32;
                a[r * d + k] = (s * apk + c * aqk) as f32;
            }
            // Zero the pivot exactly (its residual is pure round-off).
            a[p * d + r] = 0.0;
            a[r * d + p] = 0.0;

            // Q ← Q·J (columns p,r of Q rotate the same way as A's columns).
            for k in 0..d {
                let qkp = q[k * d + p] as f64;
                let qkq = q[k * d + r] as f64;
                q[k * d + p] = (c * qkp - s * qkq) as f32;
                q[k * d + r] = (s * qkp + c * qkq) as f32;
            }
        }
    }
}

/// Sort eigenpairs descending by eigenvalue and fix each eigenvector's sign
/// (largest-magnitude entry, first index on ties, made non-negative).
/// `v` is row-major `[d, d]` with eigenvectors as columns.
fn sort_and_fix_signs(evals: &mut [f32], v: &mut [f32], d: usize) {
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&i, &j| evals[j].total_cmp(&evals[i]));

    let old_e: Vec<f32> = evals.to_vec();
    let old_v: Vec<f32> = v.to_vec();
    for (new_j, &old_j) in order.iter().enumerate() {
        evals[new_j] = old_e[old_j];
        // Sign convention on the source column.
        let mut arg = 0usize;
        let mut best = -1.0f32;
        for i in 0..d {
            let m = old_v[i * d + old_j].abs();
            if m > best {
                best = m;
                arg = i;
            }
        }
        let flip = old_v[arg * d + old_j] < 0.0;
        for i in 0..d {
            let x = old_v[i * d + old_j];
            v[i * d + new_j] = if flip { -x } else { x };
        }
    }
}

impl Array<f32> {
    /// Eigendecomposition of a **symmetric** matrix (numpy `np.linalg.eigh`,
    /// but sorted **descending**): returns `(eigenvalues [D], eigenvectors
    /// [D, D])` where column `j` of the eigenvector matrix pairs with
    /// `eigenvalues[j]`, so `A ≈ V·diag(λ)·Vᵀ` and `Vᵀ·V ≈ I`.
    ///
    /// The input must be 2-D square; it is symmetrized as `(A + Aᵀ)/2` before
    /// solving, so callers whose matrices are symmetric only up to round-off
    /// (e.g. a GEMM-computed covariance) get the intended answer. Signs are
    /// deterministic: each eigenvector's largest-magnitude entry is
    /// non-negative.
    ///
    /// Cyclic Jacobi, host-driven: rotation angles on the host, `D×D` matrix
    /// updates as GPU [`Array::matmul`]s (see the module docs for the split).
    pub fn eigh_symmetric(&self) -> Result<(Array<f32>, Array<f32>), ArrayError> {
        if self.rank() != 2 || self.shape()[0] != self.shape()[1] {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "eigh_symmetric: input must be a square 2-D matrix",
            )));
        }
        let d = self.shape()[0];
        let g = self.gpu();

        // Host mirror, symmetrized; device copies of A and V.
        let mut a_host = self.contiguous_or_self()?.to_vec()?;
        symmetrize(&mut a_host, d);
        let mut a_gpu = Array::from_slice(g, &a_host, &[d, d])?;
        let mut v_gpu = Array::<f32>::eye(g, d)?;

        let frob = frob_norm(&a_host, d);
        for _sweep in 0..MAX_SWEEPS {
            if off_norm(&a_host, d) <= TOL as f64 * frob {
                break;
            }
            // Simulate the sweep on the host mirror, composing the rotations
            // into one orthogonal Q...
            let mut q = vec![0.0f32; d * d];
            for i in 0..d {
                q[i * d + i] = 1.0;
            }
            jacobi_sweep_host(&mut a_host, &mut q, d);

            // ...then replay it on the device: A ← Qᵀ·(A·Q), V ← V·Q.
            let q_arr = Array::from_slice(g, &q, &[d, d])?;
            let qt = q_arr.transpose(0, 1)?;
            a_gpu = qt.matmul(&a_gpu.matmul(&q_arr)?)?;
            v_gpu = v_gpu.matmul(&q_arr)?;

            // Re-sync the mirror from the device so next sweep's angles (and
            // the convergence test) reflect the authoritative GPU state.
            a_host = a_gpu.to_vec()?;
            symmetrize(&mut a_host, d);
        }

        // Diagonal of the (device-synced) mirror = eigenvalues; V columns =
        // eigenvectors. Order + sign are fixed host-side.
        let mut evals: Vec<f32> = (0..d).map(|i| a_host[i * d + i]).collect();
        let mut v = v_gpu.to_vec()?;
        sort_and_fix_signs(&mut evals, &mut v, d);

        let evals_arr = Array::from_slice(g, &evals, &[d])?;
        let evecs_arr = Array::from_slice(g, &v, &[d, d])?;
        Ok((evals_arr, evecs_arr))
    }
}
