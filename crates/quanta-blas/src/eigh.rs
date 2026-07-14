//! Symmetric eigendecomposition — `eigh` (`A = V·Λ·Vᵀ` for a real symmetric
//! `A`), via the cyclic **Jacobi** eigenvalue algorithm. Eigenvalues in `w`
//! (ascending), orthonormal eigenvectors as the columns of `v`.
//!
//! ## Algorithm: cyclic Jacobi rotations
//!
//! Jacobi drives the symmetric matrix to diagonal form by a sequence of
//! Givens rotations, each chosen to annihilate one off-diagonal pair
//! `(p, q)`. A sweep cycles through all `p < q`; the off-diagonal Frobenius
//! norm decreases every sweep and converges quadratically, so a handful of
//! sweeps suffices. The accumulated rotations form the eigenvector matrix.
//!
//! For a pair `(p, q)` the rotation angle comes from the three scalars
//! `A[p,p]`, `A[q,q]`, `A[p,q]`:
//!
//!   `θ = (A[q,q] − A[p,p]) / (2·A[p,q])`,
//!   `t = sign(θ) / (|θ| + √(θ² + 1))`,  `c = 1/√(t² + 1)`,  `s = t·c`,
//!
//! and the rotation `J(p,q,c,s)` applied as `A ← Jᵀ·A·J` zeroes `A[p,q]`
//! (and `A[q,p]`) exactly in real arithmetic.
//!
//! ## Kernel shape: host-orchestrated rotations
//!
//! Only three scalars determine each rotation, so the **host** reads them
//! back, computes `(c, s)` in `f64`, and dispatches kernels that apply the
//! rotation to the two affected rows/columns of `A` and the two columns of
//! `V`. Each kernel is a **single loop** over the index `k` (the
//! lowering-safe shape the other blas kernels use); the sweep loop and the
//! tiny angle computation live on the host. The row side (`A ← Jᵀ·A`, with
//! `V ← V·J`) and the column side (`A ← A·J`) are two dispatches so each is a
//! race-free single pass. `n·(n−1)/2` rotations per sweep is the inherent
//! Jacobi cost; a parallel round-robin ordering is a later optimisation.
//!
//! ## Numerical contract
//!
//! Jacobi is *iterative* — the result is exact only in the limit. What is
//! exact per step (and proven in `specs/verify/lean/Quanta/Blas/Eigh.lean`):
//! a Jacobi rotation is orthogonal (`Jᵀ·J = I`), it preserves the Frobenius
//! norm and the trace (so the eigenvalue sum is invariant), and the chosen
//! angle annihilates the target off-diagonal exactly. The per-entry rounding
//! decomposition of one rotation-update reuses the shared `roundedOp` model.
//! The whole-decomposition convergence-rate / backward-error bound is flagged
//! there as follow-up.

use crate::params::Uplo;
use quanta_core::{Field, Gpu, QuantaError};

#[allow(unused_imports)]
mod kernel {
    use quanta_core::*;

    /// Row side of the two-sided rotation: `A ← Jᵀ·A` on rows `p,q`, and
    /// `V ← V·J` on eigenvector columns `p,q`. One thread per index `k`.
    /// `z` is the address-XOR guard (0), `s_step` the loop step (1). The
    /// rotation `(c,s)` is host-computed from `A[p,p], A[q,q], A[p,q]`.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn jacobi_rot_f32(
        a: &mut [f32],
        v: &mut [f32],
        n: u32,
        p: u32,
        q: u32,
        c: f32,
        s: f32,
        z: u32,
        s_step: u32,
    ) {
        let k = quark_id();
        let active = if k < n { 1u32 } else { 0u32 };
        let steps = active * s_step;
        let mut done: u32 = 0u32;
        while done < steps {
            // Rows p,q at column k (A ← Jᵀ·A).
            let apk = a[((p * n + k) ^ z) as usize];
            let aqk = a[((q * n + k) ^ z) as usize];
            a[((p * n + k) ^ z) as usize] = c * apk - s * aqk;
            a[((q * n + k) ^ z) as usize] = s * apk + c * aqk;
            // Eigenvector columns p,q (V ← V·J).
            let vkp = v[((k * n + p) ^ z) as usize];
            let vkq = v[((k * n + q) ^ z) as usize];
            v[((k * n + p) ^ z) as usize] = c * vkp - s * vkq;
            v[((k * n + q) ^ z) as usize] = s * vkp + c * vkq;
            done = done + s_step;
        }
    }

    /// Column side of the two-sided rotation: `A ← A·J` on columns `p,q`,
    /// run after the row pass so it reads the row-updated `A`.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn jacobi_rot_cols_f32(
        a: &mut [f32],
        n: u32,
        p: u32,
        q: u32,
        c: f32,
        s: f32,
        z: u32,
        s_step: u32,
    ) {
        let k = quark_id();
        let active = if k < n { 1u32 } else { 0u32 };
        let steps = active * s_step;
        let mut done: u32 = 0u32;
        while done < steps {
            let akp = a[((k * n + p) ^ z) as usize];
            let akq = a[((k * n + q) ^ z) as usize];
            a[((k * n + p) ^ z) as usize] = c * akp - s * akq;
            a[((k * n + q) ^ z) as usize] = s * akp + c * akq;
            done = done + s_step;
        }
    }
}

/// Number of Jacobi sweeps. Cyclic Jacobi converges quadratically; for the
/// sizes here (≤ ~64) a dozen sweeps drive the off-diagonal to f32 noise.
const MAX_SWEEPS: u32 = 24;
/// Off-diagonal magnitude below which a pivot is skipped as already zero.
const EPS: f64 = 1e-12;

/// `eigh`: symmetric eigendecomposition of a real symmetric `n×n` row-major
/// matrix. Only the `uplo` triangle of `a` is read (the full symmetric
/// matrix is reconstructed internally). Writes the eigenvalues to `w`
/// (length `n`, ascending) and the orthonormal eigenvectors to `v`
/// (`n×n`, column `j` is the eigenvector for `w[j]`).
///
/// `a` itself is not modified — the routine rotates an internal device copy
/// to diagonal form. Iterative (Jacobi); see the module docs for the
/// convergence contract. Errors on a shape mismatch.
pub fn eigh(
    gpu: &Gpu,
    uplo: Uplo,
    n: u32,
    a: &Field<f32>,
    w: &Field<f32>,
    v: &Field<f32>,
) -> Result<(), QuantaError> {
    let nu = n as usize;
    if a.len() != nu * nu {
        return Err(QuantaError::invalid_param("eigh: A length must be n*n"));
    }
    if w.len() != nu {
        return Err(QuantaError::invalid_param("eigh: w length must be n"));
    }
    if v.len() != nu * nu {
        return Err(QuantaError::invalid_param("eigh: V length must be n*n"));
    }
    if nu == 0 {
        return Ok(());
    }

    // Host-side working copy of the full symmetric matrix (reconstructed from
    // the referenced triangle) and the eigenvector accumulator V = I.
    let a_in = a.read()?;
    let mut work = vec![0.0f32; nu * nu];
    for i in 0..nu {
        for j in 0..nu {
            let src = match uplo {
                Uplo::Lower => {
                    if j <= i {
                        a_in[i * nu + j]
                    } else {
                        a_in[j * nu + i]
                    }
                }
                Uplo::Upper => {
                    if j >= i {
                        a_in[i * nu + j]
                    } else {
                        a_in[j * nu + i]
                    }
                }
            };
            work[i * nu + j] = src;
        }
    }
    let mut vwork = vec![0.0f32; nu * nu];
    for i in 0..nu {
        vwork[i * nu + i] = 1.0;
    }

    if nu == 1 {
        w.write(&[work[0]])?;
        v.write(&vwork)?;
        return Ok(());
    }

    // Device buffers rotated in place.
    let af = gpu.field::<f32>(nu * nu)?;
    af.write(&work)?;
    let vf = gpu.field::<f32>(nu * nu)?;
    vf.write(&vwork)?;

    for _sweep in 0..MAX_SWEEPS {
        let mut off = 0.0f64; // largest |off-diagonal| this sweep
        for p in 0..nu {
            for q in (p + 1)..nu {
                // Read the 2×2 pivot block back to compute the angle on the host.
                let cur = af.read()?;
                let app = cur[p * nu + p] as f64;
                let aqq = cur[q * nu + q] as f64;
                let apq = cur[p * nu + q] as f64;
                if apq.abs() > off {
                    off = apq.abs();
                }
                if apq.abs() <= EPS {
                    continue;
                }
                // Jacobi rotation angle.
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                // Row pass: A ← Jᵀ·A, and V ← V·J.
                let mut rot = kernel::jacobi_rot_f32(gpu)?;
                rot.bind(0, &af);
                rot.bind(1, &vf);
                rot.set_value(2, n);
                rot.set_value(3, p as u32);
                rot.set_value(4, q as u32);
                rot.set_value(5, c as f32);
                rot.set_value(6, s as f32);
                rot.set_value(7, 0u32); // z
                rot.set_value(8, 1u32); // s_step
                gpu.dispatch(&rot, n)?.wait()?;

                // Column pass: A ← A·J.
                let mut rotc = kernel::jacobi_rot_cols_f32(gpu)?;
                rotc.bind(0, &af);
                rotc.set_value(1, n);
                rotc.set_value(2, p as u32);
                rotc.set_value(3, q as u32);
                rotc.set_value(4, c as f32);
                rotc.set_value(5, s as f32);
                rotc.set_value(6, 0u32); // z
                rotc.set_value(7, 1u32); // s_step
                gpu.dispatch(&rotc, n)?.wait()?;
            }
        }
        if off <= EPS {
            break;
        }
    }

    // Eigenvalues are the diagonal of the (now near-diagonal) A; sort
    // ascending and permute the eigenvector columns to match.
    let a_final = af.read()?;
    let v_final = vf.read()?;
    let mut order: Vec<usize> = (0..nu).collect();
    order.sort_by(|&i, &j| {
        a_final[i * nu + i]
            .partial_cmp(&a_final[j * nu + j])
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    let mut w_out = vec![0.0f32; nu];
    let mut v_out = vec![0.0f32; nu * nu];
    for (new_col, &old_col) in order.iter().enumerate() {
        w_out[new_col] = a_final[old_col * nu + old_col];
        for row in 0..nu {
            v_out[row * nu + new_col] = v_final[row * nu + old_col];
        }
    }

    w.write(&w_out)?;
    v.write(&v_out)?;
    Ok(())
}
