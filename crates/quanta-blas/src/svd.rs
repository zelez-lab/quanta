//! Singular value decomposition — `svd` (`A = U·Σ·Vᵀ` for a real `m×n`
//! matrix with `m ≥ n`), via **one-sided Jacobi**. Singular values in `s`
//! (descending), left factor `U` (`m×n`, orthonormal columns), right factor
//! `V` (`n×n`, orthonormal).
//!
//! ## Algorithm: one-sided Jacobi
//!
//! One-sided Jacobi orthogonalises the *columns* of `A` by a sequence of
//! right Givens rotations. For a column pair `(p, q)` the rotation angle
//! comes from the three inner products
//!
//!   `α = ⟨a_p, a_p⟩`, `β = ⟨a_q, a_q⟩`, `γ = ⟨a_p, a_q⟩`,
//!   `ζ = (β − α)/(2γ)`, `t = sign(ζ)/(|ζ| + √(ζ²+1))`,
//!   `c = 1/√(t²+1)`, `s = t·c`,
//!
//! and the right rotation `A ← A·J(p,q,c,s)` makes columns `p` and `q`
//! orthogonal (`⟨a_p, a_q⟩ = 0`) exactly in real arithmetic. A sweep cycles
//! all `p < q`; the columns converge to mutual orthogonality quadratically.
//! On convergence the column norms are the singular values `σ` and the
//! normalised columns are `U`; the accumulated rotations form `V`.
//!
//! This reuses the eigh Jacobi machinery — the same Givens rotation, applied
//! *one-sided* on the right instead of two-sided. Its numerical advantage is
//! high relative accuracy: small singular values are computed accurately
//! because the algorithm never forms `AᵀA` explicitly.
//!
//! ## Kernel shape: host-orchestrated rotations
//!
//! Each rotation needs only the three inner products, so the **host** reads
//! the two columns back, forms `α, β, γ` and the angle `(c, s)` in `f64`, and
//! dispatches a single kernel that applies the right rotation to columns
//! `(p, q)` of the working matrix and of `V`. The kernel is a **single loop**
//! over the row index `i` (the lowering-safe shape the other blas kernels
//! use); `z` is the address-XOR guard (0), `s_step` the loop step (1).
//!
//! ## Numerical contract
//!
//! One-sided Jacobi is *iterative* — orthogonality is reached only in the
//! limit. What is exact per step (and proven in
//! `specs/verify/lean/Quanta/Blas/Svd.lean`): the right rotation is orthogonal
//! (`JᵀJ = I`), it preserves the Frobenius norm of the acted-on pair (hence
//! `Σσ²` is invariant), and the chosen angle orthogonalises the target column
//! pair exactly. The per-entry rounding decomposition of one rotation-update
//! reuses the shared `roundedOp` model. The whole-decomposition
//! convergence-rate / backward-error bound is flagged there as follow-up.

use quanta::{Field, Gpu, QuantaError};

#[allow(unused_imports)]
mod kernel {
    use quanta::*;

    /// Right rotation of a column pair: `M ← M·J` on columns `p, q` of an
    /// `r×cols` row-major matrix. One thread per row `i`. `(c, s)` is
    /// host-computed. `z` is the address-XOR guard (0), `s_step` the loop
    /// step (1). Used for both the working matrix (`r = m`) and `V`
    /// (`r = n`), each with its own `cols`.
    #[quanta::kernel(workgroup = [256])]
    pub fn jacobi_col_rot_f32(
        mat: &mut [f32],
        rows: u32,
        cols: u32,
        p: u32,
        q: u32,
        c: f32,
        s: f32,
        z: u32,
        s_step: u32,
    ) {
        let i = quark_id();
        let active = if i < rows { 1u32 } else { 0u32 };
        let steps = active * s_step;
        let mut done: u32 = 0u32;
        while done < steps {
            let mip = mat[((i * cols + p) ^ z) as usize];
            let miq = mat[((i * cols + q) ^ z) as usize];
            mat[((i * cols + p) ^ z) as usize] = c * mip - s * miq;
            mat[((i * cols + q) ^ z) as usize] = s * mip + c * miq;
            done = done + s_step;
        }
    }
}

/// Number of one-sided Jacobi sweeps. Cyclic Jacobi converges quadratically;
/// for the sizes here a few dozen sweeps drive the column pairs to
/// orthogonality at f32 precision.
const MAX_SWEEPS: u32 = 60;
/// Relative off-diagonal magnitude below which a column pair is treated as
/// already orthogonal and skipped.
const EPS: f64 = 1e-12;

/// `svd`: economy singular value decomposition of a real `m×n` (`m ≥ n`)
/// row-major matrix. Writes the singular values to `s` (length `n`,
/// descending), the left factor to `u` (`m×n`, orthonormal columns) and the
/// right factor to `v` (`n×n`, orthonormal), with `A = U·diag(s)·Vᵀ`.
///
/// `a` itself is not modified — the routine orthogonalises an internal device
/// copy. Requires `m ≥ n` (the economy form); `m < n` returns
/// `NotSupported` (transpose the input and swap `U`/`V`). Iterative
/// (one-sided Jacobi); see the module docs for the convergence contract.
pub fn svd(
    gpu: &Gpu,
    m: u32,
    n: u32,
    a: &Field<f32>,
    u: &Field<f32>,
    s: &Field<f32>,
    v: &Field<f32>,
) -> Result<(), QuantaError> {
    let (mu, nu) = (m as usize, n as usize);
    if a.len() != mu * nu {
        return Err(QuantaError::invalid_param("svd: A length must be m*n"));
    }
    if u.len() != mu * nu {
        return Err(QuantaError::invalid_param("svd: U length must be m*n"));
    }
    if s.len() != nu {
        return Err(QuantaError::invalid_param("svd: s length must be n"));
    }
    if v.len() != nu * nu {
        return Err(QuantaError::invalid_param("svd: V length must be n*n"));
    }
    if m < n {
        return Err(QuantaError::not_supported(
            "svd: m < n not supported; transpose the input and swap U/V",
        ));
    }
    if nu == 0 {
        return Ok(());
    }

    // Working device copy of A (orthogonalised in place) and V = I.
    let wf = gpu.field::<f32>(mu * nu)?;
    wf.write(&a.read()?)?;
    let mut vwork = vec![0.0f32; nu * nu];
    for i in 0..nu {
        vwork[i * nu + i] = 1.0;
    }
    let vf = gpu.field::<f32>(nu * nu)?;
    vf.write(&vwork)?;

    if nu == 1 {
        // Single column: σ = ‖a‖, U = a/σ, V = [1].
        let col = a.read()?;
        let nrm = (col.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()).sqrt();
        let sig = nrm as f32;
        let denom = if nrm > 1e-300 { nrm } else { 1.0 };
        let u_out: Vec<f32> = col.iter().map(|&x| (x as f64 / denom) as f32).collect();
        u.write(&u_out)?;
        s.write(&[sig])?;
        v.write(&[1.0f32])?;
        return Ok(());
    }

    for _sweep in 0..MAX_SWEEPS {
        let mut off = 0.0f64; // largest |⟨a_p, a_q⟩| this sweep
        for p in 0..nu {
            for q in (p + 1)..nu {
                // Read the working matrix back and form the column inner products.
                let cur = wf.read()?;
                let mut alpha = 0.0f64;
                let mut beta = 0.0f64;
                let mut gamma = 0.0f64;
                for i in 0..mu {
                    let wip = cur[i * nu + p] as f64;
                    let wiq = cur[i * nu + q] as f64;
                    alpha += wip * wip;
                    beta += wiq * wiq;
                    gamma += wip * wiq;
                }
                let scale = (alpha * beta).sqrt().max(f64::MIN_POSITIVE);
                let rel = gamma.abs() / scale;
                if rel > off {
                    off = rel;
                }
                if rel <= EPS {
                    continue;
                }
                // One-sided Jacobi angle orthogonalising columns (p, q).
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (zeta * zeta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let sn = t * c;

                // Right rotation of the working matrix columns (p, q).
                let mut rot = kernel::jacobi_col_rot_f32(gpu)?;
                rot.bind(0, &wf);
                rot.set_value(1, m); // rows
                rot.set_value(2, n); // cols
                rot.set_value(3, p as u32);
                rot.set_value(4, q as u32);
                rot.set_value(5, c as f32);
                rot.set_value(6, sn as f32);
                rot.set_value(7, 0u32); // z
                rot.set_value(8, 1u32); // s_step
                gpu.dispatch(&rot, m)?.wait()?;

                // Right rotation of V columns (p, q).
                let mut rotv = kernel::jacobi_col_rot_f32(gpu)?;
                rotv.bind(0, &vf);
                rotv.set_value(1, n); // rows
                rotv.set_value(2, n); // cols
                rotv.set_value(3, p as u32);
                rotv.set_value(4, q as u32);
                rotv.set_value(5, c as f32);
                rotv.set_value(6, sn as f32);
                rotv.set_value(7, 0u32);
                rotv.set_value(8, 1u32);
                gpu.dispatch(&rotv, n)?.wait()?;
            }
        }
        if off <= EPS {
            break;
        }
    }

    // Column norms of the orthogonalised working matrix are the singular
    // values; the normalised columns are U.
    let w_final = wf.read()?;
    let v_final = vf.read()?;
    let mut sig = vec![0.0f64; nu];
    for j in 0..nu {
        let mut nrm = 0.0f64;
        for i in 0..mu {
            let x = w_final[i * nu + j] as f64;
            nrm += x * x;
        }
        sig[j] = nrm.sqrt();
    }
    // Sort singular values descending; permute U and V columns to match.
    let mut order: Vec<usize> = (0..nu).collect();
    order.sort_by(|&i, &j| {
        sig[j]
            .partial_cmp(&sig[i])
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    let mut u_out = vec![0.0f32; mu * nu];
    let mut s_out = vec![0.0f32; nu];
    let mut v_out = vec![0.0f32; nu * nu];
    for (newc, &oldc) in order.iter().enumerate() {
        s_out[newc] = sig[oldc] as f32;
        let denom = if sig[oldc] > 1e-300 { sig[oldc] } else { 1.0 };
        for i in 0..mu {
            u_out[i * nu + newc] = (w_final[i * nu + oldc] as f64 / denom) as f32;
        }
        for i in 0..nu {
            v_out[i * nu + newc] = v_final[i * nu + oldc];
        }
    }

    u.write(&u_out)?;
    s.write(&s_out)?;
    v.write(&v_out)?;
    Ok(())
}
