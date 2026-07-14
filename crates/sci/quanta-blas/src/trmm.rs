//! Level-3 BLAS TRMM (f32) — triangular matrix-multiply
//! `B ← α·op(A)·B` (side = Left) or `B ← α·B·op(A)` (side = Right), `A`
//! triangular ([`Uplo`]/[`Trans`]/[`Diag`]), in place on `B`.
//!
//! The multiply counterpart of `trsm`: same lane layout (one lane per RHS
//! column for Left, per row for Right), each lane computing the triangular
//! matrix-vector product for its vector serially. `B'[i] = α·Σ_{p∈tri(i)}
//! M[i,p]·B[p]` where `M = op(A)` accessed through the same stride plan as
//! `trsm` (`M[i,p] = a[i·rs + p·cs]`, `M[i,i] = a[i·(rs+cs)]`, implicit 1
//! under [`Diag::Unit`]).
//!
//! In-place safety: within a lane, output `i` reads inputs `p` in `i`'s
//! triangle. For an *upper* effective `M` (each `B'[i]` needs `B[p], p ≥ i`)
//! the lane walks `i` from high to low so a written `B'[i]` is never read by
//! a later, smaller-`i` step; for a *lower* `M` it walks low to high. This
//! is the reverse of the substitution direction `trsm` uses, and it keeps
//! the whole op in place with no scratch buffer.
//!
//! Each entry is a `gemmEntry` dot product (`α·dot(M-row, B-col) + 0`), so
//! the proven GEMM per-entry Higham decomposition transfers
//! (`specs/verify/lean/Quanta/Blas/Trmm.lean`).

use crate::params::{Diag, Side, Trans, Uplo};
use quanta_core::{Field, Gpu, QuantaError};

#[allow(unused_imports)]
mod kernel {
    use quanta_core::*;

    /// LOWER effective `M` (`B'[i] = α·Σ_{p≤i} M[i,p]·B[p]`). Each output
    /// `i` reads inputs `p ≤ i`; walking `i` from high to low keeps a
    /// written `B'[i]` from being read by a later smaller-`i` step, so the
    /// op stays in place. The high→low walk uses a separate `t` counter and
    /// `i = rows-1-t` computed only for the diagonal/bound comparison, while
    /// the accumulation walks `p` forward 0..=i (the trsm-identical address
    /// shape `base + p·ts`). `unit = 1` uses an implicit-1 diagonal.
    ///
    /// Addresses XORed with `z` (host passes 0) to stay inline
    /// `buf + (index << 2)`; `s` is the accumulation step (1). Lane guard
    /// folds into `rows` (no `if` around the loop) — the lowering-safe shape.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    #[allow(clippy::too_many_arguments)]
    pub fn trmm_lower_f32(
        a: &[f32],
        b: &mut [f32],
        nt: u32,
        nlanes: u32,
        lb: u32,
        ts: u32,
        rs: u32,
        cs: u32,
        unit: u32,
        alpha: f32,
        z: u32,
        s: u32,
    ) {
        let lane = quark_id();
        let base = lane * lb;
        let rows = if lane < nlanes { nt } else { 0u32 };
        let mut t: u32 = 0u32;
        while t < rows {
            let i = rows - 1u32 - t; // high → low (hazard-safe for lower M)
            let mut acc: f32 = 0.0f32;
            let mut p: u32 = 0u32;
            while p <= i {
                let mip = a[((i * rs + p * cs) ^ z) as usize];
                let on_diag = if p == i { 1u32 } else { 0u32 };
                let m_eff = if (unit == 1u32) & (on_diag == 1u32) {
                    1.0f32
                } else {
                    mip
                };
                let bp = b[((base + p * ts) ^ z) as usize];
                acc = acc + m_eff * bp;
                p = p + s;
            }
            b[((base + i * ts) ^ z) as usize] = alpha * acc;
            t = t + 1u32;
        }
    }

    /// UPPER effective `M` (`B'[i] = α·Σ_{p≥i} M[i,p]·B[p]`). Each output
    /// `i` reads inputs `p ≥ i`; walking `i` low → high keeps the op in
    /// place. Accumulation walks `p` from `i` to `nt`. Otherwise identical
    /// to [`trmm_lower_f32`].
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    #[allow(clippy::too_many_arguments)]
    pub fn trmm_upper_f32(
        a: &[f32],
        b: &mut [f32],
        nt: u32,
        nlanes: u32,
        lb: u32,
        ts: u32,
        rs: u32,
        cs: u32,
        unit: u32,
        alpha: f32,
        z: u32,
        s: u32,
    ) {
        let lane = quark_id();
        let base = lane * lb;
        let rows = if lane < nlanes { nt } else { 0u32 };
        let mut i: u32 = 0u32;
        while i < rows {
            let mut acc: f32 = 0.0f32;
            let mut p: u32 = i;
            while p < nt {
                let mip = a[((i * rs + p * cs) ^ z) as usize];
                let on_diag = if p == i { 1u32 } else { 0u32 };
                let m_eff = if (unit == 1u32) & (on_diag == 1u32) {
                    1.0f32
                } else {
                    mip
                };
                let bp = b[((base + p * ts) ^ z) as usize];
                acc = acc + m_eff * bp;
                p = p + s;
            }
            b[((base + i * ts) ^ z) as usize] = alpha * acc;
            i = i + 1u32;
        }
    }
}

/// `trmm`: triangular matrix-multiply `B ← α·op(A)·B` (`side = Left`, `A`
/// is `m×m`) or `B ← α·B·op(A)` (`side = Right`, `A` is `n×n`), in place on
/// `B` (`m×n` row-major). `A` is triangular; only its `uplo` triangle is
/// read, and with [`Diag::Unit`] the stored diagonal is not read. All
/// `side`/`uplo`/`trans`/`diag` combinations are supported. Errors on a
/// shape mismatch.
#[allow(clippy::too_many_arguments)]
pub fn trmm(
    gpu: &Gpu,
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    m: u32,
    n: u32,
    alpha: f32,
    a: &Field<f32>,
    b: &Field<f32>,
) -> Result<(), QuantaError> {
    let (mu, nu) = (m as usize, n as usize);
    let na = match side {
        Side::Left => mu,
        Side::Right => nu,
    };
    if a.len() != na * na {
        return Err(QuantaError::invalid_param(
            "trmm: A length must be m*m (Left) or n*n (Right)",
        ));
    }
    if b.len() != mu * nu {
        return Err(QuantaError::invalid_param("trmm: B length must be m*n"));
    }
    if mu * nu == 0 {
        return Ok(());
    }

    // Reuse the trsm stride plan: (rs, cs) map M = op(A) [Left] or op(A)ᵀ
    // [Right] into A's row-major storage; `forward` is true iff M is lower
    // triangular. trmm walks the *reverse* direction of the trsm solve.
    let (rs, cs, forward) = crate::params::trsm_plan(side, uplo, trans, na);
    // `forward` = the effective M is lower-triangular.
    let unit = matches!(diag, Diag::Unit) as u32;
    let (nt, nlanes, lb, ts) = match side {
        Side::Left => (m, n, 1u32, n),
        Side::Right => (n, m, n, 1u32),
    };

    let mut wave = if forward {
        kernel::trmm_lower_f32(gpu)?
    } else {
        kernel::trmm_upper_f32(gpu)?
    };
    wave.bind(0, a);
    wave.bind(1, b); // in place
    wave.set_value(2, nt);
    wave.set_value(3, nlanes);
    wave.set_value(4, lb);
    wave.set_value(5, ts);
    wave.set_value(6, rs as u32);
    wave.set_value(7, cs as u32);
    wave.set_value(8, unit);
    wave.set_value(9, alpha);
    wave.set_value(10, 0u32); // z: index XOR guard — always 0
    wave.set_value(11, 1u32); // s: accumulation step — always 1
    gpu.dispatch(&wave, nlanes)?.wait()?;
    Ok(())
}
