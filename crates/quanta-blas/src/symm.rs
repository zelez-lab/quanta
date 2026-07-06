//! Level-3 BLAS SYMM (f32) — `C ← α·A·B + β·C` (side = Left) or
//! `C ← α·B·A + β·C` (side = Right), where **A is symmetric** and only its
//! [`Uplo`] triangle is stored/read. `B`, `C` are general `m×n`.
//!
//! A symm entry is a gemm entry: with `side = Left`, `C[i,j] = α·Σₚ
//! Asym[i,p]·B[p,j] + β·C[i,j]`, where `Asym` is the full symmetric matrix
//! reconstructed from the stored triangle (`Asym[i,p] = A[i,p]` if `(i,p)`
//! is in the stored triangle, else `A[p,i]`). So the proven GEMM per-entry
//! Higham decomposition transfers verbatim
//! (`specs/verify/lean/Quanta/Blas/Symm.lean`).
//!
//! Kernel shape: one thread per `m×n` output entry (the naive-GEMM shape).
//! The symmetric access `Asym[i,p]` is computed inline by comparing `(i,p)`
//! against the stored triangle and picking `A[i,p]` or `A[p,i]` — a single
//! index select, no branch on the store. `side = Right` swaps the operand
//! roles host-side (`C ← α·B·A`), so one kernel covers both sides. The
//! symmetry-aware access is the only difference from a plain GEMM; skipping
//! redundant reads is a later blocked-kernel optimisation.

use crate::params::{Side, Uplo};
use quanta::{Field, Gpu, QuantaError};

#[allow(unused_imports)]
mod kernel {
    use quanta::*;

    /// One thread per C entry (`m×n`). `A` is the `d×d` symmetric operand
    /// (`d = m` for Left, `n` for Right); `bmat` is the general operand.
    /// `sideL = 1` computes `C ← α·A·B + β·C`, else `C ← α·B·A + β·C`.
    /// `lower = 1` reads A's lower triangle. The symmetric load picks
    /// `a[i·d + p]` when `(i,p)` is in the stored triangle else `a[p·d + i]`
    /// — folded into a single index with a 0/1 mask (no nested `if`s).
    ///
    /// The A-index is XORed with `z` (host passes 0 — a no-op) to keep the
    /// access an inline `buf + (index << 2)`, defeating LLVM's base-pointer
    /// hoist the WASM-route lowering refuses (same trick as syrk).
    #[quanta::kernel(workgroup = [256])]
    #[allow(clippy::too_many_arguments)]
    pub fn symm_f32(
        a: &[f32],
        bmat: &[f32],
        c: &mut [f32],
        m: u32,
        n: u32,
        d: u32,
        sidel: u32,
        lower: u32,
        alpha: f32,
        beta: f32,
        z: u32,
        s: u32,
    ) {
        let gid = quark_id();
        let total = m * n;
        let idx = if gid < total { gid } else { 0u32 };
        let row = idx / n;
        let col = idx - row * n;
        let inb = if gid < total { 1u32 } else { 0u32 };

        // Contract dimension is `d`. For Left: sum_p Asym[row,p]·B[p,col].
        // For Right: sum_p B[row,p]·Asym[p,col].
        let mut acc: f32 = 0.0f32;
        let mut p: u32 = 0u32;
        while p < d {
            // symmetric-matrix index for the pair (r, cc) given uplo:
            // Left  -> pair is (row, p); Right -> pair is (p, col).
            let sr = if sidel == 1u32 { row } else { p };
            let sc = if sidel == 1u32 { p } else { col };
            // in stored triangle? lower: sc <= sr; upper: sc >= sr.
            let le = if sc <= sr { 1u32 } else { 0u32 };
            let ge = if sr <= sc { 1u32 } else { 0u32 };
            let in_tri = if lower == 1u32 { le } else { ge };
            // stored index: in-tri -> sr·d + sc ; else -> sc·d + sr.
            let a_in = sr * d + sc;
            let a_sw = sc * d + sr;
            let a_idx = if in_tri == 1u32 { a_in } else { a_sw };
            let av = a[(a_idx ^ z) as usize];
            // general operand index.
            let b_idx = if sidel == 1u32 {
                p * n + col
            } else {
                row * d + p
            };
            let bv = bmat[(b_idx ^ z) as usize];
            acc = acc + av * bv;
            p = p + s;
        }

        let cv = c[idx as usize];
        let result = alpha * acc + beta * cv;
        if inb == 1u32 {
            c[idx as usize] = result;
        }
    }
}

/// `symm`: symmetric matrix-multiply `C ← α·A·B + β·C` (`side = Left`, `A`
/// is `m×m`) or `C ← α·B·A + β·C` (`side = Right`, `A` is `n×n`). `A` is
/// symmetric and only its `uplo` triangle is read; the opposite triangle is
/// never referenced. `B` and `C` are general `m×n` row-major; `C` is read
/// (for `β·C`) and overwritten. Errors on a shape mismatch.
#[allow(clippy::too_many_arguments)]
pub fn symm(
    gpu: &Gpu,
    side: Side,
    uplo: Uplo,
    m: u32,
    n: u32,
    alpha: f32,
    a: &Field<f32>,
    b: &Field<f32>,
    beta: f32,
    c: &Field<f32>,
) -> Result<(), QuantaError> {
    let (mu, nu) = (m as usize, n as usize);
    // A is d×d symmetric, d = m (Left) or n (Right).
    let d = match side {
        Side::Left => m,
        Side::Right => n,
    };
    let du = d as usize;
    if a.len() != du * du {
        return Err(QuantaError::invalid_param(
            "symm: A length must be d*d (d = m for Left, n for Right)",
        ));
    }
    if b.len() != mu * nu {
        return Err(QuantaError::invalid_param("symm: B length must be m*n"));
    }
    if c.len() != mu * nu {
        return Err(QuantaError::invalid_param("symm: C length must be m*n"));
    }
    if mu == 0 || nu == 0 {
        return Ok(());
    }

    let sidel = matches!(side, Side::Left) as u32;
    let lower = matches!(uplo, Uplo::Lower) as u32;

    let mut wave = kernel::symm_f32(gpu)?;
    wave.bind(0, a);
    wave.bind(1, b);
    wave.bind(2, c);
    wave.set_value(3, m);
    wave.set_value(4, n);
    wave.set_value(5, d);
    wave.set_value(6, sidel);
    wave.set_value(7, lower);
    wave.set_value(8, alpha);
    wave.set_value(9, beta);
    wave.set_value(10, 0u32); // z: index XOR guard — always 0
    wave.set_value(11, 1u32); // s: contraction-loop step — always 1
    gpu.dispatch(&wave, m * n)?.wait()?;
    Ok(())
}
