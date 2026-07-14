//! Triangular solves — Level-2 `trsv` (`op(A)·x = b`) and Level-3 `trsm`
//! (`op(A)·X = α·B` / `X·op(A) = α·B`), the LU/Cholesky building blocks.
//!
//! ## Kernel shape: one thread per independent lane
//!
//! Substitution has a sequential dependency chain *within* a solve, but a
//! `trsm` is many **independent** solves: with `side = Left` every RHS
//! column of `B` is its own length-`m` solve; with `side = Right` every RHS
//! row is its own length-`n` solve. So the kernel assigns one thread per
//! lane and each thread runs the full substitution for its lane — no
//! barriers, no shared memory, no cross-thread communication at all. For
//! the LU/Cholesky panel-solve workload (small triangular factor, many RHS)
//! this parallelises exactly where the work is. `trsv` is the one-lane
//! degenerate case (`trsm` with a single RHS column) — correct but serial;
//! a parallel single-vector solver is a later optimisation.
//!
//! ## Two kernels cover all 16 variants
//!
//! Every `side`/`uplo`/`transA` combination is a forward or a backward
//! substitution over an effective matrix `M[i,p] = a[i·rs + p·cs]` — the
//! strides encode the transpose/side, computed host-side by
//! [`crate::params::trsm_plan`]. `diag` is a scalar flag (unit diagonal
//! skips the divide). So: `trsm_fwd_f32` + `trsm_bwd_f32`, and **all**
//! BLAS variants are supported — nothing is `NotSupported`.
//!
//! Out-of-range lanes get a zero-iteration loop bound (no wrapping `if`
//! around the loop, no conditional stores — both known structured-control
//! lowering hazards).
//!
//! The numerical contract: each substitution step is the `gemmEntry`-shaped
//! formula `(α·b − Σ aₚ·xₚ)/d`; the per-step rounding decomposition and the
//! exact per-step residual are proven in
//! `specs/verify/lean/Quanta/Blas/Triangular.lean`. The full-solve Higham
//! backward-error bound (Thm 8.5) is flagged follow-up work there.

use crate::params::{Diag, Side, Trans, Uplo, trsm_plan};
use quanta_core::{Field, Gpu, QuantaError};

#[allow(unused_imports)]
mod kernel {
    use quanta_core::*;

    // Kernel-shape notes (all four kernels):
    //
    // - Every buffer index is XORed with `z`, a scalar param the host
    //   always passes as 0 — semantically a no-op. It defeats LLVM's
    //   pointer strength-reduction: without it, opt-level=3 hoists the
    //   loop-invariant lane base pointer into a local and turns the
    //   inner-loop addresses into walking pointer induction variables,
    //   and the WASM-route lowering (correctly) refuses to commit
    //   buffer-address values to registers. The XOR makes the index
    //   non-affine in the loop counters, so every access stays an inline
    //   `buf + (index << 2)` — the chained-address shape the lowering
    //   supports and proves.
    // - The inner-loop step `s` is likewise a scalar param the host
    //   always passes as 1: a runtime step makes the trip count
    //   uncomputable, so LLVM cannot unroll — the unroll epilogue is a
    //   conditional-buffer-load tail the lowering mishandles.
    // - Unit-diag is a separate kernel pair, not a flag: a loop-invariant
    //   `if unit …` inside the loop gets loop-UNSWITCHED into two loop
    //   copies cross-jumped through a multi-level `br` after the backedge
    //   `br_if` — a structured-control shape the lowering gets wrong. The
    //   split also honors the BLAS contract that the stored diagonal is
    //   never read under `Diag::Unit`.

    /// Forward substitution (non-unit diagonal), one thread per lane.
    /// Lane `l` solves the length-`nt` system whose element `t` lives at
    /// `x[l·lb + t·ts]`; the effective matrix is `a[i·rs + p·cs]`.
    /// Threads with `l ≥ nlanes` get `rows = 0` and never enter the loop
    /// (no `if` around a loop, no conditional store — lowering-safe).
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn trsm_fwd_f32(
        a: &[f32],
        x: &mut [f32],
        nt: u32,
        nlanes: u32,
        lb: u32,
        ts: u32,
        rs: u32,
        cs: u32,
        alpha: f32,
        z: u32,
        s: u32,
    ) {
        let lane = quark_id();
        let base = lane * lb;
        let rows = if lane < nlanes { nt } else { 0u32 };
        let mut i: u32 = 0u32;
        while i < rows {
            let mut acc: f32 = alpha * x[((base + i * ts) ^ z) as usize];
            let mut p: u32 = 0u32;
            while p < i {
                let av = a[((i * rs + p * cs) ^ z) as usize];
                let xv = x[((base + p * ts) ^ z) as usize];
                acc = acc - av * xv;
                p = p + s;
            }
            let d = a[((i * (rs + cs)) ^ z) as usize];
            x[((base + i * ts) ^ z) as usize] = acc / d;
            i = i + 1u32;
        }
    }

    /// Forward substitution, unit diagonal — as [`trsm_fwd_f32`] but the
    /// diagonal is implicitly 1: no diagonal load, no divide.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn trsm_fwd_unit_f32(
        a: &[f32],
        x: &mut [f32],
        nt: u32,
        nlanes: u32,
        lb: u32,
        ts: u32,
        rs: u32,
        cs: u32,
        alpha: f32,
        z: u32,
        s: u32,
    ) {
        let lane = quark_id();
        let base = lane * lb;
        let rows = if lane < nlanes { nt } else { 0u32 };
        let mut i: u32 = 0u32;
        while i < rows {
            let mut acc: f32 = alpha * x[((base + i * ts) ^ z) as usize];
            let mut p: u32 = 0u32;
            while p < i {
                let av = a[((i * rs + p * cs) ^ z) as usize];
                let xv = x[((base + p * ts) ^ z) as usize];
                acc = acc - av * xv;
                p = p + s;
            }
            x[((base + i * ts) ^ z) as usize] = acc;
            i = i + 1u32;
        }
    }

    /// Backward substitution (non-unit diagonal) — as [`trsm_fwd_f32`]
    /// but sweeping row `nt−1` down to `0`, update sum over `p > i`.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn trsm_bwd_f32(
        a: &[f32],
        x: &mut [f32],
        nt: u32,
        nlanes: u32,
        lb: u32,
        ts: u32,
        rs: u32,
        cs: u32,
        alpha: f32,
        z: u32,
        s: u32,
    ) {
        let lane = quark_id();
        let base = lane * lb;
        let rows = if lane < nlanes { nt } else { 0u32 };
        let mut ii: u32 = 0u32;
        while ii < rows {
            let i = (nt - 1u32) - ii;
            let mut acc: f32 = alpha * x[((base + i * ts) ^ z) as usize];
            let mut p: u32 = i + 1u32;
            while p < nt {
                let av = a[((i * rs + p * cs) ^ z) as usize];
                let xv = x[((base + p * ts) ^ z) as usize];
                acc = acc - av * xv;
                p = p + s;
            }
            let d = a[((i * (rs + cs)) ^ z) as usize];
            x[((base + i * ts) ^ z) as usize] = acc / d;
            ii = ii + 1u32;
        }
    }

    /// Backward substitution, unit diagonal — as [`trsm_bwd_f32`] but the
    /// diagonal is implicitly 1: no diagonal load, no divide.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn trsm_bwd_unit_f32(
        a: &[f32],
        x: &mut [f32],
        nt: u32,
        nlanes: u32,
        lb: u32,
        ts: u32,
        rs: u32,
        cs: u32,
        alpha: f32,
        z: u32,
        s: u32,
    ) {
        let lane = quark_id();
        let base = lane * lb;
        let rows = if lane < nlanes { nt } else { 0u32 };
        let mut ii: u32 = 0u32;
        while ii < rows {
            let i = (nt - 1u32) - ii;
            let mut acc: f32 = alpha * x[((base + i * ts) ^ z) as usize];
            let mut p: u32 = i + 1u32;
            while p < nt {
                let av = a[((i * rs + p * cs) ^ z) as usize];
                let xv = x[((base + p * ts) ^ z) as usize];
                acc = acc - av * xv;
                p = p + s;
            }
            x[((base + i * ts) ^ z) as usize] = acc;
            ii = ii + 1u32;
        }
    }
}

/// `trsm`: solve `op(A)·X = α·B` (`side = Left`, `A` is `m×m`) or
/// `X·op(A) = α·B` (`side = Right`, `A` is `n×n`) for `X`, **in place on
/// `b`** (`m×n`, row-major), `A` triangular. Only the `uplo` triangle of
/// `A` is read (and with [`Diag::Unit`], not the diagonal either). All
/// `side`/`uplo`/`trans`/`diag` combinations are supported.
///
/// One thread per RHS lane (column for Left, row for Right) — see the
/// module docs. As in BLAS, no singularity check is performed: a zero
/// diagonal with [`Diag::NonUnit`] yields `inf`/`nan` entries.
///
/// Errors on a shape mismatch between the declared dimensions and the
/// field lengths.
#[allow(clippy::too_many_arguments)]
pub fn trsm(
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
            "trsm: A length must be m*m (Left) or n*n (Right)",
        ));
    }
    if b.len() != mu * nu {
        return Err(QuantaError::invalid_param("trsm: B length must be m*n"));
    }
    if mu * nu == 0 {
        return Ok(());
    }

    let (rs, cs, forward) = trsm_plan(side, uplo, trans, na);
    // Lane layout: Left solves the n columns (element t of column j at
    // t·n + j), Right solves the m rows (element t of row i at i·n + t).
    let (nt, nlanes, lb, ts) = match side {
        Side::Left => (m, n, 1u32, n),
        Side::Right => (n, m, n, 1u32),
    };
    let mut wave = match (forward, diag) {
        (true, Diag::NonUnit) => kernel::trsm_fwd_f32(gpu)?,
        (true, Diag::Unit) => kernel::trsm_fwd_unit_f32(gpu)?,
        (false, Diag::NonUnit) => kernel::trsm_bwd_f32(gpu)?,
        (false, Diag::Unit) => kernel::trsm_bwd_unit_f32(gpu)?,
    };
    wave.bind(0, a);
    wave.bind(1, b); // in place: B is read (α·B) and overwritten with X
    wave.set_value(2, nt);
    wave.set_value(3, nlanes);
    wave.set_value(4, lb);
    wave.set_value(5, ts);
    wave.set_value(6, rs as u32);
    wave.set_value(7, cs as u32);
    wave.set_value(8, alpha);
    wave.set_value(9, 0u32); // z: the index XOR guard — always 0
    wave.set_value(10, 1u32); // s: the inner-loop step — always 1
    gpu.dispatch(&wave, nlanes)?.wait()?;
    Ok(())
}

/// `trsv`: solve `op(A)·x = b` for `x`, **in place on `x`** (which starts
/// holding `b`). `A` is `n×n` triangular, row-major; only the `uplo`
/// triangle is read. All `uplo`/`trans`/`diag` combinations are supported.
///
/// Exactly [`trsm`] with a single RHS column (as `gemv` is `gemm` with one
/// output column), so it reuses the same kernels and the same per-step
/// numerical contract. A single lane runs the whole substitution serially
/// on-device — correct everywhere; a parallel single-vector solver is a
/// later optimisation. Errors on a shape mismatch.
pub fn trsv(
    gpu: &Gpu,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    n: u32,
    a: &Field<f32>,
    x: &Field<f32>,
) -> Result<(), QuantaError> {
    if x.len() != n as usize {
        return Err(QuantaError::invalid_param("trsv: x length must be n"));
    }
    trsm(gpu, Side::Left, uplo, trans, diag, n, 1, 1.0, a, x)
}
