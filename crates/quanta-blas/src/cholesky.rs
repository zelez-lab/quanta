//! Cholesky factorisation — `potrf` (`A = L·Lᵀ` / `A = Uᵀ·U`) and the SPD
//! linear solve `chol_solve` built on it. The first matrix factorisation in
//! the crate; the workhorse behind symmetric-positive-definite `solve`/`inv`.
//!
//! ## Kernel shape: host-orchestrated per-column dispatch
//!
//! Cholesky's dependency chain spans the whole matrix — the diagonal `sqrt`
//! of column `j` gates every entry of column `j`, which in turn feeds all
//! later columns. There is no cross-column parallelism, so the **host loops
//! the `n` columns** and issues, per column `j`, two dispatches:
//!
//!   1. `chol_diag_f32` (one thread) — `L[j,j] = sqrt(A[j,j] − Σ_{p<j} L[j,p]²)`.
//!   2. `chol_col_f32` (one thread per row `i > j`) — the sub-diagonal entries
//!      `L[i,j] = (A[i,j] − Σ_{p<j} L[i,p]·L[j,p]) / L[j,j]`, independent
//!      across `i` once the diagonal is known.
//!
//! Each kernel is a **single loop** over `p < j` (the safe structured-control
//! shape the other blas kernels use); the multi-loop-per-lane form a single
//! all-in-one-lane kernel would need is a known lowering hazard, so the
//! sequential structure lives on the host instead. `n` sequential column
//! steps is the inherent Cholesky critical path; the within-column update is
//! parallel. A blocked panel factorisation (larger GEMM-shaped trailing
//! updates) is a later optimisation.
//!
//! ## Buffer-address idioms (shared with `triangular`)
//!
//! Every buffer index is XORed with `z` (host passes 0) so the inner-loop
//! addresses stay inline `buf + (index << 2)` rather than pointer induction
//! variables the WASM-route lowering refuses to commit; the loop step `s`
//! (always 1) blocks the unroll whose epilogue the lowering mishandles. See
//! `triangular`'s kernel notes.
//!
//! ## Numerical contract
//!
//! Each off-diagonal entry is the `gemmEntry`-shaped formula
//! `(A[i,j] − Σ_{p<j} L[i,p]·L[j,p]) / L[j,j]`; each diagonal is
//! `sqrt(A[j,j] − Σ_{p<j} L[j,p]²)`. The exact factorisation identity and the
//! per-entry rounding decomposition are in
//! `specs/verify/lean/Quanta/Blas/Cholesky.lean`; the full norm-wise Higham
//! backward-error bound (Thm 10.3) is flagged there as follow-up.

use crate::params::{Diag, Side, Trans, Uplo};
use quanta::{Field, Gpu, QuantaError};

#[allow(unused_imports)]
mod kernel {
    use quanta::*;

    /// Lower diagonal step: `L[j,j] = sqrt(A[j,j] − Σ_{p<j} L[j,p]²)`.
    /// One thread (thread 0). `z` is the address-XOR guard (0), `s` the loop
    /// step (1).
    #[quanta::kernel(workgroup = [1])]
    pub fn chol_diag_lower_f32(a: &mut [f32], n: u32, j: u32, z: u32, s: u32) {
        let lane = quark_id();
        let active = if lane < 1u32 { 1u32 } else { 0u32 };
        let steps = active * j;
        let mut d: f32 = a[((j * n + j) ^ z) as usize];
        let mut p: u32 = 0u32;
        while p < steps {
            let ljp = a[((j * n + p) ^ z) as usize];
            d = d - ljp * ljp;
            p = p + s;
        }
        if active == 1u32 {
            a[((j * n + j) ^ z) as usize] = sqrt(d);
        }
    }

    /// Lower sub-diagonal column step: thread `i` (with `i > j`, `i < n`)
    /// writes `L[i,j] = (A[i,j] − Σ_{p<j} L[i,p]·L[j,p]) / L[j,j]`. The
    /// diagonal `L[j,j]` was written by the preceding `chol_diag_lower_f32`.
    #[quanta::kernel(workgroup = [256])]
    pub fn chol_col_lower_f32(a: &mut [f32], n: u32, j: u32, z: u32, s: u32) {
        let i = quark_id();
        // Rows i in (j, n) are active; others run zero iterations and skip the store.
        let active = if i < n {
            if i > j { 1u32 } else { 0u32 }
        } else {
            0u32
        };
        let steps = active * j;
        let ljj = a[((j * n + j) ^ z) as usize];
        let mut acc: f32 = a[((i * n + j) ^ z) as usize];
        let mut p: u32 = 0u32;
        while p < steps {
            let lip = a[((i * n + p) ^ z) as usize];
            let ljp = a[((j * n + p) ^ z) as usize];
            acc = acc - lip * ljp;
            p = p + s;
        }
        if active == 1u32 {
            a[((i * n + j) ^ z) as usize] = acc / ljj;
        }
    }

    /// Upper diagonal step: `U[j,j] = sqrt(A[j,j] − Σ_{p<j} U[p,j]²)`.
    #[quanta::kernel(workgroup = [1])]
    pub fn chol_diag_upper_f32(a: &mut [f32], n: u32, j: u32, z: u32, s: u32) {
        let lane = quark_id();
        let active = if lane < 1u32 { 1u32 } else { 0u32 };
        let steps = active * j;
        let mut d: f32 = a[((j * n + j) ^ z) as usize];
        let mut p: u32 = 0u32;
        while p < steps {
            let upj = a[((p * n + j) ^ z) as usize];
            d = d - upj * upj;
            p = p + s;
        }
        if active == 1u32 {
            a[((j * n + j) ^ z) as usize] = sqrt(d);
        }
    }

    /// Upper super-diagonal row step: thread `i` (with `i > j`, `i < n`)
    /// writes `U[j,i] = (A[j,i] − Σ_{p<j} U[p,j]·U[p,i]) / U[j,j]`.
    #[quanta::kernel(workgroup = [256])]
    pub fn chol_col_upper_f32(a: &mut [f32], n: u32, j: u32, z: u32, s: u32) {
        let i = quark_id();
        let active = if i < n {
            if i > j { 1u32 } else { 0u32 }
        } else {
            0u32
        };
        let steps = active * j;
        let ujj = a[((j * n + j) ^ z) as usize];
        let mut acc: f32 = a[((j * n + i) ^ z) as usize];
        let mut p: u32 = 0u32;
        while p < steps {
            let uqj = a[((p * n + j) ^ z) as usize];
            let uqi = a[((p * n + i) ^ z) as usize];
            acc = acc - uqj * uqi;
            p = p + s;
        }
        if active == 1u32 {
            a[((j * n + i) ^ z) as usize] = acc / ujj;
        }
    }
}

/// `cholesky` (`potrf`): factor a symmetric positive-definite `n×n`
/// row-major matrix `a` **in place**. With `uplo = Lower` the lower factor
/// `L` (`A = L·Lᵀ`) is written into the lower triangle; with `uplo = Upper`
/// the upper factor `U` (`A = Uᵀ·U`) into the upper triangle. The opposite
/// triangle is left untouched.
///
/// The host issues `n` sequential column steps (a diagonal dispatch then a
/// parallel sub-column dispatch each), the inherent Cholesky critical path;
/// a blocked panel factorisation is a later optimisation. As in LAPACK, no
/// positive-definiteness check is performed — a non-SPD matrix produces
/// `nan` from the diagonal `sqrt`. Errors on a shape mismatch.
pub fn cholesky(gpu: &Gpu, uplo: Uplo, n: u32, a: &Field<f32>) -> Result<(), QuantaError> {
    let nu = n as usize;
    if a.len() != nu * nu {
        return Err(QuantaError::invalid_param("cholesky: A length must be n*n"));
    }
    if nu == 0 {
        return Ok(());
    }
    for j in 0..n {
        let mut diag = match uplo {
            Uplo::Lower => kernel::chol_diag_lower_f32(gpu)?,
            Uplo::Upper => kernel::chol_diag_upper_f32(gpu)?,
        };
        diag.bind(0, a);
        diag.set_value(1, n);
        diag.set_value(2, j);
        diag.set_value(3, 0u32); // z
        diag.set_value(4, 1u32); // s
        gpu.dispatch(&diag, 1)?.wait()?;

        let mut col = match uplo {
            Uplo::Lower => kernel::chol_col_lower_f32(gpu)?,
            Uplo::Upper => kernel::chol_col_upper_f32(gpu)?,
        };
        col.bind(0, a);
        col.set_value(1, n);
        col.set_value(2, j);
        col.set_value(3, 0u32); // z
        col.set_value(4, 1u32); // s
        gpu.dispatch(&col, n)?.wait()?;
    }
    Ok(())
}

/// `chol_solve` (`potrs` fused with `potrf`): solve `A·X = B` for a
/// symmetric positive-definite `A` (`n×n`), in place on `b` (`n×nrhs`,
/// row-major). Factors `A = L·Lᵀ` (or `Uᵀ·U`) with [`cholesky`], then runs
/// the two triangular solves.
///
/// For `uplo = Lower`: `A·X = B` ⇒ `L·(Lᵀ·X) = B`, so solve `L·Y = B`
/// (forward) then `Lᵀ·X = Y` (backward, via `trans = Trans`). For
/// `uplo = Upper` (`A = Uᵀ·U`): solve `Uᵀ·Y = B` then `U·X = Y`.
///
/// **`a` is overwritten with its Cholesky factor** (as LAPACK's `posv`
/// does). Pass a copy if the original `A` must survive. Errors on a shape
/// mismatch.
pub fn chol_solve(
    gpu: &Gpu,
    uplo: Uplo,
    n: u32,
    nrhs: u32,
    a: &Field<f32>,
    b: &Field<f32>,
) -> Result<(), QuantaError> {
    let (nu, ru) = (n as usize, nrhs as usize);
    if a.len() != nu * nu {
        return Err(QuantaError::invalid_param(
            "chol_solve: A length must be n*n",
        ));
    }
    if b.len() != nu * ru {
        return Err(QuantaError::invalid_param(
            "chol_solve: B length must be n*nrhs",
        ));
    }
    if nu == 0 || ru == 0 {
        return Ok(());
    }
    cholesky(gpu, uplo, n, a)?;
    match uplo {
        Uplo::Lower => {
            // L·Y = B, then Lᵀ·X = Y.
            crate::triangular::trsm(
                gpu,
                Side::Left,
                Uplo::Lower,
                Trans::NoTrans,
                Diag::NonUnit,
                n,
                nrhs,
                1.0,
                a,
                b,
            )?;
            crate::triangular::trsm(
                gpu,
                Side::Left,
                Uplo::Lower,
                Trans::Trans,
                Diag::NonUnit,
                n,
                nrhs,
                1.0,
                a,
                b,
            )?;
        }
        Uplo::Upper => {
            // Uᵀ·Y = B, then U·X = Y.
            crate::triangular::trsm(
                gpu,
                Side::Left,
                Uplo::Upper,
                Trans::Trans,
                Diag::NonUnit,
                n,
                nrhs,
                1.0,
                a,
                b,
            )?;
            crate::triangular::trsm(
                gpu,
                Side::Left,
                Uplo::Upper,
                Trans::NoTrans,
                Diag::NonUnit,
                n,
                nrhs,
                1.0,
                a,
                b,
            )?;
        }
    }
    Ok(())
}
