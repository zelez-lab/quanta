//! LU factorisation with partial pivoting — `lu` (`getrf`, `P·A = L·U`) and
//! the general linear solve `lu_solve` built on it. The non-symmetric
//! counterpart of `cholesky`; the workhorse behind general `solve`/`inv`.
//!
//! ## Kernel shape: host-orchestrated per-column dispatch
//!
//! LU's dependency chain spans the whole matrix — column `k` must be fully
//! pivoted and eliminated before column `k+1` can be touched. As in
//! `cholesky`, the **host loops the `n` columns** and issues per-column
//! dispatches, each a single-loop kernel (the lowering-safe shape); the
//! sequential structure lives on the host.
//!
//! Per column `k`, the right-looking algorithm does:
//!
//!   1. **Pivot search** — the row `r ≥ k` maximising `|A[r,k]|`. Done on the
//!      host: the sub-column `A[k..n, k]` is read back (one small transfer)
//!      and scanned. This keeps the device kernels branch-free and avoids an
//!      argmax-reduction dependency; the sub-column read is `O(n)` per step,
//!      `O(n²)` total — dominated by the `O(n³)` elimination.
//!   2. **Row swap** — `lu_swap_rows_f32` exchanges rows `k` and `r` across
//!      all `n` columns (one thread per column). Skipped when `r == k`.
//!   3. **Elimination** — `lu_elim_f32`: thread `i` (row `i > k`) computes the
//!      multiplier `L[i,k] = A[i,k] / A[k,k]` (stored below the diagonal) and
//!      updates the trailing row `A[i,j] -= L[i,k]·A[k,j]` for `j > k`. A
//!      single loop over `j` per lane — the safe structured-control shape.
//!
//! The pivots are recorded as an `ipiv` array of the swapped row index per
//! step (LAPACK convention: `ipiv[k]` is the row swapped with row `k`).
//!
//! ## Buffer-address idioms (shared with `cholesky` / `triangular`)
//!
//! Every buffer index is XORed with `z` (host passes 0) so the inner-loop
//! addresses stay inline `buf + (index << 2)`; the loop step `s` (always 1)
//! blocks the unroll whose epilogue the lowering mishandles. See
//! `triangular`'s kernel notes.
//!
//! ## Numerical contract
//!
//! Each eliminated entry is the `A[i,j] − L[i,k]·U[k,j]` Schur-complement
//! update, and each multiplier is `A[i,k] / A[k,k]` — the same
//! `(a − Σ …)/d` substitution shape as the triangular solve. The per-step
//! exact residual and the per-entry rounding decomposition are in
//! `specs/verify/lean/Quanta/Blas/Lu.lean`; the full norm-wise Higham bound
//! (with growth factor) is flagged there as follow-up.

use crate::params::{Diag, Side, Trans, Uplo};
use quanta_core::{Field, Gpu, QuantaError};

#[allow(unused_imports)]
mod kernel {
    use quanta_core::*;

    /// Swap rows `k` and `r` across all `n` columns. Thread `j` (column) moves
    /// `A[k,j]` and `A[r,j]`. `z` is the address-XOR guard (0), `s` the loop
    /// step (1, unused here but kept for the shared idiom).
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn lu_swap_rows_f32(a: &mut [f32], n: u32, k: u32, r: u32, z: u32, s: u32) {
        let j = quark_id();
        let active = if j < n { 1u32 } else { 0u32 };
        // Read both, write swapped, only when active. A dummy column 0 index
        // keeps the loads in-bounds for inactive lanes.
        let col = if active == 1u32 { j } else { 0u32 };
        let vk = a[((k * n + col) ^ z) as usize];
        let vr = a[((r * n + col) ^ z) as usize];
        // `s` participates so the parameter is live (blocks unroll assumptions).
        let one = if s < 2u32 { 1u32 } else { 1u32 };
        if active == one {
            a[((k * n + col) ^ z) as usize] = vr;
            a[((r * n + col) ^ z) as usize] = vk;
        }
    }

    /// Right-looking elimination for column `k`. Thread `i` (row `i`, with
    /// `k < i < n`) computes the multiplier `m = A[i,k] / A[k,k]`, stores it
    /// into `A[i,k]` (the `L` factor below the diagonal), then updates the
    /// trailing row `A[i,j] -= m·A[k,j]` for `j` in `(k, n)`. One loop over
    /// `j` — the safe single-loop shape.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn lu_elim_f32(a: &mut [f32], n: u32, k: u32, z: u32, s: u32) {
        let i = quark_id();
        let active = if i < n {
            if i > k { 1u32 } else { 0u32 }
        } else {
            0u32
        };
        let akk = a[((k * n + k) ^ z) as usize];
        let aik = a[((i * n + k) ^ z) as usize];
        let m = aik / akk;
        if active == 1u32 {
            a[((i * n + k) ^ z) as usize] = m;
        }
        // Update the trailing sub-row j in (k, n): A[i,j] -= m·A[k,j].
        let span = if active == 1u32 { n } else { 0u32 };
        let mut j: u32 = k + 1u32;
        while j < span {
            let akj = a[((k * n + j) ^ z) as usize];
            let aij = a[((i * n + j) ^ z) as usize];
            a[((i * n + j) ^ z) as usize] = aij - m * akj;
            j = j + s;
        }
    }
}

/// `lu` (`getrf`): factor an `n×n` row-major matrix `a` **in place** as
/// `P·A = L·U` with partial (row) pivoting. On return the strict lower
/// triangle of `a` holds the unit-lower multipliers `L` (the unit diagonal
/// is implicit), and the upper triangle (including the diagonal) holds `U`.
/// `ipiv` (length `n`) receives the pivot per step: `ipiv[k]` is the row that
/// was swapped with row `k` (LAPACK convention).
///
/// The host issues `n` sequential column steps — a host-side pivot search, an
/// optional row swap, then a parallel trailing-submatrix elimination — the
/// inherent LU critical path; a blocked panel factorisation is a later
/// optimisation. As in LAPACK, no singularity check is performed; a zero
/// pivot yields `inf`/`nan` in the multipliers. Errors on a shape mismatch.
pub fn lu(gpu: &Gpu, n: u32, a: &Field<f32>, ipiv: &Field<u32>) -> Result<(), QuantaError> {
    let nu = n as usize;
    if a.len() != nu * nu {
        return Err(QuantaError::invalid_param("lu: A length must be n*n"));
    }
    if ipiv.len() != nu {
        return Err(QuantaError::invalid_param("lu: ipiv length must be n"));
    }
    if nu == 0 {
        return Ok(());
    }
    let mut pivots = vec![0u32; nu];
    // `k` is a matrix coordinate (row/col index into `a` and a kernel param),
    // not merely a `pivots` index — the range loop is the right shape.
    #[allow(clippy::needless_range_loop)]
    for k in 0..nu {
        // 1. Pivot search: read back the sub-column A[k..n, k] and find the
        //    row of maximum absolute value.
        let col = a.read()?;
        let mut best_row = k;
        let mut best_abs = col[k * nu + k].abs();
        for r in (k + 1)..nu {
            let v = col[r * nu + k].abs();
            if v > best_abs {
                best_abs = v;
                best_row = r;
            }
        }
        pivots[k] = best_row as u32;

        // 2. Row swap (device), only if the pivot is a different row.
        if best_row != k {
            let mut sw = kernel::lu_swap_rows_f32(gpu)?;
            sw.bind(0, a);
            sw.set_value(1, n);
            sw.set_value(2, k as u32);
            sw.set_value(3, best_row as u32);
            sw.set_value(4, 0u32); // z
            sw.set_value(5, 1u32); // s
            gpu.dispatch(&sw, n)?.wait()?;
        }

        // 3. Elimination of the trailing submatrix for column k.
        let mut el = kernel::lu_elim_f32(gpu)?;
        el.bind(0, a);
        el.set_value(1, n);
        el.set_value(2, k as u32);
        el.set_value(3, 0u32); // z
        el.set_value(4, 1u32); // s
        gpu.dispatch(&el, n)?.wait()?;
    }
    ipiv.write(&pivots)?;
    Ok(())
}

/// `lu_solve` (`getrs` fused with `getrf`): solve `A·X = B` for a general
/// `n×n` matrix `A`, in place on `b` (`n×nrhs`, row-major). Factors
/// `P·A = L·U` with [`lu`], applies the row permutation to `B`, then runs the
/// two triangular solves: `L·Y = P·B` (forward, unit-lower) and `U·X = Y`
/// (backward, upper).
///
/// **`a` is overwritten with its LU factors** and **`ipiv` receives the
/// pivots** (as LAPACK's `gesv` does). Pass copies if the originals must
/// survive. Errors on a shape mismatch.
pub fn lu_solve(
    gpu: &Gpu,
    n: u32,
    nrhs: u32,
    a: &Field<f32>,
    ipiv: &Field<u32>,
    b: &Field<f32>,
) -> Result<(), QuantaError> {
    let (nu, ru) = (n as usize, nrhs as usize);
    if a.len() != nu * nu {
        return Err(QuantaError::invalid_param("lu_solve: A length must be n*n"));
    }
    if b.len() != nu * ru {
        return Err(QuantaError::invalid_param(
            "lu_solve: B length must be n*nrhs",
        ));
    }
    if ipiv.len() != nu {
        return Err(QuantaError::invalid_param(
            "lu_solve: ipiv length must be n",
        ));
    }
    if nu == 0 || ru == 0 {
        return Ok(());
    }
    lu(gpu, n, a, ipiv)?;

    // Apply the row permutation to B on the host: replay the swaps in order.
    let pivots = ipiv.read()?;
    let mut bh = b.read()?;
    // `k` is a matrix row coordinate (indexes `bh` rows and compares to `r`),
    // not merely a `pivots` index — the range loop is the right shape.
    #[allow(clippy::needless_range_loop)]
    for k in 0..nu {
        let r = pivots[k] as usize;
        if r != k {
            for j in 0..ru {
                bh.swap(k * ru + j, r * ru + j);
            }
        }
    }
    b.write(&bh)?;

    // L·Y = P·B (unit-lower forward), then U·X = Y (upper backward).
    crate::triangular::trsm(
        gpu,
        Side::Left,
        Uplo::Lower,
        Trans::NoTrans,
        Diag::Unit,
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
    Ok(())
}

/// `lu_inv` (`getri` via `getrf` + solve against the identity): compute
/// `A⁻¹` for a general `n×n` matrix `A`, writing the inverse into `out`
/// (`n×n`, row-major). Factors `A` (overwriting it) and solves `A·X = I`.
///
/// **`a` is overwritten with its LU factors** and **`ipiv` receives the
/// pivots**. `out` need not be initialised (it is set to the identity, then
/// solved in place). Errors on a shape mismatch.
pub fn lu_inv(
    gpu: &Gpu,
    n: u32,
    a: &Field<f32>,
    ipiv: &Field<u32>,
    out: &Field<f32>,
) -> Result<(), QuantaError> {
    let nu = n as usize;
    if a.len() != nu * nu {
        return Err(QuantaError::invalid_param("lu_inv: A length must be n*n"));
    }
    if out.len() != nu * nu {
        return Err(QuantaError::invalid_param("lu_inv: out length must be n*n"));
    }
    if ipiv.len() != nu {
        return Err(QuantaError::invalid_param("lu_inv: ipiv length must be n"));
    }
    if nu == 0 {
        return Ok(());
    }
    // out ← identity, then solve A·out = I.
    let mut id = vec![0.0f32; nu * nu];
    for i in 0..nu {
        id[i * nu + i] = 1.0;
    }
    out.write(&id)?;
    lu_solve(gpu, n, n, a, ipiv, out)
}
