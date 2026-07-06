//! Householder QR factorisation — `qr` (`geqrf`, `A = Q·R`) and the
//! least-squares solve `lstsq` built on it. The third matrix factorisation
//! in the crate (after Cholesky and LU); the workhorse behind
//! overdetermined `solve`/`lstsq` and the foundation for the SVD/eig path.
//!
//! ## Kernel shape: host-orchestrated per-column dispatch
//!
//! QR's dependency chain spans the columns — the reflector of column `k`
//! must be applied to the whole trailing submatrix before column `k+1`'s
//! reflector can be built. There is no cross-column parallelism, so the
//! **host loops the `n` columns** and, per column `k`, issues:
//!
//!   1. `qr_house_f32` (one thread) — read the sub-column `x = A[k:m, k]`,
//!      compute `α = −sign(x₀)·‖x‖`, build the Householder vector
//!      `v` (with `v[k] = x_k − α`, `v[i>k] = x_i`), store `v` into a scratch
//!      column, write `τ = 2/(vᵀv)` into `tau[k]`, and set `R[k,k] = α`.
//!   2. `qr_apply_f32` (one thread per trailing column `j > k`) — apply the
//!      reflector `H = I − τ·v·vᵀ` to column `j` of the trailing submatrix:
//!      `A[i,j] −= τ · v_i · (Σ_p v_p · A[p,j])`, independent across `j`.
//!
//! Each kernel is a **single loop** over the row range (the safe
//! structured-control shape the other blas kernels use); the multi-loop
//! all-in-one-lane form is a known lowering hazard, so the sequential
//! structure lives on the host. `n` sequential column steps is the inherent
//! QR critical path; the trailing update is parallel across columns. A
//! blocked (WY-form) panel factorisation is a later optimisation.
//!
//! ## Reflector storage
//!
//! The full Householder vectors are kept in a scratch `V` buffer (`m×n`)
//! rather than packed below `R`'s diagonal, so `Qᵀ` application (in `lstsq`)
//! and explicit `Q` formation read them directly without an implicit-unit
//! convention. `R` occupies the upper triangle of the input `a` on return;
//! the below-diagonal part of `a` holds the essential reflector tails
//! (`v_i` for `i > k`), matching the LAPACK `geqrf` layout, while the
//! scratch `V` carries the full vectors used internally.
//!
//! ## Buffer-address idioms (shared with `triangular`/`cholesky`)
//!
//! Every buffer index is XORed with `z` (host passes 0) so the inner-loop
//! addresses stay inline `buf + (index << 2)` rather than pointer-induction
//! variables the WASM-route lowering refuses to commit; the loop step `s`
//! (always 1) blocks the unroll whose epilogue the lowering mishandles.

use crate::params::{Diag, Side, Trans, Uplo};
use quanta::{Field, Gpu, QuantaError};

#[allow(unused_imports)]
mod kernel {
    use quanta::*;

    /// Column reflector: thread 0 reads sub-column `x = A[k:m, k]`, forms the
    /// Householder vector into `v` (a scratch `m×n` buffer, column `k`),
    /// writes `tau[k]`, and sets `A[k,k] = α`. Also writes the essential
    /// reflector tail `v_i` (`i > k`) below the diagonal of `A`.
    ///
    /// `α = −sign(x_k)·‖x‖`; `v_k = x_k − α`, `v_i = x_i` (`i > k`);
    /// `τ = 2/(vᵀv)` (0 when the column is already zero below the diagonal).
    #[quanta::kernel(workgroup = [1])]
    pub fn qr_house_f32(
        a: &mut [f32],
        v: &mut [f32],
        tau: &mut [f32],
        m: u32,
        n: u32,
        k: u32,
        z: u32,
        s: u32,
    ) {
        let lane = quark_id();
        let active = if lane < 1u32 { 1u32 } else { 0u32 };
        // ‖x‖² over rows i in [k, m): Σ A[i,k]².
        let mut nrm2: f32 = 0.0f32;
        let mut i: u32 = active * k;
        let ilim: u32 = active * m;
        while i < ilim {
            let xi = a[((i * n + k) ^ z) as usize];
            nrm2 = nrm2 + xi * xi;
            i = i + s;
        }
        let xnorm = sqrt(nrm2);
        let xk = a[((k * n + k) ^ z) as usize];
        // α = −sign(xk)·‖x‖ (sign(0) taken as +1 → α = −‖x‖).
        let sgn = if xk < 0.0f32 { -1.0f32 } else { 1.0f32 };
        let alpha = -sgn * xnorm;
        // v_k = xk − α; v_i = x_i (i>k). vᵀv = (xk−α)² + Σ_{i>k} x_i².
        let vk = xk - alpha;
        // Σ_{i>k} x_i² = nrm2 − xk².
        let tail = nrm2 - xk * xk;
        let vtv = vk * vk + tail;
        // τ = 2/vᵀv, guarding the degenerate all-zero-below column.
        let tau_k = if vtv > 0.0f32 { 2.0f32 / vtv } else { 0.0f32 };
        // Write v into the scratch column and the reflector tail below R.
        let mut w: u32 = active * k;
        let wlim: u32 = active * m;
        while w < wlim {
            let xw = a[((w * n + k) ^ z) as usize];
            let vw = if w == k { vk } else { xw };
            v[((w * n + k) ^ z) as usize] = vw;
            // Below-diagonal a holds the essential tail v_i (i>k); diagonal → α.
            if w > k {
                a[((w * n + k) ^ z) as usize] = vw;
            }
            w = w + s;
        }
        if active == 1u32 {
            a[((k * n + k) ^ z) as usize] = alpha;
            tau[(k ^ z) as usize] = tau_k;
        }
    }

    /// Trailing-submatrix update: thread `j` (with `k < j < n`) applies the
    /// column-`k` reflector to column `j` of `A`:
    /// `A[i,j] −= τ_k · v_i · (Σ_{p≥k} v_p · A[p,j])` for `i` in `[k, m)`.
    /// The dot `Σ v_p·A[p,j]` and the axpy share one pass each.
    #[quanta::kernel(workgroup = [256])]
    pub fn qr_apply_f32(
        a: &mut [f32],
        v: &[f32],
        tau: &[f32],
        m: u32,
        n: u32,
        k: u32,
        z: u32,
        s: u32,
    ) {
        let j = quark_id();
        // Columns j in (k, n) are active.
        let active = if j < n {
            if j > k { 1u32 } else { 0u32 }
        } else {
            0u32
        };
        let tau_k = tau[(k ^ z) as usize];
        // dot = Σ_{p in [k,m)} v[p]·A[p,j].
        let mut dot: f32 = 0.0f32;
        let mut p: u32 = active * k;
        let plim: u32 = active * m;
        while p < plim {
            let vp = v[((p * n + k) ^ z) as usize];
            let apj = a[((p * n + j) ^ z) as usize];
            dot = dot + vp * apj;
            p = p + s;
        }
        let scale = tau_k * dot;
        // A[i,j] −= scale·v[i], i in [k,m).
        let mut i: u32 = active * k;
        let ilim: u32 = active * m;
        while i < ilim {
            let vi = v[((i * n + k) ^ z) as usize];
            let aij = a[((i * n + j) ^ z) as usize];
            let upd = aij - scale * vi;
            if active == 1u32 {
                a[((i * n + j) ^ z) as usize] = upd;
            }
            i = i + s;
        }
    }

    /// Apply `Qᵀ` to a right-hand-side matrix `B` (`m×nrhs`, row-major) using
    /// reflector column `k`: thread `j` (rhs column) does
    /// `B[i,j] −= τ_k · v_i · (Σ_{p≥k} v_p · B[p,j])`. Same shape as
    /// `qr_apply_f32` but over `B`'s columns. `Qᵀ = H_{n-1}···H_0`, so the
    /// host applies these in forward `k` order.
    #[quanta::kernel(workgroup = [256])]
    pub fn qr_qt_apply_f32(
        b: &mut [f32],
        v: &[f32],
        tau: &[f32],
        m: u32,
        n: u32,
        nrhs: u32,
        k: u32,
        z: u32,
        s: u32,
    ) {
        let j = quark_id();
        let active = if j < nrhs { 1u32 } else { 0u32 };
        let tau_k = tau[(k ^ z) as usize];
        let mut dot: f32 = 0.0f32;
        let mut p: u32 = active * k;
        let plim: u32 = active * m;
        while p < plim {
            let vp = v[((p * n + k) ^ z) as usize];
            let bpj = b[((p * nrhs + j) ^ z) as usize];
            dot = dot + vp * bpj;
            p = p + s;
        }
        let scale = tau_k * dot;
        let mut i: u32 = active * k;
        let ilim: u32 = active * m;
        while i < ilim {
            let vi = v[((i * n + k) ^ z) as usize];
            let bij = b[((i * nrhs + j) ^ z) as usize];
            let upd = bij - scale * vi;
            if active == 1u32 {
                b[((i * nrhs + j) ^ z) as usize] = upd;
            }
            i = i + s;
        }
    }
}

/// `qr` (`geqrf`): factor an `m×n` (`m ≥ n`) row-major matrix `a` **in
/// place** into `A = Q·R` by Householder reflections. On return the upper
/// triangle of `a` holds `R`; the strictly-lower part holds the essential
/// reflector tails, and `tau` (length `n`) the reflector scalars.
///
/// The host issues `n` sequential column steps — a one-thread reflector
/// dispatch then a parallel trailing-column update — the inherent QR
/// critical path; a blocked WY-form update is a later optimisation. Errors
/// on a shape mismatch or `m < n`.
pub fn qr(gpu: &Gpu, m: u32, n: u32, a: &Field<f32>, tau: &Field<f32>) -> Result<(), QuantaError> {
    let (mu, nu) = (m as usize, n as usize);
    if a.len() != mu * nu {
        return Err(QuantaError::invalid_param("qr: A length must be m*n"));
    }
    if tau.len() != nu {
        return Err(QuantaError::invalid_param("qr: tau length must be n"));
    }
    if m < n {
        return Err(QuantaError::invalid_param("qr: requires m >= n"));
    }
    if nu == 0 || mu == 0 {
        return Ok(());
    }
    // Scratch full-reflector buffer V (m×n), zero-initialised.
    let v = gpu.field::<f32>(mu * nu)?;
    v.write(&vec![0.0f32; mu * nu])?;

    for k in 0..n {
        let mut house = kernel::qr_house_f32(gpu)?;
        house.bind(0, a);
        house.bind(1, &v);
        house.bind(2, tau);
        house.set_value(3, m);
        house.set_value(4, n);
        house.set_value(5, k);
        house.set_value(6, 0u32); // z
        house.set_value(7, 1u32); // s
        gpu.dispatch(&house, 1)?.wait()?;

        let mut apply = kernel::qr_apply_f32(gpu)?;
        apply.bind(0, a);
        apply.bind(1, &v);
        apply.bind(2, tau);
        apply.set_value(3, m);
        apply.set_value(4, n);
        apply.set_value(5, k);
        apply.set_value(6, 0u32); // z
        apply.set_value(7, 1u32); // s
        gpu.dispatch(&apply, n)?.wait()?;
    }
    Ok(())
}

/// `lstsq` (least-squares solve): for an `m×n` (`m ≥ n`) matrix `A` and an
/// `m×nrhs` right-hand side `B`, compute the minimiser `X` (`n×nrhs`) of
/// `‖A·X − B‖₂`. Factors `A = Q·R` with [`qr`], applies `Qᵀ` to `B`, then
/// back-substitutes with the upper-triangular `R`.
///
/// `a` is overwritten with its QR factor (as LAPACK's `gels` does). `b` is
/// `m×nrhs` on entry; on return its **top `n×nrhs` block holds the
/// solution `X`** (the remaining `m − n` rows hold the residual image and
/// are left as scratch). Errors on a shape mismatch or `m < n`.
pub fn lstsq(
    gpu: &Gpu,
    m: u32,
    n: u32,
    nrhs: u32,
    a: &Field<f32>,
    tau: &Field<f32>,
    b: &Field<f32>,
) -> Result<(), QuantaError> {
    let (mu, nu, ru) = (m as usize, n as usize, nrhs as usize);
    if a.len() != mu * nu {
        return Err(QuantaError::invalid_param("lstsq: A length must be m*n"));
    }
    if tau.len() != nu {
        return Err(QuantaError::invalid_param("lstsq: tau length must be n"));
    }
    if b.len() != mu * ru {
        return Err(QuantaError::invalid_param("lstsq: B length must be m*nrhs"));
    }
    if m < n {
        return Err(QuantaError::invalid_param("lstsq: requires m >= n"));
    }
    if nu == 0 || ru == 0 || mu == 0 {
        return Ok(());
    }
    // Factor and rebuild V (qr consumes an internal scratch; redo it here so
    // Qᵀ can be applied to B from the stored reflectors in `a` below R).
    let v = gpu.field::<f32>(mu * nu)?;
    v.write(&vec![0.0f32; mu * nu])?;
    for k in 0..n {
        let mut house = kernel::qr_house_f32(gpu)?;
        house.bind(0, a);
        house.bind(1, &v);
        house.bind(2, tau);
        house.set_value(3, m);
        house.set_value(4, n);
        house.set_value(5, k);
        house.set_value(6, 0u32);
        house.set_value(7, 1u32);
        gpu.dispatch(&house, 1)?.wait()?;

        let mut apply = kernel::qr_apply_f32(gpu)?;
        apply.bind(0, a);
        apply.bind(1, &v);
        apply.bind(2, tau);
        apply.set_value(3, m);
        apply.set_value(4, n);
        apply.set_value(5, k);
        apply.set_value(6, 0u32);
        apply.set_value(7, 1u32);
        gpu.dispatch(&apply, n)?.wait()?;

        // Apply Qᵀ column-k reflector to B (Qᵀ = H_{n-1}···H_0, forward order).
        let mut qt = kernel::qr_qt_apply_f32(gpu)?;
        qt.bind(0, b);
        qt.bind(1, &v);
        qt.bind(2, tau);
        qt.set_value(3, m);
        qt.set_value(4, n);
        qt.set_value(5, nrhs);
        qt.set_value(6, k);
        qt.set_value(7, 0u32);
        qt.set_value(8, 1u32);
        gpu.dispatch(&qt, nrhs)?.wait()?;
    }
    // Back-substitute R·X = (Qᵀ·B)[0:n], upper-triangular non-unit.
    // R lives in the upper triangle of `a` (m×n): R[i,j] = a[i*n+j] for
    // i,j < n. For m > n those rows are not the whole buffer, and trsm needs a
    // dense n×n operand, so extract R into its own field and the top n rows of
    // Qᵀ·B into an n×nrhs field, solve there, then write the solution back into
    // b's leading n×nrhs block.
    let a_host = a.read()?;
    let mut r = vec![0.0f32; nu * nu];
    for i in 0..nu {
        for j in i..nu {
            r[i * nu + j] = a_host[i * nu + j];
        }
    }
    let rf = gpu.field::<f32>(nu * nu)?;
    rf.write(&r)?;

    let b_host = b.read()?;
    let mut rhs = vec![0.0f32; nu * ru];
    rhs[..nu * ru].copy_from_slice(&b_host[..nu * ru]);
    let xf = gpu.field::<f32>(nu * ru)?;
    xf.write(&rhs)?;

    crate::triangular::trsm(
        gpu,
        Side::Left,
        Uplo::Upper,
        Trans::NoTrans,
        Diag::NonUnit,
        n,
        nrhs,
        1.0,
        &rf,
        &xf,
    )?;

    // Write the solution back into b's leading n×nrhs block.
    let x_host = xf.read()?;
    let mut b_out = b_host;
    b_out[..nu * ru].copy_from_slice(&x_host[..nu * ru]);
    b.write(&b_out)?;
    Ok(())
}
