//! Level-3 BLAS SYRK (f32) — `C ← α·op(A)·op(A)ᵀ + β·C`, C symmetric,
//! updating only the [`Uplo`] triangle.
//!
//! A syrk entry is a gemm entry with both operands drawn from `A`:
//! `C[i,j] = α·Σₚ op(A)[i,p]·op(A)[j,p] + β·C[i,j]` — the same
//! `gemmEntry` formula with row `i` and row `j` of `op(A)` as the two
//! vectors. The proven GEMM per-entry Higham decomposition therefore
//! transfers verbatim, plus the symmetry fact `C[i,j] = C[j,i]` in exact
//! arithmetic (`specs/verify/lean/Quanta/Blas/Syrk.lean`).
//!
//! Kernel shape: one thread per `n×n` output entry (the naive-GEMM shape,
//! with `B := Aᵀ` folded into strides). Off-triangle threads compute and
//! discard — the store is guarded by a single precomputed flag, so the
//! opposite triangle of `C` is **never written** (the BLAS contract).
//! `Trans::NoTrans`/`Trans::Trans` swap `A`'s access strides host-side,
//! so one kernel covers both forms. Skipping the off-triangle work (the
//! symmetry flop saving) is a later blocked-kernel optimisation; this
//! kernel is the correctness-first differential-tested baseline.

use crate::params::{Trans, Uplo};
use quanta::{Field, Gpu, QuantaError};

#[allow(unused_imports)]
mod kernel {
    use quanta::*;

    /// One thread per C entry. `op(A)[r,p] = a[r·ars + p·acs]`; `lower`
    /// selects which triangle is stored. Bounds + triangle fold into a
    /// single `u32` flag guarding the store (the lowering-safe shape —
    /// no `&&`, no nested `if`s).
    ///
    /// The two A-loads' indices are XORed with `z` (a scalar param the
    /// host always passes as 0 — semantically a no-op): both loads walk
    /// the same buffer with a shared `p·acs` term, and at opt-level=3
    /// LLVM otherwise commits a common base *pointer* to a local, which
    /// the WASM-route lowering (correctly) refuses. The XOR keeps every
    /// address an inline `buf + (index << 2)` — the supported shape.
    #[quanta::kernel(workgroup = [256])]
    pub fn syrk_f32(
        a: &[f32],
        c: &mut [f32],
        n: u32,
        k: u32,
        ars: u32,
        acs: u32,
        lower: u32,
        alpha: f32,
        beta: f32,
        z: u32,
        s: u32,
    ) {
        let gid = quark_id();
        let total = n * n;
        let idx = if gid < total { gid } else { 0u32 };
        let row = idx / n;
        let col = idx - row * n;

        // Triangle membership as 0/1 masks (no nested ifs, no &&).
        let le = if col <= row { 1u32 } else { 0u32 };
        let ge = if row <= col { 1u32 } else { 0u32 };
        let tri = if lower == 1u32 { le } else { ge };
        let inb = if gid < total { 1u32 } else { 0u32 };

        let mut acc: f32 = 0.0f32;
        let mut p: u32 = 0u32;
        while p < k {
            let av = a[((row * ars + p * acs) ^ z) as usize];
            let bv = a[((col * ars + p * acs) ^ z) as usize];
            acc = acc + av * bv;
            p = p + s;
        }

        let cv = c[idx as usize];
        let result = alpha * acc + beta * cv;
        let ok = inb * tri;
        if ok == 1u32 {
            c[idx as usize] = result;
        }
    }
}

/// `syrk`: symmetric rank-k update `C ← α·op(A)·op(A)ᵀ + β·C`, updating
/// **only** the `uplo` triangle of `C` (the opposite triangle is never
/// read as output nor written). Both forms are supported:
///
/// - [`Trans::NoTrans`]: `A` is `n×k`, `C ← α·A·Aᵀ + β·C`
/// - [`Trans::Trans`]: `A` is `k×n`, `C ← α·Aᵀ·A + β·C`
///
/// `C` is `n×n` row-major, read for the `β·C` term and overwritten in
/// place on the selected triangle. `k = 0` degenerates to `C ← β·C` on
/// the triangle. Errors on a shape mismatch.
#[allow(clippy::too_many_arguments)]
pub fn syrk(
    gpu: &Gpu,
    uplo: Uplo,
    trans: Trans,
    n: u32,
    k: u32,
    alpha: f32,
    a: &Field<f32>,
    beta: f32,
    c: &Field<f32>,
) -> Result<(), QuantaError> {
    let (nu, ku) = (n as usize, k as usize);
    if a.len() != nu * ku {
        return Err(QuantaError::invalid_param(
            "syrk: A length must be n*k (NoTrans) or k*n (Trans)",
        ));
    }
    if c.len() != nu * nu {
        return Err(QuantaError::invalid_param("syrk: C length must be n*n"));
    }
    if nu == 0 {
        return Ok(());
    }

    let (ars, acs) = match trans {
        Trans::NoTrans => (k, 1u32),
        Trans::Trans => (1u32, n),
    };
    let lower = matches!(uplo, Uplo::Lower) as u32;

    let mut wave = kernel::syrk_f32(gpu)?;
    // k = 0 means A is empty and never read (the K loop runs zero times);
    // bind C in its slot so the wave has a valid buffer there.
    if ku == 0 {
        wave.bind(0, c);
    } else {
        wave.bind(0, a);
    }
    wave.bind(1, c); // in place: C is read (β·C) and written on the triangle
    wave.set_value(2, n);
    wave.set_value(3, k);
    wave.set_value(4, ars);
    wave.set_value(5, acs);
    wave.set_value(6, lower);
    wave.set_value(7, alpha);
    wave.set_value(8, beta);
    wave.set_value(9, 0u32); // z: the index XOR guard — always 0
    wave.set_value(10, 1u32); // s: the K-loop step — always 1
    gpu.dispatch(&wave, n * n)?.wait()?;
    Ok(())
}
