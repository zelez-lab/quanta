//! Level-3 BLAS SYR2K (f32) — symmetric rank-2k update
//! `C ← α·(op(A)·op(B)ᵀ + op(B)·op(A)ᵀ) + β·C`, `C` symmetric, updating
//! only the [`Uplo`] triangle.
//!
//! A syr2k entry is a sum of two gemm entries: for the stored triangle,
//! `C[i,j] = α·(Σₚ op(A)[i,p]·op(B)[j,p] + Σₚ op(B)[i,p]·op(A)[j,p]) +
//! β·C[i,j]`. Each of the two dot products is a `gemmEntry`, so the proven
//! GEMM per-entry Higham decomposition transfers to each term
//! (`specs/verify/lean/Quanta/Blas/Syr2k.lean`); the exact symmetry
//! `C[i,j] = C[j,i]` holds because swapping `i,j` swaps the two terms.
//!
//! Kernel shape: one thread per `n×n` output entry (the syrk shape with a
//! second operand `B`). Off-triangle threads compute and discard — the
//! store is guarded by a single precomputed flag, so the opposite triangle
//! is never written (the BLAS contract). `Trans` swaps the `(A,B)` access
//! strides host-side, so one kernel covers both forms.

use crate::params::{Trans, Uplo};
use quanta::{Field, Gpu, QuantaError};

#[allow(unused_imports)]
mod kernel {
    use quanta::*;

    /// One thread per C entry (`n×n`). `op(X)[r,p] = x[r·rs + p·cs]`.
    /// `lower` selects the stored triangle. The two cross dot-products
    /// (`A·Bᵀ` and `B·Aᵀ`) accumulate in one K loop. Bounds + triangle
    /// fold into a single `u32` store flag (lowering-safe: no `&&`, no
    /// nested `if`s). Indices XORed with `z` (host passes 0) to keep the
    /// four loads inline `buf + (index << 2)` (the syrk trick).
    #[quanta::kernel(workgroup = [256])]
    #[allow(clippy::too_many_arguments)]
    pub fn syr2k_f32(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        n: u32,
        k: u32,
        rs: u32,
        cs: u32,
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

        let le = if col <= row { 1u32 } else { 0u32 };
        let ge = if row <= col { 1u32 } else { 0u32 };
        let tri = if lower == 1u32 { le } else { ge };
        let inb = if gid < total { 1u32 } else { 0u32 };

        // The two cross dot-products run in SEPARATE loops, each with only
        // two loads walking a shared `p·cs` term — the exact syrk shape the
        // lowering supports. Folding both into one loop (four loads sharing
        // `p·cs`) makes LLVM commit a common base pointer the WASM route
        // refuses, so keep them split.
        let mut acc: f32 = 0.0f32;
        let mut p: u32 = 0u32;
        while p < k {
            let ai = a[((row * rs + p * cs) ^ z) as usize];
            let bj = b[((col * rs + p * cs) ^ z) as usize];
            acc = acc + ai * bj;
            p = p + s;
        }
        let mut acc2: f32 = 0.0f32;
        let mut q: u32 = 0u32;
        while q < k {
            let bi = b[((row * rs + q * cs) ^ z) as usize];
            let aj = a[((col * rs + q * cs) ^ z) as usize];
            acc2 = acc2 + bi * aj;
            q = q + s;
        }
        let acc = acc + acc2;

        let cv = c[idx as usize];
        let result = alpha * acc + beta * cv;
        let ok = inb * tri;
        if ok == 1u32 {
            c[idx as usize] = result;
        }
    }
}

/// `syr2k`: symmetric rank-2k update
/// `C ← α·(op(A)·op(B)ᵀ + op(B)·op(A)ᵀ) + β·C`, updating **only** the
/// `uplo` triangle of `C`. Both forms are supported:
///
/// - [`Trans::NoTrans`]: `A`, `B` are `n×k`, `C ← α·(A·Bᵀ + B·Aᵀ) + β·C`
/// - [`Trans::Trans`]: `A`, `B` are `k×n`, `C ← α·(Aᵀ·B + Bᵀ·A) + β·C`
///
/// `C` is `n×n` row-major, read for `β·C` and overwritten on the selected
/// triangle only. `k = 0` degenerates to `C ← β·C` on the triangle. Errors
/// on a shape mismatch.
#[allow(clippy::too_many_arguments)]
pub fn syr2k(
    gpu: &Gpu,
    uplo: Uplo,
    trans: Trans,
    n: u32,
    k: u32,
    alpha: f32,
    a: &Field<f32>,
    b: &Field<f32>,
    beta: f32,
    c: &Field<f32>,
) -> Result<(), QuantaError> {
    let (nu, ku) = (n as usize, k as usize);
    if a.len() != nu * ku {
        return Err(QuantaError::invalid_param(
            "syr2k: A length must be n*k (NoTrans) or k*n (Trans)",
        ));
    }
    if b.len() != nu * ku {
        return Err(QuantaError::invalid_param(
            "syr2k: B length must be n*k (NoTrans) or k*n (Trans)",
        ));
    }
    if c.len() != nu * nu {
        return Err(QuantaError::invalid_param("syr2k: C length must be n*n"));
    }
    if nu == 0 {
        return Ok(());
    }

    // op(X)[r,p] = x[r·rs + p·cs].
    let (rs, cs) = match trans {
        Trans::NoTrans => (k, 1u32),
        Trans::Trans => (1u32, n),
    };
    let lower = matches!(uplo, Uplo::Lower) as u32;

    let mut wave = kernel::syr2k_f32(gpu)?;
    if ku == 0 {
        // A, B empty and never read; bind C so slots 0/1 have valid buffers.
        wave.bind(0, c);
        wave.bind(1, c);
    } else {
        wave.bind(0, a);
        wave.bind(1, b);
    }
    wave.bind(2, c);
    wave.set_value(3, n);
    wave.set_value(4, k);
    wave.set_value(5, rs);
    wave.set_value(6, cs);
    wave.set_value(7, lower);
    wave.set_value(8, alpha);
    wave.set_value(9, beta);
    wave.set_value(10, 0u32); // z
    wave.set_value(11, 1u32); // s
    gpu.dispatch(&wave, n * n)?.wait()?;
    Ok(())
}
