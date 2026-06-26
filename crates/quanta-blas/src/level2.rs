//! Level-2 BLAS GEMV (f32) — `y ← α·A·x + β·y`, A row-major `m×n`.
//!
//! A GEMV is exactly a GEMM with a single output column: with `A` as the
//! `m×n` matrix, `x` as the `n×1` right operand, and `y` as the `m×1`
//! accumulator,
//!
//!   `y[i] = α · Σⱼ A[i,j]·x[j] + β · y[i]`   =   `(A·x)ᵢ`
//!
//! is `gemm(m, 1, n, α, A, x, β, y)`. So gemv reuses the proven GEMM kernel
//! verbatim — no separate kernel, no separate lowering. The numerical
//! contract is the same Higham bound: `Quanta.Blas.gemvEntry_eq_gemmEntry`
//! shows a gemv entry IS a gemm entry, so `gemmEntry_error_decomp` transfers
//! (`specs/verify/lean/Quanta/Blas/Gemv.lean`). The differential tests pin the
//! dedicated `gemv` surface against the reference oracle independently.

use crate::gemm;
use quanta::{Field, Gpu, QuantaError};

/// `gemv`: `y ← α·A·x + β·y`, A row-major `m×n`. `a` is `m·n`, `x` is `n`,
/// `y` is `m` (read for the `β·y` term, overwritten with the result, in
/// place). Errors on a shape mismatch.
///
/// Implemented as `gemm(m, 1, n, α, A, x, β, y)` — a GEMM with one output
/// column. The matrix `A` is `m×n`, so in GEMM terms it is `m×k` with `k = n`;
/// `x` is the `k×1` operand and `y` the `m×1` accumulator.
#[allow(clippy::too_many_arguments)]
pub fn gemv(
    gpu: &Gpu,
    m: u32,
    n: u32,
    alpha: f32,
    a: &Field<f32>,
    x: &Field<f32>,
    beta: f32,
    y: &Field<f32>,
) -> Result<(), QuantaError> {
    let (mu, nu) = (m as usize, n as usize);
    if a.len() != mu * nu {
        return Err(QuantaError::invalid_param("gemv: A length must be m*n"));
    }
    if x.len() != nu {
        return Err(QuantaError::invalid_param("gemv: x length must be n"));
    }
    if y.len() != mu {
        return Err(QuantaError::invalid_param("gemv: y length must be m"));
    }
    // GEMV ≡ GEMM with N = 1, K = n (the matrix's column count). gemm
    // validates the (now exact) lengths again and short-circuits empty cases.
    gemm::gemm(gpu, m, 1, n, alpha, a, x, beta, y)
}
