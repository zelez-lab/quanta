//! Mixed-precision GEMM (narrow float inputs, f32 accumulate).
//!
//! A,B are stored in a narrow float dtype and converted to f32 on load; the
//! inner product accumulates in f32 and C is f32. This is the standard ML
//! mixed-precision path (16-/8-bit weights/activations, f32 math). The output
//! contract is the same real-arithmetic `gemmEntry`; the dtype is an
//! implementation detail of *how* the entry is computed. The forward-error
//! bound (`specs/verify/lean/Quanta/Blas/GemmMixed.lean`) splits the entry
//! error into the proven f32 GEMM error over the quantised inputs plus the
//! input-quantisation error — so each dtype reuses the GEMM proof.
//!
//! Storage width is intrinsic to the dtype: 2-byte dtypes (bf16/f16) ride a
//! `Field<u16>` via [`gemm_mixed`]; 1-byte dtypes (fp8) ride a `Field<u8>` via
//! [`gemm_mixed8`]. The naive one-thread-per-output-entry kernel is shared
//! with the quantized path (see `mixed_kernel`); the tiled / cooperative-matrix
//! paths are a later perf fork. Quantized int8/int4 inputs live in
//! `mixed_quant`.

use crate::mixed_kernel;
use quanta_core::{Field, Gpu, QuantaError};
use quanta_ir::ScalarType;

/// Input element dtype for a mixed-precision (narrow-float) GEMM. Output (C) is
/// always f32. The storage width is intrinsic to the dtype — 2-byte dtypes
/// (bf16/f16) ride a `Field<u16>` and dispatch through [`gemm_mixed`]; 1-byte
/// dtypes (fp8) ride a `Field<u8>` and dispatch through [`gemm_mixed8`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmInputType {
    /// bfloat16 — 1 sign / 8 exponent / 7 mantissa (top 16 bits of an f32).
    Bf16,
    /// IEEE half — 1 sign / 5 exponent / 10 mantissa.
    F16,
    /// fp8 E5M2 — 1 sign / 5 exponent / 2 mantissa.
    Fp8E5M2,
    /// fp8 E4M3 — 1 sign / 4 exponent / 3 mantissa.
    Fp8E4M3,
}

impl GemmInputType {
    fn scalar_type(self) -> ScalarType {
        match self {
            GemmInputType::Bf16 => ScalarType::BF16,
            GemmInputType::F16 => ScalarType::F16,
            GemmInputType::Fp8E5M2 => ScalarType::FP8E5M2,
            GemmInputType::Fp8E4M3 => ScalarType::FP8E4M3,
        }
    }

    fn tag(self) -> &'static str {
        match self {
            GemmInputType::Bf16 => "bf16",
            GemmInputType::F16 => "f16",
            GemmInputType::Fp8E5M2 => "fp8e5m2",
            GemmInputType::Fp8E4M3 => "fp8e4m3",
        }
    }

    /// Storage width in bytes — 2 for bf16/f16, 1 for fp8. Selects which public
    /// entry (`gemm_mixed` vs `gemm_mixed8`) accepts the dtype.
    fn storage_bytes(self) -> usize {
        match self {
            GemmInputType::Bf16 | GemmInputType::F16 => 2,
            GemmInputType::Fp8E5M2 | GemmInputType::Fp8E4M3 => 1,
        }
    }
}

/// `gemm_mixed`: `C ← α·A·B + β·C`, A `m×k` and B `k×n` row-major in a 2-byte
/// narrow `dtype` (`Bf16` / `F16`), C `m×n` in f32. A and B are `Field<u16>`
/// holding one element per 2-byte slot; C is `Field<f32>` (read for `β·C`,
/// overwritten). Errors on a shape mismatch, or if `dtype` is not a 2-byte
/// dtype (use [`gemm_mixed8`] for fp8).
#[allow(clippy::too_many_arguments)]
pub fn gemm_mixed(
    gpu: &Gpu,
    dtype: GemmInputType,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    a: &Field<u16>,
    b: &Field<u16>,
    beta: f32,
    c: &Field<f32>,
) -> Result<(), QuantaError> {
    if dtype.storage_bytes() != 2 {
        return Err(QuantaError::invalid_param(
            "gemm_mixed: dtype is not 2-byte — use gemm_mixed8",
        ));
    }
    mixed_kernel::dispatch(
        gpu,
        dtype.scalar_type(),
        dtype.tag(),
        m,
        n,
        k,
        alpha,
        a,
        b,
        beta,
        c,
    )
}

/// `gemm_mixed8`: like [`gemm_mixed`] but for 1-byte fp8 dtypes (`Fp8E5M2` /
/// `Fp8E4M3`). A and B are `Field<u8>` (one fp8 byte per slot); C is f32.
#[allow(clippy::too_many_arguments)]
pub fn gemm_mixed8(
    gpu: &Gpu,
    dtype: GemmInputType,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    a: &Field<u8>,
    b: &Field<u8>,
    beta: f32,
    c: &Field<f32>,
) -> Result<(), QuantaError> {
    if dtype.storage_bytes() != 1 {
        return Err(QuantaError::invalid_param(
            "gemm_mixed8: dtype is not 1-byte — use gemm_mixed",
        ));
    }
    mixed_kernel::dispatch(
        gpu,
        dtype.scalar_type(),
        dtype.tag(),
        m,
        n,
        k,
        alpha,
        a,
        b,
        beta,
        c,
    )
}

/// `gemv_mixed`: `y ← α·A·x + β·y`, A `m×n` row-major in a 2-byte `dtype`, x in
/// `dtype`, y in f32. A gemv is a gemm with one output column (N=1), so this
/// routes into `gemm_mixed(m, 1, n, …)` — same kernel, same proven bound
/// (`Quanta.Blas.gemvEntry_eq_gemmEntry`).
#[allow(clippy::too_many_arguments)]
pub fn gemv_mixed(
    gpu: &Gpu,
    dtype: GemmInputType,
    m: u32,
    n: u32,
    alpha: f32,
    a: &Field<u16>,
    x: &Field<u16>,
    beta: f32,
    y: &Field<f32>,
) -> Result<(), QuantaError> {
    gemm_mixed(gpu, dtype, m, 1, n, alpha, a, x, beta, y)
}

/// `gemv_mixed8`: [`gemv_mixed`] for 1-byte fp8 dtypes — routes into
/// `gemm_mixed8(m, 1, n, …)`.
#[allow(clippy::too_many_arguments)]
pub fn gemv_mixed8(
    gpu: &Gpu,
    dtype: GemmInputType,
    m: u32,
    n: u32,
    alpha: f32,
    a: &Field<u8>,
    x: &Field<u8>,
    beta: f32,
    y: &Field<f32>,
) -> Result<(), QuantaError> {
    gemm_mixed8(gpu, dtype, m, 1, n, alpha, a, x, beta, y)
}
