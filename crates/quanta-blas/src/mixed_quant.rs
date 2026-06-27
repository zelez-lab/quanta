//! Quantized GEMM (int8 inputs, per-tensor symmetric, f32 accumulate).
//!
//! A,B are symmetric-quantized integer codes with per-tensor scales `sa`/`sb`:
//! the real value of a code `q` is `scale·q` (`dequantize_sym`). The
//! dequantised product is `(sa·qa)·(sb·qb)` summed `= sa·sb·Σ qa·qb`, so the
//! kernel runs the shared mixed GEMM over the raw integer codes with the scales
//! folded into alpha (`alpha_eff = α·sa·sb`) — dequantisation costs nothing
//! per element. C is f32.
//!
//! int8 (Q8) codes ride a `Field<i32>` (one code per word; int8-ness is the
//! quantiser's range clamp, applied when the codes are produced, not a storage
//! width). The forward-error bound reuses `gemmEntry_narrow_error_split` (the
//! Lean `gemmEntryMixedQ8Sym_*` instance): a quantized entry is the real GEMM
//! entry over the dequantised inputs, so the split into f32-GEMM error plus
//! input-quantisation error holds exactly as for the float dtypes.

use crate::mixed_kernel;
use quanta::{Field, Gpu, QuantaError};
use quanta_ir::ScalarType;

/// Quantized input scheme for a quantized GEMM. Output (C) is always f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmQuantType {
    /// Per-tensor symmetric signed int8 (range [-128, 127]), one code per
    /// `i32` slot — use [`gemm_quant`].
    Q8Symmetric,
    /// Per-tensor symmetric signed int4 (range [-8, 7]), 8 codes packed per
    /// `u32` word — use [`gemm_quant4`].
    Q4Symmetric,
}

impl GemmQuantType {
    fn tag(self) -> &'static str {
        match self {
            GemmQuantType::Q8Symmetric => "q8sym",
            GemmQuantType::Q4Symmetric => "q4sym",
        }
    }
}

/// `gemm_quant`: `C ← α·(sa·A)·(sb·B) + β·C`, A `m×k` and B `k×n` row-major
/// int8 codes (in a `Field<i32>`), with per-tensor symmetric scales `a_scale`
/// / `b_scale`; C `m×n` in f32 (read for `β·C`, overwritten). The scales fold
/// into the effective alpha (`α·sa·sb`), so the kernel dequantises for free.
/// Errors on a shape mismatch.
#[allow(clippy::too_many_arguments)]
pub fn gemm_quant(
    gpu: &Gpu,
    qty: GemmQuantType,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    a_scale: f32,
    b_scale: f32,
    a: &Field<i32>,
    b: &Field<i32>,
    beta: f32,
    c: &Field<f32>,
) -> Result<(), QuantaError> {
    if qty != GemmQuantType::Q8Symmetric {
        return Err(QuantaError::invalid_param(
            "gemm_quant: only Q8Symmetric rides Field<i32> — use gemm_quant4 for Q4",
        ));
    }
    let alpha_eff = alpha * a_scale * b_scale;
    mixed_kernel::dispatch(
        gpu,
        ScalarType::I32,
        qty.tag(),
        m,
        n,
        k,
        alpha_eff,
        a,
        b,
        beta,
        c,
    )
}

/// `gemv_quant`: quantized GEMV — A `m×n` int8 codes, x int8 codes, y f32.
/// Routes into `gemm_quant(m, 1, n, …)`.
#[allow(clippy::too_many_arguments)]
pub fn gemv_quant(
    gpu: &Gpu,
    qty: GemmQuantType,
    m: u32,
    n: u32,
    alpha: f32,
    a_scale: f32,
    x_scale: f32,
    a: &Field<i32>,
    x: &Field<i32>,
    beta: f32,
    y: &Field<f32>,
) -> Result<(), QuantaError> {
    gemm_quant(gpu, qty, m, 1, n, alpha, a_scale, x_scale, a, x, beta, y)
}

/// `gemm_quant4`: per-tensor symmetric **int4** quantized GEMM. A,B are int4
/// codes packed 8 per `u32` word (`Field<u32>`, length `ceil(m·k/8)` /
/// `ceil(k·n/8)`); C is f32. Like `gemm_quant`, the per-tensor scales fold into
/// the effective alpha so the kernel dequantises for free; the int4 nibble
/// unpacking happens in `Load { ty: I4 }` on every backend.
#[allow(clippy::too_many_arguments)]
pub fn gemm_quant4(
    gpu: &Gpu,
    qty: GemmQuantType,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    a_scale: f32,
    b_scale: f32,
    a: &Field<u32>,
    b: &Field<u32>,
    beta: f32,
    c: &Field<f32>,
) -> Result<(), QuantaError> {
    if qty != GemmQuantType::Q4Symmetric {
        return Err(QuantaError::invalid_param(
            "gemm_quant4: only Q4Symmetric rides packed Field<u32> — use gemm_quant for Q8",
        ));
    }
    let alpha_eff = alpha * a_scale * b_scale;
    mixed_kernel::dispatch_i4(gpu, qty.tag(), m, n, k, alpha_eff, a, b, beta, c)
}

/// `gemv_quant4`: int4 quantized GEMV — routes into `gemm_quant4(m, 1, n, …)`.
/// `x` is `ceil(n/8)` packed words; `y` is f32 length `m`.
#[allow(clippy::too_many_arguments)]
pub fn gemv_quant4(
    gpu: &Gpu,
    qty: GemmQuantType,
    m: u32,
    n: u32,
    alpha: f32,
    a_scale: f32,
    x_scale: f32,
    a: &Field<u32>,
    x: &Field<u32>,
    beta: f32,
    y: &Field<f32>,
) -> Result<(), QuantaError> {
    gemm_quant4(gpu, qty, m, 1, n, alpha, a_scale, x_scale, a, x, beta, y)
}
