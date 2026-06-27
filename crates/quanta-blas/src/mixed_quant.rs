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
    /// Per-tensor symmetric signed int8 (range [-128, 127]).
    Q8Symmetric,
}

impl GemmQuantType {
    fn tag(self) -> &'static str {
        match self {
            GemmQuantType::Q8Symmetric => "q8sym",
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
