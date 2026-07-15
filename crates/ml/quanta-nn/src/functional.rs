//! Functional (stateless) neural-network ops. Currently: fused scaled
//! dot-product attention.
//!
//! [`scaled_dot_product_attention`] is the **forward-fused** entry point — it
//! dispatches the online-softmax kernel from [`crate::kernel`] and returns the
//! context array, never materialising the `seq_q × seq_k` score matrix.
//! [`sdpa_var`] is the tape-differentiable variant.
//!
//! ## Scope of this increment (honest)
//!
//! - **Single head, `f32`.** Shapes `Q:(seq_q, d)`, `K:(seq_k, d)`,
//!   `V:(seq_k, dv)`, out `(seq_q, dv)`.
//! - **Batch / multi-head = host loop.** A `[B, H, T, d]` workload is `B*H`
//!   independent calls to [`scaled_dot_product_attention`] (each head is a 2-D
//!   problem). Fusing the batch into one dispatch is a later increment; this
//!   commit ships the correct single-head core.
//! - **`sdpa_var` backward = naive recompute.** The forward *value* matches the
//!   fused kernel (differential-tested), but the tape records the composed
//!   attention ops (scale · QKᵀ → mask → softmax → ·V), so backward flows
//!   through the existing `quanta-autograd` VJPs — the `seq_q × seq_k` matrix is
//!   rematerialised on the backward path. The fused backward (consuming the
//!   `(m, l)` stats the forward kernel already saves) is the next slice.

use quanta_array::Array;
use quanta_autograd::{AutogradError, DiffScalar, Tape, Var};
use quanta_core::{Gpu, QuantaError};

/// Lift a runtime `QuantaError` (from a field/dispatch call) into
/// `AutogradError` via `ArrayError::Gpu` — the `?` operator only performs one
/// `From` hop, and `AutogradError` converts from `ArrayError`, not directly
/// from `QuantaError`.
fn lift(e: QuantaError) -> AutogradError {
    AutogradError::from(quanta_array::ArrayError::Gpu(e))
}

/// Options for [`scaled_dot_product_attention`] / [`sdpa_var`].
/// The default (`Sdpa::default()`) is full bidirectional attention with the
/// standard `1/√d` scale and no padding — the plain scaled-dot-product case.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sdpa {
    /// Multiplies the raw `Q·Kᵀ` scores. `None` → `1/√d` (the standard
    /// scaled-dot-product factor, `d` the query/key head dim).
    pub scale: Option<f32>,
    /// Apply a causal (lower-triangular) mask: query row `i` attends only to
    /// keys `j ≤ i`. `false` → full (bidirectional) attention.
    pub causal: bool,
    /// Effective (unpadded) key length. `None` → all `seq_k` keys are real;
    /// `Some(n)` restricts every query to keys `j < n` (right-padding mask).
    pub kv_len: Option<usize>,
}

impl Sdpa {
    /// Resolve the scale factor (defaulting to `1/√d`).
    fn resolve_scale(&self, d: usize) -> f32 {
        self.scale.unwrap_or_else(|| 1.0 / (d as f32).sqrt())
    }

    /// Resolve the effective key length, clamped to `[1, seq_k]`.
    fn resolve_kv_len(&self, seq_k: usize) -> usize {
        self.kv_len.unwrap_or(seq_k).clamp(1, seq_k)
    }
}

/// Result of the fused forward: the context array plus the per-row softmax
/// stats the future fused backward consumes.
pub struct SdpaOutput {
    /// Attention output, shape `(seq_q, dv)`.
    pub output: Array<f32>,
    /// Per-row `(m, l)` softmax statistics, shape `(seq_q, 2)`: column 0 is the
    /// row max `m*` of the scaled+masked scores, column 1 the normaliser
    /// `l* = Σ exp(score − m*)`. These are exactly the T9204 summary the online
    /// fold produces; the fused backward reads them to avoid recomputing the
    /// softmax denominator.
    pub stats: Array<f32>,
}

/// **Fused scaled dot-product attention (forward).** Single head, `f32`.
///
/// `q:(seq_q, d)`, `k:(seq_k, d)`, `v:(seq_k, dv)` → context `(seq_q, dv)`.
/// Streams the online-softmax recurrence (T9200–T9209) over the key sequence
/// per query row, so the score matrix is never materialised. Returns the
/// context together with the per-row `(m, l)` stats (see [`SdpaOutput`]).
///
/// Errors on a rank/shape mismatch (all three inputs must be 2-D with matching
/// `d` / `seq_k`).
pub fn scaled_dot_product_attention(
    gpu: &Gpu,
    q: &Array<f32>,
    k: &Array<f32>,
    v: &Array<f32>,
    opts: Sdpa,
) -> Result<SdpaOutput, AutogradError> {
    let map_err = |e: quanta_array::ArrayError| AutogradError::from(e);
    let bad = |msg: &str| {
        AutogradError::from(quanta_array::ArrayError::Gpu(
            quanta_core::QuantaError::invalid_param(msg),
        ))
    };

    if q.rank() != 2 || k.rank() != 2 || v.rank() != 2 {
        return Err(bad("sdpa: Q, K, V must each be 2-D"));
    }
    let (seq_q, d) = (q.shape()[0], q.shape()[1]);
    let (seq_k, dk) = (k.shape()[0], k.shape()[1]);
    let (seq_kv, dv) = (v.shape()[0], v.shape()[1]);
    if dk != d {
        return Err(bad("sdpa: K head dim must equal Q head dim"));
    }
    if seq_kv != seq_k {
        return Err(bad("sdpa: V rows must equal K rows (seq_k)"));
    }

    let scale = opts.resolve_scale(d);
    let kv_len = opts.resolve_kv_len(seq_k);

    // Host-bridge: materialise inputs, upload to fresh fields, dispatch, read
    // back. quanta-array keeps the Array→Field binding `pub(crate)`, so the ml
    // tier round-trips through host — matching how quanta-array itself
    // orchestrates its batched-gemm host loop. Batch/head fusion lives here.
    let q_host = q.contiguous().map_err(map_err)?.to_vec().map_err(map_err)?;
    let k_host = k.contiguous().map_err(map_err)?.to_vec().map_err(map_err)?;
    let v_host = v.contiguous().map_err(map_err)?.to_vec().map_err(map_err)?;

    let qf = gpu.field::<f32>(seq_q * d).map_err(lift)?;
    let kf = gpu.field::<f32>(seq_k * d).map_err(lift)?;
    let vf = gpu.field::<f32>(seq_k * dv).map_err(lift)?;
    let of = gpu.field::<f32>(seq_q * dv).map_err(lift)?;
    let sf = gpu.field::<f32>(seq_q * 2).map_err(lift)?;
    qf.write(&q_host).map_err(lift)?;
    kf.write(&k_host).map_err(lift)?;
    vf.write(&v_host).map_err(lift)?;

    crate::kernel::sdpa_forward(
        gpu,
        seq_q as u32,
        seq_k as u32,
        d as u32,
        dv as u32,
        scale,
        opts.causal,
        kv_len as u32,
        &qf,
        &kf,
        &vf,
        &of,
        &sf,
    )
    .map_err(lift)?;

    let out_host = of.read().map_err(lift)?;
    let stats_host = sf.read().map_err(lift)?;
    let output = Array::from_slice(gpu, &out_host, &[seq_q, dv]).map_err(map_err)?;
    let stats = Array::from_slice(gpu, &stats_host, &[seq_q, 2]).map_err(map_err)?;
    Ok(SdpaOutput { output, stats })
}

/// **Tape-differentiable scaled dot-product attention.** Single head.
///
/// The returned `Var` carries the attention context `(seq_q, dv)`. Its forward
/// *value* equals [`scaled_dot_product_attention`]'s (differential-tested), but
/// the graph records the composed ops (`scale·QKᵀ → mask → softmax → ·V`), so
/// backward flows through the existing `quanta-autograd` VJPs — the naive
/// recompute path this increment ships. The fused backward is the next slice.
///
/// `tape` owns the graph the `q`/`k`/`v` vars belong to; constant scale/mask
/// nodes are created on it.
pub fn sdpa_var<T: DiffScalar>(
    tape: &Tape<T>,
    q: &Var<T>,
    k: &Var<T>,
    v: &Var<T>,
    opts: Sdpa,
) -> Result<Var<T>, AutogradError> {
    let bad = |msg: &str| {
        AutogradError::from(quanta_array::ArrayError::Gpu(
            quanta_core::QuantaError::invalid_param(msg),
        ))
    };

    let qshape = q.value().shape().to_vec();
    let kshape = k.value().shape().to_vec();
    let vshape = v.value().shape().to_vec();
    if qshape.len() != 2 || kshape.len() != 2 || vshape.len() != 2 {
        return Err(bad("sdpa_var: Q, K, V must each be 2-D"));
    }
    let (seq_q, d) = (qshape[0], qshape[1]);
    let (seq_k, dk) = (kshape[0], kshape[1]);
    if dk != d {
        return Err(bad("sdpa_var: K head dim must equal Q head dim"));
    }
    if vshape[0] != seq_k {
        return Err(bad("sdpa_var: V rows must equal K rows (seq_k)"));
    }

    let gpu = q.value().gpu().clone();
    let scale = opts.resolve_scale(d) as f64;
    let kv_len = opts.resolve_kv_len(seq_k);

    // scores = (Q · Kᵀ) · scale  →  [seq_q, seq_k]
    let kt = k.transpose(0, 1)?; // [d, seq_k]
    let raw = q.matmul(&kt)?; // [seq_q, seq_k]
    let scale_arr = Array::full(&gpu, T::from_f64(scale), &[1])?
        .broadcast_to(&[seq_q, seq_k])?
        .contiguous()?;
    let mut scores = raw.mul(&tape.var(scale_arr))?;

    // Additive mask: 0 where a key is attended, −1e9 where masked (causal
    // and/or padding). Built once on host as a detached constant.
    if opts.causal || kv_len < seq_k {
        let neg = -1.0e9f64;
        let mut mask = vec![0f32; seq_q * seq_k];
        for i in 0..seq_q {
            for j in 0..seq_k {
                let causal_masked = opts.causal && j > i;
                let pad_masked = j >= kv_len;
                if causal_masked || pad_masked {
                    mask[i * seq_k + j] = neg as f32;
                }
            }
        }
        let mask_host: Vec<T> = mask.iter().map(|&x| T::from_f64(x as f64)).collect();
        let mask_arr = Array::from_slice(&gpu, &mask_host, &[seq_q, seq_k])?;
        scores = scores.add(&tape.var(mask_arr))?;
    }

    // Row-wise softmax over the key axis (scores is 2-D [seq_q, seq_k]), then
    // mix with V → [seq_q, dv].
    let attn = scores.softmax()?;
    attn.matmul(v)
}
