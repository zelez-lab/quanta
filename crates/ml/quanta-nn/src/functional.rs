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
//! - **`sdpa_var` backward = fully fused.** The forward runs the online-softmax
//!   kernel (saving `(m, l)` stats); the backward is a **custom VJP node** on
//!   the tape ([`quanta_autograd::Tape::custom_vjp`]) that dispatches the fused
//!   [`crate::kernel::sdpa_backward`] — reconstructing the softmax weights from
//!   the saved stats (T9204), so the `seq_q × seq_k` matrix is never
//!   materialised on *either* pass. The old composed path
//!   ([`sdpa_var_composed`]) is retained as the differential-test oracle.

use quanta_array::{Array, ToF64};
use quanta_autograd::{AutogradError, DiffScalar, Tape, Var};
use quanta_core::{Gpu, QuantaError};

/// Lift a runtime `QuantaError` (from a field/dispatch call) into
/// `AutogradError` via `ArrayError::Gpu` — the `?` operator only performs one
/// `From` hop, and `AutogradError` converts from `ArrayError`, not directly
/// from `QuantaError`.
pub(crate) fn lift(e: QuantaError) -> AutogradError {
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
/// The returned `Var` carries the attention context `(seq_q, dv)`. The forward
/// runs the fused online-softmax kernel (via [`scaled_dot_product_attention`],
/// saving the `(m, l)` stats); the backward is a **custom VJP node** on the
/// tape ([`quanta_autograd::Tape::custom_vjp`]) that dispatches the fused
/// [`crate::kernel::sdpa_backward`], reconstructing the softmax weights from
/// the saved stats — so the `seq_q × seq_k` score matrix is materialised on
/// *neither* pass. The composed path is kept as [`sdpa_var_composed`], the
/// differential-test oracle.
///
/// `tape` owns the graph the `q`/`k`/`v` vars belong to; the custom node is
/// pushed onto it with `[q, k, v]` as inputs.
pub fn sdpa_var<T: DiffScalar + ToF64>(
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
    let dv = vshape[1];
    if dk != d {
        return Err(bad("sdpa_var: K head dim must equal Q head dim"));
    }
    if vshape[0] != seq_k {
        return Err(bad("sdpa_var: V rows must equal K rows (seq_k)"));
    }

    let gpu = q.value().gpu().clone();
    let scale = opts.resolve_scale(d);
    let kv_len = opts.resolve_kv_len(seq_k);

    // Forward: fused online-softmax kernel over f32, producing O and the per-row
    // (m, l) stats. `DiffScalar` is f32 in practice, but stay generic by moving
    // through host f32 vectors (the fused kernel is f32-only). Capture the
    // forward tensors as f32 host vecs for the backward closure.
    let q_f32 = to_f32_host(&q.value())?;
    let k_f32 = to_f32_host(&k.value())?;
    let v_f32 = to_f32_host(&v.value())?;

    let qa = Array::from_slice(&gpu, &q_f32, &[seq_q, d]).map_err(AutogradError::from)?;
    let ka = Array::from_slice(&gpu, &k_f32, &[seq_k, d]).map_err(AutogradError::from)?;
    let va = Array::from_slice(&gpu, &v_f32, &[seq_k, dv]).map_err(AutogradError::from)?;
    let fwd = scaled_dot_product_attention(&gpu, &qa, &ka, &va, opts)?;
    let o_f32 = fwd.output.to_vec().map_err(AutogradError::from)?;
    let stats_f32 = fwd.stats.to_vec().map_err(AutogradError::from)?;

    // The forward value, back in `T` for the tape.
    let out_t: Vec<T> = o_f32.iter().map(|&x| T::from_f64(x as f64)).collect();
    let out_arr = Array::from_slice(&gpu, &out_t, &[seq_q, dv]).map_err(AutogradError::from)?;

    // Backward closure: upstream grad g == dO (shaped [seq_q, dv]). Upload the
    // captured forward tensors + g to fresh fields, dispatch the fused
    // backward, read (dq, dk, dv) back, and return them (in T) for [q, k, v].
    let gpu_b = gpu.clone();
    let backward = move |g: &Array<T>| -> Result<Vec<Array<T>>, AutogradError> {
        let do_f32 = to_f32_host(g)?;

        let qf = gpu_b.field::<f32>(seq_q * d).map_err(lift)?;
        let kf = gpu_b.field::<f32>(seq_k * d).map_err(lift)?;
        let vf = gpu_b.field::<f32>(seq_k * dv).map_err(lift)?;
        let of = gpu_b.field::<f32>(seq_q * dv).map_err(lift)?;
        let sf = gpu_b.field::<f32>(seq_q * 2).map_err(lift)?;
        let dof = gpu_b.field::<f32>(seq_q * dv).map_err(lift)?;
        let dqf = gpu_b.field::<f32>(seq_q * d).map_err(lift)?;
        let dkf = gpu_b.field::<f32>(seq_k * d).map_err(lift)?;
        let dvf = gpu_b.field::<f32>(seq_k * dv).map_err(lift)?;
        qf.write(&q_f32).map_err(lift)?;
        kf.write(&k_f32).map_err(lift)?;
        vf.write(&v_f32).map_err(lift)?;
        of.write(&o_f32).map_err(lift)?;
        sf.write(&stats_f32).map_err(lift)?;
        dof.write(&do_f32).map_err(lift)?;

        crate::kernel::sdpa_backward(
            &gpu_b,
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
            &dof,
            &dqf,
            &dkf,
            &dvf,
        )
        .map_err(lift)?;

        let dq = f32_field_to_array::<T>(&gpu_b, &dqf, &[seq_q, d])?;
        let dk = f32_field_to_array::<T>(&gpu_b, &dkf, &[seq_k, d])?;
        let dv_g = f32_field_to_array::<T>(&gpu_b, &dvf, &[seq_k, dv])?;
        Ok(vec![dq, dk, dv_g])
    };

    Ok(tape.custom_vjp(&[q, k, v], out_arr, backward))
}

/// Materialise an `Array<T>` (contiguous) into a host `Vec<f32>` — the bridge
/// into the f32-only fused kernels. `T` is `f32` in practice (`DiffScalar`),
/// but going through `ToF64` keeps the call generic.
pub(crate) fn to_f32_host<T: DiffScalar + ToF64>(a: &Array<T>) -> Result<Vec<f32>, AutogradError> {
    let host = a
        .contiguous()
        .map_err(AutogradError::from)?
        .to_vec()
        .map_err(AutogradError::from)?;
    Ok(host.into_iter().map(|x| x.to_f64() as f32).collect())
}

/// Read an `f32` field back into an `Array<T>` of `shape` (via `T::from_f64`).
pub(crate) fn f32_field_to_array<T: DiffScalar>(
    gpu: &Gpu,
    f: &quanta_core::Field<f32>,
    shape: &[usize],
) -> Result<Array<T>, AutogradError> {
    let host = f.read().map_err(lift)?;
    let t_host: Vec<T> = host.iter().map(|&x| T::from_f64(x as f64)).collect();
    Array::from_slice(gpu, &t_host, shape).map_err(AutogradError::from)
}

/// The **composed-VJP** scaled dot-product attention — the reference oracle the
/// fused [`sdpa_var`] is differential-tested against, and a fallback for callers
/// that want the materialising backward. Records the explicit ops
/// (`scale·QKᵀ → mask → softmax → ·V`) so backward flows through the existing
/// `quanta-autograd` VJPs, rematerialising the `seq_q × seq_k` score matrix on
/// the backward path. Same forward *value* as [`sdpa_var`]; prefer `sdpa_var`
/// in production (it never materialises the score matrix on either pass).
#[doc(hidden)]
pub fn sdpa_var_composed<T: DiffScalar>(
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
