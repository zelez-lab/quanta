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
//!   the tape ([`quanta_array::autograd::Tape::custom_vjp`]) that dispatches the fused
//!   [`crate::kernel::sdpa_backward`] — reconstructing the softmax weights from
//!   the saved stats (T9204), so the `seq_q × seq_k` matrix is never
//!   materialised on *either* pass. The old composed path
//!   ([`sdpa_var_composed`]) is retained as the differential-test oracle.

use quanta_array::autograd::{AutogradError, DiffScalar, Tape, Var};
use quanta_array::{Array, ToF64};
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

    // Device-resident (the form the dependency-aware lane makes
    // profitable — hazard runs let independent heads/cases encode
    // barrier-free instead of serializing): bind the inputs' own
    // buffers, adopt the kernel's outputs. History: with the OLD
    // serial batch encoder this exact shape measured ~5x slower than
    // a host bridge; the concurrent encoder + write-mask runs is what
    // flipped it.
    let qi = f32_input(gpu, q)?;
    let ki = f32_input(gpu, k)?;
    let vi = f32_input(gpu, v)?;
    let of = gpu.field::<f32>(seq_q * dv).map_err(lift)?;
    let sf = gpu.field::<f32>(seq_q * 2).map_err(lift)?;

    crate::kernel::sdpa_forward(
        gpu,
        seq_q as u32,
        seq_k as u32,
        d as u32,
        dv as u32,
        scale,
        opts.causal,
        kv_len as u32,
        qi.field(),
        ki.field(),
        vi.field(),
        &of,
        &sf,
    )
    .map_err(lift)?;

    let output = Array::from_field(gpu, of, &[seq_q, dv]).map_err(map_err)?;
    let stats = Array::from_field(gpu, sf, &[seq_q, 2]).map_err(map_err)?;
    Ok(SdpaOutput { output, stats })
}

/// **Tape-differentiable scaled dot-product attention.** Single head.
///
/// The returned `Var` carries the attention context `(seq_q, dv)`. The forward
/// runs the fused online-softmax kernel (via [`scaled_dot_product_attention`],
/// saving the `(m, l)` stats); the backward is a **custom VJP node** on the
/// tape ([`quanta_array::autograd::Tape::custom_vjp`]) that dispatches the fused
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

    // Forward: the fused online-softmax kernel, fully device-resident —
    // zero-copy f32 bindings of Q/K/V, output adopted as the tape
    // value, the (m, l) stats field captured by the backward.
    let qi = f32_input(&gpu, &q.value())?;
    let ki = f32_input(&gpu, &k.value())?;
    let vi = f32_input(&gpu, &v.value())?;
    let of = gpu.field::<f32>(seq_q * dv).map_err(lift)?;
    let sf = gpu.field::<f32>(seq_q * 2).map_err(lift)?;
    crate::kernel::sdpa_forward(
        &gpu,
        seq_q as u32,
        seq_k as u32,
        d as u32,
        dv as u32,
        scale,
        opts.causal,
        kv_len as u32,
        qi.field(),
        ki.field(),
        vi.field(),
        &of,
        &sf,
    )
    .map_err(lift)?;
    let out_arr = adopt_f32_field::<T>(&gpu, of, &[seq_q, dv])?;
    // The backward reconstructs the softmax weights from O and the
    // stats — alias the adopted output before the tape takes it.
    let oi = f32_input(&gpu, &out_arr)?;

    // Backward closure: upstream grad g == dO (shaped [seq_q, dv]).
    // Everything it needs is captured device-resident.
    let gpu_b = gpu.clone();
    let backward = move |g: &Array<T>| -> Result<Vec<Array<T>>, AutogradError> {
        let doi = f32_input(&gpu_b, g)?;
        let dqf = gpu_b.field::<f32>(seq_q * d).map_err(lift)?;
        let dkf = gpu_b.field::<f32>(seq_k * d).map_err(lift)?;
        let dvf = gpu_b.field::<f32>(seq_k * dv).map_err(lift)?;

        crate::kernel::sdpa_backward(
            &gpu_b,
            seq_q as u32,
            seq_k as u32,
            d as u32,
            dv as u32,
            scale,
            opts.causal,
            kv_len as u32,
            qi.field(),
            ki.field(),
            vi.field(),
            oi.field(),
            &sf,
            doi.field(),
            &dqf,
            &dkf,
            &dvf,
        )
        .map_err(lift)?;

        let dq = adopt_f32_field::<T>(&gpu_b, dqf, &[seq_q, d])?;
        let dk = adopt_f32_field::<T>(&gpu_b, dkf, &[seq_k, d])?;
        let dv_g = adopt_f32_field::<T>(&gpu_b, dvf, &[seq_k, dv])?;
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

/// An f32 binding for a fused-kernel input: the array's own backing
/// buffer when `T` IS f32 and the value is contiguous (zero-copy —
/// no staging field, no host round trip), a staged converted field
/// otherwise. Owns whatever storage the binding needs; keep it alive
/// until the dispatch call returns (from encode onward the deferred
/// lane keeps the buffer alive and ordered).
pub(crate) enum F32Input {
    Zero(Array<f32>),
    Staged(quanta_core::Field<f32>),
}

impl F32Input {
    pub(crate) fn field(&self) -> &quanta_core::Field<f32> {
        match self {
            F32Input::Zero(a) => a
                .backing_field()
                .expect("F32Input::Zero holds a contiguous array"),
            F32Input::Staged(f) => f,
        }
    }
}

/// Bind `a` as an f32 kernel input (see [`F32Input`]).
pub(crate) fn f32_input<T: DiffScalar + ToF64>(
    gpu: &Gpu,
    a: &Array<T>,
) -> Result<F32Input, AutogradError> {
    let contig = a.contiguous().map_err(AutogradError::from)?;
    if let Some(f32a) = T::as_f32_array(&contig) {
        // Zero-copy only when the buffer is exactly the logical
        // content — a dense prefix view (narrow of a bigger table)
        // stages instead (see `Array::backing_field`).
        if f32a.backing_field().is_some() {
            return Ok(F32Input::Zero(f32a.shallow_clone()));
        }
    }
    let host = to_f32_host(&contig)?;
    let f = gpu.field::<f32>(host.len()).map_err(lift)?;
    f.write(&host).map_err(lift)?;
    Ok(F32Input::Staged(f))
}

/// Adopt an f32 field a fused kernel wrote as the op's `Array<T>`
/// value — zero-copy for f32 (no read-back, no completion point; the
/// lane orders later consumers), element-converted otherwise.
pub(crate) fn adopt_f32_field<T: DiffScalar>(
    gpu: &Gpu,
    f: quanta_core::Field<f32>,
    shape: &[usize],
) -> Result<Array<T>, AutogradError> {
    T::array_from_f32_field(gpu, f, shape).map_err(AutogradError::from)
}

/// The **composed-VJP** scaled dot-product attention — the reference oracle the
/// fused [`sdpa_var`] is differential-tested against, and a fallback for callers
/// that want the materialising backward. Records the explicit ops
/// (`scale·QKᵀ → mask → softmax → ·V`) so backward flows through the existing
/// `quanta-array` autograd VJPs, rematerialising the `seq_q × seq_k` score matrix on
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
