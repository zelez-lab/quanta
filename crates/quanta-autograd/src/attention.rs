//! Multi-head self-attention — composed from the differentiable primitives, so
//! the tape backpropagates it with no new VJP rule.
//!
//! The heads are laid out as batch dims: a `[B, T, D]` input is projected to
//! Q/K/V, split into `H` heads of width `d = D/H` (`[B, T, H, d]`), transposed
//! to `[B, H, T, d]` so `(B, H)` are the leading batch dims of the batched
//! matmul, scored (`Q·Kᵀ/√d`), masked, softmax-normalised row-wise, mixed with
//! V, then the heads are merged back to `[B, T, D]` and mixed by the output
//! projection. Every step is an existing `Var` op, so the backward pass is free.

use quanta_array::Array;

use crate::error::AutogradError;
use crate::scalar::DiffScalar;
use crate::tape::Var;

impl<T: DiffScalar> Var<T> {
    /// Multiply this `Var` by a scalar constant `s` (broadcast over its shape),
    /// staying on the same tape. `s` is a detached constant, so it contributes
    /// no gradient of its own — only scales the upstream gradient.
    fn scale(&self, s: f64) -> Result<Var<T>, AutogradError> {
        let val = self.value();
        let c = Array::full(val.gpu(), T::from_f64(s), &[1])?
            .broadcast_to(val.shape())?
            .contiguous()?;
        self.mul(&self.tape_handle().var(c))
    }

    /// Row-wise softmax over the **last axis** of an arbitrary-rank tensor.
    /// `softmax`/`log_softmax` are defined on 2-D `[N, C]` matrices, so this
    /// flattens the leading axes to a single `N`, softmaxes, and restores the
    /// original shape — the operation transformer attention needs on the
    /// `[B, H, T, T]` score tensor.
    fn softmax_last_axis(&self) -> Result<Var<T>, AutogradError> {
        let shape = self.value().shape().to_vec();
        let rank = shape.len();
        if rank < 2 {
            // 1-D: treat as a single row.
            let c = shape[0];
            return self.reshape(&[1, c])?.softmax()?.reshape(&shape);
        }
        let c = shape[rank - 1];
        let n: usize = shape[..rank - 1].iter().product();
        self.reshape(&[n, c])?.softmax()?.reshape(&shape)
    }

    /// **Multi-head self-attention.** `self` is the input `[B, T, D]`; `wq`,
    /// `wk`, `wv`, `wo` are `[D, D]` projection weights; `n_heads` splits `D`
    /// into `H` heads of width `d = D/H`. `mask`, if given, is an additive
    /// `[T, T]` bias (e.g. `0` / `−1e9` for a causal mask) broadcast over the
    /// batch and head dims. Returns `[B, T, D]`.
    ///
    /// Composed entirely from existing differentiable ops (projection matmuls,
    /// reshape/transpose head-split, batched `Q·Kᵀ`, additive mask, row-wise
    /// softmax, `attn·V`, head-merge, output projection), so its VJP flows
    /// through the tape automatically.
    pub fn multi_head_attention(
        &self,
        wq: &Var<T>,
        wk: &Var<T>,
        wv: &Var<T>,
        wo: &Var<T>,
        n_heads: usize,
        mask: Option<&Var<T>>,
    ) -> Result<Var<T>, AutogradError> {
        let shape = self.value().shape().to_vec();
        if shape.len() != 3 {
            return Err(AutogradError::from(quanta_array::ArrayError::Gpu(
                quanta_core::QuantaError::invalid_param(
                    "multi_head_attention: input must be 3-D [B, T, D]",
                ),
            )));
        }
        let (b, t, dmodel) = (shape[0], shape[1], shape[2]);
        if n_heads == 0 || dmodel % n_heads != 0 {
            return Err(AutogradError::from(quanta_array::ArrayError::Gpu(
                quanta_core::QuantaError::invalid_param(
                    "multi_head_attention: D must be divisible by n_heads",
                ),
            )));
        }
        let h = n_heads;
        let hd = dmodel / h; // per-head width
        let scale = 1.0 / (hd as f64).sqrt();

        // Project: [B,T,D] · [D,D] → [B,T,D] (batched matmul broadcasts wq over B).
        let q = self.matmul(wq)?;
        let k = self.matmul(wk)?;
        let v = self.matmul(wv)?;

        // Split into heads: [B,T,D] → [B,T,H,hd] → [B,H,T,hd].
        let split = |x: &Var<T>| -> Result<Var<T>, AutogradError> {
            x.reshape(&[b, t, h, hd])?.transpose(1, 2)
        };
        let qh = split(&q)?; // [B,H,T,hd]
        let kh = split(&k)?;
        let vh = split(&v)?;

        // scores = Q·Kᵀ / √hd  → [B,H,T,T]
        let kt = kh.transpose(2, 3)?; // [B,H,hd,T]
        let mut scores = qh.matmul(&kt)?.scale(scale)?; // [B,H,T,T]

        // additive mask [T,T] → broadcast over [B,H,·,·] via [1,1,T,T]
        if let Some(m) = mask {
            let m4 = m.reshape(&[1, 1, t, t])?;
            scores = scores.add(&m4)?;
        }

        // row-wise softmax over the last axis, then mix with V.
        let attn = scores.softmax_last_axis()?; // [B,H,T,T]
        let ctx = attn.matmul(&vh)?; // [B,H,T,hd]

        // merge heads: [B,H,T,hd] → [B,T,H,hd] → [B,T,D]
        let merged = ctx.transpose(1, 2)?.reshape(&[b, t, dmodel])?;

        // output projection
        merged.matmul(wo)
    }
}
