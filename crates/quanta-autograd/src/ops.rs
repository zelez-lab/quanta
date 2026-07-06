//! Differentiable forward ops on [`Var`]. Each runs the real `quanta-array`
//! kernel now (define-by-run) and records a [`Op`] node so the backward pass
//! can apply the matching VJP. Binary ops require both operands to be on the
//! same tape.

use std::rc::Rc;

use quanta_array::Array;

use crate::error::AutogradError;
use crate::scalar::DiffScalar;
use crate::tape::{Op, Tape, Var};

impl<T: DiffScalar> Var<T> {
    fn same_tape(&self, other: &Var<T>) -> Result<(), AutogradError> {
        if Rc::ptr_eq(&self.tape, &other.tape) {
            Ok(())
        } else {
            Err(AutogradError::ForeignVar)
        }
    }

    /// A `Tape` handle pointing at this var's tape (to push new nodes).
    pub(crate) fn tape_handle(&self) -> Tape<T> {
        // Tape wraps the same Rc; reconstruct a handle from it.
        Tape::from_inner(Rc::clone(&self.tape))
    }

    /// Elementwise negation.
    pub fn neg(&self) -> Result<Var<T>, AutogradError> {
        let y = self.value().neg()?;
        Ok(self.tape_handle().push(Op::Neg(self.id), y))
    }

    /// Elementwise add (broadcasting; gradients un-broadcast in backward).
    pub fn add(&self, rhs: &Var<T>) -> Result<Var<T>, AutogradError> {
        self.same_tape(rhs)?;
        let av = self.value();
        let bv = rhs.value();
        let (sa, sb) = (av.shape().to_vec(), bv.shape().to_vec());
        let y = av.add(&bv)?;
        Ok(self.tape_handle().push(Op::Add(self.id, rhs.id, sa, sb), y))
    }

    /// Elementwise subtract (broadcasting; gradients un-broadcast in backward).
    pub fn sub(&self, rhs: &Var<T>) -> Result<Var<T>, AutogradError> {
        self.same_tape(rhs)?;
        let av = self.value();
        let bv = rhs.value();
        let (sa, sb) = (av.shape().to_vec(), bv.shape().to_vec());
        let y = av.sub(&bv)?;
        Ok(self.tape_handle().push(Op::Sub(self.id, rhs.id, sa, sb), y))
    }

    /// Elementwise multiply.
    pub fn mul(&self, rhs: &Var<T>) -> Result<Var<T>, AutogradError> {
        self.same_tape(rhs)?;
        let av = self.value();
        let bv = rhs.value();
        let y = av.mul(&bv)?;
        Ok(self.tape_handle().push(Op::Mul(self.id, rhs.id, av, bv), y))
    }

    /// Elementwise divide.
    pub fn div(&self, rhs: &Var<T>) -> Result<Var<T>, AutogradError> {
        self.same_tape(rhs)?;
        let av = self.value();
        let bv = rhs.value();
        let y = av.div(&bv)?;
        Ok(self.tape_handle().push(Op::Div(self.id, rhs.id, av, bv), y))
    }

    /// Elementwise natural exponential.
    pub fn exp(&self) -> Result<Var<T>, AutogradError> {
        let y = self.value().exp()?;
        // Capture y (= exp x) for the VJP g·y.
        Ok(self
            .tape_handle()
            .push(Op::Exp(self.id, y.shallow_clone()), y))
    }

    /// Elementwise natural logarithm.
    pub fn log(&self) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let y = x.log()?;
        Ok(self.tape_handle().push(Op::Log(self.id, x), y))
    }

    /// Elementwise square root.
    pub fn sqrt(&self) -> Result<Var<T>, AutogradError> {
        let y = self.value().sqrt()?;
        Ok(self
            .tape_handle()
            .push(Op::Sqrt(self.id, y.shallow_clone()), y))
    }

    /// ReLU: `max(x, 0)`. VJP `g·[x>0]` (subgradient 0 at the kink).
    pub fn relu(&self) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let zeros = Array::full(x.gpu(), T::ZERO, &[1])?.broadcast_to(x.shape())?;
        let y = x.maximum(&zeros)?;
        Ok(self.tape_handle().push(Op::Relu(self.id, x), y))
    }

    /// Sigmoid `σ(x) = 1/(1 + e⁻ˣ)`. VJP `g·y·(1−y)` (captures output y).
    pub fn sigmoid(&self) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let one = Array::full(x.gpu(), T::ONE, &[1])?.broadcast_to(x.shape())?;
        // y = 1 / (1 + exp(-x))
        let denom = one.add(&x.neg()?.exp()?)?;
        let y = one.div(&denom)?;
        Ok(self
            .tape_handle()
            .push(Op::Sigmoid(self.id, y.shallow_clone()), y))
    }

    /// Tanh. VJP `g·(1−y²)` (captures output y). Computed from exp (no array
    /// `tanh` primitive): `(e^x − e⁻ˣ)/(e^x + e⁻ˣ)`.
    pub fn tanh(&self) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let ex = x.exp()?;
        let enx = x.neg()?.exp()?;
        let y = ex.sub(&enx)?.div(&ex.add(&enx)?)?;
        Ok(self
            .tape_handle()
            .push(Op::Tanh(self.id, y.shallow_clone()), y))
    }

    /// A constant `Var` (no upstream gradient) holding `v` broadcast to this
    /// var's shape — used to fold scalar coefficients into a composed op.
    fn scalar_const(&self, v: f64) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let arr = Array::full(x.gpu(), T::from_f64(v), &[1])?.broadcast_to(x.shape())?;
        Ok(self.tape_handle().var(arr))
    }

    /// SiLU / swish `x · σ(x)`. Composed from `sigmoid`/`mul`, so the VJP flows
    /// through the tape automatically.
    pub fn silu(&self) -> Result<Var<T>, AutogradError> {
        self.mul(&self.sigmoid()?)
    }

    /// GELU (tanh approximation, GPT-2 form):
    /// `0.5·x·(1 + tanh( √(2/π)·(x + 0.044715·x³) ))`. Composed from
    /// `mul`/`add`/`tanh` — no dedicated `Op` variant, VJP is automatic.
    pub fn gelu(&self) -> Result<Var<T>, AutogradError> {
        // c = √(2/π)
        let c = self.scalar_const(0.797_884_560_802_865_4)?;
        let half = self.scalar_const(0.5)?;
        let k = self.scalar_const(0.044715)?;
        let one = self.scalar_const(1.0)?;
        let x3 = self.mul(self)?.mul(self)?;
        let inner = self.add(&x3.mul(&k)?)?;
        let tanh = inner.mul(&c)?.tanh()?;
        self.mul(&half)?.mul(&one.add(&tanh)?)
    }

    /// Reshape to `shape` (same element count) — a zero-copy view. The gradient
    /// passes straight through, reshaped back to this var's shape. Errors if the
    /// element counts don't match (delegated to `Array::reshape`).
    pub fn reshape(&self, shape: &[usize]) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let in_shape = x.shape().to_vec();
        let y = x.contiguous()?.reshape(shape)?;
        Ok(self.tape_handle().push(Op::Reshape(self.id, in_shape), y))
    }

    /// Swap two axes — a zero-copy transposing view. Transpose is its own
    /// inverse, so the backward pass just transposes the upstream gradient back
    /// over the same pair of axes. The `Kᵀ` in attention scores, and any place
    /// a computation graph needs a transpose.
    pub fn transpose(&self, d0: usize, d1: usize) -> Result<Var<T>, AutogradError> {
        let y = self.value().transpose(d0, d1)?;
        Ok(self.tape_handle().push(Op::Transpose(self.id, d0, d1), y))
    }

    /// Zero-copy sub-range along axis 0: keep `len` rows starting at `start`.
    /// The forward is a view ([`Array::narrow`]); the gradient scatters back
    /// into a zero tensor of this var's shape at the same offset
    /// ([`Array::pad_axis0`]). This is the differentiable minibatch selector —
    /// `x.narrow(b * bs, bs)` trains on batch `b`. Only axis 0 is supported
    /// (the adjoint kernel is axis-0 only); slicing other axes is a view via
    /// `Array::narrow` but not yet differentiable.
    pub fn narrow(&self, start: usize, len: usize) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let in_shape = x.shape().to_vec();
        let y = x.narrow(0, start, len)?;
        Ok(self
            .tape_handle()
            .push(Op::Narrow(self.id, start, in_shape), y))
    }

    /// Concatenate `parts` along axis 0 (the inverse of `narrow`). All parts
    /// must be on the same tape and share the trailing shape. The gradient of
    /// each part is the slice of the output gradient at that part's window, so
    /// this composes cleanly with minibatch code (e.g. re-assembling batches).
    pub fn concat_axis0(parts: &[&Var<T>]) -> Result<Var<T>, AutogradError> {
        let first = parts.first().ok_or_else(|| {
            AutogradError::from(quanta_array::ArrayError::Gpu(
                quanta::QuantaError::invalid_param("concat_axis0: need at least one var"),
            ))
        })?;
        for p in parts {
            first.same_tape(p)?;
        }
        let arrays: Vec<Array<T>> = parts.iter().map(|p| p.value()).collect();
        let refs: Vec<&Array<T>> = arrays.iter().collect();
        let y = Array::concat_axis0(&refs)?;
        let ids: Vec<usize> = parts.iter().map(|p| p.id).collect();
        let lens: Vec<usize> = arrays.iter().map(|a| a.shape()[0]).collect();
        Ok(first.tape_handle().push(Op::Concat(ids, lens), y))
    }

    /// Row-wise gather: pick column `idx[i]` from each row of this `[N, C]`
    /// table, giving `[N]` where `out[i] = self[i, idx[i]]`. `idx` is a plain
    /// `[N]` `u32` array (labels aren't differentiated). The gradient scatters
    /// back to the gathered columns — the piece a cross-entropy loss needs.
    pub fn gather_rows(&self, idx: &Array<u32>) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let d = x.shape();
        if d.len() != 2 {
            return Err(AutogradError::from(quanta_array::ArrayError::Gpu(
                quanta::QuantaError::invalid_param("gather_rows: table must be 2-D [N, C]"),
            )));
        }
        let c = d[1];
        let y = x.gather_rows(idx)?;
        Ok(self
            .tape_handle()
            .push(Op::GatherRows(self.id, idx.shallow_clone(), c), y))
    }

    /// Select elementwise: `out = mask·self + (1−mask)·other`, where `mask` is
    /// a non-differentiable `{0,1}` array (typically from a comparison). The
    /// gradient routes to `self` where the mask is 1 and to `other` where it's
    /// 0 — a differentiable conditional without control flow.
    pub fn where_mask(&self, mask: &Array<T>, other: &Var<T>) -> Result<Var<T>, AutogradError> {
        self.same_tape(other)?;
        let y = mask.where_mask(&self.value(), &other.value())?;
        Ok(self
            .tape_handle()
            .push(Op::Where(self.id, other.id, mask.shallow_clone()), y))
    }

    /// Embedding lookup: select whole rows of this `[V, E]` table by `ids`
    /// `[B]` → `[B, E]` where `out[b] = self[ids[b], :]`. `ids` isn't
    /// differentiated; the gradient scatter-adds each row back to its source
    /// row (repeated ids accumulate) — the sparse embedding update.
    pub fn embedding(&self, ids: &Array<u32>) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let d = x.shape();
        if d.len() != 2 {
            return Err(AutogradError::from(quanta_array::ArrayError::Gpu(
                quanta::QuantaError::invalid_param("embedding: table must be 2-D [V, E]"),
            )));
        }
        let v = d[0];
        let y = x.select_rows(ids)?;
        Ok(self
            .tape_handle()
            .push(Op::Embedding(self.id, ids.shallow_clone(), v), y))
    }

    /// Gather along the last axis: `self [L…, D]`, `idx [L…, K]` → `[L…, K]`
    /// where `out[l…, j] = self[l…, idx[l…, j]]`. The general data-dependent
    /// lookup (embeddings, top-k, attention) — `idx` isn't differentiated; the
    /// gradient scatter-adds back to the gathered positions, so repeated picks
    /// accumulate.
    pub fn gather_last(&self, idx: &Array<u32>) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let d = *x.shape().last().ok_or_else(|| {
            AutogradError::from(quanta_array::ArrayError::Gpu(
                quanta::QuantaError::invalid_param("gather_last: input needs at least one axis"),
            ))
        })?;
        let y = x.gather_last(idx)?;
        Ok(self
            .tape_handle()
            .push(Op::GatherLast(self.id, idx.shallow_clone(), d), y))
    }

    /// Flatten the trailing dims into one, keeping axis 0 as the batch:
    /// `[N, d1, d2, …]` → `[N, d1·d2·…]`. The standard conv/pool → linear bridge.
    pub fn flatten(&self) -> Result<Var<T>, AutogradError> {
        let d = self.value().shape().to_vec();
        if d.is_empty() {
            return Err(AutogradError::from(quanta_array::ArrayError::Gpu(
                quanta::QuantaError::invalid_param("flatten: need at least 1 dim"),
            )));
        }
        let rest: usize = d[1..].iter().product();
        self.reshape(&[d[0], rest])
    }

    /// Sum of all elements → a 1-element (scalar) `Var`. The natural way to
    /// produce a scalar loss to call `grad` on.
    pub fn sum(&self) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let s = x.sum()?;
        let in_shape = x.shape().to_vec();
        let scalar = Array::full(x.gpu(), s, &[1])?;
        Ok(self.tape_handle().push(Op::Sum(self.id, in_shape), scalar))
    }

    /// Sum over `axis`, keeping it as a size-1 dim. Gradient broadcasts back.
    pub fn sum_axis(&self, axis: usize) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let in_shape = x.shape().to_vec();
        let y = x.sum_axis(axis)?;
        Ok(self
            .tape_handle()
            .push(Op::SumAxis(self.id, axis, in_shape), y))
    }

    /// Mean over `axis`, keeping it as a size-1 dim (sum_axis / extent).
    pub fn mean_axis(&self, axis: usize) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let in_shape = x.shape().to_vec();
        let count = in_shape[axis];
        let summed = x.sum_axis(axis)?;
        // y = summed / count
        let inv = Array::full(x.gpu(), T::from_f64(1.0 / count as f64), &[1])?;
        let y = summed.mul(&inv.broadcast_to(summed.shape())?)?;
        Ok(self
            .tape_handle()
            .push(Op::MeanAxis(self.id, axis, in_shape, count), y))
    }

    /// 2-D matrix multiply `self (m×k) · rhs (k×n) → (m×n)`. The VJPs are
    /// themselves matmuls (∂A = G·Bᵀ, ∂B = Aᵀ·G), captured by recording both
    /// forward operands.
    pub fn matmul(&self, rhs: &Var<T>) -> Result<Var<T>, AutogradError> {
        self.same_tape(rhs)?;
        let av = self.value();
        let bv = rhs.value();
        let y = T::array_matmul(&av, &bv)?;
        Ok(self
            .tape_handle()
            .push(Op::Matmul(self.id, rhs.id, av, bv), y))
    }
}
