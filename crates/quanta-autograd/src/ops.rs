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

    /// Reshape to `shape` (same element count) — a zero-copy view. The gradient
    /// passes straight through, reshaped back to this var's shape. Errors if the
    /// element counts don't match (delegated to `Array::reshape`).
    pub fn reshape(&self, shape: &[usize]) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let in_shape = x.shape().to_vec();
        let y = x.contiguous()?.reshape(shape)?;
        Ok(self.tape_handle().push(Op::Reshape(self.id, in_shape), y))
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
