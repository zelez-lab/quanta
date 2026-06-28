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
    fn tape_handle(&self) -> Tape<T> {
        // Tape wraps the same Rc; reconstruct a handle from it.
        Tape::from_inner(Rc::clone(&self.tape))
    }

    /// Elementwise negation.
    pub fn neg(&self) -> Result<Var<T>, AutogradError> {
        let y = self.value().neg()?;
        Ok(self.tape_handle().push(Op::Neg(self.id), y))
    }

    /// Elementwise add.
    pub fn add(&self, rhs: &Var<T>) -> Result<Var<T>, AutogradError> {
        self.same_tape(rhs)?;
        let y = self.value().add(&rhs.value())?;
        Ok(self.tape_handle().push(Op::Add(self.id, rhs.id), y))
    }

    /// Elementwise subtract.
    pub fn sub(&self, rhs: &Var<T>) -> Result<Var<T>, AutogradError> {
        self.same_tape(rhs)?;
        let y = self.value().sub(&rhs.value())?;
        Ok(self.tape_handle().push(Op::Sub(self.id, rhs.id), y))
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

    /// Sum of all elements → a 1-element (scalar) `Var`. The natural way to
    /// produce a scalar loss to call `grad` on.
    pub fn sum(&self) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let s = x.sum()?;
        let in_shape = x.shape().to_vec();
        let scalar = Array::full(x.gpu(), s, &[1])?;
        Ok(self.tape_handle().push(Op::Sum(self.id, in_shape), scalar))
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
