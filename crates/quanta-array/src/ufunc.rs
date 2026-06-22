//! Elementwise ufuncs — the runtime-kernel engine.
//!
//! A ufunc call builds a [`quanta_ir::KernelDef`] at runtime, JIT-compiles
//! it via `quanta`'s `wave_jit`, dispatches one quark per output element,
//! and returns a fresh contiguous `Array`.
//!
//! This module is the **same-shape, contiguous** path (commit 2):
//! `out[i] = op(a[i])` / `op(a[i], b[i])`. Broadcasting over differing
//! shapes lands in the next commit (strided-index variant).

use core::ops::{Add, Div, Mul, Sub};

use quanta::GpuType;
use quanta_ir::{
    BinOp as IrBinOp, KernelDef, KernelOp, KernelParam, MathFn, Reg, UnaryOp as IrUnaryOp,
    serialize_kernel,
};

use crate::array::Array;
use crate::error::ArrayError;

impl<T: GpuType> Array<T> {
    /// Realize this array as a fresh contiguous row-major `Array` (copies
    /// through the device if it's a strided/transposed view). A no-op clone
    /// of the backing buffer when already contiguous.
    pub fn contiguous(&self) -> Result<Array<T>, ArrayError> {
        // Host round-trip: download in logical order, re-upload row-major.
        // (A copy *kernel* over strided indices is a later optimization;
        // correctness first.)
        let data = self.to_vec()?;
        Array::from_slice(self.gpu(), &data, self.shape())
    }

    /// Apply a unary IR op (`UnaryOp::Neg`) elementwise.
    fn unary_op(&self, op: IrUnaryOp) -> Result<Array<T>, ArrayError> {
        let src = self.contiguous_if_needed()?;
        let n = src.len();
        let ty = T::scalar_type();
        let def = KernelDef {
            name: "qa_unary".into(),
            params: vec![
                KernelParam::FieldRead {
                    name: "a".into(),
                    slot: 0,
                    scalar_type: ty,
                },
                KernelParam::FieldWrite {
                    name: "out".into(),
                    slot: 1,
                    scalar_type: ty,
                },
            ],
            body: vec![
                KernelOp::QuarkId { dst: Reg(0) },
                KernelOp::Load {
                    dst: Reg(1),
                    field: 0,
                    index: Reg(0),
                    ty,
                },
                KernelOp::UnaryOp {
                    dst: Reg(2),
                    a: Reg(1),
                    op,
                    ty,
                },
                KernelOp::Store {
                    field: 1,
                    index: Reg(0),
                    src: Reg(2),
                    ty,
                },
            ],
            body_source: None,
            next_reg: 3,
            opt_level: 0,
            device_sources: vec![],
            device_functions: vec![],
            workgroup_size: [1, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        };
        src.dispatch_unary(&def, n)
    }

    /// Apply a unary math function (`MathFn::Sqrt`, …) elementwise.
    fn math1(&self, func: MathFn) -> Result<Array<T>, ArrayError> {
        let src = self.contiguous_if_needed()?;
        let n = src.len();
        let ty = T::scalar_type();
        let def = KernelDef {
            name: "qa_math1".into(),
            params: vec![
                KernelParam::FieldRead {
                    name: "a".into(),
                    slot: 0,
                    scalar_type: ty,
                },
                KernelParam::FieldWrite {
                    name: "out".into(),
                    slot: 1,
                    scalar_type: ty,
                },
            ],
            body: vec![
                KernelOp::QuarkId { dst: Reg(0) },
                KernelOp::Load {
                    dst: Reg(1),
                    field: 0,
                    index: Reg(0),
                    ty,
                },
                KernelOp::MathCall {
                    dst: Reg(2),
                    func,
                    args: vec![Reg(1)],
                    ty,
                },
                KernelOp::Store {
                    field: 1,
                    index: Reg(0),
                    src: Reg(2),
                    ty,
                },
            ],
            body_source: None,
            next_reg: 3,
            opt_level: 0,
            device_sources: vec![],
            device_functions: vec![],
            workgroup_size: [1, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        };
        src.dispatch_unary(&def, n)
    }

    /// Apply a binary IR op (`BinOp::Add`, …) elementwise. Both arrays must
    /// have the same shape (broadcasting lands in the next commit).
    fn binary_op(&self, rhs: &Array<T>, op: IrBinOp) -> Result<Array<T>, ArrayError> {
        if self.shape() != rhs.shape() {
            return Err(ArrayError::LengthMismatch {
                expected: self.len(),
                got: rhs.len(),
            });
        }
        let a = self.contiguous_if_needed()?;
        let b = rhs.contiguous_if_needed()?;
        let n = a.len();
        let ty = T::scalar_type();
        let def = KernelDef {
            name: "qa_binary".into(),
            params: vec![
                KernelParam::FieldRead {
                    name: "a".into(),
                    slot: 0,
                    scalar_type: ty,
                },
                KernelParam::FieldRead {
                    name: "b".into(),
                    slot: 1,
                    scalar_type: ty,
                },
                KernelParam::FieldWrite {
                    name: "out".into(),
                    slot: 2,
                    scalar_type: ty,
                },
            ],
            body: vec![
                KernelOp::QuarkId { dst: Reg(0) },
                KernelOp::Load {
                    dst: Reg(1),
                    field: 0,
                    index: Reg(0),
                    ty,
                },
                KernelOp::Load {
                    dst: Reg(2),
                    field: 1,
                    index: Reg(0),
                    ty,
                },
                KernelOp::BinOp {
                    dst: Reg(3),
                    a: Reg(1),
                    b: Reg(2),
                    op,
                    ty,
                },
                KernelOp::Store {
                    field: 2,
                    index: Reg(0),
                    src: Reg(3),
                    ty,
                },
            ],
            body_source: None,
            next_reg: 4,
            opt_level: 0,
            device_sources: vec![],
            device_functions: vec![],
            workgroup_size: [1, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        };
        a.dispatch_binary(&b, &def, n)
    }

    /// Apply a binary math function (`MathFn::Pow`, …) elementwise.
    fn math2(&self, rhs: &Array<T>, func: MathFn) -> Result<Array<T>, ArrayError> {
        if self.shape() != rhs.shape() {
            return Err(ArrayError::LengthMismatch {
                expected: self.len(),
                got: rhs.len(),
            });
        }
        let a = self.contiguous_if_needed()?;
        let b = rhs.contiguous_if_needed()?;
        let n = a.len();
        let ty = T::scalar_type();
        let def = KernelDef {
            name: "qa_math2".into(),
            params: vec![
                KernelParam::FieldRead {
                    name: "a".into(),
                    slot: 0,
                    scalar_type: ty,
                },
                KernelParam::FieldRead {
                    name: "b".into(),
                    slot: 1,
                    scalar_type: ty,
                },
                KernelParam::FieldWrite {
                    name: "out".into(),
                    slot: 2,
                    scalar_type: ty,
                },
            ],
            body: vec![
                KernelOp::QuarkId { dst: Reg(0) },
                KernelOp::Load {
                    dst: Reg(1),
                    field: 0,
                    index: Reg(0),
                    ty,
                },
                KernelOp::Load {
                    dst: Reg(2),
                    field: 1,
                    index: Reg(0),
                    ty,
                },
                KernelOp::MathCall {
                    dst: Reg(3),
                    func,
                    args: vec![Reg(1), Reg(2)],
                    ty,
                },
                KernelOp::Store {
                    field: 2,
                    index: Reg(0),
                    src: Reg(3),
                    ty,
                },
            ],
            body_source: None,
            next_reg: 4,
            opt_level: 0,
            device_sources: vec![],
            device_functions: vec![],
            workgroup_size: [1, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        };
        a.dispatch_binary(&b, &def, n)
    }

    // ── named ufunc methods ──────────────────────────────────────────────

    /// Elementwise negation.
    pub fn neg(&self) -> Result<Array<T>, ArrayError> {
        self.unary_op(IrUnaryOp::Neg)
    }
    /// Elementwise absolute value.
    pub fn abs(&self) -> Result<Array<T>, ArrayError> {
        self.math1(MathFn::Abs)
    }
    /// Elementwise square root.
    pub fn sqrt(&self) -> Result<Array<T>, ArrayError> {
        self.math1(MathFn::Sqrt)
    }
    /// Elementwise natural exponential.
    pub fn exp(&self) -> Result<Array<T>, ArrayError> {
        self.math1(MathFn::Exp)
    }
    /// Elementwise natural logarithm.
    pub fn log(&self) -> Result<Array<T>, ArrayError> {
        self.math1(MathFn::Log)
    }
    /// Elementwise sine.
    pub fn sin(&self) -> Result<Array<T>, ArrayError> {
        self.math1(MathFn::Sin)
    }
    /// Elementwise cosine.
    pub fn cos(&self) -> Result<Array<T>, ArrayError> {
        self.math1(MathFn::Cos)
    }
    /// Elementwise floor.
    pub fn floor(&self) -> Result<Array<T>, ArrayError> {
        self.math1(MathFn::Floor)
    }
    /// Elementwise ceil.
    pub fn ceil(&self) -> Result<Array<T>, ArrayError> {
        self.math1(MathFn::Ceil)
    }

    /// Elementwise add (same shape).
    pub fn add(&self, rhs: &Array<T>) -> Result<Array<T>, ArrayError> {
        self.binary_op(rhs, IrBinOp::Add)
    }
    /// Elementwise subtract (same shape).
    pub fn sub(&self, rhs: &Array<T>) -> Result<Array<T>, ArrayError> {
        self.binary_op(rhs, IrBinOp::Sub)
    }
    /// Elementwise multiply (same shape).
    pub fn mul(&self, rhs: &Array<T>) -> Result<Array<T>, ArrayError> {
        self.binary_op(rhs, IrBinOp::Mul)
    }
    /// Elementwise divide (same shape).
    pub fn div(&self, rhs: &Array<T>) -> Result<Array<T>, ArrayError> {
        self.binary_op(rhs, IrBinOp::Div)
    }
    /// Elementwise minimum (same shape).
    pub fn minimum(&self, rhs: &Array<T>) -> Result<Array<T>, ArrayError> {
        self.math2(rhs, MathFn::Min)
    }
    /// Elementwise maximum (same shape).
    pub fn maximum(&self, rhs: &Array<T>) -> Result<Array<T>, ArrayError> {
        self.math2(rhs, MathFn::Max)
    }
    /// Elementwise power (`self ** rhs`, same shape).
    pub fn pow(&self, rhs: &Array<T>) -> Result<Array<T>, ArrayError> {
        self.math2(rhs, MathFn::Pow)
    }

    // ── dispatch plumbing ────────────────────────────────────────────────

    /// Realize into a contiguous array only when the layout isn't already
    /// the linear fast path; otherwise reuse self (cheap Arc share).
    fn contiguous_if_needed(&self) -> Result<Array<T>, ArrayError> {
        if self.is_contiguous() {
            Ok(self.shallow_clone())
        } else {
            self.contiguous()
        }
    }

    fn dispatch_unary(&self, def: &KernelDef, n: usize) -> Result<Array<T>, ArrayError> {
        let out = self.gpu().field::<T>(n)?;
        let bytes = serialize_kernel(def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, self.field_ref());
        wave.bind(1, &out);
        self.gpu().dispatch(&wave, n as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            quanta_tensor::Layout::row_major(self.shape())?,
        ))
    }

    fn dispatch_binary(
        &self,
        rhs: &Array<T>,
        def: &KernelDef,
        n: usize,
    ) -> Result<Array<T>, ArrayError> {
        let out = self.gpu().field::<T>(n)?;
        let bytes = serialize_kernel(def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, self.field_ref());
        wave.bind(1, rhs.field_ref());
        wave.bind(2, &out);
        self.gpu().dispatch(&wave, n as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            quanta_tensor::Layout::row_major(self.shape())?,
        ))
    }
}

// Operator-trait sugar: `&a + &b` etc. (panics on shape/gpu error — use
// the `.add()` methods for the fallible form).
macro_rules! impl_binop_trait {
    ($trait:ident, $method:ident, $call:ident) => {
        impl<T: GpuType> $trait<&Array<T>> for &Array<T> {
            type Output = Array<T>;
            fn $method(self, rhs: &Array<T>) -> Array<T> {
                self.$call(rhs).expect("quanta-array elementwise op failed")
            }
        }
    };
}
impl_binop_trait!(Add, add, add);
impl_binop_trait!(Sub, sub, sub);
impl_binop_trait!(Mul, mul, mul);
impl_binop_trait!(Div, div, div);
