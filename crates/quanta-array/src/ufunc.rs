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
use crate::scalar::FloatScalar;

impl<T: GpuType> Array<T> {
    /// Realize this array as a fresh contiguous row-major `Array`. Already
    /// contiguous arrays are a cheap `Arc` share; strided / transposed /
    /// broadcast views are gathered **on the device** (no host round-trip).
    pub fn contiguous(&self) -> Result<Array<T>, ArrayError> {
        if self.is_contiguous() {
            Ok(self.shallow_clone())
        } else {
            self.gather_contiguous()
        }
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
            return self.broadcast_binary(rhs, crate::broadcast::Combine::Bin(op));
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
            return self.broadcast_binary(rhs, crate::broadcast::Combine::Math(func));
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

    // ── named arithmetic ufuncs (all numeric dtypes) ─────────────────────

    /// Elementwise negation.
    pub fn neg(&self) -> Result<Array<T>, ArrayError> {
        self.unary_op(IrUnaryOp::Neg)
    }
    /// Elementwise add (broadcasting).
    pub fn add(&self, rhs: &Array<T>) -> Result<Array<T>, ArrayError> {
        self.binary_op(rhs, IrBinOp::Add)
    }
    /// Elementwise subtract (broadcasting).
    pub fn sub(&self, rhs: &Array<T>) -> Result<Array<T>, ArrayError> {
        self.binary_op(rhs, IrBinOp::Sub)
    }
    /// Elementwise multiply (broadcasting).
    pub fn mul(&self, rhs: &Array<T>) -> Result<Array<T>, ArrayError> {
        self.binary_op(rhs, IrBinOp::Mul)
    }
    /// Elementwise divide (broadcasting).
    pub fn div(&self, rhs: &Array<T>) -> Result<Array<T>, ArrayError> {
        self.binary_op(rhs, IrBinOp::Div)
    }

    // ── dispatch plumbing ────────────────────────────────────────────────

    /// Realize into a contiguous array only when the layout isn't already
    /// the linear fast path; otherwise reuse self (cheap Arc share).
    fn contiguous_if_needed(&self) -> Result<Array<T>, ArrayError> {
        self.contiguous_or_self()
    }

    /// Contiguous form of this array, reusing `self` (Arc share) when it is
    /// already contiguous. Shared with the broadcast path. (Now identical to
    /// `contiguous`, which already short-circuits the contiguous case.)
    pub(crate) fn contiguous_or_self(&self) -> Result<Array<T>, ArrayError> {
        self.contiguous()
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

// ── math-function ufuncs (floating-point dtypes only) ─────────────────
//
// Transcendentals and rounding map to `MathFn`, which every backend
// implements for floats only. Restricting them to `FloatScalar` makes
// `int_array.sqrt()` a compile error instead of a wrong result (the
// `compile_fail` doctest in lib.rs pins this contract).
impl<T: FloatScalar> Array<T> {
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
    /// Elementwise minimum (broadcasting).
    pub fn minimum(&self, rhs: &Array<T>) -> Result<Array<T>, ArrayError> {
        self.math2(rhs, MathFn::Min)
    }
    /// Elementwise maximum (broadcasting).
    pub fn maximum(&self, rhs: &Array<T>) -> Result<Array<T>, ArrayError> {
        self.math2(rhs, MathFn::Max)
    }
    /// Elementwise power (`self ** rhs`, broadcasting).
    pub fn pow(&self, rhs: &Array<T>) -> Result<Array<T>, ArrayError> {
        self.math2(rhs, MathFn::Pow)
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
