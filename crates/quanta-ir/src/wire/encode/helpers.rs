//! Scalar, enum, const, param, and register encode helpers.

use crate::{AtomicOp, BinOp, CmpOp, ConstValue, KernelParam, MathFn, Reg, ScalarType, UnaryOp};

use super::header::Writer;

// ---------------------------------------------------------------------------
// ScalarType  (12 variants, tags 0..11)
// ---------------------------------------------------------------------------

pub(crate) fn write_scalar_type(w: &mut Writer, ty: &ScalarType) {
    let tag: u8 = match ty {
        ScalarType::F16 => 0,
        ScalarType::F32 => 1,
        ScalarType::F64 => 2,
        ScalarType::U8 => 3,
        ScalarType::U16 => 4,
        ScalarType::U32 => 5,
        ScalarType::U64 => 6,
        ScalarType::I8 => 7,
        ScalarType::I16 => 8,
        ScalarType::I32 => 9,
        ScalarType::I64 => 10,
        ScalarType::Bool => 11,
        // BF16 appended (tag 12) so the existing tags stay stable.
        ScalarType::BF16 => 12,
    };
    w.u8(tag);
}

// ---------------------------------------------------------------------------
// BinOp  (14 variants, tags 0..13)
// ---------------------------------------------------------------------------

pub(in crate::wire) fn write_binop(w: &mut Writer, op: &BinOp) {
    let tag: u8 = match op {
        BinOp::Add => 0,
        BinOp::Sub => 1,
        BinOp::Mul => 2,
        BinOp::Div => 3,
        BinOp::Rem => 4,
        BinOp::BitAnd => 5,
        BinOp::BitOr => 6,
        BinOp::BitXor => 7,
        BinOp::Shl => 8,
        BinOp::Shr => 9,
        BinOp::SatAdd => 10,
        BinOp::SatSub => 11,
        BinOp::Rotl => 12,
        BinOp::Rotr => 13,
    };
    w.u8(tag);
}

// ---------------------------------------------------------------------------
// UnaryOp  (3 variants, tags 0..2)
// ---------------------------------------------------------------------------

pub(in crate::wire) fn write_unaryop(w: &mut Writer, op: &UnaryOp) {
    let tag: u8 = match op {
        UnaryOp::Neg => 0,
        UnaryOp::BitNot => 1,
        UnaryOp::LogicalNot => 2,
    };
    w.u8(tag);
}

// ---------------------------------------------------------------------------
// CmpOp  (6 variants, tags 0..5)
// ---------------------------------------------------------------------------

pub(in crate::wire) fn write_cmpop(w: &mut Writer, op: &CmpOp) {
    let tag: u8 = match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Lt => 2,
        CmpOp::Le => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
    };
    w.u8(tag);
}

// ---------------------------------------------------------------------------
// AtomicOp  (9 variants, tags 0..8)
// ---------------------------------------------------------------------------

pub(in crate::wire) fn write_atomicop(w: &mut Writer, op: &AtomicOp) {
    let tag: u8 = match op {
        AtomicOp::Add => 0,
        AtomicOp::Sub => 1,
        AtomicOp::Min => 2,
        AtomicOp::Max => 3,
        AtomicOp::And => 4,
        AtomicOp::Or => 5,
        AtomicOp::Xor => 6,
        AtomicOp::Exchange => 7,
        AtomicOp::CompareExchange => 8,
    };
    w.u8(tag);
}

// ---------------------------------------------------------------------------
// MemoryOrder  (5 variants, tags 0..4)
// ---------------------------------------------------------------------------

pub(in crate::wire) fn write_memory_order(w: &mut Writer, order: &crate::MemoryOrder) {
    use crate::MemoryOrder::*;
    let tag: u8 = match order {
        Relaxed => 0,
        Acquire => 1,
        Release => 2,
        AcqRel => 3,
        SeqCst => 4,
    };
    w.u8(tag);
}

// ---------------------------------------------------------------------------
// MathFn  (21 variants, tags 0..20)
// ---------------------------------------------------------------------------

pub(in crate::wire) fn write_mathfn(w: &mut Writer, f: &MathFn) {
    let tag: u8 = match f {
        MathFn::Sin => 0,
        MathFn::Cos => 1,
        MathFn::Tan => 2,
        MathFn::Asin => 3,
        MathFn::Acos => 4,
        MathFn::Atan => 5,
        MathFn::Atan2 => 6,
        MathFn::Sqrt => 7,
        MathFn::Rsqrt => 8,
        MathFn::Exp => 9,
        MathFn::Exp2 => 10,
        MathFn::Log => 11,
        MathFn::Log2 => 12,
        MathFn::Pow => 13,
        MathFn::Abs => 14,
        MathFn::Min => 15,
        MathFn::Max => 16,
        MathFn::Clamp => 17,
        MathFn::Floor => 18,
        MathFn::Ceil => 19,
        MathFn::Round => 20,
        MathFn::Fma => 21,
    };
    w.u8(tag);
}

// ---------------------------------------------------------------------------
// Reg
// ---------------------------------------------------------------------------

pub(in crate::wire) fn write_reg(w: &mut Writer, r: &Reg) {
    w.u32(r.0);
}

// ---------------------------------------------------------------------------
// ConstValue  (8 variants, tags 0..7)
// ---------------------------------------------------------------------------

pub(crate) fn write_const_value(w: &mut Writer, cv: &ConstValue) {
    match cv {
        ConstValue::F16(v) => {
            w.u8(0);
            w.u16(*v);
        }
        // BF16 appended (tag 8) so the existing tags stay stable.
        ConstValue::BF16(v) => {
            w.u8(8);
            w.u16(*v);
        }
        ConstValue::F32(v) => {
            w.u8(1);
            w.f32(*v);
        }
        ConstValue::F64(v) => {
            w.u8(2);
            w.f64(*v);
        }
        ConstValue::U32(v) => {
            w.u8(3);
            w.u32(*v);
        }
        ConstValue::U64(v) => {
            w.u8(4);
            w.u64(*v);
        }
        ConstValue::I32(v) => {
            w.u8(5);
            w.i32(*v);
        }
        ConstValue::I64(v) => {
            w.u8(6);
            w.i64(*v);
        }
        ConstValue::Bool(v) => {
            w.u8(7);
            w.bool_val(*v);
        }
    }
}

// ---------------------------------------------------------------------------
// KernelParam  (6 variants, tags 0..5)
// ---------------------------------------------------------------------------

pub(in crate::wire) fn write_kernel_param(w: &mut Writer, p: &KernelParam) {
    match p {
        KernelParam::FieldRead {
            name,
            slot,
            scalar_type,
        } => {
            w.u8(0);
            w.str(name);
            w.u32(*slot);
            write_scalar_type(w, scalar_type);
        }
        KernelParam::FieldWrite {
            name,
            slot,
            scalar_type,
        } => {
            w.u8(1);
            w.str(name);
            w.u32(*slot);
            write_scalar_type(w, scalar_type);
        }
        KernelParam::Constant {
            name,
            slot,
            scalar_type,
        } => {
            w.u8(2);
            w.str(name);
            w.u32(*slot);
            write_scalar_type(w, scalar_type);
        }
        KernelParam::Texture2DRead {
            name,
            slot,
            scalar_type,
        } => {
            w.u8(3);
            w.str(name);
            w.u32(*slot);
            write_scalar_type(w, scalar_type);
        }
        KernelParam::Texture2DWrite {
            name,
            slot,
            scalar_type,
        } => {
            w.u8(4);
            w.str(name);
            w.u32(*slot);
            write_scalar_type(w, scalar_type);
        }
        KernelParam::Texture3DRead {
            name,
            slot,
            scalar_type,
        } => {
            w.u8(5);
            w.str(name);
            w.u32(*slot);
            write_scalar_type(w, scalar_type);
        }
    }
}
