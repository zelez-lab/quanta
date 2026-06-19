//! Scalar, enum, const, param, and register decode helpers.

use crate::{AtomicOp, BinOp, CmpOp, ConstValue, KernelParam, MathFn, Reg, ScalarType, UnaryOp};

use super::header::Reader;

// ---------------------------------------------------------------------------
// ScalarType  (12 variants, tags 0..11)
// ---------------------------------------------------------------------------

pub(crate) fn read_scalar_type(r: &mut Reader) -> Result<ScalarType, &'static str> {
    let tag = r.u8()?;
    match tag {
        0 => Ok(ScalarType::F16),
        1 => Ok(ScalarType::F32),
        2 => Ok(ScalarType::F64),
        3 => Ok(ScalarType::U8),
        4 => Ok(ScalarType::U16),
        5 => Ok(ScalarType::U32),
        6 => Ok(ScalarType::U64),
        7 => Ok(ScalarType::I8),
        8 => Ok(ScalarType::I16),
        9 => Ok(ScalarType::I32),
        10 => Ok(ScalarType::I64),
        11 => Ok(ScalarType::Bool),
        12 => Ok(ScalarType::BF16),
        _ => Err("invalid ScalarType tag"),
    }
}

// ---------------------------------------------------------------------------
// BinOp  (14 variants, tags 0..13)
// ---------------------------------------------------------------------------

pub(in crate::wire) fn read_binop(r: &mut Reader) -> Result<BinOp, &'static str> {
    let tag = r.u8()?;
    match tag {
        0 => Ok(BinOp::Add),
        1 => Ok(BinOp::Sub),
        2 => Ok(BinOp::Mul),
        3 => Ok(BinOp::Div),
        4 => Ok(BinOp::Rem),
        5 => Ok(BinOp::BitAnd),
        6 => Ok(BinOp::BitOr),
        7 => Ok(BinOp::BitXor),
        8 => Ok(BinOp::Shl),
        9 => Ok(BinOp::Shr),
        10 => Ok(BinOp::SatAdd),
        11 => Ok(BinOp::SatSub),
        12 => Ok(BinOp::Rotl),
        13 => Ok(BinOp::Rotr),
        _ => Err("invalid BinOp tag"),
    }
}

// ---------------------------------------------------------------------------
// UnaryOp  (3 variants, tags 0..2)
// ---------------------------------------------------------------------------

pub(in crate::wire) fn read_unaryop(r: &mut Reader) -> Result<UnaryOp, &'static str> {
    let tag = r.u8()?;
    match tag {
        0 => Ok(UnaryOp::Neg),
        1 => Ok(UnaryOp::BitNot),
        2 => Ok(UnaryOp::LogicalNot),
        _ => Err("invalid UnaryOp tag"),
    }
}

// ---------------------------------------------------------------------------
// CmpOp  (6 variants, tags 0..5)
// ---------------------------------------------------------------------------

pub(in crate::wire) fn read_cmpop(r: &mut Reader) -> Result<CmpOp, &'static str> {
    let tag = r.u8()?;
    match tag {
        0 => Ok(CmpOp::Eq),
        1 => Ok(CmpOp::Ne),
        2 => Ok(CmpOp::Lt),
        3 => Ok(CmpOp::Le),
        4 => Ok(CmpOp::Gt),
        5 => Ok(CmpOp::Ge),
        _ => Err("invalid CmpOp tag"),
    }
}

// ---------------------------------------------------------------------------
// AtomicOp  (9 variants, tags 0..8)
// ---------------------------------------------------------------------------

pub(in crate::wire) fn read_atomicop(r: &mut Reader) -> Result<AtomicOp, &'static str> {
    let tag = r.u8()?;
    match tag {
        0 => Ok(AtomicOp::Add),
        1 => Ok(AtomicOp::Sub),
        2 => Ok(AtomicOp::Min),
        3 => Ok(AtomicOp::Max),
        4 => Ok(AtomicOp::And),
        5 => Ok(AtomicOp::Or),
        6 => Ok(AtomicOp::Xor),
        7 => Ok(AtomicOp::Exchange),
        8 => Ok(AtomicOp::CompareExchange),
        _ => Err("invalid AtomicOp tag"),
    }
}

// ---------------------------------------------------------------------------
// MemoryOrder  (5 variants, tags 0..4)
// ---------------------------------------------------------------------------

pub(in crate::wire) fn read_memory_order(
    r: &mut Reader,
) -> Result<crate::MemoryOrder, &'static str> {
    let tag = r.u8()?;
    match tag {
        0 => Ok(crate::MemoryOrder::Relaxed),
        1 => Ok(crate::MemoryOrder::Acquire),
        2 => Ok(crate::MemoryOrder::Release),
        3 => Ok(crate::MemoryOrder::AcqRel),
        4 => Ok(crate::MemoryOrder::SeqCst),
        _ => Err("invalid MemoryOrder tag"),
    }
}

// ---------------------------------------------------------------------------
// MathFn  (21 variants, tags 0..20)
// ---------------------------------------------------------------------------

pub(in crate::wire) fn read_mathfn(r: &mut Reader) -> Result<MathFn, &'static str> {
    let tag = r.u8()?;
    match tag {
        0 => Ok(MathFn::Sin),
        1 => Ok(MathFn::Cos),
        2 => Ok(MathFn::Tan),
        3 => Ok(MathFn::Asin),
        4 => Ok(MathFn::Acos),
        5 => Ok(MathFn::Atan),
        6 => Ok(MathFn::Atan2),
        7 => Ok(MathFn::Sqrt),
        8 => Ok(MathFn::Rsqrt),
        9 => Ok(MathFn::Exp),
        10 => Ok(MathFn::Exp2),
        11 => Ok(MathFn::Log),
        12 => Ok(MathFn::Log2),
        13 => Ok(MathFn::Pow),
        14 => Ok(MathFn::Abs),
        15 => Ok(MathFn::Min),
        16 => Ok(MathFn::Max),
        17 => Ok(MathFn::Clamp),
        18 => Ok(MathFn::Floor),
        19 => Ok(MathFn::Ceil),
        20 => Ok(MathFn::Round),
        21 => Ok(MathFn::Fma),
        _ => Err("invalid MathFn tag"),
    }
}

// ---------------------------------------------------------------------------
// Reg
// ---------------------------------------------------------------------------

pub(in crate::wire) fn read_reg(r: &mut Reader) -> Result<Reg, &'static str> {
    r.u32().map(Reg)
}

// ---------------------------------------------------------------------------
// ConstValue  (8 variants, tags 0..7)
// ---------------------------------------------------------------------------

pub(crate) fn read_const_value(r: &mut Reader) -> Result<ConstValue, &'static str> {
    let tag = r.u8()?;
    match tag {
        0 => Ok(ConstValue::F16(r.u16()?)),
        1 => Ok(ConstValue::F32(r.f32()?)),
        2 => Ok(ConstValue::F64(r.f64()?)),
        3 => Ok(ConstValue::U32(r.u32()?)),
        4 => Ok(ConstValue::U64(r.u64()?)),
        5 => Ok(ConstValue::I32(r.i32()?)),
        6 => Ok(ConstValue::I64(r.i64()?)),
        7 => Ok(ConstValue::Bool(r.bool_val()?)),
        8 => Ok(ConstValue::BF16(r.u16()?)),
        _ => Err("invalid ConstValue tag"),
    }
}

// ---------------------------------------------------------------------------
// KernelParam  (6 variants, tags 0..5)
// ---------------------------------------------------------------------------

pub(in crate::wire) fn read_kernel_param(r: &mut Reader) -> Result<KernelParam, &'static str> {
    let tag = r.u8()?;
    let name = r.str()?;
    let slot = r.u32()?;
    let scalar_type = read_scalar_type(r)?;
    match tag {
        0 => Ok(KernelParam::FieldRead {
            name,
            slot,
            scalar_type,
        }),
        1 => Ok(KernelParam::FieldWrite {
            name,
            slot,
            scalar_type,
        }),
        2 => Ok(KernelParam::Constant {
            name,
            slot,
            scalar_type,
        }),
        3 => Ok(KernelParam::Texture2DRead {
            name,
            slot,
            scalar_type,
        }),
        4 => Ok(KernelParam::Texture2DWrite {
            name,
            slot,
            scalar_type,
        }),
        5 => Ok(KernelParam::Texture3DRead {
            name,
            slot,
            scalar_type,
        }),
        _ => Err("invalid KernelParam tag"),
    }
}
