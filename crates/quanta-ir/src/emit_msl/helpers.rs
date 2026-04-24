//! Small conversion helpers: const literals, operator strings, type mappings.

use crate::*;

pub(super) fn const_msl(v: &ConstValue) -> (&'static str, String) {
    match v {
        ConstValue::F32(x) => ("float", format!("{:.6}", x)),
        ConstValue::F64(x) => ("double", format!("{:.6}", x)),
        ConstValue::U32(x) => ("uint", format!("{}u", x)),
        ConstValue::U64(x) => ("ulong", format!("{}ul", x)),
        ConstValue::I32(x) => ("int", format!("{}", x)),
        ConstValue::I64(x) => ("long", format!("{}l", x)),
        ConstValue::Bool(x) => ("bool", if *x { "true" } else { "false" }.to_string()),
        ConstValue::F16(x) => (
            "half",
            format!("(half){}", f32::from_bits((*x as u32) << 16)),
        ),
    }
}

pub(super) fn binop_str(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::BitXor => "^",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
        // Saturating ops: MSL doesn't have a native operator, use regular +/-
        BinOp::SatAdd => "+",
        BinOp::SatSub => "-",
    }
}

pub(super) fn cmpop_str(op: &CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "==",
        CmpOp::Ne => "!=",
        CmpOp::Lt => "<",
        CmpOp::Le => "<=",
        CmpOp::Gt => ">",
        CmpOp::Ge => ">=",
    }
}

pub(super) fn math_fn_str(f: &MathFn) -> &'static str {
    match f {
        MathFn::Sin => "sin",
        MathFn::Cos => "cos",
        MathFn::Tan => "tan",
        MathFn::Asin => "asin",
        MathFn::Acos => "acos",
        MathFn::Atan => "atan",
        MathFn::Atan2 => "atan2",
        MathFn::Sqrt => "sqrt",
        MathFn::Rsqrt => "fast::rsqrt",
        MathFn::Exp => "exp",
        MathFn::Exp2 => "exp2",
        MathFn::Log => "log",
        MathFn::Log2 => "log2",
        MathFn::Pow => "pow",
        MathFn::Abs => "abs",
        MathFn::Min => "min",
        MathFn::Max => "max",
        MathFn::Clamp => "clamp",
        MathFn::Floor => "floor",
        MathFn::Ceil => "ceil",
        MathFn::Round => "round",
        MathFn::Fma => "fma",
    }
}

pub(super) fn atomic_fn_str(op: &AtomicOp) -> &'static str {
    match op {
        AtomicOp::Add => "atomic_fetch_add_explicit",
        AtomicOp::Sub => "atomic_fetch_sub_explicit",
        AtomicOp::Min => "atomic_fetch_min_explicit",
        AtomicOp::Max => "atomic_fetch_max_explicit",
        AtomicOp::And => "atomic_fetch_and_explicit",
        AtomicOp::Or => "atomic_fetch_or_explicit",
        AtomicOp::Xor => "atomic_fetch_xor_explicit",
        AtomicOp::Exchange => "atomic_exchange_explicit",
        AtomicOp::CompareExchange => "atomic_compare_exchange_weak_explicit",
    }
}
