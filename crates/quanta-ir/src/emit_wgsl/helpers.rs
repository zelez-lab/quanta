//! Small conversion helpers: const literals, operator strings, type mappings.

use crate::*;

pub(super) fn const_wgsl(v: &ConstValue) -> String {
    match v {
        ConstValue::F32(x) => {
            // WGSL float literal needs `f` suffix and a decimal point.
            if x.is_nan() {
                "bitcast<f32>(0x7fc00000u)".to_string()
            } else if x.is_infinite() {
                if *x > 0.0 {
                    "bitcast<f32>(0x7f800000u)".to_string()
                } else {
                    "bitcast<f32>(0xff800000u)".to_string()
                }
            } else {
                format!("{:?}f", x)
            }
        }
        ConstValue::F64(x) => format!("{:?}f", *x as f32),
        ConstValue::U32(x) => format!("{}u", x),
        ConstValue::U64(x) => format!("{}u", x),
        ConstValue::I32(x) => format!("{}i", x),
        ConstValue::I64(x) => format!("{}i", *x as i32),
        ConstValue::Bool(x) => if *x { "true" } else { "false" }.to_string(),
        ConstValue::F16(x) => {
            // WGSL `enable f16;` uses `h` suffix.
            format!("{:?}h", f32::from_bits((*x as u32) << 16))
        }
    }
}

/// WGSL has no native saturating add/sub — emit as `clamp` over the cast
/// to the unsigned integer range.
pub(super) fn binop_wgsl(
    out: &mut String,
    pad: &str,
    dst: u32,
    a: u32,
    b: u32,
    op: &BinOp,
    ty: &ScalarType,
) {
    let ty_w = ty.wgsl_name();
    let op_str = match op {
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
        BinOp::SatAdd => {
            // Saturating: clamp(a + b, MIN, MAX) — but WGSL has no
            // overflow-saturating intrinsic. Use widening + clamp pattern.
            // For u32: `min(a + b, 0xffffffffu)` (overflow wraps to small value)
            // We emit a defensive form that approximates saturation correctly
            // for the common unsigned case.
            out.push_str(&format!(
                "{}let r{}: {} = select(r{} + r{}, ~{}(0), (r{} + r{}) < r{});\n",
                pad, dst, ty_w, a, b, ty_w, a, b, a
            ));
            return;
        }
        BinOp::SatSub => {
            out.push_str(&format!(
                "{}let r{}: {} = select(r{} - r{}, {}(0), r{} > r{});\n",
                pad, dst, ty_w, a, b, ty_w, b, a
            ));
            return;
        }
    };
    // WGSL's shift operators require unsigned RHS — cast explicitly.
    if matches!(op, BinOp::Shl | BinOp::Shr) {
        out.push_str(&format!(
            "{}let r{}: {} = r{} {} u32(r{});\n",
            pad, dst, ty_w, a, op_str, b
        ));
    } else {
        out.push_str(&format!(
            "{}let r{}: {} = r{} {} r{};\n",
            pad, dst, ty_w, a, op_str, b
        ));
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
        MathFn::Rsqrt => "inverseSqrt",
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
        AtomicOp::Add => "atomicAdd",
        AtomicOp::Sub => "atomicSub",
        AtomicOp::Min => "atomicMin",
        AtomicOp::Max => "atomicMax",
        AtomicOp::And => "atomicAnd",
        AtomicOp::Or => "atomicOr",
        AtomicOp::Xor => "atomicXor",
        AtomicOp::Exchange => "atomicExchange",
        AtomicOp::CompareExchange => "atomicCompareExchangeWeak",
    }
}

/// Translate a Rust device function source to WGSL.
///
/// WGSL uses `fn name(...) -> type` — same syntax as Rust for function
/// signatures, so only body-level translations are needed.
pub(super) fn translate_device_fn_to_wgsl(rust_source: &str) -> String {
    let mut s = rust_source.to_string();
    s = s.replace("let mut ", "var ");
    s = s.replace(" as f32", "");
    s = s.replace(" as u32", "");
    s = s.replace(" as i32", "");
    s
}
