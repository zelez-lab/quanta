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
        // bf16 has no WGSL type; emit as an f32 literal (body is f32).
        ConstValue::BF16(x) => format!("{:?}f", f32::from_bits((*x as u32) << 16)),
        // fp8 likewise → f32 literal via the format conversion.
        ConstValue::FP8E5M2(x) => format!("{:?}f", crate::dtype::fp8_to_f32(*x, 5, 2)),
        ConstValue::FP8E4M3(x) => format!("{:?}f", crate::dtype::fp8_to_f32(*x, 4, 3)),
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
        BinOp::Rotl | BinOp::Rotr => {
            // WGSL has no rotate builtin. Emit the masked manual
            // decomposition `(x << k) | (x >> (W - k))` with `k` masked
            // to [0, W) so the (W - k) shift never reaches W (UB).
            //
            // The shifts MUST run in the *unsigned* domain: a signed `>>`
            // in WGSL is arithmetic (sign-extending), which corrupts the
            // bits that should rotate in. So cast the operand to the
            // same-width unsigned type, rotate there, and bitcast back.
            let width: u32 = match ty {
                ScalarType::U8 | ScalarType::I8 => 8,
                ScalarType::U16 | ScalarType::I16 | ScalarType::F16 => 16,
                ScalarType::U32
                | ScalarType::I32
                | ScalarType::F32
                | ScalarType::BF16
                | ScalarType::FP8E5M2
                | ScalarType::FP8E4M3
                | ScalarType::I4 => 32,
                ScalarType::U64 | ScalarType::I64 | ScalarType::F64 => 64,
                ScalarType::Bool => 1,
            };
            let mask = width - 1;
            // Unsigned WGSL type of matching width, and the cast that
            // returns the rotated bits to the operand type.
            let (uw, back) = if width >= 64 {
                ("u64", format!("bitcast<{ty_w}>"))
            } else {
                ("u32", format!("bitcast<{ty_w}>"))
            };
            // For unsigned operand types the bitcast is a no-op the WGSL
            // compiler accepts; emit it uniformly for simplicity.
            let (shl_amt, shr_amt) = if matches!(op, BinOp::Rotl) {
                (
                    format!("r{dst}_k"),
                    format!("(({width}u - r{dst}_k) & {mask}u)"),
                )
            } else {
                (
                    format!("(({width}u - r{dst}_k) & {mask}u)"),
                    format!("r{dst}_k"),
                )
            };
            out.push_str(&format!(
                "{pad}let r{dst}_k: u32 = u32(r{b}) & {mask}u; \
                 let r{dst}_x: {uw} = bitcast<{uw}>(r{a}); \
                 let r{dst}_l: {uw} = r{dst}_x << {shl_amt}; \
                 let r{dst}_r: {uw} = r{dst}_x >> {shr_amt}; \
                 let r{dst}: {ty_w} = {back}(r{dst}_l | r{dst}_r);\n",
            ));
            return;
        }
    };
    // WGSL's shift operators require an unsigned RHS — cast explicitly.
    // The LHS is bitcast to the op's declared type so the shift runs in
    // the right domain: an unsigned `Shr` is logical even when the source
    // register happens to be signed (the i32.shr_u-after-cast pattern), a
    // signed `Shr` stays arithmetic.
    if matches!(op, BinOp::Shl | BinOp::Shr) {
        out.push_str(&format!(
            "{}let r{}: {} = bitcast<{}>(r{}) {} u32(r{});\n",
            pad, dst, ty_w, ty_w, a, op_str, b
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
