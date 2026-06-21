//! Evaluation functions — binop, cmp, unary, math, cast, atomic.

use quanta_ir::{AtomicOp, BinOp, CmpOp, MathFn, ScalarType, UnaryOp};

use super::value::Value;

// ── Binary operations ────────────────────────────────────────────────────────

pub(super) fn eval_binop(a: Value, b: Value, op: &BinOp, ty: &ScalarType) -> Value {
    match ty {
        ScalarType::F32
        | ScalarType::F16
        | ScalarType::BF16
        | ScalarType::FP8E5M2
        | ScalarType::FP8E4M3 => {
            let va = a.as_f32();
            let vb = b.as_f32();
            Value::F32(match op {
                BinOp::Add => va + vb,
                BinOp::Sub => va - vb,
                BinOp::Mul => va * vb,
                BinOp::Div => va / vb,
                BinOp::Rem => va % vb,
                _ => 0.0, // bitwise ops meaningless for floats
            })
        }
        ScalarType::F64 => {
            let va = a.as_f64();
            let vb = b.as_f64();
            Value::F64(match op {
                BinOp::Add => va + vb,
                BinOp::Sub => va - vb,
                BinOp::Mul => va * vb,
                BinOp::Div => va / vb,
                BinOp::Rem => va % vb,
                _ => 0.0,
            })
        }
        ScalarType::U32 | ScalarType::U16 | ScalarType::U8 => {
            let va = a.as_u32();
            let vb = b.as_u32();
            Value::U32(match op {
                BinOp::Add => va.wrapping_add(vb),
                BinOp::Sub => va.wrapping_sub(vb),
                BinOp::Mul => va.wrapping_mul(vb),
                BinOp::Div => va.checked_div(vb).unwrap_or(0),
                BinOp::Rem => {
                    if vb == 0 {
                        0
                    } else {
                        va % vb
                    }
                }
                BinOp::BitAnd => va & vb,
                BinOp::BitOr => va | vb,
                BinOp::BitXor => va ^ vb,
                BinOp::Shl => va.wrapping_shl(vb),
                BinOp::Shr => va.wrapping_shr(vb),
                BinOp::Rotl => va.rotate_left(vb),
                BinOp::Rotr => va.rotate_right(vb),
                BinOp::SatAdd => va.saturating_add(vb),
                BinOp::SatSub => va.saturating_sub(vb),
            })
        }
        ScalarType::I32 | ScalarType::I16 | ScalarType::I8 => {
            let va = a.as_i32();
            let vb = b.as_i32();
            Value::I32(match op {
                BinOp::Add => va.wrapping_add(vb),
                BinOp::Sub => va.wrapping_sub(vb),
                BinOp::Mul => va.wrapping_mul(vb),
                BinOp::Div => {
                    if vb == 0 {
                        0
                    } else {
                        va.wrapping_div(vb)
                    }
                }
                BinOp::Rem => {
                    if vb == 0 {
                        0
                    } else {
                        va.wrapping_rem(vb)
                    }
                }
                BinOp::BitAnd => va & vb,
                BinOp::BitOr => va | vb,
                BinOp::BitXor => va ^ vb,
                BinOp::Shl => va.wrapping_shl(vb as u32),
                BinOp::Shr => va.wrapping_shr(vb as u32),
                BinOp::Rotl => va.rotate_left(vb as u32),
                BinOp::Rotr => va.rotate_right(vb as u32),
                BinOp::SatAdd => va.saturating_add(vb),
                BinOp::SatSub => va.saturating_sub(vb),
            })
        }
        ScalarType::U64 => {
            let va = a.as_u64();
            let vb = b.as_u64();
            Value::U64(match op {
                BinOp::Add => va.wrapping_add(vb),
                BinOp::Sub => va.wrapping_sub(vb),
                BinOp::Mul => va.wrapping_mul(vb),
                BinOp::Div => va.checked_div(vb).unwrap_or(0),
                BinOp::Rem => {
                    if vb == 0 {
                        0
                    } else {
                        va % vb
                    }
                }
                BinOp::BitAnd => va & vb,
                BinOp::BitOr => va | vb,
                BinOp::BitXor => va ^ vb,
                BinOp::Shl => va.wrapping_shl(vb as u32),
                BinOp::Shr => va.wrapping_shr(vb as u32),
                BinOp::Rotl => va.rotate_left(vb as u32),
                BinOp::Rotr => va.rotate_right(vb as u32),
                BinOp::SatAdd => va.saturating_add(vb),
                BinOp::SatSub => va.saturating_sub(vb),
            })
        }
        ScalarType::I64 => {
            let va = a.as_i64();
            let vb = b.as_i64();
            Value::I64(match op {
                BinOp::Add => va.wrapping_add(vb),
                BinOp::Sub => va.wrapping_sub(vb),
                BinOp::Mul => va.wrapping_mul(vb),
                BinOp::Div => {
                    if vb == 0 {
                        0
                    } else {
                        va.wrapping_div(vb)
                    }
                }
                BinOp::Rem => {
                    if vb == 0 {
                        0
                    } else {
                        va.wrapping_rem(vb)
                    }
                }
                BinOp::BitAnd => va & vb,
                BinOp::BitOr => va | vb,
                BinOp::BitXor => va ^ vb,
                BinOp::Shl => va.wrapping_shl(vb as u32),
                BinOp::Shr => va.wrapping_shr(vb as u32),
                BinOp::Rotl => va.rotate_left(vb as u32),
                BinOp::Rotr => va.rotate_right(vb as u32),
                BinOp::SatAdd => va.saturating_add(vb),
                BinOp::SatSub => va.saturating_sub(vb),
            })
        }
        ScalarType::Bool => {
            let va = a.as_bool();
            let vb = b.as_bool();
            Value::Bool(match op {
                BinOp::BitAnd => va & vb,
                BinOp::BitOr => va | vb,
                BinOp::BitXor => va ^ vb,
                _ => false,
            })
        }
    }
}

pub(super) fn eval_cmp(a: Value, b: Value, op: &CmpOp, ty: &ScalarType) -> Value {
    let result = match ty {
        ScalarType::F32
        | ScalarType::F16
        | ScalarType::BF16
        | ScalarType::FP8E5M2
        | ScalarType::FP8E4M3 => {
            let va = a.as_f32();
            let vb = b.as_f32();
            match op {
                CmpOp::Eq => va == vb,
                CmpOp::Ne => va != vb,
                CmpOp::Lt => va < vb,
                CmpOp::Le => va <= vb,
                CmpOp::Gt => va > vb,
                CmpOp::Ge => va >= vb,
            }
        }
        ScalarType::F64 => {
            let va = a.as_f64();
            let vb = b.as_f64();
            match op {
                CmpOp::Eq => va == vb,
                CmpOp::Ne => va != vb,
                CmpOp::Lt => va < vb,
                CmpOp::Le => va <= vb,
                CmpOp::Gt => va > vb,
                CmpOp::Ge => va >= vb,
            }
        }
        ScalarType::U32 | ScalarType::U16 | ScalarType::U8 => {
            let va = a.as_u32();
            let vb = b.as_u32();
            match op {
                CmpOp::Eq => va == vb,
                CmpOp::Ne => va != vb,
                CmpOp::Lt => va < vb,
                CmpOp::Le => va <= vb,
                CmpOp::Gt => va > vb,
                CmpOp::Ge => va >= vb,
            }
        }
        ScalarType::I32 | ScalarType::I16 | ScalarType::I8 => {
            let va = a.as_i32();
            let vb = b.as_i32();
            match op {
                CmpOp::Eq => va == vb,
                CmpOp::Ne => va != vb,
                CmpOp::Lt => va < vb,
                CmpOp::Le => va <= vb,
                CmpOp::Gt => va > vb,
                CmpOp::Ge => va >= vb,
            }
        }
        ScalarType::U64 => {
            let va = a.as_u64();
            let vb = b.as_u64();
            match op {
                CmpOp::Eq => va == vb,
                CmpOp::Ne => va != vb,
                CmpOp::Lt => va < vb,
                CmpOp::Le => va <= vb,
                CmpOp::Gt => va > vb,
                CmpOp::Ge => va >= vb,
            }
        }
        ScalarType::I64 => {
            let va = a.as_i64();
            let vb = b.as_i64();
            match op {
                CmpOp::Eq => va == vb,
                CmpOp::Ne => va != vb,
                CmpOp::Lt => va < vb,
                CmpOp::Le => va <= vb,
                CmpOp::Gt => va > vb,
                CmpOp::Ge => va >= vb,
            }
        }
        ScalarType::Bool => {
            let va = a.as_bool();
            let vb = b.as_bool();
            match op {
                CmpOp::Eq => va == vb,
                CmpOp::Ne => va != vb,
                _ => false,
            }
        }
    };
    Value::Bool(result)
}

pub(super) fn eval_unary(a: Value, op: &UnaryOp, ty: &ScalarType) -> Value {
    match op {
        UnaryOp::Neg => match ty {
            ScalarType::F32
            | ScalarType::F16
            | ScalarType::BF16
            | ScalarType::FP8E5M2
            | ScalarType::FP8E4M3 => Value::F32(-a.as_f32()),
            ScalarType::F64 => Value::F64(-a.as_f64()),
            ScalarType::I32 | ScalarType::I16 | ScalarType::I8 => {
                Value::I32(a.as_i32().wrapping_neg())
            }
            ScalarType::I64 => Value::I64(a.as_i64().wrapping_neg()),
            ScalarType::U32 | ScalarType::U16 | ScalarType::U8 => {
                Value::U32(a.as_u32().wrapping_neg())
            }
            ScalarType::U64 => Value::U64(a.as_u64().wrapping_neg()),
            ScalarType::Bool => Value::Bool(!a.as_bool()),
        },
        UnaryOp::BitNot => match ty {
            ScalarType::U32 | ScalarType::U16 | ScalarType::U8 => Value::U32(!a.as_u32()),
            ScalarType::I32 | ScalarType::I16 | ScalarType::I8 => Value::I32(!a.as_i32()),
            ScalarType::U64 => Value::U64(!a.as_u64()),
            ScalarType::I64 => Value::I64(!a.as_i64()),
            ScalarType::Bool => Value::Bool(!a.as_bool()),
            _ => Value::U32(0),
        },
        UnaryOp::LogicalNot => Value::Bool(!a.as_bool()),
    }
}

pub(super) fn eval_math(func: &MathFn, args: &[Value], ty: &ScalarType) -> Value {
    // All math operates on f32 or f64 internally
    match ty {
        ScalarType::F64 => {
            let a = args.first().map(|v| v.as_f64()).unwrap_or(0.0);
            let b = args.get(1).map(|v| v.as_f64()).unwrap_or(0.0);
            let c = args.get(2).map(|v| v.as_f64()).unwrap_or(0.0);
            Value::F64(match func {
                MathFn::Sin => a.sin(),
                MathFn::Cos => a.cos(),
                MathFn::Tan => a.tan(),
                MathFn::Asin => a.asin(),
                MathFn::Acos => a.acos(),
                MathFn::Atan => a.atan(),
                MathFn::Atan2 => a.atan2(b),
                MathFn::Sqrt => a.sqrt(),
                MathFn::Rsqrt => 1.0 / a.sqrt(),
                MathFn::Exp => a.exp(),
                MathFn::Exp2 => a.exp2(),
                MathFn::Log => a.ln(),
                MathFn::Log2 => a.log2(),
                MathFn::Pow => a.powf(b),
                MathFn::Abs => a.abs(),
                MathFn::Min => a.min(b),
                MathFn::Max => a.max(b),
                MathFn::Clamp => a.max(b).min(c),
                MathFn::Floor => a.floor(),
                MathFn::Ceil => a.ceil(),
                MathFn::Round => a.round(),
                MathFn::Fma => a.mul_add(b, c),
            })
        }
        _ => {
            let a = args.first().map(|v| v.as_f32()).unwrap_or(0.0);
            let b = args.get(1).map(|v| v.as_f32()).unwrap_or(0.0);
            let c = args.get(2).map(|v| v.as_f32()).unwrap_or(0.0);
            Value::F32(match func {
                MathFn::Sin => a.sin(),
                MathFn::Cos => a.cos(),
                MathFn::Tan => a.tan(),
                MathFn::Asin => a.asin(),
                MathFn::Acos => a.acos(),
                MathFn::Atan => a.atan(),
                MathFn::Atan2 => a.atan2(b),
                MathFn::Sqrt => a.sqrt(),
                MathFn::Rsqrt => 1.0 / a.sqrt(),
                MathFn::Exp => a.exp(),
                MathFn::Exp2 => a.exp2(),
                MathFn::Log => a.ln(),
                MathFn::Log2 => a.log2(),
                MathFn::Pow => a.powf(b),
                MathFn::Abs => a.abs(),
                MathFn::Min => a.min(b),
                MathFn::Max => a.max(b),
                MathFn::Clamp => a.max(b).min(c),
                MathFn::Floor => a.floor(),
                MathFn::Ceil => a.ceil(),
                MathFn::Round => a.round(),
                MathFn::Fma => a.mul_add(b, c),
            })
        }
    }
}

pub(super) fn eval_cast(val: Value, from: &ScalarType, to: &ScalarType) -> Value {
    // Cast semantics: when widening from a narrower unsigned source,
    // the WASM/IR convention is **zero-extend** the bit pattern. The
    // raw `val.as_u64()` would sign-extend when `val` happens to be
    // tagged `Value::I32(-x)` even though the lowerer's `from` says
    // U32. Honour `from` explicitly so the same bit pattern gives
    // the same numeric result regardless of how the source Value
    // was tagged upstream.
    //
    // Mask-then-extend matches the WASM `i64.extend_i32_u` / Quanta
    // `Cast { from: U32, to: U64 }` semantics: drop high bits to the
    // source width, then widen.
    let zero_extended_u64: u64 = match from {
        ScalarType::U8 | ScalarType::I8 => (val.as_u32() & 0xFF) as u64,
        ScalarType::U16 | ScalarType::I16 => (val.as_u32() & 0xFFFF) as u64,
        ScalarType::U32 | ScalarType::I32 => val.as_u32() as u64,
        _ => val.as_u64(),
    };
    let sign_extended_i64: i64 = match from {
        ScalarType::I8 => (val.as_i32() as i8) as i64,
        ScalarType::I16 => (val.as_i32() as i16) as i64,
        ScalarType::I32 => val.as_i32() as i64,
        _ => val.as_i64(),
    };
    match to {
        ScalarType::F32
        | ScalarType::F16
        | ScalarType::BF16
        | ScalarType::FP8E5M2
        | ScalarType::FP8E4M3 => Value::F32(val.as_f32()),
        ScalarType::F64 => Value::F64(val.as_f64()),
        ScalarType::U32 | ScalarType::U16 | ScalarType::U8 => Value::U32(val.as_u32()),
        ScalarType::I32 | ScalarType::I16 | ScalarType::I8 => Value::I32(val.as_i32()),
        // For widening to u64: zero-extend honouring `from`.
        ScalarType::U64 => Value::U64(zero_extended_u64),
        // For widening to i64: sign-extend honouring `from`.
        ScalarType::I64 => Value::I64(sign_extended_i64),
        ScalarType::Bool => Value::Bool(val.as_bool()),
    }
}

pub(super) fn eval_atomic(
    old: Value,
    operand: Value,
    op: &AtomicOp,
    ty: &ScalarType,
) -> (Value, Value) {
    // Returns (new_value, old_value)
    match ty {
        ScalarType::U32 | ScalarType::U16 | ScalarType::U8 => {
            let o = old.as_u32();
            let v = operand.as_u32();
            let new = match op {
                AtomicOp::Add => o.wrapping_add(v),
                AtomicOp::Sub => o.wrapping_sub(v),
                AtomicOp::Min => o.min(v),
                AtomicOp::Max => o.max(v),
                AtomicOp::And => o & v,
                AtomicOp::Or => o | v,
                AtomicOp::Xor => o ^ v,
                AtomicOp::Exchange => v,
                AtomicOp::CompareExchange => unreachable!("CAS has its own op"),
            };
            (Value::U32(new), Value::U32(o))
        }
        ScalarType::I32 | ScalarType::I16 | ScalarType::I8 => {
            let o = old.as_i32();
            let v = operand.as_i32();
            let new = match op {
                AtomicOp::Add => o.wrapping_add(v),
                AtomicOp::Sub => o.wrapping_sub(v),
                AtomicOp::Min => o.min(v),
                AtomicOp::Max => o.max(v),
                AtomicOp::And => o & v,
                AtomicOp::Or => o | v,
                AtomicOp::Xor => o ^ v,
                AtomicOp::Exchange => v,
                AtomicOp::CompareExchange => unreachable!("CAS has its own op"),
            };
            (Value::I32(new), Value::I32(o))
        }
        ScalarType::U64 => {
            let o = old.as_u64();
            let v = operand.as_u64();
            let new = match op {
                AtomicOp::Add => o.wrapping_add(v),
                AtomicOp::Sub => o.wrapping_sub(v),
                AtomicOp::Min => o.min(v),
                AtomicOp::Max => o.max(v),
                AtomicOp::And => o & v,
                AtomicOp::Or => o | v,
                AtomicOp::Xor => o ^ v,
                AtomicOp::Exchange => v,
                AtomicOp::CompareExchange => unreachable!("CAS has its own op"),
            };
            (Value::U64(new), Value::U64(o))
        }
        _ => (operand, old), // unsupported types pass through
    }
}

// ── Property-based tests ────────────────────────────────────────────────────

#[cfg(test)]
mod proptest_eval {
    use super::*;
    use proptest::prelude::*;
    use quanta_ir::{BinOp, CmpOp, MathFn, ScalarType, UnaryOp};

    // ── BinOp: never panics for any input ───────────────────────────────

    proptest! {
        #[test]
        fn binop_add_u32_no_panic(a in any::<u32>(), b in any::<u32>()) {
            let _ = eval_binop(Value::U32(a), Value::U32(b), &BinOp::Add, &ScalarType::U32);
        }

        #[test]
        fn binop_sub_u32_no_panic(a in any::<u32>(), b in any::<u32>()) {
            let _ = eval_binop(Value::U32(a), Value::U32(b), &BinOp::Sub, &ScalarType::U32);
        }

        #[test]
        fn binop_mul_u32_no_panic(a in any::<u32>(), b in any::<u32>()) {
            let _ = eval_binop(Value::U32(a), Value::U32(b), &BinOp::Mul, &ScalarType::U32);
        }

        /// Division by zero must return 0, never panic.
        #[test]
        fn binop_div_by_zero_u32(a in any::<u32>()) {
            let result = eval_binop(Value::U32(a), Value::U32(0), &BinOp::Div, &ScalarType::U32);
            assert_eq!(result.as_u32(), 0);
        }

        /// Remainder by zero must return 0, never panic.
        #[test]
        fn binop_rem_by_zero_u32(a in any::<u32>()) {
            let result = eval_binop(Value::U32(a), Value::U32(0), &BinOp::Rem, &ScalarType::U32);
            assert_eq!(result.as_u32(), 0);
        }

        /// Division by zero for i32 must return 0, never panic.
        #[test]
        fn binop_div_by_zero_i32(a in any::<i32>()) {
            let result = eval_binop(Value::I32(a), Value::I32(0), &BinOp::Div, &ScalarType::I32);
            assert_eq!(result.as_i32(), 0);
        }

        /// i32::MIN / -1 must not panic (wrapping_div handles it).
        #[test]
        fn binop_div_i32_overflow(a in any::<i32>(), b in any::<i32>().prop_filter("non-zero", |b| *b != 0)) {
            let _ = eval_binop(Value::I32(a), Value::I32(b), &BinOp::Div, &ScalarType::I32);
        }

        /// i32::MIN % -1 must not panic.
        #[test]
        fn binop_rem_i32_overflow(a in any::<i32>(), b in any::<i32>().prop_filter("non-zero", |b| *b != 0)) {
            let _ = eval_binop(Value::I32(a), Value::I32(b), &BinOp::Rem, &ScalarType::I32);
        }

        /// Division by zero for u64 must return 0.
        #[test]
        fn binop_div_by_zero_u64(a in any::<u64>()) {
            let result = eval_binop(Value::U64(a), Value::U64(0), &BinOp::Div, &ScalarType::U64);
            assert_eq!(result.as_u64(), 0);
        }

        /// Division by zero for i64 must return 0.
        #[test]
        fn binop_div_by_zero_i64(a in any::<i64>()) {
            let result = eval_binop(Value::I64(a), Value::I64(0), &BinOp::Div, &ScalarType::I64);
            assert_eq!(result.as_i64(), 0);
        }

        /// i64 division with arbitrary non-zero divisors must not panic.
        #[test]
        fn binop_div_i64_no_panic(a in any::<i64>(), b in any::<i64>().prop_filter("non-zero", |b| *b != 0)) {
            let _ = eval_binop(Value::I64(a), Value::I64(b), &BinOp::Div, &ScalarType::I64);
        }

        /// Shift operations with any shift amount must not panic.
        #[test]
        fn binop_shl_u32_no_panic(a in any::<u32>(), b in any::<u32>()) {
            let _ = eval_binop(Value::U32(a), Value::U32(b), &BinOp::Shl, &ScalarType::U32);
        }

        #[test]
        fn binop_shr_u32_no_panic(a in any::<u32>(), b in any::<u32>()) {
            let _ = eval_binop(Value::U32(a), Value::U32(b), &BinOp::Shr, &ScalarType::U32);
        }

        /// Float arithmetic with special values (NaN, inf, denormal).
        #[test]
        fn binop_f32_all_ops(a_bits in any::<u32>(), b_bits in any::<u32>()) {
            let a = f32::from_bits(a_bits);
            let b = f32::from_bits(b_bits);
            for op in &[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Rem] {
                let _ = eval_binop(Value::F32(a), Value::F32(b), op, &ScalarType::F32);
            }
        }

        /// Float64 arithmetic with special values.
        #[test]
        fn binop_f64_all_ops(a_bits in any::<u64>(), b_bits in any::<u64>()) {
            let a = f64::from_bits(a_bits);
            let b = f64::from_bits(b_bits);
            for op in &[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Rem] {
                let _ = eval_binop(Value::F64(a), Value::F64(b), op, &ScalarType::F64);
            }
        }

        /// Bool binops never panic.
        #[test]
        fn binop_bool_no_panic(a in any::<bool>(), b in any::<bool>()) {
            for op in &[BinOp::BitAnd, BinOp::BitOr, BinOp::BitXor, BinOp::Add] {
                let _ = eval_binop(Value::Bool(a), Value::Bool(b), op, &ScalarType::Bool);
            }
        }
    }

    // ── CmpOp: never panics ─────────────────────────────────────────────

    proptest! {
        #[test]
        fn cmp_u32_no_panic(a in any::<u32>(), b in any::<u32>(), op_tag in 0u8..6) {
            let op = cmpop_from_tag(op_tag);
            let result = eval_cmp(Value::U32(a), Value::U32(b), &op, &ScalarType::U32);
            // Result must always be a bool.
            let _ = result.as_bool();
        }

        #[test]
        fn cmp_i32_no_panic(a in any::<i32>(), b in any::<i32>(), op_tag in 0u8..6) {
            let op = cmpop_from_tag(op_tag);
            let _ = eval_cmp(Value::I32(a), Value::I32(b), &op, &ScalarType::I32);
        }

        #[test]
        fn cmp_f32_no_panic(a_bits in any::<u32>(), b_bits in any::<u32>(), op_tag in 0u8..6) {
            let op = cmpop_from_tag(op_tag);
            let a = f32::from_bits(a_bits);
            let b = f32::from_bits(b_bits);
            let _ = eval_cmp(Value::F32(a), Value::F32(b), &op, &ScalarType::F32);
        }
    }

    // ── UnaryOp: never panics ───────────────────────────────────────────

    proptest! {
        #[test]
        fn unary_u32_no_panic(a in any::<u32>(), op_tag in 0u8..3) {
            let op = unaryop_from_tag(op_tag);
            let _ = eval_unary(Value::U32(a), &op, &ScalarType::U32);
        }

        #[test]
        fn unary_i32_no_panic(a in any::<i32>(), op_tag in 0u8..3) {
            let op = unaryop_from_tag(op_tag);
            let _ = eval_unary(Value::I32(a), &op, &ScalarType::I32);
        }

        #[test]
        fn unary_f32_no_panic(a_bits in any::<u32>(), op_tag in 0u8..3) {
            let op = unaryop_from_tag(op_tag);
            let a = f32::from_bits(a_bits);
            let _ = eval_unary(Value::F32(a), &op, &ScalarType::F32);
        }

        /// Neg of i32::MIN must not panic (wrapping_neg handles it).
        #[test]
        fn neg_i32_min_no_panic(a in any::<i32>()) {
            let _ = eval_unary(Value::I32(a), &UnaryOp::Neg, &ScalarType::I32);
        }
    }

    // ── Cast: never panics for any type pair ────────────────────────────

    proptest! {
        #[test]
        fn cast_u32_to_all_types(val in any::<u32>(), to_tag in 0u8..12) {
            let to = scalar_type_from_tag(to_tag);
            let _ = eval_cast(Value::U32(val), &ScalarType::U32, &to);
        }

        #[test]
        fn cast_f32_to_all_types(bits in any::<u32>(), to_tag in 0u8..12) {
            let to = scalar_type_from_tag(to_tag);
            let val = f32::from_bits(bits);
            let _ = eval_cast(Value::F32(val), &ScalarType::F32, &to);
        }

        #[test]
        fn cast_i32_to_all_types(val in any::<i32>(), to_tag in 0u8..12) {
            let to = scalar_type_from_tag(to_tag);
            let _ = eval_cast(Value::I32(val), &ScalarType::I32, &to);
        }
    }

    // ── Math: never panics ──────────────────────────────────────────────

    proptest! {
        /// All MathFn functions must not panic for any f32 inputs.
        #[test]
        fn math_f32_no_panic(a_bits in any::<u32>(), b_bits in any::<u32>(), c_bits in any::<u32>(), fn_tag in 0u8..22) {
            let func = mathfn_from_tag(fn_tag);
            let a = Value::F32(f32::from_bits(a_bits));
            let b = Value::F32(f32::from_bits(b_bits));
            let c = Value::F32(f32::from_bits(c_bits));
            let _ = eval_math(&func, &[a, b, c], &ScalarType::F32);
        }

        /// All MathFn functions must not panic for any f64 inputs.
        #[test]
        fn math_f64_no_panic(a_bits in any::<u64>(), b_bits in any::<u64>(), fn_tag in 0u8..22) {
            let func = mathfn_from_tag(fn_tag);
            let a = Value::F64(f64::from_bits(a_bits));
            let b = Value::F64(f64::from_bits(b_bits));
            let _ = eval_math(&func, &[a, b], &ScalarType::F64);
        }
    }

    // ── f16 roundtrip ───────────────────────────────────────────────────

    proptest! {
        /// Normal f16 values (non-zero exponent, non-max exponent) should
        /// roundtrip through f16_to_f32 -> f32_to_f16.
        #[test]
        fn f16_normal_roundtrip(bits in 0x0400u16..0x7C00) {
            // bits in [0x0400, 0x7BFF] = positive normals
            use super::super::value::{f16_to_f32, f32_to_f16};
            let f32_val = f16_to_f32(bits);
            let back = f32_to_f16(f32_val);
            assert_eq!(back, bits, "roundtrip failed for f16 bits 0x{:04X}", bits);
        }

        /// Negative normal f16 values should also roundtrip.
        #[test]
        fn f16_negative_normal_roundtrip(bits in 0x8400u16..0xFC00) {
            use super::super::value::{f16_to_f32, f32_to_f16};
            let f32_val = f16_to_f32(bits);
            let back = f32_to_f16(f32_val);
            assert_eq!(back, bits, "roundtrip failed for f16 bits 0x{:04X}", bits);
        }

        /// f16_to_f32 must never panic for any u16 input.
        #[test]
        fn f16_to_f32_no_panic(bits in any::<u16>()) {
            use super::super::value::f16_to_f32;
            let _ = f16_to_f32(bits);
        }

        /// f32_to_f16 must never panic for any f32 input.
        #[test]
        fn f32_to_f16_no_panic(bits in any::<u32>()) {
            use super::super::value::f32_to_f16;
            let val = f32::from_bits(bits);
            let _ = f32_to_f16(val);
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    fn scalar_type_from_tag(tag: u8) -> ScalarType {
        match tag {
            0 => ScalarType::F16,
            1 => ScalarType::F32,
            2 => ScalarType::F64,
            3 => ScalarType::U8,
            4 => ScalarType::U16,
            5 => ScalarType::U32,
            6 => ScalarType::U64,
            7 => ScalarType::I8,
            8 => ScalarType::I16,
            9 => ScalarType::I32,
            10 => ScalarType::I64,
            11 => ScalarType::Bool,
            _ => unreachable!(),
        }
    }

    fn cmpop_from_tag(tag: u8) -> CmpOp {
        match tag {
            0 => CmpOp::Eq,
            1 => CmpOp::Ne,
            2 => CmpOp::Lt,
            3 => CmpOp::Le,
            4 => CmpOp::Gt,
            5 => CmpOp::Ge,
            _ => unreachable!(),
        }
    }

    fn unaryop_from_tag(tag: u8) -> UnaryOp {
        match tag {
            0 => UnaryOp::Neg,
            1 => UnaryOp::BitNot,
            2 => UnaryOp::LogicalNot,
            _ => unreachable!(),
        }
    }

    fn mathfn_from_tag(tag: u8) -> MathFn {
        match tag {
            0 => MathFn::Sin,
            1 => MathFn::Cos,
            2 => MathFn::Tan,
            3 => MathFn::Asin,
            4 => MathFn::Acos,
            5 => MathFn::Atan,
            6 => MathFn::Atan2,
            7 => MathFn::Sqrt,
            8 => MathFn::Rsqrt,
            9 => MathFn::Exp,
            10 => MathFn::Exp2,
            11 => MathFn::Log,
            12 => MathFn::Log2,
            13 => MathFn::Pow,
            14 => MathFn::Abs,
            15 => MathFn::Min,
            16 => MathFn::Max,
            17 => MathFn::Clamp,
            18 => MathFn::Floor,
            19 => MathFn::Ceil,
            20 => MathFn::Round,
            21 => MathFn::Fma,
            _ => unreachable!(),
        }
    }
}
