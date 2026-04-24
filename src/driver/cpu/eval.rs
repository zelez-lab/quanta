//! Evaluation functions — binop, cmp, unary, math, cast, atomic.

use quanta_ir::{AtomicOp, BinOp, CmpOp, MathFn, ScalarType, UnaryOp};

use super::value::Value;

// ── Binary operations ────────────────────────────────────────────────────────

pub(super) fn eval_binop(a: Value, b: Value, op: &BinOp, ty: &ScalarType) -> Value {
    match ty {
        ScalarType::F32 | ScalarType::F16 => {
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
                BinOp::Div => {
                    if vb == 0 {
                        0
                    } else {
                        va / vb
                    }
                }
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
            })
        }
        ScalarType::U64 => {
            let va = a.as_u64();
            let vb = b.as_u64();
            Value::U64(match op {
                BinOp::Add => va.wrapping_add(vb),
                BinOp::Sub => va.wrapping_sub(vb),
                BinOp::Mul => va.wrapping_mul(vb),
                BinOp::Div => {
                    if vb == 0 {
                        0
                    } else {
                        va / vb
                    }
                }
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
        ScalarType::F32 | ScalarType::F16 => {
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
            ScalarType::F32 | ScalarType::F16 => Value::F32(-a.as_f32()),
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
    let _ = from; // source type is implicit in the value
    match to {
        ScalarType::F32 | ScalarType::F16 => Value::F32(val.as_f32()),
        ScalarType::F64 => Value::F64(val.as_f64()),
        ScalarType::U32 | ScalarType::U16 | ScalarType::U8 => Value::U32(val.as_u32()),
        ScalarType::I32 | ScalarType::I16 | ScalarType::I8 => Value::I32(val.as_i32()),
        ScalarType::U64 => Value::U64(val.as_u64()),
        ScalarType::I64 => Value::I64(val.as_i64()),
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
