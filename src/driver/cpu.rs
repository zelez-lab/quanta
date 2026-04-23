//! CPU software driver — executes KernelDef IR without GPU hardware.
//!
//! Simulates GPU execution on CPU by walking KernelOp instructions
//! sequentially per thread. Enables:
//! - Testing without GPU hardware
//! - CI on any machine
//! - Debugging kernels step by step
//! - CPU oracle for correctness verification
//!
//! Only supports the JIT path (`wave_jit`). Pre-compiled binaries
//! (SPIR-V, metallib) cannot be executed on CPU.
//!
//! V1 simplifications:
//! - No parallelism — threads execute sequentially (correct but slow)
//! - Shared memory: plain Vec per workgroup
//! - Barriers: no-op (sequential execution = always visible)
//! - Atomics: regular operations (sequential = no races)
//! - Texture ops: return zero with warning

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use std::collections::HashMap;
use std::sync::Mutex;

use crate::ray_tracing::{GeometryDesc, RayTracingPipelineDesc};
use crate::{
    Caps, FieldUsage, GpuDevice, Pipeline, Pulse, QuantaError, RenderPass, Texture, TextureDesc,
    Vendor, Wave,
};

use quanta_ir::{
    AtomicOp, BinOp, CmpOp, ConstValue, KernelDef, KernelOp, MathFn, Reg, ScalarType, UnaryOp,
};

// ── Value type ───────────────────────────────────────────────────────────────

/// Runtime value held in a virtual register.
#[derive(Debug, Clone, Copy)]
enum Value {
    F32(f32),
    F64(f64),
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    Bool(bool),
}

impl Value {
    fn as_u32(self) -> u32 {
        match self {
            Self::U32(v) => v,
            Self::I32(v) => v as u32,
            Self::U64(v) => v as u32,
            Self::I64(v) => v as u32,
            Self::F32(v) => v as u32,
            Self::F64(v) => v as u32,
            Self::Bool(v) => v as u32,
        }
    }

    fn as_u64(self) -> u64 {
        match self {
            Self::U64(v) => v,
            Self::U32(v) => v as u64,
            Self::I32(v) => v as u64,
            Self::I64(v) => v as u64,
            Self::F32(v) => v as u64,
            Self::F64(v) => v as u64,
            Self::Bool(v) => v as u64,
        }
    }

    fn as_i32(self) -> i32 {
        match self {
            Self::I32(v) => v,
            Self::U32(v) => v as i32,
            Self::U64(v) => v as i32,
            Self::I64(v) => v as i32,
            Self::F32(v) => v as i32,
            Self::F64(v) => v as i32,
            Self::Bool(v) => v as i32,
        }
    }

    fn as_i64(self) -> i64 {
        match self {
            Self::I64(v) => v,
            Self::I32(v) => v as i64,
            Self::U32(v) => v as i64,
            Self::U64(v) => v as i64,
            Self::F32(v) => v as i64,
            Self::F64(v) => v as i64,
            Self::Bool(v) => v as i64,
        }
    }

    fn as_f32(self) -> f32 {
        match self {
            Self::F32(v) => v,
            Self::F64(v) => v as f32,
            Self::U32(v) => v as f32,
            Self::I32(v) => v as f32,
            Self::U64(v) => v as f32,
            Self::I64(v) => v as f32,
            Self::Bool(v) => {
                if v {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    fn as_f64(self) -> f64 {
        match self {
            Self::F64(v) => v,
            Self::F32(v) => v as f64,
            Self::U32(v) => v as f64,
            Self::I32(v) => v as f64,
            Self::U64(v) => v as f64,
            Self::I64(v) => v as f64,
            Self::Bool(v) => {
                if v {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    fn as_bool(self) -> bool {
        match self {
            Self::Bool(v) => v,
            Self::U32(v) => v != 0,
            Self::I32(v) => v != 0,
            Self::U64(v) => v != 0,
            Self::I64(v) => v != 0,
            Self::F32(v) => v != 0.0,
            Self::F64(v) => v != 0.0,
        }
    }
}

// ── Scalar helpers ───────────────────────────────────────────────────────────

fn scalar_size(ty: &ScalarType) -> usize {
    match ty {
        ScalarType::Bool | ScalarType::U8 | ScalarType::I8 => 1,
        ScalarType::F16 | ScalarType::U16 | ScalarType::I16 => 2,
        ScalarType::F32 | ScalarType::U32 | ScalarType::I32 => 4,
        ScalarType::F64 | ScalarType::U64 | ScalarType::I64 => 8,
    }
}

fn read_scalar(buf: &[u8], index: u32, ty: &ScalarType) -> Value {
    let size = scalar_size(ty);
    let offset = index as usize * size;
    if offset + size > buf.len() {
        return Value::U32(0); // out-of-bounds reads zero
    }
    let bytes = &buf[offset..offset + size];
    match ty {
        ScalarType::F32 => Value::F32(f32::from_le_bytes(bytes.try_into().unwrap())),
        ScalarType::F64 => Value::F64(f64::from_le_bytes(bytes.try_into().unwrap())),
        ScalarType::U32 => Value::U32(u32::from_le_bytes(bytes.try_into().unwrap())),
        ScalarType::I32 => Value::I32(i32::from_le_bytes(bytes.try_into().unwrap())),
        ScalarType::U64 => Value::U64(u64::from_le_bytes(bytes.try_into().unwrap())),
        ScalarType::I64 => Value::I64(i64::from_le_bytes(bytes.try_into().unwrap())),
        ScalarType::U16 => Value::U32(u16::from_le_bytes(bytes.try_into().unwrap()) as u32),
        ScalarType::I16 => Value::I32(i16::from_le_bytes(bytes.try_into().unwrap()) as i32),
        ScalarType::U8 => Value::U32(bytes[0] as u32),
        ScalarType::I8 => Value::I32(bytes[0] as i32),
        ScalarType::F16 => {
            // f16: decode from u16 bits, expand to f32
            let bits = u16::from_le_bytes(bytes.try_into().unwrap());
            Value::F32(f16_to_f32(bits))
        }
        ScalarType::Bool => Value::Bool(bytes[0] != 0),
    }
}

fn write_scalar(buf: &mut [u8], index: u32, val: Value, ty: &ScalarType) {
    let size = scalar_size(ty);
    let offset = index as usize * size;
    if offset + size > buf.len() {
        return; // out-of-bounds write is silently ignored
    }
    let dest = &mut buf[offset..offset + size];
    match ty {
        ScalarType::F32 => dest.copy_from_slice(&val.as_f32().to_le_bytes()),
        ScalarType::F64 => dest.copy_from_slice(&val.as_f64().to_le_bytes()),
        ScalarType::U32 => dest.copy_from_slice(&val.as_u32().to_le_bytes()),
        ScalarType::I32 => dest.copy_from_slice(&val.as_i32().to_le_bytes()),
        ScalarType::U64 => dest.copy_from_slice(&val.as_u64().to_le_bytes()),
        ScalarType::I64 => dest.copy_from_slice(&val.as_i64().to_le_bytes()),
        ScalarType::U16 => dest.copy_from_slice(&(val.as_u32() as u16).to_le_bytes()),
        ScalarType::I16 => dest.copy_from_slice(&(val.as_i32() as i16).to_le_bytes()),
        ScalarType::U8 => dest[0] = val.as_u32() as u8,
        ScalarType::I8 => dest[0] = val.as_i32() as u8,
        ScalarType::F16 => {
            let bits = f32_to_f16(val.as_f32());
            dest.copy_from_slice(&bits.to_le_bytes());
        }
        ScalarType::Bool => dest[0] = val.as_bool() as u8,
    }
}

/// IEEE 754 half-precision to single-precision.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            // subnormal
            let mut e = 1u32;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e += 1;
            }
            f &= 0x3FF;
            let f32_exp = (127 - 15 - e + 1) as u32;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (f << 13))
        }
    } else if exp == 31 {
        if frac == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            f32::NAN
        }
    } else {
        let f32_exp = exp + (127 - 15);
        f32::from_bits((sign << 31) | (f32_exp << 23) | (frac << 13))
    }
}

/// Single-precision to IEEE 754 half-precision (round to nearest even).
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 0xFF {
        // inf/nan
        if frac == 0 {
            ((sign << 15) | 0x7C00) as u16
        } else {
            ((sign << 15) | 0x7C00 | (frac >> 13).max(1)) as u16
        }
    } else if exp > 142 {
        // overflow to inf
        ((sign << 15) | 0x7C00) as u16
    } else if exp < 113 {
        // underflow to zero
        (sign << 15) as u16
    } else {
        let new_exp = (exp - 112) as u32;
        ((sign << 15) | (new_exp << 10) | (frac >> 13)) as u16
    }
}

fn value_from_const(cv: &ConstValue) -> Value {
    match cv {
        ConstValue::F32(v) => Value::F32(*v),
        ConstValue::F64(v) => Value::F64(*v),
        ConstValue::U32(v) => Value::U32(*v),
        ConstValue::U64(v) => Value::U64(*v),
        ConstValue::I32(v) => Value::I32(*v),
        ConstValue::I64(v) => Value::I64(*v),
        ConstValue::Bool(v) => Value::Bool(*v),
        ConstValue::F16(bits) => Value::F32(f16_to_f32(*bits)),
    }
}

// ── Binary operations ────────────────────────────────────────────────────────

fn eval_binop(a: Value, b: Value, op: &BinOp, ty: &ScalarType) -> Value {
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

fn eval_cmp(a: Value, b: Value, op: &CmpOp, ty: &ScalarType) -> Value {
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

fn eval_unary(a: Value, op: &UnaryOp, ty: &ScalarType) -> Value {
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

fn eval_math(func: &MathFn, args: &[Value], ty: &ScalarType) -> Value {
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

fn eval_cast(val: Value, from: &ScalarType, to: &ScalarType) -> Value {
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

fn eval_atomic(old: Value, operand: Value, op: &AtomicOp, ty: &ScalarType) -> (Value, Value) {
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

// ── Execution context ────────────────────────────────────────────────────────

/// Signal to break out of the current loop.
struct BreakSignal;

/// Per-thread execution state.
struct ExecCtx<'a> {
    quark_id: u32,
    local_id: u32,
    group_id: u32,
    group_size: u32,
    quark_count: u32,
    regs: HashMap<u32, Value>,
    fields: &'a mut HashMap<u64, Vec<u8>>,
    /// Shared memory per workgroup, keyed by declaration id.
    shared: &'a mut HashMap<u32, Vec<u8>>,
}

fn execute_ops(ctx: &mut ExecCtx, ops: &[KernelOp]) -> Result<Option<BreakSignal>, String> {
    for op in ops {
        match op {
            KernelOp::QuarkId { dst } => {
                ctx.regs.insert(dst.0, Value::U32(ctx.quark_id));
            }
            KernelOp::QuarkCount { dst } => {
                ctx.regs.insert(dst.0, Value::U32(ctx.quark_count));
            }
            KernelOp::LocalId { dst } => {
                ctx.regs.insert(dst.0, Value::U32(ctx.local_id));
            }
            KernelOp::GroupId { dst } => {
                ctx.regs.insert(dst.0, Value::U32(ctx.group_id));
            }
            KernelOp::GroupSize { dst } => {
                ctx.regs.insert(dst.0, Value::U32(ctx.group_size));
            }
            KernelOp::Const { dst, value } => {
                ctx.regs.insert(dst.0, value_from_const(value));
            }
            KernelOp::Load {
                dst,
                field,
                index,
                ty,
            } => {
                let idx = reg(ctx, index)?;
                let slot = *field as u64;
                let buf = ctx
                    .fields
                    .get(&slot)
                    .ok_or_else(|| format!("Load: field slot {slot} not bound"))?;
                let val = read_scalar(buf, idx.as_u32(), ty);
                ctx.regs.insert(dst.0, val);
            }
            KernelOp::Store {
                field,
                index,
                src,
                ty,
            } => {
                let idx = reg(ctx, index)?;
                let val = reg(ctx, src)?;
                let slot = *field as u64;
                let buf = ctx
                    .fields
                    .get_mut(&slot)
                    .ok_or_else(|| format!("Store: field slot {slot} not bound"))?;
                write_scalar(buf, idx.as_u32(), val, ty);
            }
            KernelOp::BinOp { dst, a, b, op, ty } => {
                let va = reg(ctx, a)?;
                let vb = reg(ctx, b)?;
                ctx.regs.insert(dst.0, eval_binop(va, vb, op, ty));
            }
            KernelOp::UnaryOp { dst, a, op, ty } => {
                let va = reg(ctx, a)?;
                ctx.regs.insert(dst.0, eval_unary(va, op, ty));
            }
            KernelOp::Cmp { dst, a, b, op, ty } => {
                let va = reg(ctx, a)?;
                let vb = reg(ctx, b)?;
                ctx.regs.insert(dst.0, eval_cmp(va, vb, op, ty));
            }
            KernelOp::Branch {
                cond,
                then_ops,
                else_ops,
            } => {
                let cv = reg(ctx, cond)?;
                let branch_ops = if cv.as_bool() { then_ops } else { else_ops };
                if let Some(brk) = execute_ops(ctx, branch_ops)? {
                    return Ok(Some(brk));
                }
            }
            KernelOp::Loop {
                count,
                iter_reg,
                body,
            } => {
                let n = reg(ctx, count)?.as_u32();
                'lp: for i in 0..n {
                    ctx.regs.insert(iter_reg.0, Value::U32(i));
                    if let Some(_brk) = execute_ops(ctx, body)? {
                        break 'lp;
                    }
                }
            }
            KernelOp::Break => {
                return Ok(Some(BreakSignal));
            }
            KernelOp::MathCall {
                dst,
                func,
                args,
                ty,
            } => {
                let arg_vals: Vec<Value> =
                    args.iter().map(|r| reg(ctx, r)).collect::<Result<_, _>>()?;
                ctx.regs.insert(dst.0, eval_math(func, &arg_vals, ty));
            }
            KernelOp::Cast { dst, src, from, to } => {
                let v = reg(ctx, src)?;
                ctx.regs.insert(dst.0, eval_cast(v, from, to));
            }
            KernelOp::Copy { dst, src, .. } => {
                let v = reg(ctx, src)?;
                ctx.regs.insert(dst.0, v);
            }
            KernelOp::SharedDecl { id, ty, count } => {
                let size = scalar_size(ty) * (*count as usize);
                ctx.shared.entry(*id).or_insert_with(|| vec![0u8; size]);
            }
            KernelOp::SharedLoad { dst, id, index, ty } => {
                let idx = reg(ctx, index)?.as_u32();
                let buf = ctx
                    .shared
                    .get(id)
                    .ok_or_else(|| format!("SharedLoad: shared id {id} not declared"))?;
                let val = read_scalar(buf, idx, ty);
                ctx.regs.insert(dst.0, val);
            }
            KernelOp::SharedStore { id, index, src, ty } => {
                let idx = reg(ctx, index)?.as_u32();
                let val = reg(ctx, src)?;
                let buf = ctx
                    .shared
                    .get_mut(id)
                    .ok_or_else(|| format!("SharedStore: shared id {id} not declared"))?;
                write_scalar(buf, idx, val, ty);
            }
            KernelOp::Barrier => {
                // No-op: sequential execution means shared memory is always visible.
            }
            KernelOp::AtomicOp {
                dst,
                field,
                index,
                val,
                op,
                ty,
            } => {
                let idx = reg(ctx, index)?.as_u32();
                let operand = reg(ctx, val)?;
                let slot = *field as u64;
                let buf = ctx
                    .fields
                    .get(&slot)
                    .ok_or_else(|| format!("AtomicOp: field slot {slot} not bound"))?;
                let old = read_scalar(buf, idx, ty);
                let (new_val, old_val) = eval_atomic(old, operand, op, ty);
                let buf = ctx.fields.get_mut(&slot).unwrap();
                write_scalar(buf, idx, new_val, ty);
                ctx.regs.insert(dst.0, old_val);
            }
            KernelOp::AtomicCas {
                dst,
                field,
                index,
                expected,
                desired,
                ty,
            } => {
                let idx = reg(ctx, index)?.as_u32();
                let exp = reg(ctx, expected)?;
                let des = reg(ctx, desired)?;
                let slot = *field as u64;
                let buf = ctx
                    .fields
                    .get(&slot)
                    .ok_or_else(|| format!("AtomicCas: field slot {slot} not bound"))?;
                let old = read_scalar(buf, idx, ty);
                let old_u64 = old.as_u64();
                let exp_u64 = exp.as_u64();
                if old_u64 == exp_u64 {
                    let buf = ctx.fields.get_mut(&slot).unwrap();
                    write_scalar(buf, idx, des, ty);
                }
                ctx.regs.insert(dst.0, old);
            }
            // Wave/subgroup intrinsics: return identity values in sequential mode
            KernelOp::WaveShuffle { dst, src, .. } => {
                // Single-thread: shuffle returns own value
                let v = reg(ctx, src)?;
                ctx.regs.insert(dst.0, v);
            }
            KernelOp::WaveBallot { dst, .. } => {
                // Single-thread ballot: bit 0 set
                ctx.regs.insert(dst.0, Value::U32(1));
            }
            KernelOp::WaveAny { dst, predicate } => {
                let v = reg(ctx, predicate)?;
                ctx.regs.insert(dst.0, Value::Bool(v.as_bool()));
            }
            KernelOp::WaveAll { dst, predicate } => {
                let v = reg(ctx, predicate)?;
                ctx.regs.insert(dst.0, Value::Bool(v.as_bool()));
            }
            KernelOp::SubgroupReduceAdd { dst, src, .. }
            | KernelOp::SubgroupInclusiveAdd { dst, src, .. }
            | KernelOp::SubgroupExclusiveAdd { dst, src, .. } => {
                // Single-thread: reduce/scan = own value (exclusive = 0)
                if matches!(op, KernelOp::SubgroupExclusiveAdd { .. }) {
                    ctx.regs.insert(dst.0, Value::U32(0));
                } else {
                    let v = reg(ctx, src)?;
                    ctx.regs.insert(dst.0, v);
                }
            }
            KernelOp::SubgroupReduceMin { dst, src, .. }
            | KernelOp::SubgroupReduceMax { dst, src, .. } => {
                let v = reg(ctx, src)?;
                ctx.regs.insert(dst.0, v);
            }
            // Vector ops
            KernelOp::VecConstruct {
                dst, components, ..
            } => {
                // Store as the first component for simple use cases
                if let Some(first) = components.first() {
                    let v = reg(ctx, first)?;
                    ctx.regs.insert(dst.0, v);
                }
            }
            KernelOp::VecExtract {
                dst,
                vec,
                component,
                ..
            } => {
                // Simplified: we store vectors as their first component
                let v = reg(ctx, vec)?;
                let _ = component;
                ctx.regs.insert(dst.0, v);
            }
            KernelOp::MatMul { dst, .. } => {
                ctx.regs.insert(dst.0, Value::F32(0.0));
            }
            KernelOp::Dot {
                dst,
                a,
                b,
                ty,
                width,
            } => {
                // Simplified dot product: a * b (scalar, not vector)
                let va = reg(ctx, a)?;
                let vb = reg(ctx, b)?;
                let _ = width;
                ctx.regs.insert(dst.0, eval_binop(va, vb, &BinOp::Mul, ty));
            }
            // Texture ops: return zero with no-op
            KernelOp::TextureSample2D { dst, .. }
            | KernelOp::TextureSample3D { dst, .. }
            | KernelOp::TextureLoad2D { dst, .. } => {
                ctx.regs.insert(dst.0, Value::F32(0.0));
            }
            KernelOp::TextureWrite2D { .. } => {
                // no-op
            }
            KernelOp::TextureSize { dst_w, dst_h, .. } => {
                ctx.regs.insert(dst_w.0, Value::U32(0));
                ctx.regs.insert(dst_h.0, Value::U32(0));
            }
            // Bit manipulation
            KernelOp::Bitcast { dst, src, .. } => {
                let v = reg(ctx, src)?;
                ctx.regs.insert(dst.0, v);
            }
            KernelOp::CountTrailingZeros { dst, src, ty } => {
                let v = reg(ctx, src)?;
                let result = match ty {
                    ScalarType::U32 | ScalarType::I32 => Value::U32(v.as_u32().trailing_zeros()),
                    ScalarType::U64 | ScalarType::I64 => Value::U32(v.as_u64().trailing_zeros()),
                    _ => Value::U32(0),
                };
                ctx.regs.insert(dst.0, result);
            }
            KernelOp::CountLeadingZeros { dst, src, ty } => {
                let v = reg(ctx, src)?;
                let result = match ty {
                    ScalarType::U32 | ScalarType::I32 => Value::U32(v.as_u32().leading_zeros()),
                    ScalarType::U64 | ScalarType::I64 => Value::U32(v.as_u64().leading_zeros()),
                    _ => Value::U32(0),
                };
                ctx.regs.insert(dst.0, result);
            }
            KernelOp::PopCount { dst, src, ty } => {
                let v = reg(ctx, src)?;
                let result = match ty {
                    ScalarType::U32 | ScalarType::I32 => Value::U32(v.as_u32().count_ones()),
                    ScalarType::U64 | ScalarType::I64 => Value::U32(v.as_u64().count_ones()),
                    _ => Value::U32(0),
                };
                ctx.regs.insert(dst.0, result);
            }
            // Dynamic dispatch and device calls: unsupported in V1
            KernelOp::Dispatch { .. } => {
                // Dynamic parallelism is not supported in CPU mode.
            }
            KernelOp::DeviceCall { dst, .. } => {
                // Device function calls require linked function bodies.
                // Return zero for now.
                ctx.regs.insert(dst.0, Value::U32(0));
            }
        }
    }
    Ok(None)
}

/// Read a register, returning an error if it hasn't been set.
fn reg(ctx: &ExecCtx, r: &Reg) -> Result<Value, String> {
    ctx.regs
        .get(&r.0)
        .copied()
        .ok_or_else(|| format!("register r{} not set", r.0))
}

// ── CPU Device ───────────────────────────────────────────────────────────────

/// Internal buffer allocation.
struct CpuBuffer {
    data: Vec<u8>,
}

/// Stored kernel ready for dispatch.
struct CpuKernel {
    def: KernelDef,
}

/// CPU software device — executes GPU kernel IR without hardware.
pub struct CpuDevice {
    caps: Caps,
    next_handle: Mutex<u64>,
    buffers: Mutex<HashMap<u64, CpuBuffer>>,
    kernels: Mutex<HashMap<u64, CpuKernel>>,
}

impl CpuDevice {
    /// Create a new CPU software device.
    pub fn new() -> Self {
        Self {
            caps: Caps {
                nuclei: 1,
                protons_per_nucleus: 1,
                quarks_per_proton: 1,
                memory_bytes: 1024 * 1024 * 1024, // 1 GB virtual
                max_quarks_per_dispatch: u32::MAX,
                max_groups: [u32::MAX; 3],
                vendor: Vendor::Software,
                name: String::from("Quanta CPU (software)"),
            },
            next_handle: Mutex::new(1),
            buffers: Mutex::new(HashMap::new()),
            kernels: Mutex::new(HashMap::new()),
        }
    }

    fn alloc_handle(&self) -> u64 {
        let mut h = self.next_handle.lock().unwrap();
        let handle = *h;
        *h += 1;
        handle
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuDevice for CpuDevice {
    fn caps(&self) -> &Caps {
        &self.caps
    }

    // === Fields ===

    fn field_alloc(&self, size: usize, _usage: FieldUsage) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        let buf = CpuBuffer {
            data: vec![0u8; size],
        };
        self.buffers.lock().unwrap().insert(handle, buf);
        Ok(handle)
    }

    fn field_free(&self, handle: u64) {
        self.buffers.lock().unwrap().remove(&handle);
    }

    fn field_write_bytes(&self, handle: u64, data: &[u8]) -> Result<(), QuantaError> {
        let mut bufs = self.buffers.lock().unwrap();
        let buf = bufs
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::invalid_param("field handle not found"))?;
        let len = data.len().min(buf.data.len());
        buf.data[..len].copy_from_slice(&data[..len]);
        Ok(())
    }

    fn field_read_bytes(&self, handle: u64, size: usize) -> Result<Vec<u8>, QuantaError> {
        let bufs = self.buffers.lock().unwrap();
        let buf = bufs
            .get(&handle)
            .ok_or_else(|| QuantaError::invalid_param("field handle not found"))?;
        let len = size.min(buf.data.len());
        Ok(buf.data[..len].to_vec())
    }

    fn field_copy_bytes(&self, dst: u64, src: u64, size: usize) -> Result<(), QuantaError> {
        let mut bufs = self.buffers.lock().unwrap();
        // Copy src data first to avoid borrow conflict
        let src_data = {
            let src_buf = bufs
                .get(&src)
                .ok_or_else(|| QuantaError::invalid_param("src field not found"))?;
            let len = size.min(src_buf.data.len());
            src_buf.data[..len].to_vec()
        };
        let dst_buf = bufs
            .get_mut(&dst)
            .ok_or_else(|| QuantaError::invalid_param("dst field not found"))?;
        let len = src_data.len().min(dst_buf.data.len());
        dst_buf.data[..len].copy_from_slice(&src_data[..len]);
        Ok(())
    }

    fn field_map(&self, handle: u64, _size: usize) -> Result<*mut u8, QuantaError> {
        let mut bufs = self.buffers.lock().unwrap();
        let buf = bufs
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::invalid_param("field handle not found"))?;
        Ok(buf.data.as_mut_ptr())
    }

    fn field_unmap(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(()) // CPU memory is always mapped
    }

    fn field_create_mapped(
        &self,
        size: usize,
        _usage: FieldUsage,
    ) -> Result<(u64, *mut u8), QuantaError> {
        let handle = self.alloc_handle();
        let buf = CpuBuffer {
            data: vec![0u8; size],
        };
        self.buffers.lock().unwrap().insert(handle, buf);
        let ptr = self
            .buffers
            .lock()
            .unwrap()
            .get_mut(&handle)
            .unwrap()
            .data
            .as_mut_ptr();
        Ok((handle, ptr))
    }

    // === Textures (minimal stubs) ===

    fn texture_create(&self, desc: &TextureDesc) -> Result<Texture, QuantaError> {
        let handle = self.alloc_handle();
        let size = (desc.width * desc.height) as usize * desc.format.bytes_per_pixel();
        self.buffers.lock().unwrap().insert(
            handle,
            CpuBuffer {
                data: vec![0u8; size],
            },
        );
        Ok(Texture {
            handle,
            width: desc.width,
            height: desc.height,
            format: desc.format,
            drop_fn: None,
        })
    }

    fn texture_write(&self, texture: &Texture, data: &[u8]) -> Result<(), QuantaError> {
        self.field_write_bytes(texture.handle(), data)
    }

    fn texture_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        let size =
            (texture.width() * texture.height()) as usize * texture.format().bytes_per_pixel();
        self.field_read_bytes(texture.handle(), size)
    }

    fn sampler_create(
        &self,
        _desc: &crate::render_pass::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        Ok(crate::Sampler {
            handle: self.alloc_handle(),
            drop_fn: None,
        })
    }

    fn generate_mipmaps(&self, _texture: &Texture) -> Result<(), QuantaError> {
        Ok(()) // no-op on CPU
    }

    // === Compute ===

    fn wave(&self, _kernel: &[u8]) -> Result<Wave, QuantaError> {
        Err(QuantaError::invalid_param(
            "CPU device only supports JIT path (wave_jit). \
             Pre-compiled binaries cannot be executed on CPU.",
        ))
    }

    fn wave_jit(&self, kernel_def_bytes: &[u8]) -> Result<Wave, QuantaError> {
        let def = quanta_ir::deserialize_kernel(kernel_def_bytes)
            .map_err(|e| QuantaError::compilation_failed(e.to_string()))?;
        let handle = self.alloc_handle();
        let workgroup_size = def.workgroup_size;
        self.kernels
            .lock()
            .unwrap()
            .insert(handle, CpuKernel { def });
        Ok(Wave {
            handle,
            bindings: [0u64; 16],
            binding_count: 0,
            texture_bindings: [0u64; 16],
            texture_count: 0,
            push_data: [0u8; 256],
            push_len: 0,
            push_mask: 0,
            workgroup_size,
            drop_fn: None,
        })
    }

    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        let kernels = self.kernels.lock().unwrap();
        let kernel = kernels
            .get(&wave.handle)
            .ok_or_else(|| QuantaError::invalid_param("wave handle not found"))?;

        let wg = kernel.def.workgroup_size;
        let total_groups = groups[0] as u64 * groups[1] as u64 * groups[2] as u64;
        let threads_per_group = wg[0] as u64 * wg[1] as u64 * wg[2] as u64;
        let total_threads = total_groups * threads_per_group;

        // Snapshot bound buffer data into a working copy
        let mut field_data: HashMap<u64, Vec<u8>> = HashMap::new();
        {
            let bufs = self.buffers.lock().unwrap();
            for i in 0..wave.binding_count as usize {
                let handle = wave.bindings[i];
                if handle != 0 {
                    if let Some(buf) = bufs.get(&handle) {
                        field_data.insert(i as u64, buf.data.clone());
                    }
                }
            }
        }

        let group_size_x = wg[0];

        // Split the kernel body at top-level Barrier ops into segments.
        // For each workgroup: run all threads through segment 0, then all
        // threads through segment 1, etc. This correctly simulates GPU
        // barrier semantics where all threads synchronize at each barrier.
        let segments = split_at_barriers(&kernel.def.body);

        for gid in 0..total_groups {
            let mut shared: HashMap<u32, Vec<u8>> = HashMap::new();
            // Per-thread register state persists across barrier segments
            let mut thread_regs: Vec<HashMap<u32, Value>> =
                (0..threads_per_group).map(|_| HashMap::new()).collect();

            for segment in &segments {
                for lid in 0..threads_per_group {
                    let quark_id = (gid * threads_per_group + lid) as u32;
                    let local_id = lid as u32;
                    let group_id = gid as u32;

                    let mut ctx = ExecCtx {
                        quark_id,
                        local_id,
                        group_id,
                        group_size: group_size_x,
                        quark_count: total_threads as u32,
                        regs: core::mem::take(&mut thread_regs[lid as usize]),
                        fields: &mut field_data,
                        shared: &mut shared,
                    };

                    execute_ops(&mut ctx, segment).map_err(|e| {
                        QuantaError::compilation_failed(format!(
                            "CPU execution error (quark {quark_id}): {e}"
                        ))
                    })?;

                    // Save register state for next segment
                    thread_regs[lid as usize] = ctx.regs;
                }
            }
        }

        // Write back modified buffer data
        {
            let mut bufs = self.buffers.lock().unwrap();
            for i in 0..wave.binding_count as usize {
                let handle = wave.bindings[i];
                if handle != 0 {
                    if let Some(modified) = field_data.remove(&(i as u64)) {
                        if let Some(buf) = bufs.get_mut(&handle) {
                            buf.data = modified;
                        }
                    }
                }
            }
        }

        Ok(Pulse {
            handle: 0,
            completed: true,
            wait_fn: None,
        })
    }

    fn wave_dispatch_indirect(
        &self,
        _wave: &Wave,
        _buffer: u64,
        _offset: u64,
    ) -> Result<Pulse, QuantaError> {
        Err(QuantaError::invalid_param(
            "indirect dispatch not supported on CPU device",
        ))
    }

    // === Render (stubs) ===

    fn pipeline_create(&self, _desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        Ok(Pipeline {
            handle: self.alloc_handle(),
            drop_fn: None,
        })
    }

    fn render_begin(&self, _target: &Texture) -> Result<RenderPass, QuantaError> {
        Err(QuantaError::invalid_param(
            "render passes not supported on CPU device",
        ))
    }

    fn render_end(&self, _pass: RenderPass) -> Result<Pulse, QuantaError> {
        Err(QuantaError::invalid_param(
            "render passes not supported on CPU device",
        ))
    }

    // === Sync ===

    fn pulse_wait(&self, pulse: &mut Pulse) -> Result<(), QuantaError> {
        pulse.completed = true;
        Ok(())
    }

    fn pulse_poll(&self, _pulse: &Pulse) -> bool {
        true // CPU execution is synchronous
    }

    // === M4.2: Mesh shaders ===

    fn dispatch_mesh(&self, _pipeline: u64, _groups: [u32; 3]) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param(
            "mesh shaders not supported on CPU device",
        ))
    }

    // === M4.3: Ray tracing ===

    fn build_acceleration_structure(&self, _geometry: &[GeometryDesc]) -> Result<u64, QuantaError> {
        Err(QuantaError::invalid_param(
            "ray tracing not supported on CPU device",
        ))
    }

    fn create_ray_tracing_pipeline(
        &self,
        _desc: &RayTracingPipelineDesc,
    ) -> Result<u64, QuantaError> {
        Err(QuantaError::invalid_param(
            "ray tracing not supported on CPU device",
        ))
    }

    fn dispatch_rays(&self, _pipeline: u64, _width: u32, _height: u32) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param(
            "ray tracing not supported on CPU device",
        ))
    }

    fn destroy_acceleration_structure(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    // === M5.1: Sparse textures ===

    fn sparse_texture_create(&self, _desc: &TextureDesc) -> Result<u64, QuantaError> {
        Err(QuantaError::invalid_param(
            "sparse textures not supported on CPU device",
        ))
    }

    fn sparse_map_tile(
        &self,
        _texture: u64,
        _mip: u32,
        _x: u32,
        _y: u32,
        _backing: u64,
    ) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param(
            "sparse textures not supported on CPU device",
        ))
    }

    fn sparse_unmap_tile(
        &self,
        _texture: u64,
        _mip: u32,
        _x: u32,
        _y: u32,
    ) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param(
            "sparse textures not supported on CPU device",
        ))
    }

    // === M5.2: Indirect command buffers ===

    fn indirect_buffer_create(&self, _max_commands: u32) -> Result<u64, QuantaError> {
        Err(QuantaError::invalid_param(
            "indirect command buffers not supported on CPU device",
        ))
    }

    fn indirect_buffer_execute(&self, _handle: u64, _count: u32) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param(
            "indirect command buffers not supported on CPU device",
        ))
    }

    fn indirect_buffer_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    // === M5.3: Bindless resources ===

    fn bind_texture_array(&self, _textures: &[u64]) -> Result<u64, QuantaError> {
        Err(QuantaError::invalid_param(
            "bindless resources not supported on CPU device",
        ))
    }

    fn bind_buffer_array(&self, _buffers: &[u64]) -> Result<u64, QuantaError> {
        Err(QuantaError::invalid_param(
            "bindless resources not supported on CPU device",
        ))
    }
}

/// Split a list of ops at top-level `Barrier` instructions.
///
/// Returns a list of segments. Each segment is a slice of ops between
/// barriers. The barrier ops themselves are consumed (they serve only
/// as synchronization points).
fn split_at_barriers(ops: &[KernelOp]) -> Vec<Vec<KernelOp>> {
    let mut segments: Vec<Vec<KernelOp>> = Vec::new();
    let mut current: Vec<KernelOp> = Vec::new();

    for op in ops {
        if matches!(op, KernelOp::Barrier) {
            segments.push(core::mem::take(&mut current));
        } else {
            current.push(op.clone());
        }
    }
    // Push the final segment (ops after the last barrier, or all ops if no barriers)
    segments.push(current);
    segments
}

/// Discover CPU devices. Always returns exactly one.
pub fn discover() -> Vec<Box<dyn GpuDevice>> {
    vec![Box::new(CpuDevice::new())]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_conversions() {
        assert_eq!(Value::U32(42).as_u32(), 42);
        assert_eq!(Value::U32(42).as_f32(), 42.0);
        assert_eq!(Value::F32(3.14).as_u32(), 3);
        assert!(Value::U32(1).as_bool());
        assert!(!Value::U32(0).as_bool());
        assert!(Value::Bool(true).as_bool());
    }

    #[test]
    fn scalar_read_write_roundtrip() {
        let mut buf = vec![0u8; 16];
        write_scalar(&mut buf, 0, Value::F32(3.14), &ScalarType::F32);
        let v = read_scalar(&buf, 0, &ScalarType::F32);
        assert!((v.as_f32() - 3.14).abs() < 1e-6);

        write_scalar(&mut buf, 1, Value::U32(42), &ScalarType::U32);
        let v = read_scalar(&buf, 1, &ScalarType::U32);
        assert_eq!(v.as_u32(), 42);
    }

    #[test]
    fn binop_add() {
        let r = eval_binop(Value::U32(3), Value::U32(4), &BinOp::Add, &ScalarType::U32);
        assert_eq!(r.as_u32(), 7);

        let r = eval_binop(
            Value::F32(1.5),
            Value::F32(2.5),
            &BinOp::Add,
            &ScalarType::F32,
        );
        assert!((r.as_f32() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn binop_div_by_zero() {
        let r = eval_binop(Value::U32(10), Value::U32(0), &BinOp::Div, &ScalarType::U32);
        assert_eq!(r.as_u32(), 0);
    }

    #[test]
    fn cmp_ops() {
        let r = eval_cmp(Value::U32(3), Value::U32(5), &CmpOp::Lt, &ScalarType::U32);
        assert!(r.as_bool());

        let r = eval_cmp(Value::U32(5), Value::U32(3), &CmpOp::Lt, &ScalarType::U32);
        assert!(!r.as_bool());
    }

    #[test]
    fn f16_roundtrip() {
        let original = 1.5f32;
        let bits = f32_to_f16(original);
        let back = f16_to_f32(bits);
        assert!((back - original).abs() < 1e-3);
    }

    #[test]
    fn cpu_device_field_alloc_write_read() {
        let dev = CpuDevice::new();
        let handle = dev.field_alloc(16, FieldUsage::default_compute()).unwrap();
        dev.field_write_bytes(handle, &[1, 2, 3, 4]).unwrap();
        let data = dev.field_read_bytes(handle, 16).unwrap();
        assert_eq!(&data[..4], &[1, 2, 3, 4]);
        assert_eq!(&data[4..], &[0; 12]);
        dev.field_free(handle);
    }

    #[test]
    fn cpu_device_caps() {
        let dev = CpuDevice::new();
        assert_eq!(dev.caps().vendor, Vendor::Software);
        assert_eq!(dev.caps().name, "Quanta CPU (software)");
    }

    #[test]
    fn cpu_device_wave_rejects_binary() {
        let dev = CpuDevice::new();
        let result = dev.wave(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn cpu_device_pulse_is_synchronous() {
        let dev = CpuDevice::new();
        let pulse = Pulse {
            handle: 0,
            completed: false,
            wait_fn: None,
        };
        assert!(dev.pulse_poll(&pulse));
    }
}
