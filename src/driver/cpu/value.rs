//! Value type and scalar read/write helpers for the CPU driver.

use quanta_ir::{ConstValue, ScalarType};

// ── Value type ───────────────────────────────────────────────────────────────

/// Runtime value held in a virtual register.
#[derive(Debug, Clone, Copy)]
pub(super) enum Value {
    F32(f32),
    F64(f64),
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    Bool(bool),
}

impl Value {
    pub(super) fn as_u32(self) -> u32 {
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

    pub(super) fn as_u64(self) -> u64 {
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

    pub(super) fn as_i32(self) -> i32 {
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

    pub(super) fn as_i64(self) -> i64 {
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

    pub(super) fn as_f32(self) -> f32 {
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

    pub(super) fn as_f64(self) -> f64 {
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

    pub(super) fn as_bool(self) -> bool {
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

pub(super) fn scalar_size(ty: &ScalarType) -> usize {
    match ty {
        ScalarType::Bool | ScalarType::U8 | ScalarType::I8 => 1,
        ScalarType::F16 | ScalarType::BF16 | ScalarType::U16 | ScalarType::I16 => 2,
        ScalarType::F32 | ScalarType::U32 | ScalarType::I32 => 4,
        ScalarType::F64 | ScalarType::U64 | ScalarType::I64 => 8,
    }
}

/// Read a scalar at a byte offset (not an element index). Used by
/// the push-constant Load path on the CPU backend, where slots are
/// laid out at fixed 16-byte boundaries inside the push-data buffer.
pub(super) fn read_scalar_at_offset(buf: &[u8], offset: usize, ty: &ScalarType) -> Value {
    let size = scalar_size(ty);
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
            let bits = u16::from_le_bytes(bytes.try_into().unwrap());
            Value::F32(f16_to_f32(bits))
        }
        ScalarType::BF16 => Value::F32(bf16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap()))),
        ScalarType::Bool => Value::Bool(bytes[0] != 0),
    }
}

pub(super) fn read_scalar(buf: &[u8], index: u32, ty: &ScalarType) -> Value {
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
        ScalarType::BF16 => Value::F32(bf16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap()))),
        ScalarType::Bool => Value::Bool(bytes[0] != 0),
    }
}

pub(super) fn write_scalar(buf: &mut [u8], index: u32, val: Value, ty: &ScalarType) {
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
        ScalarType::BF16 => {
            let bits = f32_to_bf16(val.as_f32());
            dest.copy_from_slice(&bits.to_le_bytes());
        }
        ScalarType::Bool => dest[0] = val.as_bool() as u8,
    }
}

/// bfloat16 → f32: bf16 is the top 16 bits of an f32, so place them back.
pub(super) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// f32 → bfloat16, round-to-nearest-even. NaN is preserved (kept quiet by
/// forcing a mantissa bit). This is the inverse of `bf16_to_f32` for every
/// value representable in bf16, so the round-trip is exact there.
pub(super) fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    if val.is_nan() {
        // Keep it a NaN after truncation (set a high mantissa bit).
        return ((bits >> 16) as u16) | 0x0040;
    }
    // Round-to-nearest-even: add the rounding bias then truncate.
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

/// IEEE 754 half-precision to single-precision.
pub(super) fn f16_to_f32(bits: u16) -> f32 {
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
            let f32_exp = 127 - 15 - e + 1;
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
pub(super) fn f32_to_f16(val: f32) -> u16 {
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

pub(super) fn value_from_const(cv: &ConstValue) -> Value {
    match cv {
        ConstValue::F32(v) => Value::F32(*v),
        ConstValue::F64(v) => Value::F64(*v),
        ConstValue::U32(v) => Value::U32(*v),
        ConstValue::U64(v) => Value::U64(*v),
        ConstValue::I32(v) => Value::I32(*v),
        ConstValue::I64(v) => Value::I64(*v),
        ConstValue::Bool(v) => Value::Bool(*v),
        ConstValue::F16(bits) => Value::F32(f16_to_f32(*bits)),
        ConstValue::BF16(bits) => Value::F32(bf16_to_f32(*bits)),
    }
}
