//! Narrow-float bit conversions shared by the emitters and case generator.
//!
//! These are the *reference* host-side conversions. Each GPU emitter emits
//! the identical arithmetic inline (so device results match bit-for-bit),
//! and the CPU executor mirrors them. Parameterised by exponent/mantissa
//! widths so e5m2 and e4m3 share one implementation.

/// bf16 → f32: bf16 is the top 16 bits of an f32.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// f32 → bf16, round-to-nearest-even.
pub fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    if val.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + bias) >> 16) as u16
}

/// fp8 → f32. `eb`/`mb` = exponent/mantissa bit widths (e5m2: 5,2 — e4m3:
/// 4,3). IEEE-style: all-ones exponent is inf/NaN, exponent-0 is
/// zero/subnormal.
pub fn fp8_to_f32(bits: u8, eb: u32, mb: u32) -> f32 {
    let bits = bits as u32;
    let sign = (bits >> (eb + mb)) & 1;
    let exp = (bits >> mb) & ((1 << eb) - 1);
    let mant = bits & ((1 << mb) - 1);
    let bias = (1u32 << (eb - 1)) - 1;
    let exp_mask = (1u32 << eb) - 1;

    let f32_sign = sign << 31;
    if exp == exp_mask {
        let f32_mant = if mant != 0 { 0x0040_0000 } else { 0 };
        return f32::from_bits(f32_sign | (0xFFu32 << 23) | f32_mant);
    }
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(f32_sign);
        }
        let mut e: i32 = 1 - bias as i32 - mb as i32;
        let mut m = mant;
        while m & (1 << mb) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= (1 << mb) - 1;
        let f32_exp = ((e + mb as i32 + 127) as u32) << 23;
        return f32::from_bits(f32_sign | f32_exp | (m << (23 - mb)));
    }
    let f32_exp = (exp + 127 - bias) << 23;
    f32::from_bits(f32_sign | f32_exp | (mant << (23 - mb)))
}

/// f32 → fp8, round-to-nearest-even. Overflow → inf, NaN → canonical NaN.
pub fn f32_to_fp8(val: f32, eb: u32, mb: u32) -> u8 {
    let b = val.to_bits();
    let sign = ((b >> 31) & 1) as u8;
    let sign_slot = sign << (eb + mb);
    let f32_exp = ((b >> 23) & 0xFF) as i32;
    let f32_mant = b & 0x007F_FFFF;
    let bias = (1i32 << (eb - 1)) - 1;
    let exp_mask = (1u32 << eb) - 1;

    if f32_exp == 0xFF {
        let m = if f32_mant != 0 { 1u32 << (mb - 1) } else { 0 };
        return (sign_slot as u32 | (exp_mask << mb) | m) as u8;
    }
    if f32_exp == 0 && f32_mant == 0 {
        return sign_slot;
    }
    let target_exp = (f32_exp - 127) + bias;
    if target_exp >= exp_mask as i32 {
        return (sign_slot as u32 | (exp_mask << mb)) as u8;
    }
    let signif = f32_mant | 0x0080_0000;
    if target_exp <= 0 {
        let shift = (23 - mb) as i32 + (1 - target_exp);
        if shift > 31 {
            return sign_slot;
        }
        let rounded = round_shift_rne(signif, shift as u32);
        return (sign_slot as u32 | rounded) as u8;
    }
    let rounded = round_shift_rne(f32_mant, 23 - mb);
    let mut out_exp = target_exp as u32;
    let mut out_mant = rounded;
    if out_mant >> mb != 0 {
        out_mant = 0;
        out_exp += 1;
        if out_exp >= exp_mask {
            return (sign_slot as u32 | (exp_mask << mb)) as u8;
        }
    }
    (sign_slot as u32 | (out_exp << mb) | (out_mant & ((1 << mb) - 1))) as u8
}

/// Right-shift with round-to-nearest-even.
fn round_shift_rne(v: u32, s: u32) -> u32 {
    if s == 0 {
        return v;
    }
    if s >= 32 {
        return 0;
    }
    let kept = v >> s;
    let rem = v & ((1u32 << s) - 1);
    let half = 1u32 << (s - 1);
    if rem > half || (rem == half && (kept & 1) == 1) {
        kept + 1
    } else {
        kept
    }
}

/// Exponent/mantissa widths for the two fp8 formats.
pub const E5M2: (u32, u32) = (5, 2);
pub const E4M3: (u32, u32) = (4, 3);
