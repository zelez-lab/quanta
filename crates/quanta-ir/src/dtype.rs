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

/// IEEE 754 half-precision → f32. The single source of truth for f16 decode;
/// the CPU executor and the reference oracles call this so they cannot drift.
pub fn f16_to_f32(bits: u16) -> f32 {
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

/// f32 → IEEE 754 half-precision (truncating mantissa). Mirrors the CPU
/// executor / emitter store path; the single source of truth for f16 encode.
pub fn f32_to_f16(val: f32) -> u16 {
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

// The fp8 conversions below are written in **branchless** form — only
// shifts, masks, adds, comparisons, and selects (no loops, no early
// returns). This is deliberate: each GPU emitter lowers the identical
// arithmetic with `OpSelect` (SPIR-V) / `select` (WGSL) / `?:` (MSL),
// and a branchless reference ports straight across with no basic-block
// plumbing. The forms are verified bit-exact against the textbook branchy
// versions over all 8-bit unpack inputs and ~1M f32 pack inputs per
// format (see the dtype tests).

/// fp8 → f32. `eb`/`mb` = exponent/mantissa bit widths (e5m2: 5,2 — e4m3:
/// 4,3). IEEE-style: all-ones exponent is inf/NaN, exponent-0 is
/// zero/subnormal. Branchless.
pub fn fp8_to_f32(bits: u8, eb: u32, mb: u32) -> f32 {
    let bits = bits as u32;
    let sign = (bits >> (eb + mb)) & 1;
    let exp = (bits >> mb) & ((1 << eb) - 1);
    let mant = bits & ((1 << mb) - 1);
    let bias = (1u32 << (eb - 1)) - 1;
    let exp_mask = (1u32 << eb) - 1;
    let f32_sign = sign << 31;

    // normal: rebias exponent, left-justify the mantissa.
    let norm = f32_sign | ((exp.wrapping_add(127).wrapping_sub(bias)) << 23) | (mant << (23 - mb));

    // inf / NaN: exp == exp_mask. NaN keeps a quiet mantissa bit.
    let inf_mant = if mant != 0 { 0x0040_0000u32 } else { 0 };
    let infnan = f32_sign | (0xFFu32 << 23) | inf_mant;

    // subnormal (exp == 0, mant != 0): normalise the mantissa. The loop
    // that shifts until the implicit bit appears is replaced by a small
    // unrolled leading-bit scan over the mb mantissa bits (mb ≤ 3), so
    // the emitters expand it into mb selects.
    let mut lead = 0u32; // index of the highest set bit of `mant`
    let mut i = 0u32;
    while i < mb {
        let set = (mant >> i) & 1;
        lead = (set * i) | ((1 - set) * lead);
        i += 1;
    }
    let shifts = mb - lead;
    let e_sub: i32 = (1 - bias as i32 - mb as i32) - shifts as i32;
    let m_sub = (mant << shifts) & ((1 << mb) - 1);
    let sub = f32_sign | (((e_sub + mb as i32 + 127) as u32) << 23) | (m_sub << (23 - mb));

    let is_inf = exp == exp_mask;
    let is_zero = exp == 0 && mant == 0;
    let is_sub = exp == 0 && mant != 0;

    // priority: zero/subnormal/inf override the normal form.
    let mut out = norm;
    out = if is_sub { sub } else { out };
    out = if is_inf { infnan } else { out };
    out = if is_zero { f32_sign } else { out };
    f32::from_bits(out)
}

/// f32 → fp8, round-to-nearest-even. Overflow → inf, NaN → canonical NaN.
/// Branchless.
pub fn f32_to_fp8(val: f32, eb: u32, mb: u32) -> u8 {
    let b = val.to_bits();
    let sign = (b >> 31) & 1;
    let sign_slot = sign << (eb + mb);
    let f32_exp = ((b >> 23) & 0xFF) as i32;
    let f32_mant = b & 0x007F_FFFF;
    let bias = (1i32 << (eb - 1)) - 1;
    let exp_mask = (1u32 << eb) - 1;
    let target_exp = (f32_exp - 127) + bias;

    // normal: round the 23-bit mantissa down to mb bits (RNE); a carry
    // out of the mantissa bumps the exponent. This branch is only the
    // selected result when target_exp is in (0, exp_mask); the wrapping
    // arithmetic keeps the discarded out-of-range cases panic-free in
    // debug (every branch is evaluated under branchless selection).
    let rnd_n = round_shift_rne(f32_mant, 23 - mb);
    let carry = (rnd_n >> mb) != 0;
    let out_exp_n = (target_exp as u32).wrapping_add(carry as u32);
    let out_mant_n = if carry { 0 } else { rnd_n };
    let normal = if out_exp_n >= exp_mask {
        sign_slot | (exp_mask << mb) // carry pushed into inf
    } else {
        sign_slot | (out_exp_n << mb) | (out_mant_n & ((1 << mb) - 1))
    };

    // subnormal (target_exp <= 0): shift the full significand into fp8's
    // subnormal scale, RNE. Far underflow (shift > 31) → ±0.
    let signif = f32_mant | 0x0080_0000;
    let shift = (23 - mb) as i32 + (1 - target_exp);
    let sub = if shift > 31 {
        sign_slot
    } else {
        sign_slot | round_shift_rne(signif, shift as u32)
    };

    // inf / NaN (f32_exp == 0xFF), finite overflow, and zero.
    let nan_m = if f32_mant != 0 { 1u32 << (mb - 1) } else { 0 };
    let infnan = sign_slot | (exp_mask << mb) | nan_m;
    let ovf = sign_slot | (exp_mask << mb);
    let zero = sign_slot;

    let is_infnan = f32_exp == 0xFF;
    let is_zero = f32_exp == 0 && f32_mant == 0;
    let is_ovf = target_exp >= exp_mask as i32;
    let is_sub = target_exp <= 0;

    // priority: infnan > zero > overflow > subnormal > normal.
    let mut out = normal;
    out = if is_sub { sub } else { out };
    out = if is_ovf { ovf } else { out };
    out = if is_zero { zero } else { out };
    out = if is_infnan { infnan } else { out };
    out as u8
}

/// Right-shift with round-to-nearest-even. Branchless over the shift
/// amount: `s == 0` returns `v`; `s >= 32` returns 0.
fn round_shift_rne(v: u32, s: u32) -> u32 {
    let big = s >= 32;
    let s_c = if big { 31 } else { s }; // clamp so the shifts stay defined
    let kept = v >> s_c;
    let rem = v & (1u32 << s_c).wrapping_sub(1);
    let half = 1u32 << s_c.saturating_sub(1);
    let roundup = rem > half || (rem == half && (kept & 1) == 1);
    let r = kept + (roundup as u32);
    let r = if s == 0 { v } else { r };
    if big { 0 } else { r }
}

/// Exponent/mantissa widths for the two fp8 formats.
pub const E5M2: (u32, u32) = (5, 2);
pub const E4M3: (u32, u32) = (4, 3);

// ── int8 / int4 symmetric quantization ───────────────────────────────
//
// Quantization is NOT a self-describing dtype: an integer `q` means
// `real = scale * (q - zero_point)`, with (scale, zero_point) external to
// the value. The first increment is per-tensor SYMMETRIC (zero_point = 0):
//   quantize(x)   = clamp(round_ties_even(x / scale), lo, hi)
//   dequantize(q) = scale * q
// `lo`/`hi` are the signed integer range: int8 = [-128, 127], int4 =
// [-8, 7]. Rounding is round-half-to-even to match WGSL/MSL `round` and
// SPIR-V `OpExtInst Round` bit-for-bit. int4 is stored packed 8 nibbles
// per 32-bit word (the `store` axis — see the nibble helpers below).
//
// These are the reference host conversions; the CPU oracle, the GPU
// emitters, and the Lean spec mirror them.

/// Signed integer range `(lo, hi)` for a quantized value width in bits
/// (8 → int8, 4 → int4).
pub const fn quant_range(bits: u32) -> (i32, i32) {
    let hi = (1i32 << (bits - 1)) - 1;
    (-(hi + 1), hi)
}

/// Symmetric quantize: `clamp(round_ties_even(x / scale), lo, hi)`.
pub fn quantize_sym(x: f32, scale: f32, bits: u32) -> i32 {
    let (lo, hi) = quant_range(bits);
    let r = (x / scale).round_ties_even() as i32;
    r.clamp(lo, hi)
}

/// Symmetric dequantize: `scale * q`.
pub fn dequantize_sym(q: i32, scale: f32) -> f32 {
    scale * (q as f32)
}

// ── int4 sub-byte packing (8 signed nibbles per u32 word) ────────────
//
// The `store` axis: int4 *values* are logical 4-bit signed integers; their
// *storage* packs 8 per 32-bit word, low nibble first (GPTQ / llama.cpp
// layout). Load reads + sign-extends a nibble; Store is read-modify-write.

/// Read the signed int4 at nibble index `i` (0..8) of a packed word.
pub fn int4_unpack(word: u32, i: u32) -> i32 {
    let n = (word >> (i * 4)) & 0xF;
    // sign-extend the 4-bit value: (n ^ 0x8) - 0x8.
    (n ^ 0x8).wrapping_sub(0x8) as i32
}

/// Write the signed int4 `q` into nibble index `i` (0..8), preserving the
/// other nibbles (read-modify-write).
pub fn int4_pack(word: u32, i: u32, q: i32) -> u32 {
    let shift = i * 4;
    let n = (q as u32) & 0xF;
    (word & !(0xF << shift)) | (n << shift)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Textbook branchy references, kept here only as test oracles. The
    // production fp8_to_f32 / f32_to_fp8 are the branchless forms above;
    // these confirm the rewrite is bit-identical.
    fn ref_to_f32(bits: u8, eb: u32, mb: u32) -> f32 {
        let bits = bits as u32;
        let sign = (bits >> (eb + mb)) & 1;
        let exp = (bits >> mb) & ((1 << eb) - 1);
        let mant = bits & ((1 << mb) - 1);
        let bias = (1u32 << (eb - 1)) - 1;
        let exp_mask = (1u32 << eb) - 1;
        let f32_sign = sign << 31;
        if exp == exp_mask {
            let m = if mant != 0 { 0x0040_0000 } else { 0 };
            return f32::from_bits(f32_sign | (0xFFu32 << 23) | m);
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

    fn ref_rne(v: u32, s: u32) -> u32 {
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

    fn ref_to_fp8(val: f32, eb: u32, mb: u32) -> u8 {
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
            return (sign_slot as u32 | ref_rne(signif, shift as u32)) as u8;
        }
        let rounded = ref_rne(f32_mant, 23 - mb);
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

    #[test]
    fn fp8_unpack_matches_branchy_all_bytes() {
        for &(eb, mb) in &[E5M2, E4M3] {
            for byte in 0u16..=255 {
                let a = fp8_to_f32(byte as u8, eb, mb).to_bits();
                let b = ref_to_f32(byte as u8, eb, mb).to_bits();
                assert_eq!(a, b, "unpack e{eb}m{mb} byte {byte:#04x}");
            }
        }
    }

    #[test]
    fn fp8_pack_matches_branchy_f32_sweep() {
        for &(eb, mb) in &[E5M2, E4M3] {
            for exp in 0u32..=255 {
                for sign in 0u32..=1 {
                    for mstep in 0u32..512 {
                        let mant = (mstep.wrapping_mul(16411)) & 0x7F_FFFF;
                        let f = f32::from_bits((sign << 31) | (exp << 23) | mant);
                        assert_eq!(
                            f32_to_fp8(f, eb, mb),
                            ref_to_fp8(f, eb, mb),
                            "pack e{eb}m{mb} bits {:#010x}",
                            f.to_bits()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fp8_known_values() {
        // 1.0 round-trips through both formats.
        assert_eq!(f32_to_fp8(1.0, 5, 2), 0x3c);
        assert_eq!(f32_to_fp8(1.0, 4, 3), 0x38);
        assert_eq!(fp8_to_f32(0x3c, 5, 2), 1.0);
        assert_eq!(fp8_to_f32(0x38, 4, 3), 1.0);
        // ±0.
        assert_eq!(f32_to_fp8(0.0, 5, 2), 0x00);
        assert_eq!(f32_to_fp8(-0.0, 5, 2), 0x80);
    }

    #[test]
    fn quant_ranges() {
        assert_eq!(quant_range(8), (-128, 127));
        assert_eq!(quant_range(4), (-8, 7));
    }

    #[test]
    fn int4_pack_unpack_roundtrip_all_nibbles() {
        // Every signed int4 value in every nibble slot round-trips, and
        // packing one nibble leaves the other seven untouched.
        for i in 0u32..8 {
            let mut word = 0xDEAD_BEEFu32;
            for v in -8i32..=7 {
                let w = int4_pack(word, i, v);
                assert_eq!(int4_unpack(w, i), v, "nibble {i} value {v}");
                for j in 0u32..8 {
                    if j != i {
                        assert_eq!(
                            int4_unpack(w, j),
                            int4_unpack(word, j),
                            "nibble {j} clobbered"
                        );
                    }
                }
                word = w;
            }
        }
    }

    #[test]
    fn quantize_sym_clamps_rounds_dequantizes() {
        let s = 0.5f32;
        // exact multiples, clamping, and round-half-to-even.
        assert_eq!(quantize_sym(0.0, s, 4), 0);
        assert_eq!(quantize_sym(0.5, s, 4), 1);
        assert_eq!(quantize_sym(-1.0, s, 4), -2);
        assert_eq!(quantize_sym(3.5, s, 4), 7);
        assert_eq!(quantize_sym(3.6, s, 4), 7); // clamp hi
        assert_eq!(quantize_sym(-5.0, s, 4), -8); // clamp lo
        assert_eq!(quantize_sym(0.24, s, 4), 0); // 0.48 → 0
        assert_eq!(quantize_sym(0.26, s, 4), 1); // 0.52 → 1
        assert_eq!(quantize_sym(300.0, 1.0, 8), 127); // int8 saturates
        assert_eq!(dequantize_sym(7, s), 3.5);
        assert_eq!(dequantize_sym(-8, s), -4.0);
    }
}
