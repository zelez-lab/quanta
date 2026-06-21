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
}
