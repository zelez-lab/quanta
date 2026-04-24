//! Kani harnesses for f16 <-> f32 conversion precision.
//!
//! Verifies the f16_to_f32 and f32_to_f16 functions from
//! `src/driver/cpu/value.rs`.
//!
//! Run: cargo kani --harness verify_f16_roundtrip_normal
//!      cargo kani --harness verify_f16_roundtrip_special
//!      cargo kani --harness verify_f16_zero_roundtrip
//!      cargo kani --harness verify_f16_infinity_roundtrip
//!      cargo kani --harness verify_f16_nan_preserves_category
//!      cargo kani --harness verify_f16_subnormal_category
//!
//! | Theorem | What it proves |
//! |---------|----------------|
//! | T1300a  | Normal f16 values roundtrip exactly: f32_to_f16(f16_to_f32(bits)) == bits. |
//! | T1300b  | Zero roundtrips: +0 and -0 are preserved. |
//! | T1300c  | Infinity roundtrips: +inf and -inf are preserved. |
//! | T1300d  | NaN preserves category: NaN input produces NaN output. |
//! | T1300e  | Subnormal category: subnormal f16 produces finite f32. |

// ── Inline copies of the production functions ──────────────────────
//
// These are exact copies of f16_to_f32 and f32_to_f16 from
// src/driver/cpu/value.rs. Any change to the production code must
// be mirrored here (caught by human review at commit time).

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

// ── Helper: classify f16 bit patterns ──────────────────────────────

/// Returns true if the f16 bit pattern represents a normal number.
/// Normal: exponent in [1, 30] (not 0 = zero/subnormal, not 31 = inf/nan).
fn is_f16_normal(bits: u16) -> bool {
    let exp = (bits >> 10) & 0x1F;
    exp >= 1 && exp <= 30
}

/// Returns true if the f16 bit pattern represents +0 or -0.
fn is_f16_zero(bits: u16) -> bool {
    (bits & 0x7FFF) == 0
}

/// Returns true if the f16 bit pattern represents +inf or -inf.
fn is_f16_infinity(bits: u16) -> bool {
    let exp = (bits >> 10) & 0x1F;
    let frac = bits & 0x3FF;
    exp == 31 && frac == 0
}

/// Returns true if the f16 bit pattern represents NaN.
fn is_f16_nan(bits: u16) -> bool {
    let exp = (bits >> 10) & 0x1F;
    let frac = bits & 0x3FF;
    exp == 31 && frac != 0
}

/// Returns true if the f16 bit pattern represents a subnormal.
fn is_f16_subnormal(bits: u16) -> bool {
    let exp = (bits >> 10) & 0x1F;
    let frac = bits & 0x3FF;
    exp == 0 && frac != 0
}

// ── T1300a: Normal f16 roundtrip ───────────────────────────────────

/// For all normal f16 bit patterns, f32_to_f16(f16_to_f32(bits)) == bits.
///
/// Normal f16 values have exponent in [1, 30]. The mantissa has 10 bits,
/// which fit exactly in f32's 23-bit mantissa. The exponent bias
/// difference (127 - 15 = 112) is a simple offset. Therefore the
/// roundtrip is exact for all normal values.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(12)]
fn verify_f16_roundtrip_normal() {
    let bits: u16 = kani::any();
    kani::assume(is_f16_normal(bits));

    let f32_val = f16_to_f32(bits);
    let roundtrip = f32_to_f16(f32_val);

    assert!(roundtrip == bits, "Normal f16 roundtrip failed for bits={}", bits);
}

// ── T1300b: Zero roundtrip ─────────────────────────────────────────

/// +0 (0x0000) and -0 (0x8000) roundtrip exactly.
#[cfg(kani)]
#[kani::proof]
fn verify_f16_zero_roundtrip() {
    let bits: u16 = kani::any();
    kani::assume(is_f16_zero(bits));

    let f32_val = f16_to_f32(bits);
    let roundtrip = f32_to_f16(f32_val);

    assert!(roundtrip == bits, "Zero roundtrip failed");
}

// ── T1300c: Infinity roundtrip ─────────────────────────────────────

/// +inf (0x7C00) and -inf (0xFC00) roundtrip exactly.
#[cfg(kani)]
#[kani::proof]
fn verify_f16_infinity_roundtrip() {
    let bits: u16 = kani::any();
    kani::assume(is_f16_infinity(bits));

    let f32_val = f16_to_f32(bits);
    let roundtrip = f32_to_f16(f32_val);

    assert!(roundtrip == bits, "Infinity roundtrip failed");
}

// ── T1300d: NaN preserves category ─────────────────────────────────

/// Any f16 NaN, when converted to f32 and back, remains a NaN.
///
/// Note: the specific NaN payload may not be preserved (f16_to_f32
/// maps all NaNs to f32::NAN, which has a canonical payload). But
/// the NaN *category* is preserved: NaN in -> NaN out.
#[cfg(kani)]
#[kani::proof]
fn verify_f16_nan_preserves_category() {
    let bits: u16 = kani::any();
    kani::assume(is_f16_nan(bits));

    let f32_val = f16_to_f32(bits);
    // f16_to_f32 returns f32::NAN for all f16 NaN inputs
    assert!(f32_val.is_nan(), "f16 NaN did not produce f32 NaN");

    let roundtrip = f32_to_f16(f32_val);
    // f32_to_f16 maps NaN to an f16 NaN (exp=31, frac!=0)
    assert!(is_f16_nan(roundtrip), "NaN category not preserved on roundtrip");
}

// ── T1300e: Subnormal category preservation ────────────────────────

/// Subnormal f16 values produce finite, non-NaN f32 values.
///
/// Subnormals do NOT roundtrip exactly because f32_to_f16 truncates
/// the mantissa and may underflow to zero. But the category is
/// preserved: subnormal f16 -> finite f32 (never inf or NaN).
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(12)]
fn verify_f16_subnormal_category() {
    let bits: u16 = kani::any();
    kani::assume(is_f16_subnormal(bits));

    let f32_val = f16_to_f32(bits);
    // Subnormal f16 -> finite f32 (the value is small but representable)
    assert!(f32_val.is_finite(), "Subnormal f16 produced non-finite f32");
    assert!(!f32_val.is_nan(), "Subnormal f16 produced NaN f32");
}

// ── Combined: all f16 bit patterns maintain category ───────────────

/// For every possible u16 bit pattern interpreted as f16:
///   - zero -> zero
///   - normal -> same bits (exact roundtrip)
///   - infinity -> same bits (exact roundtrip)
///   - NaN -> NaN (category preserved)
///   - subnormal -> finite f32 (no inf/NaN corruption)
///
/// This is the comprehensive category-preservation theorem.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(12)]
fn verify_f16_roundtrip_special() {
    let bits: u16 = kani::any();

    let f32_val = f16_to_f32(bits);
    let roundtrip = f32_to_f16(f32_val);

    if is_f16_normal(bits) {
        // Normal values roundtrip exactly
        assert!(roundtrip == bits, "Normal roundtrip failed");
    } else if is_f16_zero(bits) {
        // Zeros roundtrip exactly
        assert!(roundtrip == bits, "Zero roundtrip failed");
    } else if is_f16_infinity(bits) {
        // Infinities roundtrip exactly
        assert!(roundtrip == bits, "Infinity roundtrip failed");
    } else if is_f16_nan(bits) {
        // NaN category preserved (payload may differ)
        assert!(is_f16_nan(roundtrip), "NaN category lost");
    } else {
        // Subnormal: f32 result is finite
        assert!(is_f16_subnormal(bits));
        assert!(f32_val.is_finite(), "Subnormal produced non-finite f32");
    }
}
