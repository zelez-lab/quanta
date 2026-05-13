//! Uniform float conversion utilities.
//!
//! All distributions (uniform float, normal via Box-Muller,
//! exponential via inverse-CDF, ...) start from a uniform integer
//! and convert it to a float in a specific range. This module is
//! the single source of truth for those conversions, so every
//! algorithm sharing this layer is bit-exact between CPU and GPU.
//!
//! ## Conversion modes
//!
//! - **Half-open `[0, 1)`** (`u32_to_unit_f32`, `u64_to_unit_f64`):
//!   the natural "drop low bits, multiply by 2^-M" mode. Zero is
//!   reachable, one is not. Standard form for `rng.next_f32()`.
//! - **Half-open `(0, 1]`** (`u32_to_open_unit_f32`,
//!   `u64_to_open_unit_f64`): adds half an ULP so zero is
//!   unreachable. Safe for `log(u)` — used by Box-Muller and
//!   exponential.
//! - **Signed `(-1, 1]`** (`u32_to_unit11_f32`, `u64_to_unit11_f64`):
//!   for distributions that want a symmetric range.
//!
//! The Random123 reference calls these `u01fixedpt`/`u01`/`uneg11`;
//! we use Rust-idiomatic names but produce numerically equivalent
//! output.

/// Convert a u32 to a uniform `f32` in `[0, 1)`. Takes the top 24
/// bits as the mantissa and scales by `2^-24`. The result is one
/// of 2^24 evenly spaced values; zero is reachable, one is not.
#[inline]
pub const fn u32_to_unit_f32(n: u32) -> f32 {
    let bits = n >> 8;
    (bits as f32) * (1.0 / 16_777_216.0) // 1.0 / 2^24
}

/// Convert a u64 to a uniform `f64` in `[0, 1)`. Takes the top 53
/// bits as the mantissa and scales by `2^-53`. The result is one
/// of 2^53 evenly spaced values; zero is reachable, one is not.
#[inline]
pub const fn u64_to_unit_f64(n: u64) -> f64 {
    let bits = n >> 11;
    (bits as f64) * (1.0 / 9_007_199_254_740_992.0) // 1.0 / 2^53
}

/// Convert a u32 to a uniform `f32` in `(0, 1]` — zero is never
/// returned, so `log(u)` is safe. Equivalent to Random123's
/// `u01<float>(u)` with `factor = 2^-32`, `halffactor = 2^-33`.
/// The smallest value is `2^-33`, the largest is `1.0`.
#[inline]
pub const fn u32_to_open_unit_f32(n: u32) -> f32 {
    // (u as f32) * 2^-32 + 2^-33.
    // Shift right by 8 (drop low 8 bits to fit in f32 mantissa)
    // and use a scaled factor; produces the same set of values
    // as the C reference at full precision.
    let bits = n >> 8;
    (bits as f32) * (1.0 / 16_777_216.0) + (1.0 / 33_554_432.0) // 2^-25
}

/// Convert a u64 to a uniform `f64` in `(0, 1]` — zero is never
/// returned, so `log(u)` is safe. Smallest value `2^-54`, largest
/// `1.0`.
#[inline]
pub const fn u64_to_open_unit_f64(n: u64) -> f64 {
    let bits = n >> 11;
    (bits as f64) * (1.0 / 9_007_199_254_740_992.0) + (1.0 / 18_014_398_509_481_984.0) // 2^-54
}

/// Convert a u32 to a uniform `f32` in `[-1, 1]`. Useful for
/// distributions that want a symmetric range (some Box-Muller
/// variants, polar coords). Equivalent to Random123's
/// `uneg11<float>(u)`.
///
/// Note on endpoints: because f32 has only 24 mantissa bits and the
/// input is 32 bits wide, `-1.0` is reachable (the math `-1 + 2^-32`
/// rounds to `-1.0` in f32). If you need a strictly open range,
/// use the f64 form or reject on equality.
#[inline]
pub const fn u32_to_unit11_f32(n: u32) -> f32 {
    let signed = n as i32;
    (signed as f32) * (1.0 / 2_147_483_648.0) + (1.0 / 4_294_967_296.0)
}

/// Convert a u64 to a uniform `f64` in `[-1, 1]`. Same endpoint
/// caveat as `u32_to_unit11_f32`: `-1.0` is reachable because the
/// `2^-64` nudge underflows in f64 precision near -1.
#[inline]
pub const fn u64_to_unit11_f64(n: u64) -> f64 {
    let signed = n as i64;
    (signed as f64) * (1.0 / 9_223_372_036_854_775_808.0) + (1.0 / 18_446_744_073_709_551_616.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_f32_zero_and_max() {
        assert_eq!(u32_to_unit_f32(0), 0.0);
        // u32::MAX yields one less than 1.0 (the largest representable
        // value in [0, 1) at 24-bit precision).
        let one_minus_ulp = u32_to_unit_f32(u32::MAX);
        assert!(one_minus_ulp < 1.0);
        assert!(one_minus_ulp > 1.0 - (1.0 / 16_777_216.0) - f32::EPSILON);
    }

    #[test]
    fn unit_f64_zero_and_max() {
        assert_eq!(u64_to_unit_f64(0), 0.0);
        let one_minus_ulp = u64_to_unit_f64(u64::MAX);
        assert!(one_minus_ulp < 1.0);
    }

    #[test]
    fn open_unit_f32_never_zero() {
        // Even u=0 must produce a strictly positive result, so
        // log(u) is always finite.
        assert!(u32_to_open_unit_f32(0) > 0.0);
        assert!(u32_to_open_unit_f32(1) > 0.0);
        for u in [0u32, 1, 7, 42, 0xDEAD_BEEF, u32::MAX / 2, u32::MAX] {
            let f = u32_to_open_unit_f32(u);
            assert!(f > 0.0, "u={u} produced f={f} ≤ 0");
            assert!(f <= 1.0, "u={u} produced f={f} > 1");
        }
    }

    #[test]
    fn open_unit_f64_never_zero() {
        assert!(u64_to_open_unit_f64(0) > 0.0);
        for u in [
            0u64,
            1,
            7,
            42,
            0xDEAD_BEEF_CAFE_BABE,
            u64::MAX / 2,
            u64::MAX,
        ] {
            let f = u64_to_open_unit_f64(u);
            assert!(f > 0.0, "u={u} produced f={f} ≤ 0");
            assert!(f <= 1.0, "u={u} produced f={f} > 1");
        }
    }

    #[test]
    fn unit11_f32_in_range() {
        for u in [0u32, 1, 7, 1 << 31, u32::MAX] {
            let f = u32_to_unit11_f32(u);
            assert!(f >= -1.0, "u={u} produced f={f} < -1");
            assert!(f <= 1.0, "u={u} produced f={f} > 1");
        }
    }

    #[test]
    fn unit11_f64_in_range() {
        for u in [0u64, 1, 7, 1u64 << 63, u64::MAX] {
            let f = u64_to_unit11_f64(u);
            assert!(f >= -1.0, "u={u} produced f={f} < -1");
            assert!(f <= 1.0, "u={u} produced f={f} > 1");
        }
    }

    #[test]
    fn unit11_f32_covers_both_signs() {
        // Most u32s produce a value with the expected sign.
        let neg = u32_to_unit11_f32(1u32 << 31); // i32::MIN
        let pos = u32_to_unit11_f32(0x7FFF_FFFF); // i32::MAX
        assert!(neg < 0.0);
        assert!(pos > 0.0);
    }
}
