//! Threefry4×32-20 counter-based RNG — pure Rust port.
//!
//! Reference: D. E. Shaw Research, "Parallel Random Numbers: As Easy
//! as 1, 2, 3" (SC11). The bit-exact algorithm matches
//! `random123/include/Random123/threefry.h`. Threefry is a
//! non-cryptographic adaptation of the Threefish block cipher from
//! the Skein hash function; output passes the full TestU01
//! BigCrush battery.
//!
//! ## Why two counter-based generators?
//!
//! Philox uses integer multiplication; Threefry uses only rotates
//! and XOR. On hardware with weak/expensive 32-bit multiply (some
//! older GPUs, scalar SIMD lanes on certain CPUs), Threefry can be
//! faster. The two also have different statistical fingerprints —
//! some applications prefer one for the same reason simulation
//! groups validate with multiple independent generators.
//!
//! ## State and output shape
//!
//! - **counter**: 4×u32 (128-bit).
//! - **key**: 4×u32 (128-bit) — same shape as the counter, unlike
//!   Philox where the key is half-width.
//! - **output**: 4×u32 per `threefry4x32_20` call.
//!
//! ## Bit-exact with the reference
//!
//! `tests/threefry_kat.rs` runs the published known-answer vectors
//! from `random123/tests/kat_vectors` through this implementation
//! and asserts equality.

/// 4×u32 counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counter(pub [u32; 4]);

/// 4×u32 key. Note the key is the same width as the counter for
/// Threefry (unlike Philox, where the key is half-width).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key(pub [u32; 4]);

/// Threefry key-schedule parity constant (from Skein Hash Function
/// specification, 32-bit variant). XORed into the extended-key word
/// `ks4` derived from the four user-supplied key words.
const SKEIN_KS_PARITY_32: u32 = 0x1BD1_1BDA;

/// Rotation constants for the 4×32 variant, indexed by round
/// number mod 8. `R[j]` = `(R_32x4_j_0, R_32x4_j_1)` from
/// `random123/include/Random123/threefry.h`.
const ROTATIONS: [(u32, u32); 8] = [
    (10, 26),
    (11, 21),
    (13, 27),
    (23, 5),
    (6, 20),
    (17, 11),
    (25, 10),
    (18, 20),
];

/// Standard round count — 20 rounds is the published variant
/// (`threefry4x32_20`) that passes BigCrush.
pub const ROUNDS: u32 = 20;

/// Threefry4×32-R bijection: run `R` rounds starting from
/// `(ctr, key)`. Each round is two parallel "mix" operations on
/// the 4-word state; every 4 rounds a key-schedule injection
/// re-stirs the state with a rotated subkey.
#[inline]
pub const fn threefry4x32_r(rounds: u32, ctr: Counter, key: Key) -> Counter {
    // Extended key: ks0..ks3 are the four user-supplied key words,
    // ks4 = parity XOR all four. The injection schedule cycles
    // ks0..ks4 with a (r+1) increment on the last word.
    let ks0 = key.0[0];
    let ks1 = key.0[1];
    let ks2 = key.0[2];
    let ks3 = key.0[3];
    let ks4 = SKEIN_KS_PARITY_32 ^ ks0 ^ ks1 ^ ks2 ^ ks3;

    // Initial key injection: ctr + key word-by-word.
    let mut x0 = ctr.0[0].wrapping_add(ks0);
    let mut x1 = ctr.0[1].wrapping_add(ks1);
    let mut x2 = ctr.0[2].wrapping_add(ks2);
    let mut x3 = ctr.0[3].wrapping_add(ks3);

    // Even rounds (j%2==0) pair (X0,X1) and (X2,X3); odd rounds
    // pair (X0,X3) and (X2,X1). The rotation index cycles 0..8.
    let mut round_idx: u32 = 0;
    while round_idx < rounds {
        let (r0, r1) = ROTATIONS[(round_idx % 8) as usize];
        if round_idx.is_multiple_of(2) {
            x0 = x0.wrapping_add(x1);
            x1 = x1.rotate_left(r0);
            x1 ^= x0;
            x2 = x2.wrapping_add(x3);
            x3 = x3.rotate_left(r1);
            x3 ^= x2;
        } else {
            x0 = x0.wrapping_add(x3);
            x3 = x3.rotate_left(r0);
            x3 ^= x0;
            x2 = x2.wrapping_add(x1);
            x1 = x1.rotate_left(r1);
            x1 ^= x2;
        }
        // Key injection every 4 rounds (after rounds 3, 7, 11, ...).
        if round_idx % 4 == 3 {
            let inject = (round_idx + 1) / 4; // 1, 2, 3, ...
            // Rotated key schedule: each injection cycles ks_(i+r) % 5.
            // Pattern matches the reference's hand-unrolled InjectKey
            // blocks at rounds>3/>7/>11/>15/>19.
            let ks = [ks0, ks1, ks2, ks3, ks4];
            x0 = x0.wrapping_add(ks[(inject as usize) % 5]);
            x1 = x1.wrapping_add(ks[((inject as usize) + 1) % 5]);
            x2 = x2.wrapping_add(ks[((inject as usize) + 2) % 5]);
            x3 = x3
                .wrapping_add(ks[((inject as usize) + 3) % 5])
                .wrapping_add(inject);
        }
        round_idx += 1;
    }
    Counter([x0, x1, x2, x3])
}

/// Standard Threefry4×32-20. Returns four u32s for one
/// `(counter, key)` pair.
#[inline]
pub const fn threefry4x32_20(ctr: Counter, key: Key) -> Counter {
    threefry4x32_r(ROUNDS, ctr, key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotations_are_in_range() {
        for &(r0, r1) in &ROTATIONS {
            assert!(r0 < 32);
            assert!(r1 < 32);
        }
    }

    #[test]
    fn threefry_is_deterministic() {
        let ctr = Counter([0xDEAD_BEEF, 0xCAFE_BABE, 0x1234_5678, 0x9ABC_DEF0]);
        let key = Key([0xA5A5_A5A5, 0x5A5A_5A5A, 0xC0FF_EE00, 0x0BAD_F00D]);
        let a = threefry4x32_20(ctr, key);
        let b = threefry4x32_20(ctr, key);
        assert_eq!(a, b);
    }

    #[test]
    fn distinct_counters_produce_distinct_output() {
        let key = Key([0xC0FF_EE00, 0x0BAD_F00D, 0x1234_5678, 0x9ABC_DEF0]);
        let a = threefry4x32_20(Counter([0, 0, 0, 0]), key);
        let b = threefry4x32_20(Counter([1, 0, 0, 0]), key);
        assert_ne!(a, b);
    }
}
