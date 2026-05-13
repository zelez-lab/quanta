//! xoshiro128++ core algorithm — u32-only variant.
//!
//! Reference: David Blackman & Sebastiano Vigna, "Scrambled Linear
//! Pseudorandom Number Generators" (2018). The state-advance and
//! output function match `https://prng.di.unimi.it/xoshiro128plusplus.c`
//! exactly.
//!
//! Departure from the reference: this v0 seeds from a single u32
//! (not the standard u64 splitmix64 ladder). The seed is expanded
//! through four rounds of splitmix32 to fill the 4×u32 state. The
//! resulting stream is still cryptographically diffuse, but a future
//! v0.2 will switch to the standard u64 ladder once the Quanta
//! WASM-route lowers i64 arithmetic.
//!
//! State is 4×u32 (16 bytes). Period of the underlying generator is
//! 2^128 − 1. Suitable for per-quark seeding via independent
//! `(seed, id) → u32 → state` hashing.

/// 4×u32 state of the xoshiro128++ generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State {
    pub s0: u32,
    pub s1: u32,
    pub s2: u32,
    pub s3: u32,
}

impl State {
    /// Initialise the state from a single u32 seed via four rounds
    /// of splitmix32 expansion. See module-level note on why u32
    /// (rather than the reference's u64) is the v0 seed type.
    #[inline]
    pub const fn from_seed_u32(seed: u32) -> Self {
        let s0 = splitmix32(seed);
        let s1 = splitmix32(s0);
        let s2 = splitmix32(s1);
        let s3 = splitmix32(s2);
        if s0 | s1 | s2 | s3 == 0 {
            Self {
                s0: 0x9E3779B9,
                s1: 0x243F6A88,
                s2: 0xB7E15162,
                s3: 0xCC9E2D51,
            }
        } else {
            Self { s0, s1, s2, s3 }
        }
    }
}

/// One advance step of the xoshiro128++ algorithm.
///
/// Returns the next u32 output and the post-step state. The output
/// is computed *before* the state advance, matching the reference C.
#[inline]
pub const fn next_u32(state: State) -> (u32, State) {
    // Output: rotl(s0 + s3, 7) + s0
    let result = rotl(state.s0.wrapping_add(state.s3), 7).wrapping_add(state.s0);

    // State advance.
    let t = state.s1 << 9;
    let s2 = state.s2 ^ state.s0;
    let s3 = state.s3 ^ state.s1;
    let s1 = state.s1 ^ s2;
    let s0 = state.s0 ^ s3;
    let s2 = s2 ^ t;
    let s3 = rotl(s3, 11);

    (result, State { s0, s1, s2, s3 })
}

/// Convert a u32 output to a uniform f32 in `[0, 1)`.
#[inline]
pub const fn u32_to_unit_f32(n: u32) -> f32 {
    let bits = n >> 8;
    (bits as f32) * (1.0 / 16777216.0)
}

/// Convert a u64 output to a uniform f64 in `[0, 1)`. Standard
/// technique: take the top 53 bits as the mantissa, scale by
/// `2^-53`. The result is one of `2^53` evenly spaced values in
/// `[0, 1)`.
#[inline]
pub fn u64_to_unit_f64(n: u64) -> f64 {
    // Top 53 bits → integer in `[0, 2^53)`.
    let bits = n >> 11;
    (bits as f64) * (1.0 / 9_007_199_254_740_992.0)
}

/// Jump-ahead constants for xoshiro128++ — published in the
/// upstream reference (`xoshiro128plusplus.c`). `JUMP` advances the
/// state by 2^64 steps; `LONG_JUMP` by 2^96 steps. Used to spawn
/// independent streams from one seed.
const JUMP: [u32; 4] = [0x8764000B, 0xF542D2D3, 0x6FA035C3, 0x77F2DB5B];
const LONG_JUMP: [u32; 4] = [0xB523952E, 0x0B6F099F, 0xCCF5A0EF, 0x1C580662];

/// Apply a polynomial-jump constant array to a State, returning the
/// new state. Used by both `jump` (2^64) and `long_jump` (2^96).
/// Algorithm from the upstream reference: walk each bit of each
/// constant word; for set bits, XOR the current state into an
/// accumulator; then advance the state by one xoshiro128++ step.
fn apply_jump(state: State, jump_const: &[u32; 4]) -> State {
    let mut s0: u32 = 0;
    let mut s1: u32 = 0;
    let mut s2: u32 = 0;
    let mut s3: u32 = 0;
    let mut st = state;
    for word in jump_const {
        for b in 0..32 {
            if (word >> b) & 1 == 1 {
                s0 ^= st.s0;
                s1 ^= st.s1;
                s2 ^= st.s2;
                s3 ^= st.s3;
            }
            let (_v, next) = next_u32(st);
            st = next;
        }
    }
    State { s0, s1, s2, s3 }
}

/// Advance the state by 2^64 steps. Equivalent to calling
/// `next_u32` 2^64 times, but constant-time. Used to spawn
/// non-overlapping streams from a common seed.
#[inline]
pub fn jump(state: State) -> State {
    apply_jump(state, &JUMP)
}

/// Advance the state by 2^96 steps. Use for outer-level parallelism
/// (e.g. one `long_jump` per worker thread, one `jump` per inner
/// stream).
#[inline]
pub fn long_jump(state: State) -> State {
    apply_jump(state, &LONG_JUMP)
}

/// 32-bit splitmix step. Murmur3 finaliser shape, 32-bit variant.
#[inline]
const fn splitmix32(mut x: u32) -> u32 {
    x = x.wrapping_add(0x9E3779B9);
    x = (x ^ (x >> 16)).wrapping_mul(0x85EBCA6B);
    x = (x ^ (x >> 13)).wrapping_mul(0xC2B2AE35);
    x ^ (x >> 16)
}

#[inline]
const fn rotl(x: u32, k: u32) -> u32 {
    // Manually unrolled to keep the function `const fn` and to mirror
    // the GPU-kernel-side rotate (which can't use `u32::rotate_left`
    // because the WASM-route doesn't yet lower `i32.rotl`).
    #[allow(clippy::manual_rotate)]
    {
        (x << k) | (x >> (32 - k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_f32_in_range() {
        let mut st = State::from_seed_u32(7);
        for _ in 0..1024 {
            let (v, next) = next_u32(st);
            let f = u32_to_unit_f32(v);
            assert!((0.0..1.0).contains(&f), "f32 out of [0, 1): {f}");
            st = next;
        }
    }

    #[test]
    fn from_seed_is_nonzero() {
        for seed in [0u32, 1, 42, u32::MAX, 0xDEAD_BEEF, 0x9E37_79B9] {
            let st = State::from_seed_u32(seed);
            assert_ne!(
                st.s0 | st.s1 | st.s2 | st.s3,
                0,
                "all-zero state for seed {seed}"
            );
        }
    }

    #[test]
    fn distinct_seeds_produce_distinct_first_outputs() {
        let mut count = 0;
        for s1 in 0u32..256 {
            let (a, _) = next_u32(State::from_seed_u32(s1));
            let (b, _) = next_u32(State::from_seed_u32(s1.wrapping_add(1)));
            if a == b {
                count += 1;
            }
        }
        assert!(
            count < 4,
            "{count} / 256 adjacent seeds produced identical first outputs"
        );
    }
}
