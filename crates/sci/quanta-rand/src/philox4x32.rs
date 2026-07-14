//! Philox4×32-10 counter-based RNG — pure Rust port.
//!
//! Reference: D. E. Shaw Research, "Parallel Random Numbers: As Easy
//! as 1, 2, 3" (SC11). The bit-exact algorithm matches
//! `random123/include/Random123/philox.h`. Output passes the full
//! TestU01 BigCrush battery and is the default counter-based
//! generator in cuRAND / rocRAND.
//!
//! ## Why Philox
//!
//! Counter-based RNGs are stateless functions `(counter, key) → output`.
//! Every quark can compute its own output from `(seed, quark_id)`
//! with zero coordination — perfect for GPU parallelism. Unlike
//! xoshiro128++, there's no state to carry between draws and no
//! `jump`/`long_jump` machinery: the counter *is* the position in
//! the stream.
//!
//! ## State and output shape
//!
//! - **counter**: 4×u32 (128-bit). Increment between draws to step
//!   through the stream.
//! - **key**: 2×u32 (64-bit). Acts as the "seed" — pick a key once,
//!   then iterate over counters.
//! - **output**: 4×u32 per `philox4x32_10` call. Each draw produces
//!   four independent uniform u32s.
//!
//! ## Bit-exact with the reference
//!
//! The `tests/philox_kat.rs` integration test runs the three
//! published known-answer vectors from
//! `random123/tests/kat_vectors` through this implementation and
//! asserts equality. If those tests pass, the algorithm matches the
//! Random123 reference C for these inputs.

/// One-counter-one-key Philox4×32 type. `counter` is a 4×u32 array;
/// `key` is a 2×u32 array. The output of `next` is a 4×u32 array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counter(pub [u32; 4]);

/// 64-bit Philox key, stored as 2×u32 to match the reference layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key(pub [u32; 2]);

/// Multiplier constants for the round function — irrational-number-
/// derived 32-bit values picked by the original Philox authors to
/// avoid weak diffusion. From `random123/include/Random123/philox.h`.
const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;

/// Key-bump constants — golden ratio (W0) and `sqrt(3) - 1` (W1)
/// scaled to 32-bit. Added to `key[0]` and `key[1]` between rounds.
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

/// Number of Philox rounds. 10 is the standard published variant
/// (`philox4x32_10`); produces BigCrush-clean output. Fewer rounds
/// trade diffusion for speed but only `_10` is sanctioned.
pub const ROUNDS: u32 = 10;

/// 32×32 → 64-bit multiply, returning `(hi, lo)` halves of the
/// 64-bit product. Compilers fold this to a single `mul` instruction
/// on every modern target (x86, ARM, RISC-V); the apparent u64
/// intermediate is free.
#[inline]
const fn mulhilo32(a: u32, b: u32) -> (u32, u32) {
    let product = (a as u64).wrapping_mul(b as u64);
    let hi = (product >> 32) as u32;
    let lo = product as u32;
    (hi, lo)
}

/// One Philox4×32 round: counter ← f(counter, key).
#[inline]
const fn round(ctr: Counter, key: Key) -> Counter {
    let (hi0, lo0) = mulhilo32(M0, ctr.0[0]);
    let (hi1, lo1) = mulhilo32(M1, ctr.0[2]);
    Counter([
        hi1 ^ ctr.0[1] ^ key.0[0],
        lo1,
        hi0 ^ ctr.0[3] ^ key.0[1],
        lo0,
    ])
}

/// Key-bump between rounds.
#[inline]
const fn bumpkey(key: Key) -> Key {
    Key([key.0[0].wrapping_add(W0), key.0[1].wrapping_add(W1)])
}

/// Philox4×32-R bijection: run `R` rounds starting from `(ctr, key)`.
/// The standard `_10` variant is exposed as `philox4x32_10` below;
/// this parameterised form is here for KAT-vector validation against
/// the reference's 7-round set.
#[inline]
pub const fn philox4x32_r(rounds: u32, mut ctr: Counter, mut key: Key) -> Counter {
    // Unrolled identical to Random123's `_philoxNxW_tpl` macro — one
    // round, then alternating bumpkey+round up to `R`.
    let mut i = 0;
    while i < rounds {
        if i > 0 {
            key = bumpkey(key);
        }
        ctr = round(ctr, key);
        i += 1;
    }
    ctr
}

/// Standard Philox4×32-10. Returns four u32s for one
/// `(counter, key)` pair.
///
/// To produce a stream, hold `key` fixed and increment `ctr` between
/// draws. To spawn independent streams from one seed, pick different
/// keys (or partition the counter space).
#[inline]
pub const fn philox4x32_10(ctr: Counter, key: Key) -> Counter {
    philox4x32_r(ROUNDS, ctr, key)
}

/// Kernel-callable scalar form of `philox4x32_10`. Same algorithm
/// and bit-exact output as `philox4x32_10`, expressed as a flat
/// `(u32, ..., u32) -> u32` so a `#[quanta::kernel]` body can call
/// it through the `#[quanta::device]` machinery (the WASM-lowering
/// pipeline doesn't lower structs or fixed-size arrays).
///
/// Returns only the first u32 of the 4-word output; that's the
/// "one random u32 per quark" case real kernels actually want. To
/// get more output words from the same counter draw, increment a
/// counter component (e.g. `c0 + 1`, `c0 + 2`, …) — counter-based
/// RNGs are designed for exactly that.
///
/// Host-side use is also supported (the attribute emits the fn
/// unchanged), so the same source serves CPU reference and GPU
/// kernel byte-for-byte.
///
/// The round multiplies compute the 64-bit product's high half via
/// the pure-32-bit 16-bit-split form (bit-identical to `mulhilo32`,
/// see `mulhilo32_split_matches_u64_form` in the tests) instead of
/// a u64 intermediate, so the spliced kernel lowers on devices
/// without `shaderInt64` (Metal, Broadcom V3D). The split is
/// inlined — not a helper call — because the splice is verbatim
/// and must stay self-contained, same reason the constants are
/// local.
#[cfg_attr(feature = "gpu", quanta_compute_dsl::device(crate = quanta_core))]
pub fn philox4x32_10_first_u32(c0: u32, c1: u32, c2: u32, c3: u32, k0: u32, k1: u32) -> u32 {
    // Constants must be local — `#[quanta::device]` splices the
    // function source verbatim into the wasm-shell crate where the
    // module-level `M0`/`M1`/`W0`/`W1` aren't in scope.
    const M0_K: u32 = 0xD251_1F53;
    const M1_K: u32 = 0xCD9E_8D57;
    const W0_K: u32 = 0x9E37_79B9;
    const W1_K: u32 = 0xBB67_AE85;

    let mut x0 = c0;
    let mut x1 = c1;
    let mut x2 = c2;
    let mut x3 = c3;
    let mut key0 = k0;
    let mut key1 = k1;

    let mut i: u32 = 0;
    while i < 10u32 {
        if i > 0 {
            key0 = key0.wrapping_add(W0_K);
            key1 = key1.wrapping_add(W1_K);
        }
        // mulhi(M0_K, x0) via 16-bit split — pure u32, bit-identical
        // to the 64-bit form (each partial fits; carry < 2^18).
        let m0_lo: u32 = M0_K & 0xFFFFu32;
        let m0_hi: u32 = M0_K >> 16u32;
        let x0_lo: u32 = x0 & 0xFFFFu32;
        let x0_hi: u32 = x0 >> 16u32;
        let p0_lolo: u32 = m0_lo.wrapping_mul(x0_lo);
        let p0_lohi: u32 = m0_lo.wrapping_mul(x0_hi);
        let p0_hilo: u32 = m0_hi.wrapping_mul(x0_lo);
        let p0_hihi: u32 = m0_hi.wrapping_mul(x0_hi);
        let p0_cross: u32 = (p0_lolo >> 16u32)
            .wrapping_add(p0_lohi & 0xFFFFu32)
            .wrapping_add(p0_hilo & 0xFFFFu32);
        let hi0: u32 = p0_hihi
            .wrapping_add(p0_lohi >> 16u32)
            .wrapping_add(p0_hilo >> 16u32)
            .wrapping_add(p0_cross >> 16u32);
        let lo0: u32 = M0_K.wrapping_mul(x0);
        // mulhi(M1_K, x2), same split.
        let m1_lo: u32 = M1_K & 0xFFFFu32;
        let m1_hi: u32 = M1_K >> 16u32;
        let x2_lo: u32 = x2 & 0xFFFFu32;
        let x2_hi: u32 = x2 >> 16u32;
        let p1_lolo: u32 = m1_lo.wrapping_mul(x2_lo);
        let p1_lohi: u32 = m1_lo.wrapping_mul(x2_hi);
        let p1_hilo: u32 = m1_hi.wrapping_mul(x2_lo);
        let p1_hihi: u32 = m1_hi.wrapping_mul(x2_hi);
        let p1_cross: u32 = (p1_lolo >> 16u32)
            .wrapping_add(p1_lohi & 0xFFFFu32)
            .wrapping_add(p1_hilo & 0xFFFFu32);
        let hi1: u32 = p1_hihi
            .wrapping_add(p1_lohi >> 16u32)
            .wrapping_add(p1_hilo >> 16u32)
            .wrapping_add(p1_cross >> 16u32);
        let lo1: u32 = M1_K.wrapping_mul(x2);
        let new_x0 = hi1 ^ x1 ^ key0;
        let new_x1 = lo1;
        let new_x2 = hi0 ^ x3 ^ key1;
        let new_x3 = lo0;
        x0 = new_x0;
        x1 = new_x1;
        x2 = new_x2;
        x3 = new_x3;
        i += 1;
    }
    x0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mulhilo32_basic() {
        // 2 * 3 = 6, hi = 0
        assert_eq!(mulhilo32(2, 3), (0, 6));
        // u32::MAX * u32::MAX = 0xFFFFFFFE_00000001
        assert_eq!(mulhilo32(u32::MAX, u32::MAX), (0xFFFF_FFFE, 0x0000_0001));
        // 1 << 16 squared = 1 << 32 = (hi=1, lo=0)
        assert_eq!(mulhilo32(1 << 16, 1 << 16), (1, 0));
    }

    #[test]
    fn round_does_not_panic() {
        let _ = round(Counter([0, 0, 0, 0]), Key([0, 0]));
        let _ = round(
            Counter([0xDEAD_BEEF, 0xCAFE_BABE, 0x1234_5678, 0x9ABC_DEF0]),
            Key([0xA5A5_A5A5, 0x5A5A_5A5A]),
        );
    }

    #[test]
    fn bumpkey_advances_both_words() {
        let k = Key([0, 0]);
        let k1 = bumpkey(k);
        assert_eq!(k1.0[0], W0);
        assert_eq!(k1.0[1], W1);
        let k2 = bumpkey(k1);
        assert_eq!(k2.0[0], W0.wrapping_mul(2));
        assert_eq!(k2.0[1], W1.wrapping_mul(2));
    }

    #[test]
    fn philox4x32_10_is_deterministic() {
        let ctr = Counter([0xDEAD_BEEF, 0xCAFE_BABE, 0x1234_5678, 0x9ABC_DEF0]);
        let key = Key([0xA5A5_A5A5, 0x5A5A_5A5A]);
        let a = philox4x32_10(ctr, key);
        let b = philox4x32_10(ctr, key);
        assert_eq!(a, b);
    }

    #[test]
    fn distinct_counters_produce_distinct_output() {
        let key = Key([0xC0FF_EE00, 0xBAD_F00D]);
        let a = philox4x32_10(Counter([0, 0, 0, 0]), key);
        let b = philox4x32_10(Counter([1, 0, 0, 0]), key);
        assert_ne!(a, b);
    }

    #[test]
    fn scalar_form_matches_struct_form_first_word() {
        // The kernel-callable `philox4x32_10_first_u32` must return
        // exactly the first u32 of the canonical `philox4x32_10`
        // output. Validated against the three published KAT inputs.
        let cases: &[([u32; 4], [u32; 2])] = &[
            ([0, 0, 0, 0], [0, 0]),
            (
                [u32::MAX, u32::MAX, u32::MAX, u32::MAX],
                [u32::MAX, u32::MAX],
            ),
            (
                [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
                [0xa4093822, 0x299f31d0],
            ),
        ];
        for &(c, k) in cases {
            let canonical = philox4x32_10(Counter(c), Key(k));
            let scalar = philox4x32_10_first_u32(c[0], c[1], c[2], c[3], k[0], k[1]);
            assert_eq!(
                scalar, canonical.0[0],
                "scalar form diverges from canonical first word for {c:?} / {k:?}"
            );
        }
    }

    /// The 16-bit-split mulhi used inside `philox4x32_10_first_u32`
    /// (and the in-kernel twin in `gpu_kernel.rs`) must be
    /// bit-identical to the u64-based `mulhilo32` for every input.
    /// Sweeps the corner lattice (all pairs of boundary-shaped
    /// words) plus a dense splitmix-driven random sweep.
    #[test]
    fn mulhilo32_split_matches_u64_form() {
        fn mulhi_split(a: u32, b: u32) -> u32 {
            let a_lo = a & 0xFFFF;
            let a_hi = a >> 16;
            let b_lo = b & 0xFFFF;
            let b_hi = b >> 16;
            let lolo = a_lo.wrapping_mul(b_lo);
            let lohi = a_lo.wrapping_mul(b_hi);
            let hilo = a_hi.wrapping_mul(b_lo);
            let hihi = a_hi.wrapping_mul(b_hi);
            let cross = (lolo >> 16)
                .wrapping_add(lohi & 0xFFFF)
                .wrapping_add(hilo & 0xFFFF);
            hihi.wrapping_add(lohi >> 16)
                .wrapping_add(hilo >> 16)
                .wrapping_add(cross >> 16)
        }

        // Corner lattice: values that stress the 16-bit halves and
        // the carry chain.
        let corners: &[u32] = &[
            0,
            1,
            2,
            0x7FFF,
            0x8000,
            0xFFFF,
            0x0001_0000,
            0x0001_0001,
            0x7FFF_FFFF,
            0x8000_0000,
            0xFFFF_0000,
            0xFFFF_FFFE,
            u32::MAX,
            M0,
            M1,
            W0,
            W1,
        ];
        for &a in corners {
            for &b in corners {
                assert_eq!(
                    mulhi_split(a, b),
                    mulhilo32(a, b).0,
                    "split mulhi diverges at a={a:#010x}, b={b:#010x}"
                );
            }
        }

        // Dense sweep: 1M pseudorandom pairs via splitmix32.
        let mut state = 0x9E37_79B9u32;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9);
            let mut z = state;
            z = (z ^ (z >> 16)).wrapping_mul(0x85EB_CA6B);
            z = (z ^ (z >> 13)).wrapping_mul(0xC2B2_AE35);
            z ^ (z >> 16)
        };
        for _ in 0..1_000_000 {
            let a = next();
            let b = next();
            assert_eq!(
                mulhi_split(a, b),
                mulhilo32(a, b).0,
                "split mulhi diverges at a={a:#010x}, b={b:#010x}"
            );
        }
    }

    /// Dense cross-check of the full scalar Philox (with its
    /// inlined split mulhi) against the canonical u64-based
    /// `philox4x32_10` — 10k pseudorandom counter/key draws.
    #[test]
    fn scalar_form_matches_struct_form_dense_sweep() {
        let mut state = 0xBB67_AE85u32;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9);
            let mut z = state;
            z = (z ^ (z >> 16)).wrapping_mul(0x85EB_CA6B);
            z = (z ^ (z >> 13)).wrapping_mul(0xC2B2_AE35);
            z ^ (z >> 16)
        };
        for _ in 0..10_000 {
            let c = [next(), next(), next(), next()];
            let k = [next(), next()];
            let canonical = philox4x32_10(Counter(c), Key(k));
            let scalar = philox4x32_10_first_u32(c[0], c[1], c[2], c[3], k[0], k[1]);
            assert_eq!(
                scalar, canonical.0[0],
                "scalar form diverges from canonical first word for {c:?} / {k:?}"
            );
        }
    }
}
