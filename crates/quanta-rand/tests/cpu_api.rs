//! CPU-only surface tests for the extended `Rng` API:
//! `next_u64`, `next_f64`, `jump`, `long_jump`, and the
//! `quark_next_u64` host reference.
//!
//! Pure-CPU — runs regardless of the `gpu` feature.

use quanta_rand::{Rng, quark_next_u32, quark_next_u64};

#[test]
fn next_u64_packs_two_u32_outputs() {
    // `next_u64` is defined as `(hi << 32) | lo` where `hi` is the
    // first u32 and `lo` is the second. Drive two parallel streams
    // and check they agree.
    let mut wide = Rng::from_seed(0xC0FFEE);
    let mut narrow = Rng::from_seed(0xC0FFEE);
    for _ in 0..64 {
        let packed = wide.next_u64();
        let hi = narrow.next_u32();
        let lo = narrow.next_u32();
        let expected = ((hi as u64) << 32) | (lo as u64);
        assert_eq!(packed, expected);
    }
}

#[test]
fn next_f64_in_unit_range() {
    let mut rng = Rng::from_seed(7);
    let mut saw_below_half = false;
    let mut saw_above_half = false;
    for _ in 0..4096 {
        let d = rng.next_f64();
        assert!((0.0..1.0).contains(&d), "f64 out of [0, 1): {d}");
        if d < 0.5 {
            saw_below_half = true;
        }
        if d >= 0.5 {
            saw_above_half = true;
        }
    }
    assert!(
        saw_below_half && saw_above_half,
        "uniform-ish sample didn't span both halves of [0, 1)",
    );
}

#[test]
fn jump_advances_state() {
    // `jump` advances by 2^64 steps. Two clones should diverge
    // immediately after a jump on one of them.
    let mut a = Rng::from_seed(42);
    let mut b = a.clone();
    b.jump();

    let mut diffs = 0;
    for _ in 0..16 {
        if a.next_u32() != b.next_u32() {
            diffs += 1;
        }
    }
    assert!(
        diffs >= 14,
        "jump should produce a divergent stream — saw {diffs} / 16 differences",
    );
}

#[test]
fn long_jump_advances_state() {
    let mut a = Rng::from_seed(42);
    let mut b = a.clone();
    b.long_jump();

    let mut diffs = 0;
    for _ in 0..16 {
        if a.next_u32() != b.next_u32() {
            diffs += 1;
        }
    }
    assert!(
        diffs >= 14,
        "long_jump should produce a divergent stream — saw {diffs} / 16 differences",
    );
}

#[test]
fn jump_is_deterministic() {
    // Same seed + same number of jumps must produce the same state.
    let mut a = Rng::from_seed(1234);
    let mut b = Rng::from_seed(1234);
    a.jump();
    b.jump();
    for _ in 0..8 {
        assert_eq!(a.next_u32(), b.next_u32());
    }
}

#[test]
fn quark_next_u64_matches_two_u32_draws() {
    // `quark_next_u64` is defined as
    //   (quark_next_u32(s, id) << 32) | quark_next_u32(s, id ^ 0x8000_0000)
    let seed_lo = 0xDEAD_BEEF;
    let seed_hi = 0xCAFE_BABE;
    for id in [0u32, 1, 7, 42, 1024, u32::MAX] {
        let packed = quark_next_u64(seed_lo, seed_hi, id);
        let hi = quark_next_u32(seed_lo, seed_hi, id);
        let lo = quark_next_u32(seed_lo, seed_hi, id ^ 0x8000_0000);
        let expected = ((hi as u64) << 32) | (lo as u64);
        assert_eq!(packed, expected, "quark id = {id}");
    }
}

#[test]
fn quark_next_u64_distinguishes_ids() {
    let seed_lo = 0xC0FFEE;
    let seed_hi = 0xBAD_F00D;
    let mut count = 0;
    for id in 0u32..256 {
        let a = quark_next_u64(seed_lo, seed_hi, id);
        let b = quark_next_u64(seed_lo, seed_hi, id.wrapping_add(1));
        if a == b {
            count += 1;
        }
    }
    assert!(
        count < 2,
        "{count} / 256 adjacent quark ids produced identical u64 outputs",
    );
}
