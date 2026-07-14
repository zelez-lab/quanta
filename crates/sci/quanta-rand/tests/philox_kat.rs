//! Philox4×32 known-answer test vectors from Random123.
//!
//! Validates the pure-Rust port against the three published test
//! vectors in `random123/tests/kat_vectors`. If these match, the
//! algorithm is bit-exact with the reference C — which has been
//! validated against TestU01 BigCrush.
//!
//! Source: `D. E. Shaw Research/random123` repository, `kat_vectors`
//! file. Each line is:
//!
//!   philox4x32 <rounds> <ctr0..3> <key0..1>   <out0..3>

use quanta_rand::philox4x32::{Counter, Key, philox4x32_r};

#[test]
fn kat_philox4x32_7_zero_inputs() {
    // philox4x32 7 00000000 00000000 00000000 00000000 00000000 00000000
    //               5f6fb709 0d893f64 4f121f81 4f730a48
    let ctr = Counter([0, 0, 0, 0]);
    let key = Key([0, 0]);
    let out = philox4x32_r(7, ctr, key);
    assert_eq!(out.0, [0x5f6fb709, 0x0d893f64, 0x4f121f81, 0x4f730a48]);
}

#[test]
fn kat_philox4x32_7_all_ones() {
    // philox4x32 7 ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff
    //               5207ddc2 45165e59 4d8ee751 8c52f662
    let ctr = Counter([u32::MAX, u32::MAX, u32::MAX, u32::MAX]);
    let key = Key([u32::MAX, u32::MAX]);
    let out = philox4x32_r(7, ctr, key);
    assert_eq!(out.0, [0x5207ddc2, 0x45165e59, 0x4d8ee751, 0x8c52f662]);
}

#[test]
fn kat_philox4x32_7_pi_inputs() {
    // philox4x32 7 243f6a88 85a308d3 13198a2e 03707344 a4093822 299f31d0
    //               4dfccaba 190a87f0 c47362ba b6b5242a
    let ctr = Counter([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344]);
    let key = Key([0xa4093822, 0x299f31d0]);
    let out = philox4x32_r(7, ctr, key);
    assert_eq!(out.0, [0x4dfccaba, 0x190a87f0, 0xc47362ba, 0xb6b5242a]);
}

#[test]
fn kat_philox4x32_10_zero_inputs() {
    // philox4x32 10 00000000 00000000 00000000 00000000 00000000 00000000
    //                6627e8d5 e169c58d bc57ac4c 9b00dbd8
    let ctr = Counter([0, 0, 0, 0]);
    let key = Key([0, 0]);
    let out = philox4x32_r(10, ctr, key);
    assert_eq!(out.0, [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]);
}

#[test]
fn kat_philox4x32_10_all_ones() {
    // philox4x32 10 ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff
    //                408f276d 41c83b0e a20bc7c6 6d5451fd
    let ctr = Counter([u32::MAX, u32::MAX, u32::MAX, u32::MAX]);
    let key = Key([u32::MAX, u32::MAX]);
    let out = philox4x32_r(10, ctr, key);
    assert_eq!(out.0, [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]);
}

#[test]
fn kat_philox4x32_10_pi_inputs() {
    // philox4x32 10 243f6a88 85a308d3 13198a2e 03707344 a4093822 299f31d0
    //                d16cfe09 94fdcceb 5001e420 24126ea1
    let ctr = Counter([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344]);
    let key = Key([0xa4093822, 0x299f31d0]);
    let out = philox4x32_r(10, ctr, key);
    assert_eq!(out.0, [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]);
}

#[test]
fn philox4x32_10_alias_matches_r10() {
    // The convenience `philox4x32_10(ctr, key)` must equal
    // `philox4x32_r(10, ctr, key)` byte-for-byte.
    let ctr = Counter([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344]);
    let key = Key([0xa4093822, 0x299f31d0]);
    let a = quanta_rand::philox4x32::philox4x32_10(ctr, key);
    let b = philox4x32_r(10, ctr, key);
    assert_eq!(a.0, b.0);
}
