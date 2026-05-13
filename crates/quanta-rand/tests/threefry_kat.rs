//! Threefry4×32 known-answer test vectors from Random123.
//!
//! Validates the pure-Rust port against the published test vectors
//! in `random123/tests/kat_vectors` at 13, 20 (standard), and 72
//! rounds.
//!
//! Source: `D. E. Shaw Research/random123` repository, `kat_vectors`
//! file. Each line is:
//!
//!   threefry4x32 <rounds> <ctr0..3> <key0..3>   <out0..3>

use quanta_rand::threefry4x32::{Counter, Key, threefry4x32_r};

#[test]
fn kat_threefry4x32_13_zero_inputs() {
    let ctr = Counter([0, 0, 0, 0]);
    let key = Key([0, 0, 0, 0]);
    let out = threefry4x32_r(13, ctr, key);
    assert_eq!(out.0, [0x531c7e4f, 0x39491ee5, 0x2c855a92, 0x3d6abf9a]);
}

#[test]
fn kat_threefry4x32_13_all_ones() {
    let ctr = Counter([u32::MAX, u32::MAX, u32::MAX, u32::MAX]);
    let key = Key([u32::MAX, u32::MAX, u32::MAX, u32::MAX]);
    let out = threefry4x32_r(13, ctr, key);
    assert_eq!(out.0, [0xc4189358, 0x1c9cc83a, 0xd5881c67, 0x6a0a89e0]);
}

#[test]
fn kat_threefry4x32_13_pi_inputs() {
    let ctr = Counter([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344]);
    let key = Key([0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89]);
    let out = threefry4x32_r(13, ctr, key);
    assert_eq!(out.0, [0x4aa71d8f, 0x734738c2, 0x431fc6a8, 0xae6debf1]);
}

#[test]
fn kat_threefry4x32_20_zero_inputs() {
    let ctr = Counter([0, 0, 0, 0]);
    let key = Key([0, 0, 0, 0]);
    let out = threefry4x32_r(20, ctr, key);
    assert_eq!(out.0, [0x9c6ca96a, 0xe17eae66, 0xfc10ecd4, 0x5256a7d8]);
}

#[test]
fn kat_threefry4x32_20_all_ones() {
    let ctr = Counter([u32::MAX, u32::MAX, u32::MAX, u32::MAX]);
    let key = Key([u32::MAX, u32::MAX, u32::MAX, u32::MAX]);
    let out = threefry4x32_r(20, ctr, key);
    assert_eq!(out.0, [0x2a881696, 0x57012287, 0xf6c7446e, 0xa16a6732]);
}

#[test]
fn kat_threefry4x32_20_pi_inputs() {
    let ctr = Counter([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344]);
    let key = Key([0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89]);
    let out = threefry4x32_r(20, ctr, key);
    assert_eq!(out.0, [0x59cd1dbb, 0xb8879579, 0x86b5d00c, 0xac8b6d84]);
}

#[test]
fn kat_threefry4x32_72_zero_inputs() {
    let ctr = Counter([0, 0, 0, 0]);
    let key = Key([0, 0, 0, 0]);
    let out = threefry4x32_r(72, ctr, key);
    assert_eq!(out.0, [0x93171da6, 0x9220326d, 0xb392b7b1, 0xff58a002]);
}

#[test]
fn kat_threefry4x32_72_all_ones() {
    let ctr = Counter([u32::MAX, u32::MAX, u32::MAX, u32::MAX]);
    let key = Key([u32::MAX, u32::MAX, u32::MAX, u32::MAX]);
    let out = threefry4x32_r(72, ctr, key);
    assert_eq!(out.0, [0x60743f3d, 0x9961e684, 0xaab21c34, 0x8c65fb7d]);
}

#[test]
fn kat_threefry4x32_72_pi_inputs() {
    let ctr = Counter([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344]);
    let key = Key([0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89]);
    let out = threefry4x32_r(72, ctr, key);
    assert_eq!(out.0, [0x09930adf, 0x7f27bd55, 0x9ed68ce1, 0x97f803f6]);
}

#[test]
fn threefry4x32_20_alias_matches_r20() {
    let ctr = Counter([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344]);
    let key = Key([0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89]);
    let a = quanta_rand::threefry4x32::threefry4x32_20(ctr, key);
    let b = threefry4x32_r(20, ctr, key);
    assert_eq!(a.0, b.0);
}
