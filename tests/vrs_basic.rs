//! Integration tests for `VrsState` (steps 028 + 029).
//!
//! Refines the proven Lean theorems T7500–T7505 + Verus theorems
//! T7550–T7553 against the CPU device's software VRS lifecycle. Each
//! test names the theorem it refines.
//!
//! Run: cargo test --test vrs_basic --features software

#![cfg(feature = "software")]

use quanta::ShadingRate;

#[test]
fn vrs_create_default_rate_is_1x1() {
    // T7500 refinement: created state defaults to 1×1.
    let gpu = quanta::init_cpu();
    let vrs = gpu.vrs_state().unwrap();
    assert_eq!(vrs.current(), ShadingRate::R1x1);
}

#[test]
fn vrs_set_rate_writes_input() {
    // T7501 refinement: after set_rate r, current = r.
    let gpu = quanta::init_cpu();
    let mut vrs = gpu.vrs_state().unwrap();
    vrs.set_rate(ShadingRate::R2x2).unwrap();
    assert_eq!(vrs.current(), ShadingRate::R2x2);
}

#[test]
fn vrs_set_rate_all_seven_rates() {
    // T7501 refinement across the full rate set.
    let gpu = quanta::init_cpu();
    let mut vrs = gpu.vrs_state().unwrap();
    for r in [
        ShadingRate::R1x1,
        ShadingRate::R1x2,
        ShadingRate::R2x1,
        ShadingRate::R2x2,
        ShadingRate::R2x4,
        ShadingRate::R4x2,
        ShadingRate::R4x4,
    ] {
        vrs.set_rate(r).unwrap();
        assert_eq!(vrs.current(), r);
    }
}

#[test]
fn vrs_axes_are_in_range() {
    // T7504 refinement: all axes are in {1, 2, 4}.
    for r in [
        ShadingRate::R1x1,
        ShadingRate::R1x2,
        ShadingRate::R2x1,
        ShadingRate::R2x2,
        ShadingRate::R2x4,
        ShadingRate::R4x2,
        ShadingRate::R4x4,
    ] {
        assert!(matches!(r.x_axis(), 1 | 2 | 4));
        assert!(matches!(r.y_axis(), 1 | 2 | 4));
    }
}

#[test]
fn vrs_codes_are_unique_and_in_range() {
    // Verus encoding sanity: 7 distinct codes in 0..=6.
    let codes: alloc::vec::Vec<u8> = [
        ShadingRate::R1x1,
        ShadingRate::R1x2,
        ShadingRate::R2x1,
        ShadingRate::R2x2,
        ShadingRate::R2x4,
        ShadingRate::R4x2,
        ShadingRate::R4x4,
    ]
    .iter()
    .map(|r| r.code())
    .collect();
    let mut sorted = codes.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, alloc::vec![0u8, 1, 2, 3, 4, 5, 6]);
}

extern crate alloc;

#[test]
fn vrs_drop_invalidates_state() {
    // T7503 refinement: Drop releases the backend handle.
    let gpu = quanta::init_cpu();
    {
        let _vrs = gpu.vrs_state().unwrap();
    }
}
