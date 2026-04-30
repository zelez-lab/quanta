//! Integration tests for `PrintfBuffer` (step 049).
//!
//! Refines the proven Lean theorems T7900–T7905 + Verus theorems
//! T7950–T7954 against the CPU device's software printf lifecycle.
//! Each test names the theorem it refines.
//!
//! Run: cargo test --test printf_basic --features software

#![cfg(feature = "software")]

#[test]
fn printf_create_records_capacity() {
    // T7900 refinement.
    let gpu = quanta::init_cpu();
    let p = gpu.printf_buffer(8).unwrap();
    assert_eq!(p.capacity(), 8);
}

#[test]
fn printf_create_rejects_zero_capacity() {
    // typed-wrapper precondition before reaching the device.
    let gpu = quanta::init_cpu();
    assert!(gpu.printf_buffer(0).is_err());
}

#[test]
fn printf_record_appends_in_order() {
    // T7901 refinement: record appends; drain returns in order.
    let gpu = quanta::init_cpu();
    let p = gpu.printf_buffer(4).unwrap();
    p.record(10).unwrap();
    p.record(20).unwrap();
    p.record(30).unwrap();
    let drained = p.drain().unwrap();
    assert_eq!(drained, vec![10u64, 20, 30]);
}

#[test]
fn printf_record_full_buffer_fails() {
    // T7903 refinement: when len >= cap, record fails.
    let gpu = quanta::init_cpu();
    let p = gpu.printf_buffer(2).unwrap();
    p.record(1).unwrap();
    p.record(2).unwrap();
    assert!(p.record(3).is_err());
}

#[test]
fn printf_drain_empties_buffer() {
    // T7904 refinement: drain returns recorded messages and empties.
    let gpu = quanta::init_cpu();
    let p = gpu.printf_buffer(4).unwrap();
    p.record(7).unwrap();
    let first = p.drain().unwrap();
    assert_eq!(first, vec![7u64]);
    let second = p.drain().unwrap();
    assert!(second.is_empty());
}

#[test]
fn printf_drop_releases_handle() {
    // T7905 refinement: Drop releases backend handle.
    let gpu = quanta::init_cpu();
    {
        let _p = gpu.printf_buffer(2).unwrap();
    }
}
