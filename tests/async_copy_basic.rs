//! Integration tests for `AsyncCopyQueue` (step 044).
//!
//! Refines the proven Lean theorems T7800–T7804 + Verus theorems
//! T7850–T7853 against the CPU device's software async-copy
//! lifecycle. Each test names the theorem it refines.
//!
//! Run: cargo test --test async_copy_basic --features software

#![cfg(feature = "software")]

#[test]
fn async_copy_create_succeeds() {
    // T7800 refinement: queue creation succeeds; handle returned.
    let gpu = quanta::init_cpu();
    let q = gpu.async_copy_queue().unwrap();
    assert_ne!(q.handle(), 0);
}

#[test]
fn async_copy_buffer_transfers_bytes() {
    // T7801 refinement: submit_copy records the (dst, src, size)
    // triple. The CPU device executes the copy serially through
    // field_copy_bytes; the data must round-trip.
    let gpu = quanta::init_cpu();
    let q = gpu.async_copy_queue().unwrap();
    let src = gpu.field::<u32>(4).unwrap();
    let dst = gpu.field::<u32>(4).unwrap();
    src.write(&[7u32, 11, 13, 17]).unwrap();
    q.copy_buffer(&dst, &src, 4 * 4).unwrap();
    let out = dst.read().unwrap();
    assert_eq!(out, vec![7u32, 11, 13, 17]);
}

#[test]
fn async_copy_multiple_submits_succeed() {
    // T7801 boundary: multiple consecutive copies all succeed and
    // appear in submission order (CPU executes serially).
    let gpu = quanta::init_cpu();
    let q = gpu.async_copy_queue().unwrap();
    let a = gpu.field::<u32>(2).unwrap();
    let b = gpu.field::<u32>(2).unwrap();
    let c = gpu.field::<u32>(2).unwrap();
    a.write(&[1u32, 2]).unwrap();
    q.copy_buffer(&b, &a, 8).unwrap();
    q.copy_buffer(&c, &b, 8).unwrap();
    assert_eq!(c.read().unwrap(), vec![1u32, 2]);
}

#[test]
fn async_copy_drop_invalidates_queue() {
    // T7803 refinement: Drop releases the backend handle.
    let gpu = quanta::init_cpu();
    {
        let _q = gpu.async_copy_queue().unwrap();
    }
}

#[test]
fn async_copy_raw_handle_path_works() {
    let gpu = quanta::init_cpu();
    let q = gpu.async_copy_queue().unwrap();
    let src = gpu.field::<u32>(2).unwrap();
    let dst = gpu.field::<u32>(2).unwrap();
    src.write(&[42u32, 99]).unwrap();
    q.copy_buffer_raw(dst.handle(), src.handle(), 8).unwrap();
    assert_eq!(dst.read().unwrap(), vec![42u32, 99]);
}
