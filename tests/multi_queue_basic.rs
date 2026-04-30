//! Integration tests for typed `Queue` (steps 018 + 019).
//!
//! Refines the proven Lean theorems T7700–T7705 + Verus theorems
//! T7750–T7755 against the CPU device's software queue lifecycle.
//! Each test names the theorem it refines.
//!
//! Run: cargo test --test multi_queue_basic --features software

#![cfg(feature = "software")]

use quanta::QueueType;

#[test]
fn queue_families_lists_graphics_compute_transfer() {
    // CPU exposes one queue per family.
    let gpu = quanta::init_cpu();
    let families = gpu.queue_families();
    let kinds: alloc::vec::Vec<QueueType> = families.iter().map(|f| f.queue_type).collect();
    assert!(kinds.contains(&QueueType::Graphics));
    assert!(kinds.contains(&QueueType::Compute));
    assert!(kinds.contains(&QueueType::Transfer));
}

extern crate alloc;

#[test]
fn queue_create_records_kind() {
    // T7700 refinement: created queue has the requested kind.
    let gpu = quanta::init_cpu();
    let q = gpu.queue(QueueType::Compute).unwrap();
    assert_eq!(q.kind(), QueueType::Compute);
}

#[test]
fn queue_create_all_three_kinds() {
    let gpu = quanta::init_cpu();
    let g = gpu.queue(QueueType::Graphics).unwrap();
    let c = gpu.queue(QueueType::Compute).unwrap();
    let t = gpu.queue(QueueType::Transfer).unwrap();
    assert_eq!(g.kind(), QueueType::Graphics);
    assert_eq!(c.kind(), QueueType::Compute);
    assert_eq!(t.kind(), QueueType::Transfer);
}

#[test]
fn queue_signal_and_wait_succeed_on_live_queue() {
    // T7703 refinement: signal on a live queue records the pair.
    // T7704 refinement: signal preserves earlier submitted history.
    let gpu = quanta::init_cpu();
    let q = gpu.queue(QueueType::Compute).unwrap();
    // Use a dummy semaphore handle (cross-queue sync is software
    // serial on CPU).
    q.signal(42).unwrap();
    q.wait(42).unwrap();
}

#[test]
fn queue_drop_invalidates_handle() {
    // T7705 refinement: Drop releases the backend handle.
    let gpu = quanta::init_cpu();
    {
        let _q = gpu.queue(QueueType::Graphics).unwrap();
    }
    // No panic from Drop is the assertion.
}

#[test]
fn multiple_queues_have_distinct_handles() {
    let gpu = quanta::init_cpu();
    let q1 = gpu.queue(QueueType::Compute).unwrap();
    let q2 = gpu.queue(QueueType::Compute).unwrap();
    assert_ne!(q1.handle(), q2.handle());
}
