//! Tier 2 -- Multi-queue operations.
//!
//! Verifies queue_families and typed Queue creation.
//! Features may return "not supported" -- that is acceptable.
//! Requires a GPU; skips gracefully if none available.

use quanta::QueueType;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[test]
fn queue_families_returns_at_least_one() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let families = gpu.queue_families();
    assert!(!families.is_empty(), "must have at least one queue family");

    // At least one family should be graphics.
    let has_graphics = families.iter().any(|f| f.queue_type == QueueType::Graphics);
    assert!(has_graphics, "must have a graphics queue family");
}

#[test]
fn queue_families_have_nonzero_count() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let families = gpu.queue_families();
    for (i, f) in families.iter().enumerate() {
        assert!(f.count > 0, "queue family {} has zero queues", i);
    }
}

#[test]
fn create_graphics_queue_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // May return not supported -- that is OK.
    let result = gpu.queue(QueueType::Graphics);
    match result {
        Ok(queue) => {
            assert!(queue.handle() != 0, "queue handle should be nonzero");
            assert_eq!(queue.kind(), QueueType::Graphics);
        }
        Err(e) => {
            eprintln!("queue(Graphics) not supported: {}", e);
        }
    }
}

#[test]
fn create_compute_queue_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let result = gpu.queue(QueueType::Compute);
    match result {
        Ok(queue) => {
            assert!(queue.handle() != 0);
        }
        Err(e) => {
            eprintln!("queue(Compute) not supported: {}", e);
        }
    }
}

#[test]
fn create_transfer_queue_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let result = gpu.queue(QueueType::Transfer);
    match result {
        Ok(queue) => {
            assert!(queue.handle() != 0);
        }
        Err(e) => {
            eprintln!("queue(Transfer) not supported: {}", e);
        }
    }
}

#[test]
fn queue_signal_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // signal/wait with dummy semaphore handles should return an error,
    // not panic. Ok or Err ("not supported") are both acceptable.
    if let Ok(queue) = gpu.queue(QueueType::Graphics) {
        let _ = queue.signal(0);
    }
}

#[test]
fn queue_wait_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Ok or Err ("not supported") are both acceptable; must not panic.
    if let Ok(queue) = gpu.queue(QueueType::Graphics) {
        let _ = queue.wait(0);
    }
}
