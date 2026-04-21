//! Tier 2 -- Multi-queue operations.
//!
//! Verifies queue_families, create_queue, and queue dispatch.
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
    let result = gpu.create_queue(QueueType::Graphics);
    match result {
        Ok(handle) => {
            assert!(handle != 0, "queue handle should be nonzero");
        }
        Err(e) => {
            eprintln!("create_queue(Graphics) not supported: {}", e);
        }
    }
}

#[test]
fn create_compute_queue_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let result = gpu.create_queue(QueueType::Compute);
    match result {
        Ok(handle) => {
            assert!(handle != 0);
        }
        Err(e) => {
            eprintln!("create_queue(Compute) not supported: {}", e);
        }
    }
}

#[test]
fn create_transfer_queue_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let result = gpu.create_queue(QueueType::Transfer);
    match result {
        Ok(handle) => {
            assert!(handle != 0);
        }
        Err(e) => {
            eprintln!("create_queue(Transfer) not supported: {}", e);
        }
    }
}

#[test]
fn queue_signal_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // queue_signal/queue_wait with dummy handles should return an error,
    // not panic.
    let result = gpu.queue_signal(0, 0);
    match result {
        Ok(()) => {}
        Err(_) => {} // expected "not supported"
    }
}

#[test]
fn queue_wait_returns_result() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let result = gpu.queue_wait(0, 0);
    match result {
        Ok(()) => {}
        Err(_) => {} // expected "not supported"
    }
}
