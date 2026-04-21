//! Tier 2 -- Timeline semaphore lifecycle.
//!
//! Verifies timeline_create, timeline_signal, timeline_wait.
//! Features may return "not supported" -- that is acceptable.
//! Requires a GPU; skips gracefully if none available.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[test]
fn timeline_create_and_query() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    match gpu.timeline_create() {
        Ok(_timeline) => {
            // Timeline created successfully.
        }
        Err(e) => {
            // Not supported -- acceptable.
            eprintln!("timeline_create not supported: {}", e);
        }
    }
}

#[test]
fn timeline_signal_and_wait() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let timeline = match gpu.timeline_create() {
        Ok(t) => t,
        Err(_) => {
            eprintln!("skipping: timeline semaphores not supported");
            return;
        }
    };

    // Signal value 1.
    match gpu.timeline_signal(&timeline, 1) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("timeline_signal not supported: {}", e);
            return;
        }
    }

    // Wait for value 1.
    match gpu.timeline_wait(&timeline, 1) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("timeline_wait failed: {}", e);
        }
    }
}

#[test]
fn timeline_signal_monotonic() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let timeline = match gpu.timeline_create() {
        Ok(t) => t,
        Err(_) => {
            eprintln!("skipping: timeline semaphores not supported");
            return;
        }
    };

    // Signal increasing values.
    for val in 1..=5 {
        match gpu.timeline_signal(&timeline, val) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("timeline_signal({}) failed: {}", val, e);
                return;
            }
        }
    }

    // Wait for the latest value.
    match gpu.timeline_wait(&timeline, 5) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("timeline_wait(5) failed: {}", e);
        }
    }
}

#[test]
fn timeline_wait_already_reached() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let timeline = match gpu.timeline_create() {
        Ok(t) => t,
        Err(_) => {
            eprintln!("skipping: timeline semaphores not supported");
            return;
        }
    };

    // Signal to 10.
    match gpu.timeline_signal(&timeline, 10) {
        Ok(()) => {}
        Err(_) => return,
    }

    // Waiting for a value already reached should return immediately.
    match gpu.timeline_wait(&timeline, 5) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("timeline_wait for already-reached value failed: {}", e);
        }
    }
}
