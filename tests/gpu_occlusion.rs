//! Tier 2 -- Occlusion query lifecycle.
//!
//! Verifies occlusion_query_create, begin/end in render pass, and read.
//! Features may return "not supported" -- that is acceptable.
//! Requires a GPU; skips gracefully if none available.

use quanta::Format;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[test]
fn occlusion_query_create() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    match gpu.occlusion_query_create(4) {
        Ok(query) => {
            assert_eq!(query.count(), 4, "query should have 4 slots");
            assert!(query.handle() != 0, "query handle should be nonzero");
        }
        Err(e) => {
            eprintln!("occlusion queries not supported: {}", e);
        }
    }
}

#[test]
fn occlusion_query_read_initial() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let query = match gpu.occlusion_query_create(2) {
        Ok(q) => q,
        Err(_) => {
            eprintln!("skipping: occlusion queries not supported");
            return;
        }
    };

    // Read before any draw -- should return zeros or an error.
    match gpu.occlusion_query_read(&query) {
        Ok(results) => {
            assert_eq!(results.len(), 2);
        }
        Err(e) => {
            eprintln!("occlusion_query_read failed: {}", e);
        }
    }
}

#[test]
fn occlusion_query_in_render_pass() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let query = match gpu.occlusion_query_create(1) {
        Ok(q) => q,
        Err(_) => {
            eprintln!("skipping: occlusion queries not supported");
            return;
        }
    };

    let target = gpu.render_target(16, 16, Format::RGBA8).unwrap();

    // Begin and end an occlusion query (no draw between -- result should be 0).
    let mut pulse = gpu
        .render(&target)
        .unwrap()
        .begin_occlusion_query(&query, 0)
        .end_occlusion_query(&query, 0)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();

    // Read the result -- should be 0 (no fragments drawn).
    match gpu.occlusion_query_read(&query) {
        Ok(results) => {
            assert_eq!(results.len(), 1);
            assert_eq!(results[0], 0, "no fragments drawn, should be 0");
        }
        Err(e) => {
            eprintln!("occlusion_query_read failed: {}", e);
        }
    }
}

#[test]
fn occlusion_query_multiple_slots() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let slot_count = 4;
    let query = match gpu.occlusion_query_create(slot_count) {
        Ok(q) => q,
        Err(_) => {
            eprintln!("skipping: occlusion queries not supported");
            return;
        }
    };

    let target = gpu.render_target(16, 16, Format::RGBA8).unwrap();

    // Use each slot.
    let mut builder = gpu.render(&target).unwrap();
    for i in 0..slot_count {
        builder = builder
            .begin_occlusion_query(&query, i)
            .end_occlusion_query(&query, i);
    }

    let mut pulse = builder.pulse().unwrap();
    pulse.wait().unwrap();

    match gpu.occlusion_query_read(&query) {
        Ok(results) => {
            assert_eq!(results.len(), slot_count as usize);
        }
        Err(e) => {
            eprintln!("occlusion_query_read failed: {}", e);
        }
    }
}
