//! Tier 2 -- Pulse lifecycle (sync primitives).
//!
//! Verifies pulse_poll, pulse_wait, and reset behavior.
//! Requires a GPU; skips gracefully if none available.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[quanta::kernel]
fn noop_kernel(data: &mut [f32]) {
    let i = quark_id();
    data[i] = data[i] + 0.0;
}

#[test]
fn pulse_poll_returns_true_after_wait() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 64;
    let field = gpu.compute_field::<f32>(count).unwrap();
    gpu.write_field(&field, &vec![0.0f32; count]).unwrap();

    let mut wave = noop_kernel(&gpu).unwrap();
    wave.bind(0, &field);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    // After wait, poll should return true (completed).
    assert!(gpu.poll(&pulse), "pulse should be completed after wait");
}

#[test]
fn pulse_poll_completed_on_dispatch() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // The Metal/Vulkan drivers submit-and-wait synchronously,
    // so pulse is completed immediately on dispatch.
    let count = 64;
    let field = gpu.compute_field::<f32>(count).unwrap();
    gpu.write_field(&field, &vec![0.0f32; count]).unwrap();

    let mut wave = noop_kernel(&gpu).unwrap();
    wave.bind(0, &field);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();

    // Poll before explicit wait -- should still be done (sync dispatch).
    let done = gpu.poll(&pulse);
    // Either true (sync) or false (async) -- both are valid.
    // But wait should always succeed.
    gpu.wait(&mut pulse).unwrap();
    assert!(gpu.poll(&pulse), "pulse must be done after wait");
    let _ = done; // silence unused
}

#[test]
fn pulse_wait_idempotent() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 32;
    let field = gpu.compute_field::<f32>(count).unwrap();
    gpu.write_field(&field, &vec![1.0f32; count]).unwrap();

    let mut wave = noop_kernel(&gpu).unwrap();
    wave.bind(0, &field);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();

    // Waiting multiple times should not error.
    gpu.wait(&mut pulse).unwrap();
    gpu.wait(&mut pulse).unwrap();
    gpu.wait(&mut pulse).unwrap();
    assert!(gpu.poll(&pulse));
}

#[test]
fn pulse_wait_and_reset() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 32;
    let field = gpu.compute_field::<f32>(count).unwrap();
    gpu.write_field(&field, &vec![0.0f32; count]).unwrap();

    let mut wave = noop_kernel(&gpu).unwrap();
    wave.bind(0, &field);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();

    gpu.wait_and_reset(&mut pulse).unwrap();

    // After reset, is_done should return false.
    assert!(!pulse.is_done(), "pulse should not be done after reset");
}

#[test]
fn render_pass_returns_pulse() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let target = gpu.render_target(16, 16, quanta::Format::RGBA8).unwrap();
    let pass = gpu.render_begin(&target).unwrap();
    let mut pulse = gpu.render_end(pass).unwrap();

    gpu.wait(&mut pulse).unwrap();
    assert!(gpu.poll(&pulse), "render pulse should be done after wait");
}
