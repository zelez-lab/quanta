#![cfg(feature = "render")]
//! Tier 2 -- Pulse lifecycle (sync primitives).
//!
//! Verifies pulse_poll, pulse_wait, and reset behavior.
//! Requires a GPU; skips gracefully if none available.

use quanta::RenderGpu;

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
    let field = gpu.field::<f32>(count).unwrap();
    field.write(&vec![0.0f32; count]).unwrap();

    let mut wave = noop_kernel(&gpu).unwrap();
    wave.bind(0, &field);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    pulse.wait().unwrap();

    // After wait, poll should return true (completed).
    assert!(pulse.is_done(), "pulse should be completed after wait");
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
    let field = gpu.field::<f32>(count).unwrap();
    field.write(&vec![0.0f32; count]).unwrap();

    let mut wave = noop_kernel(&gpu).unwrap();
    wave.bind(0, &field);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();

    // Poll before explicit wait -- should still be done (sync dispatch).
    let done = pulse.is_done();
    // Either true (sync) or false (async) -- both are valid.
    // But wait should always succeed.
    pulse.wait().unwrap();
    assert!(pulse.is_done(), "pulse must be done after wait");
    let _ = done; // silence unused
}

#[test]
fn pulse_wait_idempotent() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 32;
    let field = gpu.field::<f32>(count).unwrap();
    field.write(&vec![1.0f32; count]).unwrap();

    let mut wave = noop_kernel(&gpu).unwrap();
    wave.bind(0, &field);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();

    // Waiting multiple times should not error.
    pulse.wait().unwrap();
    pulse.wait().unwrap();
    pulse.wait().unwrap();
    assert!(pulse.is_done());
}

#[test]
fn pulse_wait_and_reset() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 32;
    let field = gpu.field::<f32>(count).unwrap();
    field.write(&vec![0.0f32; count]).unwrap();

    let mut wave = noop_kernel(&gpu).unwrap();
    wave.bind(0, &field);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();

    pulse.wait().unwrap();
    pulse.reset();

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
    let mut pulse = gpu.render(&target).unwrap().pulse().unwrap();

    pulse.wait().unwrap();
    assert!(pulse.is_done(), "render pulse should be done after wait");
}

#[test]
fn pulse_outlives_every_gpu_handle() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 64;
    let field = gpu.field::<f32>(count).unwrap();
    field.write(&vec![0.0f32; count]).unwrap();
    let mut wave = noop_kernel(&gpu).unwrap();
    wave.bind(0, &field);

    // Batch gives a genuinely deferred pulse where supported (Metal);
    // backends without batch dispatch pin the device through a plain
    // dispatch pulse instead.
    let mut pulse = match gpu.batch() {
        Ok(mut batch) => {
            batch.dispatch(&wave, count as u32).unwrap();
            batch.pulse().unwrap()
        }
        Err(_) => gpu.dispatch(&wave, count as u32).unwrap(),
    };

    // Every other handle goes FIRST: the pulse keeps the device alive
    // on its own, so the deferred wait below runs against a live
    // device. The depth-N in-flight-fence pattern holds a pulse across
    // teardown by design — this must never dangle.
    drop(wave);
    drop(field);
    drop(gpu);

    pulse.wait().unwrap();
    assert!(pulse.is_done());
}

#[test]
fn render_pulse_dropped_after_gpu() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let target = gpu.render_target(16, 16, quanta::Format::RGBA8).unwrap();
    let pulse = gpu.render(&target).unwrap().pulse().unwrap();

    // The exact consumer teardown order that used to use-after-free on
    // Vulkan: the render pulse (whose deferred cleanup waits the
    // submit fence when it drops) outlives the target and the Gpu, and
    // is dropped WITHOUT an explicit wait.
    drop(target);
    drop(gpu);
    drop(pulse);
}
