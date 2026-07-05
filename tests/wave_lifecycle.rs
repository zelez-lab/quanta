#![cfg(feature = "compute")]
//! Compute-resource lifecycle — proves waves no longer leak.
//!
//! A `Wave` releases its driver registry entry on Drop
//! (`device + live` pattern, same as the render wrappers). These tests
//! snapshot the driver's registry sizes around create+drop cycles and
//! assert the entries are freed, plus a 100-iteration
//! create+dispatch+drop loop asserting the wave registry does not grow
//! unboundedly — the test that would have caught the leak.
//!
//! Requires a GPU; skips gracefully if none available.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// --- Kernel (proc macro compiles at build time) ---

#[quanta::kernel]
fn lifecycle_add_one(data: &[f32], result: &mut [f32]) {
    let i = quark_id();
    result[i] = data[i] + 1.0;
}

// ─── Single-resource lifecycle ──────────────────────────────────────────────

#[test]
fn wave_drop_frees_registry_entry() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let before = gpu.debug_registry_counts();
    let wave = lifecycle_add_one(&gpu).unwrap();
    let during = gpu.debug_registry_counts();
    assert_eq!(
        during.waves,
        before.waves + 1,
        "wave creation should register a driver-side entry"
    );
    drop(wave);
    let after = gpu.debug_registry_counts();
    assert_eq!(before, after, "dropping a Wave must free its entry");
}

#[cfg(feature = "software")]
#[test]
fn cpu_wave_drop_frees_registry_entry() {
    let gpu = quanta::init_cpu();
    let before = gpu.debug_registry_counts();
    let wave = lifecycle_add_one(&gpu).unwrap();
    let during = gpu.debug_registry_counts();
    assert_eq!(
        during.waves,
        before.waves + 1,
        "CPU wave creation should register a kernel entry"
    );
    drop(wave);
    let after = gpu.debug_registry_counts();
    assert_eq!(before, after, "dropping a CPU Wave must free its entry");
}

// ─── Double-free safety ─────────────────────────────────────────────────────

fn pass_through(wave: quanta::Wave) -> quanta::Wave {
    wave
}

#[test]
fn moved_wave_frees_exactly_once() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let before = gpu.debug_registry_counts();

    // Move the wave through a function and a Vec: only the final
    // owner's Drop may release the handle.
    let wave = lifecycle_add_one(&gpu).unwrap();
    let wave = pass_through(wave);
    let mut owners = vec![wave];
    let wave = owners.pop().unwrap();
    drop(owners);

    let during = gpu.debug_registry_counts();
    assert_ne!(before, during, "moves must not release the entry early");

    drop(wave);
    let after = gpu.debug_registry_counts();
    assert_eq!(before, after, "moved Wave must be freed exactly once");

    // The device must still be healthy afterwards (no over-release):
    // a fresh wave dispatches and produces the right values.
    let count = 64;
    let input = gpu.field::<f32>(count).unwrap();
    let output = gpu.field::<f32>(count).unwrap();
    input.write(&vec![1.0f32; count]).unwrap();

    let mut probe = lifecycle_add_one(&gpu).unwrap();
    probe.bind(0, &input);
    probe.bind(1, &output);
    let mut pulse = gpu.dispatch(&probe, count as u32).unwrap();
    pulse.wait().unwrap();

    let result = output.read().unwrap();
    assert!(
        (result[0] - 2.0).abs() < 0.001,
        "probe dispatch after drops should produce 2.0, got {}",
        result[0]
    );
}

// ─── 100-iteration reuse loop (the test that would have caught the leak) ────

#[test]
fn hundred_wave_create_dispatch_drop_does_not_grow_registry() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 64;
    let input = gpu.field::<f32>(count).unwrap();
    let output = gpu.field::<f32>(count).unwrap();
    input.write(&vec![1.0f32; count]).unwrap();

    let before = gpu.debug_registry_counts();

    for _iter in 0..100 {
        // Per-iteration wave, dropped at the end of the iteration —
        // the hot-reload / long-session shape that leaked.
        let mut wave = lifecycle_add_one(&gpu).expect("wave");
        wave.bind(0, &input);
        wave.bind(1, &output);
        let mut pulse = gpu.dispatch(&wave, count as u32).expect("dispatch");
        pulse.wait().expect("wait");
    }

    let after = gpu.debug_registry_counts();
    assert_eq!(
        before, after,
        "100 create+dispatch+drop iterations must not grow the wave registry"
    );

    let result = output.read().unwrap();
    assert!(
        (result[0] - 2.0).abs() < 0.001,
        "final iteration should still produce 2.0, got {}",
        result[0]
    );
}
