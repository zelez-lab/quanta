#![cfg(feature = "compute")]
//! Compute-resource lifecycle — waves neither leak nor double-free,
//! and the per-device wave cache dedups pipeline construction.
//!
//! Repeat `Gpu::wave`/`wave_jit` calls with the same kernel bytes
//! share ONE driver registry entry, co-owned by the cache and every
//! outstanding `Wave` (`api::wave_cache`); the entry is released by
//! the last owner, so a dropped `Wave` leaves the cached pipeline
//! registered BY DESIGN. These tests snapshot the driver's registry
//! sizes around create+drop cycles and assert three things: creation
//! registers exactly one entry per distinct kernel (dedup), drops
//! never release an entry another owner still holds (no double-free),
//! and `__wave_cache_drain` returns the registry to baseline (no
//! leak).
//!
//! Requires a GPU; skips gracefully if none available.

/// Every test in this file asserts absolute `debug_registry_counts`
/// snapshots, which are only sound on a device no parallel test can
/// allocate on — so the whole file initializes through the
/// registry-bypassing constructor instead of the process-shared
/// `init()` device.
fn try_isolated_gpu() -> Option<quanta::Gpu> {
    quanta::init_isolated().ok()
}

// --- Kernel (proc macro compiles at build time) ---

#[quanta::kernel]
fn lifecycle_add_one(data: &[f32], result: &mut [f32]) {
    let i = quark_id();
    result[i] = data[i] + 1.0;
}

// ─── Single-resource lifecycle ──────────────────────────────────────────────

#[test]
fn wave_drop_retains_cached_entry_until_drain() {
    let Some(gpu) = try_isolated_gpu() else {
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
    let after_drop = gpu.debug_registry_counts();
    assert_eq!(
        after_drop, during,
        "dropping a Wave must leave the cache-held pipeline registered"
    );
    gpu.__wave_cache_drain();
    let after = gpu.debug_registry_counts();
    assert_eq!(before, after, "draining the cache must free the entry");
}

#[cfg(feature = "software")]
#[test]
fn cpu_wave_drop_retains_cached_entry_until_drain() {
    let gpu = quanta::init_cpu_isolated();
    let before = gpu.debug_registry_counts();
    let wave = lifecycle_add_one(&gpu).unwrap();
    let during = gpu.debug_registry_counts();
    assert_eq!(
        during.waves,
        before.waves + 1,
        "CPU wave creation should register a kernel entry"
    );
    drop(wave);
    let after_drop = gpu.debug_registry_counts();
    assert_eq!(
        after_drop, during,
        "dropping a CPU Wave must leave the cache-held entry registered"
    );
    gpu.__wave_cache_drain();
    let after = gpu.debug_registry_counts();
    assert_eq!(before, after, "draining the cache must free the entry");
}

// ─── Double-free safety ─────────────────────────────────────────────────────

fn pass_through(wave: quanta::Wave) -> quanta::Wave {
    wave
}

#[test]
fn moved_wave_frees_exactly_once() {
    let Some(gpu) = try_isolated_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let before = gpu.debug_registry_counts();

    // Move the wave through a function and a Vec: only the final
    // owner's Drop may release its co-ownership of the entry.
    let wave = lifecycle_add_one(&gpu).unwrap();
    let wave = pass_through(wave);
    let mut owners = vec![wave];
    let wave = owners.pop().unwrap();
    drop(owners);

    let during = gpu.debug_registry_counts();
    assert_ne!(before, during, "moves must not release the entry early");

    // Cache drained while the Wave is still live: the Wave's Arc
    // keeps the pipeline registered.
    gpu.__wave_cache_drain();
    let held = gpu.debug_registry_counts();
    assert_eq!(
        held, during,
        "a live Wave must keep the drained entry registered"
    );

    drop(wave);
    let after = gpu.debug_registry_counts();
    assert_eq!(
        before, after,
        "the moved Wave (last owner) must free the entry exactly once"
    );

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
fn hundred_wave_create_dispatch_drop_shares_one_entry() {
    let Some(gpu) = try_isolated_gpu() else {
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
        // the create-per-op shape the sci/ml layers use.
        let mut wave = lifecycle_add_one(&gpu).expect("wave");
        wave.bind(0, &input);
        wave.bind(1, &output);
        let mut pulse = gpu.dispatch(&wave, count as u32).expect("dispatch");
        pulse.wait().expect("wait");
    }

    let after = gpu.debug_registry_counts();
    assert_eq!(
        after.waves,
        before.waves + 1,
        "100 same-kernel creations must share ONE cached driver entry"
    );

    gpu.__wave_cache_drain();
    let drained = gpu.debug_registry_counts();
    assert_eq!(
        before, drained,
        "drain must return the registry to baseline"
    );

    let result = output.read().unwrap();
    assert!(
        (result[0] - 2.0).abs() < 0.001,
        "final iteration should still produce 2.0, got {}",
        result[0]
    );
}

// ─── Cache semantics ────────────────────────────────────────────────────────

#[test]
fn cache_hit_shares_entry_with_independent_bindings() {
    let Some(gpu) = try_isolated_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 64;
    let input = gpu.field::<f32>(count).unwrap();
    let out_a = gpu.field::<f32>(count).unwrap();
    let out_b = gpu.field::<f32>(count).unwrap();
    input.write(&vec![3.0f32; count]).unwrap();

    let before = gpu.debug_registry_counts();
    let mut wave_a = lifecycle_add_one(&gpu).unwrap();
    let mut wave_b = lifecycle_add_one(&gpu).unwrap();
    let after = gpu.debug_registry_counts();

    // One driver entry serves both waves; binding state is per-Wave.
    assert_eq!(
        wave_a.handle(),
        wave_b.handle(),
        "same kernel bytes must share one driver pipeline"
    );
    assert_eq!(
        after.waves,
        before.waves + 1,
        "two same-kernel waves must register exactly one entry"
    );

    wave_a.bind(0, &input);
    wave_a.bind(1, &out_a);
    wave_b.bind(0, &input);
    wave_b.bind(1, &out_b);
    let mut pulse_a = gpu.dispatch(&wave_a, count as u32).unwrap();
    let mut pulse_b = gpu.dispatch(&wave_b, count as u32).unwrap();
    pulse_a.wait().unwrap();
    pulse_b.wait().unwrap();

    for (name, out) in [("a", &out_a), ("b", &out_b)] {
        let result = out.read().unwrap();
        assert!(
            (result[0] - 4.0).abs() < 0.001,
            "shared-pipeline wave {name} should produce 4.0, got {}",
            result[0]
        );
    }
}

#[test]
fn drain_with_encoded_batch_pending_is_safe() {
    let Some(gpu) = try_isolated_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 64;
    let input = gpu.field::<f32>(count).unwrap();
    let output = gpu.field::<f32>(count).unwrap();
    input.write(&vec![7.0f32; count]).unwrap();

    // Encode into the deferred lane, then drop BOTH owners (wave and
    // cache slot) before the flush: destruction must defer behind the
    // encoded work (Vulkan routes it through the retire bin / batch
    // pins), and the result must still be correct.
    let mut wave = lifecycle_add_one(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &output);
    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    drop(wave);
    gpu.__wave_cache_drain();
    pulse.wait().unwrap();

    let result = output.read().unwrap();
    assert!(
        (result[0] - 8.0).abs() < 0.001,
        "dispatch surviving both owners' drops should produce 8.0, got {}",
        result[0]
    );
}

// ─── LRU bounds (CPU: cheap distinct-kernel floods) ─────────────────────────

/// Build the probe-style runtime-IR kernel `out[i] = in[i] + K`,
/// distinct per `K` — distinct serialized bytes defeat the cache on
/// purpose.
#[cfg(feature = "software")]
fn add_k_def(k: f32) -> quanta::kernel::KernelDef {
    use quanta::kernel::*;
    KernelDef {
        name: format!("wave_cache_add_{}", k as u32),
        params: vec![
            KernelParam::FieldRead {
                name: "input".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::F32(k),
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Add,
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 1,
                index: Reg(0),
                src: Reg(3),
                ty: ScalarType::F32,
            },
        ],
        body_source: None,
        next_reg: 4,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

/// Mirrors `api::wave_cache::CAP` — update together.
#[cfg(feature = "software")]
const CACHE_CAP: usize = 256;

#[cfg(feature = "software")]
#[test]
fn cpu_lru_evicts_oldest_beyond_cap() {
    let gpu = quanta::init_cpu_isolated();
    let before = gpu.debug_registry_counts();

    // Kernel 0 first, then CACHE_CAP distinct kernels: 0 is the LRU
    // entry and must be the one evicted.
    let first_bytes = quanta_ir::serialize_kernel(&add_k_def(0.0));
    let first_handle = gpu.wave_jit(&first_bytes).unwrap().handle();
    for k in 1..=CACHE_CAP {
        gpu.wave_jit(&quanta_ir::serialize_kernel(&add_k_def(k as f32)))
            .unwrap();
    }

    let full = gpu.debug_registry_counts();
    assert_eq!(
        full.waves,
        before.waves + CACHE_CAP,
        "the cache must hold exactly CAP entries after CAP+1 distinct kernels"
    );

    // Kernel CACHE_CAP is recent → hit (same driver handle); kernel 0
    // was evicted → miss (fresh handle).
    let recent_bytes = quanta_ir::serialize_kernel(&add_k_def(CACHE_CAP as f32));
    let recent_a = gpu.wave_jit(&recent_bytes).unwrap().handle();
    let recent_b = gpu.wave_jit(&recent_bytes).unwrap().handle();
    assert_eq!(recent_a, recent_b, "a retained kernel must hit the cache");
    let first_again = gpu.wave_jit(&first_bytes).unwrap().handle();
    assert_ne!(
        first_again, first_handle,
        "the LRU kernel must have been evicted (recreation compiles fresh)"
    );

    gpu.__wave_cache_drain();
    let drained = gpu.debug_registry_counts();
    assert_eq!(
        before, drained,
        "drain must return the registry to baseline"
    );
}

#[cfg(feature = "software")]
#[test]
fn cpu_evicted_but_held_wave_still_dispatches() {
    let gpu = quanta::init_cpu_isolated();
    let before = gpu.debug_registry_counts();

    let count = 16;
    let input = gpu.field::<f32>(count).unwrap();
    let output = gpu.field::<f32>(count).unwrap();
    input.write(&vec![1.0f32; count]).unwrap();

    // Hold a wave while flooding the cache past CAP: its entry gets
    // evicted from the cache, but the wave's co-ownership keeps the
    // driver pipeline alive.
    let mut held = gpu
        .wave_jit(&quanta_ir::serialize_kernel(&add_k_def(41.0)))
        .unwrap();
    for k in 100..100 + CACHE_CAP {
        gpu.wave_jit(&quanta_ir::serialize_kernel(&add_k_def(k as f32)))
            .unwrap();
    }

    held.bind(0, &input);
    held.bind(1, &output);
    let mut pulse = gpu.dispatch(&held, count as u32).unwrap();
    pulse.wait().unwrap();
    let result = output.read().unwrap();
    assert!(
        (result[0] - 42.0).abs() < 0.001,
        "evicted-but-held wave should produce 42.0, got {}",
        result[0]
    );

    drop(held);
    gpu.__wave_cache_drain();
    let drained = gpu.debug_registry_counts();
    // Only the two fields remain registered beyond the baseline.
    assert_eq!(
        drained.waves, before.waves,
        "drop + drain must free every pipeline, evicted ones included"
    );
}
