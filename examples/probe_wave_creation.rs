//! Wave-creation cost probe — the measurement the wave/pipeline-cache
//! follow-up is gated on (`roadmap/_design/deferred_dispatch.md`).
//!
//! Each `wave_jit` call resolves through the per-device wave cache
//! (`api::wave_cache`): the first creation of a kernel builds the
//! shader module + compute pipeline, repeats hand out a fresh `Wave`
//! over the cached pipeline. The probe measured the pre-cache cost
//! that justified the cache (lavapipe release: ~80% of the sync
//! per-op cost was construction) and now guards both sides of it.
//! Four numbers, printed per-op:
//!
//! - `creation only` — same-bytes `wave_jit` + drop in a loop: the
//!   cache-hit cost (pre-cache, this was the full construction cost).
//! - `creation only (cold)` — distinct kernel bytes per iteration
//!   defeat the cache: the driver-side construction cost the cache
//!   removes.
//! - `creation+dispatch+wait` — the per-op cost of the sci/ml
//!   create-per-dispatch shape (cache-hit + dispatch).
//! - `dispatch+wait (reused wave)` — the floor: no creation at all.
//!
//! Informational: the probe asserts correctness of one dispatch but
//! never fails on timing. Grep CI logs for `wave-cache probe:`.
//!
//! Phase markers go to STDERR (unbuffered — a crash cannot eat them the
//! way block-buffered stdout loses `println!` output under CI pipes):
//! the probe's first lavapipe run segfaulted with empty stdout, so the
//! markers are what localizes any recurrence.

/// Phase marker on stderr; stderr is unbuffered, so this survives a
/// crash that swallows pending stdout.
fn mark(phase: &str) {
    eprintln!("wave-cache probe [phase] {phase}");
}

fn main() {
    mark("init");
    let gpu = quanta::init().expect("probe requires a GPU device");
    mark("init done");
    println!("wave-cache probe: device = {}", gpu.name());

    let def = build_def();
    let bytes = quanta_ir::serialize_kernel(&def);

    const N: usize = 4096;
    let fin = gpu.field::<f32>(N).unwrap();
    let fout = gpu.field::<f32>(N).unwrap();
    fin.write(&vec![1.5f32; N]).unwrap();

    // Warm-up + correctness: one full create/bind/dispatch/read cycle.
    mark("warmup create");
    let mut wave = gpu.wave_jit(&bytes).unwrap();
    wave.bind(0, &fin);
    wave.bind(1, &fout);
    let mut pulse = gpu.dispatch(&wave, N as u32).unwrap();
    pulse.wait().unwrap();
    let out = fout.read().unwrap();
    mark("warmup done");
    assert!(
        (out[0] - 2.5).abs() < 1e-6,
        "probe kernel wrong: {}",
        out[0]
    );

    // Creation only, same bytes: all cache hits after the warmup —
    // the per-op creation cost the sci/ml layers actually pay now.
    mark("create loop");
    const CREATES: u32 = 32;
    let t = std::time::Instant::now();
    for _ in 0..CREATES {
        let w = gpu.wave_jit(&bytes).unwrap();
        drop(w);
    }
    let per_create = t.elapsed().as_secs_f64() * 1e6 / CREATES as f64;
    println!("wave-cache probe: creation only: {per_create:.1} us/op ({CREATES} creates)");

    mark("create loop done");

    // Cold creation: a distinct kernel per iteration defeats the
    // cache, measuring the driver-side construction cost it removes
    // (comparable to the pre-cache `creation only` history).
    mark("cold create loop");
    let t = std::time::Instant::now();
    for i in 0..CREATES {
        let mut def = build_def();
        def.name = format!("probe_add_one_{i}");
        let cold_bytes = quanta_ir::serialize_kernel(&def);
        let w = gpu.wave_jit(&cold_bytes).unwrap();
        drop(w);
    }
    let per_cold = t.elapsed().as_secs_f64() * 1e6 / CREATES as f64;
    println!(
        "wave-cache probe: creation only (cold): {per_cold:.1} us/op ({CREATES} distinct kernels)"
    );

    mark("cold create loop done");

    // Creation + dispatch + wait: today's create-per-dispatch shape.
    mark("full loop");
    const FULL: u32 = 16;
    let t = std::time::Instant::now();
    for _ in 0..FULL {
        let mut w = gpu.wave_jit(&bytes).unwrap();
        w.bind(0, &fin);
        w.bind(1, &fout);
        let mut p = gpu.dispatch(&w, N as u32).unwrap();
        p.wait().unwrap();
    }
    let per_full = t.elapsed().as_secs_f64() * 1e6 / FULL as f64;
    println!("wave-cache probe: creation+dispatch+wait: {per_full:.1} us/op ({FULL} iters)");

    mark("full loop done");

    // Reused wave: the floor a perfect cache would approach.
    mark("reuse loop");
    const REUSED: u32 = 32;
    let t = std::time::Instant::now();
    for _ in 0..REUSED {
        let mut p = gpu.dispatch(&wave, N as u32).unwrap();
        p.wait().unwrap();
    }
    let per_reuse = t.elapsed().as_secs_f64() * 1e6 / REUSED as f64;
    println!(
        "wave-cache probe: dispatch+wait (reused wave): {per_reuse:.1} us/op ({REUSED} iters)"
    );

    println!(
        "wave-cache probe: cacheable share = {:.1} us/op ({:.0}% of the per-op cost)",
        per_full - per_reuse,
        100.0 * (per_full - per_reuse) / per_full
    );
    use std::io::Write as _;
    std::io::stdout().flush().ok();
    mark("done (teardown follows)");
}

/// A minimal add-one kernel, built as runtime IR so the probe needs no
/// compiler binary: `out[i] = in[i] + 1.0`.
fn build_def() -> quanta::kernel::KernelDef {
    use quanta::kernel::*;
    KernelDef {
        name: "probe_add_one".into(),
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
                value: ConstValue::F32(1.0),
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
        opt_level: 3,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}
