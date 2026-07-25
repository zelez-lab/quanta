//! Per-op overhead layers of the define-by-run array path — the
//! measurement harness behind the deferred-dispatch increment (design
//! record: roadmap/_design/deferred_dispatch.md). Layer 0 is what an
//! autodiff tape op costs end to end; A–E isolate where the time goes.
//!
//! Layers measured (small [4096] f32 arrays, nn-test scale):
//!   0. full composed path: Array::add (deferred-lane encoded)
//!   A. wave_jit alone (per-op JIT compile cost)
//!   B. pre-JIT'd wave: dispatch + pulse-wait (= lane flush) per op
//!   C. pre-JIT'd wave: dispatch (encode) per op, single flush at end
//!   D. explicit batch API: N encodes, one commit, one wait
//!   E. 1-element field.read() (the `sum` scalar round-trip)
//!
//! Run: cargo run --release --features "sci metal jit compute" \
//!        --example bench_op_overhead

use std::time::Instant;

use quanta_ir::{BinOp, KernelDef, KernelOp, KernelParam, Reg, ScalarType, serialize_kernel};

fn qa_binary_add_def() -> KernelDef {
    // Byte-for-byte the def quanta-array's ufunc `add` builds.
    let ty = ScalarType::F32;
    KernelDef {
        name: "qa_binary".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: ty,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: ty,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 2,
                scalar_type: ty,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty,
            },
            KernelOp::Load {
                dst: Reg(2),
                field: 1,
                index: Reg(0),
                ty,
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Add,
                ty,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(3),
                ty,
            },
        ],
        body_source: None,
        next_reg: 4,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = quanta::init()?;
    let n = 4096usize;
    let iters = 300u32;

    let host = vec![1.0f32; n];
    let a = quanta::sci::Array::from_slice(&gpu, &host, &[n])?;
    let b = quanta::sci::Array::from_slice(&gpu, &host, &[n])?;

    // Layer 0: the full composed path, exactly what the tape does.
    let _ = a.add(&b)?; // warmup (first JIT may differ)
    let t = Instant::now();
    for _ in 0..iters {
        let _ = a.add(&b)?;
    }
    let full = t.elapsed();

    // Raw-core setup for the isolated layers.
    let bytes = serialize_kernel(&qa_binary_add_def());
    let fa = gpu.field::<f32>(n)?;
    let fb = gpu.field::<f32>(n)?;
    let fo = gpu.field::<f32>(n)?;

    // A: JIT alone.
    let _ = gpu.wave_jit(&bytes)?; // warmup
    let t = Instant::now();
    for _ in 0..iters {
        let _w = gpu.wave_jit(&bytes)?;
    }
    let jit = t.elapsed();

    // B: pre-JIT'd wave, dispatch + wait per op.
    let mut wave = gpu.wave_jit(&bytes)?;
    wave.bind(0, &fa);
    wave.bind(1, &fb);
    wave.bind(2, &fo);
    gpu.dispatch(&wave, n as u32)?.wait()?; // warmup
    let t = Instant::now();
    for _ in 0..iters {
        gpu.dispatch(&wave, n as u32)?.wait()?;
    }
    let dispatch_wait = t.elapsed();

    // C: dispatch per op, wait once at the end.
    let t = Instant::now();
    let mut last = None;
    for _ in 0..iters {
        last = Some(gpu.dispatch(&wave, n as u32)?);
    }
    if let Some(mut p) = last {
        p.wait()?;
    }
    gpu.wait_idle()?;
    let dispatch_nowait = t.elapsed();

    // D: batch — encode all, one commit, one wait.
    let t = Instant::now();
    let mut batch = gpu.batch()?;
    for _ in 0..iters {
        batch.dispatch(&wave, n as u32)?;
    }
    let mut pulse = batch.pulse()?;
    pulse.wait()?;
    let batched = t.elapsed();

    // E: tiny host readback (the `sum` scalar round-trip shape).
    let f1 = gpu.field::<f32>(1)?;
    let _ = f1.read()?; // warmup
    let t = Instant::now();
    for _ in 0..iters {
        let _ = f1.read()?;
    }
    let readback = t.elapsed();

    let per = |d: std::time::Duration| d.as_secs_f64() * 1e6 / iters as f64;
    println!("iters={iters}, n={n} f32");
    println!(
        "0 full composed (JIT+dispatch+wait) : {:>9.1} us/op",
        per(full)
    );
    println!(
        "A wave_jit alone                    : {:>9.1} us/op",
        per(jit)
    );
    println!(
        "B dispatch+wait (pre-JIT)           : {:>9.1} us/op",
        per(dispatch_wait)
    );
    println!(
        "C dispatch only, one wait at end    : {:>9.1} us/op",
        per(dispatch_nowait)
    );
    println!(
        "D batch encode, one commit+wait     : {:>9.1} us/op",
        per(batched)
    );
    println!(
        "E 1-elem field.read()               : {:>9.1} us/op",
        per(readback)
    );
    Ok(())
}
