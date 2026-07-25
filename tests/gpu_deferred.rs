#![cfg(all(feature = "compute", feature = "std"))]
//! Deferred dispatch — lane semantics. Deferral is THE dispatch
//! model: every `Gpu::dispatch` encodes into the per-device lane.
//!
//! The contract under test: dispatches encode instead of committing,
//! program order is preserved (including chains through distinct
//! buffers — the autodiff tape's shape), and every documented sync
//! point (`Pulse::wait`, `Gpu::flush`, `Gpu::wait_idle`, a `Field`
//! byte op on an owed buffer) completes all pending work before
//! returning. Runs on every batch-capable backend (Metal, CPU); on
//! backends without a batch path dispatch degrades to
//! commit-and-wait, and the same assertions must still hold.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[quanta::kernel]
fn add_one(data: &mut [f32]) {
    let i = quark_id();
    data[i] = data[i] + 1.0;
}

#[quanta::kernel]
fn double_into(src: &[f32], dst: &mut [f32]) {
    let i = quark_id();
    dst[i] = src[i] * 2.0;
}

#[test]
fn deferred_chain_through_distinct_buffers() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let count = 256usize;

    let a = gpu.field::<f32>(count).unwrap();
    a.write(&vec![3.0f32; count]).unwrap();
    let b = gpu.field::<f32>(count).unwrap();
    let c = gpu.field::<f32>(count).unwrap();

    // a -+1-> a, a -*2-> b, b -*2-> c: three dependent dispatches, two
    // of them chained through different buffers. Encode order must be
    // execution order or c reads stale data.
    let mut w1 = add_one(&gpu).unwrap();
    w1.bind(0, &a);
    let mut w2 = double_into(&gpu).unwrap();
    w2.bind(0, &a);
    w2.bind(1, &b);
    let mut w3 = double_into(&gpu).unwrap();
    w3.bind(0, &b);
    w3.bind(1, &c);

    let _ = gpu.dispatch(&w1, count as u32).unwrap();
    let _ = gpu.dispatch(&w2, count as u32).unwrap();
    let mut last = gpu.dispatch(&w3, count as u32).unwrap();
    last.wait().unwrap();

    let out = c.read().unwrap();
    assert!(
        out.iter().all(|&v| v == 16.0),
        "(3+1)*2*2 = 16 expected, got {:?}…",
        &out[..4]
    );
}

#[test]
fn pulse_wait_flushes_everything_encoded() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let count = 64usize;
    let field = gpu.field::<f32>(count).unwrap();
    field.write(&vec![0.0f32; count]).unwrap();

    let mut wave = add_one(&gpu).unwrap();
    wave.bind(0, &field);

    // Wait the FIRST pulse after encoding all three: a lane wait is a
    // full flush, not a per-dispatch fence.
    let mut first = gpu.dispatch(&wave, count as u32).unwrap();
    let _ = gpu.dispatch(&wave, count as u32).unwrap();
    let _ = gpu.dispatch(&wave, count as u32).unwrap();
    first.wait().unwrap();
    assert!(first.is_done());

    let out = field.read().unwrap();
    assert!(out.iter().all(|&v| v == 3.0), "got {:?}…", &out[..4]);
}

#[test]
fn gpu_flush_completes_dropped_pulses() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let count = 64usize;
    let field = gpu.field::<f32>(count).unwrap();
    field.write(&vec![0.0f32; count]).unwrap();

    let mut wave = add_one(&gpu).unwrap();
    wave.bind(0, &field);

    // The array-layer usage shape: pulses dropped un-waited, one
    // explicit flush before the host read.
    for _ in 0..5 {
        let _ = gpu.dispatch(&wave, count as u32).unwrap();
    }
    gpu.flush().unwrap();

    let out = field.read().unwrap();
    assert!(out.iter().all(|&v| v == 5.0), "got {:?}…", &out[..4]);
}

#[test]
fn wait_idle_flushes_the_lane() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let count = 64usize;
    let field = gpu.field::<f32>(count).unwrap();
    field.write(&vec![0.0f32; count]).unwrap();

    let mut wave = add_one(&gpu).unwrap();
    wave.bind(0, &field);
    let _ = gpu.dispatch(&wave, count as u32).unwrap();

    // wait_idle on the ORIGINAL (eager) handle: the lane is
    // per-device, so any handle's wait_idle must see deferred work.
    gpu.wait_idle().unwrap();

    let out = field.read().unwrap();
    assert!(out.iter().all(|&v| v == 1.0), "got {:?}…", &out[..4]);
}

#[test]
fn threshold_auto_submit_preserves_order() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let count = 64usize;
    let field = gpu.field::<f32>(count).unwrap();
    field.write(&vec![0.0f32; count]).unwrap();

    let mut wave = add_one(&gpu).unwrap();
    wave.bind(0, &field);

    // 600 chained increments cross the 512-encode auto-submit, so the
    // chain spans two submissions — ordering must hold across the
    // boundary, not only inside one batch.
    let iters = 600u32;
    for _ in 0..iters {
        let _ = gpu.dispatch(&wave, count as u32).unwrap();
    }
    gpu.flush().unwrap();

    let out = field.read().unwrap();
    assert!(
        out.iter().all(|&v| v == iters as f32),
        "expected {}, got {:?}…",
        iters,
        &out[..4]
    );
}

#[test]
fn field_dropped_between_encode_and_flush() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let count = 64usize;

    let src = gpu.field::<f32>(count).unwrap();
    src.write(&vec![5.0f32; count]).unwrap();
    let out = gpu.field::<f32>(count).unwrap();

    {
        // `tmp` is an intermediate a composed expression would drop as
        // soon as the next op consumes it — before any flush. The
        // encoded dispatches must keep the underlying buffer alive.
        let tmp = gpu.field::<f32>(count).unwrap();
        let mut w1 = double_into(&gpu).unwrap();
        w1.bind(0, &src);
        w1.bind(1, &tmp);
        let mut w2 = double_into(&gpu).unwrap();
        w2.bind(0, &tmp);
        w2.bind(1, &out);
        let _ = gpu.dispatch(&w1, count as u32).unwrap();
        let _ = gpu.dispatch(&w2, count as u32).unwrap();
    } // tmp (and both waves) drop here, flush hasn't happened yet

    gpu.flush().unwrap();
    let v = out.read().unwrap();
    assert!(
        v.iter().all(|&x| x == 20.0),
        "5*2*2 = 20, got {:?}…",
        &v[..4]
    );
}

#[test]
fn field_read_completes_owed_work() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    // No pulse wait, no explicit flush: `Field::read` itself must
    // complete the encoded producer of the buffer it reads — the
    // lane's referenced-handles contract.
    let count = 64usize;
    let field = gpu.field::<f32>(count).unwrap();
    field.write(&vec![0.0f32; count]).unwrap();
    let mut wave = add_one(&gpu).unwrap();
    wave.bind(0, &field);
    let _ = gpu.dispatch(&wave, count as u32).unwrap();
    let out = field.read().unwrap();
    assert!(out.iter().all(|&v| v == 1.0), "got {:?}…", &out[..4]);
}

#[test]
fn flush_with_nothing_pending_is_a_noop() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    gpu.flush().unwrap();
    gpu.wait_idle().unwrap();
}

#[test]
fn independent_dispatches_share_a_run() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    // Two independent chains (disjoint buffers) plus a shared
    // READ-ONLY input: under hazard-run grouping these encode without
    // barriers between them (read-read never orders) and may execute
    // concurrently — results must still be exact.
    let count = 256usize;
    let src = gpu.field::<f32>(count).unwrap();
    src.write(&vec![7.0f32; count]).unwrap();
    let a = gpu.field::<f32>(count).unwrap();
    let b = gpu.field::<f32>(count).unwrap();

    let mut wa = double_into(&gpu).unwrap();
    wa.bind(0, &src);
    wa.bind(1, &a);
    let mut wb = double_into(&gpu).unwrap();
    wb.bind(0, &src);
    wb.bind(1, &b);

    for _ in 0..8 {
        let _ = gpu.dispatch(&wa, count as u32).unwrap();
        let _ = gpu.dispatch(&wb, count as u32).unwrap();
    }
    gpu.flush().unwrap();

    let va = a.read().unwrap();
    let vb = b.read().unwrap();
    assert!(va.iter().all(|&v| v == 14.0), "a: got {:?}…", &va[..4]);
    assert!(vb.iter().all(|&v| v == 14.0), "b: got {:?}…", &vb[..4]);
}
