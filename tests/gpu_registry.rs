//! The process-wide device registry: repeated `init()` converges on
//! ONE shared device (and therefore one deferred lane), and the device
//! still tears down and rebuilds when nothing holds it.

use std::sync::Arc;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[quanta::kernel]
fn fill_double(data: &mut [f32]) {
    let i = quark_id();
    data[i] = (i as f32) * 2.0;
}

/// Two `init()` calls in one process are clones of the same device —
/// the registry's core promise, and what makes the one-lane-per-device
/// contract hold across independent initializations.
#[test]
fn init_twice_is_one_device() {
    let Some(a) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let b = quanta::init().expect("second init");
    assert!(
        Arc::ptr_eq(a.device_handle(), b.device_handle()),
        "two init() calls must share one device"
    );
}

/// `devices()` agrees with `init()` — same registry, same contexts.
#[test]
fn devices_and_init_agree() {
    let Some(g) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let listed = quanta::devices();
    assert!(!listed.is_empty());
    assert!(
        Arc::ptr_eq(g.device_handle(), listed[0].device_handle()),
        "devices()[0] and init() must be the same device"
    );
}

/// Work dispatched through one clone is visible through the other —
/// the shared lane in action, not just pointer identity.
#[test]
fn work_flows_across_init_clones() {
    let Some(a) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let b = quanta::init().expect("second init");

    let n = 64usize;
    let field = a.field::<f32>(n).unwrap();
    let mut wave = fill_double(&a).unwrap();
    wave.bind(0, &field);
    let mut pulse = b.dispatch(&wave, n as u32).unwrap();
    pulse.wait().unwrap();

    let out = field.read().unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert!((v - (i as f32) * 2.0).abs() < 1e-6, "at {i}: {v}");
    }
}

/// Dropping the last handle releases the device; the next `init()`
/// rebuilds one that works. (The registry holds `Weak`s — nothing is
/// immortal.)
#[test]
fn rebuild_after_last_drop() {
    {
        let Some(g) = try_gpu() else {
            eprintln!("skipping: no GPU available");
            return;
        };
        drop(g);
    }
    // NOTE: under parallel `cargo test`, sibling tests may still hold
    // the device, making this an alive-reuse rather than a rebuild —
    // both are correct registry behavior; the rebuild path is pinned
    // by running this file single-threaded (CI does both modes by
    // running the suite normally and the storm below).
    let g = quanta::init().expect("re-init after drop");
    let n = 32usize;
    let field = g.field::<f32>(n).unwrap();
    let mut wave = fill_double(&g).unwrap();
    wave.bind(0, &field);
    let mut pulse = g.dispatch(&wave, n as u32).unwrap();
    pulse.wait().unwrap();
    assert!((field.read().unwrap()[7] - 14.0).abs() < 1e-6);
}

/// The Iris Xe repro shape: many threads init + dispatch + drop
/// concurrently. Pre-registry this stormed the driver with an
/// instance+device per thread; now every thread shares one context and
/// the only pressure is the registry lock.
#[test]
fn parallel_init_storm() {
    if try_gpu().is_none() {
        eprintln!("skipping: no GPU available");
        return;
    }
    let threads: Vec<_> = (0..8)
        .map(|t| {
            std::thread::spawn(move || {
                for _ in 0..4 {
                    let g = quanta::init().expect("storm init");
                    let n = 16usize;
                    let field = g.field::<f32>(n).unwrap();
                    let mut wave = fill_double(&g).unwrap();
                    wave.bind(0, &field);
                    let mut pulse = g.dispatch(&wave, n as u32).unwrap();
                    pulse.wait().unwrap();
                    assert!((field.read().unwrap()[3] - 6.0).abs() < 1e-6, "thread {t}");
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
}
