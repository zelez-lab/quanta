//! One device buffer, many holders — `SharedField<T>` (= `Arc<Field<T>>`).
//!
//! The deliverable shape: two independent callers share one device
//! buffer, one writes through a kernel, the other reads, zero CPU
//! copies of the buffer between them, and the buffer is freed exactly
//! once when the last holder drops. The ghost model (T770–T774)
//! proves the state machine; these tests check the production side.
//!
//! Run: cargo test --test shared_field --features software

use quanta::QuantaErrorKind;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[quanta::kernel]
fn triple(seed: &[f32], out: &mut [f32]) {
    let i = quark_id();
    out[i] = seed[i] * 3.0f32;
}

/// Two holders on two threads: thread A dispatches a kernel writing
/// the shared buffer and waits; the main thread reads the result. No
/// CPU copy of the buffer moves between the holders — they touch the
/// same device allocation through the same handle.
#[test]
fn two_holders_one_writes_one_reads() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    const N: usize = 1024;
    let seed = gpu.field::<f32>(N).unwrap();
    seed.write(&(0..N).map(|i| i as f32).collect::<Vec<_>>())
        .unwrap();

    let shared = gpu.field::<f32>(N).unwrap().into_shared();
    let writer_clone = shared.clone();
    assert_eq!(
        writer_clone.handle(),
        shared.handle(),
        "one buffer, two holders"
    );

    let writer_gpu = gpu.clone();
    let writer = std::thread::spawn(move || {
        let mut wave = triple(&writer_gpu).expect("create wave");
        wave.bind(0, &seed);
        wave.bind(1, &writer_clone);
        writer_gpu
            .dispatch(&wave, N as u32)
            .unwrap()
            .wait()
            .unwrap();
        // writer_clone drops here — the buffer must survive: the main
        // thread still holds a clone.
    });
    writer.join().unwrap();

    let result = shared.read().unwrap();
    for (i, &r) in result.iter().enumerate() {
        assert_eq!(r, i as f32 * 3.0, "element {i}");
    }
}

/// Freed exactly once: N holders, drops in any order, exactly one
/// registry release — the leak-check idiom over the shared form.
#[test]
fn shared_freed_exactly_once() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let before = gpu.debug_registry_counts();
    let a = gpu.field::<f32>(256).unwrap().into_shared();
    let b = a.clone();
    let c = b.clone();
    let during = gpu.debug_registry_counts();
    assert_ne!(before, during, "the shared buffer holds one entry");

    drop(a);
    assert_eq!(
        during,
        gpu.debug_registry_counts(),
        "early drops must not free — two holders remain"
    );
    drop(c);
    assert_eq!(
        during,
        gpu.debug_registry_counts(),
        "early drops must not free — one holder remains"
    );
    drop(b);
    assert_eq!(
        before,
        gpu.debug_registry_counts(),
        "the last drop frees exactly once"
    );
}

/// Native buffer export mirrors the texture contract: a real object
/// where the capability says so, NotSupported elsewhere.
#[test]
fn native_handle_matches_capability() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let field = gpu.field::<f32>(64).unwrap();
    match field.native_handle() {
        Ok(handle) => {
            assert!(gpu.supports_native_handle_export());
            match handle {
                quanta::NativeBufferHandle::Metal { buffer } => assert!(!buffer.is_null()),
                quanta::NativeBufferHandle::Vulkan {
                    buffer,
                    memory,
                    size,
                } => {
                    assert!(!buffer.is_null());
                    assert!(!memory.is_null());
                    assert_eq!(size, 64 * 4);
                }
                _ => {}
            }
        }
        Err(e) => {
            assert!(!gpu.supports_native_handle_export());
            assert!(matches!(e.kind, QuantaErrorKind::NotSupported(_)));
        }
    }
}

/// The export works identically through the shared form (Deref).
#[test]
fn shared_native_handle_via_deref() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let shared = gpu.field::<f32>(64).unwrap().into_shared();
    assert_eq!(
        shared.native_handle().is_ok(),
        gpu.supports_native_handle_export()
    );
}
