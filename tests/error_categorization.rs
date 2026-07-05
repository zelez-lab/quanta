#![cfg(feature = "render")]
//! Verifies that the typed-API surface returns the right `QuantaErrorKind`
//! variant for each error class (step 070).
//!
//! - `NotSupported`: feature is genuinely unavailable on this backend.
//! - `NotFound`: handle does not refer to a live resource.
//! - `InvalidParam`: caller passed a value outside the documented range.
//!
//! Run: cargo test --test error_categorization --features software

#![cfg(feature = "software")]

use quanta::{QuantaError, QuantaErrorKind};

#[test]
fn display_strings_include_category_prefix() {
    let ns = QuantaError::not_supported("foo");
    let nf = QuantaError::not_found("bar");
    let ip = QuantaError::invalid_param("baz");
    let it = QuantaError::internal("qux");

    assert!(format!("{}", ns).contains("not supported on this backend"));
    assert!(format!("{}", nf).contains("not found"));
    assert!(format!("{}", ip).contains("invalid parameter"));
    assert!(format!("{}", it).contains("internal error"));
}

#[test]
fn error_kinds_are_distinct() {
    let ns = QuantaError::not_supported("a");
    let nf = QuantaError::not_found("a");
    let ip = QuantaError::invalid_param("a");

    assert!(matches!(ns.kind, QuantaErrorKind::NotSupported(_)));
    assert!(matches!(nf.kind, QuantaErrorKind::NotFound(_)));
    assert!(matches!(ip.kind, QuantaErrorKind::InvalidParam(_)));

    // PartialEq: same variant + payload compares equal.
    assert_eq!(ns, QuantaError::not_supported("a"));
    assert_ne!(ns, nf);
    assert_ne!(ns, ip);
    assert_ne!(nf, ip);
}

#[test]
fn typed_wrapper_oob_arg_returns_invalid_param() {
    // Caller passed bad arg → InvalidParam.
    let gpu = quanta::init_cpu();
    let r = gpu.printf_buffer(0);
    match r {
        Err(e) => assert!(
            matches!(e.kind, QuantaErrorKind::InvalidParam(_)),
            "expected InvalidParam, got {:?}",
            e.kind
        ),
        Ok(_) => panic!("zero-capacity printf_buffer should fail"),
    }
}

#[test]
fn unknown_queue_handle_returns_not_found() {
    // Unknown handle path: queue_signal on a never-allocated handle.
    // After 070 migration, "queue not found" returns NotFound. (The
    // raw-handle API is gone from `Gpu`; reach through the device
    // handle for the driver-level check.)
    let gpu = quanta::init_cpu();
    let r = gpu.device_handle().queue_signal(99_999_999, 0);
    match r {
        Err(e) => assert!(
            matches!(e.kind, QuantaErrorKind::NotFound(_)),
            "expected NotFound, got {:?}",
            e.kind
        ),
        Ok(_) => panic!("unknown queue handle should fail"),
    }
}

#[test]
fn context_propagates_with_existing_kind() {
    let e = QuantaError::not_supported("foo").with_context("rendering pass");
    assert!(matches!(e.kind, QuantaErrorKind::NotSupported(_)));
    let s = format!("{}", e);
    assert!(s.contains("not supported on this backend"));
    assert!(s.contains("rendering pass"));
}

// === Step 062 parity: default-stub features must report NotSupported ===
//
// The CPU driver doesn't override these `GpuDevice` methods, so they
// fall through to the default trait impls. Step 062 sweeps the last
// `invalid_param` defaults to `not_supported` so backends without an
// implementation of a feature surface a category that matches the
// step 070 contract.

#[test]
fn cpu_occlusion_query_create_returns_not_supported() {
    let gpu = quanta::init_cpu();
    let r = gpu.occlusion_query_create(4);
    match r {
        Err(e) => assert!(
            matches!(e.kind, QuantaErrorKind::NotSupported(_)),
            "expected NotSupported, got {:?}",
            e.kind
        ),
        Ok(_) => panic!("CPU occlusion_query_create should return NotSupported"),
    }
}

#[test]
fn cpu_timeline_create_returns_not_supported() {
    let gpu = quanta::init_cpu();
    let r = gpu.timeline_create();
    match r {
        Err(e) => assert!(
            matches!(e.kind, QuantaErrorKind::NotSupported(_)),
            "expected NotSupported, got {:?}",
            e.kind
        ),
        Ok(_) => panic!("CPU timeline_create should return NotSupported"),
    }
}

#[test]
fn cpu_render_begin_returns_not_supported() {
    // CPU has no rasterizer; opening a render pass on it must
    // surface NotSupported, not a generic invalid-param error.
    let gpu = quanta::init_cpu();
    let target = gpu
        .render_target(8, 8, quanta::Format::RGBA8)
        .expect("render_target on CPU should still allocate");
    let r = gpu.render(&target);
    match r {
        Err(e) => assert!(
            matches!(e.kind, QuantaErrorKind::NotSupported(_)),
            "expected NotSupported, got {:?}",
            e.kind
        ),
        Ok(_) => panic!("CPU render() should return NotSupported"),
    }
}
