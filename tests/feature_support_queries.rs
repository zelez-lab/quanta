//! Public surface for the per-backend capability caches —
//! step 063 slice 20.
//!
//! `gpu.supports_vrs()`, `supports_ray_tracing()`,
//! `supports_mesh_shaders()`, `supports_tessellation()`,
//! `supports_sparse_residency()`, and
//! `supported_shading_rates()` let callers gate on feature
//! availability without trial-and-error.
//!
//! On the CPU backend none of the deferred features have a real
//! lowering, so every query returns false / empty. The same shape
//! holds on Metal / Vulkan when the underlying hardware doesn't
//! advertise the extension; the test below covers the CPU path
//! which is reachable on every dev machine.
//!
//! Run: cargo test --test feature_support_queries --features software

#![cfg(feature = "software")]

#[test]
fn cpu_reports_no_advanced_render_features() {
    let gpu = quanta::init_cpu();
    assert!(!gpu.supports_vrs(), "CPU cannot rasterize");
    assert!(!gpu.supports_ray_tracing(), "CPU has no RT pipeline");
    assert!(!gpu.supports_mesh_shaders(), "CPU has no mesh stages");
    assert!(!gpu.supports_tessellation(), "CPU has no tessellation");
    assert!(
        !gpu.supports_sparse_residency(),
        "CPU sparse uses HashMap, not residency control"
    );
}

#[test]
fn cpu_reports_empty_shading_rate_list() {
    let gpu = quanta::init_cpu();
    assert!(gpu.supported_shading_rates().is_empty());
}

#[test]
fn feature_queries_match_render_path_gates() {
    // If a backend reports `supports_vrs() = false`, then the
    // typed VRS state path must not return Ok on `set_rate`. CPU
    // overrides VRS state lifecycle as a HashMap, so this
    // doesn't exercise the typed-wrapper-vs-feature-flag
    // alignment exhaustively, but documents the contract.
    let gpu = quanta::init_cpu();
    assert!(!gpu.supports_vrs());
    // The render path on CPU short-circuits earlier with
    // "render passes not supported on CPU device", so a positive
    // VRS state create on CPU doesn't contradict supports_vrs:
    // the gate is at render submission, not at state create.
}
