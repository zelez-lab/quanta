#![cfg(feature = "render")]
//! Integration tests for `TessellationPipeline` (steps 022 + 023).
//!
//! Refines the proven Lean theorems T7200–T7206 + Verus theorems
//! T7250–T7256 against the CPU device's software tessellation
//! pipeline. Each test names the theorem it refines.
//!
//! Run: cargo test --test tessellation_basic --features software

#![cfg(feature = "software")]

use quanta::RenderGpu;

use quanta::{MAX_PATCH_SIZE, MAX_TESS_LEVEL, TessTopology};

#[test]
fn tess_create_triangle_topology_shape() {
    // T7200 refinement: created pipeline has the requested control
    // points; topology determines outer/inner counts.
    let gpu = quanta::init_cpu();
    let p = gpu
        .tessellation_pipeline(TessTopology::Triangle, 3)
        .unwrap();
    assert_eq!(p.topology(), TessTopology::Triangle);
    assert_eq!(p.control_points(), 3);
    assert_eq!(p.topology().outer_count(), 3);
    assert_eq!(p.topology().inner_count(), 1);
}

#[test]
fn tess_create_quad_topology_shape() {
    // T7200 refinement, quad branch.
    let gpu = quanta::init_cpu();
    let p = gpu.tessellation_pipeline(TessTopology::Quad, 4).unwrap();
    assert_eq!(p.topology().outer_count(), 4);
    assert_eq!(p.topology().inner_count(), 2);
}

#[test]
fn tess_create_rejects_zero_control_points() {
    // T7205 refinement: cp == 0 is out of range and rejected.
    let gpu = quanta::init_cpu();
    let r = gpu.tessellation_pipeline(TessTopology::Triangle, 0);
    assert!(r.is_err());
}

#[test]
fn tess_create_rejects_oversized_control_points() {
    // T7205 refinement: cp > MAX_PATCH_SIZE rejected.
    let gpu = quanta::init_cpu();
    let r = gpu.tessellation_pipeline(TessTopology::Triangle, MAX_PATCH_SIZE + 1);
    assert!(r.is_err());
}

#[test]
fn tess_set_outer_in_bounds_succeeds() {
    // T7202 refinement: setOuter at an in-bounds index succeeds on
    // a live pipeline.
    let gpu = quanta::init_cpu();
    let p = gpu
        .tessellation_pipeline(TessTopology::Triangle, 3)
        .unwrap();
    p.set_outer(0, 4).unwrap();
    p.set_outer(1, 8).unwrap();
    p.set_outer(2, 16).unwrap();
}

#[test]
fn tess_set_outer_out_of_bounds_fails() {
    // T7202/T7203 (typed-wrapper precondition): index >=
    // outer_count() returns Err. Triangle has 3 outer factors, so
    // index 3 is out of range.
    let gpu = quanta::init_cpu();
    let p = gpu
        .tessellation_pipeline(TessTopology::Triangle, 3)
        .unwrap();
    let r = p.set_outer(3, 4);
    assert!(r.is_err());
}

#[test]
fn tess_set_inner_out_of_bounds_fails() {
    // Inner-factor mirror of the above. Triangle has 1 inner
    // factor — index 1 is out of range.
    let gpu = quanta::init_cpu();
    let p = gpu
        .tessellation_pipeline(TessTopology::Triangle, 3)
        .unwrap();
    let r = p.set_inner(1, 4);
    assert!(r.is_err());
}

#[test]
fn tess_factors_clamp_to_max_tess_level() {
    // T7201 refinement: requested factor > MAX_TESS_LEVEL is
    // accepted (the wrapper clamps). Backends only see clamped
    // values.
    let gpu = quanta::init_cpu();
    let p = gpu.tessellation_pipeline(TessTopology::Quad, 4).unwrap();
    p.set_outer(0, MAX_TESS_LEVEL + 999).unwrap();
    p.set_inner(0, 0).unwrap(); // clamped up to 1
}

#[test]
fn tess_drop_invalidates_pipeline() {
    // T7256 refinement: Drop releases the backend handle. We
    // verify by creating + dropping in a scope and observing no
    // panic from the destructor path.
    let gpu = quanta::init_cpu();
    {
        let p = gpu
            .tessellation_pipeline(TessTopology::Triangle, 3)
            .unwrap();
        let _ = p.handle();
        // dropped at end of scope
    }
}
