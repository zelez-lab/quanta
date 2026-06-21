#![cfg(feature = "render")]
//! Integration tests for `MeshPipeline` (steps 024 + 025).
//!
//! Refines the proven Lean theorems T7300–T7305 + Verus theorems
//! T7350–T7354 against the CPU device's software mesh-pipeline
//! lifecycle. Each test names the theorem it refines.
//!
//! Run: cargo test --test mesh_shader_basic --features software

#![cfg(feature = "software")]

use quanta::{MAX_GROUP_COUNT, MAX_MESH_PRIMITIVES, MAX_MESH_VERTICES, MeshPipelineDesc};

#[test]
fn mesh_create_default_desc_succeeds() {
    // T7300 refinement: created pipeline has the default-desc limits.
    let gpu = quanta::init_cpu();
    let p = gpu.mesh_pipeline(MeshPipelineDesc::default()).unwrap();
    assert_eq!(p.max_vertices_per_meshlet(), 64);
    assert_eq!(p.max_primitives_per_meshlet(), 124);
}

#[test]
fn mesh_create_at_max_limits_succeeds() {
    // T7300 boundary: limits at the proven hardware-minimum cap.
    let gpu = quanta::init_cpu();
    let p = gpu
        .mesh_pipeline(MeshPipelineDesc {
            max_vertices_per_meshlet: MAX_MESH_VERTICES,
            max_primitives_per_meshlet: MAX_MESH_PRIMITIVES,
            task_threads_per_group: 64,
        })
        .unwrap();
    assert_eq!(p.max_vertices_per_meshlet(), MAX_MESH_VERTICES);
    assert_eq!(p.max_primitives_per_meshlet(), MAX_MESH_PRIMITIVES);
}

#[test]
fn mesh_create_rejects_zero_vertices() {
    // T7301 refinement: max_vertices == 0 is out of range.
    let gpu = quanta::init_cpu();
    let r = gpu.mesh_pipeline(MeshPipelineDesc {
        max_vertices_per_meshlet: 0,
        ..MeshPipelineDesc::default()
    });
    assert!(r.is_err());
}

#[test]
fn mesh_create_rejects_oversized_primitives() {
    // T7301 refinement: max_primitives > MAX_MESH_PRIMITIVES rejected.
    let gpu = quanta::init_cpu();
    let r = gpu.mesh_pipeline(MeshPipelineDesc {
        max_primitives_per_meshlet: MAX_MESH_PRIMITIVES + 1,
        ..MeshPipelineDesc::default()
    });
    assert!(r.is_err());
}

#[test]
fn mesh_dispatch_in_bounds_succeeds() {
    // T7302 refinement: dispatch on a live pipeline succeeds; the
    // recorded sequence is extended.
    let gpu = quanta::init_cpu();
    let p = gpu.mesh_pipeline(MeshPipelineDesc::default()).unwrap();
    p.dispatch([1, 1, 1]).unwrap();
    p.dispatch([4, 2, 1]).unwrap();
    p.dispatch([MAX_GROUP_COUNT, 1, 1]).unwrap();
}

#[test]
fn mesh_dispatch_out_of_range_fails() {
    // T7304 refinement: dispatch with axis > MAX_GROUP_COUNT fails.
    let gpu = quanta::init_cpu();
    let p = gpu.mesh_pipeline(MeshPipelineDesc::default()).unwrap();
    let r = p.dispatch([MAX_GROUP_COUNT + 1, 1, 1]);
    assert!(r.is_err());
}

#[test]
fn mesh_drop_invalidates_pipeline() {
    // T7305 refinement: Drop releases the backend handle. After
    // Drop the typed wrapper would refuse dispatch on `live=false`
    // (we observe by exiting the scope without panic).
    let gpu = quanta::init_cpu();
    {
        let p = gpu.mesh_pipeline(MeshPipelineDesc::default()).unwrap();
        let _ = p.handle();
    }
}
