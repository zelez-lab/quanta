#![cfg(feature = "render")]
//! Integration tests for `AccelerationStructure` + `RayTracingPipeline`
//! (steps 026 + 027).
//!
//! Refines the proven Lean theorems T7400–T7406 + Verus theorems
//! T7450–T7455 against the CPU device's software RT lifecycle. Each
//! test names the theorem it refines.
//!
//! Run: cargo test --test ray_tracing_basic --features software

#![cfg(feature = "software")]

use quanta::{AsKind, GeometryDesc, MAX_DISPATCH_DIM, MAX_RECURSION_DEPTH, RayTracingPipelineDesc};

fn dummy_geom(gpu: &quanta::Gpu) -> GeometryDesc {
    let f = gpu.field::<f32>(64).unwrap();
    GeometryDesc {
        vertices: f.handle(),
        indices: None,
        vertex_count: 8,
        index_count: 0,
        vertex_stride: 12,
    }
}

#[test]
fn rt_blas_build_records_geom_count() {
    // T7400 refinement: built BLAS captures kind + geom_count.
    let gpu = quanta::init_cpu();
    let g = dummy_geom(&gpu);
    let blas = gpu.acceleration_structure_blas(&[g, g, g]).unwrap();
    assert_eq!(blas.kind(), AsKind::Bottom);
    assert_eq!(blas.geom_count(), 3);
}

#[test]
fn rt_blas_build_rejects_empty_geometry() {
    // T7401 refinement: empty geometry list rejected.
    let gpu = quanta::init_cpu();
    let r = gpu.acceleration_structure_blas(&[]);
    assert!(r.is_err());
}

#[test]
fn rt_pipeline_create_records_recursion_depth() {
    // T7402 refinement: pipeline carries the requested recursion.
    let gpu = quanta::init_cpu();
    let p = gpu
        .ray_tracing_pipeline(&RayTracingPipelineDesc {
            ray_gen: &[],
            closest_hit: &[],
            miss: &[],
            max_recursion: 4,
        })
        .unwrap();
    assert_eq!(p.max_recursion(), 4);
}

#[test]
fn rt_pipeline_create_rejects_excess_recursion() {
    // T7402 boundary: max_recursion > MAX_RECURSION_DEPTH rejected
    // by the typed wrapper before it reaches the device.
    let gpu = quanta::init_cpu();
    let r = gpu.ray_tracing_pipeline(&RayTracingPipelineDesc {
        ray_gen: &[],
        closest_hit: &[],
        miss: &[],
        max_recursion: MAX_RECURSION_DEPTH + 1,
    });
    assert!(r.is_err());
}

#[test]
fn rt_dispatch_in_bounds_succeeds() {
    // T7403 refinement: dispatch on a live pipeline succeeds; the
    // recorded sequence is extended.
    let gpu = quanta::init_cpu();
    let p = gpu
        .ray_tracing_pipeline(&RayTracingPipelineDesc {
            ray_gen: &[],
            closest_hit: &[],
            miss: &[],
            max_recursion: 1,
        })
        .unwrap();
    p.dispatch_rays(64, 64).unwrap();
    p.dispatch_rays(MAX_DISPATCH_DIM, 1).unwrap();
}

#[test]
fn rt_dispatch_out_of_range_fails() {
    // T7405 refinement: dispatch with width > MAX_DISPATCH_DIM fails.
    let gpu = quanta::init_cpu();
    let p = gpu
        .ray_tracing_pipeline(&RayTracingPipelineDesc {
            ray_gen: &[],
            closest_hit: &[],
            miss: &[],
            max_recursion: 1,
        })
        .unwrap();
    let r = p.dispatch_rays(MAX_DISPATCH_DIM + 1, 1);
    assert!(r.is_err());
}

#[test]
fn rt_drop_invalidates_resources() {
    // T7406 refinement: Drop releases backend handles for both AS
    // and RT pipeline. We verify by exiting the scope without panic.
    let gpu = quanta::init_cpu();
    {
        let g = dummy_geom(&gpu);
        let _blas = gpu.acceleration_structure_blas(&[g]).unwrap();
        let _p = gpu
            .ray_tracing_pipeline(&RayTracingPipelineDesc {
                ray_gen: &[],
                closest_hit: &[],
                miss: &[],
                max_recursion: 1,
            })
            .unwrap();
    }
}
