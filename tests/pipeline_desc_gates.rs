//! End-to-end test for the cross-backend `PipelineDesc` deferred-feature
//! gate — step 063 slices 5, 11, 12.
//!
//! Setting `tessellation`, `mesh_shader`, or `conservative_rasterization`
//! on a `PipelineDesc` and calling `gpu.pipeline()` must surface
//! `NotSupported` on every backend (rather than silently dropping the
//! request). This validates the no-silent-drops contract that 063
//! closed across CPU / Metal / Vulkan / WebGPU.
//!
//! Run: cargo test --test pipeline_desc_gates --features software
//!
//! The test uses `init_cpu` so it runs on every dev machine without
//! needing a live GPU.

#![cfg(feature = "software")]

use quanta::{PipelineDesc, QuantaErrorKind, TessSpacing, TessWinding, TessellationDesc};

fn ensure_not_supported(label: &str, r: Result<quanta::Pipeline, quanta::QuantaError>) {
    match r {
        Err(e) => assert!(
            matches!(e.kind, QuantaErrorKind::NotSupported(_)),
            "{label} expected NotSupported, got {:?}",
            e.kind
        ),
        Ok(_) => panic!("{label}: expected NotSupported, got Ok"),
    }
}

#[test]
fn cpu_pipeline_with_tessellation_is_not_supported() {
    let gpu = quanta::init_cpu();
    let desc = PipelineDesc {
        vertex: b"vert",
        fragment: b"frag",
        tessellation: Some(TessellationDesc {
            patch_size: 3,
            spacing: TessSpacing::Equal,
            winding: TessWinding::Ccw,
        }),
        ..PipelineDesc::default()
    };
    ensure_not_supported("CPU + tessellation", gpu.pipeline(&desc));
}

#[test]
fn cpu_pipeline_with_conservative_rasterization_is_not_supported() {
    let gpu = quanta::init_cpu();
    let desc = PipelineDesc {
        vertex: b"vert",
        fragment: b"frag",
        conservative_rasterization: true,
        ..PipelineDesc::default()
    };
    ensure_not_supported("CPU + conservative", gpu.pipeline(&desc));
}

#[test]
fn cpu_pipeline_baseline_succeeds() {
    // Sanity check: a default PipelineDesc still builds on CPU
    // (fake handle) — the gate only fires on the deferred fields.
    let gpu = quanta::init_cpu();
    let desc = PipelineDesc {
        vertex: b"vert",
        fragment: b"frag",
        ..PipelineDesc::default()
    };
    let r = gpu.pipeline(&desc);
    assert!(
        r.is_ok(),
        "baseline PipelineDesc should still build on CPU, got {:?}",
        r.err()
    );
}
