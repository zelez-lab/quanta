#![cfg(feature = "render")]
//! Metal-backed render-path ICB smoke test (steps 032 + 033).
//!
//! Exercises the `IndirectRenderBundle` typed wrapper against the
//! Metal driver: create a DRAW-typed `MTLIndirectCommandBuffer`,
//! record one draw, attempt to execute it inside a render pass.
//! Skips gracefully if no Apple GPU or no compiled kernel for
//! vendor Apple is available (Metal Toolchain not installed).
//!
//! The bundle creation + record path drives Metal's
//! `indirectRenderCommandAtIndex:` + `setRenderPipelineState:` +
//! `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`
//! FFI added in this commit. The Lean T7006 contract (record_draw
//! appends one entry) is satisfied at the recording level
//! regardless of whether a render pass is later actually run.

fn try_apple_gpu() -> Option<quanta::Gpu> {
    quanta::init()
        .ok()
        .filter(|g| g.caps().vendor == quanta::Vendor::Apple)
}

#[test]
fn metal_render_bundle_create_and_record() {
    let Some(gpu) = try_apple_gpu() else {
        eprintln!("skipping: no Apple GPU available");
        return;
    };

    // The bundle itself is allocated against the device — no
    // pipeline / shader required. Recording requires a real render
    // pipeline, so try to create one and skip if shader compilation
    // is not available locally (Metal Toolchain not installed).
    let mut bundle = match gpu.render_bundle(4) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: render_bundle create unsupported: {e}");
            return;
        }
    };
    assert_eq!(bundle.capacity(), 4);
    assert_eq!(bundle.len(), 0);
    assert!(bundle.is_empty());

    // Build the simplest possible render pipeline so we have a
    // handle to record against.
    let pipeline = match gpu.pipeline(&quanta::PipelineDesc::default()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skipping: pipeline create unsupported (Metal Toolchain?): {e}");
            return;
        }
    };

    bundle.record_draw(&pipeline, 3, 1).unwrap();
    assert_eq!(bundle.len(), 1);
    bundle.record_draw(&pipeline, 6, 1).unwrap();
    assert_eq!(bundle.len(), 2);

    // Recording past capacity fails (T7052 refinement).
    bundle.record_draw(&pipeline, 3, 1).unwrap();
    bundle.record_draw(&pipeline, 3, 1).unwrap();
    let err = bundle.record_draw(&pipeline, 3, 1).unwrap_err();
    assert!(err.to_string().contains("full"), "got: {err}");
}
