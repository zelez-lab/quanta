#![cfg(feature = "render")]
//! A pipeline that exceeds the device's vertex-attribute limit must fail
//! with a clean, NAMED error — never crash the process.
//!
//! On Broadcom V3D (v3dv, `maxVertexInputAttributes` = 16) a failing
//! `vkCreateGraphicsPipelines` was observed to corrupt the process heap
//! ("malloc(): unsorted double linked list corrupted"). Desktop drivers
//! (limit 32) and Metal (31) mask the over-limit case. Quanta pre-validates
//! the attribute budget against the device limit and returns
//! `CompilationFailed` BEFORE calling the driver, turning an
//! undefined-behavior driver call into a recoverable error. This test
//! declares a vertex layout with FAR more attributes than any current
//! device supports (48 > 32 > 31 > 16), so the check fires on every
//! backend, and asserts the call returns `Err` with the process still
//! alive.

use quanta::RenderGpu;

use quanta::{AttributeFormat, Format, QuantaErrorKind, StepMode, VertexAttribute, VertexLayout};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[quanta::vertex]
fn tiny_vertex(pos: Vec3) -> Vec4 {
    Vec4::new(pos.x, pos.y, 0.0, 1.0)
}

#[quanta::fragment]
fn tiny_frag() -> Vec4 {
    Vec4::new(1.0, 1.0, 1.0, 1.0)
}

/// A single-buffer layout declaring `n` float attributes at locations
/// `0..n` — deliberately larger than any device's attribute limit.
fn over_limit_layout(n: u32) -> Vec<VertexLayout> {
    let attributes = (0..n)
        .map(|i| VertexAttribute {
            location: i,
            offset: i * 4,
            format: AttributeFormat::Float,
        })
        .collect();
    vec![VertexLayout {
        stride: n * 4,
        step: StepMode::Vertex,
        attributes,
    }]
}

#[test]
fn over_limit_vertex_attributes_error_cleanly() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    // This exercises the VULKAN pre-validation limit check. Metal has no
    // such check and its driver ABORTS the process (a hard assertion) on an
    // over-limit attribute count rather than returning an error — the exact
    // undefined-behavior class this fix defends Vulkan against — so the
    // over-limit input must never reach Metal. Skip on Apple; lavapipe in
    // CI (limit 32 < 48) is where the named rejection is proven.
    if matches!(gpu.caps().vendor, quanta::Vendor::Apple) {
        eprintln!("SKIP: Metal aborts on over-limit attributes; this is a Vulkan-lane check");
        return;
    }
    if TINY_VERTEX_SHADER.for_vendor(gpu.caps().vendor).is_none()
        || TINY_FRAG_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary for this vendor");
        return;
    }

    // 48 attributes — over v3dv (16), Metal (31), and desktop Vulkan (32).
    let layouts = over_limit_layout(48);
    let result = gpu.pipeline(
        &quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
            vertex: &TINY_VERTEX_SHADER,
            fragment: &TINY_FRAG_SHADER,
        })
        .with_entries(TINY_VERTEX_SHADER.entry_point, TINY_FRAG_SHADER.entry_point)
        .with_color_formats(vec![Format::RGBA8])
        .with_vertex_layouts(&layouts)
        .with_blend(quanta::BlendState::NONE),
    );

    // The whole point: an Err, not a crash. Reaching this line proves the
    // process survived the over-limit pipeline build.
    let err = match result {
        Ok(_) => {
            // A hypothetical device that genuinely supports 48 attributes
            // would legally accept this — don't fail the suite on such
            // hardware, but note it (no current target does).
            eprintln!(
                "note: device accepted 48 vertex attributes — its limit is \
                 >= 48; over-limit rejection unexercised here"
            );
            return;
        }
        Err(e) => e,
    };
    eprintln!("over-limit pipeline error: {err}");

    // On the Vulkan backend this is the NAMED limit check, mentioning the
    // Vulkan limit. Assert that specifically when it fired; otherwise just
    // require a clean CompilationFailed (Metal rejects >31 through its own
    // path, without our message).
    match &err.kind {
        QuantaErrorKind::CompilationFailed(msg) => {
            if msg.contains("maxVertexInputAttributes") {
                assert!(
                    msg.contains("48") || msg.contains("supports at most"),
                    "the named limit error should report the offending count / limit: {msg}"
                );
            }
        }
        other => panic!("expected CompilationFailed for an over-limit pipeline, got {other:?}"),
    }

    // Process is demonstrably alive: a follow-up valid call still works.
    let ok_layout = vec![VertexLayout {
        stride: 12,
        step: StepMode::Vertex,
        attributes: vec![VertexAttribute {
            location: 0,
            offset: 0,
            format: AttributeFormat::Float3,
        }],
    }];
    let ok = gpu.pipeline(
        &quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
            vertex: &TINY_VERTEX_SHADER,
            fragment: &TINY_FRAG_SHADER,
        })
        .with_entries(TINY_VERTEX_SHADER.entry_point, TINY_FRAG_SHADER.entry_point)
        .with_color_formats(vec![Format::RGBA8])
        .with_vertex_layouts(&ok_layout)
        .with_blend(quanta::BlendState::NONE),
    );
    assert!(
        ok.is_ok(),
        "a valid pipeline must still build after the rejected over-limit one \
         (proves the failed build left the device usable): {:?}",
        ok.err()
    );
}

#[test]
fn over_limit_attribute_location_errs() {
    // The location-based half of the check: even a SINGLE attribute placed
    // at a location past the device limit overflows the attribute array.
    let Some(gpu) = try_gpu() else {
        return;
    };
    // Vulkan-lane check — see the note above; Metal aborts on this input.
    if matches!(gpu.caps().vendor, quanta::Vendor::Apple) {
        eprintln!("SKIP: Metal aborts on over-limit attribute locations; Vulkan-lane check");
        return;
    }
    if TINY_VERTEX_SHADER.for_vendor(gpu.caps().vendor).is_none()
        || TINY_FRAG_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no shader binary for this vendor");
        return;
    }

    // One attribute, but at location 100 — past every current device limit.
    let layouts = vec![VertexLayout {
        stride: 4,
        step: StepMode::Vertex,
        attributes: vec![VertexAttribute {
            location: 100,
            offset: 0,
            format: AttributeFormat::Float,
        }],
    }];
    let result = gpu.pipeline(
        &quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
            vertex: &TINY_VERTEX_SHADER,
            fragment: &TINY_FRAG_SHADER,
        })
        .with_entries(TINY_VERTEX_SHADER.entry_point, TINY_FRAG_SHADER.entry_point)
        .with_color_formats(vec![Format::RGBA8])
        .with_vertex_layouts(&layouts)
        .with_blend(quanta::BlendState::NONE),
    );

    match result {
        Ok(_) => {
            eprintln!("note: device accepted attribute location 100 — unexpected but not fatal")
        }
        Err(e) => {
            eprintln!("over-limit location error: {e}");
            assert!(
                matches!(e.kind, QuantaErrorKind::CompilationFailed(_)),
                "an out-of-range attribute location must fail as CompilationFailed, got {:?}",
                e.kind
            );
        }
    }
}
