#![cfg(feature = "render")]
//! A Metal pipeline that exceeds Metal's fixed vertex-attribute limit (31)
//! must fail with a clean, NAMED error — never abort the process.
//!
//! Metal's vertex attribute count is fixed at 31 (locations 0..30). Above
//! that, `newRenderPipelineStateWithDescriptor` hard-aborts in-driver (a
//! process abort, no clean error) — the Metal analog of the Broadcom V3D heap
//! corruption the Vulkan pre-check defends against. Quanta pre-validates the
//! attribute budget and returns `CompilationFailed` BEFORE touching the
//! driver, turning that abort into a recoverable error. This test declares a
//! layout with more attributes than Metal allows (32 > 31) and one with a
//! location past the limit (100 > 30), and asserts each returns a clean `Err`
//! with the process still alive. Apple-only (the Vulkan lane has its own
//! twin); a non-Apple backend has a different limit and is skipped.

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

/// A single-buffer layout declaring `n` float attributes at locations `0..n`.
fn attr_layout(n: u32) -> Vec<VertexLayout> {
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

fn build(
    gpu: &quanta::Gpu,
    layouts: &[VertexLayout],
) -> Result<quanta::Pipeline, quanta::QuantaError> {
    gpu.pipeline(
        &quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
            vertex: &TINY_VERTEX_SHADER,
            fragment: &TINY_FRAG_SHADER,
        })
        .with_entries(TINY_VERTEX_SHADER.entry_point, TINY_FRAG_SHADER.entry_point)
        .with_color_formats(vec![Format::RGBA8])
        .with_vertex_layouts(layouts)
        .with_blend(quanta::BlendState::NONE),
    )
}

fn apple_gpu_ready() -> Option<quanta::Gpu> {
    let gpu = try_gpu()?;
    if !matches!(gpu.caps().vendor, quanta::Vendor::Apple) {
        eprintln!("SKIP: Metal-only attribute-limit check (not an Apple backend)");
        return None;
    }
    if TINY_VERTEX_SHADER.for_vendor(gpu.caps().vendor).is_none()
        || TINY_FRAG_SHADER.for_vendor(gpu.caps().vendor).is_none()
    {
        eprintln!("SKIP: no Metal shader binary");
        return None;
    }
    Some(gpu)
}

#[test]
fn metal_over_limit_vertex_attributes_error_cleanly() {
    let Some(gpu) = apple_gpu_ready() else { return };

    // 32 attributes — one past Metal's fixed limit of 31.
    let over = attr_layout(32);
    let err = match build(&gpu, &over) {
        Ok(_) => panic!("Metal accepted 32 vertex attributes (over the fixed limit of 31)"),
        Err(e) => e,
    };
    eprintln!("over-limit pipeline error: {err}");
    match &err.kind {
        QuantaErrorKind::CompilationFailed(msg) => {
            assert!(
                msg.contains("31") && (msg.contains("32") || msg.contains("at most")),
                "the named limit error must report the count and the Metal limit: {msg}"
            );
        }
        other => panic!("expected CompilationFailed for an over-limit pipeline, got {other:?}"),
    }

    // Process is demonstrably alive: a valid pipeline still builds afterward.
    let ok = build(&gpu, &attr_layout(1));
    assert!(
        ok.is_ok(),
        "a valid pipeline must still build after the rejected over-limit one \
         (proves the failed build left the device usable): {:?}",
        ok.err()
    );
}

#[test]
fn metal_over_limit_attribute_location_errs() {
    let Some(gpu) = apple_gpu_ready() else { return };

    // One attribute, but at location 100 — past Metal's limit (valid 0..30).
    let layouts = vec![VertexLayout {
        stride: 4,
        step: StepMode::Vertex,
        attributes: vec![VertexAttribute {
            location: 100,
            offset: 0,
            format: AttributeFormat::Float,
        }],
    }];
    match build(&gpu, &layouts) {
        Ok(_) => panic!("Metal accepted attribute location 100 (over the fixed limit of 31)"),
        Err(e) => {
            eprintln!("over-limit location error: {e}");
            match &e.kind {
                QuantaErrorKind::CompilationFailed(msg) => assert!(
                    msg.contains("100") && msg.contains("31"),
                    "the location error must report the offending location and the limit: {msg}"
                ),
                other => {
                    panic!("expected CompilationFailed for an out-of-range location, got {other:?}")
                }
            }
        }
    }
}
