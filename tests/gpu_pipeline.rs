//! Tier 2 -- Render pipeline creation and draw operations.
//!
//! Verifies pipeline_create with various configurations.
//! Uses raw MSL source for Metal backend.
//! Requires a GPU; skips gracefully if none available.

use quanta::{BlendState, CullMode, DepthStencilState, Format, PipelineDesc, Primitive};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

/// Minimal MSL with vertex and fragment shaders.
const MSL_TRIANGLE: &[u8] = b"#include <metal_stdlib>\n\
using namespace metal;\n\
struct V { float4 pos [[position]]; };\n\
vertex V vmain(uint vid [[vertex_id]]) {\n\
    V o;\n\
    float2 p[3] = {float2(0,0.5), float2(-0.5,-0.5), float2(0.5,-0.5)};\n\
    o.pos = float4(p[vid], 0, 1);\n\
    return o;\n\
}\n\
fragment float4 fmain() { return float4(1,0,0,1); }\n";

#[test]
fn pipeline_create_basic() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = PipelineDesc {
        source: Some(MSL_TRIANGLE),
        vertex_entry: "vmain",
        fragment_entry: "fmain",
        color_formats: vec![Format::BGRA8],
        ..PipelineDesc::default()
    };

    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0, "pipeline handle should be nonzero");
}

#[test]
fn pipeline_draw_triangle() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = PipelineDesc {
        source: Some(MSL_TRIANGLE),
        vertex_entry: "vmain",
        fragment_entry: "fmain",
        color_formats: vec![Format::RGBA8],
        ..PipelineDesc::default()
    };

    let pipeline = gpu.pipeline(&desc).unwrap();
    let target = gpu.render_target(64, 64, Format::RGBA8).unwrap();

    let mut pass = gpu.render_begin(&target).unwrap();
    pass.set_pipeline(&pipeline);
    pass.draw(3);

    let mut pulse = gpu.render_end(pass).unwrap();
    gpu.wait(&mut pulse).unwrap();

    // Verify the render target has data.
    let pixels = gpu.texture_read(&target).unwrap();
    assert_eq!(pixels.len(), 64 * 64 * 4);

    // At least some pixels should be non-zero (the red triangle).
    let has_nonzero = pixels.iter().any(|&b| b != 0);
    assert!(
        has_nonzero,
        "render target should have non-zero pixels after draw"
    );
}

#[test]
fn pipeline_blend_none() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = PipelineDesc {
        source: Some(MSL_TRIANGLE),
        vertex_entry: "vmain",
        fragment_entry: "fmain",
        color_formats: vec![Format::RGBA8],
        blend: BlendState::NONE,
        ..PipelineDesc::default()
    };

    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_blend_alpha() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = PipelineDesc {
        source: Some(MSL_TRIANGLE),
        vertex_entry: "vmain",
        fragment_entry: "fmain",
        color_formats: vec![Format::RGBA8],
        blend: BlendState::ALPHA,
        ..PipelineDesc::default()
    };

    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_blend_additive() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = PipelineDesc {
        source: Some(MSL_TRIANGLE),
        vertex_entry: "vmain",
        fragment_entry: "fmain",
        color_formats: vec![Format::RGBA8],
        blend: BlendState::ADDITIVE,
        ..PipelineDesc::default()
    };

    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_cull_back() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = PipelineDesc {
        source: Some(MSL_TRIANGLE),
        vertex_entry: "vmain",
        fragment_entry: "fmain",
        color_formats: vec![Format::RGBA8],
        cull_mode: CullMode::Back,
        ..PipelineDesc::default()
    };

    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_cull_front() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = PipelineDesc {
        source: Some(MSL_TRIANGLE),
        vertex_entry: "vmain",
        fragment_entry: "fmain",
        color_formats: vec![Format::RGBA8],
        cull_mode: CullMode::Front,
        ..PipelineDesc::default()
    };

    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_depth_stencil_less() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = PipelineDesc {
        source: Some(MSL_TRIANGLE),
        vertex_entry: "vmain",
        fragment_entry: "fmain",
        color_formats: vec![Format::RGBA8],
        depth_format: Some(Format::Depth32Float),
        depth_stencil: DepthStencilState::DEPTH_LESS,
        ..PipelineDesc::default()
    };

    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_depth_read_only() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = PipelineDesc {
        source: Some(MSL_TRIANGLE),
        vertex_entry: "vmain",
        fragment_entry: "fmain",
        color_formats: vec![Format::RGBA8],
        depth_format: Some(Format::Depth32Float),
        depth_stencil: DepthStencilState::DEPTH_READ_ONLY,
        ..PipelineDesc::default()
    };

    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_triangle_strip() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = PipelineDesc {
        source: Some(MSL_TRIANGLE),
        vertex_entry: "vmain",
        fragment_entry: "fmain",
        color_formats: vec![Format::RGBA8],
        primitive: Primitive::TriangleStrip,
        ..PipelineDesc::default()
    };

    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_point_topology() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = PipelineDesc {
        source: Some(MSL_TRIANGLE),
        vertex_entry: "vmain",
        fragment_entry: "fmain",
        color_formats: vec![Format::RGBA8],
        primitive: Primitive::Point,
        ..PipelineDesc::default()
    };

    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_rgba16float_target() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = PipelineDesc {
        source: Some(MSL_TRIANGLE),
        vertex_entry: "vmain",
        fragment_entry: "fmain",
        color_formats: vec![Format::RGBA16Float],
        ..PipelineDesc::default()
    };

    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_msaa_4x() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let desc = PipelineDesc {
        source: Some(MSL_TRIANGLE),
        vertex_entry: "vmain",
        fragment_entry: "fmain",
        color_formats: vec![Format::RGBA8],
        sample_count: 4,
        ..PipelineDesc::default()
    };

    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}
