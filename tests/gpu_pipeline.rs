#![cfg(feature = "render")]
//! Tier 2 -- Render pipeline creation and draw operations.
//!
//! Verifies pipeline_create with various configurations.
//! Uses macro-compiled shaders (SPIR-V + metallib) for cross-platform support.
//! Requires a GPU; skips gracefully if none available.

use quanta::{
    BlendState, CullMode, DepthStencilState, Format, PipelineDesc, Primitive, ShaderSource,
};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[quanta::vertex]
fn pipe_vertex(pos: Vec3) -> Vec4 {
    Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

#[quanta::fragment]
fn pipe_fragment() -> Vec4 {
    Vec4::new(1.0, 0.0, 0.0, 1.0)
}

fn shaders() -> ShaderSource<'static> {
    // The driver picks the right per-vendor payload from the binaries.
    ShaderSource::Binaries {
        vertex: &PIPE_VERTEX_SHADER,
        fragment: &PIPE_FRAGMENT_SHADER,
    }
}

fn vertex_layout() -> Vec<quanta::VertexLayout> {
    vec![quanta::VertexLayout {
        stride: 12,
        step: quanta::StepMode::Vertex,
        attributes: vec![quanta::VertexAttribute {
            location: 0,
            offset: 0,
            format: quanta::AttributeFormat::Float3,
        }],
    }]
}

fn base_desc<'a>(layouts: &'a [quanta::VertexLayout]) -> PipelineDesc<'a> {
    PipelineDesc::new(shaders())
        .with_entries(
            PIPE_VERTEX_SHADER.entry_point,
            PIPE_FRAGMENT_SHADER.entry_point,
        )
        .with_vertex_layouts(layouts)
        .with_color_formats(vec![Format::RGBA8])
        .with_blend(BlendState::NONE)
}

#[test]
fn pipeline_create_basic() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let layouts = vertex_layout();
    let pipeline = gpu.pipeline(&base_desc(&layouts)).unwrap();
    assert!(pipeline.handle() != 0, "pipeline handle should be nonzero");
}

#[test]
fn pipeline_draw_triangle() {
    let Some(gpu) = try_gpu() else {
        return;
    };

    let layouts = vertex_layout();
    let pipeline = gpu.pipeline(&base_desc(&layouts)).unwrap();

    let verts: [f32; 9] = [0.0, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), quanta::FieldUsage::default_render())
        .unwrap();
    vb.write(&verts).unwrap();

    let target = gpu.render_target(64, 64, Format::RGBA8).unwrap();
    let mut pulse = gpu
        .render(&target)
        .unwrap()
        .viewport(0.0, 0.0, 64.0, 64.0)
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw(3)
        .pulse()
        .unwrap();
    pulse.wait().unwrap();

    let pixels = target.read().unwrap();
    assert_eq!(pixels.len(), 64 * 64 * 4);
    let has_nonzero = pixels.iter().any(|&b| b != 0);
    assert!(
        has_nonzero,
        "render target should have non-zero pixels after draw"
    );
}

#[test]
fn pipeline_blend_none() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(&layouts);
    desc.blend = BlendState::NONE;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_blend_alpha() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(&layouts);
    desc.blend = BlendState::ALPHA;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_blend_additive() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(&layouts);
    desc.blend = BlendState::ADDITIVE;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_cull_back() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(&layouts);
    desc.cull_mode = CullMode::Back;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_cull_front() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(&layouts);
    desc.cull_mode = CullMode::Front;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_depth_stencil_less() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(&layouts);
    desc.depth_format = Some(Format::Depth32Float);
    desc.depth_stencil = DepthStencilState::DEPTH_LESS;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_depth_read_only() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(&layouts);
    desc.depth_format = Some(Format::Depth32Float);
    desc.depth_stencil = DepthStencilState::DEPTH_READ_ONLY;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_triangle_strip() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(&layouts);
    desc.primitive = Primitive::TriangleStrip;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_point_topology() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(&layouts);
    desc.primitive = Primitive::Point;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_rgba16float_target() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(&layouts);
    desc.color_formats = vec![Format::RGBA16Float];
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_msaa_4x() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(&layouts);
    desc.sample_count = 4;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}
