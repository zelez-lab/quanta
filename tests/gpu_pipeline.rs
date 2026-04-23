//! Tier 2 -- Render pipeline creation and draw operations.
//!
//! Verifies pipeline_create with various configurations.
//! Uses macro-compiled shaders (SPIR-V + metallib) for cross-platform support.
//! Requires a GPU; skips gracefully if none available.

use quanta::{BlendState, CullMode, DepthStencilState, Format, PipelineDesc, Primitive};

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

fn shader_desc(gpu: &quanta::Gpu) -> Option<(&'static [u8], &'static [u8])> {
    let v = PIPE_VERTEX_SHADER.for_vendor(gpu.caps().vendor)?;
    let f = PIPE_FRAGMENT_SHADER.for_vendor(gpu.caps().vendor)?;
    Some((v, f))
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

fn base_desc<'a>(
    vert: &'a [u8],
    frag: &'a [u8],
    layouts: &'a [quanta::VertexLayout],
) -> PipelineDesc<'a> {
    PipelineDesc {
        vertex: vert,
        fragment: frag,
        vertex_entry: PIPE_VERTEX_SHADER.entry_point,
        fragment_entry: PIPE_FRAGMENT_SHADER.entry_point,
        vertex_layouts: layouts,
        color_formats: vec![Format::RGBA8],
        blend: BlendState::NONE,
        ..PipelineDesc::default()
    }
}

#[test]
fn pipeline_create_basic() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let Some((v, f)) = shader_desc(&gpu) else {
        return;
    };
    let layouts = vertex_layout();
    let pipeline = gpu.pipeline(&base_desc(v, f, &layouts)).unwrap();
    assert!(pipeline.handle() != 0, "pipeline handle should be nonzero");
}

#[test]
fn pipeline_draw_triangle() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let Some((v, f)) = shader_desc(&gpu) else {
        return;
    };

    let layouts = vertex_layout();
    let pipeline = gpu.pipeline(&base_desc(v, f, &layouts)).unwrap();

    let verts: [f32; 9] = [0.0, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0];
    let vb: quanta::Field<f32> = gpu.render_field(verts.len()).unwrap();
    gpu.write_field(&vb, &verts).unwrap();

    let target = gpu.render_target(64, 64, Format::RGBA8).unwrap();
    let mut pass = gpu.render_begin(&target).unwrap();
    pass.set_viewport(0.0, 0.0, 64.0, 64.0);
    pass.set_pipeline(&pipeline);
    pass.bind_vertices(0, &vb);
    pass.draw(3);
    let mut pulse = gpu.render_end(pass).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let pixels = gpu.texture_read(&target).unwrap();
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
    let Some((v, f)) = shader_desc(&gpu) else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(v, f, &layouts);
    desc.blend = BlendState::NONE;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_blend_alpha() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let Some((v, f)) = shader_desc(&gpu) else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(v, f, &layouts);
    desc.blend = BlendState::ALPHA;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_blend_additive() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let Some((v, f)) = shader_desc(&gpu) else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(v, f, &layouts);
    desc.blend = BlendState::ADDITIVE;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_cull_back() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let Some((v, f)) = shader_desc(&gpu) else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(v, f, &layouts);
    desc.cull_mode = CullMode::Back;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_cull_front() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let Some((v, f)) = shader_desc(&gpu) else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(v, f, &layouts);
    desc.cull_mode = CullMode::Front;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_depth_stencil_less() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let Some((v, f)) = shader_desc(&gpu) else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(v, f, &layouts);
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
    let Some((v, f)) = shader_desc(&gpu) else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(v, f, &layouts);
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
    let Some((v, f)) = shader_desc(&gpu) else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(v, f, &layouts);
    desc.primitive = Primitive::TriangleStrip;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_point_topology() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let Some((v, f)) = shader_desc(&gpu) else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(v, f, &layouts);
    desc.primitive = Primitive::Point;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_rgba16float_target() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let Some((v, f)) = shader_desc(&gpu) else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(v, f, &layouts);
    desc.color_formats = vec![Format::RGBA16Float];
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}

#[test]
fn pipeline_msaa_4x() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    let Some((v, f)) = shader_desc(&gpu) else {
        return;
    };
    let layouts = vertex_layout();
    let mut desc = base_desc(v, f, &layouts);
    desc.sample_count = 4;
    let pipeline = gpu.pipeline(&desc).unwrap();
    assert!(pipeline.handle() != 0);
}
