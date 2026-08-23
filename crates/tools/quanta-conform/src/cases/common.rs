//! Plumbing more than one family needs: the two stock geometries and
//! the vertex stages that consume them, the pipeline descriptor shape,
//! and the draw-and-read pair that turns a pass into a [`Frame`].
//!
//! Fragments live in the family modules — a fragment IS the case. Only
//! what several families share sits here.
//!
//! The module is private on purpose: the shader macros emit `pub`
//! statics, and `cases` is a public module, so a shader declared one
//! level up would put an undocumented `pub static` on the crate's
//! surface.

use crate::Frame;
use quanta::RenderGpu;
use quanta::render_pass::ColorTarget;
use quanta::{FieldUsage, LoadOp, StoreOp};
use quanta::{Vec2, Vec4};

/// pos(x,y,z) + uv(u,v), two triangles covering clip space. The
/// diagonal is the only interior edge and every fragment over it reads
/// a continuous varying, so both triangles agree along it.
#[rustfmt::skip]
pub(crate) const FULLSCREEN_QUAD: [f32; 30] = [
    -1.0, -1.0, 0.0,  0.0, 0.0,
     1.0, -1.0, 0.0,  1.0, 0.0,
     1.0,  1.0, 0.0,  1.0, 1.0,
    -1.0, -1.0, 0.0,  0.0, 0.0,
     1.0,  1.0, 0.0,  1.0, 1.0,
    -1.0,  1.0, 0.0,  0.0, 1.0,
];

/// The shared vertex→fragment interface: clip position + interpolated uv.
#[derive(quanta::Varyings)]
pub(crate) struct QuadVary {
    #[position]
    clip: Vec4,
    uv: Vec2,
}

/// Pass-through vertex: clip position straight from the buffer, uv
/// interpolates across the target.
#[quanta::vertex]
fn conform_quad_vertex(pos: Vec3, uv: Vec2) -> QuadVary {
    QuadVary {
        clip: Vec4::new(pos.x, pos.y, 0.0, 1.0),
        uv,
    }
}

/// Position-only vertex for the geometry families. `z` passes through
/// so a depth-tested pass can order two draws by it.
#[quanta::vertex]
fn conform_pos_vertex(pos: Vec3) -> Vec4 {
    Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

/// The [`FULLSCREEN_QUAD`] layout: position at 0, uv at 12, one buffer.
pub(crate) fn pos_uv_layout() -> Vec<quanta::VertexLayout> {
    vec![quanta::VertexLayout {
        stride: 20,
        step: quanta::StepMode::Vertex,
        attributes: vec![
            quanta::VertexAttribute {
                location: 0,
                offset: 0,
                format: quanta::AttributeFormat::Float3,
            },
            quanta::VertexAttribute {
                location: 1,
                offset: 12,
                format: quanta::AttributeFormat::Float2,
            },
        ],
    }]
}

/// A position-only vertex buffer: three floats, one buffer.
pub(crate) fn pos_layout() -> Vec<quanta::VertexLayout> {
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

/// The descriptor shape every case starts from: two stages, one RGBA8
/// color attachment, `layouts` as the vertex input, no blending. A case
/// that needs depth, MSAA, or blending adds it with the `with_*`
/// builders.
pub(crate) fn pipeline_desc<'a>(
    vert: &'a quanta::ShaderBinary,
    frag: &'a quanta::ShaderBinary,
    layouts: &'a [quanta::VertexLayout],
) -> quanta::PipelineDesc<'a> {
    quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
        vertex: vert,
        fragment: frag,
    })
    .with_entries(vert.entry_point, frag.entry_point)
    .with_color_formats(vec![quanta::Format::RGBA8])
    .with_vertex_layouts(layouts)
    .with_blend(quanta::BlendState::NONE)
}

/// A fullscreen-quad pipeline: the shared uv vertex stage in front of
/// `frag`, under `blend`.
pub(crate) fn quad_pipeline(
    gpu: &quanta::Gpu,
    frag: &quanta::ShaderBinary,
    blend: quanta::BlendState,
) -> Result<quanta::Pipeline, quanta::QuantaError> {
    let layouts = pos_uv_layout();
    gpu.pipeline(&pipeline_desc(&CONFORM_QUAD_VERTEX_SHADER, frag, &layouts).with_blend(blend))
}

/// A pipeline over `layouts` pairing the position-only vertex stage
/// with `frag`.
pub(crate) fn pos_pipeline(
    gpu: &quanta::Gpu,
    frag: &quanta::ShaderBinary,
    layouts: &[quanta::VertexLayout],
) -> Result<quanta::Pipeline, quanta::QuantaError> {
    gpu.pipeline(&pipeline_desc(&CONFORM_POS_VERTEX_SHADER, frag, layouts))
}

/// The position-only vertex stage, for the cases that build their own
/// descriptor (depth, MSAA).
pub(crate) fn pos_vertex() -> &'static quanta::ShaderBinary {
    &CONFORM_POS_VERTEX_SHADER
}

/// Upload `data` as a render-usable float field.
pub(crate) fn field_of(
    gpu: &quanta::Gpu,
    data: &[f32],
) -> Result<quanta::Field<f32>, quanta::QuantaError> {
    let f: quanta::Field<f32> = gpu.field_with_usage(data.len(), FieldUsage::default_render())?;
    f.write(data)?;
    Ok(f)
}

/// The [`FULLSCREEN_QUAD`] as a vertex buffer.
pub(crate) fn fullscreen_vb(gpu: &quanta::Gpu) -> Result<quanta::Field<f32>, quanta::QuantaError> {
    field_of(gpu, &FULLSCREEN_QUAD)
}

/// A fresh `w`×`h` RGBA8 offscreen target.
pub(crate) fn color_target(
    gpu: &quanta::Gpu,
    w: u32,
    h: u32,
) -> Result<quanta::Texture, quanta::QuantaError> {
    gpu.render_target(w, h, quanta::Format::RGBA8)
}

/// Read a drawn target back as the case's [`Frame`].
pub(crate) fn read_frame(
    target: &quanta::Texture,
    w: u32,
    h: u32,
) -> Result<Frame, quanta::QuantaError> {
    let bytes = target.read()?;
    Frame::rgba8(w, h, bytes).map_err(quanta::QuantaError::invalid_param)
}

/// Render `draw_count` vertices of `vb` through `pipe` into a fresh
/// `w`×`h` RGBA8 target cleared to `clear`, and read it back.
pub(crate) fn draw_frame(
    gpu: &quanta::Gpu,
    pipe: &quanta::Pipeline,
    vb: &quanta::Field<f32>,
    draw_count: u32,
    w: u32,
    h: u32,
    clear: quanta::Color,
) -> Result<Frame, quanta::QuantaError> {
    let target = color_target(gpu, w, h)?;
    let mut pulse = gpu
        .render(&target)?
        .color_targets(vec![
            ColorTarget::new(&target)
                .with_load_op(LoadOp::Clear(clear))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(pipe)
        .vertices(0, vb)
        .draw(draw_count)
        .pulse()?;
    pulse.wait()?;
    read_frame(&target, w, h)
}
