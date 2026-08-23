//! Fullscreen-quad cases: the interpolation path with no interior
//! primitive edge that matters (the quad's diagonal carries a
//! continuous varying, so both triangles agree along it).

use super::draw_frame;
use crate::{Case, Frame, Tolerance};
use quanta::RenderGpu;
use quanta::{FieldUsage, Vec2, Vec4};

/// pos(x,y,z) + uv(u,v), two triangles covering clip space.
#[rustfmt::skip]
const FULLSCREEN_QUAD: [f32; 30] = [
    -1.0, -1.0, 0.0,  0.0, 0.0,
     1.0, -1.0, 0.0,  1.0, 0.0,
     1.0,  1.0, 0.0,  1.0, 1.0,
    -1.0, -1.0, 0.0,  0.0, 0.0,
     1.0,  1.0, 0.0,  1.0, 1.0,
    -1.0,  1.0, 0.0,  0.0, 1.0,
];

/// The shared vertex→fragment interface: clip position + interpolated uv.
#[derive(quanta::Varyings)]
struct QuadVary {
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

/// The interpolation ramp itself: uv into the red/green channels, a
/// constant blue so a dropped channel is visible.
#[quanta::fragment]
fn conform_uv_gradient_frag(s: QuadVary) -> Vec4 {
    Vec4::new(s.uv.x, s.uv.y, 0.25, 1.0)
}

fn vertex_layout() -> Vec<quanta::VertexLayout> {
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

fn quad_pipeline(
    gpu: &quanta::Gpu,
    frag: &quanta::ShaderBinary,
) -> Result<quanta::Pipeline, quanta::QuantaError> {
    let layouts = vertex_layout();
    gpu.pipeline(
        &quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
            vertex: &CONFORM_QUAD_VERTEX_SHADER,
            fragment: frag,
        })
        .with_entries(CONFORM_QUAD_VERTEX_SHADER.entry_point, frag.entry_point)
        .with_color_formats(vec![quanta::Format::RGBA8])
        .with_vertex_layouts(&layouts)
        .with_blend(quanta::BlendState::NONE),
    )
}

fn fullscreen_vb(gpu: &quanta::Gpu) -> Result<quanta::Field<f32>, quanta::QuantaError> {
    let vb: quanta::Field<f32> =
        gpu.field_with_usage(FULLSCREEN_QUAD.len(), FieldUsage::default_render())?;
    vb.write(&FULLSCREEN_QUAD)?;
    Ok(vb)
}

/// `uv_gradient_quad` — the full-frame interpolation ramp.
pub fn run_uv_gradient(gpu: &quanta::Gpu) -> Result<Frame, quanta::QuantaError> {
    let pipe = quad_pipeline(gpu, &CONFORM_UV_GRADIENT_FRAG_SHADER)?;
    let vb = fullscreen_vb(gpu)?;
    draw_frame(
        gpu,
        &pipe,
        &vb,
        6,
        64,
        64,
        quanta::Color::rgba(0.0, 0.0, 0.0, 1.0),
    )
}

/// Further quad-family cases (the corpus grows here).
pub fn extra() -> Vec<Case> {
    let _ = Tolerance::EXACT;
    Vec::new()
}
