//! `instanced_triangles` — one small triangle drawn four times from a
//! second vertex buffer stepping per instance.

use super::common::{color_target, field_of, pipeline_desc, read_frame};
use crate::Frame;
use quanta::RenderGpu;
use quanta::render_pass::ColorTarget;
use quanta::{LoadOp, StoreOp};

/// The instanced primitive: a small triangle around the origin.
#[rustfmt::skip]
const TRIANGLE: [f32; 9] = [
     0.0,  0.3, 0.0,
    -0.3, -0.3, 0.0,
     0.3, -0.3, 0.0,
];

/// One offset per instance — the four quadrant centres. A backend that
/// steps the second buffer per vertex instead of per instance draws
/// somewhere else entirely.
#[rustfmt::skip]
const OFFSETS: [f32; 12] = [
    -0.45, -0.45, 0.0,
     0.45, -0.45, 0.0,
    -0.45,  0.45, 0.0,
     0.45,  0.45, 0.0,
];

/// Buffer 0 steps per vertex (the triangle), buffer 1 per instance
/// (the offset).
fn instanced_layouts() -> Vec<quanta::VertexLayout> {
    vec![
        quanta::VertexLayout {
            stride: 12,
            step: quanta::StepMode::Vertex,
            attributes: vec![quanta::VertexAttribute {
                location: 0,
                offset: 0,
                format: quanta::AttributeFormat::Float3,
            }],
        },
        quanta::VertexLayout {
            stride: 12,
            step: quanta::StepMode::Instance,
            attributes: vec![quanta::VertexAttribute {
                location: 1,
                offset: 0,
                format: quanta::AttributeFormat::Float3,
            }],
        },
    ]
}

#[quanta::vertex]
fn conform_instance_offset_vertex(pos: Vec3, offset: Vec3) -> Vec4 {
    Vec4::new(pos.x + offset.x, pos.y + offset.y, 0.0, 1.0)
}

#[quanta::fragment]
fn conform_instance_frag() -> Vec4 {
    Vec4::new(0.2509804, 0.8, 0.4, 1.0)
}

/// `instanced_triangles` — four instances, one draw call.
pub fn run(gpu: &quanta::Gpu) -> Result<Frame, quanta::QuantaError> {
    let (w, h) = (64u32, 64u32);
    let layouts = instanced_layouts();
    let pipe = gpu.pipeline(&pipeline_desc(
        &CONFORM_INSTANCE_OFFSET_VERTEX_SHADER,
        &CONFORM_INSTANCE_FRAG_SHADER,
        &layouts,
    ))?;
    let vb = field_of(gpu, &TRIANGLE)?;
    let instances = field_of(gpu, &OFFSETS)?;

    let target = color_target(gpu, w, h)?;
    let mut pulse = gpu
        .render(&target)?
        .color_targets(vec![
            ColorTarget::new(&target)
                .with_load_op(LoadOp::Clear(quanta::Color::rgba(0.05, 0.05, 0.1, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&pipe)
        .vertices(0, &vb)
        .vertices(1, &instances)
        .draw_instanced(3, 4)
        .pulse()?;
    pulse.wait()?;
    read_frame(&target, w, h)
}
