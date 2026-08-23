//! `depth_two_triangles` — a far triangle drawn first, a near one
//! drawn over it, and a `Less` depth test that has to let the near one
//! win where they overlap.

use super::common::{color_target, field_of, pipeline_desc, pos_layout, pos_vertex, read_frame};
use crate::Frame;
use quanta::RenderGpu;
use quanta::render_pass::{ColorTarget, DepthTarget};
use quanta::{LoadOp, StoreOp};

/// Drawn FIRST at z = 0.8 — the left two thirds of the frame.
#[rustfmt::skip]
const FAR: [f32; 9] = [
    -0.9, -0.8, 0.8,
     0.35, -0.8, 0.8,
    -0.3,  0.8, 0.8,
];

/// Drawn SECOND at z = 0.2 — the right two thirds. The overlap in the
/// middle is the pixels the test is about: they must read the near
/// colour even though the far triangle is not what came last.
#[rustfmt::skip]
const NEAR: [f32; 9] = [
    -0.35, -0.8, 0.2,
     0.9, -0.8, 0.2,
     0.3,  0.8, 0.2,
];

#[quanta::fragment]
fn conform_depth_far_frag() -> Vec4 {
    Vec4::new(0.101960786, 0.6, 0.2509804, 1.0)
}

#[quanta::fragment]
fn conform_depth_near_frag() -> Vec4 {
    Vec4::new(0.9019608, 0.2, 0.101960786, 1.0)
}

fn depth_pipeline(
    gpu: &quanta::Gpu,
    frag: &quanta::ShaderBinary,
    layouts: &[quanta::VertexLayout],
) -> Result<quanta::Pipeline, quanta::QuantaError> {
    gpu.pipeline(
        &pipeline_desc(pos_vertex(), frag, layouts)
            .with_depth_format(quanta::Format::Depth32Float)
            .with_depth_stencil(quanta::DepthStencilState::DEPTH_LESS),
    )
}

/// `depth_two_triangles` — the nearer draw wins on the overlap.
pub fn run(gpu: &quanta::Gpu) -> Result<Frame, quanta::QuantaError> {
    let (w, h) = (64u32, 64u32);
    let layouts = pos_layout();
    let far_pipe = depth_pipeline(gpu, &CONFORM_DEPTH_FAR_FRAG_SHADER, &layouts)?;
    let near_pipe = depth_pipeline(gpu, &CONFORM_DEPTH_NEAR_FRAG_SHADER, &layouts)?;
    let far_vb = field_of(gpu, &FAR)?;
    let near_vb = field_of(gpu, &NEAR)?;

    let target = color_target(gpu, w, h)?;
    let depth = gpu.create_texture(
        &quanta::TextureDesc::new(w, h, quanta::Format::Depth32Float)
            .with_usage(quanta::TextureUsage::RENDER_TARGET),
    )?;

    let mut pulse = gpu
        .render(&target)?
        .color_targets(vec![
            ColorTarget::new(&target)
                .with_load_op(LoadOp::Clear(quanta::Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .depth_target(
            // The clear colour's red channel carries the depth value.
            DepthTarget::new(&depth)
                .with_load_op(LoadOp::Clear(quanta::Color::rgba(1.0, 0.0, 0.0, 0.0)))
                .with_store_op(StoreOp::DontCare)
                .with_stencil_load_op(LoadOp::DontCare)
                .with_stencil_store_op(StoreOp::DontCare),
        )
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&far_pipe)
        .vertices(0, &far_vb)
        .draw(3)
        .pipeline(&near_pipe)
        .vertices(0, &near_vb)
        .draw(3)
        .pulse()?;
    pulse.wait()?;
    read_frame(&target, w, h)
}
