//! `indexed_quad` — the fullscreen quad drawn from four vertices and
//! six indices instead of six vertices: the index-buffer fetch path.

use super::common::{color_target, field_of, quad_pipeline, read_frame};
use crate::Frame;
use quanta::RenderGpu;
use quanta::render_pass::ColorTarget;
use quanta::{FieldUsage, LoadOp, StoreOp};

// The fragment below takes `QuadVary` — see the note in `bands`.
use super::common::__quanta_varyings_QuadVary;

/// The quad's four corners, pos(x,y,z) + uv(u,v).
#[rustfmt::skip]
const QUAD_CORNERS: [f32; 20] = [
    -1.0, -1.0, 0.0,  0.0, 0.0,
     1.0, -1.0, 0.0,  1.0, 0.0,
     1.0,  1.0, 0.0,  1.0, 1.0,
    -1.0,  1.0, 0.0,  0.0, 1.0,
];

/// Two triangles over the four corners.
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 2, 3, 0];

/// Quadrants of the uv square in the red and green channels. A
/// mis-fetched index scrambles which corner reaches which fragment, and
/// the quadrants land somewhere else; a smooth ramp would hide it. The
/// 0.5 boundaries fall between pixel centres (`(c + 0.5) / 64` is never
/// 0.5), so no fragment sits on an edge.
#[quanta::fragment]
fn conform_indexed_quadrants_frag(s: QuadVary) -> Vec4 {
    let r = if s.uv.x > 0.5 { 1.0 } else { 0.0 };
    let g = if s.uv.y > 0.5 { 1.0 } else { 0.0 };
    Vec4::new(r, g, 0.2509804, 1.0)
}

/// `indexed_quad` — `draw_indexed` over an index buffer.
pub fn run(gpu: &quanta::Gpu) -> Result<Frame, quanta::QuantaError> {
    let (w, h) = (64u32, 64u32);
    let pipe = quad_pipeline(
        gpu,
        &CONFORM_INDEXED_QUADRANTS_FRAG_SHADER,
        quanta::BlendState::NONE,
    )?;
    let vb = field_of(gpu, &QUAD_CORNERS)?;
    let ib: quanta::Field<u32> =
        gpu.field_with_usage(QUAD_INDICES.len(), FieldUsage::default_render())?;
    ib.write(&QUAD_INDICES)?;

    let target = color_target(gpu, w, h)?;
    let mut pulse = gpu
        .render(&target)?
        .color_targets(vec![
            ColorTarget::new(&target)
                .with_load_op(LoadOp::Clear(quanta::Color::rgba(0.0, 0.0, 0.0, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&pipe)
        .vertices(0, &vb)
        .indices(&ib)
        .draw_indexed(QUAD_INDICES.len() as u32)
        .pulse()?;
    pulse.wait()?;
    read_frame(&target, w, h)
}
