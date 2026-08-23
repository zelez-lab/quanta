//! `viewport_scissor` — a fullscreen-quad draw restricted to the
//! frame's central quarter by the scissor rectangle.

use super::common::{color_target, fullscreen_vb, quad_pipeline, read_frame};
use crate::Frame;
use quanta::RenderGpu;
use quanta::render_pass::ColorTarget;
use quanta::{LoadOp, StoreOp};

// The fragment below takes `QuadVary` — see the note in `bands`.
use super::common::__quanta_varyings_QuadVary;

/// A ramp, not a flat fill: the scissor must clip the draw without
/// moving it, so the surviving quarter has to carry the same uv values
/// it would have carried unclipped.
#[quanta::fragment]
fn conform_scissor_ramp_frag(s: QuadVary) -> Vec4 {
    Vec4::new(1.0 - s.uv.x, s.uv.y, 0.5, 1.0)
}

/// `viewport_scissor` — the viewport is the whole frame (as in every
/// other case); the scissor is what restricts the draw. The rectangle
/// is centred, so a backend that measured scissor Y from the other
/// edge would still land on the same rows — this case pins the clip,
/// and `orientation_bands` pins the vertical convention.
pub fn run(gpu: &quanta::Gpu) -> Result<Frame, quanta::QuantaError> {
    let (w, h) = (64u32, 64u32);
    let pipe = quad_pipeline(
        gpu,
        &CONFORM_SCISSOR_RAMP_FRAG_SHADER,
        quanta::BlendState::NONE,
    )?;
    let vb = fullscreen_vb(gpu)?;
    let target = color_target(gpu, w, h)?;
    let mut pulse = gpu
        .render(&target)?
        .color_targets(vec![
            ColorTarget::new(&target)
                .with_load_op(LoadOp::Clear(quanta::Color::rgba(0.0, 0.0, 0.2, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .viewport(0.0, 0.0, w as f32, h as f32)
        .scissor(w / 4, h / 4, w / 2, h / 2)
        .pipeline(&pipe)
        .vertices(0, &vb)
        .draw(6)
        .pulse()?;
    pulse.wait()?;
    read_frame(&target, w, h)
}
