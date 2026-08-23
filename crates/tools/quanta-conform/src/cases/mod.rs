//! The conformance corpus. Every case draws with the public API only,
//! offscreen, RGBA8, row 0 = top (the orientation contract), and
//! returns the readback as a [`Frame`].
//!
//! Template notes for new cases:
//! * 64×64 target unless the case is about size.
//! * Choose [`Tolerance::EXACT`] when no primitive edge crosses the
//!   frame interior (fullscreen quads, clears); [`Tolerance::EDGED`]
//!   when one does.
//! * Shaders are `#[quanta::vertex]` / `#[quanta::fragment]` items at
//!   module scope, one module per case family.

use crate::{Case, Frame, Tolerance};

mod clear;
mod quad;

/// The corpus, in matrix row order.
pub fn all() -> Vec<Case> {
    let mut v = vec![
        Case {
            name: "clear_only",
            tolerance: Tolerance::EXACT,
            run: clear::run,
        },
        Case {
            name: "uv_gradient_quad",
            tolerance: Tolerance::EXACT,
            run: quad::run_uv_gradient,
        },
    ];
    v.extend(quad::extra());
    v
}

/// Draw-and-read plumbing shared by every case: render `draw_count`
/// vertices of `vb` through `pipe` into a fresh `w`×`h` RGBA8 target
/// cleared to `clear`, and read it back.
pub(crate) fn draw_frame(
    gpu: &quanta::Gpu,
    pipe: &quanta::Pipeline,
    vb: &quanta::Field<f32>,
    draw_count: u32,
    w: u32,
    h: u32,
    clear: quanta::Color,
) -> Result<Frame, quanta::QuantaError> {
    use quanta::RenderGpu;
    use quanta::render_pass::ColorTarget;
    use quanta::{LoadOp, StoreOp};
    let target = gpu.render_target(w, h, quanta::Format::RGBA8)?;
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
    let bytes = target.read()?;
    Frame::rgba8(w, h, bytes).map_err(quanta::QuantaError::invalid_param)
}
