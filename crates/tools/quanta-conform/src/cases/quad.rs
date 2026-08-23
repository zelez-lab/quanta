//! Fullscreen-quad interpolation: the varying path with no interior
//! primitive edge that matters (the quad's diagonal carries a
//! continuous varying, so both triangles agree along it).

use super::common::{draw_frame, fullscreen_vb, quad_pipeline};
use crate::Frame;

// The fragment below takes `QuadVary` — see the note in `bands`.
use super::common::__quanta_varyings_QuadVary;

/// The interpolation ramp itself: uv into the red/green channels, a
/// constant blue so a dropped channel is visible.
#[quanta::fragment]
fn conform_uv_gradient_frag(s: QuadVary) -> Vec4 {
    Vec4::new(s.uv.x, s.uv.y, 0.25, 1.0)
}

/// `uv_gradient_quad` — the full-frame interpolation ramp.
pub fn run_uv_gradient(gpu: &quanta::Gpu) -> Result<Frame, quanta::QuantaError> {
    let pipe = quad_pipeline(
        gpu,
        &CONFORM_UV_GRADIENT_FRAG_SHADER,
        quanta::BlendState::NONE,
    )?;
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
