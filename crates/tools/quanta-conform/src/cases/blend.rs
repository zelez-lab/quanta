//! `alpha_blend_quad` — a half-transparent fullscreen quad through
//! `BlendState::ALPHA` over a clear the blend has to read back.

use super::common::{draw_frame, fullscreen_vb, quad_pipeline};
use crate::Frame;

/// The clear the quad blends over. Every channel is an exact 8-bit
/// value, and each one sums with the source below to an EVEN number of
/// LSBs — so `(src + dst) / 2` is an integer and no channel of the
/// blended result sits on a rounding tie.
const CLEAR: quanta::Color = quanta::Color::rgba(0.2, 0.6, 0.4, 1.0);

/// Source (255, 51, 0) at alpha 0.5 over (51, 153, 102) resolves to
/// (153, 102, 51): three distinct channels, all exact.
#[quanta::fragment]
fn conform_alpha_blend_frag() -> Vec4 {
    Vec4::new(1.0, 0.2, 0.0, 0.5)
}

/// `alpha_blend_quad` — src-alpha / one-minus-src-alpha over a clear.
pub fn run(gpu: &quanta::Gpu) -> Result<Frame, quanta::QuantaError> {
    let pipe = quad_pipeline(
        gpu,
        &CONFORM_ALPHA_BLEND_FRAG_SHADER,
        quanta::BlendState::ALPHA,
    )?;
    let vb = fullscreen_vb(gpu)?;
    draw_frame(gpu, &pipe, &vb, 6, 64, 64, CLEAR)
}
