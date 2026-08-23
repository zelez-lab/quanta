//! `triangle_flat` — one flat-colour triangle over a clear: the
//! rasterizer's coverage rule, with nothing else in the way.

use super::common::{draw_frame, field_of, pos_layout, pos_pipeline};
use crate::Frame;

/// A large centred triangle: two slanted edges plus a horizontal base,
/// all of them interior, so this is the case that pays the edge budget.
#[rustfmt::skip]
const TRIANGLE: [f32; 9] = [
     0.0,  0.8, 0.0,
    -0.85, -0.75, 0.0,
     0.85, -0.75, 0.0,
];

/// A colour whose three channels are distinct and land on exact 8-bit
/// values (230, 77, 26), so a dropped or swapped channel is obvious.
#[quanta::fragment]
fn conform_flat_frag() -> Vec4 {
    Vec4::new(0.9019608, 0.3019608, 0.101960786, 1.0)
}

/// `triangle_flat` — flat colour over a clear the triangle does not cover.
pub fn run(gpu: &quanta::Gpu) -> Result<Frame, quanta::QuantaError> {
    let layouts = pos_layout();
    let pipe = pos_pipeline(gpu, &CONFORM_FLAT_FRAG_SHADER, &layouts)?;
    let vb = field_of(gpu, &TRIANGLE)?;
    draw_frame(
        gpu,
        &pipe,
        &vb,
        3,
        64,
        64,
        quanta::Color::rgba(0.0, 0.2, 0.4, 1.0),
    )
}
