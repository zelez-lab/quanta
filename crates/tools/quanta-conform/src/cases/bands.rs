//! Banding fragments over the fullscreen quad: the orientation
//! contract as a frame, and the nested-if control flow both native
//! emitters have to agree on.

use super::common::{draw_frame, fullscreen_vb, quad_pipeline};
use crate::Frame;

// The fragments below take `QuadVary`; the shader macros reach its
// interface through the derive-generated trampoline, which has to be in
// scope here (the struct itself is erased along with the shader fn).
use super::common::__quanta_varyings_QuadVary;

/// Thirds of `uv.y`: red top, green middle, blue bottom. Under the
/// ratified convention (NDC +Y up, framebuffer origin top-left,
/// readback row 0 = top) `uv.y > 2/3` IS the top of the frame, so a
/// backend that flips vertically paints this upside down and every
/// pixel outside the middle band diverges.
///
/// No pixel centre lands on a band edge: at 64 rows the nearest are
/// `uv.y` 0.664 and 0.680 against the 0.667 boundary, ~0.003 clear.
#[quanta::fragment]
fn conform_orientation_bands_frag(s: QuadVary) -> Vec4 {
    if s.uv.y > 0.6666667 {
        Vec4::new(1.0, 0.0, 0.0, 1.0)
    } else {
        if s.uv.y > 0.3333333 {
            Vec4::new(0.0, 1.0, 0.0, 1.0)
        } else {
            Vec4::new(0.0, 0.0, 1.0, 1.0)
        }
    }
}

/// The four-band nested expression-if from `shader_parity_draw`'s D1
/// (Layer-A fixture `expr_if_nested`), as a whole frame instead of
/// eight probed texels.
#[quanta::fragment]
fn conform_nested_if_bands_frag(s: QuadVary) -> Vec4 {
    let c = if s.uv.x < 0.25 {
        Vec4::new(1.0, 0.0, 0.0, 1.0)
    } else {
        if s.uv.x < 0.5 {
            Vec4::new(0.0, 1.0, 0.0, 1.0)
        } else {
            if s.uv.x < 0.75 {
                Vec4::new(0.0, 0.0, 1.0, 1.0)
            } else {
                Vec4::new(1.0, 1.0, 1.0, 1.0)
            }
        }
    };
    c
}

/// `orientation_bands` — the vertical convention, as a frame.
pub fn run_orientation(gpu: &quanta::Gpu) -> Result<Frame, quanta::QuantaError> {
    run(gpu, &CONFORM_ORIENTATION_BANDS_FRAG_SHADER)
}

/// `bands_nested_if` — four `uv.x` bands through nested if-expressions.
pub fn run_nested_if(gpu: &quanta::Gpu) -> Result<Frame, quanta::QuantaError> {
    run(gpu, &CONFORM_NESTED_IF_BANDS_FRAG_SHADER)
}

fn run(gpu: &quanta::Gpu, frag: &quanta::ShaderBinary) -> Result<Frame, quanta::QuantaError> {
    let pipe = quad_pipeline(gpu, frag, quanta::BlendState::NONE)?;
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
