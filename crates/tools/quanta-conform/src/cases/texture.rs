//! `texture_sample_quad` — an 8×8 checkerboard sampled with nearest
//! filtering across the fullscreen quad.

use super::common::{color_target, fullscreen_vb, quad_pipeline, read_frame};
use crate::Frame;
use quanta::RenderGpu;
use quanta::render_pass::ColorTarget;
use quanta::{LoadOp, StoreOp};

// The fragment below takes `QuadVary` — see the note in `bands`.
use super::common::__quanta_varyings_QuadVary;

/// Checker side in texels. At 64 pixels the frame is an exact 8× blow-up,
/// so every fragment samples a texel CENTRE (`(c + 0.5) / 64` is
/// `(i + 0.5) / 8` for `i = c / 8`) and no filter-rounding decision is
/// ever close.
const SIDE: u32 = 8;

/// The two checker colours — distinct in all three channels, so a
/// swapped or dropped channel shows.
const LIGHT: [u8; 4] = [230, 40, 20, 255];
const DARK: [u8; 4] = [20, 120, 230, 255];

/// A checkerboard is its own vertical-flip detector: mirroring the rows
/// inverts the parity, so every texel changes colour.
fn checkerboard() -> Vec<u8> {
    let mut bytes = Vec::with_capacity((SIDE * SIDE * 4) as usize);
    for y in 0..SIDE {
        for x in 0..SIDE {
            let texel = if (x + y) % 2 == 0 { LIGHT } else { DARK };
            bytes.extend_from_slice(&texel);
        }
    }
    bytes
}

#[quanta::fragment]
fn conform_checker_sample_frag(s: QuadVary, checker: &Sampled2D) -> Vec4 {
    sample(checker, s.uv)
}

/// `texture_sample_quad` — nearest sampling, so only the fetch is
/// compared and never the filtering math.
pub fn run(gpu: &quanta::Gpu) -> Result<Frame, quanta::QuantaError> {
    let (w, h) = (64u32, 64u32);
    let pipe = quad_pipeline(
        gpu,
        &CONFORM_CHECKER_SAMPLE_FRAG_SHADER,
        quanta::BlendState::NONE,
    )?;
    let vb = fullscreen_vb(gpu)?;
    let tex = gpu.create_texture(
        &quanta::TextureDesc::new(SIDE, SIDE, quanta::Format::RGBA8)
            .with_usage(quanta::TextureUsage::SHADER_READ),
    )?;
    tex.write(&checkerboard())?;

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
        .texture(0, &tex)
        .sampler(
            0,
            quanta::SamplerDesc::default()
                .with_filters(quanta::Filter::Nearest, quanta::Filter::Nearest),
        )
        .draw(6)
        .pulse()?;
    pulse.wait()?;
    read_frame(&target, w, h)
}
