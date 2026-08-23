//! `msaa4_resolve` — a triangle rasterized 4× multisampled and
//! resolved into a single-sample target.

use super::common::{color_target, field_of, pipeline_desc, pos_layout, pos_vertex, read_frame};
use crate::{Frame, Tolerance};
use quanta::RenderGpu;

/// The comparison terms. Resolve is a per-backend average over sample
/// positions the specs do not pin, so edge pixels legitimately differ
/// by more than a clear/interior pixel ever would: two LSBs of channel
/// slack and a 10‰ budget for the edge itself.
pub const RESOLVE: Tolerance = Tolerance {
    channel: 2,
    edge_budget_permille: 10,
};

/// Deliberately off-axis on every edge: an axis-aligned triangle would
/// resolve to the same pixels a single-sample draw produces and prove
/// nothing about the multisample path.
#[rustfmt::skip]
const TRIANGLE: [f32; 9] = [
    -0.75, -0.6, 0.0,
     0.8, -0.85, 0.0,
     0.1,  0.85, 0.0,
];

#[quanta::fragment]
fn conform_msaa_frag() -> Vec4 {
    Vec4::new(0.9019608, 0.8, 0.101960786, 1.0)
}

/// `msaa4_resolve` — the builder-managed MSAA path: a pooled 4×
/// intermediate, cleared and drawn into, then subpass-resolved into the
/// single-sample target the frame is read from.
pub fn run(gpu: &quanta::Gpu) -> Result<Frame, quanta::QuantaError> {
    let (w, h) = (64u32, 64u32);
    let layouts = pos_layout();
    let pipe = gpu.pipeline(
        &pipeline_desc(pos_vertex(), &CONFORM_MSAA_FRAG_SHADER, &layouts).with_sample_count(4),
    )?;
    let vb = field_of(gpu, &TRIANGLE)?;
    let target = color_target(gpu, w, h)?;
    let mut pulse = gpu
        .render(&target)?
        .msaa(4)
        .clear(quanta::Color::rgba(0.0, 0.1, 0.3, 1.0))
        .viewport(0.0, 0.0, w as f32, h as f32)
        .pipeline(&pipe)
        .vertices(0, &vb)
        .draw(3)
        .msaa_resolve()
        .pulse()?;
    pulse.wait()?;
    read_frame(&target, w, h)
}
