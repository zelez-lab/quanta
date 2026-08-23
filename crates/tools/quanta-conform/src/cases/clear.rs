//! `clear_only` — a clear and nothing else: the load-op path, the
//! format conversion, and the readback, with no rasterizer involved.

use crate::Frame;
use quanta::RenderGpu;
use quanta::render_pass::ColorTarget;
use quanta::{LoadOp, StoreOp};

/// Clear to a non-trivial colour (values that do not round the same
/// way under every naive float→unorm conversion) and read it back.
pub fn run(gpu: &quanta::Gpu) -> Result<Frame, quanta::QuantaError> {
    let (w, h) = (64u32, 64u32);
    let target = gpu.render_target(w, h, quanta::Format::RGBA8)?;
    let mut pulse = gpu
        .render(&target)?
        .color_targets(vec![
            ColorTarget::new(&target)
                .with_load_op(LoadOp::Clear(quanta::Color::rgba(0.25, 0.5, 0.75, 1.0)))
                .with_store_op(StoreOp::Store),
        ])
        .pulse()?;
    pulse.wait()?;
    let bytes = target.read()?;
    Frame::rgba8(w, h, bytes).map_err(quanta::QuantaError::invalid_param)
}
