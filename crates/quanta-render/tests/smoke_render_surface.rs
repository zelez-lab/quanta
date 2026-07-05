//! Smoke: the render surface is reachable through quanta-render — the
//! device line and render types re-exported, and the render methods
//! callable through the sealed `RenderGpu` extension trait
//! (compile-only).
use quanta_render::{Gpu, PipelineDesc, RenderGpu};

#[allow(dead_code)]
fn uses_render_surface(gpu: &Gpu, desc: &PipelineDesc) {
    fn bound<T: RenderGpu>(_: &T) {}
    bound(gpu);
    let _ = gpu.pipeline(desc); // RenderGpu extension method
    let _ = gpu.render_target(4, 4, quanta_render::Format::RGBA8);
}
