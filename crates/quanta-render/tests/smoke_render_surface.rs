//! Smoke: the render surface is reachable through quanta-render — render
//! types re-exported, the RenderGpu marker bounds quanta::Gpu, and the
//! inherent render methods are callable (compile-only).
use quanta_render::{PipelineDesc, RenderGpu};

#[allow(dead_code)]
fn uses_render_surface(gpu: &quanta::Gpu, desc: &PipelineDesc) {
    fn bound<T: RenderGpu>(_: &T) {}
    bound(gpu);
    let _ = gpu.pipeline(desc); // inherent render method, present under `render`
}
