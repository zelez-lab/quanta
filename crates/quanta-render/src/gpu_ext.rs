//! The `RenderGpu` marker for `quanta::Gpu`.
//!
//! In this 2-crate layout the render methods (`pipeline`, `render`,
//! `render_target`, `mesh_pipeline`, `build_acceleration_structure`, …)
//! remain *inherent* methods on `quanta::Gpu`, gated behind the `render`
//! Cargo feature. Because `quanta-render` builds `quanta` with `render`
//! on, those methods are already callable directly:
//!
//! ```ignore
//! let gpu = quanta::init()?;
//! let pipe = gpu.pipeline(&desc)?;   // inherent, available under `render`
//! ```
//!
//! `RenderGpu` is a sealed marker trait implemented for `quanta::Gpu`. It
//! exists so a consumer can write `use quanta_render::RenderGpu;` to make
//! the render intent explicit and to have a stable name to bound generic
//! code on ("a GPU with the render face linked"). It deliberately adds no
//! methods — duplicating the inherent render methods as trait methods
//! would create call-ambiguity for no benefit.
//!
//! (The driver line is render-typed — the `GpuDevice` trait speaks
//! `PipelineDesc`/`RenderPass` and all four backends execute render ops —
//! so the render code physically lives in `quanta` behind the feature, not
//! in a separate crate; see roadmap 085. `quanta-render` is the front door
//! that turns the feature on and surfaces the render types + macros.)

mod sealed {
    pub trait Sealed {}
    impl Sealed for quanta::Gpu {}
}

/// Marker for a `quanta::Gpu` with the render face compiled in. Implemented
/// only for `quanta::Gpu` (sealed). See the module docs.
pub trait RenderGpu: sealed::Sealed {}

impl RenderGpu for quanta::Gpu {}
