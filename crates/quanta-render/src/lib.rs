//! # Quanta — rendering face
//!
//! The graphics half of Quanta: render passes, graphics pipelines,
//! textures-as-render-targets, tessellation, mesh shaders, ray tracing,
//! and variable-rate shading. Depends on the compute substrate crate
//! [`quanta`]; the dependency is one-directional (`quanta-render → quanta`,
//! never the reverse — step 085).
//!
//! A headless compute consumer depends on `quanta` alone and never
//! compiles or sees a single rendering type. A graphical consumer adds
//! this crate and brings the render surface into scope:
//!
//! ```ignore
//! use quanta_render::{vertex, fragment}; // render-stage shader macros
//! use quanta_render::PipelineDesc;       // render types, re-exported
//!
//! let gpu = quanta::init()?;
//! let pipeline = gpu.pipeline(&desc)?;   // inherent on Gpu under `render`
//! ```
//!
//! ## How the split works
//!
//! Quanta has one device handle (`quanta::Gpu`, wrapping one
//! `Arc<dyn GpuDevice>`). The compute/render boundary is the **`render`
//! Cargo feature**, not a separate crate, because the boundary cuts
//! *through* the driver line: the `GpuDevice` trait itself speaks
//! `PipelineDesc`/`RenderPass` and all four backends execute render ops, so
//! the render code physically lives in `quanta` behind `#[cfg(feature =
//! "render")]`. A headless consumer depends on `quanta` with
//! `default-features = false` and compiles zero render code — no render
//! module, type, or `Gpu` method exists on its surface.
//!
//! `quanta-render` is the **front door** for graphical consumers: it builds
//! `quanta` with `render` on, re-exports the render types and shader macros,
//! and offers the [`RenderGpu`] marker (see its docs) so render intent can
//! be named and bounded. The render methods themselves are inherent on
//! `quanta::Gpu` and become callable once `render` is on. See roadmap 085.

#![no_std]

// Re-export the render-stage shader macros so a render consumer pulls them
// from `quanta_render` rather than reaching into `quanta`.
pub use quanta_dsl::{
    Vertex, closest_hit, fragment, mesh, miss, ray_gen, task, tess_control, tess_eval, vertex,
};

// Re-export the render types from `quanta` (visible because this crate
// builds `quanta` with the `render` feature on). A render consumer names
// them through `quanta_render::*`.
pub use quanta::{
    AttributeFormat, BlendFactor, BlendOp, ColorTarget, CompareFunc, CullMode, DepthTarget,
    IndirectRenderBundle, MeshPipeline, MeshPipelineDesc, Pipeline, PipelineDesc, Primitive,
    RenderBuilder, RenderPass, ShadingRate, StepMode, TessTopology, TessellationPipeline, VrsState,
};

mod gpu_ext;
pub use gpu_ext::RenderGpu;
