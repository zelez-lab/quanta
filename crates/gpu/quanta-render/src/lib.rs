//! # Quanta — rendering face
//!
//! The graphics half of Quanta: render passes, graphics pipelines,
//! textures-as-render-targets, tessellation, mesh shaders, ray tracing,
//! and variable-rate shading. Builds on the shared substrate crate
//! `quanta-core` (with its `render` feature on) — never on the compute
//! stack. The dependency is one-directional and the graph proves it: a
//! render-only consumer's tree contains no kernel lowering, no JIT, no
//! WASM machinery.
//!
//! A headless compute consumer depends on the `quanta` facade with the
//! `compute` feature and never compiles or sees a single rendering
//! type. A graphical consumer depends on this crate (directly, or via
//! the facade's `render` feature) and brings the render surface into
//! scope:
//!
//! ```ignore
//! use quanta_render::{vertex, fragment}; // render-stage shader macros
//! use quanta_render::{PipelineDesc, RenderGpu};
//!
//! let gpu = quanta_render::init()?;      // device line, re-exported
//! let pipeline = gpu.pipeline(&desc)?;   // RenderGpu extension method
//! ```
//!
//! ## How the split works
//!
//! Quanta has one device handle ([`Gpu`], from `quanta-core`) wrapping
//! one `Arc<dyn GpuDevice>`. The `GpuDevice` trait itself speaks the
//! render *data model* (`PipelineDesc` / `RenderPass` / `RenderOp`) and
//! all four backends execute render ops, so that data model lives in
//! `quanta-core` behind its `render` feature. This crate adds
//! everything a render consumer touches on top of it:
//!
//! - the [`RenderGpu`] extension trait — the render methods that used
//!   to be inherent on `Gpu` (`pipeline`, `render`, `render_target`,
//!   `mesh_pipeline`, …). Sealed; implemented for `quanta_core::Gpu`.
//! - the typed wrappers whose lifecycles are proven in Lean/Verus:
//!   [`MeshPipeline`], [`TessellationPipeline`], [`VrsState`],
//!   [`AccelerationStructure`], [`RayTracingPipeline`], [`Surface`].
//! - the chainable [`RenderBuilder`].
//! - the render-stage shader macros, re-exported from `quanta-render-dsl`.
//!
//! The whole `quanta-core` surface is re-exported so this crate is
//! self-sufficient for a render-only consumer (`init`, fields for
//! vertex data, textures, sync — all reachable as `quanta_render::…`).

#![no_std]

extern crate alloc;

// Re-export the render-stage shader macros so a render consumer pulls
// them from `quanta_render` rather than reaching into the facade.
pub use quanta_render_dsl::{
    Vertex, closest_hit, fragment, mesh, miss, ray_gen, task, tess_control, tess_eval, vertex,
};

// The shared substrate, wholesale: the device line (`init`, `devices`,
// `Gpu`, `GpuDevice`), resources (fields, textures, samplers, sync),
// and — because this crate turns `quanta-core/render` on — the render
// data model (`PipelineDesc`, `RenderPass`, `ColorTarget`, shader
// binaries, surface configuration, `IndirectRenderBundle`, …).
pub use quanta_core::*;

mod gpu_ext;
mod mesh_shader;
mod ray_tracing_wrap;
mod render_builder;
mod surface_wrap;
mod tessellation;
mod vrs_wrap;

pub use gpu_ext::RenderGpu;
pub use mesh_shader::{
    MAX_GROUP_COUNT, MAX_MESH_PRIMITIVES, MAX_MESH_VERTICES, MAX_TASK_THREADS, MeshPipeline,
    MeshPipelineDesc,
};
pub use ray_tracing_wrap::{
    AccelerationStructure, AsKind, MAX_DISPATCH_DIM, MAX_RECURSION_DEPTH, RayTracingPipeline,
};
pub use render_builder::RenderBuilder;
pub use surface_wrap::{Surface, SurfaceFrame};
pub use tessellation::{MAX_PATCH_SIZE, MAX_TESS_LEVEL, TessTopology, TessellationPipeline};
pub use vrs_wrap::VrsState;
