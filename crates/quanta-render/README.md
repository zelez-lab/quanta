# quanta-render

The **graphics face of Quanta** — render passes, graphics pipelines,
textures-as-render-targets, tessellation, mesh shaders, ray tracing, and
variable-rate shading. It is the front door for graphical consumers: it builds
[`quanta`](../..) with the `render` feature on, re-exports every render type and
shader-stage macro, and adds the [`RenderGpu`] marker so render intent can be
named.

```rust,ignore
use quanta_render::{vertex, fragment, PipelineDesc};

let gpu = quanta::init()?;                  // one device handle, compute + render
let pipeline = gpu.pipeline(&desc)?;        // render methods are inherent on Gpu
let target = gpu.render_target(256, 256, Format::Rgba8)?;

gpu.render(&target)?                        // chainable RenderBuilder
    .set_pipeline(&pipeline)
    .bind_vertices(0, &verts)
    .clear(Color::BLACK)
    .draw(3)                                // a triangle
    .pulse()?;                              // submit
```

```toml
quanta-render = { version = "0.1", features = ["metal"] } # or vulkan / webgpu
```

## How the compute/render split works

Quanta has **one device handle** (`quanta::Gpu`). The compute/render boundary is
the `render` **Cargo feature**, not a separate driver — the render types live in
`quanta` behind `#[cfg(feature = "render")]`, because the boundary cuts *through*
the driver line (the `GpuDevice` trait itself speaks `PipelineDesc` / `RenderPass`
and all four backends execute render ops).

- A **headless compute** consumer depends on `quanta` alone with
  `default-features = false` and compiles **zero rendering code** — no render
  module, type, or `Gpu` method exists on its surface.
- A **graphical** consumer adds `quanta-render`, which turns the `render` feature
  on and brings the render surface into scope. The render methods become callable
  inherent methods on `quanta::Gpu`.

## What it re-exports

**Shader-stage macros** (from `quanta-dsl`) — write render shaders in Rust:
`vertex`, `fragment`, `Vertex` (attribute derive), `tess_control`, `tess_eval`,
`mesh`, `task`, `ray_gen`, `closest_hit`, `miss`.

**Render types** (re-exported from `quanta`, visible because this crate builds it
with `render` on):

| Category | Types |
|---|---|
| Pipelines / passes | `Pipeline`, `PipelineDesc`, `RenderBuilder`, `RenderPass` |
| Targets | `ColorTarget`, `DepthTarget` |
| Vertex layout | `AttributeFormat`, `StepMode`, `Primitive` |
| Blending / depth | `BlendFactor`, `BlendOp`, `CompareFunc`, `CullMode` |
| Tessellation | `TessellationPipeline`, `TessTopology` |
| Mesh shaders | `MeshPipeline`, `MeshPipelineDesc` |
| Variable rate shading | `VrsState`, `ShadingRate` |
| Indirect | `IndirectRenderBundle` |

Plus the [`RenderGpu`] marker for bounding render intent in generic code. Types
for ray tracing (`RayTracingPipeline`, `AccelerationStructure`), textures/samplers,
and MSAA are reached through `quanta::` directly, since they're shared with the
compute surface. Render methods (`gpu.render`, `gpu.pipeline`,
`gpu.tessellation_pipeline`, `gpu.mesh_pipeline`, `gpu.ray_tracing_pipeline`,
`gpu.vrs_state`, …) are inherent on `quanta::Gpu` once `render` is on.

## Backends

Render runs on **Metal**, **Vulkan**, and **WebGPU** (the software CPU lane is
compute-only). Rendering produces **identical pixel output across Metal and
Vulkan** — zero validation errors on either — and runs in the browser via WebGPU
(see the `web_triangle` / `web_textured` examples in the workspace root).

Advanced features are capability-gated: `gpu.supports_ray_tracing()`,
`supports_mesh_shaders()`, `supports_tessellation()`, `supports_vrs()`. Unsupported
features return `QuantaErrorKind::NotSupported` rather than panicking — check
before use.

## Learn it

The [Rendering tutorials](../../docs/SUMMARY.md) teach this in order — first
triangle → vertex/fragment shaders → textures → tessellation → mesh shaders →
ray tracing → VRS → indirect commands → multi-queue. The Rendering how-to
recipes cover shadow mapping, deferred rendering, and the advanced pipelines.
