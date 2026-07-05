# quanta-render

The **render face of Quanta** — render passes, graphics pipelines,
textures-as-render-targets, tessellation, mesh shaders, ray tracing,
variable-rate shading, and presentation. A real crate, not a feature shim: it
builds on the shared substrate crate [`quanta-core`](../quanta-core) (with its
`render` feature on) plus the render-stage DSL macros, and **never on the
compute stack** — a render-only consumer's dependency graph contains no kernel
lowering, no JIT, no WASM machinery.

```rust,ignore
use quanta_render::{vertex, fragment};              // render-stage shader macros
use quanta_render::{Format, PipelineDesc, RenderGpu, ShaderSource};

let gpu = quanta_render::init()?;                   // device line, re-exported from quanta-core
let target = gpu.render_target(256, 256, Format::RGBA8)?;
let pipeline = gpu.pipeline(&desc)?;                // RenderGpu extension method

let mut pulse = gpu.render(&target)?                // chainable RenderBuilder
    .clear(Color::BLACK)
    .pipeline(&pipeline)
    .vertices(0, &verts)
    .draw(3)                                        // a triangle
    .pulse()?;                                      // submit
pulse.wait()?;
```

```toml
quanta-render = { version = "0.1", features = ["metal"] } # or vulkan / webgpu
```

Consumers coming through the `quanta` facade get the same surface from the
default-on `render` feature (`quanta = { features = ["metal", "render"] }`),
which pulls this crate in and re-exports it — `use quanta::*;` covers
everything below.

## How the compute/render split works

Quanta has **one device handle** (`Gpu`, defined in `quanta-core`) wrapping one
`Arc<dyn GpuDevice>`. The compute/render boundary cuts *through* the driver
line — the `GpuDevice` trait itself speaks the render data model
(`PipelineDesc` / `RenderPass` / `RenderOp`) and the backends execute render
ops — so that data model lives in `quanta-core` behind its `render` feature.
This crate adds everything a render consumer touches on top of it:

- **`RenderGpu`** — a **sealed extension trait** carrying the render methods
  that were historically inherent on `Gpu`: `pipeline`, `render`,
  `render_target`, `msaa_target`, `resolve_texture`, `stencil_read`,
  `render_bundle`, `vrs_state`, `mesh_pipeline`, `tessellation_pipeline`,
  `create_surface`, `occlusion_query_create` / `_read`,
  `acceleration_structure_blas`, `ray_tracing_pipeline`. Bring it into scope
  (`use quanta_render::RenderGpu;` or the facade glob) to call them. Sealed:
  implemented only for `quanta_core::Gpu`, so methods can be added after the
  API freeze without a breaking change.
- The chainable **`RenderBuilder`** (`gpu.render(&target)?` → record draws →
  `.pulse()?`).
- The **typed wrappers** whose lifecycles are proven in Lean/Verus:
  `MeshPipeline`, `TessellationPipeline`, `VrsState`, `AccelerationStructure`,
  `RayTracingPipeline`. Each releases its driver resource on `Drop`, exactly
  once.
- **`Surface` / `SurfaceFrame`** — Quanta-owned presentation: `create_surface`
  → `acquire()` → render into `frame.texture()` → `frame.present()`. The
  sibling model, `Texture::native_handle()`, exports the backend-native
  texture so an external compositor owns present instead.
- The **render-stage shader macros**, re-exported from `quanta-dsl`: `vertex`,
  `fragment`, `Vertex` (attribute derive), `tess_control`, `tess_eval`,
  `mesh`, `task`, `ray_gen`, `closest_hit`, `miss`.

The whole `quanta-core` surface is re-exported, so this crate is
self-sufficient for a render-only consumer: `init`, fields for vertex data,
textures, samplers, sync — all reachable as `quanta_render::…`.

A **headless compute** consumer never sees any of this: it depends on the
`quanta` facade with `default-features = false, features = ["metal",
"compute", "jit"]` and compiles zero rendering code.

## Backends

Render runs on **Metal**, **Vulkan**, and **WebGPU** (the software CPU lane is
compute-only). Rendering produces **identical pixel output across Metal and
Vulkan** — zero validation errors on either — and runs in the browser via
WebGPU (see the `web_triangle` / `web_textured` examples in the workspace
root).

Advanced features are capability-gated: `gpu.supports_ray_tracing()`,
`supports_mesh_shaders()`, `supports_tessellation()`, `supports_vrs()`,
`supports_surface_present()` (Metal today),
`supports_native_handle_export()` (Metal + Vulkan). Unsupported features
return `QuantaErrorKind::NotSupported` rather than panicking — check before
use.

## Learn it

The [Rendering tutorials](../../docs/SUMMARY.md) teach this in order — first
triangle → vertex/fragment shaders → textures → presentation → tessellation →
mesh shaders → ray tracing → VRS → indirect commands → multi-queue. The
Rendering how-to recipes cover shadow mapping, deferred rendering, and the
advanced pipelines.
