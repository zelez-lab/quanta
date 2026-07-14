# Rendering tutorials

An ordered path through Quanta's graphics face. Each lesson builds on the one
before — start at the triangle and work down, and by the end you can drive the
full pipeline: shaders, textures, tessellation, mesh shaders, ray tracing, and
the advanced command paths.

Rendering shares one device handle with compute (`quanta::Gpu`); it lives in
the [`quanta-render`](https://github.com/zelez-lab/quanta/blob/main/crates/gpu/quanta-render/README.md)
crate, pulled in by the facade's default-on `render` feature. The render
methods (`gpu.render(...)`, `gpu.pipeline(...)`, …) come from the sealed
`RenderGpu` extension trait — `use quanta::*;` brings it into scope. A
render-only consumer (a UI toolkit, a compositor) can depend on
`quanta-render` directly and never pull in the compute stack. See
[Getting Started](../../getting-started.md#compute-only-or-compute--rendering).

## The path

1. [Your first triangle](first-triangle.md) — clear a target, draw three vertices
2. [Vertex and fragment shaders](vertex-fragment.md) — write shaders as Rust functions
3. [Rendering pipeline](rendering.md) — the pipeline, render targets, and the builder in full
4. [Textures](textures.md) — sample images, render to texture, MSAA
5. [Presenting to the screen](presentation.md) — surfaces, the frame loop, native-handle interop
6. [Tessellation](tessellation.md) — subdivide patches on the GPU
7. [Mesh shaders](mesh-shaders.md) — the modern geometry pipeline
8. [Ray tracing](ray-tracing.md) — acceleration structures and ray dispatch
9. [Variable rate shading](variable-rate-shading.md) — shade coarser where you can afford to
10. [Indirect commands](indirect-commands.md) — let the GPU drive its own draws
11. [Multi-queue](multi-queue.md) — overlap work across queues

Each lesson has a matching [how-to recipe](../how-to/) for when you already know
the concept and just want the code, and links into the [reference](../../reference/api.md).
