# Rendering tutorials

An ordered path through Quanta's graphics face. Each lesson builds on the one
before — start at the triangle and work down, and by the end you can drive the
full pipeline: shaders, textures, tessellation, mesh shaders, ray tracing, and
the advanced command paths.

Rendering shares one device handle with compute (`quanta::Gpu`); it lives behind
the default-on `render` feature. If you build headless (`default-features =
false`), add the [`quanta-render`](https://github.com/zelez-lab/quanta/blob/main/crates/quanta-render/README.md)
crate to bring the render surface back. See
[Getting Started](../../getting-started.md#compute-only-or-compute--rendering).

## The path

1. [Your first triangle](first-triangle.md) — clear a target, draw three vertices
2. [Vertex and fragment shaders](vertex-fragment.md) — write shaders as Rust functions
3. [Rendering pipeline](rendering.md) — the pipeline, render targets, and the builder in full
4. [Textures](textures.md) — sample images, render to texture, MSAA
5. [Tessellation](tessellation.md) — subdivide patches on the GPU
6. [Mesh shaders](mesh-shaders.md) — the modern geometry pipeline
7. [Ray tracing](ray-tracing.md) — acceleration structures and ray dispatch
8. [Variable rate shading](variable-rate-shading.md) — shade coarser where you can afford to
9. [Indirect commands](indirect-commands.md) — let the GPU drive its own draws
10. [Multi-queue](multi-queue.md) — overlap work across queues

Each lesson has a matching [how-to recipe](../how-to/) for when you already know
the concept and just want the code, and links into the [reference](../../reference/api.md).
