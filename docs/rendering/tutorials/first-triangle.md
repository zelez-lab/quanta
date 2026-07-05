# Your first triangle

> **You'll learn:** the shortest path to pixels — define vertices, bind a
> pipeline, draw, and present. This is the "hello world" of the rendering track.

The graphics pipeline turns geometry into pixels: vertices go in, a triangle gets
rasterized, and a fragment shader colors each pixel. Here's the whole thing end to
end. The runnable browser version is
[`examples/web_triangle`](https://github.com/zelez-lab/quanta/tree/main/examples/web_triangle).

```toml
quanta = { version = "0.1", features = ["metal"] } # or vulkan / webgpu
```

Rendering is on by default, so everything here works through `quanta` directly.
The render methods on `Gpu` (`gpu.pipeline`, `gpu.render`, `gpu.render_target`,
…) come from the sealed `RenderGpu` extension trait; `use quanta::*;` brings it
into scope (or import it explicitly: `use quanta::RenderGpu;`).

## Define the geometry

A vertex type is a plain `#[repr(C)]` struct; the `Vertex` derive figures out the
layout (strides, offsets, attribute formats) so you never compute them by hand:

```rust,ignore
use quanta::*;

#[repr(C)]
#[derive(Copy, Clone, quanta::Vertex)]
struct PosVertex {
    pos: [f32; 3],   // location 0, Float3
}

let vertices = [
    PosVertex { pos: [ 0.0,  0.5, 0.0] },  // top
    PosVertex { pos: [-0.5, -0.5, 0.0] },  // bottom-left
    PosVertex { pos: [ 0.5, -0.5, 0.0] },  // bottom-right
];
```

Upload them to a vertex buffer:

```rust,ignore
let vb = gpu.field_with_usage::<PosVertex>(3, FieldUsage::default_render())?;
vb.write(&vertices)?;
```

## Build a pipeline

A pipeline pairs a vertex shader (positions) with a fragment shader (colors) plus
the fixed-function state. The two shaders here are the smallest possible pair —
pass the position through, paint a solid color; the
[next lesson](vertex-fragment.md) goes deep on shader authoring:

```rust,ignore
#[quanta::vertex]
fn passthrough(pos: Vec3) -> Vec4 {
    Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

#[quanta::fragment]
fn solid_color() -> Vec4 {
    Vec4::new(1.0, 0.3, 0.2, 1.0)
}
```

Each macro emits a `ShaderBinary` static (`PASSTHROUGH_SHADER`,
`SOLID_COLOR_SHADER`) with per-vendor payloads; the driver picks the right
one. Descriptors are built with `PipelineDesc::new` plus `with_*` methods
(they are `#[non_exhaustive]`, so there is no struct-literal form):

```rust,ignore
let layouts = [PosVertex::vertex_layout()];
let pipeline = gpu.pipeline(
    &PipelineDesc::new(ShaderSource::Binaries {
        vertex: &PASSTHROUGH_SHADER,
        fragment: &SOLID_COLOR_SHADER,
    })
    .with_entries("passthrough", "solid_color")
    .with_vertex_layouts(&layouts)
    .with_color_formats(vec![Format::BGRA8]),
)?;
```

## Draw

Create a render target (a texture to draw into), then use the chainable builder:
clear it, bind the pipeline and vertices, draw three vertices, and submit.

```rust,ignore
let target = gpu.render_target(800, 600, Format::BGRA8)?;

gpu.render(&target)?
    .clear(Color::BLACK)      // start from a black frame
    .pipeline(&pipeline)      // bind the pipeline
    .vertices(0, &vb)         // bind the vertex buffer to slot 0
    .draw(3)                  // rasterize 3 vertices → one triangle
    .pulse()?                 // submit
    .wait()?;                 // block until the GPU finishes
```

That's a rendered triangle sitting in `target`. From here you'd read it back
(`target.read()?`), present it through a [surface](presentation.md), or export
it to a compositor via `target.native_handle()`.

## The shape of every frame

Every render you write is this same rhythm:

1. **Geometry** — vertices (and later, indices) in a buffer.
2. **Pipeline** — the shaders + state describing *how* to draw.
3. **Pass** — `gpu.render(target)` → clear → bind → draw → `pulse`.

Everything in the rest of this track — textures, tessellation, ray tracing — adds
to one of those three, never changes the rhythm.

## Next

- **[Vertex and fragment shaders](vertex-fragment.md)** — replace the built-in shaders with your own, written in Rust.
