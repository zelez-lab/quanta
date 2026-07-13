# Rendering

Quanta supports the full graphics pipeline: vertex processing, rasterization,
fragment shading, blending, and render targets. This page covers the pipeline
from a high level. See [vertex and fragment shaders](vertex-fragment.md)
for shader authoring.

> **Enabling rendering.** The render face lives in the `quanta-render`
> crate, pulled in by the facade's default-on `render` feature. With
> default features, everything on this page works through `quanta`
> directly (`use quanta::*`). The render methods on `Gpu` (`gpu.render`,
> `gpu.pipeline`, `gpu.render_target`, …) come from the sealed
> `RenderGpu` extension trait — the glob import brings it into scope,
> or add `use quanta::RenderGpu;` explicitly. A render-only consumer
> can depend on `quanta-render` directly (no compute stack in its
> graph). See
> [Getting Started](../../getting-started.md#compute-only-or-compute--rendering).

## The render pipeline

```
Vertex Buffer -> [Vertex Shader] -> Rasterizer -> [Fragment Shader] -> Blend -> Render Target
```

1. **Vertex shader** transforms vertex positions (model space to clip space).
2. **Rasterizer** generates fragments (pixels) from triangles.
3. **Fragment shader** computes the color of each fragment.
4. **Blend** combines the fragment color with the existing render target value.
5. **Render target** stores the final pixel colors (a texture).

## The Vertex derive

Define vertex layout with `#[derive(quanta::Vertex)]` instead of manually
constructing `VertexLayout` structs:

```rust
use quanta::*;

#[repr(C)]
#[derive(Copy, Clone, quanta::Vertex)]
struct MyVertex {
    pos: [f32; 3],    // location 0, Float3
    color: [f32; 4],  // location 1, Float4
}
```

The derive macro generates:

- `MyVertex::ATTRIBUTES` -- const array of `VertexAttribute` with correct
  locations, offsets, and formats
- `MyVertex::vertex_layout()` -- returns a complete `VertexLayout` with
  stride, per-vertex stepping, and attributes

No manual stride calculation. No manual offset tracking. No manual
attribute format selection.

## Render builder

Use the chainable render builder for clean, expressive render passes:

```rust
use quanta::*;

#[repr(C)]
#[derive(Copy, Clone, quanta::Vertex)]
struct PosVertex {
    pos: [f32; 3],
}

fn render_triangle(gpu: &Gpu) -> Result<(), QuantaError> {
    let vertices = [
        PosVertex { pos: [ 0.0,  0.5, 0.0] },
        PosVertex { pos: [-0.5, -0.5, 0.0] },
        PosVertex { pos: [ 0.5, -0.5, 0.0] },
    ];

    let vb = gpu.field_with_usage::<PosVertex>(3, FieldUsage::default_render())?;
    vb.write(&vertices)?;

    // `passthrough` / `solid_color` are #[quanta::vertex] / #[quanta::fragment]
    // functions — see "Your first triangle" for their bodies.
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

    let target = gpu.render_target(800, 600, Format::BGRA8)?;

    gpu.render(&target)?
        .clear(Color::BLACK)
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw(3)
        .pulse()?
        .wait()?;

    Ok(())
}
```

The builder chain:
- `.clear()` -- fill the target with a color
- `.pipeline()` -- bind a render pipeline
- `.vertices()` -- bind vertex data at a slot
- `.draw()` -- record a draw command
- `.pulse()` -- submit the render pass, returns a `Pulse`
- `.wait()` -- block until the GPU finishes

## PipelineDesc

A render pipeline bundles vertex + fragment shaders with rasterization state.
`PipelineDesc` is `#[non_exhaustive]` — construct it with `PipelineDesc::new`
and the `with_*` builder methods:

```rust
let layouts = [MyVertex::vertex_layout()];
let pipeline = gpu.pipeline(
    &PipelineDesc::new(ShaderSource::Stages {
        vertex: vertex_shader_bytes,     // native payload (or use
        fragment: fragment_shader_bytes, // ShaderSource::Binaries for
    })                                   // the macro-generated statics)
    .with_entries("vertex_main", "fragment_main")
    .with_vertex_layouts(&layouts)
    .with_color_formats(vec![Format::BGRA8])
    .with_blend(BlendState::NONE)
    .with_cull_mode(CullMode::Back)
    .with_primitive(Primitive::Triangle),
)?;
```

The shader input is a `ShaderSource`: `Stages { vertex, fragment }` for
per-stage payloads already in the backend's native format,
`Combined(&[u8])` for one payload holding both entry points, or
`Binaries { vertex, fragment }` referencing the multi-vendor
`ShaderBinary` statics that `#[quanta::vertex]` / `#[quanta::fragment]`
generate (the driver picks the right format per vendor).

`with_color_formats` is per-attachment: `color_formats[i]` types color
attachment `i` of the pass, so its length must equal the number of color
targets the pass binds (a mismatch fails `pulse()`) — it is not a list of
formats the pipeline "may be used against".

### Blend modes

| Constant                         | Behavior                              |
|----------------------------------|---------------------------------------|
| `BlendState::NONE`               | Overwrite (opaque geometry)           |
| `BlendState::PREMULTIPLIED_ALPHA`| Standard compositing                  |
| `BlendState::ALPHA`              | Non-premultiplied alpha               |
| `BlendState::ADDITIVE`           | Add to existing (particles, glow)     |

### Cull modes

| Mode            | Discards              |
|-----------------|-----------------------|
| `CullMode::None`  | Nothing (double-sided) |
| `CullMode::Front` | Front-facing triangles |
| `CullMode::Back`  | Back-facing triangles  |

### Primitive types

| Type                    | Input                           |
|-------------------------|---------------------------------|
| `Primitive::Triangle`      | Every 3 vertices = 1 triangle |
| `Primitive::TriangleStrip` | Sliding window of triangles   |
| `Primitive::Line`          | Every 2 vertices = 1 line     |
| `Primitive::LineStrip`     | Connected line segments       |
| `Primitive::Point`         | 1 vertex = 1 point            |

## Draw commands

All available via the render builder:

| Builder method                                       | Description                          |
|------------------------------------------------------|--------------------------------------|
| `.draw(vertex_count)`                               | Non-indexed draw                     |
| `.draw_instanced(verts, instances)`                 | Instanced draw                       |
| `.draw_indexed(index_count)`                        | Indexed draw (requires `.indices()`) |
| `.draw_indexed_instanced(idx, inst)`                | Indexed + instanced                  |
| `.draw_indirect(&buffer, offset)`                   | GPU-driven non-indexed draw          |
| `.draw_indexed_indirect(&buffer, offset, &indices)` | GPU-driven indexed draw              |

Render-bundle replay (`RenderPass::execute_bundle`) goes through the manual
render-pass API — see [Indirect commands](indirect-commands.md).

For indirect draws the GPU reads the draw arguments (vertex count, instance
count, etc.) out of a buffer the GPU itself wrote — useful for compute-driven
culling. See [Indirect command buffers](indirect-commands.md) for the
argument layout and `IndirectRenderBundle`.

## Depth testing

Enable depth testing for 3D scenes where closer objects occlude farther ones:

```rust
let pipeline = gpu.pipeline(
    &PipelineDesc::new(shaders)
        .with_depth_format(Format::Depth32Float)
        .with_depth_stencil(DepthStencilState::DEPTH_LESS),
)?;
```

| Constant                              | Behavior                          |
|---------------------------------------|-----------------------------------|
| `DepthStencilState::NONE`             | No depth testing (2D)             |
| `DepthStencilState::DEPTH_LESS`       | Standard 3D (closer wins)         |
| `DepthStencilState::DEPTH_READ_ONLY`  | Test but don't write (transparent)|

With a depth target:

```rust
let color = gpu.render_target(800, 600, Format::RGBA8)?;
let depth = gpu.create_texture(
    &TextureDesc::new(800, 600, Format::Depth32Float)
        .with_usage(TextureUsage::RENDER_TARGET),
)?;

// Attachments are built from typed textures — ColorTarget::new /
// DepthTarget::new plus with_* overrides; no raw handles.
gpu.render(&color)?
    .color_targets(vec![ColorTarget::new(&color)]) // Clear(BLACK) + Store
    .depth_target(
        DepthTarget::new(&depth)
            .with_load_op(LoadOp::Clear(Color::rgba(1.0, 0.0, 0.0, 0.0)))
            .with_store_op(StoreOp::DontCare),
    )
    .viewport(0.0, 0.0, 800.0, 600.0)
    .pipeline(&pipeline)
    .vertices(0, &vb)
    .indices(&ib)
    .draw_indexed(36)
    .pulse()?
    .wait()?;
```

## Textured rendering

Bind textures and samplers through the builder:

```rust
gpu.render(&target)?
    .pipeline(&pipeline)
    .vertices(0, &vb)
    .texture(0, &albedo_texture)
    .sampler(0, SamplerDesc::default())
    .draw(6)
    .pulse()?
    .wait()?;
```

See [textures](textures.md) for `sample()` usage in fragment shaders.

## Viewport and scissor

```rust
gpu.render(&target)?
    .viewport(0.0, 0.0, 800.0, 600.0)   // NDC mapping
    .scissor(100, 100, 600, 400)         // pixel clipping rect
    .pipeline(&pipeline)
    .vertices(0, &vb)
    .draw(3)
    .pulse()?
    .wait()?;
```

## Beyond vertex/fragment

Quanta exposes the full v0.1 advanced pipeline surface as typed wrappers, each
gated by a capability query:

| Feature              | Capability query                  | Chapter                                      |
|----------------------|-----------------------------------|----------------------------------------------|
| Tessellation         | `gpu.supports_tessellation()`     | [Tessellation](tessellation.md)            |
| Mesh shaders         | `gpu.supports_mesh_shaders()`     | [Mesh shaders](mesh-shaders.md)            |
| Ray tracing          | `gpu.supports_ray_tracing()`      | [Ray tracing](ray-tracing.md)              |
| Variable rate shading| `gpu.supports_vrs()`              | [VRS](variable-rate-shading.md)            |
| Indirect commands    | always (CPU fallback exists)      | [Indirect commands](indirect-commands.md)  |

When a feature isn't implemented for the active backend the constructor returns
`QuantaErrorKind::NotSupported(reason)` rather than panicking — branch on the
error to fall back to the classic vertex/fragment path.

## Next

- [Vertex and fragment shaders](vertex-fragment.md) -- writing shader code
- [Presenting to the screen](presentation.md) -- surfaces and native-handle interop
- [Device functions](../../computation/tutorials/device-functions.md) -- reusable GPU helpers
