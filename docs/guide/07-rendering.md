# Rendering

Quanta supports the full graphics pipeline: vertex processing, rasterization,
fragment shading, blending, and render targets. This page covers the pipeline
from a high level. See [vertex and fragment shaders](08-vertex-fragment.md)
for shader authoring.

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

    let vb = gpu.render_field::<PosVertex>(3)?;
    vb.write(&vertices)?;

    let pipeline = gpu.pipeline(&PipelineDesc {
        vertex: PASSTHROUGH_SHADER.for_vendor(gpu.caps().vendor).unwrap(),
        fragment: SOLID_COLOR_SHADER.for_vendor(gpu.caps().vendor).unwrap(),
        vertex_entry: "passthrough",
        fragment_entry: "solid_color",
        vertex_layouts: &[PosVertex::vertex_layout()],
        color_formats: vec![Format::BGRA8],
        ..PipelineDesc::default()
    })?;

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

A render pipeline bundles vertex + fragment shaders with rasterization state:

```rust
let pipeline = gpu.pipeline(&PipelineDesc {
    vertex: vertex_shader_bytes,
    fragment: fragment_shader_bytes,
    vertex_entry: "vertex_main",
    fragment_entry: "fragment_main",
    vertex_layouts: &[MyVertex::vertex_layout()],
    color_formats: vec![Format::BGRA8],
    depth_format: None,
    sample_count: 1,
    blend: BlendState::NONE,
    cull_mode: CullMode::Back,
    primitive: Primitive::Triangle,
    depth_stencil: DepthStencilState::NONE,
    ..PipelineDesc::default()
})?;
```

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

| Builder method                              | Description                    |
|---------------------------------------------|--------------------------------|
| `.draw(vertex_count)`                      | Non-indexed draw               |
| `.draw_instanced(verts, instances)`        | Instanced draw                 |
| `.draw_indexed(index_count)`               | Indexed draw (requires `.indices()`) |
| `.draw_indexed_instanced(idx, inst)`       | Indexed + instanced           |
| `.draw_indirect(&buffer, offset)`          | GPU-driven draw                |

## Depth testing

Enable depth testing for 3D scenes where closer objects occlude farther ones:

```rust
let pipeline = gpu.pipeline(&PipelineDesc {
    depth_format: Some(Format::Depth32Float),
    depth_stencil: DepthStencilState::DEPTH_LESS,
    ..PipelineDesc::default()
})?;
```

| Constant                              | Behavior                          |
|---------------------------------------|-----------------------------------|
| `DepthStencilState::NONE`             | No depth testing (2D)             |
| `DepthStencilState::DEPTH_LESS`       | Standard 3D (closer wins)         |
| `DepthStencilState::DEPTH_READ_ONLY`  | Test but don't write (transparent)|

With a depth target:

```rust
let color = gpu.render_target(800, 600, Format::RGBA8)?;
let depth = gpu.create_texture(&TextureDesc {
    width: 800,
    height: 600,
    format: Format::Depth32Float,
    usage: TextureUsage::RENDER_TARGET,
    ..TextureDesc::default()
})?;

gpu.render(&color)?
    .color_targets(vec![ColorTarget {
        texture: color.handle(),
        load_op: LoadOp::Clear(Color::BLACK),
        store_op: StoreOp::Store,
    }])
    .depth_target(DepthTarget {
        texture: depth.handle(),
        load_op: LoadOp::Clear(Color::rgba(1.0, 0.0, 0.0, 0.0)),
        store_op: StoreOp::DontCare,
        stencil_load_op: LoadOp::DontCare,
        stencil_store_op: StoreOp::DontCare,
    })
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

See [textures](06-textures.md) for `sample()` usage in fragment shaders.

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

## Next

- [Vertex and fragment shaders](08-vertex-fragment.md) -- writing shader code
- [Device functions](09-device-functions.md) -- reusable GPU helpers
