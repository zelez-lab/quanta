# Rendering

Quanta supports the full graphics pipeline: vertex processing, rasterization,
fragment shading, blending, and render targets. This page covers the pipeline
from a high level. See [vertex and fragment shaders](08-vertex-fragment.md) for
shader authoring.

## The render pipeline

```
Vertex Buffer ─> [Vertex Shader] ─> Rasterizer ─> [Fragment Shader] ─> Blend ─> Render Target
```

1. **Vertex shader** transforms vertex positions (model space to clip space).
2. **Rasterizer** generates fragments (pixels) from triangles.
3. **Fragment shader** computes the color of each fragment.
4. **Blend** combines the fragment color with the existing render target value.
5. **Render target** stores the final pixel colors (a texture).

## PipelineDesc

A render pipeline bundles vertex + fragment shaders with rasterization state:

```rust
use quanta::*;

let pipeline = gpu.pipeline(&PipelineDesc {
    vertex: vertex_shader_bytes,
    fragment: fragment_shader_bytes,
    vertex_entry: "vertex_main",
    fragment_entry: "fragment_main",
    vertex_layouts: &[VertexLayout {
        stride: 12, // 3 floats per vertex
        step: StepMode::Vertex,
        attributes: vec![VertexAttribute {
            location: 0,
            offset: 0,
            format: AttributeFormat::Float3,
        }],
    }],
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

## Render pass

A render pass records draw commands targeting a texture:

```rust
let target = gpu.render_target(800, 600, Format::BGRA8)?;

let mut pass = gpu.render_begin(&target)?;
pass.clear(Color::rgb(0.1, 0.1, 0.1));
pass.set_pipeline(&pipeline);
pass.bind_vertices(0, &vertex_buffer);
pass.draw(3); // Draw 3 vertices (1 triangle)
let mut pulse = gpu.render_end(pass)?;
gpu.wait(&mut pulse)?;
```

### Draw commands

| Method                                  | Description                    |
|-----------------------------------------|--------------------------------|
| `pass.draw(vertex_count)`              | Non-indexed draw               |
| `pass.draw_instanced(verts, instances)`| Instanced draw                 |
| `pass.draw_indexed(index_count)`       | Indexed draw (requires index buffer) |
| `pass.draw_indexed_instanced(idx, inst)`| Indexed + instanced           |
| `pass.draw_indirect(&buffer, offset)`  | GPU-driven draw                |

### Binding resources

```rust
pass.set_pipeline(&pipeline);
pass.bind_vertices(0, &vertex_buffer);  // slot 0 = vertex data
pass.bind_indices(&index_buffer);       // u32 index buffer
pass.set_uniform(0, &mvp_buffer);       // uniform buffer at slot 0
pass.set_texture(0, &albedo_texture);   // texture at slot 0
pass.set_value(1, &time_value);         // push constant
```

## Example: clear screen

The simplest render operation -- fill the target with a solid color:

```rust
let target = gpu.render_target(800, 600, Format::BGRA8)?;
let mut pass = gpu.render_begin(&target)?;
pass.clear(Color::rgb(0.0, 0.5, 1.0)); // sky blue
let mut pulse = gpu.render_end(pass)?;
gpu.wait(&mut pulse)?;
```

## Example: draw a triangle

```rust
use quanta::*;

// Vertex data: 3 vertices, each with position (x, y, z)
let vertices: Vec<f32> = vec![
     0.0,  0.5, 0.0,  // top
    -0.5, -0.5, 0.0,  // bottom-left
     0.5, -0.5, 0.0,  // bottom-right
];

let verts = gpu.render_field::<f32>(9)?;
gpu.write_field(&verts, &vertices)?;

let pipeline = gpu.pipeline(&PipelineDesc {
    vertex: PASSTHROUGH_SHADER.for_vendor(gpu.caps().vendor).unwrap(),
    fragment: SOLID_COLOR_SHADER.for_vendor(gpu.caps().vendor).unwrap(),
    vertex_entry: "passthrough",
    fragment_entry: "solid_color",
    vertex_layouts: &[VertexLayout {
        stride: 12,
        step: StepMode::Vertex,
        attributes: vec![VertexAttribute {
            location: 0,
            offset: 0,
            format: AttributeFormat::Float3,
        }],
    }],
    color_formats: vec![Format::BGRA8],
    ..PipelineDesc::default()
})?;

let target = gpu.render_target(800, 600, Format::BGRA8)?;
let mut pass = gpu.render_begin(&target)?;
pass.clear(Color::BLACK);
pass.set_pipeline(&pipeline);
pass.bind_vertices(0, &verts);
pass.draw(3);
let mut pulse = gpu.render_end(pass)?;
gpu.wait(&mut pulse)?;
```

## Depth testing

Enable depth testing for 3D scenes where closer objects occlude farther ones:

```rust
let pipeline = gpu.pipeline(&PipelineDesc {
    depth_format: Some(Format::Depth32Float),
    depth_stencil: DepthStencilState::DEPTH_LESS,
    ..PipelineDesc::default()
})?;
```

Available depth configurations:

| Constant                              | Behavior                          |
|---------------------------------------|-----------------------------------|
| `DepthStencilState::NONE`             | No depth testing (2D)             |
| `DepthStencilState::DEPTH_LESS`       | Standard 3D (closer wins)         |
| `DepthStencilState::DEPTH_READ_ONLY`  | Test but don't write (transparent)|

## Depth testing with a depth target

For a complete 3D scene, create a depth texture and attach it to the render pass:

```rust
let color = gpu.render_target(800, 600, Format::RGBA8)?;
let depth = gpu.create_texture(&TextureDesc {
    width: 800,
    height: 600,
    format: Format::Depth32Float,
    usage: TextureUsage::RENDER_TARGET,
    ..TextureDesc::default()
})?;

let mut pass = gpu.render_begin(&color)?;
pass.set_color_targets(vec![ColorTarget {
    texture: color.handle(),
    load_op: LoadOp::Clear(Color::BLACK),
    store_op: StoreOp::Store,
}]);
pass.set_depth_target(DepthTarget {
    texture: depth.handle(),
    load_op: LoadOp::Clear(Color::rgba(1.0, 0.0, 0.0, 0.0)),
    store_op: StoreOp::DontCare,
    stencil_load_op: LoadOp::DontCare,
    stencil_store_op: StoreOp::DontCare,
});
pass.set_viewport(0.0, 0.0, 800.0, 600.0);
pass.set_pipeline(&pipeline);
pass.bind_vertices(0, &vertices);
pass.bind_indices(&indices);
pass.draw_indexed(36); // 12 triangles
```

## Textured rendering

Bind a texture and sampler for fragment shader access:

```rust
pass.set_texture(0, &albedo_texture);
pass.set_sampler(0, SamplerDesc::default());
pass.draw(6);
```

See [textures](06-textures.md) for `sample()` usage in fragment shaders.

## Viewport and scissor

```rust
pass.set_viewport(0.0, 0.0, 800.0, 600.0);      // NDC mapping
pass.set_scissor(100, 100, 600, 400);            // pixel clipping rect
```

## Next

- [Vertex and fragment shaders](08-vertex-fragment.md) -- writing shader code
- [Device functions](09-device-functions.md) -- reusable GPU helpers
