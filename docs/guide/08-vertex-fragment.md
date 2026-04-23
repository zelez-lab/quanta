# Vertex and Fragment Shaders

Render pipeline shaders are written as annotated Rust functions, just like
compute kernels. The proc macro compiles them to SPIR-V (Vulkan) and metallib
(Metal) at build time. Both platforms get native binary shaders -- no runtime
compilation needed.

## Vertex shaders

A vertex shader transforms vertex positions into clip space. Mark with
`#[quanta::vertex]`:

```rust
use quanta::*;

#[quanta::vertex]
fn transform(pos: Vec3, mvp: &Mat4) -> Vec4 {
    mvp * Vec4::new(pos.x, pos.y, pos.z, 1.0)
}
```

**Parameters**:
- Value parameters (`pos: Vec3`) are vertex attributes, read from vertex buffers.
- Reference parameters (`mvp: &Mat4`) are uniform buffer bindings.

**Return type**: Must be `Vec4` -- the clip-space position.

The macro generates a `ShaderBinary` static and a function that returns a
reference to it:

```rust
// Generated:
pub static TRANSFORM_SHADER: ShaderBinary = ...;
pub fn transform() -> &'static ShaderBinary { &TRANSFORM_SHADER }
```

## Fragment shaders

A fragment shader computes the color of each pixel. Mark with
`#[quanta::fragment]`:

```rust
#[quanta::fragment]
fn shade(uv: Vec2, color: Vec4) -> Vec4 {
    color * Vec4::new(uv.x, uv.y, 0.0, 1.0)
}
```

**Parameters**: Interpolated values from the rasterizer (varyings). What the
vertex shader outputs, the fragment shader receives interpolated across the
triangle surface.

**Return type**: Must be `Vec4` -- the output color (RGBA).

## Supported types

Types available in vertex and fragment shaders:

| Type   | Components | Description                    |
|--------|-----------|--------------------------------|
| `f32`  | 1         | Scalar float                   |
| `Vec2` | 2         | 2D vector (x, y)              |
| `Vec3` | 3         | 3D vector (x, y, z)           |
| `Vec4` | 4         | 4D vector (x, y, z, w)        |
| `Mat4` | 16        | 4x4 matrix (column-major)     |
| `u32`  | 1         | Unsigned integer               |
| `i32`  | 1         | Signed integer                 |

Vectors support standard operations: `+`, `-`, `*`, `/`, component access (`.x`,
`.y`, `.z`, `.w`), construction (`Vec4::new(...)`).

## Uniforms

Reference parameters are bound as uniform buffers. The CPU writes transform
matrices, time values, etc. to a uniform field, and the shader reads them:

```rust
#[quanta::vertex]
fn animated(pos: Vec3, normal: Vec3, mvp: &Mat4, time: &f32) -> Vec4 {
    let offset = Vec3::new(0.0, sin(*time) * 0.1, 0.0);
    mvp * Vec4::new(pos.x + offset.x, pos.y + offset.y, pos.z + offset.z, 1.0)
}
```

On the CPU side, bind the uniform buffer:

```rust
let mvp_buffer = gpu.uniform_field::<[f32; 16]>(1)?;
gpu.write_field(&mvp_buffer, &[mvp_matrix])?;

let mut pass = gpu.render_begin(&target)?;
pass.set_pipeline(&pipeline);
pass.bind_vertices(0, &vertex_buffer);
pass.set_uniform(0, &mvp_buffer);
pass.draw(vertex_count);
```

## Example: rotating triangle with MVP

```rust
use quanta::*;

#[quanta::vertex]
fn vertex_main(pos: Vec3, color: Vec3, mvp: &Mat4) -> Vec4 {
    mvp * Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

#[quanta::fragment]
fn fragment_main(color: Vec3) -> Vec4 {
    Vec4::new(color.x, color.y, color.z, 1.0)
}

fn render_frame(gpu: &Gpu, angle: f32) -> Result<(), QuantaError> {
    // Triangle vertices: position + color interleaved
    let vertices: Vec<f32> = vec![
        // pos.x, pos.y, pos.z, color.r, color.g, color.b
         0.0,  0.5, 0.0,   1.0, 0.0, 0.0,  // top (red)
        -0.5, -0.5, 0.0,   0.0, 1.0, 0.0,  // left (green)
         0.5, -0.5, 0.0,   0.0, 0.0, 1.0,  // right (blue)
    ];

    let verts = gpu.render_field::<f32>(18)?;
    gpu.write_field(&verts, &vertices)?;

    // Build MVP matrix (rotation around Z axis)
    let cos = angle.cos();
    let sin = angle.sin();
    let mvp: [f32; 16] = [
        cos, sin, 0.0, 0.0,
       -sin, cos, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    let mvp_buf = gpu.uniform_field::<[f32; 16]>(1)?;
    gpu.write_field(&mvp_buf, &[mvp])?;

    let pipeline = gpu.pipeline(&PipelineDesc {
        vertex: vertex_main().for_vendor(gpu.caps().vendor).unwrap(),
        fragment: fragment_main().for_vendor(gpu.caps().vendor).unwrap(),
        vertex_entry: "vertex_main",
        fragment_entry: "fragment_main",
        vertex_layouts: &[VertexLayout {
            stride: 24, // 6 floats * 4 bytes
            step: StepMode::Vertex,
            attributes: vec![
                VertexAttribute { location: 0, offset: 0, format: AttributeFormat::Float3 },
                VertexAttribute { location: 1, offset: 12, format: AttributeFormat::Float3 },
            ],
        }],
        color_formats: vec![Format::BGRA8],
        ..PipelineDesc::default()
    })?;

    let target = gpu.render_target(800, 600, Format::BGRA8)?;
    let mut pass = gpu.render_begin(&target)?;
    pass.clear(Color::BLACK);
    pass.set_pipeline(&pipeline);
    pass.bind_vertices(0, &verts);
    pass.set_uniform(0, &mvp_buf);
    pass.draw(3);
    let mut pulse = gpu.render_end(pass)?;
    gpu.wait(&mut pulse)?;

    Ok(())
}
```

## Instanced rendering

Use `StepMode::Instance` for per-instance data (transforms, colors):

```rust
let pipeline = gpu.pipeline(&PipelineDesc {
    vertex_layouts: &[
        VertexLayout {
            stride: 12,
            step: StepMode::Vertex,
            attributes: vec![
                VertexAttribute { location: 0, offset: 0, format: AttributeFormat::Float3 },
            ],
        },
        VertexLayout {
            stride: 16,
            step: StepMode::Instance,
            attributes: vec![
                VertexAttribute { location: 1, offset: 0, format: AttributeFormat::Float4 },
            ],
        },
    ],
    ..PipelineDesc::default()
})?;

let mut pass = gpu.render_begin(&target)?;
pass.set_pipeline(&pipeline);
pass.bind_vertices(0, &mesh_vertices);
pass.bind_vertices(1, &instance_data);
pass.draw_instanced(36, 1000); // 36 verts * 1000 instances
```

## Vertex attribute formats

| Format                    | Components | Size  |
|---------------------------|-----------|-------|
| `AttributeFormat::Float`  | 1 f32    | 4 B   |
| `AttributeFormat::Float2` | 2 f32    | 8 B   |
| `AttributeFormat::Float3` | 3 f32    | 12 B  |
| `AttributeFormat::Float4` | 4 f32    | 16 B  |
| `AttributeFormat::Int`    | 1 i32    | 4 B   |
| `AttributeFormat::Int2`   | 2 i32    | 8 B   |
| `AttributeFormat::Int3`   | 3 i32    | 12 B  |
| `AttributeFormat::Int4`   | 4 i32    | 16 B  |
| `AttributeFormat::UInt`   | 1 u32    | 4 B   |
| `AttributeFormat::UByte4Norm` | 4 u8 (normalized) | 4 B |

## Vertex-to-fragment varyings

The first vertex parameter is always the position (goes to `gl_Position`).
All remaining parameters are automatically forwarded to the fragment shader as
interpolated varyings:

```rust
#[quanta::vertex]
fn my_vertex(pos: Vec3, uv: Vec2, color: Vec3) -> Vec4 {
    Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

// Fragment receives uv at Location 0, color at Location 1
#[quanta::fragment]
fn my_fragment(uv: Vec2, color: Vec3) -> Vec4 {
    Vec4::new(color.x * uv.x, color.y * uv.y, color.z, 1.0)
}
```

Convention:
- Vertex param 0 = position (not forwarded)
- Vertex param 1 = fragment input at Location 0
- Vertex param 2 = fragment input at Location 1
- ...

## Texture sampling

Fragment shaders can sample textures using the `sample(slot, uv)` function.
The slot number matches the `set_texture(slot, ...)` call in the render pass:

```rust
#[quanta::fragment]
fn textured(uv: Vec2) -> Vec4 {
    sample(0, uv)
}
```

On the CPU side, bind the texture and sampler:

```rust
let mut pass = gpu.render_begin(&target)?;
pass.set_pipeline(&pipeline);
pass.bind_vertices(0, &vertices);
pass.set_texture(0, &albedo_texture);
pass.set_sampler(0, SamplerDesc {
    min_filter: Filter::Linear,
    mag_filter: Filter::Linear,
    ..SamplerDesc::default()
});
pass.draw(6);
```

`sample()` returns `Vec4` (RGBA). Combine with varyings for UV-mapped rendering:

```rust
#[quanta::vertex]
fn uv_vertex(pos: Vec3, uv: Vec2) -> Vec4 {
    Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

#[quanta::fragment]
fn uv_textured(uv: Vec2) -> Vec4 {
    sample(0, uv)
}
```

## Shader expression support

The shader body supports these operations on both Metal and Vulkan:

| Feature | Example |
|---------|---------|
| Arithmetic | `a + b`, `a * b`, `a / b`, `a - b` |
| Negation | `-x` |
| Field access | `pos.x`, `color.rgb` |
| Vec constructors | `Vec4::new(x, y, z, w)` |
| Let bindings | `let t = a * 0.5;` |
| Math functions | `sin(x)`, `cos(x)`, `sqrt(x)`, `clamp(x, 0.0, 1.0)` |
| Matrix multiply | `mvp * vec4` |
| Texture sample | `sample(0, uv)` |
| Conditionals | `if x > 0.5 { a } else { b }` |

Math functions (30 total): `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sqrt`,
`abs`, `floor`, `ceil`, `round`, `fract`, `min`, `max`, `clamp`, `mix`, `step`,
`smoothstep`, `pow`, `exp`, `log`, `exp2`, `log2`, `normalize`, `length`,
`distance`, `cross`, `fma`, `atan2`, `inverseSqrt`.

## Next

- [Device functions](09-device-functions.md) -- reusable GPU helper functions
- [Advanced](10-advanced.md) -- barriers, profiling, multi-queue
