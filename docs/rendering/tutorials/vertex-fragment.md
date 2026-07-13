# Vertex and Fragment Shaders

Render pipeline shaders are written as annotated Rust functions, just like
compute kernels. The proc macro compiles them to SPIR-V (Vulkan) and metallib
(Metal) at build time. Both platforms get native binary shaders -- no runtime
compilation needed.

## Vertex derive

Define vertex layout with `#[derive(quanta::Vertex)]`:

```rust
use quanta::*;

#[repr(C)]
#[derive(Copy, Clone, quanta::Vertex)]
struct Vertex {
    pos: [f32; 3],     // location 0, Float3
    normal: [f32; 3],  // location 1, Float3
    uv: [f32; 2],      // location 2, Float2
}
```

The derive macro:
- Maps each field to a `VertexAttribute` with correct location, offset, and format
- Computes stride from `size_of::<Self>()`
- Generates `Vertex::vertex_layout()` for use in `PipelineDesc`

Supported field types:

| Type         | Format          | Size  |
|-------------|-----------------|-------|
| `f32`       | `Float`         | 4 B   |
| `[f32; 2]`  | `Float2`        | 8 B   |
| `[f32; 3]`  | `Float3`        | 12 B  |
| `[f32; 4]`  | `Float4`        | 16 B  |
| `u32`       | `UInt`          | 4 B   |
| `[u32; 2]`  | `UInt2`         | 8 B   |
| `[u32; 3]`  | `UInt3`         | 12 B  |
| `[u32; 4]`  | `UInt4`         | 16 B  |
| `i32`       | `Int`           | 4 B   |
| `[i32; 2]`  | `Int2`          | 8 B   |
| `[i32; 3]`  | `Int3`          | 12 B  |
| `[i32; 4]`  | `Int4`          | 16 B  |

## Vertex shaders

A vertex shader transforms vertex positions into clip space. Mark with
`#[quanta::vertex]`:

```rust
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

## Varyings: vertex outputs to fragment inputs

The first vertex parameter is always the position (goes to clip-space output).
All remaining parameters are automatically forwarded to the fragment shader as
interpolated varyings, matched **by name**:

```rust
#[quanta::vertex]
fn my_vertex(pos: Vec3, uv: Vec2, color: Vec3) -> Vec4 {
    Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

// Fragment receives uv and color, interpolated across the triangle surface.
// Names must match the vertex shader's non-position parameters.
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

## Coordinate conventions

Clip-space y points UP on Metal and DOWN on Vulkan, so vertically
asymmetric output (a textured quad, a gradient) renders flipped between
the two backends. Quanta does not currently normalize this — the
convention is the app's (flip `uv.y` or your projection on one backend
if you need identical output). Horizontally the backends agree.

## Texture parameters

A fragment shader samples textures through `&Texture2D` parameters and the
`sample` intrinsic:

```rust
#[quanta::fragment]
fn glyph(uv: Vec2, atlas: &Texture2D) -> Vec4 {
    let texel = sample(atlas, uv);
    Vec4::new(1.0, 1.0, 1.0, texel.x)
}
```

- Texture slots follow declaration order among texture params: the first
  `&Texture2D` is slot 0, the second slot 1, and so on (at most 8).
- Bind at draw time with the matching slot:
  `.texture(0, &atlas).sampler(0, SamplerDesc::default())`. Every texture
  gets its own sampler at the same slot number.
- `sample(param, uv)` returns `Vec4`; for single-channel formats (`R8`
  glyph atlases) read `.x`.
- Texture params are fragment-only; a `&Texture2D` in a vertex shader is a
  compile error.

## Uniforms derive

For uniform data shared across all vertices/fragments, use
`#[derive(quanta::Uniforms)]`:

```rust
#[repr(C)]
#[derive(Copy, Clone, quanta::Uniforms)]
struct Camera {
    view: [f32; 16],      // mat4x4
    proj: [f32; 16],      // mat4x4
    eye_pos: [f32; 3],    // vec3
    fov: f32,
}
```

The derive generates:
- `Camera::GPU_SIZE` -- byte size of the struct
- `Camera::GPU_FIELDS` -- `(name, type, byte_offset)` metadata
- `impl GpuType for Camera` -- enables use in fields
- MSL and WGSL struct declarations

Bind uniforms through the render builder:

```rust
let camera_buf = gpu.field_with_usage::<Camera>(1, FieldUsage::default_uniform())?;
camera_buf.write(&[camera_data])?;

gpu.render(&target)?
    .pipeline(&pipeline)
    .vertices(0, &vb)
    .uniform(0, &camera_buf)
    .draw(vertex_count)
    .pulse()?
    .wait()?;
```

Or in the vertex shader as a reference parameter:

```rust
#[quanta::vertex]
fn animated(pos: Vec3, normal: Vec3, mvp: &Mat4, time: &f32) -> Vec4 {
    let offset = Vec3::new(0.0, sin(*time) * 0.1, 0.0);
    mvp * Vec4::new(pos.x + offset.x, pos.y + offset.y, pos.z + offset.z, 1.0)
}
```

## Array params

When a shader needs a *table* of values -- gradient stops, a palette LUT,
more colour stops than you want to spend varyings on -- take a `&[T]` slice
parameter instead. The element type is `&[f32]`, `&[Vec2]`, or `&[Vec4]`,
and the body reads it with `name[index]`:

```rust
#[quanta::fragment]
fn gradient(uv: Vec2, stops: &[Vec4]) -> Vec4 {
    // Pick one of four colour stops by the horizontal coordinate.
    let idx = if uv.x < 0.25 { 0.0 }
              else { if uv.x < 0.5 { 1.0 }
              else { if uv.x < 0.75 { 2.0 } else { 3.0 } } };
    stops[idx]
}
```

A slice is backed by a storage buffer and binds with the same `.uniform`
call as a `&T` uniform -- slices and uniforms share one slot space, counted
in declaration order:

```rust
let stops = gpu.field_with_usage::<f32>(16, FieldUsage::default_render())?;
stops.write(&[
    1.0, 0.0, 0.0, 1.0,   // stop 0 -- red
    0.0, 1.0, 0.0, 1.0,   // stop 1 -- green
    0.0, 0.0, 1.0, 1.0,   // stop 2 -- blue
    1.0, 1.0, 1.0, 1.0,   // stop 3 -- white
])?;

gpu.render(&target)?
    .pipeline(&pipeline)
    .vertices(0, &vb)
    .uniform(0, &stops)   // the &[Vec4] param at slot 0
    .draw(vertex_count)
    .pulse()?
    .wait()?;
```

The index is truncated to an integer, so `stops[uv.x * 4.0]` selects stop
`floor(uv.x * 4.0)`. Bounds are unchecked (the GPU storage-buffer contract),
and at most 8 combined uniform + slice params are allowed -- texture bindings
occupy the slots above that.

## Supported shader types

| Type   | Components | Description                    |
|--------|-----------|--------------------------------|
| `f32`  | 1         | Scalar float                   |
| `Vec2` | 2         | 2D vector (x, y)              |
| `Vec3` | 3         | 3D vector (x, y, z)           |
| `Vec4` | 4         | 4D vector (x, y, z, w)        |
| `Mat4` | 16        | 4x4 matrix (column-major)     |
| `u32`  | 1         | Unsigned integer               |
| `i32`  | 1         | Signed integer                 |
| `&Texture2D` | -   | Sampled texture (fragment param only) |

## Example: rotating triangle with MVP

```rust
use quanta::*;

#[repr(C)]
#[derive(Copy, Clone, quanta::Vertex)]
struct ColorVertex {
    pos: [f32; 3],
    color: [f32; 3],
}

#[quanta::vertex]
fn vertex_main(pos: Vec3, color: Vec3, mvp: &Mat4) -> Vec4 {
    mvp * Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

#[quanta::fragment]
fn fragment_main(color: Vec3) -> Vec4 {
    Vec4::new(color.x, color.y, color.z, 1.0)
}

fn render_frame(gpu: &Gpu, angle: f32) -> Result<(), QuantaError> {
    let vertices = [
        ColorVertex { pos: [ 0.0,  0.5, 0.0], color: [1.0, 0.0, 0.0] },  // red
        ColorVertex { pos: [-0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0] },  // green
        ColorVertex { pos: [ 0.5, -0.5, 0.0], color: [0.0, 0.0, 1.0] },  // blue
    ];

    let vb = gpu.field_with_usage::<ColorVertex>(3, FieldUsage::default_render())?;
    vb.write(&vertices)?;

    // Build MVP matrix (rotation around Z axis)
    let cos = angle.cos();
    let sin = angle.sin();
    let mvp: [f32; 16] = [
        cos, sin, 0.0, 0.0,
       -sin, cos, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    let mvp_buf = gpu.field_with_usage::<[f32; 16]>(1, FieldUsage::default_uniform())?;
    mvp_buf.write(&[mvp])?;

    // ShaderSource::Binaries hands the driver the multi-vendor statics
    // the macros generated; it picks the right payload per backend.
    let layouts = [ColorVertex::vertex_layout()];
    let pipeline = gpu.pipeline(
        &PipelineDesc::new(ShaderSource::Binaries {
            vertex: &VERTEX_MAIN_SHADER,
            fragment: &FRAGMENT_MAIN_SHADER,
        })
        .with_entries("vertex_main", "fragment_main")
        .with_vertex_layouts(&layouts)
        .with_color_formats(vec![Format::BGRA8]),
    )?;

    let target = gpu.render_target(800, 600, Format::BGRA8)?;

    gpu.render(&target)?
        .clear(Color::BLACK)
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .uniform(0, &mvp_buf)
        .draw(3)
        .pulse()?
        .wait()?;

    Ok(())
}
```

`with_color_formats` is per-attachment: `color_formats[i]` types color
attachment `i` of the pass, so its length must match the number of color
targets the pass binds — here one `BGRA8` format for the one `BGRA8`
target. It is not a list of formats the pipeline may be used against; a
mismatch is caught when the pass is submitted.

## Instanced rendering

Use `StepMode::Instance` for per-instance data. With the Vertex derive,
define separate structs for per-vertex and per-instance data:

```rust
#[repr(C)]
#[derive(Copy, Clone, quanta::Vertex)]
struct MeshVertex {
    pos: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, quanta::Vertex)]
struct InstanceData {
    offset: [f32; 4],
}

// Override the step mode for the instance buffer layout:
let layouts = vec![
    MeshVertex::vertex_layout(),
    {
        let mut layout = InstanceData::vertex_layout();
        layout.step = StepMode::Instance;
        layout
    },
];

gpu.render(&target)?
    .pipeline(&pipeline)
    .vertices(0, &mesh_vb)
    .vertices(1, &instance_vb)
    .draw_instanced(36, 1000)
    .pulse()?
    .wait()?;
```

## Texture sampling

Fragment shaders sample textures through a `&Texture2D` parameter and
`sample(param, uv)`. The macro rewrites the parameter to the texture slot
it occupies (declaration order among texture params), so the shader body
never names a raw slot number:

```rust
#[quanta::fragment]
fn textured(uv: Vec2, albedo: &Texture2D) -> Vec4 {
    sample(albedo, uv)
}
```

Bind the texture and its sampler at the matching slot through the builder:

```rust
gpu.render(&target)?
    .pipeline(&pipeline)
    .vertices(0, &vb)
    .texture(0, &albedo_texture)
    .sampler(0, SamplerDesc::default().with_filters(Filter::Linear, Filter::Linear))
    .draw(6)
    .pulse()?
    .wait()?;
```

`sample()` returns `Vec4` (RGBA).

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
| Texture sample | `sample(albedo, uv)` |
| Conditionals | `if x > 0.5 { a } else { b }` |

Math functions (30 total): `sin`, `cos`, `tan`, `asin`, `acos`, `atan`,
`sqrt`, `abs`, `floor`, `ceil`, `round`, `fract`, `min`, `max`, `clamp`,
`mix`, `step`, `smoothstep`, `pow`, `exp`, `log`, `exp2`, `log2`,
`normalize`, `length`, `distance`, `cross`, `fma`, `atan2`, `inverse_sqrt`.

### Grammar the parser accepts

The shader body is a small Rust subset. A few conveniences are allowed
beyond the bare minimum, on both backends:

- **Trailing commas** in constructor calls: `Vec4::new(x, y, z, w,)`.
- **Calls split across lines**: a `Vec4::new(...)` (or any call) may wrap
  onto the next line — whitespace, including newlines, is not significant.
- **Branch-local `let`s**: a `let` declared inside an `if` branch is scoped
  to that branch and never escapes it.

### One limitation: outer-local assignment in an `if`-expression

Assigning to an *outer* local from inside an `if`-**expression** branch does
not compile — the branch runs on a copy of the surrounding locals, so the
write would silently vanish at the merge. Use a statement-level `if` (which
does write the outer local back through its merge) instead:

```rust
// Rejected: `acc` is assigned inside an if-EXPRESSION branch.
let mut acc = 0.0;
let v = if uv.x > 0.5 { acc = 1.0; uv.x } else { uv.y };

// Works: statement-level `if` mutates the outer local.
let mut acc = 0.0;
if uv.x > 0.5 { acc = 1.0; }
```

`if`-expressions used for a value must have an `else` branch (both backends
require it, so code stays portable).

## Other shader stages

`#[quanta::vertex]` and `#[quanta::fragment]` are the two stages most projects
need. v0.1 also ships these stage attributes for advanced pipelines:

| Macro                      | Stage                                            |
|----------------------------|--------------------------------------------------|
| `#[quanta::tess_control]`  | Tessellation control (M4.1)                      |
| `#[quanta::tess_eval]`     | Tessellation evaluation (M4.1)                   |
| `#[quanta::task]`          | Task / amplification shader (M4.2 mesh pipeline) |
| `#[quanta::mesh]`          | Mesh shader (M4.2)                               |
| `#[quanta::ray_gen]`       | Ray generation (M4.3)                            |
| `#[quanta::closest_hit]`   | Closest-hit shader (M4.3)                        |
| `#[quanta::miss]`          | Miss shader (M4.3)                               |

Each is gated by a capability query — see the corresponding chapter
([Tessellation](tessellation.md), [Mesh shaders](mesh-shaders.md),
[Ray tracing](ray-tracing.md)) for the typed pipeline that consumes them.

## Next

- [Device functions](../../computation/tutorials/device-functions.md) -- reusable GPU helper functions
- [Expert: Manual API](../../expert/manual-api.md) -- manual binding, raw handles
