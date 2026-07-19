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
- Value parameters (`pos: Vec3`) are vertex attributes, read from vertex
  buffers. They are pure inputs -- nothing is auto-forwarded to the fragment
  stage.
- Reference parameters (`mvp: &Mat4`) are uniform buffer bindings.

**Return type**: two forms.
- `-> Vec4` -- a *position-only* vertex: the tail expression is the clip-space
  position and the shader has **no** varyings (pair it with a fragment that
  takes no varying struct).
- `-> MyVaryings` -- to hand interpolated data to the fragment stage, return a
  [`#[derive(quanta::Varyings)]`](#varyings-the-shared-vertexfragment-interface)
  struct instead. Its `#[position]` field becomes the clip-space position; its
  other fields are the varyings.

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
fn solid(tint: &Vec4) -> Vec4 {
    *tint
}
```

**Parameters**:
- The interpolated **varyings** arrive as a single
  [`#[derive(quanta::Varyings)]`](#varyings-the-shared-vertexfragment-interface)
  struct parameter (the same struct the vertex returns); read each varying by
  field, `s.uv`. A fragment with no varyings, like `solid` above, omits the
  struct.
- `&T` / `&Sampled2D` / `&[T]` reference parameters are uniforms, sampled
  textures, and storage-buffer slices -- they stay separate parameters, they
  are **not** part of the varying struct.

A plain value parameter (e.g. `uv: Vec2`) is now a compile error -- fragment
stage inputs come from the varying struct, not from positional parameters.

**Return type**: Must be `Vec4` -- the output color (RGBA).

## Varyings: the shared vertex↔fragment interface

The vertex stage outputs values; the rasterizer interpolates them across the
triangle; the fragment stage reads them back. Quanta makes that contract
**explicit and shared**: you declare it once, in a struct that carries
`#[derive(quanta::Varyings)]`, and both stages name that one struct. This is
the WGSL / HLSL model -- there is no positional "auto-forward".

```rust
use quanta::{Vec2, Vec3, Vec4};

// Declare the interface ONCE, before the shaders that use it.
#[derive(quanta::Varyings)]
struct Surface {
    #[position] clip: Vec4, // gl_Position — the vertex writes it
    uv: Vec2,               // Location 0 (field-declaration order)
    color: Vec3,            // Location 1
}

#[quanta::vertex]
fn my_vertex(pos: Vec3, in_uv: Vec2, in_color: Vec3) -> Surface {
    Surface {
        clip: Vec4::new(pos.x, pos.y, pos.z, 1.0),
        uv: in_uv,
        color: in_color,
    }
}

#[quanta::fragment]
fn my_fragment(s: Surface) -> Vec4 {
    Vec4::new(s.color.x * s.uv.x, s.color.y * s.uv.y, s.color.z, 1.0)
}
```

The rules the derive enforces:

- **Exactly one `#[position]` field**, of type `Vec4`. It becomes
  `gl_Position` (`[[position]]` on Metal, `@builtin(position)` on WGSL). The
  vertex writes it; it is not a varying.
- **Every other field is a varying.** Supported types: `f32`, `u32`, `Vec2`,
  `Vec3`, `Vec4`. Each is assigned `Location 0, 1, …` in **field-declaration
  order** -- reorder the fields and you reorder the locations.
- **`u32` varyings are flat-interpolated automatically** (integers cannot be
  perspective-interpolated). See the [`u32` shader
  type](../../reference/shader-language.md#the-u32-scalar-type).
- **A fragment may read the `#[position]` field** -- it yields the interpolated
  **window-space** position (WGSL semantics: `x`/`y` in pixels), the same value
  [`frag_coord()`](../../reference/shader-language.md#frag_coord) returns.
- **Declaration order matters:** the `#[derive(quanta::Varyings)]` struct must
  appear **before** the shader functions that use it (a proc macro can only see
  its own item, so the interface reaches the shader macros through a generated
  trampoline that follows normal name scoping). If the struct lives in another
  module, import it *and* its generated `__quanta_varyings_<Name>` macro
  together.

A *position-only* vertex needs no struct: return `-> Vec4` and pair it with a
fragment that takes no varying struct (see [Your first
triangle](first-triangle.md)).

The derive also emits two introspection consts on the struct --
`Surface::POSITION_FIELD` and `Surface::VARYING_FIELDS` (the `(name, type)`
pairs in Location order) -- so host code and tests can see the interface
without re-parsing it.

## Coordinate conventions

Quanta normalizes render orientation across every backend: **the same
shader source produces the same pixels everywhere.** The convention is

- clip-space (NDC) **+Y points up**,
- the framebuffer **origin is top-left**,
- texture coordinates run **+Y down** (v = 0 is the top row),
- readback **row 0 is the top row** of the image.

Metal and WebGPU/WGSL already behave this way natively. Vulkan — whose
default NDC is y-down — conforms internally via a negative-viewport
y-flip, so a vertically asymmetric draw (a textured quad, a gradient)
lands identically on Metal, Vulkan, and WebGPU. Horizontally the backends
have always agreed.

If you are migrating code that flipped `uv.y` or negated a projection row
to compensate for the old per-backend divergence, **delete that
compensation** — it now double-flips. Author your geometry once against the
convention above and every backend matches.

## Texture parameters

A fragment shader samples textures through `&Sampled2D` parameters and the
`sample` intrinsic. The UV comes from the varying struct (`s.uv`); the texture
stays a separate parameter (reusing the `Surface` interface from above):

```rust
#[quanta::fragment]
fn glyph(s: Surface, atlas: &Sampled2D) -> Vec4 {
    let texel = sample(atlas, s.uv);
    Vec4::new(1.0, 1.0, 1.0, texel.x)
}
```

- Texture slots follow declaration order among texture params: the first
  `&Sampled2D` is slot 0, the second slot 1, and so on (at most 8).
- Bind at draw time with the matching slot:
  `.texture(0, &atlas).sampler(0, SamplerDesc::default())`. Every texture
  gets its own sampler at the same slot number.
- `sample(param, uv)` returns `Vec4`; for single-channel formats (`R8`
  glyph atlases) read `.x`.
- Texture params are fragment-only; a `&Sampled2D` in a vertex shader is a
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
fn animated(pos: Vec3, mvp: &Mat4, time: &f32) -> Vec4 {
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
// `s.uv` is a varying (from the `Surface` interface above); `stops` is the slice.
#[quanta::fragment]
fn gradient(s: Surface, stops: &[Vec4]) -> Vec4 {
    // Pick one of four colour stops by the horizontal coordinate.
    let idx = if s.uv.x < 0.25 { 0.0 }
              else { if s.uv.x < 0.5 { 1.0 }
              else { if s.uv.x < 0.75 { 2.0 } else { 3.0 } } };
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

The index is truncated to an integer, so `stops[s.uv.x * 4.0]` selects stop
`floor(s.uv.x * 4.0)`. Bounds are unchecked (the GPU storage-buffer contract),
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
| `u32`  | 1         | Unsigned integer (attribute, uniform, or flat varying) |
| `i32`  | 1         | Signed integer                 |
| `&Sampled2D` | -   | Sampled texture (fragment param only) |

`u32` is a real unsigned-integer scalar: use it for integer vertex attributes,
flat-interpolated varyings, and real comparisons (`kind == 3u32`, `k < 2`) --
no more smuggling an integer through an `f32` and testing `> 0.5`. The window
builtins (`frag_coord`, `vertex_id`, `instance_id`) and bounded `for` loops
also live in the shader body -- all of them are documented in the [shader
language reference](../../reference/shader-language.md).

## Example: coloured triangle with MVP

The vertex feeds a per-vertex colour to the fragment through the shared
`Tri` varying interface:

```rust
use quanta::*;

#[repr(C)]
#[derive(Copy, Clone, quanta::Vertex)]
struct ColorVertex {
    pos: [f32; 3],
    color: [f32; 3],
}

// The vertex↔fragment interface, declared before both shaders.
#[derive(quanta::Varyings)]
struct Tri {
    #[position] clip: Vec4, // gl_Position
    color: Vec3,            // Location 0
}

#[quanta::vertex]
fn vertex_main(pos: Vec3, in_color: Vec3, mvp: &Mat4) -> Tri {
    Tri {
        clip: mvp * Vec4::new(pos.x, pos.y, pos.z, 1.0),
        color: in_color,
    }
}

#[quanta::fragment]
fn fragment_main(s: Tri) -> Vec4 {
    Vec4::new(s.color.x, s.color.y, s.color.z, 1.0)
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

Fragment shaders sample textures through a `&Sampled2D` parameter and
`sample(param, uv)`. The macro rewrites the parameter to the texture slot
it occupies (declaration order among texture params), so the shader body
never names a raw slot number. The UV is a varying read from the interface
struct (`s.uv`); the texture is a separate parameter:

```rust
#[derive(quanta::Varyings)]
struct Uv {
    #[position] clip: Vec4,
    uv: Vec2,               // Location 0
}

#[quanta::vertex]
fn uv_vertex(pos: Vec3, in_uv: Vec2) -> Uv {
    Uv { clip: Vec4::new(pos.x, pos.y, pos.z, 1.0), uv: in_uv }
}

#[quanta::fragment]
fn textured(s: Uv, albedo: &Sampled2D) -> Vec4 {
    sample(albedo, s.uv)
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

## Builtins, `u32`, and loops

The shader body has a few more tools, all covered in depth in the [shader
language reference](../../reference/shader-language.md):

- **`frag_coord()`** (fragment only) -- the window-space position `Vec4`
  (`x`/`y` in pixels, `z` = depth, `w` = `1/w`). Equivalent to reading the
  `#[position]` field of the varying struct.
- **`vertex_id()` / `instance_id()`** (vertex only) -- the current vertex and
  instance index as `u32`. They let a shader **synthesize geometry with no
  vertex buffer** -- e.g. a fullscreen triangle from three vertices.
- **The `u32` scalar** -- unsigned attributes, flat varyings, and real integer
  comparisons.
- **Bounded `for` loops** -- `for i in 0..N { … }` where `N` is a compile-time
  constant integer literal and the counter `i` is `u32`. A non-constant bound
  is a hard error (a shader loop must always terminate).

> The WebGPU/WGSL backend does not yet emit these four (`frag_coord`, `u32`
> varyings, `vertex_id`/`instance_id`, shader `for` loops): a shader using them
> ships with `wgsl: None` and a build-time note, matching Quanta's "WebGPU is
> largely `NotSupported`" posture. Metal and Vulkan are the supported render
> backends.

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
