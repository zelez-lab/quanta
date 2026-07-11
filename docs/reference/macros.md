# Macro Reference

All proc macros and derive macros in Quanta. The three derive macros
(`Fields`, `Vertex`, `Uniforms`) are the recommended entry point for new code.

---

## Derive macros

### `#[derive(quanta::Fields)]`

Generate GPU dispatch metadata from a struct. Classifies each field as either
a GPU storage buffer (`Vec<T>`) or a push constant (scalar).

#### Syntax

```rust
#[derive(quanta::Fields)]
struct MyData {
    input: Vec<f32>,     // GPU storage buffer (slot 0)
    output: Vec<f32>,    // GPU storage buffer (slot 1)
    count: u32,          // push constant (slot 2)
    threshold: f32,      // push constant (slot 3)
}
```

#### What it generates

| Generated item | Type | Description |
|----------------|------|-------------|
| `FIELD_COUNT` | `usize` | Number of `Vec<T>` fields (GPU storage buffers) |
| `PUSH_CONSTANT_COUNT` | `usize` | Number of scalar fields (push constants) |
| `field_names()` | `&'static [&'static str]` | Names of Vec fields |
| `field_types()` | `&'static [&'static str]` | Element type names ("f32", "u32", etc.) |
| `push_constant_names()` | `&'static [&'static str]` | Names of scalar fields |
| `push_constant_types()` | `&'static [&'static str]` | Type names of scalar fields |

#### Supported field types

| Type | Classification | Notes |
|------|---------------|-------|
| `Vec<f32>` | GPU storage buffer | Most common for compute data |
| `Vec<u32>` | GPU storage buffer | Indices, counts, iteration data |
| `Vec<i32>` | GPU storage buffer | Signed integer data |
| `Vec<f64>` | GPU storage buffer | Double precision |
| `Vec<u8>` | GPU storage buffer | Byte data |
| `f32`, `f64` | Push constant | Per-dispatch scalar |
| `u32`, `i32` | Push constant | Per-dispatch integer |
| `u8`, `u16`, `i16` | Push constant | Small integers |
| `u64`, `i64` | Push constant | Large integers |
| `bool` | Push constant | Flag |
| `usize` | Push constant | Size/count |
| `[f32; N]` | Push constant | Fixed-size array (e.g., vec4 uniform) |

#### Constraints

- Only structs with named fields (no tuples, no enums)
- `Vec<T>` fields must use a supported element type
- Scalar fields must be GPU-compatible primitives

#### Example

```rust
#[derive(quanta::Fields)]
struct ScanData {
    input: Vec<f32>,
    output: Vec<f32>,
    count: u32,
    threshold: f32,
}

// Compile-time introspection:
assert_eq!(ScanData::FIELD_COUNT, 2);
assert_eq!(ScanData::PUSH_CONSTANT_COUNT, 2);
assert_eq!(ScanData::field_names(), &["input", "output"]);
assert_eq!(ScanData::field_types(), &["f32", "f32"]);
assert_eq!(ScanData::push_constant_names(), &["count", "threshold"]);
assert_eq!(ScanData::push_constant_types(), &["u32", "f32"]);
```

---

### `#[derive(quanta::Vertex)]`

Generate vertex layout metadata from a struct. Maps Rust types to GPU attribute
formats and computes byte offsets automatically.

#### Syntax

```rust
#[repr(C)]
#[derive(Copy, Clone, quanta::Vertex)]
struct MyVertex {
    position: [f32; 3],  // location 0, Float3
    normal: [f32; 3],    // location 1, Float3
    uv: [f32; 2],        // location 2, Float2
    color: [f32; 4],     // location 3, Float4
}
```

**Requires `#[repr(C)]`** on the struct (compile error otherwise). This
guarantees the field layout matches GPU expectations.

#### What it generates

| Generated item | Type | Description |
|----------------|------|-------------|
| `ATTRIBUTES` | `[VertexAttribute; N]` | Static array of attribute descriptors |
| `vertex_layout()` | `VertexLayout` | Complete layout with stride, step mode, attributes |

The `VertexLayout` returned by `vertex_layout()` contains:
- `stride`: `size_of::<Self>()` (total bytes per vertex)
- `step`: `StepMode::Vertex` (per-vertex, not per-instance)
- `attributes`: all attribute descriptors with location, offset, and format

#### Supported attribute types

| Rust type | Attribute format | Bytes |
|-----------|-----------------|-------|
| `f32` | `Float` | 4 |
| `[f32; 2]` | `Float2` | 8 |
| `[f32; 3]` | `Float3` | 12 |
| `[f32; 4]` | `Float4` | 16 |
| `u32` | `UInt` | 4 |
| `[u32; 2]` | `UInt2` | 8 |
| `[u32; 3]` | `UInt3` | 12 |
| `[u32; 4]` | `UInt4` | 16 |
| `i32` | `Int` | 4 |
| `[i32; 2]` | `Int2` | 8 |
| `[i32; 3]` | `Int3` | 12 |
| `[i32; 4]` | `Int4` | 16 |

#### Example

```rust
#[repr(C)]
#[derive(Copy, Clone, quanta::Vertex)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 4],
}

// Use in a pipeline (gpu.pipeline is a RenderGpu method —
// `use quanta::RenderGpu;` or `use quanta::*;`):
let layouts = [Vertex::vertex_layout()];
let pipeline = gpu.pipeline(
    &PipelineDesc::new(ShaderSource::Binaries {
        vertex: vs_main(),
        fragment: fs_main(),
    })
    .with_vertex_layouts(&layouts),
)?;

// Bind vertex data:
let verts = gpu.field_with_usage::<Vertex>(3, FieldUsage::default_render())?;
verts.write(&[
    Vertex { position: [0.0, 0.5, 0.0], color: [1.0, 0.0, 0.0, 1.0] },
    Vertex { position: [-0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0, 1.0] },
    Vertex { position: [0.5, -0.5, 0.0], color: [0.0, 0.0, 1.0, 1.0] },
])?;
```

---

### `#[derive(quanta::Uniforms)]`

Mark a struct as GPU-compatible for use as a uniform buffer. Generates byte-level
metadata, MSL/WGSL struct declarations, and a `GpuType` trait impl.

#### Syntax

```rust
#[repr(C)]
#[derive(Copy, Clone, quanta::Uniforms)]
struct Camera {
    view: [f32; 16],       // mat4x4
    projection: [f32; 16], // mat4x4
    position: [f32; 3],    // vec3
    fov: f32,
}
```

**Requires `#[repr(C)]`** on the struct (compile error otherwise).

#### What it generates

| Generated item | Type | Description |
|----------------|------|-------------|
| `GPU_SIZE` | `usize` | Byte size (`size_of::<Self>()`) |
| `GPU_FIELDS` | `&[(&str, &str, usize)]` | (name, type_string, byte_offset) per field |
| `impl GpuType` | trait impl | Enables `gpu.field_with_usage::<Self>(n, FieldUsage::default_uniform())` |
| `__QUANTA_UNIFORMS_*` | `&str` | MSL struct declaration (hidden) |
| `__QUANTA_UNIFORMS_*_WGSL` | `&str` | WGSL struct declaration (hidden) |

#### Supported field types

Same scalar and array types as `#[quanta::gpu_type]`:

| Rust type | MSL | WGSL |
|-----------|-----|------|
| `f32` | `float` | `f32` |
| `f64` | `double` | `f64` |
| `u32` | `uint` | `u32` |
| `i32` | `int` | `i32` |
| `bool` | `bool` | `bool` |
| `[f32; 2]` | `float2` | `vec2<f32>` |
| `[f32; 3]` | `float3` | `vec3<f32>` |
| `[f32; 4]` | `float4` | `vec4<f32>` |
| `[f32; 9]` | `float3x3` | `mat3x3<f32>` |
| `[f32; 16]` | `float4x4` | `mat4x4<f32>` |
| `[u32; 4]` | `uint4` | `vec4<u32>` |

#### Example

```rust
#[repr(C)]
#[derive(Copy, Clone, quanta::Uniforms)]
struct Light {
    position: [f32; 3],
    intensity: f32,
    color: [f32; 4],
}

// Allocate and write:
let light_buf = gpu.field_with_usage::<Light>(1, FieldUsage::default_uniform())?;
light_buf.write(&[Light {
    position: [10.0, 20.0, 5.0],
    intensity: 1.5,
    color: [1.0, 0.95, 0.9, 1.0],
}])?;

// Bind to a render pass (gpu.render is a RenderGpu method):
gpu.render(&target)?
    .pipeline(&pipeline)
    .uniform(0, &light_buf)
    .vertices(0, &verts)
    .draw(mesh_count)
    .pulse()?;

// Introspect:
assert_eq!(Light::GPU_SIZE, 32);
assert_eq!(Light::GPU_FIELDS[0], ("position", "[f32; 3]", 0));
assert_eq!(Light::GPU_FIELDS[1], ("intensity", "f32", 12));
assert_eq!(Light::GPU_FIELDS[2], ("color", "[f32; 4]", 16));
```

---

## Proc macros

### `#[quanta::kernel]`

Compile a Rust function into a GPU compute kernel.

#### Syntax

```rust
#[quanta::kernel]                              // Default: O3, driver-chosen workgroup
#[quanta::kernel(opt = "O2")]                  // Explicit optimization level
#[quanta::kernel(opt = "O0")]                  // No optimization (debug)
#[quanta::kernel(workgroup = [256, 1, 1])]     // 1D workgroup size
#[quanta::kernel(workgroup = [16, 16, 1])]     // 2D workgroup size
#[quanta::kernel(workgroup = [8, 8, 4])]       // 3D workgroup size
#[quanta::kernel(jit)]                         // JIT: serialize IR, compile at runtime
#[quanta::kernel(workgroup = [256, 1, 1], opt = "O2")]  // Combined attributes
fn name(params...) { body }
```

#### Attributes

| Attribute | Values | Default | Description |
|-----------|--------|---------|-------------|
| `opt` | `"O0"`, `"O1"`, `"O2"`, `"O3"` | `"O3"` | LLVM optimization level |
| `workgroup` | `[x, y, z]` | driver-chosen | Workgroup dimensions (1D/2D/3D) |
| `jit` | flag | off | Serialize KernelDef for runtime compilation |

#### Parameters

- `&[T]` -- read-only GPU buffer (bound at slot by declaration order)
- `&mut [T]` -- read-write GPU buffer
- Scalar values (`u32`, `f32`, etc.) -- push constants (set via `wave.set_value`)

#### Produces

Without `jit`:
- `static NAME_BINARY: KernelBinary` -- compiled native binaries (SPIR-V, metallib, PTX, GCN) for all backends
- `fn name(gpu: &Gpu) -> Result<Wave, QuantaError>` -- creates a bound wave

With `jit`:
- `static NAME_KERNEL_DEF: &[u8]` -- serialized KernelDef IR (compile at runtime via `gpu.wave_jit()`)
- `fn name(gpu: &Gpu) -> Result<Wave, QuantaError>` -- creates a bound wave (triggers runtime compilation)

#### Built-in functions available in kernel body

| Function | Returns | Description |
|----------|---------|-------------|
| `quark_id()` | `u32` | Global thread index |
| `quark_count()` | `u32` | Total dispatched quarks |
| `local_id()` | `u32` | Thread index within workgroup |
| `group_id()` | `u32` | Workgroup index |
| `group_size()` | `u32` | Workgroup size |
| `barrier()` | `()` | Workgroup synchronization |
| `atomic_add(dst, val)` | old value | Atomic add |
| `atomic_sub(dst, val)` | old value | Atomic subtract |
| `atomic_min(dst, val)` | old value | Atomic minimum |
| `atomic_max(dst, val)` | old value | Atomic maximum |
| `atomic_and(dst, val)` | old value | Atomic AND |
| `atomic_or(dst, val)` | old value | Atomic OR |
| `atomic_xor(dst, val)` | old value | Atomic XOR |
| `atomic_exchange(dst, val)` | old value | Atomic swap |
| `atomic_compare_exchange(dst, expected, desired)` | old value | CAS |
| `sin(x)`, `cos(x)`, `tan(x)` | `f32` | Trigonometry |
| `sqrt(x)`, `rsqrt(x)` | `f32` | Square root / reciprocal sqrt |
| `exp(x)`, `exp2(x)`, `log(x)`, `log2(x)` | `f32` | Exponential / logarithm |
| `pow(base, exp)` | `f32` | Power |
| `abs(x)` | `f32` | Absolute value |
| `min(a, b)`, `max(a, b)` | `f32` | Min / max |
| `clamp(x, lo, hi)` | `f32` | Clamp to range |
| `floor(x)`, `ceil(x)`, `round(x)` | `f32` | Rounding |
| `fma(a, b, c)` | `f32` | Fused multiply-add |

#### Example

```rust
#[derive(quanta::Fields)]
struct SaxpyData { x: Vec<f32>, y: Vec<f32>, a: f32 }

#[quanta::kernel]
fn saxpy(x: &[f32], y: &mut [f32], a: f32) {
    let i = quark_id();
    y[i] = a * x[i] + y[i];
}

fn main() -> Result<(), quanta::QuantaError> {
    let gpu = quanta::init()?;
    let x_field = gpu.field::<f32>(n)?;
    let y_field = gpu.field::<f32>(n)?;
    x_field.write(&x_data)?;
    y_field.write(&y_data)?;

    let mut wave = saxpy(&gpu)?;
    wave.bind(0, &x_field);
    wave.bind(1, &y_field);
    wave.set_value(2, 2.0f32);
    let mut pulse = gpu.dispatch(&wave, n as u32)?;
    pulse.wait()?;
    Ok(())
}
```

---

### `#[quanta::device]`

Mark a function as a GPU device function (callable from kernels, not launchable).

#### Syntax

```rust
#[quanta::device]
fn name(params...) -> ReturnType { body }
```

#### Produces

- `const __QUANTA_DEVICE_NAME: &str` -- captured source for kernel compilation

Device functions are inlined by LLVM. They cannot be dispatched from the CPU.

#### Example

```rust
#[quanta::device]
fn activate(x: f32, threshold: f32) -> f32 {
    if x > threshold { x } else { x * 0.01 }
}

#[quanta::kernel]
fn neural_layer(input: &[f32], output: &mut [f32], threshold: f32) {
    let i = quark_id();
    output[i] = activate(input[i], threshold);
}
```

---

### `#[quanta::shared]`

Declare workgroup-local (shared) memory inside a kernel.

#### Syntax

```rust
#[quanta::shared]
let name: [T; SIZE];
```

#### Produces

`SharedDecl` in the kernel IR. Access generates `SharedLoad`/`SharedStore`.

#### Constraints

- Must be a fixed-size array
- Only valid inside `#[quanta::kernel]` bodies
- Size is shared across all quarks in the workgroup

#### Example

```rust
#[quanta::kernel]
fn prefix_sum(data: &[f32], output: &mut [f32]) {
    #[quanta::shared]
    let scratch: [f32; 512];

    let lid = local_id();
    scratch[lid] = data[quark_id()];
    barrier();

    // Blelloch scan...
}
```

---

### `#[quanta::vertex]`

Compile a function into a vertex shader.

#### Syntax

```rust
#[quanta::vertex]
fn name(attributes..., uniforms: &T) -> OutputType { body }
```

#### Parameters

- Value params: vertex attributes (per-vertex data from vertex buffers)
- `&T` reference params: uniform buffer bindings

#### Produces

- `static NAME_SHADER: ShaderBinary` -- compiled SPIR-V + metallib (+ WGSL) payloads
- `fn name() -> &'static ShaderBinary` -- accessor

#### Example

```rust
#[quanta::vertex]
fn transform(position: Vec3, color: Vec4, mvp: &Mat4) -> Vec4 {
    mvp * vec4(position.x, position.y, position.z, 1.0)
}
```

---

### `#[quanta::fragment]`

Compile a function into a fragment shader.

#### Syntax

```rust
#[quanta::fragment]
fn name(varyings..., textures: &Texture2D, uniforms: &T) -> Vec4 { body }
```

#### Parameters

- Value params: interpolated varyings from the vertex shader (matched by name)
- `&Texture2D` reference params: sampled textures. Slots follow declaration
  order among texture params -- the first texture param is slot 0, bound at
  draw time with `.texture(0, &tex).sampler(0, desc)`. Sample with the
  `sample(param_name, uv)` intrinsic (returns `Vec4`).
- Other `&T` reference params (`&Vec4`, `&Mat4`, ...): uniform buffer
  bindings. Fragment uniforms number their slots by declaration order among
  uniform params -- the first uniform binds with `.uniform(0, &field)`.
  Fragment-stage uniforms currently reach Metal only (the SPIR-V fragment
  emitter does not declare them yet); texture sampling works on both.

#### Produces

- `static NAME_SHADER: ShaderBinary` -- compiled SPIR-V + metallib (+ WGSL) payloads
- `fn name() -> &'static ShaderBinary` -- accessor

#### Example

```rust
#[quanta::fragment]
fn shade(uv: Vec2, albedo: &Texture2D) -> Vec4 {
    sample(albedo, uv)
}
```

Textured fragments emit metallib and SPIR-V; the WGSL payload does not
support texture sampling yet.

---

### `#[quanta::tess_control]`

Tessellation control (hull) shader. Determines tessellation factors per patch.

#### Syntax

```rust
#[quanta::tess_control]
fn name(patch_id: u32, ...) -> TessFactors { body }
```

#### Example

```rust
#[quanta::tess_control]
fn adaptive_tess(patch_id: u32, camera_dist: f32) -> TessFactors {
    let level = clamp(10.0 / camera_dist, 1.0, 64.0);
    TessFactors { edge: [level; 4], inside: [level; 2] }
}
```

---

### `#[quanta::tess_eval]`

Tessellation evaluation (domain) shader. Runs per generated vertex.

#### Syntax

```rust
#[quanta::tess_eval]
fn name(uv: Vec2, patch: &[Vec3; N]) -> Vec4 { body }
```

#### Example

```rust
#[quanta::tess_eval]
fn terrain_eval(uv: Vec2, patch: &[Vec3; 4], heightmap: &Texture2D<f32>) -> Vec4 {
    let pos = bilinear(patch, uv);
    let height = texture_sample(heightmap, uv.x, uv.y).x;
    vec4(pos.x, pos.y + height, pos.z, 1.0)
}
```

---

### `#[quanta::task]`

Task (amplification) shader. Performs coarse culling and launches mesh shader
threadgroups.

#### Example

```rust
#[quanta::task]
fn frustum_cull(group_id: u32, bounds: &[BoundingSphere], frustum: &Frustum) {
    if sphere_in_frustum(bounds[group_id], frustum) {
        emit_mesh_threadgroups(1);
    }
}
```

---

### `#[quanta::mesh]`

Mesh shader. Generates vertices and primitives directly, replacing vertex
input assembly.

#### Example

```rust
#[quanta::mesh]
fn procedural_quad(group_id: u32) {
    set_vertex(0, vec4(-1.0, -1.0, 0.0, 1.0));
    set_vertex(1, vec4( 1.0, -1.0, 0.0, 1.0));
    set_vertex(2, vec4( 1.0,  1.0, 0.0, 1.0));
    set_vertex(3, vec4(-1.0,  1.0, 0.0, 1.0));
    set_primitive(0, [0, 1, 2]);
    set_primitive(1, [0, 2, 3]);
}
```

---

### `#[quanta::ray_gen]`

Ray generation shader. Entry point for ray tracing -- launched once per pixel/ray.

#### Example

```rust
#[quanta::ray_gen]
fn camera_rays(pixel: UVec2, scene: &AccelerationStructure, output: &mut Texture2D<f32>) {
    let ray = compute_camera_ray(pixel);
    let color = trace_ray(scene, ray, 0.0, 1000.0);
    texture_write(output, pixel.x, pixel.y, color);
}
```

---

### `#[quanta::closest_hit]`

Closest-hit shader. Invoked when a ray intersects the nearest surface.

#### Example

```rust
#[quanta::closest_hit]
fn pbr_shade(hit: HitInfo, ray: Ray) -> Vec4 {
    let albedo = sample_texture(hit.uv);
    let n_dot_l = max(dot(hit.normal, light_dir), 0.0);
    albedo * n_dot_l
}
```

---

### `#[quanta::miss]`

Miss shader. Invoked when a ray hits no geometry.

#### Example

```rust
#[quanta::miss]
fn sky_gradient(ray: Ray) -> Vec4 {
    let t = 0.5 * (ray.direction.y + 1.0);
    lerp(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.5, 0.7, 1.0, 1.0), t)
}
```

---

### `#[quanta::gpu_type]`

Mark a struct as GPU-compatible. Generates layout metadata, shader declarations,
and a `GpuType` trait impl.

#### Syntax

```rust
#[quanta::gpu_type]
struct Name {
    field1: Type1,
    field2: Type2,
}
```

#### Produces

- `#[repr(C)]` attribute (if not already present)
- `#[derive(Copy, Clone)]` (if not already present)
- `impl GpuType for Name` -- enables `gpu.field::<Name>(n)`
- `Name::GPU_SIZE: usize` -- compile-time struct size
- `Name::GPU_FIELDS: &[(&str, &str, usize)]` -- (name, type, byte_offset) tuples
- `const __QUANTA_GPU_TYPE_NAME: &str` -- MSL struct declaration
- `const __QUANTA_GPU_TYPE_NAME_WGSL: &str` -- WGSL struct declaration

#### Supported field types

| Rust type | MSL | WGSL |
|-----------|-----|------|
| `f32` | `float` | `f32` |
| `f64` | `double` | `f64` |
| `u32` | `uint` | `u32` |
| `i32` | `int` | `i32` |
| `u8` | `uint8_t` | `u32` |
| `u16` | `ushort` | `u32` |
| `u64` | `ulong` | `u32` |
| `bool` | `bool` | `bool` |
| `[f32; 2]` | `float2` | `vec2<f32>` |
| `[f32; 3]` | `float3` | `vec3<f32>` |
| `[f32; 4]` | `float4` | `vec4<f32>` |
| `[u32; 4]` | `uint4` | `vec4<u32>` |
| `[f32; 9]` | `float3x3` | `mat3x3<f32>` |
| `[f32; 16]` | `float4x4` | `mat4x4<f32>` |
| `[T; N]` (other) | `T [N]` | `array<T, N>` |

#### Constraints

- Only named-field structs (no tuples, no enums)
- All fields must be scalar, fixed-size array, or another `#[quanta::gpu_type]` struct
- No `String`, `Vec`, `Box`, or other heap types (use offset+length pattern)

#### Example

```rust
#[quanta::gpu_type]
struct Body {
    pos: [f32; 3],
    vel: [f32; 3],
    mass: f32,
}

let bodies = gpu.field::<Body>(65536)?;
assert_eq!(Body::GPU_SIZE, 28);
assert_eq!(Body::GPU_FIELDS[0], ("pos", "[f32; 3]", 0));
```

> **Note:** For uniform buffer structs, prefer `#[derive(quanta::Uniforms)]` which
> additionally generates MSL/WGSL declarations tailored for uniform binding.
> `#[quanta::gpu_type]` is best for storage buffer element types.
