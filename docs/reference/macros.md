# Macro Reference

All proc macros and derive macros in Quanta. The four derive macros
(`Fields`, `Vertex`, `Uniforms`, `Varyings`) are the recommended entry point
for new code.

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

### `#[derive(quanta::Varyings)]`

Declare the **shared vertex↔fragment varying interface** as a struct. This is
the single explicit contract between the two render stages (the WGSL/HLSL
model): the vertex returns it, the fragment takes it, and both name the same
struct. There is no positional "auto-forward".

#### Syntax

```rust
#[derive(quanta::Varyings)]
struct Surface {
    #[position] clip: Vec4, // gl_Position — the vertex writes it
    uv: Vec2,               // Location 0 (field-declaration order)
    kind: u32,              // Location 1, flat-interpolated
}
```

#### Rules

- **Exactly one `#[position]` field**, of type `Vec4`. It becomes
  `gl_Position` (`[[position]]` on Metal, `@builtin(position)` on WGSL). Read
  from a fragment it yields the interpolated **window-space** position.
- **Every other field is a varying.** Supported types: `f32`, `u32`, `Vec2`,
  `Vec3`, `Vec4`. Each is assigned `Location 0, 1, …` in **field-declaration
  order**.
- **`u32` varyings are flat-interpolated automatically** (integers cannot be
  perspective-interpolated).
- **Named-field structs only**; no generics, no tuple structs.

#### What it generates

| Generated item | Type | Description |
|----------------|------|-------------|
| `POSITION_FIELD` | `&'static str` | Name of the `#[position]` field |
| `VARYING_FIELDS` | `[(&'static str, &'static str); N]` | `(name, type-name)` per varying, in Location order |
| `__quanta_varyings_<Name>!` | hidden `macro_rules!` | The trampoline the shader macros expand through (re-exported beside the struct) |

Because a proc macro sees only its own item, the field metadata reaches
`#[quanta::vertex]` / `#[quanta::fragment]` through the generated trampoline.
**Declare the struct before the shaders that use it** (normal macro-scoping); if
it lives in another module, import it *and* its `__quanta_varyings_<Name>`
re-export together.

#### Example

```rust
#[derive(quanta::Varyings)]
struct Surface {
    #[position] clip: Vec4,
    uv: Vec2,
    kind: u32,
}

// Compile-time introspection:
assert_eq!(Surface::POSITION_FIELD, "clip");
assert_eq!(Surface::VARYING_FIELDS, [("uv", "Vec2"), ("kind", "u32")]);
```

A *position-only* interface (`struct P { #[position] clip: Vec4 }`, no
varyings) is legal, but a position-only vertex can simply return `-> Vec4`
without a struct.

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
- `&Texture2D<f32>` -- sampled texture (read via `texture_sample_2d` / `texture_load_2d`; bound with `wave.bind_texture`)
- `&mut Texture2D<f32>` -- **read-write storage image**, R32Float format. Write with `texture_write_2d`; read the same slot with `texture_load_2d` (a storage read, not a sampled fetch). Sampling a storage image is rejected at compile time. Slot = positional across the buffer/texture/constant namespace; bound with `wave.bind_texture`.
- `&mut Texture2D<u32>` -- **read-write storage image**, RGBA8-unorm format, with the texel packed into one `u32`. Each texel crosses the kernel boundary as a `0xAABBGGRR` u32 (little-endian byte order R,G,B,A); build/split it with bit math (see the example below). A read-only, sampled `&Texture2D<u32>` is a different, unwired meaning and is **rejected at compile time** -- use `&Texture2D<f32>` to sample. There is **no read-only RGBA8 spelling**: an RGBA8 source you only *read* (e.g. the `src` half of a src→dst ping-pong) must still be declared `&mut Texture2D<u32>` and read with `texture_load_2d` (a storage read). Because the slot is a read_write storage image either way, it is subject to the RGBA8 Tier-2 requirement below **even for a pure read**. **BGRA8 is not supported**: effect layers must be RGBA8 (there is no SPIR-V storage format for BGRA8; an in-kernel byte-shuffle is the escape hatch). Split and rebuild the texel with the `pack_unorm4x8` / `unpack_unorm4x8_*` intrinsics (preferred) or the equivalent bit math (see the example below).
- Scalar values (`u32`, `f32`, etc.) -- push constants (set via `wave.set_value`)

A GPU buffer param **must be a reference to a slice** (`&[T]` / `&mut [T]`). A
bare fixed-size array `[f32; N]` as a kernel parameter is **rejected at compile
time** (`unsupported parameter type`) -- it is not an inline uniform array; pass
`&[f32]` (read) or `&mut [f32]` (write) instead. (`[f32; N]` *is* a valid field
type inside a `#[derive(quanta::Fields)]` / `#[quanta::gpu_type]` struct, where
it is a push-constant array -- that is a different position from a top-level
kernel parameter.)

The texture format contract is scalar-driven and enforced per storage-slot kind
at dispatch: a `&mut Texture2D<f32>` slot must be bound to an `R32Float`
texture, and a `&mut Texture2D<u32>` slot to an `RGBA8` texture (both created
with `SHADER_WRITE` usage). A format mismatch -- either direction -- returns
`QuantaErrorKind::InvalidParam`. Storage compute textures are supported on
Metal, the CPU reference device, and native Vulkan (load/write); sampling in
compute (`texture_sample_2d` on a `&Texture2D` read slot) works on Metal, the
CPU device, and native Vulkan through a fixed nearest / clamp-to-edge /
unnormalized compute sampler, and WebGPU returns `NotSupported` for any
compute texture binding. Query support with `gpu.supports_compute_textures()`.
RGBA8 (`&mut Texture2D<u32>`) storage additionally needs
`MTLReadWriteTextureTier2` on Metal; a device below tier 2 returns
`QuantaErrorKind::NotSupported` at dispatch (RGBA8 storage is a mandatory format
on Vulkan, so there is no equivalent gate there). This tier nuance surfaces as
the `NotSupported` error, not as a separate capability query.

Working on a packed texel — the preferred form uses the `pack_unorm4x8` /
`unpack_unorm4x8_*` intrinsics, which read/write the four RGBA8 channels as
normalized `f32` in `[0, 1]`:

```rust
#[quanta::kernel]
fn tint(image: &mut Texture2D<u32>, width: u32) {
    let i = quark_id();
    let (x, y) = (i % width, i / width);
    let v = texture_load_2d(image, x, y);
    // Unpack to unorm floats, operate, repack. (Here: halve the red channel.)
    let r = unpack_unorm4x8_r(v);
    let g = unpack_unorm4x8_g(v);
    let b = unpack_unorm4x8_b(v);
    let a = unpack_unorm4x8_a(v);
    let out = pack_unorm4x8(r * 0.5, g, b, a);
    texture_write_2d(image, x, y, out);
}
```

The intrinsics are conveniences: they lower to the same integer bit math you
would otherwise write by hand. `unpack_unorm4x8_g(v)` compiles to
`(((v >> 8) & 0xFF) as f32) / 255.0`, and `pack_unorm4x8(r, g, b, a)` to the
per-channel `(round(clamp(c, 0, 1) * 255) as u32 & 0xFF)` shifted into place and
OR-ed together. When you want to manipulate the raw bytes directly (e.g. a pure
channel swizzle with no unorm arithmetic), the bit math is equally valid:

```rust
// Swap R and B directly on the packed bytes — no intrinsic needed.
let v = texture_load_2d(image, x, y);
let r = v & 0xFF;
let g = (v >> 8) & 0xFF;
let b = (v >> 16) & 0xFF;
let a = (v >> 24) & 0xFF;
let out = b | (g << 8) | (r << 16) | (a << 24);
texture_write_2d(image, x, y, out);
```

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
| `exp(x)`, `ln(x)` | `f32` | Exponential / natural logarithm |
| `powf(base, exp)` | `f32` | Power (**not** `pow`) |
| `fabs(x)` | `f32` | Absolute value (**not** `abs`) |
| `fmin(a, b)`, `fmax(a, b)` | `f32` | Min / max (**not** `min` / `max`) |
| `clamp_f(x, lo, hi)` | `f32` | Clamp to range (**not** `clamp`) |
| `floor(x)`, `ceil(x)`, `round(x)` | `f32` | Rounding |
| `fma(a, b, c)` | `f32` | Fused multiply-add |
| `texture_load_2d(tex, x, y)` | `f32` / `u32` | Read texel `(x, y)`. On `&mut/&Texture2D<f32>`: the `.x` channel as `f32` (storage read on `&mut`, texel fetch on `&`). On `&mut Texture2D<u32>`: the whole RGBA8 texel as a packed `0xAABBGGRR` u32 |
| `texture_sample_2d(tex, x, y)` | `f32` | Sample a `&Texture2D` read slot through the fixed compute sampler: **nearest**, **clamp-to-edge**, **unnormalized** coords — `(x, y)` are texel coordinates and an out-of-bounds coord reads the edge texel. Returns the `.x` channel. Portable across Metal / Vulkan / CPU; rejected on a storage slot. Contrast `texture_load_2d`, which requires in-bounds coords (out-of-bounds is undefined on the GPU) |
| `texture_write_2d(tex, x, y, v)` | `()` | Write `v` into texel `(x, y)`. On `&mut Texture2D<f32>`: `v: f32` into the R channel. On `&mut Texture2D<u32>`: `v: u32` as a packed `0xAABBGGRR` RGBA8 texel |
| `texture_size(tex)` | `(u32, u32)` | Texture `(width, height)` (CPU device) |
| `pack_unorm4x8(r, g, b, a)` | `u32` | Pack four `f32` unorm channels into a `0xAABBGGRR` RGBA8 u32: each is clamped to `[0, 1]`, scaled by 255, rounded, and stored (R = byte 0). Byte-identical to the `Texture2D<u32>` texel `texture_write_2d` stores |
| `unpack_unorm4x8_r/_g/_b/_a(v)` | `f32` | Unpack one channel of a packed RGBA8 u32 as `channel_byte / 255.0` (byte 0 = R). Inverts `pack_unorm4x8`; the round-trip `pack_unorm4x8(unpack_r(v), unpack_g(v), unpack_b(v), unpack_a(v)) == v` is exact for every `v` |

The kernel (compute) intrinsic surface above is a **different set** from
the shader (vertex/fragment) intrinsics documented under
[`#[quanta::fragment]`](#quantafragment) — e.g. kernels have
`atomic_*` and `barrier`; shaders have `sample`, `mix`, and the
derivatives. Which Rust forms are guaranteed to lower to a kernel (and
which fail loudly) is the
[kernel-lowering contract](kernel-lowering.md).

**The two surfaces also spell their shared math differently, and the
names do not cross over.** The kernel body uses the C-style free
functions `fmin` / `fmax` / `clamp_f` / `fabs` / `powf` (plus `fma`),
whereas the shader grammar uses the GLSL/WGSL names `min` / `max` /
`clamp` / `abs` / `pow`. In a `#[quanta::kernel]`, `max(a, b)` or
`clamp(x, lo, hi)` is **not** defined and fails to build (rustc reports
the unresolved name at compile time, before lowering) — write `fmax` and
`clamp_f`. The unprefixed spellings that *do* work in a kernel are the
method forms on `f32` / `f64` (`x.abs()`, `x.floor()`, `b.powf(e)`,
`x.sqrt()`, …); there are no `.min` / `.max` / `.clamp` methods, so use
the free `fmin` / `fmax` / `clamp_f`. Conversely the shader `min` / `max`
/ `clamp` spellings do not exist in a kernel.

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

- Value params: vertex attributes (per-vertex data from vertex buffers). They
  are **pure inputs** -- nothing is auto-forwarded to the fragment stage.
- `&T` reference params: uniform buffer bindings
- `&[T]` slice params (`&[f32]`, `&[Vec2]`, `&[Vec4]` only): storage-buffer
  arrays, indexed in the body with `name[index]`. They share the uniform
  binding index space and bind the same way -- see the fragment table below.
- A `&Texture2D` param is a **compile error** in a vertex (textures are
  fragment-only).

#### Return type

Two forms:

- `-> Vec4` -- a **position-only** vertex: the body's tail expression is the
  clip-space position, and the shader has no varyings.
- `-> MyVaryings` -- a [`#[derive(quanta::Varyings)]`](#derivequantavaryings)
  struct: the body ends in the struct literal, whose `#[position]` field
  becomes `gl_Position` and whose other fields are the varyings the paired
  fragment reads.

#### Produces

- `static NAME_SHADER: ShaderBinary` -- compiled SPIR-V + metallib (+ WGSL) payloads
- `fn name() -> &'static ShaderBinary` -- accessor

#### Example

```rust
// The shared interface, declared before the shader.
#[derive(quanta::Varyings)]
struct Surface {
    #[position] clip: Vec4,
    uv: Vec2,
}

#[quanta::vertex]
fn transform(position: Vec3, in_uv: Vec2, mvp: &Mat4) -> Surface {
    Surface {
        clip: mvp * Vec4::new(position.x, position.y, position.z, 1.0),
        uv: in_uv,
    }
}
```

---

### `#[quanta::fragment]`

Compile a function into a fragment shader.

#### Syntax

```rust
#[quanta::fragment]
fn name(s: MyVaryings, textures: &Texture2D, uniforms: &T) -> Vec4 { body }
```

#### Parameters

- The **varying struct**: a single parameter typed as the paired vertex's
  [`#[derive(quanta::Varyings)]`](#derivequantavaryings) struct. Read each
  varying by field (`s.uv`); reading the `#[position]` field yields the
  interpolated window position. A fragment with no varyings simply omits this
  parameter. A **plain value param** (e.g. `uv: Vec2`) is a compile error --
  stage inputs come from the struct, not from positional params.
- `&Texture2D` reference params: sampled textures. Slots follow declaration
  order among texture params -- the first texture param is slot 0, bound at
  draw time with `.texture(0, &tex).sampler(0, desc)`. Sample with the
  `sample(param_name, uv)` intrinsic (returns `Vec4`).
- Other `&T` reference params (`&Vec4`, `&Mat4`, ...): uniform buffer
  bindings. Fragment uniforms number their slots by declaration order among
  uniform params -- the first uniform binds with `.uniform(0, &field)`.
  On Vulkan they are declared as storage-buffer descriptors at the same
  binding the runtime uses, so fragment uniforms work on both backends.
- `&[T]` slice params (`&[f32]`, `&[Vec2]`, `&[Vec4]` only -- any other
  element type is a compile error): a fragment-readable table backed by a
  storage buffer, e.g. gradient stops or a palette LUT instead of a fixed
  set of varyings. **Binding.** Slices and `&T` uniforms draw slots from
  **one** declaration-index space: walking the params in order, each uniform
  or slice consumes the next index, so a slice declared between two uniforms
  takes the middle slot. Bind it at draw time with the same
  `.uniform(slot, &field)` (it binds a storage buffer at `binding=slot` on
  both stages) -- so at most **8 combined** uniform + slice params fit
  (bindings 0-7; texture bindings start at 8), and exceeding that is a
  named compile error. **Reading.** Index with `name[index]` in the body
  (see below).

#### Body language

Shader bodies are a Rust subset compiled by quanta's own emitters (SPIR-V
and MSL from one grammar): `let` / `let mut` bindings, assignments to
mutable locals, statement and expression `if`/`else` (both branches
required), **bounded `for i in 0..N` loops** (constant integer bound, `u32`
counter), arithmetic and comparisons (including `u32` integer comparisons),
`VecN::new`, swizzle field access (`.x`/`.rgba`, incl. on parenthesized
expressions), uniform deref (`*viewport`, `(*viewport).x`),
`sample(texture_param, uv)`, the window builtins `frag_coord()` (fragment) and
`vertex_id()` / `instance_id()` (vertex), the GLSL-style math intrinsics
(`sin`, `mix`, `smoothstep`, `clamp`, `inverse_sqrt`, ...), the fragment-stage
derivatives `fwidth` / `dpdx` / `dpdy`, and `&[T]` slice indexing `name[index]`
(see below). Both emitters accept the artifacts rustfmt and the token printer
produce in real builds: a trailing comma before `)` in a call, a call wrapped
and split across lines (`Vec4 ::\nnew(`), and a `let` declared inside an
`if`/`else` branch. Anything outside this surface is a compile error naming the
construct -- nothing silently miscompiles. The full body grammar, the `u32`
type, the builtins, and the loop rules are the [Shader Language
reference](shader-language.md).

**Slice indexing.** `name[index]` reads element `index` of a `&[T]` slice
param. `index` is any scalar expression the grammar accepts; it is f32 in
this grammar and **truncates** toward zero to an integer (`stops[uv.x * 4.0]`
selects stop `floor(uv.x * 4.0)`). Bounds are **unchecked** -- indexing out
of range is undefined, the GPU storage-buffer contract. Indexing anything
that is not a slice param (a value, a uniform, an unknown name) is a compile
error.

```rust,ignore
// `Surface` is the paired vertex's #[derive(Varyings)] struct; `s.uv` is a varying.
#[quanta::fragment]
fn gradient(s: Surface, stops: &[Vec4]) -> Vec4 {
    let i = s.uv.x * 4.0;    // 0.0 .. 4.0 across the quad
    stops[i]                 // truncates to stop 0/1/2/3
}
```

Assigning to a local declared *outside* an `if` from *inside an
expression-`if` branch* now works identically on both emitters -- the
SPIR-V expression-`if` merge phis a mutated outer local exactly like a
statement-`if` -- so both forms below are portable:

```rust,ignore
// statement-if reassigns the outer local
let mut c = 0.0;
if uv.x > 0.5 { c = 1.0; } else { c = 0.5; }

// expression-if branch reassigns the outer local (also portable)
let mut c = 0.0;
let v = if uv.x > 0.5 { c = 1.0; uv.x } else { uv.y };
```

#### Produces

- `static NAME_SHADER: ShaderBinary` -- compiled SPIR-V + metallib (+ WGSL) payloads
- `fn name() -> &'static ShaderBinary` -- accessor

#### Example

```rust
#[quanta::fragment]
fn shade(s: Surface, albedo: &Texture2D) -> Vec4 {
    sample(albedo, s.uv)
}
```

WGSL shader emission has **full grammar parity with MSL and SPIR-V**: a
real statement walker handles `let` / `let mut` (→ `var`) / assignment,
statement- and expression-`if`/`else`, swizzles, intrinsics, and
constructors regardless of token spacing. `&[T]` slice params lower to
`@group(0) @binding(slot) var<storage, read> name: array<ELEM>`, `&T`
uniforms to `var<uniform> name: T` at their shared decl-index, and
`sample(slot, uv)` to `textureSample(tex_N, smp_N, uv)` with textures
and samplers at bindings `8 + slot` — the same shared uniform+slice
index space (cap 8) and f32→u32 index truncation as the other two
emitters. Constructs every emitter rejects (loops, method calls,
else-less `if` expressions) are rejected by WGSL with the same error
wording. The parity corpus asserts all three emitters agree, and every
translated fixture validates under `naga` when the CLI is installed.

#### Platform-targeted metallibs (Apple)

iOS rejects a macOS-platform metallib, so a shader or kernel that ships to
iOS needs the platform-correct Metal library. The compiler emits up to
three variants per shader/kernel — macOS, iOS device, and iOS simulator —
and the macros embed **every** variant that was produced into the
`ShaderBinary` / `KernelBinary` static (the fields `metallib`,
`metallib_ios`, `metallib_ios_sim`). A proc macro cannot see the
consumer's compile target (that information only reaches build scripts), so
the choice is deferred to the runtime: `for_vendor(Apple)` resolves the
metallib by the build target, most-specific first —

- an **iOS-simulator** build picks `metallib_ios_sim`, then `metallib_ios`,
  then `metallib`;
- an **iOS-device** build picks `metallib_ios`, then `metallib`;
- a **macOS / desktop** build picks `metallib` (shaders then fall back to
  SPIR-V; kernels fall back to the JIT path).

Each build compiles only its own chain, so this adds nothing to a desktop
binary. Producing the iOS variants requires the iOS SDKs (full Xcode). On a
mac with only the Command-Line Tools the iOS SDKs are absent, so those
variants soft-skip and the build ships **macOS-only** — no error, exactly
as before this feature existed. `QUANTA_METAL_PLATFORMS` (see the
[environment reference](environment.md)) narrows which variants are
attempted.

---

> **The seven stage attributes below are placeholders, not compiled
> pipelines.** `#[quanta::tess_control]`, `tess_eval`, `task`, `mesh`,
> `ray_gen`, `closest_hit`, and `miss` compile — each expands to a valid
> `ShaderBinary` carrying the correct `ShaderStage`, with `wgsl: None`
> and **all binary payloads `None`** (no SPIR-V, no metallib). The
> function body is **not** lowered to a shader; the examples below show
> the intended stage signature, not code that runs today. The compiled
> render stages are `#[quanta::vertex]` and `#[quanta::fragment]`; the
> tessellation / mesh / ray-tracing *pipelines* are driven through the
> typed render API (`gpu.tessellation_pipeline`, `gpu.mesh_pipeline`,
> `gpu.ray_tracing_pipeline`), not these attribute stubs.

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
