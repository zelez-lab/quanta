# Macro Reference

All 13 `#[quanta::*]` proc macros. The `kernel` macro accepts additional
attributes (`workgroup`, `jit`, `opt`) described below.

---

## `#[quanta::kernel]`

Compile a Rust function into a GPU compute kernel.

### Syntax

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

### Attributes

| Attribute | Values | Default | Description |
|-----------|--------|---------|-------------|
| `opt` | `"O0"`, `"O1"`, `"O2"`, `"O3"` | `"O3"` | LLVM optimization level |
| `workgroup` | `[x, y, z]` | driver-chosen | Workgroup dimensions (1D/2D/3D) |
| `jit` | flag | off | Serialize KernelDef for runtime compilation |

### Parameters

- `&[T]` — read-only GPU buffer (bound at slot by declaration order)
- `&mut [T]` — read-write GPU buffer
- Scalar values (`u32`, `f32`, etc.) — push constants (set via `wave.set_value`)

### Produces

Without `jit`:
- `static NAME_BINARY: KernelBinary` — compiled native binaries (SPIR-V, metallib, PTX, GCN) for all backends
- `fn name(gpu: &Gpu) -> Result<Wave, QuantaError>` — creates a bound wave

With `jit`:
- `static NAME_KERNEL_DEF: &[u8]` — serialized KernelDef IR (compile at runtime via `gpu.wave_jit()`)
- `fn name(gpu: &Gpu) -> Result<Wave, QuantaError>` — creates a bound wave (triggers runtime compilation)

### Built-in functions available in kernel body

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

### Example

```rust
#[quanta::kernel]
fn saxpy(x: &[f32], y: &mut [f32], a: f32) {
    let i = quark_id();
    y[i] = a * x[i] + y[i];
}

fn main() {
    let gpu = quanta::init().unwrap();
    let mut wave = saxpy(&gpu).unwrap();
    wave.bind(0, &x_field);
    wave.bind(1, &y_field);
    wave.set_value(2, 2.0f32);
    let mut pulse = gpu.dispatch(&wave, n).unwrap();
    gpu.wait(&mut pulse).unwrap();
}
```

---

## `#[quanta::device]`

Mark a function as a GPU device function (callable from kernels, not launchable).

### Syntax

```rust
#[quanta::device]
fn name(params...) -> ReturnType { body }
```

### Produces

- `const __QUANTA_DEVICE_NAME: &str` — captured source for kernel compilation

Device functions are inlined by LLVM. They cannot be dispatched from the CPU.

### Example

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

## `#[quanta::shared]`

Declare workgroup-local (shared) memory inside a kernel.

### Syntax

```rust
#[quanta::shared]
let name: [T; SIZE];
```

### Produces

`SharedDecl` in the kernel IR. Access generates `SharedLoad`/`SharedStore`.

### Constraints

- Must be a fixed-size array
- Only valid inside `#[quanta::kernel]` bodies
- Size is shared across all quarks in the workgroup

### Example

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

## `#[quanta::vertex]`

Compile a function into a vertex shader.

### Syntax

```rust
#[quanta::vertex]
fn name(attributes..., uniforms: &T) -> OutputType { body }
```

### Parameters

- Value params: vertex attributes (per-vertex data from vertex buffers)
- `&T` reference params: uniform buffer bindings

### Produces

- `static NAME_SHADER: ShaderBinary` — compiled SPIR-V + metallib binaries
- `fn name() -> &'static ShaderBinary` — accessor

### Example

```rust
#[quanta::vertex]
fn transform(position: Vec3, color: Vec4, mvp: &Mat4) -> Vec4 {
    mvp * vec4(position.x, position.y, position.z, 1.0)
}
```

---

## `#[quanta::fragment]`

Compile a function into a fragment shader.

### Syntax

```rust
#[quanta::fragment]
fn name(varyings..., textures: &Texture2D, uniforms: &T) -> Vec4 { body }
```

### Parameters

- Value params: interpolated varyings from vertex shader
- `&Texture2D` reference params: texture bindings
- `&T` reference params: uniform buffer bindings

### Produces

- `static NAME_SHADER: ShaderBinary` — compiled SPIR-V + metallib binaries
- `fn name() -> &'static ShaderBinary` — accessor

### Example

```rust
#[quanta::fragment]
fn shade(uv: Vec2, albedo: &Texture2D<f32>) -> Vec4 {
    texture_sample(albedo, uv.x, uv.y)
}
```

---

## `#[quanta::tess_control]`

Tessellation control (hull) shader. Determines tessellation factors per patch.

### Syntax

```rust
#[quanta::tess_control]
fn name(patch_id: u32, ...) -> TessFactors { body }
```

### Produces

- `static NAME_SHADER: ShaderBinary` (stage = `TessControl`)
- `fn name() -> &'static ShaderBinary`

### Example

```rust
#[quanta::tess_control]
fn adaptive_tess(patch_id: u32, camera_dist: f32) -> TessFactors {
    let level = clamp(10.0 / camera_dist, 1.0, 64.0);
    TessFactors { edge: [level; 4], inside: [level; 2] }
}
```

---

## `#[quanta::tess_eval]`

Tessellation evaluation (domain) shader. Runs per generated vertex.

### Syntax

```rust
#[quanta::tess_eval]
fn name(uv: Vec2, patch: &[Vec3; N]) -> Vec4 { body }
```

### Produces

- `static NAME_SHADER: ShaderBinary` (stage = `TessEval`)
- `fn name() -> &'static ShaderBinary`

### Example

```rust
#[quanta::tess_eval]
fn terrain_eval(uv: Vec2, patch: &[Vec3; 4], heightmap: &Texture2D<f32>) -> Vec4 {
    let pos = bilinear(patch, uv);
    let height = texture_sample(heightmap, uv.x, uv.y).x;
    vec4(pos.x, pos.y + height, pos.z, 1.0)
}
```

---

## `#[quanta::task]`

Task (amplification) shader. Performs coarse culling and launches mesh shader
threadgroups.

### Syntax

```rust
#[quanta::task]
fn name(group_id: u32, ...) { body }
```

### Produces

- `static NAME_SHADER: ShaderBinary` (stage = `Task`)
- `fn name() -> &'static ShaderBinary`

### Example

```rust
#[quanta::task]
fn frustum_cull(group_id: u32, bounds: &[BoundingSphere], frustum: &Frustum) {
    if sphere_in_frustum(bounds[group_id], frustum) {
        emit_mesh_threadgroups(1);
    }
}
```

---

## `#[quanta::mesh]`

Mesh shader. Generates vertices and primitives directly, replacing vertex
input assembly.

### Syntax

```rust
#[quanta::mesh]
fn name(group_id: u32, ...) { body }
```

### Produces

- `static NAME_SHADER: ShaderBinary` (stage = `Mesh`)
- `fn name() -> &'static ShaderBinary`

### Example

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

## `#[quanta::ray_gen]`

Ray generation shader. Entry point for ray tracing — launched once per pixel/ray.

### Syntax

```rust
#[quanta::ray_gen]
fn name(pixel: UVec2, ...) { body }
```

### Produces

- `static NAME_SHADER: ShaderBinary` (stage = `RayGen`)
- `fn name() -> &'static ShaderBinary`

### Example

```rust
#[quanta::ray_gen]
fn camera_rays(pixel: UVec2, scene: &AccelerationStructure, output: &mut Texture2D<f32>) {
    let ray = compute_camera_ray(pixel);
    let color = trace_ray(scene, ray, 0.0, 1000.0);
    texture_write(output, pixel.x, pixel.y, color);
}
```

---

## `#[quanta::closest_hit]`

Closest-hit shader. Invoked when a ray intersects the nearest surface.

### Syntax

```rust
#[quanta::closest_hit]
fn name(hit: HitInfo, ray: Ray) -> Vec4 { body }
```

### Produces

- `static NAME_SHADER: ShaderBinary` (stage = `ClosestHit`)
- `fn name() -> &'static ShaderBinary`

### Example

```rust
#[quanta::closest_hit]
fn pbr_shade(hit: HitInfo, ray: Ray) -> Vec4 {
    let albedo = sample_texture(hit.uv);
    let n_dot_l = max(dot(hit.normal, light_dir), 0.0);
    albedo * n_dot_l
}
```

---

## `#[quanta::miss]`

Miss shader. Invoked when a ray hits no geometry.

### Syntax

```rust
#[quanta::miss]
fn name(ray: Ray) -> Vec4 { body }
```

### Produces

- `static NAME_SHADER: ShaderBinary` (stage = `Miss`)
- `fn name() -> &'static ShaderBinary`

### Example

```rust
#[quanta::miss]
fn sky_gradient(ray: Ray) -> Vec4 {
    let t = 0.5 * (ray.direction.y + 1.0);
    lerp(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.5, 0.7, 1.0, 1.0), t)
}
```

---

## `#[quanta::gpu_type]`

Mark a struct as GPU-compatible. Generates layout metadata, shader declarations,
and a `GpuType` trait impl.

### Syntax

```rust
#[quanta::gpu_type]
struct Name {
    field1: Type1,
    field2: Type2,
    // ...
}
```

### Produces

- `#[repr(C)]` attribute (if not already present)
- `#[derive(Copy, Clone)]` (if not already present)
- `impl GpuType for Name` -- enables `gpu.compute_field::<Name>(n)`
- `Name::GPU_SIZE: usize` -- compile-time struct size
- `Name::GPU_FIELDS: &[(&str, &str, usize)]` -- (name, type, byte_offset) tuples
- `const __QUANTA_GPU_TYPE_NAME: &str` -- MSL struct declaration
- `const __QUANTA_GPU_TYPE_NAME_WGSL: &str` -- WGSL struct declaration

### Supported field types

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

### Constraints

- Only named-field structs (no tuples, no enums)
- All fields must be scalar, fixed-size array, or another `#[quanta::gpu_type]` struct
- No `String`, `Vec`, `Box`, or other heap types (use offset+length pattern)

### Example

```rust
#[quanta::gpu_type]
struct Body {
    pos: [f32; 3],
    vel: [f32; 3],
    mass: f32,
}

// Use in a field
let bodies = gpu.compute_field::<Body>(65536)?;

// Inspect generated metadata
assert_eq!(Body::GPU_SIZE, 28);
assert_eq!(Body::GPU_FIELDS[0], ("pos", "[f32; 3]", 0));
assert_eq!(Body::GPU_FIELDS[1], ("vel", "[f32; 3]", 12));
assert_eq!(Body::GPU_FIELDS[2], ("mass", "f32", 24));

// MSL declaration is available at compile time:
// struct Body {
//     float3 pos;
//     float3 vel;
//     float mass;
// };
```
