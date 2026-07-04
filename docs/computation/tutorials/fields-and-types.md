# Fields and Types

GPU memory management in Quanta. Fields are typed GPU buffers that quarks
operate on.

## The derive-based API

The primary way to work with GPU data is `#[derive(quanta::Fields)]`. You
define a struct, the framework handles allocation, upload, binding, and
readback.

```rust
use quanta::*;

#[derive(quanta::Fields)]
struct MyData {
    input: Vec<f32>,      // GPU storage buffer (read-write)
    output: Vec<f32>,     // GPU storage buffer (read-write)
    scale: f32,           // Push constant (per-dispatch scalar)
    count: u32,           // Push constant (per-dispatch scalar)
}
```

Rules:

- **`Vec<T>`** fields become GPU storage buffers. Data is uploaded before
  dispatch and read back after. The GPU sees them as arrays.
- **Scalar** fields (`f32`, `u32`, `i32`, `u64`, `i64`, `f64`, `u8`,
  `u16`, `i16`, `bool`) become push constants -- small values baked
  directly into the command stream. Faster than buffer reads for
  single values like thresholds, dimensions, or timestep.
- **Fixed-size arrays** (`[f32; 4]`, `[u32; 2]`, etc.) are also push
  constants.

The derive macro generates metadata at compile time:

```rust
MyData::FIELD_COUNT           // 2 (input, output)
MyData::PUSH_CONSTANT_COUNT   // 2 (scale, count)
MyData::field_names()         // &["input", "output"]
MyData::field_types()         // &["f32", "f32"]
MyData::push_constant_names() // &["scale", "count"]
MyData::push_constant_types() // &["f32", "u32"]
```

## Supported scalar types

Any type implementing `GpuType` can appear in fields. Built-in:

| Rust type | GPU size | Notes                              |
|-----------|----------|------------------------------------|
| `f32`     | 4 bytes  | Standard GPU float                 |
| `f64`     | 8 bytes  | Not all GPUs support double        |
| `f16`     | 2 bytes  | Half-precision float               |
| `bf16`    | 2 bytes  | bfloat16 — f32 range, half width; the modern training/inference dtype. Computed in f32, packed on store. |
| `fp8` (e5m2 / e4m3) | 1 byte | 8-bit floats for quantized inference. Computed in f32, packed on store. |
| `u32`     | 4 bytes  | Standard GPU unsigned integer      |
| `i32`     | 4 bytes  | Standard GPU signed integer        |
| `u64`     | 8 bytes  | Atomics may not support 64-bit     |
| `i64`     | 8 bytes  |                                    |
| `u16`     | 2 bytes  | Half-width integer                 |
| `i16`     | 2 bytes  |                                    |
| `u8`      | 1 byte   | Byte-level access                  |
| `i8`      | 1 byte   |                                    |
| `i4`      | 4 bits   | Signed 4-bit, packed 8/word — int4 quantization storage |

`f32` and `u32` are universally supported and fastest on all GPUs. The
narrow floats (`bf16`/`fp8`) and quantized ints (`i8`/`i4` via the
quantization scheme) are emulated in f32 where the backend lacks native
support — the conversions live in `quanta-ir`'s `dtype` module and are
proven bit-exact across all backends.

Narrow-float buffers are stored at their **native stride** — 2 bytes per
`bf16`, 1 byte per `fp8` element — on the host, the CPU executor, Metal
(`ushort`/`uchar` slots) and Vulkan alike, so a tight host array binds
directly. The one exception is WebGPU: WGSL storage buffers cannot hold
16-/8-bit array elements, so that backend keeps one element per 32-bit word
and the host repacks before binding. Query `gpu.narrow_storage_u32_slot()`
to detect it. The addressing contract (every backend's element `i` reads
source element `i`) is proven in Lean
(`specs/verify/lean/Quanta/Dtype/StorageAddressing.lean`).

## Field operations (resource-owned)

Fields own their operations. Write, read, and copy are methods on the
field itself, not on the GPU handle:

```rust
let gpu = quanta::init()?;

// Allocate
let field = gpu.field::<f32>(1024, FieldUsage::default_compute())?;

// Write CPU data to GPU
field.write(&data)?;

// Read GPU data back to CPU
let result: Vec<f32> = field.read()?;

// Copy between fields
dst.copy_from(&src)?;
```

Convenience constructors for common usage patterns:

```rust
let compute = gpu.compute_field::<f32>(n)?;   // READ | WRITE | COMPUTE | TRANSFER
let render  = gpu.render_field::<f32>(n)?;    // READ | RENDER | TRANSFER
let uniform = gpu.uniform_field::<f32>(n)?;   // READ | UNIFORM | TRANSFER
```

When using `#[derive(quanta::Fields)]`, you do not call these directly --
the generated dispatch code allocates, writes, and reads for you.

## Field properties

```rust
let field = gpu.compute_field::<f32>(1024)?;
assert_eq!(field.len(), 1024);           // element count
assert_eq!(field.byte_size(), 4096);     // total bytes (1024 * 4)
assert!(!field.is_empty());
```

## User-defined structs (`#[quanta::gpu_type]`)

For multi-component data, define a GPU-compatible struct with
`#[quanta::gpu_type]`:

```rust
#[quanta::gpu_type]
struct Particle {
    pos: [f32; 3],
    vel: [f32; 3],
    mass: f32,
}
```

The macro generates:

1. **`#[repr(C)]`** -- deterministic memory layout matching GPU expectations
2. **`Copy + Clone`** -- required for GPU data transfer
3. **`GpuType` impl** -- enables use in fields and buffers
4. **MSL declaration** -- e.g., `struct Particle { float3 pos; float3 vel; float mass; };`
5. **WGSL declaration** -- e.g., `struct Particle { pos: vec3<f32>, vel: vec3<f32>, mass: f32, };`
6. **`GPU_FIELDS`** -- field name, type, and byte offset metadata
7. **`GPU_SIZE`** -- compile-time struct size constant

Use it in a Fields struct:

```rust
#[derive(quanta::Fields)]
struct Simulation {
    particles: Vec<Particle>,
    dt: f32,
}

#[quanta::kernel]
fn integrate(sim: &Simulation) {
    let i = quark_id();
    sim.particles[i].pos[0] += sim.particles[i].vel[0] * sim.dt;
    sim.particles[i].pos[1] += sim.particles[i].vel[1] * sim.dt;
    sim.particles[i].pos[2] += sim.particles[i].vel[2] * sim.dt;
}
```

### Supported field types in gpu_type

| Type | GPU representation |
|------|--------------------|
| `f32`, `u32`, `i32` | Scalar |
| `f64`, `u64`, `i64` | 64-bit scalar |
| `u8`, `i8`, `u16`, `i16` | Narrow scalar |
| `[f32; 2]`, `[f32; 3]`, `[f32; 4]` | float2/3/4 (vec2/3/4) |
| `[u32; 2]`, `[u32; 3]`, `[u32; 4]` | uint2/3/4 |
| `[f32; 9]` | float3x3 (mat3x3) |
| `[f32; 16]` | float4x4 (mat4x4) |
| `[T; N]` (other sizes) | Plain array |

### Variable-length data

GPU structs must have fixed size. For variable-length data (strings, dynamic
arrays), use the **offset + length** pattern:

```rust
#[quanta::gpu_type]
struct TextSpan {
    offset: u32,    // byte offset into a separate data field
    length: u32,    // number of bytes/elements
}
```

Store the variable data in a separate `Vec<u8>` or `Vec<f32>` field in your
Fields struct, and index into it using the offset and length.

## Memory layout

Fields store elements contiguously in memory. A `Field<f32>` with 4 elements:

```
[f32][f32][f32][f32]
 0    4    8    12   bytes
```

For multi-component data, you have two layout strategies:

**AOS** (Array of Structures) -- one Vec with a struct per element:

```rust
#[derive(quanta::Fields)]
struct AOS {
    particles: Vec<Particle>,  // [Particle, Particle, ...]
}
```

**SOA** (Structure of Arrays) -- separate Vecs per component:

```rust
#[derive(quanta::Fields)]
struct SOA {
    x: Vec<f32>,   // [x0, x1, x2, ...]
    y: Vec<f32>,   // [y0, y1, y2, ...]
    vx: Vec<f32>,  // [vx0, vx1, vx2, ...]
    vy: Vec<f32>,  // [vy0, vy1, vy2, ...]
}
```

SOA typically gives better GPU memory coalescing when kernels access one
component at a time. AOS is simpler when kernels access all components
together. The N-body example in `api_design_dream.rs` uses SOA for
exactly this reason.

## Mapped fields (zero-copy)

For data that changes every frame (uniforms, streaming vertices), mapped
fields avoid the copy overhead:

```rust
let mut mapped = gpu.field_mapped::<f32>(256)?;

// Write directly to GPU-visible memory
mapped.write(0, 42.0);
mapped.write(1, 99.0);

// Read back without a GPU transfer
let val = mapped.read(0); // 42.0

// Slice access
let slice = mapped.as_slice();
let mut_slice = mapped.as_mut_slice();
```

On unified memory architectures (Apple Silicon), writes are immediately
visible to the GPU. On discrete GPUs (NVIDIA, AMD), the driver synchronizes
automatically on the next dispatch.

## Lifetime

Fields are freed when dropped. The GPU handle is released and memory is
reclaimed. Do not dispatch a wave that references a dropped field -- this
is undefined behavior at the driver level.

When using `#[derive(quanta::Fields)]`, field lifetime is managed
automatically by the generated dispatch code.

## Next

- [Shared memory](shared-memory.md) -- fast workgroup-local storage
- [Atomics](atomics.md) -- thread-safe GPU operations
