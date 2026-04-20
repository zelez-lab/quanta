# Fields and Types

GPU memory management in Quanta. Fields are typed GPU buffers.

## Supported scalar types

Any type implementing `GpuType` can be used in fields. Built-in implementations:

| Rust type | GPU size | Notes                              |
|-----------|----------|------------------------------------|
| `f32`     | 4 bytes  | Standard GPU float                 |
| `f64`     | 8 bytes  | Not all GPUs support double        |
| `u32`     | 4 bytes  | Standard GPU unsigned integer      |
| `i32`     | 4 bytes  | Standard GPU signed integer        |
| `u64`     | 8 bytes  | Atomics may not support 64-bit     |
| `i64`     | 8 bytes  |                                    |
| `u16`     | 2 bytes  | Half-width integer                 |
| `i16`     | 2 bytes  |                                    |
| `u8`      | 1 byte   | Byte-level access                  |
| `i8`      | 1 byte   |                                    |

`f32` and `u32` are universally supported and fastest on all GPUs. Prefer them
unless you need the extra range of 64-bit types.

## Creating fields

### From size (uninitialized)

```rust
let field = gpu.compute_field::<f32>(1024)?;
```

Allocates space for 1024 `f32` values with default compute usage flags
(read + write + compute + transfer). Data is uninitialized.

### With explicit usage flags

```rust
let field = gpu.field::<f32>(1024, FieldUsage::READ.union(FieldUsage::COMPUTE))?;
```

Usage flags tell the driver how the field will be accessed, enabling placement
optimizations.

| Flag                | Meaning                               |
|---------------------|---------------------------------------|
| `FieldUsage::READ`     | GPU will read from this field     |
| `FieldUsage::WRITE`    | GPU will write to this field      |
| `FieldUsage::COMPUTE`  | Used in compute dispatches        |
| `FieldUsage::RENDER`   | Used as vertex/index data         |
| `FieldUsage::TRANSFER` | Transferred to/from CPU           |
| `FieldUsage::UNIFORM`  | Used as a uniform buffer          |

### Convenience constructors

```rust
let compute = gpu.compute_field::<f32>(n)?;   // READ | WRITE | COMPUTE | TRANSFER
let render  = gpu.render_field::<f32>(n)?;    // READ | RENDER | TRANSFER
let uniform = gpu.uniform_field::<f32>(n)?;   // READ | UNIFORM | TRANSFER
```

## Writing data to a field

```rust
let data = vec![1.0f32, 2.0, 3.0, 4.0];
let field = gpu.compute_field::<f32>(4)?;
gpu.write_field(&field, &data)?;
```

`write_field` copies CPU data into GPU memory. The slice length must not exceed
the field's element count.

## Reading data back

```rust
let output = gpu.read_field::<f32>(&field)?;
// output: Vec<f32> with field.len() elements
```

`read_field` copies GPU memory back to a CPU `Vec<T>`. This is a synchronous
operation -- ensure the GPU has finished writing (wait on the pulse) before
reading.

## Copying between fields

```rust
gpu.copy_field(&dst, &src)?;
```

Copies data GPU-to-GPU. Copies `min(dst.byte_size(), src.byte_size())` bytes.
Faster than reading to CPU and writing back.

## Resizing fields

```rust
let bigger = gpu.resize_field(&original, new_count, FieldUsage::default_compute())?;
```

Allocates a new field and copies existing data. The old field remains valid
until dropped.

## Mapped fields (zero-copy)

For data that changes every frame (uniforms, streaming vertices), mapped fields
avoid the copy overhead:

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

On unified memory architectures (Apple Silicon), writes are immediately visible
to the GPU. On discrete GPUs (NVIDIA, AMD), the driver synchronizes
automatically on the next dispatch.

## Field properties

```rust
let field = gpu.compute_field::<f32>(1024)?;
assert_eq!(field.len(), 1024);           // element count
assert_eq!(field.byte_size(), 4096);     // total bytes (1024 * 4)
assert!(!field.is_empty());
```

## Memory layout

Fields store elements contiguously in memory. A `Field<f32>` with 4 elements
is laid out as:

```
[f32][f32][f32][f32]
 0    4    8    12   bytes
```

This is Array of Structures (AOS) layout when each element is a scalar. For
multi-component data (like particles with position + velocity), you have two
choices:

**AOS** -- one field with a struct per element:

```rust
#[derive(Copy, Clone)]
#[repr(C)]
struct Particle { x: f32, y: f32, vx: f32, vy: f32 }

let particles = gpu.compute_field::<Particle>(1000)?;
```

**SOA** -- separate fields per component:

```rust
let x  = gpu.compute_field::<f32>(1000)?;
let y  = gpu.compute_field::<f32>(1000)?;
let vx = gpu.compute_field::<f32>(1000)?;
let vy = gpu.compute_field::<f32>(1000)?;
```

SOA typically gives better GPU memory coalescing when kernels access one
component at a time. AOS is simpler when kernels access all components together.

## Lifetime

Fields are freed when dropped. The GPU handle is released and memory is
reclaimed. Do not dispatch a wave that references a dropped field -- this is
undefined behavior at the driver level.

## Next

- [Shared memory](03-shared-memory.md) -- fast workgroup-local storage
- [Atomics](04-atomics.md) -- thread-safe GPU operations
