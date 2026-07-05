# GPU Memory: Fields, Shared, and Constants

## The memory hierarchy

GPU memory is layered. Faster memory is smaller and closer to the quarks
(threads). Choosing the right layer is one of the most important performance
decisions in GPU programming.

```
                    Speed        Size          Scope
                  --------    ---------    ---------------
Registers          fastest     ~256KB*     per-quark (local variables)
    |
Shared memory      fast        ~48KB       per-workgroup (#[quanta::shared])
    |
Global memory      slow        8-80GB      all quarks (gpu.field())
    |
CPU memory         slowest     system RAM  host only (Vec<T>)

*total register file per nucleus, divided among active quarks
```

Think of it like CPU caches: registers are L1, shared memory is L2, global
memory is main RAM. The difference: on a GPU, you control the "cache" (shared
memory) explicitly.

## Fields: GPU-resident typed buffers

A Field is a typed buffer in GPU global memory. This is where your data lives
on the GPU -- the equivalent of a `Vec<T>` but on the graphics card.

```rust
// Allocate 1 million floats on the GPU
let data = gpu.field::<f32>(1_000_000)?;

// Upload from CPU to GPU
data.write(&cpu_vec)?;

// Run a kernel that reads/writes the field
gpu.dispatch(&wave, 1_000_000)?;

// Download results back to CPU
let result = data.read()?;
```

Fields are the primary bridge between CPU and GPU. Your kernel parameters
(`&[f32]`, `&mut [f32]`) bind to Fields.

### Structured fields

Fields can hold user-defined structs, not just scalars:

```rust
#[quanta::gpu_type]
struct Particle {
    pos: [f32; 3],
    vel: [f32; 3],
    mass: f32,
}

let particles = gpu.field::<Particle>(100_000)?;
```

The `#[quanta::gpu_type]` macro ensures the struct layout matches what the GPU
expects (`#[repr(C)]`, no padding surprises).

See [Guide: Fields and Types](../computation/tutorials/fields-and-types.md) for details.

## Push constants: small values sent per dispatch

Push constants are small scalar values (up to ~128 bytes) sent with each
dispatch. They avoid allocating a Field for values like `width`, `dt`, or
`frame_count`.

```rust
#[quanta::kernel]
fn step(particles: &mut [Particle], dt: f32, gravity: f32) {
    let i = quark_id();
    particles[i].vel[1] += gravity * dt;  // dt and gravity are push constants
    particles[i].pos[0] += particles[i].vel[0] * dt;
}
```

In this kernel, `particles` binds to a Field (large, on GPU). `dt` and
`gravity` are push constants (small values, sent from the CPU each frame).

Rule of thumb: if it fits in a few floats/ints that change per dispatch, use a
push constant. If it is an array or large struct, use a Field.

## Shared memory: fast per-workgroup cache

Shared memory is a small (~48KB), fast block of memory accessible to all quarks
in the same workgroup. It is roughly 100x faster than global memory for random
access patterns.

```rust
#[quanta::kernel]
fn reduce(data: &[f32], result: &mut [f32]) {
    #[quanta::shared] let cache: [f32; 256];

    let lid = local_id();
    let gid = quark_id();

    // Load from slow global memory into fast shared memory
    cache[lid] = data[gid];
    barrier();   // wait for all quarks to finish loading

    // Now all quarks can read any element in cache[] cheaply
    if lid == 0 {
        let mut sum = 0.0;
        for i in 0..256 { sum += cache[i]; }
        result[group_id()] = sum;
    }
}
```

Shared memory only lives for the duration of the dispatch, and is only visible
within one workgroup.

See [Guide: Shared Memory](../computation/tutorials/shared-memory.md) for tiling patterns.

## Textures: 2D/3D images with hardware sampling

A texture is an image stored on the GPU with hardware support for:
- **Filtering:** bilinear/trilinear interpolation between pixels
- **Addressing:** wrapping, clamping, mirroring at edges
- **Format conversion:** the hardware decodes RGBA8, R16F, etc.

```rust
let texture = gpu.texture_2d(width, height, Format::Rgba8Unorm)?;
gpu.write_texture(&texture, &pixel_data)?;
```

Inside a fragment shader, sample a texture with UV coordinates:

```rust
#[quanta::fragment]
fn textured(uv: Vec2, tex: &Texture) -> Vec4 {
    tex.sample(uv)
}
```

Compute kernels can also read and write textures for image processing.

See [Guide: Textures](../rendering/tutorials/textures.md) for formats and sampling modes.

## Mapped memory: zero-copy CPU-GPU sharing

For data that changes every frame, mapped memory avoids upload/download:

```rust
let mapped = gpu.field_mapped::<f32>(1024)?;
mapped.write(0, 3.14);   // CPU writes directly to GPU-visible memory
// GPU sees the data on next dispatch -- no copy needed
```

Use mapped memory for small, frequently-updated data (camera matrices, UI
state). For large compute buffers, dedicated Fields are faster for the GPU.

## Transfers

Data crosses the PCIe bus (or unified memory controller) when moving between
CPU and GPU. Minimize transfers by keeping data on the GPU between dispatches.

## When to use what

```
Is it a large array processed by the kernel?
  YES --> Field (gpu.field / gpu.field_with_usage)

Is it a small value that changes per dispatch? (dt, frame, resolution)
  YES --> Push constant (scalar parameter in your kernel)

Is it a 2D/3D image needing interpolation?
  YES --> Texture (gpu.texture_2d)

Is it data shared between quarks in one workgroup?
  YES --> Shared memory (#[quanta::shared])

Does the CPU update it every frame?
  YES --> Mapped memory (gpu.field_mapped)

Is it a local variable in your kernel?
  YES --> Register (automatic, no annotation needed)
```

## Registers

Every local variable in your kernel is a register -- the fastest storage on the
GPU. Unlike CPUs (~16 registers), a GPU nucleus has thousands shared among its
quarks. You never allocate them explicitly; `let x = a[i]` puts `x` in a
register automatically. Using too many reduces how many quarks can run
simultaneously (occupancy), so keep kernel functions lean.
