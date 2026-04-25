# Compute

How GPU execution works in Quanta. Read [getting started](../getting-started.md) first.

## The derive-based API

Define your data layout with `#[derive(quanta::Fields)]`, write the kernel,
call it. The framework handles allocation, binding, and readback.

```rust
use quanta::*;

#[derive(quanta::Fields)]
struct Scale {
    data: Vec<f32>,
    factor: f32,
}

#[quanta::kernel]
fn scale(s: &Scale) {
    let i = quark_id();
    s.data[i] = s.data[i] * s.factor;
}

fn main() -> Result<(), QuantaError> {
    let gpu = init()?;

    let mut s = Scale {
        data: vec![1.0, 2.0, 3.0, 4.0],
        factor: 10.0,
    };

    scale(&gpu, &mut s, 4)?.wait()?;

    assert_eq!(s.data, vec![10.0, 20.0, 30.0, 40.0]);
    Ok(())
}
```

`Vec<T>` fields become GPU storage buffers. Scalar fields (`f32`, `u32`, etc.)
become push constants -- small per-dispatch values baked into the command
stream, faster than buffer reads for single values.

## Execution hierarchy

GPUs execute in a strict hierarchy. Quanta uses physics names instead of
vendor jargon:

| Quanta   | NVIDIA        | AMD          | Metal           | What it is                   |
|----------|---------------|--------------|-----------------|------------------------------|
| Quark    | Thread        | Work item    | Thread          | Smallest unit of execution   |
| Proton   | Warp (32)     | Wave (64)    | SIMD group (32) | Quarks that run in lockstep  |
| Nucleus  | SM            | CU           | GPU core        | Group of protons             |
| Field    | Buffer        | Buffer       | Buffer          | GPU-resident data            |
| Wave     | Kernel launch | Dispatch     | Compute command | A bound kernel ready to fire |
| Pulse    | Fence         | Fence        | MTLEvent        | GPU completion signal        |
| Quanta   | The GPU       | The GPU      | The GPU         | All nuclei together          |

When you dispatch 1024 quarks, the GPU schedules them across its nuclei. Each
quark runs the same kernel code with a different `quark_id()`.

Quarks within a proton execute in lockstep (SIMD). If your proton has 32
quarks and one takes a branch, all 32 execute both paths (divergence).
Minimize branching in hot code.

## Thread indexing functions

| Function         | Returns                                          |
|------------------|--------------------------------------------------|
| `quark_id()`     | Global thread index (0..total dispatched quarks) |
| `quark_count()`  | Total number of dispatched quarks                |
| `proton_id()`    | Thread index within the workgroup (0..proton_size)|
| `nucleus_id()`   | Workgroup index                                  |
| `proton_size()`  | Number of quarks per workgroup                   |

## 2D and 3D dispatch

For image processing or volume work, use 2D/3D workgroups and combine
indexing functions to compute coordinates:

```rust
#[derive(quanta::Fields)]
struct Image {
    input: Vec<f32>,
    output: Vec<f32>,
    width: u32,
}

#[quanta::kernel(workgroup = [16, 16, 1])]
fn blur(img: &Image) {
    let gid = nucleus_id();
    let lid = proton_id();
    let x = gid * proton_size() + lid;
    // Derive row/col from x and img.width
}
```

For explicit 2D/3D dispatch sizes, use `wave_dispatch` on the manual API
(see [Expert: Manual API](../expert/manual-api.md)).

## Workgroup size

By default, the driver picks a workgroup size. To set it explicitly:

```rust
// 1D workgroup (256 quarks per group)
#[quanta::kernel(workgroup = [256, 1, 1])]

// 2D workgroup (16x16 = 256 quarks per group, good for image processing)
#[quanta::kernel(workgroup = [16, 16, 1])]

// 3D workgroup (8x8x4 = 256 quarks per group, good for volume processing)
#[quanta::kernel(workgroup = [8, 8, 4])]
```

The workgroup size is baked into the compiled binary. Choose a size that is
a multiple of the proton width (32 for NVIDIA/Apple, 64 for AMD) for best
occupancy. When in doubt, `[256, 1, 1]` is a safe default for 1D work, and
`[16, 16, 1]` for 2D.

Combine `workgroup` with `opt`:

```rust
#[quanta::kernel(workgroup = [16, 16, 1], opt = "O2")]
fn blur(img: &Image) {
    // ...
}
```

## Optimization levels

```rust
#[quanta::kernel]              // default: O3 (aggressive optimization)
#[quanta::kernel(opt = "O2")]  // balanced
#[quanta::kernel(opt = "O0")]  // no optimization (for debugging)
```

O3 is the default. It enables LLVM's full optimization pipeline for the GPU
target, including loop unrolling, vectorization, and register allocation.
Use O0 only when debugging kernel correctness.

## Const generics

Kernels support const generic parameters:

```rust
#[derive(quanta::Fields)]
struct TiledData {
    data: Vec<f32>,
    output: Vec<f32>,
}

#[quanta::kernel]
fn tiled_reduce<const TILE: u32>(d: &TiledData) {
    let lid = proton_id();
    if lid < TILE {
        // ...
    }
}

// Call with specific tile size:
let gpu = quanta::init()?;
tiled_reduce::<256>(&gpu, &mut data, n)?.wait()?;
```

## Error handling

All GPU operations return `Result<T, QuantaError>`. Error kinds:

| Kind                | Meaning                                |
|---------------------|----------------------------------------|
| `NoDevice`          | No GPU found on the system             |
| `OutOfMemory`       | GPU memory allocation failed           |
| `CompilationFailed` | Kernel compilation error (with message)|
| `SubmitFailed`      | Command submission failed              |
| `Timeout`           | GPU operation timed out                |
| `DeviceLost`        | GPU disconnected or driver crashed     |
| `InvalidParam`      | Bad parameter (with message)           |

Attach context to errors for debugging:

```rust
let gpu = quanta::init()
    .map_err(|e| e.with_context("initializing GPU for particle sim"))?;
```

## Device information

Query GPU capabilities at runtime:

```rust
let gpu = quanta::init()?;
println!("GPU: {}", gpu.name());
println!("Nuclei: {}", gpu.nuclei());
println!("Protons per nucleus: {}", gpu.protons_per_nucleus());
println!("Quarks per proton: {}", gpu.quarks_per_proton());
println!("Total quarks: {}", gpu.total_quarks());
println!("Memory: {} MB", gpu.caps().memory_bytes / 1_000_000);
```

## Multiple GPUs

`quanta::devices()` returns all available GPUs:

```rust
let gpus = quanta::devices();
for gpu in &gpus {
    println!("{}: {} nuclei", gpu.name(), gpu.nuclei());
}
let gpu = gpus.into_iter().next().expect("no GPU");
```

## Saturation arithmetic

Use `.saturating_add()` and `.saturating_sub()` for clamped arithmetic
that never wraps:

```rust
#[derive(quanta::Fields)]
struct Clamped {
    a: Vec<u32>,
    b: Vec<u32>,
    out: Vec<u32>,
}

#[quanta::kernel]
fn clamp_add(c: &Clamped) {
    let i = quark_id();
    c.out[i] = c.a[i].saturating_add(c.b[i]); // clamps to u32::MAX on overflow
}
```

## Next

- [Fields and types](02-fields-and-types.md) -- memory management and supported types
- [Shared memory](03-shared-memory.md) -- fast workgroup-local storage
