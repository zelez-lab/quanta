# Compute Basics

How GPU execution works in Quanta. Read [getting started](../getting-started.md) first.

## Execution hierarchy

GPUs execute in a strict hierarchy. Quanta uses physics names instead of vendor jargon:

| Quanta   | NVIDIA        | AMD          | Metal           | What it is                   |
|----------|---------------|--------------|-----------------|------------------------------|
| Quark    | Thread        | Work item    | Thread          | Smallest unit of execution   |
| Proton   | Warp (32)     | Wave (64)    | SIMD group (32) | Quarks that run in lockstep  |
| Nucleus  | SM            | CU           | GPU core        | Group of protons             |
| Field    | Buffer        | Buffer       | Buffer          | GPU-resident data            |
| Wave     | Kernel launch | Dispatch     | Compute command | A bound kernel ready to fire |
| Pulse    | Fence         | Fence        | MTLEvent        | GPU completion signal        |
| Quanta   | The GPU       | The GPU      | The GPU         | All nuclei together          |

When you call `gpu.dispatch(&wave, 1024)`, the GPU schedules 1024 quarks across
its nuclei. Each quark runs the same kernel code with a different `quark_id()`.

Quarks within a proton execute in lockstep (SIMD). If your proton has 32 quarks
and one takes a branch, all 32 execute both paths (divergence). Minimize branching
in hot code.

## What is a field

A field is a typed GPU buffer. It holds a contiguous array of `T: Copy` elements.

```rust
let data = gpu.compute_field::<f32>(1_000_000)?;
```

This allocates space for 1 million `f32` values on the GPU. The data is not
initialized -- you must write to it before reading.

Fields are typed at the Rust level (`Field<f32>`, `Field<u32>`) but the GPU
sees raw bytes. The type parameter prevents accidentally binding a `Field<u32>`
where a `Field<f32>` is expected.

See [Fields and types](02-fields-and-types.md) for the full field API.

## The dispatch model

A kernel maps one quark to one piece of work. The standard pattern:

```rust
#[quanta::kernel]
fn scale(data: &mut [f32], factor: f32) {
    let i = quark_id();
    data[i] = data[i] * factor;
}
```

`quark_id()` is the global thread index. For 1D dispatches, it equals the
element index directly.

For 2D or 3D work, use `wave_dispatch` with explicit group counts:

```rust
let mut pulse = gpu.wave_dispatch(&wave, [width, height, 1])?;
```

Inside the kernel, combine `group_id()`, `local_id()`, and `group_size()` to
compute 2D coordinates:

```rust
#[quanta::kernel]
fn fill_2d(output: &mut [f32], width: u32) {
    let gid = group_id();
    let lid = local_id();
    let x = gid * group_size() + lid;
    // Derive row/col from x and width
}
```

## Thread indexing functions

| Function        | Returns                                          |
|-----------------|--------------------------------------------------|
| `quark_id()`    | Global thread index (0..total dispatched quarks) |
| `quark_count()` | Total number of dispatched quarks                |
| `local_id()`    | Thread index within the workgroup (0..group_size)|
| `group_id()`    | Workgroup index                                  |
| `group_size()`  | Number of quarks per workgroup                   |

## Push constants

Scalar values (not arrays) can be passed as push constants instead of fields.
Faster for small, per-dispatch data like a single `f32` threshold.

```rust
let mut wave = my_kernel(&gpu)?;
wave.bind(0, &data_field);
wave.set_value(1, 0.5f32);  // push constant at slot 1
```

Inside the kernel, the scalar parameter maps to the push constant slot:

```rust
#[quanta::kernel]
fn threshold_filter(data: &mut [f32], threshold: f32) {
    let i = quark_id();
    if data[i] < threshold {
        data[i] = 0.0;
    }
}
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
let field = gpu.compute_field::<f32>(n)
    .map_err(|e| e.with_context("allocating particle buffer"))?;
```

## Optimization levels

The `#[quanta::kernel]` macro accepts an optimization level:

```rust
#[quanta::kernel]              // default: O3 (aggressive optimization)
#[quanta::kernel(opt = "O2")]  // balanced
#[quanta::kernel(opt = "O0")]  // no optimization (for debugging)
```

O3 is the default. It enables LLVM's full optimization pipeline for the GPU
target, including loop unrolling, vectorization, and register allocation. Use
O0 only when debugging kernel correctness -- it produces significantly slower
code.

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

This information is useful for tuning dispatch sizes and workgroup dimensions.

## Multiple GPUs

`quanta::devices()` returns all available GPUs. Use this to select a specific
device or distribute work:

```rust
let gpus = quanta::devices();
for gpu in &gpus {
    println!("{}: {} nuclei", gpu.name(), gpu.nuclei());
}
let gpu = gpus.into_iter().next().expect("no GPU");
```

## Next

- [Fields and types](02-fields-and-types.md) -- memory management and supported types
- [Shared memory](03-shared-memory.md) -- fast workgroup-local storage
