# Memory Model

GPU memory is a hierarchy. Faster memory is smaller and closer to the quarks.

```
Registers        per-quark     fastest   implicit (local variables)
     |
Shared memory    per-nucleus   ~48KB     #[quanta::shared]
     |
Global memory    all quarks    8-80GB    gpu.field() / gpu.field_mapped()
     |
CPU memory       host only     system    Vec<T>, &[T]
```

## Fields (global memory)

A Field is a typed GPU buffer. Large, visible to all quarks, but high latency.

```rust
// Allocate 1M floats on the GPU
let data = gpu.compute_field::<f32>(1_000_000)?;

// Upload from CPU
gpu.write_field(&data, &cpu_vec)?;

// Download to CPU
let result = gpu.read_field(&data)?;
```

Fields are the primary way to move data between CPU and GPU.

## Mapped memory

Zero-copy buffers visible to both CPU and GPU simultaneously.

```rust
let mapped = gpu.field_mapped::<f32>(1024)?;

// CPU writes directly to GPU-visible memory
mapped.write(0, 3.14);
mapped.write(1, 2.71);

// No copy needed — GPU sees the data on next dispatch
```

Use mapped memory for data that changes every frame (uniforms, streaming vertices).

## Shared memory

Small, fast memory shared by all quarks within one nucleus.

```rust
#[quanta::kernel]
fn reduce(data: &[f32], result: &mut [f32]) {
    #[quanta::shared] let local: [f32; 256];

    let lid = local_id();   // 0..255 within this nucleus
    let gid = quark_id();   // global index

    // Each quark loads one element into shared memory
    local[lid] = data[gid];

    // BARRIER: wait for all quarks in this nucleus to finish writing
    barrier();

    // Now every quark can read any element in local[]
    if lid == 0 {
        let mut sum = 0.0;
        for i in 0..256 {
            sum += local[i];
        }
        result[group_id()] = sum;
    }
}
```

## Registers

Per-quark, fastest storage. Every local variable in your kernel lives in a register.

```rust
#[quanta::kernel]
fn compute(a: &[f32], b: &[f32], out: &mut [f32]) {
    let i = quark_id();        // register
    let x = a[i];             // register (loaded from global)
    let y = b[i];             // register
    let result = x * y + x;   // register
    out[i] = result;           // write back to global
}
```

Registers are implicit. The GPU has thousands of them (unlike CPUs).

## Barriers

`barrier()` synchronizes all quarks within a nucleus. Required between shared memory writes and reads.

```
Without barrier:
  Quark 0: write local[0]           Quark 1: read local[0]  <-- RACE!

With barrier:
  Quark 0: write local[0]     Quark 1: write local[1]
                  |_________________________|
                          barrier()
  Quark 0: read local[1]      Quark 1: read local[0]  <-- safe
```

Rules:
- `barrier()` must be reached by ALL quarks in the nucleus (no conditional barriers).
- Only needed for shared memory. Fields have no cross-quark races (each quark writes its own index).

## Resource transitions

When a texture or field is used for different purposes between dispatches, insert a barrier:

```rust
// Compute writes to texture
gpu.dispatch(&compute_wave, 1024)?;

// Transition: compute-write -> shader-read
gpu.barrier_texture(&texture, ResourceState::ComputeWrite, ResourceState::ShaderRead)?;

// Fragment shader reads the texture
gpu.render_end(pass)?;
```

On Vulkan, this inserts a pipeline barrier with correct stage/access masks.
On Metal, this is a no-op (Metal tracks hazards automatically).
