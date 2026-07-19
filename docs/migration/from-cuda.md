# Migration from CUDA

## Terminology

| CUDA | Quanta | Notes |
|------|--------|-------|
| Thread | Quark | Smallest execution unit |
| Warp | Proton | 32 quarks in lockstep |
| Block / SM | Nucleus | Compute unit with shared memory |
| Grid | Dispatch | All quarks in one launch |
| Device memory | Field | Typed GPU buffer |

## Side-by-side: vector addition

### CUDA (15 lines of host boilerplate)

```c
__global__ void vector_add(float *a, float *b, float *result, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) result[i] = a[i] + b[i];
}

int main() {
    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, N * sizeof(float));          // 1. allocate
    cudaMalloc(&d_b, N * sizeof(float));          // 2. allocate
    cudaMalloc(&d_result, N * sizeof(float));     // 3. allocate
    cudaMemcpy(d_a, h_a, N * sizeof(float),       // 4. upload
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float),       // 5. upload
               cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b,     // 6. dispatch
                                     d_result, N);
    cudaDeviceSynchronize();                       // 7. wait
    cudaMemcpy(h_result, d_result,                 // 8. download
               N * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_a);                                 // 9. free
    cudaFree(d_b);                                 // 10. free
    cudaFree(d_result);                            // 11. free
}
```

### Quanta (5 lines of host code)

```rust
#[derive(quanta::Fields)]
struct VectorData { a: Vec<f32>, b: Vec<f32>, result: Vec<f32> }

#[quanta::kernel]
fn vector_add(a: &[f32], b: &[f32], result: &mut [f32]) {
    let i = quark_id();
    result[i] = a[i] + b[i];
}

fn main() -> Result<(), quanta::QuantaError> {
    let gpu = quanta::init()?;
    let a = gpu.field::<f32>(N)?;          // allocate
    let b = gpu.field::<f32>(N)?;
    let result = gpu.field::<f32>(N)?;
    a.write(&h_a)?;                                // upload
    b.write(&h_b)?;

    let mut wave = vector_add(&gpu)?;              // compile
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &result);

    let mut pulse = gpu.dispatch(&wave, N as u32)?; // dispatch
    pulse.wait()?;                                  // wait

    let h_result = result.read()?;                  // download
    Ok(())                                          // free is automatic (RAII)
}
```

**What disappears:**
- `cudaMalloc`/`cudaFree` -- RAII handles allocation and deallocation
- `cudaMemcpyHostToDevice`/`cudaMemcpyDeviceToHost` -- `field.write()`/`field.read()`
- Block/grid dimension math -- `gpu.dispatch(&wave, N)` handles it
- `cudaError_t` checking -- Rust `Result<T, QuantaError>` everywhere
- Separate `.cu` files -- kernel is Rust, same compiler

## API mapping

| CUDA | Quanta |
|------|--------|
| `__global__ void kernel(...)` | `#[quanta::kernel] fn kernel(...)` |
| `__device__ void helper(...)` | `#[quanta::device] fn helper(...)` |
| `struct` in device code | `#[quanta::gpu_type] struct` |
| `__shared__ float s[256]` | `#[quanta::shared] let s: [f32; 256]` |
| `threadIdx.x` | `local_id()` |
| `blockIdx.x` | `group_id()` |
| `blockDim.x * blockIdx.x + threadIdx.x` | `quark_id()` |
| `gridDim.x * blockDim.x` | `quark_count()` |
| `__syncthreads()` | `barrier()` |
| `atomicAdd(&x, val)` | `atomic_add(&mut x, val)` |
| `atomicCAS(&x, expected, desired)` | `atomic_compare_exchange(&mut x, expected, desired)` |
| `__shfl_xor_sync(mask, val, delta)` | `shuffle_f32(val, delta)` (and `_u32` / `_i32`) |
| `__ballot_sync(mask, pred)` | `ballot_u32(pred)` |
| `__any_sync(mask, pred)` | `any_u32(pred)` |
| `__all_sync(mask, pred)` | `all_u32(pred)` |
| `surf2Dwrite(v, surf, x, y)` | `texture_write_2d(tex, x, y, v)` (param `&mut Texture2D<f32>`, R32Float) |
| `surf2Dread(&v, surf, x, y)` | `texture_load_2d(tex, x, y)` (texel read; `&Texture2D` for a read-only surface, `&mut` if the kernel also writes it) |
| `surf2Dwrite(uchar4_v, surf, x, y)` | `texture_write_2d(tex, x, y, packed)` (param `&mut Texture2D<u32>`, RGBA8 texel as one packed `0xAABBGGRR` u32) |
| `tex2D(tex, u, v)` | `texture_sample_2d(tex, x, y)` (param `&Sampled2D<f32>` — nearest/clamp sampled access) |
| `cudaMemcpy2D` (to a texture sub-region) | `texture.write_region(origin, size, &data)` (texel offset + extent, tightly packed rows; gated on `supports_texture_write_region`) |
| `cudaMalloc` + `cudaMemcpy` | `gpu.field::<T>(n)` + `field.write(&data)` |
| `cudaMallocManaged` | `gpu.field_mapped::<T>(n)` |
| `kernel<<<blocks, threads>>>(...)` | `gpu.dispatch(&wave, n)` |
| `cudaDeviceSynchronize()` | `pulse.wait()` / `gpu.wait_idle()` |
| `cudaLaunchHostFunc` / `cudaStreamAddCallback` | `pulse.on_complete(f)` (run `f` on a background waiter at completion — the event-driven alternative to `wait()`) |
| `cudaGetDeviceProperties` | `gpu.caps()` |
| `cudaFree` | automatic (Field drops when it goes out of scope) |

## CUB block primitives

The `quanta::prims` module covers the `cub::Block*` surface —
see [Block Primitives](../computation/tutorials/block-primitives.md):

| CUB | quanta::prims |
|-----|--------------|
| `cub::BlockReduce<T, 256>::Sum(v)` | `block_reduce_add_u32_kernel(v)` (and `min`/`max` × `u32`/`i32`/`f32`) |
| `cub::BlockScan<T, 256>::InclusiveSum(v)` | `block_scan_add_u32_kernel(v)` |
| `cub::BlockRadixSort<u32, 256, 1>::Sort` | `block_radix_sort_u32_buffer` kernel |
| `cub::DeviceReduce::Sum` | `device_reduce_add_u32(&gpu, &data)` |
| `cub::DeviceRadixSort::SortKeys` | `device_sort_u32(&gpu, &data)` |
| `cub::BlockHistogram` | `block_histogram_u32_buffer` |

## Workgroup (block) size

CUDA uses `dim3` for block dimensions:
```c
dim3 block(16, 16, 1);
dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
kernel<<<grid, block>>>(...);
```

Quanta uses the `workgroup` attribute:
```rust
#[quanta::kernel(workgroup = [16, 16, 1])]
fn process_image(input: &[f32], output: &mut [f32], width: u32) {
    let i = quark_id();
    output[i] = input[i] * 2.0;
}
```

| CUDA | Quanta |
|------|--------|
| `dim3 block(256)` | `#[quanta::kernel(workgroup = [256, 1, 1])]` |
| `dim3 block(16, 16)` | `#[quanta::kernel(workgroup = [16, 16, 1])]` |
| `dim3 block(8, 8, 4)` | `#[quanta::kernel(workgroup = [8, 8, 4])]` |

## Subgroup (warp) operations

CUDA warp intrinsics map directly to Quanta wave/subgroup operations:

| CUDA | Quanta | Description |
|------|--------|-------------|
| `__shfl_xor_sync(mask, val, delta)` | `shuffle_u32/i32/f32(val, delta)` | Cross-lane XOR (butterfly) exchange |
| `__ballot_sync(mask, pred)` | `ballot_u32(pred)` | Lane predicate bitmask |
| `__any_sync(mask, pred)` | `any_u32(pred)` | Any lane satisfies predicate |
| `__all_sync(mask, pred)` | `all_u32(pred)` | All lanes satisfy predicate |
| `__reduce_add_sync(mask, val)` | `reduce_add_u32/i32/f32(val)` | Sum across warp |
| `__reduce_min_sync(mask, val)` | `reduce_min_u32/i32/f32(val)` | Min across warp |
| `__reduce_max_sync(mask, val)` | `reduce_max_u32/i32/f32(val)` | Max across warp |
| (manual scan loop) | `scan_add_exclusive_u32/i32/f32(val)` | Exclusive prefix sum |
| (manual scan loop) | `scan_add_u32/i32/f32(val)` | Inclusive prefix sum |

The intrinsics are `unsafe extern` imports inside a `#[quanta::kernel]` body
(wrap calls in `unsafe {}`). Quanta does not require an explicit `mask`
parameter -- all active lanes participate. This matches the SPIR-V and Metal
subgroup model.

Backend note: `any` / `all` / `ballot` run on every backend, including
Broadcom V3D (Raspberry Pi 5). The reduce / scan / shuffle intrinsics need
the subgroup ARITHMETIC / SHUFFLE feature classes, which V3D does not
advertise -- gate on `gpu.supports_subgroups()` or provide a shared-memory
fallback (see the [wave intrinsics
tutorial](../computation/tutorials/wave-intrinsics.md#backend-support)).

## Shared memory

### CUDA
```c
__global__ void reduce(float *data, float *result) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    sdata[tid] = data[i];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) result[blockIdx.x] = sdata[0];
}
```

### Quanta
```rust
#[quanta::kernel]
fn reduce(data: &[f32], result: &mut [f32]) {
    #[quanta::shared] let sdata: [f32; 256];
    let tid = local_id();
    let i = quark_id();
    sdata[tid] = data[i];
    barrier();

    let mut s = group_size() / 2;
    loop {
        if s == 0 { break; }
        if tid < s {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        barrier();
        s = s / 2;
    }
    if tid == 0 {
        result[group_id()] = sdata[0];
    }
}
```

## Structured data

### CUDA
```c
struct Particle {
    float3 pos;
    float3 vel;
    float mass;
};

__global__ void update(Particle *particles, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        particles[i].pos.x += particles[i].vel.x * dt;
        // ...
    }
}
```

### Quanta
```rust
#[quanta::gpu_type]
struct Particle {
    pos: [f32; 3],
    vel: [f32; 3],
    mass: f32,
}

let particles = gpu.field::<Particle>(n)?;
```

`#[quanta::gpu_type]` is the Quanta equivalent of a CUDA `struct` used in device
code. It ensures `repr(C)` layout matching GPU expectations and generates
MSL/WGSL struct declarations automatically. No separate `.cuh` header needed.

## Key differences

**Build-time compilation.** CUDA compiles kernels with nvcc at build time (or PTX at runtime
via the driver). Quanta compiles at build time via proc macros -- the kernel binary is embedded
in your Rust binary. No CUDA toolkit needed at runtime.

**Cross-vendor.** The same `#[quanta::kernel]` compiles to PTX (NVIDIA), GCN ELF (AMD),
metallib (Apple), and SPIR-V (Vulkan). One source, all GPUs. All output is native binary.

**No bounds checking in the example.** CUDA requires manual `if (i < n)` guards because
you launch more threads than elements (grid must be a multiple of block size). Quanta
dispatches the exact number of quarks needed.

**No manual memory management.** Fields drop automatically (RAII). No `cudaFree`.

**Type safety.** `Field<f32>` cannot be bound where `Field<u32>` is expected. CUDA uses
`void*` everywhere.

## v0.1 advanced features

CUDA developers reaching for OptiX, mesh shaders, or multi-stream work will
find typed wrappers in Quanta. The render-side constructors
(`acceleration_structure_blas`, `ray_tracing_pipeline`, `mesh_pipeline`,
`tessellation_pipeline`) are `RenderGpu` extension-trait methods — add
`use quanta::RenderGpu;` (or `use quanta::*;`) and build with the
`render` feature on:

| CUDA / NVIDIA            | Quanta                                                       |
|--------------------------|--------------------------------------------------------------|
| OptiX `optixAccel*` BLAS | `gpu.acceleration_structure_blas(&[GeometryDesc { .. }])`    |
| OptiX pipeline           | `gpu.ray_tracing_pipeline(&RayTracingPipelineDesc { .. })`   |
| `optixLaunch`            | `pipeline.dispatch_rays(width, height)`                       |
| Mesh shader extension    | `gpu.mesh_pipeline(MeshPipelineDesc { .. })` + `dispatch`     |
| Tessellation             | `gpu.tessellation_pipeline(TessTopology::Triangle, n)`        |
| `cudaStreamCreate`       | `gpu.queue(QueueType::Compute)?` (one queue per stream)       |
| `cudaMemcpyAsync` on stream | `gpu.async_copy_queue()?.copy_buffer(&dst, &src, n)`       |
| `printf` from kernel     | `gpu.printf_buffer(cap)?.drain()?`                            |
| Indirect launch (CUDA Graphs) | `gpu.indirect_command_buffer(cap)?` + `record_dispatch` |
| Sparse memory (`cuMemMap`)| `gpu.sparse_texture(&desc)?.map_tile(mip, x, y, backing)`    |

Each typed wrapper has a capability query — call
`gpu.supports_ray_tracing()`, `supports_mesh_shaders()`,
`supports_tessellation()`, `supports_vrs()`, `supports_sparse_residency()`
before constructing, and branch on `QuantaErrorKind::NotSupported` to fall
back when the active backend doesn't implement the feature.

See [Compute basics: Capability queries](../getting-started.md#capability-queries-and-graceful-fallback)
for the full pattern.

## What you won't miss

- `cudaError_t` checking after every call (Quanta uses `Result<T, QuantaError>`)
- Separate `.cu` files with a different compiler
- CUDA toolkit installation and version management
- Platform lock-in to NVIDIA hardware
