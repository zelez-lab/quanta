# Migration from CUDA

## Terminology

| CUDA | Quanta | Notes |
|------|--------|-------|
| Thread | Quark | Smallest execution unit |
| Warp | Proton | 32 quarks in lockstep |
| Block / SM | Nucleus | Compute unit with shared memory |
| Grid | Dispatch | All quarks in one launch |
| Device memory | Field | Typed GPU buffer |

## API mapping

| CUDA | Quanta |
|------|--------|
| `__global__ void kernel(...)` | `#[quanta::kernel] fn kernel(...)` |
| `__device__ void helper(...)` | `#[quanta::device] fn helper(...)` |
| `__shared__ float s[256]` | `#[quanta::shared] let s: [f32; 256]` |
| `threadIdx.x` | `local_id()` |
| `blockIdx.x` | `group_id()` |
| `blockDim.x * blockIdx.x + threadIdx.x` | `quark_id()` |
| `gridDim.x * blockDim.x` | `quark_count()` |
| `__syncthreads()` | `barrier()` |
| `atomicAdd(&x, val)` | `atomic_add(&mut x, val)` |
| `atomicCAS(&x, expected, desired)` | `atomic_cas(&mut x, expected, desired)` |
| `__shfl_xor_sync(mask, val, delta)` | `wave_shuffle(val, delta)` |
| `__ballot_sync(mask, pred)` | `wave_ballot(pred)` |
| `__any_sync(mask, pred)` | `wave_any(pred)` |
| `__all_sync(mask, pred)` | `wave_all(pred)` |
| `cudaMalloc` + `cudaMemcpy` | `gpu.compute_field::<T>(n)` + `gpu.write_field(&f, &data)` |
| `cudaMallocManaged` | `gpu.field_mapped::<T>(n)` |
| `kernel<<<blocks, threads>>>(...)` | `gpu.dispatch(&wave, n)` |
| `cudaDeviceSynchronize()` | `gpu.wait(&mut pulse)` |
| `cudaGetDeviceProperties` | `gpu.caps()` |

## Example: vector addition

### CUDA

```c
__global__ void vector_add(float *a, float *b, float *result, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        result[i] = a[i] + b[i];
    }
}

int main() {
    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_result, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_result);
}
```

### Quanta

```rust
#[quanta::kernel]
fn vector_add(a: &[f32], b: &[f32], result: &mut [f32]) {
    let i = quark_id();
    result[i] = a[i] + b[i];
}

fn main() -> Result<(), quanta::QuantaError> {
    let gpu = quanta::init()?;

    let a = gpu.compute_field::<f32>(N)?;
    let b = gpu.compute_field::<f32>(N)?;
    let result = gpu.compute_field::<f32>(N)?;
    gpu.write_field(&a, &h_a)?;
    gpu.write_field(&b, &h_b)?;

    let mut wave = vector_add(&gpu)?;
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &result);

    let mut pulse = gpu.dispatch(&wave, N as u32)?;
    gpu.wait(&mut pulse)?;

    let h_result = gpu.read_field(&result)?;
    Ok(())
}
```

## Key differences

**Build-time compilation.** CUDA compiles kernels with nvcc at build time (or PTX at runtime
via the driver). Quanta compiles at build time via proc macros — the kernel binary is embedded
in your Rust binary. No CUDA toolkit needed at runtime.

**Cross-vendor.** The same `#[quanta::kernel]` compiles to PTX (NVIDIA), GCN ELF (AMD),
MSL (Apple), WGSL (browsers), and SPIR-V (Vulkan). One source, all GPUs.

**No bounds checking in the example.** CUDA requires manual `if (i < n)` guards because
you launch more threads than elements (grid must be a multiple of block size). Quanta
dispatches the exact number of quarks needed.

**No manual memory management.** Fields drop automatically (RAII). No `cudaFree`.

**Type safety.** `Field<f32>` cannot be bound where `Field<u32>` is expected. CUDA uses
`void*` everywhere.

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

## What you won't miss

- `cudaError_t` checking after every call (Quanta uses `Result<T, QuantaError>`)
- Separate `.cu` files with a different compiler
- CUDA toolkit installation and version management
- Platform lock-in to NVIDIA hardware
