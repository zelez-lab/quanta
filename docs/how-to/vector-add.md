# Vector Add

Add two arrays element-wise on the GPU.

## Kernel

```rust
#[derive(quanta::Fields)]
struct VecAdd {
    a: Vec<f32>,
    b: Vec<f32>,
    result: Vec<f32>,
}

#[quanta::kernel]
fn vector_add(d: &VecAdd) {
    let i = quark_id();
    d.result[i] = d.a[i] + d.b[i];
}
```

Each quark (GPU thread) computes one element. `quark_id()` returns the
global thread index. The struct ties data layout to the kernel — every
`Vec<T>` field becomes a GPU storage buffer, allocated and bound
automatically.

## Host code

```rust
use quanta::*;

fn main() -> Result<(), QuantaError> {
    let gpu = quanta::init()?;

    let mut data = VecAdd {
        a: (0..1024).map(|i| i as f32).collect(),
        b: (0..1024).map(|i| (i * 2) as f32).collect(),
        result: vec![0.0f32; 1024],
    };

    // One line: upload, bind, dispatch, readback.
    vector_add(&gpu, &mut data, 1024)?.wait()?;

    assert_eq!(data.result[1023], 3069.0);
    println!("first 5: {:?}", &data.result[..5]);
    println!("last 5:  {:?}", &data.result[data.result.len() - 5..]);
    Ok(())
}
```

The single call `vector_add(&gpu, &mut data, 1024)?.wait()?` does
*everything*: allocates the three GPU buffers, uploads `a` and `b`,
dispatches 1024 quarks, blocks until they finish, and reads `result`
back into `data.result`.

## Comparison with CUDA

```c
// CUDA — 15 lines of boilerplate
__global__ void vector_add(float *a, float *b, float *result, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) result[i] = a[i] + b[i];
}

int main() {
    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    vector_add<<<(N+255)/256, 256>>>(d_a, d_b, d_result, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_result);
}
```

## Comparison with wgpu

```rust
// wgpu — 50+ lines of ceremony
let instance = wgpu::Instance::new(Backends::all());
let adapter = instance.request_adapter(&Default::default()).await.unwrap();
let (device, queue) = adapter.request_device(&Default::default(), None).await.unwrap();
let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: None,
    source: wgpu::ShaderSource::Wgsl(include_str!("vector_add.wgsl").into()),
});
let bgl = device.create_bind_group_layout(/* … 8 fields per binding … */);
let pipeline_layout = device.create_pipeline_layout(/* … */);
let pipeline = device.create_compute_pipeline(/* … */);
// … create buffers, bind group, encoder, pass, dispatch, submit, readback …
```

**Quanta**: kernel + ~10 lines of host code. No WGSL file, no bind group
layouts, no pipeline layouts, no command encoders.

## How it works

1. `#[quanta::kernel]` compiles the function to GPU machine code at build
   time (MSL for Apple, PTX for NVIDIA, GCN for AMD, WGSL for WebGPU,
   SPIR-V for Vulkan).
2. `#[derive(quanta::Fields)]` generates metadata for `VecAdd` — three
   `Vec<f32>` fields become three GPU storage buffer slots.
3. `vector_add(&gpu, &mut data, N)` is the auto-dispatch wrapper the
   `#[quanta::kernel]` macro emits for struct-ref kernels. It allocates,
   uploads, dispatches `N` quarks, and on `wait()` reads results back.
4. The returned `Pulse` is a GPU completion signal — `.wait()?` blocks
   until done.

## Slot mapping

The fields map to binding slots by declaration order:

| Slot | Field    | Direction |
|------|----------|-----------|
| 0    | `a`      | read      |
| 1    | `b`      | read      |
| 2    | `result` | write     |

## Manual API (advanced)

For finer control over allocation, binding, and dispatch — useful when
you want to reuse buffers across many dispatches, or call `wave_jit`
with a runtime-built `KernelDef` — see
[Expert: Manual API](../expert/manual-api.md).
