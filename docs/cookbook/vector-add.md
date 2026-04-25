# Vector Add

Add two arrays element-wise on the GPU.

## Kernel

```rust
#[quanta::kernel]
fn vector_add(a: &[f32], b: &[f32], result: &mut [f32]) {
    let i = quark_id();
    result[i] = a[i] + b[i];
}
```

Each quark (GPU thread) computes one element. `quark_id()` returns the global
thread index, so element `i` is handled by quark `i`.

## Host code (derive API)

```rust
#[derive(quanta::Fields)]
struct VectorData {
    a: Vec<f32>,
    b: Vec<f32>,
    result: Vec<f32>,
}

fn main() -> Result<(), quanta::QuantaError> {
    let gpu = quanta::init()?;

    let count = 1_000_000;
    let a_data: Vec<f32> = (0..count).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..count).map(|i| (i * 2) as f32).collect();

    // Allocate GPU fields
    let a = gpu.compute_field::<f32>(count)?;
    let b = gpu.compute_field::<f32>(count)?;
    let result = gpu.compute_field::<f32>(count)?;

    // Upload data
    a.write(&a_data)?;
    b.write(&b_data)?;

    // Create a wave (bound kernel) and bind fields
    let mut wave = vector_add(&gpu)?;
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &result);

    // Dispatch one quark per element and wait
    let mut pulse = gpu.dispatch(&wave, count as u32)?;
    pulse.wait()?;

    // Read back
    let output = result.read()?;
    assert!((output[42] - (42.0 + 84.0)).abs() < 0.001);
    Ok(())
}
```

The `#[derive(quanta::Fields)]` macro generates compile-time metadata for the
struct: which fields are `Vec<T>` (GPU storage buffers) and which are scalars
(push constants). This metadata drives automatic slot assignment and type
checking.

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
let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    entries: &[
        wgpu::BindGroupLayoutEntry { binding: 0, /* ... 8 fields ... */ },
        wgpu::BindGroupLayoutEntry { binding: 1, /* ... 8 fields ... */ },
        wgpu::BindGroupLayoutEntry { binding: 2, /* ... 8 fields ... */ },
    ],
    label: None,
});
let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    bind_group_layouts: &[&bgl], push_constant_ranges: &[], label: None,
});
let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    layout: Some(&pipeline_layout), module: &shader, entry_point: Some("main"),
    ..Default::default()
});
// ... create buffers, bind group, encoder, pass, dispatch, submit, readback ...
```

**Quanta**: kernel + 10 lines of host code. No WGSL file, no bind group layouts,
no pipeline layouts, no command encoders.

## How it works

1. `#[quanta::kernel]` compiles the function to GPU machine code at build time
   (MSL for Apple, PTX for NVIDIA, GCN for AMD, WGSL for WebGPU).
2. `#[derive(quanta::Fields)]` generates metadata for `VectorData` — three
   `Vec<f32>` fields become three GPU storage buffer slots.
3. `vector_add(&gpu)` returns a `Wave` — the compiled kernel bound to this device.
4. `wave.bind(slot, &field)` attaches GPU buffers to kernel parameters by position.
5. `gpu.dispatch(&wave, N)` launches N quarks in parallel.
6. `pulse.wait()` blocks until the GPU finishes.

## Slot mapping

Kernel parameters map to binding slots by declaration order:

| Slot | Parameter | Direction |
|------|-----------|-----------|
| 0    | `a`       | read      |
| 1    | `b`       | read      |
| 2    | `result`  | write     |

> **Note on deprecated methods.** Older examples may use `gpu.write_field(&f, &data)`,
> `gpu.read_field(&f)`, and `gpu.wait(&mut pulse)`. These still work but are
> deprecated. Prefer `field.write(data)`, `field.read()`, and `pulse.wait()`.
