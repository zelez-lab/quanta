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

## Host code

```rust
fn main() {
    let gpu = quanta::init().expect("no GPU found");

    let count = 1_000_000;
    let a_data: Vec<f32> = (0..count).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..count).map(|i| (i * 2) as f32).collect();

    // Allocate GPU fields
    let a = gpu.compute_field::<f32>(count).unwrap();
    let b = gpu.compute_field::<f32>(count).unwrap();
    let result = gpu.compute_field::<f32>(count).unwrap();

    // Upload data
    gpu.write_field(&a, &a_data).unwrap();
    gpu.write_field(&b, &b_data).unwrap();

    // Create a wave (bound kernel) and bind fields
    let mut wave = vector_add(&gpu).expect("compile kernel");
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &result);

    // Dispatch one quark per element and wait
    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    // Read back
    let output = gpu.read_field(&result).unwrap();
    assert!((output[42] - (42.0 + 84.0)).abs() < 0.001);
}
```

## How it works

1. `#[quanta::kernel]` compiles the function to GPU machine code at build time
   (MSL for Apple, PTX for NVIDIA, GCN for AMD, WGSL for WebGPU).
2. `vector_add(&gpu)` returns a `Wave` — the compiled kernel bound to this device.
3. `wave.bind(slot, &field)` attaches GPU buffers to kernel parameters by position.
4. `gpu.dispatch(&wave, N)` launches N quarks in parallel.
5. `gpu.wait(&mut pulse)` blocks until the GPU finishes.

## Slot mapping

Kernel parameters map to binding slots by declaration order:

| Slot | Parameter | Direction |
|------|-----------|-----------|
| 0    | `a`       | read      |
| 1    | `b`       | read      |
| 2    | `result`  | write     |
