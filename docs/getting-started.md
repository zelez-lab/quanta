# Getting Started

GPU compute in 5 minutes. You know Rust. You may not know GPUs.

## Add dependency

```sh
cargo add quanta
```

## Write a kernel

A kernel is a function that runs on the GPU. Thousands of copies run in parallel,
each on a different element.

```rust
use quanta::*;

#[quanta::kernel]
fn vector_add(a: &[f32], b: &[f32], result: &mut [f32]) {
    let i = quark_id();
    result[i] = a[i] + b[i];
}
```

`quark_id()` returns this thread's index. If you dispatch 1024 quarks,
`quark_id()` ranges from 0 to 1023.

The `#[quanta::kernel]` attribute compiles this function to GPU machine code
at build time. On your CPU, `vector_add` becomes a function that creates a
`Wave` (a bound, dispatchable kernel).

## Run it

```rust
use quanta::*;

#[quanta::kernel]
fn vector_add(a: &[f32], b: &[f32], result: &mut [f32]) {
    let i = quark_id();
    result[i] = a[i] + b[i];
}

fn main() -> Result<(), QuantaError> {
    let gpu = init()?;

    let data_a = vec![1.0f32; 1024];
    let data_b = vec![2.0f32; 1024];

    // Allocate GPU fields and upload data
    let a = gpu.compute_field::<f32>(1024)?;
    let b = gpu.compute_field::<f32>(1024)?;
    let mut result = gpu.compute_field::<f32>(1024)?;

    gpu.write_field(&a, &data_a)?;
    gpu.write_field(&b, &data_b)?;

    // Create a wave and bind fields to kernel parameters
    let mut wave = vector_add(&gpu)?;
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &result);

    // Dispatch 1024 quarks, wait for completion
    let mut pulse = gpu.dispatch(&wave, 1024)?;
    gpu.wait(&mut pulse)?;

    let output = gpu.read_field::<f32>(&result)?;
    assert_eq!(output[0], 3.0);
    println!("GPU computed: {} elements", output.len());
    Ok(())
}
```

## What happened

1. **Build time**: `#[quanta::kernel]` compiled `vector_add` to native GPU binaries.
   On Apple GPUs it generates MSL. On AMD it generates GCN machine code via LLVM.
   On NVIDIA it generates PTX. All formats are embedded in your binary at compile time.

2. **`init()`**: Discovered the first available GPU and returned a `Gpu` handle.

3. **`compute_field`**: Allocated typed GPU memory (a field). `write_field` uploaded
   CPU data into it.

4. **`vector_add(&gpu)`**: Selected the right pre-compiled binary for your GPU vendor
   and created a `Wave` -- a kernel ready to dispatch.

5. **`wave.bind(slot, &field)`**: Bound each field to the corresponding kernel
   parameter by slot index (0 = `a`, 1 = `b`, 2 = `result`).

6. **`dispatch(&wave, 1024)`**: Launched 1024 quarks on the GPU. Each quark
   executed `vector_add` with its own `quark_id()`. Returns a `Pulse` (a
   completion signal).

7. **`wait(&mut pulse)`**: Blocked until the GPU finished.

8. **`read_field`**: Copied results back to CPU memory.

No shader files. No intermediate representations. No runtime compilation.
The GPU binary is baked into your Rust binary at `cargo build`.

## Next

- [Compute basics](guide/01-compute-basics.md) -- execution model, error handling, optimization
- [Fields and types](guide/02-fields-and-types.md) -- GPU memory management
