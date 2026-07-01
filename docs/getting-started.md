# Getting Started

GPU compute in 3 minutes. You know Rust. You may not know GPUs.

## Add dependency

Quanta is not on crates.io yet, so add it from the git repository:

```sh
cargo add quanta --git https://github.com/zelez-lab/quanta
```

That produces the following line in your `Cargo.toml`:

```toml
[dependencies]
quanta = { git = "https://github.com/zelez-lab/quanta" }
```

Pin to a specific revision with `--rev <sha>` (or `--tag <tag>` once
tagged releases exist) for reproducible builds.

### Compute only, or compute + rendering?

Quanta has two faces: **GPU compute** (the CUDA-like side — kernels,
fields, dispatch) and **rendering** (render passes, graphics pipelines,
mesh/tessellation/ray-tracing). They live behind a Cargo feature so a
headless consumer never compiles a line of rendering code.

- **Compute only** (a database, an ML runtime, any GPGPU app) — depend on
  `quanta` with the default `render` feature turned off:

  ```toml
  [dependencies]
  quanta = { git = "...", default-features = false, features = ["metal"] }
  # or "vulkan" / "software" instead of "metal"
  ```

  Your build has **no rendering types on its surface** and skips compiling
  the render code entirely (~37% less of the `quanta` crate to compile).

- **Compute + rendering** (a graphical app, a UI) — keep the default
  feature on, and add the `quanta-render` crate for the render surface:

  ```toml
  [dependencies]
  quanta = { git = "..." }                       # render on by default
  quanta-render = { git = "...", features = ["metal"] }
  ```

  `quanta-render` brings render types (`RenderPass`, `Pipeline`, …), the
  shader-stage macros (`#[quanta::vertex]`, `#[quanta::fragment]`, …), and
  the render methods on `Gpu` (`gpu.render_target(...)`, `gpu.render(...)`).

The rest of this guide is pure compute and works either way.

## Write a kernel

A kernel is a function that runs on the GPU. Thousands of copies run in parallel,
each on a different element.

```rust
use quanta::*;

#[derive(quanta::Fields)]
struct VecAdd {
    a: Vec<f32>,
    b: Vec<f32>,
    result: Vec<f32>,
}

#[quanta::kernel]
fn vector_add(data: &VecAdd) {
    let i = quark_id();
    data.result[i] = data.a[i] + data.b[i];
}
```

`quark_id()` returns this quark's index. If you dispatch 1024 quarks,
`quark_id()` ranges from 0 to 1023.

`#[derive(quanta::Fields)]` tells the framework how to map your struct
to the GPU. `Vec<T>` fields become GPU storage buffers. Scalar fields
become push constants. No manual slot numbers, no manual binding.

`#[quanta::kernel]` compiles the function to all 5 GPU targets at build
time: metallib (Apple), SPIR-V (Vulkan), PTX (NVIDIA), GCN (AMD), and
WGSL (WebGPU). All are embedded in your binary. At runtime, the right
one runs on whatever GPU is present.

## Run it

```rust
use quanta::*;

#[derive(quanta::Fields)]
struct VecAdd {
    a: Vec<f32>,
    b: Vec<f32>,
    result: Vec<f32>,
}

#[quanta::kernel]
fn vector_add(data: &VecAdd) {
    let i = quark_id();
    data.result[i] = data.a[i] + data.b[i];
}

fn main() -> Result<(), QuantaError> {
    let gpu = init()?;

    // Per-lane inputs so the output varies across the 1024 quarks —
    // result[i] = i + 2*i = 3*i.
    let mut data = VecAdd {
        a: (0..1024).map(|i| i as f32).collect(),
        b: (0..1024).map(|i| (i * 2) as f32).collect(),
        result: vec![0.0f32; 1024],
    };

    // One line: upload, bind, dispatch, readback
    vector_add(&gpu, &mut data, 1024)?.wait()?;

    assert_eq!(data.result[0], 0.0);
    assert_eq!(data.result[1023], 3069.0);
    println!("GPU computed: {} elements", data.result.len());
    println!("first 5: {:?}", &data.result[..5]);
    println!("last 5:  {:?}", &data.result[data.result.len() - 5..]);
    Ok(())
}
```

The dispatch call is one line:

```rust
vector_add(&gpu, &mut data, 1024)?.wait()?;
```

That is the entire GPU interaction. Define your data, call the kernel, read
the results back from the same struct.

> **Note:** `main` returns `Result<(), QuantaError>` so the `?` operator can
> propagate errors out of the program. If you copy only the body of `main`,
> keep the signature too — `?` does not work inside a `fn main()` that
> returns `()`.

## What happened

1. **Build time**: `#[quanta::kernel]` compiled `vector_add` to native GPU
   binaries for all 5 targets. `#[derive(quanta::Fields)]` generated the
   metadata that maps struct fields to GPU buffer slots and push constant
   slots. Both happen at `cargo build` -- zero runtime compilation.

2. **`init()`**: Discovered the first available GPU and returned a `Gpu` handle.

3. **`vector_add(&gpu, &mut data, 1024)?`**: This single call did everything:
   - Allocated GPU storage buffers for each `Vec<T>` field
   - Uploaded CPU data to the GPU
   - Selected the right pre-compiled binary for your GPU vendor
   - Created a Wave (a kernel ready to dispatch) and bound all fields
   - Dispatched 1024 quarks on the GPU
   - Read results back into `data.result`

4. **`.wait()?`**: Blocked until the GPU finished. Returns a `Pulse`
   (a completion signal) that you wait on.

No shader files. No intermediate representations. No runtime compilation.
No manual slot numbers. No `gpu.write_field`. The GPU binary is baked into
your Rust binary at `cargo build`.

Quanta has **zero runtime dependencies** -- no `metal-rs`, no `ash`, no
`objc` crate. Drivers use raw FFI (`objc_msgSend`, `extern "C"` Vulkan
functions) for minimal binary size and maximum build speed.

## Workgroup sizes

Control how quarks are grouped for optimal GPU utilization:

```rust
#[quanta::kernel(workgroup = [256, 1, 1])]
fn vector_add(data: &VecAdd) {
    let i = quark_id();
    data.result[i] = data.a[i] + data.b[i];
}
```

See [Compute basics](computation/tutorials/compute-basics.md) for details on 1D/2D/3D
workgroup sizes.

## Structured GPU data

For kernels that operate on multi-field data (particles, vertices, game
entities), use `#[quanta::gpu_type]` to define a GPU-compatible struct:

```rust
#[quanta::gpu_type]
struct Particle {
    pos: [f32; 3],
    vel: [f32; 3],
    mass: f32,
}
```

The macro generates `#[repr(C)]`, `Copy`, `Clone`, `GpuType` impl, and
backend struct declarations (MSL/WGSL). Then use it as an element type
in your Fields struct:

```rust
#[derive(quanta::Fields)]
struct Simulation {
    particles: Vec<Particle>,
    dt: f32,
}
```

See [Fields and types](computation/tutorials/fields-and-types.md) for the full reference.

## Manual API

For power users who need fine-grained control over GPU memory, binding,
and dispatch, Quanta exposes the full manual API:

```rust
let a = gpu.field::<f32>(1024)?;
a.write(&data_a)?;
// ... see Expert: Manual API for the full story
```

See [Expert: Manual API](expert/manual-api.md) for manual field allocation,
explicit wave binding, double-buffering, and raw handles.

## Capability queries and graceful fallback

Some features only exist on certain backends or device families: ray tracing,
mesh shaders, variable rate shading, sparse residency. Query before you build:

```rust
let gpu = quanta::init()?;

if gpu.supports_ray_tracing() {
    // build BLAS, dispatch rays
} else if gpu.supports_mesh_shaders() {
    // mesh-shader culling fast path
} else {
    // classic vertex/fragment fallback
}
```

When a feature is not implemented for the active backend, the call returns
`QuantaError { kind: NotSupported(reason), .. }` rather than panicking. Branch on
the error kind to fall back without giving up:

```rust
match gpu.sparse_texture(&desc) {
    Ok(tex) => /* use sparse */,
    Err(e) if matches!(e.kind, QuantaErrorKind::NotSupported(_)) => {
        // fall back to a regular texture
    }
    Err(e) => return Err(e),
}
```

`NotFound` is the partner variant — it means a handle no longer points at a
live resource (typically a double-free or use-after-drop).

See [Compute basics: Error handling](computation/tutorials/compute-basics.md#error-handling)
and [Reference: Errors](reference/errors.md) for the full kind list.

## Next

- [Compute basics](computation/tutorials/compute-basics.md) -- execution model, workgroups, optimization
- [Fields and types](computation/tutorials/fields-and-types.md) -- GPU memory management
- [Multi-queue](rendering/tutorials/multi-queue.md), [Indirect commands](rendering/tutorials/indirect-commands.md),
  [Tessellation](rendering/tutorials/tessellation.md), [Mesh shaders](rendering/tutorials/mesh-shaders.md),
  [Ray tracing](rendering/tutorials/ray-tracing.md), [VRS](rendering/tutorials/variable-rate-shading.md),
  [Async copy + printf](computation/tutorials/async-copy-and-printf.md) -- v0.1 advanced surface
