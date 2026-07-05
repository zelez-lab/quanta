# Expert: Manual API

The manual API gives you full control over GPU memory allocation, binding,
dispatch, and synchronization. Use this when you need:

- Custom memory layouts or usage flags
- Double-buffering or ping-pong patterns
- Explicit field lifetimes across frames
- Raw handles for interop with external libraries
- Per-field read/write timing control

Most users should start with `#[derive(quanta::Fields)]` (see
[Getting Started](../getting-started.md)). Come here when you outgrow it.

## Manual field allocation

Allocate GPU memory with explicit usage flags:

```rust
use quanta::*;

let gpu = init()?;

// Full control: specify element count and usage flags
let field = gpu.field_with_usage::<f32>(
    1024,
    FieldUsage::READ.union(FieldUsage::WRITE).union(FieldUsage::COMPUTE),
)?;

// Or use the preset usage profiles
let compute = gpu.field::<f32>(1024)?; // default_compute(): READ | WRITE | COMPUTE | TRANSFER
let render  = gpu.field_with_usage::<f32>(1024, FieldUsage::default_render())?;  // READ | RENDER | TRANSFER
let uniform = gpu.field_with_usage::<f32>(1, FieldUsage::default_uniform())?;    // READ | UNIFORM | TRANSFER
```

### FieldUsage flags

| Flag                | Meaning                               |
|---------------------|---------------------------------------|
| `FieldUsage::READ`     | GPU will read from this field     |
| `FieldUsage::WRITE`    | GPU will write to this field      |
| `FieldUsage::COMPUTE`  | Used in compute dispatches        |
| `FieldUsage::RENDER`   | Used as vertex/index data         |
| `FieldUsage::TRANSFER` | Transferred to/from CPU           |
| `FieldUsage::UNIFORM`  | Used as a uniform buffer          |

Usage flags tell the driver how the field will be accessed, enabling
placement optimizations on the hardware.

## Manual wave binding

Create a wave (compiled kernel) and bind fields by slot index:

```rust
#[quanta::kernel]
fn vector_add(a: &[f32], b: &[f32], result: &mut [f32]) {
    let i = quark_id();
    result[i] = a[i] + b[i];
}

fn main() -> Result<(), QuantaError> {
    let gpu = init()?;

    let a = gpu.field::<f32>(1024)?;
    let b = gpu.field::<f32>(1024)?;
    let result = gpu.field::<f32>(1024)?;

    a.write(&vec![1.0f32; 1024])?;
    b.write(&vec![2.0f32; 1024])?;

    // Create the wave (selects the right binary for this GPU)
    let mut wave = vector_add(&gpu)?;

    // Bind fields to kernel parameter slots
    wave.bind(0, &a);       // slot 0 = parameter `a`
    wave.bind(1, &b);       // slot 1 = parameter `b`
    wave.bind(2, &result);  // slot 2 = parameter `result`

    // Dispatch and wait
    let mut pulse = gpu.dispatch(&wave, 1024)?;
    pulse.wait()?;

    let output = result.read()?;
    assert_eq!(output[0], 3.0);
    Ok(())
}
```

### Push constants (manual)

For scalar values that are not array-backed:

```rust
wave.bind(0, &data_field);
wave.set_value(1, 0.5f32);  // push constant at slot 1
```

## Render passes with conditional logic

Render passes are recorded through the chainable `RenderBuilder`
(`gpu.render(&target)?` — a `RenderGpu` extension-trait method, in scope
via `use quanta::*;`). Every builder method consumes and returns `self`,
so a pass does not have to be one expression: hold the builder in a
variable and reassign it for conditional logic between draw calls or
dynamic pipeline switching.

```rust
let target = gpu.render_target(800, 600, Format::BGRA8)?;

let mut pass = gpu.render(&target)?
    .clear(Color::BLACK)
    .pipeline(&pipeline)
    .vertices(0, &vertex_buffer)
    .indices(&index_buffer)
    .uniform(0, &mvp_buffer)
    .texture(0, &albedo_texture)
    .sampler(0, SamplerDesc::default())
    .viewport(0.0, 0.0, 800.0, 600.0)
    .scissor(100, 100, 600, 400);

if draw_indexed_geometry {
    pass = pass.draw_indexed(36);
} else {
    pass = pass.pipeline(&fallback_pipeline).draw(3);
}

let mut pulse = pass.pulse()?;
pulse.wait()?;
```

### Manual VertexLayout construction

When `#[derive(quanta::Vertex)]` does not fit (interleaved multi-struct
layouts, non-standard attribute packing):

```rust
let layout = VertexLayout {
    stride: 24, // 6 floats * 4 bytes
    step: StepMode::Vertex,
    attributes: vec![
        VertexAttribute { location: 0, offset: 0, format: AttributeFormat::Float3 },
        VertexAttribute { location: 1, offset: 12, format: AttributeFormat::Float3 },
    ],
};
```

## Double-buffering

Render to one target while the GPU reads from another, then swap:

```rust
let target_a = gpu.render_target(1920, 1080, Format::RGBA8)?;
let target_b = gpu.render_target(1920, 1080, Format::RGBA8)?;
let mut front = &target_a;
let mut back = &target_b;

loop {
    // Render to back buffer
    gpu.render(back)?
        .clear(Color::BLACK)
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw(vertex_count)
        .pulse()?
        .wait()?;

    // Swap
    core::mem::swap(&mut front, &mut back);

    // Present `front`: export it to a compositor with
    // `front.native_handle()`, or — when rendering to a window —
    // use a `Surface` (`gpu.create_surface`) and render into
    // acquired frames instead of your own targets.
}
```

## Ping-pong compute

Process data back and forth between two fields:

```rust
let mut src = gpu.field::<f32>(n)?;
let mut dst = gpu.field::<f32>(n)?;
src.write(&initial_data)?;

for _step in 0..iterations {
    let mut wave = blur_kernel(&gpu)?;
    wave.bind(0, &src);
    wave.bind(1, &dst);
    gpu.dispatch(&wave, n as u32)?.wait()?;

    // Swap roles
    core::mem::swap(&mut src, &mut dst);
}

let result = src.read()?; // final result is in src after even number of swaps
```

## Raw handles for interop

Access the underlying driver handles for interop with Metal, Vulkan, or
other GPU libraries:

```rust
// Field handle (Metal: MTLBuffer pointer, Vulkan: VkBuffer)
let raw_handle: u64 = field.handle();

// Texture handle
let raw_tex: u64 = texture.handle();

// Pipeline handle
let raw_pipe: u64 = pipeline.handle();
```

These handles are driver-specific. Use them only when integrating with
code that operates at the Metal/Vulkan level directly.

For texture interop, prefer `texture.native_handle()` — it returns a typed
`NativeTextureHandle` (the actual `id<MTLTexture>` / `VkImage` plus import
metadata) instead of an opaque registry id, with a documented
borrow/lifetime contract. Query `gpu.supports_native_handle_export()`
first; the CPU software driver has no native object to export.

## Custom memory layouts

Control exactly how GPU memory is organized:

```rust
// Read-only field (driver can place in faster read-optimized memory)
let weights = gpu.field_with_usage::<f32>(n, FieldUsage::READ.union(FieldUsage::COMPUTE))?;

// Write-only output (driver can skip read caches)
let output = gpu.field_with_usage::<f32>(
    n,
    FieldUsage::WRITE.union(FieldUsage::COMPUTE).union(FieldUsage::TRANSFER),
)?;

// Staging buffer (CPU-accessible, used for transfer only)
let staging = gpu.field_with_usage::<f32>(n, FieldUsage::TRANSFER)?;
```

## Mapped buffers (zero-copy)

For data that changes every frame, mapped buffers eliminate the staging copy:

```rust
let mut uniforms = gpu.field_mapped::<[f32; 16]>(1)?;

// Each frame: write directly to GPU-visible memory
uniforms.as_mut_slice()[0] = compute_mvp_matrix();

// No field.write() needed -- the GPU reads the updated data directly
```

On unified memory (Apple Silicon), the write is immediate. On discrete GPUs,
the driver syncs on the next command buffer submission.

## Memory barriers and resource transitions

GPUs execute work out of order. When one operation writes a resource and
another reads it, you must insert a barrier:

```rust
// Full barrier (heavyweight, use sparingly)
gpu.barrier()?;

// Fine-grained: after compute writes, before render reads
gpu.barrier_field(&field, ResourceState::ComputeWrite, ResourceState::ShaderRead)?;

// After rendering to texture, before sampling it
gpu.barrier_texture(&texture, ResourceState::RenderTarget, ResourceState::ShaderRead)?;
```

Available resource states:

| State              | Meaning                                    |
|--------------------|--------------------------------------------|
| `General`          | Any usage (suboptimal but valid)           |
| `ComputeWrite`     | Being written by a compute shader          |
| `ComputeRead`      | Being read by a compute shader             |
| `RenderTarget`     | Being drawn to (color attachment)          |
| `DepthStencil`     | Used as depth/stencil attachment           |
| `ShaderRead`       | Read by any shader stage                   |
| `TransferSrc`      | Source of a copy operation                 |
| `TransferDst`      | Destination of a copy operation            |
| `Present`          | Ready for display (swapchain)              |

On Metal, resource transitions are no-ops (automatic hazard tracking). On
Vulkan, they map to pipeline barriers with correct stage/access masks.

## Indirect dispatch (GPU-driven)

Let the GPU decide how much work to launch:

```rust
// A field containing [group_x: u32, group_y: u32, group_z: u32]
let indirect_args = gpu.field::<u32>(3)?;

// First pass: a kernel writes the dispatch dimensions
gpu.dispatch(&setup_wave, 1)?.wait()?;

// Second pass: dispatch with GPU-computed group counts
let mut pulse = gpu.dispatch_indirect(&work_wave, &indirect_args, 0)?;
pulse.wait()?;
```

## Timestamp queries (profiling)

Measure GPU execution time:

```rust
let query = gpu.timestamp_query(4)?;

gpu.write_timestamp(&query, 0)?;
// ... dispatch or render ...
gpu.write_timestamp(&query, 1)?;

// After waiting for completion:
let timestamps = gpu.read_timestamps(&query)?;
let elapsed_ns = gpu.timestamp_to_ns(timestamps[1] - timestamps[0]);
println!("Kernel time: {} us", elapsed_ns / 1000);
```

## Debug labels

Mark GPU work for profilers (Xcode GPU Capture, RenderDoc, NSight):

```rust
gpu.debug_push("Shadow pass");
// ... render shadow maps ...
gpu.debug_pop();
```

Via the render builder:

```rust
gpu.render(&target)?
    .debug_push("Skybox")
    .draw(36)
    .debug_pop()
    .debug_push("Geometry")
    .draw(vertex_count)
    .debug_pop()
    .pulse()?
    .wait()?;
```

## Validation layer

Enable runtime validation during development:

```sh
QUANTA_VALIDATE=1 cargo run --example my_app
```

Checks field bindings, resource states, and use-after-free. Disable in
production (adds overhead per call).

## Hot reload

Replace a wave's kernel binary without recreating bindings:

```rust
let new_binary: &[u8] = load_updated_kernel();
gpu.reload_wave(&mut wave, new_binary)?;
// Bindings and push constants are preserved
```

## JIT compilation

For kernels that adapt at runtime:

```rust
#[quanta::kernel(jit)]
fn dynamic_filter(data: &mut [f32], threshold: f32) {
    let i = quark_id();
    if data[i] < threshold { data[i] = 0.0; }
}

// At runtime:
let wave = gpu.wave_jit(&DYNAMIC_FILTER_KERNEL_DEF)?;
wave.bind(0, &data);
wave.set_value(1, 0.5f32);
gpu.dispatch(&wave, n)?.wait()?;
```

Feature-gated behind `jit`:

```toml
[dependencies]
quanta = { version = "0.1", features = ["jit"] }
```

## CPU software executor

For testing without GPU hardware:

```rust
let cpu = quanta::init_cpu();
```

Or via environment variable:

```sh
QUANTA_CPU=1 cargo run --example my_compute
```

Feature-gated behind `software`. Interprets kernel IR sequentially --
correctness reference, not performance target.

## Feature queries

### Device capabilities

```rust
let caps = gpu.caps();
println!("Vendor: {:?}", caps.vendor);
println!("Name: {}", caps.name);
println!("Nuclei: {}", caps.nuclei);
println!("Memory: {} MB", caps.memory_bytes / 1_000_000);
println!("Max quarks per dispatch: {}", caps.max_quarks_per_dispatch);
```

### Format support

```rust
let caps = gpu.format_caps(Format::RGBA16Float);
println!("Filterable: {}", caps.filterable);
println!("Renderable: {}", caps.renderable);
println!("Storage: {}", caps.storage);
println!("Blendable: {}", caps.blendable);
```
