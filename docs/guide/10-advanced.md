# Advanced

Topics for production GPU applications: synchronization, profiling, multi-queue,
and feature queries.

## Memory barriers and resource transitions

GPUs execute work out of order. When one operation writes a resource and another
reads it, you must insert a barrier to guarantee ordering.

### Full barrier

```rust
gpu.barrier()?;
```

Waits for all prior GPU work to complete. Heavyweight. Use sparingly.

### Resource transitions

Fine-grained barriers that tell the driver how a resource's usage changes:

```rust
use quanta::ResourceState;

// After compute writes to a field, before rendering reads it as vertex data
gpu.barrier_buffer(&field, ResourceState::ComputeWrite, ResourceState::ShaderRead)?;

// After rendering to a texture, before sampling it in a fragment shader
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
| `ShaderRead`       | Read by any shader stage (sampled/storage) |
| `TransferSrc`      | Source of a copy operation                 |
| `TransferDst`      | Destination of a copy operation            |
| `Present`          | Ready for display (swapchain)              |

On Metal, resource transitions are no-ops (automatic hazard tracking). On Vulkan,
they map to pipeline barriers with correct stage/access masks.

## Multiple render targets (MRT)

Render to several textures simultaneously (deferred rendering):

```rust
use quanta::*;

let albedo = gpu.render_target(1920, 1080, Format::RGBA8)?;
let normal = gpu.render_target(1920, 1080, Format::RGBA16Float)?;
let depth  = gpu.render_target(1920, 1080, Format::R32Float)?;

let pipeline = gpu.pipeline(&PipelineDesc {
    color_formats: vec![Format::RGBA8, Format::RGBA16Float, Format::R32Float],
    ..PipelineDesc::default()
})?;

let mut pass = gpu.render_begin(&albedo)?;
pass.set_color_targets(vec![
    ColorTarget { texture: albedo.handle(), load_op: LoadOp::Clear(Color::BLACK), store_op: StoreOp::Store },
    ColorTarget { texture: normal.handle(), load_op: LoadOp::Clear(Color::BLACK), store_op: StoreOp::Store },
    ColorTarget { texture: depth.handle(), load_op: LoadOp::Clear(Color::BLACK), store_op: StoreOp::Store },
]);
pass.set_pipeline(&pipeline);
pass.bind_vertices(0, &geometry);
pass.draw(vertex_count);
let mut pulse = gpu.render_end(pass)?;
gpu.wait(&mut pulse)?;
```

The fragment shader outputs to multiple targets by returning a struct or using
MRT-specific output syntax.

## Mapped buffers (zero-copy)

For data that changes every frame, mapped buffers eliminate the staging copy:

```rust
let mut uniforms = gpu.field_mapped::<[f32; 16]>(1)?;

// Each frame: write directly to GPU-visible memory
uniforms.as_mut_slice()[0] = compute_mvp_matrix();

// No gpu.write_field() needed -- dispatch reads the updated data directly
```

On unified memory (Apple Silicon), the write is immediate. On discrete GPUs,
the driver syncs on the next command buffer submission.

Use mapped buffers for:
- Per-frame uniform data (camera matrix, time)
- Streaming vertex data (particle systems)
- CPU-readback of small results

## Timestamp queries (profiling)

Measure GPU execution time:

```rust
let query = gpu.timestamp_query(4)?;

// Write timestamps around GPU work
gpu.write_timestamp(&query, 0)?;
// ... dispatch or render ...
gpu.write_timestamp(&query, 1)?;
// ... more work ...
gpu.write_timestamp(&query, 2)?;

// After waiting for completion:
let timestamps = gpu.read_timestamps(&query)?;
let elapsed_ns = gpu.timestamp_to_ns(timestamps[1] - timestamps[0]);
println!("First pass: {} us", elapsed_ns / 1000);
```

`timestamp_to_ns()` converts raw GPU clock ticks to nanoseconds using the
device's timestamp frequency.

## Multi-queue (async compute)

Modern GPUs have multiple hardware queues. Overlap compute and render work:

```rust
if gpu.supports_async_compute() {
    // Dispatch compute on the async queue while rendering continues
    let mut compute_pulse = gpu.async_compute_dispatch(&wave, [64, 1, 1])?;

    // ... render pass on the main queue simultaneously ...

    gpu.wait(&mut compute_pulse)?;
}
```

For full control, create explicit queues:

```rust
let families = gpu.queue_families();
for fam in &families {
    println!("{:?}: {} queues", fam.queue_type, fam.count);
}

let compute_queue = gpu.create_queue(QueueType::Compute)?;
gpu.queue_dispatch(compute_queue, &wave, [256, 1, 1])?;
```

## Feature queries

### Device capabilities

```rust
let caps = gpu.caps();
println!("Vendor: {:?}", caps.vendor);
println!("Name: {}", caps.name);
println!("Nuclei: {}", caps.nuclei);
println!("Memory: {} MB", caps.memory_bytes / 1_000_000);
println!("Max quarks per dispatch: {}", caps.max_quarks_per_dispatch);
println!("Max groups: {:?}", caps.max_groups);
```

### Format support

Query what a format can do on this device:

```rust
let caps = gpu.format_caps(Format::RGBA16Float);
println!("Filterable: {}", caps.filterable);   // can use linear filtering?
println!("Renderable: {}", caps.renderable);   // can render to it?
println!("Storage: {}", caps.storage);         // can read-write from compute?
println!("Blendable: {}", caps.blendable);     // supports blend operations?
println!("MSAA: {}", caps.msaa);               // supports multisampling?
println!("Depth: {}", caps.depth);             // usable as depth buffer?
```

Use this to select texture formats at runtime based on device support.

## Debug labels

Mark GPU work for profilers (Xcode GPU Capture, RenderDoc, NSight):

```rust
gpu.debug_push("Shadow pass");
// ... render shadow maps ...
gpu.debug_pop();

gpu.debug_push("Lighting");
// ... lighting compute ...
gpu.debug_pop();
```

Inside render passes:

```rust
let mut pass = gpu.render_begin(&target)?;
pass.debug_push("Skybox");
pass.draw(36);
pass.debug_pop();
pass.debug_push("Geometry");
pass.draw(vertex_count);
pass.debug_pop();
```

## Validation layer

Enable runtime validation by setting the environment variable:

```sh
QUANTA_VALIDATE=1 cargo run --example hello_quanta
```

The validation layer checks:
- Field bindings match kernel expectations
- Resource states are correct before use
- No use-after-free of GPU resources

Useful during development. Disable in production (adds overhead per call).

## Hot reload

Replace a wave's kernel binary without recreating bindings:

```rust
// Recompile the kernel (e.g., from a file watcher)
let new_binary: &[u8] = load_updated_kernel();

gpu.reload_wave(&mut wave, new_binary)?;
// Bindings and push constants are preserved
```

This enables live shader editing during development.

## Indirect dispatch (GPU-driven)

Let the GPU decide how much work to launch:

```rust
// A field containing [group_x: u32, group_y: u32, group_z: u32]
let indirect_args = gpu.compute_field::<u32>(3)?;

// First pass: a kernel writes the dispatch dimensions
let mut pulse = gpu.dispatch(&setup_wave, 1)?;
gpu.wait(&mut pulse)?;

// Second pass: dispatch with GPU-computed group counts
let mut pulse = gpu.dispatch_indirect(&work_wave, &indirect_args, 0)?;
gpu.wait(&mut pulse)?;
```

## Occlusion queries

Test if geometry is visible (for culling):

```rust
let query = gpu.occlusion_query_create(16)?; // 16 query slots

let mut pass = gpu.render_begin(&target)?;
pass.set_pipeline(&pipeline);

pass.begin_occlusion_query(&query, 0);
pass.bind_vertices(0, &object_bounds);
pass.draw(36); // draw bounding box
pass.end_occlusion_query(&query, 0);

let mut pulse = gpu.render_end(pass)?;
gpu.wait(&mut pulse)?;

let results = gpu.occlusion_query_read(&query)?;
if results[0] > 0 {
    // Object is visible -- draw the full mesh
}
```

## Timeline semaphores

Multi-frame pipelining without per-frame fences:

```rust
let timeline = gpu.timeline_create()?;

// Frame N: signal value N after rendering
gpu.timeline_signal(&timeline, frame_number)?;

// Frame N+2: wait for frame N to finish before reusing its resources
gpu.timeline_wait(&timeline, frame_number - 2)?;
```

Timelines increase monotonically. The GPU signals them after work completes;
the CPU (or another queue) waits until the timeline reaches a threshold.

---

## JIT compilation

For kernels that need to adapt at runtime (user-defined expressions, dynamic
filter pipelines), use the `jit` attribute:

```rust
#[quanta::kernel(jit)]
fn dynamic_filter(data: &mut [f32], threshold: f32) {
    let i = quark_id();
    if data[i] < threshold {
        data[i] = 0.0;
    }
}
```

At build time, `#[quanta::kernel(jit)]` serializes the kernel's `KernelDef`
(IR) into the binary instead of compiling it to GPU machine code. At runtime,
call `gpu.wave_jit()` to compile and dispatch:

```rust
let wave = gpu.wave_jit(&DYNAMIC_FILTER_KERNEL_DEF)?;
wave.bind(0, &data);
wave.set_value(1, 0.5f32);
let mut pulse = gpu.dispatch(&wave, n)?;
gpu.wait(&mut pulse)?;
```

JIT compilation is feature-gated behind `jit`:

```toml
[dependencies]
quanta = { version = "0.1", features = ["jit"] }
```

Use JIT when:
- Kernel logic depends on runtime configuration
- You need to generate kernels programmatically from IR
- Hot-reload during development needs source-level changes

For all other cases, prefer the default build-time compilation. It produces
identical binaries with zero runtime compilation overhead.

---

## CPU software executor

For testing and development without a GPU, Quanta provides a full CPU
software executor that interprets the kernel IR:

```rust
// Initialize the CPU executor instead of a GPU
let cpu = quanta::init_cpu();
```

Or set the environment variable:

```sh
QUANTA_CPU=1 cargo run --example my_compute
```

The CPU executor:
- Interprets all KernelOp instructions
- Emulates workgroup barriers correctly (all quarks in a workgroup reach
  the barrier before any proceed past it)
- Supports shared memory, atomics, and wave operations
- Runs quarks sequentially within each workgroup, workgroups sequentially

This is feature-gated behind `software`:

```toml
[dependencies]
quanta = { version = "0.1", features = ["software"] }
```

The CPU executor is not designed for performance -- it is a correctness
reference. Use it for:
- CI/CD pipelines without GPU hardware
- Debugging kernel logic with standard Rust tools
- Validating results against a known-correct sequential execution

---

## Subgroup (wave) reduce and scan

Subgroup operations perform reductions and scans across lanes within a proton
(warp/SIMD group) without shared memory or barriers.

### Reduce

Combine a value across all lanes in the subgroup:

```rust
#[quanta::kernel]
fn subgroup_sum(data: &[f32], result: &mut [f32]) {
    let i = quark_id();
    let val = data[i];

    // Sum across all lanes in the proton
    let sum = wave_reduce_add(val);

    // Min/max variants
    let min_val = wave_reduce_min(val);
    let max_val = wave_reduce_max(val);

    if local_id() == 0 {
        result[group_id()] = sum;
    }
}
```

### Scan (prefix sum)

Compute running totals across lanes:

```rust
#[quanta::kernel]
fn compact(data: &[f32], output: &mut [f32], count: &mut [u32]) {
    let i = quark_id();
    let val = data[i];
    let keep = if val > 0.0 { 1u32 } else { 0u32 };

    // Exclusive prefix sum: each lane gets the sum of all preceding lanes
    let offset = wave_exclusive_add(keep);

    // Inclusive prefix sum: each lane gets the sum including itself
    let inclusive = wave_inclusive_add(keep);

    if keep == 1 {
        output[offset] = val;
    }
}
```

Available subgroup operations:

| Function | Description |
|----------|-------------|
| `wave_reduce_add(val)` | Sum across all lanes |
| `wave_reduce_min(val)` | Minimum across all lanes |
| `wave_reduce_max(val)` | Maximum across all lanes |
| `wave_exclusive_add(val)` | Exclusive prefix sum (excludes current lane) |
| `wave_inclusive_add(val)` | Inclusive prefix sum (includes current lane) |

These compile to native subgroup instructions on all backends (SPIR-V
`OpGroupNonUniform*`, Metal SIMD-group functions, PTX `redux`/`shfl`).

---

## Exclusive scan library

For larger-than-subgroup prefix sums, Quanta provides a host-side scan utility:

```rust
use quanta::scan::exclusive_scan_f32_bytes;

let input: &[u8] = /* serialized f32 array */;
let output = exclusive_scan_f32_bytes(input);
// output[i] = sum of input[0..i]
```

`exclusive_scan_f32_bytes` operates on raw byte slices (as returned by
`gpu.read_field`) and produces an exclusive prefix sum of the `f32` values.
Use this for stream compaction, histogram equalization, and radix sort
building blocks.
