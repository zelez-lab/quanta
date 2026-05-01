# API Reference

All public types in the Quanta GPU framework.

## `Gpu`

The main entry point. All GPU operations go through this type.

```rust
let gpu = quanta::init()?;
```

### Device info

| Method | Returns | Description |
|--------|---------|-------------|
| `caps()` | `&Caps` | Full device capabilities |
| `nuclei()` | `u32` | Number of compute units (SM / CU) |
| `protons_per_nucleus()` | `u32` | Cores per compute unit |
| `quarks_per_proton()` | `u32` | Threads per core (warp / wavefront width) |
| `total_quarks()` | `u32` | Total parallel execution units |
| `name()` | `&str` | Device name string |

### Feature support queries

Cheap, side-effect-free booleans reading the per-driver capability cache
populated at device discovery. Always check before constructing the
matching typed wrapper — the constructors return `NotSupported` on a
no-capability device, but checking up front lets you take a fallback
path without throwing.

| Method | Returns | Description |
|--------|---------|-------------|
| `supports_vrs()` | `bool` | Variable rate shading available |
| `supports_ray_tracing()` | `bool` | Hardware ray-tracing extensions present |
| `supports_mesh_shaders()` | `bool` | Mesh + task shader stages available |
| `supports_tessellation()` | `bool` | Tessellation control / evaluation stages |
| `supports_sparse_residency()` | `bool` | Sparse textures (`vkQueueBindSparse` / `MTLHeap` placement) |
| `supported_shading_rates()` | `Vec<(u32, u32)>` | Concrete (x,y) shading rates the device exposes (e.g. `[(1,1), (2,2), (4,4)]`). Empty when VRS is not supported. |

### Fields (typed GPU memory)

| Method | Returns | Description |
|--------|---------|-------------|
| `field::<T>(count, usage)` | `Result<Field<T>>` | Allocate with explicit usage flags |
| `compute_field::<T>(count)` | `Result<Field<T>>` | Storage + transfer (compute workloads) |
| `render_field::<T>(count)` | `Result<Field<T>>` | Vertex + transfer (render workloads) |
| `uniform_field::<T>(count)` | `Result<Field<T>>` | Uniform + transfer (constant data) |
| `field_mapped::<T>(count)` | `Result<MappedField<T>>` | CPU-mapped buffer (zero-copy) |

### Textures

| Method | Returns | Description |
|--------|---------|-------------|
| `texture(width, height)` | `Result<Texture>` | Simple RGBA8 texture |
| `create_texture(desc)` | `Result<Texture>` | Full-control creation |
| `render_target(w, h, fmt)` | `Result<Texture>` | Can be drawn to + sampled |
| `msaa_target(w, h, fmt, samples)` | `Result<Texture>` | Multi-sampled render target |
| `sampler(desc)` | `Result<Sampler>` | Create reusable sampler |
| `resolve_texture(msaa, dst)` | `Result<()>` | Resolve MSAA to single-sample |
| `texture_view_create(tex, desc)` | `Result<TextureView>` | Sub-range view |
| `format_caps(format)` | `FormatCaps` | Query format capabilities |

### Compute

| Method | Returns | Description |
|--------|---------|-------------|
| `wave(kernel_bytes)` | `Result<Wave>` | Create wave from compiled kernel |
| `wave_jit(kernel_def)` | `Result<Wave>` | JIT-compile KernelDef and create wave |
| `dispatch(wave, quarks)` | `Result<Pulse>` | Dispatch 1D (exact thread count) |
| `wave_dispatch(wave, [x,y,z])` | `Result<Pulse>` | Dispatch with group counts |
| `dispatch_indirect(wave, buf, off)` | `Result<Pulse>` | GPU-driven dispatch |
| `reload_wave(wave, kernel)` | `Result<()>` | Hot-reload kernel binary |
| `batch()` | `Result<Batch>` | Begin multi-dispatch batch |
| `indirect_command_buffer(cap)` | `Result<IndirectCommandBuffer>` | Pre-record `cap` dispatch / draw commands then `execute(n)` |
| `async_copy_queue()` | `Result<AsyncCopyQueue>` | Transfer queue concurrent with compute / graphics |
| `printf_buffer(cap)` | `Result<PrintfBuffer>` | Capacity-bounded shader printf ring |
| `queue(QueueType)` | `Result<Queue>` | Typed queue wrapper (graphics / compute / transfer) |
| `create_queue(QueueType)` | `Result<u64>` | Raw queue handle (escape hatch — prefer `queue`) |

### Render

| Method | Returns | Description |
|--------|---------|-------------|
| `pipeline(desc)` | `Result<Pipeline>` | Create render pipeline |
| `render(target)` | `Result<RenderBuilder>` | Begin render pass (builder chain) |
| `dispatch_mesh(pipeline, groups)` | `Result<()>` | Mesh shader dispatch |
| `mesh_pipeline(desc)` | `Result<Pipeline>` | Create a mesh-shader pipeline (gated on `supports_mesh_shaders`) |
| `tessellation_pipeline(desc)` | `Result<Pipeline>` | Create a tessellation pipeline (gated on `supports_tessellation`) |
| `sparse_texture(desc)` | `Result<SparseTexture>` | Virtual texture with on-demand tile residency (2D, single-mip in v0.1) |
| `acceleration_structure_blas(geoms)` | `Result<AccelerationStructure>` | Build a bottom-level BVH (foundation only in v0.1 — build dispatch returns `NotSupported`) |
| `ray_tracing_pipeline(desc)` | `Result<RayTracingPipeline>` | Construct a ray-tracing pipeline; `dispatch_rays(w, h)` traces |
| `vrs_state()` | `Result<VrsState>` | Variable rate shading handle — `set_rate(ShadingRate)` to switch |

### Sync

| Method | Returns | Description |
|--------|---------|-------------|
| `barrier()` | `Result<()>` | Full pipeline barrier |
| `barrier_field(field, from, to)` | `Result<()>` | Field state transition |
| `barrier_texture(tex, from, to)` | `Result<()>` | Texture state transition |

### Timeline semaphores

| Method | Returns | Description |
|--------|---------|-------------|
| `timeline_create()` | `Result<Timeline>` | Create timeline semaphore |
| `timeline_signal(tl, value)` | `Result<()>` | Signal value |
| `timeline_wait(tl, value)` | `Result<()>` | Block until value reached |

### Queries

| Method | Returns | Description |
|--------|---------|-------------|
| `timestamp_query(count)` | `Result<TimestampQuery>` | Create timestamp query set |
| `write_timestamp(query, idx)` | `Result<()>` | Record timestamp |
| `read_timestamps(query)` | `Result<Vec<u64>>` | Read all timestamps |
| `timestamp_to_ns(ticks)` | `u64` | Convert ticks to nanoseconds |
| `occlusion_query_create(count)` | `Result<OcclusionQuery>` | Create occlusion query |
| `occlusion_query_read(query)` | `Result<Vec<u64>>` | Read fragment counts (synchronous, native backends only) |

> **WebGPU note.** WebGPU has no synchronous readback of query results.
> On the WebGPU backend, `occlusion_query_read` returns
> `NotSupported`; use `occlusion_query_read_async(query).await` on
> the WebGPU driver directly for the Promise-based path.

### Multi-queue

| Method | Returns | Description |
|--------|---------|-------------|
| `queue_families()` | `Vec<QueueFamily>` | Available queue families |
| `create_queue(type)` | `Result<u64>` | Create queue |
| `queue_dispatch(q, wave, groups)` | `Result<()>` | Submit to specific queue |
| `queue_signal(q, sem)` | `Result<()>` | Signal from queue |
| `queue_wait(q, sem)` | `Result<()>` | Wait on queue |

### Debug

| Method | Returns | Description |
|--------|---------|-------------|
| `debug_push(label)` | `()` | Push debug group |
| `debug_pop()` | `()` | Pop debug group |

### Deprecated methods

These methods still work but have preferred alternatives:

| Deprecated | Use instead |
|------------|-------------|
| `gpu.write_field(&field, &data)` | `field.write(&data)` |
| `gpu.read_field(&field)` | `field.read()` |
| `gpu.copy_field(&dst, &src)` | `dst.copy_from(&src)` |
| `gpu.resize_field(&old, n, usage)` | Allocate new field + `dst.copy_from(&old)` |
| `gpu.texture_write(&tex, &data)` | `texture.write(&data)` |
| `gpu.texture_read(&tex)` | `texture.read()` |
| `gpu.generate_mipmaps(&tex)` | `texture.generate_mipmaps()` |
| `gpu.wait(&mut pulse)` | `pulse.wait()` |
| `gpu.wait_and_reset(&mut pulse)` | `pulse.wait()` + `pulse.reset()` |
| `gpu.poll(&pulse)` | `pulse.is_done()` |
| `gpu.begin_batch()` | `gpu.batch()` |
| `gpu.render_begin(&target)` | `gpu.render(&target)` (builder) |
| `gpu.render_end(pass)` | `.pulse()` on `RenderBuilder` |
| `gpu.barrier_buffer(field, from, to)` | `gpu.barrier_field(field, from, to)` |
| `gpu.compute_field::<T>(n)` | Still valid (not deprecated) |

---

## `Field<T>`

GPU-resident typed buffer (storage buffer). Created via `gpu.compute_field()`
or `gpu.field()`. Freed automatically when dropped (RAII).

| Method | Returns | Description |
|--------|---------|-------------|
| `write(&data)` | `Result<()>` | Upload data from CPU to GPU |
| `read()` | `Result<Vec<T>>` | Download data from GPU to CPU |
| `copy_from(&src)` | `Result<()>` | GPU-to-GPU copy from another field |
| `len()` | `usize` | Element count |
| `is_empty()` | `bool` | True if count is 0 |
| `byte_size()` | `usize` | Size in bytes |
| `handle()` | `u64` | Raw GPU handle (for driver use) |

---

## `MappedField<T>`

CPU-mapped GPU buffer for zero-copy writes. Created via `gpu.field_mapped()`.

| Method | Returns | Description |
|--------|---------|-------------|
| `write(index, value)` | `()` | Write single element at index |
| `read(index)` | `T` | Read single element at index |
| `as_slice()` | `&[T]` | Immutable slice view of mapped memory |
| `as_mut_slice()` | `&mut [T]` | Mutable slice view of mapped memory |
| `len()` | `usize` | Element count |
| `byte_size()` | `usize` | Size in bytes |
| `handle()` | `u64` | Raw GPU handle |

---

## `Texture`

GPU-resident 2D image. Created via `gpu.texture()` or `gpu.create_texture()`.

| Method | Returns | Description |
|--------|---------|-------------|
| `write(&data)` | `Result<()>` | Upload pixel data |
| `read()` | `Result<Vec<u8>>` | Download pixel data |
| `generate_mipmaps()` | `Result<()>` | Auto-generate mip chain |
| `width()` | `u32` | Width in pixels |
| `height()` | `u32` | Height in pixels |
| `format()` | `Format` | Pixel format |
| `handle()` | `u64` | Raw GPU handle |

---

## `Wave`

A bound compute pipeline -- compiled kernel with field bindings.
Created via `kernel_fn(&gpu)` (the function generated by `#[quanta::kernel]`).

| Method | Returns | Description |
|--------|---------|-------------|
| `bind(slot, &field)` | `()` | Bind a field at a slot |
| `bind_texture(slot, &texture)` | `()` | Bind a texture at a slot |
| `set_value(slot, value)` | `()` | Set push constant (any `Copy` type) |
| `set_bytes(slot, &data)` | `()` | Set raw push constant bytes |
| `handle()` | `u64` | Raw GPU handle |

Waves are reusable: rebind fields and dispatch again with different data.
All binding state is stored inline (no heap allocation on the hot path).

---

## `Pulse`

GPU completion signal returned by dispatch/render operations.

| Method | Returns | Description |
|--------|---------|-------------|
| `wait()` | `Result<()>` | Block until GPU completes this operation |
| `is_done()` | `bool` | Non-blocking completion check |
| `reset()` | `()` | Reset for reuse |
| `handle()` | `u64` | Raw GPU handle |

---

## `RenderBuilder`

Chainable render pass builder. Created by `gpu.render(&target)`. Every method
consumes and returns `self`, so the entire pass is a single expression ending
in `.pulse()`.

```rust
let mut pulse = gpu.render(&target)?
    .clear(Color::BLACK)
    .pipeline(&pipeline)
    .vertices(0, &verts)
    .draw(3)
    .pulse()?;
pulse.wait()?;
```

### Pipeline

| Method | Description |
|--------|-------------|
| `.pipeline(&p)` | Bind render pipeline |

### Geometry

| Method | Description |
|--------|-------------|
| `.vertices(slot, &field)` | Bind vertex buffer at slot |
| `.vertices_offset(slot, &field, offset)` | Bind with byte offset |
| `.indices(&field)` | Bind index buffer (u32) |

### Shader resources

| Method | Description |
|--------|-------------|
| `.field(slot, &field)` | Bind storage buffer at slot |
| `.uniform(slot, &field)` | Bind uniform buffer at slot |
| `.texture(slot, &tex)` | Bind texture at slot |
| `.sampler(slot, desc)` | Set sampler state |
| `.value(slot, &val)` | Set push constant |

### Draw commands

| Method | Description |
|--------|-------------|
| `.draw(vertex_count)` | Draw non-indexed |
| `.draw_instanced(verts, instances)` | Instanced draw |
| `.draw_indexed(index_count)` | Draw indexed |
| `.draw_indexed_instanced(idxs, insts)` | Indexed + instanced |
| `.draw_indirect(&buffer, offset)` | GPU-driven draw |
| `.draw_indexed_indirect(&buffer, offset, &indices)` | GPU-driven indexed draw |

### Render state

| Method | Description |
|--------|-------------|
| `.clear(color)` | Clear color attachment |
| `.clear_depth(depth)` | Clear depth attachment |
| `.clear_stencil(value)` | Clear stencil |
| `.stencil_ref(value)` | Set stencil reference |
| `.scissor(x, y, w, h)` | Set scissor rect (pixels) |
| `.viewport(x, y, w, h)` | Set viewport |
| `.viewport_depth(x, y, w, h, min, max)` | Viewport with depth range |
| `.shading_rate(rate)` | Variable rate shading |
| `.shading_rate_image(&tex)` | Per-pixel shading rate |
| `.color_targets(targets)` | MRT color targets |
| `.depth_target(target)` | Depth/stencil target |

### Queries and debug

| Method | Description |
|--------|-------------|
| `.begin_occlusion_query(&q, idx)` | Start occlusion query |
| `.end_occlusion_query(&q, idx)` | End occlusion query |
| `.debug_push(label)` | Push debug group |
| `.debug_pop()` | Pop debug group |

### Terminal

| Method | Returns | Description |
|--------|---------|-------------|
| `.pulse()` | `Result<Pulse>` | Submit and return completion signal |

---

## `Batch`

A batch of GPU dispatches recorded into a single command buffer.
Multiple kernels are encoded without per-dispatch commit overhead.

```rust
let mut batch = gpu.batch()?;
batch.dispatch(&wave1, n)?;
batch.dispatch(&wave2, n)?;
let mut pulse = batch.pulse()?;
pulse.wait()?;
```

| Method | Returns | Description |
|--------|---------|-------------|
| `dispatch(&wave, quarks)` | `Result<()>` | Encode a dispatch into the batch |
| `pulse()` | `Result<Pulse>` | Submit all dispatches, return one completion signal |

---

## `Pipeline`

Compiled render pipeline (vertex + fragment + state).

| Method | Returns | Description |
|--------|---------|-------------|
| `handle()` | `u64` | Raw GPU handle |

Created via `gpu.pipeline(&PipelineDesc { ... })`.

---

## `IndirectCommandBuffer`

Pre-recorded sequence of GPU dispatches / draws. Created via
`gpu.indirect_command_buffer(capacity)`. Drop releases the backend
handle.

| Method | Returns | Description |
|--------|---------|-------------|
| `record_dispatch(&wave, [x,y,z])` | `Result<()>` | Append a compute dispatch |
| `record_draw(&pipeline, vc, ic)` | `Result<()>` | Append a draw |
| `execute(count)` | `Result<()>` | Replay the first `count` recorded commands |
| `execute_all()` | `Result<()>` | Replay every recorded command |
| `len()` / `capacity()` / `is_empty()` | `u32` / `bool` | Sizes |
| `handle()` | `u64` | Raw GPU handle |

`record_*` returns `InvalidParam` if full or destroyed. `execute(count)`
returns `InvalidParam` if `count > len()`.

---

## `IndirectRenderBundle`

Render-path equivalent of `IndirectCommandBuffer` — replayed inside
an active render pass via `pass.execute_bundle(&bundle, count)`.

| Method | Returns | Description |
|--------|---------|-------------|
| `record_draw(&pipeline, vc, ic)` | `Result<()>` | Append a draw |
| `len()` / `capacity()` / `is_empty()` | `u32` / `bool` | Sizes |
| `handle()` | `u64` | Raw GPU handle |

---

## `SparseTexture`

Virtual texture with on-demand page residency. Created via
`gpu.sparse_texture(&TextureDesc)`. Drop walks remaining tile bindings,
waits for the queue, and releases the heap / image.

| Method | Returns | Description |
|--------|---------|-------------|
| `map_tile(mip, x, y, backing)` | `Result<()>` | Commit physical pages for tile `(mip, x, y)` |
| `unmap_tile(mip, x, y)` | `Result<()>` | Release the binding (idempotent) |
| `width()` / `height()` | `u32` | Virtual extent in pixels |
| `handle()` | `u64` | Raw GPU handle |

v0.1 limit: 2D color textures with single mip only (3D / Cube / Array
return `NotSupported` at create time). See `docs/expert/sparse-textures.md`.

---

## `AccelerationStructure`

Bottom-level (BLAS) or top-level (TLAS) BVH over geometry. Created via
`gpu.acceleration_structure_blas(&[GeometryDesc])`. Drop tears down
the AS handle + storage buffer.

| Method | Returns | Description |
|--------|---------|-------------|
| `kind()` | `AsKind` | `Bottom` (BLAS) or `Top` (TLAS) |
| `geom_count()` | `u32` | Number of geometries (BLAS) / instances (TLAS) |
| `handle` | `u64` | Public field — raw GPU handle |

v0.1 ships the AS proc-addr foundation; the GPU-side build dispatch
returns `NotSupported` until the AMDGPU runner validates the path.
See `docs/expert/ray-tracing.md`.

---

## `RayTracingPipeline`

Three-stage ray-tracing pipeline (raygen / closest-hit / miss).
Created via `gpu.ray_tracing_pipeline(&desc)`. Drop releases the
pipeline.

| Method | Returns | Description |
|--------|---------|-------------|
| `dispatch_rays(w, h)` | `Result<()>` | Trace `w × h` rays |
| `max_recursion()` | `u32` | Recursion depth this pipeline was built with |
| `handle()` | `u64` | Raw GPU handle |

`MAX_DISPATCH_DIM = 65535`, `MAX_RECURSION_DEPTH = 31`.

---

## `Queue`

Typed multi-queue submission handle. Created via `gpu.queue(QueueType)`.
Drop releases the backend handle.

| Method | Returns | Description |
|--------|---------|-------------|
| `submit(&wave, [x,y,z])` | `Result<()>` | Submit compute dispatch on this queue |
| `signal(semaphore)` | `Result<()>` | Signal a semaphore from this queue |
| `wait(semaphore)` | `Result<()>` | Wait on a semaphore before continuing |
| `kind()` | `QueueType` | Capability tier (graphics / compute / transfer) |
| `handle()` | `u64` | Raw GPU handle |

---

## `AsyncCopyQueue`

Transfer queue running concurrently with compute / graphics. Created
via `gpu.async_copy_queue()`. Drop releases the backend handle.

| Method | Returns | Description |
|--------|---------|-------------|
| `copy_buffer::<T>(&dst, &src, size)` | `Result<()>` | Buffer-to-buffer copy on this queue |
| `copy_buffer_raw(dst, src, size)` | `Result<()>` | Raw-handle copy (escape hatch) |
| `handle()` | `u64` | Raw GPU handle |

Cross-queue ordering must be established via `Queue::signal` /
`Queue::wait` if other queues need to observe the copy.

---

## `PrintfBuffer`

Capacity-bounded shader-printf ring drained by the host. Created via
`gpu.printf_buffer(capacity)`. Drop releases the backend handle.

| Method | Returns | Description |
|--------|---------|-------------|
| `record(msg_id)` | `Result<()>` | Host-side record (testing / shim path) |
| `drain()` | `Result<Vec<u64>>` | Drain all recorded messages, leaving the buffer empty |
| `capacity()` | `u32` | Maximum recorded messages |
| `handle()` | `u64` | Raw GPU handle |

---

## `VrsState`

Variable rate shading state. Created via `gpu.vrs_state()`. Drop
releases the backend handle.

| Method | Returns | Description |
|--------|---------|-------------|
| `set_rate(ShadingRate)` | `Result<()>` | Switch to a new shading rate |
| `current()` | `ShadingRate` | Currently bound rate |
| `handle()` | `u64` | Raw GPU handle |

---

## `ShadingRate`

```rust
enum ShadingRate { R1x1, R1x2, R2x1, R2x2, R2x4, R4x2, R4x4 }
```

Cross-vendor shading rate. `R2x2` means one fragment-shader invocation
covers a 2×2 pixel block. `x_axis()` / `y_axis()` return the per-axis
factor. Use `gpu.supported_shading_rates()` to enumerate concrete
rates the device exposes.

---

## `ShaderBinary`

Compiled shader output from `#[quanta::vertex]` or `#[quanta::fragment]`.
Contains native binaries (SPIR-V + metallib), not text sources.

| Field | Type | Description |
|-------|------|-------------|
| `spirv` | `Option<&'static [u8]>` | SPIR-V binary |
| `metallib` | `Option<&'static [u8]>` | Pre-compiled metallib |
| `entry_point` | `&'static str` | Shader entry point name |
| `stage` | `ShaderStage` | Pipeline stage |

---

## `KernelBinary`

Compiled kernel output from `#[quanta::kernel]`. All fields are native
binaries -- no text sources (MSL/WGSL) are included in the build path.

| Field | Type | Description |
|-------|------|-------------|
| `amd` | `Option<&'static [u8]>` | AMD GCN ELF binary |
| `nvidia` | `Option<&'static [u8]>` | NVIDIA PTX binary |
| `spirv` | `Option<&'static [u8]>` | SPIR-V binary (Vulkan) |
| `metallib` | `Option<&'static [u8]>` | Pre-compiled metallib (Apple) |
| `llvm_ir` | `Option<&'static [u8]>` | LLVM IR fallback |

---

## `GpuType` trait

Marker trait for types usable in GPU fields. Implemented for all scalar types
(`f32`, `u32`, `i32`, `f64`, `u64`, `i64`, `u16`, `i16`, `u8`, `i8`).

```rust
pub trait GpuType: Copy + 'static {
    fn gpu_size() -> usize;
    fn scalar_type() -> ScalarType;
}
```

Automatic `GpuType` implementation is generated by:
- `#[quanta::gpu_type]` -- for storage buffer element types
- `#[derive(quanta::Uniforms)]` -- for uniform buffer structs

---

## Free functions

### Initialization

| Function | Returns | Description |
|----------|---------|-------------|
| `quanta::init()` | `Result<Gpu>` | Discover and initialize the first available GPU |
| `quanta::init_cpu()` | `Gpu` | Create a CPU software executor (requires `software` feature) |
| `quanta::devices()` | `Vec<Gpu>` | List all available GPUs |

Set the environment variable `QUANTA_CPU=1` as an alternative to calling
`init_cpu()`. When set, `init()` includes the CPU software executor.

---

## `quanta::scan` module

Prefix sum utilities (requires `software` feature).

| Function | Returns | Description |
|----------|---------|-------------|
| `exclusive_scan_f32_bytes(input)` | `Vec<u8>` | Exclusive prefix sum on raw f32 byte slice |

---

## Design decisions

Features Quanta deliberately does not include:

| Feature | Rationale |
|---------|-----------|
| **Swapchain / window management** | Quanta renders to textures. The host application owns the window, surface, and presentation. |
| **Geometry shaders** | Deprecated in Metal and Vulkan best practices. Mesh shaders (`#[quanta::mesh]`) are the replacement. |
| **HLSL / GLSL input** | Rust is the shader language. One language for CPU and GPU. |
| **Dynamic parallelism** | Not supported by Metal or Vulkan compute. Use multiple `gpu.dispatch()` calls or `gpu.batch()`. |
