# API Reference

All public types in the Quanta GPU framework.

Quanta is split into three consumable surfaces: **`quanta-core`** (the
shared substrate — `Gpu`, drivers, fields, textures, errors, capability
queries), the **compute face** of the `quanta` facade (kernels, `Wave`
dispatch, the scan library — behind the `compute` feature), and
**`quanta-render`** (render passes, pipelines, typed
mesh/tessellation/RT/VRS wrappers, `Surface` — pulled in by the facade's
`render` feature). The facade re-exports everything, so `use quanta::*;`
covers the whole surface listed here.

> **Render methods live on the `RenderGpu` extension trait.** The
> render methods below (`gpu.pipeline()`, `gpu.render()`, …) are not
> inherent on `Gpu` — they come from the sealed `RenderGpu` trait in
> `quanta-render`. Bring it into scope with `use quanta::RenderGpu;`
> (or `use quanta::*;`).

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
| `supports_cooperative_matrix()` | `bool` | Cooperative-matrix / `simdgroup_matrix` support. True on Metal Apple GPU family 7+; **false on Vulkan** (`VK_KHR_cooperative_matrix` is not yet wired) and the software lane |
| `supports_f64()` | `bool` | Kernels may use 64-bit floats. True on the software lane and llvmpipe; false on Metal (MSL has no `double`) and Broadcom V3D |
| `supports_i64()` | `bool` | Kernels may use 64-bit integers (`shaderInt64` on Vulkan). True on the software lane and llvmpipe; false on Metal and Broadcom V3D |
| `supports_subgroups()` | `bool` | Subgroup *arithmetic* intrinsics (`reduce_*` / `scan_add_*` / `shuffle_*`). True on the software lane, Metal, and llvmpipe; false on Broadcom V3D (vote/ballot still work there) |
| `supports_async_compute()` | `bool` | Whether a dedicated async-compute queue is available. **Returns `false` on every backend today** — no driver overrides it yet. For overlapping submission use `gpu.queue(QueueType::Compute)` |
| `supports_compute_textures()` | `bool` | Compute kernels may bind textures (`&Texture2D` sampled reads / `&mut Texture2D<f32>` storage writes). True on Metal, the software driver, and native Vulkan (storage load/write; sampling in compute is not yet wired on Vulkan); false on WebGPU |
| `supports_native_handle_export()` | `bool` | `Texture::native_handle()` returns a real backend object. True on Metal and Vulkan; false on the CPU software driver and WebGPU |
| `supports_surface_present()` | `bool` | Presentation surfaces (`create_surface` + acquire/present). True on Metal, and on Vulkan when the loader offers VK_KHR_surface + VK_KHR_swapchain |
| `supports_texture_write_region()` | `bool` | Sub-region texture uploads (`Texture::write_region`). True on Metal, Vulkan, and the software driver; false on WebGPU |
| `narrow_storage_u32_slot()` | `bool` | Whether bf16/fp8 buffers use the portable u32-slot layout (one element per 32-bit word) instead of native 2-/1-byte stride. True only on WebGPU — WGSL storage buffers cannot hold 16-/8-bit array elements; the host must repack tight data one-element-per-word before binding |
| `supported_shading_rates()` | `Vec<(u32, u32)>` | Concrete (x,y) shading rates the device exposes (e.g. `[(1,1), (2,2), (4,4)]`). Empty when VRS is not supported. |

#### Per-backend summary

`Metal` = Apple Silicon; `Vulkan` = a real GPU (llvmpipe matches except
where noted); `CPU` = the software lane; `WebGPU` = WGSL. Feature-gated
rows (VRS, ray tracing, mesh, tessellation, sparse) are also
device-family- and extension-dependent within a backend.

| Query | Metal | Vulkan | CPU | WebGPU |
|-------|:-----:|:------:|:---:|:------:|
| `supports_vrs` | family 7+ | ext | ✗ | ✗ |
| `supports_ray_tracing` | family 6+ | ext | ✗ | ✗ |
| `supports_mesh_shaders` | Metal 3 | ext | ✗ | ✗ |
| `supports_tessellation` | family 4+ | feature | ✗ | ✗ |
| `supports_sparse_residency` | family 7+ | feature | ✗ | ✗ |
| `supports_cooperative_matrix` | family 7+ | ✗ (not wired) | ✗ | ✗ |
| `supports_f64` | ✗ | driver | ✓ | ✗ |
| `supports_i64` | ✗ | driver | ✓ | ✗ |
| `supports_subgroups` | ✓ | driver | ✓ | ✗ |
| `supports_async_compute` | ✗ | ✗ | ✗ | ✗ |
| `supports_compute_textures` | ✓ | ✓ | ✓ | ✗ |
| `supports_native_handle_export` | ✓ | ✓ | ✗ | ✗ |
| `supports_surface_present` | ✓ | WSI | ✗ | ✗ |
| `supports_texture_write_region` | ✓ | ✓ | ✓ | ✗ |
| `narrow_storage_u32_slot` | ✗ | ✗ | ✗ | ✓ |

### Fields (typed GPU memory)

| Method | Returns | Description |
|--------|---------|-------------|
| `field::<T>(count)` | `Result<Field<T>>` | Allocate with default compute usage (storage + transfer) |
| `field_with_usage::<T>(count, usage)` | `Result<Field<T>>` | Allocate with explicit `FieldUsage` flags (`default_compute()` / `default_render()` / `default_uniform()` or a custom union) |
| `field_mapped::<T>(count)` | `Result<MappedField<T>>` | CPU-mapped buffer (zero-copy) |

### Textures

| Method | Returns | Description |
|--------|---------|-------------|
| `texture(width, height)` | `Result<Texture>` | Simple RGBA8 texture |
| `create_texture(&desc)` | `Result<Texture>` | Full-control creation (`TextureDesc::new(w, h, fmt).with_*(…)`) |
| `sampler(&desc)` | `Result<Sampler>` | Create reusable sampler (`SamplerDesc::default().with_*(…)`) |
| `texture_view_create(tex, desc)` | `Result<TextureView>` | Sub-range view |
| `format_caps(format)` | `FormatCaps` | Query format capabilities |
| `sparse_texture(&desc)` | `Result<SparseTexture>` | Virtual texture with on-demand tile residency (2D, single-mip in v0.1) |

Render targets (`render_target`, `msaa_target`, `resolve_texture`) moved
to the [`RenderGpu`](#render-the-rendergpu-extension-trait) extension
trait below.

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

### Render (the `RenderGpu` extension trait)

The render methods are a **sealed extension trait** implemented for
`Gpu` by the `quanta-render` crate — bring it into scope with
`use quanta::RenderGpu;` (or `use quanta::*;`):

```rust
use quanta::RenderGpu;

let target = gpu.render_target(640, 480, Format::RGBA8)?;
let pipe = gpu.pipeline(&desc)?;
```

| Method | Returns | Description |
|--------|---------|-------------|
| `pipeline(&desc)` | `Result<Pipeline>` | Create render pipeline (`PipelineDesc::new(shader).with_*(…)`) |
| `render(&target)` | `Result<RenderBuilder>` | Begin render pass (builder chain) |
| `render_target(w, h, fmt)` | `Result<Texture>` | Can be drawn to + sampled |
| `msaa_target(w, h, fmt, samples)` | `Result<Texture>` | Multi-sampled render target |
| `resolve_texture(&msaa, &dst)` | `Result<()>` | Resolve MSAA to single-sample |
| `stencil_read(&tex)` | `Result<Vec<u8>>` | Read stencil buffer contents |
| `render_bundle(max_commands)` | `Result<IndirectRenderBundle>` | Render-path indirect command bundle |
| `mesh_pipeline(desc)` | `Result<MeshPipeline>` | Create a mesh-shader pipeline (gated on `supports_mesh_shaders`); `dispatch(groups)` on the wrapper dispatches |
| `tessellation_pipeline(topology, control_points)` | `Result<TessellationPipeline>` | Create a tessellation pipeline (gated on `supports_tessellation`) |
| `create_surface(&target, &config)` | `Result<Surface>` | Presentation surface — see [`Surface`](#surface) below |
| `acceleration_structure_blas(geoms)` | `Result<AccelerationStructure>` | Build a bottom-level BVH (foundation only in v0.1 — build dispatch returns `NotSupported`) |
| `ray_tracing_pipeline(&desc)` | `Result<RayTracingPipeline>` | Construct a ray-tracing pipeline; `dispatch_rays(w, h)` traces |
| `vrs_state()` | `Result<VrsState>` | Variable rate shading handle — `set_rate(ShadingRate)` to switch |
| `occlusion_query_create(count)` | `Result<OcclusionQuery>` | Create occlusion query |
| `occlusion_query_read(&query)` | `Result<Vec<u64>>` | Read fragment counts (synchronous, native backends only) |

### Sync

| Method | Returns | Description |
|--------|---------|-------------|
| `barrier()` | `Result<()>` | Full pipeline barrier |
| `barrier_field(field, from, to)` | `Result<()>` | Field state transition |
| `barrier_texture(tex, from, to)` | `Result<()>` | Texture state transition |
| `wait_idle()` | `Result<()>` | Host-blocking drain: waits until every submitted operation completes. Use before CPU-side reads when the pulse wasn't kept |

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

Occlusion queries (`occlusion_query_create` / `occlusion_query_read`)
are render-path methods on the `RenderGpu` trait above.

> **WebGPU note.** WebGPU has no synchronous readback of query results.
> On the WebGPU backend, `occlusion_query_read` returns
> `NotSupported`; use `occlusion_query_read_async(query).await` on
> the WebGPU driver directly for the Promise-based path.

### Multi-queue

| Method | Returns | Description |
|--------|---------|-------------|
| `queue_families()` | `Vec<QueueFamily>` | Available queue families |
| `queue(QueueType)` | `Result<Queue>` | Typed queue wrapper — submit/signal/wait via [`Queue`](#queue) methods |

The raw-handle variants (`create_queue`, `queue_dispatch`,
`queue_signal`, `queue_wait`) were removed in the v0.1 API scrub — use
the typed [`Queue`](#queue) wrapper.

### Debug

| Method | Returns | Description |
|--------|---------|-------------|
| `debug_push(label)` | `()` | Push debug group |
| `debug_pop()` | `()` | Pop debug group |

### Removed methods

These duplicate / raw-handle methods were removed in the v0.1 API
scrub. The replacements:

| Removed | Use instead |
|---------|-------------|
| `gpu.write_field(&field, &data)` | `field.write(&data)` |
| `gpu.read_field(&field)` | `field.read()` |
| `gpu.copy_field(&dst, &src)` | `dst.copy_from(&src)` |
| `gpu.resize_field(&old, n, usage)` | Allocate new field + `dst.copy_from(&old)` |
| `gpu.compute_field::<T>(n)` | `gpu.field::<T>(n)` (same default usage) |
| `gpu.render_field::<T>(n)` | `gpu.field_with_usage::<T>(n, FieldUsage::default_render())` |
| `gpu.uniform_field::<T>(n)` | `gpu.field_with_usage::<T>(n, FieldUsage::default_uniform())` |
| `gpu.texture_write(&tex, &data)` | `texture.write(&data)` |
| `gpu.texture_read(&tex)` | `texture.read()` |
| `gpu.generate_mipmaps(&tex)` | `texture.generate_mipmaps()` |
| `gpu.wait(&mut pulse)` | `pulse.wait()` |
| `gpu.wait_and_reset(&mut pulse)` | `pulse.wait()` + `pulse.reset()` |
| `gpu.poll(&pulse)` | `pulse.is_done()` |
| `gpu.begin_batch()` | `gpu.batch()` |
| `gpu.render_begin(&target)` | `gpu.render(&target)` (builder, via `RenderGpu`) |
| `gpu.render_end(pass)` | `.pulse()` on `RenderBuilder` |
| `gpu.barrier_buffer(field, from, to)` | `gpu.barrier_field(field, from, to)` |
| `gpu.create_queue(type)` / `queue_dispatch` / `queue_signal` / `queue_wait` | `gpu.queue(QueueType)` + `Queue::submit` / `signal` / `wait` |
| `gpu.dispatch_mesh(pipeline, groups)` | `MeshPipeline::dispatch(groups)` |

---

## `Field<T>`

GPU-resident typed buffer (storage buffer). Created via `gpu.field()`
or `gpu.field_with_usage()`. Freed automatically when dropped (RAII).

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

GPU-resident 2D image. Created via `gpu.texture()` or
`gpu.create_texture()`. Dropping a `Texture` releases the underlying
driver resource (exactly once) — the same holds for `TextureView`,
`Sampler`, and `Pipeline`.

| Method | Returns | Description |
|--------|---------|-------------|
| `write(&data)` | `Result<()>` | Upload pixel data |
| `write_region(origin, size, &data)` | `Result<()>` | Upload a sub-region: `origin`/`size` in texels, `data` tightly packed region rows (gated on `supports_texture_write_region`) |
| `read()` | `Result<Vec<u8>>` | Download pixel data |
| `generate_mipmaps()` | `Result<()>` | Auto-generate mip chain |
| `native_handle()` | `Result<NativeTextureHandle>` | Export the backend-native object for zero-copy interop (see below) |
| `width()` | `u32` | Width in pixels |
| `height()` | `u32` | Height in pixels |
| `format()` | `Format` | Pixel format |
| `handle()` | `u64` | Raw GPU handle |

### `NativeTextureHandle`

Backend-native handle exported from a `Texture` for zero-copy interop —
a compositor, the OS, or another graphics runtime imports the rendered
texture directly. The exported handle is a **borrow**: valid exactly as
long as the `Texture` (and its `Gpu`) are alive; ownership is not
transferred. An importer that needs it longer must take its own native
reference (e.g. ObjC `retain`) before the `Texture` drops.

```rust
match texture.native_handle()? {
    NativeTextureHandle::Metal { texture } => { /* id<MTLTexture> pointer */ }
    NativeTextureHandle::Vulkan { image, memory, vk_format, layout } => { /* VkImage + backing */ }
    _ => { /* non-exhaustive — new variants can be added */ }
}
```

Supported on **Metal and Vulkan**; the CPU software driver has no
native object and WebGPU export is reserved (both return
`NotSupported`). Query `gpu.supports_native_handle_export()` to branch
ahead of time. GPU work producing the texture's contents must be
complete (`pulse.wait()`) before the importer samples it.

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
| `on_complete(f)` | `Result<()>` | Consume the pulse; run `f` on a background waiter thread at completion — the event-driven alternative to `wait()` for actor mailboxes / ports / event loops |
| `is_done()` | `bool` | Non-blocking check: has `wait()` already observed completion (local state, not live GPU progress) |
| `reset()` | `()` | Reset for reuse |
| `handle()` | `u64` | Raw GPU handle |

---

## `RenderBuilder`

Chainable render pass builder. Created by `gpu.render(&target)`. Every method
consumes and returns `self`, so the entire pass is a single expression ending
in `.pulse()`.

Bindings record the resource's handle, not ownership: everything bound to the
pass (`Field`, `Texture`, `Pipeline`) must outlive `pulse()`. Dropping a bound
resource early makes `pulse()` fail with `NotFound` (dead handles are detected
before encoding; nothing is silently skipped).

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
| `.value(slot, &val)` | Set push constant. Metal: binds the bytes at buffer index `slot` on both stages (DSL uniforms can read them). Vulkan: real push constants (slots 0-7, 4-byte-aligned, ≤128 bytes total) — reachable only by hand-authored SPIR-V with a push-constant block; DSL shaders read uniforms from `.uniform()` descriptors instead |

Buffers and values are visible to **both stages**: a fragment shader reading a
uniform sees the same slot the vertex stage does (Metal binds both stages;
Vulkan descriptors are vertex+fragment visible). Slots 0-15 are user space —
vertex-attribute buffers live in a separate internal index space and never
collide. For a DSL fragment, `&Texture2D` params take texture slots in
declaration order (first texture param ↔ `.texture(0, …)`/`.sampler(0, …)`),
and uniform params take buffer slots in declaration order among uniforms
(first uniform ↔ `.uniform(0, …)`).

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
| `.scissor(x, y, w, h)` | Set scissor rect, pixels (clamped — see below) |
| `.viewport(x, y, w, h)` | Set viewport |
| `.viewport_depth(x, y, w, h, min, max)` | Viewport with depth range |
| `.shading_rate(rate)` | Variable rate shading |
| `.shading_rate_image(&tex)` | Per-pixel shading rate |
| `.color_targets(targets)` | MRT color targets |
| `.depth_target(target)` | Depth/stencil target |

Scissor offsets are **clamped to the render area on every backend**. An
offset that falls outside the target — including a negative offset passed
as a wrapped-in `u32`, the common "clip a child scrolled past its parent"
case — is pulled to the render-area edge and the extent shrinks to match;
a rectangle that clamps entirely away disables drawing for the pass
without raising an error. This gives identical results across Metal
(which tolerates such rectangles natively) and Vulkan (which would
otherwise reject a negative offset).

Viewport and scissor share one **cross-backend orientation convention**:
NDC +Y up, framebuffer origin top-left, readback row 0 = the top row, so
the same `x, y, width, height` places output identically on every backend
(Vulkan conforms via an internal negative-viewport y-flip; the viewport
`x/y` you pass are framebuffer-space top-left coordinates, unchanged). See
the [vertex/fragment coordinate conventions](../rendering/tutorials/vertex-fragment.md#coordinate-conventions).

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

Compiled render pipeline (vertex + fragment + state). Dropping a
pipeline releases the driver resource exactly once.

| Method | Returns | Description |
|--------|---------|-------------|
| `handle()` | `u64` | Raw GPU handle |

Created via `RenderGpu::pipeline`. `PipelineDesc` is
`#[non_exhaustive]` — construct it with the builder, not a struct
literal:

```rust
let desc = PipelineDesc::new(ShaderSource::Binaries {
        vertex: &VERTEX_SHADER,     // &ShaderBinary from #[quanta::vertex]
        fragment: &FRAGMENT_SHADER, // &ShaderBinary from #[quanta::fragment]
    })
    .with_color_formats(vec![Format::RGBA8])
    .with_cull_mode(CullMode::Back);
let pipeline = gpu.pipeline(&desc)?;
```

`with_color_formats` is **per-attachment**: entry `i` types color
attachment `i` of every pass the pipeline is used in — it is not a
candidate list of formats the pipeline may be used against. The count
must equal the pass's color-target count and format `i` must match bound
target `i`; both are enforced when the pass is submitted (a mismatch
fails `pulse()`), and a descriptor declaring more attachments than the
fragment writes is rejected at creation when a SPIR-V fragment is present
(a metallib-only shader can't be pre-reflected, so it skips that one).

`ShaderSource` supplies the shader payloads:
`Stages { vertex, fragment }` (raw per-stage bytes in the backend's
native format), `Combined(&[u8])` (one payload, both entry points), or
`Binaries { vertex, fragment }` (`&ShaderBinary` pairs — the driver
picks the right per-vendor format).

---

## `Surface`

A swapchain over a platform presentation target — the "Quanta owns
present" model. Created via `RenderGpu::create_surface(&SurfaceTarget,
&SurfaceConfig)`. Dropping the `Surface` releases the swapchain.

Supported on **Metal** (via a `CAMetalLayer`) and on **Vulkan** when the
loader offers the WSI extensions (`VK_KHR_surface` + `VK_KHR_swapchain`)
— on X11 through `SurfaceTarget::Xlib`, on Android through
`SurfaceTarget::AndroidWindow`, and on Windows through
`SurfaceTarget::Win32`, plus the windowless
`SurfaceTarget::Headless` on both. Backends without a present path return
`NotSupported`; query `gpu.supports_surface_present()` to branch ahead of
time. A swapchain that becomes suboptimal (e.g. after a resize the app
hasn't reconfigured yet) is self-healed on the next `acquire()` rather
than failing.

| Method | Returns | Description |
|--------|---------|-------------|
| `acquire()` | `Result<SurfaceFrame>` | Next presentable frame. `Timeout` if none free; `SurfaceOutdated` if the target was resized |
| `configure(config)` | `Result<()>` | Reconfigure — resize, format, or present-mode change |
| `config()` | `&SurfaceConfig` | Active configuration |
| `format()` | `Result<Format>` | The **negotiated** frame format (see below). Call after create, before building pipelines; pass to `with_color_formats` |
| `width()` / `height()` | `u32` | Current frame extent |

**Format negotiation.** `SurfaceConfig::format` is a *preference* on
Vulkan: the swapchain picks the first offered SRGB-nonlinear format from
`[requested, BGRA8, RGBA8]`, then any other format Quanta can express, so
a surface that only offers `RGBA8` (Android) still works with a `BGRA8`
request. Only a surface offering nothing expressible fails, with an error
listing what it offered. On Metal the configured format is exact (Quanta
sets the layer format). `surface.format()` returns what was actually
negotiated; a frame's `texture().format()` reports the same, and building
a pipeline for a different format is rejected at draw time. The chain
order is fixed — for a different fallback preference, type the pipeline
per frame from `frame.texture().format()`.

### `SurfaceFrame`

One acquired, presentable frame. `texture()` aliases the swapchain's
backing image — a borrow valid only until the frame is presented or
dropped; do not store it (or its `native_handle`) beyond the frame.
Dropping an unpresented frame discards it.

| Method | Returns | Description |
|--------|---------|-------------|
| `texture()` | `&Texture` | The frame's target — render into it with `gpu.render(frame.texture())` |
| `present()` | `Result<()>` | Present, consuming the frame. Call after `.pulse()` returned — no CPU wait needed between submit and present |

### Configuration types

- `SurfaceConfig::new(width, height)` — portable defaults: `BGRA8`,
  `PresentMode::Fifo`, `RENDER_TARGET` usage. `#[non_exhaustive]`;
  adjust fields by assignment. `format` is a *preference* on Vulkan (the
  swapchain may negotiate another offered format — read the result with
  `Surface::format()`) and *exact* on Metal.
- `SurfaceTarget::MetalLayer { layer }` — an existing `CAMetalLayer*`
  provided by the windowing environment.
  `SurfaceTarget::Xlib { display, window }` — an Xlib `Display*`
  and `Window` id (`VK_KHR_xlib_surface`); both must outlive the
  surface. `SurfaceTarget::AndroidWindow { a_native_window }` — an
  `ANativeWindow*` from the embedder (`VK_KHR_android_surface`); must
  outlive the surface. `SurfaceTarget::Win32 { hinstance, hwnd }` — an
  `HWND` and its owning module's `HINSTANCE` from the embedder's window
  (`VK_KHR_win32_surface`); both must outlive the surface.
  `SurfaceTarget::Headless` — no window attached; full
  acquire/present machinery (Metal off-screen layer /
  `VK_EXT_headless_surface`) for tests and compositor-fed consumers.
  The enum is `#[non_exhaustive]` — match with a wildcard arm.
- `PresentMode::{Fifo, Immediate, Mailbox}` — vsync (default; always
  supported where presenting works at all), lowest-latency tearing,
  triple-buffered. Unsupported modes are rejected at create/configure
  time.

The frame loop:

```rust
use quanta::RenderGpu;

let mut surface = gpu.create_surface(&target, &SurfaceConfig::new(1280, 720))?;
loop {
    let frame = match surface.acquire() {
        Ok(frame) => frame,
        Err(e) if matches!(e.kind, QuantaErrorKind::SurfaceOutdated(_)) => {
            surface.configure(new_config)?; // window resized
            continue;
        }
        Err(e) => return Err(e),
    };
    let mut pulse = gpu.render(frame.texture())?.clear(color).pulse()?;
    frame.present()?; // ordered after the submitted GPU work
}
```

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
| `wgsl` | `Option<&'static str>` | WGSL source (WebGPU) |
| `entry_point` | `&'static str` | Shader entry point name |
| `stage` | `ShaderStage` | Pipeline stage |

Pass shader binaries to a pipeline through
`ShaderSource::Binaries { vertex, fragment }` in `PipelineDesc::new` —
the driver selects the right per-vendor payload
(`ShaderBinary::for_vendor`).

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
| `quanta::init()` | `Result<Gpu>` | Discover and initialize the first available GPU, in the fixed per-OS order below |
| `quanta::init_cpu()` | `Gpu` | Create a CPU software executor (requires `software` feature) |
| `quanta::devices()` | `Vec<Gpu>` | List all available GPUs in discovery order — **the programmatic selection path**: enumerate and pick by index, name, or capability query. There is no separate "choose a device" API |

**Discovery order (the contract).** The probe order is deliberate and
stable — a given machine always initializes the same backend — not an
accident of enumeration:

- **Apple (macOS / iOS):** Metal, then the CPU device when `QUANTA_CPU=1`.
- **Linux / Android / Windows:** Vulkan, then the CPU device when `QUANTA_CPU=1`.
- **wasm:** WebGPU, via the async `init_webgpu_async()` — sync `init()`
  never returns a WebGPU device (the platform requires an async adapter
  handshake).

`init()` returns the first device `devices()` yields in that order; to
choose another, enumerate with `devices()` and pick from the list.

Set `QUANTA_CPU=1` as an alternative to calling `init_cpu()` — when set,
discovery includes the CPU software executor.

**Forcing a backend (`QUANTA_BACKEND`).** Set `QUANTA_BACKEND` to
`metal`, `vulkan`, or `cpu` (case-insensitive) to restrict discovery to
exactly that backend. A forced-but-unavailable backend does **not** fall
through to another: `devices()` returns an empty list and `init()` fails
with an error naming the env var, so CI never silently runs on the wrong
backend. An unrecognized value fails `init()` with a message listing the
accepted values. `cpu` includes the software device regardless of
`QUANTA_CPU`.

---

## `quanta::scan` module

Prefix sum utilities (requires `software` feature).

| Function | Returns | Description |
|----------|---------|-------------|
| `exclusive_scan_f32_bytes(input)` | `Vec<u8>` | Exclusive prefix sum on raw f32 byte slice |

---

## `quanta::nn` module

The neural stack (feature `nn`), built over the `autograd` tape and the
`sci` array. Completeness contract: `PARITY.md` at the crate root. Fused
kernels are theorem-backed; IDs link into `specs/THEOREMS.md`.

### `nn::layer` — the Layer model

| Item | Description |
|------|-------------|
| `trait Layer<T>` | Configuration + shapes: `in_dim() -> Option<usize>`, `out_dim(in)`, `init(&gpu, key) -> Params`, `apply(&tape, &vars, &x) -> Var` |
| `trait ParamTree<T>` | Typed parameter tree: `bind(&tape) -> Vars`, `flatten() -> Vec<Array<T>>`, `unflatten(iter)`, `grads(vars, loss)` / `grads_from`, `map(f)` |
| `Key` | Splittable PRNG key; `split(self)` and `uniform(self, …)` **consume** it — linear by ownership |
| `Linear { in_dim, out_dim, bias }` | Dense affine `[N, in] → [N, out]`; Kaiming-uniform init; params `LinearParams { w, b: Option }` |
| `LayerNorm { dim, eps }` / `RmsNorm { dim, eps }` | Norm layers over the fused kernels; params `NormParams { gamma, beta: Option }` |
| tuples `(L1, …, L6)` | Tuple stacking (arity ≤ 6): the tuple IS a layer; width contracts checked at `init`; `Params` = tuple of member trees |
| `()` as `ParamTree` | The empty tree — zero-parameter layers occupy stack slots for free |

### `nn::functional` — fused attention

| Item | Description |
|------|-------------|
| `scaled_dot_product_attention(gpu, q, k, v, Sdpa)` | Fused online-softmax forward (T9200–T9209); never materialises the score matrix; returns context + `(m, l)` row stats |
| `sdpa_var(tape, q, k, v, Sdpa)` | Tape-differentiable, fused BOTH directions (FlashAttention-style backward off the saved stats) |
| `sdpa_var_composed(…)` | The composed reference path — the differential-test oracle |
| `Sdpa` | Options: scale override, causal mask, padding masks |

### `nn::norm` / `nn::rope` — fused normalization & rotary

| Item | Description |
|------|-------------|
| `layer_norm_var(tape, x, gamma, beta, eps)` | Fused LayerNorm fwd/bwd; three-term adjoint backward (T9210) |
| `rms_norm_var(tape, x, gamma, eps)` | Fused RMSNorm (T9211, no centering term) |
| `rope_var(tape, x, cache)` | Fused rotary embedding; backward = same kernel with `sign = −1` (T9216–T9218) |

### `nn::activation` — fused + zero-param module forms

| Item | Description |
|------|-------------|
| `softmax_var` / `log_softmax_var` | Fused rowwise, max-stabilized (T9223); proven-adjoint backwards (T9224/T9225) |
| `gelu_var` | Fused tanh-form GeLU; backward reuses the forward's tanh (T9227) |
| `swiglu_var` | Fused gate `[N, 2H] → [N, H]`; σ′ from the forward's sigmoid (T9226) |
| `Relu, Gelu, Silu, Sigmoid, Tanh, Softmax, LogSoftmax, SwiGlu` | Zero-parameter layers (`Params = ()`) for tuple stacks; `SwiGlu` halves the width through the contracts |

### `nn::loss`

| Item | Description |
|------|-------------|
| `cross_entropy_var(tape, logits, &[u32], Reduction)` | FUSED stable CE: `lse(x) − x_y` forward (nonnegative, T9228), `softmax − onehot` backward |
| `mse_loss` / `l1_loss` / `huber_loss(δ)` | Composed; Huber gradient is globally `clamp(z, −δ, δ)` (T9230) |
| `bce_with_logits_loss` | Overflow-free spelling, proven equal to the textbook form (T9229) |
| `bce_loss` | Textbook BCE over probabilities in `(0, 1)` |
| `Reduction::{Mean, Sum}` | Scalar collapse for every loss |

### `nn::optim` — fused optimizers as tree operations

| Item | Description |
|------|-------------|
| `Sgd { lr, momentum, weight_decay, nesterov }` | One fused kernel per leaf: decay + velocity (T9219) + step |
| `Adam { lr, beta1, beta2, eps, weight_decay, decoupled }` | One fused kernel per leaf; exact bias correction (T9220); `decoupled: true` = AdamW (T9221) |
| `SgdState` / `AdamState` | State trees mirroring the params (flatten order); `step(&params, &grads, state)` **consumes** the state and returns `(new_params, new_state)` |
| `Schedule::{Constant, Step, LinearWarmup, Cosine}` | Pure `lr(t)`; feed back by rebuilding the `Copy` config |
| `clip_grad_norm(&grads, max)` | Global L2 over ALL leaves; returns `(clipped_tree, pre_clip_norm)` |
| `clip_grad_value(&grads, max_abs)` | Elementwise clamp over the tree |

---

## Design decisions

Features Quanta deliberately does not include:

| Feature | Rationale |
|---------|-----------|
| **Window management** | Quanta never creates windows. Presentation is supported two ways: a `Surface` over a platform target the host hands in (`SurfaceTarget::MetalLayer`), or exporting the rendered texture via `Texture::native_handle()` so an external compositor owns present. |
| **Geometry shaders** | Deprecated in Metal and Vulkan best practices. Mesh shaders (`#[quanta::mesh]`) are the replacement. |
| **HLSL / GLSL input** | Rust is the shader language. One language for CPU and GPU. |
| **Dynamic parallelism** | Not supported by Metal or Vulkan compute. Use multiple `gpu.dispatch()` calls or `gpu.batch()`. |
