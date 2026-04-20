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
| `nuclei()` | `u32` | Number of compute units |
| `protons_per_nucleus()` | `u32` | Cores per compute unit |
| `quarks_per_proton()` | `u32` | Threads per core (warp width) |
| `total_quarks()` | `u32` | Total parallel execution units |
| `name()` | `&str` | Device name string |

### Fields (typed GPU memory)

| Method | Returns | Description |
|--------|---------|-------------|
| `field::<T>(count, usage)` | `Result<Field<T>>` | Allocate with explicit usage |
| `compute_field::<T>(count)` | `Result<Field<T>>` | Read + write + compute + transfer |
| `render_field::<T>(count)` | `Result<Field<T>>` | Read + render + transfer |
| `uniform_field::<T>(count)` | `Result<Field<T>>` | Read + uniform + transfer |
| `field_mapped::<T>(count)` | `Result<MappedField<T>>` | CPU-mapped buffer (zero-copy) |
| `write_field(field, data)` | `Result<()>` | Upload data to GPU |
| `read_field(field)` | `Result<Vec<T>>` | Download data from GPU |
| `copy_field(dst, src)` | `Result<()>` | GPU-to-GPU copy |
| `resize_field(old, new_count, usage)` | `Result<Field<T>>` | Resize with data copy |

### Textures

| Method | Returns | Description |
|--------|---------|-------------|
| `texture(width, height)` | `Result<Texture>` | Simple RGBA8 texture |
| `create_texture(desc)` | `Result<Texture>` | Full-control creation |
| `render_target(w, h, fmt)` | `Result<Texture>` | Can be drawn to + sampled |
| `msaa_target(w, h, fmt, samples)` | `Result<Texture>` | Multi-sampled render target |
| `texture_write(tex, data)` | `Result<()>` | Upload pixel data |
| `texture_read(tex)` | `Result<Vec<u8>>` | Download pixel data |
| `sampler(desc)` | `Result<Sampler>` | Create reusable sampler |
| `generate_mipmaps(tex)` | `Result<()>` | Auto-generate mip chain |
| `resolve_texture(msaa, dst)` | `Result<()>` | Resolve MSAA to single-sample |
| `texture_view_create(tex, desc)` | `Result<TextureView>` | Sub-range view |
| `format_caps(format)` | `FormatCaps` | Query format capabilities |

### Compute

| Method | Returns | Description |
|--------|---------|-------------|
| `wave(kernel_bytes)` | `Result<Wave>` | Create wave from compiled kernel |
| `dispatch(wave, quarks)` | `Result<Pulse>` | Dispatch 1D (convenience) |
| `wave_dispatch(wave, [x,y,z])` | `Result<Pulse>` | Dispatch with group counts |
| `dispatch_indirect(wave, buf, off)` | `Result<Pulse>` | GPU-driven dispatch |
| `reload_wave(wave, kernel)` | `Result<()>` | Hot-reload kernel binary |

### Render

| Method | Returns | Description |
|--------|---------|-------------|
| `pipeline(desc)` | `Result<Pipeline>` | Create render pipeline |
| `render_begin(target)` | `Result<RenderPass>` | Begin render pass |
| `render_end(pass)` | `Result<Pulse>` | Submit render pass |
| `dispatch_mesh(pipeline, groups)` | `Result<()>` | Mesh shader dispatch |

### Sync

| Method | Returns | Description |
|--------|---------|-------------|
| `wait(pulse)` | `Result<()>` | Block until GPU completes |
| `wait_and_reset(pulse)` | `Result<()>` | Wait then reset for reuse |
| `poll(pulse)` | `bool` | Non-blocking completion check |
| `barrier()` | `Result<()>` | Full pipeline barrier |
| `barrier_buffer(field, from, to)` | `Result<()>` | Buffer state transition |
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
| `occlusion_query_read(query)` | `Result<Vec<u64>>` | Read fragment counts |

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

---

## `Field<T>`

GPU-resident typed buffer. Created via `gpu.compute_field()` or `gpu.field()`.

| Method | Returns | Description |
|--------|---------|-------------|
| `len()` | `usize` | Element count |
| `is_empty()` | `bool` | True if count is 0 |
| `byte_size()` | `usize` | Size in bytes |
| `handle()` | `u64` | Raw GPU handle |

Dropped automatically when it goes out of scope.

---

## `MappedField<T>`

CPU-mapped GPU buffer for zero-copy writes.

| Method | Returns | Description |
|--------|---------|-------------|
| `write(index, value)` | `()` | Write single element |
| `read(index)` | `T` | Read single element |
| `as_slice()` | `&[T]` | Immutable slice view |
| `as_mut_slice()` | `&mut [T]` | Mutable slice view |
| `len()` | `usize` | Element count |
| `byte_size()` | `usize` | Size in bytes |
| `handle()` | `u64` | Raw GPU handle |

---

## `Texture`

GPU-resident 2D image.

| Method | Returns | Description |
|--------|---------|-------------|
| `width()` | `u32` | Width in pixels |
| `height()` | `u32` | Height in pixels |
| `format()` | `Format` | Pixel format |
| `handle()` | `u64` | Raw GPU handle |

---

## `Wave`

A bound compute pipeline — compiled kernel with field bindings.

| Method | Returns | Description |
|--------|---------|-------------|
| `bind(slot, field)` | `()` | Bind buffer at slot |
| `bind_texture(slot, texture)` | `()` | Bind texture at slot |
| `set_value(slot, value)` | `()` | Set push constant (any Copy type) |
| `set_bytes(slot, data)` | `()` | Set raw push constant bytes |
| `handle()` | `u64` | Raw GPU handle |

---

## `Pulse`

GPU completion signal returned by dispatch/render operations.

| Method | Returns | Description |
|--------|---------|-------------|
| `wait()` | `Result<()>` | Block until completed |
| `is_done()` | `bool` | Non-blocking check |
| `reset()` | `()` | Reset for reuse |
| `handle()` | `u64` | Raw GPU handle |

---

## `RenderPass`

Active render pass — record draw commands, then submit via `gpu.render_end()`.

### Pipeline

| Method | Description |
|--------|-------------|
| `set_pipeline(pipeline)` | Bind render pipeline |

### Geometry

| Method | Description |
|--------|-------------|
| `bind_vertices(slot, field)` | Bind vertex buffer |
| `bind_vertices_offset(slot, field, offset)` | Bind with byte offset |
| `bind_indices(field)` | Bind index buffer (u32) |

### Resources

| Method | Description |
|--------|-------------|
| `set_field(slot, field)` | Bind storage buffer |
| `set_uniform(slot, field)` | Bind uniform buffer |
| `set_texture(slot, texture)` | Bind texture |
| `set_sampler(slot, desc)` | Set sampler state |
| `set_value(slot, value)` | Set push constant |

### Draw

| Method | Description |
|--------|-------------|
| `draw(vertex_count)` | Draw non-indexed |
| `draw_instanced(verts, instances)` | Instanced draw |
| `draw_indexed(index_count)` | Draw indexed |
| `draw_indexed_instanced(idxs, insts)` | Indexed + instanced |
| `draw_indirect(buffer, offset)` | GPU-driven draw |

### State

| Method | Description |
|--------|-------------|
| `clear(color)` | Clear color attachment |
| `clear_depth(depth)` | Clear depth attachment |
| `clear_stencil(value)` | Clear stencil attachment |
| `set_stencil_ref(value)` | Set stencil reference |
| `set_scissor(x, y, w, h)` | Set scissor rect |
| `set_viewport(x, y, w, h)` | Set viewport |
| `set_shading_rate(rate)` | Set VRS rate |
| `set_color_targets(targets)` | Set MRT targets |
| `set_depth_target(target)` | Set depth target |

---

## `Pipeline`

Compiled render pipeline (vertex + fragment + state).

| Method | Returns | Description |
|--------|---------|-------------|
| `handle()` | `u64` | Raw GPU handle |

Created via `gpu.pipeline(&PipelineDesc { ... })`.

---

## `ShaderBinary`

Compiled shader output from `#[quanta::vertex]` or `#[quanta::fragment]`.

| Field | Type | Description |
|-------|------|-------------|
| `msl` | `Option<&'static str>` | Metal Shading Language source |
| `wgsl` | `Option<&'static str>` | WebGPU Shading Language source |
| `spirv` | `Option<&'static [u8]>` | SPIR-V binary |
| `entry_point` | `&'static str` | Shader entry point name |
| `stage` | `ShaderStage` | Which pipeline stage |

| Method | Returns | Description |
|--------|---------|-------------|
| `for_vendor(vendor)` | `Option<&[u8]>` | Select best format for vendor |

---

## `KernelBinary`

Compiled kernel output from `#[quanta::kernel]`.

| Field | Type | Description |
|-------|------|-------------|
| `amd` | `Option<&'static [u8]>` | AMD GCN binary |
| `nvidia` | `Option<&'static [u8]>` | NVIDIA PTX |
| `spirv` | `Option<&'static [u8]>` | SPIR-V binary |
| `metallib` | `Option<&'static [u8]>` | Pre-compiled metallib |
| `msl` | `Option<&'static str>` | Metal source |
| `wgsl` | `Option<&'static str>` | WGSL source |
| `llvm_ir` | `Option<&'static [u8]>` | LLVM IR fallback |

| Method | Returns | Description |
|--------|---------|-------------|
| `for_vendor(vendor)` | `Option<&[u8]>` | Select best format for vendor |
