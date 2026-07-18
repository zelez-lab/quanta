# Migration from wgpu

## Side-by-side: compute shader

### wgpu (50+ lines)

```rust
// 1. Device setup (3 async calls)
let instance = wgpu::Instance::new(Backends::all());
let adapter = instance.request_adapter(&Default::default()).await.unwrap();
let (device, queue) = adapter.request_device(&Default::default(), None).await.unwrap();

// 2. Shader module (separate WGSL file)
let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: None,
    source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
});

// 3. Bind group layout (describe every binding manually)
let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    entries: &[
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ],
    label: None,
});

// 4. Pipeline layout
let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    bind_group_layouts: &[&bgl],
    push_constant_ranges: &[],
    label: None,
});

// 5. Pipeline
let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    layout: Some(&pipeline_layout),
    module: &shader,
    entry_point: Some("main"),
    ..Default::default()
});

// 6. Buffers
let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: None,
    contents: bytemuck::cast_slice(&data),
    usage: wgpu::BufferUsages::STORAGE,
});
let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
    label: None,
    size: (N * 4) as u64,
    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    mapped_at_creation: false,
});

// 7. Bind group
let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    layout: &bgl,
    entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding() },
    ],
    label: None,
});

// 8. Encode + dispatch
let mut encoder = device.create_command_encoder(&Default::default());
{
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(64, 1, 1);
}
queue.submit(Some(encoder.finish()));

// 9. Read back (async map, poll, copy)
// ... another 10+ lines ...
```

### Quanta (5 lines of host code)

```rust
#[derive(quanta::Fields)]
struct DoubleData { input: Vec<f32>, output: Vec<f32> }

#[quanta::kernel]
fn double(input: &[f32], output: &mut [f32]) {
    let i = quark_id();
    output[i] = input[i] * 2.0;
}

fn main() -> Result<(), quanta::QuantaError> {
    let gpu = quanta::init()?;                      // 1. init
    let input = gpu.field::<f32>(1024)?;            // 2. allocate
    let output = gpu.field::<f32>(1024)?;
    input.write(&data)?;                            // 3. upload

    let mut wave = double(&gpu)?;                   // 4. compile
    wave.bind(0, &input);
    wave.bind(1, &output);
    let mut pulse = gpu.dispatch(&wave, 1024)?;     // 5. dispatch
    pulse.wait()?;

    let result = output.read()?;                    // 6. download
    Ok(())
}
```

**What disappears:**
- Shader module creation (WGSL is a separate language, separate file)
- Bind group layout descriptors (8 fields per binding)
- Pipeline layout descriptors
- Pipeline creation descriptors
- Bind group creation
- Command encoder + compute pass management
- `queue.submit()` (dispatch submits automatically)
- Async buffer mapping + polling

## API mapping

| wgpu | Quanta |
|------|--------|
| `wgpu::Instance::new(Backends::all())` | `quanta::init()` |
| `instance.request_adapter()` | `quanta::init()` (automatic) |
| `adapter.request_device()` | `quanta::init()` (automatic) |
| `device.create_buffer(...)` | `gpu.field::<T>(n)` |
| `device.create_buffer_init(...)` | `gpu.field(n)` + `field.write(&data)` |
| `device.create_shader_module(wgsl)` | `#[quanta::kernel]` (compile-time) |
| `device.create_compute_pipeline(...)` | automatic (inside `#[quanta::kernel]`) |
| `device.create_bind_group_layout(...)` | not needed |
| `device.create_bind_group(...)` | `wave.bind(slot, &field)` |
| `device.create_pipeline_layout(...)` | not needed |
| `encoder.begin_compute_pass()` | not needed |
| `pass.set_pipeline(&pipeline)` | automatic |
| `pass.set_bind_group(0, &bg, &[])` | `wave.bind(slot, &field)` |
| `pass.dispatch_workgroups(x, y, z)` | `gpu.dispatch(&wave, n)` |
| `encoder.copy_buffer_to_buffer(...)` | `dst.copy_from(&src)` |
| `queue.submit(...)` | automatic (dispatch submits) |
| `buffer.slice(..).map_async(...)` | `field.read()` |
| `device.poll(Maintain::Wait)` | `pulse.wait()` / `gpu.wait_idle()` |
| `queue.on_submitted_work_done(callback)` | `pulse.on_complete(\|\| { .. })` (runs on a waiter thread; consumes the pulse) |
| `queue.write_texture(origin, data, layout, size)` | `texture.write(&data)` / `texture.write_region(origin, size, &data)` |
| `var<storage, read> t: array<vec4<f32>>` read in a shader | `table: &[Vec4]` shader param, indexed `table[i]`; bound with `.uniform(slot, &field)` at the declaration index shared with `&T` uniforms |
| `texture_storage_2d<rgba8unorm, read_write>` in compute | `&mut Texture2D<u32>` kernel param (texels as packed `0xAABBGGRR` u32) |

## Key differences

**No WGSL.** You write Rust. The proc macro compiles to WGSL/MSL/PTX/GCN at build time.
No separate shader files. No string literals. No runtime compilation.

**No pipeline/bind group ceremony.** wgpu requires you to manually describe every binding
layout, create bind groups, create pipeline layouts, then create pipelines. Quanta infers
all of this from the kernel function signature.

**No async adapter/device creation.** `quanta::init()` discovers and initializes synchronously.

**No command encoder.** Dispatch submits immediately. No manual encoder/pass management.

**Typed buffers.** `Field<f32>` vs wgpu's untyped `Buffer`. Read/write operations are
type-safe -- no manual byte slicing.

## Render pipeline

### wgpu
```rust
let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
    vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs_main"),
        buffers: &[vertex_buffer_layout],
        ..Default::default()
    },
    fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: Some("fs_main"),
        targets: &[Some(wgpu::ColorTargetState { ... })],
        ..Default::default()
    }),
    primitive: wgpu::PrimitiveState { ... },
    depth_stencil: None,
    multisample: wgpu::MultisampleState::default(),
    multiview: None,
    layout: Some(&pipeline_layout),
    cache: None,
    label: None,
});
```

### Quanta
```rust
use quanta::RenderGpu; // render methods on Gpu come from this trait

// WGSL's shader-I/O struct becomes a #[derive(quanta::Varyings)] struct:
// one #[position] field (@builtin(position)), the rest @location in order.
#[derive(quanta::Varyings)]
struct VsOut {
    #[position] clip: Vec4, // @builtin(position)
    color: Vec4,            // @location(0)
}

#[quanta::vertex]
fn vs_main(pos: Vec3, in_color: Vec4, mvp: &Mat4) -> VsOut {
    VsOut {
        clip: mvp * Vec4::new(pos.x, pos.y, pos.z, 1.0),
        color: in_color,
    }
}

#[quanta::fragment]
fn fs_main(v: VsOut) -> Vec4 {
    v.color
}

let pipeline = gpu.pipeline(&PipelineDesc::new(ShaderSource::Binaries {
    vertex: vs_main(),
    fragment: fs_main(),
}))?;
```

**Shader I/O maps almost 1:1 to WGSL.** The vertex↔fragment varying interface
is a shared struct on both sides -- the same model WGSL uses, minus the manual
attribute annotations (Quanta assigns `@location` from field order):

| WGSL | Quanta shader DSL |
|------|-------------------|
| `struct VOut { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> }` | `#[derive(quanta::Varyings)] struct VOut { #[position] p: Vec4, uv: Vec2 }` |
| `@builtin(position)` output | the one `#[position]` field (a `Vec4`) |
| `@location(n)` varying | struct field `n` (field-declaration order) |
| integer varying + `@interpolate(flat)` | a `u32` field (flat applied automatically) |
| `fn vs(..) -> VOut` | `#[quanta::vertex] fn vs(..) -> VOut` |
| `fn fs(in: VOut) -> @location(0) vec4<f32>` | `#[quanta::fragment] fn fs(in: VOut) -> Vec4` |
| `@builtin(position)` read in the fragment | `frag_coord()`, or read the `#[position]` field |
| `@builtin(vertex_index)` | `vertex_id()` |
| `@builtin(instance_index)` | `instance_id()` |
| `textureSample(t, s, uv)` | `sample(t, in.uv)` (`t: &Texture2D` param) |

> The Quanta **WebGPU/WGSL backend does not yet emit** `frag_coord`, `u32`
> varyings, `vertex_id`/`instance_id`, or shader `for` loops -- a shader using
> them ships with `wgsl: None` and a build-time note (Metal and Vulkan still
> emit natively). This matches the `NotSupported` posture in [v0.1 advanced
> features](#v01-advanced-features) below.

**Coordinate convention — nothing to change.** Quanta uses WebGPU's
convention on every backend: NDC **+Y up**, framebuffer origin **top-left**,
texture coords **+Y down** (`v = 0` is the top row), readback **row 0 is the
top row**. Geometry, UVs, and projections you authored for wgpu carry over
unchanged — Quanta normalizes the Vulkan y-down default internally, so the
same shader source produces the same pixels on Metal, Vulkan, and WebGPU.

### Render pass: wgpu vs Quanta

wgpu render passes require manual encoder management:
```rust
let mut encoder = device.create_command_encoder(&Default::default());
{
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor { ... });
    pass.set_pipeline(&pipeline);
    pass.set_vertex_buffer(0, verts.slice(..));
    pass.draw(0..3, 0..1);
}
queue.submit(Some(encoder.finish()));
```

Quanta uses a builder chain:
```rust
let mut pulse = gpu.render(&target)?
    .clear(Color::BLACK)
    .pipeline(&pipeline)
    .vertices(0, &verts)
    .draw(3)
    .pulse()?;
pulse.wait()?;
```

### Presentation: wgpu vs Quanta

wgpu's surface loop maps directly onto Quanta's `Surface`. Native
present is real on Metal (`CAMetalLayer`) and Vulkan (`VkSwapchainKHR` —
X11 via `SurfaceTarget::Xlib`, plus a windowless `Headless`
target); query `gpu.supports_surface_present()` first (on Vulkan it is
gated on loader WSI support).

Like wgpu, Quanta takes a `raw-window-handle` window straight to a
surface: `SurfaceTarget::from_window(&window)` is the analog of
`instance.create_surface(window)`, mapping the same rwh 0.6 handles (the
crate is re-exported as `quanta::rwh`, so you add no dependency line). And
`surface.render_frame(|frame| …)` folds `get_current_texture` + render +
`present` into one call, self-healing the swapchain on resize — the
one-closure equivalent of wgpu's acquire/present cycle.

| wgpu | Quanta |
|------|--------|
| `instance.create_surface(window)` | `gpu.create_surface(&SurfaceTarget::from_window(&window)?, &config)` (feature `raw-window-handle`; or name the variant by hand — `SurfaceTarget::MetalLayer { .. }` / `Xlib { .. }`) |
| `surface.configure(&device, &config)` | `surface.configure(SurfaceConfig::new(w, h))` |
| `get_current_texture()` + render + `present()` | `surface.render_frame(f)` — acquire → render → present in one call, with resize self-heal |
| `surface.get_current_texture()` | `surface.acquire()` (the primitive `render_frame` builds on) |
| `SurfaceError::Outdated` | `QuantaErrorKind::SurfaceOutdated(_)` — reconfigure, retry |
| `SurfaceError::Lost` / suboptimal reconfigure | self-heals on Vulkan: a swapchain reported *suboptimal* finishes the frame and rebuilds on the next `acquire`, no error |
| `frame.texture` + `create_view()` | `frame.texture()` (render into it directly) |
| `frame.present()` | `frame.present()` |
| `PresentMode::{Fifo, Immediate, Mailbox}` | `PresentMode::{Fifo, Immediate, Mailbox}` |

A hard `VK_ERROR_OUT_OF_DATE_KHR` (like wgpu's `Outdated`) still surfaces
as `SurfaceOutdated` — reconfigure with the new extent and retry. See
[Presenting to the screen](../rendering/tutorials/presentation.md) for the
full frame loop.

Alternatively, export the rendered texture to an external compositor
with `texture.native_handle()` (zero-copy; Metal + Vulkan) and let it
own present.

## v0.1 advanced features

WebGPU intentionally exposes a narrower surface than Vulkan or Metal. Quanta
follows the spec where it can and returns `QuantaErrorKind::NotSupported` for
features WebGPU doesn't include, so the same code compiles on the desktop
backends and degrades gracefully on the web.

| WebGPU                                  | Quanta                                                             |
|-----------------------------------------|--------------------------------------------------------------------|
| `pass.drawIndirect(buffer, offset)`     | `render_pass.draw_indirect(&buffer, offset)`                       |
| `pass.drawIndexedIndirect(...)`         | `render_pass.draw_indexed_indirect(&buffer, offset, &indices)`     |
| `GPURenderBundle` + `executeBundles`    | `gpu.render_bundle(cap)` + `render_pass.execute_bundle(&b, count)` |
| `GPUQueue.copyBufferToBuffer`           | `gpu.async_copy_queue().copy_buffer(&dst, &src, n)`                |
| Single global queue                     | `gpu.queue_families()` reports one `Graphics` family               |
| (not in spec)                           | `gpu.acceleration_structure_blas(...)` -- `NotSupported` on web    |
| (not in spec)                           | `gpu.mesh_pipeline(...)` -- `NotSupported` on web                  |
| (not in spec)                           | `gpu.tessellation_pipeline(...)` -- `NotSupported` on web          |
| (not in spec)                           | `gpu.vrs_state()` -- `NotSupported` on web                         |
| (not in spec)                           | `gpu.sparse_texture(...)` -- `NotSupported` on web                 |

Always pair an advanced-feature call with the matching capability query
(`gpu.supports_ray_tracing()`, `supports_mesh_shaders()`,
`supports_tessellation()`, `supports_vrs()`, `supports_sparse_residency()`)
or branch on `QuantaErrorKind::NotSupported` to fall back gracefully on the
web target.

## When to stay with wgpu

- You need browser surface/swapchain management. Quanta's `Surface`
  presents natively on Metal and Vulkan, but the WebGPU backend's surface
  is a reserved `NotSupported` variant — there is no canvas presentation
  path yet.
- You are building a rendering engine that needs fine-grained control over every descriptor.
- You need the wgpu ecosystem (winit integration, egui backends, etc.).

## When to use Quanta

- GPU compute (the primary use case).
- You want cross-vendor without writing WGSL.
- You want type safety and Rust-native kernel authoring.
- You want build-time compilation (no runtime shader compile latency).
