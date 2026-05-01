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
    let input = gpu.compute_field::<f32>(1024)?;    // 2. allocate
    let output = gpu.compute_field::<f32>(1024)?;
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
| `device.create_buffer(...)` | `gpu.compute_field::<T>(n)` |
| `device.create_buffer_init(...)` | `gpu.compute_field(n)` + `field.write(&data)` |
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
| `device.poll(Maintain::Wait)` | `pulse.wait()` |

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
#[quanta::vertex]
fn vs_main(pos: Vec3, mvp: &Mat4) -> Vec4 {
    mvp * Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

#[quanta::fragment]
fn fs_main(color: Vec4) -> Vec4 {
    color
}

let pipeline = gpu.pipeline(&PipelineDesc {
    vertex: vs_main(),
    fragment: fs_main(),
    ..Default::default()
})?;
```

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

- You need WebGPU-specific features (surface/swapchain management for browsers).
- You are building a rendering engine that needs fine-grained control over every descriptor.
- You need the wgpu ecosystem (winit integration, egui backends, etc.).

## When to use Quanta

- GPU compute (the primary use case).
- You want cross-vendor without writing WGSL.
- You want type safety and Rust-native kernel authoring.
- You want build-time compilation (no runtime shader compile latency).
