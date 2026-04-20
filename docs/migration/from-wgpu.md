# Migration from wgpu

## API mapping

| wgpu | Quanta |
|------|--------|
| `wgpu::Instance::new(Backends::all())` | `quanta::init()` |
| `instance.request_adapter()` | `quanta::init()` (automatic) |
| `adapter.request_device()` | `quanta::init()` (automatic) |
| `device.create_buffer(...)` | `gpu.compute_field::<T>(n)` |
| `device.create_buffer_init(...)` | `gpu.compute_field(n)` + `gpu.write_field(&f, &data)` |
| `device.create_shader_module(wgsl)` | `#[quanta::kernel]` (compile-time) |
| `device.create_compute_pipeline(...)` | automatic (inside `#[quanta::kernel]`) |
| `device.create_bind_group_layout(...)` | not needed |
| `device.create_bind_group(...)` | `wave.bind(slot, &field)` |
| `device.create_pipeline_layout(...)` | not needed |
| `encoder.begin_compute_pass()` | not needed |
| `pass.set_pipeline(&pipeline)` | automatic |
| `pass.set_bind_group(0, &bg, &[])` | `wave.bind(slot, &field)` |
| `pass.dispatch_workgroups(x, y, z)` | `gpu.dispatch(&wave, n)` |
| `encoder.copy_buffer_to_buffer(...)` | `gpu.copy_field(&dst, &src)` |
| `queue.submit(...)` | automatic (dispatch submits) |
| `buffer.slice(..).map_async(...)` | `gpu.read_field(&field)` |
| `device.poll(Maintain::Wait)` | `gpu.wait(&mut pulse)` |

## Example: compute shader

### wgpu (70 lines)

```rust
let instance = wgpu::Instance::new(Backends::all());
let adapter = instance.request_adapter(&Default::default()).await.unwrap();
let (device, queue) = adapter.request_device(&Default::default(), None).await.unwrap();

// Shader (separate WGSL string)
let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: None,
    source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
});

// Bind group layout
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

// Pipeline layout
let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    bind_group_layouts: &[&bgl],
    push_constant_ranges: &[],
    label: None,
});

// Pipeline
let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    layout: Some(&pipeline_layout),
    module: &shader,
    entry_point: Some("main"),
    ..Default::default()
});

// Buffers + bind group
let input_buf = device.create_buffer_init(...);
let output_buf = device.create_buffer(...);
let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    layout: &bgl,
    entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding() },
    ],
    label: None,
});

// Dispatch
let mut encoder = device.create_command_encoder(&Default::default());
{
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(64, 1, 1);
}
queue.submit(Some(encoder.finish()));
```

### Quanta (15 lines)

```rust
#[quanta::kernel]
fn double(input: &[f32], output: &mut [f32]) {
    let i = quark_id();
    output[i] = input[i] * 2.0;
}

fn main() -> Result<(), quanta::QuantaError> {
    let gpu = quanta::init()?;
    let input = gpu.compute_field::<f32>(1024)?;
    let output = gpu.compute_field::<f32>(1024)?;
    gpu.write_field(&input, &data)?;

    let mut wave = double(&gpu)?;
    wave.bind(0, &input);
    wave.bind(1, &output);
    let mut pulse = gpu.dispatch(&wave, 1024)?;
    gpu.wait(&mut pulse)?;

    let result = gpu.read_field(&output)?;
    Ok(())
}
```

## Key differences

**No WGSL.** You write Rust. The proc macro compiles to WGSL/MSL/PTX/GCN at build time.
No separate shader files. No string literals. No runtime compilation.

**No pipeline/bind group ceremony.** wgpu requires you to manually describe every binding
layout, create bind groups, create pipeline layouts, then create pipelines. Quanta infers
all of this from the kernel function signature.

**No async adapter/device creation.** `quanta::init()` discovers and initializes synchronously.

**No command encoder.** Dispatch submits immediately. No manual encoder/pass management.

**Typed buffers.** `Field<f32>` vs wgpu's untyped `Buffer`. Read/write operations are
type-safe — no manual byte slicing.

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

## When to stay with wgpu

- You need WebGPU-specific features (surface/swapchain management for browsers).
- You are building a rendering engine that needs fine-grained control over every descriptor.
- You need the wgpu ecosystem (winit integration, egui backends, etc.).

## When to use Quanta

- GPU compute (the primary use case).
- You want cross-vendor without writing WGSL.
- You want type safety and Rust-native kernel authoring.
- You want build-time compilation (no runtime shader compile latency).
