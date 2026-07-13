# Migration from Metal / Vulkan

If you have written raw Metal or Vulkan, you know the ceremony. Quanta eliminates it
while maintaining the same performance (it generates the same API calls under the hood).

## What disappears

| Metal/Vulkan concept | Quanta equivalent |
|---------------------|-------------------|
| Descriptor sets / argument buffers | `wave.bind(slot, &field)` |
| Pipeline state objects | `#[quanta::kernel]` (automatic) |
| Shader compilation at runtime | Build-time (proc macro) |
| Platform `#ifdef` / `#if __METAL__` | One source, all backends |
| Command buffer/encoder management | Automatic submit on dispatch |
| Memory type selection (Vulkan) | Automatic (driver picks optimal) |
| `MTLLibrary` / `VkShaderModule` | Embedded in binary as `KernelBinary` |
| Fence/semaphore creation | `Pulse` returned from dispatch |
| Buffer/image layout transitions (Vulkan) | `gpu.barrier_texture()` / `gpu.barrier_field()` |
| `replaceRegion:` / staging + `vkCmdCopyBufferToImage` | `texture.write(&data)` / `texture.write_region(origin, size, &data)` |
| `waitUntilCompleted` / `vkDeviceWaitIdle` | `pulse.wait()` / `gpu.wait_idle()` |
| `[cmdBuf addCompletedHandler:]` / fence-poll thread | `pulse.on_complete(\|\| { .. })` — runs on a waiter thread, no caller park |

## Example: buffer creation + compute dispatch

### Metal (Objective-C, ~25 lines)

```objc
id<MTLDevice> device = MTLCreateSystemDefaultDevice();
id<MTLCommandQueue> queue = [device newCommandQueue];

// Create buffer
id<MTLBuffer> buf = [device newBufferWithBytes:data
                                        length:N * sizeof(float)
                                       options:MTLResourceStorageModeShared];

// Load shader library
NSError *error;
id<MTLLibrary> lib = [device newLibraryWithSource:@"..."
                                          options:nil
                                            error:&error];
id<MTLFunction> func = [lib newFunctionWithName:@"my_kernel"];

// Create pipeline
id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func
                                                                        error:&error];

// Encode + dispatch
id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
[enc setComputePipelineState:pso];
[enc setBuffer:buf offset:0 atIndex:0];
[enc dispatchThreads:MTLSizeMake(N, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
[enc endEncoding];
[cmdBuf commit];
[cmdBuf waitUntilCompleted];
```

### Vulkan (C, ~100+ lines)

```c
// Instance creation (15 lines)
// Physical device selection (10 lines)
// Logical device + queue creation (20 lines)
// Buffer creation + memory allocation (20 lines)
// Descriptor set layout + pool + set allocation (25 lines)
// Shader module creation (5 lines)
// Pipeline layout + compute pipeline (15 lines)
// Command pool + buffer allocation (10 lines)
// Record: bind pipeline, bind descriptor set, dispatch (8 lines)
// Submit + fence wait (10 lines)
// Cleanup (15 lines)
```

### Quanta (10 lines)

```rust
#[quanta::kernel]
fn my_kernel(data: &mut [f32]) {
    let i = quark_id();
    data[i] = data[i] * 2.0;
}

let gpu = quanta::init()?;
let field = gpu.field::<f32>(N)?;
field.write(&data)?;

let mut wave = my_kernel(&gpu)?;
wave.bind(0, &field);
let mut pulse = gpu.dispatch(&wave, N as u32)?;
pulse.wait()?;
```

## Performance

Quanta does not add overhead. Under the hood:

- **Metal**: `objc_msgSend` calls to the same Metal APIs you would call directly.
- **Vulkan**: the same `vkCmd*` functions through `extern "C"` FFI.

The shader is pre-compiled at build time (not interpreted at runtime), so pipeline
creation is faster than Metal's `newLibraryWithSource:` or Vulkan's
`vkCreateShaderModule` + `vkCreateComputePipelines`.

## What you keep

- Full GPU performance (no abstraction overhead).
- Explicit resource management (Fields are RAII, but you control lifetime).
- Multi-queue support (`gpu.queue_families()` to enumerate, `gpu.queue(QueueType::Compute)` for a dedicated queue).
- Resource state transitions (`gpu.barrier_texture()`, `gpu.barrier_field()`).
- Timestamp queries for profiling (`gpu.timestamp_query()`).
- Debug labels for GPU capture tools (`gpu.debug_push()` / `gpu.debug_pop()`).

## What you give up

- Per-call control over command buffer submission order.
- Custom memory allocators (Quanta picks optimal memory types automatically).
- Window creation (Quanta never creates windows — you hand it a
  presentation target). Presentation itself is covered: a `Surface`
  over a `CAMetalLayer` (Metal) or a `VkSwapchainKHR` (Vulkan — X11 via
  `SurfaceTarget::Xlib`, or a windowless `Headless` target),
  created through `gpu.create_surface`; or exporting the rendered texture
  to your own compositor via `texture.native_handle()` (Metal + Vulkan).

## Migrating incrementally

Quanta's `Field` exposes a raw `handle() -> u64` which is the underlying
`MTLBuffer` pointer or `VkBuffer` handle. You can use this for interop with
existing Metal/Vulkan code during migration:

```rust
let field = gpu.field::<f32>(1024)?;
let raw_handle = field.handle();
// Pass raw_handle to your existing Metal/Vulkan code
```

For textures, prefer the typed export: `texture.native_handle()`
returns `NativeTextureHandle::Metal { texture }` (an `id<MTLTexture>`
pointer) or `NativeTextureHandle::Vulkan { image, memory, vk_format,
layout }`. The handle is a borrow — valid while the `Texture` lives;
`retain` it natively if your code needs it longer.

## Platform-specific notes

### Metal developers
- No more `MTLLibrary` management. Kernels are pre-compiled metallib embedded in your binary.
- `StorageModeShared` vs `StorageModePrivate` is chosen automatically based on `FieldUsage`.
- Argument buffers are created internally — you just call `wave.bind()`.

### Vulkan developers
- No more `VkDescriptorSetLayout` / `VkDescriptorPool` / `VkDescriptorSet` dance.
- No more memory type enumeration — the driver picks `HOST_VISIBLE` or `DEVICE_LOCAL`.
- Pipeline barriers are still explicit (`gpu.barrier_texture()`) because Metal cannot
  infer the source stage. On Metal these are no-ops (automatic hazard tracking).

## v0.1 advanced features

Quanta wraps the modern advanced surface as typed handles. Each is gated by a
capability query and returns `QuantaErrorKind::NotSupported` on backends that
don't implement it. The render-side methods (`gpu.mesh_pipeline`,
`gpu.render_bundle`, …) come from the `RenderGpu` extension trait —
`use quanta::RenderGpu;` (or `use quanta::*;`).

| Vulkan / Metal extension                                  | Quanta                                                           |
|-----------------------------------------------------------|------------------------------------------------------------------|
| `VK_KHR_acceleration_structure` / `MTLAccelerationStructure` | `gpu.acceleration_structure_blas(&[GeometryDesc { .. }])`     |
| `VK_KHR_ray_tracing_pipeline` / Metal intersector tables  | `gpu.ray_tracing_pipeline(&RayTracingPipelineDesc { .. })`       |
| `vkCmdTraceRaysKHR` / `dispatchThreads` on intersector    | `pipeline.dispatch_rays(w, h)`                                   |
| `VK_EXT_mesh_shader` / `MTLMeshRenderPipelineDescriptor`  | `gpu.mesh_pipeline(MeshPipelineDesc { .. })`                     |
| `vkCmdDrawMeshTasksEXT` / `drawMeshThreadgroups:`         | `pipeline.dispatch([gx, gy, gz])`                                |
| Tessellation control / evaluation stages                  | `gpu.tessellation_pipeline(TessTopology::Triangle, control_pts)` |
| `VK_KHR_fragment_shading_rate` / `MTLRasterizationRateMap`| `gpu.vrs_state()` + `set_rate(ShadingRate::R2x2)`                |
| `VK_EXT_sparse_binding` / `MTLHeap`                       | `gpu.sparse_texture(&desc)` + `map_tile` / `unmap_tile`          |
| `vkQueueBindSparse`                                       | (transparent — `map_tile` does the bind)                          |
| Multi-queue (graphics / compute / transfer)               | `gpu.queue(QueueType::Compute)`, `gpu.queue_families()`          |
| Transfer queue + `vkCmdCopyBuffer` / `MTLBlitCommandEncoder`| `gpu.async_copy_queue().copy_buffer(&dst, &src, n)`             |
| Secondary command buffers / `MTLIndirectCommandBuffer`    | `gpu.render_bundle(cap)`, `gpu.indirect_command_buffer(cap)`     |
| `vkCmdDrawIndirect` / `drawPrimitives:indirectBuffer:`    | `render_pass.draw_indirect(&buffer, offset)`                     |
| `VK_EXT_debug_printf`                                     | `gpu.printf_buffer(cap)?.drain()?`                               |
| `CAMetalLayer` + `nextDrawable` / `VkSwapchainKHR` + `vkAcquireNextImageKHR` | `gpu.create_surface(&SurfaceTarget::MetalLayer { layer } /* or Xlib / Headless */, &config)` + `surface.acquire()` → `frame.present()` (native on Metal + Vulkan; a *suboptimal* Vulkan swapchain self-heals on the next acquire, hard `OUT_OF_DATE` → `SurfaceOutdated`) |
| `MTLTexture` / `VkImage` handed to external code           | `texture.native_handle()` → `NativeTextureHandle::{Metal, Vulkan}` |
| Fragment buffer table (`const device float4* [[buffer(n)]]` / fragment SSBO descriptor) | `table: &[Vec4]` shader param, indexed `table[i]`; bound with `.uniform(slot, &field)` at the declaration index shared with `&T` uniforms |
| RGBA8 read-write storage texture in compute (`access::read_write` / storage image) | `&mut Texture2D<u32>` kernel param (texels as packed `0xAABBGGRR` u32; Metal needs `MTLReadWriteTextureTier2`) |

The argument layout for indirect draws follows the Vulkan / Metal convention
exactly — see [Guide: Indirect commands](../rendering/tutorials/indirect-commands.md).

### Memory ordering note for Vulkan developers

Vulkan's `OpMemoryBarrier` accepts arbitrary semantics. Metal's `device atomic_*`
pointers accept *only* `memory_order_relaxed` — `xcrun metal` rejects anything
stronger. Quanta clamps device-atomic ordering to `Relaxed` at MSL emission and
leans on explicit `threadgroup_barrier` / device barriers for cross-queue
visibility. See [Guide: Atomics: Memory ordering](../computation/tutorials/atomics.md#memory-ordering).
