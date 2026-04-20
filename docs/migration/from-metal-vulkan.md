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
| Buffer/image layout transitions (Vulkan) | `gpu.barrier_texture()` / `gpu.barrier_buffer()` |

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
let field = gpu.compute_field::<f32>(N)?;
gpu.write_field(&field, &data)?;

let mut wave = my_kernel(&gpu)?;
wave.bind(0, &field);
let mut pulse = gpu.dispatch(&wave, N as u32)?;
gpu.wait(&mut pulse)?;
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
- Multi-queue support (`gpu.async_compute_dispatch()`).
- Resource state transitions (`gpu.barrier_texture()`, `gpu.barrier_buffer()`).
- Timestamp queries for profiling (`gpu.timestamp_query()`).
- Debug labels for GPU capture tools (`gpu.debug_push()` / `gpu.debug_pop()`).

## What you give up

- Per-call control over command buffer submission order.
- Custom memory allocators (Quanta picks optimal memory types automatically).
- Swapchain management (Quanta is compute/render-to-texture focused, not windowing).

## Migrating incrementally

Quanta's `Field` exposes a raw `handle() -> u64` which is the underlying
`MTLBuffer` pointer or `VkBuffer` handle. You can use this for interop with
existing Metal/Vulkan code during migration:

```rust
let field = gpu.compute_field::<f32>(1024)?;
let raw_handle = field.handle();
// Pass raw_handle to your existing Metal/Vulkan code
```

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
