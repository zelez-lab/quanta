# Async copy and GPU printf

Two utility wrappers that don't fit the render or compute story but matter
during development.

## Async copy

`AsyncCopyQueue` runs buffer copies on the dedicated transfer queue (when the
backend has one). Use it to overlap data uploads with rendering or compute,
or to keep your main queue uncluttered.

```rust
use quanta::*;

let async_copy = gpu.async_copy_queue()?;
let dst = gpu.field::<f32>(N)?;
let src = gpu.field::<f32>(N)?;

async_copy.copy_buffer(&dst, &src, N)?;
```

`copy_buffer` is generic over the field element type:

```rust
async_copy.copy_buffer::<Particle>(&dst_particles, &src_particles, count)?;
```

For the raw-handle variant (when you have `u64` handles, e.g. from FFI):

```rust
async_copy.copy_buffer_raw(dst_handle, src_handle, byte_count)?;
```

### Backend matrix

| Backend | Implementation                                     |
|---------|----------------------------------------------------|
| Vulkan  | Transfer queue (`VK_QUEUE_TRANSFER_BIT`) + `vkCmdCopyBuffer` |
| Metal   | `MTLCommandQueue` + `MTLBlitCommandEncoder`        |
| WebGPU  | `GPUQueue.copyBufferToBuffer` (single queue)       |
| CPU     | Serial `memcpy`                                    |

The transfer queue may run concurrently with `Graphics` and `Compute` queues
on Vulkan and Metal — see [Multi-queue](../../rendering/tutorials/multi-queue.md) for the
synchronization model.

## GPU printf

`PrintfBuffer` is a capacity-bounded ring you record `u64` message IDs into
from inside a kernel, then drain on the host. It's a debugging tool — not
something you ship in a release build.

```rust
let printf = gpu.printf_buffer(/*capacity=*/256)?;

// After dispatching kernels that recorded into `printf`:
let drained: Vec<u64> = printf.drain()?;
for msg_id in drained {
    println!("kernel emitted message {msg_id}");
}
```

| Method            | Effect                                              |
|-------------------|-----------------------------------------------------|
| `record(msg_id)`  | Append a u64 message ID (called from host or shim)  |
| `drain()`         | Read out and clear all recorded messages            |
| `capacity()`      | The cap passed to `printf_buffer`                   |

`gpu.printf_buffer(0)` returns `InvalidParam` — capacity must be at least 1.

The intent is to encode `printf!("kernel X iter {}", i)` calls as small numeric
IDs at compile time, drain them after each frame, and look the IDs up in a
side table. The kernel-side recording API is still under design — for now,
`record(msg_id)` is callable from host code as a transport test.

### Backend matrix

| Backend | Implementation                                            |
|---------|-----------------------------------------------------------|
| Vulkan  | `VK_EXT_debug_printf` + `debug_printfEXT` SPIR-V intrinsic|
| Metal   | `os_log` from MSL via Metal Debugger                      |
| WebGPU  | Software ring through a storage buffer                    |
| CPU     | Software ring buffer                                      |

## Next

- [Multi-queue](../../rendering/tutorials/multi-queue.md) -- the queue model these wrappers live on
- [Reference: Errors](../../reference/errors.md) -- `InvalidParam` vs `NotSupported`
