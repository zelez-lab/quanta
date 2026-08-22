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
| Vulkan  | `NotSupported` — no transfer-queue path yet        |
| Metal   | `NotSupported` — no blit-encoder path yet          |
| WebGPU  | `NotSupported`                                     |
| CPU     | Serial `memcpy` on the host thread                 |

Today the typed wrapper is real on the CPU device only; every GPU backend
returns `NotSupported` from `gpu.async_copy_queue()`. The transfer-queue
designs above (Vulkan `VK_QUEUE_TRANSFER_BIT` + `vkCmdCopyBuffer`, Metal
`MTLBlitCommandEncoder`) are the intended lowerings, not shipped ones —
see [Multi-queue](../../rendering/tutorials/multi-queue.md) for the queue
model they will sit on.

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
| Vulkan  | `NotSupported` (host ring) / kernel `gpu_print` refused at validation |
| Metal   | `NotSupported` (host ring) / kernel `gpu_print` refused at validation |
| WebGPU  | `NotSupported` (host ring) / kernel `gpu_print` refused at validation |
| CPU     | Host ring buffer; in-kernel `DebugPrint` writes to stderr   |

The host-side ring (`printf_buffer` / `record` / `drain`) exists on the CPU
device only. The in-kernel `DebugPrint` op has no working GPU lowering
(SPIR-V and WGSL emit nothing; the MSL debug buffer is never bound), so the
validator refuses it for every GPU backend rather than let it run as a
silent no-op. `VK_EXT_debug_printf` and an MSL/WGSL debug-buffer scheme are
the intended lowerings.

## Next

- [Multi-queue](../../rendering/tutorials/multi-queue.md) -- the queue model these wrappers live on
- [Reference: Errors](../../reference/errors.md) -- `InvalidParam` vs `NotSupported`
