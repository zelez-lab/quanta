# Async copy and GPU printf

Two utility wrappers: one for overlapping uploads with rendering, one for
debugging from inside a kernel.

## Async copy

`AsyncCopyQueue` runs buffer copies on the dedicated transfer queue
(when the backend has one). Use it to overlap data uploads with rendering
or compute, or to keep your main queue uncluttered.

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

For raw `u64` handles (e.g. from FFI):

```rust
async_copy.copy_buffer_raw(dst_handle, src_handle, byte_count)?;
```

### Overlapping with render

The transfer queue runs concurrently with `Graphics` and `Compute` on
Vulkan and Metal — so the next frame's upload happens *while* this
frame is rendering:

```rust
let graphics = gpu.queue(QueueType::Graphics)?;
let async_copy = gpu.async_copy_queue()?;

// Kick off render of frame N
graphics.submit(&render_wave, [w, h, 1])?;

// Concurrent upload of frame N+1's data
async_copy.copy_buffer(&next_frame_dst, &next_frame_src, count)?;
```

### Backend notes

| Backend | Path |
|---|---|
| Vulkan  | Transfer queue (`VK_QUEUE_TRANSFER_BIT`) + `vkCmdCopyBuffer` |
| Metal   | `MTLCommandQueue` + `MTLBlitCommandEncoder` |
| WebGPU  | `GPUQueue.copyBufferToBuffer` (single queue, serialized) |

## GPU printf

`PrintfBuffer` is a capacity-bounded ring you record `u64` message IDs
into from inside a kernel, then drain on the host. It's a debugging
tool — not something you ship in a release build.

```rust
let printf = gpu.printf_buffer(/*capacity=*/256)?;

// After dispatching kernels that recorded into `printf`:
let drained: Vec<u64> = printf.drain()?;
for msg_id in drained {
    println!("kernel emitted message {msg_id}");
}
```

| Method | Effect |
|---|---|
| `record(msg_id)` | Append a `u64` message ID |
| `drain()` | Read out and clear all recorded messages |
| `capacity()` | The cap passed to `printf_buffer` |

`gpu.printf_buffer(0)` returns `InvalidParam` — capacity must be at
least 1.

The intent is to encode `printf!("kernel X iter {}", i)` calls as small
numeric IDs at compile time, drain them after each frame, and look the
IDs up in a side table.

### Backend notes

| Backend | Path |
|---|---|
| Vulkan  | `VK_EXT_debug_printf` + `debug_printfEXT` SPIR-V intrinsic |
| Metal   | `os_log` from MSL via Metal Debugger |
| WebGPU  | Software ring through a storage buffer |

## See also

- [Multi-queue](../../rendering/how-to/multi-queue.md) — the queue model these wrappers live on
- [Guide: Async copy and printf](../../computation/tutorials/async-copy-and-printf.md) — full reference
