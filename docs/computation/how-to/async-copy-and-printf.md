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
| Vulkan | Real `vkCmdCopyBuffer` submission (same `VkQueue` today; a dedicated transfer family is the follow-up DMA path) |
| Metal | Dedicated `MTLCommandQueue` + blit encoder — overlaps main-queue compute |
| WebGPU | `NotSupported` from `gpu.async_copy_queue()` |
| CPU | `memcpy` on the host thread |

Check `gpu.supports_async_copy()` first when the backend is not known
at build time.

## GPU printf

To print a value from inside a kernel, call `gpu_print_u32` /
`gpu_print_i32` / `gpu_print_f32` in a `#[quanta::kernel(jit)]` body:

```rust
#[quanta::kernel(jit)]
fn probe(data: &[f32], out: &mut [f32]) {
    let i = quark_id();
    if i == 0u32 {
        gpu_print_f32(data[i]);
    }
    out[i] = data[i];
}
```

After the dispatch completes the driver drains the records to stderr
as `[quanta gpu_print] quark=<thread> = <value>` — identical output on
the CPU device, Metal and Vulkan (WebGPU refuses at validation). A
printing dispatch completes synchronously. Guard prints behind a
thread-index check: the record buffer holds ~5,400 prints per drain.

### Host ring: `PrintfBuffer`

`PrintfBuffer` is a capacity-bounded ring you record `u64` message IDs
into, then drain on the host (CPU device only). It's a debugging
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

| Backend | In-kernel `gpu_print_*` | Host ring |
|---|---|---|
| Vulkan / Metal | ✅ driver-drained record buffer (binding 30 / `buffer(30)`) | `NotSupported` |
| WebGPU | Refused at validation (no WGSL scheme yet) | `NotSupported` |
| CPU | Executor prints inline | ✅ |

## See also

- [Multi-queue](../../rendering/how-to/multi-queue.md) — the queue model these wrappers live on
- [Guide: Async copy and printf](../../computation/tutorials/async-copy-and-printf.md) — full reference
