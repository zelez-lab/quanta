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
| Vulkan  | Real `vkCmdCopyBuffer` submission — on the same `VkQueue` today (one family at device creation); a dedicated transfer family is the follow-up DMA path |
| Metal   | Dedicated `MTLCommandQueue` + blit encoder — can overlap main-queue compute |
| WebGPU  | `NotSupported`                                     |
| CPU     | Serial `memcpy` on the host thread                 |

Check `gpu.supports_async_copy()` before creating the queue — WebGPU
returns `NotSupported`. See
[Multi-queue](../../rendering/tutorials/multi-queue.md) for the queue
model these copies sit on.

## GPU printf

Printing from inside a kernel is a two-part story: the `gpu_print_*`
intrinsics (the real thing, JIT-only), and the host-side `PrintfBuffer`
ring (a CPU-device transport). Both are debugging tools — not something
you ship in a release build.

### In-kernel printing: `gpu_print_*`

```rust
#[quanta::kernel(jit)]
fn probe(input: &[f32], output: &mut [f32]) {
    let i = quark_id();
    let v = input[i] * input[i];
    if i < 2u32 {
        gpu_print_f32(v);
        gpu_print_u32(i);
    }
    output[i] = v;
}
```

Each call records its value into a driver-owned debug buffer; after the
dispatch completes, the driver drains it to stderr:

```text
[quanta gpu_print] quark=0 = 1
[quanta gpu_print] quark=1 = 4
[quanta gpu_print] quark=0 = 0
[quanta gpu_print] quark=1 = 1
```

`quark` is the printing thread's global index; records appear in
completion order, not program order. A printing dispatch completes
**synchronously** — the driver waits so it can drain before returning.
The kernel must be `#[quanta::kernel(jit)]` (the macro rejects the
AOT form: the driver keys the buffer machinery off the JIT kernel
def). Guard prints behind a thread-index check as above — a full-grid
print overflows the record buffer (~5,400 records per drain; overflow
drops records, never corrupts them).

### Host ring: `PrintfBuffer`

`PrintfBuffer` is a capacity-bounded ring you record `u64` message IDs into,
then drain on the host.

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

| Backend | In-kernel `gpu_print_*`                          | Host ring |
|---------|--------------------------------------------------|-----------|
| Vulkan  | ✅ record buffer at descriptor binding 30, drained after the fence | `NotSupported` |
| Metal   | ✅ same scheme at `buffer(30)`, drained after the wait | `NotSupported` |
| WebGPU  | Refused at validation (no WGSL scheme yet)       | `NotSupported` |
| CPU     | Executor prints inline                            | ✅ |

The host-side ring (`printf_buffer` / `record` / `drain`) exists on the
CPU device only. In-kernel printing is real on the CPU device, Metal
and Vulkan; WebGPU refuses at validation rather than run the print as
a silent no-op.

## Next

- [Multi-queue](../../rendering/tutorials/multi-queue.md) -- the queue model these wrappers live on
- [Reference: Errors](../../reference/errors.md) -- `InvalidParam` vs `NotSupported`
