# Multi-queue

Modern GPUs expose multiple hardware queues that execute work in parallel:
graphics for the main render path, compute for async culling or
post-processing, and transfer for overlapping memory copies.

## Inspect available queues

```rust
use quanta::*;

let gpu = quanta::init()?;
for fam in gpu.queue_families() {
    println!("{:?}: {} queue(s)", fam.queue_type, fam.count);
}
```

Default backends report at least one `Graphics` queue. On Vulkan you'll
often see distinct `Graphics`, `Compute`, and `Transfer` families;
WebGPU exposes a single global queue.

## Acquire a queue

```rust
let compute = gpu.queue(QueueType::Compute)?;
```

Returns `QuantaErrorKind::NotSupported` if the requested type isn't
available on this device.

| Variant | Use |
|---|---|
| `QueueType::Graphics` | Render passes + general compute |
| `QueueType::Compute`  | Compute-only async work |
| `QueueType::Transfer` | Memory copies (pair with `AsyncCopyQueue`) |

## Submit work

```rust
let mut wave = my_kernel(&gpu)?;
wave.bind(0, &input);

compute.submit(&wave, [1024, 1, 1])?;
```

Submitted work runs concurrently with anything on other queues.

## Cross-queue ordering

Use semaphores when one queue's work depends on another's:

```rust
let sem: u64 = /* obtain a semaphore handle */;

let graphics = gpu.queue(QueueType::Graphics)?;
let compute  = gpu.queue(QueueType::Compute)?;

// Render the G-buffer on graphics, then run lighting on compute.
graphics.submit(&gbuffer_wave, [w, h, 1])?;
graphics.signal(sem)?;

compute.wait(sem)?;
compute.submit(&lighting_wave, [w, h, 1])?;
```

| Method | Effect |
|---|---|
| `submit(&wave, groups)` | Dispatch on this queue |
| `signal(sem)` | Signal `sem` once prior work on this queue completes |
| `wait(sem)` | Block this queue's next submit on `sem` |

## Async-copy parallel to render

The transfer queue is where async uploads belong — it can copy in
parallel with the graphics queue rendering the previous frame:

```rust
let xfer = gpu.queue(QueueType::Transfer)?;
let async_copy = gpu.async_copy_queue()?;

// Frame N: render
graphics.submit(&render_wave, [w, h, 1])?;

// Frame N+1: upload data in parallel
async_copy.copy_buffer(&dst, &src, count)?;
```

## Dropping is safe

`Queue` is a `Drop`-safe handle. The underlying hardware queue lives as
long as the `Gpu`; dropping a `Queue` only releases Quanta's
bookkeeping.

## Backend notes

| Backend | Queues |
|---|---|
| Vulkan  | One `VkQueue` per matching family |
| Metal   | One `MTLCommandQueue` per family |
| WebGPU  | Single global queue (compute and graphics serialize) |

## See also

- [Async copy and printf](../../computation/how-to/async-copy-and-printf.md) — `AsyncCopyQueue` builds on the transfer queue
- [Indirect commands](indirect-commands.md) — record on one queue, replay on another
- [Guide: Multi-queue](../../rendering/tutorials/multi-queue.md) — full reference
