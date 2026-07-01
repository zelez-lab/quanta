# Multi-queue

Modern GPUs expose multiple hardware queues that can execute work in parallel:
a graphics queue for the main render path, a compute queue for async culling
or post-processing, and a transfer queue for memory copies that can overlap
with the others. Quanta surfaces this through the typed `Queue` wrapper.

## Inspecting available queues

```rust
use quanta::*;

let gpu = quanta::init()?;
for fam in gpu.queue_families() {
    println!("{:?}: {} queue(s)", fam.queue_type, fam.count);
}
```

Default backends report at least one `Graphics` queue. On Vulkan you'll often
see distinct `Graphics`, `Compute`, and `Transfer` families; on WebGPU there's
a single global queue (the API exposes only one).

## Acquiring a queue

```rust
let compute = gpu.queue(QueueType::Compute)?;
```

`QueueType` is a small enum:

| Variant              | Use                                                |
|----------------------|----------------------------------------------------|
| `QueueType::Graphics`| Render passes + general compute                    |
| `QueueType::Compute` | Compute-only async work                            |
| `QueueType::Transfer`| Memory copies (pair with `AsyncCopyQueue`)         |

If the requested type isn't available, you get `QuantaErrorKind::NotSupported`.

## Submitting work

```rust
let mut wave = my_kernel(&gpu)?;
wave.bind(0, &input);

compute.submit(&wave, [1024, 1, 1])?;
```

Submitted work runs concurrently with anything on other queues. Use semaphores
to enforce ordering:

```rust
let sem: u64 = /* obtain a semaphore handle */;

graphics.signal(sem)?;   // graphics queue signals when its work is done
compute.wait(sem)?;      // compute queue blocks until the signal arrives
compute.submit(&wave, [256, 1, 1])?;
```

| Method                | Effect                                                |
|-----------------------|-------------------------------------------------------|
| `submit(&wave, groups)`| Dispatch on this queue (compute/graphics)            |
| `signal(sem)`         | Signal `sem` once prior work on this queue completes  |
| `wait(sem)`           | Block this queue's next submit on `sem`               |

## Dropping is safe

`Queue` is a `Drop`-safe handle. The underlying hardware queue lives as long
as the `Gpu`; dropping a `Queue` only releases Quanta's bookkeeping.

## Backend matrix

| Backend | Queues                                              |
|---------|-----------------------------------------------------|
| Vulkan  | One `VkQueue` per matching family                   |
| Metal   | One `MTLCommandQueue` per family                    |
| WebGPU  | Single global queue (compute and graphics serialize)|
| CPU     | Software FIFO model                                 |

## See also

- [Async copy + printf](../../computation/tutorials/async-copy-and-printf.md) -- `AsyncCopyQueue` builds on
  the transfer queue
- [Indirect commands](indirect-commands.md) -- record on one queue, replay on another
- [Expert: Multi-queue](../../expert/multi-queue.md) -- semaphore patterns and pitfalls
