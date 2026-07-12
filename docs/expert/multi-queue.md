# Expert: Multi-Queue

Modern GPUs have multiple hardware queues. Use them to place compute and
render work on separate queues for higher throughput.

## Query the queue families

Start by asking the device what queue families it exposes. `queue_families()`
never fails — it returns whatever the backend reports (a single `Graphics`
family on the software fallbacks and WebGPU):

```rust
use quanta::*;

let families = gpu.queue_families();
for fam in &families {
    println!("{:?}: {} queues", fam.queue_type, fam.count);
}
```

## Create a dedicated queue

`gpu.queue(QueueType)` allocates a typed [`Queue`](../reference/api.md) for a
capability tier. Backends that don't expose multi-queue (the single-queue
software fallbacks, WebGPU's global queue) return `NotSupported` here, so
branch on the result rather than assuming a second queue exists:

```rust
match gpu.queue(QueueType::Compute) {
    Ok(compute_queue) => {
        // Dispatch work on the dedicated compute queue.
        compute_queue.submit(&wave, [256, 1, 1])?;
    }
    Err(e) if matches!(e.kind, QuantaErrorKind::NotSupported(_)) => {
        // Single-queue backend — run the work on the main queue instead.
        gpu.dispatch(&wave, n)?.wait()?;
    }
    Err(e) => return Err(e),
}
```

## Async compute (capability not yet provided)

Quanta reserves a dedicated *async-compute* surface —
`gpu.supports_async_compute()` and `gpu.async_compute_dispatch(&wave, groups)`
— for overlapping compute on a background queue while the main queue renders.
**No backend provides it today**: `supports_async_compute()` returns `false`
everywhere and `async_compute_dispatch` returns `NotSupported`. Gate on the
query so code stays correct when a backend gains the capability:

```rust
if gpu.supports_async_compute() {
    let mut compute_pulse = gpu.async_compute_dispatch(&wave, [64, 1, 1])?;
    gpu.render(&target)?
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw(vertex_count)
        .pulse()?
        .wait()?;
    compute_pulse.wait()?;
} else {
    // Fall back to the main queue (the path taken on every backend today).
    gpu.dispatch(&wave, n)?.wait()?;
    gpu.render(&target)?
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw(vertex_count)
        .pulse()?
        .wait()?;
}
```

Until then, use `gpu.queue(QueueType::Compute)` for an explicit second queue
where the backend supports one.

## Resource synchronization across queues

When a resource is written on one queue and read on another, insert
barriers:

```rust
// After compute writes to a field on the async queue
gpu.barrier_field(&field, ResourceState::ComputeWrite, ResourceState::ShaderRead)?;

// Now safe to read in a render pass on the main queue
```

On Metal, hazard tracking handles this automatically. On Vulkan, the
barrier maps to a pipeline barrier with the correct queue family transfer.
