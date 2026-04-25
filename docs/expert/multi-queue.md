# Expert: Multi-Queue

Modern GPUs have multiple hardware queues. Use them to overlap compute and
render work for higher throughput.

## Async compute

Run compute work on a dedicated queue while the main queue renders:

```rust
use quanta::*;

fn async_compute_example(gpu: &Gpu) -> Result<(), QuantaError> {
    if gpu.supports_async_compute() {
        // Dispatch compute on the async queue
        let mut compute_pulse = gpu.async_compute_dispatch(&wave, [64, 1, 1])?;

        // Render on the main queue simultaneously
        gpu.render(&target)?
            .pipeline(&pipeline)
            .vertices(0, &vb)
            .draw(vertex_count)
            .pulse()?
            .wait()?;

        // Wait for async compute to finish
        compute_pulse.wait()?;
    }

    Ok(())
}
```

## Explicit queue management

For full control, query available queue families and create dedicated queues:

```rust
let families = gpu.queue_families();
for fam in &families {
    println!("{:?}: {} queues", fam.queue_type, fam.count);
}

// Create a dedicated compute queue
let compute_queue = gpu.create_queue(QueueType::Compute)?;

// Dispatch work on the dedicated queue
gpu.queue_dispatch(compute_queue, &wave, [256, 1, 1])?;
```

## Compute-render overlap pattern

A typical frame that overlaps next frame's compute with current frame's
rendering:

```rust
// Frame N: kick off compute for NEXT frame's data
let compute_pulse = gpu.async_compute_dispatch(&particle_wave, [n / 256, 1, 1])?;

// Frame N: render CURRENT frame using previously computed data
gpu.render(&target)?
    .pipeline(&render_pipeline)
    .vertices(0, &current_particles)
    .draw(particle_count)
    .pulse()?
    .wait()?;

// Ensure compute is done before we use its output next frame
compute_pulse.wait()?;
```

## Resource synchronization across queues

When a resource is written on one queue and read on another, insert
barriers:

```rust
// After compute writes to a field on the async queue
gpu.barrier_buffer(&field, ResourceState::ComputeWrite, ResourceState::ShaderRead)?;

// Now safe to read in a render pass on the main queue
```

On Metal, hazard tracking handles this automatically. On Vulkan, the
barrier maps to a pipeline barrier with the correct queue family transfer.
