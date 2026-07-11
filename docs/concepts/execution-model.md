# How GPU Threads Execute

## The dispatch model

When you call `gpu.dispatch()`, the GPU launches a grid of quarks (threads).
These quarks are organized into a three-level hierarchy.

```
Dispatch: 1,048,576 quarks
|
+-- Grid (the entire dispatch)
     |
     +-- Workgroup 0            Workgroup 1            ...
     |   (256 quarks)           (256 quarks)
     |   |                      |
     |   +-- Proton 0           +-- Proton 0
     |   |   Quark 0..31        |   Quark 0..31
     |   +-- Proton 1           +-- Proton 1
     |   |   Quark 32..63       |   Quark 32..63
     |   +-- ...                +-- ...
     |   +-- Proton 7           +-- Proton 7
     |       Quark 224..255         Quark 224..255
     |
     +-- Workgroup 2            ...            Workgroup 4095
```

| Level     | Quanta name | CUDA name    | Size        | Shares memory? |
|-----------|-------------|--------------|-------------|----------------|
| Grid      | --          | Grid         | All quarks  | Global only    |
| Workgroup | --          | Thread block | 64-1024     | Shared memory  |
| Proton    | Proton      | Warp         | 32 or 64    | Lockstep exec  |
| Thread    | Quark       | Thread       | 1           | Registers      |

The workgroup maps to one nucleus (compute unit) on the hardware. All quarks
in a workgroup share the same fast shared memory and can synchronize with
`barrier()`.

## Thread indexing

Every quark can ask "which one am I?" using built-in functions.

```
quark_id()     Global index across the entire dispatch (0..N)
local_id()     Index within the workgroup (0..group_size)
group_id()     Which workgroup this quark belongs to
group_size()   Number of quarks per workgroup (typically 64-256)
```

The relationship: `quark_id() == group_id() * group_size() + local_id()`

```rust
#[quanta::kernel]
fn example(data: &mut [f32]) {
    let global = quark_id();    // 0..1048576 (unique across dispatch)
    let local  = local_id();    // 0..255 (unique within workgroup)
    let group  = group_id();    // 0..4095 (which workgroup)
    data[global] = (group * 1000 + local) as f32;
}
```

For 2D/3D problems (images, volumes), use multi-dimensional dispatch:

```rust
// 64x64 workgroups, each 16x16 quarks = 1,048,576 total quarks
gpu.wave_dispatch(&wave, [64, 64, 1])?;
```

## SIMD execution: protons run in lockstep

Within a proton (warp), all 32 quarks execute the same instruction at the same
time. This is SIMD -- Single Instruction, Multiple Data. The hardware has one
instruction decoder shared across 32 arithmetic units.

This is what makes GPUs fast: 32 multiplications happen with one instruction
fetch. But it also means branching works differently than on a CPU.

## Branch divergence

On a CPU, an `if/else` jumps to one branch. On a GPU proton, quarks that take
different branches cause divergence: both paths execute, and quarks on the
"wrong" path are masked (their results discarded).

```rust
#[quanta::kernel]
fn divergent(data: &[f32], out: &mut [f32]) {
    let i = quark_id();
    if data[i] > 0.0 {
        out[i] = data[i] * 2.0;    // path A
    } else {
        out[i] = 0.0;              // path B
    }
}
```

```
Proton with 4 quarks (simplified from 32):
  data = [0.5, -1.0, 0.3, -0.2]

  Step 1: execute path A (quarks where data[i] > 0.0)
    Quark 0: ACTIVE  -> out[0] = 1.0
    Quark 1: MASKED  (sits idle)
    Quark 2: ACTIVE  -> out[2] = 0.6
    Quark 3: MASKED  (sits idle)

  Step 2: execute path B (remaining quarks)
    Quark 0: MASKED
    Quark 1: ACTIVE  -> out[1] = 0.0
    Quark 2: MASKED
    Quark 3: ACTIVE  -> out[3] = 0.0
```

Cost: divergence within a proton runs both paths (2x time in the worst case).
Divergence across different protons is free -- each proton decides independently.

Practical rule: structure your data so quarks in the same proton take the same
branch. Sort by category, not by ID.

## Barriers: synchronizing a workgroup

`barrier()` makes every quark in the workgroup wait until all quarks reach the
same point. This is required when quarks write to shared memory and other quarks
need to read those writes.

```rust
#[quanta::kernel]
fn prefix_sum(data: &[f32], out: &mut [f32]) {
    #[quanta::shared] let cache: [f32; 256];

    cache[local_id()] = data[quark_id()];  // every quark writes
    barrier();                              // wait for all writes
    // now every quark can safely read any element in cache[]
    if local_id() > 0 {
        out[quark_id()] = cache[local_id()] + cache[local_id() - 1];
    }
}
```

Rules:
- `barrier()` must be reached by ALL quarks in the workgroup (never put it
  inside a conditional where some quarks skip it -- this is undefined behavior)
- Only needed for shared memory; global memory has no cross-quark ordering
  guarantees within a single dispatch

## Shared memory: fast per-workgroup cache

Each workgroup has ~48KB of fast memory, declared with `#[quanta::shared]`.
It is roughly 100x faster than global memory for random access.

```rust
#[quanta::shared] let tile: [f32; 256];
tile[local_id()] = data[quark_id()];   // load from slow global
barrier();
let val = tile[(local_id() + 1) % 256]; // read from fast shared
```

Shared memory is visible only within a workgroup. Quarks in other workgroups
cannot see it. It exists only for the lifetime of the dispatch.

See [Guide: Shared Memory](../computation/tutorials/shared-memory.md) for patterns like
tiled matrix multiply and reductions.

## Execution order guarantees

| Scope                  | Ordering                                       |
|------------------------|-------------------------------------------------|
| Quarks within a proton | Lockstep (same instruction, same cycle)         |
| Protons in a workgroup | No guaranteed order (use `barrier()` to sync)   |
| Workgroups             | No guaranteed order (fully independent)          |
| Dispatches on one queue| Sequential (same queue serializes)              |
| Dispatches on different queues | Concurrent (use semaphores to order)     |

The only way to enforce ordering is explicit synchronization: `barrier()` within
a workgroup, `pulse.wait()` between dispatches on the same queue, and
queue-level signal/wait semaphores between dispatches on different queues.

GPU-to-CPU visibility is a separate concern from queue ordering: dispatch and
render submissions are **asynchronous**, so the returned `Pulse` is still in
flight when you get it back. A CPU-side read (`field.read()`, `texture.read()`)
must `pulse.wait()` first — or call `gpu.wait_idle()`, which blocks until
everything submitted so far has completed. Event-driven runtimes that must not
block a thread use `pulse.on_complete(f)` instead: the callback fires from a
background waiter thread at completion. Presenting a surface frame needs no
wait; same-queue ordering covers it.

## Queues

Most desktop GPUs expose three hardware queue families that can run in
parallel:

| Family   | Typical use                                         |
|----------|-----------------------------------------------------|
| Graphics | Render passes, general compute                      |
| Compute  | Async compute (culling, post-process, simulation)   |
| Transfer | Memory copies overlapped with the other two queues  |

Quanta's `Gpu::queue_families()` reports what the active backend exposes;
`gpu.queue(QueueType::Compute)` returns a typed `Queue` you can `submit` on.
WebGPU has only a single global queue; multiple `Queue` handles there share
that queue. See [Guide: Multi-queue](../rendering/tutorials/multi-queue.md).

The mental model: a queue is a single timeline. Two submits on the same queue
happen in submit order. Two submits on different queues happen in arbitrary
order unless you stitch them with `signal` / `wait` on a shared semaphore.
