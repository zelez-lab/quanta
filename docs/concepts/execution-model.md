# Execution Model

## One kernel, many quarks

A kernel is a function that runs once per quark. Every quark executes the same code
but operates on different data. This is SIMT (Single Instruction, Multiple Threads).

```rust
#[quanta::kernel]
fn square(input: &[f32], output: &mut [f32]) {
    let i = quark_id();       // each quark gets a unique ID
    output[i] = input[i] * input[i];
}
```

When you dispatch 1000 quarks, this function runs 1000 times simultaneously — each
invocation sees a different value from `quark_id()`.

## Thread indexing

```
quark_id()    Global index (0..N). Unique across the entire dispatch.
local_id()    Index within the nucleus (0..group_size). Resets per group.
group_id()    Which group (nucleus) this quark belongs to.
group_size()  Number of quarks per group (typically 64-256).
```

Relationship: `quark_id() == group_id() * group_size() + local_id()`

## Dispatch

```rust
let wave = my_kernel(&gpu)?;
wave.bind(0, &input_field);
wave.bind(1, &output_field);

// Launch N quarks, hardware groups them into protons and nuclei
gpu.dispatch(&wave, 1_000_000)?;
```

`gpu.dispatch(&wave, N)` launches N quarks. The GPU:
1. Divides N quarks into groups (one group per nucleus).
2. Groups are divided into protons (32 or 64 quarks each).
3. Protons execute in lockstep on the hardware.

For multi-dimensional dispatches:

```rust
// 3D dispatch: groups_x * groups_y * groups_z workgroups
gpu.wave_dispatch(&wave, [64, 64, 1])?;
```

## Branching and divergence

Within a proton, all quarks execute in lockstep. When quarks take different branches,
both paths execute — quarks on the "wrong" path are masked (their results discarded).

```rust
#[quanta::kernel]
fn divergent(data: &[f32], out: &mut [f32]) {
    let i = quark_id();
    if data[i] > 0.0 {
        // Quarks where data[i] <= 0.0 are masked here
        out[i] = data[i] * 2.0;
    } else {
        // Quarks where data[i] > 0.0 are masked here
        out[i] = 0.0;
    }
}
```

```
Proton (32 quarks):
  data = [0.5, -1.0, 0.3, -0.2, ...]

  Step 1: execute "then" branch
    Quark 0: ACTIVE  (0.5 > 0.0)   -> out[0] = 1.0
    Quark 1: MASKED  (-1.0 <= 0.0)
    Quark 2: ACTIVE  (0.3 > 0.0)   -> out[2] = 0.6
    Quark 3: MASKED  (-0.2 <= 0.0)

  Step 2: execute "else" branch
    Quark 0: MASKED
    Quark 1: ACTIVE  -> out[1] = 0.0
    Quark 2: MASKED
    Quark 3: ACTIVE  -> out[3] = 0.0
```

Performance rule: divergence within a proton costs double (both paths run).
Divergence across protons is free (different protons take different paths independently).

## Synchronization

### Within a nucleus: `barrier()`

Waits for all quarks in the nucleus to reach this point.

```rust
#[quanta::shared] let cache: [f32; 256];
cache[local_id()] = data[quark_id()];
barrier();  // all quarks have written before any quark reads
let neighbor = cache[(local_id() + 1) % 256];
```

### Within a proton: wave operations

Quarks in the same proton can exchange data without shared memory:

```rust
let my_val = data[quark_id()];
let neighbor_val = wave_shuffle(my_val, 1);  // get value from lane+1
```

### CPU-GPU: `gpu.wait()`

```rust
let pulse = gpu.dispatch(&wave, n)?;
// ... do CPU work ...
gpu.wait(&mut pulse)?;  // block until GPU finishes
```

Or non-blocking:

```rust
if gpu.poll(&pulse) {
    // GPU finished
}
```

## Execution order

- Quarks within a proton: execute in lockstep (same instruction, same cycle).
- Protons within a nucleus: scheduled independently (no guaranteed order).
- Nuclei: fully independent (no guaranteed order).
- Dispatches: sequential unless using async compute queues.

The only ordering guarantees come from explicit synchronization:
`barrier()`, `gpu.wait()`, `gpu.barrier_texture()`, `gpu.barrier_buffer()`.
