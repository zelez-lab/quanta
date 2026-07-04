# How a GPU Works

## Why GPUs exist

A CPU has a few powerful cores. Each core can do anything: branch prediction,
speculative execution, out-of-order pipelines. A CPU core is a Swiss Army knife.

A GPU has thousands of simple cores. Each core does one thing: arithmetic. No
branch prediction, no speculation, no complex control logic. A GPU core is a
single blade -- but there are thousands of them.

```
CPU (4-16 cores)                  GPU (thousands of cores)
+--------+--------+              +--+--+--+--+--+--+--+--+
|  Core  |  Core  |              |  |  |  |  |  |  |  |  |
|  ~~~~  |  ~~~~  |              +--+--+--+--+--+--+--+--+
| branch | branch |              |  |  |  |  |  |  |  |  |
|predict |predict |              +--+--+--+--+--+--+--+--+
|  OoO   |  OoO   |              |  |  |  |  |  |  |  |  |
| cache  | cache  |              +--+--+--+--+--+--+--+--+
+--------+--------+              |  |  |  |  |  |  |  |  |
|  Core  |  Core  |              +--+--+--+--+--+--+--+--+
|  ~~~~  |  ~~~~  |              (each box = one simple core)
| branch | branch |
|predict |predict |
|  OoO   |  OoO   |              Same operation on 1000 elements:
| cache  | cache  |                CPU: 1 core loops 1000 times
+--------+--------+                GPU: 1000 cores do it once each
```

The tradeoff: one GPU core is slower than one CPU core. But when you have the
same operation applied to millions of elements, the GPU wins by raw parallelism.

## Physical structure

Quanta uses physics-inspired names for GPU hardware. Here is the hierarchy,
from largest to smallest.

```
GPU (the whole chip)
 +-- Nucleus 0  (compute unit / SM / CU)
 |    +-- Proton 0  (warp / wavefront)
 |    |    +-- Quark 0..31  (threads / lanes)
 |    +-- Proton 1
 |    |    +-- Quark 32..63
 |    +-- ...
 |    +-- [shared memory: ~48KB]
 +-- Nucleus 1
 |    +-- ...
 +-- ...
 +-- [global memory: 8-80GB]
```

### Quarks (threads)

The smallest unit. One quark runs your kernel function once, with its own
unique ID. If you dispatch 10,000 quarks, your function runs 10,000 times
in parallel. Each quark has its own registers (local variables).

### Protons (warps / wavefronts)

A group of 32 or 64 quarks that execute in lockstep -- the same instruction,
at the same time. This is SIMD (Single Instruction, Multiple Data). You never
create protons explicitly; the hardware groups quarks into protons for you.

Lockstep matters because of branching. If quark 0 takes the `if` branch and
quark 1 takes the `else` branch, both paths run sequentially. See
[Execution Model](execution-model.md) for details.

Quarks in a proton can also exchange data directly through
[wave intrinsics](../computation/tutorials/wave-intrinsics.md). Note that the
cross-lane *arithmetic* intrinsics (reduce/scan/shuffle) are a separate
hardware feature class from vote/ballot — some Vulkan devices (Broadcom V3D)
have only the latter. Query `gpu.supports_subgroups()` before relying on them.

### Nuclei (compute units / SMs / CUs)

A nucleus contains multiple protons and a block of fast shared memory (~48KB).
Quarks within the same nucleus can share data through this memory. Quarks in
different nuclei cannot.

## Quanta naming table

| Quanta     | NVIDIA        | AMD         | Apple   | What it is                       |
|------------|---------------|-------------|---------|----------------------------------|
| Quark      | Thread        | Lane        | Thread  | One execution of your function   |
| Proton     | Warp (32)     | Wave (64)   | SIMD group (32) | Quarks in lockstep        |
| Nucleus    | SM            | CU          | GPU core | Protons + shared memory         |
| GPU        | Device        | Device      | Device  | The whole chip                   |

## Real hardware numbers

| GPU              | Nuclei | Quarks/Proton | Protons/Nucleus | Total quarks |
|------------------|--------|---------------|-----------------|--------------|
| Apple M1 Pro     | 16     | 32            | 4               | 2,048        |
| Apple M4         | 10     | 32            | 4               | 1,280        |
| NVIDIA RTX 4090  | 128    | 32            | 4               | 16,384       |
| AMD RX 7900 XTX  | 96     | 64            | 2               | 12,288       |

You do not need to memorize these. Quanta queries them at runtime:

```rust
let gpu = quanta::init()?;
println!("{}: {} quarks", gpu.name(), gpu.total_quarks());
```

## When to use a GPU

**Good for GPUs:** the same operation on millions of independent elements.
Matrix math, image processing, physics simulation, machine learning.

**Bad for GPUs:** branchy control flow, sequential algorithms, small datasets.
A `for` loop with complex conditionals runs faster on a CPU.

Rule of thumb: if you can write it as `for each element, do the same thing`,
it probably belongs on the GPU.

## How this maps to Quanta code

```rust
#[quanta::kernel]
fn double(data: &mut [f32]) {
    let i = quark_id();           // unique ID: 0, 1, 2, ...
    data[i] = data[i] * 2.0;     // each quark handles one element
}

// Launch 1024 quarks -- the GPU maps them to protons and nuclei
gpu.dispatch(&wave, 1024)?;
```

See [Guide: Compute Basics](../computation/tutorials/compute-basics.md) for a full walkthrough.

## Key insight

One quark is slower than one CPU core. But you can run tens of thousands at
once. GPU programming is about expressing your problem so that thousands of
quarks can each do a small piece of it, independently, simultaneously.
