# Shared Memory

Shared memory is a small, fast scratchpad local to a workgroup. All quarks in
the same workgroup can read and write it. Quarks in different workgroups cannot
see each other's shared memory.

## Why shared memory

Global field accesses have high latency (hundreds of cycles). Shared memory is
on-chip SRAM -- 10-100x faster. Use it when multiple quarks in a workgroup need
the same data, or when quarks need to communicate intermediate results.

Typical sizes: 32 KB - 64 KB per nucleus.

## Declaring shared memory

Use the `#[quanta::shared]` attribute on a `let` binding inside a kernel:

```rust
#[quanta::kernel]
fn reduce_sum(input: &[f32], output: &mut [f32]) {
    #[quanta::shared] let local: [f32; 256];

    let lid = local_id();
    let gid = quark_id();

    // Load from global to shared
    local[lid] = input[gid];
    barrier();

    // Parallel reduction in shared memory
    let mut stride = 128;
    while stride > 0 {
        if lid < stride {
            local[lid] = local[lid] + local[lid + stride];
        }
        barrier();
        stride = stride / 2;
    }

    // First quark writes the block result
    if lid == 0 {
        output[group_id()] = local[0];
    }
}
```

The array size must be a compile-time constant. The type must be a scalar
(`f32`, `u32`, `i32`, etc.).

## Barriers

`barrier()` synchronizes all quarks in the workgroup. It guarantees:

1. All shared memory writes before the barrier are visible to all quarks after it.
2. No quark crosses the barrier until all quarks in the workgroup reach it.

Without barriers, shared memory reads may see stale data.

**Rule**: Every shared memory write that will be read by a different quark must
be followed by a `barrier()` before the read.

```rust
// WRONG: race condition
local[lid] = input[gid];
let neighbor = local[lid + 1];  // may read stale data

// CORRECT
local[lid] = input[gid];
barrier();
let neighbor = local[lid + 1];  // guaranteed fresh
```

## When to use shared memory

### Reduction

Sum, min, max over a large array. Each workgroup reduces its portion in shared
memory, writes one result to global. A second pass reduces the per-group results.

### Tiling (matrix multiply)

Load tiles of two matrices into shared memory. Each quark computes one output
element using the shared tiles. Reduces global memory traffic by the tile size
factor.

```rust
#[quanta::kernel]
fn matmul_tiled(a: &[f32], b: &[f32], c: &mut [f32], n: u32) {
    #[quanta::shared] let tile_a: [f32; 256]; // 16x16
    #[quanta::shared] let tile_b: [f32; 256]; // 16x16

    let row = group_id() / (n / 16) * 16 + local_id() / 16;
    let col = group_id() % (n / 16) * 16 + local_id() % 16;
    let lid = local_id();

    let mut sum = 0.0f32;
    let mut t = 0u32;
    while t < n {
        tile_a[lid] = a[row * n + t + lid % 16];
        tile_b[lid] = b[(t + lid / 16) * n + col];
        barrier();

        let mut k = 0u32;
        while k < 16 {
            sum = sum + tile_a[lid / 16 * 16 + k] * tile_b[k * 16 + lid % 16];
            k = k + 1;
        }
        barrier();
        t = t + 16;
    }

    c[row * n + col] = sum;
}
```

### Stencil patterns

Convolution, blur, edge detection. Load a tile plus its halo into shared memory.
Each quark reads its neighbors from the shared tile without redundant global loads.

## Limitations

- Size is fixed at compile time. Cannot allocate dynamically.
- Maximum size depends on hardware (query `gpu.caps()` for exact limits).
- All quarks in the workgroup must execute the same `barrier()`. Conditional
  barriers (inside divergent branches) are undefined behavior.
- Shared memory does not persist across dispatches.

## Next

- [Atomics](atomics.md) -- thread-safe operations across quarks
- [Wave intrinsics](wave-intrinsics.md) -- communication within a proton
