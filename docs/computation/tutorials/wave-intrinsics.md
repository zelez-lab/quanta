# Wave Intrinsics

A proton (warp/wave) is a group of quarks that execute in lockstep. On NVIDIA,
a proton is 32 quarks. On AMD, 64 quarks. On Apple, 32 quarks.

Wave intrinsics let quarks within a proton exchange data and evaluate conditions
collectively -- without shared memory or atomics. They compile to single hardware
instructions.

The intrinsics live in the kernel-only `quanta` import namespace — the
`#[quanta::kernel]` macro injects their declarations, and calls are wrapped in
`unsafe` (they are `extern "C"` imports from the kernel's point of view). Two
lane-identity helpers come with them: `subgroup_size()` (proton width) and
`subgroup_id()` (this quark's lane within its proton).

## shuffle

Exchange values across lanes with an XOR (butterfly) pattern:

```rust
#[quanta::kernel]
fn shuffle_example(data: &mut [f32]) {
    let i = quark_id();
    let my_val = data[i as usize];

    // Read the value held by lane (my_lane ^ 1) — adjacent pairs swap
    let neighbor = unsafe { shuffle_f32(my_val, 1u32) };

    data[i as usize] = my_val + neighbor;
}
```

`shuffle_u32` / `shuffle_i32` / `shuffle_f32(value, lane_mask)` return the
`value` held by lane `my_lane ^ lane_mask` — the standard butterfly pattern
used by tree reductions (`lane_mask = 1` swaps adjacent lanes, `2` swaps pairs
of pairs, and so on). It mirrors Metal's `simd_shuffle_xor` and WGSL's
`subgroupShuffleXor`.

No memory access occurs. Data moves through register lanes.

## ballot_u32

Ask "which quarks in my proton satisfy a condition?"

```rust
#[quanta::kernel]
fn ballot_example(data: &[f32], masks: &mut [u32]) {
    let i = quark_id();
    let active = if data[i as usize] > 0.0 { 1u32 } else { 0u32 };

    // Returns a bitmask: bit N is set if lane N passed a non-zero predicate
    let mask = unsafe { ballot_u32(active) };

    masks[i as usize] = mask;
}
```

`ballot_u32(predicate)` takes a `u32` predicate (any non-zero value counts as
true) and returns a bitmask where bit N is set if lane N's predicate was
non-zero.

## any_u32 / all_u32

Collective boolean tests:

```rust
#[quanta::kernel]
fn early_exit(data: &[f32], output: &mut [f32]) {
    let i = quark_id();
    let val = data[i as usize];

    // 1 for every lane if ANY lane's predicate is non-zero
    let big = if val > 100.0 { 1u32 } else { 0u32 };
    if unsafe { any_u32(big) } != 0u32 {
        // At least one quark found a large value
        output[i as usize] = val;
    }

    // 1 for every lane only if ALL lanes' predicates are non-zero
    let nonneg = if val >= 0.0 { 1u32 } else { 0u32 };
    if unsafe { all_u32(nonneg) } != 0u32 {
        // Every quark in the proton has non-negative data
        output[i as usize] = val;
    }
}
```

- `any_u32(pred)` -- returns `1` for **all** quarks if at least one quark
  passed a non-zero `pred`, else `0`.
- `all_u32(pred)` -- returns `1` for **all** quarks only if every quark
  passed a non-zero `pred`, else `0`.

These are uniform within a proton: every quark gets the same answer.

## Reductions and scans

The most common cross-lane patterns ship as single intrinsics — no manual
shuffle tree needed:

- `reduce_add_{u32,i32,f32}(value)` — every lane gets the proton-wide sum.
- `reduce_min_*` / `reduce_max_*` — proton-wide minimum / maximum.
- `scan_add_{u32,i32,f32}(value)` — inclusive prefix sum (lanes `0..=self`).
- `scan_add_exclusive_*` — exclusive prefix sum (lane 0 receives 0).

```rust
#[quanta::kernel]
fn warp_reduce(input: &[f32], output: &mut [f32]) {
    let i = quark_id();
    let lane = unsafe { subgroup_id() };
    let width = unsafe { subgroup_size() };
    let val = unsafe { reduce_add_f32(input[i as usize]) };

    // Lane 0 of each proton writes the proton's sum
    if lane == 0u32 {
        output[(i / width) as usize] = val;
    }
}
```

This is faster than a shared-memory reduction because:
- No memory writes/reads (registers only)
- No barriers needed
- Compiles to a single hardware instruction sequence

The downside: it only works within a single proton (32 or 64 quarks). For larger
reductions, combine with [shared memory](shared-memory.md):
1. Warp-reduce within each proton (wave intrinsics)
2. Write proton results to shared memory
3. Final reduction in shared memory (one proton's worth of data)

`quanta::prims` packages exactly this recipe as
[block primitives](block-primitives.md).

## Backend support

Not every device implements the whole family. The vote/ballot group and the
arithmetic group are separate hardware feature classes, and they diverge on
real devices:

| Intrinsics | Requirement | Where it holds |
|------------|-------------|----------------|
| `any_u32` / `all_u32` / `ballot_u32` | SPIR-V `GroupNonUniformVote` / `GroupNonUniformBallot` (emitted automatically) | CPU, Metal, and every Vulkan device Quanta targets — **including Broadcom V3D** (Raspberry Pi 5) |
| `reduce_*` / `scan_add_*` | Subgroup **ARITHMETIC** feature class | CPU, Metal, llvmpipe, desktop Vulkan. **Not V3D** — its Mesa driver advertises only BASIC/VOTE/BALLOT and aborts at pipeline creation |
| `shuffle_*` | Subgroup **SHUFFLE** feature class | Same as arithmetic: everywhere except V3D |

Query `gpu.supports_subgroups()` before building a wave that uses the
arithmetic or shuffle intrinsics — it reports whether the device advertises
the subgroup ARITHMETIC class (always true on Metal and the CPU backend).
Kernels with a subgroup-free fallback (shared memory + barriers) should select
on it at dispatch-build time; `quanta::prims` does exactly that for its
device-wide reductions.

On WebGPU, all of these lower to the WGSL `subgroup*` builtins behind an
`enable subgroups;` directive — availability depends on the browser exposing
the `subgroups` feature.

## When to use wave intrinsics

| Pattern             | Intrinsic     | Why                                    |
|---------------------|---------------|----------------------------------------|
| Small reduction     | `reduce_add_*`| Fastest possible sum/min/max for <=64 elements |
| Prefix sum (scan)   | `scan_add_*`  | Single-instruction inclusive/exclusive scan |
| Early termination   | `all_u32`     | Skip work when all lanes are done      |
| Sparse processing   | `ballot_u32`  | Count active lanes, compute offsets    |
| Butterfly exchange  | `shuffle_*`   | Pairwise swap steps for custom tree patterns |

## Hardware differences

| GPU       | Proton width | Implication                          |
|-----------|-------------|---------------------------------------|
| NVIDIA    | 32          | 5 butterfly steps for a manual reduction |
| AMD       | 64          | 6 butterfly steps for a manual reduction |
| Apple     | 32          | Same as NVIDIA                        |

Write kernels that work for both widths. Use `subgroup_size()` at runtime
rather than hard-coding 32.

## Next

- [Block Primitives](block-primitives.md) -- the packaged block-wide reduce/scan family
- [Textures](../../rendering/tutorials/textures.md) -- image data and sampling
- [Rendering](../../rendering/tutorials/rendering.md) -- the graphics pipeline
