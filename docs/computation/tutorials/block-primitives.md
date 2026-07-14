# Block primitives (quanta::prims)

`quanta::prims` is the block-cooperative primitives module:
reduce, scan, sort, compact, histogram, and top-k that run *across the
threads of a workgroup*, the way `cub::BlockReduce` or
`rocprim::block_scan` do — except the same Rust source targets Metal,
Vulkan, WebGPU, and the software CPU backend.

```toml
[dependencies]
quanta = { version = "0.1", features = ["prims", "metal"] }  # or vulkan
```

The crate ships three layers per primitive:

1. a `#[quanta::device]` function you call from inside your own kernel,
2. a top-level `*_buffer` convenience kernel for the standalone case,
3. a CPU reference implementation in `quanta::prims::reference` — the
   correctness oracle its differential tests (and yours) compare against.

## Calling a primitive inside your kernel

The device functions consume a per-thread value and cooperate across the
workgroup. Block reduce needs a `[T; 32]` shared scratch array at slot 0
for its cross-warp stage:

```rust,ignore
use quanta::*;
use quanta::prims::block_reduce_add_u32_kernel;

#[quanta::kernel(workgroup_size = [256, 1, 1])]
fn my_reduce(data: &[u32], out: &mut [u32]) {
    #[quanta::shared] let scratch: [u32; 32];

    let i = quark_id();
    let lane = proton_id();
    if lane < 32u32 { scratch[lane] = 0u32; }
    barrier();

    let block_sum = block_reduce_add_u32_kernel(data[i as usize]);
    if lane == 0u32 {
        out[nucleus_id() as usize] = block_sum;
    }
}
```

The full device-fn family: `block_reduce_{add,min,max}_{u32,i32,f32}_kernel`
and `block_scan_add_{u32,i32,f32}_kernel` (inclusive prefix sum).

## The `*_buffer` convenience kernels

When you just need the operation standalone, every primitive has a
ready-made kernel. Dispatch with `quark_count = 256 * num_blocks`:

```rust,ignore
use quanta::prims::block_reduce_add_u32_buffer;

let mut wave = block_reduce_add_u32_buffer(&gpu)?;
wave.bind(0, &input);   // [u32; 256 * num_blocks]
wave.bind(1, &output);  // [u32; num_blocks] — one partial per block
gpu.dispatch(&wave, (256 * num_blocks) as u32)?.wait()?;
```

Tier 2 adds `block_compact_u32_buffer` (stream compaction),
`block_histogram_u32_buffer` (256-bucket histogram),
`block_top_k_u32_buffer` (runtime-`k` selection),
`block_segmented_{scan,reduce}_add_u32_buffer` (head-flag
segmented prefix sums), `block_sort_kv_u32_buffer` (unstable
key-value sort), `block_radix_sort_kv_u32_buffer` (stable
key-value sort), and `block_segmented_sort_u32_buffer` (stable
per-segment sort). All results are **per-block**: each
256-element block is processed independently.

## Device-wide one-liners

For "I have a slice and want the whole-buffer answer", the Tier-3
wrappers handle upload, padding, multi-pass orchestration, and readback:

```rust,ignore
use quanta::prims::{device_reduce_add_u32, device_sort_u32};

let total: u32 = device_reduce_add_u32(&gpu, &data)?;   // any length ≥ 1
let sorted: Vec<u32> = device_sort_u32(&gpu, &data)?;   // any length
```

Reduce runs multi-pass block reduces until one value remains; sort runs a
device-wide bitonic network, one launch per compare-exchange pass. Each
call round-trips host ↔ GPU — inside a larger pipeline, prefer the
`block_*` kernels and keep intermediates resident.

## Backend matrix

| Primitive                  | Metal | Vulkan | WebGPU | CPU |
|----------------------------|-------|--------|--------|-----|
| reduce / scan / sort / compact / top-k / segmented / device-wide | ✅ | ✅ | ✅ | ✅ |
| histogram (shared atomics) | ✅    | ✅*    | ✅*    | ❌ `NotSupported` |

\* SPIR-V / WGSL emission is validator-clean (`spirv-val`, `naga`);
on-device validation pends the lavapipe and browser lanes. The CPU
backend refuses: its shared memory is per-thread scratch, so
atomics alone can't make the kernel cooperative.

One Vulkan caveat: the `block_*` device functions and kernels use subgroup
arithmetic (`reduce_add_*` and friends), which needs the subgroup ARITHMETIC
feature class. Broadcom V3D (Raspberry Pi 5) advertises only
BASIC/VOTE/BALLOT and aborts at pipeline creation on those kernels. The
`device_wide` host wrappers handle this for you — they check
`gpu.supports_subgroups()` and fall back to a subgroup-free shared-memory
tree-reduce family on such devices (same dispatch contract, tree order
differs so f32 sums agree only within a few ULP). If you call the subgroup
`block_*` kernels directly, gate on `gpu.supports_subgroups()` yourself.

## Next

- [Shared Memory](shared-memory.md) — the `#[quanta::shared]` scratch the primitives rely on
- [Device Functions](device-functions.md) — how `#[quanta::device]` calls are inlined
- The crate's own `README.md`, `GETTING_STARTED.md`, `COOKBOOK.md`, and
  `PERFORMANCE.md` (in `crates/sci/quanta-prims/`) for recipes and perf numbers
