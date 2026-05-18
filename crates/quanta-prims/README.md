# quanta-prims

Block-cooperative GPU primitives for
[Quanta](https://github.com/zelez-lab/quanta) kernels.

## What this crate is

A library of GPU primitives that user kernels call cooperatively
across a workgroup: reduce, scan, sort, plus the warp-shuffle
utilities they share. The shape mirrors CUB / rocPRIM /
moderngpu — with one important difference: Quanta's primitives
run on Metal, Vulkan, WebGPU, and the software CPU backend from
the same Rust source.

## Status

**v0.1.0-alpha.2** — Tier 1 shipped:

| Primitive                                  | Status      |
| ------------------------------------------ | ----------- |
| `block_reduce_add` × {u32, i32, f32}       | ✅ verified |
| `block_reduce_min` × {u32, i32, f32}       | ✅ verified |
| `block_reduce_max` × {u32, i32, f32}       | ✅ verified |
| `block_scan_add`   × {u32, i32, f32}       | ✅ verified |
| `block_radix_sort_u32` (bitonic, 256 keys) | ✅ verified |
| Block histogram                            | Tier 2      |
| Block top-k                                | Tier 2      |
| Block compact / partition                  | Tier 2      |
| Segmented reduce / scan                    | Tier 2      |

13 GPU kernels. 34 differential tests on Metal. 8 Lean
correctness theorems + 12 Verus operational invariants. See
[CHANGELOG.md](CHANGELOG.md) for the release history.

## Quick example — block reduce inside your own kernel

```rust,ignore
use quanta::*;
use quanta_prims::block_reduce_add_u32_kernel;

#[quanta::kernel(workgroup_size = [256, 1, 1])]
fn my_reduce(data: &[u32], out: &mut [u32]) {
    // The block_reduce_add_*_kernel device fn requires a
    // [u32; 32] shared scratch array at slot 0 (see
    // BLOCK_REDUCE_SCRATCH_SLOT).
    #[quanta::shared] let scratch: [u32; 32];

    let i = quark_id();
    let block = nucleus_id();
    let lane = proton_id();

    // Identity-init the unused warps' scratch slots.
    if lane < 32u32 { scratch[lane] = 0u32; }
    barrier();

    let value = data[i as usize];
    let block_sum = block_reduce_add_u32_kernel(value);

    if lane == 0u32 {
        out[block as usize] = block_sum;
    }
}
```

## Quick example — sort a buffer (top-level convenience kernel)

```rust,ignore
use quanta_prims::block_radix_sort_u32_buffer;

let gpu = quanta::init()?;
let n = 256;  // workgroup size; one block per dispatch
let input = gpu.field::<u32>(n)?;
let output = gpu.field::<u32>(n)?;
input.write(&unsorted_data)?;

let mut wave = block_radix_sort_u32_buffer(&gpu)?;
wave.bind(0, &input);
wave.bind(1, &output);
gpu.dispatch(&wave, n as u32)?.wait()?;
// `output` now holds the input sorted ascending.
```

## Why a "block-cooperative" library

Standalone GPU sort / scan / reduce as a top-level "process
this buffer" API has limited real users — by the time you have
GPU data and want it sorted, you're already inside a larger
pipeline. The valuable shape is *device functions your kernel
can call cooperatively*, mirroring `cub::BlockReduceT::Sum` or
`rocprim::block_scan`.

Each primitive in this crate ships three layers:

1. A `#[quanta::device]` device-callable function — the
   cooperative kernel-body fragment users splice into their
   own kernels.
2. A reference single-thread CPU implementation in
   `quanta_prims::reference` — the correctness oracle for
   differential testing.
3. A top-level kernel wrapper (`*_buffer`) for the common "do
   this op on a whole buffer" case.

## Algorithm overview

**Block reduce** uses a two-stage pattern:

1. **Warp-level reduction** via the `reduce_add_X` /
   `reduce_min_X` / `reduce_max_X` subgroup intrinsic. Every
   lane in a subgroup gets the warp-wide result.
2. **Cross-warp reduction** via workgroup-shared memory. Lane
   0 of each warp publishes its partial; warp 0 re-reduces over
   the partials. After the second warp-reduce, lane 0 of the
   workgroup holds the block-wide total.

Constraint: `workgroup_size ≤ subgroup_size²`. On Apple/NVIDIA
(subgroup_size = 32) that's 1024 lanes — comfortably above the
typical 256.

**Block scan** uses a three-stage variant:

1. **Warp scan** — `scan_add_X` gives each lane its warp-local
   inclusive prefix sum.
2. **Warp totals** — lane (sub_size − 1) of each warp publishes
   its total to scratch[warp_id]; warp 0 then runs
   `scan_add_exclusive_X` over the totals to produce per-warp
   prefix offsets.
3. **Apply prefix** — every lane adds `scratch[warp_id]` to its
   warp-local result.

**Block sort** ships as bitonic (not LSD radix) for v0.1:
36 compare-exchange stages over 256 keys; data-independent
access pattern (`partner = lane ^ k` for each stage). LSD radix
variants are queued for Tier 2 once the device-fn inliner
handles nested control flow.

## Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)** — 10-minute
  walkthrough from `cargo new` to your first cooperative
  primitive.
- **[COOKBOOK.md](COOKBOOK.md)** — recipe catalogue: per-block
  histograms, threshold filtering, GEMV reductions, …
- **[PERFORMANCE.md](PERFORMANCE.md)** — honest perf numbers
  on Apple M1 Pro, with reproducible bench commands.
- **[CHANGELOG.md](CHANGELOG.md)** — release history.

## Related crates

- `quanta-tensor` — layout algebra. Substrate; future
  quanta-prims segmented variants will consume it.
- `quanta-rand` — counter-based RNG. Sibling primitive crate.
- `quanta-blas` (planned) — BLAS. Consumes quanta-prims for
  reductions inside dot products and matrix-vector ops.
- `quanta-fft` (planned) — FFT. Consumes quanta-prims for
  normalisation reductions.

## License

MIT OR Apache-2.0.
