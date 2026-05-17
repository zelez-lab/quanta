# quanta-prims

Block-cooperative GPU primitives for [Quanta](https://github.com/zelez-lab/quanta) kernels.

## What this crate is

A library of GPU primitives that user kernels call cooperatively
across a workgroup: reduce, scan, sort, plus the warp-shuffle
utilities they share. The shape mirrors CUB / rocPRIM / moderngpu
— with one important difference: Quanta's primitives run on
Metal, Vulkan, WebGPU, and the software CPU backend from the same
Rust source.

## Status

**v0.1.0-alpha.2** — initial scaffold. Tier 1 is being built
incrementally:

| Primitive                    | Status        |
| ---------------------------- | ------------- |
| Block reduce (sum, u32)      | first cut     |
| Block scan (sum, u32)        | planned       |
| Block radix sort (keys, u32) | planned       |
| Warp shuffle utilities       | via Quanta    |
| Block histogram              | Tier 2, later |
| Block top-k                  | Tier 2, later |

## Quick example

```rust,ignore
use quanta::*;
use quanta_prims::block_reduce_add_u32_kernel;

#[quanta::kernel(workgroup_size = [32, 1, 1])]
fn my_reduce(data: &[u32], out: &mut [u32]) {
    let i = quark_id();
    let block = nucleus_id();

    let value = data[i as usize];
    let block_sum = block_reduce_add_u32_kernel(value);

    if proton_id() == 0 {
        out[block as usize] = block_sum;
    }
}
```

## Why a "block-cooperative" library

Standalone GPU sort / scan / reduce as a top-level "process this
buffer" API has limited real users — by the time you have GPU
data and want it sorted, you're already inside a larger pipeline.
The valuable shape is *device functions your kernel can call
cooperatively*, mirroring `cub::BlockReduceT::Sum` or
`rocprim::block_scan`.

Each primitive in this crate ships three layers:

1. A `#[quanta::device]` device-callable function — the
   cooperative kernel-body fragment.
2. A reference single-thread CPU implementation in
   `quanta_prims::reference` — the correctness oracle for
   differential testing.
3. (Where useful) a top-level kernel wrapper that calls the
   device function for the common "reduce a whole buffer" case.

## Algorithm overview

**Block reduce** uses a two-stage pattern:

1. **Warp-level reduction** via the `reduce_add_u32` subgroup
   intrinsic — every lane contributes; warp leaders end up
   holding the warp sum.
2. **Cross-warp reduction** (planned) via workgroup-shared
   memory — warp leaders write their partial sum to shared, then
   the first warp re-reduces over those partials.

The result is the workgroup-wide sum, replicated in every lane
that participated.

## Related crates

- `quanta-tensor` — layout algebra. Substrate; future quanta-prims
  segmented variants will consume it.
- `quanta-rand` — counter-based RNG. Sibling primitive crate.
- `quanta-blas` (planned) — BLAS. Consumes quanta-prims for
  reductions inside reductions / matrix vector dots / row scans.

## License

MIT OR Apache-2.0.
