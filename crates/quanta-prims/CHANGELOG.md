# Changelog

All notable changes to `quanta-prims` are recorded here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0-alpha.2] — 2026-05-18

Initial Tier-1 cut: block-cooperative reduce / scan / sort
across the portable `{u32, i32, f32}` type set.

### Added

- **`block_reduce_add_{u32, i32, f32}_kernel`** — block-wide
  sum reduction; result in lane 0. Two-stage warp/cross-warp
  algorithm, supports workgroup_size up to 1024 on Apple/NVIDIA
  (subgroup_size = 32) and 4096 on AMD (subgroup_size = 64).
- **`block_reduce_min_{u32, i32, f32}_kernel`** — block-wide
  min reduction. f32 variant uses a large finite sentinel
  (`1e38`) as the identity element; otherwise type::MAX.
- **`block_reduce_max_{u32, i32, f32}_kernel`** — block-wide
  max reduction. f32 variant uses `-1e38` as the identity.
- **`block_scan_add_{u32, i32, f32}_kernel`** — inclusive
  prefix-sum scan. Three-stage warp / warp-totals / apply-prefix.
- **`block_radix_sort_u32_buffer`** — block-cooperative sort of
  256 u32 keys per workgroup. Bitonic sort with the standard
  XOR-partner compare-exchange. Named `radix_sort` for forward
  API compatibility; the algorithm choice will become an
  internal detail in a later release.
- **Top-level convenience kernels** for every primitive:
  `*_buffer` reads N inputs and writes one (reduce) or N
  (scan/sort) outputs per block.
- **`reference` module** — pure-Rust single-thread oracle
  implementations of every primitive. Available without the
  `gpu` feature.
- **`BLOCK_REDUCE_SCRATCH_SLOT`** constant — documents the
  shared-memory slot every cooperative primitive uses for
  cross-warp scratch.

### Substrate additions (in the parent `quanta` crate)

- 14 new subgroup intrinsics declared in `src/intrinsics.rs`:
  `reduce_add_{i32, f32}`, `reduce_min_*`, `reduce_max_*`,
  `scan_add_{i32, f32}`, `scan_add_exclusive_*`,
  `shuffle_{i32, f32}`. All wired through the WASM-route
  lowering. u64 / i64 / f64 deliberately excluded because
  Metal's simdgroup instructions don't support 64-bit and
  WGSL's subgroupAdd family is 32-bit/16-bit only.

### Documentation

- `README.md` — crate overview + status + quick examples.
- `GETTING_STARTED.md` — 10-minute walkthrough from `cargo new`
  through every primitive.
- `COOKBOOK.md` — recipe catalogue covering dot product,
  max-magnitude, histogram, compaction, block-local sort.

### Verification

- **Lean** (`specs/verify/lean/Quanta/Prims/Reference.lean`) —
  8 theorems on the reference implementations:
  - `reduceAdd_eq_sum`: reduce equals `List.sum`.
  - `reduceAdd_perm`: reduce is permutation-invariant — the
    load-bearing theorem for parallel-order-independence.
  - `scanAdd_length`: scan preserves length.
  - `sortAsc_perm`: sort produces a permutation of the input.
  - `sortAsc_sum`: sort preserves the input's sum.
  - Plus structural identities (`*_nil` for each primitive).
- **Verus** (`specs/verify/verus/quanta/prims_invariants.rs`)
  — 12 operational invariants on the same reference impls:
  length preservation, empty-input handling, nat return for
  reduce.

### Tests

34 GPU-differential tests against the reference implementations:
- 9 reduce-family cases (add / min / max × u32 / i32 / f32).
- 6 scan cases (ramp, uniform, alternating sign, f32
  tolerance, first-output, last-output).
- 7 sort cases (descending, sorted, uniform, pseudo-random,
  ties, extreme values, multi-block independence).
- 5 cross-warp reduce stress cases.
- 3 reference-module unit tests.
- 4 doctests across `reference` and `lib.rs`.

All tests pass on Metal. CPU-software-backend coverage falls
out for free since every kernel routes through the same
WASM-route IR.

### Status

API may still change before the first stable release. Tier 2
work (block histogram, top-k, compact, segmented reduce/scan,
multi-bit LSD radix) is queued.

[Unreleased]: https://github.com/zelez-lab/quanta/compare/quanta-prims-v0.1.0-alpha.2...HEAD
[0.1.0-alpha.2]: https://github.com/zelez-lab/quanta/releases/tag/quanta-prims-v0.1.0-alpha.2
