# Changelog

All notable changes to `quanta-prims` are recorded here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Tier 2 — `block_compact_u32_buffer`** — per-block stream
  compaction with explicit predicate array. 5 differential
  tests on Metal.
- **Tier 2 — `block_histogram_u32_buffer`** — per-block
  256-bucket histogram via shared-memory atomic increment.
  Metal-only today (WGSL/SPIR-V/software return NotSupported
  for shared atomics; tests skip on those). 4 differential
  tests.
- **Tier 2 — `block_top_k_u32_buffer`** — per-block top-K
  selection via inlined bitonic sort + conditional write.
  K is a runtime push-constant up to 256. 5 differential
  tests.
- **`reference::{compact_u32_blocks, histogram_u32_blocks,
  top_k_u32_blocks}`** — CPU oracle for each Tier-2 kernel,
  used by the differential tests.

### Changed

- **All block-reduce and block-scan device fns** gain a
  `core::hint::assert_unchecked(sub_size > 0)` after the
  `subgroup_size()` call. Lets LLVM elide the div-by-zero
  panic guard whose lowering shape currently misroutes the
  kernel epilogue in the wasm-route lowerer. The hint becomes
  redundant once the underlying redirect-chain handling for
  multi-frame `br_if N; br M` patterns is fixed.

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

### Examples

Six runnable examples in `examples/`:

- `block_sum`, `prefix_scan`, `bucket_sort` — per-primitive
  demos with println output and a CPU-reference correctness
  check.
- `cpu_oracle` — pure-Rust tour of every reference function
  (no GPU required).
- `bench_throughput` — sweep reduce/scan/sort over varying N;
  print median latency and M-elem/sec.
- `bench_vs_cpu` — head-to-head vs the single-thread CPU
  reference at a fixed N; honest framing including the
  "N-core CPU" parallel upper bound.

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

34 GPU-differential tests passing on real Metal (Apple M1 Pro):
- 9 reduce-family cases (add / min / max × u32 / i32 / f32).
- 6 scan cases (ramp, uniform, alternating sign, f32
  tolerance, first-output, last-output).
- 7 sort cases (descending, sorted, uniform, pseudo-random,
  ties, extreme values, multi-block independence).
- 5 cross-warp reduce stress cases.
- 3 reference-module unit tests.
- 4 doctests across `reference` and `lib.rs`.

Discovery note from the docs-sweep cycle: earlier session
commit messages claimed "all 34 pass on Metal" but the prims
dep only enabled `quanta/software`, so `quanta::init()`
returned `NoDevice` and the differential tests silently
returned early. Adding the `gpu-metal` / `gpu-vulkan`
convenience features surfaced real-backend execution. Two
substrate-adjacent bugs fell out:

1. `workgroup_size = [256, 1, 1]` attribute name was silently
   ignored — the correct form is `workgroup = [256]`. Fixed
   all 13 prims kernel decls.
2. The bitonic sort's `let want_smaller = ascending == i_am_lower`
   (bool equality) compiled through LLVM to a constant-true
   tautology in the unrolled Metal output, killing the
   compare-exchange body. Rewriting as a u32 bit comparison
   sidesteps the optimizer pathology.

Both fixes land in this release.

### Status

API may still change before the first stable release. Tier 2
work (block histogram, top-k, compact, segmented reduce/scan,
multi-bit LSD radix) is queued.

[Unreleased]: https://github.com/zelez-lab/quanta/compare/quanta-prims-v0.1.0-alpha.2...HEAD
[0.1.0-alpha.2]: https://github.com/zelez-lab/quanta/releases/tag/quanta-prims-v0.1.0-alpha.2
