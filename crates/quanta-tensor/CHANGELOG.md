# Changelog

All notable changes to `quanta-tensor` are recorded here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0-alpha.2] — 2026-05-17

Initial public substrate for the Quanta math-crate program. Locked
public surface for the alpha cycle.

### Added

- **Types**
  - `Shape` — multi-dimensional extents (`Vec<usize>`, every extent ≥ 1).
  - `Layout` — function-style index from a coordinate tuple to a
    flat-buffer offset; carries a signed stride vector and a base
    offset.
  - `ShapeError`, `LayoutError` — categorised error variants for
    every fallible op.
- **`Layout` constructors**
  - `Layout::row_major(&[usize])` — C / NumPy default.
  - `Layout::column_major(&[usize])` — Fortran / BLAS default.
- **`Layout` accessors**
  - `shape()`, `strides()`, `base_offset()`, `rank()`,
    `linear_size()`, `at(&[usize])`.
- **Local ops** (returning a new layout, no buffer copy)
  - `transpose(d0, d1)` — swap two axes.
  - `permute(&[usize])` — general axis permutation.
  - `slice(axis, start, end)` — half-open clip on one axis.
  - `broadcast(&[usize])` — pad with size-1 leading axes,
    stride-0 along broadcast axes.
- **Layout algebra** (the load-bearing API for downstream tiling)
  - `compose(other)` — apply `self` to the output of `other`.
  - `complement(cosize)` — the layout filling the gaps within
    `cosize`. Rank-0, rank-1 (closed form), and rank ≥ 2
    (stride-sort fold ported from CUTLASS CuTe).
  - `logical_divide(tiler)` — partition `self` by `tiler` into
    tile-modes + residual-modes.
  - `tiled_divide(tiler)` — flat-tuple alias of `logical_divide`.

### Documentation

- `GETTING_STARTED.md` — 10-minute walkthrough from `cargo new`
  through all 16 public methods.
- `COOKBOOK.md` — 10 layout recipes covering row/column major,
  transpose, broadcast, GEMM block tiling (rank-1 + rank-2),
  iterated block-then-warp, identity compose, subtensor slice,
  NHWC↔NCHW.
- `tests/cookbook_examples.rs` — 17 integration tests that mirror
  every cookbook snippet, so the docs cannot silently drift from
  the API.
- Doctest examples on `complement`, `compose`, `logical_divide`.
- `docs/concepts/tensor-layout.md` in the main Quanta mdBook.

### Verification

- **Lean** (`specs/verify/lean/Quanta/Tensor/Layout.lean`) — 48
  theorems on the layout algebra. Includes:
  - Structural facts (linear size = product of dims, strides
    length = rank, indexer well-formedness, `dot` cons law).
  - `tile_offset_bound`: every coordinate produced by
    `logical_divide` lands inside the original linear size.
  - `permutation_bijective` via `List.Perm.map` (mathlib).
  - `compose_assoc` for the rank-1 case.
- **Verus** (`specs/verify/verus/quanta/tensor_invariants.rs`) —
  42 proof obligations, covering:
  - The same structural facts as the Lean side.
  - Rank-1 `complement_rank1` closed-form (six theorems).
  - Rank-N `complement_general` recursive model — length
    invariants on the stride-sort fold and the
    leading-size-1 cleanup, plus base-offset preservation.

### Status

API may still change before the first stable release. The
`v0.1.0` line targets stabilisation once at least one downstream
math crate (`quanta-sort` is up next) is shipping against this
substrate.

[Unreleased]: https://github.com/zelez-lab/quanta/compare/quanta-tensor-v0.1.0-alpha.2...HEAD
[0.1.0-alpha.2]: https://github.com/zelez-lab/quanta/releases/tag/quanta-tensor-v0.1.0-alpha.2
