# quanta-tensor

Layout algebra substrate for the Quanta math-crate program.

## What this crate is

A pure-Rust types-and-functions library: **no GPU runtime, no
proc-macro, no kernels**. Two types and two layers of ops:

- **[`Shape`](src/shape.rs)** — multi-dimensional extents
  (`Vec<usize>`, every extent ≥ 1).
- **[`Layout`](src/layout.rs)** — function-style index from a
  coordinate tuple to a flat-buffer offset. Carries a shape + a
  signed stride vector + a base offset, exposed only through the
  indexer `at(coord) -> offset` so downstream code never depends on
  the stride representation.
- **Local ops** — `transpose`, `permute`, `slice`, `broadcast`.
  Each returns a new `Layout` without touching the buffer; they
  transform a single layout per-axis.
- **Algebra** — `compose`, `complement`, `logical_divide`,
  `tiled_divide`. These are the load-bearing combinators of two
  layouts; downstream GEMM / sort / FFT tilings express their work
  as `layout.logical_divide(&tile)`.

## What this crate is *for*

Every downstream Quanta math crate (`quanta-sort`, `quanta-blas`,
`quanta-fft`, `quanta-rand`) has shape-correctness obligations:

- GEMM: `M×K @ K×N → M×N`.
- Sort: result is a length-preserving permutation of the input.
- FFT: power-of-2 length, conjugate-symmetric for real input.
- RNG: output length matches the requested count.

Without a substrate, each crate proves shapes from scratch. With
quanta-tensor, the shape proofs (composition associativity,
permutation bijectivity, tile-offset bounds) live in one place and
every downstream crate inherits them. This mirrors how CUTLASS's
CuTe library treats layouts as algebraic objects.

## Quick example

```rust
use quanta_tensor::Layout;

// 2×3×4 row-major tensor.
let src = Layout::row_major(&[2, 3, 4]).unwrap();

// Compose three ops: swap axes 1↔2, then permute, then slice.
let view = src
    .transpose(1, 2).unwrap()       // 2×4×3
    .permute(&[2, 0, 1]).unwrap()   // 3×2×4
    .slice(0, 1, 3).unwrap();       // 2×2×4 (starting at row 1)

// `view.at([a, b, c])` returns a flat offset into the original
// buffer. No data was copied.
assert_eq!(view.shape().dims(), &[2, 2, 4]);
let offset = view.at(&[0, 0, 0]).unwrap();
println!("first element at offset {}", offset);
```

## Tiling example (the load-bearing case)

The local ops compose into per-axis transformations. For GEMM /
sort / FFT tilings — splitting a tensor into block-and-warp shapes
— reach for `logical_divide`.

```rust
use quanta_tensor::Layout;

// A contiguous 72-element buffer.
let buffer = Layout::row_major(&[72]).unwrap();

// Divide by a block tile of 36 elements: 2 blocks total.
let block_tile = Layout::row_major(&[36]).unwrap();
let blocked = buffer.logical_divide(&block_tile).unwrap();
assert_eq!(blocked.shape().dims(), &[36, 2]);

// blocked.at([elem, block]) is the offset of element `elem` in
// block `block` against the original 72-element buffer.
assert_eq!(blocked.at(&[0, 0]).unwrap(), 0);
assert_eq!(blocked.at(&[35, 1]).unwrap(), 71);
```

## Design notes

- **Dynamic rank only.** Shapes and strides live in `Vec<usize>` /
  `Vec<isize>` at runtime, not in the type system. CuTe uses
  compile-time integer tuples to enable kernel-time specialisation;
  quanta-tensor deliberately doesn't, so the dynamic-shape paths
  every downstream math crate needs interop cleanly. Divisibility
  checks become runtime errors (`LayoutError::DivisibilityFailed`).
- **Public surface is the accessor pair.** Downstream code should
  use `.shape()` / `.strides()` (and `Shape::dims`) rather than the
  private struct fields. The internal representation may shift
  without breaking accessor callers.
- **Algebra over ops for tiling.** Local ops handle per-axis
  transformations; they don't compose into GEMM-style tile
  patterns on their own. Reach for `logical_divide` / `compose`
  whenever you'd otherwise be tempted to chain `slice`s by hand.

## Related crates

Downstream math crates depend on this substrate:

| Crate            | Status                                |
| ---------------- | ------------------------------------- |
| `quanta-tensor`  | **this crate — runtime substrate**    |
| `quanta-sort`    | planned — radix sort + scan + reduce  |
| `quanta-blas`    | planned — GEMM, GEMV, axpy            |
| `quanta-fft`     | planned — Stockham FFT                |
| `quanta-rand`    | shipped (`v0.1.0-alpha.2`)            |

## Status

`v0.1.0-alpha.2` — initial scaffold, local ops, layout algebra
(compose / complement / divide), and structural Lean / Verus
invariants. API will change before the first stable release. 43
unit + integration tests passing.

## License

MIT OR Apache-2.0.
