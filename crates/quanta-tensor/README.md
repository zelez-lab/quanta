# quanta-tensor

Layout algebra substrate for the Quanta math-crate program.

## What this crate is

A pure-Rust types-and-functions library: **no GPU runtime, no
proc-macro, no kernels**. Just two types and a small set of
composable operations:

- **[`Shape`](src/shape.rs)** — multi-dimensional extents
  (`Vec<usize>`, every extent ≥ 1).
- **[`Layout`](src/layout.rs)** — function-style index from a
  coordinate tuple to a flat-buffer offset. Carries a shape + a
  signed stride vector + a base offset, exposed only through the
  indexer `at(coord) -> offset` so downstream code never depends on
  the stride representation.
- **Composable ops** — `transpose`, `permute`, `slice`,
  `broadcast`. Each returns a new `Layout` without touching the
  underlying buffer.

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

## What's coming

| Phase | Crate            | Status                                |
| ----- | ---------------- | ------------------------------------- |
| 1     | `quanta-tensor`  | **this crate — runtime substrate**    |
| 1.b   | proof artifacts  | Lean theorems + Verus invariants      |
| 2     | `quanta-sort`    | radix sort + scan + reduce            |
| 3     | `quanta-blas`    | GEMM, GEMV, axpy                      |
| 4     | `quanta-fft`     | Stockham FFT                          |
| —     | `quanta-rand`    | already shipped (`v0.1.0-alpha.2`)    |

See `roadmap/081_companion_crates/README.md` for the full plan.

## Status

`v0.1.0-alpha.2` — Phase 1 scaffold + composable ops. API will
change before the first stable release. 31 unit + integration
tests passing.

## License

MIT OR Apache-2.0.
