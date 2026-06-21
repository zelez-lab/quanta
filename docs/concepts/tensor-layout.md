# Tensor Layout

`quanta-tensor` is the **shape-correctness substrate** for Quanta's
math-crate program. It ships pure-Rust types and functions: no GPU
runtime, no proc-macro, no kernels. Downstream math crates
(`quanta-prims` and `quanta-rand` today; `quanta-blas`, `quanta-fft`
to come) depend on it and inherit its shape proofs.

## Why a separate crate

GPU compute kernels constantly reason about how multi-dimensional
data maps to flat memory:

- A matrix-multiply needs strides for the LHS, the RHS, and the
  output, and the strides have to compose correctly under tiling.
- A sort kernel needs to prove that its output is a length-
  preserving permutation of its input.
- An FFT kernel needs the bijection between time-domain and
  frequency-domain indices.

If every kernel author rolls their own shape arithmetic, each
crate carries its own bugs. CUTLASS solved this once for NVIDIA's
ecosystem with **CuTe** — layouts as algebraic objects with
associative composition, bijective permutations, and provable
tile-offset bounds. `quanta-tensor` is the Quanta equivalent,
sitting beneath every math companion crate.

## The two core types

### `Shape`

A `Shape` is an ordered list of axis extents. Each extent is at
least 1. A rank-0 shape (empty extent list) represents a scalar.

```rust
use quanta_tensor::Shape;

let s = Shape::new(&[2, 3, 4]).unwrap();
assert_eq!(s.rank(), 3);
assert_eq!(s.linear_size(), 24);
```

### `Layout`

A `Layout` pairs a shape with a stride vector and a base offset.
It exposes a function-style indexer `at(coord) -> offset` and
hides the strides behind that function — so downstream code never
takes a hard dependency on the stride representation.

```rust
use quanta_tensor::Layout;

let row_major = Layout::row_major(&[2, 3]).unwrap();
//   at([0,0]) = 0,  at([0,1]) = 1,  at([0,2]) = 2
//   at([1,0]) = 3,  at([1,1]) = 4,  at([1,2]) = 5

let col_major = Layout::column_major(&[2, 3]).unwrap();
//   at([0,0]) = 0,  at([1,0]) = 1,  at([0,1]) = 2
//   at([1,1]) = 3,  at([0,2]) = 4,  at([1,2]) = 5
```

## The four composable ops

Each returns a new `Layout` without touching data.

### `transpose(d0, d1)`

Swap two axes. Both the shape and the strides exchange positions.

```rust
# use quanta_tensor::Layout;
let l = Layout::row_major(&[2, 3]).unwrap();
let t = l.transpose(0, 1).unwrap();
assert_eq!(t.shape().dims(), &[3, 2]);
// t.at([i, j]) == l.at([j, i])
```

### `permute(perm)`

General axis permutation. `perm[i] = j` means new axis `i` is old
axis `j`. The permutation must use each index in `0..rank`
exactly once.

```rust
# use quanta_tensor::Layout;
let l = Layout::row_major(&[2, 3, 4]).unwrap();
// Reverse axes: (i, j, k) -> (k, j, i).
let p = l.permute(&[2, 1, 0]).unwrap();
assert_eq!(p.shape().dims(), &[4, 3, 2]);
```

### `slice(axis, start, end)`

Clip one axis to a half-open `[start, end)` range. Shape on that
axis becomes `end - start`; base offset advances by
`start * stride[axis]`. Other strides are unchanged.

```rust
# use quanta_tensor::Layout;
let l = Layout::row_major(&[4, 3]).unwrap();
let s = l.slice(0, 1, 3).unwrap();      // rows 1, 2
assert_eq!(s.shape().dims(), &[2, 3]);
// s.at([i, j]) == l.at([i + 1, j])
```

### `broadcast(target_shape)`

Pad `self` with size-1 leading axes to match the target rank, then
zero the stride on any axis where `self` has extent 1 and the
target wants extent N. The result reuses the same source elements
for every coordinate along the broadcast axis.

```rust
# use quanta_tensor::Layout;
let l = Layout::row_major(&[1, 3]).unwrap();
let b = l.broadcast(&[4, 3]).unwrap();
assert_eq!(b.shape().dims(), &[4, 3]);
// b.at([i, j]) == l.at([0, j]) for every i ∈ 0..4
```

## Composition

The local ops compose freely. Each is a pure function of the input
layout:

```rust
# use quanta_tensor::Layout;
let src = Layout::row_major(&[2, 3, 4, 5]).unwrap();
let view = src
    .transpose(1, 2).unwrap()
    .permute(&[3, 0, 1, 2]).unwrap()
    .slice(0, 1, 4).unwrap();
assert_eq!(view.shape().dims(), &[3, 2, 4, 3]);
```

For any coordinate, `view.at(coord)` produces the same flat
offset as a hand-rolled stride dot product against the original
`src`. The integration test
[`composition_round_trip.rs`](../../crates/quanta-tensor/tests/composition_round_trip.rs)
asserts this for the full 72-coordinate sweep.

## The algebra (tiling — the load-bearing case)

The local ops above are per-axis transformations. They don't
compose into GEMM-style tile patterns on their own. For tiling,
reach for the algebra:

- **`compose(A, B)`** — apply `A` to the output of `B`. Imagine
  using `B` as an indexer that selects elements from `A`. Includes
  divisibility checks; returns `LayoutError::DivisibilityFailed`
  on violation.
- **`complement(A, cosize)`** — the layout that fills the
  remaining space after `A` within `cosize`. Used to build the
  residual modes of a tile.
- **`logical_divide(A, tiler)`** — partition `A` by `tiler` into a
  layout whose first modes are the tile and whose later modes are
  the residual. The workhorse for GEMM block/warp tiling, sort
  block radix, FFT bit-reversal.
- **`tiled_divide(A, tiler)`** — a convenience alias of
  `logical_divide` for our flat representation. Distinct method so
  call sites read clearly.

```rust
# use quanta_tensor::Layout;
// 72-element buffer divided by a 36-element block tile.
let buffer = Layout::row_major(&[72]).unwrap();
let block_tile = Layout::row_major(&[36]).unwrap();
let blocked = buffer.logical_divide(&block_tile).unwrap();
assert_eq!(blocked.shape().dims(), &[36, 2]);

// `blocked.at([elem, block])` is the offset of element `elem` in
// block `block` against the original buffer.
assert_eq!(blocked.at(&[0, 0]).unwrap(), 0);
assert_eq!(blocked.at(&[35, 1]).unwrap(), 71);
```

The algebra is ported from CUTLASS CuTe (see
`include/cute/layout.hpp` in the cutlass repo). CuTe encodes
layouts as compile-time integer tuples for kernel-time
specialisation; quanta-tensor uses runtime `Vec<usize>` so its
divisibility checks become runtime `Result`s. The trade-off is
covered under "Design notes" below.

## Design notes

- **Dynamic rank only.** Shapes and strides live in `Vec<usize>`
  / `Vec<isize>` at runtime, not in the type system. CuTe's
  compile-time-tuple form is more powerful but doesn't interop
  with the dynamic-shape paths every downstream math crate
  eventually needs. The companion crates (`quanta-prims` today;
  `quanta-blas`, `quanta-fft` to come) accept this trade-off in
  exchange for one runtime type that all four backends agree on.
- **Downstream proc-macros should consume accessors.** Use
  `Layout::shape()` and `Layout::strides()` (and `Shape::dims()`)
  rather than the private struct fields. The internal
  representation may shift without breaking accessor callers.

## What this enables downstream

Two layers of formal artifacts ship with the substrate:

**Lean** — 68 theorems across three layers:

- **Symbolic layer** (`specs/verify/lean/Quanta/Tensor/Layout.lean`,
  54 theorems): structural facts (linear size = product of dims,
  strides length = rank, indexer well-formedness, dot's cons
  distribution law) plus the algebraic theorems each downstream
  math crate inherits. Includes `tile_offset_bound`,
  `permutation_bijective` (via `List.Perm.map`), and
  `compose_assoc` for the rank-1 LHS × rank-1 middle × rank-N
  RHS case (T8048-T8052) proven at the layout-record level.

- **Denotational layer**
  (`specs/verify/lean/Quanta/Tensor/Denotational.lean`, 5
  theorems): canonical mathematical content — a layout is *defined*
  as its index function `Coord → Int`. Associativity
  (`composeD_assoc`) closes by `rfl` because composition of index
  functions *is* function composition. Built on `Fin n`-indexed
  shapes, no list machinery, foundational primitives only.

- **Bridge layer**
  (`specs/verify/lean/Quanta/Tensor/Bridge.lean`, 9 theorems):
  agreement between symbolic and denotational. Covers
  rank-1×rank-N and rank-M×rank-1 directions of compose, with
  full denotational-level associativity (`t8218`) instantiated
  at the bridge so any tower of symbolic compositions inherits
  associativity for free.

The fully general rank-M × rank-N × rank-K **symbolic** compose
still requires lifting CuTe's divisibility-checking fold into
Lean — modelled on the Verus side as `complement_general`,
deferred in Lean. The denotational layer covers it; closing the
remaining gap means proving more agreement theorems on top of
the bridge.

**Verus** (`specs/verify/verus/quanta/tensor_invariants.rs`, 42
verified) — the same structural facts as the Lean side plus a
closed-form spec for `complement_rank1` (six rank-1 theorems
covering all four branches) and a recursive `complement_fold`
spec for the rank-N case backed by length-invariant and length-
growth theorems plus base-offset preservation. The rank-N spec
processes the working `(shape, stride)` sequence head-first
rather than pick-min-each-step; both orderings produce the same
multiset of output modes, so the structural invariants hold for
either.

Each downstream math crate can lean on the proven half for its
own shape obligations without re-proving the algebra. As more
theorems land, the IR will eventually gain a `range_narrow`
rewrite that consumes layout invariants to elide bounds checks
in inner loops — a real performance win that benefits every
kernel using a `Layout`-typed parameter, not just the math ones.
