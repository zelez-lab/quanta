# Tensor Layout

`quanta-tensor` is the **shape-correctness substrate** for Quanta's
math-crate program. It ships pure-Rust types and functions: no GPU
runtime, no proc-macro, no kernels. Downstream math crates
(`quanta-sort`, `quanta-blas`, `quanta-fft`, `quanta-rand`) depend
on it and inherit its shape proofs.

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

The ops compose freely. Each is a pure function of the input
layout, so:

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

## What this enables downstream

Structural Lean theorems and Verus invariants ship alongside the
runtime substrate (linear size = product of dims, strides length =
rank, indexer well-formedness, dot's cons distribution law). The
deeper algebraic theorems land in follow-up commits:

- `compose_assoc`: `compose(compose(A, B), C) = compose(A, compose(B, C))`
- `permutation_bijective`: every `permute` is a bijection on `0..rank`
- `tile_offset_bound`: a tile offset stays within the linear size

Once those proofs land, each downstream math crate can lean on
them for its own shape obligations without re-proving the algebra.
The IR also gains a `range_narrow` rewrite that consumes layout
invariants to elide bounds checks in inner loops — a real
performance win that benefits every kernel using a `Layout`-typed
parameter, not just the math ones.
