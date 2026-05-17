# Getting started with quanta-tensor

A 10-minute walkthrough: from `cargo new` to a working program that
reasons about tensor layouts without copying any data.

## What this crate is *for*

`quanta-tensor` is the **layout algebra substrate** the other Quanta
math crates plug into. It answers one question:

> *Given a multi-dimensional shape and a flat buffer, what's the byte
> offset of the element at coordinate `(i, j, k, …)`?*

The crate ships:

- [`Shape`](src/shape.rs) — multi-dimensional extents.
- [`Layout`](src/layout.rs) — a function-style index from a coordinate
  tuple to a flat-buffer offset.
- **Local ops** — `transpose`, `permute`, `slice`, `broadcast`.
- **Algebra** — `compose`, `complement`, `logical_divide`,
  `tiled_divide`. The load-bearing combinators downstream GEMM / sort
  / FFT tilings reduce to.

**No GPU runtime. No proc-macro. No kernels.** Pure Rust types and
functions — works on any target Rust supports.

## What you need

- **Rust 1.85+** (`rustup show` to check)
- **git** on `PATH` (Cargo uses it to fetch the dependency below).

## Step 1 — Create a new project

```sh
cargo new --bin my_layout_app
cd my_layout_app
```

## Step 2 — Add quanta-tensor

```sh
cargo add quanta-tensor --git https://github.com/zelez-lab/quanta
```

That produces the following `[dependencies]` block in `Cargo.toml`:

```toml
[dependencies]
quanta-tensor = { git = "https://github.com/zelez-lab/quanta" }
```

Pin to a specific revision with `--rev <sha>` (or `--tag <tag>` once
tagged releases exist) if you want reproducible builds.

No feature flags. No optional dependencies. The substrate is one
small crate with zero external dependencies.

## Step 3 — Your first `Layout`

Replace `src/main.rs` with:

```rust
use quanta_tensor::Layout;

fn main() -> Result<(), quanta_tensor::layout::LayoutError> {
    // A 2x3 row-major tensor — six elements laid out as:
    //   (0,0) (0,1) (0,2) (1,0) (1,1) (1,2)
    // at flat offsets 0, 1, 2, 3, 4, 5.
    let m = Layout::row_major(&[2, 3])?;

    assert_eq!(m.shape().dims(), &[2, 3]);
    assert_eq!(m.strides(), &[3, 1]);
    assert_eq!(m.linear_size(), 6);

    // Look up the offset of element (1, 2):
    let o = m.at(&[1, 2])?;
    println!("element (1, 2) is at flat offset {o}");  // -> 5

    Ok(())
}
```

```sh
cargo run
```

Expected output:

```
element (1, 2) is at flat offset 5
```

## Step 4 — Local ops compose without copying

Layout ops produce a *new layout* over the *same flat buffer*. No
data is moved.

```rust
use quanta_tensor::Layout;

let src = Layout::row_major(&[2, 3, 4])?;       // 2x3x4 row-major

// transpose swaps two axes:
let t = src.transpose(1, 2)?;                   // 2x4x3
assert_eq!(t.shape().dims(), &[2, 4, 3]);

// permute reorders all axes:
let p = src.permute(&[2, 0, 1])?;               // 4x2x3
assert_eq!(p.shape().dims(), &[4, 2, 3]);

// slice clips one axis to a half-open range:
let s = src.slice(0, 1, 2)?;                    // 1x3x4 (just row 1)
assert_eq!(s.shape().dims(), &[1, 3, 4]);

// broadcast adds size-1 axes / replicates a size-1 axis to size N:
let b = Layout::row_major(&[3])?.broadcast(&[2, 3])?;
assert_eq!(b.shape().dims(), &[2, 3]);
assert_eq!(b.strides(), &[0, 1]);               // row 0 == row 1
```

Every op chains:

```rust
let view = Layout::row_major(&[2, 3, 4])?
    .transpose(1, 2)?       // 2x4x3
    .permute(&[2, 0, 1])?   // 3x2x4
    .slice(0, 1, 3)?;       // 2x2x4 starting at row 1
```

`view.at(&[a, b, c])` returns a flat offset into the original 24-
element buffer. No allocation, no memcpy.

## Step 5 — Row-major vs column-major

Two stock constructors:

```rust
let r = Layout::row_major(&[2, 3])?;            // strides [3, 1]
let c = Layout::column_major(&[2, 3])?;         // strides [1, 2]

assert_eq!(r.at(&[1, 2])?, 5);                  // 1*3 + 2*1 = 5
assert_eq!(c.at(&[1, 2])?, 5);                  // 1*1 + 2*2 = 5
//                                                  ^^^^^^^^^
// Same coordinate, same offset for this particular cell — but
// strolling axis 1 in row-major walks +1 per step; in column-major
// it walks +2.
```

Row-major is the C / NumPy default. Column-major is Fortran / BLAS.
Pick whichever matches your kernel's loop order.

## Step 6 — Tiling: the load-bearing case

Local ops handle per-axis transformations. For block / warp / tile
patterns — splitting a tensor into smaller logical tensors —
`logical_divide` is the right primitive:

```rust
use quanta_tensor::Layout;

// A contiguous 72-element buffer.
let buffer = Layout::row_major(&[72])?;

// Divide by a block tile of 36 elements: 2 blocks total.
let block_tile = Layout::row_major(&[36])?;
let blocked = buffer.logical_divide(&block_tile)?;

assert_eq!(blocked.shape().dims(), &[36, 2]);
// blocked.at([elem_in_block, block]) is the flat offset of element
// `elem_in_block` within block `block` against the original buffer.
assert_eq!(blocked.at(&[0,  0])?, 0);
assert_eq!(blocked.at(&[35, 1])?, 71);
```

The output's first mode walks *within* a tile, the trailing modes
walk *across* tiles. That's the exact shape a GEMM block scheduler
or radix-sort histogram pass wants.

For rank-N tilers (2-D block tiles, 3-D warp tiles, …), pass a rank-
N `tile` layout:

```rust
let buffer = Layout::row_major(&[24])?;
let tile   = Layout::row_major(&[2, 3])?;       // 2x3 tile, footprint 6
let tiled  = buffer.logical_divide(&tile)?;

assert_eq!(tiled.shape().dims(), &[2, 3, 4]);   // 4 tiles of 2x3
assert_eq!(tiled.strides(),      &[3, 1, 6]);
```

## Step 7 — When to reach for `compose` directly

`compose` is what `logical_divide` is built on. Use it directly when
you've already constructed the "read pattern" as its own layout:

```rust
// A 2x3 row-major matrix, viewed *as a 6-element flat vector*.
let mat  = Layout::row_major(&[2, 3])?;
let flat = Layout::row_major(&[6])?;
let view = mat.compose(&flat)?;

// view has the same shape and offsets as `mat` — composition with
// the trivial 6-element read pattern is the identity.
assert_eq!(view.shape().dims(), &[2, 3]);
```

The CuTe slogan: **composition is the type-system-equivalent of
nested loops**.

## Step 8 — Errors are first-class

Every fallible op returns `Result<Layout, LayoutError>`. The error
variants are explicit, not strings:

```rust
use quanta_tensor::Layout;
use quanta_tensor::layout::LayoutError;

// Axis 5 doesn't exist in a rank-2 layout:
let err = Layout::row_major(&[2, 3])?.transpose(0, 5);
assert!(matches!(err, Err(LayoutError::AxisOutOfRange { .. })));

// Broadcast can't expand a non-size-1 axis to a different extent:
let err = Layout::row_major(&[3])?.broadcast(&[5]);
assert!(matches!(err, Err(LayoutError::BroadcastIncompatible { .. })));
```

The full list:

- `Shape(ShapeError)` — bad shape passed to a constructor.
- `RankMismatch` — `at` got the wrong number of coordinates.
- `OutOfBounds` — coordinate ≥ axis extent.
- `AxisOutOfRange` — op referenced an axis ≥ rank.
- `InvalidPermutation` — `permute` got a non-permutation.
- `InvalidSlice` — empty / reversed / oversized range.
- `BroadcastIncompatible` — extents don't align.
- `DivisibilityFailed` — composition divisibility law violated.
- `ComplementInfeasible` — complement is undefined (zero stride,
  non-injective layout).
- `UnsupportedRank` — reserved for ops that don't yet handle the
  given rank (currently only `compose` with both rank ≥ 2).

## Where to go next

- **[README.md](README.md)** — full crate overview + status.
- **[COOKBOOK.md](COOKBOOK.md)** — recipe catalogue: GEMM tiling,
  FFT butterflies, sort permutations, …
- **`tests/tile_pattern.rs`** — end-to-end iterated tiling against
  manual row-major offsets, the load-bearing integration test.
- **`tests/composition_round_trip.rs`** — composing with the
  identity should be the identity; tested at every rank.

## Troubleshooting

**`error[E0432]: unresolved import quanta_tensor::layout`**
The `layout` module is public but error types live inside it.
Either `use quanta_tensor::layout::LayoutError;` or fully qualify.

**`UnsupportedRank { op: "compose", rank: N }`**
`compose` currently requires at least one of the two layouts to be
rank ≤ 1. The full rank-N case is on the roadmap but not yet shipped.
For tiling, prefer `logical_divide` (which handles rank-N tilers).

**`ComplementInfeasible { reason: "stride 0 has no complement" }`**
A broadcast layout has stride 0 along the broadcast axis; complement
isn't defined there. Apply complement before broadcasting.
