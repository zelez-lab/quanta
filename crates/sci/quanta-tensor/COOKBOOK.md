# quanta-tensor cookbook

Recipe catalogue: layout patterns that come up over and over in
downstream math kernels. Each recipe states the workload, the
layout it produces, and the offset formula it implements.

> All snippets compile against `quanta-tensor` alone. No GPU runtime,
> no proc-macro, no feature flags.

---

## Row-major dense matrix

**Use case:** the default C / NumPy layout. Last axis is contiguous.

**Pattern:**

```rust
use quanta_tensor::Layout;

let m = Layout::row_major(&[4, 8])?;            // 4 rows, 8 cols
assert_eq!(m.strides(), &[8, 1]);
// Offset of (row, col) = row*8 + col.
assert_eq!(m.at(&[2, 5])?, 21);
# Ok::<_, quanta_tensor::LayoutError>(())
```

When to reach for it: every operator that walks rows from left to
right — convolutions, matmul A operand, attention queries.

---

## Column-major dense matrix

**Use case:** the Fortran / BLAS layout. First axis is contiguous.

**Pattern:**

```rust
use quanta_tensor::Layout;

let m = Layout::column_major(&[4, 8])?;         // 4 rows, 8 cols
assert_eq!(m.strides(), &[1, 4]);
// Offset of (row, col) = row + col*4.
assert_eq!(m.at(&[2, 5])?, 22);
# Ok::<_, quanta_tensor::LayoutError>(())
```

When to reach for it: BLAS GEMM B operand, FORTRAN-style linear
algebra, any kernel that walks columns inside its innermost loop.

---

## Transposed matrix view

**Use case:** read `A^T` from the same buffer that holds `A`.

**Pattern:**

```rust
use quanta_tensor::Layout;

let a  = Layout::row_major(&[4, 8])?;
let at = a.transpose(0, 1)?;                    // 8 rows, 4 cols
assert_eq!(at.shape().dims(), &[8, 4]);
assert_eq!(at.strides(), &[1, 8]);
// at.at(j, i) == a.at(i, j).
assert_eq!(at.at(&[5, 2])?, a.at(&[2, 5])?);
# Ok::<_, quanta_tensor::LayoutError>(())
```

Free transposes are the foundation of GEMM kernel families: pick
the layout that puts the contiguous dimension on the inner loop.

---

## Broadcast vector against matrix

**Use case:** add a length-N bias vector to every row of an MxN
matrix without allocating MxN copies.

**Pattern:**

```rust
use quanta_tensor::Layout;

let bias = Layout::row_major(&[8])?;            // [8]
let view = bias.broadcast(&[4, 8])?;            // 4x8 "tiled" view
assert_eq!(view.shape().dims(), &[4, 8]);
assert_eq!(view.strides(), &[0, 1]);            // stride 0 on rows
// Every row reads the same 8 bias elements.
for row in 0..4 {
    for col in 0..8 {
        assert_eq!(view.at(&[row, col])?, col);
    }
}
# Ok::<_, quanta_tensor::LayoutError>(())
```

The stride-0 axis is the broadcasting trick: every coordinate on
that axis lands on the same buffer offset.

---

## GEMM-style block tile

**Use case:** split an MxN output into BMxBN blocks for a tiled
matrix-multiply kernel. Each block becomes one work-group.

**Pattern:**

```rust
use quanta_tensor::Layout;

// 4096-element row-major output (flatten an MxN matrix to 1-D).
let out = Layout::row_major(&[4096])?;

// Block size: 64 elements per block, 64 blocks total.
let tile = Layout::row_major(&[64])?;
let tiled = out.logical_divide(&tile)?;

assert_eq!(tiled.shape().dims(), &[64, 64]);
//                                  ^^  ^^
//                          per-block  block index
//                          element

// tiled.at([elem_in_block, block_idx])
//   = block_idx * 64 + elem_in_block.
assert_eq!(tiled.at(&[0, 0])?,    0);
assert_eq!(tiled.at(&[63, 0])?,  63);
assert_eq!(tiled.at(&[0, 1])?,   64);
assert_eq!(tiled.at(&[63, 63])?, 4095);
# Ok::<_, quanta_tensor::LayoutError>(())
```

For 2-D tilings (BMxBK over MxK), pass a rank-2 `tile`:

```rust
use quanta_tensor::Layout;

let buffer = Layout::row_major(&[24])?;
let tile   = Layout::row_major(&[2, 3])?;       // 2x3 footprint
let tiled  = buffer.logical_divide(&tile)?;
assert_eq!(tiled.shape().dims(), &[2, 3, 4]);   // 4 tiles
assert_eq!(tiled.strides(),      &[3, 1, 6]);
# Ok::<_, quanta_tensor::LayoutError>(())
```

The tiler's modes come first (within-tile coordinates), the
complement's modes come last (across-tile coordinates). That's
exactly the loop ordering a GEMM kernel wants — outer loop over
blocks, inner loops over within-block elements.

---

## Iterated block-then-warp tiling

**Use case:** nested tiling. Tile the M dimension by 64 first
(block tile), then tile each block by 8 (warp tile). Three loop
levels: warp inside block, block inside output.

**Pattern:**

```rust
use quanta_tensor::Layout;

const ELEMS_PER_WARP: usize  = 12;
const WARPS_PER_BLOCK: usize =  3;
const BLOCKS: usize          =  2;
const BLOCK_SIZE: usize = ELEMS_PER_WARP * WARPS_PER_BLOCK;  // 36
const TOTAL: usize      = BLOCK_SIZE * BLOCKS;               // 72

let buffer = Layout::row_major(&[TOTAL])?;

// Step 1: divide by block tile.
let blocked = buffer.logical_divide(&Layout::row_major(&[BLOCK_SIZE])?)?;
assert_eq!(blocked.shape().dims(), &[BLOCK_SIZE, BLOCKS]);

// Step 2: within one block, divide by warp tile.
let inner   = Layout::row_major(&[BLOCK_SIZE])?;
let warped  = inner.logical_divide(&Layout::row_major(&[ELEMS_PER_WARP])?)?;
assert_eq!(warped.shape().dims(), &[ELEMS_PER_WARP, WARPS_PER_BLOCK]);

// End-to-end: (block, warp, elem) -> flat offset.
for block in 0..BLOCKS {
    for warp in 0..WARPS_PER_BLOCK {
        for elem in 0..ELEMS_PER_WARP {
            let inner_elem = warp * ELEMS_PER_WARP + elem;
            let got      = blocked.at(&[inner_elem, block])?;
            let expected = block * BLOCK_SIZE + warp * ELEMS_PER_WARP + elem;
            assert_eq!(got, expected);
        }
    }
}
# Ok::<_, quanta_tensor::LayoutError>(())
```

This is the "block then warp" recipe every GEMM kernel ends up
walking. Each `logical_divide` adds one more mode of indirection;
the offsets compose by addition because the strides multiply
through correctly.

---

## Identity composition (round-trip)

**Use case:** sanity-check. Composing any layout with a flat read
pattern over its `linear_size()` should reproduce it.

**Pattern:**

```rust
use quanta_tensor::Layout;

let m    = Layout::row_major(&[2, 3, 4])?;
let flat = Layout::row_major(&[m.linear_size()])?;
let v    = m.compose(&flat)?;

assert_eq!(v.shape().dims(),  m.shape().dims());
assert_eq!(v.strides(),       m.strides());
assert_eq!(v.base_offset(),   m.base_offset());
# Ok::<_, quanta_tensor::LayoutError>(())
```

The composition algebra has a left identity in the flat layout —
this property is one of the Lean theorems
(`Quanta.Tensor.Layout.compose11_id_left` in `Layout.lean`).

---

## Subtensor extraction (slice along one axis)

**Use case:** take rows 2..5 of a matrix without copying.

**Pattern:**

```rust
use quanta_tensor::Layout;

let m   = Layout::row_major(&[8, 16])?;
let sub = m.slice(0, 2, 5)?;                    // 3x16 view

assert_eq!(sub.shape().dims(), &[3, 16]);
assert_eq!(sub.strides(),       &[16, 1]);
assert_eq!(sub.base_offset(),    32);            // 2 * 16

// sub.at([0, 0]) reads the original buffer offset 32 = (2, 0).
assert_eq!(sub.at(&[0, 0])?, 32);
assert_eq!(sub.at(&[2, 15])?, 79);              // 4 * 16 + 15
# Ok::<_, quanta_tensor::LayoutError>(())
```

`slice` shifts the `base_offset` to the first element of the
slice; the strides are unchanged. Combining `slice` + `at` is the
generic "view into a sub-region" primitive.

---

## Axis permutation (NHWC <-> NCHW)

**Use case:** convert between memory layouts that frameworks
disagree on — channels-last (TF) vs channels-first (PyTorch).

**Pattern:**

```rust
use quanta_tensor::Layout;

// NHWC (TF default): batch, height, width, channels.
let nhwc = Layout::row_major(&[2, 32, 32, 3])?;

// NCHW (PyTorch default): batch, channels, height, width.
let nchw = nhwc.permute(&[0, 3, 1, 2])?;

assert_eq!(nchw.shape().dims(), &[2, 3, 32, 32]);
// Same buffer, different walk order. No copy needed.
# Ok::<_, quanta_tensor::LayoutError>(())
```

A permutation is a bijection of axes; the round-trip property
(`permute(perm).permute(inverse(perm)) == identity`) is one of
the proven Lean theorems.

---

## Where to go next

- **[README.md](README.md)** — full crate overview.
- **[GETTING_STARTED.md](GETTING_STARTED.md)** — first walkthrough.
- **`tests/tile_pattern.rs`** — end-to-end iterated tiling.
- **`tests/composition_round_trip.rs`** — composition algebra
  round-trips at every rank.
- **`tests/bench_compose.rs`** — micro-benchmarks for the core
  composition path.
