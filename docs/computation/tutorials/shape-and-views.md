# Shape and views

> **You'll learn:** how to reshape, transpose, and permute arrays without copying
> data, and why that matters. Builds on [Reductions](reductions.md).

An array is a flat buffer plus a *layout* — the shape and the strides that map
coordinates to buffer offsets. Changing the layout is free: it produces a new
*view* over the same GPU buffer, no data moved.

## Reshape

Reinterpret the same elements under a new shape (the element count must match):

```rust,ignore
use quanta::sci::Array;
let gpu = quanta::init_cpu();

let flat = Array::<f32>::arange(&gpu, 0.0, 1.0, 6)?; // [0,1,2,3,4,5]
let m = flat.reshape(&[2, 3])?;                       // [[0,1,2],[3,4,5]]
let v = m.reshape(&[6])?;                             // back to a vector
```

## Transpose and permute

`transpose` swaps two axes; `permute` reorders all of them. Both are zero-copy
views — they rewrite the strides, not the data:

```rust,ignore
let mt = m.transpose(0, 1)?;      // [3, 2] view — rows and columns swapped
let t  = a.permute(&[2, 0, 1])?;  // move the last axis to the front (any rank)
```

Because a transposed view is *non-contiguous* (its strides no longer march in
row-major order), some operations that need contiguous memory will ask you to
materialize it first with `.contiguous()` — which is the one place a copy happens,
and only when you request it.

## Broadcast as a view

`broadcast_to` stretches size-1 axes up to a target shape, again without copying
— it sets the stride of the broadcast axis to zero, so every "row" reads the same
underlying data:

```rust,ignore
let bias = Array::from_slice(&gpu, &[10.0f32, 20.0, 30.0], &[1, 3])?;
let big = bias.broadcast_to(&[4, 3])?; // 4 identical rows, zero extra memory
```

This is exactly the machinery behind the broadcasting in
[lesson 1](arrays-and-broadcasting.md) — now you can invoke it explicitly.

## Narrow: a sub-range view

`narrow(axis, start, len)` keeps `len` elements along one axis starting at
`start`, sharing the buffer with the extent clipped and the base offset
advanced — no copy:

```rust,ignore
let x = Array::from_slice(&gpu, &(0..12).map(|i| i as f32).collect::<Vec<_>>(), &[4, 3])?;
let mid = x.narrow(0, 1, 2)?;   // rows 1 and 2 → a [2, 3] view
let cols = x.narrow(1, 0, 2)?;  // first two columns → a strided [4, 2] view
```

This is the minibatch selector: `x.narrow(0, b * batch, batch)` picks batch `b`
out of a stacked `[N, …]` array for free. The autograd version,
[`Var::narrow`](autodiff-basics.md), makes it differentiable — the gradient
flows back only to the selected rows.

## Why zero-copy matters

On a GPU, a copy is a real cost: bandwidth and a synchronization point. Keeping
`reshape` / `transpose` / `permute` free means you can express a computation in
whatever shape reads clearly — flatten for a matmul, unflatten for a
convolution — and pay only for the arithmetic. The layout algebra underneath is
proven correct in Lean (via `quanta-tensor`), so a view can never alias out of
bounds.

## Next

- **[Linear algebra](linear-algebra.md)** — matmul and friends, where shape discipline pays off.
