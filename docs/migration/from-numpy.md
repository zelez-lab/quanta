# Migration from NumPy

`quanta-array` mirrors the NumPy array surface, so most NumPy code maps line
for line. The big difference is that every array lives in GPU memory and every
operation runs on the GPU (or the software CPU backend) — and that the math
functions are typed: float-only ops simply don't compile on integer arrays.

## Terminology

| NumPy | Quanta | Notes |
|-------|--------|-------|
| `np.ndarray` | `Array<T>` | host handle to a GPU buffer |
| `dtype` | the `T` in `Array<T>` | `f32` / `f64` / `i32` / `u32` / … |
| `arr.shape` | `arr.shape()` | `&[usize]` |
| `arr.ndim` | `arr.rank()` | |
| `arr.size` | `arr.len()` | |
| `arr.strides` | `arr.strides()` | in elements, not bytes |
| (implicit device) | the `Gpu` you pass to constructors | pick the backend at init |

## Construction

| NumPy | quanta-array |
|-------|--------------|
| `np.array(xs).reshape(s)` | `Array::from_slice(&gpu, &xs, &s)` |
| `np.zeros(s)` | `Array::<f32>::zeros(&gpu, &s)` |
| `np.ones(s)` | `Array::<f32>::ones(&gpu, &s)` |
| `np.full(s, v)` | `Array::full(&gpu, v, &s)` |
| `np.arange(start, stop, step)` | `Array::arange(&gpu, start, step, n)` |
| `np.linspace(a, b, n)` | `Array::linspace(&gpu, a, b, n)` |
| `np.eye(n)` | `Array::<f32>::eye(&gpu, n)` |

`arange` takes an explicit count `n` instead of a `stop`, so the length is
unambiguous on the host.

## Elementwise + reductions

| NumPy | quanta-array |
|-------|--------------|
| `a + b`, `a - b`, `a * b`, `a / b` | `&a + &b`, … or `a.add(&b)?`, … |
| `-a` | `a.neg()?` |
| `np.sqrt(a)`, `np.exp(a)`, `np.sin(a)`, … | `a.sqrt()?`, `a.exp()?`, `a.sin()?`, … (float only) |
| `np.abs`, `np.floor`, `np.ceil` | `a.abs()?`, `a.floor()?`, `a.ceil()?` |
| `np.minimum(a, b)`, `np.maximum`, `a ** b` | `a.minimum(&b)?`, `a.maximum(&b)?`, `a.pow(&b)?` |
| `a.sum()`, `a.mean()`, `a.min()`, `a.max()` | `a.sum()?`, `a.mean()?`, `a.min()?`, `a.max()?` |

Broadcasting follows the NumPy rule (trailing dimensions align; size-1 axes
stretch). It is lowered into strided indexing in the generated kernel, so no
operand is ever physically expanded.

## Views

| NumPy | quanta-array |
|-------|--------------|
| `a.reshape(s)` | `a.reshape(&s)?` |
| `a.T` / `a.transpose(i, j)` | `a.transpose(i, j)?` |
| `np.transpose(a, perm)` | `a.permute(&perm)?` |
| `np.broadcast_to(a, s)` | `a.broadcast_to(&s)?` |
| `a.tolist()` / `np.asarray(a)` | `a.to_vec()?` (logical row-major) |

All four view operations are zero-copy — they rewrite the layout over the same
GPU buffer.

## Side-by-side: min-max normalize

### NumPy

```python
import numpy as np
x = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
out = (x - x.mean()) / (x.max() - x.min())
```

### quanta-array

```rust,ignore
use quanta_array::Array;
let gpu = quanta::init();
let x = Array::from_slice(&gpu, &[2.0f32, 4.0, 6.0, 8.0], &[4])?;
let mean = x.mean()?;
let span = x.max()? - x.min()?;
let centered = x.sub(&Array::full(&gpu, mean, &[4])?)?;
let out = centered.div(&Array::full(&gpu, span, &[4])?)?.to_vec()?;
```

## What's different on purpose

- **Typed math.** `int_array.sqrt()` is a compile error. NumPy returns a float
  array; quanta-array makes you convert dtype first. This catches a whole class
  of silent-precision bugs at build time.
- **Reduction dtypes.** `sum`/`min`/`max` exist for `f32`/`i32`/`u32` (the
  reduces `quanta-prims` ships); `mean` is float-only. `f64` has math functions
  but no device reduce yet.
- **No per-axis reductions or fancy indexing yet.** `arr.sum(axis=0)`,
  `arr[mask]`, and `arr[idx]` are planned increments, not in this release.
- **Explicit `Gpu` + `Result`.** Every constructor takes the `Gpu` and every
  operation returns a `Result` (the operators panic for ergonomics; the named
  methods don't).
