# Arrays (quanta-array)

`quanta-array` is the NumPy-equivalent layer of the stack. `Array<T>` is a
host-side N-dimensional array backed by GPU memory: it owns a
[`Field`](02-fields-and-types.md) and carries a shape + strides, and it
gives you numpy-style construction, broadcasting elementwise math, whole-array
reductions, and zero-copy reshaping — **without writing a single kernel**.

```toml
[dependencies]
quanta-array = { version = "0.1", features = ["metal"] } # or vulkan / software
```

It is a **compute-only** consumer of `quanta` — the render half of the crate
is feature-gated off, so a program that only does array math compiles no
rendering code.

## A first array

```rust,ignore
use quanta_array::Array;

let gpu = quanta::init_cpu();              // or quanta::init() for a real GPU
let a = Array::from_slice(&gpu, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;

assert_eq!(a.shape(), &[2, 2]);
assert_eq!(a.rank(), 2);
assert_eq!(a.len(), 4);
```

`Array<T>` is generic over the numeric dtype. The element type comes from the
data you give it (`f32`, `f64`, `i32`, `u32`, …), so it follows the same
[GpuType](02-fields-and-types.md) set as the rest of the stack.

## Construction

| Builder | Meaning | numpy |
|---------|---------|-------|
| `Array::from_slice(gpu, &data, &shape)` | wrap a host slice | `np.array(data).reshape(shape)` |
| `Array::full(gpu, v, &shape)` | filled with `v` | `np.full(shape, v)` |
| `Array::zeros(gpu, &shape)` | all zeros | `np.zeros(shape)` |
| `Array::ones(gpu, &shape)` | all ones | `np.ones(shape)` |
| `Array::arange(gpu, start, step, n)` | `n` values from `start` by `step` | `np.arange` |
| `Array::linspace(gpu, start, stop, n)` | `n` evenly-spaced, inclusive | `np.linspace` |
| `Array::eye(gpu, n)` | `n × n` identity | `np.eye(n)` |

The numeric builders are generic, so spell the dtype when inference can't see
it from the data:

```rust,ignore
let z = Array::<i32>::zeros(&gpu, &[3]);        // [0, 0, 0]
let r = Array::<f32>::arange(&gpu, 0.0, 0.5, 4); // [0.0, 0.5, 1.0, 1.5]
```

## Elementwise math (ufuncs)

Arithmetic works on every dtype and **broadcasts** when shapes differ — the
generated kernel walks both operands with strided indexing:

```rust,ignore
let a = Array::from_slice(&gpu, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
let col = Array::from_slice(&gpu, &[10.0f32, 20.0], &[2, 1])?;
let c = a.add(&col)?;   // [2,1] broadcasts against [2,2] → [[11,12],[23,24]]
```

The `+ - * /` operators are also implemented on `&Array<T>` (they panic on a
shape error; call `.add()`/`.sub()`/… for the `Result` form):

```rust,ignore
let d = &a * &a;        // elementwise square
```

`minimum` / `maximum` / `pow` take a second array the same way.

### Math functions are floating-point only

The transcendental and rounding ufuncs — `abs`, `sqrt`, `exp`, `log`, `sin`,
`cos`, `floor`, `ceil` — exist only on float arrays. Calling one on an integer
array is a **compile error**, not a silently wrong result:

```rust,ignore
let i = Array::from_slice(&gpu, &[4i32, 9, 16], &[3])?;
let _ = i.sqrt();   // ❌ does not compile: i32 is not a FloatScalar
```

That boundary is part of the API contract: every backend implements these
functions for floats only, so quanta-array refuses to pretend otherwise.

## Reductions

`sum`, `min`, `max`, and `mean` reduce the **whole** array to a scalar, routed
to the matching `quanta-prims` device reduce:

```rust,ignore
let a = Array::from_slice(&gpu, &[3.0f32, -1.0, 7.0, 2.0], &[2, 2])?;
assert_eq!(a.max()?, 7.0);
let m = a.mean()?;      // 2.75
```

`sum` / `min` / `max` are available for the dtypes `quanta-prims` provides
reduces for — `f32`, `i32`, `u32`. `mean` is float-only (it divides). `f64`
arrays keep their math functions but have no device reduce, so `f64.sum()`
does not compile — again, an honest boundary rather than a hidden fallback.

> Per-axis reductions (`arr.sum(axis=0)`) and `prod` are a later increment —
> they need a segmented/strided reduce shape that prims doesn't ship yet.

## Linear algebra (f32)

`Array<f32>` gets `matmul`, `dot`, and `norm`, which call down into the
verified `quanta-blas` ops — so they carry the same mechanically-proven
forward-error bounds (Higham). Strided/transposed operands are gathered to
contiguous **on the device** first, so no host round-trip happens for the
math.

```rust,ignore
let a = Array::from_slice(&gpu, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
let b = Array::from_slice(&gpu, &[5.0f32, 6.0, 7.0, 8.0], &[2, 2])?;

let c = a.matmul(&b)?;              // [2,2] · [2,2] → [2,2]  (numpy a @ b)
let d = a.reshape(&[4])?.dot(&b.reshape(&[4])?)?; // 1-D inner product → f32
let n = a.norm()?;                 // L2 norm over all elements → f32
```

`matmul` requires 2-D operands with a matching inner dimension; `dot`
requires equal-length 1-D operands. It's f32-only for now (the blas Level-1
+ GEMM surface is f32). The GEMM underneath is the naive kernel — correct on
every backend; the tiled/tensor-core path is a later perf increment.

## Views are zero-copy

`reshape`, `permute`, `transpose`, and `broadcast_to` only rewrite the
layout; they share the same underlying `Field`, so they allocate nothing:

```rust,ignore
let a = Array::<f32>::arange(&gpu, 0.0, 1.0, 6)?;  // [0,1,2,3,4,5]
let m = a.reshape(&[2, 3])?;                       // view, no copy
let t = m.transpose(0, 1)?;                        // [3,2], strided view
```

When you need the values back on the host, `to_vec` materializes them in
logical row-major order, gathering through the strides for non-contiguous
views:

```rust,ignore
assert_eq!(t.to_vec()?, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
```

A ufunc on a strided view first compacts it to a contiguous buffer. That
compaction runs as a gather **kernel on the device** — the data never round-
trips through host memory.

## Everything stays on the device

`Array<T>` is a GPU-resident contract, not "an ndarray that visits the GPU".
The hot-path operations keep their data on the device:

- **ufuncs** dispatch a kernel; nothing is downloaded.
- **strided-view compaction** (`contiguous`) is an on-device gather kernel.
- **reductions** hand the array's `Field` straight to the `quanta-prims`
  device reduce — the whole array is never downloaded; only the tiny
  per-block partials (256× smaller) touch host memory between passes.

The only host transfers are the ones you ask for: building from a host slice
(`from_slice`) and reading back (`to_vec`). This matters because the layers
above (autograd, nn ops) chain many operations — a host round-trip per op
would dominate. Keeping the contract device-resident is what makes those
layers viable.

## Where the algebra is proven

The shape/stride layer is `quanta-tensor`, whose layout algebra (`reshape`,
`permute`, `broadcast`, `at`) is verified in Lean. quanta-array builds its
views on top of that, so the index math behind every zero-copy view rests on
the proven [tensor layout](../concepts/tensor-layout.md).
