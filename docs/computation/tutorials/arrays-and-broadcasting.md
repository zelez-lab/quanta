# Arrays and broadcasting

> **You'll learn:** how to put data on the GPU as an array, do elementwise math
> with broadcasting, and read the result back — all without writing a kernel.
> This is the foundation every later lesson builds on.

If you've used NumPy, you already know the shape of this. `quanta-array`'s
`Array<T>` is an N-dimensional array that lives in GPU memory. You build it,
compute with it, and the work runs on whatever backend you compiled for.

```toml
[dependencies]
quanta       = { version = "0.1", features = ["metal"] } # or vulkan / software
quanta-array = { version = "0.1", features = ["metal"] }
```

## Open a device

Everything starts with a GPU handle. Use `init()` for a real GPU, or `init_cpu()`
for the software backend — handy on a machine without a GPU, and identical in
behaviour.

```rust,ignore
use quanta_array::Array;

let gpu = quanta::init_cpu(); // or quanta::init() for real hardware
```

## Build some data

An array is data plus a shape. Build one from a slice, or from a constructor:

```rust,ignore
// A 2×3 matrix.  (NumPy: np.array([[1,2,3],[4,5,6]], dtype=np.float32))
let a = Array::from_slice(&gpu, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

let z = Array::<f32>::zeros(&gpu, &[2, 3])?;   // np.zeros((2,3))
let o = Array::<f32>::ones(&gpu, &[2, 3])?;    // np.ones((2,3))
let f = Array::full(&gpu, 7.0f32, &[2, 3])?;   // np.full((2,3), 7)

let r = Array::<f32>::arange(&gpu, 0.0, 2.0, 5)?;   // 5 values stepping by 2
let l = Array::<f32>::linspace(&gpu, 0.0, 1.0, 5)?; // np.linspace(0,1,5)
```

The `&[2, 3]` is the shape. The element type comes from the data (`f32` here) —
`Array` is generic over every numeric dtype (`f32`, `f64`, `i32`, `u32`, …).

## Elementwise math

Operations apply to every element. Each returns a new `Array` (a `Result`,
because a GPU dispatch can fail):

```rust,ignore
let b = a.mul(&a)?;   // square every element
let c = a.add(&b)?;   // a + a²
```

Read a result back to the host with `to_vec`:

```rust,ignore
assert_eq!(a.add(&a)?.to_vec()?, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
```

## Broadcasting

You rarely have two arrays of exactly the same shape. **Broadcasting** stretches
a smaller array over a larger one along size-1 axes — the same rule as NumPy. Add
a per-row bias of shape `[2, 1]` to our `[2, 3]` matrix:

```rust,ignore
let bias = Array::from_slice(&gpu, &[10.0f32, 20.0], &[2, 1])?;
let biased = a.add(&bias)?;
// row 0 gets +10, row 1 gets +20:
// [[11,12,13],
//  [24,25,26]]
```

The bias is never physically expanded — broadcasting is lowered into the indexing
of the generated kernel, so there's no wasted memory or copy.

## Math functions

The transcendental functions (`sqrt`, `exp`, `log`, `sin`, `cos`, …) work on
float arrays:

```rust,ignore
let x = Array::from_slice(&gpu, &[1.0f32, 4.0, 9.0, 16.0], &[4])?;
let roots = x.sqrt()?;   // [1, 2, 3, 4]
let e = x.exp()?;        // elementwise eˣ
```

These are **float-only** by design. Calling `.sqrt()` on an integer array is a
*compile error*, not a silent wrong answer — the dtype is checked at compile time.

## What just happened

You never wrote a kernel, allocated a buffer, or launched a dispatch. Each
operation built GPU IR at runtime, JIT-compiled it for your backend, and ran it —
the same code path on Metal, Vulkan, WebGPU, or the CPU. Shapes are tracked and
checked; the math is where your attention goes.

## Next

- **[Reductions](reductions.md)** — collapse an array to a sum, a mean, or along one axis.
- Already know this? The **[Array math how-to](../how-to/arrays-numpy.md)** is the copy-paste version.
- Full surface: the **[quanta-array README](https://github.com/zelez-lab/quanta/blob/main/crates/quanta-array/README.md)** and the [API reference](../../reference/api.md).
