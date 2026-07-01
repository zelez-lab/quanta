# Array math (NumPy on the GPU)

`quanta-array` covers the everyday NumPy surface — build an array, do
elementwise math with broadcasting, reduce it — on whatever backend you
compiled for, no kernels involved. This page is a task-by-task recipe; the
[Arrays guide chapter](../../computation/tutorials/arrays.md) is the narrative version.

```toml
[dependencies]
quanta-array = { version = "0.1", features = ["metal"] } # vulkan / software
```

## Setup

```rust,ignore
use quanta_array::Array;

let gpu = quanta::init();        // real GPU; init_cpu() for the CPU backend
```

## Build some data

```rust,ignore
// np.array([[1,2,3],[4,5,6]], dtype=np.float32)
let a = Array::from_slice(&gpu, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

// np.zeros((2,3)) / np.ones((2,3)) / np.full((2,3), 7.0)
let z = Array::<f32>::zeros(&gpu, &[2, 3])?;
let o = Array::<f32>::ones(&gpu, &[2, 3])?;
let f = Array::full(&gpu, 7.0f32, &[2, 3])?;

// np.arange(0, 10, 2) -> 5 values; np.linspace(0, 1, 5); np.eye(3)
let r = Array::<f32>::arange(&gpu, 0.0, 2.0, 5)?;
let l = Array::<f32>::linspace(&gpu, 0.0, 1.0, 5)?;
let i = Array::<f32>::eye(&gpu, 3)?;
```

## Elementwise + broadcasting

```rust,ignore
// c = a + a, a * 2, etc. — operator form panics on shape mismatch
let c = &a + &a;
let d = a.mul(&a)?;          // Result form

// Broadcasting: add a per-row bias of shape [2,1] to a [2,3] array
let bias = Array::from_slice(&gpu, &[10.0f32, 20.0], &[2, 1])?;
let biased = a.add(&bias)?; // [[11,12,13],[24,25,26]]
```

## Math functions (float arrays)

```rust,ignore
let x = Array::from_slice(&gpu, &[1.0f32, 4.0, 9.0, 16.0], &[4])?;
let roots = x.sqrt()?;       // [1, 2, 3, 4]
let activ = x.exp()?;        // elementwise e^x
let clamped = x.minimum(&Array::full(&gpu, 10.0f32, &[4])?)?;
```

## Reduce to a scalar

```rust,ignore
let total = a.sum()?;        // a.sum()
let avg   = a.mean()?;       // a.mean()
let lo    = a.min()?;        // a.min()
let hi    = a.max()?;        // a.max()
```

Integer reductions work too:

```rust,ignore
let counts = Array::from_slice(&gpu, &[3i32, 1, 4, 1, 5, 9], &[6])?;
assert_eq!(counts.sum()?, 23);
assert_eq!(counts.max()?, 9);
```

## Reshape and transpose (no copy)

```rust,ignore
let flat = Array::<f32>::arange(&gpu, 0.0, 1.0, 6)?; // [0..6)
let m = flat.reshape(&[2, 3])?;
let t = m.transpose(0, 1)?;          // [3,2] strided view
let back = t.to_vec()?;              // gathers logical order: [0,3,1,4,2,5]
```

## A small end-to-end: normalize a vector

```rust,ignore
// (x - mean) / (max - min) — standard min-max feature scaling
let x = Array::from_slice(&gpu, &[2.0f32, 4.0, 6.0, 8.0], &[4])?;
let mean = x.mean()?;
let span = x.max()? - x.min()?;
let centered = x.sub(&Array::full(&gpu, mean, &[4])?)?;
let scaled = centered.div(&Array::full(&gpu, span, &[4])?)?;
let out = scaled.to_vec()?;
```

Everything above runs identically on Metal, Vulkan, and the software CPU
backend — switch the Cargo feature, not the code.
