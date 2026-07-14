# Linear algebra

> **You'll learn:** matrix multiply, dot products, and norms on the GPU, plus the
> lower-level BLAS entry points. Builds on [Shape and views](shape-and-views.md).

Linear algebra is the workhorse of numerical computing. `quanta-array` gives you
the everyday operations directly on arrays; `quanta-blas` sits underneath with
the classic BLAS surface and machine-proven error bounds.

## Matrix multiply

`matmul` is the 2-D matrix product `A(m×k) · B(k×n) → (m×n)`:

```rust,ignore
use quanta_array::Array;
let gpu = quanta::init_cpu();

let a = Array::from_slice(&gpu, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?; // 2×3
let b = Array::from_slice(&gpu, &[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2])?; // 3×2
let c = a.matmul(&b)?;   // 2×2
```

Combine it with the views from the last lesson — a transposed operand is a
zero-copy view, so `a.matmul(&b.transpose(0, 1)?.contiguous()?)` computes `A·Bᵀ`.

## Dot products and norms

For vectors:

```rust,ignore
let x = Array::from_slice(&gpu, &[1.0f32, 2.0, 3.0], &[3])?;
let y = Array::from_slice(&gpu, &[4.0f32, 5.0, 6.0], &[3])?;

let d = x.dot(&y)?;    // 32.0  — Σ xᵢ·yᵢ
let n = x.norm()?;     // √14   — the L2 norm
```

These reduce on the device (the vectors never leave the GPU) — the same
device-resident pattern as [reductions](reductions.md).

## Dropping to BLAS

When you want the named BLAS operations — in-place scaling, fused multiply-add,
`C ← α·A·B + β·C` — reach for `quanta-blas` directly. It operates on raw
`Field`s (GPU buffers) rather than `Array`s, which is what you want inside a
performance-critical inner loop:

```rust,ignore
// y ← α·x + y   (the classic axpy)
quanta_blas::axpy(&gpu, 2.0, &x_field, &y_field)?;

// C ← α·A·B + β·C   (general matrix multiply)
quanta_blas::gemm(&gpu, m, n, k, 1.0, &a, &b, 0.0, &c)?;
```

Every `quanta-blas` op ships a **proven forward-error bound** (a Higham-style
`(1+δ)` bound formalised in Lean), and `gemm` has mixed-precision (bf16/f16/fp8)
and quantized (int8/int4) variants for when you trade accuracy for throughput.
See the [quanta-blas README](https://github.com/zelez-lab/quanta/blob/main/crates/sci/quanta-blas/README.md).

## A note on dtypes

The array-level `matmul` / `dot` / `norm` are **f32** today (they reuse the f32
GEMM path). That's the same precision the autodiff track uses, so everything
composes.

## Next

- **[FFT](fft.md)** — the Fourier transform, another verified numerical primitive.
- How-to: **[Matrix multiply](../how-to/matrix-multiply.md)** for the tiled-kernel details.
