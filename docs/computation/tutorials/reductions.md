# Reductions

> **You'll learn:** how to collapse an array to a single number, or reduce along
> one axis while keeping the others. Builds on [Arrays and broadcasting](arrays-and-broadcasting.md).

A *reduction* combines many values into fewer — a sum, a maximum, an average.
They run entirely on the device (the data never comes back to the host until you
ask for it), so they're cheap to chain after other array work.

## Whole-array reductions

The simplest reductions collapse the entire array to one scalar:

```rust,ignore
use quanta::sci::Array;
let gpu = quanta::init_cpu();

let a = Array::from_slice(&gpu, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

let total = a.sum()?;   // 21.0     (np.sum)
let avg   = a.mean()?;  // 3.5      (np.mean)
let lo    = a.min()?;   // 1.0      (np.min)
let hi    = a.max()?;   // 6.0      (np.max)
```

These return a host scalar directly — the device reduces, then hands you the one
value. Integer arrays reduce too:

```rust,ignore
let counts = Array::from_slice(&gpu, &[3i32, 1, 4, 1, 5, 9], &[6])?;
assert_eq!(counts.sum()?, 23);
assert_eq!(counts.max()?, 9);
```

## Reducing along an axis

Often you want to reduce *one* dimension and keep the rest — sum each column, or
average each row. `sum_axis` reduces the given axis and keeps it as a size-1
dimension (NumPy's `keepdims=True`), so the result still broadcasts cleanly
against the original:

```rust,ignore
let m = Array::from_slice(&gpu, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
let m = m?;

let col_sums = m.sum_axis(0)?;  // sum down each column → shape [1, 3] = [5, 7, 9]
let row_sums = m.sum_axis(1)?;  // sum across each row   → shape [2, 1] = [6, 15]
```

Because the reduced axis stays as size 1, you can immediately use the result as a
broadcast operand — for example, subtracting each column's mean:

```rust,ignore
let n_rows = 2.0f32;
let col_mean = m.sum_axis(0)?.mul(&Array::full(&gpu, 1.0 / n_rows, &[1])?)?;
let centered = m.sub(&col_mean)?;   // [1,3] broadcasts over the 2 rows
```

## Why this matters

Reductions are how you turn a computation into an *answer* — a loss, a norm, a
count, a probability. Every training loop later in this track ends in a reduction
(the scalar loss you differentiate). Keeping them on-device means a whole pipeline
— generate data, transform it, reduce it — runs without a single host round-trip
until the final read.

## Next

- **[Shape and views](shape-and-views.md)** — reshape and transpose without copying, so reductions land on the axis you mean.
- How-to: **[Parallel reduce](../how-to/parallel-reduce.md)** shows the hand-written kernel version for custom reductions.
