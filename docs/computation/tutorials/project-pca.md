# Project: find principal components (PCA)

> **You'll build:** a standalone Cargo project that runs Principal Component
> Analysis — finding the directions of greatest variance in a dataset — the same
> `sklearn.decomposition.PCA`, on the GPU.
>
> **You'll need:** [arrays](arrays.md) and [linear algebra](linear-algebra.md).
> No autodiff — PCA is a closed-form eigenproblem, not a trained model.

PCA finds the axes along which data varies most. The recipe: center the data,
form its covariance matrix, and take the top eigenvectors of that matrix — those
are the *principal components*, and their eigenvalues are how much variance each
explains. The covariance is symmetric, so a symmetric eigensolver does the work;
`Array::pca` wraps the whole thing.

## 1. Create the project

```sh
cargo new pca
cd pca
```

## 2. Dependencies

```toml
[dependencies]
quanta = { git = "https://github.com/zelez-lab/quanta", features = ["sci", "metal"] }
```

## 3. Make some data

Points scattered along a known direction with a little noise — so we can check
that PCA recovers that direction. Here the data stretches along the `(0.6, 0.8)`
axis. `src/main.rs`:

```rust,ignore
use quanta::sci::Array;

fn main() {
    let gpu = quanta::init().expect("a GPU");
    let n = 200usize;

    // Points along (0.6, 0.8), spread out, with small orthogonal noise.
    let mut xs = Vec::new();
    for i in 0..n {
        let s = (i as f32 / n as f32 - 0.5) * 10.0;      // position along the axis
        let noise = ((i * 7 % 13) as f32 / 13.0 - 0.5) * 0.1;
        xs.push(0.6 * s + 0.8 * noise);                   // x
        xs.push(0.8 * s - 0.6 * noise);                   // y
    }
    let data = Array::from_slice(&gpu, &xs, &[n, 2]).unwrap();
```

## 4. Run PCA

One call. `pca(k)` returns the top `k` components (each a unit direction in
feature space) and the variance each one explains:

```rust,ignore
    let (components, explained) = data.pca(1).unwrap();

    let c = components.to_vec().unwrap();       // [1, 2] — the top direction
    let ev = explained.to_vec().unwrap();       // [1]    — its variance

    println!("top component:      [{:.3}, {:.3}]", c[0], c[1]);
    println!("explained variance: {:.3}", ev[0]);
}
```

```sh
cargo run --release
```

```text
top component:      [0.600, 0.800]
explained variance: 8.375
```

The recovered component points along `(0.6, 0.8)` — the axis the data was built
on — and captures essentially all the variance, because the noise perpendicular
to it is tiny. (The sign is arbitrary: `(−0.6, −0.8)` is the same axis; PCA fixes
one convention so the result is deterministic.)

## 5. What it did under the hood

`pca(k)` is three steps you could write yourself with the array ops:

```rust,ignore
// 1. center: subtract the column means
let mean = data.sum_axis(0).unwrap() /* · (1/n) */;
let centered = data.sub(&mean.broadcast_to(data.shape()).unwrap()).unwrap();

// 2. covariance: Cᵀ·C / (n − 1)   — a symmetric D×D matrix
let cov = centered.transpose(0, 1).unwrap()
    .matmul(&centered).unwrap() /* · (1/(n-1)) */;

// 3. eigendecompose and keep the top k
let (eigenvalues, eigenvectors) = cov.eigh_symmetric().unwrap();
```

`eigh_symmetric` is a **cyclic Jacobi** eigensolver: it repeatedly applies
rotations that shrink the off-diagonal entries until the matrix is
(numerically) diagonal — the diagonal holds the eigenvalues, the accumulated
rotations hold the eigenvectors. Each sweep's rotations become one orthogonal
matrix applied with a GPU matmul, so the heavy arithmetic runs on the device.

## 6. What you built

A PCA that recovers the structure of a dataset from its covariance — dimensional
reduction, decorrelation, and the first step of many classical pipelines. The
symmetric eigensolver it's built on (`Array::eigh_symmetric`) is reusable on its
own for any symmetric matrix.

- Coming from scikit-learn? `data.pca(k)` is `PCA(n_components=k).fit(data)`;
  `components` matches `pca.components_` (up to per-row sign) and `explained` is
  `pca.explained_variance_`.
- Go further: project the data onto the components (`data_centered · componentsᵀ`)
  for a low-dimensional embedding, or use the explained-variance ratios to pick
  how many components to keep.
