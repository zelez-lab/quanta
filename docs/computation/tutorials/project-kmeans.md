# Project: cluster points (k-means)

> **You'll build:** a standalone Cargo project that groups 2-D points into `k`
> clusters with Lloyd's algorithm — the same k-means scikit-learn ships, running
> entirely on the GPU.
>
> **You'll need:** [arrays](arrays.md), [shape and views](shape-and-views.md),
> and [linear algebra](linear-algebra.md). No autodiff — k-means isn't trained
> by gradient descent; it alternates two closed-form steps until it settles.

k-means partitions `N` points into `k` clusters, each represented by its
centroid. Lloyd's algorithm repeats two steps until the centroids stop moving:

1. **Assign** each point to its nearest centroid.
2. **Update** each centroid to the mean of the points assigned to it.

Both steps are pure array algebra — no loops over points, no gradients.

## 1. Create the project

```sh
cargo new kmeans
cd kmeans
```

## 2. Dependencies

Just Quanta — pick `metal` on Apple silicon, `vulkan` on Linux/Windows, or the
software backend to run without a GPU. Edit `Cargo.toml`:

```toml
[dependencies]
quanta = { git = "https://github.com/zelez-lab/quanta", features = ["sci", "metal"] }
```

## 3. Make some data

Three well-separated blobs of 2-D points, stacked into an `[N, 2]` array. In a
real program these would be your dataset; here we hand-write them so the result
is easy to eyeball. `src/main.rs`:

```rust,ignore
use quanta::sci::Array;

fn main() {
    let gpu = quanta::init().expect("a GPU");
    let (k, d) = (3usize, 2usize);

    // 12 points: four around (0,0), four around (5,5), four around (0,9).
    let pts: Vec<f32> = vec![
        0.0, 0.0,  0.5, 0.3,  -0.3, 0.2,  0.1, -0.4,
        5.0, 5.0,  5.3, 4.8,   4.7, 5.2,  5.1,  4.9,
        0.0, 9.0,  0.2, 9.3,  -0.2, 8.8,  0.1,  9.1,
    ];
    let n = pts.len() / d;
    let points = Array::from_slice(&gpu, &pts, &[n, d]).unwrap();

    // Initial centroids — the first point of each blob. (Real code would use
    // k-means++ or random points; a fixed init keeps this deterministic.)
    let mut centers =
        Array::from_slice(&gpu, &[0.0f32, 0.0, 5.0, 5.0, 0.0, 9.0], &[k, d]).unwrap();
```

## 4. The assign step

`cdist_sq` gives the squared distance from every point to every centroid as an
`[N, k]` matrix; `argmin_last` reads off the nearest centroid per row. That's the
whole assignment:

```rust,ignore
    // one Lloyd iteration
    for _ in 0..10 {
        // [N] u32: index of the nearest centroid for each point
        let assign = points.cdist_sq(&centers).unwrap().argmin_last().unwrap();
```

## 5. The update step

The new centroid of cluster `k` is the mean of the points assigned to it. The
trick that keeps this branch-free is a **one-hot** matrix `onehot[n, k] = 1` when
point `n` belongs to cluster `k`, else `0`. Then:

- `onehotᵀ · points` sums the points in each cluster → `[k, d]`,
- `onehotᵀ · ones` counts them → `[k, 1]`,
- dividing gives the per-cluster mean.

Build the one-hot by comparing the assignment against `0..k`, then convert the
`{0,1}` mask to `f32` so it can feed a matmul:

```rust,ignore
        // one-hot [N, k] as f32
        let cols = Array::<u32>::arange(&gpu, 0.0, 1.0, k).unwrap();
        let a2 = assign.reshape(&[n, 1]).unwrap().broadcast_to(&[n, k]).unwrap();
        let c2 = cols.reshape(&[1, k]).unwrap().broadcast_to(&[n, k]).unwrap();
        let onehot: Array<f32> = a2.eq(&c2).unwrap().astype().unwrap();

        // sums [k, d] and counts [k, 1] via one-hotᵀ matmuls
        let oht = onehot.transpose(0, 1).unwrap();
        let sums = oht.matmul(&points).unwrap();
        let ones = Array::<f32>::ones(&gpu, &[n, 1]).unwrap();
        let counts = oht.matmul(&ones).unwrap();

        // new centers = sums / counts  (broadcast [k,1] over [k,d])
        centers = sums.div(&counts.broadcast_to(&[k, d]).unwrap()).unwrap();
    }
```

## 6. Read the result

After a handful of iterations the centroids sit on the blob means. Print the
final assignment:

```rust,ignore
    let assign = points.cdist_sq(&centers).unwrap().argmin_last().unwrap();
    println!("centers: {:?}", centers.to_vec().unwrap());
    println!("labels:  {:?}", assign.to_vec().unwrap());
}
```

```sh
cargo run --release
```

```text
centers: [0.07, 0.02, 5.03, 4.98, 0.03, 9.05]
labels:  [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
```

The three blobs each got their own label, and the centroids converged to their
means — a correct clustering, computed with nothing but distance, argmin, a
one-hot, and two matmuls.

## 7. What you built

An unsupervised clusterer, entirely in array algebra:

- **assign** = `cdist_sq` + `argmin_last`,
- **update** = one-hot (`eq` + `astype`) + `matmul` + `div`.

The same shape scales straight up: swap the toy blobs for real feature vectors,
raise `k`, and the loop is unchanged. For higher dimensions or many clusters,
the one-hot matmul is still the whole update — the GPU does the per-cluster
reduction in one pass.

- Coming from scikit-learn? `cdist_sq` is `pairwise_distances(squared=True)`,
  `argmin_last` is `.argmin(axis=1)`, and the one-hot matmul is what
  `KMeans._update_centers` does under the hood.
- Next: k-means++ initialization (a weighted `gather` from the points), or an
  early-stop when the centroids move less than a tolerance.
