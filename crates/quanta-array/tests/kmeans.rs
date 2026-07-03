//! k-means (Lloyd's algorithm) end-to-end on the array ops — the parity
//! example for scikit's `KMeans`. Proves the assignment (`cdist_sq` +
//! `argmin_last`) and update (one-hot via `eq`/`astype`, then `matmul` for the
//! per-cluster sums and counts) steps compose into a converging clusterer whose
//! GPU assignment matches a host oracle.

use quanta_array::Array;

/// The device these tests run on: the real GPU under a hardware backend
/// feature (metal / vulkan), else the CPU JIT (portable, no GPU needed).
fn gpu() -> quanta::Gpu {
    #[cfg(any(feature = "metal", feature = "vulkan"))]
    {
        quanta::init().expect("a GPU device")
    }
    #[cfg(not(any(feature = "metal", feature = "vulkan")))]
    {
        quanta::init_cpu()
    }
}

/// One Lloyd iteration: assign points to the nearest center, recompute centers
/// as the mean of each cluster.
fn kmeans_step(points: &Array<f32>, centers: &Array<f32>, k: usize, d: usize) -> Array<f32> {
    let g = points.gpu();
    let n = points.shape()[0];
    // assign: nearest centroid per point → [N] u32
    let assign = points.cdist_sq(centers).unwrap().argmin_last().unwrap();
    // one-hot [N, K] as f32: onehot[n, k] = (assign[n] == k)
    let cols = Array::<u32>::arange(&g, 0.0, 1.0, k).unwrap();
    let a2 = assign
        .reshape(&[n, 1])
        .unwrap()
        .broadcast_to(&[n, k])
        .unwrap();
    let c2 = cols
        .reshape(&[1, k])
        .unwrap()
        .broadcast_to(&[n, k])
        .unwrap();
    let onehot: Array<f32> = a2.eq(&c2).unwrap().astype().unwrap();
    // sums [K, D] = onehotᵀ·points ; counts [K, 1] = onehotᵀ·ones
    let oht = onehot.transpose(0, 1).unwrap();
    let sums = oht.matmul(points).unwrap();
    let ones = Array::<f32>::ones(&g, &[n, 1]).unwrap();
    let counts = oht.matmul(&ones).unwrap();
    // new centers = sums / counts
    sums.div(&counts.broadcast_to(&[k, d]).unwrap()).unwrap()
}

fn host_assign(points: &[f32], centers: &[f32], n: usize, k: usize, d: usize) -> Vec<u32> {
    (0..n)
        .map(|i| {
            let mut best = 0u32;
            let mut bd = f32::INFINITY;
            for c in 0..k {
                let dist: f32 = (0..d)
                    .map(|dd| {
                        let x = points[i * d + dd] - centers[c * d + dd];
                        x * x
                    })
                    .sum();
                if dist < bd {
                    bd = dist;
                    best = c as u32;
                }
            }
            best
        })
        .collect()
}

#[test]
fn kmeans_converges_and_matches_host() {
    let g = gpu();
    let (n, k, d) = (12usize, 3usize, 2usize);
    let pts: Vec<f32> = vec![
        0.0, 0.0, 0.5, 0.3, -0.3, 0.2, 0.1, -0.4, // blob near (0, 0)
        5.0, 5.0, 5.3, 4.8, 4.7, 5.2, 5.1, 4.9, // blob near (5, 5)
        0.0, 9.0, 0.2, 9.3, -0.2, 8.8, 0.1, 9.1, // blob near (0, 9)
    ];
    let points = Array::from_slice(&g, &pts, &[n, d]).unwrap();
    let mut centers = Array::from_slice(&g, &[0.0f32, 0.0, 5.0, 5.0, 0.0, 9.0], &[k, d]).unwrap();

    for _ in 0..8 {
        centers = kmeans_step(&points, &centers, k, d);
    }

    // GPU assignment must match the host oracle.
    let gpu_assign = points
        .cdist_sq(&centers)
        .unwrap()
        .argmin_last()
        .unwrap()
        .to_vec()
        .unwrap();
    let host = host_assign(&pts, &centers.to_vec().unwrap(), n, k, d);
    assert_eq!(gpu_assign, host, "GPU assignment must match host");
    // The three blobs cluster cleanly: each group of 4 shares a label.
    assert_eq!(gpu_assign, vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);

    // Centers land near the true blob means.
    let c = centers.to_vec().unwrap();
    let near = |a: f32, b: f32| (a - b).abs() < 0.5;
    for (tx, ty) in [(0.075f32, 0.025), (5.025, 4.975), (0.025, 9.05)] {
        assert!(
            (0..k).any(|kk| near(c[kk * d], tx) && near(c[kk * d + 1], ty)),
            "no center near true mean ({tx}, {ty}); centers = {c:?}"
        );
    }
}
