//! `cdist_sq` — pairwise squared Euclidean distance, the k-means metric.

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

fn host_cdist(p: &[f32], c: &[f32], n: usize, d: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * k];
    for i in 0..n {
        for j in 0..k {
            let mut s = 0.0f32;
            for dd in 0..d {
                let diff = p[i * d + dd] - c[j * d + dd];
                s += diff * diff;
            }
            out[i * k + j] = s;
        }
    }
    out
}

#[test]
fn cdist_sq_matches_host() {
    let g = gpu();
    let p = vec![0.0f32, 0.0, 1.0, 1.0, 2.0, 0.0]; // [3, 2]
    let c = vec![0.0f32, 0.0, 2.0, 2.0]; // [2, 2]
    let (n, d, k) = (3, 2, 2);
    let pa = Array::from_slice(&g, &p, &[n, d]).unwrap();
    let ca = Array::from_slice(&g, &c, &[k, d]).unwrap();
    let dist = pa.cdist_sq(&ca).unwrap();
    assert_eq!(dist.shape(), &[n, k]);
    assert_eq!(dist.to_vec().unwrap(), host_cdist(&p, &c, n, d, k));
}

#[test]
fn cdist_then_argmin_is_kmeans_assignment() {
    // The k-means assignment step: assign each point to its nearest centroid.
    let g = gpu();
    // 4 points near two clusters; 2 centroids at (0,0) and (10,10).
    let points = vec![
        0.5f32, 0.5, // near c0
        9.0, 11.0, // near c1
        1.0, -0.5, // near c0
        10.5, 9.5, // near c1
    ];
    let centers = vec![0.0f32, 0.0, 10.0, 10.0];
    let pa = Array::from_slice(&g, &points, &[4, 2]).unwrap();
    let ca = Array::from_slice(&g, &centers, &[2, 2]).unwrap();
    let assign = pa.cdist_sq(&ca).unwrap().argmin_last().unwrap();
    assert_eq!(assign.to_vec().unwrap(), vec![0u32, 1, 0, 1]);
}

#[test]
fn cdist_sq_dimension_mismatch_errors() {
    let g = gpu();
    let p = Array::<f32>::zeros(&g, &[3, 2]).unwrap();
    let c = Array::<f32>::zeros(&g, &[2, 3]).unwrap(); // D mismatch
    assert!(p.cdist_sq(&c).is_err());
}
