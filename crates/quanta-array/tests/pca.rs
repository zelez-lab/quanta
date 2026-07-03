//! PCA end-to-end — the parity example for scikit-learn's `PCA`.
//!
//! Data with a known dominant direction goes in; the top component must
//! align with that direction (|dot| ≈ 1) and `explained_variance` must
//! capture most of the total variance. A host reference (center → covariance
//! → the same checks in plain Rust) pins the numbers.

use quanta_array::Array;

fn gpu() -> quanta::Gpu {
    quanta::init_cpu()
}

/// Deterministic noise in [−1, 1] (Knuth hash).
fn noise(i: usize, seed: u32) -> f32 {
    ((((i as u32).wrapping_mul(2654435761) ^ seed) % 2001) as f32 - 1000.0) / 1000.0
}

/// N points stretched along unit direction `u` (2-D): x = t·u + ε·n_perp.
fn line_cloud_2d(n: usize, u: [f32; 2], eps: f32) -> Vec<f32> {
    let perp = [-u[1], u[0]];
    let mut pts = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = -5.0 + 10.0 * (i as f32) / ((n - 1) as f32); // spread [−5, 5]
        let w = eps * noise(i, 77);
        pts.push(t * u[0] + w * perp[0]);
        pts.push(t * u[1] + w * perp[1]);
    }
    pts
}

/// Host total variance (sum of per-feature sample variances = trace of the
/// sample covariance), f64 accumulate.
fn host_total_variance(data: &[f32], n: usize, d: usize) -> f64 {
    let mut total = 0.0f64;
    for j in 0..d {
        let mean: f64 = (0..n).map(|i| data[i * d + j] as f64).sum::<f64>() / n as f64;
        let var: f64 = (0..n)
            .map(|i| {
                let x = data[i * d + j] as f64 - mean;
                x * x
            })
            .sum::<f64>()
            / (n - 1) as f64;
        total += var;
    }
    total
}

fn check_dominant_direction_2d(g: &quanta::Gpu) {
    let n = 64usize;
    let u = [0.6f32, 0.8]; // known unit axis (3-4-5)
    let pts = line_cloud_2d(n, u, 0.05);
    let data = Array::from_slice(g, &pts, &[n, 2]).unwrap();

    let (components, explained) = data.pca(1).unwrap();
    assert_eq!(components.shape(), &[1, 2]);
    assert_eq!(explained.shape(), &[1]);
    let c = components.to_vec().unwrap();
    let ev = explained.to_vec().unwrap();

    // Top component is unit-length and aligns with the known axis.
    let norm = (c[0] * c[0] + c[1] * c[1]).sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "component norm {norm}");
    let dot = c[0] * u[0] + c[1] * u[1];
    assert!(
        dot.abs() > 0.999,
        "top component misaligned: |dot| = {}",
        dot.abs()
    );

    // It captures (nearly) all the variance: noise is 0.05 vs spread ~10.
    let total = host_total_variance(&pts, n, 2);
    assert!(ev[0] > 0.0);
    let ratio = ev[0] as f64 / total;
    assert!(ratio > 0.99, "explained-variance ratio {ratio}");
    assert!(
        ratio < 1.0 + 1e-3,
        "explained variance exceeds total: {ratio}"
    );
}

fn check_3d_two_components(g: &quanta::Gpu) {
    // 3-D cloud: big spread along x, medium along y, tiny along z.
    let n = 48usize;
    let mut pts = Vec::with_capacity(n * 3);
    for i in 0..n {
        let t = -1.0 + 2.0 * (i as f32) / ((n - 1) as f32);
        pts.push(10.0 * t + 0.01 * noise(i, 1));
        pts.push(3.0 * noise(i, 2));
        pts.push(0.05 * noise(i, 3));
    }
    let data = Array::from_slice(g, &pts, &[n, 3]).unwrap();
    let (components, explained) = data.pca(2).unwrap();
    assert_eq!(components.shape(), &[2, 3]);
    let c = components.to_vec().unwrap();
    let ev = explained.to_vec().unwrap();

    // Component 0 ≈ ±x̂, component 1 ≈ ±ŷ; variances ordered.
    assert!(c[0].abs() > 0.999, "PC0 not along x: {:?}", &c[0..3]);
    assert!(c[4].abs() > 0.99, "PC1 not along y: {:?}", &c[3..6]);
    assert!(
        ev[0] > ev[1] && ev[1] > 0.0,
        "variances not descending: {ev:?}"
    );
    // Components are orthonormal rows.
    let dot: f32 = (0..3).map(|j| c[j] * c[3 + j]).sum();
    assert!(dot.abs() < 1e-3, "components not orthogonal: {dot}");
    // Both together capture ≈ everything (z-noise is negligible).
    let total = host_total_variance(&pts, n, 3);
    let ratio = (ev[0] as f64 + ev[1] as f64) / total;
    assert!(ratio > 0.999, "2-component ratio {ratio}");
}

// ---------------------------------------------------------------------------
// Software-lane tests
// ---------------------------------------------------------------------------

#[test]
fn pca_recovers_dominant_axis_2d() {
    let g = gpu();
    check_dominant_direction_2d(&g);
}

#[test]
fn pca_3d_top_two_components() {
    let g = gpu();
    check_3d_two_components(&g);
}

#[test]
fn pca_matches_host_covariance_eigenvalues() {
    // explained_variance == eigenvalues of the host-computed sample
    // covariance (the sklearn definition), on a generic 4-feature cloud.
    let g = gpu();
    let (n, d) = (32usize, 4usize);
    let pts: Vec<f32> = (0..n * d)
        .map(|i| noise(i, 9) * ((i % d) as f32 + 1.0) + (i % d) as f32)
        .collect();
    let data = Array::from_slice(&g, &pts, &[n, d]).unwrap();
    let (_, explained) = data.pca(d).unwrap();
    let ev = explained.to_vec().unwrap();

    // Host: center, covariance (N−1), trace check + PSD + descending.
    let total = host_total_variance(&pts, n, d);
    let sum_ev: f64 = ev.iter().map(|&x| x as f64).sum();
    assert!(
        (sum_ev - total).abs() <= 1e-3 * (1.0 + total),
        "Σλ = {sum_ev} vs trace(C) = {total}"
    );
    for j in 0..d {
        assert!(
            ev[j] >= -1e-4,
            "covariance eigenvalue {j} negative: {}",
            ev[j]
        );
        if j + 1 < d {
            assert!(ev[j] >= ev[j + 1], "explained_variance not descending");
        }
    }
}

#[test]
fn pca_rejects_bad_args() {
    let g = gpu();
    let data = Array::<f32>::zeros(&g, &[8, 3]).unwrap();
    assert!(data.pca(0).is_err(), "k = 0");
    assert!(data.pca(4).is_err(), "k > D");
    let one = Array::<f32>::zeros(&g, &[1, 3]).unwrap();
    assert!(one.pca(1).is_err(), "N < 2");
    let flat = Array::<f32>::zeros(&g, &[9]).unwrap();
    assert!(flat.pca(1).is_err(), "1-D data");
}

// ---------------------------------------------------------------------------
// Metal lane — the parity claim on hardware.
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
#[test]
fn metal_pca_end_to_end() {
    let g = quanta::init().expect("metal device");
    check_dominant_direction_2d(&g);
    check_3d_two_components(&g);
}
