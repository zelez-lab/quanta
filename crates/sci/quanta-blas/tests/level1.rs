//! Level-1 BLAS differential tests: GPU ops vs the pure-Rust reference
//! oracle, on the software lane.

#![cfg(feature = "gpu")]

use quanta_blas::reference;

/// The device these tests run on: the real GPU under a hardware backend
/// feature (gpu-metal / gpu-vulkan), else the CPU JIT (portable, no GPU needed).
fn gpu() -> quanta::Gpu {
    #[cfg(any(feature = "gpu-metal", feature = "gpu-vulkan"))]
    {
        quanta::init().expect("a GPU device")
    }
    #[cfg(not(any(feature = "gpu-metal", feature = "gpu-vulkan")))]
    {
        quanta::init_cpu()
    }
}

const SIZES: [usize; 5] = [1, 7, 256, 257, 1000];

/// Build a deterministic f32 vector.
fn vec_a(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 13) as f32 - 6.0).collect()
}
fn vec_b(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 7) as f32 * 0.5 - 1.5).collect()
}

#[test]
fn scal_matches_reference() {
    let g = gpu();
    for &n in &SIZES {
        let host = vec_a(n);
        let alpha = 2.5f32;

        let f = g.field::<f32>(n).unwrap();
        f.write(&host).unwrap();
        quanta_blas::scal(&g, alpha, &f).unwrap();
        let got = f.read().unwrap();

        let mut want = host.clone();
        reference::scal(alpha, &mut want);
        assert_eq!(got, want, "scal n={n}");
    }
}

#[test]
fn axpy_matches_reference() {
    let g = gpu();
    for &n in &SIZES {
        let xh = vec_a(n);
        let yh = vec_b(n);
        let alpha = -1.25f32;

        let x = g.field::<f32>(n).unwrap();
        let y = g.field::<f32>(n).unwrap();
        x.write(&xh).unwrap();
        y.write(&yh).unwrap();
        quanta_blas::axpy(&g, alpha, &x, &y).unwrap();
        let got = y.read().unwrap();

        let mut want = yh.clone();
        reference::axpy(alpha, &xh, &mut want);
        assert_eq!(got, want, "axpy n={n}");
    }
}

#[test]
fn dot_matches_reference() {
    let g = gpu();
    for &n in &SIZES {
        let xh = vec_a(n);
        let yh = vec_b(n);

        let x = g.field::<f32>(n).unwrap();
        let y = g.field::<f32>(n).unwrap();
        x.write(&xh).unwrap();
        y.write(&yh).unwrap();
        let got = quanta_blas::dot(&g, &x, &y).unwrap();

        let want = reference::dot(&xh, &yh);
        // tree-order reduction vs f64 reference: relative tolerance
        assert!(
            (got - want).abs() <= 1e-3 * (1.0 + want.abs()),
            "dot n={n}: {got} vs {want}"
        );
    }
}

#[test]
fn nrm2_matches_reference() {
    let g = gpu();
    for &n in &SIZES {
        let xh = vec_a(n);
        let x = g.field::<f32>(n).unwrap();
        x.write(&xh).unwrap();
        let got = quanta_blas::nrm2(&g, &x).unwrap();
        let want = reference::nrm2(&xh);
        assert!(
            (got - want).abs() <= 1e-3 * (1.0 + want.abs()),
            "nrm2 n={n}: {got} vs {want}"
        );
    }
}

#[test]
fn length_mismatch_errors() {
    let g = gpu();
    let x = g.field::<f32>(4).unwrap();
    let y = g.field::<f32>(5).unwrap();
    assert!(quanta_blas::axpy(&g, 1.0, &x, &y).is_err());
    assert!(quanta_blas::dot(&g, &x, &y).is_err());
}
