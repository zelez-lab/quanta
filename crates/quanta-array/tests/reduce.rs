//! Whole-array reduction tests vs CPU reference (software lane).

use quanta_array::Array;

fn gpu() -> quanta::Gpu {
    quanta::init_cpu()
}

#[test]
fn sum_mean() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    let s = a.sum().unwrap();
    assert!((s - 15.0).abs() <= 1e-4, "sum {s}");
    let m = a.mean().unwrap();
    assert!((m - 3.0).abs() <= 1e-4, "mean {m}");
}

#[test]
fn min_max() {
    let g = gpu();
    let a = Array::from_slice(&g, &[3.0f32, -1.0, 7.0, 2.0, -5.0, 4.0], &[2, 3]).unwrap();
    assert_eq!(a.min().unwrap(), -5.0);
    assert_eq!(a.max().unwrap(), 7.0);
}

#[test]
fn sum_2d_matches_host_fold() {
    let g = gpu();
    let data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5).collect();
    let a = Array::from_slice(&g, &data, &[16, 16]).unwrap();
    let want: f32 = data.iter().sum();
    let got = a.sum().unwrap();
    // tree-order reduction → allow a small relative drift
    assert!(
        (got - want).abs() <= 1e-3 * (1.0 + want.abs()),
        "sum {got} vs {want}"
    );
}

#[test]
fn reduce_on_strided_view() {
    let g = gpu();
    // transpose then reduce — to_vec gathers logical order; sum is
    // order-independent so it must match the original.
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let t = a.transpose(0, 1).unwrap();
    assert!((t.sum().unwrap() - 21.0).abs() <= 1e-4);
    assert_eq!(t.min().unwrap(), 1.0);
    assert_eq!(t.max().unwrap(), 6.0);
}

#[test]
fn reduce_after_ufunc() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Array::ones(&g, &[2, 2]).unwrap();
    let c = a.add(&b).unwrap(); // [2,3,4,5]
    assert!((c.sum().unwrap() - 14.0).abs() <= 1e-4);
    assert_eq!(c.max().unwrap(), 5.0);
}
