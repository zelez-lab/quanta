//! `astype` — elementwise dtype conversion.

use quanta_array::Array;

fn gpu() -> quanta::Gpu {
    quanta::init_cpu()
}

#[test]
fn astype_u32_to_f32() {
    let g = gpu();
    let a = Array::from_slice(&g, &[0u32, 1, 5, 255], &[4]).unwrap();
    let f: Array<f32> = a.astype().unwrap();
    assert_eq!(f.shape(), &[4]);
    assert_eq!(f.to_vec().unwrap(), vec![0.0f32, 1.0, 5.0, 255.0]);
}

#[test]
fn astype_f32_to_u32_truncates() {
    let g = gpu();
    let a = Array::from_slice(&g, &[0.0f32, 1.9, 3.2, 7.99], &[4]).unwrap();
    let u: Array<u32> = a.astype().unwrap();
    assert_eq!(u.to_vec().unwrap(), vec![0u32, 1, 3, 7]);
}

#[test]
fn astype_preserves_shape() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1u32, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    let f: Array<f32> = a.astype().unwrap();
    assert_eq!(f.shape(), &[2, 3]);
    assert_eq!(f.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}
