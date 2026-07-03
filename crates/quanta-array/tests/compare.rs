//! Elementwise comparison → {0,1} masks, and `where_mask` selection.

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

#[test]
fn compare_ops_produce_masks() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 5.0, 3.0, 2.0], &[4]).unwrap();
    let b = Array::from_slice(&g, &[3.0f32, 3.0, 3.0, 3.0], &[4]).unwrap();
    assert_eq!(
        a.lt(&b).unwrap().to_vec().unwrap(),
        vec![1.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(
        a.gt(&b).unwrap().to_vec().unwrap(),
        vec![0.0, 1.0, 0.0, 0.0]
    );
    assert_eq!(
        a.eq(&b).unwrap().to_vec().unwrap(),
        vec![0.0, 0.0, 1.0, 0.0]
    );
    assert_eq!(
        a.ge(&b).unwrap().to_vec().unwrap(),
        vec![0.0, 1.0, 1.0, 0.0]
    );
    assert_eq!(
        a.le(&b).unwrap().to_vec().unwrap(),
        vec![1.0, 0.0, 1.0, 1.0]
    );
    assert_eq!(
        a.ne(&b).unwrap().to_vec().unwrap(),
        vec![1.0, 1.0, 0.0, 1.0]
    );
}

#[test]
fn compare_broadcasts_against_a_scalar() {
    let g = gpu();
    let a = Array::from_slice(&g, &[-1.0f32, 0.0, 2.0, 5.0], &[4]).unwrap();
    let thresh = Array::from_slice(&g, &[1.0f32], &[1]).unwrap(); // broadcast
    // a >= 1.0
    let mask = a.ge(&thresh.broadcast_to(&[4]).unwrap()).unwrap();
    assert_eq!(mask.to_vec().unwrap(), vec![0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn where_mask_selects() {
    let g = gpu();
    // relu via where: max(x, 0) = where(x > 0, x, 0).
    let x = Array::from_slice(&g, &[-2.0f32, 3.0, -1.0, 4.0], &[4]).unwrap();
    let zero = Array::<f32>::zeros(&g, &[4]).unwrap();
    let mask = x.gt(&zero).unwrap();
    let out = mask.where_mask(&x, &zero).unwrap();
    assert_eq!(out.to_vec().unwrap(), vec![0.0, 3.0, 0.0, 4.0]);
}

#[test]
fn where_mask_is_fft_denoise_shape() {
    let g = gpu();
    // Keep spectrum bins whose magnitude ≥ threshold, zero the rest.
    let mag = Array::from_slice(&g, &[0.1f32, 2.0, 0.05, 3.0, 0.2], &[5]).unwrap();
    let spectrum = Array::from_slice(&g, &[10.0f32, 20.0, 30.0, 40.0, 50.0], &[5]).unwrap();
    let thr = Array::from_slice(&g, &[0.5f32], &[1])
        .unwrap()
        .broadcast_to(&[5])
        .unwrap();
    let keep = mag.ge(&thr).unwrap();
    let zero = Array::<f32>::zeros(&g, &[5]).unwrap();
    let filtered = keep.where_mask(&spectrum, &zero).unwrap();
    assert_eq!(filtered.to_vec().unwrap(), vec![0.0, 20.0, 0.0, 40.0, 0.0]);
}
