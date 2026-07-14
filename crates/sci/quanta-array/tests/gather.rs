//! Device-side strided-gather (`contiguous`) correctness (software lane).
//!
//! These force the on-device gather path — a `neg` after a view materializes
//! via `gather_contiguous`, and the *result* is a fresh contiguous field, so
//! the assertion reads it back directly (no host gather masking a kernel bug).

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
fn gather_transpose_2d() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    // transpose -> [3,2] logical [[1,4],[2,5],[3,6]]
    let t = a.transpose(0, 1).unwrap();
    // double via add forces a device gather of the strided view first.
    let r = t.add(&t).unwrap();
    assert_eq!(r.shape(), &[3, 2]);
    assert_eq!(r.to_vec().unwrap(), vec![2.0, 8.0, 4.0, 10.0, 6.0, 12.0]);
}

#[test]
fn gather_permute_3d() {
    let g = gpu();
    // 2x2x2 row-major 0..8
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let a = Array::from_slice(&g, &data, &[2, 2, 2]).unwrap();
    // permute axes (2,0,1): new[i,j,k] = old[j,k,i]
    let p = a.permute(&[2, 0, 1]).unwrap();
    assert_eq!(p.shape(), &[2, 2, 2]);
    let r = p.neg().unwrap(); // device gather, then negate
    // logical order of the permuted view, negated:
    let want: Vec<f32> = p.to_vec().unwrap().iter().map(|x| -x).collect();
    assert_eq!(r.to_vec().unwrap(), want);
}

#[test]
fn gather_broadcast_zero_stride() {
    let g = gpu();
    // [1,3] broadcast to [2,3] — the row axis has stride 0 in the view.
    let a = Array::from_slice(&g, &[10.0f32, 20.0, 30.0], &[1, 3]).unwrap();
    let b = a.broadcast_to(&[2, 3]).unwrap();
    // mul by itself forces a device gather over the zero-stride axis.
    let r = b.mul(&b).unwrap();
    assert_eq!(r.shape(), &[2, 3]);
    assert_eq!(
        r.to_vec().unwrap(),
        vec![100.0, 400.0, 900.0, 100.0, 400.0, 900.0]
    );
}

#[test]
fn gather_matches_host_to_vec() {
    let g = gpu();
    // Cross-check: the device gather and the host to_vec gather agree.
    let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect();
    let a = Array::from_slice(&g, &data, &[2, 3, 4]).unwrap();
    let v = a.transpose(0, 2).unwrap(); // [4,3,2] strided
    let host = v.to_vec().unwrap(); // host gather
    let dev = v.neg().unwrap().to_vec().unwrap(); // device gather (negated)
    let want: Vec<f32> = host.iter().map(|x| -x).collect();
    assert_eq!(dev, want);
}
