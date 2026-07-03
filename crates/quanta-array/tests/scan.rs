//! `cumsum_last` — inclusive prefix sum along the last axis.

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
fn cumsum_last_1d() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    let c = a.cumsum_last().unwrap();
    assert_eq!(c.shape(), &[4]);
    assert_eq!(c.to_vec().unwrap(), vec![1.0, 3.0, 6.0, 10.0]);
}

#[test]
fn cumsum_last_per_row() {
    let g = gpu();
    // [2, 3]: each row scanned independently.
    let a = Array::from_slice(&g, &[1.0f32, 1.0, 1.0, 2.0, 0.0, 3.0], &[2, 3]).unwrap();
    let c = a.cumsum_last().unwrap();
    assert_eq!(c.shape(), &[2, 3]);
    assert_eq!(c.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 2.0, 2.0, 5.0]);
}

#[test]
fn cumsum_last_matches_host() {
    let g = gpu();
    let data: Vec<f32> = (0..20).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let a = Array::from_slice(&g, &data, &[4, 5]).unwrap();
    let got = a.cumsum_last().unwrap().to_vec().unwrap();
    let mut want = vec![0.0f32; 20];
    for r in 0..4 {
        let mut acc = 0.0;
        for j in 0..5 {
            acc += data[r * 5 + j];
            want[r * 5 + j] = acc;
        }
    }
    assert_eq!(got, want);
}
