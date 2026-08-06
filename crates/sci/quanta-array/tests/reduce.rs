//! Whole-array reduction tests vs CPU reference (software lane).

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

#[test]
fn reduce_i32() {
    let g = gpu();
    let a = Array::from_slice(&g, &[3i32, -1, 7, 2, -5, 4], &[2, 3]).unwrap();
    assert_eq!(a.sum().unwrap(), 10);
    assert_eq!(a.min().unwrap(), -5);
    assert_eq!(a.max().unwrap(), 7);
}

#[test]
fn reduce_u32() {
    let g = gpu();
    let a = Array::from_slice(&g, &[3u32, 1, 7, 2, 5, 4], &[6]).unwrap();
    assert_eq!(a.sum().unwrap(), 22);
    assert_eq!(a.min().unwrap(), 1);
    assert_eq!(a.max().unwrap(), 7);
}

#[test]
fn sum_device_matches_sum_bitwise() {
    let g = gpu();
    // Sizes crossing every pass boundary: unpadded single block, aligned
    // single block, 2-block, aligned multi-pass, 3-pass with padding.
    for n in [1usize, 5, 256, 257, 4096, 65536, 70000] {
        let data: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) * 0.37 - 11.0).collect();
        let a = Array::from_slice(&g, &data, &[n]).unwrap();
        let host = a.sum().unwrap();
        let dev = a.sum_device().unwrap();
        assert_eq!(dev.shape(), &[1]);
        let dv = dev.to_vec().unwrap()[0];
        // Identical pass structure over identical padded values —
        // bit-equal, not merely close.
        assert_eq!(
            dv.to_bits(),
            host.to_bits(),
            "n={n}: device {dv} vs host {host}"
        );
    }
}

#[test]
fn sum_device_strided_and_int() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let t = a.transpose(0, 1).unwrap();
    assert_eq!(t.sum_device().unwrap().to_vec().unwrap()[0], 21.0);
    let b = Array::from_slice(&g, &[3i32, -1, 7, 2, -5, 4], &[6]).unwrap();
    assert_eq!(b.sum_device().unwrap().to_vec().unwrap()[0], 10);
    let u = Array::from_slice(&g, &[3u32, 1, 7, 2, 5, 4], &[6]).unwrap();
    assert_eq!(u.sum_device().unwrap().to_vec().unwrap()[0], 22);
}

#[test]
fn sum_device_after_ufunc_chain() {
    let g = gpu();
    let data: Vec<f32> = (0..512).map(|i| i as f32 * 0.25).collect();
    let a = Array::from_slice(&g, &data, &[512]).unwrap();
    let b = a.mul(&a).unwrap(); // pending deferred producer
    let dev = b.sum_device().unwrap(); // encodes behind it — no flush yet
    let host = b.sum().unwrap(); // the host reduce completes the lane
    assert_eq!(dev.to_vec().unwrap()[0].to_bits(), host.to_bits());
}

// Note: f64 has the math ufuncs (FloatScalar) but no device reduce —
// quanta-prims only provides reduces for u32/i32/f32, so `f64_array.sum()`
// does not compile. That's the honest boundary, not a silent fallback.
//
// Empty arrays aren't tested here because the layout layer rejects a
// zero-extent shape at construction (`ZeroExtent`), so an empty `Array`
// is unrepresentable; the empty guards in `sum`/`min`/`max` remain as
// defensive cover for any future empty-view path.
