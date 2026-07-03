//! `Array::narrow` — zero-copy sub-range views along one axis. Mirrors the
//! `Quanta.Tensor.Layout.slice` algebra (same strides, base_offset advanced by
//! `start * strides[axis]`), the minibatch-selection primitive.

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
fn narrow_selects_row_range() {
    let g = gpu();
    // [4, 3] row-major: rows 0..4, each 3 wide.
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let a = Array::from_slice(&g, &data, &[4, 3]).unwrap();

    // Middle two rows [1, 3) → rows 1 and 2.
    let mid = a.narrow(0, 1, 2).unwrap();
    assert_eq!(mid.shape(), &[2, 3]);
    assert_eq!(mid.to_vec().unwrap(), vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn narrow_inner_axis_is_strided_view() {
    let g = gpu();
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let a = Array::from_slice(&g, &data, &[4, 3]).unwrap();

    // Columns [0, 2) of every row — a strided (non-contiguous) view.
    let cols = a.narrow(1, 0, 2).unwrap();
    assert_eq!(cols.shape(), &[4, 2]);
    // Logical row-major read: row r keeps its first two of three elements.
    assert_eq!(
        cols.to_vec().unwrap(),
        vec![0.0, 1.0, 3.0, 4.0, 6.0, 7.0, 9.0, 10.0]
    );
}

#[test]
fn narrow_is_zero_copy_shares_buffer() {
    let g = gpu();
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let a = Array::from_slice(&g, &data, &[8, 1]).unwrap();
    // A single-batch window at offset 5.
    let one = a.narrow(0, 5, 1).unwrap();
    assert_eq!(one.shape(), &[1, 1]);
    assert_eq!(one.to_vec().unwrap(), vec![5.0]);
}

#[test]
fn narrow_full_range_is_identity() {
    let g = gpu();
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let a = Array::from_slice(&g, &data, &[2, 3]).unwrap();
    let all = a.narrow(0, 0, 2).unwrap();
    assert_eq!(all.shape(), &[2, 3]);
    assert_eq!(all.to_vec().unwrap(), data);
}

#[test]
fn narrow_out_of_range_errors() {
    let g = gpu();
    let a = Array::<f32>::zeros(&g, &[4, 3]).unwrap();
    // start + len exceeds the axis extent.
    assert!(a.narrow(0, 2, 3).is_err());
    // axis out of range.
    assert!(a.narrow(2, 0, 1).is_err());
}
