//! `concat_axis0` / `stack_axis0` — join tensors along axis 0.

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
fn concat_two_matrices() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Array::from_slice(&g, &[5.0f32, 6.0], &[1, 2]).unwrap();
    let c = Array::concat_axis0(&[&a, &b]).unwrap();
    assert_eq!(c.shape(), &[3, 2]);
    assert_eq!(c.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn concat_three_parts_preserves_order() {
    let g = gpu();
    let a = Array::from_slice(&g, &[0.0f32, 1.0], &[1, 2]).unwrap();
    let b = Array::from_slice(&g, &[2.0f32, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
    let c = Array::from_slice(&g, &[6.0f32, 7.0], &[1, 2]).unwrap();
    let out = Array::concat_axis0(&[&a, &b, &c]).unwrap();
    assert_eq!(out.shape(), &[4, 2]);
    assert_eq!(
        out.to_vec().unwrap(),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    );
}

#[test]
fn concat_single_is_identity() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0], &[3, 1]).unwrap();
    let out = Array::concat_axis0(&[&a]).unwrap();
    assert_eq!(out.shape(), &[3, 1]);
    assert_eq!(out.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn concat_mismatched_trailing_shape_errors() {
    let g = gpu();
    let a = Array::<f32>::zeros(&g, &[2, 3]).unwrap();
    let b = Array::<f32>::zeros(&g, &[1, 2]).unwrap();
    assert!(Array::concat_axis0(&[&a, &b]).is_err());
}

#[test]
fn concat_is_inverse_of_narrow() {
    let g = gpu();
    let full: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let x = Array::from_slice(&g, &full, &[6, 2]).unwrap();
    // split into [0,2), [2,5), [5,6) then re-join → original.
    let p0 = x.narrow(0, 0, 2).unwrap();
    let p1 = x.narrow(0, 2, 3).unwrap();
    let p2 = x.narrow(0, 5, 1).unwrap();
    let rejoined = Array::concat_axis0(&[&p0, &p1, &p2]).unwrap();
    assert_eq!(rejoined.shape(), &[6, 2]);
    assert_eq!(rejoined.to_vec().unwrap(), full);
}

#[test]
fn stack_adds_a_leading_axis() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0], &[3]).unwrap();
    let b = Array::from_slice(&g, &[4.0f32, 5.0, 6.0], &[3]).unwrap();
    let s = Array::stack_axis0(&[&a, &b]).unwrap();
    assert_eq!(s.shape(), &[2, 3]);
    assert_eq!(s.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn stack_mismatched_shape_errors() {
    let g = gpu();
    let a = Array::<f32>::zeros(&g, &[3]).unwrap();
    let b = Array::<f32>::zeros(&g, &[4]).unwrap();
    assert!(Array::stack_axis0(&[&a, &b]).is_err());
}
