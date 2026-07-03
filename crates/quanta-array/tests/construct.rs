//! Construction + zero-copy view + materialization tests (software lane).

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
fn from_slice_roundtrip() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    assert_eq!(a.shape(), &[2, 3]);
    assert_eq!(a.rank(), 2);
    assert_eq!(a.len(), 6);
    assert!(a.is_contiguous());
    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn zeros_ones_full() {
    let g = gpu();
    assert_eq!(
        Array::<f32>::zeros(&g, &[2, 2]).unwrap().to_vec().unwrap(),
        vec![0.0; 4]
    );
    assert_eq!(
        Array::<f32>::ones(&g, &[3]).unwrap().to_vec().unwrap(),
        vec![1.0; 3]
    );
    assert_eq!(
        Array::full(&g, 7.0f32, &[2]).unwrap().to_vec().unwrap(),
        vec![7.0, 7.0]
    );
}

#[test]
fn construct_int_dtypes() {
    let g = gpu();
    // The numeric builders are generic over ArrayScalar — integer dtypes too.
    assert_eq!(
        Array::<i32>::zeros(&g, &[3]).unwrap().to_vec().unwrap(),
        vec![0; 3]
    );
    assert_eq!(
        Array::<u32>::ones(&g, &[2]).unwrap().to_vec().unwrap(),
        vec![1u32; 2]
    );
    assert_eq!(
        Array::<i32>::arange(&g, 0.0, 2.0, 4)
            .unwrap()
            .to_vec()
            .unwrap(),
        vec![0i32, 2, 4, 6]
    );
    assert_eq!(
        Array::<i32>::eye(&g, 2).unwrap().to_vec().unwrap(),
        vec![1i32, 0, 0, 1]
    );
}

#[test]
fn arange_linspace_eye() {
    let g = gpu();
    assert_eq!(
        Array::<f32>::arange(&g, 0.0, 2.0, 4)
            .unwrap()
            .to_vec()
            .unwrap(),
        vec![0.0, 2.0, 4.0, 6.0]
    );
    assert_eq!(
        Array::<f32>::linspace(&g, 0.0, 1.0, 5)
            .unwrap()
            .to_vec()
            .unwrap(),
        vec![0.0, 0.25, 0.5, 0.75, 1.0]
    );
    assert_eq!(
        Array::<f32>::eye(&g, 3).unwrap().to_vec().unwrap(),
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn reshape_is_contiguous_view() {
    let g = gpu();
    let a = Array::<f32>::arange(&g, 0.0, 1.0, 6).unwrap(); // [0,1,2,3,4,5]
    let b = a.reshape(&[2, 3]).unwrap();
    assert_eq!(b.shape(), &[2, 3]);
    assert_eq!(b.to_vec().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn transpose_gathers_logical_order() {
    let g = gpu();
    // 2x3 row-major: [[0,1,2],[3,4,5]] ; transpose -> 3x2 [[0,3],[1,4],[2,5]]
    let a = Array::from_slice(&g, &[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]).unwrap();
    let t = a.transpose(0, 1).unwrap();
    assert_eq!(t.shape(), &[3, 2]);
    assert!(!t.is_contiguous());
    // to_vec walks logical row-major over the transposed layout.
    assert_eq!(t.to_vec().unwrap(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}

#[test]
fn broadcast_to_view() {
    let g = gpu();
    // shape [1,3] broadcast to [2,3] — both rows see the same data.
    let a = Array::from_slice(&g, &[10.0f32, 20.0, 30.0], &[1, 3]).unwrap();
    let b = a.broadcast_to(&[2, 3]).unwrap();
    assert_eq!(b.shape(), &[2, 3]);
    assert_eq!(
        b.to_vec().unwrap(),
        vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
    );
}

#[test]
fn length_mismatch_errors() {
    let g = gpu();
    assert!(Array::from_slice(&g, &[1.0f32, 2.0], &[3]).is_err());
}
