//! `pad_axis0` — scatter a tensor into a larger zero tensor along axis 0.
//! The adjoint of `narrow(0, start, len)`.

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

/// Host oracle: place `src` ([len, rest]) into zeros([out_rows, rest]) at
/// rows [start, start+len).
fn host_pad(src: &[f32], len: usize, rest: usize, out_rows: usize, start: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_rows * rest];
    for r in 0..len {
        for c in 0..rest {
            out[(start + r) * rest + c] = src[r * rest + c];
        }
    }
    out
}

#[test]
fn pad_axis0_places_block() {
    let g = gpu();
    // src [2, 3] into [5, 3] at rows [1, 3).
    let src: Vec<f32> = (1..=6).map(|i| i as f32).collect();
    let a = Array::from_slice(&g, &src, &[2, 3]).unwrap();
    let padded = a.pad_axis0(5, 1).unwrap();
    assert_eq!(padded.shape(), &[5, 3]);
    assert_eq!(padded.to_vec().unwrap(), host_pad(&src, 2, 3, 5, 1));
}

#[test]
fn pad_axis0_at_start_and_end() {
    let g = gpu();
    let src: Vec<f32> = (0..4).map(|i| (i + 10) as f32).collect(); // [4,1]
    let a = Array::from_slice(&g, &src, &[4, 1]).unwrap();
    // window at the very start
    assert_eq!(
        a.pad_axis0(6, 0).unwrap().to_vec().unwrap(),
        host_pad(&src, 4, 1, 6, 0)
    );
    // window flush at the end
    assert_eq!(
        a.pad_axis0(4, 0).unwrap().to_vec().unwrap(),
        host_pad(&src, 4, 1, 4, 0)
    );
}

#[test]
fn pad_axis0_is_narrow_adjoint_roundtrip() {
    let g = gpu();
    // narrow a block out of a full tensor, pad it back → original block kept,
    // rest zeroed.
    let full: Vec<f32> = (0..24).map(|i| i as f32).collect(); // [8, 3]
    let a = Array::from_slice(&g, &full, &[8, 3]).unwrap();
    let block = a.narrow(0, 2, 3).unwrap(); // rows 2,3,4
    let back = block.pad_axis0(8, 2).unwrap();
    let mut want = vec![0.0f32; 24];
    want[6..15].copy_from_slice(&full[6..15]); // rows 2..5 → flat 6..15
    assert_eq!(back.to_vec().unwrap(), want);
}

#[test]
fn pad_axis0_out_of_range_errors() {
    let g = gpu();
    let a = Array::<f32>::from_slice(&g, &[1.0, 2.0], &[2, 1]).unwrap();
    assert!(a.pad_axis0(2, 1).is_err()); // start+len = 3 > 2
}
