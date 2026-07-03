//! gather_rows / scatter_rows_add tests (software lane).
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

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= 1e-5 * (1.0 + y.abs()),
            "elem {i}: {x} vs {y}"
        );
    }
}

#[test]
fn gather_rows_basic() {
    let g = gpu();
    // table[3,4]; pick col idx[i] from each row.
    let t: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let table = Array::from_slice(&g, &t, &[3, 4]).unwrap();
    let idx = Array::from_slice(&g, &[2u32, 0, 3], &[3]).unwrap();
    let out = table.gather_rows(&idx).unwrap();
    assert_eq!(out.shape(), &[3]);
    // row0 col2 = 2 ; row1 col0 = 4 ; row2 col3 = 11
    approx(&out.to_vec().unwrap(), &[2.0, 4.0, 11.0]);
}

#[test]
fn scatter_rows_basic() {
    let g = gpu();
    let grad = Array::from_slice(&g, &[5.0f32, 7.0, 9.0], &[3]).unwrap();
    let idx = Array::from_slice(&g, &[2u32, 0, 3], &[3]).unwrap();
    let out = grad.scatter_rows_add(&idx, 4).unwrap();
    assert_eq!(out.shape(), &[3, 4]);
    // row0: 5 at col2; row1: 7 at col0; row2: 9 at col3
    approx(
        &out.to_vec().unwrap(),
        &[0.0, 0.0, 5.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0],
    );
}

#[test]
fn gather_scatter_adjoint() {
    // <gather(table), g> == <table, scatter(g)>  for random table, g, idx.
    let g = gpu();
    let (n, c) = (5usize, 6usize);
    let t: Vec<f32> = (0..n * c).map(|i| (i % 7) as f32 - 3.0).collect();
    let gr: Vec<f32> = (0..n).map(|i| ((i * 3) % 5) as f32 - 2.0).collect();
    let idx_v = vec![1u32, 5, 0, 3, 2];
    let table = Array::from_slice(&g, &t, &[n, c]).unwrap();
    let gg = Array::from_slice(&g, &gr, &[n]).unwrap();
    let idx = Array::from_slice(&g, &idx_v, &[n]).unwrap();

    let gathered = table.gather_rows(&idx).unwrap().to_vec().unwrap();
    let scattered = gg.scatter_rows_add(&idx, c).unwrap().to_vec().unwrap();

    let lhs: f32 = gathered.iter().zip(gr.iter()).map(|(a, b)| a * b).sum();
    let rhs: f32 = t.iter().zip(scattered.iter()).map(|(a, b)| a * b).sum();
    assert!(
        (lhs - rhs).abs() <= 1e-4 * (1.0 + rhs.abs()),
        "adjoint: {lhs} vs {rhs}"
    );
}

#[test]
fn max_axis_last_basic() {
    let g = gpu();
    // [3,4]; row max
    let t = vec![
        1.0f32, 5.0, 2.0, 3.0, // max 5
        -1.0, -3.0, -2.0, -0.5, // max -0.5
        4.0, 4.0, 4.0, 4.0, // max 4
    ];
    let m = Array::from_slice(&g, &t, &[3, 4]).unwrap();
    let out = m.max_axis_last().unwrap();
    assert_eq!(out.shape(), &[3, 1]);
    approx(&out.to_vec().unwrap(), &[5.0, -0.5, 4.0]);
}

#[test]
fn argmax_last_basic() {
    let g = gpu();
    let t = vec![
        1.0f32, 5.0, 2.0, 3.0, // argmax 1
        -1.0, -3.0, -2.0, -0.5, // argmax 3
        7.0, 4.0, 7.0, 4.0, // tie at 0 and 2 → first wins → 0
    ];
    let m = Array::from_slice(&g, &t, &[3, 4]).unwrap();
    let out = m.argmax_last().unwrap();
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.to_vec().unwrap(), vec![1u32, 3, 0]);
}

#[test]
fn min_axis_last_basic() {
    let g = gpu();
    let t = vec![
        1.0f32, 5.0, 2.0, 3.0, // min 1
        -1.0, -3.0, -2.0, -0.5, // min -3
        4.0, 4.0, 4.0, 4.0, // min 4
    ];
    let m = Array::from_slice(&g, &t, &[3, 4]).unwrap();
    let out = m.min_axis_last().unwrap();
    assert_eq!(out.shape(), &[3, 1]);
    approx(&out.to_vec().unwrap(), &[1.0, -3.0, 4.0]);
}

#[test]
fn argmin_last_basic() {
    let g = gpu();
    let t = vec![
        1.0f32, 5.0, 2.0, 3.0, // argmin 0
        -1.0, -3.0, -2.0, -0.5, // argmin 1
        4.0, 7.0, 4.0, 7.0, // tie at 0 and 2 → first wins → 0
    ];
    let m = Array::from_slice(&g, &t, &[3, 4]).unwrap();
    let out = m.argmin_last().unwrap();
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.to_vec().unwrap(), vec![0u32, 1, 0]);
}
