//! Linear-algebra tests for Array<f32> (matmul/dot/norm), software lane.

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

/// Host matmul reference (row-major), f64 accumulate.
fn host_matmul(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += (a[row * k + p] as f64) * (b[p * n + col] as f64);
            }
            c[row * n + col] = acc as f32;
        }
    }
    c
}

fn approx(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-3 * (1.0 + w.abs()),
            "{ctx}: entry {i}: {g} vs {w}"
        );
    }
}

#[test]
fn matmul_square() {
    let g = gpu();
    // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Array::from_slice(&g, &[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    approx(
        &c.to_vec().unwrap(),
        &[19.0, 22.0, 43.0, 50.0],
        "matmul_square",
    );
}

#[test]
fn matmul_rectangular() {
    let g = gpu();
    let (m, k, n) = (3usize, 4usize, 2usize);
    let ah: Vec<f32> = (0..m * k).map(|i| (i % 5) as f32 - 2.0).collect();
    let bh: Vec<f32> = (0..k * n).map(|i| (i % 3) as f32 - 1.0).collect();
    let a = Array::from_slice(&g, &ah, &[m, k]).unwrap();
    let b = Array::from_slice(&g, &bh, &[k, n]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[m, n]);
    approx(
        &c.to_vec().unwrap(),
        &host_matmul(m, n, k, &ah, &bh),
        "matmul_rect",
    );
}

#[test]
fn matmul_identity() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let id = Array::<f32>::eye(&g, 3).unwrap();
    let c = a.matmul(&id).unwrap();
    approx(
        &c.to_vec().unwrap(),
        &a.to_vec().unwrap(),
        "matmul_identity",
    );
}

#[test]
fn matmul_on_transposed_view() {
    let g = gpu();
    // A is 2×3; Aᵀ is 3×2. (Aᵀ)·A is 3×3. Exercises the device-gather of a
    // strided operand before the gemm.
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let at = a.transpose(0, 1).unwrap(); // 3×2 strided
    let c = at.matmul(&a).unwrap();
    assert_eq!(c.shape(), &[3, 3]);
    // reference: gather Aᵀ to contiguous, multiply
    let at_host = at.to_vec().unwrap();
    let a_host = a.to_vec().unwrap();
    approx(
        &c.to_vec().unwrap(),
        &host_matmul(3, 3, 2, &at_host, &a_host),
        "matmul_transposed",
    );
}

#[test]
fn matmul_inner_dim_mismatch_errors() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert!(a.matmul(&b).is_err(), "3 != 2 inner dim");
}

#[test]
fn matmul_rank_error() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0], &[3]).unwrap(); // 1-D
    let b = Array::from_slice(&g, &[1.0f32, 2.0, 3.0], &[3]).unwrap();
    assert!(a.matmul(&b).is_err(), "1-D matmul must error");
}

#[test]
fn dot_vectors() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0], &[3]).unwrap();
    let b = Array::from_slice(&g, &[4.0f32, 5.0, 6.0], &[3]).unwrap();
    let d = a.dot(&b).unwrap(); // 4+10+18 = 32
    assert!((d - 32.0).abs() <= 1e-4, "dot {d}");
}

#[test]
fn dot_rank_and_length_errors() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0], &[3]).unwrap();
    let short = Array::from_slice(&g, &[1.0f32, 2.0], &[2]).unwrap();
    assert!(a.dot(&short).is_err(), "length mismatch");
    let mat = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert!(a.dot(&mat).is_err(), "2-D dot must error");
}

#[test]
fn norm_l2() {
    let g = gpu();
    let a = Array::from_slice(&g, &[3.0f32, 4.0], &[2]).unwrap();
    assert!((a.norm().unwrap() - 5.0).abs() <= 1e-4);
    // shape doesn't matter — flattens
    let m = Array::from_slice(&g, &[1.0f32, 2.0, 2.0, 4.0], &[2, 2]).unwrap();
    assert!((m.norm().unwrap() - 5.0).abs() <= 1e-4); // sqrt(1+4+4+16)=5
}
