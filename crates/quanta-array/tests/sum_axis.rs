//! `sum_axis` tests vs a CPU reference (software lane).

use quanta_array::Array;

fn gpu() -> quanta::Gpu {
    quanta::init_cpu()
}

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= 1e-4 * (1.0 + y.abs()),
            "elem {i}: {x} vs {y}"
        );
    }
}

/// Host reference: sum a row-major [m,n] over `axis`, keepdims.
fn host_sum_axis(data: &[f32], m: usize, n: usize, axis: usize) -> Vec<f32> {
    if axis == 0 {
        // → [1, n]
        let mut out = vec![0.0f32; n];
        for i in 0..m {
            for j in 0..n {
                out[j] += data[i * n + j];
            }
        }
        out
    } else {
        // → [m, 1]
        let mut out = vec![0.0f32; m];
        for i in 0..m {
            for j in 0..n {
                out[i] += data[i * n + j];
            }
        }
        out
    }
}

#[test]
fn sum_axis0_2d() {
    let g = gpu();
    let (m, n) = (3usize, 4usize);
    let data: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
    let a = Array::from_slice(&g, &data, &[m, n]).unwrap();
    let r = a.sum_axis(0).unwrap();
    assert_eq!(r.shape(), &[1, n]);
    approx(&r.to_vec().unwrap(), &host_sum_axis(&data, m, n, 0));
}

#[test]
fn sum_axis1_2d() {
    let g = gpu();
    let (m, n) = (3usize, 4usize);
    let data: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.5 - 2.0).collect();
    let a = Array::from_slice(&g, &data, &[m, n]).unwrap();
    let r = a.sum_axis(1).unwrap();
    assert_eq!(r.shape(), &[m, 1]);
    approx(&r.to_vec().unwrap(), &host_sum_axis(&data, m, n, 1));
}

#[test]
fn sum_axis_3d_middle() {
    // [2,3,2] summed over axis 1 → [2,1,2]; check against a manual fold.
    let g = gpu();
    let dims = [2usize, 3, 2];
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let a = Array::from_slice(&g, &data, &dims).unwrap();
    let r = a.sum_axis(1).unwrap();
    assert_eq!(r.shape(), &[2, 1, 2]);
    // want[i,0,k] = Σ_j data[i,j,k]  (output shape [2,1,2] → 4 elements)
    let mut want = vec![0.0f32; 4];
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..2 {
                want[i * 2 + k] += data[(i * 3 + j) * 2 + k];
            }
        }
    }
    approx(&r.to_vec().unwrap(), &want);
}

#[test]
fn sum_axis_strided_input() {
    // Transposed (strided) input must gather correctly before reducing.
    let g = gpu();
    let (m, n) = (2usize, 3usize);
    let data: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
    let a = Array::from_slice(&g, &data, &[m, n]).unwrap();
    let at = a.transpose(0, 1).unwrap(); // [n, m], strided
    let r = at.sum_axis(0).unwrap(); // → [1, m]
    assert_eq!(r.shape(), &[1, m]);
    // Σ over the (transposed) axis 0 = Σ over original axis 1 = row sums.
    let mut want = vec![0.0f32; m];
    for i in 0..m {
        for j in 0..n {
            want[i] += data[i * n + j];
        }
    }
    approx(&r.to_vec().unwrap(), &want);
}

#[test]
fn sum_axis_out_of_range_errors() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0], &[2]).unwrap();
    assert!(a.sum_axis(1).is_err());
}
