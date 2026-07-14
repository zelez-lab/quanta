//! Broadcasting ufunc tests vs a CPU numpy-style reference (software lane).

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
    assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= 1e-5 * (1.0 + y.abs()),
            "elem {i}: {x} vs {y}"
        );
    }
}

/// CPU reference: numpy-broadcast `op` over row-major a (shape sa) and
/// b (shape sb).
fn bc_ref(
    a: &[f32],
    sa: &[usize],
    b: &[f32],
    sb: &[usize],
    op: impl Fn(f32, f32) -> f32,
) -> Vec<f32> {
    let r = sa.len().max(sb.len());
    let out_shape: Vec<usize> = (0..r)
        .map(|i| {
            let ad = if i < r - sa.len() {
                1
            } else {
                sa[i - (r - sa.len())]
            };
            let bd = if i < r - sb.len() {
                1
            } else {
                sb[i - (r - sb.len())]
            };
            ad.max(bd)
        })
        .collect();
    let n: usize = out_shape.iter().product();
    // row-major strides for a/b after right-align + 1->0
    let stride = |s: &[usize]| -> Vec<usize> {
        let mut st = vec![0usize; r];
        // contiguous strides of s
        let mut cs = vec![1usize; s.len()];
        for k in (0..s.len().saturating_sub(1)).rev() {
            cs[k] = cs[k + 1] * s[k + 1];
        }
        for (i, sti) in st.iter_mut().enumerate() {
            if i >= r - s.len() {
                let ax = i - (r - s.len());
                *sti = if s[ax] == 1 { 0 } else { cs[ax] };
            }
        }
        st
    };
    let sta = stride(sa);
    let stb = stride(sb);
    let mut os = vec![1usize; r];
    for k in (0..r.saturating_sub(1)).rev() {
        os[k] = os[k + 1] * out_shape[k + 1];
    }
    let mut out = Vec::with_capacity(n);
    for q in 0..n {
        let mut ao = 0usize;
        let mut bo = 0usize;
        for k in 0..r {
            let c = (q / os[k]) % out_shape[k];
            ao += c * sta[k];
            bo += c * stb[k];
        }
        out.push(op(a[ao], b[bo]));
    }
    out
}

#[test]
fn scalar_broadcast() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let s = Array::from_slice(&g, &[10.0f32], &[1]).unwrap();
    let got = a.add(&s).unwrap().to_vec().unwrap();
    let want = bc_ref(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &[10.0], &[1], |x, y| x + y);
    approx(&got, &want);
}

#[test]
fn row_vector_broadcast() {
    let g = gpu();
    // [2,3] * [3] (row vector) -> [2,3]
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let v = Array::from_slice(&g, &[10.0f32, 100.0, 1000.0], &[3]).unwrap();
    let got = a.mul(&v).unwrap().to_vec().unwrap();
    let want = bc_ref(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        &[10.0, 100.0, 1000.0],
        &[3],
        |x, y| x * y,
    );
    approx(&got, &want);
    assert_eq!(got, vec![10.0, 200.0, 3000.0, 40.0, 500.0, 6000.0]);
}

#[test]
fn column_vector_broadcast() {
    let g = gpu();
    // [2,3] + [2,1] (column vector) -> [2,3]
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let c = Array::from_slice(&g, &[10.0f32, 20.0], &[2, 1]).unwrap();
    let got = a.add(&c).unwrap().to_vec().unwrap();
    let want = bc_ref(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        &[10.0, 20.0],
        &[2, 1],
        |x, y| x + y,
    );
    approx(&got, &want);
    assert_eq!(got, vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);
}

#[test]
fn outer_product_via_broadcast() {
    let g = gpu();
    // [3,1] * [1,4] -> [3,4]
    let col = Array::from_slice(&g, &[1.0f32, 2.0, 3.0], &[3, 1]).unwrap();
    let row = Array::from_slice(&g, &[1.0f32, 10.0, 100.0, 1000.0], &[1, 4]).unwrap();
    let got = col.mul(&row).unwrap();
    assert_eq!(got.shape(), &[3, 4]);
    let want = bc_ref(
        &[1.0, 2.0, 3.0],
        &[3, 1],
        &[1.0, 10.0, 100.0, 1000.0],
        &[1, 4],
        |x, y| x * y,
    );
    approx(&got.to_vec().unwrap(), &want);
}

#[test]
fn rank_mismatch_broadcast() {
    let g = gpu();
    // [4] + [2,3,4] -> [2,3,4]
    let v = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    let big: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let b = Array::from_slice(&g, &big, &[2, 3, 4]).unwrap();
    let got = v.add(&b).unwrap();
    assert_eq!(got.shape(), &[2, 3, 4]);
    let want = bc_ref(&[1.0, 2.0, 3.0, 4.0], &[4], &big, &[2, 3, 4], |x, y| x + y);
    approx(&got.to_vec().unwrap(), &want);
}

#[test]
fn incompatible_shapes_error() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0], &[2]).unwrap();
    let b = Array::from_slice(&g, &[1.0f32, 2.0, 3.0], &[3]).unwrap();
    assert!(a.add(&b).is_err());
}

#[test]
fn broadcast_pow_and_min() {
    let g = gpu();
    let base = Array::from_slice(&g, &[2.0f32, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
    let ex = Array::from_slice(&g, &[2.0f32], &[1]).unwrap();
    approx(
        &base.pow(&ex).unwrap().to_vec().unwrap(),
        &[4.0, 9.0, 16.0, 25.0],
    );
    let cap = Array::from_slice(&g, &[3.5f32], &[1]).unwrap();
    approx(
        &base.minimum(&cap).unwrap().to_vec().unwrap(),
        &[2.0, 3.0, 3.5, 3.5],
    );
}
