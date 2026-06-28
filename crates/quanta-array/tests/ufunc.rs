//! Elementwise ufunc tests — quanta-array vs a CPU reference (software lane).

use quanta_array::Array;

fn gpu() -> quanta::Gpu {
    quanta::init_cpu()
}

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= 1e-5 * (1.0 + y.abs()),
            "elem {i}: {x} vs {y}"
        );
    }
}

#[test]
fn binary_add_sub_mul_div() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Array::from_slice(&g, &[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

    approx(
        &a.add(&b).unwrap().to_vec().unwrap(),
        &[6.0, 8.0, 10.0, 12.0],
    );
    approx(
        &a.sub(&b).unwrap().to_vec().unwrap(),
        &[-4.0, -4.0, -4.0, -4.0],
    );
    approx(
        &a.mul(&b).unwrap().to_vec().unwrap(),
        &[5.0, 12.0, 21.0, 32.0],
    );
    approx(
        &a.div(&b).unwrap().to_vec().unwrap(),
        &[1.0 / 5.0, 2.0 / 6.0, 3.0 / 7.0, 4.0 / 8.0],
    );
}

#[test]
fn operator_traits() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0], &[3]).unwrap();
    let b = Array::from_slice(&g, &[10.0f32, 20.0, 30.0], &[3]).unwrap();
    let c = &a + &b;
    approx(&c.to_vec().unwrap(), &[11.0, 22.0, 33.0]);
    let d = &b - &a;
    approx(&d.to_vec().unwrap(), &[9.0, 18.0, 27.0]);
}

#[test]
fn min_max_pow() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 5.0, 3.0], &[3]).unwrap();
    let b = Array::from_slice(&g, &[4.0f32, 2.0, 3.0], &[3]).unwrap();
    approx(&a.minimum(&b).unwrap().to_vec().unwrap(), &[1.0, 2.0, 3.0]);
    approx(&a.maximum(&b).unwrap().to_vec().unwrap(), &[4.0, 5.0, 3.0]);
    let e = Array::from_slice(&g, &[2.0f32, 3.0], &[2]).unwrap();
    let f = Array::from_slice(&g, &[10.0f32, 2.0], &[2]).unwrap();
    approx(&e.pow(&f).unwrap().to_vec().unwrap(), &[1024.0, 9.0]);
}

#[test]
fn unary_neg_abs_sqrt() {
    let g = gpu();
    let a = Array::from_slice(&g, &[-1.0f32, 4.0, -9.0, 16.0], &[4]).unwrap();
    approx(
        &a.neg().unwrap().to_vec().unwrap(),
        &[1.0, -4.0, 9.0, -16.0],
    );
    approx(&a.abs().unwrap().to_vec().unwrap(), &[1.0, 4.0, 9.0, 16.0]);
    let p = Array::from_slice(&g, &[1.0f32, 4.0, 9.0, 16.0], &[4]).unwrap();
    approx(&p.sqrt().unwrap().to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn unary_math_vs_libm() {
    let g = gpu();
    let xs: Vec<f32> = (0..16).map(|i| 0.1 + i as f32 * 0.2).collect();
    let a = Array::from_slice(&g, &xs, &[16]).unwrap();

    let exp_ref: Vec<f32> = xs.iter().map(|x| x.exp()).collect();
    approx(&a.exp().unwrap().to_vec().unwrap(), &exp_ref);
    let log_ref: Vec<f32> = xs.iter().map(|x| x.ln()).collect();
    approx(&a.log().unwrap().to_vec().unwrap(), &log_ref);
    let sin_ref: Vec<f32> = xs.iter().map(|x| x.sin()).collect();
    approx(&a.sin().unwrap().to_vec().unwrap(), &sin_ref);
    let cos_ref: Vec<f32> = xs.iter().map(|x| x.cos()).collect();
    approx(&a.cos().unwrap().to_vec().unwrap(), &cos_ref);
}

#[test]
fn ufunc_on_strided_view_materializes() {
    let g = gpu();
    // transpose -> strided; a ufunc must contiguous-ify then compute.
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let t = a.transpose(0, 1).unwrap(); // logical [[1,4],[2,5],[3,6]]
    let r = t.neg().unwrap();
    assert_eq!(r.shape(), &[3, 2]);
    approx(&r.to_vec().unwrap(), &[-1.0, -4.0, -2.0, -5.0, -3.0, -6.0]);
}

#[test]
fn shape_mismatch_errors() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0], &[2]).unwrap();
    let b = Array::from_slice(&g, &[1.0f32, 2.0, 3.0], &[3]).unwrap();
    assert!(a.add(&b).is_err());
}

#[test]
fn step_positive_mask() {
    let g = gpu();
    let a = Array::from_slice(&g, &[-2.0f32, -0.0, 0.0, 0.5, 3.0], &[5]).unwrap();
    let r = a.step_positive().unwrap();
    // > 0 → 1, else 0 (0 and -0 are not > 0).
    approx(&r.to_vec().unwrap(), &[0.0, 0.0, 0.0, 1.0, 1.0]);
}
