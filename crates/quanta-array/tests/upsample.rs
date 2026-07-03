//! `upsample2d` (nearest-neighbour spatial upsample) + its adjoint.

use quanta_array::Array;

fn gpu() -> quanta::Gpu {
    quanta::init_cpu()
}

#[test]
fn upsample2d_replicates_pixels() {
    let g = gpu();
    // [1, 1, 2, 2] → 2× → [1, 1, 4, 4]; each pixel becomes a 2×2 block.
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
    let up = a.upsample2d(2).unwrap();
    assert_eq!(up.shape(), &[1, 1, 4, 4]);
    assert_eq!(
        up.to_vec().unwrap(),
        vec![
            1.0, 1.0, 2.0, 2.0, //
            1.0, 1.0, 2.0, 2.0, //
            3.0, 3.0, 4.0, 4.0, //
            3.0, 3.0, 4.0, 4.0,
        ]
    );
}

#[test]
fn upsample2d_multichannel_multibatch() {
    let g = gpu();
    // [2, 2, 1, 1] → 3× → [2, 2, 3, 3]; each of the 4 (n,c) planes is one pixel
    // replicated to a 3×3 block.
    let data: Vec<f32> = (0..4).map(|i| i as f32).collect();
    let a = Array::from_slice(&g, &data, &[2, 2, 1, 1]).unwrap();
    let up = a.upsample2d(3).unwrap();
    assert_eq!(up.shape(), &[2, 2, 3, 3]);
    let v = up.to_vec().unwrap();
    for plane in 0..4 {
        for cell in 0..9 {
            assert_eq!(
                v[plane * 9 + cell],
                data[plane],
                "plane {plane} cell {cell}"
            );
        }
    }
}

#[test]
fn upsample2d_backward_sums_blocks() {
    let g = gpu();
    // grad [1,1,4,4] → downsample-sum 2× → [1,1,2,2]; each output = sum of a 2×2.
    let grad: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let ga = Array::from_slice(&g, &grad, &[1, 1, 4, 4]).unwrap();
    let down = ga.upsample2d_backward(2, 2, 2).unwrap();
    assert_eq!(down.shape(), &[1, 1, 2, 2]);
    // top-left block = 1+2+5+6 = 14 ; TR = 3+4+7+8 = 22 ; BL = 9+10+13+14 = 46 ;
    // BR = 11+12+15+16 = 54
    assert_eq!(down.to_vec().unwrap(), vec![14.0, 22.0, 46.0, 54.0]);
}

#[test]
fn upsample_downsample_are_adjoint() {
    // <upsample(x), g> == <x, downsample_sum(g)> — the VJP identity.
    let g = gpu();
    let (n, c, h, w, k) = (2usize, 3, 3, 4, 2);
    let x: Vec<f32> = (0..n * c * h * w).map(|i| (i as f32) * 0.3 - 2.0).collect();
    let og: Vec<f32> = (0..n * c * h * k * w * k)
        .map(|i| (i as f32) * 0.1 + 0.5)
        .collect();
    let xa = Array::from_slice(&g, &x, &[n, c, h, w]).unwrap();
    let ga = Array::from_slice(&g, &og, &[n, c, h * k, w * k]).unwrap();

    let up = xa.upsample2d(k).unwrap().to_vec().unwrap();
    let down = ga.upsample2d_backward(k, h, w).unwrap().to_vec().unwrap();
    let lhs: f32 = up.iter().zip(&og).map(|(a, b)| a * b).sum();
    let rhs: f32 = down.iter().zip(&x).map(|(a, b)| a * b).sum();
    assert!(
        (lhs - rhs).abs() <= 1e-2 * (1.0 + lhs.abs()),
        "adjoint {lhs} vs {rhs}"
    );
}
