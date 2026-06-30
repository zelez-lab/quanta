//! Pooling tests vs CPU references (software lane).

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

/// Host avgpool (count_include_pad: divide by kh·kw, pad reads 0).
#[allow(clippy::too_many_arguments)]
fn host_avgpool(
    x: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
) -> (Vec<f32>, usize, usize) {
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (w + 2 * pad - kw) / stride + 1;
    let mut y = vec![0.0f32; n * c * oh * ow];
    for ni in 0..n {
        for ci in 0..c {
            for ohi in 0..oh {
                for owi in 0..ow {
                    let mut acc = 0.0f32;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = ohi * stride + ki;
                            let iw = owi * stride + kj;
                            if ih >= pad && ih < h + pad && iw >= pad && iw < w + pad {
                                acc += x[((ni * c + ci) * h + (ih - pad)) * w + (iw - pad)];
                            }
                        }
                    }
                    y[((ni * c + ci) * oh + ohi) * ow + owi] = acc / (kh * kw) as f32;
                }
            }
        }
    }
    (y, oh, ow)
}

#[allow(clippy::too_many_arguments)]
fn check_avg(
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
) {
    let g = gpu();
    let x: Vec<f32> = (0..n * c * h * w).map(|i| (i % 13) as f32 - 6.0).collect();
    let arr = Array::from_slice(&g, &x, &[n, c, h, w]).unwrap();
    let got = arr.avgpool2d(kh, kw, stride, pad).unwrap();
    let (want, oh, ow) = host_avgpool(&x, n, c, h, w, kh, kw, stride, pad);
    assert_eq!(got.shape(), &[n, c, oh, ow]);
    approx(&got.to_vec().unwrap(), &want);
}

#[test]
fn avgpool_basic() {
    check_avg(1, 1, 4, 4, 2, 2, 2, 0);
}
#[test]
fn avgpool_padded_strided() {
    check_avg(2, 3, 5, 5, 3, 3, 2, 1);
}
#[test]
fn avgpool_overlap() {
    check_avg(1, 2, 5, 5, 3, 3, 1, 0);
}
#[test]
fn avgpool_requires_4d() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0], &[2]).unwrap();
    assert!(a.avgpool2d(2, 2, 1, 0).is_err());
}

/// Host avgpool backward: scatter grad/(kh·kw) to each input pixel its windows covered.
#[allow(clippy::too_many_arguments)]
fn host_avgpool_bwd(
    grad: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (w + 2 * pad - kw) / stride + 1;
    let inv = 1.0f32 / (kh * kw) as f32;
    let mut dx = vec![0.0f32; n * c * h * w];
    for ni in 0..n {
        for ci in 0..c {
            for ohi in 0..oh {
                for owi in 0..ow {
                    let gv = grad[((ni * c + ci) * oh + ohi) * ow + owi] * inv;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = ohi * stride + ki;
                            let iw = owi * stride + kj;
                            if ih >= pad && ih < h + pad && iw >= pad && iw < w + pad {
                                dx[((ni * c + ci) * h + (ih - pad)) * w + (iw - pad)] += gv;
                            }
                        }
                    }
                }
            }
        }
    }
    dx
}

#[allow(clippy::too_many_arguments)]
fn check_avg_bwd(
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
) {
    let g = gpu();
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (w + 2 * pad - kw) / stride + 1;
    let grad: Vec<f32> = (0..n * c * oh * ow).map(|i| (i % 9) as f32 - 4.0).collect();
    let ga = Array::from_slice(&g, &grad, &[n, c, oh, ow]).unwrap();
    let dx = ga.avgpool2d_backward(h, w, kh, kw, stride, pad).unwrap();
    assert_eq!(dx.shape(), &[n, c, h, w]);
    let want = host_avgpool_bwd(&grad, n, c, h, w, kh, kw, stride, pad);
    approx(&dx.to_vec().unwrap(), &want);
}

#[test]
fn avgpool_bwd_basic() {
    check_avg_bwd(1, 1, 4, 4, 2, 2, 2, 0);
}
#[test]
fn avgpool_bwd_padded_strided() {
    check_avg_bwd(2, 3, 5, 5, 3, 3, 2, 1);
}
#[test]
fn avgpool_bwd_overlap() {
    check_avg_bwd(1, 2, 5, 5, 3, 3, 1, 0);
}

/// avgpool / avgpool_backward are adjoints: <avgpool(x), g> == <x, avgpool_bwd(g)>.
#[test]
fn avgpool_adjoint() {
    let g = gpu();
    let (n, c, h, w, kh, kw, stride, pad) = (1usize, 2, 5, 5, 3, 3, 2, 1);
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (w + 2 * pad - kw) / stride + 1;
    let x: Vec<f32> = (0..n * c * h * w).map(|i| (i % 7) as f32 - 3.0).collect();
    let gr: Vec<f32> = (0..n * c * oh * ow)
        .map(|i| ((i * 3) % 5) as f32 - 2.0)
        .collect();
    let xa = Array::from_slice(&g, &x, &[n, c, h, w]).unwrap();
    let ga = Array::from_slice(&g, &gr, &[n, c, oh, ow]).unwrap();
    let fwd = xa.avgpool2d(kh, kw, stride, pad).unwrap().to_vec().unwrap();
    let bwd = ga
        .avgpool2d_backward(h, w, kh, kw, stride, pad)
        .unwrap()
        .to_vec()
        .unwrap();
    let lhs: f32 = fwd.iter().zip(gr.iter()).map(|(a, b)| a * b).sum();
    let rhs: f32 = x.iter().zip(bwd.iter()).map(|(a, b)| a * b).sum();
    assert!(
        (lhs - rhs).abs() <= 1e-2 * (1.0 + rhs.abs()),
        "avgpool adjoint: {lhs} vs {rhs}"
    );
}

// ── maxpool ──────────────────────────────────────────────────────────────

/// Host maxpool: per-window max + flat input index of the winner (first max on
/// ties, scanning ki then kj — matches the kernel's strict-greater tap order).
#[allow(clippy::too_many_arguments)]
fn host_maxpool(
    x: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
) -> (Vec<f32>, Vec<u32>, usize, usize) {
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (w + 2 * pad - kw) / stride + 1;
    let mut vals = vec![0.0f32; n * c * oh * ow];
    let mut arg = vec![0u32; n * c * oh * ow];
    for ni in 0..n {
        for ci in 0..c {
            for ohi in 0..oh {
                for owi in 0..ow {
                    let mut best = f32::MIN;
                    let mut bidx = 0u32;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = ohi * stride + ki;
                            let iw = owi * stride + kj;
                            if ih >= pad && ih < h + pad && iw >= pad && iw < w + pad {
                                let fi = ((ni * c + ci) * h + (ih - pad)) * w + (iw - pad);
                                if x[fi] > best {
                                    best = x[fi];
                                    bidx = fi as u32;
                                }
                            }
                        }
                    }
                    let o = ((ni * c + ci) * oh + ohi) * ow + owi;
                    vals[o] = best;
                    arg[o] = bidx;
                }
            }
        }
    }
    (vals, arg, oh, ow)
}

#[allow(clippy::too_many_arguments)]
fn check_max(
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
) {
    let g = gpu();
    // Distinct values (no ties) so the argmax is unambiguous.
    let x: Vec<f32> = (0..n * c * h * w).map(|i| (i as f32) * 0.5 - 7.0).collect();
    let arr = Array::from_slice(&g, &x, &[n, c, h, w]).unwrap();
    let (vals, arg) = arr.maxpool2d(kh, kw, stride, pad).unwrap();
    let (wv, wa, oh, ow) = host_maxpool(&x, n, c, h, w, kh, kw, stride, pad);
    assert_eq!(vals.shape(), &[n, c, oh, ow]);
    approx(&vals.to_vec().unwrap(), &wv);
    assert_eq!(arg.to_vec().unwrap(), wa, "argmax mismatch");
}

#[test]
fn maxpool_basic() {
    check_max(1, 1, 4, 4, 2, 2, 2, 0);
}
#[test]
fn maxpool_padded_strided() {
    check_max(2, 3, 5, 5, 3, 3, 2, 1);
}
#[test]
fn maxpool_overlap() {
    check_max(1, 2, 5, 5, 3, 3, 1, 0);
}

/// Host maxpool backward: scatter grad[out] to argmax[out].
fn host_maxpool_bwd(grad: &[f32], arg: &[u32], n: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut dx = vec![0.0f32; n * c * h * w];
    for (o, &g) in grad.iter().enumerate() {
        dx[arg[o] as usize] += g;
    }
    dx
}

#[allow(clippy::too_many_arguments)]
fn check_max_bwd(
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
) {
    let g = gpu();
    let x: Vec<f32> = (0..n * c * h * w).map(|i| (i as f32) * 0.5 - 7.0).collect();
    let arr = Array::from_slice(&g, &x, &[n, c, h, w]).unwrap();
    let (_vals, arg) = arr.maxpool2d(kh, kw, stride, pad).unwrap();
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (w + 2 * pad - kw) / stride + 1;
    let grad: Vec<f32> = (0..n * c * oh * ow)
        .map(|i| ((i * 3) % 11) as f32 - 5.0)
        .collect();
    let ga = Array::from_slice(&g, &grad, &[n, c, oh, ow]).unwrap();
    let dx = ga
        .maxpool2d_backward(&arg, h, w, kh, kw, stride, pad)
        .unwrap();
    assert_eq!(dx.shape(), &[n, c, h, w]);
    let want = host_maxpool_bwd(&grad, &arg.to_vec().unwrap(), n, c, h, w);
    approx(&dx.to_vec().unwrap(), &want);
}

#[test]
fn maxpool_bwd_basic() {
    check_max_bwd(1, 1, 4, 4, 2, 2, 2, 0);
}
#[test]
fn maxpool_bwd_padded_strided() {
    check_max_bwd(2, 3, 5, 5, 3, 3, 2, 1);
}
#[test]
fn maxpool_bwd_overlap() {
    check_max_bwd(1, 2, 5, 5, 3, 3, 1, 0);
}
