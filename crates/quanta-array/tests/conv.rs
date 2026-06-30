//! im2col tests vs a CPU reference (software lane).

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

/// Host im2col reference: NCHW x → [N·OH·OW, Cin·kh·kw], zero-padded.
#[allow(clippy::too_many_arguments)]
fn host_im2col(
    x: &[f32],
    n: usize,
    cin: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (w + 2 * pad - kw) / stride + 1;
    let kdim = cin * kh * kw;
    let rows = n * oh * ow;
    let mut out = vec![0.0f32; rows * kdim];
    for ni in 0..n {
        for ohi in 0..oh {
            for owi in 0..ow {
                let row = (ni * oh + ohi) * ow + owi;
                for ci in 0..cin {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let kk = (ci * kh + ki) * kw + kj;
                            let ih = ohi * stride + ki;
                            let iw = owi * stride + kj;
                            let val = if ih >= pad && ih < h + pad && iw >= pad && iw < w + pad {
                                let ihx = ih - pad;
                                let iwx = iw - pad;
                                x[((ni * cin + ci) * h + ihx) * w + iwx]
                            } else {
                                0.0
                            };
                            out[row * kdim + kk] = val;
                        }
                    }
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn check(
    n: usize,
    cin: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
) {
    let g = gpu();
    let count = n * cin * h * w;
    let x: Vec<f32> = (0..count).map(|i| (i % 13) as f32 - 6.0).collect();
    let arr = Array::from_slice(&g, &x, &[n, cin, h, w]).unwrap();
    let cols = arr.im2col(kh, kw, stride, pad).unwrap();
    let want = host_im2col(&x, n, cin, h, w, kh, kw, stride, pad);
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (w + 2 * pad - kw) / stride + 1;
    assert_eq!(cols.shape(), &[n * oh * ow, cin * kh * kw]);
    approx(&cols.to_vec().unwrap(), &want);
}

#[test]
fn im2col_basic() {
    check(1, 1, 4, 4, 3, 3, 1, 0); // single channel, 3×3, valid
}

#[test]
fn im2col_padded() {
    check(1, 2, 5, 5, 3, 3, 1, 1); // 2 channels, same-padding
}

#[test]
fn im2col_strided() {
    check(1, 1, 6, 6, 2, 2, 2, 0); // stride 2
}

#[test]
fn im2col_batch_multichannel() {
    check(2, 3, 4, 5, 3, 2, 1, 1);
}

#[test]
fn im2col_requires_4d() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0], &[3]).unwrap();
    assert!(a.im2col(2, 2, 1, 0).is_err());
}

/// Host col2im reference: fold cols [N·OH·OW, K] back to [N,Cin,H,W], summing.
#[allow(clippy::too_many_arguments)]
fn host_col2im(
    cols: &[f32],
    n: usize,
    cin: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (w + 2 * pad - kw) / stride + 1;
    let kdim = cin * kh * kw;
    let mut out = vec![0.0f32; n * cin * h * w];
    for ni in 0..n {
        for ohi in 0..oh {
            for owi in 0..ow {
                let row = (ni * oh + ohi) * ow + owi;
                for ci in 0..cin {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let kk = (ci * kh + ki) * kw + kj;
                            let ih = ohi * stride + ki;
                            let iw = owi * stride + kj;
                            if ih >= pad && ih < h + pad && iw >= pad && iw < w + pad {
                                let ihx = ih - pad;
                                let iwx = iw - pad;
                                out[((ni * cin + ci) * h + ihx) * w + iwx] += cols[row * kdim + kk];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn check_col2im(
    n: usize,
    cin: usize,
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
    let kdim = cin * kh * kw;
    let rows = n * oh * ow;
    let cols: Vec<f32> = (0..rows * kdim).map(|i| (i % 11) as f32 - 5.0).collect();
    let arr = Array::from_slice(&g, &cols, &[rows, kdim]).unwrap();
    let im = arr.col2im(n, cin, h, w, kh, kw, stride, pad).unwrap();
    assert_eq!(im.shape(), &[n, cin, h, w]);
    let want = host_col2im(&cols, n, cin, h, w, kh, kw, stride, pad);
    approx(&im.to_vec().unwrap(), &want);
}

#[test]
fn col2im_basic() {
    check_col2im(1, 1, 4, 4, 3, 3, 1, 0);
}

#[test]
fn col2im_padded_strided() {
    check_col2im(2, 3, 5, 5, 3, 3, 2, 1);
}

/// im2col and col2im are adjoints: <im2col(x), y> == <x, col2im(y)>.
#[test]
fn im2col_col2im_adjoint() {
    let g = gpu();
    let (n, cin, h, w, kh, kw, stride, pad) = (1usize, 2, 5, 5, 3, 3, 1, 1);
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (w + 2 * pad - kw) / stride + 1;
    let kdim = cin * kh * kw;
    let x: Vec<f32> = (0..n * cin * h * w).map(|i| (i % 7) as f32 - 3.0).collect();
    let y: Vec<f32> = (0..n * oh * ow * kdim)
        .map(|i| ((i * 3) % 9) as f32 - 4.0)
        .collect();

    let xa = Array::from_slice(&g, &x, &[n, cin, h, w]).unwrap();
    let ya = Array::from_slice(&g, &y, &[n * oh * ow, kdim]).unwrap();
    let im = xa.im2col(kh, kw, stride, pad).unwrap().to_vec().unwrap(); // im2col(x)
    let cm = ya
        .col2im(n, cin, h, w, kh, kw, stride, pad)
        .unwrap()
        .to_vec()
        .unwrap(); // col2im(y)

    let lhs: f32 = im.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let rhs: f32 = x.iter().zip(cm.iter()).map(|(a, b)| a * b).sum();
    assert!(
        (lhs - rhs).abs() <= 1e-2 * (1.0 + rhs.abs()),
        "adjoint: {lhs} vs {rhs}"
    );
}
