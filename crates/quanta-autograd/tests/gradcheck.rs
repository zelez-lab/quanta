//! Finite-difference gradient checking — the binding correctness test for
//! reverse-mode autodiff. For each op we compute the analytic gradient via the
//! tape and compare it to the central difference
//! `(L(x+h) − L(x−h)) / 2h` per input element. Agreement to a small tolerance
//! means the recorded VJP matches the true derivative (the same fact proven
//! analytically in `specs/verify/lean/Quanta/Autograd/Vjp.lean`).

use quanta_array::Array;
use quanta_autograd::Tape;

fn gpu() -> quanta::Gpu {
    quanta::init_cpu()
}

/// Scalar loss `L = sum(f(x))` for the analytic path, returning d L / d x.
fn analytic_grad(
    g: &quanta::Gpu,
    x: &[f32],
    f: impl Fn(&quanta_autograd::Var<f32>) -> quanta_autograd::Var<f32>,
) -> Vec<f32> {
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(g, x, &[x.len()]).unwrap());
    let loss = f(&xv).sum().unwrap();
    loss.grad(&xv).unwrap().to_vec().unwrap()
}

/// Numerical `L = sum(f(x))` evaluated on a host closure, central-difference
/// gradient. `host_f` is the same scalar function applied elementwise on the
/// CPU, so we never depend on the autograd path for the reference.
fn numeric_grad(x: &[f32], host_f: impl Fn(f32) -> f32) -> Vec<f32> {
    let h = 1e-3f32;
    // L = Σ f(xᵢ), so ∂L/∂xⱼ = f'(xⱼ) — purely local, central difference.
    x.iter()
        .map(|&xi| (host_f(xi + h) - host_f(xi - h)) / (2.0 * h))
        .collect()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= tol * (1.0 + y.abs()),
            "{what}: grad[{i}] = {x} vs numeric {y}"
        );
    }
}

#[test]
fn grad_square() {
    // f(x) = x*x ⇒ f' = 2x
    let g = gpu();
    let x = vec![1.0f32, 2.0, -3.0, 0.5];
    let an = analytic_grad(&g, &x, |v| v.mul(v).unwrap());
    let num = numeric_grad(&x, |x| x * x);
    assert_close(&an, &num, 1e-2, "square");
}

#[test]
fn grad_add_chain() {
    // f(x) = (x + x) * x = 2x² ⇒ f' = 4x
    let g = gpu();
    let x = vec![1.5f32, -2.0, 3.0];
    let an = analytic_grad(&g, &x, |v| v.add(v).unwrap().mul(v).unwrap());
    let num = numeric_grad(&x, |x| (x + x) * x);
    assert_close(&an, &num, 1e-2, "add_chain");
}

#[test]
fn grad_sub_neg() {
    // f(x) = -(x - (x*x)) ⇒ host: -(x - x²)
    let g = gpu();
    let x = vec![0.5f32, 1.0, 2.0];
    let an = analytic_grad(&g, &x, |v| {
        v.sub(&v.mul(v).unwrap()).unwrap().neg().unwrap()
    });
    let num = numeric_grad(&x, |x| -(x - x * x));
    assert_close(&an, &num, 1e-2, "sub_neg");
}

#[test]
fn grad_div() {
    // f(x) = x / (x + x) ... trivially 0.5, but exercises div VJP with a
    // non-constant denominator: use f(x) = (x*x) / x = x ⇒ f' = 1.
    let g = gpu();
    let x = vec![1.0f32, 2.0, 4.0, 8.0];
    let an = analytic_grad(&g, &x, |v| v.mul(v).unwrap().div(v).unwrap());
    let num = numeric_grad(&x, |x| (x * x) / x);
    assert_close(&an, &num, 1e-2, "div");
}

#[test]
fn grad_exp() {
    // f(x) = exp(x) ⇒ f' = exp(x)
    let g = gpu();
    let x = vec![0.0f32, 0.5, 1.0, -1.0];
    let an = analytic_grad(&g, &x, |v| v.exp().unwrap());
    let num = numeric_grad(&x, |x| x.exp());
    assert_close(&an, &num, 1e-2, "exp");
}

#[test]
fn grad_log() {
    // f(x) = log(x) ⇒ f' = 1/x   (positive inputs)
    let g = gpu();
    let x = vec![0.5f32, 1.0, 2.0, 5.0];
    let an = analytic_grad(&g, &x, |v| v.log().unwrap());
    let num = numeric_grad(&x, |x| x.ln());
    assert_close(&an, &num, 1e-2, "log");
}

#[test]
fn grad_sqrt() {
    // f(x) = sqrt(x) ⇒ f' = 1/(2√x)   (positive inputs)
    let g = gpu();
    let x = vec![0.25f32, 1.0, 4.0, 9.0];
    let an = analytic_grad(&g, &x, |v| v.sqrt().unwrap());
    let num = numeric_grad(&x, |x| x.sqrt());
    assert_close(&an, &num, 1e-2, "sqrt");
}

#[test]
fn grad_exp_of_square() {
    // f(x) = exp(x*x) ⇒ f' = 2x·exp(x²)  — chains mul → exp.
    let g = gpu();
    let x = vec![0.3f32, 0.7, -0.5];
    let an = analytic_grad(&g, &x, |v| v.mul(v).unwrap().exp().unwrap());
    let num = numeric_grad(&x, |x| (x * x).exp());
    assert_close(&an, &num, 1e-2, "exp_of_square");
}

#[test]
fn grad_foreign_var_errors() {
    let g = gpu();
    let t1 = Tape::<f32>::new();
    let t2 = Tape::<f32>::new();
    let a = t1.var(Array::from_slice(&g, &[1.0], &[1]).unwrap());
    let b = t2.var(Array::from_slice(&g, &[1.0], &[1]).unwrap());
    assert!(a.grad(&b).is_err());
}

// ── matmul VJP: ∂A = G·Bᵀ, ∂B = Aᵀ·G, gradient-checked numerically ──────

/// Host reference: loss = sum(A·B) for A (m×k), B (k×n), row-major flat.
fn matmul_loss(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> f32 {
    let mut s = 0.0f32;
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            s += acc;
        }
    }
    s
}

#[test]
fn grad_matmul() {
    let g = gpu();
    let (m, k, n) = (2usize, 3, 2);
    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
    let b = vec![0.5f32, -1.0, 2.0, 1.0, -0.5, 0.0]; // 3×2

    // Analytic grads via the tape.
    let tape = Tape::<f32>::new();
    let av = tape.var(Array::from_slice(&g, &a, &[m, k]).unwrap());
    let bv = tape.var(Array::from_slice(&g, &b, &[k, n]).unwrap());
    let loss = av.matmul(&bv).unwrap().sum().unwrap();
    let ga = loss.grad(&av).unwrap().to_vec().unwrap();
    let gb = loss.grad(&bv).unwrap().to_vec().unwrap();

    // Central-difference grad w.r.t. each element of A and B.
    let h = 1e-2f32;
    for idx in 0..a.len() {
        let mut ap = a.clone();
        ap[idx] += h;
        let mut am = a.clone();
        am[idx] -= h;
        let num = (matmul_loss(&ap, &b, m, k, n) - matmul_loss(&am, &b, m, k, n)) / (2.0 * h);
        assert!(
            (ga[idx] - num).abs() <= 1e-2 * (1.0 + num.abs()),
            "∂A[{idx}] = {} vs numeric {num}",
            ga[idx]
        );
    }
    for idx in 0..b.len() {
        let mut bp = b.clone();
        bp[idx] += h;
        let mut bm = b.clone();
        bm[idx] -= h;
        let num = (matmul_loss(&a, &bp, m, k, n) - matmul_loss(&a, &bm, m, k, n)) / (2.0 * h);
        assert!(
            (gb[idx] - num).abs() <= 1e-2 * (1.0 + num.abs()),
            "∂B[{idx}] = {} vs numeric {num}",
            gb[idx]
        );
    }
}

#[test]
fn grad_matmul_chain() {
    // loss = sum((A·B) * (A·B)) — matmul feeding an elementwise square, so the
    // upstream gradient into matmul is non-uniform (2·(A·B)), exercising the
    // real G·Bᵀ / Aᵀ·G path rather than the all-ones special case.
    let g = gpu();
    let (m, k, n) = (2usize, 2, 2);
    let a = vec![1.0f32, 2.0, -1.0, 0.5];
    let b = vec![0.5f32, 1.0, 2.0, -1.0];
    let tape = Tape::<f32>::new();
    let av = tape.var(Array::from_slice(&g, &a, &[m, k]).unwrap());
    let bv = tape.var(Array::from_slice(&g, &b, &[k, n]).unwrap());
    let y = av.matmul(&bv).unwrap();
    let loss = y.mul(&y).unwrap().sum().unwrap();
    let ga = loss.grad(&av).unwrap().to_vec().unwrap();

    let host_loss = |a: &[f32], b: &[f32]| -> f32 {
        let mut s = 0.0;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                s += acc * acc;
            }
        }
        s
    };
    let h = 1e-2f32;
    for idx in 0..a.len() {
        let mut ap = a.clone();
        ap[idx] += h;
        let mut am = a.clone();
        am[idx] -= h;
        let num = (host_loss(&ap, &b) - host_loss(&am, &b)) / (2.0 * h);
        assert!(
            (ga[idx] - num).abs() <= 2e-2 * (1.0 + num.abs()),
            "chain ∂A[{idx}] = {} vs numeric {num}",
            ga[idx]
        );
    }
}

// ── broadcast + axis-reduction grads ────────────────────────────────────

/// loss = sum( a[1,n] + b[m,n] ). grad_a sums g over the broadcast axis 0,
/// so each a[j] gets m (= every row contributed 1). grad_b is all ones.
#[test]
fn grad_broadcast_add() {
    let g = gpu();
    let (m, n) = (3usize, 4usize);
    let a = vec![1.0f32, 2.0, 3.0, 4.0]; // [1,n]
    let b: Vec<f32> = (0..m * n).map(|i| i as f32 * 0.1).collect(); // [m,n]
    let tape = Tape::<f32>::new();
    let av = tape.var(Array::from_slice(&g, &a, &[1, n]).unwrap());
    let bv = tape.var(Array::from_slice(&g, &b, &[m, n]).unwrap());
    let loss = av.add(&bv).unwrap().sum().unwrap();
    let ga = loss.grad(&av).unwrap();
    let gb = loss.grad(&bv).unwrap();
    assert_eq!(ga.shape(), &[1, n]); // un-broadcast back to a's shape
    assert_eq!(gb.shape(), &[m, n]);
    // ∂loss/∂a[j] = Σ_i 1 = m  (the broadcast axis summed).
    for v in ga.to_vec().unwrap() {
        assert!((v - m as f32).abs() <= 1e-4, "grad_a = {v}, want {m}");
    }
    for v in gb.to_vec().unwrap() {
        assert!((v - 1.0).abs() <= 1e-4, "grad_b = {v}, want 1");
    }
}

/// loss = sum( (a[1,n] * b[m,n]) ). grad_a[j] = Σ_i b[i,j] (broadcast mul VJP
/// un-broadcast over axis 0); check vs a host column-sum of b.
#[test]
fn grad_broadcast_mul() {
    let g = gpu();
    let (m, n) = (2usize, 3usize);
    let a = vec![2.0f32, -1.0, 0.5];
    let b: Vec<f32> = (0..m * n).map(|i| (i as f32) - 2.0).collect();
    let tape = Tape::<f32>::new();
    let av = tape.var(Array::from_slice(&g, &a, &[1, n]).unwrap());
    let bv = tape.var(Array::from_slice(&g, &b, &[m, n]).unwrap());
    let loss = av.mul(&bv).unwrap().sum().unwrap();
    let ga = loss.grad(&av).unwrap().to_vec().unwrap();
    // want[j] = Σ_i b[i,j]
    let want: Vec<f32> = (0..n).map(|j| (0..m).map(|i| b[i * n + j]).sum()).collect();
    for (j, (&x, &y)) in ga.iter().zip(want.iter()).enumerate() {
        assert!(
            (x - y).abs() <= 1e-3 * (1.0 + y.abs()),
            "grad_a[{j}] {x} vs {y}"
        );
    }
}

/// loss = sum( sum_axis(x*x, 0) ) = sum(x*x) ⇒ grad = 2x (the sum_axis VJP
/// broadcasts the upstream grad back over the reduced axis).
#[test]
fn grad_sum_axis() {
    let g = gpu();
    let (m, n) = (3usize, 2usize);
    let x: Vec<f32> = (1..=m * n).map(|i| i as f32 * 0.5).collect();
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[m, n]).unwrap());
    let sq = xv.mul(&xv).unwrap();
    let loss = sq.sum_axis(0).unwrap().sum().unwrap();
    let gx = loss.grad(&xv).unwrap().to_vec().unwrap();
    for (i, (&gi, &xi)) in gx.iter().zip(x.iter()).enumerate() {
        assert!(
            (gi - 2.0 * xi).abs() <= 1e-2 * (1.0 + xi.abs()),
            "grad[{i}] {gi} vs {}",
            2.0 * xi
        );
    }
}

/// loss = sum( mean_axis(x, 0) ) over [m,n] ⇒ ∂/∂x[i,j] = 1/m.
#[test]
fn grad_mean_axis() {
    let g = gpu();
    let (m, n) = (4usize, 3usize);
    let x: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[m, n]).unwrap());
    let loss = xv.mean_axis(0).unwrap().sum().unwrap();
    let gx = loss.grad(&xv).unwrap().to_vec().unwrap();
    for (i, &gi) in gx.iter().enumerate() {
        assert!(
            (gi - 1.0 / m as f32).abs() <= 1e-4,
            "grad[{i}] = {gi}, want {}",
            1.0 / m as f32
        );
    }
}

// ── activations ─────────────────────────────────────────────────────────

#[test]
fn grad_relu() {
    // f(x) = relu(x); f' = 1 (x>0), 0 (x<0). Avoid x=0 (the kink).
    let g = gpu();
    let x = vec![-2.0f32, -0.5, 0.5, 3.0];
    let an = analytic_grad(&g, &x, |v| v.relu().unwrap());
    let num = numeric_grad(&x, |x| x.max(0.0));
    assert_close(&an, &num, 1e-2, "relu");
}

#[test]
fn grad_sigmoid() {
    // f(x) = σ(x); f' = σ(1−σ).
    let g = gpu();
    let x = vec![-1.0f32, 0.0, 0.5, 2.0];
    let an = analytic_grad(&g, &x, |v| v.sigmoid().unwrap());
    let num = numeric_grad(&x, |x| 1.0 / (1.0 + (-x).exp()));
    assert_close(&an, &num, 1e-2, "sigmoid");
}

#[test]
fn grad_tanh() {
    // f(x) = tanh(x); f' = 1 − tanh².
    let g = gpu();
    let x = vec![-1.5f32, -0.3, 0.4, 1.2];
    let an = analytic_grad(&g, &x, |v| v.tanh().unwrap());
    let num = numeric_grad(&x, |x| x.tanh());
    assert_close(&an, &num, 1e-2, "tanh");
}

#[test]
fn grad_relu_chain() {
    // f(x) = relu(x)·relu(x): smooth except the x=0 kink; ⇒ f' = 2·relu(x)·[x>0]
    // = 2x for x>0, 0 for x<0. Chains relu into a mul, so the upstream gradient
    // into relu is non-uniform (2·relu(x)), not all-ones.
    let g = gpu();
    let x = vec![-2.0f32, -0.5, 0.5, 3.0];
    let an = analytic_grad(&g, &x, |v| {
        let r = v.relu().unwrap();
        r.mul(&r).unwrap()
    });
    let num = numeric_grad(&x, |x| x.max(0.0) * x.max(0.0));
    assert_close(&an, &num, 1e-2, "relu_chain");
}

// ── conv2d (im2col + matmul) ─────────────────────────────────────────────

/// Host naive NCHW conv2d: x[N,Cin,H,W] ⊛ w[Cout,Cin,kh,kw] → y[N,Cout,OH,OW],
/// zero-padded. The reference the autograd forward must match.
#[allow(clippy::too_many_arguments)]
fn host_conv2d(
    x: &[f32],
    w: &[f32],
    n: usize,
    cin: usize,
    h: usize,
    wd: usize,
    cout: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
) -> (Vec<f32>, usize, usize) {
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (wd + 2 * pad - kw) / stride + 1;
    let mut y = vec![0.0f32; n * cout * oh * ow];
    for ni in 0..n {
        for co in 0..cout {
            for ohi in 0..oh {
                for owi in 0..ow {
                    let mut acc = 0.0f32;
                    for ci in 0..cin {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = ohi * stride + ki;
                                let iw = owi * stride + kj;
                                if ih >= pad && ih < h + pad && iw >= pad && iw < wd + pad {
                                    let xv =
                                        x[((ni * cin + ci) * h + (ih - pad)) * wd + (iw - pad)];
                                    let wv = w[((co * cin + ci) * kh + ki) * kw + kj];
                                    acc += xv * wv;
                                }
                            }
                        }
                    }
                    y[((ni * cout + co) * oh + ohi) * ow + owi] = acc;
                }
            }
        }
    }
    (y, oh, ow)
}

#[test]
fn conv2d_forward_matches_host() {
    let g = gpu();
    let (n, cin, h, wd, cout, kh, kw, stride, pad) = (2usize, 3, 5, 5, 4, 3, 3, 1, 1);
    let x: Vec<f32> = (0..n * cin * h * wd)
        .map(|i| (i % 7) as f32 - 3.0)
        .collect();
    let w: Vec<f32> = (0..cout * cin * kh * kw)
        .map(|i| ((i * 3) % 5) as f32 - 2.0)
        .collect();

    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[n, cin, h, wd]).unwrap());
    let wv = tape.var(Array::from_slice(&g, &w, &[cout, cin, kh, kw]).unwrap());
    let y = xv.conv2d(&wv, stride, pad).unwrap();
    let (want, oh, ow) = host_conv2d(&x, &w, n, cin, h, wd, cout, kh, kw, stride, pad);
    assert_eq!(y.value().shape(), &[n, cout, oh, ow]);
    let got = y.value().to_vec().unwrap();
    assert_close(&got, &want, 1e-4, "conv2d_forward");
}

#[test]
fn grad_conv2d() {
    // loss = sum(conv2d(x, w)); gradient-check both ∂x and ∂w against central
    // differences of the host conv2d, exercising col2im (∂x) and the weight
    // VJP (∂w) — both via the matmul backward.
    let g = gpu();
    let (n, cin, h, wd, cout, kh, kw, stride, pad) = (1usize, 2, 4, 4, 3, 3, 3, 1, 1);
    let x: Vec<f32> = (0..n * cin * h * wd)
        .map(|i| (i % 5) as f32 - 2.0)
        .collect();
    let w: Vec<f32> = (0..cout * cin * kh * kw)
        .map(|i| ((i * 2) % 7) as f32 - 3.0)
        .collect();

    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[n, cin, h, wd]).unwrap());
    let wv = tape.var(Array::from_slice(&g, &w, &[cout, cin, kh, kw]).unwrap());
    let loss = xv.conv2d(&wv, stride, pad).unwrap().sum().unwrap();
    let gx = loss.grad(&xv).unwrap().to_vec().unwrap();
    let gw = loss.grad(&wv).unwrap().to_vec().unwrap();

    let host_loss = |x: &[f32], w: &[f32]| -> f32 {
        host_conv2d(x, w, n, cin, h, wd, cout, kh, kw, stride, pad)
            .0
            .iter()
            .sum()
    };
    let hh = 1e-2f32;
    for idx in 0..x.len() {
        let mut xp = x.clone();
        xp[idx] += hh;
        let mut xm = x.clone();
        xm[idx] -= hh;
        let num = (host_loss(&xp, &w) - host_loss(&xm, &w)) / (2.0 * hh);
        assert!(
            (gx[idx] - num).abs() <= 1e-2 * (1.0 + num.abs()),
            "∂x[{idx}] = {} vs {num}",
            gx[idx]
        );
    }
    for idx in 0..w.len() {
        let mut wp = w.clone();
        wp[idx] += hh;
        let mut wm = w.clone();
        wm[idx] -= hh;
        let num = (host_loss(&x, &wp) - host_loss(&x, &wm)) / (2.0 * hh);
        assert!(
            (gw[idx] - num).abs() <= 1e-2 * (1.0 + num.abs()),
            "∂w[{idx}] = {} vs {num}",
            gw[idx]
        );
    }
}

/// Real-Metal lane: the conv2d forward + both gradients must match the same
/// host references on hardware (the CPU lane runs the interpreter, so this is
/// the binding check that im2col/col2im/matmul emit correct MSL).
#[cfg(feature = "metal")]
#[test]
fn conv2d_metal_matches_host() {
    let g = match quanta::init() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("skip: no Metal device");
            return;
        }
    };
    let (n, cin, h, wd, cout, kh, kw, stride, pad) = (2usize, 3, 5, 5, 4, 3, 3, 1, 1);
    let x: Vec<f32> = (0..n * cin * h * wd)
        .map(|i| (i % 7) as f32 - 3.0)
        .collect();
    let w: Vec<f32> = (0..cout * cin * kh * kw)
        .map(|i| ((i * 3) % 5) as f32 - 2.0)
        .collect();

    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[n, cin, h, wd]).unwrap());
    let wv = tape.var(Array::from_slice(&g, &w, &[cout, cin, kh, kw]).unwrap());
    let loss = xv.conv2d(&wv, stride, pad).unwrap().sum().unwrap();

    // Forward.
    let y = xv.conv2d(&wv, stride, pad).unwrap();
    let (want, oh, ow) = host_conv2d(&x, &w, n, cin, h, wd, cout, kh, kw, stride, pad);
    assert_eq!(y.value().shape(), &[n, cout, oh, ow]);
    assert_close(
        &y.value().to_vec().unwrap(),
        &want,
        1e-3,
        "conv2d_metal_fwd",
    );

    // Gradients vs central differences of the host conv.
    let gx = loss.grad(&xv).unwrap().to_vec().unwrap();
    let gw = loss.grad(&wv).unwrap().to_vec().unwrap();
    let host_loss = |x: &[f32], w: &[f32]| -> f32 {
        host_conv2d(x, w, n, cin, h, wd, cout, kh, kw, stride, pad)
            .0
            .iter()
            .sum()
    };
    let hh = 1e-2f32;
    for idx in 0..x.len() {
        let mut xp = x.clone();
        xp[idx] += hh;
        let mut xm = x.clone();
        xm[idx] -= hh;
        let num = (host_loss(&xp, &w) - host_loss(&xm, &w)) / (2.0 * hh);
        assert!(
            (gx[idx] - num).abs() <= 2e-2 * (1.0 + num.abs()),
            "metal ∂x[{idx}] = {} vs {num}",
            gx[idx]
        );
    }
    for idx in 0..w.len() {
        let mut wp = w.clone();
        wp[idx] += hh;
        let mut wm = w.clone();
        wm[idx] -= hh;
        let num = (host_loss(&x, &wp) - host_loss(&x, &wm)) / (2.0 * hh);
        assert!(
            (gw[idx] - num).abs() <= 2e-2 * (1.0 + num.abs()),
            "metal ∂w[{idx}] = {} vs {num}",
            gw[idx]
        );
    }
}

#[test]
fn conv2d_bias_broadcasts() {
    // [N,Cout,OH,OW] + [1,Cout,1,1] must broadcast (per-channel bias), and the
    // bias gradient sums over N, OH, OW back to [1,Cout,1,1].
    let g = gpu();
    let (n, cin, h, wd, cout) = (2usize, 2, 4, 4, 3);
    let x: Vec<f32> = (0..n * cin * h * wd)
        .map(|i| (i % 5) as f32 - 2.0)
        .collect();
    let w: Vec<f32> = (0..cout * cin * 3 * 3)
        .map(|i| (i % 4) as f32 - 1.0)
        .collect();
    let b: Vec<f32> = vec![0.5, -1.0, 2.0];
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[n, cin, h, wd]).unwrap());
    let wv = tape.var(Array::from_slice(&g, &w, &[cout, cin, 3, 3]).unwrap());
    let bv = tape.var(Array::from_slice(&g, &b, &[1, cout, 1, 1]).unwrap());
    let y = xv.conv2d(&wv, 1, 1).unwrap().add(&bv).unwrap();
    let loss = y.sum().unwrap();
    let gb = loss.grad(&bv).unwrap();
    assert_eq!(gb.shape(), &[1, cout, 1, 1]);
    // ∂(Σ y)/∂b[c] = number of (n,oh,ow) positions = n*oh*ow (oh=ow=4 here).
    let positions = (n * 4 * 4) as f32;
    for v in gb.to_vec().unwrap() {
        assert!(
            (v - positions).abs() <= 1e-3,
            "bias grad {v} vs {positions}"
        );
    }
}

// ── pooling (avg / max) ──────────────────────────────────────────────────

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
) -> Vec<f32> {
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
    y
}

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
) -> Vec<f32> {
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (w + 2 * pad - kw) / stride + 1;
    let mut y = vec![0.0f32; n * c * oh * ow];
    for ni in 0..n {
        for ci in 0..c {
            for ohi in 0..oh {
                for owi in 0..ow {
                    let mut best = f32::MIN;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = ohi * stride + ki;
                            let iw = owi * stride + kj;
                            if ih >= pad && ih < h + pad && iw >= pad && iw < w + pad {
                                let v = x[((ni * c + ci) * h + (ih - pad)) * w + (iw - pad)];
                                if v > best {
                                    best = v;
                                }
                            }
                        }
                    }
                    y[((ni * c + ci) * oh + ohi) * ow + owi] = best;
                }
            }
        }
    }
    y
}

#[test]
fn grad_avgpool() {
    // loss = sum(avgpool(x)); ∂x checked against central differences of the
    // host avgpool (avgpool is linear, so the gradient is exact).
    let g = gpu();
    let (n, c, h, w, kh, kw, stride, pad) = (2usize, 2, 5, 5, 3, 3, 2, 1);
    let x: Vec<f32> = (0..n * c * h * w).map(|i| (i % 9) as f32 - 4.0).collect();
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[n, c, h, w]).unwrap());
    let loss = xv.avgpool2d(kh, kw, stride, pad).unwrap().sum().unwrap();
    let gx = loss.grad(&xv).unwrap().to_vec().unwrap();
    let host_loss = |x: &[f32]| -> f32 {
        host_avgpool(x, n, c, h, w, kh, kw, stride, pad)
            .iter()
            .sum()
    };
    let hh = 1e-2f32;
    for idx in 0..x.len() {
        let mut xp = x.clone();
        xp[idx] += hh;
        let mut xm = x.clone();
        xm[idx] -= hh;
        let num = (host_loss(&xp) - host_loss(&xm)) / (2.0 * hh);
        assert!(
            (gx[idx] - num).abs() <= 1e-2 * (1.0 + num.abs()),
            "avgpool ∂x[{idx}] = {} vs {num}",
            gx[idx]
        );
    }
}

#[test]
fn grad_maxpool() {
    // loss = sum(maxpool(x)·maxpool(x)) — square so the upstream gradient into
    // maxpool is non-uniform (2·max). Distinct x (no ties) and a small h⁻¹
    // perturbation keep the argmax fixed across the central difference.
    let g = gpu();
    let (n, c, h, w, kh, kw, stride, pad) = (1usize, 2, 4, 4, 2, 2, 2, 0);
    let x: Vec<f32> = (0..n * c * h * w)
        .map(|i| (i as f32) * 0.37 - 3.0)
        .collect();
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[n, c, h, w]).unwrap());
    let mp = xv.maxpool2d(kh, kw, stride, pad).unwrap();
    let loss = mp.mul(&mp).unwrap().sum().unwrap();
    let gx = loss.grad(&xv).unwrap().to_vec().unwrap();
    let host_loss = |x: &[f32]| -> f32 {
        host_maxpool(x, n, c, h, w, kh, kw, stride, pad)
            .iter()
            .map(|v| v * v)
            .sum()
    };
    let hh = 1e-3f32;
    for idx in 0..x.len() {
        let mut xp = x.clone();
        xp[idx] += hh;
        let mut xm = x.clone();
        xm[idx] -= hh;
        let num = (host_loss(&xp) - host_loss(&xm)) / (2.0 * hh);
        assert!(
            (gx[idx] - num).abs() <= 2e-2 * (1.0 + num.abs()),
            "maxpool ∂x[{idx}] = {} vs {num}",
            gx[idx]
        );
    }
}

#[test]
fn pool_requires_4d() {
    let g = gpu();
    let tape = Tape::<f32>::new();
    let a = tape.var(Array::from_slice(&g, &[1.0f32, 2.0], &[2]).unwrap());
    assert!(a.avgpool2d(2, 2, 1, 0).is_err());
    assert!(a.maxpool2d(2, 2, 1, 0).is_err());
}

/// Real-Metal lane for pooling: forward + gradient must match the host
/// references on hardware (the CPU lane is the interpreter).
#[cfg(feature = "metal")]
#[test]
fn pool_metal_matches_host() {
    let g = match quanta::init() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("skip: no Metal device");
            return;
        }
    };
    let (n, c, h, w, kh, kw, stride, pad) = (2usize, 2, 5, 5, 3, 3, 2, 1);
    let x: Vec<f32> = (0..n * c * h * w).map(|i| (i % 9) as f32 - 4.0).collect();

    // avgpool forward + grad.
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[n, c, h, w]).unwrap());
    let yv = xv.avgpool2d(kh, kw, stride, pad).unwrap();
    assert_close(
        &yv.value().to_vec().unwrap(),
        &host_avgpool(&x, n, c, h, w, kh, kw, stride, pad),
        1e-3,
        "avgpool_metal_fwd",
    );
    let loss = yv.sum().unwrap();
    let gx = loss.grad(&xv).unwrap().to_vec().unwrap();
    let host_loss = |x: &[f32]| -> f32 {
        host_avgpool(x, n, c, h, w, kh, kw, stride, pad)
            .iter()
            .sum()
    };
    let hh = 1e-2f32;
    for idx in 0..x.len() {
        let mut xp = x.clone();
        xp[idx] += hh;
        let mut xm = x.clone();
        xm[idx] -= hh;
        let num = (host_loss(&xp) - host_loss(&xm)) / (2.0 * hh);
        assert!(
            (gx[idx] - num).abs() <= 2e-2 * (1.0 + num.abs()),
            "metal avgpool ∂x[{idx}] = {} vs {num}",
            gx[idx]
        );
    }

    // maxpool forward on hardware (distinct values).
    let xm: Vec<f32> = (0..n * c * h * w)
        .map(|i| (i as f32) * 0.37 - 3.0)
        .collect();
    let xv2 = tape.var(Array::from_slice(&g, &xm, &[n, c, h, w]).unwrap());
    let yv2 = xv2.maxpool2d(kh, kw, stride, pad).unwrap();
    assert_close(
        &yv2.value().to_vec().unwrap(),
        &host_maxpool(&xm, n, c, h, w, kh, kw, stride, pad),
        1e-3,
        "maxpool_metal_fwd",
    );
}

// ── reshape / flatten ────────────────────────────────────────────────────

#[test]
fn grad_reshape() {
    // loss = sum( reshape(x*x) ): reshape is linear & shape-only, so ∂x = 2x
    // regardless of the intermediate shape.
    let g = gpu();
    let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[2, 3]).unwrap());
    let y = xv.mul(&xv).unwrap().reshape(&[3, 2]).unwrap();
    assert_eq!(y.value().shape(), &[3, 2]);
    let loss = y.sum().unwrap();
    let gx = loss.grad(&xv).unwrap();
    assert_eq!(gx.shape(), &[2, 3]); // gradient comes back in x's shape
    for (i, (&gi, &xi)) in gx.to_vec().unwrap().iter().zip(x.iter()).enumerate() {
        assert!(
            (gi - 2.0 * xi).abs() <= 1e-3,
            "reshape ∂x[{i}] {gi} vs {}",
            2.0 * xi
        );
    }
}

#[test]
fn grad_flatten() {
    // flatten [N,C,H,W] → [N, C·H·W]; loss = sum(flatten(x)·flatten(x)) ⇒ ∂x = 2x.
    let g = gpu();
    let (n, c, h, w) = (2usize, 2, 2, 2);
    let x: Vec<f32> = (0..n * c * h * w).map(|i| i as f32 - 4.0).collect();
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[n, c, h, w]).unwrap());
    let f = xv.flatten().unwrap();
    assert_eq!(f.value().shape(), &[n, c * h * w]);
    let loss = f.mul(&f).unwrap().sum().unwrap();
    let gx = loss.grad(&xv).unwrap();
    assert_eq!(gx.shape(), &[n, c, h, w]);
    for (i, (&gi, &xi)) in gx.to_vec().unwrap().iter().zip(x.iter()).enumerate() {
        assert!(
            (gi - 2.0 * xi).abs() <= 1e-3,
            "flatten ∂x[{i}] {gi} vs {}",
            2.0 * xi
        );
    }
}

// ── gather_rows (label pick, the cross-entropy piece) ────────────────────

#[test]
fn grad_gather_rows() {
    // loss = sum( gather_rows(x, idx) )  ⇒  ∂x[i,c] = 1 iff c==idx[i], else 0.
    let g = gpu();
    let (n, c) = (4usize, 5usize);
    let x: Vec<f32> = (0..n * c).map(|i| (i % 7) as f32 - 3.0).collect();
    let idx = Array::from_slice(&g, &[2u32, 0, 4, 1], &[n]).unwrap();
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[n, c]).unwrap());
    let picked = xv.gather_rows(&idx).unwrap();
    assert_eq!(picked.value().shape(), &[n]);
    let loss = picked.sum().unwrap();
    let gx = loss.grad(&xv).unwrap().to_vec().unwrap();
    // Expected: one-hot per row at the label column.
    let idx_v = [2usize, 0, 4, 1];
    for i in 0..n {
        for cc in 0..c {
            let want = if cc == idx_v[i] { 1.0 } else { 0.0 };
            assert!(
                (gx[i * c + cc] - want).abs() <= 1e-5,
                "∂x[{i},{cc}] {} vs {want}",
                gx[i * c + cc]
            );
        }
    }
}

#[test]
fn grad_gather_rows_weighted() {
    // loss = sum( gather_rows(x·x, idx) )  — nonlinear upstream, ∂ = 2x at label col.
    let g = gpu();
    let (n, c) = (3usize, 4usize);
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.5 - 2.0).collect();
    let idx = Array::from_slice(&g, &[1u32, 3, 0], &[n]).unwrap();
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[n, c]).unwrap());
    let loss = xv
        .mul(&xv)
        .unwrap()
        .gather_rows(&idx)
        .unwrap()
        .sum()
        .unwrap();
    let gx = loss.grad(&xv).unwrap().to_vec().unwrap();
    let idx_v = [1usize, 3, 0];
    for i in 0..n {
        for cc in 0..c {
            let want = if cc == idx_v[i] {
                2.0 * x[i * c + cc]
            } else {
                0.0
            };
            assert!(
                (gx[i * c + cc] - want).abs() <= 1e-4 * (1.0 + want.abs()),
                "∂x[{i},{cc}] {} vs {want}",
                gx[i * c + cc]
            );
        }
    }
}

// ── cross-entropy loss (the classification head) ─────────────────────────

/// Host cross-entropy: mean over rows of −log_softmax(logits)[label].
fn host_cross_entropy(logits: &[f32], labels: &[u32], n: usize, c: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..n {
        let row = &logits[i * c..(i + 1) * c];
        let m = row.iter().cloned().fold(f32::MIN, f32::max);
        let sum_exp: f32 = row.iter().map(|&x| (x - m).exp()).sum();
        let logp = row[labels[i] as usize] - m - sum_exp.ln();
        total += -logp;
    }
    total / n as f32
}

#[test]
fn cross_entropy_value_matches_host() {
    let g = gpu();
    let (n, c) = (4usize, 3usize);
    let logits: Vec<f32> = (0..n * c).map(|i| (i % 5) as f32 - 2.0).collect();
    let labels = [0u32, 2, 1, 2];
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &logits, &[n, c]).unwrap());
    let lab = Array::from_slice(&g, &labels, &[n]).unwrap();
    let loss = xv.cross_entropy(&lab).unwrap();
    let got = loss.value().to_vec().unwrap()[0];
    let want = host_cross_entropy(&logits, &labels, n, c);
    assert!(
        (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
        "CE {got} vs {want}"
    );
}

#[test]
fn grad_cross_entropy() {
    // gradient-check ∂loss/∂logits against central differences of host CE.
    let g = gpu();
    let (n, c) = (3usize, 4usize);
    let logits: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.3 - 1.5).collect();
    let labels = [2u32, 0, 3];
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &logits, &[n, c]).unwrap());
    let lab = Array::from_slice(&g, &labels, &[n]).unwrap();
    let loss = xv.cross_entropy(&lab).unwrap();
    let gx = loss.grad(&xv).unwrap().to_vec().unwrap();

    let hh = 1e-2f32;
    for idx in 0..logits.len() {
        let mut lp = logits.clone();
        lp[idx] += hh;
        let mut lm = logits.clone();
        lm[idx] -= hh;
        let num = (host_cross_entropy(&lp, &labels, n, c) - host_cross_entropy(&lm, &labels, n, c))
            / (2.0 * hh);
        assert!(
            (gx[idx] - num).abs() <= 1e-2 * (1.0 + num.abs()),
            "∂CE[{idx}] = {} vs {num}",
            gx[idx]
        );
    }
}

#[test]
fn log_softmax_normalizes() {
    // exp(log_softmax(x)) rows must sum to 1.
    let g = gpu();
    let (n, c) = (2usize, 5usize);
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.7 - 3.0).collect();
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[n, c]).unwrap());
    let lp = xv.log_softmax().unwrap().value().to_vec().unwrap();
    for i in 0..n {
        let s: f32 = (0..c).map(|j| lp[i * c + j].exp()).sum();
        assert!((s - 1.0).abs() <= 1e-4, "row {i} sums to {s}, want 1");
    }
}
