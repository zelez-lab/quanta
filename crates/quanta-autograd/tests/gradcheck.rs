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

#[test]
fn grad_narrow_is_window_mask() {
    // L = sum(narrow(x, start=1, len=2)) over x[4,3].
    // ∂L/∂x = 1 for rows 1,2 (the window), 0 elsewhere.
    let g = gpu();
    let (n, c) = (4usize, 3usize);
    let x: Vec<f32> = (0..n * c).map(|i| i as f32).collect();
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[n, c]).unwrap());
    let loss = xv.narrow(1, 2).unwrap().sum().unwrap();
    let gx = loss.grad(&xv).unwrap().to_vec().unwrap();
    let mut want = vec![0.0f32; n * c];
    for cell in want.iter_mut().take(3 * c).skip(c) {
        *cell = 1.0; // rows 1 and 2
    }
    assert_close(&gx, &want, 1e-6, "narrow window mask");
}

#[test]
fn grad_narrow_composes_with_matmul() {
    // A real minibatch shape: narrow the batch, run it through a linear layer,
    // and check the sliced-out rows get zero gradient.
    let g = gpu();
    let (n, k, m) = (5usize, 3usize, 2usize);
    let x: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.1 - 0.7).collect();
    let w: Vec<f32> = (0..k * m).map(|i| (i as f32) * 0.2 - 0.3).collect();
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[n, k]).unwrap());
    let wv = tape.var(Array::from_slice(&g, &w, &[k, m]).unwrap());
    // train on rows [1, 4): a 3-row minibatch
    let loss = xv.narrow(1, 3).unwrap().matmul(&wv).unwrap().sum().unwrap();
    let gx = loss.grad(&xv).unwrap().to_vec().unwrap();
    // rows 0 and 4 are outside the window → zero gradient.
    for j in 0..k {
        assert!(
            gx[j].abs() <= 1e-6,
            "row 0 col {j} should be 0, got {}",
            gx[j]
        );
        assert!(
            gx[4 * k + j].abs() <= 1e-6,
            "row 4 col {j} should be 0, got {}",
            gx[4 * k + j]
        );
    }
    // in-window rows get the standard matmul gradient (row-sum of W).
    let wsum: Vec<f32> = (0..k)
        .map(|kk| (0..m).map(|mm| w[kk * m + mm]).sum())
        .collect();
    for r in 1..4 {
        for j in 0..k {
            assert!(
                (gx[r * k + j] - wsum[j]).abs() <= 1e-4,
                "row {r} col {j}: {} vs {}",
                gx[r * k + j],
                wsum[j]
            );
        }
    }
}

#[test]
fn grad_concat_routes_slices_to_inputs() {
    // out = concat([a, b]); L = sum(out * out) = sum(a²) + sum(b²).
    // ∂L/∂a = 2a, ∂L/∂b = 2b — each input gets exactly its slice of the grad.
    let g = gpu();
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
    let b_data = vec![5.0f32, 6.0]; // [1, 2]
    let tape = Tape::<f32>::new();
    let a = tape.var(Array::from_slice(&g, &a_data, &[2, 2]).unwrap());
    let b = tape.var(Array::from_slice(&g, &b_data, &[1, 2]).unwrap());
    let out = quanta_autograd::Var::concat_axis0(&[&a, &b]).unwrap();
    let loss = out.mul(&out).unwrap().sum().unwrap();
    let ga = loss.grad(&a).unwrap().to_vec().unwrap();
    let gb = loss.grad(&b).unwrap().to_vec().unwrap();
    let want_a: Vec<f32> = a_data.iter().map(|x| 2.0 * x).collect();
    let want_b: Vec<f32> = b_data.iter().map(|x| 2.0 * x).collect();
    assert_close(&ga, &want_a, 1e-2, "concat grad a");
    assert_close(&gb, &want_b, 1e-2, "concat grad b");
}

#[test]
fn grad_concat_narrow_roundtrip_is_identity_grad() {
    // Split x with narrow, concat the pieces back, sum. Every element flows
    // through exactly once → ∂L/∂x is all ones.
    let g = gpu();
    let x_data: Vec<f32> = (0..8).map(|i| i as f32).collect(); // [4, 2]
    let tape = Tape::<f32>::new();
    let x = tape.var(Array::from_slice(&g, &x_data, &[4, 2]).unwrap());
    let top = x.narrow(0, 2).unwrap();
    let bot = x.narrow(2, 2).unwrap();
    let rejoined = quanta_autograd::Var::concat_axis0(&[&top, &bot]).unwrap();
    let loss = rejoined.sum().unwrap();
    let gx = loss.grad(&x).unwrap().to_vec().unwrap();
    assert_close(&gx, &[1.0f32; 8], 1e-6, "concat/narrow roundtrip grad");
}

#[test]
fn grad_gather_last_scatters_back() {
    // out = gather_last(x, idx); L = sum(out²).
    // ∂L/∂x[r,c] = Σ_j 2·out[r,j]·[idx[r,j]==c] — the scatter-add of 2·out.
    let g = gpu();
    let (r, d, k) = (2usize, 3usize, 4usize);
    let x_data: Vec<f32> = (0..r * d).map(|i| (i as f32) - 2.0).collect();
    let idx_data: Vec<u32> = vec![0, 2, 0, 1, 2, 2, 1, 0]; // [2,4], has repeats
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x_data, &[r, d]).unwrap());
    let idx = Array::from_slice(&g, &idx_data, &[r, k]).unwrap();
    let out = xv.gather_last(&idx).unwrap();
    let loss = out.mul(&out).unwrap().sum().unwrap();
    let gx = loss.grad(&xv).unwrap().to_vec().unwrap();

    // host reference: scatter-add 2·out back through idx.
    let mut want = vec![0.0f32; r * d];
    for row in 0..r {
        for j in 0..k {
            let col = idx_data[row * k + j] as usize;
            let ov = x_data[row * d + col]; // out[row,j] = x[row, col]
            want[row * d + col] += 2.0 * ov;
        }
    }
    assert_close(&gx, &want, 1e-2, "gather_last grad");
}

#[test]
fn grad_where_routes_by_mask() {
    // out = where(mask, a, b); L = sum(out² ). ∂L/∂a = 2a·mask, ∂L/∂b = 2b·(1-mask).
    let g = gpu();
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0];
    let mask_data = vec![1.0f32, 0.0, 1.0, 0.0];
    let tape = Tape::<f32>::new();
    let a = tape.var(Array::from_slice(&g, &a_data, &[4]).unwrap());
    let b = tape.var(Array::from_slice(&g, &b_data, &[4]).unwrap());
    let mask = Array::from_slice(&g, &mask_data, &[4]).unwrap();
    let out = a.where_mask(&mask, &b).unwrap();
    // value: pick a where mask=1 else b → [1, 20, 3, 40]
    assert_close(
        &out.value().to_vec().unwrap(),
        &[1.0, 20.0, 3.0, 40.0],
        1e-6,
        "where value",
    );
    let loss = out.mul(&out).unwrap().sum().unwrap();
    let ga = loss.grad(&a).unwrap().to_vec().unwrap();
    let gb = loss.grad(&b).unwrap().to_vec().unwrap();
    // ∂L/∂a = 2·a·mask ; ∂L/∂b = 2·b·(1-mask)
    let wa: Vec<f32> = a_data
        .iter()
        .zip(&mask_data)
        .map(|(x, m)| 2.0 * x * m)
        .collect();
    let wb: Vec<f32> = b_data
        .iter()
        .zip(&mask_data)
        .map(|(x, m)| 2.0 * x * (1.0 - m))
        .collect();
    assert_close(&ga, &wa, 1e-2, "where grad a");
    assert_close(&gb, &wb, 1e-2, "where grad b");
}

#[test]
fn softmax_rows_sum_to_one() {
    let g = gpu();
    let (n, c) = (3usize, 4usize);
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.5 - 2.0).collect();
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[n, c]).unwrap());
    let p = xv.softmax().unwrap().value().to_vec().unwrap();
    for i in 0..n {
        let s: f32 = (0..c).map(|j| p[i * c + j]).sum();
        assert!((s - 1.0).abs() <= 1e-5, "row {i} sums to {s}");
        assert!(
            (0..c).all(|j| p[i * c + j] > 0.0),
            "row {i} has non-positive prob"
        );
    }
}

#[test]
fn mean_matches_host() {
    let g = gpu();
    let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[5]).unwrap());
    let m = xv.mean().unwrap().value().to_vec().unwrap()[0];
    assert!((m - 3.0).abs() <= 1e-5, "mean {m} != 3.0");
    // ∂mean/∂xᵢ = 1/n
    let gx = xv.mean().unwrap().grad(&xv).unwrap().to_vec().unwrap();
    assert_close(&gx, &[0.2f32; 5], 1e-4, "mean grad");
}

#[test]
fn mse_loss_and_grad() {
    // L = mean((x - t)²); ∂L/∂x = 2(x - t)/n.
    let g = gpu();
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let t = vec![1.5f32, 1.0, 3.0, 5.0];
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[4]).unwrap());
    let target = Array::from_slice(&g, &t, &[4]).unwrap();
    let loss = xv.mse_loss(&target).unwrap();
    let lv = loss.value().to_vec().unwrap()[0];
    let want_l: f32 = x.iter().zip(&t).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / 4.0;
    assert!((lv - want_l).abs() <= 1e-4, "mse {lv} vs {want_l}");
    let gx = xv
        .mse_loss(&target)
        .unwrap()
        .grad(&xv)
        .unwrap()
        .to_vec()
        .unwrap();
    let want_g: Vec<f32> = x.iter().zip(&t).map(|(a, b)| 2.0 * (a - b) / 4.0).collect();
    assert_close(&gx, &want_g, 1e-3, "mse grad");
}

#[test]
fn grad_embedding_is_sparse_scatter() {
    // out = embedding(table, ids); L = sum(out²).
    // ∂L/∂table[r,:] = Σ_b 2·out[b,:]·[ids[b]==r] — sparse: only looked-up rows,
    // and a row looked up twice gets both contributions.
    let g = gpu();
    let (v, e) = (4usize, 2usize);
    let table_data: Vec<f32> = (0..v * e).map(|i| (i as f32) - 3.0).collect();
    let ids_data = vec![1u32, 3, 1]; // row 1 looked up twice, row 3 once, rows 0/2 never
    let tape = Tape::<f32>::new();
    let tv = tape.var(Array::from_slice(&g, &table_data, &[v, e]).unwrap());
    let ids = Array::from_slice(&g, &ids_data, &[3]).unwrap();
    let out = tv.embedding(&ids).unwrap();
    let loss = out.mul(&out).unwrap().sum().unwrap();
    let gt = loss.grad(&tv).unwrap().to_vec().unwrap();

    // host: scatter-add 2·(gathered value) back through ids.
    let mut want = vec![0.0f32; v * e];
    for &r in ids_data.iter() {
        let r = r as usize;
        for c in 0..e {
            let ov = table_data[r * e + c]; // out[b,c] = table[r,c]
            want[r * e + c] += 2.0 * ov;
        }
    }
    assert_close(&gt, &want, 1e-2, "embedding grad");
    // rows never looked up (0 and 2) must be exactly zero.
    #[allow(clippy::erasing_op)] // `0 * e + c` kept parallel to `2 * e + c` for readability
    for c in 0..e {
        assert!(gt[0 * e + c].abs() < 1e-6, "row 0 should be 0");
        assert!(gt[2 * e + c].abs() < 1e-6, "row 2 should be 0");
    }
}

#[test]
fn grad_transpose_routes_back() {
    // out = xᵀ; L = sum(out · W) with W a constant. ∂L/∂x = Wᵀ (the grad of a
    // transpose is the transpose of the upstream grad).
    let g = gpu();
    let x_data: Vec<f32> = (0..6).map(|i| i as f32).collect(); // [2, 3]
    let w_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.5 - 1.0).collect(); // [3, 2]
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x_data, &[2, 3]).unwrap());
    let wv = tape.var(Array::from_slice(&g, &w_data, &[3, 2]).unwrap());
    let xt = xv.transpose(0, 1).unwrap();
    assert_eq!(xt.value().shape(), &[3, 2]);
    // value: xᵀ
    assert_eq!(
        xt.value().to_vec().unwrap(),
        vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
    );
    // L = sum(xᵀ ⊙ W)  → ∂L/∂xᵀ = W → ∂L/∂x = Wᵀ
    let loss = xt.mul(&wv).unwrap().sum().unwrap();
    let gx = loss.grad(&xv).unwrap().to_vec().unwrap();
    // Wᵀ as a [2,3] flat vector.
    let mut want = vec![0.0f32; 6];
    for i in 0..3 {
        for j in 0..2 {
            want[j * 3 + i] = w_data[i * 2 + j]; // Wᵀ[j,i] = W[i,j]
        }
    }
    assert_close(&gx, &want, 1e-4, "transpose grad");
}

#[test]
fn layer_norm_normalizes_rows() {
    // With γ=1, β=0, each row of the output should be ~zero-mean, unit-var.
    let g = gpu();
    let (n, c) = (3usize, 5usize);
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32) * 0.7 - 4.0).collect();
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[n, c]).unwrap());
    let gamma = tape.var(Array::from_slice(&g, &vec![1.0f32; c], &[c]).unwrap());
    let beta = tape.var(Array::from_slice(&g, &vec![0.0f32; c], &[c]).unwrap());
    let out = xv
        .layer_norm(&gamma, &beta, 1e-5)
        .unwrap()
        .value()
        .to_vec()
        .unwrap();
    for i in 0..n {
        let row = &out[i * c..(i + 1) * c];
        let mean: f32 = row.iter().sum::<f32>() / c as f32;
        let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / c as f32;
        assert!(mean.abs() < 1e-3, "row {i} mean {mean} not ~0");
        assert!((var - 1.0).abs() < 1e-2, "row {i} var {var} not ~1");
    }
}

#[test]
fn layer_norm_affine_and_grad() {
    // γ scales, β shifts; check the gradient flows to all three (x, γ, β).
    let g = gpu();
    let (n, c) = (2usize, 4usize);
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32) - 3.0).collect();
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[n, c]).unwrap());
    let gamma = tape.var(Array::from_slice(&g, &[2.0f32, 2.0, 2.0, 2.0], &[c]).unwrap());
    let beta = tape.var(Array::from_slice(&g, &[0.5f32, 0.5, 0.5, 0.5], &[c]).unwrap());
    let out = xv.layer_norm(&gamma, &beta, 1e-5).unwrap();
    // rows still standardized then ·2 +0.5 → mean ≈ 0.5, var ≈ 4.
    let ov = out.value().to_vec().unwrap();
    for i in 0..n {
        let row = &ov[i * c..(i + 1) * c];
        let mean: f32 = row.iter().sum::<f32>() / c as f32;
        assert!((mean - 0.5).abs() < 1e-2, "row {i} mean {mean} != 0.5");
    }
    // gradients exist and are finite for all three parameters.
    let loss = out.mul(&out).unwrap().sum().unwrap();
    for (name, var) in [("x", &xv), ("gamma", &gamma), ("beta", &beta)] {
        let grad = loss.grad(var).unwrap().to_vec().unwrap();
        assert!(
            grad.iter().all(|v| v.is_finite()),
            "{name} grad has non-finite"
        );
        assert!(grad.iter().any(|&v| v.abs() > 1e-6), "{name} grad all ~0");
    }
}

#[test]
fn rms_norm_and_grad() {
    // rmsnorm(x)·γ, no mean-subtraction. Each row has unit RMS before scaling.
    let g = gpu();
    let (n, c) = (2usize, 4usize);
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32) - 3.0 + 0.5).collect();
    let eps = 1e-5f64;

    // Forward: check each row's RMS is ≈ 1 with γ = 1.
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x, &[n, c]).unwrap());
    let gamma1 = tape.var(Array::from_slice(&g, &vec![1.0f32; c], &[c]).unwrap());
    let out = xv.rms_norm(&gamma1, eps).unwrap();
    let ov = out.value().to_vec().unwrap();
    for i in 0..n {
        let row = &ov[i * c..(i + 1) * c];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / c as f32;
        assert!(
            (ms.sqrt() - 1.0).abs() < 1e-2,
            "row {i} rms {} != 1",
            ms.sqrt()
        );
    }

    // Finite-difference gradient check on x (loss = sum(out²), γ = 1).
    let host_loss = |xin: &[f32]| -> f32 {
        let mut l = 0.0f32;
        for i in 0..n {
            let row = &xin[i * c..(i + 1) * c];
            let ms = row.iter().map(|v| v * v).sum::<f32>() / c as f32 + eps as f32;
            let rms = ms.sqrt();
            for &v in row {
                let o = v / rms;
                l += o * o;
            }
        }
        l
    };
    let mut numeric = vec![0.0f32; n * c];
    let h = 1e-3f32;
    for j in 0..n * c {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[j] += h;
        xm[j] -= h;
        numeric[j] = (host_loss(&xp) - host_loss(&xm)) / (2.0 * h);
    }
    let tape2 = Tape::<f32>::new();
    let xv2 = tape2.var(Array::from_slice(&g, &x, &[n, c]).unwrap());
    let gamma2 = tape2.var(Array::from_slice(&g, &vec![1.0f32; c], &[c]).unwrap());
    let out2 = xv2.rms_norm(&gamma2, eps).unwrap();
    let loss2 = out2.mul(&out2).unwrap().sum().unwrap();
    let gx = loss2.grad(&xv2).unwrap().to_vec().unwrap();
    assert_close(&gx, &numeric, 3e-2, "rms_norm grad x");

    // γ gradient exists and is finite.
    let ggamma = loss2.grad(&gamma2).unwrap().to_vec().unwrap();
    assert!(ggamma.iter().all(|v| v.is_finite()) && ggamma.iter().any(|&v| v.abs() > 1e-6));
}

#[test]
fn grad_upsample2d() {
    // out = upsample2d(x, 2); L = sum(out · W). ∂L/∂x[p] = sum of W over the 2×2
    // block p maps to (upsample's adjoint is a block-sum).
    let g = gpu();
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0]; // [1,1,2,2]
    let tape = Tape::<f32>::new();
    let xv = tape.var(Array::from_slice(&g, &x_data, &[1, 1, 2, 2]).unwrap());
    let up = xv.upsample2d(2).unwrap();
    assert_eq!(up.value().shape(), &[1, 1, 4, 4]);
    // weight the 4×4 output by a fixed pattern, sum → scalar loss
    let w: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
    let wv = tape.var(Array::from_slice(&g, &w, &[1, 1, 4, 4]).unwrap());
    let loss = up.mul(&wv).unwrap().sum().unwrap();
    let gx = loss.grad(&xv).unwrap().to_vec().unwrap();
    // ∂L/∂x[i,j] = sum of w over the 2×2 block rows [2i,2i+2) cols [2j,2j+2)
    let mut want = vec![0.0f32; 4];
    for ii in 0..2 {
        for jj in 0..2 {
            let mut s = 0.0;
            for ki in 0..2 {
                for kj in 0..2 {
                    s += w[(2 * ii + ki) * 4 + (2 * jj + kj)];
                }
            }
            want[ii * 2 + jj] = s;
        }
    }
    assert_close(&gx, &want, 1e-4, "upsample2d grad");
}

#[test]
fn grad_silu() {
    // f(x) = x·σ(x) ⇒ f'(x) = σ(x)·(1 + x·(1 − σ(x)))
    let g = gpu();
    let x = vec![-2.0f32, -0.5, 0.0, 0.5, 1.0, 3.0];
    let an = analytic_grad(&g, &x, |v| v.silu().unwrap());
    let num = numeric_grad(&x, |x| {
        let s = 1.0 / (1.0 + (-x).exp());
        x * s
    });
    assert_close(&an, &num, 2e-2, "silu");
}

#[test]
fn grad_gelu() {
    // f(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))  (tanh approximation).
    let g = gpu();
    let x = vec![-2.0f32, -0.5, 0.0, 0.5, 1.0, 3.0];
    let an = analytic_grad(&g, &x, |v| v.gelu().unwrap());
    let num = numeric_grad(&x, |x| {
        let c = (2.0f32 / std::f32::consts::PI).sqrt();
        let inner = c * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    });
    assert_close(&an, &num, 2e-2, "gelu");
}

/// operand `which` ("a" or "b"), evaluated purely on the host via the batched
/// matmul forward — the reference the tape must match.
fn numeric_matmul_grad(
    g: &quanta::Gpu,
    ash: &[usize],
    bsh: &[usize],
    a: &[f32],
    b: &[f32],
    which: &str,
) -> Vec<f32> {
    use quanta_array::Array;
    let loss = |a: &[f32], b: &[f32]| -> f32 {
        let av = Array::from_slice(g, a, ash).unwrap();
        let bv = Array::from_slice(g, b, bsh).unwrap();
        av.matmul(&bv).unwrap().to_vec().unwrap().iter().sum()
    };
    let h = 1e-3f32;
    let target = if which == "a" { a } else { b };
    (0..target.len())
        .map(|i| {
            let mut pp = target.to_vec();
            let mut pm = target.to_vec();
            pp[i] += h;
            pm[i] -= h;
            let (lp, lm) = if which == "a" {
                (loss(&pp, b), loss(&pm, b))
            } else {
                (loss(a, &pp), loss(a, &pm))
            };
            (lp - lm) / (2.0 * h)
        })
        .collect()
}

fn matmul_gradcheck(ash: &[usize], bsh: &[usize], seed: u32) {
    use quanta_array::Array;
    let g = gpu();
    let asz: usize = ash.iter().product();
    let bsz: usize = bsh.iter().product();
    let a: Vec<f32> = (0..asz)
        .map(|i| (((i as u32 + seed) % 7) as f32 - 3.0) * 0.5)
        .collect();
    let b: Vec<f32> = (0..bsz)
        .map(|i| (((i as u32 + seed * 3) % 5) as f32 - 2.0) * 0.5)
        .collect();

    let tape = Tape::<f32>::new();
    let av = tape.var(Array::from_slice(&g, &a, ash).unwrap());
    let bv = tape.var(Array::from_slice(&g, &b, bsh).unwrap());
    let loss = av.matmul(&bv).unwrap().sum().unwrap();
    let ga = loss.grad(&av).unwrap().to_vec().unwrap();
    let gb = loss.grad(&bv).unwrap().to_vec().unwrap();

    let na = numeric_matmul_grad(&g, ash, bsh, &a, &b, "a");
    let nb = numeric_matmul_grad(&g, ash, bsh, &a, &b, "b");
    // gradient shapes must match the operands (broadcast reduction applied)
    assert_eq!(ga.len(), asz, "∂A shape");
    assert_eq!(gb.len(), bsz, "∂B shape");
    assert_close(&ga, &na, 2e-2, "matmul ∂A");
    assert_close(&gb, &nb, 2e-2, "matmul ∂B");
}

#[test]
fn grad_matmul_2d() {
    matmul_gradcheck(&[3, 4], &[4, 2], 1);
}

#[test]
fn grad_matmul_batched() {
    matmul_gradcheck(&[2, 3, 4], &[2, 4, 5], 7);
}

#[test]
fn grad_matmul_broadcast_rhs() {
    // (B,m,k)·(k,n): ∂B must be summed back over the broadcast batch.
    matmul_gradcheck(&[3, 2, 4], &[4, 2], 13);
}

// ── Multi-head self-attention ───────────────────────────────────────────────

/// Deterministic small init in [-0.1, 0.1].
fn mha_init(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i as f32) * 12.9898 + seed).sin() * 43758.547).fract() * 0.2 - 0.1)
        .collect()
}

/// Build multi-head attention and return `L = sum(out²)`, on a fresh tape from
/// the given host tensors. `xh` is the perturbable input [B,T,D]; the weights
/// are fixed. Used both for the analytic (tape) and numeric (host-rebuild) grad.
#[allow(clippy::too_many_arguments)]
fn mha_loss(
    g: &quanta::Gpu,
    xh: &[f32],
    b: usize,
    t: usize,
    d: usize,
    heads: usize,
    wqh: &[f32],
    wkh: &[f32],
    wvh: &[f32],
    woh: &[f32],
) -> f32 {
    let tape = Tape::<f32>::new();
    let x = tape.var(Array::from_slice(g, xh, &[b, t, d]).unwrap());
    let wq = tape.var(Array::from_slice(g, wqh, &[d, d]).unwrap());
    let wk = tape.var(Array::from_slice(g, wkh, &[d, d]).unwrap());
    let wv = tape.var(Array::from_slice(g, wvh, &[d, d]).unwrap());
    let wo = tape.var(Array::from_slice(g, woh, &[d, d]).unwrap());
    let out = x
        .multi_head_attention(&wq, &wk, &wv, &wo, heads, None)
        .unwrap();
    out.mul(&out)
        .unwrap()
        .sum()
        .unwrap()
        .value()
        .to_vec()
        .unwrap()[0]
}

#[test]
fn grad_multi_head_attention() {
    use quanta_autograd::Var;
    let g = gpu();
    let (b, t, d, heads) = (1usize, 3usize, 4usize, 2usize);
    let xh = mha_init(b * t * d, 1.0);
    let wqh = mha_init(d * d, 2.0);
    let wkh = mha_init(d * d, 3.0);
    let wvh = mha_init(d * d, 4.0);
    let woh = mha_init(d * d, 5.0);

    // Analytic gradient on x via the tape.
    let tape = Tape::<f32>::new();
    let x = tape.var(Array::from_slice(&g, &xh, &[b, t, d]).unwrap());
    let wq = tape.var(Array::from_slice(&g, &wqh, &[d, d]).unwrap());
    let wk = tape.var(Array::from_slice(&g, &wkh, &[d, d]).unwrap());
    let wv = tape.var(Array::from_slice(&g, &wvh, &[d, d]).unwrap());
    let wo = tape.var(Array::from_slice(&g, &woh, &[d, d]).unwrap());
    let out: Var<f32> = x
        .multi_head_attention(&wq, &wk, &wv, &wo, heads, None)
        .unwrap();
    let loss = out.mul(&out).unwrap().sum().unwrap();
    let gx = loss.grad(&x).unwrap().to_vec().unwrap();
    // also confirm a weight gradient is finite & non-trivial
    let gwq = loss.grad(&wq).unwrap().to_vec().unwrap();
    assert!(
        gwq.iter().all(|v| v.is_finite()) && gwq.iter().any(|&v| v.abs() > 1e-6),
        "wq grad degenerate"
    );

    // Numeric gradient on x via central difference.
    let h = 1e-3f32;
    let mut numeric = vec![0.0f32; xh.len()];
    for j in 0..xh.len() {
        let mut xp = xh.clone();
        let mut xm = xh.clone();
        xp[j] += h;
        xm[j] -= h;
        let lp = mha_loss(&g, &xp, b, t, d, heads, &wqh, &wkh, &wvh, &woh);
        let lm = mha_loss(&g, &xm, b, t, d, heads, &wqh, &wkh, &wvh, &woh);
        numeric[j] = (lp - lm) / (2.0 * h);
    }
    assert_close(&gx, &numeric, 2e-2, "multi_head_attention grad x");
}
