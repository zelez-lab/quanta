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
