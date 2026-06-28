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
