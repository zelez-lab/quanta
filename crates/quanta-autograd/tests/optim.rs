//! Optimizer tests: SGD + Adam both minimize a simple quadratic.
use quanta_array::Array;
use quanta_autograd::{
    Tape,
    optim::{Adam, Sgd},
};

fn gpu() -> quanta::Gpu {
    quanta::init_cpu()
}

/// loss = sum((p - target)²); gradient = 2(p - target). Minimizer is p = target.
fn quad_grad(p: &Array<f32>, target: &Array<f32>) -> (f32, Array<f32>) {
    let tape = Tape::<f32>::new();
    let pv = tape.var(p.shallow_clone());
    let tv = tape.var(target.shallow_clone());
    let diff = pv.sub(&tv).unwrap();
    let loss = diff.mul(&diff).unwrap().sum().unwrap();
    let g = loss.grad(&pv).unwrap();
    (loss.value().to_vec().unwrap()[0], g)
}

#[test]
fn sgd_minimizes_quadratic() {
    let g = gpu();
    let target = Array::from_slice(&g, &[3.0f32, -1.0, 2.0], &[3]).unwrap();
    let mut p = Array::<f32>::zeros(&g, &[3]).unwrap();
    let opt = Sgd::new(0.1);
    for _ in 0..100 {
        let (_l, grad) = quad_grad(&p, &target);
        p = opt.step(&p, &grad).unwrap();
    }
    let pv = p.to_vec().unwrap();
    for (a, b) in pv.iter().zip([3.0, -1.0, 2.0]) {
        assert!((a - b).abs() <= 1e-2, "sgd got {a}, want {b}");
    }
}

#[test]
fn adam_minimizes_quadratic() {
    let g = gpu();
    let target = Array::from_slice(&g, &[3.0f32, -1.0, 2.0], &[3]).unwrap();
    let mut p = Array::<f32>::zeros(&g, &[3]).unwrap();
    let mut opt = Adam::new(0.2);
    opt.register(&p).unwrap();
    for _ in 0..300 {
        opt.advance();
        let (_l, grad) = quad_grad(&p, &target);
        p = opt.step(0, &p, &grad).unwrap();
    }
    let pv = p.to_vec().unwrap();
    for (a, b) in pv.iter().zip([3.0, -1.0, 2.0]) {
        assert!((a - b).abs() <= 1e-2, "adam got {a}, want {b}");
    }
}
