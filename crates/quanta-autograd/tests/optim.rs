//! Optimizer tests: SGD + Adam both minimize a simple quadratic.
use quanta_array::Array;
use quanta_autograd::{
    Tape,
    optim::{Adam, Sgd},
};

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

#[test]
fn adamw_weight_decay_pulls_toward_zero() {
    // AdamW's decoupled decay shrinks parameters toward 0, so the converged
    // point lands *short of* the plain-Adam target (the loss minimum) on the
    // same problem — the observable signature of decoupled weight decay.
    let g = gpu();
    let target = Array::from_slice(&g, &[3.0f32, -1.0, 2.0], &[3]).unwrap();

    let solve = |wd: f32| -> Vec<f32> {
        let mut p = Array::<f32>::zeros(&g, &[3]).unwrap();
        let mut opt = Adam::adamw(0.2, wd);
        opt.register(&p).unwrap();
        for _ in 0..300 {
            opt.advance();
            let (_l, grad) = quad_grad(&p, &target);
            p = opt.step(0, &p, &grad).unwrap();
        }
        p.to_vec().unwrap()
    };

    let plain = solve(0.0); // AdamW with wd=0 == Adam → reaches the target
    let decayed = solve(0.1); // nonzero decay → pulled toward zero

    // wd=0 recovers plain Adam (hits the target).
    for (a, b) in plain.iter().zip([3.0, -1.0, 2.0]) {
        assert!((a - b).abs() <= 1e-2, "adamw(wd=0) got {a}, want {b}");
    }
    // With decay, each converged coordinate is strictly smaller in magnitude
    // than the target (shrunk toward 0), and still same-signed.
    for (d, t) in decayed.iter().zip([3.0f32, -1.0, 2.0]) {
        assert!(
            d.abs() < t.abs() && d.signum() == t.signum(),
            "adamw(wd=0.1) coord {d} should be shrunk toward 0 from {t}"
        );
    }
}
