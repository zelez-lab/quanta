//! End-to-end training sanity: gradient descent actually fits a model.
//!
//! Per-op gradient checks prove each VJP is right; this proves the *loop* is
//! right — forward → scalar loss → backward → parameter update → repeat
//! converges to the known solution. A sign error or mis-accumulation that
//! survives the per-op checks would show up here as a loss that doesn't drop or
//! parameters that don't reach ground truth.

use quanta_array::Array;
use quanta_autograd::Tape;

fn gpu() -> quanta::Gpu {
    quanta::init_cpu()
}

/// `p ← p - lr·g`, as plain array ops (the optimizer step lives outside the
/// tape; each training step builds a fresh tape from the current params).
fn sgd_step(p: &Array<f32>, g: &Array<f32>, lr: f32) -> Array<f32> {
    let lr_a = Array::full(p.gpu(), lr, &[1]).unwrap();
    let scaled = g.mul(&lr_a.broadcast_to(g.shape()).unwrap()).unwrap();
    p.sub(&scaled).unwrap()
}

#[test]
fn fits_linear_regression() {
    // Ground truth: y = 2·x + 0.5. Fit w, b by GD on MSE.
    let g = gpu();
    let n = 16usize;
    let xs: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
    let ys: Vec<f32> = xs.iter().map(|&x| 2.0 * x + 0.5).collect();

    let x = Array::from_slice(&g, &xs, &[n, 1]).unwrap();
    let y = Array::from_slice(&g, &ys, &[n, 1]).unwrap();

    // Parameters (start away from the solution).
    let mut w = Array::from_slice(&g, &[0.0f32], &[1, 1]).unwrap();
    let mut b = Array::from_slice(&g, &[0.0f32], &[1, 1]).unwrap();

    // High learning rate + a small problem converges fast; keep the iteration
    // count low so the CPU-lane test (a fresh JIT per op per step) stays quick.
    let lr = 0.8f32;
    let mut first_loss = None;
    let mut last_loss = 0.0f32;

    for _ in 0..120 {
        let tape = Tape::<f32>::new();
        let xv = tape.var(x.shallow_clone());
        let yv = tape.var(y.shallow_clone());
        let wv = tape.var(w.shallow_clone());
        let bv = tape.var(b.shallow_clone());

        // pred = x·w + b   (matmul [n,1]·[1,1] → [n,1]; b [1,1] broadcasts)
        let pred = xv.matmul(&wv).unwrap().add(&bv).unwrap();
        // loss = mean((pred − y)²)
        let diff = pred.sub(&yv).unwrap();
        let sq = diff.mul(&diff).unwrap();
        // mean over rows then the (size-1) column → scalar.
        let loss = sq.mean_axis(0).unwrap().sum().unwrap();

        last_loss = loss.value().to_vec().unwrap()[0];
        first_loss.get_or_insert(last_loss);

        let gw = loss.grad(&wv).unwrap();
        let gb = loss.grad(&bv).unwrap();
        w = sgd_step(&w, &gw, lr);
        b = sgd_step(&b, &gb, lr);
    }

    let wf = w.to_vec().unwrap()[0];
    let bf = b.to_vec().unwrap()[0];
    let first = first_loss.unwrap();

    // The loop must actually learn: loss collapses and params approach truth.
    assert!(
        last_loss < first * 1e-2,
        "loss didn't drop: {first} → {last_loss}"
    );
    assert!((wf - 2.0).abs() < 1e-1, "w = {wf}, want 2.0");
    assert!((bf - 0.5).abs() < 1e-1, "b = {bf}, want 0.5");
}
