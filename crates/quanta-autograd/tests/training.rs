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

/// Softmax (multinomial logistic) regression — a single linear layer trained
/// with cross-entropy separates three linearly-separable classes to ~100%.
/// The parity example for scikit's `LogisticRegression(multi_class="multinomial")`.
#[test]
fn softmax_regression_separates_three_classes() {
    use quanta_autograd::optim::Adam;

    let g = gpu();
    let (n_per, d, k) = (10usize, 2usize, 3usize);
    let n = n_per * k;
    // widely separated blobs → separable in few epochs (keeps the CPU test fast)
    let centers = [(0.0f32, 0.0), (6.0, 0.0), (3.0, 6.0)];
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (c, (cx, cy)) in centers.iter().enumerate() {
        for i in 0..n_per {
            let a = i as f32 * 2.399;
            let r = 0.6 * ((i % 7) as f32 / 7.0);
            xs.push(cx + r * a.cos());
            xs.push(cy + r * a.sin());
            ys.push(c as u32);
        }
    }
    let x = Array::from_slice(&g, &xs, &[n, d]).unwrap();
    let y = Array::from_slice(&g, &ys, &[n]).unwrap();

    let init = |shape: &[usize], s: f32| {
        let c: usize = shape.iter().product();
        let v: Vec<f32> = (0..c).map(|i| s * (i as f32 * 1.3).sin()).collect();
        Array::from_slice(&g, &v, shape).unwrap()
    };
    let mut w = init(&[d, k], 0.1);
    let mut b = Array::<f32>::zeros(&g, &[1, k]).unwrap();
    let mut opt = Adam::new(0.05);
    opt.register(&w).unwrap();
    opt.register(&b).unwrap();

    for _ in 0..40 {
        opt.advance();
        let tape = Tape::<f32>::new();
        let xv = tape.var(x.shallow_clone());
        let wv = tape.var(w.shallow_clone());
        let bv = tape.var(b.shallow_clone());
        let logits = xv.matmul(&wv).unwrap().add(&bv).unwrap();
        let loss = logits.cross_entropy(&y).unwrap();
        let gw = loss.grad(&wv).unwrap();
        let gb = loss.grad(&bv).unwrap();
        w = opt.step(0, &w, &gw).unwrap();
        b = opt.step(1, &b, &gb).unwrap();
    }

    let tape = Tape::<f32>::new();
    let logits = tape
        .var(x.shallow_clone())
        .matmul(&tape.var(w.shallow_clone()))
        .unwrap()
        .add(&tape.var(b.shallow_clone()))
        .unwrap();
    let pred = logits.value().argmax_last().unwrap().to_vec().unwrap();
    let correct = pred.iter().zip(ys.iter()).filter(|(a, b)| a == b).count();
    assert!(
        correct as f32 / n as f32 > 0.95,
        "softmax regression only got {correct}/{n}"
    );
}
