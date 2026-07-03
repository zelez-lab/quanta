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

/// Word2vec-style skip-gram: a single embedding table trained so co-occurring
/// token pairs have a high dot-product and cross-topic pairs low. The parity
/// example for PyTorch `nn.Embedding` + a dot-product objective — proves the
/// differentiable embedding lookup and its sparse gradient train end to end.
#[test]
fn word2vec_embeddings_separate_topics() {
    use quanta_autograd::optim::Adam;

    let g = gpu();
    let (vocab, e) = (6usize, 4usize);
    // tokens 0,1,2 = topic A; 3,4,5 = topic B. positives within, negatives across.
    let centers: Vec<u32> = vec![0, 1, 2, 0, 3, 4, 5, 3, 0, 1, 3, 4];
    let contexts: Vec<u32> = vec![1, 2, 0, 2, 4, 5, 3, 5, 3, 4, 0, 1];
    let labels: Vec<f32> = vec![1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.];
    let b = centers.len();

    let cen = Array::from_slice(&g, &centers, &[b]).unwrap();
    let ctx = Array::from_slice(&g, &contexts, &[b]).unwrap();
    let lab = Array::from_slice(&g, &labels, &[b, 1]).unwrap();
    let init: Vec<f32> = (0..vocab * e)
        .map(|i| 0.1 * (i as f32 * 1.7).sin())
        .collect();
    let mut emb = Array::from_slice(&g, &init, &[vocab, e]).unwrap();

    let mut opt = Adam::new(0.1);
    opt.register(&emb).unwrap();

    for _ in 0..60 {
        opt.advance();
        let tape = Tape::<f32>::new();
        let ev = tape.var(emb.shallow_clone());
        let ce = ev.embedding(&cen).unwrap();
        let co = ev.embedding(&ctx).unwrap();
        let dot = ce.mul(&co).unwrap().sum_axis(1).unwrap();
        let loss = dot.sigmoid().unwrap().mse_loss(&lab).unwrap();
        let ge = loss.grad(&ev).unwrap();
        emb = opt.step(0, &emb, &ge).unwrap();
    }

    let tape = Tape::<f32>::new();
    let ev = tape.var(emb.shallow_clone());
    let score = ev
        .embedding(&cen)
        .unwrap()
        .mul(&ev.embedding(&ctx).unwrap())
        .unwrap()
        .sum_axis(1)
        .unwrap()
        .sigmoid()
        .unwrap()
        .value()
        .to_vec()
        .unwrap();
    let correct = (0..b)
        .filter(|&i| ((if score[i] > 0.5 { 1.0 } else { 0.0 }) - labels[i]).abs() < 0.5)
        .count();
    assert!(
        correct >= b - 1,
        "embeddings didn't separate topics: {correct}/{b}"
    );
}

/// Single-head self-attention, composed from matmul + transpose + softmax +
/// scale — the core of a transformer. Trains an induction task: each position
/// must output the payload of the *next* position, solvable only by attending
/// there. Proves the attention pattern trains end to end through the tape.
#[test]
fn self_attention_learns_an_induction_shift() {
    use quanta_autograd::optim::Adam;

    let g = gpu();
    let (s, d, dh) = (4usize, 8usize, 8usize);

    // input: each position = a position one-hot (dims 0..4) + a payload (4..8)
    let mut xd = vec![0.0f32; s * d];
    for i in 0..s {
        xd[i * d + i] = 1.0;
        xd[i * d + 4 + (i % 4)] = 0.5 + 0.1 * i as f32;
    }
    let x = Array::from_slice(&g, &xd, &[s, d]).unwrap();
    // target: position i outputs the payload of position (i+1)%S — a shift.
    let mut target = vec![0.0f32; s * dh];
    for i in 0..s {
        let src = (i + 1) % s;
        target[i * dh + 4 + (src % 4)] = 0.5 + 0.1 * src as f32;
    }
    let tgt = Array::from_slice(&g, &target, &[s, dh]).unwrap();

    let init = |shape: &[usize], sc: f32| {
        let c: usize = shape.iter().product();
        let v: Vec<f32> = (0..c).map(|j| sc * ((j as f32) * 1.3).sin()).collect();
        Array::from_slice(&g, &v, shape).unwrap()
    };
    let (mut wq, mut wk, mut wv) = (
        init(&[d, dh], 0.2),
        init(&[d, dh], 0.2),
        init(&[d, dh], 0.2),
    );
    let inv_sqrt_d = 1.0 / (dh as f32).sqrt();

    let mut opt = Adam::new(0.02);
    opt.register(&wq).unwrap();
    opt.register(&wk).unwrap();
    opt.register(&wv).unwrap();

    let mut final_loss = 1.0f32;
    for _ in 0..50 {
        opt.advance();
        let tape = Tape::<f32>::new();
        let xv = tape.var(x.shallow_clone());
        let wqv = tape.var(wq.shallow_clone());
        let wkv = tape.var(wk.shallow_clone());
        let wvv = tape.var(wv.shallow_clone());
        let scale = tape.var(
            Array::full(&g, inv_sqrt_d, &[1])
                .unwrap()
                .broadcast_to(&[s, s])
                .unwrap()
                .contiguous()
                .unwrap(),
        );

        // out = softmax((X·Wq)(X·Wk)ᵀ / √d)·(X·Wv)
        let q = xv.matmul(&wqv).unwrap();
        let k = xv.matmul(&wkv).unwrap();
        let v = xv.matmul(&wvv).unwrap();
        let scores = q
            .matmul(&k.transpose(0, 1).unwrap())
            .unwrap()
            .mul(&scale)
            .unwrap();
        let out = scores.softmax().unwrap().matmul(&v).unwrap();
        let loss = out.mse_loss(&tgt).unwrap();

        let gq = loss.grad(&wqv).unwrap();
        let gk = loss.grad(&wkv).unwrap();
        let gv = loss.grad(&wvv).unwrap();
        wq = opt.step(0, &wq, &gq).unwrap();
        wk = opt.step(1, &wk, &gk).unwrap();
        wv = opt.step(2, &wv, &gv).unwrap();
        final_loss = loss.value().to_vec().unwrap()[0];
    }

    assert!(
        final_loss < 0.02,
        "self-attention didn't learn the shift: final loss {final_loss}"
    );
}

/// Convolutional autoencoder — encode an image to a spatial bottleneck, decode
/// it back with the new upsample path, and train to reconstruct. Exercises
/// upsample2d in a real decoder with gradients flowing through it.
///
/// `#[ignore]`: conv2d over 70 epochs is minutes on the CPU-interpreter lane.
/// Run it explicitly (`cargo test conv_autoencoder -- --ignored`) or on a real
/// GPU (`--features metal`), where it finishes in seconds. The upsample op
/// itself is covered fast by `gradcheck::grad_upsample2d` and the array tests.
#[test]
#[ignore = "slow on the CPU interpreter; run with --ignored or --features metal"]
fn conv_autoencoder_reconstructs() {
    use quanta_autograd::optim::Adam;

    let g = gpu();
    let (n, h, w) = (4usize, 8usize, 8usize);
    let mut imgs = vec![0.0f32; n * h * w];
    for b in 0..n {
        for i in 0..h {
            for j in 0..w {
                imgs[(b * h + i) * w + j] = ((i + j + b) as f32 * 0.3).sin() * 0.5 + 0.5;
            }
        }
    }
    let x = Array::from_slice(&g, &imgs, &[n, 1, h, w]).unwrap();

    let init = |shape: &[usize], s: f32| {
        let c: usize = shape.iter().product();
        let v: Vec<f32> = (0..c).map(|i| s * ((i as f32) * 1.7).sin()).collect();
        Array::from_slice(&g, &v, shape).unwrap()
    };
    let mut we = init(&[4, 1, 3, 3], 0.3); // encoder conv
    let mut wd = init(&[1, 4, 3, 3], 0.2); // decoder conv

    let mut opt = Adam::new(0.01);
    opt.register(&we).unwrap();
    opt.register(&wd).unwrap();

    let mut final_loss = 1.0f32;
    for _ in 0..70 {
        opt.advance();
        let tape = Tape::<f32>::new();
        let xv = tape.var(x.shallow_clone());
        let wev = tape.var(we.shallow_clone());
        let wdv = tape.var(wd.shallow_clone());

        // encode → [n,4,4,4] bottleneck ; decode → [n,1,8,8]
        let z = xv
            .conv2d(&wev, 1, 1)
            .unwrap()
            .relu()
            .unwrap()
            .maxpool2d(2, 2, 2, 0)
            .unwrap();
        let recon = z.upsample2d(2).unwrap().conv2d(&wdv, 1, 1).unwrap();
        let loss = recon.mse_loss(&x).unwrap();

        we = opt.step(0, &we, &loss.grad(&wev).unwrap()).unwrap();
        wd = opt.step(1, &wd, &loss.grad(&wdv).unwrap()).unwrap();
        final_loss = loss.value().to_vec().unwrap()[0];
    }

    // reconstruct far better than predicting the per-image mean.
    let mean: f32 = imgs.iter().sum::<f32>() / imgs.len() as f32;
    let baseline: f32 = imgs.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / imgs.len() as f32;
    assert!(
        final_loss < baseline * 0.3,
        "autoencoder didn't reconstruct: {final_loss} vs baseline {baseline}"
    );
}
