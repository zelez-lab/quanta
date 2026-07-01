//! A tiny convolutional network, trained end-to-end on quanta-autograd.
//!
//! The task: classify 4×4 single-channel images as a **horizontal** stripe
//! (one bright row) vs a **vertical** stripe (one bright column). It's the
//! smallest task where convolution earns its keep — a 3×3 filter learns an
//! oriented-edge detector, which a plain linear model on raw pixels can't
//! express position-invariantly.
//!
//! Architecture:
//!   x[N,1,4,4] → conv2d(1→C, 3×3, pad 1) → relu → maxpool2d(2×2, stride 2)
//!             → flatten[N, C·2·2] → linear → sigmoid → MSE vs {0,1}
//!
//! This exercises the whole conv stack composing into a network: conv2d, relu,
//! maxpool2d (with its argmax-routed gradient), flatten, matmul, a broadcast
//! bias, a sigmoid, an MSE loss, the backward pass, and SGD updates — every
//! gradient being the proven analytic one.
//!
//! Run it:
//!   cargo run --example cnn_training -p quanta-autograd --release
//! (release matters — the software lane JITs a kernel per op per step.)

use quanta_array::Array;
use quanta_autograd::{Tape, Var};

/// `p ← p − lr·g` — the optimizer step, plain array ops outside the tape.
fn sgd(p: &Array<f32>, g: &Array<f32>, lr: f32) -> Array<f32> {
    let lr_a = Array::full(p.gpu(), lr, &[1])
        .unwrap()
        .broadcast_to(g.shape())
        .unwrap();
    p.sub(&g.mul(&lr_a).unwrap()).unwrap()
}

/// One 4×4 image: a single bright line. `horizontal` puts it on row `idx`,
/// otherwise on column `idx`. Background 0, line 1.
fn stripe(horizontal: bool, idx: usize) -> Vec<f32> {
    let mut img = vec![0.0f32; 16];
    for k in 0..4 {
        let (r, c) = if horizontal { (idx, k) } else { (k, idx) };
        img[r * 4 + c] = 1.0;
    }
    img
}

fn main() {
    let gpu = quanta::init_cpu();
    let channels = 4usize; // conv output channels

    // Dataset: every horizontal stripe (4) and every vertical stripe (4),
    // labelled 0 / 1. 8 images, fully deterministic.
    let mut imgs: Vec<f32> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();
    for idx in 0..4 {
        imgs.extend(stripe(true, idx));
        labels.push(0.0);
    }
    for idx in 0..4 {
        imgs.extend(stripe(false, idx));
        labels.push(1.0);
    }
    let n = labels.len();
    let x = Array::from_slice(&gpu, &imgs, &[n, 1, 4, 4]).unwrap();
    let y = Array::from_slice(&gpu, &labels, &[n, 1]).unwrap();

    // Parameters. A fixed spread-out init keeps the run deterministic.
    let init = |shape: &[usize], scale: f32, phase: f32| {
        let count: usize = shape.iter().product();
        let v: Vec<f32> = (0..count)
            .map(|i| scale * (i as f32 * 1.7 + phase).sin())
            .collect();
        Array::from_slice(&gpu, &v, shape).unwrap()
    };
    // conv weight [Cout, Cin, kh, kw] and the linear head [C·2·2, 1].
    let mut wc = init(&[channels, 1, 3, 3], 0.5, 0.0);
    let flat = channels * 2 * 2;
    let mut wl = init(&[flat, 1], 0.4, 0.5);
    let mut bl = Array::zeros(&gpu, &[1, 1]).unwrap();

    let lr = 0.5f32;
    println!("epoch    loss");
    for epoch in 0..400 {
        let tape = Tape::<f32>::new();
        let xv = tape.var(x.shallow_clone());
        let yv = tape.var(y.shallow_clone());
        let wcv = tape.var(wc.shallow_clone());
        let wlv = tape.var(wl.shallow_clone());
        let blv = tape.var(bl.shallow_clone());

        // Forward: conv → relu → maxpool → flatten → linear+bias → sigmoid.
        let feat = xv
            .conv2d(&wcv, 1, 1)
            .unwrap()
            .relu()
            .unwrap()
            .maxpool2d(2, 2, 2, 0)
            .unwrap()
            .flatten()
            .unwrap();
        let pred = feat
            .matmul(&wlv)
            .unwrap()
            .add(&blv)
            .unwrap()
            .sigmoid()
            .unwrap();
        // loss = mean((pred − y)²)
        let diff = pred.sub(&yv).unwrap();
        let loss = diff
            .mul(&diff)
            .unwrap()
            .mean_axis(0)
            .unwrap()
            .sum()
            .unwrap();

        if epoch % 50 == 0 || epoch == 399 {
            println!("{epoch:5}  {:.6}", loss.value().to_vec().unwrap()[0]);
        }

        let vars: [&Var<f32>; 3] = [&wcv, &wlv, &blv];
        let grads: Vec<Array<f32>> = vars.iter().map(|v| loss.grad(v).unwrap()).collect();
        wc = sgd(&wc, &grads[0], lr);
        wl = sgd(&wl, &grads[1], lr);
        bl = sgd(&bl, &grads[2], lr);
    }

    // Evaluate: predicted probability per image, and the accuracy.
    let tape = Tape::<f32>::new();
    let xv = tape.var(x.shallow_clone());
    let wcv = tape.var(wc.shallow_clone());
    let wlv = tape.var(wl.shallow_clone());
    let blv = tape.var(bl.shallow_clone());
    let probs = xv
        .conv2d(&wcv, 1, 1)
        .unwrap()
        .relu()
        .unwrap()
        .maxpool2d(2, 2, 2, 0)
        .unwrap()
        .flatten()
        .unwrap()
        .matmul(&wlv)
        .unwrap()
        .add(&blv)
        .unwrap()
        .sigmoid()
        .unwrap()
        .value()
        .to_vec()
        .unwrap();

    println!("\n  image          label  pred  ✓");
    let mut correct = 0;
    for i in 0..n {
        let kind = if labels[i] == 0.0 {
            "horizontal"
        } else {
            "vertical  "
        };
        let hit = (probs[i] >= 0.5) == (labels[i] >= 0.5);
        correct += hit as usize;
        println!(
            "  {kind}    {:.0}    {:.3}  {}",
            labels[i],
            probs[i],
            if hit { "✓" } else { "✗" }
        );
    }
    println!("\naccuracy: {correct}/{n}");
}
