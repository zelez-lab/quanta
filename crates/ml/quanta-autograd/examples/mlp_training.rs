//! A tiny neural network, trained end-to-end on quanta-autograd.
//!
//! A 2-layer MLP — `h = tanh(x·W1 + b1); ŷ = h·W2 + b2` — learns the nonlinear
//! target `y = x²` on `[-1, 1]` by SGD on mean-squared error. The hidden layer
//! (with a tanh nonlinearity) is what lets a linear output represent a curve.
//!
//! This is the tutorial's worked example (docs: Training an MLP) — it exercises
//! the whole stack composing into a network: matmul, broadcast bias, a tanh
//! activation, an MSE loss (sub / mul / mean), the backward pass, and an SGD
//! parameter update.
//!
//! Runs on the real GPU when built with a hardware backend feature; otherwise
//! it falls back to the portable CPU JIT (slower, but no GPU required):
//!   cargo run --example mlp_training -p quanta-autograd --release --features metal
//!   cargo run --example mlp_training -p quanta-autograd --release            # CPU JIT
//! (release matters — every op JITs a kernel per step.)
//!
//! Bounded by design: 16 samples, 300 epochs — well under a minute either way.
//! To make it a bigger fit, raise `n` (more samples), `hidden` (wider net), or
//! the epoch count; drop `lr` if a wider net starts to diverge.

use quanta_array::Array;
use quanta_autograd::{Tape, Var};

/// The device this example runs on. With a hardware backend feature
/// (`metal` / `vulkan`) it trains on the real GPU; otherwise it falls back to
/// the portable CPU JIT interpreter.
fn gpu() -> quanta::Gpu {
    #[cfg(any(feature = "metal", feature = "vulkan"))]
    {
        quanta::init().expect("a GPU device (metal/vulkan feature is on)")
    }
    #[cfg(not(any(feature = "metal", feature = "vulkan")))]
    {
        quanta::init_cpu()
    }
}

/// `p ← p − lr·g` — the optimizer step, plain array ops outside the tape.
fn sgd(p: &Array<f32>, g: &Array<f32>, lr: f32) -> Array<f32> {
    let lr_a = Array::full(p.gpu(), lr, &[1])
        .unwrap()
        .broadcast_to(g.shape())
        .unwrap();
    p.sub(&g.mul(&lr_a).unwrap()).unwrap()
}

fn main() {
    let gpu = gpu();
    let n = 16usize;
    let hidden = 6usize;

    // Dataset: x ∈ [-1, 1], y = x².
    let xs: Vec<f32> = (0..n)
        .map(|i| -1.0 + 2.0 * i as f32 / (n - 1) as f32)
        .collect();
    let ys: Vec<f32> = xs.iter().map(|&x| x * x).collect();
    let x = Array::from_slice(&gpu, &xs, &[n, 1]).unwrap();
    let y = Array::from_slice(&gpu, &ys, &[n, 1]).unwrap();

    // Parameters. A fixed, spread-out init keeps the run deterministic and the
    // tanh units from starting saturated or all-identical.
    let init = |rows: usize, cols: usize, scale: f32, phase: f32| {
        let v: Vec<f32> = (0..rows * cols)
            .map(|i| scale * (i as f32 * 1.3 + phase).sin())
            .collect();
        Array::from_slice(&gpu, &v, &[rows, cols]).unwrap()
    };
    let mut w1 = init(1, hidden, 1.0, 0.0);
    let mut b1 = Array::zeros(&gpu, &[1, hidden]).unwrap();
    let mut w2 = init(hidden, 1, 0.8, 0.7);
    let mut b2 = Array::zeros(&gpu, &[1, 1]).unwrap();

    let lr = 0.2f32;
    let mut final_loss = f32::NAN;
    println!("epoch    loss");
    for epoch in 0..300 {
        // Each step builds a fresh tape from the current parameters.
        let tape = Tape::<f32>::new();
        let xv = tape.var(x.shallow_clone());
        let yv = tape.var(y.shallow_clone());
        let w1v = tape.var(w1.shallow_clone());
        let b1v = tape.var(b1.shallow_clone());
        let w2v = tape.var(w2.shallow_clone());
        let b2v = tape.var(b2.shallow_clone());

        // Forward: h = tanh(x·W1 + b1); pred = h·W2 + b2.
        let h = xv.matmul(&w1v).unwrap().add(&b1v).unwrap().tanh().unwrap();
        let pred = h.matmul(&w2v).unwrap().add(&b2v).unwrap();
        // loss = mean((pred − y)²)
        let diff = pred.sub(&yv).unwrap();
        let loss = diff
            .mul(&diff)
            .unwrap()
            .mean_axis(0)
            .unwrap()
            .sum()
            .unwrap();

        final_loss = loss.value().to_vec().unwrap()[0];
        if !final_loss.is_finite() {
            eprintln!("diverged: loss = {final_loss} at epoch {epoch}");
            std::process::exit(1);
        }
        if epoch % 30 == 0 || epoch == 299 {
            println!("{epoch:5}  {final_loss:.6}");
        }

        // Backward + SGD update for every parameter.
        let vars: [&Var<f32>; 4] = [&w1v, &b1v, &w2v, &b2v];
        let grads: Vec<Array<f32>> = vars.iter().map(|v| loss.grad(v).unwrap()).collect();
        w1 = sgd(&w1, &grads[0], lr);
        b1 = sgd(&b1, &grads[1], lr);
        w2 = sgd(&w2, &grads[2], lr);
        b2 = sgd(&b2, &grads[3], lr);
    }

    // Show the learned fit at a few points.
    println!("\n   x      y=x²    pred");
    let probe = [-1.0f32, -0.5, 0.0, 0.5, 1.0];
    let px = Array::from_slice(&gpu, &probe, &[probe.len(), 1]).unwrap();
    let h = px
        .matmul(&w1)
        .unwrap()
        .add(&b1.broadcast_to(&[probe.len(), hidden]).unwrap())
        .unwrap();
    // tanh on the host-visible array path (forward only, no tape needed).
    let ex = h.exp().unwrap();
    let enx = h.neg().unwrap().exp().unwrap();
    let ht = ex.sub(&enx).unwrap().div(&ex.add(&enx).unwrap()).unwrap();
    let pred = ht
        .matmul(&w2)
        .unwrap()
        .add(&b2.broadcast_to(&[probe.len(), 1]).unwrap())
        .unwrap()
        .to_vec()
        .unwrap();
    for (i, &xi) in probe.iter().enumerate() {
        println!("{xi:5.1}   {:6.3}   {:6.3}", xi * xi, pred[i]);
    }

    println!("\nfinal loss: {final_loss:.6}");
    // A converged fit lands well under this; a loop bug that stalls training
    // would leave the loss high, so fail CI on it.
    if final_loss > 0.05 {
        eprintln!("did not converge: final loss {final_loss:.6} > 0.05");
        std::process::exit(1);
    }
}
