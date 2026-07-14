# Training an MLP

> **You'll learn:** how to turn gradients into a training loop that fits a small
> neural network. Builds on [Autodiff basics](autodiff-basics.md).

A training loop is just: run the forward pass, compute a scalar loss, get the
gradients, nudge the parameters against them, repeat. With autodiff in hand,
that's a dozen lines. We'll fit a 2-layer MLP to the curve `y = x²`.

The complete, runnable version is
[`examples/mlp_training.rs`](https://github.com/zelez-lab/quanta/blob/main/crates/ml/quanta-autograd/examples/mlp_training.rs)
— run it on the GPU with `cargo run --example mlp_training -p quanta-autograd
--release --features metal` (use `--features vulkan` off Apple, or drop the flag
for the portable CPU lane). (The example lives inside the crate, so it imports
`quanta_autograd::` directly; in your own app use `quanta::autograd`.)

## The optimizer step

The parameter update is plain array math, *outside* the tape — the tape is only
for the forward pass you differentiate:

```rust,ignore
use quanta::sci::Array;

// p ← p − lr·g
fn sgd(p: &Array<f32>, g: &Array<f32>, lr: f32) -> Array<f32> {
    let lr_a = Array::full(p.gpu(), lr, &[1]).unwrap()
        .broadcast_to(g.shape()).unwrap();
    p.sub(&g.mul(&lr_a).unwrap()).unwrap()
}
```

## The loop

Each step builds a **fresh tape** from the current parameters, runs the forward
pass, and reads out one gradient per parameter:

```rust,ignore
use quanta::autograd::{Tape, Var};

let mut w1 = /* [1, hidden] */;  let mut b1 = /* [1, hidden] */;
let mut w2 = /* [hidden, 1] */;  let mut b2 = /* [1, 1] */;
let lr = 0.2;

for epoch in 0..300 {
    let tape = Tape::<f32>::new();
    let xv  = tape.var(x.shallow_clone());
    let yv  = tape.var(y.shallow_clone());
    let w1v = tape.var(w1.shallow_clone());
    let b1v = tape.var(b1.shallow_clone());
    let w2v = tape.var(w2.shallow_clone());
    let b2v = tape.var(b2.shallow_clone());

    // Forward: h = tanh(x·W1 + b1);  pred = h·W2 + b2
    let h = xv.matmul(&w1v)?.add(&b1v)?.tanh()?;
    let pred = h.matmul(&w2v)?.add(&b2v)?;

    // loss = mean((pred − y)²)
    let diff = pred.sub(&yv)?;
    let loss = diff.mul(&diff)?.mean_axis(0)?.sum()?;

    // Backward + SGD update for every parameter.
    let grads: Vec<Array<f32>> = [&w1v, &b1v, &w2v, &b2v]
        .iter().map(|v| loss.grad(v).unwrap()).collect();
    w1 = sgd(&w1, &grads[0], lr);
    b1 = sgd(&b1, &grads[1], lr);
    w2 = sgd(&w2, &grads[2], lr);
    b2 = sgd(&b2, &grads[3], lr);
}
```

Running it, the loss falls and the fit forms:

```text
epoch    loss
    0  0.981187
  150  0.012775
  299  0.002606
```

## The three pieces

Every training loop, no matter how large the model, is these three moves:

1. **Forward** — build `Var`s from the parameters, compute a scalar loss.
2. **Backward** — `loss.grad(&param)` for each parameter.
3. **Update** — `sgd` (or any rule) on the raw arrays, outside the tape.

The hidden `tanh` layer is what lets a linear output bend into a curve — without a
nonlinearity, two matmuls collapse into one. That's the whole idea of a network:
stack linear maps with nonlinearities between them.

## Next

- **[Convolution and pooling](convolution-and-pooling.md)** — the operations that make a network *convolutional*.
