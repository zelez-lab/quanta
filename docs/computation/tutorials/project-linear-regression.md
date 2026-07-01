# Project: linear regression from scratch

> **You'll build:** a complete, standalone Cargo project that fits a linear model
> `y = X·w + b` by gradient descent on the GPU — the "hello world" of machine
> learning, and your first end-to-end Quanta program outside this repo.
>
> **You'll need:** the earlier lessons on [arrays](arrays-and-broadcasting.md),
> [linear algebra](linear-algebra.md), and [autodiff](autodiff-basics.md). No new
> Quanta features — everything here already exists.

This is the Rust/Quanta equivalent of scikit-learn's `SGDRegressor` or a one-layer
PyTorch net: recover the weights of a linear relationship from noisy data.

## 1. Create the project

```sh
cargo new linear-regression
cd linear-regression
```

## 2. Add the Quanta dependencies

Quanta isn't on crates.io yet, so depend on it directly from the git repository.
Edit `Cargo.toml`:

```toml
[package]
name = "linear-regression"
version = "0.1.0"
edition = "2024"

[dependencies]
quanta          = { git = "https://github.com/zelez-lab/quanta" }
quanta-array    = { git = "https://github.com/zelez-lab/quanta" }
quanta-autograd = { git = "https://github.com/zelez-lab/quanta" }
```

By default these build the **software (CPU) backend** — no GPU required, and the
behaviour is identical to hardware. To run on a real GPU instead, add the backend
feature to each crate:

```toml
quanta          = { git = "https://github.com/zelez-lab/quanta", features = ["metal"] } # or vulkan
quanta-array    = { git = "https://github.com/zelez-lab/quanta", features = ["metal"] }
quanta-autograd = { git = "https://github.com/zelez-lab/quanta", features = ["metal"] }
```

## 3. The plan

We'll generate data from a known line, then let gradient descent recover it:

- **Data.** Pick a true weight vector `w*` and bias `b*`, generate `X`, compute
  `y = X·w* + b*` (plus a little noise if you like). The model should rediscover
  `w*` and `b*`.
- **Model.** `pred = X·w + b` — one matmul and a broadcast bias.
- **Loss.** Mean squared error, `mean((pred − y)²)`.
- **Train.** Each step: forward on a fresh tape, get `∂loss/∂w` and `∂loss/∂b`,
  nudge the parameters against the gradients.

## 4. File layout

A single-file program is fine for this size. Everything goes in `src/main.rs`.
For a larger project you'd split the model and the training loop into modules; we
keep it flat so the whole flow reads top to bottom.

```
linear-regression/
  Cargo.toml
  src/
    main.rs        ← everything below
```

## 5. The code

### The optimizer step

The parameter update is plain array math, done *outside* the tape — the tape is
only for the forward pass you differentiate. Put this above `main`:

```rust,ignore
use quanta_array::Array;
use quanta_autograd::Tape;

/// One SGD step: p ← p − lr·g.
fn sgd(p: &Array<f32>, g: &Array<f32>, lr: f32) -> Array<f32> {
    let lr_a = Array::full(p.gpu(), lr, &[1]).unwrap()
        .broadcast_to(g.shape()).unwrap();
    p.sub(&g.mul(&lr_a).unwrap()).unwrap()
}
```

### Generate the data

Inside `main`, open a device and build a dataset from a known line. We use a
2-feature problem so `w` is a `[2, 1]` matrix:

```rust,ignore
fn main() {
    let gpu = quanta::init_cpu(); // or quanta::init() for a real GPU

    // Ground truth: y = 2·x0 − 3·x1 + 1
    let n = 64usize; // samples
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..n {
        let x0 = (i as f32) / n as f32;          // 0 .. 1
        let x1 = ((i * 7 % n) as f32) / n as f32; // a second, decorrelated feature
        xs.push(x0);
        xs.push(x1);
        ys.push(2.0 * x0 - 3.0 * x1 + 1.0);
    }
    let x = Array::from_slice(&gpu, &xs, &[n, 2]).unwrap();  // [N, 2]
    let y = Array::from_slice(&gpu, &ys, &[n, 1]).unwrap();  // [N, 1]
```

### Initialize the parameters

Start the weights and bias at zero — gradient descent will move them:

```rust,ignore
    let mut w = Array::<f32>::zeros(&gpu, &[2, 1]).unwrap(); // [in, out]
    let mut b = Array::<f32>::zeros(&gpu, &[1, 1]).unwrap(); // broadcasts over rows
    let lr = 0.5f32;
```

### The training loop

Each iteration builds a fresh tape, runs the forward pass, reads one gradient per
parameter, and updates:

```rust,ignore
    for epoch in 0..200 {
        let tape = Tape::<f32>::new();
        let xv = tape.var(x.shallow_clone());
        let yv = tape.var(y.shallow_clone());
        let wv = tape.var(w.shallow_clone());
        let bv = tape.var(b.shallow_clone());

        // Forward: pred = X·w + b
        let pred = xv.matmul(&wv).unwrap().add(&bv).unwrap();

        // Loss: mean((pred − y)²)
        let diff = pred.sub(&yv).unwrap();
        let loss = diff.mul(&diff).unwrap().mean_axis(0).unwrap().sum().unwrap();

        // Backward: one gradient per parameter.
        let gw = loss.grad(&wv).unwrap();
        let gb = loss.grad(&bv).unwrap();

        // Update.
        w = sgd(&w, &gw, lr);
        b = sgd(&b, &gb, lr);

        if epoch % 40 == 0 {
            println!("epoch {epoch:4}  loss {:.6}", loss.value().to_vec().unwrap()[0]);
        }
    }
```

### Report the learned parameters

```rust,ignore
    let wv = w.to_vec().unwrap();
    let bv = b.to_vec().unwrap();
    println!("\nlearned  w = [{:.3}, {:.3}]  b = {:.3}", wv[0], wv[1], bv[0]);
    println!("true     w = [2.000, -3.000]  b = 1.000");
}
```

## 6. Run it

```sh
cargo run --release
```

`--release` matters — the software backend JIT-compiles a kernel per operation, and
release makes that fast. You should see the loss fall and the parameters converge
on the truth:

```text
epoch    0  loss 1.264038
epoch   40  loss 0.002201
epoch   80  loss 0.000007
epoch  120  loss 0.000000
epoch  160  loss 0.000000

learned  w = [2.000, -3.000]  b = 1.000
true     w = [2.000, -3.000]  b = 1.000
```

## 7. What you built, and what's next

You wrote a complete ML program — data, model, loss, gradients, optimizer — in one
file, against a GPU library, with every gradient mechanically proven correct. The
loop is the same three moves as every model: **forward** on a tape, **backward**
with `grad`, **update** outside it.

- Make it nonlinear: add a hidden layer with a `tanh`, and you have the
  [MLP](training-an-mlp.md).
- Make it a classifier: swap MSE for a classification head — that's the road to
  [training a CNN](training-a-cnn.md) on images.
- Coming from Python? The [From NumPy](../../migration/from-numpy.md) guide maps
  the APIs.
