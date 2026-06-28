# Training with autodiff

`quanta-autograd` adds reverse-mode gradients to the GPU array ops — record a
forward computation on a `Tape`, then ask for the gradient of a scalar w.r.t.
any input. Every gradient rule is the proven analytic derivative (see the
[verification page](../verification/index.md)). This is a task-by-task recipe;
the runnable end-to-end example is
[`examples/mlp_training.rs`](https://github.com/zelez-lab/quanta/blob/main/crates/quanta-autograd/examples/mlp_training.rs).

```toml
[dependencies]
quanta-autograd = { version = "0.1", features = ["metal"] } # vulkan / software
quanta-array    = { version = "0.1", features = ["metal"] }
```

## A first gradient

```rust,ignore
use quanta_autograd::Tape;
use quanta_array::Array;

let gpu = quanta::init_cpu();          // or quanta::init() for a real GPU
let tape = Tape::<f32>::new();

// A leaf we want the gradient for (PyTorch: requires_grad=True).
let x = tape.var(Array::from_slice(&gpu, &[1.0, 2.0, 3.0], &[3])?);

// loss = sum(x * x)   ⇒   d loss / d x = 2x
let loss = x.mul(&x)?.sum()?;
let gx = loss.grad(&x)?;
assert_eq!(gx.to_vec()?, vec![2.0, 4.0, 6.0]);
```

`grad` walks the tape backward from the scalar `loss` and returns the gradient
shaped like `x`. Run it again with a different `wrt` to get another input's
gradient — the backward pass is recomputed per call.

## Chaining ops

Any chain of recorded ops differentiates by the chain rule automatically:

```rust,ignore
// loss = sum(exp(x * x))   ⇒   d/dx = 2x · exp(x²)
let loss = x.mul(&x)?.exp()?.sum()?;
let gx = loss.grad(&x)?;
```

Available ops: `neg`, `add`, `sub`, `mul`, `div` (all broadcasting), `exp`,
`log`, `sqrt`, the activations `relu` / `sigmoid` / `tanh`, `matmul`, and the
reductions `sum` / `sum_axis` / `mean_axis`.

## Broadcasting (bias-style)

Add a row vector to a matrix — the gradient is summed back over the broadcast
axis, so each operand's gradient matches its own shape:

```rust,ignore
let w = tape.var(Array::from_slice(&gpu, &[/* m×n */], &[m, n])?);
let b = tape.var(Array::from_slice(&gpu, &[/* 1×n */], &[1, n])?); // bias row
let out = w.add(&b)?;          // [m,n]; b broadcasts over the m rows
let loss = out.sum()?;
let gb = loss.grad(&b)?;       // shape [1,n] — summed over the m broadcast rows
```

## A linear layer

`matmul` is the workhorse — its VJP is `∂A = G·Bᵀ`, `∂B = Aᵀ·G`, both proven:

```rust,ignore
// pred = x · W + b   (x: [N,in], W: [in,out], b: [1,out])
let pred = x.matmul(&w)?.add(&b)?;
// MSE loss against targets y
let diff = pred.sub(&y)?;
let loss = diff.mul(&diff)?.mean_axis(0)?.sum()?;
```

## The training loop

The optimizer step is plain array ops *outside* the tape; build a fresh tape
each step from the current parameters:

```rust,ignore
fn sgd(p: &Array<f32>, g: &Array<f32>, lr: f32) -> Array<f32> {
    let lr_a = Array::full(p.gpu(), lr, &[1]).unwrap()
        .broadcast_to(g.shape()).unwrap();
    p.sub(&g.mul(&lr_a).unwrap()).unwrap()       // p ← p − lr·g
}

let mut w = /* initial weights */;
for _ in 0..epochs {
    let tape = Tape::<f32>::new();
    let wv = tape.var(w.shallow_clone());
    // … forward using wv, compute scalar `loss` …
    let gw = loss.grad(&wv)?;
    w = sgd(&w, &gw, lr);                         // update for next step
}
```

## A whole network

[`examples/mlp_training.rs`](https://github.com/zelez-lab/quanta/blob/main/crates/quanta-autograd/examples/mlp_training.rs)
puts it together: a 2-layer MLP `h = tanh(x·W1 + b1); ŷ = h·W2 + b2` learning
`y = x²`. Running it (`cargo run --example mlp_training -p quanta-autograd
--release`) shows the loss falling and the fit forming:

```text
epoch    loss
    0  0.981187
  150  0.012775
  299  0.002606

   x      y=x²    pred
 -1.0    1.000    0.891
  0.0    0.000   -0.046
  1.0    1.000    0.913
```

## Notes

- **f32 only** today (matmul reuses the f32 GEMM); the tape is generic over a
  `DiffScalar` bound that `f32` satisfies.
- **`grad` recomputes** the backward pass each call — cache the result if you
  need several inputs' gradients from one loss.
- All backends are equivalent — `init_cpu()` runs the software lane (used by the
  tests); `init()` picks a real GPU.
