# Training with autodiff

`quanta::autograd` adds reverse-mode gradients to the GPU array ops — record a
forward computation on a `Tape`, then ask for the gradient of a scalar w.r.t.
any input. Every gradient rule is the proven analytic derivative (see the
[verification page](../../verification/index.md)). This is a task-by-task recipe;
the runnable end-to-end example is
[`examples/mlp_training.rs`](https://github.com/zelez-lab/quanta/blob/main/crates/ml/quanta-autograd/examples/mlp_training.rs).
(The example lives inside the crate, so it imports `quanta_autograd::` directly; in your own app use `quanta::autograd`.)

```toml
[dependencies]
quanta = { version = "0.1", features = ["sci", "autograd", "metal"] } # vulkan / software
```

## A first gradient

```rust,ignore
use quanta::autograd::Tape;
use quanta::sci::Array;

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
`log`, `sqrt`, the activations `relu` / `sigmoid` / `tanh`, `matmul`, `conv2d`,
`avgpool2d` / `maxpool2d`, and the reductions `sum` / `sum_axis` / `mean_axis`.

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

## A convolution layer

`conv2d` is a 2-D NCHW convolution — the CNN workhorse. Input is
`[N, Cin, H, W]`, the weight is `[Cout, Cin, kh, kw]`, and the output is
`[N, Cout, OH, OW]` with `OH = (H + 2·pad − kh)/stride + 1` (likewise `OW`):

```rust,ignore
// x: [N, Cin, H, W]   w: [Cout, Cin, kh, kw]
let x = tape.var(Array::from_slice(&gpu, &input, &[n, cin, h, w])?);
let w = tape.var(Array::from_slice(&gpu, &kernel, &[cout, cin, kh, kw])?);

let y = x.conv2d(&w, /* stride */ 1, /* pad */ 1)?;   // [N, Cout, OH, OW]
let loss = y.sum()?;
let gx = loss.grad(&x)?;   // [N, Cin, H, W]
let gw = loss.grad(&w)?;   // [Cout, Cin, kh, kw]
```

Under the hood `conv2d` is `im2col → matmul → reshape`: the input is unfolded
into a patch matrix, multiplied by the flattened weight, and reshaped back. So
its backward is just `matmul`'s VJP plus `col2im` (the unfold's adjoint) for the
input gradient — nothing new to trust. The one extra fact, *col2im is the
transpose of im2col*, is [proven in Lean](../../verification/index.md); the kernels
are gradient-checked against a host convolution on both the software lane and a
real GPU.

A bias adds per output channel — broadcast a `[1, Cout, 1, 1]` term:

```rust,ignore
let b = tape.var(Array::from_slice(&gpu, &bias, &[1, cout, 1, 1])?);
let y = x.conv2d(&w, 1, 1)?.add(&b)?;     // bias broadcasts over N, OH, OW
let act = y.relu()?;                       // conv → bias → activation
```

`stride` and `pad` are symmetric (same in H and W); zero-padding reads 0 outside
the input. Both `x` and `w` must be 4-D — anything else is an error, not a silent
reshape.

## Pooling (downsampling)

After a conv → activation, pooling halves the spatial size. Both poolers take an
NCHW input `[N, C, H, W]` and a `kh×kw` window with `stride` / `pad`, producing
`[N, C, OH, OW]` (channels unchanged — pooling is per-channel):

```rust,ignore
let h = x.conv2d(&w, 1, 1)?.relu()?;
let pooled = h.maxpool2d(2, 2, /* stride */ 2, /* pad */ 0)?;  // 2×2 max, halves H,W
// average pooling instead:
let avg = h.avgpool2d(2, 2, 2, 0)?;
```

The gradients differ by kind. `avgpool2d` is linear: each input pixel's gradient
is the sum of `g/(kh·kw)` over the windows containing it. `maxpool2d` is the
nonlinear one — the forward records which input pixel won each window (its
argmax), and the backward routes that window's gradient to exactly that pixel
and nowhere else (the subgradient). Both backwards are atomic-free gather
kernels, so they're deterministic.

`avgpool2d` counts padding toward the `kh·kw` divisor (`count_include_pad`), and
`maxpool2d` assumes each window covers at least one real pixel (the usual
`pad ≤ k/2`). As with `conv2d`, both require a 4-D input.

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

[`examples/mlp_training.rs`](https://github.com/zelez-lab/quanta/blob/main/crates/ml/quanta-autograd/examples/mlp_training.rs)
puts it together: a 2-layer MLP `h = tanh(x·W1 + b1); ŷ = h·W2 + b2` learning
`y = x²`. Running it on the GPU (`cargo run --example mlp_training -p
quanta-autograd --release --features metal` — `vulkan` off Apple, or drop the
flag for the CPU lane) shows the loss falling and the fit forming:

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
