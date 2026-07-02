# Autodiff basics

> **You'll learn:** how to compute gradients automatically with a tape — the
> foundation of training. Builds on everything in the array track so far.

To train a model you need gradients: how the loss changes as you nudge each
parameter. `quanta-autograd` computes them automatically. You run a forward
computation on a *tape*, ask for the gradient of a scalar, and it walks the
computation backward for you. Every gradient rule is **proven equal to the
analytic derivative** in Lean.

```toml
quanta-autograd = { version = "0.1", features = ["metal"] } # or vulkan / software
quanta-array    = { version = "0.1", features = ["metal"] }
```

## A first gradient

Wrap the values you want gradients for as `Var`s on a `Tape`, compute a scalar,
then call `grad`:

```rust,ignore
use quanta_autograd::Tape;
use quanta_array::Array;

let gpu = quanta::init_cpu();
let tape = Tape::<f32>::new();

// A leaf we want the gradient for (PyTorch: requires_grad=True).
let x = tape.var(Array::from_slice(&gpu, &[1.0, 2.0, 3.0], &[3])?);

// loss = sum(x · x)   ⇒   d loss / d x = 2x
let loss = x.mul(&x)?.sum()?;
let gx = loss.grad(&x)?;
assert_eq!(gx.to_vec()?, vec![2.0, 4.0, 6.0]);
```

`grad` seeds the output gradient, walks the tape in reverse applying each op's
rule, and returns the gradient shaped like `x`. Ask again with a different `wrt`
to get another input's gradient.

## The chain rule is automatic

Chain any recorded ops and differentiation follows the chain rule for you:

```rust,ignore
// loss = sum(exp(x · x))   ⇒   d/dx = 2x · exp(x²)
let loss = x.mul(&x)?.exp()?.sum()?;
let gx = loss.grad(&x)?;
```

The differentiable ops are the same array operations you already know —
`add` / `sub` / `mul` / `div` (broadcasting), `exp` / `log` / `sqrt`, the
activations `relu` / `sigmoid` / `tanh`, `matmul`, `conv2d`, `avgpool2d` /
`maxpool2d`, `reshape` / `flatten`, and the reductions `sum` / `sum_axis` /
`mean_axis` — now on `Var`s instead of `Array`s.

`Var::narrow(start, len)` is the differentiable minibatch selector: it takes a
zero-copy row window of a `[N, …]` batch and, in the backward pass, routes the
gradient back only to those rows (the sliced-out rows get zero). That's what
lets a training loop step over `x.narrow(b * batch, batch)` one minibatch at a
time — see the [MNIST walkthrough](project-mnist.md).

## Gradients of a broadcast

Broadcasting differentiates correctly: when a small operand is stretched over a
larger one, its gradient is summed back over the broadcast axes so it matches the
operand's own shape:

```rust,ignore
let w = tape.var(/* [m, n] */);
let b = tape.var(/* [1, n] bias row */);
let out = w.add(&b)?;        // b broadcasts over the m rows
let loss = out.sum()?;
let gb = loss.grad(&b)?;     // shape [1, n] — summed over the m rows
```

## The mental model

A `Tape` records operations as they run (define-by-run, like PyTorch). Each `Var`
is a handle into that recording. `grad` replays it backward. There's no separate
"compile the model" step — you write ordinary forward code, and the gradient
falls out. That's what makes the next two lessons — training an [MLP](training-an-mlp.md)
and a [CNN](training-a-cnn.md) — short.

## Next

- **[Training an MLP](training-an-mlp.md)** — assemble gradients into a learning loop.
- How-to: **[Training with autodiff](../how-to/autograd-training.md)**.
