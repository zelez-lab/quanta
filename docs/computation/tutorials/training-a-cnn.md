# Training a CNN

> **You'll learn:** how to assemble convolution, pooling, and a dense head into a
> convolutional network and train it end to end. The capstone of the array track.

You now have every piece: [arrays](arrays-and-broadcasting.md),
[linear algebra](linear-algebra.md), [autodiff](autodiff-basics.md), a
[training loop](training-an-mlp.md), and [conv/pool](convolution-and-pooling.md).
This lesson puts them together into a real CNN that classifies images.

The complete, runnable version is
[`examples/cnn_training.rs`](https://github.com/zelez-lab/quanta/blob/main/crates/quanta-autograd/examples/cnn_training.rs)
— run it on the GPU with `cargo run --example cnn_training -p quanta-autograd
--release --features metal` (use `--features vulkan` off Apple; the CPU lane
works too but is far slower on this conv-heavy model).

## The task

Classify 4×4 single-channel images as a **horizontal** stripe (one bright row) or
a **vertical** stripe (one bright column). It's the smallest task where
convolution earns its keep — a 3×3 filter learns an oriented-edge detector, which
a plain dense model on raw pixels can't express position-invariantly.

## The architecture

```text
x[N,1,4,4] → conv2d(1→C, 3×3, pad 1) → relu → maxpool2d(2×2, stride 2)
          → flatten[N, C·2·2] → linear → sigmoid → MSE vs {0,1}
```

Every arrow is a `Var` op you met in the previous lessons.

## The forward pass

```rust,ignore
let feat = xv
    .conv2d(&wcv, 1, 1)?    // convolve
    .relu()?               // nonlinearity
    .maxpool2d(2, 2, 2, 0)? // downsample
    .flatten()?;           // [N, C·2·2] for the dense head
let pred = feat.matmul(&wlv)?.add(&blv)?.sigmoid()?; // logits → probability
let diff = pred.sub(&yv)?;
let loss = diff.mul(&diff)?.mean_axis(0)?.sum()?;    // MSE
```

## The loop is the same shape

The training loop is *identical in structure* to the [MLP](training-an-mlp.md) —
fresh tape, forward, one `grad` per parameter, SGD update. Only the forward pass
and the parameter list changed:

```rust,ignore
let grads: Vec<Array<f32>> = [&wcv, &wlv, &blv]
    .iter().map(|v| loss.grad(v).unwrap()).collect();
wc = sgd(&wc, &grads[0], lr);   // conv filters
wl = sgd(&wl, &grads[1], lr);   // linear head
bl = sgd(&bl, &grads[2], lr);   // head bias
```

That's the payoff of the whole track: once you can differentiate the ops, a
convolutional network is no harder to train than a linear one.

## It learns

Running it, the loss falls and every image is classified correctly:

```text
epoch    loss
    0  0.246425
  399  0.001858

  image          label  pred  ✓
  horizontal    0    0.064  ✓   (×4)
  vertical      1    0.967  ✓   (×4)
accuracy: 8/8
```

## Where you are now

You've gone from putting a slice on the GPU to training a convolutional network —
without writing a kernel, and with every gradient mechanically proven correct.
The same building blocks compose into whatever you need next: deeper networks,
other losses, other data.

## Beyond the array track

- Need an operation that isn't a building block yet? The **kernel track** (starting
  at [Compute Basics](compute-basics.md)) shows how to write and dispatch your own.
- Coming from Python? The **[From NumPy](../../migration/from-numpy.md)** migration
  guide maps the APIs you know.
