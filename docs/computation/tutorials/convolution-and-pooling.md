# Convolution and pooling

> **You'll learn:** the two operations that turn a network into a *convolutional*
> one — `conv2d` and pooling. Builds on [Training an MLP](training-an-mlp.md).

A dense (matmul) layer treats every input independently of position. Images have
*structure* — a feature looks the same wherever it appears. Convolution captures
that: a small filter slides over the image, detecting the same pattern
everywhere. Pooling then shrinks the spatial size, keeping the strongest
responses. Both are differentiable `Var` operations, so they drop straight into
the training loop from the last lesson.

## Convolution

`conv2d` convolves an NCHW input `[N, Cin, H, W]` with a weight
`[Cout, Cin, kh, kw]`, producing `[N, Cout, OH, OW]`:

```rust,ignore
use quanta_autograd::Tape;
use quanta_array::Array;

let tape = Tape::<f32>::new();
// x: one batch of N single-channel 4×4 images
let x = tape.var(Array::from_slice(&gpu, &input, &[n, 1, 4, 4])?);
// w: `cout` filters, each 1 input channel × 3×3
let w = tape.var(Array::from_slice(&gpu, &kernel, &[cout, 1, 3, 3])?);

let y = x.conv2d(&w, /* stride */ 1, /* pad */ 1)?; // [n, cout, 4, 4]
```

Under the hood `conv2d` is an `im2col` unfold followed by a `matmul` — so its
backward reuses the proven matmul gradient plus `col2im` (the unfold's adjoint,
also proven). You get a real convolution with gradients you can trust, and
nothing new to learn: it's a `Var` op like `matmul`.

A per-channel bias broadcasts naturally:

```rust,ignore
let b = tape.var(Array::from_slice(&gpu, &bias, &[1, cout, 1, 1])?);
let y = x.conv2d(&w, 1, 1)?.add(&b)?.relu()?;  // conv → bias → activation
```

## Pooling

Pooling downsamples each channel with a sliding window. **Max** pooling keeps the
strongest response in each window; **average** pooling takes the mean:

```rust,ignore
let pooled = y.maxpool2d(2, 2, /* stride */ 2, /* pad */ 0)?; // halves H and W
let avg    = y.avgpool2d(2, 2, 2, 0)?;
```

Their gradients differ by kind, and both are proven. Average pooling is linear —
each input pixel's gradient is the shared window average. Max pooling is
nonlinear — the forward remembers which pixel won each window (its argmax), and
the backward routes the gradient to exactly that pixel and nowhere else.

## Flattening to a classifier head

After the conv/pool stack you flatten the spatial dimensions to feed a dense
layer — `[N, C, H, W] → [N, C·H·W]`:

```rust,ignore
let feat = pooled.flatten()?;          // [N, C·H·W]
let logits = feat.matmul(&w_head)?.add(&b_head)?;
```

`flatten` is a shape-only view (its gradient just reshapes back), so it costs
nothing but connects the convolutional front end to a linear head.

## Next

- **[Training a CNN](training-a-cnn.md)** — assemble conv, pool, and flatten into a network that learns.
