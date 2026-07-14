# Layer normalization

> **You'll learn:** what layer normalization does, why deep nets and
> transformers need it, and how it's just a reduction plus a few elementwise ops.
>
> **You'll need:** [autodiff basics](autodiff-basics.md).

As a network gets deeper, the scale of its activations drifts layer to layer —
some blow up, some vanish — and training stalls. **Layer normalization** fixes
each layer's activations to a stable scale by standardizing every row to zero
mean and unit variance, then letting the network re-scale with two learnable
parameters. It's the normalization transformers are built on (and, unlike batch
norm, it doesn't depend on the batch, so it works the same at train and
inference time).

## The formula

For a `[N, C]` matrix — `N` examples, `C` features — layer norm operates
**per row**, over the feature axis:

```text
μ  = mean(x)          over C          (per-row mean,      [N, 1])
σ² = mean((x − μ)²)   over C          (per-row variance,  [N, 1])
out = (x − μ) / √(σ² + eps) · γ + β
```

`γ` and `β` are learnable `[C]` vectors: after standardizing, `γ` rescales each
feature and `β` re-shifts it, so the network can undo the normalization where it
helps. `eps` keeps the divide stable when a row is nearly constant.

## Using it

`Var::layer_norm` is one call — you supply the learnable `γ` and `β`:

```rust,ignore
use quanta::sci::Array;
use quanta::autograd::Tape;

let tape = Tape::<f32>::new();
let xv = tape.var(activations);                       // [N, C]
let gamma = tape.var(Array::ones(&gpu, &[c]).unwrap());   // learnable scale
let beta  = tape.var(Array::zeros(&gpu, &[c]).unwrap());  // learnable shift

let normed = xv.layer_norm(&gamma, &beta, 1e-5).unwrap();
```

With `γ = 1`, `β = 0`, every row of `normed` comes out zero-mean, unit-variance
— check it yourself by reducing the output. `γ` and `β` are parameters like any
other: register them with your optimizer and they train alongside the weights.

## It's a composition

Layer norm needs no special kernel — it's the reduction and elementwise ops you
already have, wired together:

```rust,ignore
let mean = xv.mean_axis(1)?;              // μ, keepdims [N, 1]
let centered = xv.sub(&mean)?;            // x − μ  (μ broadcasts over C)
let var = centered.mul(&centered)?.mean_axis(1)?;   // σ²  [N, 1]
let std = var.add(&eps)?.sqrt()?;         // √(σ² + eps)
let normed = centered.div(&std)?.mul(&gamma)?.add(&beta)?;
```

Because `mean_axis` keeps the reduced dimension as size 1, `μ` and `σ²`
broadcast straight back over the features. And because every step is a
differentiable `Var` op, the tape produces the (notoriously fiddly) layer-norm
gradient automatically — you never hand-derive it.

## Where it goes

In a transformer block, layer norm wraps each sub-layer:

```text
x = x + attention(layer_norm(x))          (pre-norm residual)
x = x + feedforward(layer_norm(x))
```

Combined with [self-attention](attention.md) and a feed-forward `matmul → relu →
matmul`, that block — normalized, residual, attended — is the repeating unit of
every transformer. You now have all three pieces.
