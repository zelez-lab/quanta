# Self-attention

> **You'll learn:** how to build a self-attention block — the core computation of
> a transformer — from the array ops you already have, and train it end to end.
>
> **You'll need:** [autodiff basics](autodiff-basics.md) and
> [linear algebra](linear-algebra.md).

Attention lets each position in a sequence look at every other position and pull
in what's relevant. For a sequence `X` of `S` positions, self-attention is:

```text
Q = X·Wq    K = X·Wk    V = X·Wv          (project to queries, keys, values)
scores = softmax( Q·Kᵀ / √d )              (how much each position attends to each)
out    = scores · V                        (a weighted mix of the values)
```

Every step is an op Quanta already has — three matmuls, a transpose, a scale,
and a row-wise softmax.

## The block

```rust,ignore
use quanta_array::Array;
use quanta_autograd::{Tape, Var};

/// Single-head self-attention: X [S, D] → [S, d].
fn attention(
    tape: &Tape<f32>,
    x: &Var<f32>,
    wq: &Var<f32>, wk: &Var<f32>, wv: &Var<f32>,
    s: usize, d_head: usize,
) -> Var<f32> {
    let q = x.matmul(wq).unwrap();   // [S, d]
    let k = x.matmul(wk).unwrap();
    let v = x.matmul(wv).unwrap();

    // scores[i, j] = how much position i attends to position j
    let scale = tape.var(
        Array::full(x.value().gpu(), 1.0 / (d_head as f32).sqrt(), &[1]).unwrap()
            .broadcast_to(&[s, s]).unwrap().contiguous().unwrap(),
    );
    let scores = q
        .matmul(&k.transpose(0, 1).unwrap()).unwrap()   // Q·Kᵀ  → [S, S]
        .mul(&scale).unwrap();                          // / √d

    // row-softmax turns each row into attention weights that sum to 1
    scores.softmax().unwrap().matmul(v).unwrap()        // weighted mix → [S, d]
}
```

The one piece that isn't obvious is `k.transpose(0, 1)` — a **differentiable**
transpose. It's a zero-copy view in the forward pass, and because transposing
twice is the identity, its backward is just transposing the gradient back. That
one op is what lets `Q·Kᵀ` sit inside a trained graph.

## Why the softmax axis matters

`scores` is `[S, S]` — row `i` holds position `i`'s affinity to every position.
`Var::softmax` normalizes over the **last axis**, so each row becomes a
probability distribution over keys: position `i`'s attention weights. That's
exactly what you want — after the softmax, `scores · V` is a convex mixture of
the value vectors, weighted by attention.

## Training it

Attention is differentiable throughout, so you train the `Wq`/`Wk`/`Wv`
projections like any other parameters. A task that *requires* attention — one a
position-wise linear map can't solve — is the honest test. For example, an
**induction shift**: each position must output the payload of the *next*
position, which forces the model to learn to attend one step over.

```rust,ignore
for _ in 0..epochs {
    opt.advance();
    let tape = Tape::<f32>::new();
    let xv = tape.var(x.shallow_clone());
    let (wqv, wkv, wvv) = (tape.var(wq.clone()), tape.var(wk.clone()), tape.var(wv.clone()));

    let out = attention(&tape, &xv, &wqv, &wkv, &wvv, s, d_head);
    let loss = out.mse_loss(&target).unwrap();

    wq = opt.step(0, &wq, &loss.grad(&wqv).unwrap()).unwrap();
    wk = opt.step(1, &wk, &loss.grad(&wkv).unwrap()).unwrap();
    wv = opt.step(2, &wv, &loss.grad(&wvv).unwrap()).unwrap();
}
```

The loss drops to near zero: the projections learn a query/key geometry where
each position's query matches the next position's key, so the softmax routes the
right value through. That's attention learning *where to look*.

## Where this goes

The single head above is the core; a full transformer block adds a few pieces,
all of which the stack has:

- **Multi-head** attention runs several heads in parallel on slices of the
  feature dim — `Var::multi_head_attention(wq, wk, wv, wo, heads, mask)` does the
  split → per-head attention (batched matmul over the heads) → merge for you.
- A **transformer block** wraps attention with a residual add, a GELU
  feed-forward `matmul → gelu → matmul`, and normalization
  (`rms_norm` / [layer_norm](layer-norm.md)).
- **Causal** attention masks out future positions before the softmax — an
  additive lower-triangular mask (`0` allowed, a large negative blocked) added to
  `scores`.

The **[Project: train a GPT decoder block](project-transformer.md)** assembles
all of these into a real decoder block and trains it end to end — the natural
next step from here.

You now have the piece every transformer is built from.
