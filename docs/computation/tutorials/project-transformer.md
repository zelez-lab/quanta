# Project: train a GPT decoder block

> **You'll build:** a complete decoder-only transformer block — token embedding,
> multi-head causal self-attention, RMSNorm, a GELU feed-forward network, and a
> cross-entropy language-modelling loss — and train it end to end until the loss
> drops. Every piece is an op Quanta already gives you; the tape supplies the
> gradients.
>
> **You'll need:** [autodiff basics](autodiff-basics.md),
> [self-attention](attention.md), and [layer normalization](layer-norm.md).
>
> **Dependencies:** just `quanta` (with the `sci` + `autograd` features). That's
> the whole stack — no Python, no external ML runtime.

A GPT block is the single unit a decoder-only language model stacks `L` times.
This tutorial builds one, wires it into a training loop, and watches it learn a
tiny next-token task. It's the capstone of the compute tutorials: it uses
attention, normalization, activations, and autodiff all at once.

## 1. The shape of a block

A decoder block transforms a sequence `x` of shape `[T, D]` (`T` tokens, each a
`D`-dim vector) into an equally-shaped output, two residual sub-layers deep:

```text
x                                     # [T, D]  token embeddings
h   = rmsnorm( x + multihead_attn(x) )   # attention sub-layer + residual
out = rmsnorm( h + ffn(h) )              # feed-forward sub-layer + residual
logits = out · Wout                      # [T, V]  vocabulary scores
loss   = cross_entropy(logits, targets)
```

`multihead_attn` is causal (each position only attends to earlier ones) and
`ffn(h) = gelu(h·W1)·W2`. Everything here is differentiable, so one `.grad(...)`
call per parameter gives you the whole backward pass.

## 2. Setup

```rust,ignore
use quanta::sci::Array;
use quanta::autograd::Tape;

// A device: the CPU JIT by default; a real GPU under the metal/vulkan feature.
let g = quanta::init_cpu();

// Sizes. Tiny — this is a correctness demo, not a scale test.
let (t, v, d, heads, f) = (4usize, 5usize, 8usize, 2usize, 16usize);
```

`T=4` positions, vocab `V=5`, model width `D=8`, `heads=2` (so each head is
`D/heads = 4` wide), and a feed-forward hidden size `F=16`.

## 3. A next-token task

The simplest language-model task: predict the next token in a fixed sequence.
Input tokens `[0,1,2,3]`, targets shifted by one, `[1,2,3,4]`.

```rust,ignore
let tokens:  Vec<u32> = vec![0, 1, 2, 3];
let targets: Vec<u32> = vec![1, 2, 3, 4];
let tok_arr = Array::from_slice(&g, &tokens,  &[t]).unwrap();
let tgt_arr = Array::from_slice(&g, &targets, &[t]).unwrap();
```

## 4. The causal mask

In a decoder, position `i` may only attend to positions `j ≤ i`. We enforce that
with an **additive** mask added to the attention scores before the softmax: `0`
where attention is allowed, a large negative number where it isn't (so those
entries become ~0 after the softmax).

```rust,ignore
let mut mask_h = vec![0.0f32; t * t];
for i in 0..t {
    for j in 0..t {
        if j > i { mask_h[i * t + j] = -1.0e9; } // block the future
    }
}
let mask = Array::from_slice(&g, &mask_h, &[t, t]).unwrap();
```

## 5. Parameters

Ten tensors: the embedding table, the four attention projections
(`Wq`,`Wk`,`Wv`,`Wo`), the two FFN matrices (`W1`,`W2`), the output projection
(`Wout`), and two RMSNorm gains (`g1`,`g2`). Initialize them small.

```rust,ignore
// A cheap deterministic small-random init.
let init = |rows: usize, cols: usize, seed: f32| -> Array<f32> {
    let n = rows * cols;
    let data: Vec<f32> = (0..n)
        .map(|i| ((((i as f32) * 12.9898 + seed).sin() * 43758.547).fract() - 0.5) * 0.2)
        .collect();
    Array::from_slice(&g, &data, &[rows, cols]).unwrap()
};

let mut emb  = init(v, d, 1.0);
let mut wq   = init(d, d, 2.0);
let mut wk   = init(d, d, 3.0);
let mut wv   = init(d, d, 4.0);
let mut wo   = init(d, d, 8.0);
let mut w1   = init(d, f, 5.0);
let mut w2   = init(f, d, 6.0);
let mut wout = init(d, v, 7.0);
let mut g1 = Array::from_slice(&g, &vec![1.0f32; d], &[d]).unwrap();
let mut g2 = Array::from_slice(&g, &vec![1.0f32; d], &[d]).unwrap();
```

## 6. The training loop

Quanta's autograd is **define-by-run**: build a fresh tape each step, run the
forward pass (which records the op graph), then call `.grad(param)` per
parameter and take an SGD step. The forward pass *is* the block from section 1.

```rust,ignore
fn sgd_step(p: &Array<f32>, grad: &Array<f32>, lr: f32) -> Array<f32> {
    let lr_a = Array::full(p.gpu(), lr, &[1]).unwrap();
    p.sub(&grad.mul(&lr_a.broadcast_to(grad.shape()).unwrap()).unwrap()).unwrap()
}

let lr = 0.3f32;
let (mut first, mut last) = (None, 0.0f32);

for _ in 0..40 {
    let tape = Tape::<f32>::new();

    // Lift every parameter onto this step's tape.
    let embv  = tape.var(emb.shallow_clone());
    let wqv   = tape.var(wq.shallow_clone());
    let wkv   = tape.var(wk.shallow_clone());
    let wvv   = tape.var(wv.shallow_clone());
    let wov   = tape.var(wo.shallow_clone());
    let w1v   = tape.var(w1.shallow_clone());
    let w2v   = tape.var(w2.shallow_clone());
    let woutv = tape.var(wout.shallow_clone());
    let g1v   = tape.var(g1.shallow_clone());
    let g2v   = tape.var(g2.shallow_clone());
    let maskv = tape.var(mask.shallow_clone());

    // --- forward ---
    // embed, then add a batch dim of 1 so multi_head_attention sees [B,T,D].
    let x = embv.embedding(&tok_arr).unwrap().reshape(&[1, t, d]).unwrap();

    // multi-head causal self-attention → [B,T,D]
    let attn = x
        .multi_head_attention(&wqv, &wkv, &wvv, &wov, heads, Some(&maskv))
        .unwrap();

    // attention sub-layer: residual + RMSNorm (collapse the batch of 1 to [T,D])
    let x_flat    = x.reshape(&[t, d]).unwrap();
    let attn_flat = attn.reshape(&[t, d]).unwrap();
    let h = x_flat.add(&attn_flat).unwrap().rms_norm(&g1v, 1e-5).unwrap();

    // feed-forward sub-layer: gelu FFN + residual + RMSNorm
    let ff  = h.matmul(&w1v).unwrap().gelu().unwrap().matmul(&w2v).unwrap();
    let out = h.add(&ff).unwrap().rms_norm(&g2v, 1e-5).unwrap();

    // language-model head + loss
    let logits = out.matmul(&woutv).unwrap();          // [T, V]
    let loss   = logits.cross_entropy(&tgt_arr).unwrap();

    last = loss.value().to_vec().unwrap()[0];
    first.get_or_insert(last);

    // --- backward + update ---
    emb  = sgd_step(&emb,  &loss.grad(&embv ).unwrap(), lr);
    wq   = sgd_step(&wq,   &loss.grad(&wqv  ).unwrap(), lr);
    wk   = sgd_step(&wk,   &loss.grad(&wkv  ).unwrap(), lr);
    wv   = sgd_step(&wv,   &loss.grad(&wvv  ).unwrap(), lr);
    wo   = sgd_step(&wo,   &loss.grad(&wov  ).unwrap(), lr);
    w1   = sgd_step(&w1,   &loss.grad(&w1v  ).unwrap(), lr);
    w2   = sgd_step(&w2,   &loss.grad(&w2v  ).unwrap(), lr);
    wout = sgd_step(&wout, &loss.grad(&woutv).unwrap(), lr);
    g1   = sgd_step(&g1,   &loss.grad(&g1v  ).unwrap(), lr);
    g2   = sgd_step(&g2,   &loss.grad(&g2v  ).unwrap(), lr);
}

println!("loss {} → {}", first.unwrap(), last);
assert!(last < first.unwrap() * 0.5, "the block should learn");
```

Run it and the loss drops well below half its starting value — the block learns
to predict the next token. Every gradient in that update came from the tape; you
never wrote a backward pass.

## 7. What you built

A real decoder block, from Quanta primitives:

- **Multi-head causal self-attention** — `multi_head_attention` splits `D` into
  `heads`, runs attention batched over the heads (a batched N-D matmul), applies
  the causal mask, and merges the heads back.
- **RMSNorm** residual sub-layers — the LLaMA-style normalizer.
- **A GELU feed-forward network** — `gelu(h·W1)·W2`, the transformer FFN.
- **A cross-entropy language-model head** — trained by reverse-mode autodiff.

Everything ran through **one dependency graph** (`quanta`, `sci` + `autograd`
features), on the CPU JIT here and unchanged on Metal or Vulkan under the backend
feature.

## 8. Where to take it

The block scales up with pieces the stack already has:

- **Stack `L` blocks** — the loop body is the block; wrap it in a function and
  call it `L` times, threading `out` back in as the next block's `x`.
- **Rotary position embeddings** — apply `RopeCache` + `Var::rope` to the
  queries and keys after the head split, so the model encodes relative position.
- **AdamW instead of SGD** — swap `sgd_step` for `Adam::adamw(lr, wd)`
  (register each parameter once, then `step`), the standard transformer optimizer.
- **Dropout** — `Var::dropout(p, seed)` on the attention and FFN outputs during
  training for regularization.

These are additive: the block above trains as-is, and each of these makes it more
like a production model without changing its shape.
