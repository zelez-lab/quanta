# Fused attention

> **You'll learn:** how to run scaled dot-product attention through `quanta::nn`'s
> **fused** kernel — one call that streams the softmax and never builds the
> `seq_q × seq_k` score matrix — read its `(m, l)` stats, differentiate it, and
> understand why the fused form is numerically stable.
>
> **You'll need:** [autodiff basics](autodiff-basics.md) and, for the *idea* of
> attention built from primitives, [self-attention](attention.md). This page is
> the shipped-kernel counterpart to that conceptual lesson.

The [self-attention](attention.md) tutorial builds attention out of the array
ops you already have — three matmuls, a transpose, a scale, a row softmax. That
composition materialises the full `[seq_q, seq_k]` score matrix in memory. The
`quanta::nn` crate ships a **fused** alternative:
`functional::scaled_dot_product_attention` runs the whole `softmax(scale·QKᵀ +
mask)·V` in a single streaming kernel that carries a running `(max, sum,
accumulator)` per query row and **never allocates the score matrix**. It is the
FlashAttention-style online-softmax form, and its correctness is proven in Lean
(theorems T9200–T9209 on the [verification dashboard](../../verification/index.md)).

```toml
quanta = { version = "0.1", features = ["nn", "metal"] } # or vulkan / software
```

`nn` mounts the `quanta::nn` module; it pulls the `sci` array and `autograd`
tape in as its substrate, so `quanta::sci::Array` and `quanta::autograd::Tape`
are available alongside it. Pair it with a backend (`metal` / `vulkan` /
`software`) for execution.

## A first call

`scaled_dot_product_attention` takes three 2-D arrays — `q:(seq_q, d)`,
`k:(seq_k, d)`, `v:(seq_k, dv)` — and an options struct, and returns the context
`(seq_q, dv)` plus per-row stats:

```rust,ignore
use quanta::nn::functional::{Sdpa, scaled_dot_product_attention};
use quanta::sci::Array;

let gpu = quanta::init_cpu();   // or quanta::init()? for a real device

// A tiny sequence: 3 query rows, 3 keys/values, head dim d = 4, value dim dv = 4.
let (seq_q, seq_k, d, dv) = (3, 3, 4, 4);
let q = Array::from_slice(&gpu, &[
    1.0f32, 0.0, 0.0, 0.0,   0.0, 1.0, 0.0, 0.0,   0.0, 0.0, 1.0, 0.0,
], &[seq_q, d])?;
let k = Array::from_slice(&gpu, &[
    1.0f32, 0.0, 0.0, 0.0,   0.0, 1.0, 0.0, 0.0,   0.0, 0.0, 1.0, 0.0,
], &[seq_k, d])?;
let v = Array::from_slice(&gpu, &[
    10.0f32, 0.0, 0.0, 0.0,   0.0, 20.0, 0.0, 0.0,   0.0, 0.0, 30.0, 0.0,
], &[seq_k, dv])?;

// Full bidirectional attention with the standard 1/√d scale.
let out = scaled_dot_product_attention(&gpu, &q, &k, &v, Sdpa::default())?;

let context = out.output.to_vec()?;   // [seq_q, dv] flattened
let stats   = out.stats.to_vec()?;    // [seq_q, 2] — (m, l) per row
```

Running that prints:

```text
context = [4.5186276, 5.4813724, 8.222058, 0.0,
           2.7406862, 9.037255,  8.222058, 0.0,
           2.7406862, 5.4813724, 13.555882, 0.0]
stats (m, l) per row = [0.5, 2.2130613,  0.5, 2.2130613,  0.5, 2.2130613]
```

Each query row is a convex mixture of the value rows, weighted by the softmax of
its scaled dot products against the keys. `out.output` is a `quanta::sci::Array`
of shape `[seq_q, dv]`; `out.stats` is `[seq_q, 2]` (more on it below).

## The `Sdpa` options

`Sdpa` is a small `Copy` struct; `Sdpa::default()` is plain bidirectional
attention with the standard `1/√d` scale and no padding. Three knobs:

| Field | Type | Meaning |
|---|---|---|
| `scale` | `Option<f32>` | Multiplies the raw `Q·Kᵀ` scores. `None` → `1/√d` (`d` = the query/key head dim). |
| `causal` | `bool` | `true` applies a lower-triangular mask: query row `i` attends only to keys `j ≤ i`. `false` (default) is full attention. |
| `kv_len` | `Option<usize>` | Effective (unpadded) key count. `Some(n)` restricts every query to keys `j < n` (right-padding mask), clamped to `[1, seq_k]`. `None` → all `seq_k` keys are real. |

Use struct-update to set one and keep the rest at their defaults:

```rust,ignore
// Causal (decoder) attention.
let out = scaled_dot_product_attention(&gpu, &q, &k, &v, Sdpa {
    causal: true,
    ..Sdpa::default()
})?;

// Right-padded batch: only the first `n` keys are real.
let out = scaled_dot_product_attention(&gpu, &q, &k, &v, Sdpa {
    kv_len: Some(n),
    ..Sdpa::default()
})?;
```

With `causal: true` on the example above, row 0 attends only to key 0 and its
context comes out as exactly `v[0]` (`[10.0, 0.0, 0.0, 0.0]`), because a query
that can see a single key returns that key's value unchanged — a quick sanity
check that the mask is doing its job. `causal` and `kv_len` compose: both fold
into one additive bias inside the kernel, so causal-and-padded attention is a
single fused pass, not two.

## The `(m, l)` stats

Alongside the context, the forward returns `out.stats` of shape `[seq_q, 2]`:
for each query row, column 0 is `m` — the maximum scaled-and-masked score in that
row — and column 1 is `l` — the softmax normaliser `Σⱼ exp(scoreⱼ − m)`. These
are exactly the running summary the online fold produces (the `l*`, `m*` of
theorem T9204). Two reasons they are surfaced:

- **`m` is the log-sum-exp shift.** `log Σ exp(score) = m + log l`, so `(m, l)`
  is everything you need to recover the per-row partition function without
  recomputing the scores — useful for perplexity, for mixing attention across
  key blocks, or for a KV-cache that appends new keys to an existing summary.
- **The fused backward will consume them.** The next increment's fused backward
  reads `(m, l)` to get the softmax denominator for free instead of
  recomputing it (see [what's shipped](#whats-shipped-and-whats-next)).

## Differentiating it

For training, use `sdpa_var` — the tape-differentiable variant. You wrap `q`,
`k`, `v` as `Var`s on a `Tape`, call `sdpa_var`, and the returned `Var` carries
the attention context ready to feed a loss:

```rust,ignore
use quanta::autograd::Tape;
use quanta::nn::functional::{Sdpa, sdpa_var};
use quanta::sci::Array;

let gpu = quanta::init_cpu();
let (seq_q, seq_k, d, dv) = (2, 2, 2, 2);
let q = Array::from_slice(&gpu, &[1.0f32, 0.0, 0.0, 1.0], &[seq_q, d])?;
let k = Array::from_slice(&gpu, &[1.0f32, 0.5, 0.2, 1.0], &[seq_k, d])?;
let v = Array::from_slice(&gpu, &[1.0f32, 2.0, 3.0, 4.0], &[seq_k, dv])?;

let tape = Tape::<f32>::new();
let qv = tape.var(q);
let kv = tape.var(k);
let vv = tape.var(v);

let ctx  = sdpa_var(&tape, &qv, &kv, &vv, Sdpa::default())?;
let loss = ctx.sum()?;               // any scalar loss of the context
let dq   = loss.grad(&qv)?;          // dL/dQ, shape [seq_q, d]
```

which produces:

```text
loss  = 9.798848
dL/dQ = [-0.52273905, 0.32671204, -0.5483697, 0.3427308]
```

`sdpa_var`'s forward *value* equals the fused kernel's (they are
differential-tested against each other), and it accepts the same `Sdpa` options.
It differentiates with respect to all of `q`, `k`, `v` — ask `loss.grad(&kv)` /
`loss.grad(&vv)` for the others — so the `Wq`/`Wk`/`Wv` projections feeding it
train like any other parameters.

## Why fused

The composed path (`self-attention`, above) computes `scores = scale·Q·Kᵀ`, a
full `[seq_q, seq_k]` matrix, applies the mask and softmax to it, then multiplies
by `V`. For a long sequence that matrix dominates memory: it is quadratic in the
sequence length. The fused kernel avoids ever creating it.

**The online-softmax idea.** A softmax-weighted sum
`(Σⱼ exp(sⱼ − m)·vⱼ) / Σⱼ exp(sⱼ − m)` normally needs two passes over the scores:
one to find the max `m`, one to sum. The *online* form does it in a single pass by
carrying a running `(m, l, acc)` state and rescaling it whenever a new, larger
score arrives:

```text
m'   = max(m, sₖ)
l    = l·exp(m − m') + exp(sₖ − m')
acc  = acc·exp(m − m') + exp(sₖ − m')·vₖ
```

The kernel streams this recurrence over the key sequence per query row, so the
scores exist only transiently in a register — **no N² materialization**. That
the online fold computes *exactly* the two-pass result (for any block schedule)
is the load-bearing invariant proven in Lean: T9204 (the fold summarises the whole
list), T9205 (online `acc/l` equals the direct two-pass output), and T9206 (any
block partition gives the same state).

**±80-logit stability.** Because every step subtracts the running max before
`exp`, every exponent the kernel evaluates is `≤ 0`, so every weight lands in
`(0, 1]` — no overflow to `inf`, no `0·inf = NaN`. That is theorem **T9207**
(every `exp` argument in a step is non-positive) and **T9208** (every weight is
in `(0, 1]`). The crate's test suite makes the property empirical: it drives
logits to ±80 and asserts every output stays finite and matches the shifted f64
reference. The whole T9200–T9209 block is exact over ℝ with **zero axioms** — see
the [verification dashboard](../../verification/index.md).

## What's shipped, and what's next

This is the first slice of `quanta::nn`. The honest bounds today, each tracked
against the crate's completeness contract in
[`PARITY.md`](https://github.com/zelez-lab/quanta/blob/main/crates/ml/quanta-nn/PARITY.md):

- **Single head, `f32`.** `q:(seq_q, d)`, `k:(seq_k, d)`, `v:(seq_k, dv)`. Wider
  dtypes are a later increment.
- **Batch and heads are a host loop.** A `[B, H, T, d]` workload is `B·H`
  independent `scaled_dot_product_attention` calls (each head is a 2-D problem);
  fusing the batch into one dispatch comes later, as does the batched/multi-head
  `MultiheadAttention` *module*.
- **`sdpa_var`'s backward is a naive recompute.** Its forward value matches the
  fused kernel, but the tape records the composed ops (`scale·QKᵀ → mask →
  softmax → ·V`), so backward flows through the existing `quanta-autograd` VJPs
  and *does* rematerialise the score matrix on the backward path. The fused
  backward — consuming the `(m, l)` stats the forward already saves — is the next
  slice.

Looking for the `Module` API, optimizers, or other layers? On the declared
surface they are **not yet shipped** — `PARITY.md` is the map of what's planned
versus what ships, with a documented reason on every row.

> **Migration from PyTorch.** A `migration/from-torch.md` table will land when
> the `Module`/optimizer layer ships — a one-op migration table (just
> `scaled_dot_product_attention`) would serve nobody. The decision is recorded
> here rather than left silent.

## Next

- **[Self-attention](attention.md)** — the same computation built from array
  primitives, and how to train the `Wq`/`Wk`/`Wv` projections around it.
- **[Project: train a GPT decoder block](project-transformer.md)** — assembles
  attention, normalization, and a feed-forward into a real decoder block.
