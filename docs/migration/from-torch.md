# From PyTorch

`quanta::nn` covers the torch.nn capability surface (the crate's
`PARITY.md` is the row-by-row contract), but it is deliberately **not** an
API mirror. The differences are architectural, and most torch idioms have a
sharper equivalent rather than a lookalike. This page is the idiom map.

## The mental model shift

| PyTorch | `quanta::nn` | Why it's different |
|---|---|---|
| `nn.Module` holds its parameters | A layer is *configuration only*; parameters live in an explicit typed **tree** you own (`ParamTree`) | Params-as-data: trees checkpoint, compare, and shard as plain values |
| `model(x)` (implicit params, implicit autograd) | `model.apply(&tape, &vars, &x)` — tape and params are arguments | Nothing ambient; a forward is reproducible from its arguments alone |
| `loss.backward()` fills `.grad` fields | `params.grads_from(&vars, &loss)?` returns a **gradient tree** | Gradients are values in the tree's shape, not mutable slots |
| `optimizer.step()` mutates params in place; `zero_grad()` | `(params, state) = opt.step(&params, &grads, state)?` | Functional update; the state is *consumed* — replaying a stale state is a compile error |
| `torch.manual_seed(0)` (global RNG) | `Key::new(seed)`, split per layer, **consumed on use** | No global RNG, no init-order hazard; same key ⇒ same params, always |
| `nn.Sequential(l1, l2)` | the tuple `(l1, l2)` *is* a layer | Width contracts checked at `init` (build time), not at first forward |
| `model.state_dict()` | `params.flatten()` → ordered `Vec<Array<T>>` (+ `unflatten`) | The same leaf view optimizers and checkpointing share |
| `requires_grad` / `no_grad()` | binding to a tape is what makes something differentiable | Differentiability is structural, not a flag on a tensor |
| lr schedulers wrap the optimizer | `Schedule` is a pure `lr(t)`; rebuild the `Copy` config: `Adam { lr: sched.lr(t), ..opt }` | The learning rate is data, not hidden mutable state |

## Layers

| torch.nn | quanta |
|---|---|
| `nn.Linear(i, o, bias=…)` | `layer::Linear { in_dim, out_dim, bias }` |
| `nn.LayerNorm(c)` | `layer::LayerNorm { dim, eps }` — fused fwd/bwd (T9210) |
| `nn.RMSNorm(c)` | `layer::RmsNorm { dim, eps }` — fused (T9211) |
| `nn.ReLU / GELU / SiLU / Sigmoid / Tanh` | `activation::{Relu, Gelu, Silu, Sigmoid, Tanh}` — zero-param layers; GeLU fused (tanh form) |
| `nn.Softmax(dim=-1)` / `LogSoftmax` | `activation::{Softmax, LogSoftmax}` — rowwise, fused, proven-adjoint backwards |
| SwiGLU (hand-rolled in torch) | `activation::SwiGlu` — one fused kernel per direction, halves the width |
| `F.scaled_dot_product_attention` | `functional::sdpa_var` — FlashAttention-style, fused both directions |
| rotary embeddings (hand-rolled) | `rope::rope_var` + `RopeCache` — one sign-flagged kernel both directions |
| `nn.MultiheadAttention` | next increment — the fused SDPA + projections as a layer |
| `nn.Embedding` | `Var::embedding` today; module form arrives with the state/derive increment |
| `nn.Dropout` | planned (`Key`-seeded, deterministic) — see `PARITY.md` |

## Losses

All losses are free functions taking the tape and a `Reduction`
(`Mean` / `Sum`):

| torch | quanta |
|---|---|
| `F.mse_loss(p, t)` | `loss::mse_loss(&tape, &p, &t, Reduction::Mean)` |
| `F.l1_loss` | `loss::l1_loss(…)` |
| `F.huber_loss(delta=δ)` | `loss::huber_loss(&tape, &p, &t, δ, …)` — gradient is provably `clamp(z, −δ, δ)` (T9230) |
| `F.cross_entropy(logits, labels)` | `loss::cross_entropy_var(&tape, &logits, &labels_u32, …)` — fused, stable `lse − x_y` form, nonnegative by T9228 |
| `F.binary_cross_entropy_with_logits` | `loss::bce_with_logits_loss` — overflow-free spelling, proven equal to the textbook form (T9229) |
| `F.binary_cross_entropy` | `loss::bce_loss` (prefer the logits form) |

`cross_entropy` takes integer labels as `&[u32]` — class indices, exactly
like torch's `target` tensor.

## Optimizers

| torch | quanta |
|---|---|
| `optim.SGD(lr, momentum, weight_decay, nesterov)` | `optim::Sgd { lr, momentum, weight_decay, nesterov }` |
| `optim.Adam(lr, betas, eps, weight_decay)` | `optim::Adam { lr, beta1, beta2, eps, weight_decay, decoupled: false }` |
| `optim.AdamW(…)` | `optim::Adam::adamw(lr, wd)` — same kernel, `decoupled: true` (the equivalence of the two decay spellings is T9221) |
| `clip_grad_norm_(params, max)` | `optim::clip_grad_norm(&grads, max)` — global L2 over all leaves, returns the pre-clip norm |
| `clip_grad_value_` | `optim::clip_grad_value(&grads, max_abs)` |
| `CosineAnnealingLR` + warmup wrappers | `Schedule::Cosine { base, min_lr, warmup, total }` |
| `StepLR(gamma, step_size)` | `Schedule::Step { lr, gamma, every }` |
| `LinearLR` warmup | `Schedule::LinearWarmup { lr, warmup }` |

The optimizer update is **one fused kernel per parameter leaf** — moments,
bias correction, weight decay, and the step in a single dispatch. Bias
correction uses the exact `1/(1−βᵗ)` factors (T9220: exact at every step,
not just asymptotically).

## A full loop, side by side

PyTorch:

```python
model = nn.Sequential(nn.Linear(2, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 3))
opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
for t in range(200):
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
```

quanta:

```rust,ignore
let model = (
    Linear { in_dim: 2, out_dim: 32, bias: true },
    LayerNorm { dim: 32, eps: 1e-5 },
    Gelu,
    Linear { in_dim: 32, out_dim: 3, bias: true },
);
let mut params = Layer::<f32>::init(&model, &gpu, Key::new(0))?;
let opt = Adam::adamw(1e-2, 0.01);
let mut state = opt.init(&params)?;
for t in 0..200u64 {
    let tape: Tape<f32> = Tape::new();
    let vars = params.bind(&tape);
    let loss = cross_entropy_var(&tape, &model.apply(&tape, &vars, &x)?, &y, Reduction::Mean)?;
    let grads = params.grads_from(&vars, &loss)?;
    let (grads, _) = clip_grad_norm(&grads, 1.0)?;
    let (p, s) = opt.step(&params, &grads, state)?;
    params = p; state = s;
}
```

The [Train a model](../computation/tutorials/nn-training.md) tutorial walks
this loop line by line.

## Gotchas for torch hands

- **`Mean` averages over all elements**, like torch's default `reduction='mean'`.
- **Weight-decay semantics are explicit.** `decoupled: true` is AdamW
  (decay outside the moments); `false` folds it into the gradient
  (classic L2). Torch's `Adam(weight_decay=…)` is the coupled one.
- **2-D cores.** The fused rowwise kernels (norms, softmax family, CE,
  RoPE, SDPA) operate on `[N, C]` / `[T, d]`; batch and head dimensions
  are host loops for now, matching the single-head attention core.
- **No `.to(device)`.** Arrays are created on a `Gpu` handle and stay
  there; there is one device per handle rather than a global default.
- **No DataLoader.** Data loading and tokenizers are a separate step of
  the numeric stack (see the structural exclusions in `PARITY.md`);
  minibatch selection inside a graph is `Var::narrow` on axis 0.
