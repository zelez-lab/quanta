# Train a model with `quanta::nn`

> **You'll learn:** the whole `quanta::nn` training story — layers as plain
> values, parameters as typed trees, seeded initialization with consumable
> keys, stacking with build-time width contracts, losses, and fused
> optimizers with schedules and gradient clipping — by training a small
> classifier end to end.
>
> **You'll need:** [autodiff basics](autodiff-basics.md). The
> [fused attention](fused-attention.md) and
> [layer normalization](layer-norm.md) pages cover the kernels this stack
> is built from; you don't need them to follow along.

`quanta::nn` is deliberately **not** a torch mirror. Four decisions shape
everything you'll touch on this page:

1. **A layer is configuration, parameters are data.** `Linear { in_dim,
   out_dim, bias }` holds no tensors. Its parameters live in an explicit,
   typed **parameter tree** you create, own, and pass around like any other
   value — so they checkpoint, compare, and (upstream, in dija) shard and
   migrate as values.
2. **Ownership is the effect system.** The RNG key and the optimizer state
   are *consumed* by the operations that advance them. Reusing a stale key
   or replaying an old optimizer state is a compile error, not a silent bug.
3. **Stacking is tuple composition.** `(l1, l2, l3)` *is* a layer. Width
   contracts between neighbours are checked once, when you build the
   parameters — never per forward pass.
4. **Effects are arguments.** The tape, the GPU, the key: everything a
   computation touches arrives as a parameter. Nothing is ambient, global,
   or implicit.

Setup:

```toml
quanta = { version = "0.1", features = ["nn", "metal"] } # or vulkan / software
```

## Layers are values, parameters are trees

A model is a plain tuple of layer configurations:

```rust,ignore
use quanta::nn::layer::{Key, Layer, LayerNorm, Linear, ParamTree};
use quanta::nn::activation::Gelu;

let gpu = quanta::init_cpu(); // or quanta::init()? for a real device

let model = (
    Linear { in_dim: 2,  out_dim: 32, bias: true },
    LayerNorm { dim: 32, eps: 1e-5 },
    Gelu,
    Linear { in_dim: 32, out_dim: 3, bias: true },
);
```

Nothing has been allocated yet. `init` walks the tuple, checks every
neighbour's width contract, and builds the parameter tree — Kaiming-uniform
weights, zero biases, unit gammas:

```rust,ignore
let mut params = Layer::<f32>::init(&model, &gpu, Key::new(42))?;
```

Two things happened in that one line:

- **The width contracts ran.** If the second `Linear` had said `in_dim: 16`,
  `init` would have failed *here*, at model construction — you never get a
  shape error three layers deep in a forward pass. Layers that change width
  participate: `SwiGlu` halves it, and the contract propagates the halving.
- **The key was split and consumed.** `Key::new(42)` is a splittable PRNG
  key. Every layer got its own independent subkey; the same seed always
  builds the same parameters. `split` takes the key *by value* — after
  handing a key to `init` you cannot accidentally reuse it, because the
  borrow checker owns that discipline (decision 2). There is no global RNG
  to seed and no ordering hazard between layers.

`params` is a tuple mirroring the model: `(LinearParams, NormParams, (),
LinearParams)`. Zero-parameter layers like `Gelu` contribute the unit type —
they occupy a stack slot but add no tensors, consume no keys.

Every parameter tree gives you the **ordered leaf view** that optimizers,
checkpointing, and gradient tooling all share:

```rust,ignore
let leaves = params.flatten();            // Vec<Array<f32>>, stable order
let rebuilt = params.unflatten(&mut leaves.into_iter())?; // same shape back
```

## The training step

One step reads: bind the tree to a fresh tape, run the model, take a loss,
pull the gradient *tree*, step the optimizer.

```rust,ignore
use quanta::autograd::Tape;
use quanta::nn::loss::{cross_entropy_var, Reduction};
use quanta::nn::optim::{clip_grad_norm, Adam, Schedule};
use quanta::sci::Array;

// A toy 3-class problem: 64 points in the plane, labels 0/1/2.
let xs: Array<f32> = Array::from_slice(&gpu, &points, &[64, 2])?;
let labels: Vec<u32> = make_labels();

let opt = Adam::adamw(0.01, 0.01);          // AdamW: decoupled weight decay
let mut state = opt.init(&params)?;          // moment trees shaped like params
let sched = Schedule::Cosine { base: 0.01, min_lr: 1e-4, warmup: 10, total: 200 };

for t in 0..200u64 {
    let tape: Tape<f32> = Tape::new();
    let vars = params.bind(&tape);           // tape-bound twin of the tree

    let logits = model.apply(&tape, &vars, &tape.var(xs.shallow_clone()))?;
    let loss = cross_entropy_var(&tape, &logits, &labels, Reduction::Mean)?;

    let grads = params.grads_from(&vars, &loss)?;      // a tree, same shape
    let (grads, _pre_clip_norm) = clip_grad_norm(&grads, 1.0)?;

    let opt_t = Adam { lr: sched.lr(t), ..opt };       // schedule by rebuild
    let (next_params, next_state) = opt_t.step(&params, &grads, state)?;
    params = next_params;
    state = next_state;
}
```

Walk the loop once slowly:

- **`bind` per step.** The tape records one step's computation; binding the
  tree gives you `vars`, the same tree shape with `Var` leaves. The tape is
  an explicit argument to `apply` — decision 4.
- **Gradients are a tree, not a bag.** `grads_from` extracts the gradient
  of the loss with respect to every leaf, *in the tree's own shape*. There
  is no `.grad` field hiding on a tensor and no `zero_grad()` — each step's
  gradients are freshly returned values.
- **The optimizer is functional.** `step` takes `(params, grads, state)`
  and returns `(new_params, new_state)`. The state — momentum / Adam
  moments, one buffer per leaf, plus the step counter — is **consumed**:
  you cannot apply the same state twice. Parameters stay borrowable
  because keeping an old tree around (for a checkpoint or a comparison) is
  legitimate; only the state is linear.
- **The learning rate is data.** Optimizer configs are small `Copy`
  structs. A schedule is a pure function `lr(t)`; you feed it back by
  rebuilding the config with struct-update syntax. No parameter groups, no
  scheduler objects mutating an optimizer behind your back.
- **Clipping is a tree op.** `clip_grad_norm` computes the global L2 norm
  over *all* leaves (the torch semantic) and rescales the whole tree if it
  exceeds the threshold, returning the pre-clip norm for logging.

## The loss menu

All losses take the tape, reduce with `Reduction::Mean` or `Reduction::Sum`,
and return a scalar `Var`:

```rust,ignore
use quanta::nn::loss::{
    bce_with_logits_loss, cross_entropy_var, huber_loss, l1_loss, mse_loss, Reduction,
};

let l = mse_loss(&tape, &pred, &target, Reduction::Mean)?;
let l = l1_loss(&tape, &pred, &target, Reduction::Mean)?;
let l = huber_loss(&tape, &pred, &target, /*delta*/ 1.0, Reduction::Mean)?;
let l = bce_with_logits_loss(&tape, &logits, &targets01, Reduction::Mean)?;
let l = cross_entropy_var(&tape, &logits, &labels_u32, Reduction::Mean)?;
```

Two of these are worth a note:

- **`cross_entropy_var` is fused.** The forward computes the numerically
  stable `lse(x) − x_y` per row directly from max-stabilized row stats, and
  the backward is a single elementwise kernel producing
  `softmax − onehot` — the `N×C` log-softmax intermediate is never
  materialised, in either direction. The stable form is provably
  nonnegative (theorem T9228), so a negative CE reading is a bug, never a
  rounding artifact.
- **`bce_with_logits_loss` cannot saturate.** It uses the overflow-free
  spelling `max(x,0) − x·y + log(1 + e^{−|x|})`, proven equal to the
  textbook `−(y·log σ + (1−y)·log(1−σ))` for every logit (T9229). At
  `x = ±100`, where `σ` rounds to exactly 0 or 1 in f32 and the textbook
  form returns infinity, this one returns the right answer with finite
  gradients. Prefer it over `bce_loss` whenever you have logits.

## What's fused underneath

You don't call kernels from this API, but it's worth knowing what runs when
you do the idiomatic thing:

| You write | What executes |
|---|---|
| `LayerNorm` / `RmsNorm` in a stack | fused forward saving `(μ, rstd)` row stats; fused three-term backward (T9210/T9211) |
| `Gelu` in a stack | fused tanh-form GeLU; the backward reuses the forward's saved tanh (T9227) |
| `SwiGlu` in a stack | one gate kernel per direction; σ′ derived from the forward's sigmoid (T9226) |
| `Softmax` / `LogSoftmax` | max-stabilized rowwise kernels with proven-adjoint backwards (T9223–T9225) |
| `cross_entropy_var` | stable CE off shared row stats, `softmax − onehot` backward (T9228) |
| `Adam::step` / `Sgd::step` | ONE elementwise kernel per leaf: moments, exact bias correction (T9220), decay, and the update in a single dispatch |
| `functional::sdpa_var` / `MultiheadAttention` in a stack | FlashAttention-style streaming attention, fused both directions (T9200–T9209), one fused head at a time |

Every fused path is differentially tested against a composed reference
built from per-op-proven VJPs, and against an f64 host oracle. The theorem
IDs link into the [verification dashboard](../../verification/index.md).

## Where next

- [Fused attention](fused-attention.md) — the streaming kernel underneath
  `attention::MultiheadAttention` (which slots into stacks like any layer,
  with causal and rotary options).
- [Rotary embeddings](rotary-embeddings.md) — positions for attention, one
  sign-flagged kernel for both directions.
- [From PyTorch](../../migration/from-torch.md) — the idiom-by-idiom map if
  you're arriving from torch.
- `PARITY.md` at the crate root — the completeness contract: every declared
  item either ships or carries a documented deferral.
