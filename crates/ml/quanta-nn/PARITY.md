# quanta-nn — declared completeness surface

This file is the crate's definition of done. Every row is either SHIPPED or
a DOCUMENTED DEFERRAL with its reasoning. Nothing on the reference surface
(the torch.nn / burn core set) is silently absent; "complete" is a claim you
can check against this table. Kernels marked ⚗ follow the verified-track
recipe (Lean proof foundation, then implementation with differential tests).

## Layers / modules

| Item | Status | Notes |
|---|---|---|
| `Linear` | planned | blas gemm-backed; bias optional |
| `Embedding` | planned | array `select_rows` backed |
| `LayerNorm` ⚗ | **shipped (functional)** | `norm::layer_norm_var` — fused fwd (saves `(μ, rstd)` stats) + the proven T9210 three-term backward via `custom_vjp`; composed `Var::layer_norm` retained as oracle. Module form arrives with the Layer slice. |
| `RMSNorm` ⚗ | **shipped (functional)** | `norm::rms_norm_var` — fused fwd/bwd (T9211, no centering term); composed `Var::rms_norm` as oracle. Module form with the Layer slice. |
| `GroupNorm` | planned | via LayerNorm machinery |
| `BatchNorm2d` | planned | running stats owned by the module |
| `Dropout` | planned | quanta-rand backed, seeded/deterministic mode |
| `Conv2d` | planned | im2col path exists in quanta-array |
| `MaxPool2d` / `AvgPool2d` | planned | exists in array/autograd; wrapped as modules |
| `MultiheadAttention` ⚗ | **shipped (module)** | `attention::MultiheadAttention` — four `Linear` projections around H fused streaming heads (`functional::sdpa_var`, T9200–T9209: the N² score matrix exists on neither pass). Optional per-head rotary (`rope: true`, T9216–T9218) and causal masking; `Layer::apply` = self-attention, inherent `attend` = cross-attention; head-divisibility contract fails at `init`. Params = `MhaParams`, the first `#[derive(ParamTree)]` consumer. Tested against the composed `Var::multi_head_attention` oracle (values + gradients, both mask modes), gradchecked with biases, future-leak probed, trained in a tuple stack. 2-D `[T, E]` core (batch = host loop); per-call rope cache + padding-mask passthrough = next increments. |
| `RotaryEmbedding` ⚗ | **shipped (functional)** | `rope::rope_var` — one sign-flagged elementwise kernel serves forward AND backward (T9216-T9218: the VJP of a rotation is the rotation by −θ; isometry per pair). Composed `Var::rope` as oracle. 2-D core; batch/heads = host loop. |
| `Sequential` / containers | **shipped (tuple stacking)** | `layer::Layer` + tuple composition `(l1, l2, …)` per the architecture record (D3): Params = tuple of trees, width contracts checked at `init` (build time). **`#[derive(ParamTree)]`** (quanta-nn-derive) generates user trees — `Vars` twin + bind/flatten/unflatten/grads; `Option<P>` subtrees supported. Named traversal = state increment. |
| `TransformerEncoderLayer`-style block | planned | composed from the above; doubles as the SUMMIT A example |
| Conv1d / Conv3d / grouped / depthwise | **deferred** | no consumer, no reference workload; take the next one on demand |
| RNN / LSTM / GRU | **deferred** | legacy family; attention-era stack ships first; revisit on demand |
| InstanceNorm / LocalResponseNorm | **deferred** | rare in modern nets; GroupNorm covers the practical cases |

## Activations (fns + module forms)

| Item | Status | Notes |
|---|---|---|
| ReLU, GeLU ⚗, SiLU, Sigmoid, Tanh | **shipped** | zero-param module forms (`activation::{Relu, Gelu, Silu, Sigmoid, Tanh}`, `Params = ()`) over the composed per-op-proven VJPs; GeLU is FUSED (tanh-approx GPT-2 form; backward reuses the forward's tanh via T9227 — no cosh) with the composed `Var::gelu` as oracle. |
| Softmax / LogSoftmax ⚗ | **shipped** | `activation::{softmax_var, log_softmax_var}` + module forms — fused rowwise max-stabilized forward (T9223 exactness) and the proven-adjoint backwards (T9224/T9225); extreme-logit stability tested at ±1e4. Composed ops as oracles. |
| SwiGLU ⚗ | **shipped** | `activation::swiglu_var` + `SwiGlu` layer (`[N, 2H] → [N, H]` — the width contract propagates the halving through stacks); backward derives σ′ from the forward's sigmoid (T9226). Composed split/silu/mul path as oracle. |
| Leaky/PReLU/ELU family | **deferred** | thin wrappers, added on first ask; documented to keep the table honest |

## Losses

| Item | Status | Notes |
|---|---|---|
| MSE, L1, Huber | **shipped** | `loss::{mse_loss, l1_loss, huber_loss}` — composed from per-op-proven VJPs, Mean/Sum reductions; Huber's knee constants and clamp-gradient continuity are T9230, checked empirically across the knee. |
| CrossEntropy (from logits, stable) ⚗ | **shipped** | `loss::cross_entropy_var` — FUSED both directions off the shared max-stabilized stats kernel: forward `lse(x) − x_y` per row (nonnegative, T9228), backward one elementwise `scale·(softmax − onehot)`; the N×C log-softmax intermediate exists on neither pass. Composed `Var::cross_entropy` + f64 host as oracles; ±1e4-logit stability tested. |
| BCE / BCEWithLogits | **shipped** | `loss::{bce_loss, bce_with_logits_loss}` — the logits form uses the overflow-free spelling proven equal to the textbook one (T9229); exact at ±100 logits where σ rounds to 0/1. |
| KLDiv, CTC, Triplet, etc. | **deferred** | specialist losses on demand |

## Optimizers / training

| Item | Status | Notes |
|---|---|---|
| SGD (momentum, nesterov, weight decay) ⚗ | **shipped** | `optim::Sgd` — tree-shaped state-passing `step` over `ParamTree` (consumes the state, D2); ONE fused kernel per leaf folds decay + the T9219 momentum recurrence + the classical/nesterov direction. f64-host differential oracle. |
| Adam / AdamW ⚗ | **shipped** | `optim::Adam` (`decoupled` flag = AdamW) — one fused kernel per leaf: both moment recurrences, exact bias correction (T9220), both weight-decay spellings (T9221 licenses the shared kernel); step magnitude scale-invariance (T9222) checked empirically. Supersedes `autograd::optim`'s composed per-slot Adam (kept as-is where it's used). |
| Grad clipping (by norm, by value) | **shipped** | `optim::{clip_grad_norm, clip_grad_value}` — global L2 norm over ALL leaves (torch semantics), returns the pre-clip norm. |
| LR schedulers: constant, step, cosine, linear-warmup | **shipped** | `optim::Schedule` — a pure `lr(t)` enum; feed back by rebuilding the `Copy` config (`Adam { lr: sched.lr(t), ..opt }`). Warmup+cosine is the transformer default. |
| Dynamic loss scaling (mixed precision, bf16) | planned | dtypes shipped in 084.1; arrives with the mixed-precision training increment |
| RMSprop / Adagrad / LAMB / Lion | **deferred** | on demand — the fused-kernel + tree-state recipe above extends directly |

## Initialization (quanta-rand backed)

| Item | Status |
|---|---|
| zeros/ones/uniform (kaiming) | **shipped (in layer init)** — full standalone init family with the derive increment |
| Xavier/Glorot (u+n), Kaiming/He (u+n) | planned |

## State

| Item | Status |
|---|---|
| Named parameter traversal (`state_dict` equivalent) | planned |
| In-memory + bytes round-trip save/load | planned |
| safetensors / npy interop | **deferred to step 084.8** — file-format IO is the numeric-stack step's scope; this crate exposes the traversal it will consume |
| torch/ONNX checkpoint import | **deferred** — interop lane, post-084.8 |

## Structural exclusions (not deferrals — they live elsewhere by design)

- Anything distributed (sharding, collectives, actor-scheduled backward):
  dija-nn's scope per the placement policy; it wraps this crate.
- Data loading / tokenizers: step 084.8 (quanta-data lane).
- Inference-serving optimizations (paged KV cache, batching servers):
  a future inference step; the attention kernel exposes the kv-cache hook.
