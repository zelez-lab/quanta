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
| `MultiheadAttention` ⚗ | forward+backward fused shipped (functional) | `functional::scaled_dot_product_attention` — fused online-softmax forward (no N² materialize; T9200–T9209), single head f32, scale + causal + padding masks, self- and cross-attention; saves per-row `(m, l)` stats. `functional::sdpa_var` is tape-differentiable with a **fully fused backward**: a custom VJP node (`Tape::custom_vjp`) dispatches the FlashAttention-style `kernel::sdpa_backward`, reconstructing the softmax weights from the saved `(m, l)` stats (T9204) so the N² score matrix is materialised on *neither* pass. The composed-VJP path is retained as `functional::sdpa_var_composed`, the differential-test oracle. **Next increments:** the batched/multi-head `MultiheadAttention` module, and dtype coverage. |
| `RotaryEmbedding` ⚗ | **shipped (functional)** | `rope::rope_var` — one sign-flagged elementwise kernel serves forward AND backward (T9216-T9218: the VJP of a rotation is the rotation by −θ; isometry per pair). Composed `Var::rope` as oracle. 2-D core; batch/heads = host loop. |
| `Sequential` / containers | **shipped (tuple stacking)** | `layer::Layer` + tuple composition `(l1, l2, …)` per the architecture record (D3): Params = tuple of trees, width contracts checked at `init` (build time). Named traversal + derive = next increment. |
| `TransformerEncoderLayer`-style block | planned | composed from the above; doubles as the SUMMIT A example |
| Conv1d / Conv3d / grouped / depthwise | **deferred** | no consumer, no reference workload; take the next one on demand |
| RNN / LSTM / GRU | **deferred** | legacy family; attention-era stack ships first; revisit on demand |
| InstanceNorm / LocalResponseNorm | **deferred** | rare in modern nets; GroupNorm covers the practical cases |

## Activations (fns + module forms)

| Item | Status |
|---|---|
| ReLU, GeLU ⚗, SiLU, Sigmoid, Tanh | planned (ufuncs exist in array; VJPs in autograd) |
| Softmax / LogSoftmax ⚗ | planned (numerically stable, fused) |
| SwiGLU ⚗ | planned (fused gate kernel) |
| Leaky/PReLU/ELU family | **deferred** — thin wrappers, added on first ask; documented to keep the table honest |

## Losses

| Item | Status |
|---|---|
| MSE, L1, Huber | planned |
| CrossEntropy (from logits, stable) ⚗ | planned |
| BCE / BCEWithLogits | planned |
| KLDiv, CTC, Triplet, etc. | **deferred** — specialist losses on demand |

## Optimizers / training

| Item | Status |
|---|---|
| SGD (momentum, nesterov, weight decay) | planned |
| Adam / AdamW | planned (Adam exists in autograd::optim; lifted + completed) |
| Grad clipping (by norm, by value) | planned |
| LR schedulers: constant, step, cosine, linear-warmup | planned |
| Dynamic loss scaling (mixed precision, bf16) | planned (dtypes shipped in 084.1) |
| RMSprop / Adagrad / LAMB / Lion | **deferred** — on demand |

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
