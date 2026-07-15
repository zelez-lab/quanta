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
| `LayerNorm` ⚗ | planned | fused kernel, fwd + VJP |
| `RMSNorm` ⚗ | planned | fused kernel, fwd + VJP |
| `GroupNorm` | planned | via LayerNorm machinery |
| `BatchNorm2d` | planned | running stats owned by the module |
| `Dropout` | planned | quanta-rand backed, seeded/deterministic mode |
| `Conv2d` | planned | im2col path exists in quanta-array |
| `MaxPool2d` / `AvgPool2d` | planned | exists in array/autograd; wrapped as modules |
| `MultiheadAttention` ⚗ | forward-fused shipped (functional) | `functional::scaled_dot_product_attention` — fused online-softmax forward (no N² materialize; T9200–T9209), single head f32, scale + causal + padding masks, self- and cross-attention; saves per-row `(m, l)` stats. `functional::sdpa_var` is tape-differentiable via the fused forward + composed-VJP backward. **Next increments:** fused backward (consuming the stats), the batched/multi-head `MultiheadAttention` module, and dtype coverage. |
| `RotaryEmbedding` ⚗ | planned | RopeCache exists in autograd; lifted |
| `Sequential` / module containers | planned | with named-parameter traversal |
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
| zeros/ones/constant/uniform/normal | planned |
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
