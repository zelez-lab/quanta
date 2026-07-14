# quanta-autograd

Reverse-mode automatic differentiation for Quanta. The headline claim:
**every VJP rule is proven to be the analytic derivative** — via Mathlib's
`HasDerivAt` in Lean (`specs/verify/lean/Quanta/Autograd/`) — and
gradient-checked against finite differences on real GPU execution.

The tier-2 *differentiation primitive*: it adds gradients to Quanta's GPU
array ops, the substrate ML consumers (tier-5, e.g. `ai_project`) build training
on. It is **not** an ML framework — no layers, optimizers, or datasets, just
correct gradients. Built on `quanta-array` (the differentiable values) and
`quanta-blas` (the matmul VJP).

## Status — tape-based reverse mode (f32)

| op | forward | VJP |
|----|---------|-----|
| `neg` | `-x` | `-g` |
| `add` / `sub` | `a ± b` (broadcasting) | `g`, `±g` un-broadcast to each operand |
| `mul` / `div` | `a·b`, `a/b` (broadcasting) | `g·b`, `g·a` / `g/b`, `-g·a/b²` |
| `exp` / `log` / `sqrt` | elementwise | `g·y`, `g/x`, `g/(2y)` |
| `relu` / `sigmoid` / `tanh` | activations | `g·[x>0]`, `g·y·(1-y)`, `g·(1-y²)` |
| `matmul` | `A·B` (2-D) | `G·Bᵀ`, `Aᵀ·G` |
| `conv2d` | NCHW conv (im2col·matmul) | matmul VJP + `col2im` (∂x), reshape (∂w) |
| `avgpool2d` / `maxpool2d` | NCHW pooling | scatter `g/(kh·kw)` / route `g` to the argmax pixel |
| `reshape` / `flatten` | shape-only view | `g` reshaped back to the input shape |
| `sum` / `sum_axis` / `mean_axis` | reductions | broadcast `g` (mean: `g/count`) |

A `Tape` records the forward ops as they run (define-by-run); `Var` is a handle
into it. `var.grad(&wrt)` seeds the output gradient and walks the tape in
reverse, applying each op's VJP and accumulating into inputs.

```rust,no_run
use quanta_autograd::Tape;
use quanta_array::Array;

let gpu = quanta::init_cpu();
let tape = Tape::<f32>::new();
let x = tape.var(Array::from_slice(&gpu, &[1.0, 2.0, 3.0], &[3]).unwrap());
// loss = sum(x * x)  ⇒  d loss / d x = 2x
let loss = x.mul(&x).unwrap().sum().unwrap();
assert_eq!(loss.grad(&x).unwrap().to_vec().unwrap(), vec![2.0, 4.0, 6.0]);
```

The VJP rules live as pure functions (`vjp` module) so a future graph/fusion
layer can reuse them; `tape` owns the graph + reverse sweep. matmul is f32-only
(it reuses `quanta-blas`'s f32 GEMM), so the tape's scalar bound is `DiffScalar`
— in practice `f32`.

```toml
[dependencies]
quanta-autograd = { version = "0.1", features = ["metal"] } # vulkan / software
```

## Verification (honest framing)

Three layers of confidence, all green:

- **Lean** — each VJP multiplier is proven equal to the analytic derivative
  (`HasDerivAt`): elementwise + activations in `Vjp.lean` / `ActivationVjp.lean`,
  matmul in `MatmulVjp.lean`, reductions in `ReduceVjp.lean`. `conv2d` reduces to
  matmul plus the im2col/col2im adjoint, and `ConvVjp.lean` proves that adjoint
  (`⟨im2col x, y⟩ = ⟨x, col2im y⟩`) — so its `∂x` is the true gradient.
  `PoolVjp.lean` proves the pooling backwards: avgpool's is its adjoint
  (`⟨avgpool x, y⟩ = ⟨x, avgpoolBack y⟩`) and maxpool's routes each output's
  gradient to its window argmax. 0 sorry; rests on Mathlib calculus (no new
  axioms).
- **Per-op gradient checks** — every op cross-checked against central
  finite differences on real GPU execution (`tests/gradcheck.rs`); `conv2d` and
  the pooling ops add real-Metal lanes against host references (maxpool is
  checked through a squared loss so the upstream gradient is non-uniform).
- **End-to-end** — `tests/training.rs` fits a model by SGD (the loop composes),
  `examples/mlp_training.rs` trains a 2-layer MLP to learn `y = x²`, and
  `examples/cnn_training.rs` trains a conv → relu → maxpool → linear net to
  classify horizontal vs vertical stripes (8/8) — the whole conv stack composed.

Run them: `cargo test -p quanta-autograd` and
`cargo run --example cnn_training -p quanta-autograd --release`.

## Coming next

More activations; broadcasting beyond the right-aligned numpy cases; the
graph/fusion layer (fuse forward + backward) the pure VJP functions were
factored for. f16/bf16 differentiation once `quanta-blas`'s mixed-precision GEMM
is wired through the matmul VJP.
