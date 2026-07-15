# Rotary embeddings

> **You'll learn:** what rotary position embeddings (RoPE) do, how to apply
> the fused `quanta::nn` kernel and differentiate through it, and why one
> kernel with a sign flag serves both the forward and the backward pass.
>
> **You'll need:** [autodiff basics](autodiff-basics.md); the
> [fused attention](fused-attention.md) page for where RoPE slots in.

Attention is permutation-blind: swap two key rows and the scores swap with
them. **RoPE** injects position by rotating each query/key row *before* the
score matmul — pairs of feature channels are treated as 2-D points and
rotated by an angle that grows with the token's position, at a different
frequency per pair. Relative offsets then fall out of the dot product
automatically: the score between positions `p` and `q` depends on `p − q`.

## The cache and the fused kernel

Angles depend only on position and channel, so they're precomputed once
into a `RopeCache` and reused every step:

```rust,ignore
use quanta::autograd::{RopeCache, Tape};
use quanta::nn::rope::rope_var;
use quanta::sci::Array;

let gpu = quanta::init_cpu(); // or quanta::init()?

let (max_t, d) = (128, 64);                   // sequence capacity, head dim (even)
let cache = RopeCache::<f32>::new(&gpu, max_t, d, 10_000.0)?;

let tape: Tape<f32> = Tape::new();
let q = tape.var(Array::from_slice(&gpu, &q_host, &[t, d])?); // [T, d], T ≤ max_t
let q_rot = rope_var(&tape, &q, &cache)?;      // fused rotation, differentiable
```

`rope_var` follows the rotate-half convention: channel `j` pairs with
channel `j + d/2`, and each pair `(x_j, x_{j+d/2})` rotates by `θ_{p,j} =
p · base^{−2j/d}` at row `p`. The kernel is a single loop-free elementwise
pass — one thread per element, the pair partner sits a fixed `d/2` away.

Gradients flow through it like any other op:

```rust,ignore
let loss = q_rot.mul(&weights)?.sum()?;
let dq = loss.grad(&q)?;                       // rotation adjoint, same kernel
```

## One kernel, both directions

A rotation is orthogonal, so its vector-Jacobian product is *the rotation by
the opposite angle*. That is a theorem (T9216), and it has a very concrete
consequence: the backward pass is **the same kernel** invoked with
`sign = −1`, flipping the sine term. Nothing else changes — no transpose, no
second code path, no separate adjoint kernel to keep in sync.

Two companion theorems round out the picture (both in
`specs/verify/lean/Quanta/Nn/RotationVjp.lean`, linked from the
[verification dashboard](../../verification/index.md)):

- **T9218 — inversion.** Running the sign-flipped rotation on the forward's
  output recovers the input exactly: the flag really is the inverse, not an
  approximation of it.
- **T9217 — isometry.** `cos² + sin² = 1`, so every frequency pair keeps its
  norm through the rotation — forward and backward alike. RoPE can amplify
  neither activations nor gradients; it is one of the few layers that is
  *provably* neutral to training stability.

The 2-D `[T, d]` case is the fused core; batch and head dimensions are a
host loop over it, the same convention the attention and norm kernels use.
The composed `Var::rope` (built from matmuls with per-op-proven VJPs) stays
available as the differential-test oracle for the fused path.

## Where it slots in

In a transformer block, RoPE rotates queries and keys — not values — right
before [fused attention](fused-attention.md):

```rust,ignore
let q_rot = rope_var(&tape, &q, &cache)?;
let k_rot = rope_var(&tape, &k, &cache)?;
let ctx = quanta::nn::functional::sdpa_var(&tape, &q_rot, &k_rot, &v, opts)?;
```

The upcoming `MultiheadAttention` layer wires this in as a constructor
option; until then the functional composition above is the idiom.
