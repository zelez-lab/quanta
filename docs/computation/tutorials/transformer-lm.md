# A tiny transformer, end to end

This is the summit walk of `quanta::nn`: a character-level language
model built from every shipped piece — `Embedding` at the chain head,
causal + rotary `TransformerEncoderLayer` blocks, a `Linear` head,
fused cross-entropy, Adam, key-threaded dropout, and a named
checkpoint — trained on a toy sequence in seconds on any backend.

The runnable twin of this page is `examples/cookbook_transformer.rs`.

## The model is data

There is no `nn.Module`. A model is a *configuration* (plain structs)
plus a *parameter tree* you own:

```rust
use quanta::nn::attention::MultiheadAttention;
use quanta::nn::embedding::Embedding;
use quanta::nn::layer::{Key, Layer, Linear, LayerNorm, ParamTree};
use quanta::nn::transformer::{EncoderLayerParams, TransformerEncoderLayer};

const V: usize = 8;   // vocab
const E: usize = 32;  // embedding width
const T: usize = 24;  // sequence length

let block = TransformerEncoderLayer {
    // decoder() = causal masking + per-head rotary embeddings — the
    // rope IS the model's sense of position, so no positional table.
    attn: MultiheadAttention::decoder(E, 4),
    ffn_hidden: E,     // SwiGLU width (the ffn projects E → 2H → H → E)
    dropout: 0.1,
    eps: 1e-5,
};
let emb = Embedding { vocab: V, dim: E };
let norm = LayerNorm { dim: E, eps: 1e-5 };
let head = Linear { in_dim: E, out_dim: V, bias: true };
```

The parameter tree is a derived struct — `#[derive(ParamTree)]` gives
it binding, flattening, gradients, and hierarchical NAMES:

```rust
use quanta::nn::layer::{LinearParams, NormParams};
use quanta::nn::{Array, DiffScalar};

#[derive(quanta::nn::layer::ParamTree)]
struct LmParams<S: DiffScalar> {
    emb: Array<S>,                                    // the [V, E] table
    blocks: (EncoderLayerParams<S>, EncoderLayerParams<S>),
    norm: NormParams<S>,
    head: LinearParams<S>,
}
```

Initialization is a fold over one `Key` — splitting CONSUMES it, so
the same seed always builds the same model, with no global RNG and no
init-order hazard:

```rust
let gpu = quanta::init()?;
let key = Key::new(42);
let (k1, rest) = key.split();
let (k2, rest) = rest.split();
let (k3, k4) = rest.split();
let mut params = LmParams::<f32> {
    emb: emb.init(&gpu, k1)?,
    blocks: (block.init(&gpu, k2)?, block.init(&gpu, k3)?),
    norm: norm.init(&gpu, Key::new(0))?,   // norms ignore their key
    head: head.init(&gpu, k4)?,
};
```

## Two forwards, no mode flag

`apply` is the eval forward. `apply_train` is the training forward: it
takes a `Key` and returns the remainder — dropout layers split it, and
tuple stacks thread it member to member. The *signature* says which
semantics you are running; there is nothing to remember to toggle.

```rust
use quanta::nn::Tape;

fn forward_train(
    /* … */ ids: &Array<u32>, key: Key,
) -> /* (logits, remainder key) */ {
    let tape = Tape::<f32>::new();
    let vars = params.bind(&tape);              // tree → tape-bound twin
    let x = emb.apply(&vars.emb, ids)?;         // [T, E] — the chain head
    let (x, key) = block.apply_train(&tape, &vars.blocks.0, &x, key)?;
    let (x, key) = block.apply_train(&tape, &vars.blocks.1, &x, key)?;
    let x = norm.apply(&tape, &vars.norm, &x)?;
    let logits = head.apply(&tape, &vars.head, &x)?;   // [T, V]
    // …
}
```

Dropout here is *deterministic per key*: the mask is a pure function of
(key, element index) — one Philox word per element, regenerated in the
backward, never stored. Rerun the same step with the same key and you
get the same masks on every backend.

## The training loop

Losses are free functions; gradients come back as a tree of the
parameters' shape; the optimizer step *consumes* its state:

```rust
use quanta::nn::loss::{cross_entropy_var, Reduction};
use quanta::nn::optim::Adam;

let opt = Adam::new(3e-3);
let mut state = opt.init(&params)?;
let mut key = Key::new(7);

for step in 0..300 {
    let tape = Tape::<f32>::new();
    let vars = params.bind(&tape);

    let x = emb.apply(&vars.emb, &ids)?;                 // ids: [T] u32
    let (k_step, rest) = key.split();
    key = rest;
    let (x, _spent) = block.apply_train(&tape, &vars.blocks.0, &x, k_step)?;
    // … second block, norm, head as above …
    let loss = cross_entropy_var(&tape, &logits, &next_ids, Reduction::Mean)?;

    let grads = params.grads_from(&vars, &loss)?;        // same tree shape
    (params, state) = opt.step(&params, &grads, state)?; // state consumed
}
```

Everything on the tape is the proven, fused machinery: the streaming
attention never materialises the score matrix on either pass
(T9200–T9209), the LayerNorm backward is the proven three-term adjoint
(T9210), SwiGLU derives σ′ from the forward's sigmoid (T9226), the
cross-entropy is the stable `lse − x_y` form (T9228), and dropout is
exactly unbiased at the rate the kernel implements (T9231–T9233).

## Checkpoint by name

`named_flatten` gives every leaf a stable hierarchical path
(`"blocks.0.attn.wq.w"`, `"norm.gamma"`, `"emb"`), and the state module
serializes through it:

```rust
use quanta::nn::state::{save_state, load_state};

let bytes = save_state(&params)?;                 // dependency-free QNNS bytes
let restored: LmParams<f32> = load_state(&params, &gpu, &bytes)?;
```

Loading matches leaves **by name, never by position** — reorder the
struct's fields and the checkpoint still lands correctly — and a
missing, extra, or wrong-shape leaf fails loudly, naming its path.
Optimizer states are trees of the same shape, so checkpointing training
is the same two calls again.

## Generate

Inference uses `apply` — the eval forward. Inverted dropout means there
is nothing to rescale: eval is simply the identity where training
dropped.

```rust
let tape = Tape::<f32>::new();
let vars = params.bind(&tape);
let x = emb.apply(&vars.emb, &prompt_ids)?;
let x = block.apply(&tape, &vars.blocks.0, &x)?;
let x = block.apply(&tape, &vars.blocks.1, &x)?;
let logits = head.apply(&tape, &norm.apply(&tape, &vars.norm, &x)?)?;
// greedy: argmax of the last row, append, repeat.
```

Run the full program:

```sh
cargo run --release --example cookbook_transformer
```

## Where to go next

* [`nn` API reference](../../reference/api.md) — every module's table.
* [From PyTorch](../../migration/from-torch.md) — the idiom map
  (`state_dict` → named trees, `train()/eval()` → two forwards,
  global RNG → keys).
* `crates/ml/quanta-nn/PARITY.md` — the row-by-row completeness
  contract, including documented deferrals.
* [Verification dashboard](../../verification/index.md) — what the
  T92xx theorems actually claim.
