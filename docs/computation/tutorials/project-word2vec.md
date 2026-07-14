# Project: learn word embeddings (word2vec)

> **You'll build:** a standalone Cargo project that learns a vector per token so
> that tokens which co-occur end up close together — the core of word2vec's
> skip-gram, the same idea behind PyTorch's `nn.Embedding`.
>
> **You'll need:** [autodiff basics](autodiff-basics.md). This introduces the
> *embedding lookup* — selecting rows of a table by index, differentiably.

Word2vec learns a `[V, E]` table: one `E`-dimensional vector per token in a
vocabulary of size `V`. Training pushes the vectors of tokens that appear
together to have a large dot-product, and random pairs to have a small one. The
only new op is the **embedding lookup**: `embedding(ids)` selects whole rows of
the table, and — because it's differentiable — the gradient flows back only to
the rows you looked up (the *sparse update*).

## 1. Create the project

```sh
cargo new word2vec
cd word2vec
```

## 2. Dependencies

```toml
[dependencies]
quanta = { git = "https://github.com/zelez-lab/quanta", features = ["sci", "autograd", "metal"] }
```

## 3. A tiny corpus

Real word2vec slides a window over a text corpus to produce `(center, context)`
pairs. We'll fake it: tokens `0,1,2` are "topic A" and co-occur; `3,4,5` are
"topic B". Positive pairs (label `1`) are within a topic; negatives (label `0`)
cross topics. `src/main.rs`:

```rust,ignore
use quanta::sci::Array;
use quanta::autograd::{optim::Adam, Tape};

fn main() {
    let gpu = quanta::init().expect("a GPU");
    let (vocab, e) = (6usize, 4usize);

    let centers:  Vec<u32> = vec![0, 1, 2, 0, 3, 4, 5, 3,  0, 1, 3, 4];
    let contexts: Vec<u32> = vec![1, 2, 0, 2, 4, 5, 3, 5,  3, 4, 0, 1];
    let labels:   Vec<f32> = vec![1.,1.,1.,1.,1.,1.,1.,1., 0.,0.,0.,0.];
    let b = centers.len();

    let cen = Array::from_slice(&gpu, &centers, &[b]).unwrap();
    let ctx = Array::from_slice(&gpu, &contexts, &[b]).unwrap();
    let lab = Array::from_slice(&gpu, &labels, &[b, 1]).unwrap();
```

## 4. The embedding table

One `[vocab, E]` table of parameters, initialized small and registered with
Adam:

```rust,ignore
    let init: Vec<f32> = (0..vocab * e).map(|i| 0.1 * (i as f32 * 1.7).sin()).collect();
    let mut emb = Array::from_slice(&gpu, &init, &[vocab, e]).unwrap();

    let mut opt = Adam::new(0.1);
    opt.register(&emb).unwrap();
```

## 5. Train

Each step looks up the center and context vectors with `embedding`, scores each
pair by their dot-product (`elementwise multiply → sum over the feature axis`),
squashes with `sigmoid`, and fits the labels with an MSE loss. The gradient
flows back through `embedding` to exactly the rows that were looked up:

```rust,ignore
    for epoch in 0..200 {
        opt.advance();
        let tape = Tape::<f32>::new();
        let ev = tape.var(emb.shallow_clone());

        let ce = ev.embedding(&cen).unwrap();   // [B, E] center vectors
        let co = ev.embedding(&ctx).unwrap();   // [B, E] context vectors

        // score = sigmoid( Σ_E ce·co )  → [B, 1]
        let dot = ce.mul(&co).unwrap().sum_axis(1).unwrap();
        let loss = dot.sigmoid().unwrap().mse_loss(&lab).unwrap();

        let ge = loss.grad(&ev).unwrap();
        emb = opt.step(0, &emb, &ge).unwrap();

        if epoch % 40 == 0 {
            println!("epoch {epoch:3}  loss {:.4}", loss.value().to_vec().unwrap()[0]);
        }
    }
```

## 6. Check what it learned

Score every pair again: positives should sit above 0.5, negatives below.

```rust,ignore
    let tape = Tape::<f32>::new();
    let ev = tape.var(emb.shallow_clone());
    let score = ev.embedding(&cen).unwrap()
        .mul(&ev.embedding(&ctx).unwrap()).unwrap()
        .sum_axis(1).unwrap()
        .sigmoid().unwrap()
        .value().to_vec().unwrap();
    println!("scores: {score:?}");
}
```

```sh
cargo run --release
```

```text
epoch   0  loss 0.2476
epoch  40  loss 0.0000
scores: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
```

The eight in-topic pairs score ~1, the four cross-topic pairs ~0 — the
embeddings learned which tokens belong together, with no supervision beyond
co-occurrence.

## 7. What you built

A word-embedding trainer built on one new op: `Var::embedding`, a differentiable
row lookup whose backward is a sparse scatter-add — only the rows a batch
touches get updated, which is exactly why embedding training scales to huge
vocabularies.

- Coming from PyTorch? `Var::embedding` is `nn.Embedding.forward`; its sparse
  gradient is what `nn.Embedding(sparse=True)` gives you.
- Scale it up: a real corpus of `(center, context)` pairs, negative sampling
  from the token-frequency distribution (a weighted draw), and a second output
  table for the context vectors.
