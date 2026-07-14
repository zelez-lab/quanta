# Project: classify with softmax regression

> **You'll build:** a standalone Cargo project that trains a one-layer softmax
> (multinomial logistic) classifier to separate points into classes — the
> simplest neural classifier, the same `LogisticRegression(multi_class=
> "multinomial")` scikit-learn ships.
>
> **You'll need:** [autodiff basics](autodiff-basics.md) and, ideally, the
> [MNIST project](project-mnist.md) — this is that classifier with the
> convolutions removed.

Softmax regression is a single linear layer followed by a softmax:
`logits = X·W + b`, trained to minimize cross-entropy against integer labels.
No hidden layers, no convolutions — just the classification head. It's the
right first classifier to build by hand because every piece is one op.

## 1. Create the project

```sh
cargo new softmax-reg
cd softmax-reg
```

## 2. Dependencies

```toml
[dependencies]
quanta = { git = "https://github.com/zelez-lab/quanta", features = ["sci", "autograd", "metal"] }
```

## 3. Make some data

Three well-separated blobs of 2-D points, one per class, stacked into `[N, 2]`
with an `[N]` label array. `src/main.rs`:

```rust,ignore
use quanta::sci::Array;
use quanta::autograd::{optim::Adam, Tape};

fn main() {
    let gpu = quanta::init().expect("a GPU");
    let (n_per, d, k) = (30usize, 2usize, 3usize);
    let n = n_per * k;

    let centers = [(0.0f32, 0.0), (6.0, 0.0), (3.0, 6.0)];
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for (c, (cx, cy)) in centers.iter().enumerate() {
        for i in 0..n_per {
            let a = i as f32 * 2.399;                 // spread points out
            let r = 0.6 * ((i % 7) as f32 / 7.0);
            xs.push(cx + r * a.cos());
            xs.push(cy + r * a.sin());
            ys.push(c as u32);
        }
    }
    let x = Array::from_slice(&gpu, &xs, &[n, d]).unwrap();
    let y = Array::from_slice(&gpu, &ys, &[n]).unwrap();
```

## 4. The model

Two parameters: a weight matrix `W [d, k]` mapping features to per-class scores,
and a bias `b [1, k]`. Initialize them small and register them with Adam:

```rust,ignore
    let init = |shape: &[usize], s: f32| {
        let c: usize = shape.iter().product();
        let v: Vec<f32> = (0..c).map(|i| s * (i as f32 * 1.3).sin()).collect();
        Array::from_slice(&gpu, &v, shape).unwrap()
    };
    let mut w = init(&[d, k], 0.1);
    let mut b = Array::<f32>::zeros(&gpu, &[1, k]).unwrap();

    let mut opt = Adam::new(0.05);
    opt.register(&w).unwrap();
    opt.register(&b).unwrap();
```

## 5. Train

Each epoch: forward `X·W + b` to logits, take the cross-entropy loss against the
labels, backprop, and step. `cross_entropy` folds in the softmax and the
numerically-stable log — you never call softmax yourself during training:

```rust,ignore
    for epoch in 0..100 {
        opt.advance();
        let tape = Tape::<f32>::new();
        let xv = tape.var(x.shallow_clone());
        let wv = tape.var(w.shallow_clone());
        let bv = tape.var(b.shallow_clone());

        let logits = xv.matmul(&wv).unwrap().add(&bv).unwrap();
        let loss = logits.cross_entropy(&y).unwrap();

        let gw = loss.grad(&wv).unwrap();
        let gb = loss.grad(&bv).unwrap();
        w = opt.step(0, &w, &gw).unwrap();
        b = opt.step(1, &b, &gb).unwrap();

        if epoch % 20 == 0 {
            println!("epoch {epoch:3}  loss {:.4}", loss.value().to_vec().unwrap()[0]);
        }
    }
```

## 6. Predict

For inference, `argmax` over the logits gives the predicted class (softmax is
monotonic, so you don't need to normalize to compare). Count how many the model
gets right:

```rust,ignore
    let logits = {
        let tape = Tape::<f32>::new();
        tape.var(x.shallow_clone())
            .matmul(&tape.var(w.shallow_clone())).unwrap()
            .add(&tape.var(b.shallow_clone())).unwrap()
    };
    let pred = logits.value().argmax_last().unwrap().to_vec().unwrap();
    let correct = pred.iter().zip(&ys).filter(|(a, b)| a == b).count();
    println!("accuracy: {correct}/{n}");
}
```

```sh
cargo run --release
```

```text
epoch   0  loss 0.9563
epoch  20  loss 0.1012
epoch  40  loss 0.0315
epoch  60  loss 0.0185
epoch  80  loss 0.0133
accuracy: 90/90
```

Linearly-separable classes → the loss drops to near zero and the classifier
gets every point right.

## 7. What you built

The full classification recipe, minus the feature extractor: `matmul → add →
cross_entropy → Adam`. Swap the blobs for real feature vectors and it's a
working softmax classifier; put a `conv → relu → pool` stack in front of it and
it's the [MNIST network](project-mnist.md).

- Coming from scikit-learn? This is `LogisticRegression(multi_class=
  "multinomial", solver="...")` — same model, trained with your own gradient
  loop instead of a black-box solver.
- Want class probabilities instead of a label? Call `logits.softmax()` (rows sum
  to 1) instead of `argmax`.
