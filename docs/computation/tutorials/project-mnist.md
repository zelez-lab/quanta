# Project: recognize handwritten digits (MNIST)

> **You'll build:** a standalone Cargo project that downloads the MNIST dataset,
> trains a small convolutional network to recognize handwritten digits 0–9, and
> reports its accuracy. The "hello world" of computer vision, end to end on the
> GPU.
>
> **You'll need:** the [CNN lesson](training-a-cnn.md) and [autodiff basics](autodiff-basics.md).
> This uses the classification stack — `cross_entropy`, `argmax`, `Adam` — that
> a real classifier trains on.

MNIST is 70,000 28×28 grayscale images of handwritten digits. We'll train a
`conv → relu → maxpool → flatten → linear` network with a cross-entropy loss —
the same architecture PyTorch tutorials use.

## 1. Create the project

```sh
cargo new mnist
cd mnist
```

## 2. Dependencies

Quanta from git, plus two small crates to fetch and unzip the dataset — those are
your project's concern, not Quanta's. Edit `Cargo.toml`:

```toml
[package]
name = "mnist"
version = "0.1.0"
edition = "2024"

[dependencies]
# The GPU stack. Use metal on Apple silicon; vulkan on Linux/Windows.
quanta = { git = "https://github.com/zelez-lab/quanta", features = ["sci", "autograd", "metal"] }

# Data plumbing.
ureq   = "2"   # blocking HTTP download
flate2 = "1"   # gunzip the .gz files
```

> MNIST is a real GPU workload — train it on hardware (`metal` / `vulkan`), not
> the software backend, or it will be slow.

> **Backend note:** this model runs fully on Vulkan and Metal (the weights
> below use an `f32` deterministic init, and every op is `f32`). If you later
> swap in random weight init from `quanta::sci::random`, use the **`f32`**
> distributions (`fill_normal_f32_gpu`, …) — they run bit-exact on every
> backend, including devices without 64-bit support. The **`f64`** twins need
> real 64-bit device support (`gpu.supports_i64()` + `gpu.supports_f64()`)
> and return `NotSupported` where it's missing — Metal (no `double` in MSL)
> and the Raspberry Pi's V3D. The CPU backend and llvmpipe run them natively.

## 3. Get the data

MNIST ships as four gzip'd files in the **IDX** format: a big-endian header
(magic number, then the dimension sizes) followed by raw `u8` bytes. Add a small
module `src/data.rs` to download, decompress, and parse them:

```rust,ignore
use std::io::Read;

const BASE: &str = "https://ossci-datasets.s3.amazonaws.com/mnist";

/// Download `name.gz`, gunzip it, return the raw bytes.
fn fetch(name: &str) -> Vec<u8> {
    let url = format!("{BASE}/{name}.gz");
    let resp = ureq::get(&url).call().expect("download");
    let mut gz = Vec::new();
    resp.into_reader().read_to_end(&mut gz).unwrap();
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(&gz[..]).read_to_end(&mut out).unwrap();
    out
}

/// IDX images → (pixels normalized to [0,1], count). 16-byte header, then N·28·28 bytes.
pub fn images(name: &str) -> (Vec<f32>, usize) {
    let b = fetch(name);
    let n = u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize;
    let px = b[16..].iter().map(|&v| v as f32 / 255.0).collect();
    (px, n)
}

/// IDX labels → Vec<u32>. 8-byte header, then N bytes.
pub fn labels(name: &str) -> Vec<u32> {
    let b = fetch(name);
    b[8..].iter().map(|&v| v as u32).collect()
}
```

The four files:

- `train-images-idx3-ubyte` / `train-labels-idx1-ubyte` — 60,000 training pairs
- `t10k-images-idx3-ubyte` / `t10k-labels-idx1-ubyte` — 10,000 test pairs

## 4. The model and training loop

`src/main.rs`:

```rust,ignore
mod data;

use quanta::sci::Array;
use quanta::autograd::{Tape, optim::Adam};

fn main() {
    let gpu = quanta::init().expect("a GPU");

    // Load. Images become [N, 1, 28, 28]; labels become [N] u32.
    let (train_px, n) = data::images("train-images-idx3-ubyte");
    let train_lab = data::labels("train-labels-idx1-ubyte");
    let x = Array::from_slice(&gpu, &train_px, &[n, 1, 28, 28]).unwrap();
    let y = Array::from_slice(&gpu, &train_lab, &[n]).unwrap();
    println!("training on {n} images");

    // Parameters: 8 conv filters (3×3), then a linear head over the pooled map.
    let cout = 8usize;
    let flat = cout * 14 * 14; // 28→28 (pad 1) → maxpool 2 → 14
    let init = |shape: &[usize], scale: f32| {
        let c: usize = shape.iter().product();
        let v: Vec<f32> = (0..c).map(|i| scale * (i as f32 * 1.7).sin()).collect();
        Array::from_slice(&gpu, &v, shape).unwrap()
    };
    let mut wc = init(&[cout, 1, 3, 3], 0.3);
    let mut wl = init(&[flat, 10], 0.05);
    let mut bl = Array::<f32>::zeros(&gpu, &[1, 10]).unwrap();

    // Adam optimizer — register each parameter slot in order.
    let mut opt = Adam::new(0.005);
    opt.register(&wc).unwrap();
    opt.register(&wl).unwrap();
    opt.register(&bl).unwrap();

    let batch = 64;
    let n_batches = n / batch; // drop the ragged tail for simplicity

    for epoch in 0..20 {
        let mut epoch_loss = 0.0f32;
        for b in 0..n_batches {
            opt.advance();
            let tape = Tape::<f32>::new();

            // Select this minibatch — `narrow` is a zero-copy view over the
            // full [N, 1, 28, 28] batch, differentiable so gradients flow only
            // to the selected rows. Labels ride as a plain `Array<u32>`.
            let xb = tape.var(x.shallow_clone()).narrow(b * batch, batch).unwrap();
            let yb = y.narrow(0, b * batch, batch).unwrap();

            let wcv = tape.var(wc.shallow_clone());
            let wlv = tape.var(wl.shallow_clone());
            let blv = tape.var(bl.shallow_clone());

            // Forward: conv → relu → maxpool → flatten → linear.
            let logits = xb
                .conv2d(&wcv, 1, 1).unwrap()
                .relu().unwrap()
                .maxpool2d(2, 2, 2, 0).unwrap()
                .flatten().unwrap()
                .matmul(&wlv).unwrap()
                .add(&blv).unwrap();

            // Cross-entropy loss against this batch's labels.
            let loss = logits.cross_entropy(&yb).unwrap();

            // Backward + Adam update.
            let gwc = loss.grad(&wcv).unwrap();
            let gwl = loss.grad(&wlv).unwrap();
            let gbl = loss.grad(&blv).unwrap();
            wc = opt.step(0, &wc, &gwc).unwrap();
            wl = opt.step(1, &wl, &gwl).unwrap();
            bl = opt.step(2, &bl, &gbl).unwrap();

            epoch_loss += loss.value().to_vec().unwrap()[0];
        }
        println!("epoch {epoch:2}  loss {:.4}", epoch_loss / n_batches as f32);
    }
```

Minibatch SGD: each step trains on a `batch`-row window that
[`Var::narrow`](https://docs.rs/quanta-autograd) selects as a zero-copy view of
the full array — the gradient flows only to the selected rows, so slicing costs
nothing and stays differentiable. `n / batch` steps make an epoch; the reported
loss is the epoch average. (For a stronger model, shuffle the row order each
epoch; the loop body is unchanged.)

### Measure test accuracy

Run the trained network on the held-out test set and count correct predictions
with `argmax_last`:

```rust,ignore
    let (test_px, tn) = data::images("t10k-images-idx3-ubyte");
    let test_lab = data::labels("t10k-labels-idx1-ubyte");
    let tx = Array::from_slice(&gpu, &test_px, &[tn, 1, 28, 28]).unwrap();

    // Forward pass only (a fresh tape; we don't need gradients here).
    let tape = Tape::<f32>::new();
    let xv = tape.var(tx);
    let wcv = tape.var(wc.shallow_clone());
    let wlv = tape.var(wl.shallow_clone());
    let blv = tape.var(bl.shallow_clone());
    let logits = xv
        .conv2d(&wcv, 1, 1).unwrap().relu().unwrap()
        .maxpool2d(2, 2, 2, 0).unwrap().flatten().unwrap()
        .matmul(&wlv).unwrap().add(&blv).unwrap();

    let pred = logits.value().argmax_last().unwrap().to_vec().unwrap();
    let correct = pred.iter().zip(test_lab.iter()).filter(|(a, b)| a == b).count();
    println!("\ntest accuracy: {correct}/{tn} = {:.1}%", 100.0 * correct as f32 / tn as f32);
}
```

## 5. Run it

```sh
cargo run --release
```

The reported loss is the per-epoch average over the minibatches; it starts
below `ln 10 ≈ 2.30` (each step has already updated the weights) and falls
steadily as the network learns:

```text
training on 60000 images
epoch  0  loss 1.0...
epoch  4  loss 0.16...
epoch  9  loss 0.04...

test accuracy: 9xxx/10000 = 9x.x%
```

This small single-conv model lands in the mid-to-high 90s% on the test set — a
real digit recognizer. Exact numbers vary with the init, batch size, and epoch
count; more conv channels, a second layer, and more epochs push it toward 99%,
the standard MNIST ceiling for convnets.

## 6. What you built

A convolutional digit recognizer — download, model, cross-entropy loss, Adam,
accuracy — against a GPU library, with every gradient mechanically proven
correct. The whole classification stack (`conv2d`, `maxpool2d`, `flatten`,
`matmul`, `log_softmax`, `cross_entropy`, `argmax`, `Adam`) is what any image
classifier is made of; MNIST is just the first dataset you point it at.

- Push accuracy up: more conv channels, a second conv layer, a shuffled row
  order, more epochs.
- Coming from PyTorch? The [From NumPy](../../migration/from-numpy.md) guide's
  "Beyond NumPy" table maps `F.cross_entropy`, `F.conv2d`, and friends.
