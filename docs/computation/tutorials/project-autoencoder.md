# Project: compress images (autoencoder)

> **You'll build:** a standalone Cargo project that trains a convolutional
> autoencoder — a network that squeezes an image down to a small bottleneck and
> reconstructs it — learning a compact representation with no labels.
>
> **You'll need:** the [CNN lesson](training-a-cnn.md) and
> [autodiff basics](autodiff-basics.md). This adds the *decoder* — how you grow
> a small feature map back up to an image.

An autoencoder has two halves. The **encoder** shrinks the input through convs
and pooling to a small bottleneck; the **decoder** grows it back to the original
size and tries to reproduce the input. Training on the reconstruction error
forces the bottleneck to keep only what matters — unsupervised representation
learning. The new piece is `upsample2d`, the decoder's answer to pooling.

## 1. Create the project

```sh
cargo new autoencoder
cd autoencoder
```

## 2. Dependencies

```toml
[dependencies]
quanta          = { git = "https://github.com/zelez-lab/quanta", features = ["metal"] }
quanta-array    = { git = "https://github.com/zelez-lab/quanta", features = ["metal"] }
quanta-autograd = { git = "https://github.com/zelez-lab/quanta", features = ["metal"] }
```

## 3. Some images

For a self-contained demo, generate a few small structured images — real code
would load a dataset. Each is `1×8×8`. `src/main.rs`:

```rust,ignore
use quanta_array::Array;
use quanta_autograd::{optim::Adam, Tape};

fn main() {
    let gpu = quanta::init().expect("a GPU");
    let (n, h, w) = (4usize, 8usize, 8usize);

    let mut imgs = vec![0.0f32; n * h * w];
    for b in 0..n {
        for i in 0..h {
            for j in 0..w {
                imgs[(b * h + i) * w + j] = ((i + j + b) as f32 * 0.3).sin() * 0.5 + 0.5;
            }
        }
    }
    let x = Array::from_slice(&gpu, &imgs, &[n, 1, h, w]).unwrap();
```

## 4. Encoder and decoder

Two conv weights. The **encoder** convolves to 4 feature maps, ReLUs, and pools
2× down to the `[n, 4, 4, 4]` bottleneck. The **decoder** upsamples 2× back to
`8×8` and convolves down to a single reconstructed channel:

```rust,ignore
    let init = |shape: &[usize], s: f32| {
        let c: usize = shape.iter().product();
        let v: Vec<f32> = (0..c).map(|i| s * ((i as f32) * 1.7).sin()).collect();
        Array::from_slice(&gpu, &v, shape).unwrap()
    };
    let mut we = init(&[4, 1, 3, 3], 0.3);   // encoder conv (4 filters)
    let mut wd = init(&[1, 4, 3, 3], 0.2);   // decoder conv (back to 1 channel)

    let mut opt = Adam::new(0.01);
    opt.register(&we).unwrap();
    opt.register(&wd).unwrap();
```

## 5. Train on reconstruction error

Each step: encode to the bottleneck, decode back, and minimize the MSE between
the reconstruction and the original. There's no label — the *input is the
target*:

```rust,ignore
    for epoch in 0..120 {
        opt.advance();
        let tape = Tape::<f32>::new();
        let xv = tape.var(x.shallow_clone());
        let wev = tape.var(we.shallow_clone());
        let wdv = tape.var(wd.shallow_clone());

        // encode → [n, 4, 4, 4] bottleneck
        let z = xv
            .conv2d(&wev, 1, 1).unwrap()
            .relu().unwrap()
            .maxpool2d(2, 2, 2, 0).unwrap();

        // decode → [n, 1, 8, 8] reconstruction
        let recon = z
            .upsample2d(2).unwrap()          // grow 4×4 → 8×8
            .conv2d(&wdv, 1, 1).unwrap();

        let loss = recon.mse_loss(&x).unwrap();   // reconstruct the input

        we = opt.step(0, &we, &loss.grad(&wev).unwrap()).unwrap();
        wd = opt.step(1, &wd, &loss.grad(&wdv).unwrap()).unwrap();

        if epoch % 30 == 0 {
            println!("epoch {epoch:3}  recon loss {:.5}", loss.value().to_vec().unwrap()[0]);
        }
    }
}
```

```sh
cargo run --release
```

```text
epoch   0  recon loss 0.55399
epoch  30  recon loss 0.01863
epoch  60  recon loss 0.00484
epoch  90  recon loss 0.00385
```

The loss drops far below the variance of the data — the bottleneck learned to
carry enough of each image through the 2× spatial squeeze to rebuild it. That's
the autoencoder compressing and reconstructing.

## 6. How upsample fits

`maxpool2d(2, …)` in the encoder halves the spatial size; `upsample2d(2)` in the
decoder doubles it back. Upsampling replicates each pixel over a `k×k` block,
and it's differentiable — in the backward pass the gradient sums each block back
to its source pixel (the adjoint of replication), so the encoder learns *what*
to put in the bottleneck for the decoder to expand. It's the mirror image of
pooling, which is exactly the symmetry an autoencoder is built on.

## 7. What you built

An unsupervised image compressor: an encoder that finds a compact code and a
decoder that reconstructs from it, trained end to end on reconstruction error.

- Coming from PyTorch? The decoder here uses nearest-neighbour `upsample2d`
  (like `nn.Upsample`) followed by a conv — the common, stable alternative to a
  transposed convolution.
- Go further: a deeper encoder/decoder for a tighter bottleneck, a denoising
  autoencoder (feed corrupted input, target the clean image), or a variational
  bottleneck for a generative model.
