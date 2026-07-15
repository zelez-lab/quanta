# Computation tutorials

An ordered path. Each lesson builds on the one before, so read top to bottom the
first time through — by the end you can build a full GPU numerical program (load
data, do linear algebra, train a model) without writing a single kernel.

There are two tracks. Most scientists want the **array stack** — the NumPy /
PyTorch-shaped surface. The **kernel track** is for when you need something the
building blocks don't cover yet, or you want to understand what runs underneath.

## The array stack

Work with data, not threads. No kernels, no dispatch — just arrays.

1. [Arrays and broadcasting](arrays-and-broadcasting.md) — build data, elementwise math
2. [Reductions](reductions.md) — sum, mean, and per-axis reductions
3. [Shape and views](shape-and-views.md) — reshape, transpose, broadcast (all zero-copy)
4. [Linear algebra](linear-algebra.md) — matmul, dot, norms, and BLAS
5. [FFT](fft.md) — Fourier transforms
6. [Random numbers](random-numbers.md) — reproducible RNG and distributions
7. [Autodiff basics](autodiff-basics.md) — gradients with a tape
8. [Training an MLP](training-an-mlp.md) — a first neural network
9. [Convolution and pooling](convolution-and-pooling.md) — the CNN building blocks
10. [Training a CNN](training-a-cnn.md) — a convolutional network end to end

### The neural stack (`quanta::nn`)

The MLP/CNN lessons above train through raw tape ops. `quanta::nn` is the
layer above: typed parameter trees, stackable layers, fused theorem-backed
kernels, losses, and fused optimizers.

11. [Self-attention](attention.md) — attention built from array primitives (the idea)
12. [Fused attention](fused-attention.md) — the shipped streaming kernel (the practice)
13. [Layer normalization](layer-norm.md) — the composition, then the fused pair
14. [Rotary embeddings](rotary-embeddings.md) — positions for attention, one kernel both directions
15. [Train a model with quanta::nn](nn-training.md) — the whole training story, end to end

### Build a whole project

Full walkthroughs that end with a complete, standalone Cargo project you run
yourself:

- [Project: linear regression](project-linear-regression.md) — fit a line by gradient descent, from `cargo new` to a working program
- [Project: recognize digits (MNIST)](project-mnist.md) — download MNIST and train a convnet to classify handwritten digits

## The kernel track

Drop to the metal: write your own GPU kernel in Rust and dispatch it.

- [Compute Basics](compute-basics.md)
- [Fields and Types](fields-and-types.md)
- [Shared Memory](shared-memory.md)
- [Atomics](atomics.md)
- [Wave Intrinsics](wave-intrinsics.md)
- [Device Functions](device-functions.md)
- [Async Copy and Printf](async-copy-and-printf.md)
- [Block Primitives](block-primitives.md)
- [Arrays (under the hood)](arrays.md)

Every lesson has a matching [how-to recipe](../how-to/) for when you already know
the concept and just need the code, and links into the [API reference](../../reference/api.md).
