# Random numbers

> **You'll learn:** how to generate reproducible random numbers and draw from
> common distributions on the GPU. Builds on [FFT](fft.md).

Randomness shows up everywhere in numerical work — initializing weights, dropout,
Monte-Carlo integration, sampling. `quanta-rand` gives you **counter-based**
generators whose output is *bit-exact across every backend* for a given seed, so
a run is reproducible whether it lands on Metal, Vulkan, or the CPU.

```toml
quanta-rand = { version = "0.1", features = ["gpu-metal"] } # or gpu-vulkan / gpu
```

## Filling a buffer

The GPU generators fill a `Field` (a device buffer) in one launch. Uniform
values in `[0, 1)`:

```rust,ignore
let n = 1024;
let out = gpu.field::<f32>(n)?;
quanta_rand::fill_uniform_f32_gpu(&gpu, &out, /* seed */ 42)?;
```

Same seed → same bytes, every time, on every backend. That determinism is the
whole point: a bug reproduces, a paper's result reproduces.

## Distributions

Beyond uniform, there are six distributions ready to fill:

```rust,ignore
quanta_rand::fill_normal_f32_gpu(&gpu, &out, seed)?;      // N(0, 1)
quanta_rand::fill_exponential_f32_gpu(&gpu, &out, seed)?; // Exp(1)
quanta_rand::fill_lognormal_f32_gpu(&gpu, &out, seed)?;   // exp(N(μ,σ))
quanta_rand::fill_bernoulli_u32_gpu(&gpu, &out_u32, seed, 0.3)?; // 1 w.p. 0.3
quanta_rand::fill_poisson_u32_gpu(&gpu, &out_u32, seed, 4.0)?;   // Poisson(λ=4)
```

Normal weights for a layer, a Bernoulli dropout mask, Poisson event counts — the
building blocks are one call each.

## Host-side streaming

When you need random numbers on the CPU (a quick reference value, a seed) there's
a streaming generator too:

```rust,ignore
let mut rng = quanta_rand::Rng::from_seed(7);
let u = rng.next_f32();          // uniform
let z = rng.next_normal_f32();   // standard normal
rng.jump();                      // skip 2⁶⁴ ahead — independent stream
```

## Why counter-based

Classic streaming RNGs carry mutable state, which is awkward on a GPU where
thousands of threads run at once. Counter-based generators (Philox, Threefry)
compute the *n*-th value directly from `(seed, index)` with no shared state — so
every thread produces its own independent stream, and the whole buffer fills in
parallel with a reproducible result.

## Next

- **[Autodiff basics](autodiff-basics.md)** — now you have data, math, and randomness; time to compute gradients.
- Full surface: the **[quanta-rand README](https://github.com/zelez-lab/quanta/blob/main/crates/quanta-rand/README.md)**.
