# Random numbers

> **You'll learn:** how to generate reproducible random numbers and draw from
> common distributions on the GPU. Builds on [FFT](fft.md).

Randomness shows up everywhere in numerical work — initializing weights, dropout,
Monte-Carlo integration, sampling. `quanta::sci::random` gives you **counter-based**
generators whose `u32`/`f32` output is *bit-exact across every backend* for a
given seed, so a run is reproducible whether it lands on Metal, Vulkan (real
hardware or lavapipe, including the Raspberry Pi's V3D), or the CPU. The
64-bit variants (`u64` fill, `f64` distributions) are the one exception — see
the note below.

```toml
quanta = { version = "0.1", features = ["sci", "metal"] } # or vulkan / software
```

## Filling a buffer

The GPU generators produce a whole buffer in one kernel launch and return it
as a `Vec`. Uniform values in `[0, 1)`:

```rust,ignore
let n = 1024;
let vals = quanta::sci::random::fill_uniform_f32_gpu(&gpu, n, /* seed */ 42)?; // Vec<f32>
```

Same seed → same bytes, every time, on every backend. That determinism is the
whole point: a bug reproduces, a paper's result reproduces.

## Distributions

Beyond uniform, there are six distributions ready to fill:

```rust,ignore
quanta::sci::random::fill_normal_f32_gpu(&gpu, n, seed)?;              // N(0, 1)
quanta::sci::random::fill_exponential_f32_gpu(&gpu, n, seed, 1.0)?;    // Exp(λ=1)
quanta::sci::random::fill_lognormal_f32_gpu(&gpu, n, seed, 0.0, 1.0)?; // exp(N(μ,σ))
quanta::sci::random::fill_bernoulli_u32_gpu(&gpu, n, seed, 0.3)?;      // 1 w.p. 0.3
quanta::sci::random::fill_poisson_u32_gpu(&gpu, n, seed, 4.0)?;        // Poisson(λ=4)
```

Normal weights for a layer, a Bernoulli dropout mask, Poisson event counts — the
building blocks are one call each.

> The Philox core is pure 32-bit arithmetic, so the `u32`/`f32` fill and every
> `f32` distribution run — bit-exact against the CPU oracle — on every
> backend, including devices without 64-bit support (Metal, the Pi's V3D).
> The 64-bit *outputs* are gated on real device capability instead: the `u64`
> fill needs `gpu.supports_i64()` (its `(hi << 32) | lo` pack), and each
> `f64` twin (`fill_normal_f64_gpu`, …) additionally needs
> `gpu.supports_f64()`. On a device without those features — Metal has no
> `double`, V3D has neither `shaderInt64` nor `shaderFloat64` — the call
> returns `NotSupported` rather than crashing the driver or silently
> truncating bits. The CPU backend and llvmpipe run the 64-bit variants
> natively.

## Host-side streaming

When you need random numbers on the CPU (a quick reference value, a seed) there's
a streaming generator too:

```rust,ignore
let mut rng = quanta::sci::random::Rng::from_seed(7);
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
- Full surface: the **[quanta-rand README](https://github.com/zelez-lab/quanta/blob/main/crates/sci/quanta-rand/README.md)**.
