# quanta-rand

GPU-first random number generation for Quanta kernels: counter-based
RNGs, six probability distributions, host↔GPU bit-exact reproducibility.

**New here?** Start with [GETTING_STARTED.md](GETTING_STARTED.md) — a
10-minute walkthrough from `cargo new` to a working program.

## What you get

- **Three algorithms**, all bit-exact with their published references:
  - `xoshiro128++` (state-based, CPU-side `Rng` API)
  - `Philox4×32-10` (counter-based, BigCrush-clean, in-kernel default)
  - `Threefry4×32-20` (counter-based, no integer multiply)

- **Six distributions** with one-call host-side fill kernels:

  | Distribution | Fill function                    | Output type   | Notes                |
  |--------------|----------------------------------|---------------|----------------------|
  | Uniform u32  | `fill_uniform_u32_gpu`           | `Vec<u32>`    | raw Philox words     |
  | Uniform u64  | `fill_uniform_u64_gpu`           | `Vec<u64>`    | two Philox draws     |
  | Uniform f32  | `fill_uniform_f32_gpu`           | `Vec<f32>`    | `[0, 1)`             |
  | Uniform f64  | `fill_uniform_f64_gpu`           | `Vec<f64>`    | `[0, 1)`             |
  | Normal       | `fill_normal_f32_gpu`            | `Vec<f32>`    | Box-Muller, N(0, 1)  |
  | Exponential  | `fill_exponential_f32_gpu`       | `Vec<f32>`    | inverse-CDF          |
  | LogNormal    | `fill_lognormal_f32_gpu`         | `Vec<f32>`    | `exp(μ + σN)`        |
  | Bernoulli    | `fill_bernoulli_u32_gpu`         | `Vec<u32>`    | 1 with prob p        |
  | Poisson      | `fill_poisson_u32_gpu`           | `Vec<u32>`    | Knuth, λ ≤ ~30       |

- **Determinism**: every call is a pure function of `(seed, len)`.
  Re-running with the same seed produces bit-identical output. Host
  and GPU agree byte-for-byte.

## Install

In `Cargo.toml`:

```toml
[dependencies]
quanta-rand = { path = "../quanta/crates/quanta-rand", features = ["gpu"] }
quanta = { path = "../quanta" }   # for the Gpu handle
```

The `gpu` feature pulls in the Quanta runtime so the `fill_*_gpu`
helpers are available. Without it, only the host-side `Rng` and the
pure-Rust generator modules are exposed — useful as a correctness
oracle for downstream tests.

## Quick start

```rust,no_run
use quanta_rand::{fill_uniform_f32_gpu, fill_normal_f32_gpu};

let gpu = quanta::init()?;                    // real GPU
// let gpu = quanta::init_cpu();              // or software backend for tests
let seed = 0xCAFE_BABE_DEAD_BEEFu64;

let unif: Vec<f32> = fill_uniform_f32_gpu(&gpu, 1_000_000, seed)?;
let norm: Vec<f32> = fill_normal_f32_gpu(&gpu, 1_000_000, seed)?;
# Ok::<(), quanta::QuantaError>(())
```

## CPU-side API

For single-thread CPU work the `Rng` type wraps xoshiro128++:

```rust
use quanta_rand::Rng;

let mut rng = Rng::from_seed(0xC0FFEE);
let x: u32 = rng.next_u32();
let y: u64 = rng.next_u64();
let f: f32 = rng.next_f32();
let d: f64 = rng.next_f64();
let z: f32 = rng.next_normal_f32();   // Box-Muller, N(0, 1)

// Spawn an independent stream by advancing 2^64 steps:
let mut other = rng.clone();
other.jump();
```

## Worked examples

Each is runnable end-to-end:

- **`fill_buffer`** — tour of all 9 fill kernels (one per distribution)
- **`nn_weight_init`** — Glorot / He initialization for an MLP
- **`monte_carlo_pi`** — classic Monte Carlo estimate of π
- **`dropout_mask`** — inverted-dropout training mask
- **`bench_distributions`** — timing sweep over the full surface

Run any of them with:

```sh
cargo run --example <name> -p quanta-rand --features gpu --release
```

## Choosing a distribution

- **Uniform u32 / u64** — when you want raw bits (hashing, partitioning,
  bloom filters, sketching).
- **Uniform f32 / f64** — when you want `[0, 1)`. Most common scalar
  random draw.
- **Normal** — neural network weights, Gaussian noise injection,
  Brownian motion in physics sims.
- **Exponential** — inter-arrival times, Poisson processes, half-life
  decay sims.
- **LogNormal** — financial returns, particle size distributions,
  positive-only quantities with multiplicative noise.
- **Bernoulli** — dropout masks, coin flips, A/B test simulation,
  random sampling decisions.
- **Poisson** — count of events in a fixed interval (rare events,
  packet arrivals, photon counts).

## Reproducibility

`fill_*_gpu(&gpu, len, seed)` is a *pure function of (len, seed)*.
For the parameterised distributions (`fill_normal_f32_gpu`,
`fill_exponential_f32_gpu`, ...) it's a pure function of all the
inputs taken together. Specifically:

- Re-running with the same arguments produces bit-identical output.
- The output for a given `(seed, quark_id)` does **not** depend on
  `len`. So `fill_uniform_u32_gpu(gpu, 100, s)[7]` equals
  `fill_uniform_u32_gpu(gpu, 1_000_000, s)[7]`.
- Host and GPU produce the same bytes. The integration tests
  (`tests/uniform_fill_correctness.rs`) validate this on every
  call signature.

To get **independent** streams from the same workflow, use
different seeds:

```rust,ignore
let layer_a_init = fill_normal_f32_gpu(&gpu, n, seed ^ 0)?;
let layer_b_init = fill_normal_f32_gpu(&gpu, n, seed ^ 1)?;
```

## Using quanta-rand's device fns from your own kernels

If you're writing your own `#[quanta::kernel]` and want to call
quanta-rand's Philox / Threefry primitives directly (instead of
the `fill_*_gpu` host-side wrappers), splice the source in with
one line:

```rust,ignore
quanta::import_devices!(quanta_rand::philox4x32_10_first_u32_kernel);

#[quanta::kernel]
fn my_kernel(d: &MyData) {
    let id = quark_id();
    let r = philox4x32_10_first_u32_kernel(id, 0, 0, 0, d.seed_lo, d.seed_hi);
    d.out[id as usize] = r;
}
```

The `import_devices!` macro expands to per-fn source-injection
macros that `#[quanta::device]` auto-generates on the library
side. The downstream effect is identical to a same-crate device
fn — bare-name calls work, LLVM inlines at -O3.

Multiple imports in one call:

```rust,ignore
quanta::import_devices!(
    quanta_rand::philox4x32_10_first_u32_kernel,
    quanta_rand::threefry4x32_20_first_u32_kernel,
);
```

See `crates/quanta-rand-import-test/` in the Quanta workspace for
a working end-to-end example with bit-exact validation.

## Why counter-based?

State-based RNGs like xoshiro need to walk through a sequence — each
draw depends on the previous. That serializes across threads, which
is exactly the opposite of what a GPU wants. Counter-based
generators are stateless functions `(counter, key) → output`; each
quark computes its own output from `(seed, quark_id)` with zero
coordination, so the whole grid runs in lock-step parallel.

Philox4×32-10 (the in-kernel default) is what cuRAND and rocRAND
both use. It passes the full TestU01 BigCrush statistical battery
and produces ~5× more random bytes per round than xoshiro.

## v0.1 limits, deferred to v0.2

- **f64 normal / exponential / lognormal**: blocked on f64 math
  intrinsics (`sqrt_f64`, `ln_f64`, …) which only exist for f32
  today. Cross-backend emitter work.
- **Large-λ Poisson** (transformed-rejection / PTRD): the current
  Knuth kernel caps at 64 inner iterations — fine for λ ≤ ~30,
  truncates above.
- **GPU-side jump-ahead**: constant-time on CPU; requires a long
  fixed loop on GPU. Pending a use case.
- **Other distributions**: gamma, beta, Dirichlet, geometric,
  categorical, multinomial.

## Validation

68 tests, all green:

- 7 Random123 known-answer vectors for Philox (3 inputs × 2 round
  counts + alias check)
- 10 KAT vectors for Threefry (3 inputs × 3 round counts + alias)
- 8 uniform-conversion unit tests
- 7 CPU `Rng` surface tests
- 18 host↔GPU bit-exact correctness tests across all 9 fill kernels
- 6 Kolmogorov-Smirnov goodness-of-fit checks at n=50,000

Run with:

```sh
cargo test -p quanta-rand --features gpu
```
