# quanta-rand cookbook

When to use which distribution. Each recipe is a short paragraph,
the matching API call, and the workload it targets.

> Backend assumption: all snippets use `quanta::init_cpu()` so they
> run on any machine. Replace with `quanta::init()?` once you have a
> real GPU — same kernels, faster.

---

## ML model initialization

**Use case:** initialize neural network weights at the start of
training.

**Schemes:**
- **Glorot / Xavier**: `W ~ N(0, 2 / (fan_in + fan_out))`. Default
  for tanh / sigmoid activations.
- **He / Kaiming**: `W ~ N(0, 2 / fan_in)`. Default for ReLU.

**API:**

```rust,ignore
use quanta_rand::fill_normal_f32_gpu;

let gpu = quanta::init_cpu();
let (fan_in, fan_out) = (784, 256);
let n = fan_in * fan_out;
let sigma = (2.0_f32 / fan_in as f32).sqrt();  // He
let z = fill_normal_f32_gpu(&gpu, n, seed)?;
let weights: Vec<f32> = z.iter().map(|x| x * sigma).collect();
```

See `examples/nn_weight_init.rs` for a runnable version.

---

## Dropout masks

**Use case:** regularize NN training by zeroing a fraction of
activations each forward pass.

**API:**

```rust,ignore
use quanta_rand::fill_bernoulli_u32_gpu;

let keep_prob = 1.0 - p_drop;
let mask = fill_bernoulli_u32_gpu(&gpu, n, step_seed, keep_prob)?;
// Apply: a_out = a * mask / keep_prob  (inverted dropout)
```

Bump `step_seed` each training step (e.g. `seed ^ step_idx`) so
masks differ between steps but are still reproducible.

See `examples/dropout_mask.rs`.

---

## Monte Carlo integration

**Use case:** estimate the value of an integral or geometric
quantity by uniform sampling.

**API:**

```rust,ignore
use quanta_rand::fill_uniform_f32_gpu;

let xs = fill_uniform_f32_gpu(&gpu, n, seed)?;
let ys = fill_uniform_f32_gpu(&gpu, n, seed.wrapping_add(1))?;
let inside = xs.iter().zip(&ys)
    .filter(|(&x, &y)| x*x + y*y < 1.0)
    .count();
let pi_estimate = 4.0 * inside as f64 / n as f64;
```

When you need a 1-σ error around 0.1%, use `n ≈ 1M`. Throughput
scales linearly with `n` — for science-grade precision (~0.01%),
use `n ≈ 100M` on a native GPU.

For full f64 precision use `fill_uniform_f64_gpu` (same shape,
twice as many random bits per sample).

See `examples/monte_carlo_pi.rs`.

---

## Diffusion models / Gaussian noise injection

**Use case:** add scaled Gaussian noise to an image / latent
representation at each diffusion timestep.

**API:**

```rust,ignore
use quanta_rand::fill_normal_f32_gpu;

let noise = fill_normal_f32_gpu(&gpu, h * w * 3, step_seed)?;
let sigma_t = sigma_schedule[step];
// y[i] = sqrt(1 - sigma_t²) * x[i] + sigma_t * noise[i]
```

Each step needs its own seed so the noise patterns differ; a
deterministic seed schedule (`base_seed ^ step_idx`) keeps the
whole generation reproducible.

---

## Langevin dynamics / MD thermostat

**Use case:** add random kicks to particle velocities to maintain
a thermal bath in molecular dynamics or Langevin sampling.

**API:**

```rust,ignore
use quanta_rand::fill_normal_f64_gpu;  // f64 for energy conservation

let n_particles = 100_000;
let kicks = fill_normal_f64_gpu(&gpu, 3 * n_particles, step_seed)?;
let scale = (2.0 * gamma * kbT * dt).sqrt();
// v[i] += sqrt(scale) * kicks[i]  (per dimension)
```

Energy conservation requires f64 — f32 drift over 10⁶ timesteps is
visible.

---

## Sampling from probabilistic policies (RL)

**Use case:** action selection from softmax policy logits.

**API (Bernoulli, binary action):**

```rust,ignore
use quanta_rand::fill_bernoulli_u32_gpu;

let p_action_a = softmax(logits)[0];
let actions = fill_bernoulli_u32_gpu(&gpu, batch_size, seed, p_action_a)?;
// actions[i] == 1 means "take action A", 0 means "take action B"
```

For categorical with N actions, use uniform draws and the inverse-CDF
trick — `fill_uniform_f32_gpu` then host-side bucketize. (A native
categorical kernel is on the v0.2 roadmap.)

---

## Poisson processes / event simulation

**Use case:** simulate arrivals (customers, packets, photon
detections) where events happen independently at a known rate.

**API:**

```rust,ignore
use quanta_rand::fill_exponential_f32_gpu;

let lambda = 10.0;  // 10 events per unit time
let inter_arrivals = fill_exponential_f32_gpu(&gpu, n, seed, lambda)?;
// Cumulative sum gives event times.
```

Or count events per fixed window:

```rust,ignore
use quanta_rand::fill_poisson_u32_gpu;

let counts = fill_poisson_u32_gpu(&gpu, n_windows, seed, lambda * window_len)?;
// counts[i] is the number of events in window i.
```

Poisson kernel caps inner iterations at 64; works for `λ ≲ 30`.
For larger λ, fall back to a normal approximation (`N(λ, λ)`
rounded) or use the host-side CPU API.

---

## Financial Monte Carlo (option pricing, VaR)

**Use case:** simulate asset price paths under geometric Brownian
motion or jump-diffusion.

**API:**

```rust,ignore
use quanta_rand::fill_normal_f64_gpu;

let n_paths = 1_000_000;
let n_steps = 252;  // daily over 1 year
let z = fill_normal_f64_gpu(&gpu, n_paths * n_steps, seed)?;
// S_t+1 = S_t * exp((r - sigma²/2)*dt + sigma*sqrt(dt)*z[i])
```

Use f64 — option prices propagate cumulative log-returns, and f32
drift over 252 steps × 1M paths shows up in deep OTM tails.

---

## Synthetic data generation

**Use case:** generate test inputs matching a known distribution
for benchmarks, robustness tests, or data augmentation.

| Want | Use |
|---|---|
| Integers in `[0, N)` | `fill_uniform_u32_gpu` + modulo |
| Categories with weights | `fill_uniform_f32_gpu` + inverse-CDF lookup |
| Heavy-tailed (e.g. token lengths) | `fill_lognormal_f32_gpu` |
| Sparse boolean masks | `fill_bernoulli_u32_gpu` with small `p` |
| Counts per bucket | `fill_poisson_u32_gpu` |
| Inter-arrival gaps | `fill_exponential_f32_gpu` |

---

## When to use the CPU `Rng` instead

The GPU dispatch overhead per call is in the millisecond range for
the software backend; on native GPUs it's ~10-100µs. If you only
need a handful of draws, use the streaming CPU API:

```rust,no_run
use quanta_rand::Rng;

let mut rng = Rng::from_seed(42);
let x: u32 = rng.next_u32();
let f: f32 = rng.next_f32();
let z: f32 = rng.next_normal_f32();
```

Switch to `fill_*_gpu` when you need ≳ 10k draws per call.

---

## Choosing the right precision

| Workload | f32 OK? | Use f64? |
|---|---|---|
| NN training (forward + back) | ✓ | rare |
| Diffusion / image gen | ✓ | rare |
| Monte Carlo for π / volumes | ✓ until n ≳ 10M | ✓ at higher n |
| Financial MC | ✗ | ✓ |
| Langevin / MD / energy conservation | ✗ | ✓ |
| Lattice QCD / spin systems | ✗ | ✓ |
| Bayesian inference / MCMC | usually ✓ | ✓ for tight likelihoods |

f64 kernels are ~2-3× slower than f32 on most hardware (more bytes
per draw, more bandwidth-bound work). Use the precision your
application's error budget actually needs.

---

## Seeding strategies

- **Reproducible training**: pin one `seed: u64` per training run.
  Derive per-step seeds with `seed ^ step_idx as u64` for time-varying
  noise (dropout masks, diffusion noise).
- **Parallel workers**: each worker gets `seed = master_seed ^ worker_id`.
  Philox streams from different seeds are independent.
- **A/B tests**: pin the seed for both arms so the only difference is
  the algorithm under test.
- **CI determinism**: hardcode the seed; commit expected outputs.
  Same-seed runs are bit-exact across CPU and GPU.

---

## See also

- **[README.md](README.md)** — install + full API reference.
- **[GETTING_STARTED.md](GETTING_STARTED.md)** — 10-minute first-app
  walkthrough.
- **`examples/`** — runnable demos:
  - `fill_buffer` — tour of all fill kernels
  - `nn_weight_init` — Glorot/He init
  - `monte_carlo_pi` — π estimation
  - `dropout_mask` — inverted dropout
  - `bench_distributions` — throughput sweep
