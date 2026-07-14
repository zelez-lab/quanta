# Project: price an option (Monte-Carlo)

> **You'll build:** a standalone Cargo project that prices a European call option
> by Monte-Carlo simulation — draw thousands of random price outcomes, average
> the payoff, and check it against the closed-form Black-Scholes price. The same
> `numpy` MC pricer, on the GPU.
>
> **You'll need:** [arrays](arrays.md) and [random numbers](random-numbers.md).
> No autodiff — this is simulation, not learning.

Under the Black-Scholes model, a stock's price at expiry is
`S_T = S₀·exp((r − σ²/2)T + σ√T·Z)` for a standard normal `Z`. A European call
pays `max(S_T − K, 0)`; its fair price is the discounted average payoff over
many random `Z`. That's the whole simulation: draw `Z`, transform, average.

## 1. Create the project

```sh
cargo new mc-option
cd mc-option
```

## 2. Dependencies

```toml
[dependencies]
quanta = { git = "https://github.com/zelez-lab/quanta", features = ["sci", "metal"] }
```

## 3. Draw the randomness

Parameters for an at-the-money one-year option, and a million standard-normal
draws straight onto the GPU. `src/main.rs`:

```rust,ignore
use quanta::sci::Array;

fn splat(g: &quanta::Gpu, c: f32, n: usize) -> Array<f32> {
    Array::full(g, c, &[1]).unwrap().broadcast_to(&[n]).unwrap()
}

fn main() {
    let gpu = quanta::init().expect("a GPU");
    let (s0, k, r, sigma, t) = (100.0f32, 100.0f32, 0.05f32, 0.2f32, 1.0f32);
    let n = 1_000_000usize;

    // Z ~ N(0, 1)
    let z = quanta::sci::random::fill_normal_f32_gpu(&gpu, n, 0xBEEF).unwrap();
    let zarr = Array::from_slice(&gpu, &z, &[n]).unwrap();
```

## 4. Simulate and price

Transform every draw into a terminal price, take the call payoff, and discount
the average. Each line is one array op:

```rust,ignore
    // S_T = S0 · exp((r − σ²/2)T + σ√T·Z)
    let drift = (r - 0.5 * sigma * sigma) * t;
    let vol = sigma * t.sqrt();
    let st = zarr
        .mul(&splat(&gpu, vol, n)).unwrap()
        .add(&splat(&gpu, drift, n)).unwrap()
        .exp().unwrap()
        .mul(&splat(&gpu, s0, n)).unwrap();

    // payoff = max(S_T − K, 0) ; price = e^{−rT} · mean(payoff)
    let payoff = st
        .sub(&splat(&gpu, k, n)).unwrap()
        .maximum(&Array::<f32>::zeros(&gpu, &[n]).unwrap()).unwrap();
    let price = (-r * t).exp() * payoff.mean().unwrap();

    println!("MC call price: {price:.4}");   // ≈ 10.45 (Black-Scholes)
}
```

```sh
cargo run --release
```

```text
MC call price: 10.44
```

Black-Scholes gives 10.45 for these parameters — the million-path simulation
lands within a few cents, and the error shrinks as `1/√n`.

## 5. Path-dependent options: a cumulative sum

A European option only cares about the *terminal* price, so one draw per path
suffices. An **Asian** option pays on the *average* price along the path, which
means simulating the whole trajectory. Build each path from per-step log-returns
with a **prefix sum** (`cumsum_last`) — one row per path, one column per time
step:

```rust,ignore
    let (paths, steps) = (20_000usize, 12usize);
    let dt = t / steps as f32;

    // per-step log-returns [paths, steps], then the cumulative log-price
    let incr = /* (r − σ²/2)·dt + σ·√dt·Z, shaped [paths, steps] */;
    let prices = incr.cumsum_last().unwrap()   // running log-price per path
        .exp().unwrap()
        .mul(&s0_grid).unwrap();               // S at each step

    // average price per path, then the usual discounted payoff
    let avg = prices.sum_axis(1).unwrap().reshape(&[paths]).unwrap()
        .mul(&splat(&gpu, 1.0 / steps as f32, paths)).unwrap();
    // payoff = max(avg − K, 0) ; discount and mean as before
```

`cumsum_last` turns a grid of increments into a grid of running totals in one
pass — the primitive behind any path average, running maximum window, or
empirical CDF.

## 6. What you built

A Monte-Carlo pricer: draw, transform, reduce. The European call is pure
elementwise math plus a `mean`; the Asian call adds a `cumsum_last` to walk each
price path. The same shape prices any payoff you can write as a function of the
simulated prices.

- Coming from NumPy? This is `np.random.randn(n)` → vectorized payoff →
  `.mean()`. `fill_normal_f32_gpu` is `np.random.randn` on the device.
- Go further: antithetic variates or a control variate to cut the variance, or
  a barrier option (a running-max check along the path, another `cumsum`-shaped
  scan).
