//! Monte-Carlo European option pricing under Black-Scholes — the parity example
//! for a numpy MC pricer. Simulates terminal prices from normal draws and
//! averages the discounted payoff; the result matches the closed-form
//! Black-Scholes price to well under 1%.
//!
//! Every step after the draws is array algebra: `exp`, `maximum`, `mean` and a
//! couple of scalar broadcasts.

use quanta_array::Array;
use quanta_rand::Rng;

/// The device these tests run on: the real GPU under a hardware backend
/// feature (metal / vulkan), else the CPU JIT (portable, no GPU needed).
fn gpu() -> quanta::Gpu {
    #[cfg(any(feature = "metal", feature = "vulkan"))]
    {
        quanta::init().expect("a GPU device")
    }
    #[cfg(not(any(feature = "metal", feature = "vulkan")))]
    {
        quanta::init_cpu()
    }
}

/// A `[n]` array filled with the constant `c` (a scalar broadcast).
fn splat(g: &quanta::Gpu, c: f32, n: usize) -> Array<f32> {
    Array::full(g, c, &[1]).unwrap().broadcast_to(&[n]).unwrap()
}

/// Black-Scholes closed-form call price — the reference.
fn bs_call(s0: f32, k: f32, r: f32, sigma: f32, t: f32) -> f32 {
    fn erf(x: f32) -> f32 {
        // Abramowitz-Stegun 7.1.26
        let t = 1.0 / (1.0 + 0.3275911 * x.abs());
        let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t;
        let y = 1.0 - poly * (-x * x).exp();
        if x >= 0.0 { y } else { -y }
    }
    let n_cdf = |x: f32| 0.5 * (1.0 + erf(x / std::f32::consts::SQRT_2));
    let d1 = ((s0 / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    s0 * n_cdf(d1) - k * (-r * t).exp() * n_cdf(d2)
}

#[test]
fn monte_carlo_call_matches_black_scholes() {
    let g = gpu();
    let (s0, k, r, sigma, t) = (100.0f32, 100.0f32, 0.05f32, 0.2f32, 1.0f32);
    let n = 200_000usize;

    // Z ~ N(0, 1), generated host-side.
    let mut rng = Rng::from_seed(0xC0FFEE);
    let z: Vec<f32> = (0..n).map(|_| rng.next_normal_f32()).collect();
    let zarr = Array::from_slice(&g, &z, &[n]).unwrap();

    // S_T = S0 · exp((r − σ²/2)·T + σ·√T·Z)
    let drift = (r - 0.5 * sigma * sigma) * t;
    let vol = sigma * t.sqrt();
    let exponent = zarr
        .mul(&splat(&g, vol, n))
        .unwrap()
        .add(&splat(&g, drift, n))
        .unwrap();
    let st = exponent.exp().unwrap().mul(&splat(&g, s0, n)).unwrap();

    // payoff = max(S_T − K, 0) ; price = e^{−rT}·mean(payoff)
    let payoff = st
        .sub(&splat(&g, k, n))
        .unwrap()
        .maximum(&Array::<f32>::zeros(&g, &[n]).unwrap())
        .unwrap();
    let mc = (-r * t).exp() * payoff.mean().unwrap();

    let bs = bs_call(s0, k, r, sigma, t);
    let rel = (mc - bs).abs() / bs;
    assert!(
        rel < 0.01,
        "MC price {mc} vs BS {bs} — relative error {:.3}% too high",
        rel * 100.0
    );
}

/// Asian (average-price) call — a *path-dependent* option whose payoff depends
/// on the average price along each path, not just the terminal price. Exercises
/// `cumsum_last` to build each price path from its per-step log-returns. There's
/// no simple closed form, so we sanity-check: the Asian price is positive, and
/// below the European call (averaging dampens the payoff's upside).
#[test]
fn monte_carlo_asian_call_uses_cumsum() {
    let g = gpu();
    let (s0, k, r, sigma, t) = (100.0f32, 100.0f32, 0.05f32, 0.2f32, 1.0f32);
    let (paths, steps) = (20_000usize, 12usize);
    let dt = t / steps as f32;
    let n = paths * steps;

    // per-step log-return increments: (r − σ²/2)·dt + σ·√dt·Z, shape [paths, steps]
    let mut rng = Rng::from_seed(0x5EED);
    let z: Vec<f32> = (0..n).map(|_| rng.next_normal_f32()).collect();
    let zarr = Array::from_slice(&g, &z, &[paths, steps]).unwrap();
    let step_drift = (r - 0.5 * sigma * sigma) * dt;
    let step_vol = sigma * dt.sqrt();
    // scalar broadcasts shaped to the [paths, steps] path grid
    let splat2 = |c: f32| {
        Array::full(&g, c, &[1])
            .unwrap()
            .broadcast_to(&[paths, steps])
            .unwrap()
    };
    let incr = zarr
        .mul(&splat2(step_vol))
        .unwrap()
        .add(&splat2(step_drift))
        .unwrap();

    // cumulative log-price per path, then prices S_{t} = S0·exp(cumlogret)
    let cumlog = incr.cumsum_last().unwrap();
    let prices = cumlog.exp().unwrap().mul(&splat2(s0)).unwrap();

    // average price per path = (sum over the step axis) / steps → [paths]
    let avg = prices
        .sum_axis(1)
        .unwrap()
        .reshape(&[paths])
        .unwrap()
        .mul(&splat(&g, 1.0 / steps as f32, paths))
        .unwrap();

    // payoff = max(avg − K, 0) ; price = e^{−rT}·mean
    let payoff = avg
        .sub(&splat(&g, k, paths))
        .unwrap()
        .maximum(&Array::<f32>::zeros(&g, &[paths]).unwrap())
        .unwrap();
    let asian = (-r * t).exp() * payoff.mean().unwrap();
    let euro = bs_call(s0, k, r, sigma, t);

    assert!(asian > 0.0, "Asian price {asian} must be positive");
    assert!(
        asian < euro,
        "Asian {asian} should be below European {euro} (averaging dampens upside)"
    );
    // sanity band: a 1y ATM Asian call is roughly half to two-thirds the European.
    assert!(
        asian > 0.3 * euro && asian < 0.8 * euro,
        "Asian {asian} out of expected band vs European {euro}"
    );
}
