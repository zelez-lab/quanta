//! Monte Carlo estimation of π — the canonical RNG demo.
//!
//! Draw N points uniformly in the unit square `[0, 1)²` and count
//! how many land inside the quarter unit circle (`x² + y² < 1`).
//! The fraction inside, times 4, approximates π.
//!
//! Standard error of π̂ after N draws is `4 * sqrt(p(1-p)/N)`
//! where p = π/4, which works out to `~1.64 / sqrt(N)`. At N=100k
//! expect ±0.005-ish; at N=1M expect ±0.0016 ish.
//!
//! Run: cargo run --example monte_carlo_pi -p quanta-rand --features gpu --release

use quanta_rand::fill_uniform_f32_gpu;

fn main() {
    let gpu = quanta::init_cpu();
    // 100k samples is enough to demonstrate the technique on the
    // CPU-JIT backend. On a real GPU, raise this to 100M+; throughput
    // scales linearly with sample count.
    let n: usize = 100_000;
    let seed = 0xC0FFEEu64;

    // Draw 2N uniforms — first half is x-coordinates, second half y.
    // The Philox stream is deterministic, so using the same seed
    // with twice the length deterministically produces 2N values.
    let xs = fill_uniform_f32_gpu(&gpu, n, seed).expect("draw xs");
    let ys = fill_uniform_f32_gpu(&gpu, n, seed.wrapping_add(1)).expect("draw ys");

    let inside = xs
        .iter()
        .zip(ys.iter())
        .filter(|&(&x, &y)| x * x + y * y < 1.0)
        .count();

    let pi_est = 4.0 * (inside as f64) / (n as f64);
    let err = (pi_est - std::f64::consts::PI).abs();
    // Stderr of π̂ = 4 * sqrt(p(1-p)/N) with p = π/4 ≈ 1.6418 / sqrt(N).
    let stderr = 1.6418 / (n as f64).sqrt();

    println!("Monte Carlo π estimation");
    println!("  samples : {n}");
    println!("  inside  : {inside}");
    println!("  π̂      : {pi_est:.6}");
    println!("  actual π: {:.6}", std::f64::consts::PI);
    println!("  |error| : {err:.6} (theoretical stderr ≈ {stderr:.6})");

    // 4σ tolerance — for a correctly-distributed estimator, getting
    // farther than 4σ happens roughly 1 in 16,000 times. With a
    // fixed seed and known-good generator this is comfortably safe.
    assert!(
        err < 4.0 * stderr,
        "estimate {pi_est} is more than 4σ from π — generator may be broken"
    );
}
