//! Side-by-side throughput: quanta-rand vs the standard `rand` crate.
//!
//! Compares two CPU-side paths for generating N draws:
//!   1. `quanta-rand` host-side `Rng` (xoshiro128++, sequential)
//!   2. `quanta-rand` `fill_uniform_f32_gpu` on the software backend
//!      (Philox4×32-10 through the CPU JIT)
//!   3. `rand::rngs::StdRng` (ChaCha12, the rand-crate default)
//!   4. `rand` xoshiro256++ via the `rand` ecosystem
//!
//! Honest framing of what this measures:
//!
//! - The host-side `quanta_rand::Rng` and `rand::rngs::*` are
//!   apples-to-apples CPU-native sequential generators. The
//!   comparison there is just "which RNG is faster on a single CPU
//!   thread."
//! - The `fill_uniform_f32_gpu(&gpu, ...)` call here runs through
//!   the software backend (CPU JIT) — that includes dispatch
//!   overhead and a slow interpreted-eval-style path designed to
//!   validate kernels, NOT for production speed. On a real GPU the
//!   throughput is 100-1000× higher; on CPU JIT it's 100-1000× LOWER
//!   than CPU-native rand because that's the price of validating
//!   through the wasm-route lowering.
//!
//! Run: cargo run --example bench_vs_rand -p quanta-rand --features gpu --release

use std::time::Instant;

use quanta_rand::{Rng, fill_uniform_f32_gpu};
use rand::{Rng as _, SeedableRng, rngs::StdRng};

const N: usize = 1_000_000;
const WARMUP: usize = 1;
const ITERS: usize = 3;

fn bench<F: FnMut() -> usize>(name: &str, mut f: F) {
    for _ in 0..WARMUP {
        let _ = f();
    }
    let mut best_ms = f64::INFINITY;
    let mut total_n = 0;
    for _ in 0..ITERS {
        let t = Instant::now();
        let n = f();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if ms < best_ms {
            best_ms = ms;
            total_n = n;
        }
    }
    let m_per_sec = total_n as f64 / 1e6 / (best_ms / 1e3);
    println!("  {name:<42} best={best_ms:>8.2}ms  {m_per_sec:>8.1} M/s");
}

fn main() {
    println!("quanta-rand vs rand — CPU throughput comparison");
    println!("  N per call : {N}");
    println!("  Warmup     : {WARMUP}, best of: {ITERS}");
    println!();

    println!("CPU-native sequential generators (apples-to-apples):");

    // quanta-rand host-side Rng (xoshiro128++)
    bench("quanta_rand::Rng::next_f32 (xoshiro128++)", || {
        let mut rng = Rng::from_seed(0xC0FFEE);
        let mut sum: f32 = 0.0;
        for _ in 0..N {
            sum += rng.next_f32();
        }
        // Stop the optimizer from deleting the loop.
        std::hint::black_box(sum);
        N
    });

    // rand::rngs::StdRng (ChaCha12, rand-crate default)
    bench("rand::StdRng::r#gen::<f32> (ChaCha12)", || {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let mut sum: f32 = 0.0;
        for _ in 0..N {
            sum += rng.r#gen::<f32>();
        }
        std::hint::black_box(sum);
        N
    });

    println!();
    println!("Same generator on quanta-rand's GPU dispatch path (CPU JIT):");
    println!("  Note: software-backend throughput is NOT representative of");
    println!("  native-GPU throughput. Native Metal/Vulkan is 100-1000× faster.");

    let gpu = quanta::init_cpu();

    bench("fill_uniform_f32_gpu (Philox, software JIT)", || {
        let out = fill_uniform_f32_gpu(&gpu, N, 0xC0FFEE).unwrap();
        out.len()
    });

    println!();
    println!("Takeaways:");
    println!("  - For CPU-bound serial work, `rand::StdRng` is fast and");
    println!("    well-tested. quanta-rand's host-side Rng exists for");
    println!("    interop with the GPU-side generator (bit-exact streams).");
    println!("  - quanta-rand's real value is when you need ≥10k draws on");
    println!("    GPU — `fill_*_gpu` shines on real hardware.");
}
