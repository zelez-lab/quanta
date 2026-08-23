//! 1-D FFT micro-benchmarks: the radix-2 sweep, the Bluestein path and
//! `rfft` — device vs the CPU software baseline.
//!
//! The headline number is GFLOP/s under the standard complex-FFT model,
//! `5·N·log2(N)`. Run with a real backend:
//! `cargo bench -p quanta-fft --features gpu-metal --bench fft`. Results land
//! in `PERFORMANCE.md`.
//!
//! Every size is cross-checked against the CPU baseline before it is timed
//! (and the radix-2 sweep also checks the one-shot `fft` against a reused
//! `FftPlan`), so a perf run doubles as a differential correctness check.

use std::hint::black_box;
use std::time::Instant;

/// Radix-2 sweep: 2^10 … 2^20 points.
const POW2_MIN_LOG: u32 = 10;
const POW2_MAX_LOG: u32 = 20;

/// Non-power-of-2 sizes — every one routes through Bluestein at
/// `M = next_pow2(2N−1)`.
const BLUESTEIN: [usize; 5] = [1_000, 3_000, 10_000, 30_000, 100_000];

/// Real-input sizes for the `rfft`-vs-complex comparison.
const REAL: [usize; 5] = [1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20];

/// Sizes at or below this are differentially checked against the CPU
/// baseline; above it a single software transform costs seconds, and the
/// test suite already covers correctness.
const CHECK_MAX: usize = 1 << 14;

/// The standard complex-FFT FLOP model, `5·N·log2(N)`: radix-2 Cooley-Tukey
/// runs `(N/2)·log2(N)` butterflies and a butterfly is one complex
/// multiply-add — 10 flops. It counts the arithmetic the *algorithm* owes,
/// not instructions retired, which is what makes two implementations of the
/// same transform comparable.
fn cx_flops(n: usize) -> f64 {
    5.0 * n as f64 * (n as f64).log2()
}

/// Seconds for the **fastest** of several calls. A shared developer
/// workstation is never idle, so the mean of a few repeats largely measures
/// whatever else is running; the minimum is the closest honest estimate of
/// what the call itself costs. The first run warms the JIT and the pipeline
/// cache and is discarded — unless one call already exceeds the budget, where
/// only a second sample is affordable.
fn time_best(mut f: impl FnMut()) -> f64 {
    const MAX_ITERS: u32 = 20;
    const BUDGET_S: f64 = 1.0;

    let t = Instant::now();
    f();
    let first = t.elapsed().as_secs_f64();
    if first >= BUDGET_S {
        let t = Instant::now();
        f();
        return t.elapsed().as_secs_f64().min(first);
    }

    let start = Instant::now();
    let mut best = f64::INFINITY;
    let mut iters = 0u32;
    loop {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
        iters += 1;
        if iters >= MAX_ITERS || start.elapsed().as_secs_f64() >= BUDGET_S {
            break;
        }
    }
    best
}

/// Deterministic split-complex signal of length `n`.
fn signal(n: usize) -> (Vec<f32>, Vec<f32>) {
    let re = (0..n).map(|i| ((i % 17) as f32) * 0.25 - 2.0).collect();
    let im = (0..n).map(|i| ((i % 13) as f32) * 0.5 - 3.0).collect();
    (re, im)
}

/// Assert two split-complex spectra agree. The bar is the error relative to
/// the reference's **joint** peak magnitude: both parts of one spectrum share
/// a single error scale, so holding a near-zero component to its own peak
/// would be an impossible standard.
fn agree(got: (&[f32], &[f32]), want: (&[f32], &[f32]), what: &str) {
    assert_eq!(
        got.0.len(),
        want.0.len(),
        "{what}: length {} vs {}",
        got.0.len(),
        want.0.len()
    );
    let reference = || want.0.iter().chain(want.1.iter());
    let peak = reference().fold(1.0f32, |m, &y| m.max(y.abs()));
    let worst = got
        .0
        .iter()
        .chain(got.1.iter())
        .zip(reference())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst <= 1e-3 * peak,
        "{what}: max abs error {worst} against peak {peak}"
    );
}

fn main() {
    let Ok(gpu) = quanta::init() else {
        eprintln!("no GPU backend available — skipping quanta-fft benches");
        return;
    };
    let cpu = quanta::init_cpu();

    println!("quanta-fft 1-D: {} vs {}", gpu.name(), cpu.name());
    if gpu.name() == cpu.name() {
        println!(
            "  NOTE: this build has no hardware backend — both columns are the \
             same software device, so the comparison says nothing."
        );
    }
    println!(
        "GFLOP/s = 5·N·log2(N) / time; times are the fastest of several repeats \
         and include the host round trip."
    );

    radix2(&gpu, &cpu);
    bluestein(&gpu, &cpu);
    real_input(&gpu);
}

/// The power-of-2 sweep: one-shot `fft`, a reused `FftPlan`, and the CPU
/// baseline on the same sizes.
fn radix2(gpu: &quanta::Gpu, cpu: &quanta::Gpu) {
    println!("\n-- radix-2 Cooley-Tukey (complex f32)");
    println!(
        "{:>9} | {:>10} {:>9} | {:>10} {:>9} | {:>11} {:>9} | {:>8}",
        "N", "fft ms", "GFLOP/s", "plan ms", "GFLOP/s", "cpu ms", "GFLOP/s", "speedup"
    );

    for log2n in POW2_MIN_LOG..=POW2_MAX_LOG {
        let n = 1usize << log2n;
        let (re, im) = signal(n);
        let flops = cx_flops(n);

        let (dr, di) = quanta_fft::fft(gpu, &re, &im).unwrap();
        if n <= CHECK_MAX {
            let (cr, ci) = quanta_fft::fft(cpu, &re, &im).unwrap();
            agree((&dr, &di), (&cr, &ci), &format!("device vs cpu n={n}"));
        }

        // The one-shot column is timed before any plan for this size exists,
        // so it is the cost of `fft` alone — nothing this bench allocated for
        // the plan column is resident while it runs.
        let one_shot = time_best(|| {
            black_box(quanta_fft::fft(gpu, &re, &im).unwrap());
        });

        let mut plan = quanta_fft::FftPlan::new(gpu, n, false).unwrap();
        let (pr, pi) = plan.execute(&re, &im).unwrap();
        agree((&dr, &di), (&pr, &pi), &format!("fft vs plan n={n}"));
        let planned = time_best(|| {
            black_box(plan.execute(&re, &im).unwrap());
        });
        let baseline = time_best(|| {
            black_box(quanta_fft::fft(cpu, &re, &im).unwrap());
        });

        println!(
            "{:>9} | {:>10.3} {:>9.3} | {:>10.3} {:>9.3} | {:>11.3} {:>9.3} | {:>7.1}x",
            n,
            one_shot * 1e3,
            flops / one_shot / 1e9,
            planned * 1e3,
            flops / planned / 1e9,
            baseline * 1e3,
            flops / baseline / 1e9,
            baseline / planned,
        );
    }
}

/// The chirp-z path. Its cost is three length-`M` transforms plus O(N) chirp
/// work, so the `5·N·log2(N)` model does not describe it — this table is
/// milliseconds only, with `M` spelled out.
fn bluestein(gpu: &quanta::Gpu, cpu: &quanta::Gpu) {
    println!("\n-- Bluestein chirp-z (non-power-of-2 N)");
    println!(
        "{:>9} {:>9} | {:>11} | {:>13} | {:>8}",
        "N", "M", "device ms", "cpu ms", "speedup"
    );

    for &n in &BLUESTEIN {
        let (re, im) = signal(n);
        let m = (2 * n - 1).next_power_of_two();

        let (dr, di) = quanta_fft::fft(gpu, &re, &im).unwrap();
        if n <= CHECK_MAX {
            let (cr, ci) = quanta_fft::fft(cpu, &re, &im).unwrap();
            agree((&dr, &di), (&cr, &ci), &format!("bluestein n={n}"));
        }

        let device = time_best(|| {
            black_box(quanta_fft::fft(gpu, &re, &im).unwrap());
        });
        let baseline = time_best(|| {
            black_box(quanta_fft::fft(cpu, &re, &im).unwrap());
        });

        println!(
            "{:>9} {:>9} | {:>11.3} | {:>13.3} | {:>7.1}x",
            n,
            m,
            device * 1e3,
            baseline * 1e3,
            baseline / device,
        );
    }
}

/// `rfft` against transforming the same real signal as complex-with-zero-imag
/// — the shape the crate docs claim is ~2× slower.
fn real_input(gpu: &quanta::Gpu) {
    println!("\n-- rfft vs the same signal as complex (device only)");
    println!(
        "{:>9} | {:>10} | {:>13} | {:>8}",
        "N", "rfft ms", "complex ms", "speedup"
    );

    for &n in &REAL {
        let real: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.25 - 2.0).collect();
        let zeros = vec![0.0f32; n];

        // The half-spectrum must be the first n/2+1 bins of the complex
        // transform of the same signal.
        let (hr, hi) = quanta_fft::rfft(gpu, &real).unwrap();
        let (cr, ci) = quanta_fft::fft(gpu, &real, &zeros).unwrap();
        agree(
            (&hr, &hi),
            (&cr[..n / 2 + 1], &ci[..n / 2 + 1]),
            &format!("rfft n={n}"),
        );

        let half = time_best(|| {
            black_box(quanta_fft::rfft(gpu, &real).unwrap());
        });
        let full = time_best(|| {
            black_box(quanta_fft::fft(gpu, &real, &zeros).unwrap());
        });

        println!(
            "{:>9} | {:>10.3} | {:>13.3} | {:>7.2}x",
            n,
            half * 1e3,
            full * 1e3,
            full / half,
        );
    }
}
