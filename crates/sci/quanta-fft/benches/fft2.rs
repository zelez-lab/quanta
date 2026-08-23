//! 2-D FFT micro-benchmark: `fft2` on square power-of-2 grids, device vs the
//! CPU software baseline.
//!
//! The 2-D transform is separable, so its FLOP count is the 1-D model applied
//! to the whole grid: `H` row transforms of length `W` plus `W` column
//! transforms of length `H` is `5·H·W·log2(W) + 5·W·H·log2(H)` =
//! `5·N·log2(N)` with `N = H·W`. Run with a real backend:
//! `cargo bench -p quanta-fft --features gpu-metal --bench fft2`. Results land
//! in `PERFORMANCE.md`.
//!
//! The smallest grid is cross-checked against the CPU baseline before it is
//! timed, so a perf run doubles as a differential correctness check.

use std::hint::black_box;
use std::time::Instant;

/// Square grids, `(height, width)`.
const GRIDS: [(usize, usize); 4] = [(64, 64), (128, 128), (256, 256), (512, 512)];

/// Grids at or below this many points are differentially checked against the
/// CPU baseline; the test suite covers the rest.
const CHECK_MAX: usize = 1 << 12;

/// The standard complex-FFT FLOP model, `5·N·log2(N)`, with `N = H·W` — see
/// the module docs for why separability makes it the right count here.
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

/// Deterministic split-complex grid of `h·w` points.
fn grid(h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let n = h * w;
    let re = (0..n).map(|i| ((i % 17) as f32) * 0.25 - 2.0).collect();
    let im = (0..n).map(|i| ((i % 13) as f32) * 0.5 - 3.0).collect();
    (re, im)
}

/// Assert two split-complex spectra agree, relative to the reference's
/// **joint** peak magnitude — both parts of one spectrum share a single error
/// scale.
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

    println!("quanta-fft 2-D: {} vs {}", gpu.name(), cpu.name());
    if gpu.name() == cpu.name() {
        println!(
            "  NOTE: this build has no hardware backend — both columns are the \
             same software device, so the comparison says nothing."
        );
    }
    println!(
        "GFLOP/s = 5·N·log2(N) / time, N = H·W; times are the fastest of several \
         repeats and include the host round trip."
    );
    println!(
        "{:>11} | {:>11} {:>9} | {:>11} {:>9} | {:>8}",
        "H×W", "device ms", "GFLOP/s", "cpu ms", "GFLOP/s", "speedup"
    );

    for &(h, w) in &GRIDS {
        let n = h * w;
        let (re, im) = grid(h, w);
        let flops = cx_flops(n);

        if n <= CHECK_MAX {
            let (dr, di) = quanta_fft::fft2(&gpu, &re, &im, h, w).unwrap();
            let (cr, ci) = quanta_fft::fft2(&cpu, &re, &im, h, w).unwrap();
            agree((&dr, &di), (&cr, &ci), &format!("fft2 {h}×{w}"));
        }

        let device = time_best(|| {
            black_box(quanta_fft::fft2(&gpu, &re, &im, h, w).unwrap());
        });
        let baseline = time_best(|| {
            black_box(quanta_fft::fft2(&cpu, &re, &im, h, w).unwrap());
        });

        println!(
            "{:>11} | {:>11.3} {:>9.3} | {:>11.3} {:>9.3} | {:>7.2}x",
            format!("{h}×{w}"),
            device * 1e3,
            flops / device / 1e9,
            baseline * 1e3,
            flops / baseline / 1e9,
            baseline / device,
        );
    }
}
