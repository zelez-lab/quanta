//! GPU FFT differential tests: the radix-2 kernel and the Bluestein chirp-z
//! path vs the direct-DFT oracle (software lane), plus the inverse round
//! trips.

#![cfg(feature = "gpu")]

use quanta_fft::reference;

/// The device these tests run on: the real GPU under a hardware backend
/// feature (gpu-metal / gpu-vulkan), else the CPU JIT (portable, no GPU needed).
fn gpu() -> quanta::Gpu {
    #[cfg(any(feature = "gpu-metal", feature = "gpu-vulkan"))]
    {
        quanta::init().expect("a GPU device")
    }
    #[cfg(not(any(feature = "gpu-metal", feature = "gpu-vulkan")))]
    {
        quanta::init_cpu()
    }
}

/// Deterministic complex signal of length `n`.
fn signal(n: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
    let re: Vec<f32> = (0..n)
        .map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 17) as f32 - 8.0)
        .collect();
    let im: Vec<f32> = (0..n)
        .map(|i| (((i as u32).wrapping_mul(40503) ^ seed.wrapping_add(7)) % 13) as f32 - 6.0)
        .collect();
    (re, im)
}

fn close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: len {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= 1e-2 * (1.0 + y.abs()),
            "{what}: [{i}] {x} vs {y}"
        );
    }
}

/// GPU forward FFT must match the direct DFT for every power-of-2 size.
#[test]
fn fft_matches_dft() {
    let g = gpu();
    for &n in &[1usize, 2, 4, 8, 16, 32, 64, 256] {
        let (re, im) = signal(n, n as u32);
        let (gr, gi) = quanta_fft::fft(&g, &re, &im).unwrap();
        let (wr, wi) = reference::dft(&re, &im);
        close(&gr, &wr, &format!("fft re n={n}"));
        close(&gi, &wi, &format!("fft im n={n}"));
    }
}

/// ifft(fft(x)) == x.
#[test]
fn fft_round_trip() {
    let g = gpu();
    for &n in &[2usize, 8, 64, 128] {
        let (re, im) = signal(n, 1234 + n as u32);
        let (fr, fi) = quanta_fft::fft(&g, &re, &im).unwrap();
        let (rr, ri) = quanta_fft::ifft(&g, &fr, &fi).unwrap();
        close(&rr, &re, &format!("round-trip re n={n}"));
        close(&ri, &im, &format!("round-trip im n={n}"));
    }
}

/// Inverse FFT must match the direct inverse DFT.
#[test]
fn ifft_matches_idft() {
    let g = gpu();
    let n = 32;
    let (re, im) = signal(n, 99);
    let (gr, gi) = quanta_fft::ifft(&g, &re, &im).unwrap();
    let (wr, wi) = reference::idft(&re, &im);
    close(&gr, &wr, "ifft re");
    close(&gi, &wi, "ifft im");
}

// === Bluestein (non-power-of-2 sizes) ===

/// Every non-power-of-2 size the sweep exercises: small primes, composites,
/// a large prime, and a large composite. N = 127 and N = 1000 are the phase
/// canaries — a chirp built from an unreduced n² phase drifts visibly there,
/// so passing at crate tolerance demonstrates the mod-2N reduction holds.
const NON_POW2_SIZES: &[usize] = &[3, 5, 7, 11, 13, 6, 9, 10, 12, 15, 100, 127, 1000];

/// The Bluestein forward path must match the direct DFT for every
/// non-power-of-2 size in the sweep, at the same tolerance as the radix-2
/// differential test.
#[test]
fn bluestein_fft_matches_dft() {
    let g = gpu();
    for &n in NON_POW2_SIZES {
        let (re, im) = signal(n, n as u32);
        let (gr, gi) = quanta_fft::fft(&g, &re, &im).unwrap();
        let (wr, wi) = reference::dft(&re, &im);
        close(&gr, &wr, &format!("bluestein fft re n={n}"));
        close(&gi, &wi, &format!("bluestein fft im n={n}"));
    }
}

/// The Bluestein inverse path must match the direct inverse DFT.
#[test]
fn bluestein_ifft_matches_idft() {
    let g = gpu();
    for &n in &[7usize, 12, 100, 1000] {
        let (re, im) = signal(n, 31 + n as u32);
        let (gr, gi) = quanta_fft::ifft(&g, &re, &im).unwrap();
        let (wr, wi) = reference::idft(&re, &im);
        close(&gr, &wr, &format!("bluestein ifft re n={n}"));
        close(&gi, &wi, &format!("bluestein ifft im n={n}"));
    }
}

/// ifft(fft(x)) == x through the chirp-z path, for every sweep size.
#[test]
fn bluestein_round_trip() {
    let g = gpu();
    for &n in NON_POW2_SIZES {
        let (re, im) = signal(n, 4321 + n as u32);
        let (fr, fi) = quanta_fft::fft(&g, &re, &im).unwrap();
        let (rr, ri) = quanta_fft::ifft(&g, &fr, &fi).unwrap();
        close(&rr, &re, &format!("bluestein round-trip re n={n}"));
        close(&ri, &im, &format!("bluestein round-trip im n={n}"));
    }
}

#[test]
fn fft_length_mismatch_errors() {
    let g = gpu();
    assert!(quanta_fft::fft(&g, &[1.0, 2.0], &[1.0]).is_err());
}

// === FftPlan (plan-based dispatch) ===

/// A reused plan is deterministic: two executes on the same input are
/// bit-for-bit identical (the cache serves the same compiled kernels and the
/// same twiddle table both times).
#[test]
fn plan_reuse_is_deterministic() {
    let g = gpu();
    for &n in &[8usize, 64, 256] {
        let (re, im) = signal(n, 42 + n as u32);
        let mut plan = quanta_fft::FftPlan::new(&g, n, false).unwrap();
        let (r1, i1) = plan.execute(&re, &im).unwrap();
        let (r2, i2) = plan.execute(&re, &im).unwrap();
        assert_eq!(r1, r2, "plan re-execute re n={n}");
        assert_eq!(i1, i2, "plan re-execute im n={n}");
    }
}

/// A plan's output matches the one-shot `fft()` bit-for-bit (the one-shot IS
/// a single-use plan), and the same for `ifft()`.
#[test]
fn plan_matches_one_shot_bit_for_bit() {
    let g = gpu();
    for &n in &[4usize, 32, 128] {
        let (re, im) = signal(n, 7 + n as u32);

        let mut fwd = quanta_fft::FftPlan::new(&g, n, false).unwrap();
        let (pr, pi) = fwd.execute(&re, &im).unwrap();
        let (or, oi) = quanta_fft::fft(&g, &re, &im).unwrap();
        assert_eq!(pr, or, "plan vs fft() re n={n}");
        assert_eq!(pi, oi, "plan vs fft() im n={n}");

        let mut inv = quanta_fft::FftPlan::new(&g, n, true).unwrap();
        let (pr, pi) = inv.execute(&re, &im).unwrap();
        let (or, oi) = quanta_fft::ifft(&g, &re, &im).unwrap();
        assert_eq!(pr, or, "plan vs ifft() re n={n}");
        assert_eq!(pi, oi, "plan vs ifft() im n={n}");
    }
}

/// The cache engages: one plan serves many different inputs, each still
/// matching the DFT oracle at the standard tolerance.
#[test]
fn plan_reuse_across_inputs_matches_dft() {
    let g = gpu();
    let n = 64usize;
    let mut plan = quanta_fft::FftPlan::new(&g, n, false).unwrap();
    for seed in [1u32, 2, 3, 500] {
        let (re, im) = signal(n, seed);
        let (gr, gi) = plan.execute(&re, &im).unwrap();
        let (wr, wi) = reference::dft(&re, &im);
        close(&gr, &wr, &format!("plan fft re seed={seed}"));
        close(&gi, &wi, &format!("plan fft im seed={seed}"));
    }
}

/// Round trip through two reused plans: inv.execute(fwd.execute(x)) == x.
#[test]
fn plan_round_trip() {
    let g = gpu();
    let n = 128usize;
    let mut fwd = quanta_fft::FftPlan::new(&g, n, false).unwrap();
    let mut inv = quanta_fft::FftPlan::new(&g, n, true).unwrap();
    for seed in [10u32, 20] {
        let (re, im) = signal(n, seed);
        let (fr, fi) = fwd.execute(&re, &im).unwrap();
        let (rr, ri) = inv.execute(&fr, &fi).unwrap();
        close(&rr, &re, &format!("plan round-trip re seed={seed}"));
        close(&ri, &im, &format!("plan round-trip im seed={seed}"));
    }
}

/// Structural check: the twiddle table is precomputed on the device —
/// `tw[k] = exp(sign·2πi·k/n)` for `k < n/2`, before any execute.
#[test]
fn plan_precomputes_twiddles() {
    let g = gpu();
    let n = 16usize;
    let plan = quanta_fft::FftPlan::new(&g, n, false).unwrap();
    let (twr, twi) = plan.twiddles().unwrap();
    assert_eq!(twr.len(), n / 2);
    assert_eq!(twi.len(), n / 2);
    for k in 0..n / 2 {
        let theta = -2.0 * std::f32::consts::PI * k as f32 / n as f32;
        assert!((twr[k] - theta.cos()).abs() <= 1e-6, "tw_re[{k}]");
        assert!((twi[k] - theta.sin()).abs() <= 1e-6, "tw_im[{k}]");
    }
}

/// Plan construction rejects non-power-of-2 sizes; execute rejects inputs of
/// the wrong length.
#[test]
fn plan_size_errors() {
    let g = gpu();
    assert!(quanta_fft::FftPlan::new(&g, 0, false).is_err());
    assert!(quanta_fft::FftPlan::new(&g, 6, false).is_err());
    let mut plan = quanta_fft::FftPlan::new(&g, 8, false).unwrap();
    assert!(plan.execute(&[0.0; 4], &[0.0; 4]).is_err()); // wrong length
    assert!(plan.execute(&[0.0; 8], &[0.0; 4]).is_err()); // re/im mismatch
    assert_eq!(plan.len(), 8);
    assert!(!plan.inverse());
}
