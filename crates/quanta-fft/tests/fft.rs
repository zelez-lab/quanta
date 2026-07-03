//! GPU FFT differential tests: the radix-2 kernel vs the direct-DFT oracle
//! (software lane), plus the inverse round trip.

#![cfg(feature = "gpu")]

use quanta_fft::reference;

fn gpu() -> quanta::Gpu {
    // Pinned to the CPU JIT: this kernel trips a mutable-register scope bug in
    // the MSL emitter on real Metal (register redefinition / use-before-decl).
    // Runs on the software backend until that emitter bug is fixed.
    quanta::init_cpu()
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

#[test]
fn fft_non_power_of_two_errors() {
    let g = gpu();
    let (re, im) = signal(6, 0);
    assert!(quanta_fft::fft(&g, &re, &im).is_err());
}

#[test]
fn fft_length_mismatch_errors() {
    let g = gpu();
    assert!(quanta_fft::fft(&g, &[1.0, 2.0], &[1.0]).is_err());
}
