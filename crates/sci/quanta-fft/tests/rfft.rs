//! Real-input FFT differential tests: rfft vs the direct real-DFT oracle and
//! vs the full complex FFT's first N/2+1 bins, the irfft round trip, and the
//! conjugate-symmetry reconstruction.

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

/// Deterministic real signal of length `n`.
fn signal(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 17) as f32 - 8.0)
        .collect()
}

/// The crate's differential tolerance (same as the complex-FFT tests). The
/// packed rfft reorders f32 summations relative to both the oracle and the
/// full complex FFT, so comparisons are within tolerance, not bit-for-bit.
fn close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: len {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= 1e-2 * (1.0 + y.abs()),
            "{what}: [{i}] {x} vs {y}"
        );
    }
}

/// rfft must match the direct real-DFT oracle for every power-of-2 size.
#[test]
fn rfft_matches_rdft() {
    let g = gpu();
    for &n in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256] {
        let x = signal(n, n as u32);
        let (gr, gi) = quanta_fft::rfft(&g, &x).unwrap();
        let (wr, wi) = reference::rdft(&x);
        assert_eq!(gr.len(), n / 2 + 1, "rfft bin count n={n}");
        close(&gr, &wr, &format!("rfft re n={n}"));
        close(&gi, &wi, &format!("rfft im n={n}"));
    }
}

/// rfft must match the first N/2+1 bins of the full complex fft([x, zeros]).
#[test]
fn rfft_matches_full_fft_half() {
    let g = gpu();
    for &n in &[2usize, 4, 8, 16, 64, 256] {
        let x = signal(n, 77 + n as u32);
        let zeros = vec![0.0f32; n];
        let (fr, fi) = quanta_fft::fft(&g, &x, &zeros).unwrap();
        let (hr, hi) = quanta_fft::rfft(&g, &x).unwrap();
        close(&hr, &fr[..n / 2 + 1], &format!("rfft vs fft re n={n}"));
        close(&hi, &fi[..n / 2 + 1], &format!("rfft vs fft im n={n}"));
    }
}

/// irfft(rfft(x), N) ≈ x for every power-of-2 size.
#[test]
fn rfft_round_trip() {
    let g = gpu();
    for &n in &[1usize, 2, 4, 8, 16, 64, 128, 256] {
        let x = signal(n, 1234 + n as u32);
        let (hr, hi) = quanta_fft::rfft(&g, &x).unwrap();
        let back = quanta_fft::irfft(&g, &hr, &hi, n).unwrap();
        close(&back, &x, &format!("rfft round-trip n={n}"));
    }
}

/// irfft must match the direct inverse real-DFT oracle on a valid
/// half-spectrum (one produced by the oracle itself).
#[test]
fn irfft_matches_irdft() {
    let g = gpu();
    for &n in &[2usize, 8, 32, 128] {
        let x = signal(n, 99 + n as u32);
        let (hr, hi) = reference::rdft(&x);
        let gpu_x = quanta_fft::irfft(&g, &hr, &hi, n).unwrap();
        let ref_x = reference::irdft(&hr, &hi, n);
        close(&gpu_x, &ref_x, &format!("irfft vs irdft n={n}"));
    }
}

/// Conjugate-symmetry sanity: the full spectrum reconstructed from the half
/// (X[n−k] = conj(X[k])) must match fft([x, zeros]) across all N bins, and
/// the DC/Nyquist bins must be exactly real.
#[test]
fn rfft_conjugate_symmetry_reconstruction() {
    let g = gpu();
    for &n in &[2usize, 8, 16, 64, 256] {
        let x = signal(n, 5 + n as u32);
        let (hr, hi) = quanta_fft::rfft(&g, &x).unwrap();
        assert_eq!(hi[0], 0.0, "DC bin real n={n}");
        assert_eq!(hi[n / 2], 0.0, "Nyquist bin real n={n}");

        let mut full_re = vec![0.0f32; n];
        let mut full_im = vec![0.0f32; n];
        full_re[..hr.len()].copy_from_slice(&hr);
        full_im[..hi.len()].copy_from_slice(&hi);
        for k in 1..n / 2 {
            full_re[n - k] = hr[k];
            full_im[n - k] = -hi[k];
        }

        let zeros = vec![0.0f32; n];
        let (fr, fi) = quanta_fft::fft(&g, &x, &zeros).unwrap();
        close(&full_re, &fr, &format!("symmetry re n={n}"));
        close(&full_im, &fi, &format!("symmetry im n={n}"));
    }
}

#[test]
fn rfft_empty_is_empty() {
    let g = gpu();
    let (r, i) = quanta_fft::rfft(&g, &[]).unwrap();
    assert!(r.is_empty() && i.is_empty());
    assert!(quanta_fft::irfft(&g, &[], &[], 0).unwrap().is_empty());
}

#[test]
fn rfft_non_power_of_two_errors() {
    let g = gpu();
    assert!(quanta_fft::rfft(&g, &signal(6, 0)).is_err());
    assert!(quanta_fft::irfft(&g, &[0.0; 4], &[0.0; 4], 6).is_err());
}

#[test]
fn irfft_length_errors() {
    let g = gpu();
    // re/im mismatch.
    assert!(quanta_fft::irfft(&g, &[0.0; 5], &[0.0; 4], 8).is_err());
    // Wrong half-spectrum length for n (needs n/2 + 1 = 5).
    assert!(quanta_fft::irfft(&g, &[0.0; 4], &[0.0; 4], 8).is_err());
    assert!(quanta_fft::irfft(&g, &[0.0; 8], &[0.0; 8], 8).is_err());
    // Non-empty spectrum for n = 0.
    assert!(quanta_fft::irfft(&g, &[0.0], &[0.0], 0).is_err());
}
