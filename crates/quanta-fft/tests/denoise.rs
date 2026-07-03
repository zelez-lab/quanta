//! FFT signal denoising — the parity example for a `numpy.fft` low-pass filter.
//!
//! Transform a noisy signal, zero every frequency bin whose magnitude is below
//! a threshold, then invert. The clean low-frequency tone survives; the
//! high-frequency noise, spread thin across many small-magnitude bins, is
//! filtered out. Magnitude is `√(re² + im²)`; the keep-mask is a threshold
//! compare — both also available as GPU array ops (`sqrt`/`ge`/`where_mask`),
//! here done host-side over the FFT's split-complex output.

#![cfg(feature = "gpu")]

use quanta_fft::{fft, ifft};

fn gpu() -> quanta::Gpu {
    // Pinned to the CPU JIT: this kernel trips a mutable-register scope bug in
    // the MSL emitter on real Metal (register redefinition / use-before-decl).
    // Runs on the software backend until that emitter bug is fixed.
    quanta::init_cpu()
}

#[test]
fn fft_lowpass_recovers_the_clean_tone() {
    let g = gpu();
    let n = 64usize;

    // Clean signal: a single low-frequency cosine.
    let clean: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 2.0 * i as f32 / n as f32).cos())
        .collect();

    // Add a high-frequency wiggle as "noise".
    let noisy: Vec<f32> = clean
        .iter()
        .enumerate()
        .map(|(i, &c)| c + 0.4 * (2.0 * std::f32::consts::PI * 20.0 * i as f32 / n as f32).cos())
        .collect();

    let im0 = vec![0.0f32; n];
    let (re, im) = fft(&g, &noisy, &im0).unwrap();

    // Per-bin magnitude, and a keep-mask: drop everything below a threshold.
    let mag: Vec<f32> = re
        .iter()
        .zip(&im)
        .map(|(r, i)| (r * r + i * i).sqrt())
        .collect();
    let peak = mag.iter().cloned().fold(0.0f32, f32::max);
    let thr = 0.5 * peak;
    let (re_f, im_f): (Vec<f32>, Vec<f32>) = re
        .iter()
        .zip(&im)
        .zip(&mag)
        .map(|((&r, &i), &m)| if m >= thr { (r, i) } else { (0.0, 0.0) })
        .unzip();

    // Invert; the recovered real part should track the clean tone.
    let (rec, _) = ifft(&g, &re_f, &im_f).unwrap();

    let err: f32 = rec
        .iter()
        .zip(&clean)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(err < 0.1, "denoised signal max error {err} too high");

    // And it's genuinely closer to clean than the noisy input was.
    let noisy_err: f32 = noisy
        .iter()
        .zip(&clean)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        err < noisy_err * 0.5,
        "denoise didn't help: {err} vs {noisy_err}"
    );
}
