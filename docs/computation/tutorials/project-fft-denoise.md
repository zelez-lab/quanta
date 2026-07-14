# Project: denoise a signal (FFT low-pass)

> **You'll build:** a standalone Cargo project that cleans a noisy signal by
> transforming it to the frequency domain, dropping the low-energy (noise) bins,
> and transforming back — the classic `numpy.fft` low-pass filter.
>
> **You'll need:** the [FFT lesson](fft.md). No autodiff — this is signal
> processing, not learning.

A clean periodic signal concentrates its energy in a few frequency bins; noise
smears across many small ones. So the recipe is: FFT the signal, keep only the
bins whose **magnitude** is large, zero the rest, and inverse-FFT. What comes
back is the signal with the noise removed.

## 1. Create the project

```sh
cargo new fft-denoise
cd fft-denoise
```

## 2. Dependencies

```toml
[dependencies]
quanta = { git = "https://github.com/zelez-lab/quanta", features = ["sci", "metal"] }
```

## 3. A noisy signal

A low-frequency cosine (the signal we want) plus a high-frequency wiggle (the
noise). Quanta's FFT takes **split complex** input — a real part and an
imaginary part as separate slices, both length a power of two. `src/main.rs`:

```rust,ignore
use quanta::sci::fft::{fft, ifft};

fn main() {
    let gpu = quanta::init().expect("a GPU");
    let n = 64usize;
    let tau = std::f32::consts::TAU;

    // clean tone at frequency 2, noise at frequency 20
    let clean: Vec<f32> = (0..n).map(|i| (tau * 2.0 * i as f32 / n as f32).cos()).collect();
    let noisy: Vec<f32> = clean.iter().enumerate()
        .map(|(i, &c)| c + 0.4 * (tau * 20.0 * i as f32 / n as f32).cos())
        .collect();
    let zeros = vec![0.0f32; n];
```

## 4. Transform and filter

FFT the signal, compute each bin's magnitude `√(re² + im²)`, and build a
keep-mask — zero out every bin below half the peak magnitude:

```rust,ignore
    let (re, im) = fft(&gpu, &noisy, &zeros).unwrap();

    let mag: Vec<f32> = re.iter().zip(&im).map(|(r, i)| (r*r + i*i).sqrt()).collect();
    let thr = 0.5 * mag.iter().cloned().fold(0.0, f32::max);

    let (re_f, im_f): (Vec<f32>, Vec<f32>) = re.iter().zip(&im).zip(&mag)
        .map(|((&r, &i), &m)| if m >= thr { (r, i) } else { (0.0, 0.0) })
        .unzip();
```

> On the GPU. When the spectrum is an `Array<f32>`, the same filter is
> `mag = re.mul(&re)?.add(&im.mul(&im)?)?.sqrt()?`, the mask is
> `mag.ge(&thr_broadcast)?`, and applying it is `mask.where_mask(&spectrum,
> &zeros)?`. Here the FFT already hands us host vectors, so we filter them
> directly.

## 5. Invert and check

Inverse-FFT the filtered spectrum; the real part is the denoised signal:

```rust,ignore
    let (recovered, _) = ifft(&gpu, &re_f, &im_f).unwrap();

    let err = recovered.iter().zip(&clean)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("max error vs clean tone: {err:.4}");
}
```

```sh
cargo run --release
```

```text
max error vs clean tone: 0.03
```

The high-frequency noise lived in bins the filter zeroed, so the recovered
signal tracks the clean cosine to a few percent — a working low-pass filter, in
about a dozen lines.

## 6. What you built

A frequency-domain denoiser: `fft → magnitude → threshold mask → ifft`. The FFT
does the heavy lifting; the filter is just a comparison. The same shape covers
high-pass (keep the *high* bins), band-pass (a windowed mask), and compression
(keep the top-k bins by magnitude).

- Coming from NumPy? This is `np.fft.fft` → mask on `np.abs(spectrum)` →
  `np.fft.ifft`. `quanta::sci::fft::fft` is the split-complex equivalent of
  `np.fft.fft` on a real input.
- Sharpen it: taper the mask instead of a hard cutoff (a Butterworth response),
  or keep a fixed number of the largest bins for a compression ratio.
