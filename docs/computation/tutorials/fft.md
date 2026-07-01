# FFT

> **You'll learn:** how to take a Fourier transform on the GPU and read the
> spectrum. Builds on [Linear algebra](linear-algebra.md).

The Fast Fourier Transform turns a signal into its frequency components. `quanta-fft`
implements the radix-2 Cooley-Tukey FFT — forward and inverse — for power-of-2
sizes, and it's **proven equal to the direct DFT** in Lean.

```toml
quanta-fft = { version = "0.1", features = ["gpu-metal"] } # or gpu-vulkan / gpu
```

## Complex data as split arrays

Quanta represents complex data as **two real `f32` buffers** — a real part and an
imaginary part of equal length. A real-valued signal just passes zeros for the
imaginary part:

```rust,ignore
// np.fft.fft([1,2,3,4])
let re = vec![1.0f32, 2.0, 3.0, 4.0];
let im = vec![0.0f32; 4];                        // real input → zero imaginary
let (fr, fi) = quanta_fft::fft(&gpu, &re, &im)?; // fr, fi: real/imag spectrum
```

The length must be a power of 2; other sizes return `NotSupported`.

## The inverse round-trips

`ifft` undoes `fft` (it divides by N), so transforming and inverting recovers the
input up to floating-point rounding:

```rust,ignore
let (rr, ri) = quanta_fft::ifft(&gpu, &fr, &fi)?;
// rr ≈ re, ri ≈ im
```

## Reading the spectrum

For a real input of length `N`, bin `k` is the component at frequency `k/N`
cycles per sample; bins `k` and `N−k` are complex conjugates, and bin 0 is the DC
(average) component. A pure tone at one cycle puts its energy in bins 1 and N−1:

```rust,ignore
let n = 8;
let re: Vec<f32> = (0..n)
    .map(|j| (2.0 * std::f32::consts::PI * j as f32 / n as f32).cos())
    .collect();
let im = vec![0.0f32; n];
let (fr, _) = quanta_fft::fft(&gpu, &re, &im)?;
// |fr[1]| and |fr[n-1]| carry the tone; the rest are ≈ 0.
```

## Where it fits

The FFT is the bridge to signal processing, spectral methods, and convolution-by-
multiplication. Because it's a plain function over arrays, it drops into the same
pipelines as everything else in this track — generate a signal with
[arrays](arrays-and-broadcasting.md), transform it, reduce the spectrum.

## Next

- **[Random numbers](random-numbers.md)** — reproducible RNG, the last building block before autodiff.
- How-to: **[FFT](../how-to/fft.md)** for the copy-paste version.
