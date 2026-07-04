# FFT

> **You'll learn:** how to take a Fourier transform on the GPU and read the
> spectrum. Builds on [Linear algebra](linear-algebra.md).

The Fast Fourier Transform turns a signal into its frequency components.
`quanta-fft` transforms any length — power-of-2 sizes run the radix-2
Cooley-Tukey kernels, which are **proven equal to the direct DFT** in Lean;
other sizes run Bluestein's chirp-z algorithm on top of the same radix-2
machinery, differential-tested against the direct-DFT oracle.

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

Any length works. A power of 2 is the fastest shape (straight radix-2);
anything else — 1000 samples, a prime bin count — goes through the Bluestein
convolution, which costs about three power-of-2 transforms.

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

## Real signals: rfft

Most signals you'll transform are real — audio, sensor data, image rows. For
those, the negative-frequency half of the spectrum is redundant (bins `k` and
`N−k` are conjugates), and `rfft` exploits that: it takes the real slice
directly and returns just the first `N/2 + 1` bins, computed with a **half-size**
complex FFT under the hood (the packed method — about 2× the throughput and
half the device memory). `irfft` goes back to the real signal:

```rust,ignore
// np.fft.rfft / np.fft.irfft
let x: Vec<f32> = samples();                       // real, N a power of 2
let (hr, hi) = quanta_fft::rfft(&gpu, &x)?;        // N/2 + 1 bins
let back = quanta_fft::irfft(&gpu, &hr, &hi, x.len())?;  // back ≈ x
```

Bin 0 (DC) and bin `N/2` (Nyquist) of a real signal are themselves real —
`hi[0]` and `hi[N/2]` come back as exactly `0.0`.

## Repeated transforms: build a plan

`fft`/`ifft` are one-shot: each call compiles the kernels and runs once.
Transforming many signals of the same size — audio frames, rows of an image —
build an `FftPlan` instead. The plan JIT-compiles the kernels once and
precomputes the twiddle factors into a device buffer (the VkFFT pattern);
every `execute` after that just binds and dispatches:

```rust,ignore
let mut plan = quanta_fft::FftPlan::new(&gpu, 1024, false)?; // forward, N=1024
for frame in frames {
    let (fr, fi) = plan.execute(&frame.re, &frame.im)?;      // no re-JIT
    // ...
}
```

The size and direction are fixed at `new` (inverse plans pass `true` and fold
in the `1/N` scale); `execute` rejects inputs of any other length. A one-shot
`fft()` and a plan `execute` produce identical results. Plans are the radix-2
engine, so they take power-of-2 sizes only — the arbitrary-N support lives in
`fft`/`ifft`, which build the Bluestein convolution out of these same plans.

## Where it fits

The FFT is the bridge to signal processing, spectral methods, and convolution-by-
multiplication. Because it's a plain function over arrays, it drops into the same
pipelines as everything else in this track — generate a signal with
[arrays](arrays-and-broadcasting.md), transform it, reduce the spectrum.

## Next

- **[Random numbers](random-numbers.md)** — reproducible RNG, the last building block before autodiff.
- How-to: **[FFT](../how-to/fft.md)** for the copy-paste version.
