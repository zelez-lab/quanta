# FFT (Fourier transforms on the GPU)

`quanta-fft` is a GPU FFT — forward and inverse, complex data as split
real/imag `f32` arrays, **any length** — on whatever backend you compiled
for. Power-of-2 sizes run the radix-2 Cooley-Tukey kernels (mechanically
proven equal to the direct DFT — see the
[verification page](../../verification/index.md)); other sizes run
Bluestein's chirp-z algorithm on top of the same radix-2 plans,
differential-tested against the direct-DFT oracle. This page is a
task-by-task recipe.

```toml
[dependencies]
quanta-fft = { version = "0.1", features = ["gpu-metal"] } # gpu-vulkan / gpu
```

## Setup

```rust,ignore
let gpu = quanta::init_cpu();    // real GPU: quanta::init()
```

## Forward transform

Complex input is **split** — a real-part slice and an imag-part slice of equal
length. A real signal just passes zeros for the imaginary part:

```rust,ignore
// np.fft.fft([1,2,3,4])
let re = vec![1.0f32, 2.0, 3.0, 4.0];
let im = vec![0.0f32; 4];                 // real input
let (fr, fi) = quanta_fft::fft(&gpu, &re, &im)?;   // fr/fi: real/imag spectrum
```

Any length works. Powers of 2 go straight to the radix-2 kernels;
non-power-of-2 sizes (e.g. 1000 samples, or a prime bin count) go through
Bluestein's chirp-z convolution at `next_pow2(2N−1)` — about three power-of-2
transforms, so a power-of-2 N is the faster shape when you can choose.

## Inverse transform

`ifft` undoes `fft` (it divides by N), so the round trip recovers the input:

```rust,ignore
// np.fft.ifft(spectrum)
let (rr, ri) = quanta_fft::ifft(&gpu, &fr, &fi)?;
// rr ≈ re, ri ≈ im  (to f32 rounding)
```

## Reading the spectrum

For a real input of length `N`, bin `k` holds the component at frequency
`k/N` cycles per sample; bins `k` and `N−k` are complex conjugates. The DC
(average) component is bin 0:

```rust,ignore
// A pure tone at 1 cycle: cos(2π·j/N) → energy split between bins 1 and N−1.
let n = 8;
let re: Vec<f32> = (0..n)
    .map(|j| (2.0 * std::f32::consts::PI * j as f32 / n as f32).cos())
    .collect();
let im = vec![0.0f32; n];
let (fr, _) = quanta_fft::fft(&gpu, &re, &im)?;
// fr[0] ≈ 0 (no DC); fr[1] ≈ fr[n-1] ≈ N/2.
```

## Real signals: `rfft` / `irfft`

A real signal's spectrum is conjugate-symmetric (`X[N−k] = conj(X[k])`), so
the first `N/2 + 1` bins carry everything. `rfft` takes the real slice
directly — no zero imaginary part to allocate — and returns just those bins;
`irfft` goes back:

```rust,ignore
// np.fft.rfft(x) / np.fft.irfft(spectrum, n)
let x = vec![1.0f32, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, -1.0]; // real, N = 8
let (hr, hi) = quanta_fft::rfft(&gpu, &x)?;               // 5 bins (N/2 + 1)
let back = quanta_fft::irfft(&gpu, &hr, &hi, 8)?;         // back ≈ x
```

Under the hood this is the packed real-FFT: one half-size complex transform
on the device plus an O(N) split pass — about twice the throughput and half
the device memory of `fft(&x, &zeros)`. `hi[0]` and `hi[N/2]` (DC, Nyquist)
are exactly `0.0`. Length must be a power of 2 (`NotSupported` otherwise);
`irfft` checks that the half-spectrum holds exactly `n/2 + 1` bins.

## 2-D transforms (images, grids)

`fft2`/`ifft2` transform a row-major `H×W` grid — both dimensions powers of 2
(others return `NotSupported`), split re/im of length `H·W`. Internally it is
the separable row-column method: a length-`W` FFT of every row, a transpose,
a length-`H` FFT of every (former) column, and a transpose back; each pass
reuses one `FftPlan`.

```rust,ignore
// np.fft.fft2(img)
let (h, w) = (256, 512);                      // powers of 2
let (sr, si) = quanta_fft::fft2(&gpu, &re, &im, h, w)?;  // 2-D spectrum, same layout
// np.fft.ifft2(spectrum) — divides by H·W, so the round trip recovers the input
let (rr, ri) = quanta_fft::ifft2(&gpu, &sr, &si, h, w)?; // rr ≈ re, ri ≈ im
```

Bin `(ky, kx)` holds the component at `ky/H` cycles per row-step and `kx/W`
cycles per column-step; for real input, `(ky, kx)` and `(H−ky, W−kx)` are
conjugates. The reference oracle is `reference::dft2` / `idft2` — the direct
2-D double sum, any sizes.

## Repeated same-size transforms

`fft`/`ifft` build and run a plan per call. For many transforms of one size,
hold an `FftPlan`: kernels are JIT-compiled once and the twiddle table is
precomputed into a device buffer at `new`; `execute` only binds and
dispatches. (`FftPlan` is the radix-2 engine, so it takes power-of-2 sizes
only — Bluestein builds on it internally.)

```rust,ignore
let mut plan = quanta_fft::FftPlan::new(&gpu, 1024, false)?; // inverse: true
for (re, im) in frames {
    let (fr, fi) = plan.execute(&re, &im)?;
}
```

## Checking against the reference

The crate ships a pure-Rust direct DFT (the differential-test oracle) — handy
to sanity-check any result without a GPU in the loop:

```rust,ignore
use quanta_fft::reference;
let (wr, wi) = reference::dft(&re, &im);   // direct O(N²) DFT, any N
let (hr, hi) = reference::rdft(&x);        // direct real DFT, N/2+1 bins
```

## Notes

- **f32, split re/im.** `fft`/`ifft` take **any N** (radix-2 for powers of 2,
  Bluestein chirp-z otherwise; the chirp phases are computed in f64 with the
  `n²` argument reduced mod 2N before the trig, so large non-power-of-2 N —
  e.g. 1000 — stays well inside tolerance). `rfft` and 2-D `fft2` are
  power-of-2 today.
- The reference module is always available (no `gpu` feature); the GPU `fft` /
  `ifft` need a backend feature.
- All backends are equivalent — `init_cpu()` runs the software lane (used by the
  tests); `init()` picks a real GPU.
