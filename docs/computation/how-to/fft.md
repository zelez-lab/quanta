# FFT (Fourier transforms on the GPU)

`quanta-fft` is a radix-2 Cooley-Tukey FFT — forward and inverse, complex data
as split real/imag `f32` arrays, sizes a power of 2 — on whatever backend you
compiled for. Cooley-Tukey is mechanically proven equal to the direct DFT (see
the [verification page](../../verification/index.md)). This page is a task-by-task
recipe.

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

Length must be a power of 2 — other sizes return `NotSupported`.

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

## Repeated same-size transforms

`fft`/`ifft` build and run a plan per call. For many transforms of one size,
hold an `FftPlan`: kernels are JIT-compiled once and the twiddle table is
precomputed into a device buffer at `new`; `execute` only binds and
dispatches:

```rust,ignore
let mut plan = quanta_fft::FftPlan::new(&gpu, 1024, false)?; // inverse: true
for (re, im) in frames {
    let (fr, fi) = plan.execute(&re, &im)?;
}
```

## Checking against the reference

The crate ships a pure-Rust direct DFT (the differential-test oracle) — handy
to sanity-check a result or to transform a non-power-of-2 size the GPU path
doesn't take yet:

```rust,ignore
use quanta_fft::reference;
let (wr, wi) = reference::dft(&re, &im);   // direct O(N²) DFT, any N
```

## Notes

- **f32, split re/im, power-of-2** today. Mixed-radix / arbitrary-N and a
  real-input `rfft` are later increments.
- The reference module is always available (no `gpu` feature); the GPU `fft` /
  `ifft` need a backend feature.
- All backends are equivalent — `init_cpu()` runs the software lane (used by the
  tests); `init()` picks a real GPU.
