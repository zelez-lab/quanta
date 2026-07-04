# quanta-fft

GPU fast Fourier transform for Quanta. Radix-2 Cooley-Tukey, forward and
inverse, complex data split into real/imag `f32` arrays, sizes a power of 2.
One kernel, every backend (Metal / Vulkan / CPU).

The headline claim: **Cooley-Tukey correctness is mechanically proven** — the
radix-2 decomposition equals the direct DFT, end to end, in Lean
(`specs/verify/lean/Quanta/Fft/`), on top of being differential-tested against
a direct-DFT oracle and validated on real Metal.

## Status — radix-2 (split re/im, power-of-2, forward + inverse)

| op | signature | notes |
|----|-----------|-------|
| `fft`  | `fft(gpu, &re, &im) -> (Vec<f32>, Vec<f32>)`  | forward DFT, N a power of 2 (one-shot plan) |
| `ifft` | `ifft(gpu, &re, &im) -> (Vec<f32>, Vec<f32>)` | inverse (÷N); `ifft(fft(x)) == x` |
| `rfft` | `rfft(gpu, &x) -> (Vec<f32>, Vec<f32>)` | real input → the `N/2 + 1` half-spectrum (packed method: one half-size complex FFT + O(N) split — ~2× the throughput, half the device memory) |
| `irfft` | `irfft(gpu, &re, &im, n) -> Vec<f32>` | half-spectrum → real signal; `irfft(rfft(x), N) ≈ x` |
| `FftPlan` | `FftPlan::new(gpu, n, inverse)` → `plan.execute(&re, &im)` | plan-based dispatch (VkFFT pattern): kernels JIT-compiled once, twiddle table precomputed into a device buffer, reusable across executes |
| `reference::dft` / `idft` | `dft(&re, &im) -> (Vec<f32>, Vec<f32>)` | pure-Rust direct O(N²) DFT — the oracle (always available, no `gpu` feature) |
| `reference::rdft` / `irdft` | `rdft(&x) -> (Vec<f32>, Vec<f32>)` | direct real DFT (half-spectrum) + inverse — the `rfft`/`irfft` oracle |

Complex data is **split**: a real-part slice and an imag-part slice of equal
length. The GPU transform runs a bit-reversal kernel (in-kernel `log₂N`-bit
reversal) then `log₂N` butterfly stages — `N/2` threads each, twiddles loaded
from a precomputed table `tw[k] = exp(sign·2πi·k/N)` (`k < N/2`; stage `m`
reads `tw[j·N/m]`), in place (each butterfly owns a disjoint index pair).
Inverse flips the twiddle sign and scales by `1/N`.

`fft`/`ifft` are one-shot plans. Transforming many same-size signals? Build
the plan once — repeated `execute`s skip the kernel rebuild + re-JIT and the
per-butterfly `sin`/`cos`:

```rust,ignore
let gpu = quanta::init_cpu();
let mut plan = quanta_fft::FftPlan::new(&gpu, 1024, false).unwrap(); // forward
for (re, im) in frames {
    let (fr, fi) = plan.execute(&re, &im).unwrap();
    // ...
}
```

The plan owns the twiddle table and the compiled waves; I/O buffers are
allocated per `execute`, so executes stay independent.

### Real-input FFT (`rfft` / `irfft`)

A real signal's spectrum is conjugate-symmetric (`X[N−k] = conj(X[k])`), so
`rfft` returns just the first `N/2 + 1` bins — the whole spectrum's
information. It is the **packed real-FFT**: the N reals are packed as N/2
complex pairs `z[k] = x[2k] + i·x[2k+1]`, one half-size complex FFT runs on
the device (half the butterflies, half the device memory — the ~2× win over
transforming the signal as complex-with-zero-imag), and an O(N) split pass
separates the even/odd spectra (`X[k] = Fe[k] + e^(−2πik/N)·Fo[k]`). `irfft`
applies the exact algebraic inverse of the split, then the half-size inverse
plan:

```rust,ignore
let x = vec![1.0f32, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, -1.0]; // real, N = 8
let (hr, hi) = quanta_fft::rfft(&gpu, &x).unwrap();       // 5 bins (N/2 + 1)
let back = quanta_fft::irfft(&gpu, &hr, &hi, 8).unwrap(); // back ≈ x
```

`hi[0]` and `hi[N/2]` (DC and Nyquist) are exactly `0.0` — those bins of a
real signal are real. The split/merge pass runs on the host in `f64` (the
transform I/O is host vectors anyway; O(N) is negligible next to the
O(N log N) device work), so `rfft` matches the oracle at the same tolerance
as the complex path but is not bit-identical to slicing `fft([x, zeros])` —
packing reorders the f32 summations.

Sizes must be a power of 2; others return `NotSupported` (mixed-radix is a later
increment).

```rust,no_run
let gpu = quanta::init_cpu();
let re = vec![1.0f32, 2.0, 3.0, 4.0];
let im = vec![0.0f32; 4];
let (fr, fi) = quanta_fft::fft(&gpu, &re, &im).unwrap();
let (rr, _)  = quanta_fft::ifft(&gpu, &fr, &fi).unwrap();   // rr ≈ re
```

```toml
[dependencies]
quanta-fft = { version = "0.1", features = ["gpu-metal"] } # gpu-vulkan / gpu
```

Off by default the crate is the pure-Rust reference library (`reference::dft`);
enable `gpu` (+ a backend) for the device FFT.

## Verification (honest framing)

- **Lean** — Cooley-Tukey proven correct end to end (0 sorry): the radix-2
  butterfly identity `X[k] = Xe[k] + ω·Xo[k]` (`dft_radix2`) and its `log₂N`
  iteration to the full DFT (`fftRec_eq_dftN`), built from scratch over an
  ℕ-indexed DFT (Mathlib has the DFT but no radix-2 decomposition).
- **Differential** — the GPU FFT matches the direct DFT for every power-of-2
  size up to 256, `ifft` matches the direct inverse DFT, and `ifft(fft(x)) == x`
  round-trips. `rfft` matches the direct real-DFT oracle and the first
  `N/2+1` bins of the full complex FFT; `irfft(rfft(x), N) ≈ x` round-trips;
  reconstructing the full spectrum from the half by conjugate symmetry
  matches `fft([x, zeros])`. All validated on the software lane **and real
  Metal**.

## Coming next

Mixed-radix / arbitrary-N (Bluestein for primes) and batched/multi-dimensional
transforms.
