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
| `fft`  | `fft(gpu, &re, &im) -> (Vec<f32>, Vec<f32>)`  | forward DFT, N a power of 2 |
| `ifft` | `ifft(gpu, &re, &im) -> (Vec<f32>, Vec<f32>)` | inverse (÷N); `ifft(fft(x)) == x` |
| `reference::dft` / `idft` | `dft(&re, &im) -> (Vec<f32>, Vec<f32>)` | pure-Rust direct O(N²) DFT — the oracle (always available, no `gpu` feature) |

Complex data is **split**: a real-part slice and an imag-part slice of equal
length. The GPU transform runs a bit-reversal kernel (in-kernel `log₂N`-bit
reversal) then `log₂N` butterfly stages — `N/2` threads each, twiddles computed
in-kernel via `sin`/`cos`, in place (each butterfly owns a disjoint index pair).
Inverse flips the twiddle sign and scales by `1/N`.

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
  round-trips. Validated on the software lane **and real Metal**.

## Coming next

Mixed-radix / arbitrary-N (Bluestein for primes), a real-input `rfft`
(half-spectrum), a precomputed `Plan` that caches twiddles (the VkFFT pattern),
and batched/multi-dimensional transforms.
