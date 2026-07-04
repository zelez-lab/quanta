# quanta-fft

GPU fast Fourier transform for Quanta. Forward and inverse, complex data
split into real/imag `f32` arrays, **any length N ≥ 1**: radix-2 Cooley-Tukey
for power-of-2 sizes, Bluestein's chirp-z algorithm (built on the same
radix-2 plans) for everything else. One kernel, every backend
(Metal / Vulkan / CPU).

The headline claim: **Cooley-Tukey correctness is mechanically proven** — the
radix-2 decomposition equals the direct DFT, end to end, in Lean
(`specs/verify/lean/Quanta/Fft/`), on top of being differential-tested against
a direct-DFT oracle and validated on real Metal. The Bluestein path's bar is
the same differential oracle (the Lean proof models the radix-2 recursion
only).

## Status — 1-D complex, any N (split re/im, forward + inverse)

| op | signature | notes |
|----|-----------|-------|
| `fft`  | `fft(gpu, &re, &im) -> (Vec<f32>, Vec<f32>)`  | forward DFT, any N (radix-2 for 2^k, Bluestein otherwise) |
| `ifft` | `ifft(gpu, &re, &im) -> (Vec<f32>, Vec<f32>)` | inverse (÷N); `ifft(fft(x)) == x`, any N |
| `rfft` | `rfft(gpu, &x) -> (Vec<f32>, Vec<f32>)` | real input → the `N/2 + 1` half-spectrum (packed method: one half-size complex FFT + O(N) split — ~2× the throughput, half the device memory); power-of-2 N |
| `irfft` | `irfft(gpu, &re, &im, n) -> Vec<f32>` | half-spectrum → real signal; `irfft(rfft(x), N) ≈ x` |
| `fft2`  | `fft2(gpu, &re, &im, h, w) -> (Vec<f32>, Vec<f32>)`  | 2-D forward, row-major H×W, both dims powers of 2 (row-column decomposition over `FftPlan`) |
| `ifft2` | `ifft2(gpu, &re, &im, h, w) -> (Vec<f32>, Vec<f32>)` | 2-D inverse (÷(H·W)); `ifft2(fft2(x)) == x` |
| `FftPlan` | `FftPlan::new(gpu, n, inverse)` → `plan.execute(&re, &im)` | plan-based dispatch (VkFFT pattern): kernels JIT-compiled once, twiddle table precomputed into a device buffer, reusable across executes |
| `reference::dft` / `idft` | `dft(&re, &im) -> (Vec<f32>, Vec<f32>)` | pure-Rust direct O(N²) DFT — the oracle (always available, no `gpu` feature) |
| `reference::rdft` / `irdft` | `rdft(&x) -> (Vec<f32>, Vec<f32>)` | direct real DFT (half-spectrum) + inverse — the `rfft`/`irfft` oracle |
| `reference::dft2` / `idft2` | `dft2(&re, &im, h, w) -> (Vec<f32>, Vec<f32>)` | direct O((HW)²) 2-D DFT double sum (NOT row-column composed) — the independent 2-D oracle |

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
allocated per `execute`, so executes stay independent. `FftPlan` itself is
power-of-2 only — it IS the radix-2 engine Bluestein builds on.

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
packing reorders the f32 summations. `rfft` takes a power-of-2 length.

### Non-power-of-2 sizes: Bluestein's chirp-z

`fft`/`ifft` accept any N. Non-power-of-2 sizes are rewritten as a
power-of-2 convolution via `nk = (n² + k² − (k−n)²)/2`:

```text
X[k] = exp(−πi·k²/N) · Σₙ [x[n]·exp(−πi·n²/N)] · exp(+πi·(k−n)²/N)
```

The chirped input `a[n] = x[n]·exp(−πi·n²/N)` is convolved with the chirp
kernel `b[m] = exp(+πi·m²/N)` at `M = next_pow2(2N−1)` using the radix-2
plans (forward-FFT both, pointwise-multiply, inverse-FFT), then the output
chirp is applied. The inverse conjugates the chirps and scales by `1/N`.
Chirps are host-computed in f64 with the `n²` phase reduced `mod 2N` in exact
integer arithmetic before the trig — that reduction is what keeps the phase
accurate at large N (N = 1000 lands ~160× inside the crate tolerance;
unreduced `n²` phases would drift there). Cost is three length-M transforms,
so a power-of-2 N is always the faster shape.

**2-D**: `fft2`/`ifft2` transform a row-major H×W grid (both dims powers of 2)
by the separable row-column method — a W-point row pass (one reused `FftPlan`,
H executes), a transpose so columns become rows, an H-point pass (W executes),
and a transpose back. The inverse folds `1/W`·`1/H` = `1/(H·W)` into the two
passes. Differential-tested against a direct 2-D DFT double sum (an oracle that
does **not** share the row-column decomposition) plus a separability check:
`fft2` of `f(y)·g(x)` equals the complex outer product of the 1-D spectra.

```rust,ignore
let (h, w) = (8, 16);
let (sr, si) = quanta_fft::fft2(&gpu, &re, &im, h, w)?;   // 2-D spectrum, H×W row-major
let (rr, ri) = quanta_fft::ifft2(&gpu, &sr, &si, h, w)?;  // rr ≈ re, ri ≈ im
```

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
  size up to 256 AND for the Bluestein sweep (primes 3/5/7/11/13/127,
  composites 6/9/10/12/15/100/1000); `ifft` matches the direct inverse DFT and
  `ifft(fft(x)) == x` round-trips for all of them. `rfft` matches the direct
  real-DFT oracle and the first `N/2+1` bins of the full complex FFT;
  `irfft(rfft(x), N) ≈ x` round-trips; reconstructing the full spectrum from
  the half by conjugate symmetry matches `fft([x, zeros])`. The 2-D `fft2`
  matches the direct 2-D double-sum DFT (2×2 through 16×16, square and
  rectangular), round-trips through `ifft2`, and passes the separability
  outer-product check. All validated on the software lane **and real Metal**.
  The Lean proof covers the radix-2 recursion only; Bluestein's correctness
  claim rests on the differential oracle.

## Coming next

Batched / multi-dimensional (3-D) transforms.
