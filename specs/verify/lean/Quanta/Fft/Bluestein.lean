/-
Bluestein's chirp-z algorithm — arbitrary-N DFT as a chirp convolution.

`Dft.lean`/`Recursion.lean` handle the power-of-2 radix-2 path. For
arbitrary `N`, `quanta-fft` routes through Bluestein: the DFT is rewritten
as a convolution of a chirped input against a chirp kernel, wrapped by an
output chirp. The mathematical heart is the exponent identity

  2·n·k = n² + k² − (k − n)²      (over ℤ)

which factors the DFT twiddle `exp(-2πi·nk/N)` into three chirps:

  exp(-2πi·nk/N)
    = exp(-πi·n²/N) · exp(-πi·k²/N) · exp(+πi·(k−n)²/N).

Writing `chirp N j = exp(-πi·j²/N)` (so `exp(+πi·m²/N) = (chirp N m)⁻¹`),
the direct DFT becomes

  X[k] = chirp N k · Σ_{n<N} (chirp N n · x n) · (chirp N (k−n))⁻¹,

an `N`-point convolution of the chirped input `a n = chirp N n · x n` with
the kernel `b m = (chirp N m)⁻¹`, post-multiplied by the output chirp
`chirp N k`. This file proves that identity (`bluestein_eq_dftN`) — the
spec-level correctness of the chirp-z rewrite, independent of the f32
device numerics or the power-of-2 convolution length `M ≥ 2N−1` (which
only affects how the linear convolution is realised circularly, not the
value of the linear convolution itself).

The chirp phase uses a `(k − n)` that ranges over ℤ (it is negative when
`n > k`), so the chirp is defined on ℤ indices; `chirp_neg` records that it
is even (`chirp N (−j) = chirp N j`), matching the kernel's wrap-around
layout `b[M−m] = b[m]`.
-/

import Quanta.Fft.Dft

namespace Quanta.Fft

open scoped BigOperators
open Complex

/-- The Bluestein chirp `exp(-πi·j²/N)` on ℤ indices. Squaring makes it
    even in `j`; `chirp N k · (chirp N n) · (chirp N (k−n))⁻¹` reassembles
    the DFT twiddle `tw N (n*k)`. -/
noncomputable def chirp (N : ℕ) (j : ℤ) : ℂ :=
  Complex.exp (-Real.pi * Complex.I * ((j : ℂ) ^ 2) / (N : ℂ))

/-- The chirp is even: `exp(-πi·(−j)²/N) = exp(-πi·j²/N)`. This is why the
    convolution kernel can be laid out wrap-around (`b[M−m] = b[m]`). -/
theorem chirp_neg (N : ℕ) (j : ℤ) : chirp N (-j) = chirp N j := by
  unfold chirp
  congr 2
  push_cast
  ring

/-- The chirp never vanishes (it is a `Complex.exp`), so `(chirp N m)⁻¹` is a
    genuine two-sided inverse — used to move the kernel chirp across an
    equality without a division side-condition. -/
theorem chirp_ne_zero (N : ℕ) (j : ℤ) : chirp N j ≠ 0 := by
  unfold chirp
  exact Complex.exp_ne_zero _

/-- **Chirp factorisation of the twiddle.** For indices `n, k : ℕ`, the DFT
    twiddle factors into the three chirps via `2nk = n² + k² − (k−n)²`:

      tw N (n·k) = chirp N k · chirp N n · (chirp N (k − n))⁻¹.

    This is the load-bearing algebraic identity; everything else is summation
    bookkeeping. The `(k − n)` is taken over ℤ, so it is correct even when
    `n > k`. -/
theorem tw_eq_chirp_factor (N n k : ℕ) :
    tw N (n * k)
      = chirp N k * chirp N n * (chirp N ((k : ℤ) - (n : ℤ)))⁻¹ := by
  unfold tw chirp
  rw [← Complex.exp_neg, ← Complex.exp_add, ← Complex.exp_add]
  congr 1
  push_cast
  ring

/-- The Bluestein expression: output chirp times the chirp convolution of the
    input, summed over `n < N`.

      B[k] = chirp N k · Σ_{n<N} (chirp N n · x n) · (chirp N (k − n))⁻¹.

    `x` is sampled on ℕ indices as in `dftN`; the chirp indices are cast to ℤ
    so the kernel argument `k − n` is a true (possibly negative) integer. -/
noncomputable def bluestein (N : ℕ) (x : ℕ → ℂ) (k : ℕ) : ℂ :=
  chirp N k
    * ∑ n ∈ Finset.range N,
        (chirp N n * x n) * (chirp N ((k : ℤ) - (n : ℤ)))⁻¹

/-- **Bluestein computes the DFT.** The chirp-z expression equals the direct
    DFT for every output index `k`:

      bluestein N x k = dftN N x k.

    Proof: distribute the output chirp into the sum and match summands. Each
    term `tw N (n·k) · x n` of `dftN` equals `chirp N k · (chirp N n · x n) ·
    (chirp N (k−n))⁻¹` by `tw_eq_chirp_factor`; the `chirp N k` factor is the
    one pulled out front. No power-of-2 / convolution-length hypothesis is
    needed — this is the exact linear-convolution identity. -/
theorem bluestein_eq_dftN (N : ℕ) (x : ℕ → ℂ) (k : ℕ) :
    bluestein N x k = dftN N x k := by
  unfold bluestein dftN
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro n _
  rw [tw_eq_chirp_factor N n k]
  ring

/-- Un-normalized inverse DFT: `x̌[k] = Σ_{j<N} ω_N^{-jk}·x[j]` — the conjugate
    twiddle `(tw N (j*k))⁻¹`. The `1/N` normalization of the true inverse is a
    scalar convention applied separately (exactly as the forward `bluestein`
    proof left the pow2 convolution length `M` out of scope); this file proves
    the transform identity, not the normalization. -/
noncomputable def idftN (N : ℕ) (x : ℕ → ℂ) (k : ℕ) : ℂ :=
  ∑ j ∈ Finset.range N, (tw N (j * k))⁻¹ * x j

/-- **Inverse chirp factorisation.** Taking `⁻¹` of `tw_eq_chirp_factor`: since
    each chirp is a unit (`chirp_ne_zero`), inversion distributes and flips every
    factor —

      (tw N (n·k))⁻¹ = (chirp N k)⁻¹ · (chirp N n)⁻¹ · chirp N (k − n).

    This is the mirror identity the inverse Bluestein form rests on. -/
theorem tw_inv_eq_chirp_factor (N n k : ℕ) :
    (tw N (n * k))⁻¹
      = (chirp N k)⁻¹ * (chirp N n)⁻¹ * chirp N ((k : ℤ) - (n : ℤ)) := by
  rw [tw_eq_chirp_factor N n k]
  rw [mul_inv, mul_inv, inv_inv]

/-- The inverse Bluestein expression: the inverse output chirp times the
    inverse-chirp convolution of the input.

      B̌[k] = (chirp N k)⁻¹ · Σ_{n<N} ((chirp N n)⁻¹ · x n) · chirp N (k − n).

    Every chirp of the forward `bluestein` is replaced by its inverse (and the
    kernel chirp, formerly inverted, is now upright) — the exact conjugate mirror. -/
noncomputable def ibluestein (N : ℕ) (x : ℕ → ℂ) (k : ℕ) : ℂ :=
  (chirp N k)⁻¹
    * ∑ n ∈ Finset.range N,
        ((chirp N n)⁻¹ * x n) * chirp N ((k : ℤ) - (n : ℤ))

/-- **Inverse Bluestein computes the inverse DFT.** The conjugate chirp-z
    expression equals the (un-normalized) inverse DFT for every `k`:

      ibluestein N x k = idftN N x k.

    Proof mirrors `bluestein_eq_dftN`: distribute the inverse output chirp into
    the sum and match summands via `tw_inv_eq_chirp_factor`. -/
theorem ibluestein_eq_idftN (N : ℕ) (x : ℕ → ℂ) (k : ℕ) :
    ibluestein N x k = idftN N x k := by
  unfold ibluestein idftN
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro n _
  rw [tw_inv_eq_chirp_factor N n k]
  ring

end Quanta.Fft
