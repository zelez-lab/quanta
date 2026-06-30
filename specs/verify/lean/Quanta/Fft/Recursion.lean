/-
Cooley-Tukey full recursion correctness — `fftRec = dftN` on `2^p` points.

`Dft.lean` proved the single radix-2 step (`dft_radix2`). Here we iterate it: a
recursively-defined radix-2 FFT on `2^p` points equals the direct DFT, by
induction on the depth `p`. This is the end-to-end correctness of the
divide-and-conquer algorithm the kernel realises — every butterfly stage is one
application of `dft_radix2`, and `fftRec_eq_dftN` is the composition of all `p`
of them.
-/

import Quanta.Fft.Dft

namespace Quanta.Fft

open scoped BigOperators
open Complex

/-- Recursive radix-2 FFT on `2^p` points, mirroring `dft_radix2`: the base case
    (`p = 0`, one point) is the identity, and each level splits into the
    even/odd subsequences combined with the twiddle `ω_{2^{p}}^k`. The data `x`
    is sampled by index, so the even/odd split is `x ∘ (2·)` and `x ∘ (2·+1)`. -/
noncomputable def fftRec : ℕ → (ℕ → ℂ) → ℕ → ℂ
  | 0, x, _ => x 0
  | p + 1, x, k =>
      fftRec p (fun j => x (2 * j)) k
        + tw (2 ^ (p + 1)) k * fftRec p (fun j => x (2 * j + 1)) k

/-- **Cooley-Tukey is correct.** The recursive radix-2 FFT on `2^p` points
    computes the direct DFT. Induction on `p`: the base case is the
    1-point DFT (`= x 0`), and the step is exactly the butterfly identity
    `dft_radix2` with `M = 2^p`, applied to the two recursively-computed
    sub-DFTs (which match the direct sub-DFTs by the induction hypothesis). -/
theorem fftRec_eq_dftN (p : ℕ) (x : ℕ → ℂ) (k : ℕ) :
    fftRec p x k = dftN (2 ^ p) x k := by
  induction p generalizing x with
  | zero =>
    -- N = 1: dftN 1 x k = tw 1 (0·k) · x 0 = x 0.
    simp only [fftRec, pow_zero, dftN, Finset.range_one, Finset.sum_singleton, zero_mul, tw,
      Nat.cast_zero, mul_zero, zero_div, Complex.exp_zero, one_mul]
  | succ p ih =>
    -- N = 2·2^p: butterfly split, then the two sub-FFTs are sub-DFTs by `ih`.
    have hM : (2 : ℕ) ^ p ≠ 0 := pow_ne_zero p two_ne_zero
    have hpow : 2 ^ (p + 1) = 2 * 2 ^ p := by rw [pow_succ]; ring
    rw [fftRec, ih, ih, hpow, dft_radix2 (2 ^ p) hM x k]

end Quanta.Fft
