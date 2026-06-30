/-
Cooley-Tukey radix-2 correctness — the butterfly identity.

`quanta-fft` computes the DFT

  X[k] = Σ_{j<N} exp(-2πi·j·k/N) · x[j]

by the radix-2 Cooley-Tukey algorithm. The mathematical heart is the
**butterfly decomposition**: for `N = 2M`, splitting the sum into even
(`j = 2j'`) and odd (`j = 2j'+1`) indices gives

  X[k] = Xe[k] + W·Xo[k]

where `Xe`/`Xo` are the size-`M` DFTs of the even/odd subsequences and
`W = exp(-2πi·k/(2M))` is the twiddle. This file proves that identity
(`dft_radix2`), the load-bearing step the kernel's butterfly stage realises.
Iterating it `log₂N` times — the full recursion — is a separate induction
built on this lemma.

The DFT is stated over `ℕ` indices (`Finset.range N`), matching what the kernel
and the `quanta_fft::reference` oracle compute, so the proof is about the actual
algorithm. Mathlib's `ZMod` DFT (`Mathlib.Analysis.Fourier.ZMod`) is the
abstract counterpart; this is the index-explicit form.
-/

import Mathlib.Data.Complex.Exponential
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Complex.Circle

namespace Quanta.Fft

open scoped BigOperators
open Complex

/-- The DFT twiddle `ω_N^{jk} = exp(-2πi·j·k/N)`. -/
noncomputable def tw (N : ℕ) (m : ℕ) : ℂ :=
  Complex.exp (-2 * Real.pi * Complex.I * (m : ℂ) / (N : ℂ))

/-- Direct DFT over `ℕ` indices: `X[k] = Σ_{j<N} ω_N^{jk}·x[j]`. -/
noncomputable def dftN (N : ℕ) (x : ℕ → ℂ) (k : ℕ) : ℂ :=
  ∑ j ∈ Finset.range N, tw N (j * k) * x j

/-- Twiddle multiplicativity in the exponent: `ω_N^{a+b} = ω_N^a · ω_N^b`. -/
theorem tw_add (N a b : ℕ) : tw N (a + b) = tw N a * tw N b := by
  unfold tw
  rw [← Complex.exp_add]
  congr 1
  push_cast
  ring

/-- The even-index twiddle halves the modulus: `ω_{2M}^{2t} = ω_M^t`. (The key
    algebraic fact that turns the even subsequence into a size-`M` DFT.) -/
theorem tw_even (M t : ℕ) (hM : M ≠ 0) : tw (2 * M) (2 * t) = tw M t := by
  unfold tw
  congr 1
  have h2 : (2 : ℂ) * (M : ℂ) ≠ 0 := by
    simp [hM]
  have hMne : (M : ℂ) ≠ 0 := by exact_mod_cast hM
  field_simp
  ring

/-- A sum over `range (2M)` splits into the even-indexed and odd-indexed
    sub-sums, each reindexed over `range M`. Pure combinatorics — the geometric
    decomposition the radix-2 step rests on. -/
theorem sum_range_even_odd {α : Type*} [AddCommMonoid α] (M : ℕ) (g : ℕ → α) :
    ∑ j ∈ Finset.range (2 * M), g j
      = (∑ i ∈ Finset.range M, g (2 * i)) + ∑ i ∈ Finset.range M, g (2 * i + 1) := by
  induction M with
  | zero => simp
  | succ m ih =>
    -- range (2(m+1)) = range (2m) ∪ {2m, 2m+1}; peel the two new terms.
    have e : 2 * (m + 1) = (2 * m + 1) + 1 := by ring
    rw [e, Finset.sum_range_succ, Finset.sum_range_succ, ih,
        Finset.sum_range_succ, Finset.sum_range_succ]
    abel

/-- **Radix-2 butterfly identity.** For `N = 2M` (`M ≠ 0`), the DFT splits into
    the size-`M` DFTs of the even and odd subsequences, combined with the
    twiddle `ω_{2M}^k`:

      X[k] = Xe[k] + ω_{2M}^k · Xo[k]

    with `xe j = x(2j)`, `xo j = x(2j+1)`. This is the butterfly the kernel's
    stage computes; iterating it is Cooley-Tukey. -/
theorem dft_radix2 (M : ℕ) (hM : M ≠ 0) (x : ℕ → ℂ) (k : ℕ) :
    dftN (2 * M) x k
      = dftN M (fun j => x (2 * j)) k
        + tw (2 * M) k * dftN M (fun j => x (2 * j + 1)) k := by
  unfold dftN
  rw [sum_range_even_odd M (fun j => tw (2 * M) (j * k) * x j)]
  congr 1
  · -- Even part: ω_{2M}^{(2i)k} = ω_{2M}^{2(ik)} = ω_M^{ik}.
    apply Finset.sum_congr rfl
    intro i _
    have : 2 * i * k = 2 * (i * k) := by ring
    rw [this, tw_even M (i * k) hM]
  · -- Odd part: ω_{2M}^{(2i+1)k} = ω_{2M}^{2(ik)} · ω_{2M}^k = ω_M^{ik} · ω_{2M}^k.
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    have hsplit : (2 * i + 1) * k = 2 * (i * k) + k := by ring
    rw [hsplit, tw_add (2 * M) (2 * (i * k)) k, tw_even M (i * k) hM]
    ring

end Quanta.Fft
