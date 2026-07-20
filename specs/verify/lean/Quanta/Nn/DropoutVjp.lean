/-
Dropout — Lean proof foundation for quanta-nn's key-based dropout.

The kernel draws a uniform 32-bit word per element from a counter-based
Philox stream (key, element-index) and KEEPS the element iff `t ≤ u`,
where `t = ⌊rate · N⌋` and `N = 2³²` — then scales kept elements by
`N / (N − t)` (inverted dropout). The mask is a pure function of
(key, index), so the backward regenerates it instead of storing it, and
one kernel serves both directions. What that rests on:

* **T9231 — unbiasedness, exactly, at the quantized rate.** Averaged
  over all `N` equally-likely words, `keep · x · N/(N−t)` sums to
  `N · x` — the estimator is unbiased for the rate the kernel actually
  implements (`t/N`), not just the requested real-valued rate.
* **T9232 — the VJP is the forward map.** The mask-scale map is a
  diagonal linear operator, hence self-adjoint:
  `⟨m·s·x, g⟩ = ⟨x, m·s·g⟩`. The backward pass is the SAME masked
  scaling applied to the cotangent — the fact that licenses regenerating
  the mask and reusing the one kernel.
* **T9233 — threshold quantization.** With `t = ⌊rate·N⌋`, the
  implemented drop-rate `t/N` satisfies `t/N ≤ rate < t/N + 1/N`: the
  kernel's rate never exceeds the requested one and undershoots by less
  than `2⁻³²`.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.Order.Floor
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

namespace Quanta.Nn.DropoutVjp

open Finset

/-- T9231 — inverted dropout is exactly unbiased at the quantized rate:
over the `N` equally-likely words, keeping `u ≥ t` with scale `N/(N−t)`
averages to the identity (`∑ = N · x`; divide both sides by `N`). -/
theorem t9231_dropout_unbiased (N t : ℕ) (ht : t < N) (x : ℝ) :
    (∑ u ∈ Finset.range N, if t ≤ u then x * (N : ℝ) / ((N : ℝ) - t) else 0)
      = (N : ℝ) * x := by
  have hfilter : (Finset.range N).filter (fun u => t ≤ u) = Finset.Ico t N := by
    ext u
    simp [Finset.mem_filter, Finset.mem_range, Finset.mem_Ico, and_comm]
  have hne : ((N : ℝ) - t) ≠ 0 := by
    have : (t : ℝ) < N := by exact_mod_cast ht
    linarith
  rw [← Finset.sum_filter, hfilter, Finset.sum_const, Nat.card_Ico,
    nsmul_eq_mul, Nat.cast_sub (le_of_lt ht)]
  rw [mul_comm ((N : ℝ) - (t : ℝ)) (x * (N : ℝ) / ((N : ℝ) - (t : ℝ))),
    div_mul_cancel₀ _ hne]
  ring

/-- T9232 — the mask-scale map is self-adjoint: applying it to the input
or to the cotangent gives the same inner product. The dropout backward is
therefore the forward kernel run on the cotangent, with the mask
regenerated from the same key. -/
theorem t9232_dropout_self_adjoint {n : ℕ} (m : Fin n → ℝ) (s : ℝ)
    (x g : Fin n → ℝ) :
    ∑ i, (m i * s * x i) * g i = ∑ i, x i * (m i * s * g i) := by
  refine Finset.sum_congr rfl fun i _ => ?_
  ring

/-- T9233 — the floor threshold quantizes the rate from below, by less
than one word: `t/N ≤ rate < t/N + 1/N` for `t = ⌊rate·N⌋`. -/
theorem t9233_threshold_quantization (N : ℕ) (hN : 0 < N) (rate : ℝ) :
    ((⌊rate * (N : ℝ)⌋ : ℝ) / (N : ℝ) ≤ rate)
      ∧ (rate < ((⌊rate * (N : ℝ)⌋ : ℝ) + 1) / (N : ℝ)) := by
  have hNR : (0 : ℝ) < N := by exact_mod_cast hN
  constructor
  · have h := Int.floor_le (rate * (N : ℝ))
    have := (div_le_div_iff_of_pos_right hNR).mpr h
    rwa [mul_div_cancel_right₀ rate (ne_of_gt hNR)] at this
  · have h := Int.lt_floor_add_one (rate * (N : ℝ))
    have := (div_lt_div_iff_of_pos_right hNR).mpr h
    rwa [mul_div_cancel_right₀ rate (ne_of_gt hNR)] at this

end Quanta.Nn.DropoutVjp
