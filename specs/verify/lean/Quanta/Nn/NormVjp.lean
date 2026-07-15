/-
LayerNorm / RMSNorm VJP identities — Lean proof foundation for
`quanta-nn`'s fused normalization kernels.

The fused backward kernels implement closed-form three-term gradients
instead of backpropagating through the composed mean/var/sqrt/div chain:

* LayerNorm:  `dx = (1/s) · (h − mean h − x̂ · mean (h∘x̂))`
* RMSNorm:    `dx = (1/r) · (h − x̂ · mean (h∘x̂))`

where `h = g∘γ` is the cotangent pulled through the scale, `x̂` the
normalized row, and `s` / `r` the regularized std / RMS.

## What we prove (T9210–T9215)

* **T9210 / T9211 — the adjoint identities.** For the standard
  linearization of `x ↦ x̂` (matrix `(δᵢⱼ − 1/C)/s − x̂ⱼx̂ᵢ/(C·s)` for
  LayerNorm; `δᵢⱼ/r − x̂ⱼx̂ᵢ/(C·r)` for RMSNorm), the pairing
  `⟨h, L v⟩` equals `⟨dx-formula(h), v⟩` for every direction `v`:
  the kernels' three-term formula IS the adjoint of the linearization.
* **T9212 — centering invariant**: the LayerNorm-normalized row sums
  to zero.
* **T9213 / T9214 — stability**: `√ε ≤ s` (resp. `≤ r`) whenever the
  regularizer sits under the square root with a nonnegative moment, so
  with `ε > 0` no division in forward or backward can blow up — the
  reason the fused kernels carry no guards on `1/s`.
* **T9215 — derivative anchor**: the directional derivative of the row
  mean is the mean of the direction (the affine base case of the
  composite chain).

The full composite Fréchet chain (that the documented linearization is
THE derivative of the mean/var/sqrt pipeline) is the declared next
increment. Analytically the fused path is additionally cross-checked in
tests against the composed tape path, whose per-op VJPs are proven in
`Quanta/Autograd`.
-/

import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

namespace Quanta.Nn.NormVjp

open Finset

variable {C : ℕ}

/-- Row mean over `Fin C`. -/
noncomputable def mean (f : Fin C → ℝ) : ℝ := (∑ i, f i) / C

/-! ### The adjoint identities (pure algebra over one row)

Both proofs follow the same route: rewrite each side into the atoms
`∑ h∘v`, `∑ h`, `∑ v`, `∑ h∘x̂`, `∑ x̂∘v`, then close with `ring`. -/

/-- T9210 — LayerNorm VJP adjoint identity. -/
theorem t9210_layer_norm_vjp_adjoint (s : ℝ) (xh h v : Fin C → ℝ) :
    (∑ j, h j * ((v j - mean v) / s - xh j * (mean (fun i => xh i * v i) / s)))
      = ∑ i, (h i - mean h - xh i * mean (fun j => h j * xh j)) * v i / s := by
  have lhs_step : ∀ j,
      h j * ((v j - mean v) / s - xh j * (mean (fun i => xh i * v i) / s))
        = (h j * v j - mean v * h j - mean (fun i => xh i * v i) * (h j * xh j)) / s := by
    intro j; simp only [div_eq_mul_inv]; ring
  have rhs_step : ∀ i,
      (h i - mean h - xh i * mean (fun j => h j * xh j)) * v i / s
        = (h i * v i - mean h * v i - mean (fun j => h j * xh j) * (xh i * v i)) / s := by
    intro i; simp only [div_eq_mul_inv]; ring
  rw [Finset.sum_congr rfl fun j _ => lhs_step j,
      Finset.sum_congr rfl fun i _ => rhs_step i,
      ← Finset.sum_div, ← Finset.sum_div]
  congr 1
  rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib,
      Finset.sum_sub_distrib, Finset.sum_sub_distrib,
      ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum]
  simp only [mean]
  ring

/-- T9211 — RMSNorm VJP adjoint identity (the centering term drops). -/
theorem t9211_rms_norm_vjp_adjoint (r : ℝ) (xh h v : Fin C → ℝ) :
    (∑ j, h j * (v j / r - xh j * (mean (fun i => xh i * v i) / r)))
      = ∑ i, (h i - xh i * mean (fun j => h j * xh j)) * v i / r := by
  have lhs_step : ∀ j,
      h j * (v j / r - xh j * (mean (fun i => xh i * v i) / r))
        = (h j * v j - mean (fun i => xh i * v i) * (h j * xh j)) / r := by
    intro j; simp only [div_eq_mul_inv]; ring
  have rhs_step : ∀ i,
      (h i - xh i * mean (fun j => h j * xh j)) * v i / r
        = (h i * v i - mean (fun j => h j * xh j) * (xh i * v i)) / r := by
    intro i; simp only [div_eq_mul_inv]; ring
  rw [Finset.sum_congr rfl fun j _ => lhs_step j,
      Finset.sum_congr rfl fun i _ => rhs_step i,
      ← Finset.sum_div, ← Finset.sum_div]
  congr 1
  rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib,
      ← Finset.mul_sum, ← Finset.mul_sum]
  simp only [mean]
  ring

/-! ### The centering invariant -/

/-- T9212 — the LayerNorm-normalized row sums to zero: centering by the
row mean kills the row sum, and the common divisor `s` preserves that. -/
theorem t9212_centered_row_sums_to_zero (hC : 0 < C) (s : ℝ) (x : Fin C → ℝ) :
    (∑ i, (x i - mean x) / s) = 0 := by
  have hCr : (C : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hC.ne'
  rw [← Finset.sum_div, Finset.sum_sub_distrib, Finset.sum_const,
      Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  simp only [mean]
  have h1 : (C : ℝ) * ((∑ i, x i) / C) = ∑ i, x i := by field_simp
  rw [h1, sub_self, zero_div]

/-! ### Stability -/

/-- T9213 — LayerNorm stability: the regularized std dominates `√ε`
because the variance moment is a mean of squares, hence nonnegative.
With `ε > 0` every division by `s` in the forward and the backward is
division by at least `√ε`. -/
theorem t9213_layer_norm_std_lower_bound (hC : 0 < C) (ε : ℝ) (x : Fin C → ℝ) :
    Real.sqrt ε ≤ Real.sqrt (mean (fun i => (x i - mean x) ^ 2) + ε) := by
  apply Real.sqrt_le_sqrt
  have hnn : 0 ≤ mean fun i => (x i - mean x) ^ 2 := by
    have : 0 ≤ ∑ i, (x i - mean x) ^ 2 :=
      Finset.sum_nonneg fun i _ => sq_nonneg _
    exact div_nonneg this (Nat.cast_nonneg C)
  linarith

/-- T9214 — RMSNorm stability: the regularized RMS dominates `√ε`. -/
theorem t9214_rms_norm_lower_bound (hC : 0 < C) (ε : ℝ) (x : Fin C → ℝ) :
    Real.sqrt ε ≤ Real.sqrt (mean (fun i => x i ^ 2) + ε) := by
  apply Real.sqrt_le_sqrt
  have hnn : 0 ≤ mean fun i => x i ^ 2 := by
    have : 0 ≤ ∑ i, x i ^ 2 := Finset.sum_nonneg fun i _ => sq_nonneg _
    exact div_nonneg this (Nat.cast_nonneg C)
  linarith

/-! ### Derivative anchor -/

/-- T9215 — the directional derivative of the row mean along `v` is
`mean v`: the mean of the perturbed row `x + t·v` is affine in `t` with
slope `mean v`. The affine base case of the composite chain whose full
Fréchet treatment is the declared next increment. -/
theorem t9215_mean_directional_derivative (x v : Fin C → ℝ) (t₀ : ℝ) :
    HasDerivAt (fun t => mean (fun i => x i + t * v i)) (mean v) t₀ := by
  have hfun : (fun t => mean (fun i => x i + t * v i))
      = fun t => mean x + mean v * t := by
    funext t
    simp only [mean, div_eq_mul_inv]
    rw [Finset.sum_add_distrib, ← Finset.mul_sum]
    ring
  rw [hfun]
  simpa using ((hasDerivAt_id t₀).const_mul (mean v)).const_add (mean x)

end Quanta.Nn.NormVjp
