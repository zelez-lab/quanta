/-
Activation identities — Lean proof foundation for quanta-nn's fused
activation kernels (rowwise softmax / log-softmax, GeLU, SwiGLU).

* **T9223 — log-sum-exp stability pair.** Shifting the logits shifts the
  lse by the same amount — THE trick the fused kernels implement (subtract
  the row max before exponentiating); and the resulting softmax weights
  sum to exactly one.
* **T9224 — softmax VJP adjoint.** The fused backward formula
  `dx = p ⊙ (g − ⟨g, p⟩)` is the adjoint of the softmax Jacobian
  `diag(p) − p pᵀ`, for any direction.
* **T9225 — log-softmax VJP adjoint.** `dx = g − (∑g)·p` is the adjoint
  of `I − 1 pᵀ`.
* **T9226 — sigmoid derivative algebra.** With `s = 1/(1+e)` (`e = e^{−x}`),
  `s·(1−s) = e/(1+e)²` — the factor the SwiGLU backward computes from the
  forward's sigmoid without re-exponentiating.
* **T9227 — the sech² identity.** `(1 − tanh²u)·cosh²u = 1`: the GeLU
  backward needs `sech²u`, and this identity lets it reuse the tanh value
  the forward already computed — the kernel never evaluates cosh.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

namespace Quanta.Nn.ActivationVjp

open Finset

/-- T9223 (shift invariance) — subtracting a constant from every logit
subtracts it from the log-sum-exp: the max-subtraction stabilization is
exact, not an approximation. -/
theorem t9223_lse_shift_invariant {C : ℕ} (hC : 0 < C) (x : Fin C → ℝ) (c : ℝ) :
    Real.log (∑ i, Real.exp (x i - c)) = Real.log (∑ i, Real.exp (x i)) - c := by
  have hne : (univ : Finset (Fin C)).Nonempty :=
    univ_nonempty_iff.mpr (Fin.pos_iff_nonempty.mp hC)
  have hpos : 0 < ∑ i, Real.exp (x i) :=
    sum_pos (fun i _ => Real.exp_pos _) hne
  have hsum : ∑ i, Real.exp (x i - c) = (∑ i, Real.exp (x i)) * Real.exp (-c) := by
    rw [sum_mul]
    exact sum_congr rfl fun i _ => by
      rw [← Real.exp_add]
      ring_nf
  rw [hsum, Real.log_mul (ne_of_gt hpos) (Real.exp_ne_zero _), Real.log_exp]
  ring

/-- T9223 (unit sum) — the stabilized softmax weights sum to exactly one. -/
theorem t9223_softmax_sums_to_one {C : ℕ} (hC : 0 < C) (x : Fin C → ℝ) :
    ∑ i, Real.exp (x i - Real.log (∑ j, Real.exp (x j))) = 1 := by
  have hne : (univ : Finset (Fin C)).Nonempty :=
    univ_nonempty_iff.mpr (Fin.pos_iff_nonempty.mp hC)
  have hpos : 0 < ∑ j, Real.exp (x j) :=
    sum_pos (fun j _ => Real.exp_pos _) hne
  have h : ∀ i : Fin C,
      Real.exp (x i - Real.log (∑ j, Real.exp (x j)))
        = Real.exp (x i) / ∑ j, Real.exp (x j) := fun i => by
    rw [Real.exp_sub, Real.exp_log hpos]
  rw [sum_congr rfl fun i _ => h i, ← sum_div, div_self (ne_of_gt hpos)]

/-- T9224 — the softmax VJP is the adjoint of the softmax Jacobian
`diag(p) − p pᵀ`: pairing `g` with the Jacobian applied to any direction
`v` equals pairing the fused formula `p ⊙ (g − ⟨g, p⟩)` with `v`. -/
theorem t9224_softmax_vjp_adjoint {C : ℕ} (p g v : Fin C → ℝ) :
    ∑ i, g i * (p i * v i - p i * ∑ j, p j * v j)
      = ∑ i, (p i * (g i - ∑ j, g j * p j)) * v i := by
  have hl : ∀ i : Fin C,
      g i * (p i * v i - p i * ∑ j, p j * v j)
        = p i * g i * v i - (g i * p i) * ∑ j, p j * v j := fun i => by ring
  have hr : ∀ i : Fin C,
      (p i * (g i - ∑ j, g j * p j)) * v i
        = p i * g i * v i - (∑ j, g j * p j) * (p i * v i) := fun i => by ring
  rw [sum_congr rfl fun i _ => hl i, sum_congr rfl fun i _ => hr i,
    sum_sub_distrib, sum_sub_distrib, ← sum_mul, ← mul_sum]

/-- T9225 — the log-softmax VJP is the adjoint of `I − 1 pᵀ`:
`dx = g − (∑g)·p`. -/
theorem t9225_log_softmax_vjp_adjoint {C : ℕ} (p g v : Fin C → ℝ) :
    ∑ i, g i * (v i - ∑ j, p j * v j)
      = ∑ i, (g i - (∑ j, g j) * p i) * v i := by
  have hl : ∀ i : Fin C,
      g i * (v i - ∑ j, p j * v j)
        = g i * v i - g i * ∑ j, p j * v j := fun i => by ring
  have hr : ∀ i : Fin C,
      (g i - (∑ j, g j) * p i) * v i
        = g i * v i - (∑ j, g j) * (p i * v i) := fun i => by ring
  rw [sum_congr rfl fun i _ => hl i, sum_congr rfl fun i _ => hr i,
    sum_sub_distrib, sum_sub_distrib, ← sum_mul, ← mul_sum]

/-- T9226 — sigmoid derivative algebra: with `s = 1/(1+e)` (where
`e = e^{−x} > 0`), `s·(1−s) = e/(1+e)²`. The SwiGLU backward computes
`σ'` from the forward's sigmoid value without re-exponentiating. -/
theorem t9226_sigmoid_derivative_algebra (e : ℝ) (he : 0 < e) :
    (1 / (1 + e)) * (1 - 1 / (1 + e)) = e / (1 + e) ^ 2 := by
  have h : (1 : ℝ) + e ≠ 0 := by positivity
  have key : (1 : ℝ) - 1 / (1 + e) = e / (1 + e) := by
    rw [eq_div_iff h, sub_mul, one_mul, one_div, inv_mul_cancel₀ h]
    ring
  rw [key, div_mul_div_comm, one_mul, sq]

/-- T9227 — the sech² identity: `(1 − tanh²u)·cosh²u = 1`. The GeLU
backward needs `sech²u` and gets it as `1 − tanh²u` from the tanh value
the forward already computed — no cosh is ever evaluated. -/
theorem t9227_sech_sq_identity (u : ℝ) :
    (1 - Real.tanh u ^ 2) * Real.cosh u ^ 2 = 1 := by
  have hc : Real.cosh u ≠ 0 := ne_of_gt (Real.cosh_pos u)
  rw [Real.tanh_eq_sinh_div_cosh]
  field_simp

end Quanta.Nn.ActivationVjp
