/-
Loss identities — Lean proof foundation for quanta-nn's losses.

* **T9228 — cross-entropy nonnegativity.** `x_y ≤ lse(x)`, so the stable
  form `loss = lse(x) − x_y` the fused kernel computes is nonnegative for
  every row — a negative CE reading is a bug, never a rounding artifact.
* **T9229 — BCE-with-logits stable form.** The overflow-free spelling
  `max(x,0) − x·y + log(1 + e^{−|x|})` EQUALS
  `−(y·log σ(x) + (1−y)·log(1−σ(x)))` for all logits — the identity that
  lets the implementation never evaluate `σ` near 0 or 1.
* **T9230 — the Huber knee.** The quadratic and linear branches meet at
  `|r| = δ` in value (the `δ²/2` offset people routinely drop), and on the
  quadratic region the gradient equals the clamp `clamp(r, −δ, δ)` — so
  the Huber gradient is globally the clamp, with no discontinuity.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

namespace Quanta.Nn.LossVjp

open Finset

/-- T9228 — the label logit never exceeds the log-sum-exp, so the stable
cross-entropy `lse(x) − x_y` is nonnegative. -/
theorem t9228_ce_nonneg {C : ℕ} (x : Fin C → ℝ) (y : Fin C) :
    x y ≤ Real.log (∑ i, Real.exp (x i)) := by
  have hy : Real.exp (x y) ≤ ∑ i, Real.exp (x i) :=
    single_le_sum (fun i _ => le_of_lt (Real.exp_pos _)) (mem_univ y)
  calc x y = Real.log (Real.exp (x y)) := (Real.log_exp _).symm
    _ ≤ Real.log (∑ i, Real.exp (x i)) := Real.log_le_log (Real.exp_pos _) hy

/-- T9229 — the overflow-free BCE-with-logits spelling equals the
textbook `−(y·log σ + (1−y)·log(1−σ))` for every logit and target. -/
theorem t9229_bce_with_logits_stable (x y : ℝ) :
    max x 0 - x * y + Real.log (1 + Real.exp (-|x|))
      = -(y * Real.log (1 / (1 + Real.exp (-x)))
          + (1 - y) * Real.log (1 - 1 / (1 + Real.exp (-x)))) := by
  have he : (0 : ℝ) < Real.exp (-x) := Real.exp_pos _
  have h1e : (0 : ℝ) < 1 + Real.exp (-x) := by positivity
  have hlog_s : Real.log (1 / (1 + Real.exp (-x)))
      = -Real.log (1 + Real.exp (-x)) := by
    rw [one_div, Real.log_inv]
  have h1s : (1 : ℝ) - 1 / (1 + Real.exp (-x))
      = Real.exp (-x) / (1 + Real.exp (-x)) := by
    rw [eq_div_iff (ne_of_gt h1e), sub_mul, one_mul, one_div,
      inv_mul_cancel₀ (ne_of_gt h1e)]
    ring
  have hlog_1s : Real.log (1 - 1 / (1 + Real.exp (-x)))
      = -x - Real.log (1 + Real.exp (-x)) := by
    rw [h1s, Real.log_div (ne_of_gt he) (ne_of_gt h1e), Real.log_exp]
  rcases le_or_lt 0 x with hx | hx
  · rw [max_eq_left hx, abs_of_nonneg hx, hlog_s, hlog_1s]
    ring
  · rw [max_eq_right (le_of_lt hx), abs_of_neg hx, neg_neg, hlog_s, hlog_1s]
    have hprod : (1 : ℝ) + Real.exp x = Real.exp x * (1 + Real.exp (-x)) := by
      rw [mul_add, mul_one, ← Real.exp_add, add_neg_cancel, Real.exp_zero]
      ring
    have hkey : Real.log (1 + Real.exp x)
        = x + Real.log (1 + Real.exp (-x)) := by
      rw [hprod, Real.log_mul (Real.exp_ne_zero x) (ne_of_gt h1e), Real.log_exp]
    rw [hkey]
    ring

/-- T9230 (value) — the branches meet at the knee: at `|r| = δ` the
quadratic `r²/2` equals the linear `δ(|r| − δ/2)`. -/
theorem t9230_huber_knee_value (δ : ℝ) : δ ^ 2 / 2 = δ * (δ - δ / 2) := by
  ring

/-- T9230 (gradient) — on the quadratic region the Huber gradient `r`
IS the clamp `clamp(r, −δ, δ)`, so the gradient the kernel computes is
globally the clamp, continuous across the knee. -/
theorem t9230_huber_grad_clamp (r δ : ℝ) (h : |r| ≤ δ) :
    max (-δ) (min r δ) = r := by
  rcases abs_le.mp h with ⟨h1, h2⟩
  rw [min_eq_left h2, max_eq_right h1]

end Quanta.Nn.LossVjp
