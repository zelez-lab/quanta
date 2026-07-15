/-
Optimizer step identities — Lean proof foundation for quanta-nn's fused
optimizer kernels (SGD with momentum, Adam, AdamW).

The fused kernels compute one whole update per element in a single
dispatch. These theorems pin down the algebra the kernels implement and
the properties the tests check empirically (T9219/T9220/T9222 under a
constant gradient, T9221 exactly):

* **T9219 — momentum unrolling.** The velocity recurrence `v₀ = 0`,
  `vₜ₊₁ = μ·vₜ + g` under a constant gradient has the closed form
  `vₜ = g·∑_{k<t} μᵏ`: momentum is a geometrically-weighted gradient
  memory, nothing more.
* **T9220 — bias-correction exactness.** The Adam moment recurrence
  `m₀ = 0`, `mₜ₊₁ = β·mₜ + (1−β)·g` gives `mₜ = (1−βᵗ)·g`, so the
  corrected moment `mₜ/(1−βᵗ)` recovers `g` EXACTLY at every step —
  the reason the kernel divides by `1−βᵗ` (passed as a host scalar).
  The second moment is the same recurrence in `g²`.
* **T9221 — AdamW decoupling.** The shrink-then-step form the kernel
  computes, `(1−lr·wd)·p − lr·u`, equals the paper's add-a-decay-term
  form `p − lr·(u + wd·p)`. One kernel serves both spellings.
* **T9222 — scale invariance.** With exact moments (constant gradient,
  ε = 0) the Adam step magnitude is `lr` regardless of the gradient's
  scale — only its sign survives. This is the property that makes the
  ε-floor a tie-breaker rather than a tuning knob.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

namespace Quanta.Nn.OptimStep

open Finset

/-- Momentum velocity under a constant gradient `g`:
`v₀ = 0`, `vₜ₊₁ = μ·vₜ + g`. -/
def vel (μ g : ℝ) : ℕ → ℝ
  | 0 => 0
  | t + 1 => μ * vel μ g t + g

/-- Adam moment (exponential moving average) under a constant gradient:
`m₀ = 0`, `mₜ₊₁ = β·mₜ + (1−β)·g`. -/
def ema (β g : ℝ) : ℕ → ℝ
  | 0 => 0
  | t + 1 => β * ema β g t + (1 - β) * g

/-- T9219 — momentum is a geometric gradient memory:
`vₜ = g·∑_{k<t} μᵏ`. -/
theorem t9219_momentum_geometric (μ g : ℝ) (t : ℕ) :
    vel μ g t = g * ∑ k ∈ range t, μ ^ k := by
  induction t with
  | zero => simp [vel]
  | succ n ih =>
      rw [vel, ih, Finset.sum_range_succ']
      simp only [pow_succ, pow_zero]
      rw [← Finset.sum_mul]
      ring

/-- T9220 — the Adam moment closed form: `mₜ = (1−βᵗ)·g`. -/
theorem t9220_ema_closed_form (β g : ℝ) (t : ℕ) :
    ema β g t = (1 - β ^ t) * g := by
  induction t with
  | zero => simp [ema]
  | succ n ih =>
      rw [ema, ih, pow_succ]
      ring

/-- T9220 (corollary) — bias correction is exact: whenever the correction
denominator is nonzero, the corrected moment recovers the gradient at
EVERY step, not merely in the limit. -/
theorem t9220_bias_correction_exact (β g : ℝ) (t : ℕ) (h : 1 - β ^ t ≠ 0) :
    ema β g t / (1 - β ^ t) = g := by
  rw [t9220_ema_closed_form]
  field_simp

/-- T9221 — AdamW decoupling: the kernel's shrink-then-step form equals
the add-a-decay-term form. -/
theorem t9221_adamw_decoupling (p u lr wd : ℝ) :
    (1 - lr * wd) * p - lr * u = p - lr * (u + wd * p) := by
  ring

/-- T9222 — Adam step scale invariance: with exact moments (`m̂ = g`,
`v̂ = g²`) and `ε = 0`, the step magnitude is `lr` for every nonzero
gradient — the scale cancels, only the sign survives. -/
theorem t9222_adam_step_scale_invariant (g lr : ℝ) (hg : g ≠ 0) :
    |lr * (g / Real.sqrt (g ^ 2))| = |lr| := by
  rw [Real.sqrt_sq_eq_abs, abs_mul, abs_div, abs_abs,
    div_self (abs_ne_zero.mpr hg), mul_one]

end Quanta.Nn.OptimStep
