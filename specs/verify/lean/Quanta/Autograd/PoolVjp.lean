/-
Pooling VJP correctness — avgpool (linear, an adjoint) and maxpool (the argmax
subgradient).

The Rust `quanta-autograd` pooling backwards are gather kernels. This file
proves the two facts they rest on, abstracting over the window geometry (the
concrete NCHW stride/pad arithmetic is the tested kernel; its *shape* is what we
abstract).

A pooling layer is a family of windows: each output `o` reads the inputs `i`
with `mem o i` (membership in `o`'s window). `K` is the window size (`kh·kw`).

* **avgpool** `avgpool x o = (1/K)·Σ_{i ∈ o} x i` is **linear**, and its backward
  `avgpoolBack y i = (1/K)·Σ_{o ∋ i} y o` is its **adjoint**:

      ⟨avgpool x, y⟩ = ⟨x, avgpoolBack y⟩

  a double-sum swap over `mem`. So in reverse mode the gradient into the input
  is exactly `avgpoolBack g` — the correctness of the `∂x` step.

* **maxpool** picks, per window, a winning input `arg o` (the argmax). The
  subgradient of the window max w.r.t. an input `i` is the indicator
  `[i = arg o]`, so the reverse-mode gradient routes `g o` to `arg o` and
  nowhere else — exactly what `maxpool2d_backward` does.
-/

import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring

namespace Quanta.Autograd

open scoped BigOperators

variable {O I : Type*} [Fintype O] [Fintype I] [DecidableEq I]

section AvgPool

variable (mem : O → I → Prop) [DecidableRel mem] (K : ℝ)

/-- `avgpool`: output `o` is `1/K` times the sum of the inputs in its window. -/
noncomputable def avgpool (x : I → ℝ) : O → ℝ :=
  fun o => K⁻¹ * ∑ i ∈ Finset.univ.filter (fun i => mem o i), x i

/-- `avgpoolBack`: input `i` accumulates `1/K · y o` over every window `o` that
    contains it — the gather backward. -/
noncomputable def avgpoolBack (y : O → ℝ) : I → ℝ :=
  fun i => K⁻¹ * ∑ o ∈ Finset.univ.filter (fun o => mem o i), y o

omit [DecidableEq I] in
/-- **The avgpool adjoint.** `⟨avgpool x, y⟩ = ⟨x, avgpoolBack y⟩`. Pull each
    side back to the full double sum `Σ_o Σ_i [mem o i] · K⁻¹ · x i · y o`, swap
    the order, and refold. So `avgpoolBack g` is the true input gradient. -/
theorem avgpool_adjoint (x : I → ℝ) (y : O → ℝ) :
    (∑ o, avgpool mem K x o * y o) = ∑ i, x i * avgpoolBack mem K y i := by
  -- Each side equals the guarded double sum Σ Σ [mem o i] · K⁻¹·x i·y o; the two
  -- orders are equal by `Finset.sum_comm`.
  have key : ∀ o i, (if mem o i then x i else 0) * (K⁻¹ * y o)
      = (if mem o i then K⁻¹ * x i * y o else 0) := by
    intro o i
    by_cases h : mem o i
    · rw [if_pos h, if_pos h]
      ring
    · rw [if_neg h, if_neg h, zero_mul]
  have hL : (∑ o, avgpool mem K x o * y o)
      = ∑ o, ∑ i, (if mem o i then K⁻¹ * x i * y o else 0) := by
    apply Finset.sum_congr rfl
    intro o _
    unfold avgpool
    rw [Finset.sum_filter]
    -- (K⁻¹ · Σ ite) · y = Σ (ite · (K⁻¹ · y)) = Σ ite(…)
    have : (K⁻¹ * ∑ i, (if mem o i then x i else 0)) * y o
        = ∑ i, (if mem o i then x i else 0) * (K⁻¹ * y o) := by
      rw [← Finset.sum_mul]
      ring
    rw [this]
    apply Finset.sum_congr rfl
    intro i _
    exact key o i
  have hR : (∑ i, x i * avgpoolBack mem K y i)
      = ∑ o, ∑ i, (if mem o i then K⁻¹ * x i * y o else 0) := by
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro i _
    unfold avgpoolBack
    rw [Finset.sum_filter]
    -- x i · (K⁻¹ · Σ ite) = Σ (ite · (K⁻¹ · y))
    have : x i * (K⁻¹ * ∑ o, (if mem o i then y o else 0))
        = ∑ o, (if mem o i then x i else 0) * (K⁻¹ * y o) := by
      rw [← mul_assoc, Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro o _
      by_cases h : mem o i
      · rw [if_pos h, if_pos h]
        ring
      · simp [h]
    rw [this]
    apply Finset.sum_congr rfl
    intro o _
    exact key o i
  rw [hL, hR]

end AvgPool

section MaxPool

variable (arg : O → I)

/-- `maxpoolBack`: route each output's gradient to its window's argmax input.
    `maxpoolBack g i = Σ_{o : arg o = i} g o`. -/
def maxpoolBack (g : O → ℝ) : I → ℝ :=
  fun i => ∑ o ∈ Finset.univ.filter (fun o => arg o = i), g o

omit [Fintype I] in
/-- The window max is differentiable in the winning input with derivative 1 and
    in every other input with derivative 0: the subgradient is the indicator
    `[i = arg o]`. Reverse mode therefore sends `g o` to `arg o`. We state this
    as: pairing `g` against that indicator over the inputs recovers
    `maxpoolBack`, i.e. the routed gradient **is** the indicator-weighted sum. -/
theorem maxpool_routes_to_argmax (g : O → ℝ) (i : I) :
    maxpoolBack arg g i = ∑ o, (if arg o = i then g o else 0) := by
  unfold maxpoolBack
  rw [Finset.sum_filter]

/-- Equivalently, the maxpool VJP is the adjoint of the (linear-on-the-active-
    cell) forward selector `sel x o = x (arg o)`: `⟨sel x, g⟩ = ⟨x, maxpoolBack g⟩`.
    This is the exact reverse-mode duality the backward kernel implements. -/
theorem maxpool_selector_adjoint (x : I → ℝ) (g : O → ℝ) :
    (∑ o, x (arg o) * g o) = ∑ i, x i * maxpoolBack arg g i := by
  have hR : (∑ i, x i * maxpoolBack arg g i)
      = ∑ i, ∑ o, (if arg o = i then x i * g o else 0) := by
    apply Finset.sum_congr rfl
    intro i _
    rw [maxpool_routes_to_argmax, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro o _
    by_cases h : arg o = i <;> simp [h]
  rw [hR, Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro o _
  -- ∑ i, [arg o = i] · x i · g o  =  x (arg o) · g o
  rw [Finset.sum_eq_single (arg o)]
  · simp
  · intro b _ hb
    simp [Ne.symm hb]
  · intro h; exact absurd (Finset.mem_univ (arg o)) h

end MaxPool

end Quanta.Autograd
