/-
Triangular solves (trsv / trsm) — Lean formalisation of the `quanta-blas`
substitution-step numerical contract.

Both `trsv` and `trsm` reduce to independent substitution lanes; each lane
computes, per solved element `i` (forward or backward order):

  x[i] = (α·b[i] − Σₚ M[i,p]·x[p]) / d       (d = M[i,i], or 1 for unit diag)

over the already-solved prefix (forward) / suffix (backward). This file
proves, for that **substitution step**:

1. `trsvStep_residual` — exact-arithmetic correctness: the computed `x[i]`
   makes row `i`'s equation hold exactly (`Σₚ M[i,p]·x[p] + d·x[i] = α·b[i]`).
   This is the invariant each substitution step maintains; composing it
   along the (forward or backward) sweep yields `op(A)·x = α·b`.
2. `subAcc_exact` — the kernel's subtract-fold accumulation shape equals
   `init − dot as xs`, tying the kernel's op order to the `dot`
   formalisation from `Quanta.Blas.Reference`.
3. `trsvStep_error_decomp` — the per-step rounding decomposition: the
   floating-point step (each elementary op rounded, in the kernel's exact
   order: rounded α·b, sequential rounded subtract-fold, rounded divide)
   stays within the final-divide rounding plus the accumulated numerator
   error scaled by `1/|d|`. Triangle-inequality structural bound, same
   shape as `gemmEntry_error_decomp`.
4. `trsvStep_unit` — the unit-diagonal case drops the divide.

FLAGGED FOLLOW-UP (structural statement shipped, full bound not yet
proven — deliberately NOT stated with `sorry`): the **whole-solve** Higham
backward-error bound (Higham, *Accuracy and Stability of Numerical
Algorithms*, Thm 8.5: `(T + ΔT)·x̂ = b` with `|ΔT| ≤ γₙ·|T|`) requires
propagating the per-step bound through the sequential dependency chain
(each later step consumes earlier computed x̂ values). The per-step
decomposition proven here is the induction step's local content; the
chain composition is future work.
-/

import Quanta.Blas.Reference

namespace Quanta.Blas

/-- Exact substitution step (real arithmetic): solve element `i` of a
    triangular system given the already-solved values `xs` paired with
    the row coefficients `as`: `x_i = (α·b − dot as xs) / d`. Mirrors one
    iteration of the `trsm_fwd_f32` / `trsm_bwd_f32` kernel loops and of
    `quanta_blas::reference::trsm`'s lane solve. -/
noncomputable def trsvStep (α b : ℝ) (as xs : List ℝ) (d : ℝ) : ℝ :=
  (α * b - dot as xs) / d

/-- **Per-step exact correctness (residual form).** For a nonzero
    diagonal, the computed element satisfies row `i`'s equation exactly:
    the off-diagonal contribution plus `d` times the solved value
    reconstructs the scaled RHS. Composing this along the sweep is what
    makes the whole substitution a solve of `op(A)·x = α·b`. -/
theorem trsvStep_residual (α b d : ℝ) (as xs : List ℝ) (hd : d ≠ 0) :
    dot as xs + d * trsvStep α b as xs d = α * b := by
  unfold trsvStep
  field_simp

/-- Unit-diagonal step: dividing by 1 is the identity, so the step is the
    bare numerator — exactly what the `trsm_*_unit_f32` kernels compute
    (no diagonal load, no divide). -/
theorem trsvStep_unit (α b : ℝ) (as xs : List ℝ) :
    trsvStep α b as xs 1 = α * b - dot as xs := by
  unfold trsvStep
  rw [div_one]

-- ── The kernel's accumulation shape ─────────────────────────────────

/-- Exact subtract-fold: the kernel's accumulation order
    (`acc ← acc − aₚ·xₚ`, left to right), in real arithmetic. -/
def subAcc : ℝ → List ℝ → List ℝ → ℝ
  | acc, [], _ => acc
  | acc, _, [] => acc
  | acc, a :: as, x :: xs => subAcc (acc - a * x) as xs

/-- The subtract-fold equals `init − dot as xs`: the kernel's op order
    computes exactly the substitution numerator. -/
theorem subAcc_exact (init : ℝ) :
    ∀ (as xs : List ℝ), subAcc init as xs = init - dot as xs := by
  intro as
  induction as generalizing init with
  | nil => intro xs; cases xs <;> simp [subAcc, dot]
  | cons a as ih =>
      intro xs
      cases xs with
      | nil => simp [subAcc, dot]
      | cons x xs =>
          unfold subAcc dot
          simp only [List.zipWith_cons_cons, List.sum_cons]
          rw [ih (init - a * x) xs]
          unfold dot
          ring

/-- Floating-point subtract-fold: every multiply and every subtract is
    rounded, in the kernel's sequential order. -/
noncomputable def subAccRounded : ℝ → List ℝ → List ℝ → ℝ
  | acc, [], _ => acc
  | acc, _, [] => acc
  | acc, a :: as, x :: xs =>
      subAccRounded (roundedOp (acc - roundedOp (a * x))) as xs

/-- Floating-point substitution step, rounding each elementary op exactly
    as the kernel does: `round(α·b)` seeds the accumulator, the
    subtract-fold rounds each multiply and subtract, and the divide is
    rounded once at the end. -/
noncomputable def trsvStepRounded (α b : ℝ) (as xs : List ℝ) (d : ℝ) : ℝ :=
  roundedOp (subAccRounded (roundedOp (α * b)) as xs / d)

/-- **Per-step forward-error decomposition (structural).** The rounded
    step strays from the exact step by at most the final-divide rounding
    plus the accumulated numerator error scaled by `1/|d|`:

      |step̂ − step| ≤ |round(q/d) − q/d| + |q − (α·b − dot as xs)| / |d|

    where `q` is the rounded numerator. Pure triangle inequality — the
    numerator term decomposes further via `subAcc_exact` and the
    `dotRounded` machinery; the whole-solve chain composition is the
    flagged follow-up (see the file header). -/
theorem trsvStep_error_decomp (α b d : ℝ) (as xs : List ℝ) :
    |trsvStepRounded α b as xs d - trsvStep α b as xs d|
      ≤ |roundedOp (subAccRounded (roundedOp (α * b)) as xs / d)
            - subAccRounded (roundedOp (α * b)) as xs / d|
        + |subAccRounded (roundedOp (α * b)) as xs - (α * b - dot as xs)| / |d| := by
  set q := subAccRounded (roundedOp (α * b)) as xs with hq
  unfold trsvStepRounded trsvStep
  rw [← hq]
  have e : roundedOp (q / d) - (α * b - dot as xs) / d
      = (roundedOp (q / d) - q / d) + (q - (α * b - dot as xs)) / d := by
    ring
  calc |roundedOp (q / d) - (α * b - dot as xs) / d|
      = |(roundedOp (q / d) - q / d) + (q - (α * b - dot as xs)) / d| := by rw [e]
    _ ≤ |roundedOp (q / d) - q / d| + |(q - (α * b - dot as xs)) / d| :=
        abs_add _ _
    _ = |roundedOp (q / d) - q / d| + |q - (α * b - dot as xs)| / |d| := by
        rw [abs_div]

end Quanta.Blas
