/-
Cholesky factorisation (potrf) — Lean formalisation of the `quanta-blas`
per-entry numerical contract.

The factorisation `A = L·Lᵀ` (lower) computes, column by column:

  L[j,j] = sqrt( A[j,j] − Σ_{p<j} L[j,p]² )          (diagonal step)
  L[i,j] = ( A[i,j] − Σ_{p<j} L[i,p]·L[j,p] ) / L[j,j]   (i > j, off-diagonal)

Each off-diagonal entry has exactly the `(α·b − Σ aₚ·xₚ)/d` shape of the
triangular-solve substitution step (with α = 1, b = A[i,j], the fold over the
already-computed columns, d = L[j,j]) — so the exact residual and the
per-entry rounding decomposition reuse the `Quanta.Blas.Reference` `dot` /
`roundedOp` machinery and mirror `Triangular.lean`. The diagonal step is the
new content: its exact residual is the defining Cholesky identity
`L[j,j]² + Σ_{p<j} L[j,p]² = A[j,j]`.

This file proves, for the factorisation steps:

1. `cholDiagStep_residual` — the diagonal step reconstructs `A[j,j]` exactly:
   `L[j,j]² + (Σ_{p<j} L[j,p]²) = A[j,j]` when the radicand is nonnegative.
2. `cholColStep_residual` — the off-diagonal step makes row `i` column `j`'s
   defining equation hold exactly: `dot Li Lj + L[j,j]·L[i,j] = A[i,j]`.
3. `cholColStep_error_decomp` — the per-entry rounding decomposition (same
   triangle-inequality structural bound as `trsvStep_error_decomp` /
   `gemmEntry_error_decomp`): the rounded off-diagonal entry strays from the
   exact one by at most the final-divide rounding plus the accumulated
   numerator error scaled by `1/|L[j,j]|`.

The whole-factorisation norm-wise Higham backward-error bound (Thm 10.3,
`‖ΔA‖ ≤ γ_{n+1}‖A‖` under the accumulated-error composition) is the flagged
follow-up, exactly as the trsv whole-solve chain is in `Triangular.lean`.
-/

import Quanta.Blas.Reference
import Quanta.Blas.Triangular

namespace Quanta.Blas

/-- Exact off-diagonal Cholesky step (real arithmetic): the sub-diagonal
    entry `L[i,j]` from the row-`i` and row-`j` already-computed columns
    (`li`, `lj`) and the diagonal `d = L[j,j]`. Identical shape to
    `trsvStep 1 a li lj d`; kept as its own definition for readability. -/
noncomputable def cholColStep (a : ℝ) (li lj : List ℝ) (d : ℝ) : ℝ :=
  (a - dot li lj) / d

/-- The off-diagonal step is exactly a unit-`α` triangular substitution
    step — so everything proved about `trsvStep` transfers. -/
theorem cholColStep_eq_trsvStep (a : ℝ) (li lj : List ℝ) (d : ℝ) :
    cholColStep a li lj d = trsvStep 1 a li lj d := by
  unfold cholColStep trsvStep
  rw [one_mul]

/-- **Off-diagonal per-entry exact correctness (residual form).** For a
    nonzero diagonal, the computed `L[i,j]` satisfies the defining equation
    of `A = L·Lᵀ` at position `(i,j)` exactly:

      Σ_{p<j} L[i,p]·L[j,p] + L[j,j]·L[i,j] = A[i,j].

    Composing this over the lower triangle reconstructs `A` — the exact
    factorisation identity. -/
theorem cholColStep_residual (a d : ℝ) (li lj : List ℝ) (hd : d ≠ 0) :
    dot li lj + d * cholColStep a li lj d = a := by
  rw [cholColStep_eq_trsvStep]
  have h := trsvStep_residual 1 a d li lj hd
  rwa [one_mul] at h

/-- Exact diagonal Cholesky step (real arithmetic): `L[j,j] = sqrt(A[j,j] −
    Σ_{p<j} L[j,p]²)`. The radicand is `a − dot lj lj` (the dot of the
    already-computed row-`j` column with itself is `Σ L[j,p]²`). -/
noncomputable def cholDiagStep (a : ℝ) (lj : List ℝ) : ℝ :=
  Real.sqrt (a - dot lj lj)

/-- **Diagonal per-entry exact correctness (residual form).** When the
    radicand is nonnegative (the positive-definiteness precondition at this
    step), the diagonal reconstructs `A[j,j]` exactly:

      L[j,j]² + Σ_{p<j} L[j,p]² = A[j,j].

    This is the defining Cholesky identity on the diagonal; together with
    `cholColStep_residual` it gives `A = L·Lᵀ` entrywise. -/
theorem cholDiagStep_residual (a : ℝ) (lj : List ℝ) (h : 0 ≤ a - dot lj lj) :
    cholDiagStep a lj ^ 2 + dot lj lj = a := by
  unfold cholDiagStep
  rw [Real.sq_sqrt h]
  ring

/-- Floating-point off-diagonal step, rounding each elementary op exactly as
    the `chol_col_*_f32` kernel does: the accumulator is seeded with the
    (already-stored) `A[i,j]`, the subtract-fold rounds each multiply and
    subtract over the two columns, and the divide by `L[j,j]` is rounded
    once. Reuses `subAccRounded` from `Triangular`. -/
noncomputable def cholColStepRounded (a : ℝ) (li lj : List ℝ) (d : ℝ) : ℝ :=
  roundedOp (subAccRounded a li lj / d)

/-- **Off-diagonal per-entry forward-error decomposition (structural).** The
    rounded entry strays from the exact entry by at most the final-divide
    rounding plus the accumulated numerator error scaled by `1/|d|`:

      |ĉ − c| ≤ |round(q/d) − q/d| + |q − (a − dot li lj)| / |d|

    where `q = subAccRounded a li lj` is the rounded numerator. Pure triangle
    inequality — same shape as `trsvStep_error_decomp` and
    `gemmEntry_error_decomp`. The numerator term decomposes further via
    `subAcc_exact` + the `dotRounded` machinery; the whole-factorisation
    chain is the flagged follow-up. -/
theorem cholColStep_error_decomp (a d : ℝ) (li lj : List ℝ) :
    |cholColStepRounded a li lj d - cholColStep a li lj d|
      ≤ |roundedOp (subAccRounded a li lj / d) - subAccRounded a li lj / d|
        + |subAccRounded a li lj - (a - dot li lj)| / |d| := by
  set q := subAccRounded a li lj with hq
  unfold cholColStepRounded cholColStep
  rw [← hq]
  have e : roundedOp (q / d) - (a - dot li lj) / d
      = (roundedOp (q / d) - q / d) + (q - (a - dot li lj)) / d := by
    ring
  rw [e]
  refine (abs_add _ _).trans ?_
  gcongr
  rw [abs_div]

end Quanta.Blas
