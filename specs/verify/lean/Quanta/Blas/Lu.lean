/-
LU factorisation with partial pivoting (getrf) — Lean formalisation of the
`quanta-blas` per-step numerical contract.

The right-looking factorisation `P·A = L·U` computes, column by column, two
kinds of entry:

  U[k,j] = A[k,j] − Σ_{p<k} L[k,p]·U[p,j]              (row of U, no divide)
  L[i,k] = ( A[i,k] − Σ_{p<k} L[i,p]·U[p,k] ) / U[k,k]  (column of L, divide)

Both are the `(α·b − Σ aₚ·xₚ)/d` shape of the triangular-solve substitution
step (with α = 1): the U-row entry is the unit-`d` case (no divide), the
L-column entry the general case (divide by the pivot `U[k,k]`). So the exact
residuals and the per-entry rounding decomposition reuse the
`Quanta.Blas.Reference` / `Quanta.Blas.Triangular` (`trsvStep`, `dot`,
`subAccRounded`, `roundedOp`) machinery, exactly as `Cholesky.lean` does for
its off-diagonal step. Partial pivoting is a permutation of the input rows and
does not change the per-entry arithmetic — it is orthogonal to this contract.

This file proves, for the factorisation steps:

1. `luRowStep_residual` — the U-row step reconstructs `A[k,j]` exactly:
   `Σ_{p<k} L[k,p]·U[p,j] + U[k,j] = A[k,j]`.
2. `luColStep_residual` — the L-column step makes entry `(i,k)`'s defining
   equation hold exactly: `Σ_{p<k} L[i,p]·U[p,k] + U[k,k]·L[i,k] = A[i,k]`.
   Composing the two over the matrix gives `P·A = L·U` entrywise.
3. `luColStep_error_decomp` — the per-entry rounding decomposition (same
   triangle-inequality structure as `trsvStep_error_decomp` /
   `cholColStep_error_decomp`): the rounded multiplier strays from the exact
   one by at most the final-divide rounding plus the accumulated numerator
   error scaled by `1/|U[k,k]|`.

The whole-factorisation norm-wise Higham backward-error bound (Thm 9.3,
`‖ΔA‖ ≤ γ_n · |L|·|U|` with the growth factor) is the flagged follow-up,
exactly as the trsv whole-solve chain is in `Triangular.lean`.
-/

import Quanta.Blas.Reference
import Quanta.Blas.Triangular

namespace Quanta.Blas

/-- Exact U-row LU step (real arithmetic): `U[k,j] = A[k,j] − Σ_{p<k}
    L[k,p]·U[p,j]`. No divide — the unit-`d` triangular substitution step. -/
noncomputable def luRowStep (a : ℝ) (lk up : List ℝ) : ℝ :=
  a - dot lk up

/-- The U-row step is the unit-diagonal triangular substitution step. -/
theorem luRowStep_eq_trsvStep (a : ℝ) (lk up : List ℝ) :
    luRowStep a lk up = trsvStep 1 a lk up 1 := by
  rw [trsvStep_unit, one_mul, luRowStep]

/-- **U-row per-entry exact correctness (residual form).** The computed
    `U[k,j]` satisfies the `P·A = L·U` equation at `(k,j)` exactly:

      Σ_{p<k} L[k,p]·U[p,j] + U[k,j] = A[k,j]. -/
theorem luRowStep_residual (a : ℝ) (lk up : List ℝ) :
    dot lk up + luRowStep a lk up = a := by
  unfold luRowStep
  ring

/-- Exact L-column LU step (real arithmetic): the multiplier `L[i,k] =
    (A[i,k] − Σ_{p<k} L[i,p]·U[p,k]) / U[k,k]`. Identical shape to
    `trsvStep 1 a li uk d`. -/
noncomputable def luColStep (a : ℝ) (li uk : List ℝ) (d : ℝ) : ℝ :=
  (a - dot li uk) / d

/-- The L-column step is exactly a unit-`α` triangular substitution step. -/
theorem luColStep_eq_trsvStep (a : ℝ) (li uk : List ℝ) (d : ℝ) :
    luColStep a li uk d = trsvStep 1 a li uk d := by
  unfold luColStep trsvStep
  rw [one_mul]

/-- **L-column per-entry exact correctness (residual form).** For a nonzero
    pivot, the computed `L[i,k]` satisfies the `P·A = L·U` equation at
    `(i,k)` exactly:

      Σ_{p<k} L[i,p]·U[p,k] + U[k,k]·L[i,k] = A[i,k].

    Together with `luRowStep_residual` this gives `P·A = L·U` entrywise. -/
theorem luColStep_residual (a d : ℝ) (li uk : List ℝ) (hd : d ≠ 0) :
    dot li uk + d * luColStep a li uk d = a := by
  rw [luColStep_eq_trsvStep]
  have h := trsvStep_residual 1 a d li uk hd
  rwa [one_mul] at h

/-- Floating-point L-column step, rounding each elementary op exactly as the
    `lu_elim_f32` kernel does: the subtract-fold rounds each multiply and
    subtract over the two already-computed lists, and the divide by the pivot
    is rounded once. Reuses `subAccRounded` from `Triangular`. -/
noncomputable def luColStepRounded (a : ℝ) (li uk : List ℝ) (d : ℝ) : ℝ :=
  roundedOp (subAccRounded a li uk / d)

/-- **L-column per-entry forward-error decomposition (structural).** The
    rounded multiplier strays from the exact one by at most the final-divide
    rounding plus the accumulated numerator error scaled by `1/|d|`:

      |L̂[i,k] − L[i,k]| ≤ |round(q/d) − q/d| + |q − (a − dot li uk)| / |d|

    where `q = subAccRounded a li uk` is the rounded numerator. Pure triangle
    inequality — same shape as `trsvStep_error_decomp`,
    `gemmEntry_error_decomp`, and `cholColStep_error_decomp`. The numerator
    term decomposes further via `subAcc_exact` + the `dotRounded` machinery;
    the whole-factorisation growth-factor bound is the flagged follow-up. -/
theorem luColStep_error_decomp (a d : ℝ) (li uk : List ℝ) :
    |luColStepRounded a li uk d - luColStep a li uk d|
      ≤ |roundedOp (subAccRounded a li uk / d) - subAccRounded a li uk / d|
        + |subAccRounded a li uk - (a - dot li uk)| / |d| := by
  set q := subAccRounded a li uk with hq
  unfold luColStepRounded luColStep
  rw [← hq]
  have e : roundedOp (q / d) - (a - dot li uk) / d
      = (roundedOp (q / d) - q / d) + (q - (a - dot li uk)) / d := by
    ring
  rw [e]
  refine (abs_add _ _).trans ?_
  gcongr
  rw [abs_div]

end Quanta.Blas
