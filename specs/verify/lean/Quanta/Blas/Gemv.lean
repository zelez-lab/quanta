/-
Level-2 BLAS GEMV — Lean formalisation of the `quanta-blas` `gemv`
numerical contract.

GEMV computes, per output entry `i`:

  y'[i] = α · (Σⱼ A[i,j]·x[j]) + β · y[i]

This is **per-entry identical to GEMM**: `y'[i]` is `α · dot(row_i, x) +
β · y[i]`, the same `gemmEntry` formula with the matrix row as `a`, the
vector `x` as `b`, and the incoming `y[i]` as `c`. So the GEMM per-entry
forward-error bound (Higham §3.5, `Quanta.Blas.gemmEntry_error_decomp`)
transfers verbatim. No new axioms, no new analysis — gemv reuses the gemm
entry contract through a definitional equality.
-/

import Quanta.Blas.Gemm

namespace Quanta.Blas

/-- Exact gemv entry (real arithmetic): `α · dot(row, x) + β · y`, where
    `row` is a row of A, `x` the input vector, `y` the incoming `y[i]`.
    Mirrors the per-entry math of `quanta_blas::reference::gemv`. -/
def gemvEntry (α β : ℝ) (row x : List ℝ) (y : ℝ) : ℝ :=
  α * dot row x + β * y

/-- The gemv entry IS the gemm entry — same formula, with the matrix row as
    `a`, the input vector as `b`, and the incoming `y[i]` as `c`. -/
theorem gemvEntry_eq_gemmEntry (α β : ℝ) (row x : List ℝ) (y : ℝ) :
    gemvEntry α β row x y = gemmEntry α β row x y := rfl

/-- Floating-point gemv entry — defined as the gemm entry, so the rounded
    computation and its error analysis are shared. -/
noncomputable def gemvEntryRounded (α β : ℝ) (row x : List ℝ) (y : ℝ) : ℝ :=
  gemmEntryRounded α β row x y

/-- The gemv magnitude budget is the gemm magnitude budget. -/
def gemvEntryMagnitude (α β : ℝ) (row x : List ℝ) (y : ℝ) : ℝ :=
  gemmEntryMagnitude α β row x y

/-- The gemv-entry magnitude budget is non-negative (inherited from gemm). -/
theorem gemvEntryMagnitude_nonneg (α β : ℝ) (row x : List ℝ) (y : ℝ) :
    0 ≤ gemvEntryMagnitude α β row x y :=
  gemmEntryMagnitude_nonneg α β row x y

/-- **gemv per-entry forward-error decomposition.** Identical to the gemm
    decomposition (`gemmEntry_error_decomp`), since a gemv entry is a gemm
    entry. The total forward error is bounded by the final rounded-add error,
    the α-scaling error, the inner-product error of `dotRounded` vs `dot`, and
    the β·y error — each a single rounded op or the carried inner-product
    budget. -/
theorem gemvEntry_error_decomp (α β : ℝ) (row x : List ℝ) (y : ℝ) :
    |gemvEntryRounded α β row x y - gemvEntry α β row x y|
      ≤ |roundedOp (roundedOp (α * dotRounded row x) + roundedOp (β * y))
            - (roundedOp (α * dotRounded row x) + roundedOp (β * y))|
        + |roundedOp (α * dotRounded row x) - α * dotRounded row x|
        + |α * dotRounded row x - α * dot row x|
        + |roundedOp (β * y) - β * y| :=
  gemmEntry_error_decomp α β row x y

end Quanta.Blas
