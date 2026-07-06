/-
Level-3 BLAS SYMM — Lean formalisation of the `quanta-blas` `symm`
numerical contract.

SYMM computes, per output entry `(i, j)`:

  C'[i,j] = α · (Σₚ Asym[i,p]·B[p,j]) + β · C[i,j]      (side = Left)

where `Asym` is the full symmetric matrix reconstructed from the stored
triangle. This is **per-entry identical to GEMM**: `C'[i,j]` is
`α · dot(aᵢ, bⱼ) + β · C[i,j]`, the `gemmEntry` formula with row `i` of
`Asym` and column `j` of `B` as the two vectors. So the GEMM per-entry
forward-error bound (`Quanta.Blas.gemmEntry_error_decomp`) transfers
verbatim — no new axioms, no new analysis.
-/

import Quanta.Blas.Gemm

namespace Quanta.Blas

/-- Exact symm entry (real arithmetic): `α · dot(ai, bj) + β · c`, where
    `ai` is row `i` of the reconstructed symmetric `A` and `bj` is column
    `j` of `B`. Mirrors the per-entry math of `quanta_blas::reference::symm`. -/
def symmEntry (α β : ℝ) (ai bj : List ℝ) (c : ℝ) : ℝ :=
  α * dot ai bj + β * c

/-- The symm entry IS the gemm entry — same formula. -/
theorem symmEntry_eq_gemmEntry (α β : ℝ) (ai bj : List ℝ) (c : ℝ) :
    symmEntry α β ai bj c = gemmEntry α β ai bj c := rfl

/-- Floating-point symm entry — defined as the gemm entry, so the rounded
    computation and its error analysis are shared. -/
noncomputable def symmEntryRounded (α β : ℝ) (ai bj : List ℝ) (c : ℝ) : ℝ :=
  gemmEntryRounded α β ai bj c

/-- The symm magnitude budget is the gemm magnitude budget. -/
def symmEntryMagnitude (α β : ℝ) (ai bj : List ℝ) (c : ℝ) : ℝ :=
  gemmEntryMagnitude α β ai bj c

/-- The symm-entry magnitude budget is non-negative (inherited). -/
theorem symmEntryMagnitude_nonneg (α β : ℝ) (ai bj : List ℝ) (c : ℝ) :
    0 ≤ symmEntryMagnitude α β ai bj c :=
  gemmEntryMagnitude_nonneg α β ai bj c

/-- **symm per-entry forward-error decomposition.** Identical to the gemm
    decomposition, since a symm entry is a gemm entry. -/
theorem symmEntry_error_decomp (α β : ℝ) (ai bj : List ℝ) (c : ℝ) :
    |symmEntryRounded α β ai bj c - symmEntry α β ai bj c|
      ≤ |roundedOp (roundedOp (α * dotRounded ai bj) + roundedOp (β * c))
            - (roundedOp (α * dotRounded ai bj) + roundedOp (β * c))|
        + |roundedOp (α * dotRounded ai bj) - α * dotRounded ai bj|
        + |α * dotRounded ai bj - α * dot ai bj|
        + |roundedOp (β * c) - β * c| :=
  gemmEntry_error_decomp α β ai bj c

end Quanta.Blas
