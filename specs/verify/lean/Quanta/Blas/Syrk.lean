/-
Level-3 BLAS SYRK — Lean formalisation of the `quanta-blas` `syrk`
numerical contract.

SYRK computes, per output entry `(i, j)` of the selected triangle:

  C'[i,j] = α · (Σₚ op(A)[i,p]·op(A)[j,p]) + β · C[i,j]

This is **per-entry identical to GEMM**: `C'[i,j]` is `α · dot(aᵢ, aⱼ) +
β · C[i,j]`, the `gemmEntry` formula with row `i` and row `j` of `op(A)`
as the two vectors. So the GEMM per-entry forward-error bound (Higham
§3.5, `Quanta.Blas.gemmEntry_error_decomp`) transfers verbatim — no new
axioms, no new analysis.

On top of the transfer, this file proves the symmetry fact specific to
syrk: in exact arithmetic the entry is symmetric in the two rows
(`syrkEntry_symm`, via `dot_comm`) — the formal content of "C is
symmetric, computing one triangle suffices".
-/

import Quanta.Blas.Gemm

namespace Quanta.Blas

/-- The inner product is symmetric: `dot xs ys = dot ys xs`. Induction on
    both lists; the cons case is `mul_comm` plus the tail. -/
theorem dot_comm : ∀ (xs ys : List ℝ), dot xs ys = dot ys xs
  | [], [] => rfl
  | [], _ :: _ => by simp [dot]
  | _ :: _, [] => by simp [dot]
  | x :: xs, y :: ys => by
      unfold dot
      simp only [List.zipWith_cons_cons, List.sum_cons]
      have ih := dot_comm xs ys
      unfold dot at ih
      rw [ih, mul_comm]

/-- Exact syrk entry (real arithmetic): `α · dot(aᵢ, aⱼ) + β · c`, where
    `ai`/`aj` are rows `i` and `j` of `op(A)` and `c` is the incoming
    `C[i,j]`. Mirrors the per-entry math of
    `quanta_blas::reference::syrk`. -/
def syrkEntry (α β : ℝ) (ai aj : List ℝ) (c : ℝ) : ℝ :=
  α * dot ai aj + β * c

/-- The syrk entry IS the gemm entry — same formula, with row `i` of
    `op(A)` as `a` and row `j` of `op(A)` as `b`. -/
theorem syrkEntry_eq_gemmEntry (α β : ℝ) (ai aj : List ℝ) (c : ℝ) :
    syrkEntry α β ai aj c = gemmEntry α β ai aj c := rfl

/-- **Exact symmetry.** Swapping the two rows leaves the exact syrk entry
    unchanged — the reason computing one triangle of C suffices. -/
theorem syrkEntry_symm (α β : ℝ) (ai aj : List ℝ) (c : ℝ) :
    syrkEntry α β ai aj c = syrkEntry α β aj ai c := by
  unfold syrkEntry
  rw [dot_comm]

/-- Floating-point syrk entry — defined as the gemm entry, so the rounded
    computation and its error analysis are shared. -/
noncomputable def syrkEntryRounded (α β : ℝ) (ai aj : List ℝ) (c : ℝ) : ℝ :=
  gemmEntryRounded α β ai aj c

/-- The syrk magnitude budget is the gemm magnitude budget. -/
def syrkEntryMagnitude (α β : ℝ) (ai aj : List ℝ) (c : ℝ) : ℝ :=
  gemmEntryMagnitude α β ai aj c

/-- The syrk-entry magnitude budget is non-negative (inherited). -/
theorem syrkEntryMagnitude_nonneg (α β : ℝ) (ai aj : List ℝ) (c : ℝ) :
    0 ≤ syrkEntryMagnitude α β ai aj c :=
  gemmEntryMagnitude_nonneg α β ai aj c

/-- **syrk per-entry forward-error decomposition.** Identical to the gemm
    decomposition (`gemmEntry_error_decomp`), since a syrk entry is a gemm
    entry: the total forward error is bounded by the final rounded-add
    error, the α-scaling error, the inner-product error of `dotRounded`
    vs `dot`, and the β·c error. -/
theorem syrkEntry_error_decomp (α β : ℝ) (ai aj : List ℝ) (c : ℝ) :
    |syrkEntryRounded α β ai aj c - syrkEntry α β ai aj c|
      ≤ |roundedOp (roundedOp (α * dotRounded ai aj) + roundedOp (β * c))
            - (roundedOp (α * dotRounded ai aj) + roundedOp (β * c))|
        + |roundedOp (α * dotRounded ai aj) - α * dotRounded ai aj|
        + |α * dotRounded ai aj - α * dot ai aj|
        + |roundedOp (β * c) - β * c| :=
  gemmEntry_error_decomp α β ai aj c

end Quanta.Blas
