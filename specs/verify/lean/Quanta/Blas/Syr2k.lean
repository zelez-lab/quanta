/-
Level-3 BLAS SYR2K — Lean formalisation of the `quanta-blas` `syr2k`
numerical contract.

SYR2K computes, per output entry `(i, j)` of the selected triangle:

  C'[i,j] = α · (Σₚ op(A)[i,p]·op(B)[j,p] + Σₚ op(B)[i,p]·op(A)[j,p])
              + β · C[i,j]

i.e. `α · (dot(aᵢ, bⱼ) + dot(bᵢ, aⱼ)) + β · C[i,j]` — a sum of two dot
products, each a `gemmEntry`-style inner product. Two facts:

* **Exact symmetry** (`syr2kEntry_symm`): swapping `i ↔ j` swaps the two
  dot products (via `dot_comm`), leaving the entry unchanged — the reason
  computing one triangle of `C` suffices.
* The rounded entry reuses `dotRounded` per dot product, so the inner-
  product rounding analysis is the shared one (no new axiom); the
  two-term structure is a pair of gemm inner products.
-/

import Quanta.Blas.Gemm
import Quanta.Blas.Syrk

namespace Quanta.Blas

/-- Exact syr2k entry (real arithmetic): `α · (dot aᵢ bⱼ + dot bᵢ aⱼ) +
    β · c`. Mirrors `quanta_blas::reference::syr2k`. -/
def syr2kEntry (α β : ℝ) (ai bj bi aj : List ℝ) (c : ℝ) : ℝ :=
  α * (dot ai bj + dot bi aj) + β * c

/-- **Exact symmetry.** Swapping the two rows (`i ↔ j`) sends
    `(ai, bj, bi, aj)` to `(aj, bi, bj, ai)`, which by `dot_comm` on each
    term leaves the entry unchanged. This is the formal content of "C is
    symmetric, computing one triangle suffices". -/
theorem syr2kEntry_symm (α β : ℝ) (ai bj bi aj : List ℝ) (c : ℝ) :
    syr2kEntry α β ai bj bi aj c = syr2kEntry α β aj bi bj ai c := by
  unfold syr2kEntry
  rw [dot_comm ai bj, dot_comm bi aj]
  ring

/-- Floating-point syr2k entry: each of the two dot products is the
    sequentially-rounded `dotRounded` (as the two kernel loops compute), the
    two partial sums are added, then `α··`, `β·c`, and the final add are
    rounded. -/
noncomputable def syr2kEntryRounded (α β : ℝ) (ai bj bi aj : List ℝ) (c : ℝ) : ℝ :=
  roundedOp
    (roundedOp (α * (dotRounded ai bj + dotRounded bi aj)) + roundedOp (β * c))

/-- **syr2k per-entry forward-error decomposition (structural).** The total
    forward error splits, by the triangle inequality, into the final
    rounded-add error, the α-scaling error, the two inner-product errors
    (`dotRounded` vs `dot`, one per cross term), and the β·c error. Each
    inner-product term is bounded by the shared `cDot` machinery; the α/β/add
    terms are single rounded ops. No new axiom beyond the shared rounding
    model. -/
theorem syr2kEntry_error_decomp (α β : ℝ) (ai bj bi aj : List ℝ) (c : ℝ) :
    |syr2kEntryRounded α β ai bj bi aj c - syr2kEntry α β ai bj bi aj c|
      ≤ |roundedOp (roundedOp (α * (dotRounded ai bj + dotRounded bi aj))
              + roundedOp (β * c))
            - (roundedOp (α * (dotRounded ai bj + dotRounded bi aj))
              + roundedOp (β * c))|
        + |roundedOp (α * (dotRounded ai bj + dotRounded bi aj))
            - α * (dotRounded ai bj + dotRounded bi aj)|
        + |α * (dotRounded ai bj + dotRounded bi aj)
            - α * (dot ai bj + dot bi aj)|
        + |roundedOp (β * c) - β * c| := by
  unfold syr2kEntryRounded syr2kEntry
  set sR := dotRounded ai bj + dotRounded bi aj with hsR
  set sX := dot ai bj + dot bi aj with hsX
  set p := roundedOp (α * sR) with hp
  set q := roundedOp (β * c) with hq
  have e1 : roundedOp (p + q) - (α * sX + β * c)
      = (roundedOp (p + q) - (p + q))
        + (p - α * sR)
        + (α * sR - α * sX)
        + (q - β * c) := by rw [hp, hq]; ring
  calc |roundedOp (p + q) - (α * sX + β * c)|
      = |(roundedOp (p + q) - (p + q))
          + (p - α * sR)
          + (α * sR - α * sX)
          + (q - β * c)| := by rw [e1]
    _ ≤ |roundedOp (p + q) - (p + q)|
          + |p - α * sR|
          + |α * sR - α * sX|
          + |q - β * c| := by
        apply (abs_add _ _).trans
        gcongr
        apply (abs_add _ _).trans
        gcongr
        apply abs_add _ _

end Quanta.Blas
