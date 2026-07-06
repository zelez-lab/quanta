/-
Level-3 BLAS TRMM — Lean formalisation of the `quanta-blas` `trmm`
numerical contract.

TRMM computes, per output entry `i` of a B-vector:

  B'[i] = α · (Σ_{p ∈ tri(i)} M[i,p]·B[p])

where `M = op(A)` is triangular and the sum ranges over `i`'s triangle.
This is a `gemmEntry` with `β = 0`: `α · dot(mᵢ, bvec) + 0 · c`, where `mᵢ`
is row `i` of `M` (restricted to the triangle) and `bvec` the B-vector. So
the GEMM per-entry forward-error bound (`Quanta.Blas.gemmEntry_error_decomp`)
specialises verbatim — no new axioms.
-/

import Quanta.Blas.Gemm

namespace Quanta.Blas

/-- Exact trmm entry (real arithmetic): `α · dot(mi, bvec)` — the triangular
    matrix-vector product scaled by `α`. Written as a `gemmEntry` with
    `β = 0` and `c = 0`. Mirrors `quanta_blas::reference::trmm`. -/
def trmmEntry (α : ℝ) (mi bvec : List ℝ) : ℝ :=
  α * dot mi bvec

/-- The trmm entry IS a gemm entry with `β = c = 0`. -/
theorem trmmEntry_eq_gemmEntry (α : ℝ) (mi bvec : List ℝ) :
    trmmEntry α mi bvec = gemmEntry α 0 mi bvec 0 := by
  unfold trmmEntry gemmEntry
  ring

/-- Floating-point trmm entry — the sequentially-rounded inner product
    scaled and rounded, matching the kernel (no `β·c` term). -/
noncomputable def trmmEntryRounded (α : ℝ) (mi bvec : List ℝ) : ℝ :=
  roundedOp (α * dotRounded mi bvec)

/-- **trmm per-entry forward-error decomposition.** The forward error splits
    into the α-scaling rounding error and the inner-product error
    (`dotRounded` vs `dot`) — the `β·c` terms of the gemm decomposition
    vanish. -/
theorem trmmEntry_error_decomp (α : ℝ) (mi bvec : List ℝ) :
    |trmmEntryRounded α mi bvec - trmmEntry α mi bvec|
      ≤ |roundedOp (α * dotRounded mi bvec) - α * dotRounded mi bvec|
        + |α * dotRounded mi bvec - α * dot mi bvec| := by
  unfold trmmEntryRounded trmmEntry
  set p := roundedOp (α * dotRounded mi bvec) with hp
  have e1 : p - α * dot mi bvec
      = (p - α * dotRounded mi bvec) + (α * dotRounded mi bvec - α * dot mi bvec) := by
    ring
  calc |p - α * dot mi bvec|
      = |(p - α * dotRounded mi bvec) + (α * dotRounded mi bvec - α * dot mi bvec)| := by
        rw [e1]
    _ ≤ |p - α * dotRounded mi bvec| + |α * dotRounded mi bvec - α * dot mi bvec| :=
        abs_add _ _

end Quanta.Blas
