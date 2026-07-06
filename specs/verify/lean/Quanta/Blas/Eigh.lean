/-
Symmetric eigendecomposition (eigh) — Lean formalisation of the per-step
Jacobi contract.

The cyclic Jacobi algorithm drives a symmetric `A` to diagonal form by a
sequence of Givens rotations `J(p,q,c,s)` with `c² + s² = 1`, each applied
two-sided (`A ← Jᵀ·A·J`) to annihilate the off-diagonal pair `(p,q)`. Unlike
the exact factorisations (Cholesky/LU/QR), Jacobi is *iterative* — the result
is diagonal only in the limit. This file proves what is exactly true per step,
which is the numerical backbone of the method:

1. `givens_orthogonal` — a rotation with `c² + s² = 1` is orthogonal: the
   `2×2` matrix `[[c,−s],[s,c]]` satisfies `JᵀJ = I` (both diagonal entries
   `c²+s² = 1`, both off-diagonal `0`). Orthogonality is why the accumulated
   `V` is orthonormal and the transform is a similarity (eigenvalue-preserving).

2. `givens_norm_preserving` — a rotation preserves the sum of squares of the
   pair it acts on: `(c·x − s·y)² + (s·x + c·y)² = x² + y²`. Applied across all
   rows/columns this is Frobenius-norm invariance; on the diagonal it is the
   trace / eigenvalue-sum invariant.

3. `jacobi_annihilates` — the Jacobi angle zeroes the target off-diagonal
   exactly. For the symmetric `2×2` block `[[app, apq],[apq, aqq]]`, with `t`
   the smaller root of `t² + 2θt − 1 = 0` (`θ = (aqq−app)/(2·apq)`) and
   `c = 1/√(t²+1)`, `s = t·c`, the rotated off-diagonal
   `(c²−s²)·apq + c·s·(app−aqq)` vanishes. Proved from the defining relation
   `apq·(1 − t²) = t·(aqq − app)` (i.e. the angle equation), which is the
   exact-arithmetic content of the host's angle computation.

4. `jacobiUpdateRounded` / `jacobiUpdate_error_decomp` — the per-entry rounding
   decomposition of one rotated entry `c·x − s·y`, reusing the shared
   `roundedOp` model: the rounded result strays from the exact one by at most
   the sum of the elementary-op rounding errors. Same structural triangle-
   inequality shape as the other blas error decompositions.

The whole-decomposition convergence-rate and backward-error bounds (the
off-diagonal norm decreases each sweep; the computed eigenpairs satisfy a
`γ`-scaled residual) are the flagged follow-up, as with the factorisation
whole-op bounds elsewhere in `Quanta.Blas`.
-/

import Quanta.Blas.Reference

namespace Quanta.Blas

/-- **Givens rotation is orthogonal.** With `c² + s² = 1` the `2×2` rotation
    `[[c, −s], [s, c]]` satisfies `JᵀJ = I`: the two diagonal entries of the
    product are `c² + s² = 1` and the two off-diagonal entries are
    `c·s − s·c = 0` (stated here as the three scalar identities). This is why
    the accumulated eigenvector matrix is orthonormal and each rotation is a
    similarity transform (preserving the spectrum). -/
theorem givens_orthogonal (c s : ℝ) (h : c ^ 2 + s ^ 2 = 1) :
    c * c + s * s = 1 ∧ (-s) * (-s) + c * c = 1 ∧ c * (-s) + s * c = 0 := by
  refine ⟨?_, ?_, ?_⟩
  · have : c ^ 2 + s ^ 2 = c * c + s * s := by ring
    rwa [this] at h
  · have : c ^ 2 + s ^ 2 = (-s) * (-s) + c * c := by ring
    rwa [this] at h
  · ring

/-- **Norm preservation.** A Givens rotation with `c² + s² = 1` preserves the
    sum of squares of the pair it rotates:

      (c·x − s·y)² + (s·x + c·y)² = x² + y².

    Summed over every column this is Frobenius-norm invariance of the
    two-sided rotation; on the diagonal entries it is the trace (eigenvalue
    sum) invariant. -/
theorem givens_norm_preserving (c s x y : ℝ) (h : c ^ 2 + s ^ 2 = 1) :
    (c * x - s * y) ^ 2 + (s * x + c * y) ^ 2 = x ^ 2 + y ^ 2 := by
  have hc : c ^ 2 = 1 - s ^ 2 := by linarith
  ring_nf
  nlinarith [h, sq_nonneg x, sq_nonneg y]

/-- The Jacobi tangent relation: the chosen `t` solves `t² + 2θt − 1 = 0`,
    equivalently `apq·(1 − t²) = t·(aqq − app)` when `θ = (aqq−app)/(2·apq)`.
    This is the exact-arithmetic content of the host's angle computation and
    the hypothesis under which the rotation annihilates the off-diagonal. -/
def jacobiTangentRel (app aqq apq t : ℝ) : Prop :=
  apq * (1 - t ^ 2) = t * (aqq - app)

/-- **The Jacobi angle annihilates the target off-diagonal (exactly).** For a
    symmetric block with diagonal `app, aqq` and off-diagonal `apq`, writing
    `c = 1/√(t²+1)`, `s = t·c` (so `c² + s² = 1` and `s = t·c`), the rotated
    off-diagonal entry

      apq·(c² − s²) + c·s·(app − aqq)

    is zero whenever `t` satisfies the Jacobi tangent relation. This is the
    defining property of the rotation the algorithm applies at each step. -/
theorem jacobi_annihilates (app aqq apq c s t : ℝ)
    (hs : s = t * c) (hc2 : c ^ 2 + s ^ 2 = 1)
    (hrel : jacobiTangentRel app aqq apq t) :
    apq * (c ^ 2 - s ^ 2) + c * s * (app - aqq) = 0 := by
  unfold jacobiTangentRel at hrel
  -- Substitute s = t·c and factor out c².
  subst hs
  -- apq·(c² − t²c²) + c·(t·c)·(app − aqq)
  --   = c²·(apq·(1 − t²) + t·(app − aqq))
  --   = c²·(apq·(1 − t²) − t·(aqq − app)) = c²·0 = 0.
  have e : apq * (c ^ 2 - (t * c) ^ 2) + c * (t * c) * (app - aqq)
      = c ^ 2 * (apq * (1 - t ^ 2) - t * (aqq - app)) := by ring
  rw [e, hrel]
  ring

/-- Floating-point rotated entry, rounding each elementary op as the
    `jacobi_rot_f32` kernel does: `round(round(c·x) − round(s·y))`. -/
noncomputable def jacobiUpdateRounded (c s x y : ℝ) : ℝ :=
  roundedOp (roundedOp (c * x) - roundedOp (s * y))

/-- Exact rotated entry `c·x − s·y`. -/
noncomputable def jacobiUpdate (c s x y : ℝ) : ℝ := c * x - s * y

/-- **Per-entry forward-error decomposition (structural).** The rounded
    rotated entry strays from the exact one by at most the sum of the three
    elementary-op rounding errors — the outer subtract, and each of the two
    products:

      |ĉ − (c·x − s·y)|
        ≤ |round(a−b) − (a−b)| + |round(c·x) − c·x| + |round(s·y) − s·y|

    with `a = round(c·x)`, `b = round(s·y)`. Pure triangle inequality, same
    shape as `gemmEntry_error_decomp` / `trsvStep_error_decomp`; reuses the
    shared `roundedOp` model. -/
theorem jacobiUpdate_error_decomp (c s x y : ℝ) :
    |jacobiUpdateRounded c s x y - jacobiUpdate c s x y|
      ≤ |roundedOp (roundedOp (c * x) - roundedOp (s * y))
            - (roundedOp (c * x) - roundedOp (s * y))|
        + |roundedOp (c * x) - c * x|
        + |roundedOp (s * y) - s * y| := by
  unfold jacobiUpdateRounded jacobiUpdate
  set a := roundedOp (c * x) with ha
  set b := roundedOp (s * y) with hb
  -- round(a−b) − (c·x − s·y)
  --   = (round(a−b) − (a−b)) + (a − c·x) − (b − s·y)
  have e : roundedOp (a - b) - (c * x - s * y)
      = (roundedOp (a - b) - (a - b)) + (a - c * x) - (b - s * y) := by ring
  rw [e]
  calc
    |(roundedOp (a - b) - (a - b)) + (a - c * x) - (b - s * y)|
        ≤ |(roundedOp (a - b) - (a - b)) + (a - c * x)| + |b - s * y| :=
          abs_sub _ _
    _ ≤ (|roundedOp (a - b) - (a - b)| + |a - c * x|) + |b - s * y| := by
          gcongr; exact abs_add _ _
    _ = |roundedOp (a - b) - (a - b)| + |a - c * x| + |b - s * y| := by ring

end Quanta.Blas
