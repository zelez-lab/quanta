/-
Singular value decomposition (svd) — Lean formalisation of the per-step
one-sided Jacobi contract.

One-sided Jacobi computes `A = U·Σ·Vᵀ` by orthogonalising the *columns* of
`A` with a sequence of right Givens rotations `J(p,q,c,s)` (`c² + s² = 1`),
each chosen to make columns `p` and `q` orthogonal. Like the symmetric
eigendecomposition it is *iterative* — the columns are mutually orthogonal
only in the limit. This file proves what is exactly true per step:

1. `givens_orthogonal` — a rotation with `c² + s² = 1` is orthogonal
   (`JᵀJ = I`). The right factor `V` accumulated from these rotations is
   therefore orthonormal.

2. `givens_pair_norm_preserving` — a right rotation preserves the sum of
   squares of the two entries it acts on:
   `(c·x − s·y)² + (s·x + c·y)² = x² + y²`. Summed over the rows of the two
   columns this is Frobenius-norm invariance of the rotation, so the total
   `Σσ²` (the squared singular values) is invariant across every step.

3. `jacobi_orthogonalises_columns` — the chosen angle makes the target column
   pair orthogonal exactly. For columns with Gram entries `α = ⟨a_p,a_p⟩`,
   `β = ⟨a_q,a_q⟩`, `γ = ⟨a_p,a_q⟩`, with `t` the smaller root of the angle
   equation (`γ·(1 − t²) = t·(β − α)`, i.e. `ζ = (β−α)/(2γ)`), `c = 1/√(t²+1)`,
   `s = t·c`, the rotated cross inner product
   `(c²−s²)·γ + c·s·(α − β)` vanishes. Proved from the angle relation — the
   exact-arithmetic content of the host's angle computation.

4. `svdUpdate_error_decomp` — the per-entry rounding decomposition of one
   rotated entry `c·x − s·y`, reusing the shared `roundedOp` model.

The whole-decomposition convergence-rate and backward-error bounds (the
off-diagonal Gram norm decreases each sweep; the computed factors satisfy a
`γ`-scaled residual) are the flagged follow-up, as with the other iterative
and factorisation whole-op bounds in `Quanta.Blas`.

The Givens invariants (1)(2) coincide with those proved for `eigh`; they are
restated here so `Svd` is self-contained (the algorithms share the rotation
but not the file).
-/

import Quanta.Blas.Reference

namespace Quanta.Blas

/-- **Givens rotation is orthogonal.** With `c² + s² = 1` the `2×2` rotation
    `[[c, −s], [s, c]]` satisfies `JᵀJ = I`: the diagonal entries of the
    product are `c² + s² = 1`, the off-diagonal entries `0`. This is why the
    accumulated right factor `V` is orthonormal. -/
theorem givens_orthogonal (c s : ℝ) (h : c ^ 2 + s ^ 2 = 1) :
    c * c + s * s = 1 ∧ (-s) * (-s) + c * c = 1 ∧ c * (-s) + s * c = 0 := by
  refine ⟨?_, ?_, ?_⟩
  · have : c ^ 2 + s ^ 2 = c * c + s * s := by ring
    rwa [this] at h
  · have : c ^ 2 + s ^ 2 = (-s) * (-s) + c * c := by ring
    rwa [this] at h
  · ring

/-- **Pair norm preservation.** A Givens rotation with `c² + s² = 1` preserves
    the sum of squares of the pair it rotates:

      (c·x − s·y)² + (s·x + c·y)² = x² + y².

    Summed over the rows of the two columns it acts on, this is Frobenius-norm
    invariance of the right rotation — so `Σσ²` (the sum of squared singular
    values, i.e. `‖A‖_F²`) is invariant across every step. -/
theorem givens_pair_norm_preserving (c s x y : ℝ) (h : c ^ 2 + s ^ 2 = 1) :
    (c * x - s * y) ^ 2 + (s * x + c * y) ^ 2 = x ^ 2 + y ^ 2 := by
  nlinarith [h, sq_nonneg x, sq_nonneg y, sq_nonneg (c * x - s * y),
    sq_nonneg (s * x + c * y)]

/-- The one-sided Jacobi angle relation: the chosen `t` solves the angle
    equation `γ·(1 − t²) = t·(β − α)` (with `ζ = (β−α)/(2γ)` the standard
    parameter). This is the exact-arithmetic content of the host's angle
    computation and the hypothesis under which the rotation orthogonalises the
    column pair. -/
def jacobiColAngleRel (alpha beta gamma t : ℝ) : Prop :=
  gamma * (1 - t ^ 2) = t * (beta - alpha)

/-- **The Jacobi angle orthogonalises the column pair (exactly).** For columns
    with Gram entries `α = ⟨a_p,a_p⟩`, `β = ⟨a_q,a_q⟩`, `γ = ⟨a_p,a_q⟩`,
    writing `c = 1/√(t²+1)`, `s = t·c` (so `c² + s² = 1` and `s = t·c`), the
    rotated cross inner product

      (c² − s²)·γ + c·s·(α − β)

    is zero whenever `t` satisfies the angle relation. This is the defining
    property of the right rotation the algorithm applies at each step: after
    it, columns `p` and `q` are orthogonal. -/
theorem jacobi_orthogonalises_columns (alpha beta gamma c s t : ℝ)
    (hs : s = t * c) (hc2 : c ^ 2 + s ^ 2 = 1)
    (hrel : jacobiColAngleRel alpha beta gamma t) :
    (c ^ 2 - s ^ 2) * gamma + c * s * (alpha - beta) = 0 := by
  unfold jacobiColAngleRel at hrel
  subst hs
  have e : (c ^ 2 - (t * c) ^ 2) * gamma + c * (t * c) * (alpha - beta)
      = c ^ 2 * (gamma * (1 - t ^ 2) - t * (beta - alpha)) := by ring
  rw [e, hrel]
  ring

/-- Floating-point rotated entry, rounding each elementary op as the
    `jacobi_col_rot_f32` kernel does: `round(round(c·x) − round(s·y))`. -/
noncomputable def svdUpdateRounded (c s x y : ℝ) : ℝ :=
  roundedOp (roundedOp (c * x) - roundedOp (s * y))

/-- Exact rotated entry `c·x − s·y`. -/
noncomputable def svdUpdate (c s x y : ℝ) : ℝ := c * x - s * y

/-- **Per-entry forward-error decomposition (structural).** The rounded
    rotated entry strays from the exact one by at most the sum of the three
    elementary-op rounding errors (the outer subtract and each product):

      |ĉ − (c·x − s·y)|
        ≤ |round(a−b) − (a−b)| + |round(c·x) − c·x| + |round(s·y) − s·y|

    with `a = round(c·x)`, `b = round(s·y)`. Pure triangle inequality, the
    same shape as `gemmEntry_error_decomp`; reuses the shared `roundedOp`. -/
theorem svdUpdate_error_decomp (c s x y : ℝ) :
    |svdUpdateRounded c s x y - svdUpdate c s x y|
      ≤ |roundedOp (roundedOp (c * x) - roundedOp (s * y))
            - (roundedOp (c * x) - roundedOp (s * y))|
        + |roundedOp (c * x) - c * x|
        + |roundedOp (s * y) - s * y| := by
  unfold svdUpdateRounded svdUpdate
  set a := roundedOp (c * x) with ha
  set b := roundedOp (s * y) with hb
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
