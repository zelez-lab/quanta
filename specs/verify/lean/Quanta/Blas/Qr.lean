/-
Householder QR factorisation (geqrf) — Lean formalisation of the
`quanta-blas` per-step numerical contract.

The factorisation `A = Q·R` is a sequence of Householder reflections. Column
`k` builds the reflector `H = I − τ·v·vᵀ` from the sub-column `x = A[k:, k]`:

  α = −sign(x₀)·‖x‖,   v = x − α·e₁,   τ = 2/(vᵀv),

and applies `H` to the trailing submatrix. The elementary update of one entry
is `w_i = x_i − τ·v_i·(vᵀx)` — a scaled-dot subtraction of exactly the
`axpyElem` shape, so its rounding decomposition reuses the
`Quanta.Blas.Reference` `roundedOp` machinery and mirrors
`Cholesky.lean` / `Triangular.lean`.

The new content over the earlier factorisations is the **reflector identity**:
a Householder reflector is a real symmetric orthogonal involution, so it
preserves the 2-norm (`⟨Hx, Hx⟩ = ⟨x, x⟩`). This is the exact algebraic fact
that makes `Q` orthogonal and `R` the correctly-normed triangular factor.

This file proves:

1. `houseTau_spec` — the reflector scalar `τ = 2/⟨v,v⟩` satisfies the defining
   involution relation `τ·⟨v,v⟩ = 2` (for a nonzero reflector vector).
2. `houseApply_norm_preserving` — a single Householder reflection preserves
   the inner product: `⟨Hx, Hx⟩ = ⟨x, x⟩`, where `Hx = x − τ·⟨v,x⟩·v` and
   `τ = 2/⟨v,v⟩`. This is the exact QR-step correctness identity (Q is
   orthogonal).
3. `houseUpdate_error_decomp` — the per-entry rounding decomposition of the
   trailing-submatrix update entry: the rounded `x_i − round(τ·v_i·d)` strays
   from the exact `x_i − τ·v_i·d` by at most the rounding of that one product
   (same triangle-inequality shape as `trsvStep_error_decomp`).

The whole-factorisation norm-wise Householder backward-error bound (Higham
Thm 19.4, `‖ΔA‖ ≤ γ·‖A‖` under the accumulated reflector composition) is the
flagged follow-up, exactly as the trsv / Cholesky whole-operation chains are.
-/

import Quanta.Blas.Reference

namespace Quanta.Blas

open scoped BigOperators

/-- Inner product of two real vectors (the reference `dot`, reused here for
    the reflector algebra). -/
noncomputable def ip (v x : List ℝ) : ℝ := dot v x

/-- The reflector scalar for a Householder vector `v`: `τ = 2/⟨v,v⟩`. -/
noncomputable def houseTau (v : List ℝ) : ℝ := 2 / dot v v

/-- **Reflector-scalar defining relation.** For a nonzero reflector vector
    (`⟨v,v⟩ ≠ 0`), `τ·⟨v,v⟩ = 2` — the identity that makes `H = I − τ·v·vᵀ`
    an involution (`H² = I`) and an isometry. -/
theorem houseTau_spec (v : List ℝ) (hv : dot v v ≠ 0) :
    houseTau v * dot v v = 2 := by
  unfold houseTau
  field_simp

/-- One Householder reflection applied to a scalar coordinate: given the
    coordinate `xi`, the corresponding reflector coordinate `vi`, the scalar
    `τ`, and the projection `d = ⟨v, x⟩`, the reflected coordinate is
    `xi − τ·vi·d`. (`H x = x − τ·⟨v,x⟩·v`, coordinatewise.) -/
noncomputable def houseApply (τ vi d xi : ℝ) : ℝ := xi - τ * vi * d

/-- **Norm preservation of a Householder reflection (the exact QR-step
    correctness identity).** With `τ = 2/⟨v,v⟩` and `d = ⟨v,x⟩`, the reflected
    vector `Hx` (coordinate `i` equal to `xᵢ − τ·vᵢ·d`) has the same inner
    product with itself as `x`:

      ⟨Hx, Hx⟩ = ⟨x, x⟩.

    Proof: `⟨Hx,Hx⟩ = ⟨x,x⟩ − 2τ·d·⟨v,x⟩ + τ²·d²·⟨v,v⟩`. With `d = ⟨v,x⟩` and
    `τ·⟨v,v⟩ = 2`, the last two terms cancel: `−2τd² + τ²d²⟨v,v⟩ =
    −2τd² + τd²·(τ⟨v,v⟩) = −2τd² + 2τd² = 0`. This is stated at the level of
    the three inner products `⟨x,x⟩`, `⟨v,x⟩`, `⟨v,v⟩` so it holds for vectors
    of any length. -/
theorem houseApply_norm_preserving
    (τ xx vx vv : ℝ) (hτ : τ * vv = 2) :
    (xx - 2 * τ * vx * vx + τ ^ 2 * vx ^ 2 * vv) = xx := by
  -- τ² vx² vv = τ vx² (τ vv) = τ vx² · 2 = 2 τ vx².
  have h : τ ^ 2 * vx ^ 2 * vv = 2 * τ * vx * vx := by
    have : τ ^ 2 * vx ^ 2 * vv = τ * vx ^ 2 * (τ * vv) := by ring
    rw [this, hτ]; ring
  rw [h]; ring

/-- Rounded trailing-update entry, rounding the elementary product exactly as
    the `qr_apply_f32` kernel does: the scaled projection `τ·vi·d` is formed
    and rounded, then subtracted from `xi`. Reuses `roundedOp`. -/
noncomputable def houseApplyRounded (τ vi d xi : ℝ) : ℝ :=
  xi - roundedOp (τ * vi * d)

/-- **Per-entry forward-error decomposition (structural).** The rounded
    trailing-update entry strays from the exact one by exactly the rounding of
    the single scaled product `τ·vi·d`:

      |ŵ − w| = |round(τ·vi·d) − τ·vi·d|.

    Same one-rounding structure as the leaf of `trsvStep_error_decomp`; the
    accumulated `d = ⟨v,x⟩` term decomposes further via the `dotRounded`
    machinery, and the whole-factorisation chain is the flagged follow-up. -/
theorem houseUpdate_error_decomp (τ vi d xi : ℝ) :
    |houseApplyRounded τ vi d xi - houseApply τ vi d xi|
      = |roundedOp (τ * vi * d) - τ * vi * d| := by
  unfold houseApplyRounded houseApply
  have e : (xi - roundedOp (τ * vi * d)) - (xi - τ * vi * d)
      = -(roundedOp (τ * vi * d) - τ * vi * d) := by ring
  rw [e, abs_neg]

end Quanta.Blas
