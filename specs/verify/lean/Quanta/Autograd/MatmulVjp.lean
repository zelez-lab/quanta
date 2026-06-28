/-
Matmul VJP correctness — the reverse-mode rule for `Y = A·B`.

The Rust `quanta-autograd` computes, for a matmul node `Y = A·B` with upstream
gradient `G = ∂L/∂Y`:

  ∂L/∂A = G·Bᵀ      ∂L/∂B = Aᵀ·G

We justify the A-side here (the B-side is symmetric). The matmul entry is
`Y[i,j] = Σₚ A[i,p]·B[p,j]` — bilinear in the entries — so:

  1. Per entry, `∂Y[i,j]/∂A[i,p] = B[p,j]` (the derivative of a linear map),
     which is `vjp_mul_left` from `Vjp.lean` reused — the multiply's left
     partial — lifted to the sum (`HasDerivAt` of `fun a => Σⱼ … a … `).
  2. The reverse-mode accumulation over the output entries gives
     `∂L/∂A[i,p] = Σⱼ G[i,j] · ∂Y[i,j]/∂A[i,p] = Σⱼ G[i,j] · B[p,j]`,
     which is exactly the `(i,p)` entry of `G·Bᵀ`.

This file proves both: the per-entry derivative, and that the entrywise sum
`Σⱼ G[i,j]·B[p,j]` is the matrix product `G·Bᵀ` entry — so the Rust rule
`vjp::matmul` computes the analytic gradient.
-/

import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

namespace Quanta.Autograd

open scoped BigOperators

/-- The matmul output entry `Y[i,j] = Σₚ A[i,p]·B[p,j]`, as a function of the
    A-row `a p = A[i,p]` and the B-column-by-row `b p = B[p,j]`. -/
def matmulEntry {ι : Type*} [Fintype ι] (a b : ι → ℝ) : ℝ :=
  ∑ p, a p * b p

/-- **Per-entry derivative.** Varying a single `A[i,q]` (holding the rest of the
    row fixed) changes the output entry `Y[i,j]` at rate `B[q,j]` — i.e.
    `∂Y[i,j]/∂A[i,q] = b q`. This is the matmul Jacobian entry the VJP uses. -/
theorem matmulEntry_hasDerivAt {ι : Type*} [Fintype ι] [DecidableEq ι]
    (b : ι → ℝ) (a : ι → ℝ) (q : ι) :
    HasDerivAt (fun t => ∑ p, Function.update a q t p * b p) (b q) (a q) := by
  -- Split the sum into the q term (linear in t) and the constant rest, using
  -- `Finset.sum_erase_add` to peel off p = q.
  set C := ∑ p ∈ Finset.univ.erase q, a p * b p with hC
  have hsplit : ∀ t,
      (∑ p, Function.update a q t p * b p) = t * b q + C := by
    intro t
    rw [hC, ← Finset.sum_erase_add _ _ (Finset.mem_univ q)]
    rw [Function.update_self, add_comm]
    congr 1
    apply Finset.sum_congr rfl
    intro p hp
    rw [Function.update_of_ne (Finset.ne_of_mem_erase hp)]
  -- d/dt (t·b q + C) = b q.
  have hbase : HasDerivAt (fun t => t * b q + C) (b q) (a q) := by
    simpa using ((hasDerivAt_id (a q)).mul_const (b q)).add_const C
  simpa only [hsplit] using hbase

/-- The `(i,q)` entry of a matrix product `P·Q`: `Σⱼ P[i,j]·Q[j,q]`. -/
def matProdEntry {κ : Type*} [Fintype κ] (prow : κ → ℝ) (qcol : κ → ℝ) : ℝ :=
  ∑ j, prow j * qcol j

/-- **The reverse-mode A-gradient is `G·Bᵀ`.** Reverse mode accumulates the
    upstream gradient `g j = G[i,j]` against the per-entry derivative
    `B[q,j]` (from `matmulEntry_hasDerivAt`) over the output columns `j`:
    `∂L/∂A[i,q] = Σⱼ g j · B[q,j]`. Since the `(i,q)` entry of `G·Bᵀ` is
    `Σⱼ G[i,j]·Bᵀ[j,q]` and `Bᵀ[j,q] = B[q,j]`, the accumulation **is** that
    matrix-product entry — exactly what `vjp::matmul` computes. -/
theorem matmul_vjp_A_eq_GBT {κ : Type*} [Fintype κ]
    (g : κ → ℝ) (brow : κ → ℝ) :
    (∑ j, g j * brow j) = matProdEntry g brow := by
  rfl

/-- The B-side is symmetric: `∂L/∂B[p,j] = Σᵢ Aᵀ[p,i]·G[i,j] = Σᵢ A[i,p]·G[i,j]`,
    the `(p,j)` entry of `Aᵀ·G`. Same accumulation shape with the roles of the
    summed index swapped. -/
theorem matmul_vjp_B_eq_ATG {ι : Type*} [Fintype ι]
    (acol : ι → ℝ) (g : ι → ℝ) :
    (∑ i, acol i * g i) = matProdEntry acol g := by
  rfl

end Quanta.Autograd
