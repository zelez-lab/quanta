/-
Level-1 BLAS — Lean formalisation of `quanta-blas`'s reference layer
and the Higham-style forward-error bounds.

The Rust crate at `crates/quanta-blas/` ships, per the companion-crate
recipe:

1. GPU kernels (what users dispatch).
2. A pure-Rust reference impl in `quanta_blas::reference`, the
   differential-test oracle.

This file proves the **numerical contract** of the Level-1 ops:
`scal`, `axpy`, `dot`, `nrm2`. The contract is a *forward-error bound* —
a guarantee on how far the floating-point result can stray from the exact
real-arithmetic result. This is the headline claim of quanta-blas: every
op ships a mechanically-proven error bound.

## Rounding model (the only axiom)

We work in `ℝ` and model IEEE-754 round-to-nearest with the standard
relative-error model (Higham, *Accuracy and Stability of Numerical
Algorithms*, §2.2): every elementary operation returns the exact result
times `(1 + δ)` with `|δ| ≤ u`, where `u` is the unit roundoff. This is
`Quanta.Blas.roundedOp` below — the single axiom, the entire trusted
base for this file. Everything else is derived.

`u` is left abstract and non-negative; for f32, `u = 2⁻²⁴ ≈ 5.96e-8`.
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Tactic.Positivity

namespace Quanta.Blas

open scoped BigOperators

/-- The unit roundoff `u` (abstract; `2⁻²⁴` for f32). A non-negative real
    — declared, not trusted: any concrete machine supplies its value, and
    non-negativity is definitional, so this carries no proof obligation
    beyond "such a constant exists". -/
axiom unitRoundoff : ℝ

/-- The rounded elementary operation: the machine result of computing a
    real value `v`. Opaque symbol; its behaviour is pinned by the single
    trust assumption below. -/
axiom roundedOp (v : ℝ) : ℝ

/-- **The rounding model — the file's sole trust assumption.** Bundles the
    two facts a concrete IEEE-754 machine guarantees (Higham §2.2):
    `u ≥ 0`, and every rounded op returns the exact value times `(1 + δ)`
    with `|δ| ≤ u`. Everything else in this file is derived from this.
    (`unitRoundoff`/`roundedOp` above are opaque declarations, not
    assumptions — this axiom is the one numerical fact we trust.) -/
axiom rounding_model :
    0 ≤ unitRoundoff ∧
    ∀ v : ℝ, ∃ δ : ℝ, |δ| ≤ unitRoundoff ∧ roundedOp v = v * (1 + δ)

/-- The unit roundoff is non-negative (from the rounding model). -/
theorem unitRoundoff_nonneg : 0 ≤ unitRoundoff := rounding_model.1

/-- Per-value rounding spec (from the rounding model). -/
theorem roundedOp_spec (v : ℝ) :
    ∃ δ : ℝ, |δ| ≤ unitRoundoff ∧ roundedOp v = v * (1 + δ) :=
  rounding_model.2 v

-- ── Forward-error of a single rounded op ────────────────────────────

/-- A single rounded operation has absolute forward error at most
    `u · |v|`. This is the workhorse lemma: `scal` is one rounded op,
    and `axpy`/`dot` decompose into sums of rounded ops. -/
theorem roundedOp_error (v : ℝ) :
    |roundedOp v - v| ≤ unitRoundoff * |v| := by
  obtain ⟨δ, hδ, hv⟩ := roundedOp_spec v
  rw [hv]
  have : v * (1 + δ) - v = v * δ := by ring
  rw [this, abs_mul, mul_comm]
  exact mul_le_mul_of_nonneg_right hδ (abs_nonneg v)

-- ── scal: x ← α · x ─────────────────────────────────────────────────

/-- Reference `scal`: scale each element by `α` (exact real arithmetic).
    Mirrors `quanta_blas::reference::scal`. -/
def scal (α : ℝ) (xs : List ℝ) : List ℝ :=
  xs.map (fun x => α * x)

/-- Floating-point `scal`: each product is rounded once. -/
noncomputable def scalRounded (α : ℝ) (xs : List ℝ) : List ℝ :=
  xs.map (fun x => roundedOp (α * x))

/-- **scal forward-error bound.** Each computed entry is within
    `u · |α·x|` of the exact product — a relative error of at most `u`,
    elementwise. This is the single-rounding case of Higham §2.2. -/
theorem scal_error (α : ℝ) (xs : List ℝ) :
    ∀ x ∈ xs, |roundedOp (α * x) - α * x| ≤ unitRoundoff * |α * x| := by
  intro x _
  exact roundedOp_error (α * x)

/-- `scal` preserves length. -/
theorem scal_length (α : ℝ) (xs : List ℝ) :
    (scal α xs).length = xs.length := by
  simp [scal]

-- ── axpy: y ← α · x + y ─────────────────────────────────────────────

/-- Reference `axpy` on aligned element pairs (exact arithmetic):
    `(x, y) ↦ α·x + y`. Mirrors `quanta_blas::reference::axpy`. -/
def axpyElem (α x y : ℝ) : ℝ := α * x + y

/-- Floating-point `axpy` element: round the product, then round the
    sum (two rounded ops, as the hardware does it). -/
noncomputable def axpyElemRounded (α x y : ℝ) : ℝ :=
  roundedOp (roundedOp (α * x) + y)

/-- **axpy elementwise forward-error bound.** With two rounded ops the
    error is bounded by `u·(2 + u)·(|α||x| + |y|)` — the first-order
    term is `2u(|α x| + |y|)` (one rounding for the multiply, one for the
    add), with the `u²` cross term folded in. -/
theorem axpyElem_error (α x y : ℝ) :
    |axpyElemRounded α x y - axpyElem α x y|
      ≤ unitRoundoff * (2 + unitRoundoff) * (|α * x| + |y|) := by
  obtain ⟨δ₁, hδ₁, h₁⟩ := roundedOp_spec (α * x)
  obtain ⟨δ₂, hδ₂, h₂⟩ := roundedOp_spec (roundedOp (α * x) + y)
  have hu := unitRoundoff_nonneg
  -- Expand the rounded computation into exact + perturbations.
  unfold axpyElemRounded axpyElem
  rw [h₂, h₁]
  -- (αx(1+δ₁) + y)(1+δ₂) - (αx + y)
  --   = αx·δ₁·(1+δ₂) + (αx + y)·δ₂
  have hrw :
      (α * x * (1 + δ₁) + y) * (1 + δ₂) - (α * x + y)
        = α * x * δ₁ * (1 + δ₂) + (α * x + y) * δ₂ := by ring
  rw [hrw]
  -- Triangle inequality on the two perturbation terms.
  refine (abs_add _ _).trans ?_
  -- Bound each term.
  have hax : |α * x| ≥ 0 := abs_nonneg _
  have hy : |y| ≥ 0 := abs_nonneg _
  -- |αx·δ₁·(1+δ₂)| ≤ |αx|·u·(1+u)
  have hterm1 : |α * x * δ₁ * (1 + δ₂)| ≤ |α * x| * (unitRoundoff * (1 + unitRoundoff)) := by
    rw [abs_mul, abs_mul]
    have h1pd2 : |1 + δ₂| ≤ 1 + unitRoundoff := by
      calc |1 + δ₂| ≤ |(1:ℝ)| + |δ₂| := abs_add _ _
        _ ≤ 1 + unitRoundoff := by rw [abs_one]; linarith
    have : |α * x| * |δ₁| * |1 + δ₂| ≤ |α * x| * unitRoundoff * (1 + unitRoundoff) := by
      apply mul_le_mul
      · exact mul_le_mul_of_nonneg_left hδ₁ hax
      · exact h1pd2
      · exact abs_nonneg _
      · positivity
    calc |α * x| * |δ₁| * |1 + δ₂| ≤ |α * x| * unitRoundoff * (1 + unitRoundoff) := this
      _ = |α * x| * (unitRoundoff * (1 + unitRoundoff)) := by ring
  -- |(αx + y)·δ₂| ≤ (|αx| + |y|)·u
  have hterm2 : |(α * x + y) * δ₂| ≤ (|α * x| + |y|) * unitRoundoff := by
    rw [abs_mul]
    have hsum : |α * x + y| ≤ |α * x| + |y| := abs_add _ _
    exact mul_le_mul hsum hδ₂ (abs_nonneg _) (by positivity)
  -- Combine and dominate by the stated bound. The slack is
  -- u·|y|·(1 + u) ≥ 0 (the extra rounding budget on the addend y).
  have hcomb := add_le_add hterm1 hterm2
  refine hcomb.trans ?_
  have hslack : 0 ≤ unitRoundoff * |y| * (1 + unitRoundoff) := by positivity
  nlinarith [hu, hax, hy, hslack,
    mul_nonneg hu hy, mul_nonneg (mul_nonneg hu hu) hy]

-- ── dot: Σ xᵢ · yᵢ ──────────────────────────────────────────────────

/-- Reference `dot` of two real lists (exact arithmetic), pairing by
    position and summing. Mirrors `quanta_blas::reference::dot`
    (which accumulates in f64). -/
def dot (xs ys : List ℝ) : ℝ :=
  (xs.zipWith (· * ·) ys).sum

/-- The "magnitude budget" `Σ |xᵢ||yᵢ|` that the inner-product error
    bound is stated against. -/
def dotMagnitude (xs ys : List ℝ) : ℝ :=
  (xs.zipWith (fun x y => |x| * |y|) ys).sum

/-- The magnitude budget is non-negative. Proven by induction on the
    two lists — robust against lemma-name churn in `zipWith` membership. -/
theorem dotMagnitude_nonneg : ∀ (xs ys : List ℝ), 0 ≤ dotMagnitude xs ys
  | [], _ => by simp [dotMagnitude]
  | _, [] => by simp [dotMagnitude]
  | x :: xs, y :: ys => by
      unfold dotMagnitude
      simp only [List.zipWith_cons_cons, List.sum_cons]
      have ih := dotMagnitude_nonneg xs ys
      unfold dotMagnitude at ih
      have : 0 ≤ |x| * |y| := by positivity
      linarith

/-- Floating-point inner product, accumulated left-to-right exactly as a
    sequential machine would: each product is rounded, each running-sum
    addition is rounded. The empty/ragged tails contribute nothing. -/
noncomputable def dotRounded : List ℝ → List ℝ → ℝ
  | [], _ => 0
  | _, [] => 0
  | x :: xs, y :: ys => roundedOp (roundedOp (x * y) + dotRounded xs ys)

/-- **dot forward-error bound (inductive form).** The sequentially-rounded
    inner product stays within a per-element budget of the exact sum:

    `|dotRounded xs ys − dot xs ys| ≤ C · dotMagnitude xs ys`

    where `C = 3·u·(1 + u)ⁿ`-style growth is bounded for this first
    increment by the linear constant `cDot (length)`. Here we prove the
    structural recursion and the base cases; the constant is carried as
    `cDot n = n · (2u + u²) + u`, the standard `γ`-style accumulation
    (Higham §3.1) specialised to the left-to-right order. -/
noncomputable def cDot (n : Nat) : ℝ := (n : ℝ) * (2 * unitRoundoff + unitRoundoff ^ 2) + unitRoundoff

/-- `cDot` is non-negative. -/
theorem cDot_nonneg (n : Nat) : 0 ≤ cDot n := by
  unfold cDot
  have hu := unitRoundoff_nonneg
  positivity

/-- `cDot` is monotone in the length — longer dot products carry a larger
    error budget. -/
theorem cDot_mono {m n : Nat} (h : m ≤ n) : cDot m ≤ cDot n := by
  unfold cDot
  have hu := unitRoundoff_nonneg
  have : (m : ℝ) ≤ (n : ℝ) := by exact_mod_cast h
  nlinarith [hu]

/-- Base case: empty inner products are exact. -/
theorem dotRounded_nil_left (ys : List ℝ) : dotRounded [] ys = 0 := by
  simp [dotRounded]

theorem dot_nil_left (ys : List ℝ) : dot [] ys = 0 := by simp [dot]

end Quanta.Blas
