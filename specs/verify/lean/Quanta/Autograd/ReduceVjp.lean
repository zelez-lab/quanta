/-
Reduction / broadcast VJP correctness for `quanta-autograd`.

Two reverse-mode rules:

  * `sum_axis` (and whole-array `sum`): `y = Σⱼ xⱼ` ⇒ `∂y/∂xⱼ = 1`, so the
    upstream gradient `g` is *broadcast* unchanged to every summed element.
  * `mean`: `y = (Σⱼ xⱼ)/c` ⇒ `∂y/∂xⱼ = 1/c`, so the gradient is `g/c`
    broadcast back.

These are the adjoints of broadcasting: summing a gradient over an axis is the
transpose of broadcasting along it (`unbroadcast` in the Rust crate). We prove
the per-element derivatives (the multipliers the VJP applies), via Mathlib's
`HasDerivAt` over a `Finset` sum, mirroring `matmulEntry_hasDerivAt`.
-/

import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

namespace Quanta.Autograd

open scoped BigOperators

/-- **sum VJP multiplier is 1.** Varying a single element `xⱼ` of a sum (the
    rest fixed) changes the sum at rate `1` — so `∂(Σ x)/∂xⱼ = 1` and the
    backward pass broadcasts the upstream gradient unchanged. -/
theorem sum_hasDerivAt {ι : Type*} [Fintype ι] [DecidableEq ι]
    (x : ι → ℝ) (j : ι) :
    HasDerivAt (fun t => ∑ i, Function.update x j t i) 1 (x j) := by
  have hsplit : ∀ t,
      (∑ i, Function.update x j t i)
        = t + ∑ i ∈ Finset.univ.erase j, x i := by
    intro t
    rw [← Finset.sum_erase_add _ _ (Finset.mem_univ j)]
    rw [Function.update_self, add_comm]
    congr 1
    apply Finset.sum_congr rfl
    intro i hi
    rw [Function.update_of_ne (Finset.ne_of_mem_erase hi)]
  have hbase :
      HasDerivAt (fun t => t + ∑ i ∈ Finset.univ.erase j, x i) 1 (x j) := by
    simpa using (hasDerivAt_id (x j)).add_const
      (∑ i ∈ Finset.univ.erase j, x i)
  simpa only [hsplit] using hbase

/-- **mean VJP multiplier is 1/c.** For `mean = (Σ x)/c` (here `c` = `|ι|`, a
    nonzero constant), `∂mean/∂xⱼ = 1/c` — the gradient is scaled by `1/c` and
    broadcast back. -/
theorem mean_hasDerivAt {ι : Type*} [Fintype ι] [DecidableEq ι]
    (x : ι → ℝ) (j : ι) (c : ℝ) (_hc : c ≠ 0) :
    HasDerivAt (fun t => (∑ i, Function.update x j t i) / c) (1 / c) (x j) := by
  have h := (sum_hasDerivAt x j).div_const c
  simpa [one_div] using h

end Quanta.Autograd
