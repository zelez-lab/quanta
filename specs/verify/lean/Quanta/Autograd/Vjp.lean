/-
Reverse-mode autodiff — Lean correctness of the per-op VJP (vector-Jacobian
product) rules used by `quanta-autograd`.

A VJP rule says: given the upstream gradient `g = ∂L/∂y` of a scalar loss `L`
w.r.t. an op's output `y = f(x)`, the gradient w.r.t. the input is
`∂L/∂x = g · f'(x)` (the chain rule). So a VJP rule is *correct* exactly when
its multiplier is the analytic derivative `f'(x)`. We prove that for every
elementwise op `quanta-autograd` differentiates by exhibiting `HasDerivAt f
(vjpMul) x` from Mathlib's calculus — the VJP multiplier IS the derivative,
mechanically checked, not asserted.

For binary ops the two partials are proven separately (the derivative in each
argument with the other held fixed). The reverse-mode accumulation
`gradᵢ = g · ∂y/∂xᵢ` then follows from these by `HasDerivAt.scomp` / the chain
rule (`vjp_chain` below).

These are the scalar (per-element) rules. The array-level VJPs in the Rust
crate apply them elementwise (and sum over broadcast/reduction axes); the
elementwise correctness is what this file pins down.
-/

import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.Sqrt

namespace Quanta.Autograd

open Real

/-! ## Unary ops: the VJP multiplier is the derivative. -/

/-- `neg`: `y = -x`, VJP multiplier `-1`. -/
theorem vjp_neg (x : ℝ) : HasDerivAt (fun a => -a) (-1) x := by
  simpa using (hasDerivAt_id x).neg

/-- `exp`: `y = exp x`, VJP multiplier `exp x` (= the output itself). -/
theorem vjp_exp (x : ℝ) : HasDerivAt (fun a => Real.exp a) (Real.exp x) x :=
  Real.hasDerivAt_exp x

/-- `log`: `y = log x` (x ≠ 0), VJP multiplier `1/x`. -/
theorem vjp_log {x : ℝ} (hx : x ≠ 0) : HasDerivAt (fun a => Real.log a) x⁻¹ x :=
  Real.hasDerivAt_log hx

/-- `sqrt`: `y = √x` (x > 0), VJP multiplier `1 / (2√x)`. -/
theorem vjp_sqrt {x : ℝ} (hx : x ≠ 0) :
    HasDerivAt (fun a => Real.sqrt a) (1 / (2 * Real.sqrt x)) x :=
  Real.hasDerivAt_sqrt hx

/-- `scale` by a constant `c`: `y = c·x`, VJP multiplier `c`. The `α`-scaling
    in axpy/scal and the constant-multiply ufunc. -/
theorem vjp_scale (c x : ℝ) : HasDerivAt (fun a => c * a) c x := by
  simpa using (hasDerivAt_id x).const_mul c

/-! ## Binary ops: the two partial derivatives. -/

/-- `add` ∂/∂a: `y = a + b`, VJP multiplier `1` (the b-branch is symmetric). -/
theorem vjp_add_left (a b : ℝ) : HasDerivAt (fun x => x + b) 1 a := by
  simpa using (hasDerivAt_id a).add_const b

/-- `add` ∂/∂b. -/
theorem vjp_add_right (a b : ℝ) : HasDerivAt (fun x => a + x) 1 b := by
  simpa using (hasDerivAt_const b a).add (hasDerivAt_id b)

/-- `sub` ∂/∂a: multiplier `1`. -/
theorem vjp_sub_left (a b : ℝ) : HasDerivAt (fun x => x - b) 1 a := by
  simpa using (hasDerivAt_id a).sub_const b

/-- `sub` ∂/∂b: multiplier `-1`. -/
theorem vjp_sub_right (a b : ℝ) : HasDerivAt (fun x => a - x) (-1) b := by
  simpa using (hasDerivAt_const b a).sub (hasDerivAt_id b)

/-- `mul` ∂/∂a: `y = a·b`, multiplier `b`. -/
theorem vjp_mul_left (a b : ℝ) : HasDerivAt (fun x => x * b) b a := by
  simpa using (hasDerivAt_id a).mul_const b

/-- `mul` ∂/∂b: multiplier `a`. -/
theorem vjp_mul_right (a b : ℝ) : HasDerivAt (fun x => a * x) a b := by
  simpa using (hasDerivAt_const b a).mul (hasDerivAt_id b)

/-- `div` ∂/∂a (b ≠ 0): `y = a/b`, multiplier `1/b`. (The hypothesis records
    that the rule is meaningful only for `b ≠ 0`; the derivative itself holds
    for any constant divisor.) -/
theorem vjp_div_left {a b : ℝ} (_hb : b ≠ 0) :
    HasDerivAt (fun x => x / b) (1 / b) a := by
  have h := (hasDerivAt_id a).div_const b
  simpa [one_div] using h

/-- `div` ∂/∂b (b ≠ 0): `y = a/b = a·b⁻¹`, multiplier `-a/b²`. -/
theorem vjp_div_right {a b : ℝ} (hb : b ≠ 0) :
    HasDerivAt (fun x => a / x) (-a / b ^ 2) b := by
  -- a/x = a · x⁻¹; d/dx x⁻¹ = -x⁻², so the derivative is a · (-b⁻²) = -a/b².
  have hinv : HasDerivAt (fun x => x⁻¹) (-(b ^ 2)⁻¹) b := hasDerivAt_inv hb
  have h := hinv.const_mul a
  have heq : a * -(b ^ 2)⁻¹ = -a / b ^ 2 := by
    field_simp
  rw [heq] at h
  simpa [div_eq_mul_inv] using h

/-! ## Chain rule: reverse-mode accumulation is correct. -/

/-- **The reverse-mode step.** If `y = f(x)` has derivative `f'`, then for a
    downstream scalar `L = h(y)` with upstream gradient `g = h'(y)`, the
    gradient w.r.t. `x` is `g · f'(x)` — exactly what a VJP rule computes. This
    is `HasDerivAt.comp` specialised to the reverse-mode convention. -/
theorem vjp_chain {f h : ℝ → ℝ} {x f' g : ℝ}
    (hf : HasDerivAt f f' x) (hh : HasDerivAt h g (f x)) :
    HasDerivAt (fun t => h (f t)) (g * f') x :=
  hh.comp x hf

end Quanta.Autograd
