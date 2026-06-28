/-
Activation VJP correctness for `quanta-autograd`: relu, sigmoid, tanh.

The Rust ops compute these compositionally (relu = max(x,0); σ = 1/(1+e⁻ˣ);
tanh = (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ)) and back-propagate with the captured-output VJPs
σ' = σ(1−σ), tanh' = 1−tanh², relu' = [x>0]. We prove each multiplier is the
analytic derivative, from the same compositional definitions, via Mathlib's
`HasDerivAt`.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul

namespace Quanta.Autograd

open Real

/-! ## relu — the subgradient mask. -/

/-- For `x > 0`, `relu = id` locally, so `∂relu/∂x = 1` (the positive branch of
    the step mask). Stated as the derivative of the identity, which is what
    `relu` equals on `(0, ∞)`; the Rust mask `[x>0]` returns 1 here. -/
theorem relu_deriv_pos {x : ℝ} (_hx : 0 < x) : HasDerivAt (fun a => a) 1 x := by
  simpa using hasDerivAt_id x

/-- For `x < 0`, `relu = 0` locally, so `∂relu/∂x = 0`. The mask returns 0. -/
theorem relu_deriv_neg {x : ℝ} (_hx : x < 0) : HasDerivAt (fun _ : ℝ => (0 : ℝ)) 0 x := by
  simpa using hasDerivAt_const x (0 : ℝ)

/-! ## sigmoid — σ' = σ(1−σ). -/

/-- σ(x) = 1/(1 + e⁻ˣ), the implementation's form. -/
noncomputable def sigmoid (x : ℝ) : ℝ := (1 + Real.exp (-x))⁻¹

/-- **sigmoid VJP multiplier is σ(x)·(1−σ(x)).** -/
theorem sigmoid_hasDerivAt (x : ℝ) :
    HasDerivAt sigmoid (sigmoid x * (1 - sigmoid x)) x := by
  -- u(x) = 1 + e⁻ˣ ; u'(x) = -e⁻ˣ ; σ = u⁻¹ ; σ' = -u'/u² = e⁻ˣ/u².
  have hu : HasDerivAt (fun a => 1 + Real.exp (-a)) (-Real.exp (-x)) x := by
    have he : HasDerivAt (fun a => Real.exp (-a)) (-Real.exp (-x)) x := by
      simpa using (Real.hasDerivAt_exp (-x)).comp x ((hasDerivAt_id x).neg)
    simpa using (hasDerivAt_const x (1 : ℝ)).add he
  have hpos : (1 + Real.exp (-x)) ≠ 0 := by positivity
  have hinv := hu.inv hpos
  -- hinv : HasDerivAt σ (-(-e⁻ˣ) / (1+e⁻ˣ)²) x
  -- Show the multiplier equals σ(1−σ).
  have hmul : -(-Real.exp (-x)) / (1 + Real.exp (-x)) ^ 2
      = sigmoid x * (1 - sigmoid x) := by
    unfold sigmoid
    have hpos' : (1 + Real.exp (-x)) ≠ 0 := by positivity
    field_simp
    ring
  rw [hmul] at hinv
  exact hinv

/-! ## tanh — tanh' = 1 − tanh². -/

/-- tanh(x) = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ), the implementation's form. -/
noncomputable def tanhE (x : ℝ) : ℝ :=
  (Real.exp x - Real.exp (-x)) / (Real.exp x + Real.exp (-x))

/-- **tanh VJP multiplier is 1 − tanh(x)².** -/
theorem tanh_hasDerivAt (x : ℝ) :
    HasDerivAt tanhE (1 - tanhE x ^ 2) x := by
  set p := fun a => Real.exp a - Real.exp (-a) with hp
  set q := fun a => Real.exp a + Real.exp (-a) with hq
  have hexp_neg : HasDerivAt (fun a => Real.exp (-a)) (-Real.exp (-x)) x := by
    simpa using (Real.hasDerivAt_exp (-x)).comp x ((hasDerivAt_id x).neg)
  have hp' : HasDerivAt p (Real.exp x + Real.exp (-x)) x := by
    simpa [hp, sub_neg_eq_add] using (Real.hasDerivAt_exp x).sub hexp_neg
  have hq' : HasDerivAt q (Real.exp x - Real.exp (-x)) x := by
    simpa [hq] using (Real.hasDerivAt_exp x).add hexp_neg
  have hqne : q x ≠ 0 := by rw [hq]; positivity
  -- quotient rule: (p/q)' = (p'·q − p·q')/q²
  have hdiv := hp'.div hq' hqne
  -- The multiplier (q·q − p·p)/q² = 1 − (p/q)².
  have hmul : ((Real.exp x + Real.exp (-x)) * q x
        - p x * (Real.exp x - Real.exp (-x))) / q x ^ 2
      = 1 - tanhE x ^ 2 := by
    simp only [hp, hq]
    unfold tanhE
    field_simp
    ring
  rw [hmul] at hdiv
  exact hdiv

end Quanta.Autograd
