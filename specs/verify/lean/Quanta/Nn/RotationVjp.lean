/-
Rotary position embedding (RoPE) — VJP identities for the fused kernel.

RoPE rotates each `(xₑ, xₒ)` pair (rotate-half convention: indices `j` and
`j + d/2` share a frequency) by a position-dependent angle:
`yₑ = xₑ·c − xₒ·s`, `yₒ = xₑ·s + xₒ·c` with `c = cos θ`, `s = sin θ`.

Because a rotation is orthogonal, its VJP is the rotation by `−θ` — which
lets ONE kernel with a sign flag serve both the forward and the backward.
Three scalar lemmas license this (T9216–T9218):

* **T9216 — adjoint identity**: `⟨g, R v⟩ = ⟨R(−θ) g, v⟩` per pair, i.e.
  the backward formula `(gₑ·c + gₒ·s, gₒ·c − gₑ·s)` is exactly the adjoint.
* **T9217 — norm preservation**: `c² + s² = 1 → |R x|² = |x|²` per pair —
  the stability story: RoPE cannot amplify activations or gradients.
* **T9218 — inverse composition**: applying the sign-flipped rotation to
  the rotated pair returns the original pair — the algebraic fact behind
  the shared kernel's `sign` parameter.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.LinearCombination

namespace Quanta.Nn.RotationVjp

/-- T9216 — the rotation VJP is the transpose rotation: pairing the
upstream gradient `(gₑ, gₒ)` with the rotation of a direction `(vₑ, vₒ)`
equals pairing `(gₑ·c + gₒ·s, gₒ·c − gₑ·s)` with the direction. -/
theorem t9216_rotation_vjp_adjoint (c s ge go ve vo : ℝ) :
    ge * (ve * c - vo * s) + go * (ve * s + vo * c)
      = (ge * c + go * s) * ve + (go * c - ge * s) * vo := by
  ring

/-- T9217 — rotation preserves the pair norm when `c² + s² = 1`: RoPE is an
isometry on every frequency pair, in forward and backward alike. -/
theorem t9217_rotation_preserves_norm (c s a b : ℝ) (h : c ^ 2 + s ^ 2 = 1) :
    (a * c - b * s) ^ 2 + (a * s + b * c) ^ 2 = a ^ 2 + b ^ 2 := by
  linear_combination (a ^ 2 + b ^ 2) * h

/-- T9218 — the sign-flipped rotation inverts the rotation: the shared
kernel run with `sign = −1` on the forward's output recovers the input
(first component; the second is symmetric). -/
theorem t9218_inverse_rotation_fst (c s a b : ℝ) (h : c ^ 2 + s ^ 2 = 1) :
    (a * c - b * s) * c - (a * s + b * c) * (-s) = a := by
  linear_combination a * h

/-- T9218 (second component). -/
theorem t9218_inverse_rotation_snd (c s a b : ℝ) (h : c ^ 2 + s ^ 2 = 1) :
    (a * c - b * s) * (-s) + (a * s + b * c) * c = b := by
  linear_combination b * h

end Quanta.Nn.RotationVjp
