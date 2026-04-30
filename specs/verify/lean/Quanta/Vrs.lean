/-
Variable rate shading (steps 028 + 029).

VRS lets the renderer reduce shading rate per region. A "shading
rate" of `(x, y)` means one fragment-shader invocation covers an
`x × y` block of pixels. The standard cross-vendor rate set
(Vulkan + D3D12 Tier 1 + Metal rate maps) is:

  1×1, 1×2, 2×1, 2×2, 2×4, 4×2, 4×4

Backends:

- Metal: `MTLRasterizationRateMap` per-tile rates on Apple Silicon
  (variable rate per region) and a per-render-pass rate set.
- Vulkan: `VK_KHR_fragment_shading_rate` +
  `vkCmdSetFragmentShadingRateKHR(rate, combiner_op)`.
- WebGPU: not in W3C — `NotSupported`.
- CPU: software lifecycle model.

Proof shape mirrors `Quanta.MeshShader`: lifecycle structure with
a single mutable field (current rate) + bounded validity check.
-/

namespace Quanta.Vrs

/-- The seven cross-vendor shading rates. -/
inductive ShadingRate
  | r1x1
  | r1x2
  | r2x1
  | r2x2
  | r2x4
  | r4x2
  | r4x4
  deriving Repr, DecidableEq

/-- The horizontal axis of a shading rate, in pixels per fragment. -/
def ShadingRate.xAxis : ShadingRate → Nat
  | .r1x1 | .r1x2          => 1
  | .r2x1 | .r2x2 | .r2x4   => 2
  | .r4x2 | .r4x4           => 4

/-- The vertical axis, in pixels per fragment. -/
def ShadingRate.yAxis : ShadingRate → Nat
  | .r1x1 | .r2x1          => 1
  | .r1x2 | .r2x2 | .r4x2  => 2
  | .r2x4 | .r4x4          => 4

/-- VRS state — the current rate the next draw will use. -/
structure State where
  current : ShadingRate
  live    : Bool
  deriving Repr

/-- Create a fresh VRS state. Default rate is 1×1 (no reduction). -/
def State.create : State :=
  { current := .r1x1, live := true }

/-- Set the current shading rate. Fails if the state is destroyed. -/
def State.setRate (s : State) (rate : ShadingRate) : Option State :=
  if s.live then
    some { s with current := rate }
  else
    none

/-- Mark the state destroyed. -/
def State.destroy (s : State) : State :=
  { s with live := false }

/- ============================================================ -/
/-                          THEOREMS                              -/
/- ============================================================ -/

/-- T7500 — `create` produces a live state at default rate 1×1. -/
theorem t7500_create_shape :
    State.create.current = .r1x1
    ∧ State.create.live = true := by
  unfold State.create; exact ⟨rfl, rfl⟩

/-- T7501 — after `setRate r` on a live state, `current` is `r`. -/
theorem t7501_set_rate_get
    (s s' : State) (r : ShadingRate)
    (h_set : s.setRate r = some s')
    : s'.current = r := by
  unfold State.setRate at h_set
  by_cases h_live : s.live
  · rw [if_pos h_live] at h_set
    have h_eq : s' = { s with current := r } := (Option.some.inj h_set).symm
    rw [h_eq]
  · rw [if_neg h_live] at h_set
    exact absurd h_set (by simp)

/-- T7502 — `setRate` preserves the live flag. -/
theorem t7502_set_rate_preserves_live
    (s s' : State) (r : ShadingRate)
    (h_set : s.setRate r = some s')
    : s'.live = s.live := by
  unfold State.setRate at h_set
  by_cases h_live : s.live
  · rw [if_pos h_live] at h_set
    have h_eq : s' = { s with current := r } := (Option.some.inj h_set).symm
    rw [h_eq]
  · rw [if_neg h_live] at h_set
    exact absurd h_set (by simp)

/-- T7503 — `setRate` on a destroyed state fails. -/
theorem t7503_destroy_blocks_set
    (s : State) (r : ShadingRate)
    : (s.destroy).setRate r = none := by
  unfold State.destroy State.setRate
  simp

/-- T7504 — every shading rate has axes drawn from {1, 2, 4}. -/
theorem t7504_axes_in_range (r : ShadingRate) :
    (r.xAxis = 1 ∨ r.xAxis = 2 ∨ r.xAxis = 4)
    ∧ (r.yAxis = 1 ∨ r.yAxis = 2 ∨ r.yAxis = 4) := by
  cases r <;> exact ⟨by simp [ShadingRate.xAxis], by simp [ShadingRate.yAxis]⟩

/-- T7505 — destroy is idempotent on the live flag. -/
theorem t7505_destroy_idempotent (s : State) :
    (s.destroy).destroy = s.destroy := by
  unfold State.destroy
  rfl

end Quanta.Vrs
