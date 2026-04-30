/-
Ray tracing pipelines + acceleration structures (steps 026 + 027).

A ray-tracing pipeline carries a small bundle of shader stages
(`ray_gen`, `closest_hit`, `miss`, optional `intersection`) plus a
`max_recursion_depth`. It runs against a built bottom-level
acceleration structure (BLAS) — a BVH over user-provided geometry.

Backends:

- Metal: `MTLAccelerationStructureDescriptor` + intersector tables
  invoked from a compute kernel (no dedicated RT pipeline state).
- Vulkan: `VK_KHR_acceleration_structure` +
  `VK_KHR_ray_tracing_pipeline`; `vkCmdTraceRaysKHR(width, height, 1)`
  dispatches.
- WebGPU: not in W3C — `NotSupported`.
- CPU: software lifecycle model.

Proof shape mirrors `Quanta.MeshShader`: lifecycle structures with
bounded fields, parametric theorems on the create / dispatch /
destroy flow, and an in-order dispatch history every backend's
`dispatch_rays` path must respect.
-/

namespace Quanta.RayTracing

/-- Maximum ray recursion depth. Vulkan
    `VkPhysicalDeviceRayTracingPipelinePropertiesKHR.maxRayRecursionDepth`
    minimum guarantee + Metal cap. -/
def MAX_RECURSION_DEPTH : Nat := 31

/-- Maximum dispatch dimension (width or height). Conservative
    cross-vendor minimum — Vulkan + Metal expose 2¹⁶ at minimum. -/
def MAX_DISPATCH_DIM : Nat := 65535

/-- The two acceleration-structure tiers. -/
inductive AsKind
  | bottom  -- BLAS: BVH over geometry
  | top     -- TLAS: BVH over BLAS instances
  deriving Repr, DecidableEq

/-- A built acceleration structure. `geom_count` is the number of
    geometries embedded (BLAS) or instance count (TLAS). -/
structure AccelerationStructure where
  kind        : AsKind
  geom_count  : Nat
  live        : Bool
  deriving Repr

/-- Build a fresh AS over `geom_count` geometries. Returns `none`
    when no geometry is supplied — both Metal and Vulkan reject
    empty descriptors. -/
def AccelerationStructure.build (kind : AsKind) (geom_count : Nat)
    : Option AccelerationStructure :=
  if 1 ≤ geom_count then
    some { kind := kind, geom_count := geom_count, live := true }
  else
    none

/-- A ray-tracing pipeline. -/
structure Pipeline where
  max_recursion_depth : Nat
  dispatched          : List (Nat × Nat)  -- (width, height) per launch
  live                : Bool
  deriving Repr

/-- Whether a recursion depth is in range. -/
def depthOk (d : Nat) : Prop :=
  d ≤ MAX_RECURSION_DEPTH

instance (d : Nat) : Decidable (depthOk d) := by
  unfold depthOk; exact inferInstance

/-- Create an RT pipeline. Returns `none` if `max_recursion_depth`
    exceeds the proven hardware-minimum bound. -/
def Pipeline.create (max_d : Nat) : Option Pipeline :=
  if depthOk max_d then
    some {
      max_recursion_depth := max_d
      dispatched := []
      live := true
    }
  else
    none

/-- Whether a dispatch dimension is in range. -/
def dimOk (w h : Nat) : Prop :=
  w ≤ MAX_DISPATCH_DIM ∧ h ≤ MAX_DISPATCH_DIM

instance (w h : Nat) : Decidable (dimOk w h) := by
  unfold dimOk; exact inferInstance

/-- Dispatch `(width, height)` rays on this pipeline. Fails if the
    pipeline is destroyed or any dimension exceeds
    `MAX_DISPATCH_DIM`. -/
def Pipeline.dispatch (p : Pipeline) (w h : Nat) : Option Pipeline :=
  if ¬ p.live then none
  else if dimOk w h then
    some { p with dispatched := p.dispatched ++ [(w, h)] }
  else
    none

/-- Mark the pipeline destroyed. -/
def Pipeline.destroy (p : Pipeline) : Pipeline :=
  { p with live := false }

/-- Mark the AS destroyed. -/
def AccelerationStructure.destroy (a : AccelerationStructure) : AccelerationStructure :=
  { a with live := false }

/- ============================================================ -/
/-                          THEOREMS                              -/
/- ============================================================ -/

/-- T7400 — `AS.build` produces an AS with the requested kind +
    geometry count, marked live. -/
theorem t7400_as_build_shape (kind : AsKind) (gc : Nat)
    (h : 1 ≤ gc)
    (a : AccelerationStructure) (h_b : AccelerationStructure.build kind gc = some a)
    : a.kind = kind ∧ a.geom_count = gc ∧ a.live = true := by
  unfold AccelerationStructure.build at h_b
  rw [if_pos h] at h_b
  have h_eq : a = { kind := kind, geom_count := gc, live := true } :=
    (Option.some.inj h_b).symm
  rw [h_eq]; exact ⟨rfl, rfl, rfl⟩

/-- T7401 — `AS.build` rejects empty geometry. -/
theorem t7401_as_build_empty_fails (kind : AsKind) (gc : Nat)
    (h : ¬ 1 ≤ gc)
    : AccelerationStructure.build kind gc = none := by
  unfold AccelerationStructure.build; rw [if_neg h]

/-- T7402 — `Pipeline.create` produces a pipeline with the requested
    recursion depth + empty dispatch history. -/
theorem t7402_pipeline_create_shape (max_d : Nat)
    (h : depthOk max_d)
    (p : Pipeline) (h_c : Pipeline.create max_d = some p)
    : p.max_recursion_depth = max_d
      ∧ p.dispatched = []
      ∧ p.live = true := by
  unfold Pipeline.create at h_c
  rw [if_pos h] at h_c
  have h_eq : p = {
    max_recursion_depth := max_d
    dispatched := []
    live := true
  } := (Option.some.inj h_c).symm
  rw [h_eq]; exact ⟨rfl, rfl, rfl⟩

/-- T7403 — `Pipeline.dispatch` extends the recorded sequence by
    exactly one (width, height) pair, preserving earlier order. -/
theorem t7403_dispatch_appends
    (p p' : Pipeline) (w h : Nat)
    (h_d : dimOk w h)
    (h_disp : p.dispatch w h = some p')
    : p'.dispatched = p.dispatched ++ [(w, h)] := by
  unfold Pipeline.dispatch at h_disp
  by_cases h_live : ¬ p.live
  · rw [if_pos h_live] at h_disp; exact absurd h_disp (by simp)
  · rw [if_neg h_live] at h_disp
    rw [if_pos h_d] at h_disp
    have h_eq : p' = { p with dispatched := p.dispatched ++ [(w, h)] } :=
      (Option.some.inj h_disp).symm
    rw [h_eq]

/-- T7404 — `Pipeline.dispatch` preserves recursion depth + live. -/
theorem t7404_dispatch_preserves
    (p p' : Pipeline) (w h : Nat)
    (h_disp : p.dispatch w h = some p')
    : p'.max_recursion_depth = p.max_recursion_depth
      ∧ p'.live = p.live := by
  unfold Pipeline.dispatch at h_disp
  by_cases h_live : ¬ p.live
  · rw [if_pos h_live] at h_disp; exact absurd h_disp (by simp)
  · rw [if_neg h_live] at h_disp
    by_cases h_d : dimOk w h
    · rw [if_pos h_d] at h_disp
      have h_eq : p' = { p with dispatched := p.dispatched ++ [(w, h)] } :=
        (Option.some.inj h_disp).symm
      rw [h_eq]; exact ⟨rfl, rfl⟩
    · rw [if_neg h_d] at h_disp; exact absurd h_disp (by simp)

/-- T7405 — `dispatch` fails when w or h exceeds MAX_DISPATCH_DIM. -/
theorem t7405_dispatch_oob_fails
    (p : Pipeline) (w h : Nat)
    (h_live : p.live)
    (h_oob : ¬ dimOk w h)
    : p.dispatch w h = none := by
  unfold Pipeline.dispatch
  rw [if_neg (not_not_intro h_live)]
  rw [if_neg h_oob]

/-- T7406 — `destroy` invalidates dispatch + AS access. -/
theorem t7406_destroy_blocks_dispatch
    (p : Pipeline) (w h : Nat)
    : (p.destroy).dispatch w h = none := by
  unfold Pipeline.destroy Pipeline.dispatch
  simp

theorem t7406b_as_destroy_invalidates
    (a : AccelerationStructure)
    : (a.destroy).live = false := by
  unfold AccelerationStructure.destroy
  rfl

end Quanta.RayTracing
