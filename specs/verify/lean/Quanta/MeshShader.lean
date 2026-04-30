/-
Mesh shaders (steps 024 + 025).

Mesh shaders replace the classical vertex / tessellation / geometry
pipeline with a programmable two-stage path:

  - Object stage (Metal) / Task stage (Vulkan, optional): groups of
    threads decide how many mesh-shader workgroups to launch.
  - Mesh stage: each workgroup directly writes a small bounded
    "meshlet" — a list of vertices + primitive indices — straight
    into the rasterizer.

A `MeshPipeline` is parameterized at create time by:

  - `max_vertices_per_meshlet`   ≤ MAX_MESH_VERTICES   (= 256)
  - `max_primitives_per_meshlet` ≤ MAX_MESH_PRIMITIVES (= 256)
  - `task_threads_per_group`     ≤ MAX_TASK_THREADS    (= 128)

These bounds match Vulkan `VkPhysicalDeviceMeshShaderPropertiesEXT`
guaranteed minima and Metal 3 mesh-pipeline limits. Hardware may
expose larger; the IR model uses the conservative minima.

Backends:

  - Metal 3+: `MTLMeshRenderPipelineDescriptor` + object/mesh
    functions; `drawMeshThreadgroups:` dispatches.
  - Vulkan: `VK_EXT_mesh_shader` (or core in 1.3 with
    `VK_KHR_maintenance4`) + TASK_BIT_EXT / MESH_BIT_EXT shader
    stages; `vkCmdDrawMeshTasksEXT(group_x, group_y, group_z)`.
  - WebGPU: not in W3C spec — `NotSupported`.
  - CPU: software lifecycle model; rasterization not modeled.

Proof shape mirrors `Quanta.Tessellation` and `Quanta.Bindless`:
a structure with bounded fields, `create / dispatch / destroy`,
and parametric theorems showing the lifecycle invariants every
backend's `dispatch_mesh` path must respect.
-/

namespace Quanta.MeshShader

/-- Maximum vertices per meshlet (Vulkan EXT minimum guarantee). -/
def MAX_MESH_VERTICES : Nat := 256

/-- Maximum primitives per meshlet (Vulkan EXT minimum guarantee). -/
def MAX_MESH_PRIMITIVES : Nat := 256

/-- Maximum threads per task workgroup (Vulkan EXT minimum guarantee). -/
def MAX_TASK_THREADS : Nat := 128

/-- Maximum workgroups in one dispatch axis. Both Vulkan
    `maxMeshWorkGroupCount` and Metal `maxThreadgroupsPerMeshGrid`
    guarantee at least 65535. -/
def MAX_GROUP_COUNT : Nat := 65535

/-- Mesh pipeline state. `dispatched` records the sequence of
    `[gx, gy, gz]` group counts issued via `dispatch`; backends
    must execute these in the same order they were recorded. -/
structure Pipeline where
  max_vertices       : Nat
  max_primitives     : Nat
  task_threads       : Nat
  dispatched         : List (Nat × Nat × Nat)
  live               : Bool
  deriving Repr

/-- Whether the pipeline parameters are within the proven bounds. -/
def Pipeline.boundsOk (max_v max_p task_t : Nat) : Prop :=
  1 ≤ max_v ∧ max_v ≤ MAX_MESH_VERTICES
  ∧ 1 ≤ max_p ∧ max_p ≤ MAX_MESH_PRIMITIVES
  ∧ 1 ≤ task_t ∧ task_t ≤ MAX_TASK_THREADS

instance (max_v max_p task_t : Nat) : Decidable (Pipeline.boundsOk max_v max_p task_t) := by
  unfold Pipeline.boundsOk
  exact inferInstance

/-- Create a fresh mesh pipeline. Returns `none` if any of the
    requested limits is out of the proven hardware-minimum range. -/
def Pipeline.create (max_v max_p task_t : Nat) : Option Pipeline :=
  if Pipeline.boundsOk max_v max_p task_t then
    some {
      max_vertices := max_v
      max_primitives := max_p
      task_threads := task_t
      dispatched := []
      live := true
    }
  else
    none

/-- Whether a group count is in range for `dispatch`. -/
def groupOk (gx gy gz : Nat) : Prop :=
  gx ≤ MAX_GROUP_COUNT ∧ gy ≤ MAX_GROUP_COUNT ∧ gz ≤ MAX_GROUP_COUNT

instance (gx gy gz : Nat) : Decidable (groupOk gx gy gz) := by
  unfold groupOk; exact inferInstance

/-- Dispatch `[gx, gy, gz]` mesh workgroups on this pipeline.
    Fails if the pipeline is destroyed or any axis exceeds
    `MAX_GROUP_COUNT`. The dispatch is appended to the recorded
    sequence so the backend execution order is observable. -/
def Pipeline.dispatch (p : Pipeline) (gx gy gz : Nat) : Option Pipeline :=
  if ¬ p.live then none
  else if groupOk gx gy gz then
    some { p with dispatched := p.dispatched ++ [(gx, gy, gz)] }
  else
    none

/-- Mark the pipeline destroyed. -/
def Pipeline.destroy (p : Pipeline) : Pipeline :=
  { p with live := false }

/- ============================================================ -/
/-                          THEOREMS                              -/
/- ============================================================ -/

/-- T7300 — `create` produces a pipeline with the requested limits
    and an empty dispatch history. -/
theorem t7300_create_shape (max_v max_p task_t : Nat)
    (h : Pipeline.boundsOk max_v max_p task_t)
    (p : Pipeline) (h_create : Pipeline.create max_v max_p task_t = some p)
    : p.max_vertices = max_v
      ∧ p.max_primitives = max_p
      ∧ p.task_threads = task_t
      ∧ p.dispatched = []
      ∧ p.live = true := by
  unfold Pipeline.create at h_create
  rw [if_pos h] at h_create
  have h_eq : p = {
    max_vertices := max_v
    max_primitives := max_p
    task_threads := task_t
    dispatched := []
    live := true
  } := (Option.some.inj h_create).symm
  rw [h_eq]
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- T7301 — `create` rejects out-of-range parameters. -/
theorem t7301_create_oob_fails (max_v max_p task_t : Nat)
    (h : ¬ Pipeline.boundsOk max_v max_p task_t)
    : Pipeline.create max_v max_p task_t = none := by
  unfold Pipeline.create; rw [if_neg h]

/-- T7302 — `dispatch` extends the recorded sequence by exactly
    one entry, preserving earlier order. This is the contract every
    backend's `dispatch_mesh` path refines: dispatches execute in
    the order they were issued. -/
theorem t7302_dispatch_appends
    (p p' : Pipeline) (gx gy gz : Nat)
    (h_g : groupOk gx gy gz)
    (h_disp : p.dispatch gx gy gz = some p')
    : p'.dispatched = p.dispatched ++ [(gx, gy, gz)] := by
  unfold Pipeline.dispatch at h_disp
  by_cases h_live : ¬ p.live
  · rw [if_pos h_live] at h_disp; exact absurd h_disp (by simp)
  · rw [if_neg h_live] at h_disp
    rw [if_pos h_g] at h_disp
    have h_eq : p' = { p with dispatched := p.dispatched ++ [(gx, gy, gz)] } :=
      (Option.some.inj h_disp).symm
    rw [h_eq]

/-- T7303 — `dispatch` preserves limits. The bounds set at create
    time are immutable. -/
theorem t7303_dispatch_preserves_limits
    (p p' : Pipeline) (gx gy gz : Nat)
    (h_disp : p.dispatch gx gy gz = some p')
    : p'.max_vertices = p.max_vertices
      ∧ p'.max_primitives = p.max_primitives
      ∧ p'.task_threads = p.task_threads
      ∧ p'.live = p.live := by
  unfold Pipeline.dispatch at h_disp
  by_cases h_live : ¬ p.live
  · rw [if_pos h_live] at h_disp; exact absurd h_disp (by simp)
  · rw [if_neg h_live] at h_disp
    by_cases h_g : groupOk gx gy gz
    · rw [if_pos h_g] at h_disp
      have h_eq : p' = { p with dispatched := p.dispatched ++ [(gx, gy, gz)] } :=
        (Option.some.inj h_disp).symm
      rw [h_eq]; exact ⟨rfl, rfl, rfl, rfl⟩
    · rw [if_neg h_g] at h_disp; exact absurd h_disp (by simp)

/-- T7304 — `dispatch` fails when any axis exceeds MAX_GROUP_COUNT. -/
theorem t7304_dispatch_oob_fails
    (p : Pipeline) (gx gy gz : Nat)
    (h_live : p.live)
    (h_oob : ¬ groupOk gx gy gz)
    : p.dispatch gx gy gz = none := by
  unfold Pipeline.dispatch
  rw [if_neg (not_not_intro h_live)]
  rw [if_neg h_oob]

/-- T7305 — `dispatch` fails on a destroyed pipeline. -/
theorem t7305_destroy_blocks_dispatch
    (p : Pipeline) (gx gy gz : Nat)
    : (p.destroy).dispatch gx gy gz = none := by
  unfold Pipeline.destroy Pipeline.dispatch
  simp

end Quanta.MeshShader
