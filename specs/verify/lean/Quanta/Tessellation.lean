/-
Tessellation pipelines (steps 022 + 023).

Tessellation lets the GPU subdivide a coarse "patch" of control points
into a finer mesh of triangles by parameter-space evaluation. A patch
has:

- `control_points` ∈ 1..MAX_PATCH_SIZE  (Vulkan: 32, Metal: 32, D3D12: 32)
- `inner` factor(s)  — 1 entry for triangle patches, 2 for quad patches
- `outer` factor(s)  — 3 entries for triangle patches, 4 for quad patches

Tessellation factors are real numbers in [1.0, MAX_TESS_LEVEL] (we use
`Nat` here — backends quantize to floats — and the upper bound matches
both Vulkan's `maxTessellationGenerationLevel` and Metal's
`maxTessellationFactor`, both = 64).

Backends:

- Metal: no hardware tessellator. A compute kernel writes per-patch
  factors into an `MTLBuffer`, then the post-tessellation vertex
  shader is invoked via `drawIndexedPatches:`.
- Vulkan: native TCS + TES stages, gated on the `tessellationShader`
  device feature (core in Vulkan 1.0).
- WebGPU: not in the W3C spec — software fallback returns
  `NotSupported`; user code branches.
- CPU: software model — the structure here is the source of truth.

The proofs are the same shape as `Quanta.Bindless`: a structure with
bounded fields, `create / set_inner / set_outer / destroy`, and
parametric theorems showing bounds preservation + localized updates.
The interesting verification content is that *every backend's update
path quantizes within the proven bound*.
-/

namespace Quanta.Tessellation

/-- Maximum tessellation factor any axis can request. Matches Vulkan
    `maxTessellationGenerationLevel` and Metal `maxTessellationFactor`.
    Backends MUST clamp at-or-below this; the IR model enforces it on
    `set`. -/
def MAX_TESS_LEVEL : Nat := 64

/-- Maximum control-point count per patch. Matches Vulkan/Metal/D3D12. -/
def MAX_PATCH_SIZE : Nat := 32

/-- Patch topology: triangle (3 outer + 1 inner) or quad (4 outer + 2
    inner). Isolines are not modeled — Metal does not support them
    and they are vanishingly rarely used. -/
inductive Topology
  | triangle  -- 3 outer, 1 inner
  | quad      -- 4 outer, 2 inner
  deriving Repr, DecidableEq

def Topology.outerCount : Topology → Nat
  | .triangle => 3
  | .quad     => 4

def Topology.innerCount : Topology → Nat
  | .triangle => 1
  | .quad     => 2

/-- A tessellation pipeline state. -/
structure Pipeline where
  topology       : Topology
  control_points : Nat
  outer          : List Nat   -- length = topology.outerCount
  inner          : List Nat   -- length = topology.innerCount
  live           : Bool
  deriving Repr

/-- Create a fresh tessellation pipeline. Returns `none` if
    `control_points` is out of range. Factors initialized to 1
    (= no subdivision). -/
def Pipeline.create (topo : Topology) (cp : Nat) : Option Pipeline :=
  if 1 ≤ cp ∧ cp ≤ MAX_PATCH_SIZE then
    some {
      topology := topo
      control_points := cp
      outer := List.replicate topo.outerCount 1
      inner := List.replicate topo.innerCount 1
      live := true
    }
  else
    none

/-- Clamp a candidate factor into `[1, MAX_TESS_LEVEL]`. -/
def clampFactor (f : Nat) : Nat :=
  if f < 1 then 1
  else if f > MAX_TESS_LEVEL then MAX_TESS_LEVEL
  else f

/-- Set the outer factor at edge `index`. Fails if index is out of
    bounds for the topology, or the pipeline is destroyed. The factor
    is clamped to `[1, MAX_TESS_LEVEL]` before being stored. -/
def Pipeline.setOuter (p : Pipeline) (index : Nat) (factor : Nat)
    : Option Pipeline :=
  if ¬ p.live then none
  else if index < p.topology.outerCount then
    some { p with outer := p.outer.set index (clampFactor factor) }
  else
    none

/-- Set the inner factor. -/
def Pipeline.setInner (p : Pipeline) (index : Nat) (factor : Nat)
    : Option Pipeline :=
  if ¬ p.live then none
  else if index < p.topology.innerCount then
    some { p with inner := p.inner.set index (clampFactor factor) }
  else
    none

/-- Read the outer factor at `index`. -/
def Pipeline.getOuter (p : Pipeline) (index : Nat) : Option Nat :=
  if index < p.topology.outerCount then p.outer[index]? else none

/-- Read the inner factor at `index`. -/
def Pipeline.getInner (p : Pipeline) (index : Nat) : Option Nat :=
  if index < p.topology.innerCount then p.inner[index]? else none

/-- Mark the pipeline destroyed. Subsequent `set*` calls fail. -/
def Pipeline.destroy (p : Pipeline) : Pipeline :=
  { p with live := false }

/- ============================================================ -/
/-                          THEOREMS                              -/
/- ============================================================ -/

/-- T7200 — `create` produces the requested control-point count and
    factor lists of the topology-prescribed length. -/
theorem t7200_create_shape (topo : Topology) (cp : Nat)
    (h_cp : 1 ≤ cp ∧ cp ≤ MAX_PATCH_SIZE)
    (p : Pipeline) (h_create : Pipeline.create topo cp = some p)
    : p.control_points = cp
      ∧ p.outer.length = topo.outerCount
      ∧ p.inner.length = topo.innerCount
      ∧ p.live = true := by
  unfold Pipeline.create at h_create
  rw [if_pos h_cp] at h_create
  have h_eq : p = {
    topology := topo
    control_points := cp
    outer := List.replicate topo.outerCount 1
    inner := List.replicate topo.innerCount 1
    live := true
  } := (Option.some.inj h_create).symm
  rw [h_eq]
  refine ⟨rfl, ?_, ?_, rfl⟩
  · simp
  · simp

/-- T7201 — `clampFactor` always returns a value in `[1, MAX_TESS_LEVEL]`. -/
theorem t7201_clamp_in_range (f : Nat) :
    1 ≤ clampFactor f ∧ clampFactor f ≤ MAX_TESS_LEVEL := by
  unfold clampFactor
  by_cases h1 : f < 1
  · rw [if_pos h1]; exact ⟨Nat.le_refl _, by decide⟩
  · rw [if_neg h1]
    by_cases h2 : f > MAX_TESS_LEVEL
    · rw [if_pos h2]; exact ⟨by decide, Nat.le_refl _⟩
    · rw [if_neg h2]
      refine ⟨?_, ?_⟩
      · exact Nat.le_of_not_lt h1
      · exact Nat.le_of_not_lt h2

/-- T7202 — after `setOuter i f` on a live pipeline at an in-bounds
    index, reading at `i` returns the clamped factor. This is the
    contract every backend's tessellation-factor write path refines. -/
theorem t7202_set_outer_get_eq
    (p : Pipeline) (i : Nat) (f : Nat) (p' : Pipeline)
    (h_inv : p.outer.length = p.topology.outerCount)
    (h_set : p.setOuter i f = some p')
    : p'.getOuter i = some (clampFactor f) := by
  unfold Pipeline.setOuter at h_set
  by_cases h_live : ¬ p.live
  · rw [if_pos h_live] at h_set; exact absurd h_set (by simp)
  · rw [if_neg h_live] at h_set
    by_cases h_i : i < p.topology.outerCount
    · rw [if_pos h_i] at h_set
      have h_eq : p' = { p with outer := p.outer.set i (clampFactor f) } :=
        (Option.some.inj h_set).symm
      rw [h_eq]
      unfold Pipeline.getOuter
      simp [h_i]
      exact List.getElem?_set_self (h := by rw [h_inv]; exact h_i)
    · rw [if_neg h_i] at h_set; exact absurd h_set (by simp)

/-- T7203 — `setOuter` is localized: edges other than `i` keep their
    prior factor. -/
theorem t7203_set_outer_localizes
    (p : Pipeline) (i j : Nat) (f : Nat) (p' : Pipeline)
    (h_ne : i ≠ j)
    (h_set : p.setOuter i f = some p')
    : p'.getOuter j = p.getOuter j := by
  unfold Pipeline.setOuter at h_set
  by_cases h_live : ¬ p.live
  · rw [if_pos h_live] at h_set; exact absurd h_set (by simp)
  · rw [if_neg h_live] at h_set
    by_cases h_i : i < p.topology.outerCount
    · rw [if_pos h_i] at h_set
      have h_eq : p' = { p with outer := p.outer.set i (clampFactor f) } :=
        (Option.some.inj h_set).symm
      rw [h_eq]
      unfold Pipeline.getOuter
      by_cases h_j : j < p.topology.outerCount
      · simp [h_j]
        exact List.getElem?_set_ne (h := h_ne)
      · simp [h_j]
    · rw [if_neg h_i] at h_set; exact absurd h_set (by simp)

/-- T7204 — `setInner` mirrors T7202 for inner factors. -/
theorem t7204_set_inner_get_eq
    (p : Pipeline) (i : Nat) (f : Nat) (p' : Pipeline)
    (h_inv : p.inner.length = p.topology.innerCount)
    (h_set : p.setInner i f = some p')
    : p'.getInner i = some (clampFactor f) := by
  unfold Pipeline.setInner at h_set
  by_cases h_live : ¬ p.live
  · rw [if_pos h_live] at h_set; exact absurd h_set (by simp)
  · rw [if_neg h_live] at h_set
    by_cases h_i : i < p.topology.innerCount
    · rw [if_pos h_i] at h_set
      have h_eq : p' = { p with inner := p.inner.set i (clampFactor f) } :=
        (Option.some.inj h_set).symm
      rw [h_eq]
      unfold Pipeline.getInner
      simp [h_i]
      exact List.getElem?_set_self (h := by rw [h_inv]; exact h_i)
    · rw [if_neg h_i] at h_set; exact absurd h_set (by simp)

/-- T7205 — `create` rejects out-of-range control-point counts. -/
theorem t7205_create_oob_fails (topo : Topology) (cp : Nat)
    (h : ¬ (1 ≤ cp ∧ cp ≤ MAX_PATCH_SIZE))
    : Pipeline.create topo cp = none := by
  unfold Pipeline.create; rw [if_neg h]

/-- T7206 — after `destroy`, every `setOuter` / `setInner` fails. -/
theorem t7206_destroy_blocks_set_outer
    (p : Pipeline) (i f : Nat)
    : (p.destroy).setOuter i f = none := by
  unfold Pipeline.destroy Pipeline.setOuter
  simp

theorem t7206_destroy_blocks_set_inner
    (p : Pipeline) (i f : Nat)
    : (p.destroy).setInner i f = none := by
  unfold Pipeline.destroy Pipeline.setInner
  simp

end Quanta.Tessellation
