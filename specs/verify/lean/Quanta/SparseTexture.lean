/-
Sparse (virtual) textures (steps 030 + 031).

Sparse textures are virtual textures whose physical memory is
allocated lazily, one tile at a time. A tile is a fixed-size page
(typically 64×64 or 128×128 pixels per mip level). Backends expose
"map this tile to that backing buffer" and "unmap this tile".

Backends:

- Metal: `MTLHeap` + `replaceRegion:`-style sparse residency on
  GPU families that support it.
- Vulkan: `VK_EXT_sparse_binding` + `vkQueueBindSparse` to bind
  backing memory ranges to texture tiles.
- WebGPU: not in W3C — `NotSupported`.
- CPU: software model — the structure here is the source of truth.

Proof shape: a structure with a tile-association list, `create /
map / unmap / destroy` operations, and parametric theorems showing
the tile-table invariants every backend must respect.
-/

namespace Quanta.SparseTexture

/-- A `(mip, x, y)` tile coordinate. -/
abbrev TileKey := Nat × Nat × Nat

/-- Sparse texture state. `tiles` is the association list of bound
    tiles; each entry maps a `TileKey` to a backing handle (0 means
    unbound, but we use list membership to distinguish bound). -/
structure Texture where
  width  : Nat
  height : Nat
  tiles  : List (TileKey × Nat)
  live   : Bool
  deriving Repr

/-- Build a fresh sparse texture with the given pixel dimensions. -/
def Texture.create (w h : Nat) : Option Texture :=
  if 1 ≤ w ∧ 1 ≤ h then
    some { width := w, height := h, tiles := [], live := true }
  else
    none

/-- Look up the backing handle for the given tile, if any. -/
def Texture.lookup (t : Texture) (k : TileKey) : Option Nat :=
  match t.tiles.find? (fun e => e.fst = k) with
  | some (_, h) => some h
  | none        => none

/-- Bind tile `k` to backing `b`. If the tile was already bound the
    new binding replaces the old one (filter+prepend). -/
def Texture.mapTile (t : Texture) (k : TileKey) (b : Nat) : Option Texture :=
  if t.live then
    let withoutK := t.tiles.filter (fun e => e.fst ≠ k)
    some { t with tiles := (k, b) :: withoutK }
  else
    none

/-- Release the binding for tile `k`. -/
def Texture.unmapTile (t : Texture) (k : TileKey) : Option Texture :=
  if t.live then
    some { t with tiles := t.tiles.filter (fun e => e.fst ≠ k) }
  else
    none

/-- Mark the texture destroyed. -/
def Texture.destroy (t : Texture) : Texture :=
  { t with live := false }

/- ============================================================ -/
/-                          THEOREMS                              -/
/- ============================================================ -/

/-- T7600 — `create` produces a fresh texture with an empty tile
    table and the requested dimensions. -/
theorem t7600_create_shape (w ht : Nat)
    (hwh : 1 ≤ w ∧ 1 ≤ ht)
    (t : Texture) (h_c : Texture.create w ht = some t)
    : t.width = w ∧ t.height = ht ∧ t.tiles = [] ∧ t.live = true := by
  unfold Texture.create at h_c
  rw [if_pos hwh] at h_c
  have h_eq : t = { width := w, height := ht, tiles := [], live := true } :=
    (Option.some.inj h_c).symm
  rw [h_eq]; exact ⟨rfl, rfl, rfl, rfl⟩

/-- T7601 — `create` rejects zero-sized textures. -/
theorem t7601_create_oob_fails (w ht : Nat)
    (hcond : ¬ (1 ≤ w ∧ 1 ≤ ht))
    : Texture.create w ht = none := by
  unfold Texture.create; rw [if_neg hcond]

/-- T7602 — after `mapTile k b`, the lookup at `k` returns `b`. -/
theorem t7602_map_then_lookup
    (t t' : Texture) (k : TileKey) (b : Nat)
    (h_map : t.mapTile k b = some t')
    : t'.lookup k = some b := by
  unfold Texture.mapTile at h_map
  by_cases h_live : t.live
  · rw [if_pos h_live] at h_map
    have h_eq : t' = { t with tiles := (k, b) :: t.tiles.filter (fun e => e.fst ≠ k) } :=
      (Option.some.inj h_map).symm
    rw [h_eq]
    unfold Texture.lookup
    simp [List.find?]
  · rw [if_neg h_live] at h_map
    exact absurd h_map (by simp)

/-- T7603 — `mapTile k b` preserves the dimensions + live flag. -/
theorem t7603_map_preserves
    (t t' : Texture) (k : TileKey) (b : Nat)
    (h_map : t.mapTile k b = some t')
    : t'.width = t.width ∧ t'.height = t.height ∧ t'.live = t.live := by
  unfold Texture.mapTile at h_map
  by_cases h_live : t.live
  · rw [if_pos h_live] at h_map
    have h_eq : t' = { t with tiles := (k, b) :: t.tiles.filter (fun e => e.fst ≠ k) } :=
      (Option.some.inj h_map).symm
    rw [h_eq]; exact ⟨rfl, rfl, rfl⟩
  · rw [if_neg h_live] at h_map
    exact absurd h_map (by simp)

/-- T7604 — after `unmapTile k`, lookup at `k` returns none. -/
theorem t7604_unmap_then_no_lookup
    (t t' : Texture) (k : TileKey)
    (h_unmap : t.unmapTile k = some t')
    : t'.lookup k = none := by
  unfold Texture.unmapTile at h_unmap
  by_cases h_live : t.live
  · rw [if_pos h_live] at h_unmap
    have h_eq : t' = { t with tiles := t.tiles.filter (fun e => e.fst ≠ k) } :=
      (Option.some.inj h_unmap).symm
    rw [h_eq]
    unfold Texture.lookup
    have h_not_found :
        (t.tiles.filter (fun e => e.fst ≠ k)).find? (fun e => e.fst = k) = none := by
      apply List.find?_eq_none.mpr
      intro x hx
      have hfilter := List.mem_filter.mp hx
      have hne : x.fst ≠ k := by
        have : decide (x.fst ≠ k) = true := hfilter.2
        exact of_decide_eq_true this
      simp; exact hne
    rw [h_not_found]
  · rw [if_neg h_live] at h_unmap
    exact absurd h_unmap (by simp)

/-- T7605 — `destroy` blocks `mapTile` and `unmapTile`. -/
theorem t7605_destroy_blocks_map
    (t : Texture) (k : TileKey) (b : Nat)
    : (t.destroy).mapTile k b = none := by
  unfold Texture.destroy Texture.mapTile
  simp

theorem t7605b_destroy_blocks_unmap
    (t : Texture) (k : TileKey)
    : (t.destroy).unmapTile k = none := by
  unfold Texture.destroy Texture.unmapTile
  simp

end Quanta.SparseTexture
