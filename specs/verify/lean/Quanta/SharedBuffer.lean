/-
Shared device buffers — one buffer, many holders.

A device buffer becomes a shareable resource: independent holders
(actors, threads, subsystems) each keep a reference to one buffer;
kernels dispatched by any holder read/write it with no host
round-trip; the buffer is freed exactly once, when the last holder
releases. Native-handle export (the buffer sibling of the texture
export) is a borrow: it never affects ownership.

Production shape: `SharedField<T> = Arc<Field<T>>` — the reference
count is `Arc`'s, the exactly-once free is `Field`'s affine `Drop`.
This model states the invariants that composition must satisfy;
Rust's `Arc` + affine types discharge them in production.

Proof shape: a `{ handle, refs, freed }` state with `create / clone /
drop / export` operations and theorems for the reference-count
algebra, exactly-once free, no-use-after-last-drop, and
ownership-neutral export.
-/

namespace Quanta.SharedBuffer

/-- Shared-buffer state. `refs` counts live holders; `freed` records
    whether the device buffer has been released. -/
structure Shared where
  handle : Nat
  refs   : Nat
  freed  : Bool
  deriving Repr

/-- Create with a single holder. -/
def Shared.create (h : Nat) : Shared :=
  { handle := h, refs := 1, freed := false }

/-- Add a holder. Defined only while the buffer is live with at least
    one holder — production makes cloning a freed buffer unreachable
    (you clone through a live reference). -/
def Shared.cloneRef (s : Shared) : Option Shared :=
  if ¬ s.freed ∧ 1 ≤ s.refs then
    some { s with refs := s.refs + 1 }
  else
    none

/-- Release one holder. The device buffer is freed exactly when the
    last holder releases. -/
def Shared.dropRef (s : Shared) : Option Shared :=
  if ¬ s.freed ∧ 1 ≤ s.refs then
    some { s with refs := s.refs - 1, freed := s.refs = 1 }
  else
    none

/-- Export the native handle — a borrow: state-neutral. -/
def Shared.export (s : Shared) : Shared :=
  s

/-- A buffer may be bound to a wave slot only while live. -/
def Shared.bindable (s : Shared) : Bool :=
  ¬ s.freed ∧ 1 ≤ s.refs

/- ============================================================ -/
/-                          THEOREMS                              -/
/- ============================================================ -/

/-- T770 — cloning preserves the handle: sharing never mints a second
    device buffer. -/
theorem t770_clone_preserves_handle (s s' : Shared)
    (h_c : s.cloneRef = some s')
    : s'.handle = s.handle := by
  unfold Shared.cloneRef at h_c
  by_cases h : ¬ s.freed ∧ 1 ≤ s.refs
  · rw [if_pos h] at h_c
    have h_eq : s' = { s with refs := s.refs + 1 } := (Option.some.inj h_c).symm
    rw [h_eq]
  · rw [if_neg h] at h_c
    exact absurd h_c (by simp)

/-- T771 — the reference-count algebra: create starts at one, clone
    adds one, drop removes one. -/
theorem t771_create_one_holder (h : Nat)
    : (Shared.create h).refs = 1 ∧ (Shared.create h).freed = false := by
  unfold Shared.create; exact ⟨rfl, rfl⟩

theorem t771b_clone_increments (s s' : Shared)
    (h_c : s.cloneRef = some s')
    : s'.refs = s.refs + 1 := by
  unfold Shared.cloneRef at h_c
  by_cases h : ¬ s.freed ∧ 1 ≤ s.refs
  · rw [if_pos h] at h_c
    have h_eq : s' = { s with refs := s.refs + 1 } := (Option.some.inj h_c).symm
    rw [h_eq]
  · rw [if_neg h] at h_c
    exact absurd h_c (by simp)

theorem t771c_drop_decrements (s s' : Shared)
    (h_d : s.dropRef = some s')
    : s'.refs = s.refs - 1 := by
  unfold Shared.dropRef at h_d
  by_cases h : ¬ s.freed ∧ 1 ≤ s.refs
  · rw [if_pos h] at h_d
    have h_eq : s' = { s with refs := s.refs - 1, freed := s.refs = 1 } :=
      (Option.some.inj h_d).symm
    rw [h_eq]
  · rw [if_neg h] at h_d
    exact absurd h_d (by simp)

/-- T772 — freed exactly once: the drop that frees is exactly the
    last holder's (refs = 1 before, 0 after), and a freed buffer
    admits no further clone or drop — nothing can free it a second
    time. -/
theorem t772_last_drop_frees (s s' : Shared)
    (h_d : s.dropRef = some s')
    : s'.freed = true ↔ s.refs = 1 := by
  unfold Shared.dropRef at h_d
  by_cases h : ¬ s.freed ∧ 1 ≤ s.refs
  · rw [if_pos h] at h_d
    have h_eq : s' = { s with refs := s.refs - 1, freed := s.refs = 1 } :=
      (Option.some.inj h_d).symm
    rw [h_eq]
    simp
  · rw [if_neg h] at h_d
    exact absurd h_d (by simp)

theorem t772b_freed_blocks_clone (s : Shared)
    (h_f : s.freed = true)
    : s.cloneRef = none := by
  unfold Shared.cloneRef
  rw [if_neg (by simp [h_f])]

theorem t772c_freed_blocks_drop (s : Shared)
    (h_f : s.freed = true)
    : s.dropRef = none := by
  unfold Shared.dropRef
  rw [if_neg (by simp [h_f])]

/-- T773 — no use after the last drop: a freed buffer is not
    bindable. Production discharge: `Arc` cannot yield a reference
    after the last clone dropped; affine types make it unreachable. -/
theorem t773_freed_not_bindable (s : Shared)
    (h_f : s.freed = true)
    : s.bindable = false := by
  unfold Shared.bindable
  simp [h_f]

/-- T774 — export is ownership-neutral: a borrowed native handle
    changes neither the holder count nor liveness. -/
theorem t774_export_ownership_neutral (s : Shared)
    : s.export = s := by
  rfl

end Quanta.SharedBuffer
