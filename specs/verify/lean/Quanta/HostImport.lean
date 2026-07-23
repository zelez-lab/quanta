/-
Zero-copy host-memory Field import.

The host owns page-aligned memory (an mmap'd file region); the GPU
imports a view of it without copying — the inverse of `MappedField`,
where the driver allocates and the host gets the view. Where import is
unavailable the same API succeeds through a staged copy, and the
difference is queryable (`is_imported`), never silent.

Backends:

- Metal: `newBufferWithBytesNoCopy` (shared storage, nil deallocator —
  Metal never frees the pages).
- Vulkan: `VK_EXT_external_memory_host`, honoring
  `minImportedHostPointerAlignment`.
- CPU: borrowed-pointer passthrough (the backend is synchronous, so
  the borrow is live across every dispatch).
- WebGPU: no host import exists — staged copy.

Proof shape: a state structure with `importView / staged / drop`
operations plus a closed post-creation op set, and theorems showing
the invariants every backend must respect: zero-copy is a property of
the import path (not a comment), the staged path copies exactly once,
the view is released exactly once, releasing the view NEVER frees the
caller's pages, a freed view is not bindable, unaligned import is
rejected (not silently copied), and no operation writes host bytes
(the v1 read-only coherence contract).

The `'a`-outlives and no-writable-alias halves of the contract are
discharged by Rust's borrow checker on the safe path (`&'a [T]` held
by the wrapper) — documented in scope.md, not ghost-modeled, the same
split `field_lifetime.rs` uses for use-after-free.
-/

namespace Quanta.HostImport

/-- Host-import view state. `copies` counts host→device copies made at
    creation; `hostFreed` records whether the import machinery ever
    freed the caller's pages (the invariant is that it never does). -/
structure Field where
  handle    : Nat
  count     : Nat
  imported  : Bool
  live      : Bool
  copies    : Nat
  hostFreed : Bool
  deriving Repr

/-- Import a view over caller-owned memory. Defined only when the base
    pointer and byte length are multiples of the backend's import
    granularity (Metal `vm_page_size`, Vulkan
    `minImportedHostPointerAlignment`). Zero copies. -/
def Field.importView (h c ptr len g : Nat) : Option Field :=
  if 0 < g ∧ ptr % g = 0 ∧ len % g = 0 then
    some { handle := h, count := c, imported := true,
           live := true, copies := 0, hostFreed := false }
  else
    none

/-- The fallback path where no native import exists: allocate a
    device field and copy the host bytes into it — exactly once. -/
def Field.staged (h c : Nat) : Field :=
  { handle := h, count := c, imported := false,
    live := true, copies := 1, hostFreed := false }

/-- Release the view (the production `Drop`). Frees the driver-side
    buffer object; NEVER the caller's pages. -/
def Field.drop (f : Field) : Field :=
  { f with live := false }

/-- A field may be bound to a wave slot only while live. -/
def Field.bindable (f : Field) : Bool :=
  f.live

/-- The complete post-creation operation set at v1: release the view,
    or bind it (a handle read). Paired with the host region's version
    counter so "no op writes host bytes" is a closed-world statement. -/
inductive Op
  | drop
  | bind

def applyOp : Op → Field × Nat → Field × Nat
  | Op.drop, (f, v) => (f.drop, v)
  | Op.bind, (f, v) => (f, v)

/- ============================================================ -/
/-                          THEOREMS                              -/
/- ============================================================ -/

/-- T760 — the import path is zero-copy by construction: an imported
    view has `copies = 0`, is live, and has not touched the caller's
    pages. -/
theorem t760_import_zero_copy (h c ptr len g : Nat)
    (haligned : 0 < g ∧ ptr % g = 0 ∧ len % g = 0)
    (f : Field) (h_c : Field.importView h c ptr len g = some f)
    : f.imported = true ∧ f.copies = 0 ∧ f.live = true ∧ f.hostFreed = false := by
  unfold Field.importView at h_c
  rw [if_pos haligned] at h_c
  have h_eq : f = { handle := h, count := c, imported := true,
                    live := true, copies := 0, hostFreed := false } :=
    (Option.some.inj h_c).symm
  rw [h_eq]; exact ⟨rfl, rfl, rfl, rfl⟩

/-- T761 — the staged fallback copies exactly once and reports
    `imported = false`: the cost is queryable, never silent. -/
theorem t761_staged_one_copy (h c : Nat)
    : (Field.staged h c).imported = false ∧ (Field.staged h c).copies = 1 ∧
      (Field.staged h c).live = true ∧ (Field.staged h c).hostFreed = false := by
  unfold Field.staged; exact ⟨rfl, rfl, rfl, rfl⟩

/-- T762 — drop releases the view: the field is no longer live. -/
theorem t762_drop_releases (f : Field)
    : (f.drop).live = false := by
  rfl

/-- T762b — release-exactly-once at the state level: dropping an
    already-dropped view changes nothing. Production makes a second
    drop unreachable (affine types); the model restates the
    invariant. -/
theorem t762b_drop_idempotent (f : Field)
    : (f.drop).drop = f.drop := by
  rfl

/-- T763 — releasing the view never frees the caller's pages:
    `hostFreed` is preserved by drop, and both creation paths start
    it at `false` (T760/T761), so it is `false` forever. -/
theorem t763_drop_preserves_host_bytes (f : Field)
    : (f.drop).hostFreed = f.hostFreed := by
  rfl

/-- T764 — a released view is not bindable. -/
theorem t764_freed_not_bindable (f : Field)
    : (f.drop).bindable = false := by
  rfl

/-- T765 — unaligned import is rejected, not silently staged: the
    alignment contract is a hard precondition of the import path. -/
theorem t765_unaligned_import_rejected (h c ptr len g : Nat)
    (hbad : ¬ (0 < g ∧ ptr % g = 0 ∧ len % g = 0))
    : Field.importView h c ptr len g = none := by
  unfold Field.importView; rw [if_neg hbad]

/-- T766 — the v1 read-only coherence contract: no operation in the
    closed post-creation op set writes the host region (its version
    counter is invariant under every op). -/
theorem t766_no_host_write (op : Op) (f : Field) (v : Nat)
    : (applyOp op (f, v)).2 = v := by
  cases op <;> rfl

/-- T766b — no operation in the op set frees the caller's pages. -/
theorem t766b_ops_never_free_host (op : Op) (f : Field) (v : Nat)
    (hf : f.hostFreed = false)
    : (applyOp op (f, v)).1.hostFreed = false := by
  cases op
  · exact (t763_drop_preserves_host_bytes f).trans hf
  · exact hf

end Quanta.HostImport
