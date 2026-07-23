//! Verus mirror — SharedField lifecycle (step 088, T770–T774).
//!
//! Mirrors `SharedField<T> = Arc<Field<T>>`: one device buffer, many
//! holders. The reference count is `Arc`'s, the exactly-once free is
//! `Field`'s affine `Drop`; this mirror states the invariants that
//! composition must satisfy. Native-handle export (the buffer sibling
//! of the texture export) is a borrow — ownership-neutral.
//!
//! Theorems (paired with the Lean model in
//! `specs/verify/lean/Quanta/SharedBuffer.lean` — same range):
//!   T770 — cloning preserves the handle (no second buffer minted).
//!   T771 — refcount algebra: create = 1, clone +1, drop −1.
//!   T772 — freed exactly once: only the last holder's drop frees,
//!          and a freed state blocks clone and drop.
//!   T773 — a freed buffer is not bindable (no use-after-last-drop).
//!   T774 — export is ownership-neutral.

use vstd::prelude::*;

verus! {

// ── Ghost state ─────────────────────────────────────────────────────────────

/// Mirror of the shared-ownership state: `refs` counts live holders,
/// `freed` records whether the device buffer was released.
pub struct Shared {
    pub handle: u64,
    pub refs: nat,
    pub freed: bool,
}

// ── Operations ─────────────────────────────────────────────────────────────

pub open spec fn create(handle: u64) -> Shared {
    Shared { handle, refs: 1, freed: false }
}

/// Whether the state admits clone/drop: live with at least one holder.
pub open spec fn live(s: Shared) -> bool {
    !s.freed && s.refs >= 1
}

/// Mirror of `Clone for Arc<Field<T>>` — one more holder.
pub open spec fn clone_ref(s: Shared) -> Shared
    recommends live(s),
{
    Shared { refs: s.refs + 1, ..s }
}

/// Mirror of dropping one holder: the device buffer is freed exactly
/// when the last holder releases.
pub open spec fn drop_ref(s: Shared) -> Shared
    recommends live(s),
{
    Shared { refs: (s.refs - 1) as nat, freed: s.refs == 1, ..s }
}

/// Mirror of `native_handle()` — a borrow, state-neutral.
pub open spec fn export(s: Shared) -> Shared {
    s
}

pub open spec fn bindable(s: Shared) -> bool {
    live(s)
}

// ── T770: cloning preserves the handle ────────────────────────────────────

proof fn t770_clone_preserves_handle(s: Shared)
    requires live(s),
    ensures clone_ref(s).handle == s.handle,
{
}

// ── T771: refcount algebra ────────────────────────────────────────────────

proof fn t771_create_one_holder(handle: u64)
    ensures
        create(handle).refs == 1,
        !create(handle).freed,
        bindable(create(handle)),
{
}

proof fn t771b_clone_increments(s: Shared)
    requires live(s),
    ensures
        clone_ref(s).refs == s.refs + 1,
        !clone_ref(s).freed,
{
}

proof fn t771c_drop_decrements(s: Shared)
    requires live(s),
    ensures drop_ref(s).refs == s.refs - 1,
{
}

// ── T772: freed exactly once ──────────────────────────────────────────────

/// The drop that frees is exactly the last holder's.
proof fn t772_last_drop_frees(s: Shared)
    requires live(s),
    ensures drop_ref(s).freed <==> s.refs == 1,
{
}

/// A non-last drop leaves the buffer live for the remaining holders —
/// no early free.
proof fn t772b_early_drop_keeps_live(s: Shared)
    requires live(s), s.refs >= 2,
    ensures live(drop_ref(s)),
{
}

// ── T773: no use after the last drop ──────────────────────────────────────

proof fn t773_freed_not_bindable(s: Shared)
    requires s.freed,
    ensures !bindable(s),
{
}

/// After the last holder's drop, nothing is bindable. Production
/// discharge: `Arc` cannot yield a reference once the count hits
/// zero; affine types make use-after-drop unreachable.
proof fn t773b_last_drop_ends_bindability(s: Shared)
    requires live(s), s.refs == 1,
    ensures !bindable(drop_ref(s)),
{
}

// ── T774: export is ownership-neutral ─────────────────────────────────────

proof fn t774_export_ownership_neutral(s: Shared)
    ensures
        export(s).refs == s.refs,
        export(s).freed == s.freed,
        export(s).handle == s.handle,
{
}

}  // verus!
