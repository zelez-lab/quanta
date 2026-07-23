//! Verus mirror — HostField import lifecycle (step 094, T760–T766).
//!
//! Mirrors `crates/gpu/quanta-core/src/api/field.rs::HostField`: a
//! field over caller-owned, page-aligned host memory. The host owns
//! the pages (an mmap'd region); the device imports a *view* — Metal
//! `newBufferWithBytesNoCopy`, Vulkan `VK_EXT_external_memory_host`,
//! CPU borrowed-pointer passthrough. Where no import path exists the
//! same API succeeds through a staged copy, and the difference is
//! queryable (`is_imported`), never silent.
//!
//! Theorems (paired with the Lean model in
//! `specs/verify/lean/Quanta/HostImport.lean` — same range):
//!   T760 — the import path is zero-copy by construction.
//!   T761 — the staged fallback copies exactly once, `imported=false`.
//!   T762 — drop releases the view exactly once (state-level
//!          idempotence; affine types make a second drop unreachable).
//!   T763 — releasing the view never frees the caller's pages.
//!   T764 — a released view is not bindable.
//!   T765 — unaligned import is rejected, not silently staged.
//!   T766 — the closed post-creation op set never writes host bytes
//!          (the v1 read-only coherence contract).
//!
//! The `'a`-outlives and no-writable-alias halves are discharged by
//! Rust's borrow checker on the safe path (`&'a [T]` held by the
//! wrapper) — the same split `field_lifetime.rs` documents for
//! use-after-free. The raw-pointer path carries the contract in its
//! `unsafe` documentation.

use vstd::prelude::*;

verus! {

// ── Ghost state ─────────────────────────────────────────────────────────────

pub enum HostFieldState { Allocated, Freed }

/// Mirror of `HostField<'a, T>`. `T` and `'a` are erased at the
/// mirror level; `copies` counts host→device copies made at creation
/// and `host_bytes_freed` records whether the import machinery ever
/// freed the caller's pages (the invariant: it never does).
pub struct HostField {
    pub handle: u64,
    pub count: nat,
    pub imported: bool,
    pub state: HostFieldState,
    pub copies: nat,
    pub host_bytes_freed: bool,
}

// ── Operations ─────────────────────────────────────────────────────────────

/// The import-alignment contract: base pointer AND byte length are
/// multiples of the backend's granularity (Metal `vm_page_size`,
/// Vulkan `minImportedHostPointerAlignment`).
pub open spec fn import_aligned(ptr: nat, len: nat, granularity: nat) -> bool {
    granularity > 0 && ptr % granularity == 0 && len % granularity == 0
}

/// Mirror of the native import path: a zero-copy view.
pub open spec fn create_import(handle: u64, count: nat) -> HostField {
    HostField {
        handle,
        count,
        imported: true,
        state: HostFieldState::Allocated,
        copies: 0,
        host_bytes_freed: false,
    }
}

/// Mirror of `Gpu::field_from_host`'s driver call: import when the
/// alignment contract holds, reject otherwise (the API layer maps the
/// rejection to `InvalidParam` — it does NOT silently stage).
pub open spec fn try_import(
    handle: u64,
    count: nat,
    ptr: nat,
    len: nat,
    granularity: nat,
) -> Option<HostField> {
    if import_aligned(ptr, len, granularity) {
        Option::Some(create_import(handle, count))
    } else {
        Option::None
    }
}

/// Mirror of the staged-copy fallback (backend without import):
/// allocate + copy the host bytes exactly once.
pub open spec fn create_staged(handle: u64, count: nat) -> HostField {
    HostField {
        handle,
        count,
        imported: false,
        state: HostFieldState::Allocated,
        copies: 1,
        host_bytes_freed: false,
    }
}

/// Mirror of `Drop for HostField` — releases the driver-side view
/// (`field_free(handle)`) and transitions Allocated → Freed. The
/// caller's pages are untouched: `host_bytes_freed` is preserved.
pub open spec fn drop_host_field(f: HostField) -> HostField {
    HostField { state: HostFieldState::Freed, ..f }
}

/// A view may be bound to a wave slot only while Allocated.
pub open spec fn bindable(f: HostField) -> bool {
    matches!(f.state, HostFieldState::Allocated)
}

/// The complete post-creation op set at v1, paired with the host
/// region's version counter: drop releases the view, bind reads the
/// handle. Neither writes host bytes.
pub open spec fn host_version_after_drop(f: HostField, v: nat) -> nat {
    v
}

pub open spec fn host_version_after_bind(f: HostField, v: nat) -> nat {
    v
}

// ── T760: the import path is zero-copy by construction ────────────────────

proof fn t760_import_zero_copy(handle: u64, count: nat)
    ensures
        create_import(handle, count).imported,
        create_import(handle, count).copies == 0,
        !create_import(handle, count).host_bytes_freed,
        bindable(create_import(handle, count)),
{
}

/// The aligned path really is the import path.
proof fn t760b_aligned_imports(handle: u64, count: nat, ptr: nat, len: nat, g: nat)
    requires
        import_aligned(ptr, len, g),
    ensures
        matches!(try_import(handle, count, ptr, len, g), Option::Some(_)),
{
}

// ── T761: the staged fallback copies exactly once ─────────────────────────

proof fn t761_staged_one_copy(handle: u64, count: nat)
    ensures
        !create_staged(handle, count).imported,
        create_staged(handle, count).copies == 1,
        !create_staged(handle, count).host_bytes_freed,
        bindable(create_staged(handle, count)),
{
}

// ── T762: drop releases the view exactly once ─────────────────────────────

proof fn t762_drop_releases(f: HostField)
    requires
        matches!(f.state, HostFieldState::Allocated),
    ensures
        matches!(drop_host_field(f).state, HostFieldState::Freed),
{
}

/// Once Freed, further drops change nothing — release-exactly-once at
/// the state level. Production makes a second drop unreachable
/// (affine types); the mirror restates the invariant.
proof fn t762b_drop_idempotent_at_state_level(f: HostField)
    requires
        matches!(f.state, HostFieldState::Freed),
    ensures
        matches!(drop_host_field(f).state, HostFieldState::Freed),
{
}

// ── T763: releasing the view never frees the caller's pages ───────────────

proof fn t763_drop_preserves_host_bytes(f: HostField)
    ensures
        drop_host_field(f).host_bytes_freed == f.host_bytes_freed,
{
}

/// Combined with T760/T761 (both creation paths start it `false`),
/// `host_bytes_freed` is `false` over every reachable state.
proof fn t763b_host_bytes_never_freed(f: HostField)
    requires
        !f.host_bytes_freed,
    ensures
        !drop_host_field(f).host_bytes_freed,
{
}

// ── T764: a released view is not bindable ─────────────────────────────────

proof fn t764_freed_not_bindable(f: HostField)
    ensures
        !bindable(drop_host_field(f)),
{
}

// ── T765: unaligned import is rejected, not silently staged ───────────────

proof fn t765_unaligned_rejected(handle: u64, count: nat, ptr: nat, len: nat, g: nat)
    requires
        !import_aligned(ptr, len, g),
    ensures
        matches!(try_import(handle, count, ptr, len, g), Option::None),
{
}

// ── T766: the closed op set never writes host bytes ───────────────────────

proof fn t766_no_host_write(f: HostField, v: nat)
    ensures
        host_version_after_drop(f, v) == v,
        host_version_after_bind(f, v) == v,
{
}

}  // verus!
