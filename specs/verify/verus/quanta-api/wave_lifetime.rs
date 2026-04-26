//! Verus mirror — Wave lifecycle + binding invariants (step 075).
//!
//! Mirrors `src/api/wave.rs::Wave`. The two load-bearing properties:
//!   - the `bindings: [u64; 16]` array indexed by slot is a sparse
//!     map; slot 0 of every unbound entry equals 0 (the reserved
//!     "unbound" sentinel observed by the WebGPU driver's render-pass
//!     bind-group construction).
//!   - `binding_count` is monotonic per `bind` / `bind_handle` call:
//!     binding to a higher slot raises the count, never lowers it.
//!     This is the **capability-monotonicity** property step 075
//!     names explicitly.
//!
//! Theorems:
//!   T740 — bindings are slot-indexed: writing slot s leaves slots ≠ s
//!          unchanged.
//!   T741 — binding_count is monotonic non-decreasing across bind ops.
//!   T742 — handle 0 is the unbound sentinel: reading a slot that was
//!          never bound returns 0.
//!   T743 — push_data alignment: push slot s occupies bytes
//!          [16*s, 16*s + size_of(value)).

use vstd::prelude::*;

verus! {

// ── Constants mirroring src/api/wave.rs ────────────────────────────────────

pub open spec fn max_bindings() -> nat { 16 }
pub open spec fn max_textures() -> nat { 16 }
pub open spec fn push_data_cap() -> nat { 256 }
pub open spec fn unbound() -> u64 { 0 }

// ── Ghost state ─────────────────────────────────────────────────────────────

/// Mirror of `Wave`. We model only the binding-relevant fields; the
/// drop_fn closure is consumed by Drop just like Pulse — see T722
/// in pulse_lifetime.rs for the parallel argument.
pub struct Wave {
    pub handle: u64,
    pub bindings: Seq<u64>,    // length = max_bindings()
    pub binding_count: nat,
    pub texture_bindings: Seq<u64>, // length = max_textures()
    pub texture_count: nat,
    pub push_len: nat,
    pub push_mask: u32,
}

// ── Operations ─────────────────────────────────────────────────────────────

/// Mirror of `Wave::bind_handle(slot, handle)`. Same shape as the
/// public `bind<T>(slot, &Field<T>)` — both write into bindings[slot]
/// and bump binding_count if slot >= binding_count.
pub open spec fn bind_handle(w: Wave, slot: nat, h: u64) -> Wave {
    Wave {
        bindings: w.bindings.update(slot as int, h),
        binding_count: if slot + 1 > w.binding_count { slot + 1 } else { w.binding_count },
        ..w
    }
}

/// Mirror of `Wave::bind_texture(slot, &Texture)`.
pub open spec fn bind_texture(w: Wave, slot: nat, h: u64) -> Wave {
    Wave {
        texture_bindings: w.texture_bindings.update(slot as int, h),
        texture_count: if slot + 1 > w.texture_count { slot + 1 } else { w.texture_count },
        ..w
    }
}

/// Mirror of `Wave::set_value` / `set_bytes` — writes `data_size`
/// bytes starting at `slot * 16` and updates push_len accordingly.
pub open spec fn set_push_value(w: Wave, slot: nat, data_size: nat) -> Wave {
    let end = slot * 16 + data_size;
    Wave {
        push_len: if end > w.push_len { end } else { w.push_len },
        push_mask: w.push_mask | (1u32 << (slot as u32)),
        ..w
    }
}

// ── T740: bindings are slot-indexed; only target slot changes ─────────────

proof fn t740_bind_only_target_slot_changes(w: Wave, slot: nat, h: u64)
    requires
        w.bindings.len() == max_bindings(),
        slot < max_bindings(),
    ensures
        forall|i: int| 0 <= i < max_bindings() && i != slot as int
            ==> #[trigger] bind_handle(w, slot, h).bindings[i] == w.bindings[i],
        bind_handle(w, slot, h).bindings[slot as int] == h,
{
    assert(forall|i: int| 0 <= i < max_bindings() && i != slot as int
        ==> #[trigger] bind_handle(w, slot, h).bindings[i] == w.bindings[i]);
}

// ── T741: binding_count is monotonic non-decreasing ───────────────────────

proof fn t741_binding_count_monotonic(w: Wave, slot: nat, h: u64)
    ensures bind_handle(w, slot, h).binding_count >= w.binding_count,
{}

proof fn t741_texture_count_monotonic(w: Wave, slot: nat, h: u64)
    ensures bind_texture(w, slot, h).texture_count >= w.texture_count,
{}

/// Capability monotonicity (the property step 075 names): no operation
/// on Wave ever shrinks `binding_count` or `texture_count`.
proof fn t741_capability_monotonicity(w: Wave, slot: nat, h: u64)
    ensures
        bind_handle(w, slot, h).binding_count >= w.binding_count,
        bind_handle(w, slot, h).texture_count == w.texture_count,
        bind_texture(w, slot, h).texture_count >= w.texture_count,
        bind_texture(w, slot, h).binding_count == w.binding_count,
{}

// ── T742: handle 0 is the unbound sentinel ────────────────────────────────

/// A "fresh" Wave has every slot set to the unbound sentinel.
pub open spec fn fresh_wave(handle: u64) -> Wave {
    Wave {
        handle,
        bindings: Seq::new(max_bindings(), |_i| unbound()),
        binding_count: 0,
        texture_bindings: Seq::new(max_textures(), |_i| unbound()),
        texture_count: 0,
        push_len: 0,
        push_mask: 0,
    }
}

proof fn t742_fresh_slots_unbound(w: Wave, slot: nat)
    requires
        w == fresh_wave(w.handle),
        slot < max_bindings(),
    ensures w.bindings[slot as int] == unbound(),
{}

// ── T743: push_data slot alignment ────────────────────────────────────────

proof fn t743_push_slot_offset(slot: nat)
    requires slot < 16,
    ensures slot * 16 < push_data_cap(),
{}

/// After set_push_value, push_len covers the written region.
proof fn t743_push_len_covers_write(w: Wave, slot: nat, data_size: nat)
    ensures set_push_value(w, slot, data_size).push_len >= slot * 16 + data_size,
{}

}  // verus!
