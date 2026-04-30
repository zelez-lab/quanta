//! Verus mirror — variable rate shading state invariants
//! (steps 028 + 029).
//!
//! Mirrors `Quanta.Vrs.{ShadingRate, State}` from Lean. Every backend
//! that implements VRS (Metal `MTLRasterizationRateMap`, Vulkan
//! `VK_KHR_fragment_shading_rate`) refines this contract:
//!
//! - `create()` returns a live state at default rate 1×1.
//! - `set_rate(rate)` succeeds iff the state is live; the stored
//!   rate equals the input.
//! - `destroy(s)` flips `live` to false; subsequent `set_rate` fails.
//!
//! Theorems mirror Lean T7500-T7505:
//!   T7550 — fresh state matches Lean shape.
//!   T7551 — set_rate writes the input value.
//!   T7552 — set_rate preserves live.
//!   T7553 — destroy invalidates + blocks set.

use vstd::prelude::*;

verus! {

// Rate code: 0 = 1x1, 1 = 1x2, 2 = 2x1, 3 = 2x2, 4 = 2x4, 5 = 4x2, 6 = 4x4.
pub type ShadingRate = u8;

pub spec const RATE_1X1: ShadingRate = 0u8;

pub open spec fn rate_valid(r: ShadingRate) -> bool {
    r <= 6u8
}

pub struct VrsState {
    pub handle: u64,
    pub current: ShadingRate,
    pub live: bool,
}

pub open spec fn create(handle: u64) -> VrsState {
    VrsState { handle, current: RATE_1X1, live: true }
}

pub open spec fn set_rate(s: VrsState, rate: ShadingRate) -> Option<VrsState> {
    if s.live && rate_valid(rate) {
        Option::Some(VrsState { current: rate, ..s })
    } else {
        Option::None
    }
}

pub open spec fn destroy(s: VrsState) -> VrsState {
    VrsState { live: false, ..s }
}

// ── T7550: create produces a well-formed state ────────────────────────────

proof fn t7550_create_fresh(handle: u64)
    ensures
        create(handle).handle == handle,
        create(handle).current == RATE_1X1,
        create(handle).live == true,
{}

// ── T7551: set_rate writes the input value ────────────────────────────────

proof fn t7551_set_rate_writes(s: VrsState, rate: ShadingRate, s2: VrsState)
    requires
        s.live,
        rate_valid(rate),
        set_rate(s, rate) == Option::<VrsState>::Some(s2),
    ensures
        s2.current == rate,
        s2.handle == s.handle,
{}

// ── T7552: set_rate preserves live ────────────────────────────────────────

proof fn t7552_set_rate_preserves_live(s: VrsState, rate: ShadingRate, s2: VrsState)
    requires
        set_rate(s, rate) == Option::<VrsState>::Some(s2),
    ensures
        s2.live == s.live,
{}

// ── T7553: destroy invalidates + blocks set + idempotent ─────────────────

proof fn t7553_destroy_invalidates(s: VrsState)
    ensures
        destroy(s).live == false,
        destroy(s).current == s.current,
        destroy(s).handle == s.handle,
{}

proof fn t7553b_destroy_blocks_set(s: VrsState, rate: ShadingRate)
    ensures
        set_rate(destroy(s), rate) == Option::<VrsState>::None,
{}

proof fn t7553c_destroy_idempotent(s: VrsState)
    ensures
        destroy(destroy(s)) == destroy(s),
{}

}  // verus!
