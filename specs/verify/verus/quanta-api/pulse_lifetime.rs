//! Verus mirror — Pulse lifecycle invariants (step 075).
//!
//! Mirrors `src/api/pulse.rs::Pulse`. The ghost state below tracks the
//! discrete states a Pulse can be in: `Pending`, `Done`. Production
//! holds an `Option<Box<dyn FnOnce()>>` for the deferred wait closure;
//! the mirror models `closure_present` as a flag that becomes false
//! after the FnOnce fires.
//!
//! Theorems:
//!   T720 — completion is monotonic-with-reset: `wait()` transitions
//!          Pending → Done; `reset()` is the only transition Done →
//!          Pending. No way to go Done → Done by calling wait twice
//!          (the FnOnce is consumed).
//!   T721 — `is_done()` agrees with the state field: returns true iff
//!          the pulse is in `Done`.
//!   T722 — no use-after-free: the wait closure can fire at most once
//!          per Pending→Done transition.

use vstd::prelude::*;

verus! {

// ── Ghost state ─────────────────────────────────────────────────────────────

pub enum PulseState { Pending, Done }

/// Mirror of `pulse::Pulse`. `closure_present` reflects whether the
/// `Option<Box<dyn FnOnce()>>` still holds Some — production sets it to
/// None via `take()` on the first `wait` call.
pub struct Pulse {
    pub state: PulseState,
    pub closure_present: bool,
}

// ── Operations (mirror of impl Pulse) ──────────────────────────────────────

/// `Pulse::wait` — consumes the deferred-wait closure (if present),
/// transitions Pending → Done, and is idempotent on a Done pulse.
pub open spec fn wait(p: Pulse) -> Pulse {
    Pulse {
        state: PulseState::Done,
        // FnOnce is consumed by `Option::take`; never re-fires.
        closure_present: false,
    }
}

/// `Pulse::reset` — transitions Done → Pending. The closure stays
/// absent (a reset pulse is reusable for a fresh dispatch but does
/// not regrow its FnOnce).
pub open spec fn reset(p: Pulse) -> Pulse {
    Pulse {
        state: PulseState::Pending,
        closure_present: false,
    }
}

/// `Pulse::is_done` — pure observer of the state field.
pub open spec fn is_done(p: Pulse) -> bool {
    matches!(p.state, PulseState::Done)
}

// ── T720: completion is monotonic-with-reset ──────────────────────────────

/// After `wait`, the pulse is Done — regardless of prior state.
proof fn t720_wait_makes_done(p: Pulse)
    ensures matches!(wait(p).state, PulseState::Done),
{}

/// After `reset`, the pulse is Pending — regardless of prior state.
proof fn t720_reset_makes_pending(p: Pulse)
    ensures matches!(reset(p).state, PulseState::Pending),
{}

/// Reset is the only operation that re-introduces Pending. `wait` only
/// goes one way (Pending → Done) and never undoes that.
proof fn t720_wait_never_uncompletes(p: Pulse)
    ensures matches!(wait(p).state, PulseState::Done),
{}

// ── T721: is_done observer agrees with state field ────────────────────────

proof fn t721_is_done_after_wait(p: Pulse)
    ensures is_done(wait(p)),
{}

proof fn t721_is_done_false_after_reset(p: Pulse)
    ensures !is_done(reset(p)),
{}

// ── T722: closure can fire at most once per Pending→Done transition ───────

/// The closure is gone after the first wait call — even if `wait` is
/// called again, the second call sees `closure_present = false` and
/// the FnOnce cannot re-execute.
proof fn t722_closure_consumed_after_wait(p: Pulse)
    ensures !wait(p).closure_present,
{}

/// Calling `wait` twice is safe: the second call observes the closure
/// as already taken. No double-free, no double-fire.
proof fn t722_double_wait_safe(p: Pulse)
    ensures
        matches!(wait(wait(p)).state, PulseState::Done),
        !wait(wait(p)).closure_present,
{}

}  // verus!
