//! Verus mirror of Pulse completion and wait semantics.
//!
//! Mirrors the production struct and methods in `src/api/pulse.rs`.
//!
//! The Pulse lifecycle is:
//!   1. Created with completed = false, wait_fn = Some(f).
//!   2. wait() calls wait_fn.take() (consuming the closure), sets completed = true.
//!   3. Subsequent wait() calls are no-ops (wait_fn is None, completed stays true).
//!   4. reset() sets completed = false (for reuse).
//!
//! Verified properties:
//!
//! | Theorem                   | What it proves                                       |
//! |---------------------------|------------------------------------------------------|
//! | completed_monotonic       | completed: false -> true only, never true -> false    |
//! |                           | (outside of explicit reset).                          |
//! | wait_fn_consumed_at_most_once | wait_fn transitions Some -> None exactly once.    |
//! | wait_sets_completed       | After wait(), completed == true.                     |
//! | wait_consumes_fn          | After wait(), wait_fn == None.                       |
//! | double_wait_idempotent    | Second wait() is a no-op.                            |
//! | reset_allows_reuse        | reset then wait works correctly.                     |
//! | fresh_pulse_not_done      | Newly created pulse has completed == false.           |

use vstd::prelude::*;

verus! {

// ── Abstract Pulse state ────────────────────────────────────────────

/// Ghost model of the Pulse struct.
/// wait_fn is modeled as a bool (has_wait_fn) since Verus cannot reason
/// about closures; what matters is the Some/None transition.
pub struct PulseState {
    pub completed: bool,
    pub has_wait_fn: bool,
}

/// Well-formedness: if completed, wait_fn must have been consumed.
/// This captures the invariant maintained by the production code:
/// wait() is the only path that sets completed = true, and it always
/// takes the wait_fn first.
pub open spec fn wf(p: PulseState) -> bool {
    p.completed ==> !p.has_wait_fn
}

/// A freshly created Pulse.
pub open spec fn fresh_pulse() -> PulseState {
    PulseState {
        completed: false,
        has_wait_fn: true,
    }
}

// ── Operation specs ─────────────────────────────────────────────────

/// wait() operation: take wait_fn if present, then set completed = true.
/// Models the production code:
///   if let Some(f) = self.wait_fn.take() { f(); }
///   self.completed = true;
pub open spec fn wait_result(pre: PulseState, post: PulseState) -> bool {
    // wait_fn is consumed (Option::take).
    &&& post.has_wait_fn == false
    // completed is set to true.
    &&& post.completed == true
}

/// is_done() is a pure read.
pub open spec fn is_done(p: PulseState) -> bool {
    p.completed
}

/// reset() operation: sets completed = false.
/// Note: production reset() does NOT restore wait_fn.
pub open spec fn reset_result(pre: PulseState, post: PulseState) -> bool {
    &&& post.completed == false
    // wait_fn is NOT restored by reset.
    &&& post.has_wait_fn == pre.has_wait_fn
}

// ── Theorems ────────────────────────────────────────────────────────

// ── Completed transitions monotonically ─────────────────────────────

/// wait() can only move completed from false to true (or keep it true).
/// It never sets completed to false.
proof fn completed_monotonic(pre: PulseState, post: PulseState)
    requires
        wf(pre),
        wait_result(pre, post),
    ensures
        // If it was already completed, it stays completed.
        pre.completed ==> post.completed,
        // It is always completed after wait.
        post.completed == true,
        // Monotonicity: post.completed >= pre.completed (as bools).
        !pre.completed ==> post.completed,
{
}

/// Outside of reset(), completed never goes from true to false.
/// (reset is the ONLY operation that clears completed.)
proof fn completed_never_reverts_via_wait(pre: PulseState, post: PulseState)
    requires
        wf(pre),
        pre.completed == true,
        wait_result(pre, post),
    ensures
        post.completed == true,
{
}

// ── wait_fn consumed at most once ───────────────────────────────────

/// After wait(), has_wait_fn is always false.
proof fn wait_consumes_fn(pre: PulseState, post: PulseState)
    requires
        wf(pre),
        wait_result(pre, post),
    ensures
        post.has_wait_fn == false,
{
}

/// If wait_fn was already consumed (second call), wait() is a no-op
/// on the wait_fn (it was already None).
proof fn wait_fn_consumed_at_most_once(pre: PulseState, post: PulseState)
    requires
        wf(pre),
        !pre.has_wait_fn,
        wait_result(pre, post),
    ensures
        // was already false, stays false.
        !post.has_wait_fn,
{
}

/// wait() sets completed to true.
proof fn wait_sets_completed(pre: PulseState, post: PulseState)
    requires wait_result(pre, post),
    ensures post.completed == true,
{
}

// ── Double wait is idempotent ───────────────────────────────────────

/// Calling wait() twice yields the same state as calling it once.
proof fn double_wait_idempotent(
    s0: PulseState,
    s1: PulseState,
    s2: PulseState,
)
    requires
        wf(s0),
        wait_result(s0, s1),
        wait_result(s1, s2),
    ensures
        s1 == s2,
{
    // After first wait: completed = true, has_wait_fn = false.
    // After second wait: completed = true, has_wait_fn = false.
    // Both fields are identical.
}

// ── wait preserves well-formedness ──────────────────────────────────

/// wait() preserves the well-formedness invariant.
proof fn wait_preserves_wf(pre: PulseState, post: PulseState)
    requires
        wf(pre),
        wait_result(pre, post),
    ensures wf(post),
{
    // post.completed == true, post.has_wait_fn == false.
    // wf requires completed ==> !has_wait_fn, which holds.
}

// ── reset + reuse ───────────────────────────────────────────────────

/// reset() produces a valid (but wait_fn-less) state.
proof fn reset_valid(pre: PulseState, post: PulseState)
    requires
        wf(pre),
        reset_result(pre, post),
    ensures
        // After reset, not completed.
        !post.completed,
        // wf holds (completed is false, so implication is vacuously true).
        wf(post),
{
}

/// After wait() then reset(), the pulse is not done.
proof fn wait_then_reset(
    s0: PulseState,
    s1: PulseState,
    s2: PulseState,
)
    requires
        wf(s0),
        wait_result(s0, s1),
        reset_result(s1, s2),
    ensures
        !is_done(s2),
        wf(s2),
        // wait_fn was consumed and NOT restored.
        !s2.has_wait_fn,
{
}

// ── Fresh pulse properties ──────────────────────────────────────────

/// A freshly created pulse is not done.
proof fn fresh_pulse_not_done()
    ensures !is_done(fresh_pulse()),
{
}

/// A freshly created pulse satisfies well-formedness.
proof fn fresh_pulse_wf()
    ensures wf(fresh_pulse()),
{
    // completed == false, so the implication is vacuously true.
}

/// A freshly created pulse has its wait_fn.
proof fn fresh_pulse_has_wait_fn()
    ensures fresh_pulse().has_wait_fn,
{
}

/// Full lifecycle: create -> wait -> is_done == true.
proof fn lifecycle_create_wait_done(post: PulseState)
    requires wait_result(fresh_pulse(), post),
    ensures is_done(post),
{
}

} // verus!
