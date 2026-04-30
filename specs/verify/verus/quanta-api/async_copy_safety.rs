//! Verus mirror — async memory copy invariants (step 044).
//!
//! Mirrors `Quanta.AsyncCopy.Queue` from Lean. Every backend that
//! implements an async-copy queue (Vulkan transfer-bit queue +
//! vkCmdCopyBuffer, Metal MTLBlitCommandEncoder) refines this
//! contract:
//!
//! - `create()` returns a live queue with empty FIFO.
//! - `submit_copy(q, dst, src, size)` succeeds iff live; appends
//!   the copy to the recorded sequence.
//! - `destroy(q)` flips live to false; subsequent submit fails.
//!
//! Theorems mirror Lean T7800-T7804:
//!   T7850 — fresh queue matches Lean shape.
//!   T7851 — submit_copy appends.
//!   T7852 — submit_copy preserves live.
//!   T7853 — destroy blocks submit + idempotent.

use vstd::prelude::*;

verus! {

pub struct CopyOp {
    pub dst: u64,
    pub src: u64,
    pub size: nat,
}

pub struct CopyQueue {
    pub handle: u64,
    pub submitted: Seq<CopyOp>,
    pub live: bool,
}

pub open spec fn create(handle: u64) -> CopyQueue {
    CopyQueue { handle, submitted: Seq::empty(), live: true }
}

pub open spec fn submit_copy(q: CopyQueue, dst: u64, src: u64, size: nat)
    -> Option<CopyQueue>
{
    if q.live {
        Option::Some(CopyQueue {
            submitted: q.submitted.push(CopyOp { dst, src, size }),
            ..q
        })
    } else {
        Option::None
    }
}

pub open spec fn destroy(q: CopyQueue) -> CopyQueue {
    CopyQueue { live: false, ..q }
}

// ── T7850: fresh queue matches Lean shape ────────────────────────────────

proof fn t7850_create_fresh(handle: u64)
    ensures
        create(handle).handle == handle,
        create(handle).submitted.len() == 0,
        create(handle).live == true,
{}

// ── T7851: submit_copy appends ───────────────────────────────────────────

proof fn t7851_submit_appends(q: CopyQueue, dst: u64, src: u64, size: nat, q2: CopyQueue)
    requires
        q.live,
        submit_copy(q, dst, src, size) == Option::<CopyQueue>::Some(q2),
    ensures
        q2.submitted.len() == q.submitted.len() + 1,
        q2.submitted.last() == (CopyOp { dst, src, size }),
{}

// ── T7852: submit_copy preserves live + handle ───────────────────────────

proof fn t7852_submit_preserves(q: CopyQueue, dst: u64, src: u64, size: nat, q2: CopyQueue)
    requires
        submit_copy(q, dst, src, size) == Option::<CopyQueue>::Some(q2),
    ensures
        q2.live == q.live,
        q2.handle == q.handle,
{}

// ── T7853: destroy invalidates + blocks submit + idempotent ──────────────

proof fn t7853_destroy_invalidates(q: CopyQueue)
    ensures
        destroy(q).live == false,
        destroy(q).handle == q.handle,
        destroy(q).submitted == q.submitted,
{}

proof fn t7853b_destroy_blocks_submit(q: CopyQueue, dst: u64, src: u64, size: nat)
    ensures
        submit_copy(destroy(q), dst, src, size) == Option::<CopyQueue>::None,
{}

proof fn t7853c_destroy_idempotent(q: CopyQueue)
    ensures
        destroy(destroy(q)) == destroy(q),
{}

}  // verus!
