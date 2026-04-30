//! Verus mirror — multi-queue invariants (steps 018 + 019).
//!
//! Mirrors `Quanta.MultiQueue.Queue` from Lean. Every backend that
//! implements multi-queue (Vulkan VkQueue, Metal MTLCommandQueue)
//! refines this contract:
//!
//! - `create(kind)` returns a fresh queue with empty history,
//!   no signal, live = true.
//! - `submit(q, cmd)` succeeds iff live; appends cmd to submitted.
//! - `signal(q, sem, value)` succeeds iff live; records the pair.
//! - `destroy(q)` flips live to false; subsequent submit/signal fail.
//!
//! Theorems mirror Lean T7700-T7705:
//!   T7750 — fresh queue matches Lean shape.
//!   T7751 — submit appends to submitted history.
//!   T7752 — submit preserves kind / last_signal / live.
//!   T7753 — signal records (sem, value).
//!   T7754 — signal preserves submitted history.
//!   T7755 — destroy invalidates + blocks submit/signal.

use vstd::prelude::*;

verus! {

// 0 = graphics, 1 = compute, 2 = transfer
pub type QueueKind = u8;

pub struct Queue {
    pub handle: u64,
    pub kind: QueueKind,
    pub submitted: Seq<nat>,
    pub last_signal: Option<(nat, nat)>,
    pub live: bool,
}

pub open spec fn create(handle: u64, kind: QueueKind) -> Queue {
    Queue {
        handle,
        kind,
        submitted: Seq::empty(),
        last_signal: Option::None,
        live: true,
    }
}

pub open spec fn submit(q: Queue, cmd: nat) -> Option<Queue> {
    if q.live {
        Option::Some(Queue { submitted: q.submitted.push(cmd), ..q })
    } else {
        Option::None
    }
}

pub open spec fn signal(q: Queue, sem: nat, value: nat) -> Option<Queue> {
    if q.live {
        Option::Some(Queue { last_signal: Option::Some((sem, value)), ..q })
    } else {
        Option::None
    }
}

pub open spec fn destroy(q: Queue) -> Queue {
    Queue { live: false, ..q }
}

// ── T7750: fresh queue matches Lean shape ────────────────────────────────

proof fn t7750_create_fresh(handle: u64, kind: QueueKind)
    ensures
        create(handle, kind).handle == handle,
        create(handle, kind).kind == kind,
        create(handle, kind).submitted.len() == 0,
        create(handle, kind).last_signal.is_None(),
        create(handle, kind).live == true,
{}

// ── T7751: submit appends ────────────────────────────────────────────────

proof fn t7751_submit_appends(q: Queue, cmd: nat, q2: Queue)
    requires
        q.live,
        submit(q, cmd) == Option::<Queue>::Some(q2),
    ensures
        q2.submitted == q.submitted.push(cmd),
        q2.submitted.len() == q.submitted.len() + 1,
{}

// ── T7752: submit preserves kind / last_signal / live ────────────────────

proof fn t7752_submit_preserves(q: Queue, cmd: nat, q2: Queue)
    requires
        submit(q, cmd) == Option::<Queue>::Some(q2),
    ensures
        q2.handle == q.handle,
        q2.kind == q.kind,
        q2.last_signal == q.last_signal,
        q2.live == q.live,
{}

// ── T7753: signal records (sem, value) ───────────────────────────────────

proof fn t7753_signal_records(q: Queue, sem: nat, value: nat, q2: Queue)
    requires
        q.live,
        signal(q, sem, value) == Option::<Queue>::Some(q2),
    ensures
        q2.last_signal == Option::<(nat, nat)>::Some((sem, value)),
{}

// ── T7754: signal preserves submitted ────────────────────────────────────

proof fn t7754_signal_preserves_submitted(q: Queue, sem: nat, value: nat, q2: Queue)
    requires
        signal(q, sem, value) == Option::<Queue>::Some(q2),
    ensures
        q2.submitted == q.submitted,
        q2.kind == q.kind,
{}

// ── T7755: destroy invalidates + blocks ──────────────────────────────────

proof fn t7755_destroy_invalidates(q: Queue)
    ensures
        destroy(q).live == false,
        destroy(q).handle == q.handle,
        destroy(q).submitted == q.submitted,
{}

proof fn t7755b_destroy_blocks_submit(q: Queue, cmd: nat)
    ensures
        submit(destroy(q), cmd) == Option::<Queue>::None,
{}

proof fn t7755c_destroy_blocks_signal(q: Queue, sem: nat, value: nat)
    ensures
        signal(destroy(q), sem, value) == Option::<Queue>::None,
{}

}  // verus!
