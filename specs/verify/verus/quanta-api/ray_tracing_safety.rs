//! Verus mirror — ray-tracing pipeline + AS invariants (steps 026 + 027).
//!
//! Mirrors `Quanta.RayTracing.{AccelerationStructure, Pipeline}` from
//! Lean. Every backend that implements ray tracing (Metal
//! MTLAccelerationStructureDescriptor, Vulkan
//! VK_KHR_ray_tracing_pipeline) refines this contract:
//!
//! - `as_build(kind, geom_count)` returns a fresh AS iff
//!   geom_count >= 1.
//! - `pipeline_create(max_d)` returns a fresh pipeline iff
//!   max_d <= MAX_RECURSION_DEPTH.
//! - `dispatch(w, h)` succeeds iff the pipeline is live and each
//!   dimension is <= MAX_DISPATCH_DIM. The (w, h) pair is appended
//!   to the recorded sequence; backends execute in order.
//! - `destroy(*)` flips live to false.
//!
//! Theorems mirror Lean T7400–T7406:
//!   T7450 — fresh AS matches Lean shape.
//!   T7451 — fresh pipeline matches Lean shape.
//!   T7452 — dispatch appends one entry.
//!   T7453 — dispatch preserves max_recursion_depth + handle.
//!   T7454 — dispatch fails OOB or destroyed.
//!   T7455 — destroy invalidates pipeline + AS.

use vstd::prelude::*;

verus! {

pub spec const MAX_RECURSION_DEPTH: nat = 31nat;
pub spec const MAX_DISPATCH_DIM: nat = 65535nat;

// 0 = bottom (BLAS), 1 = top (TLAS)
pub type AsKind = u8;

pub struct AccelerationStructure {
    pub handle: u64,
    pub kind: AsKind,
    pub geom_count: nat,
    pub live: bool,
}

pub struct RtPipeline {
    pub handle: u64,
    pub max_recursion_depth: nat,
    pub dispatched: Seq<(nat, nat)>,
    pub live: bool,
}

pub open spec fn depth_ok(d: nat) -> bool {
    d <= MAX_RECURSION_DEPTH
}

pub open spec fn dim_ok(w: nat, h: nat) -> bool {
    w <= MAX_DISPATCH_DIM && h <= MAX_DISPATCH_DIM
}

pub open spec fn as_build(handle: u64, kind: AsKind, geom_count: nat)
    -> Option<AccelerationStructure>
{
    if 1nat <= geom_count {
        Option::Some(AccelerationStructure {
            handle, kind, geom_count, live: true,
        })
    } else {
        Option::None
    }
}

pub open spec fn pipeline_create(handle: u64, max_d: nat) -> Option<RtPipeline> {
    if depth_ok(max_d) {
        Option::Some(RtPipeline {
            handle,
            max_recursion_depth: max_d,
            dispatched: Seq::empty(),
            live: true,
        })
    } else {
        Option::None
    }
}

pub open spec fn dispatch(p: RtPipeline, w: nat, h: nat) -> Option<RtPipeline> {
    if p.live && dim_ok(w, h) {
        Option::Some(RtPipeline {
            dispatched: p.dispatched.push((w, h)),
            ..p
        })
    } else {
        Option::None
    }
}

pub open spec fn destroy_pipeline(p: RtPipeline) -> RtPipeline {
    RtPipeline { live: false, ..p }
}

pub open spec fn destroy_as(a: AccelerationStructure) -> AccelerationStructure {
    AccelerationStructure { live: false, ..a }
}

// ── T7450: fresh AS matches Lean shape ───────────────────────────────────

proof fn t7450_as_build_fresh(handle: u64, kind: AsKind, gc: nat)
    requires
        1nat <= gc,
    ensures
        as_build(handle, kind, gc) matches Option::Some(a) ==>
            a.handle == handle
            && a.kind == kind
            && a.geom_count == gc
            && a.live == true,
{}

// ── T7451: fresh pipeline matches Lean shape ─────────────────────────────

proof fn t7451_pipeline_create_fresh(handle: u64, max_d: nat)
    requires
        depth_ok(max_d),
    ensures
        pipeline_create(handle, max_d) matches Option::Some(p) ==>
            p.handle == handle
            && p.max_recursion_depth == max_d
            && p.dispatched.len() == 0
            && p.live == true,
{}

// ── T7452: dispatch appends one entry ────────────────────────────────────

proof fn t7452_dispatch_appends(p: RtPipeline, w: nat, h: nat, p2: RtPipeline)
    requires
        p.live,
        dim_ok(w, h),
        dispatch(p, w, h) == Option::<RtPipeline>::Some(p2),
    ensures
        p2.dispatched == p.dispatched.push((w, h)),
        p2.dispatched.len() == p.dispatched.len() + 1,
{}

// ── T7453: dispatch preserves recursion depth + handle ───────────────────

proof fn t7453_dispatch_preserves(p: RtPipeline, w: nat, h: nat, p2: RtPipeline)
    requires
        dispatch(p, w, h) == Option::<RtPipeline>::Some(p2),
    ensures
        p2.max_recursion_depth == p.max_recursion_depth,
        p2.handle == p.handle,
        p2.live == p.live,
{}

// ── T7454: dispatch fails OOB or destroyed ───────────────────────────────

proof fn t7454_dispatch_fails(p: RtPipeline, w: nat, h: nat)
    requires
        !p.live || !dim_ok(w, h),
    ensures
        dispatch(p, w, h) == Option::<RtPipeline>::None,
{}

// ── T7455: destroy invalidates pipeline + AS ─────────────────────────────

proof fn t7455_destroy_pipeline_invalidates(p: RtPipeline)
    ensures
        destroy_pipeline(p).live == false,
        destroy_pipeline(p).max_recursion_depth == p.max_recursion_depth,
        destroy_pipeline(p).dispatched == p.dispatched,
{}

proof fn t7455b_destroy_pipeline_blocks_dispatch(p: RtPipeline, w: nat, h: nat)
    ensures
        dispatch(destroy_pipeline(p), w, h) == Option::<RtPipeline>::None,
{}

proof fn t7455c_destroy_as_invalidates(a: AccelerationStructure)
    ensures
        destroy_as(a).live == false,
        destroy_as(a).handle == a.handle,
        destroy_as(a).kind == a.kind,
        destroy_as(a).geom_count == a.geom_count,
{}

}  // verus!
