//! Verus mirror — mesh shader pipeline invariants (steps 024 + 025).
//!
//! Mirrors `Quanta.MeshShader.Pipeline` from Lean. Every backend that
//! implements mesh shaders (Metal `MTLMeshRenderPipelineDescriptor`,
//! Vulkan `VK_EXT_mesh_shader`) refines this contract:
//!
//! - `create(max_v, max_p, task_t)` returns a fresh pipeline iff all
//!   three params are within hardware-minimum bounds (256/256/128).
//! - `dispatch(gx, gy, gz)` succeeds iff the pipeline is live and
//!   each axis is ≤ MAX_GROUP_COUNT (= 65535). The dispatch is
//!   appended to the recorded sequence; backends execute in order.
//! - `destroy(p)` flips `live` to false; subsequent `dispatch` fails.
//!
//! Theorems mirror the Lean proofs T7300–T7305:
//!   T7350 — fresh pipeline matches Lean shape.
//!   T7351 — dispatch appends one entry, preserves earlier order.
//!   T7352 — dispatch preserves limits.
//!   T7353 — dispatch fails OOB or destroyed.
//!   T7354 — destroy is idempotent + blocks dispatch.

use vstd::prelude::*;

verus! {

pub spec const MAX_MESH_VERTICES: nat = 256nat;
pub spec const MAX_MESH_PRIMITIVES: nat = 256nat;
pub spec const MAX_TASK_THREADS: nat = 128nat;
pub spec const MAX_GROUP_COUNT: nat = 65535nat;

pub struct MeshPipeline {
    pub handle: u64,
    pub max_vertices: nat,
    pub max_primitives: nat,
    pub task_threads: nat,
    pub dispatched: Seq<(nat, nat, nat)>,
    pub live: bool,
}

pub open spec fn bounds_ok(max_v: nat, max_p: nat, task_t: nat) -> bool {
    1nat <= max_v && max_v <= MAX_MESH_VERTICES
    && 1nat <= max_p && max_p <= MAX_MESH_PRIMITIVES
    && 1nat <= task_t && task_t <= MAX_TASK_THREADS
}

pub open spec fn group_ok(gx: nat, gy: nat, gz: nat) -> bool {
    gx <= MAX_GROUP_COUNT && gy <= MAX_GROUP_COUNT && gz <= MAX_GROUP_COUNT
}

pub open spec fn create(handle: u64, max_v: nat, max_p: nat, task_t: nat)
    -> Option<MeshPipeline>
{
    if bounds_ok(max_v, max_p, task_t) {
        Option::Some(MeshPipeline {
            handle,
            max_vertices: max_v,
            max_primitives: max_p,
            task_threads: task_t,
            dispatched: Seq::empty(),
            live: true,
        })
    } else {
        Option::None
    }
}

pub open spec fn dispatch(p: MeshPipeline, gx: nat, gy: nat, gz: nat)
    -> Option<MeshPipeline>
{
    if p.live && group_ok(gx, gy, gz) {
        Option::Some(MeshPipeline {
            dispatched: p.dispatched.push((gx, gy, gz)),
            ..p
        })
    } else {
        Option::None
    }
}

pub open spec fn destroy(p: MeshPipeline) -> MeshPipeline {
    MeshPipeline { live: false, ..p }
}

// ── T7350: create produces a well-formed pipeline ────────────────────────

proof fn t7350_create_fresh(handle: u64, max_v: nat, max_p: nat, task_t: nat)
    requires
        bounds_ok(max_v, max_p, task_t),
    ensures
        create(handle, max_v, max_p, task_t) matches Option::Some(p) ==>
            p.handle == handle
            && p.max_vertices == max_v
            && p.max_primitives == max_p
            && p.task_threads == task_t
            && p.dispatched.len() == 0
            && p.live == true,
{}

// ── T7351: dispatch appends one entry, preserves earlier order ───────────

proof fn t7351_dispatch_appends(
    p: MeshPipeline, gx: nat, gy: nat, gz: nat, p2: MeshPipeline,
)
    requires
        p.live,
        group_ok(gx, gy, gz),
        dispatch(p, gx, gy, gz) == Option::<MeshPipeline>::Some(p2),
    ensures
        p2.dispatched == p.dispatched.push((gx, gy, gz)),
        p2.dispatched.len() == p.dispatched.len() + 1,
{}

// ── T7352: dispatch preserves limits ─────────────────────────────────────

proof fn t7352_dispatch_preserves_limits(
    p: MeshPipeline, gx: nat, gy: nat, gz: nat, p2: MeshPipeline,
)
    requires
        dispatch(p, gx, gy, gz) == Option::<MeshPipeline>::Some(p2),
    ensures
        p2.max_vertices == p.max_vertices,
        p2.max_primitives == p.max_primitives,
        p2.task_threads == p.task_threads,
        p2.live == p.live,
        p2.handle == p.handle,
{}

// ── T7353: dispatch fails OOB or destroyed ───────────────────────────────

proof fn t7353_dispatch_fails(p: MeshPipeline, gx: nat, gy: nat, gz: nat)
    requires
        !p.live || !group_ok(gx, gy, gz),
    ensures
        dispatch(p, gx, gy, gz) == Option::<MeshPipeline>::None,
{}

// ── T7354: destroy invalidates + blocks dispatch ─────────────────────────

proof fn t7354_destroy_invalidates(p: MeshPipeline)
    ensures
        destroy(p).live == false,
        destroy(p).dispatched == p.dispatched,
        destroy(p).max_vertices == p.max_vertices,
{}

proof fn t7354b_destroy_idempotent(p: MeshPipeline)
    ensures
        destroy(destroy(p)) == destroy(p),
{}

proof fn t7354c_destroy_blocks_dispatch(p: MeshPipeline, gx: nat, gy: nat, gz: nat)
    ensures
        dispatch(destroy(p), gx, gy, gz) == Option::<MeshPipeline>::None,
{}

}  // verus!
