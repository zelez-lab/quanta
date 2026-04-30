//! Verus mirror — tessellation pipeline invariants (steps 022 + 023).
//!
//! Mirrors `Quanta.Tessellation.Pipeline` from Lean. Every backend that
//! implements tessellation (Metal compute-kernel + drawIndexedPatches,
//! Vulkan TCS+TES stages, software fallback) refines this contract:
//!
//! - `create(topo, cp)` returns a fresh pipeline with the requested
//!   patch size, factor lists initialized to 1, live = true.
//! - `set_outer(i, f)` / `set_inner(i, f)` succeed iff the index is
//!   in-bounds for the topology and the pipeline is live; the stored
//!   factor is clamped into `[1, MAX_TESS_LEVEL]`.
//! - `destroy(p)` flips `live` to false; subsequent `set_*` fail.
//!
//! Theorems mirror the Lean proofs T7200–T7206:
//!   T7250 — fresh pipeline from create has the requested cp + cap +
//!           factor lists of the topology length.
//!   T7251 — clamp_factor stays in `[1, MAX_TESS_LEVEL]`.
//!   T7252 — set_outer extends only the target edge.
//!   T7253 — set_outer fails when out-of-bounds or destroyed.
//!   T7254 — set_inner mirrors set_outer.
//!   T7255 — destroy is idempotent on the live flag.
//!   T7256 — destroy blocks set_outer + set_inner.

use vstd::prelude::*;

verus! {

pub spec const MAX_TESS_LEVEL: nat = 64nat;
pub spec const MAX_PATCH_SIZE: nat = 32nat;

// 0 = triangle (3 outer + 1 inner)
// 1 = quad     (4 outer + 2 inner)
pub type Topology = u8;

pub open spec fn outer_count(t: Topology) -> nat {
    if t == 0u8 { 3nat } else { 4nat }
}

pub open spec fn inner_count(t: Topology) -> nat {
    if t == 0u8 { 1nat } else { 2nat }
}

pub struct TessPipeline {
    pub handle: u64,
    pub topology: Topology,
    pub control_points: nat,
    pub outer: Seq<nat>,
    pub inner: Seq<nat>,
    pub live: bool,
}

pub open spec fn clamp_factor(f: nat) -> nat {
    if f < 1nat {
        1nat
    } else if f > MAX_TESS_LEVEL {
        MAX_TESS_LEVEL
    } else {
        f
    }
}

pub open spec fn create(handle: u64, topo: Topology, cp: nat) -> Option<TessPipeline> {
    if 1nat <= cp && cp <= MAX_PATCH_SIZE {
        Option::Some(TessPipeline {
            handle,
            topology: topo,
            control_points: cp,
            outer: Seq::new(outer_count(topo), |_i: int| 1nat),
            inner: Seq::new(inner_count(topo), |_i: int| 1nat),
            live: true,
        })
    } else {
        Option::None
    }
}

pub open spec fn set_outer(p: TessPipeline, i: nat, f: nat) -> Option<TessPipeline> {
    if p.live && i < outer_count(p.topology) {
        Option::Some(TessPipeline {
            outer: p.outer.update(i as int, clamp_factor(f)),
            ..p
        })
    } else {
        Option::None
    }
}

pub open spec fn set_inner(p: TessPipeline, i: nat, f: nat) -> Option<TessPipeline> {
    if p.live && i < inner_count(p.topology) {
        Option::Some(TessPipeline {
            inner: p.inner.update(i as int, clamp_factor(f)),
            ..p
        })
    } else {
        Option::None
    }
}

pub open spec fn destroy(p: TessPipeline) -> TessPipeline {
    TessPipeline { live: false, ..p }
}

// ── T7250: create produces a well-formed pipeline ────────────────────────

proof fn t7250_create_fresh(handle: u64, topo: Topology, cp: nat)
    requires
        1nat <= cp,
        cp <= MAX_PATCH_SIZE,
    ensures
        create(handle, topo, cp) matches Option::Some(p) ==>
            p.handle == handle
            && p.topology == topo
            && p.control_points == cp
            && p.outer.len() == outer_count(topo)
            && p.inner.len() == inner_count(topo)
            && p.live == true,
{}

// ── T7251: clamp_factor is always in [1, MAX_TESS_LEVEL] ─────────────────

proof fn t7251_clamp_in_range(f: nat)
    ensures
        1nat <= clamp_factor(f),
        clamp_factor(f) <= MAX_TESS_LEVEL,
{}

// ── T7252: set_outer updates only the target edge ────────────────────────

proof fn t7252_set_outer_localizes(p: TessPipeline, i: nat, f: nat, p2: TessPipeline)
    requires
        p.live,
        i < outer_count(p.topology),
        p.outer.len() == outer_count(p.topology),
        i < p.outer.len(),
        set_outer(p, i, f) == Option::<TessPipeline>::Some(p2),
    ensures
        p2.handle == p.handle,
        p2.topology == p.topology,
        p2.control_points == p.control_points,
        p2.live == p.live,
        p2.outer.len() == p.outer.len(),
        p2.inner == p.inner,
        p2.outer.index(i as int) == clamp_factor(f),
{}

// ── T7253: set_outer fails OOB or when destroyed ─────────────────────────

proof fn t7253_set_outer_fails_oob(p: TessPipeline, i: nat, f: nat)
    requires
        !p.live || i >= outer_count(p.topology),
    ensures
        set_outer(p, i, f) == Option::<TessPipeline>::None,
{}

// ── T7254: set_inner mirrors set_outer ───────────────────────────────────

proof fn t7254_set_inner_localizes(p: TessPipeline, i: nat, f: nat, p2: TessPipeline)
    requires
        p.live,
        i < inner_count(p.topology),
        p.inner.len() == inner_count(p.topology),
        i < p.inner.len(),
        set_inner(p, i, f) == Option::<TessPipeline>::Some(p2),
    ensures
        p2.outer == p.outer,
        p2.inner.len() == p.inner.len(),
        p2.inner.index(i as int) == clamp_factor(f),
{}

// ── T7255: destroy invalidates and is idempotent ─────────────────────────

proof fn t7255_destroy_invalidates(p: TessPipeline)
    ensures
        destroy(p).live == false,
        destroy(p).handle == p.handle,
        destroy(p).topology == p.topology,
        destroy(p).control_points == p.control_points,
{}

proof fn t7255b_destroy_idempotent(p: TessPipeline)
    ensures
        destroy(destroy(p)) == destroy(p),
{}

// ── T7256: destroy blocks set_outer + set_inner ──────────────────────────

proof fn t7256_destroy_blocks_set_outer(p: TessPipeline, i: nat, f: nat)
    ensures
        set_outer(destroy(p), i, f) == Option::<TessPipeline>::None,
{}

proof fn t7256b_destroy_blocks_set_inner(p: TessPipeline, i: nat, f: nat)
    ensures
        set_inner(destroy(p), i, f) == Option::<TessPipeline>::None,
{}

}  // verus!
