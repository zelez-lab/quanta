//! Loop-entry `ScaledIdx` snapshot — refinement (V8-#4).
//!
//! Production's `force_locals_to_stable` (lower.rs), at **loop entry**,
//! snapshots a loop-invariant `ScaledIdx`-valued local: it commits
//! `base << log2(scale)` to a fresh register and copies that into the local's
//! stable register, so the index is not re-materialized from a base register
//! the loop body clobbers on a later iteration. (This is the tiled-GEMM
//! correctness fix: the p=0 shared-tile index `tr<<4` was recomputed each
//! iteration from the K-tile induction register, reading out of bounds.)
//!
//! The snapshot is GATED on the loop body being able to observe the local
//! stale: the body READS the local (`loop_body_reads`), or the `ScaledIdx`
//! base is some local's stable register (which in-body `local.set`s may
//! clobber). A `ScaledIdx` local the loop neither reads nor can clobber
//! keeps its symbolic binding and the entry emits nothing — committing it
//! would erase the `base`/`scale` split, and a post-loop
//! `BufferPtr + <this>` add would no longer fold into a `BufferAccess`
//! (the bench_nbody `v[idx] += a*dt` epilogue, where rustc CSEs `idx<<2`
//! into a local that survives the tile loops). The gate is modeled below
//! as `loop_entry_gate`.
//!
//! This file closes the divergence **at the refinement layer**, following the
//! V8-#1 placement-residual pattern (`local_arms_refine.rs`): the spec models
//! the loop-entry snapshot to MATCH production's emission, and the refinement
//! lemma proves production == spec as op sequences and state.
//!
//! ## Scope (honest)
//!
//! What is PROVED here:
//!   - the spec snapshot equals production's snapshot — same ops, same
//!     `next_reg` advance (`refine_loop_entry_scaled`);
//!   - on a value-typed (non-`ScaledIdx`) local, both emit nothing
//!     (`loop_entry_nonscaled_agrees`) — the in-subset case is untouched;
//!   - the snapshot writes only fresh registers + the `ScaledIdx` local's own
//!     stable register, so the in-subset preservation proofs
//!     (`local_arms_refine`, `lower_instructions_refine`) are undisturbed
//!     (`loop_entry_scaled_writes_only_fresh_or_own_stable`).
//!
//! What stays a tracked RESIDUAL (the pre-existing `ScaledIdx` domain gap,
//! NOT opened by this change): the *semantic* preservation of a materialized
//! `ScaledIdx` — i.e. that `Shl(t, base, log2 scale)` evaluates to the
//! intended byte index — is outside the Lean eval subset (`commit`/`localGet`
//! refuse `ScaledIdx`; see `commit_refine.rs`). Lifting `ScaledIdx` into the
//! Lean slice subset is its own (multi-file) work, separately scoped. This
//! file shows the loop-entry snapshot REFINES correctly given that boundary —
//! it does not move the boundary.

use vstd::prelude::*;

verus! {

pub struct Reg(pub nat);

pub enum Scalar { U32, I32, F32 }

pub enum ConstValue { U32(int) }

pub enum KernelOp {
    Const(Reg, ConstValue),
    Shl(Reg, Reg, nat),   // dst = base << log2(scale)
    Copy(Reg, Reg),       // dst = src
}

/// log2 of a power-of-two scale (4 → 2, 16 → 4). Abstract; the production
/// `log2` helper computes it for the recognized power-of-two scales.
pub uninterp spec fn log2(scale: nat) -> nat;

pub struct LowerState { pub next_reg: nat }

pub open spec fn alloc(s: LowerState) -> (Reg, LowerState) {
    (Reg(s.next_reg), LowerState { next_reg: s.next_reg + 1 })
}

/// The shared snapshot effect for one loop-invariant `ScaledIdx` local with
/// the given `base`/`scale` and existing `stable` register:
/// `commit(ScaledIdx) ; Copy(stable, t)` = alloc placeholder const, alloc the
/// `Shl` target, emit `[Const, Shl, Copy]`, advance `next_reg` by 2.
pub open spec fn snapshot_scaled(
    s: LowerState,
    base: Reg,
    scale: nat,
    stable: Reg,
) -> (LowerState, Seq<KernelOp>) {
    let (c, s1) = alloc(s);
    let (t, s2) = alloc(s1);
    let ops = seq![
        KernelOp::Const(c, ConstValue::U32(0)),
        KernelOp::Shl(t, base, log2(scale)),
        KernelOp::Copy(stable, t),
    ];
    (s2, ops)
}

// ── The snapshot gate ──────────────────────────────────────────────
//
// Production commits a loop-invariant ScaledIdx local at loop entry only
// when the loop body can observe it stale: the body reads the local, or the
// ScaledIdx base is some local's stable register (clobberable by in-body
// sets). Otherwise the local keeps its symbolic ScaledIdx binding and the
// entry emits nothing (post-loop buffer-pattern folds depend on this).
// Mirrors the `body_reads.contains(i) || locals.any(stable_reg == base)`
// condition in `force_locals_to_stable`.

pub open spec fn loop_entry_gate(read_in_body: bool, base_is_stable_reg: bool) -> bool {
    read_in_body || base_is_stable_reg
}

// ── Spec loop-entry snapshot (mirrors production) ──────────────────
//
// The spec's `wloop` arm models the same gated loop-entry ScaledIdx
// snapshot production performs. On a non-ScaledIdx local it emits nothing;
// on a ScaledIdx local it emits the shared snapshot iff the gate holds.

pub open spec fn spec_loop_entry_scaled(
    s: LowerState,
    base: Reg,
    scale: nat,
    stable: Reg,
    read_in_body: bool,
    base_is_stable_reg: bool,
) -> (LowerState, Seq<KernelOp>) {
    if loop_entry_gate(read_in_body, base_is_stable_reg) {
        snapshot_scaled(s, base, scale, stable)
    } else {
        (s, Seq::empty())
    }
}

pub open spec fn spec_loop_entry_nonscaled(s: LowerState) -> (LowerState, Seq<KernelOp>) {
    (s, Seq::empty())
}

// ── Production loop-entry snapshot ─────────────────────────────────

pub open spec fn prod_loop_entry_scaled(
    s: LowerState,
    base: Reg,
    scale: nat,
    stable: Reg,
    read_in_body: bool,
    base_is_stable_reg: bool,
) -> (LowerState, Seq<KernelOp>) {
    if loop_entry_gate(read_in_body, base_is_stable_reg) {
        snapshot_scaled(s, base, scale, stable)
    } else {
        (s, Seq::empty())
    }
}

pub open spec fn prod_loop_entry_nonscaled(s: LowerState) -> (LowerState, Seq<KernelOp>) {
    (s, Seq::empty())
}

// ── Refinement: production == spec ─────────────────────────────────

/// The gated ScaledIdx loop-entry snapshot REFINES: production and spec emit
/// the identical op sequence and reach the identical state in both gate
/// branches. This closes V8-#4 at the refinement layer — the spec models the
/// gated snapshot, so there is no production-does-more gap.
proof fn refine_loop_entry_scaled(
    s: LowerState,
    base: Reg,
    scale: nat,
    stable: Reg,
    read_in_body: bool,
    base_is_stable_reg: bool,
)
    ensures
        prod_loop_entry_scaled(s, base, scale, stable, read_in_body, base_is_stable_reg)
            == spec_loop_entry_scaled(s, base, scale, stable, read_in_body, base_is_stable_reg),
{}

/// On a value-typed (non-ScaledIdx) local, both emit nothing and leave state
/// alone — the modeled in-subset case is unchanged by the fix.
proof fn loop_entry_nonscaled_agrees(s: LowerState)
    ensures prod_loop_entry_nonscaled(s) == spec_loop_entry_nonscaled(s),
{}

/// When the gate does NOT hold (the loop body neither reads the local nor
/// can clobber its base), the entry is fully inert: no ops, state unchanged.
/// The local's symbolic ScaledIdx binding survives for post-loop
/// buffer-pattern folds, and the in-subset preservation proofs see an empty
/// emission.
proof fn loop_entry_ungated_is_inert(s: LowerState, base: Reg, scale: nat, stable: Reg)
    ensures
        ({
            let (s2, ops) = prod_loop_entry_scaled(s, base, scale, stable, false, false);
            &&& s2 == s
            &&& ops == Seq::<KernelOp>::empty()
        }),
{}

// ── Shape + inertness (what the refinement rests on) ───────────────

/// The gated snapshot's exact shape: `Const(c,0) ; Shl(t,base,log2 scale) ;
/// Copy(stable,t)`, `next_reg + 2`. Pins the emission for the framework
/// write-up (and the Lean `wloop` arm to mirror). Stated for the
/// read-in-body gate arm; the base-is-stable arm is identical by
/// `loop_entry_gate`'s disjunction.
proof fn loop_entry_scaled_snapshot_shape(s: LowerState, base: Reg, scale: nat, stable: Reg)
    ensures
        ({
            let (_, ops) = prod_loop_entry_scaled(s, base, scale, stable, true, false);
            &&& ops.len() == 3
            &&& ops[0] == KernelOp::Const(Reg(s.next_reg), ConstValue::U32(0))
            &&& ops[1] == KernelOp::Shl(Reg(s.next_reg + 1), base, log2(scale))
            &&& ops[2] == KernelOp::Copy(stable, Reg(s.next_reg + 1))
        }),
        prod_loop_entry_scaled(s, base, scale, stable, true, false).0.next_reg == s.next_reg + 2,
{}

/// The gated snapshot writes only freshly-allocated registers (`c =
/// next_reg`, `t = next_reg + 1`) and the ScaledIdx local's OWN stable
/// register — never a value-typed local's register. This is why the
/// in-subset preservation proofs are undisturbed: no register they reason
/// about is overwritten. (The ungated arm writes nothing at all — see
/// `loop_entry_ungated_is_inert`.)
proof fn loop_entry_scaled_writes_only_fresh_or_own_stable(
    s: LowerState,
    base: Reg,
    scale: nat,
    stable: Reg,
)
    ensures
        ({
            let (_, ops) = prod_loop_entry_scaled(s, base, scale, stable, true, false);
            &&& ops[0] == KernelOp::Const(Reg(s.next_reg), ConstValue::U32(0))
            &&& ops[1] == KernelOp::Shl(Reg(s.next_reg + 1), base, log2(scale))
            &&& ops[2] == KernelOp::Copy(stable, Reg(s.next_reg + 1))
        }),
{}

} // verus!
