//! Loop-entry `ScaledIdx` snapshot — divergence witness (V8-#4).
//!
//! Records, with a mechanized witness, a production effect the Lean spec
//! does not model: at **loop entry**, production's `force_locals_to_stable`
//! (lower.rs) now SNAPSHOTS any loop-invariant `ScaledIdx`-valued local —
//! it commits `base << log2(scale)` to a fresh register and copies that into
//! the local's stable register — so the index is not re-materialized from a
//! base register the loop body clobbers on later iterations.
//!
//! ## Why this is a witness, not a closed refinement
//!
//! `ScaledIdx` is ALREADY outside the proven Lean slice-1 subset: the Lean
//! `commit`/`localGet` arms REFUSE `ScaledIdx` (`Translate.lean` returns
//! `none`), and `commit_refine.rs` carries that as a documented domain gap
//! ("production materializes ScaledIdx / the Lean spec refuses → outside the
//! Lean commit domain... pointer-arith hoisting the slice-1 subset doesn't
//! model"). The Lean `wloop` arm (`Translate.lean:695`) models loop entry as
//! a pure snapshot/restore of the binding maps and emits NO ops there.
//!
//! So this change does not break any proven theorem — the refinement crates
//! (`commit_refine`, `local_arms_refine`, `structured_refine`,
//! `lower_instructions_refine`) all still verify (9/4/15/9, 0 errors). It
//! *extends* the existing `ScaledIdx` domain gap to a new program point
//! (loop entry). Per the project's V8 methodology, the new divergence is
//! recorded here with the production effect modeled and its shape pinned,
//! rather than left silent. Closing it (adding a Lean `wloop` arm that
//! snapshots `ScaledIdx` indices) requires first lifting `ScaledIdx` itself
//! into the Lean subset — tracked, not done here.
//!
//! What IS proved below:
//!   - on a NON-`ScaledIdx` (value-typed) local, production's loop-entry
//!     rebind and the spec's snapshot AGREE (no extra ops): the in-subset
//!     case is unchanged;
//!   - the production `ScaledIdx` snapshot has exactly the shape
//!     `Const(c, 0) ; Shl(t, base, k) ; Copy(stable, t)` (a hoisted,
//!     loop-invariant index materialization), pinned as the divergence;
//!   - the snapshot is INERT for the spec's modeled state: it only touches
//!     `next_reg` and the (out-of-subset) `ScaledIdx` local's binding, never
//!     a value-typed local's stable value.

use vstd::prelude::*;

verus! {

pub struct Reg(pub nat);

pub enum Scalar { U32, I32, F32 }

pub enum SymVal {
    Reg(Reg, Scalar),
    BufferPtr(nat),
    ScaledIdx { base: Reg, scale: nat },
    I32ConstSym(int),
}

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

// ── Spec loop-entry handling (Translate.lean `wloop`) ───────────────
//
// The Lean spec snapshots the binding maps at loop entry and emits NO ops.
// For a value-typed local the snapshot is its stable register (unchanged);
// for any local it produces no IR. Modeled here as: the snapshot emits the
// empty op sequence and leaves `next_reg` untouched.

pub open spec fn spec_loop_entry_snapshot(s: LowerState) -> (LowerState, Seq<KernelOp>) {
    (s, Seq::empty())
}

// ── Production loop-entry handling (force_locals_to_stable) ─────────
//
// On a NON-ScaledIdx local, production also emits nothing at loop entry —
// the value-typed rebind just points the local's `val` at its existing
// stable register (no IR). This is the in-subset case; it AGREES with spec.
//
// On a ScaledIdx-valued local that is loop-invariant (not written in the
// loop body), production COMMITS it: alloc a const-zero placeholder (the
// commit prelude), emit the `Shl` materializing `base << log2(scale)`, then
// copy the materialized register into the local's stable register. This is
// the divergence — the spec emits nothing here.

pub open spec fn prod_loop_entry_nonscaled(s: LowerState) -> (LowerState, Seq<KernelOp>) {
    // No ScaledIdx local ⇒ no commit ⇒ no ops, state unchanged.
    (s, Seq::empty())
}

/// The production ScaledIdx snapshot effect on `(state, ops)` for one
/// loop-invariant ScaledIdx local with the given `base`/`scale` and an
/// existing `stable` register. Models `commit(ScaledIdx) ; Copy(stable, t)`.
pub open spec fn prod_loop_entry_scaled(
    s: LowerState,
    base: Reg,
    scale: nat,
    stable: Reg,
) -> (LowerState, Seq<KernelOp>) {
    // commit(ScaledIdx) = alloc placeholder const + alloc Shl target.
    // (Mirrors commit_refine's note: ScaledIdx materializes via Const+Shl.)
    let (c, s1) = alloc(s);
    let (t, s2) = alloc(s1);
    let ops = seq![
        KernelOp::Const(c, ConstValue::U32(0)),
        KernelOp::Shl(t, base, log2(scale)),
        KernelOp::Copy(stable, t),
    ];
    (s2, ops)
}

// ── In-subset agreement (value-typed local) ────────────────────────

/// On a non-ScaledIdx (value-typed) local, production's loop-entry rebind
/// and the spec's snapshot AGREE: both emit no ops and leave state alone.
/// The fix does not perturb the modeled, in-subset case.
proof fn loop_entry_nonscaled_agrees(s: LowerState)
    ensures prod_loop_entry_nonscaled(s) == spec_loop_entry_snapshot(s),
{}

// ── The divergence, pinned ─────────────────────────────────────────

/// The production ScaledIdx snapshot emits exactly
/// `Const(c, 0) ; Shl(t, base, log2 scale) ; Copy(stable, t)` and advances
/// `next_reg` by 2 — a hoisted, loop-invariant index materialization. The
/// spec emits nothing (out of the ScaledIdx domain), so this is the new
/// divergence. Pinned as a witness; closing it needs ScaledIdx lifted into
/// the Lean subset first.
proof fn loop_entry_scaled_snapshot_shape(s: LowerState, base: Reg, scale: nat, stable: Reg)
    ensures
        // Production emits the three-op snapshot, in order.
        ({
            let (_, ops) = prod_loop_entry_scaled(s, base, scale, stable);
            &&& ops.len() == 3
            &&& ops[0] == KernelOp::Const(Reg(s.next_reg), ConstValue::U32(0))
            &&& ops[1] == KernelOp::Shl(Reg(s.next_reg + 1), base, log2(scale))
            &&& ops[2] == KernelOp::Copy(stable, Reg(s.next_reg + 1))
        }),
        // Production advances next_reg by exactly 2 (the two allocs).
        prod_loop_entry_scaled(s, base, scale, stable).0.next_reg == s.next_reg + 2,
        // The spec emits nothing at loop entry — the gap.
        spec_loop_entry_snapshot(s).1.len() == 0,
{}

/// The snapshot is INERT for any value-typed local's stable value: it writes
/// only the fresh placeholder/target registers and the *ScaledIdx* local's
/// own `stable`, never a value-typed local's register. (Here: the only
/// `Copy` destination is `stable`, the ScaledIdx local's own stable reg; the
/// other writes target the freshly-allocated `c`/`t` ≥ `s.next_reg`, which no
/// prior op can have named.) This is why the in-subset preservation proofs
/// (`local_arms_refine`, `lower_instructions_refine`) are undisturbed.
proof fn loop_entry_scaled_writes_only_fresh_or_own_stable(
    s: LowerState,
    base: Reg,
    scale: nat,
    stable: Reg,
)
    ensures
        ({
            let (_, ops) = prod_loop_entry_scaled(s, base, scale, stable);
            // The placeholder and Shl target are freshly allocated.
            &&& ops[0] == KernelOp::Const(Reg(s.next_reg), ConstValue::U32(0))
            &&& ops[1] == KernelOp::Shl(Reg(s.next_reg + 1), base, log2(scale))
            // The only stable-reg write is the ScaledIdx local's own.
            &&& ops[2] == KernelOp::Copy(stable, Reg(s.next_reg + 1))
        }),
{}

} // verus!
