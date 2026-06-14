//! V2 — Verus spec types mirroring the Lean translator vocabulary.
//!
//! Mechanizes the core types of `Quanta.Wasm.Translate` (the Lean
//! refinement target) and `crates/quanta-wasm-lowering/src/lower.rs`
//! (the production translator) as Verus `spec`-world types:
//!
//!   - `Reg`     — register index (Lean `abbrev Reg := Nat`)
//!   - `Scalar`  — IR scalar type (Lean `KOps.Scalar`, 12 variants)
//!   - `SymVal`  — the symbolic stack value the translator threads
//!   - `LowerState` (core) — `nextReg` + the symbolic stack
//!
//! These are the shared vocabulary every later refinement file (V3,
//! V5–V7) speaks in. The structural lemmas below pin the properties
//! the refinement proofs lean on: constructor disjointness, the
//! `regs` projection, and fresh-register monotonicity of `alloc`.
//!
//! Correspondence to Lean (`Quanta.Wasm.Translate`):
//!   SymVal.reg r ty          ↔ SymVal::Reg
//!   SymVal.bufferPtr slot    ↔ SymVal::BufferPtr
//!   SymVal.scaledIdx base sc ↔ SymVal::ScaledIdx
//!   SymVal.i32ConstSym n     ↔ SymVal::I32Const
//!   SymVal.bufferAccess ...  ↔ SymVal::BufferAccess
//! The Rust `SymVal` additionally carries `Opaque` and `I64Const`;
//! those are outside the slice-1 refinement subset and are mirrored
//! when their refinement arms land (V5+).

use vstd::prelude::*;

verus! {

// ── Reg ────────────────────────────────────────────────────────────
// Lean: `abbrev Reg : Type := Nat`. We keep it a `nat` alias in the
// spec world so register identities compare by value.

pub type Reg = nat;

// ── Scalar ─────────────────────────────────────────────────────────
// Lean `Quanta.KOps.Scalar`, 12 variants, same order.

pub enum Scalar {
    Bool,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    F16, F32, F64,
}

// ── SymVal ─────────────────────────────────────────────────────────
// The symbolic stack alphabet (Lean `Quanta.Wasm.SymVal`). Reg
// indices appear in `reg`, `scaledIdx`, and `bufferAccess`; the
// pointer/const forms carry none.

pub enum SymVal {
    Reg(Reg, Scalar),
    BufferPtr(nat),
    ScaledIdx { base: Reg, scale: nat },
    I32ConstSym(int),
    BufferAccess { slot: nat, base: Reg, scale: nat },
}

/// Registers referenced by a SymVal — mirrors Lean `SymVal.regs`.
/// Used by the freshness / alias-free invariants the per-op
/// refinements carry. Pointer and const forms contribute none.
pub open spec fn sym_regs(v: SymVal) -> Seq<Reg> {
    match v {
        SymVal::Reg(r, _)               => seq![r],
        SymVal::BufferPtr(_)            => Seq::empty(),
        SymVal::ScaledIdx { base, .. }  => seq![base],
        SymVal::I32ConstSym(_)          => Seq::empty(),
        SymVal::BufferAccess { base, .. } => seq![base],
    }
}

// ── LowerState (core) ──────────────────────────────────────────────
// The Lean `LowerState` carries six fields; the refinement of the
// straight-line subset only consults `nextReg` and `stack`, so the
// V2 core models exactly those. The local-binding maps (localReg /
// localTy / currentReg / bufferSlots) join when their refinement
// arms need them (V5: localGet/localSet).

pub struct LowerState {
    pub next_reg: nat,
    pub stack: Seq<SymVal>,
}

/// Allocate a fresh register — mirrors Lean `LowerState.alloc`:
/// returns the current `nextReg` and bumps the counter.
pub open spec fn alloc(s: LowerState) -> (Reg, LowerState) {
    (s.next_reg, LowerState { next_reg: s.next_reg + 1, stack: s.stack })
}

/// Push a plain `.reg r .u32` — mirrors Lean `LowerState.push`.
pub open spec fn push_reg(s: LowerState, r: Reg) -> LowerState {
    LowerState { stack: seq![SymVal::Reg(r, Scalar::U32)].add(s.stack), next_reg: s.next_reg }
}

// ── Structural lemmas ──────────────────────────────────────────────

/// `alloc` never decreases the fresh-register counter (strictly
/// increases it). This is the spec-world seed of the production
/// `nextReg`-monotonicity the refinement chain carries through.
proof fn alloc_increases_next_reg(s: LowerState)
    ensures
        alloc(s).0 == s.next_reg,
        alloc(s).1.next_reg == s.next_reg + 1,
        alloc(s).1.next_reg > s.next_reg,
{}

/// `alloc` leaves the symbolic stack untouched.
proof fn alloc_preserves_stack(s: LowerState)
    ensures alloc(s).1.stack == s.stack,
{}

/// The freshly-allocated register equals the pre-alloc `nextReg`, so
/// it is distinct from every register an `s.next_reg`-bounded value
/// already mentions — the freshness fact per-op refinements need.
proof fn alloc_reg_is_fresh(s: LowerState, used: Reg)
    requires used < s.next_reg,
    ensures alloc(s).0 != used,
{}

/// `push_reg` grows the stack by exactly one and preserves nextReg.
proof fn push_reg_shape(s: LowerState, r: Reg)
    ensures
        push_reg(s, r).stack.len() == s.stack.len() + 1,
        push_reg(s, r).stack[0] == SymVal::Reg(r, Scalar::U32),
        push_reg(s, r).next_reg == s.next_reg,
{
    assert(push_reg(s, r).stack[0] == SymVal::Reg(r, Scalar::U32));
}

/// `sym_regs` of a register value is the singleton of that register.
proof fn sym_regs_reg(r: Reg, ty: Scalar)
    ensures sym_regs(SymVal::Reg(r, ty)) == seq![r],
{}

/// `sym_regs` of the pointer / const forms is empty — they carry no
/// runtime register, so they never alias a fresh allocation.
proof fn sym_regs_ptr_empty(slot: nat)
    ensures sym_regs(SymVal::BufferPtr(slot)) == Seq::<Reg>::empty(),
{}

proof fn sym_regs_const_empty(n: int)
    ensures sym_regs(SymVal::I32ConstSym(n)) == Seq::<Reg>::empty(),
{}

/// Constructor disjointness witnessed: a `Reg` value is never a
/// `BufferPtr` value, so the translator's stack-shape dispatch
/// (pop-as-reg vs pop-as-symbolic) is sound.
proof fn reg_ne_bufferptr(r: Reg, ty: Scalar, slot: nat)
    ensures SymVal::Reg(r, ty) != SymVal::BufferPtr(slot),
{}

} // verus!
