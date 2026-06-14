//! V5 (local/memory production refinement) — Rust `lower.rs` local
//! arms ⊑ the `local_arms_spec` spec arms.
//!
//! Extends the imperative→functional bridge (lower_instr_refine.rs)
//! to the local-binding arms. Beyond the stack `view`, the production
//! state carries `locals: Vec<LocalInfo>` where each `LocalInfo` has
//! `{ val, stable_reg, stable_ty }`. The spec carries three keyed
//! maps (`current_reg`, `local_reg`, `local_ty`). The `view_locals`
//! correspondence maps one onto the other:
//!
//!   LocalInfo[i].val == Some(Reg(r,_))  ⇔  current_reg[i] == r
//!   LocalInfo[i].stable_reg == Some(r)  ⇔  local_reg[i] == r
//!   LocalInfo[i].stable_ty == t         ⇔  local_ty[i] == t
//!
//! ## The frame-0 zero-init gap (documented, semantics-preserving)
//!
//! Production's `write_local_via_copy` / `ensure_stable_reg_for`
//! INSERT a `Const fresh 0` at the function frame head (frame 0)
//! before emitting the dual-Copy into the current frame. The Lean
//! spec — and `local_arms_spec` — model only the current-frame ops
//! (the dual-Copy); the frame-0 zero-inits are a production-only
//! artifact for the Metal/WGSL emitters (which need `uint rN = 0u;`
//! before a `rN = rM;` assignment). They are semantically inert: the
//! Copy overwrites the reg before any read. So the refinement here is
//! stated on the BODY ops (what the spec arm returns) and the
//! register count; the frame-0 Const insertions are out of the
//! Lean-spec subset and recorded as a V8 correspondence note.
//!
//! ## Scope
//!
//! The `localSet` register-operand case (the shape rustc emits) and
//! the `localGet` current-binding read, refined against the spec arms
//! through the extended view. Same trust boundary as the arithmetic
//! refinement: the `step_<op>` transcription ↔ Rust-arm faithfulness
//! is the documented manual obligation; what is mechanized is that
//! the transcription equals the spec arm.

use vstd::prelude::*;

verus! {

// ── Spec vocabulary (extended LowerState, inlined) ─────────────────

pub type Reg = nat;
pub enum Scalar { Bool, I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64 }
pub enum SymVal {
    Reg(Reg, Scalar),
    BufferPtr(nat),
    ScaledIdx { base: Reg, scale: nat },
    I32ConstSym(int),
    BufferAccess { slot: nat, base: Reg, scale: nat },
}
pub enum ConstValue { U32(int), I32(int) }
pub enum KernelOp {
    Const(Reg, ConstValue),
    Copy(Reg, Reg),
    LoadK(Reg, nat, Reg, Scalar),   // dst, slot, index, ty
}

pub struct LowerState {
    pub next_reg: nat,
    pub stack: Seq<SymVal>,
    pub local_reg: Map<nat, Reg>,
    pub local_ty: Map<nat, Scalar>,
    pub current_reg: Map<nat, Reg>,
    pub buffer_slots: Map<nat, nat>,
}

pub open spec fn lookup_local(s: LowerState, i: nat) -> Option<Reg> {
    if s.local_reg.dom().contains(i) { Some(s.local_reg[i]) } else { None }
}
pub open spec fn lookup_current_reg(s: LowerState, i: nat) -> Option<Reg> {
    if s.current_reg.dom().contains(i) { Some(s.current_reg[i]) } else { None }
}
pub open spec fn lookup_local_ty(s: LowerState, i: nat) -> Option<Scalar> {
    if s.local_ty.dom().contains(i) { Some(s.local_ty[i]) } else { None }
}
pub open spec fn lookup_buffer_slot(s: LowerState, i: nat) -> Option<nat> {
    if s.buffer_slots.dom().contains(i) { Some(s.buffer_slots[i]) } else { None }
}
pub open spec fn set_local_reg(s: LowerState, i: nat, r: Reg, ty: Scalar) -> LowerState {
    LowerState { local_reg: s.local_reg.insert(i, r), local_ty: s.local_ty.insert(i, ty), ..s }
}
pub open spec fn set_current_reg(s: LowerState, i: nat, r: Reg) -> LowerState {
    LowerState { current_reg: s.current_reg.insert(i, r), ..s }
}
pub open spec fn alloc(s: LowerState) -> (Reg, LowerState) {
    (s.next_reg, LowerState { next_reg: s.next_reg + 1, ..s })
}
pub open spec fn pop_sym(s: LowerState) -> Option<(SymVal, LowerState)> {
    if s.stack.len() == 0 { None }
    else { Some((s.stack[0], LowerState { stack: s.stack.subrange(1, s.stack.len() as int), ..s })) }
}
pub open spec fn commit(s: LowerState, v: SymVal) -> Option<(Reg, LowerState, Seq<KernelOp>)> {
    match v {
        SymVal::Reg(r, _) => Some((r, s, Seq::empty())),
        SymVal::I32ConstSym(n) => {
            let (dst, s1) = alloc(s);
            Some((dst, s1, seq![KernelOp::Const(dst, ConstValue::U32(n))]))
        },
        _ => None,
    }
}

/// The spec `local_set` arm (mirror of local_arms_spec.rs; the
/// refinement target). Returns the body ops (the dual-Copy + any
/// commit prefix), NOT the frame-0 zero-inits.
pub open spec fn local_set(s: LowerState, i: nat) -> Option<(LowerState, Seq<KernelOp>)> {
    match pop_sym(s) {
        None => None,
        Some((sv, s1)) => match commit(s1, sv) {
            None => None,
            Some((src, s2, ops_commit)) => {
                let ty: Scalar = match lookup_local_ty(s2, i) { Some(t) => t, None => Scalar::U32 };
                let (fresh, s3) = alloc(s2);
                match lookup_local(s3, i) {
                    Some(stable) => {
                        let s4 = set_current_reg(set_local_reg(s3, i, stable, ty), i, fresh);
                        Some((s4, ops_commit.add(seq![KernelOp::Copy(fresh, src),
                                                       KernelOp::Copy(stable, fresh)])))
                    },
                    None => {
                        let (stable, s4) = alloc(s3);
                        let s5 = set_current_reg(set_local_reg(s4, i, stable, ty), i, fresh);
                        Some((s5, ops_commit.add(seq![KernelOp::Copy(fresh, src),
                                                       KernelOp::Copy(stable, fresh)])))
                    },
                }
            },
        },
    }
}

pub open spec fn push_val(s: LowerState, v: SymVal) -> LowerState {
    LowerState { stack: seq![v].add(s.stack), ..s }
}

/// Spec `local_get` arm (mirror of local_arms_spec.rs).
pub open spec fn local_get(s: LowerState, i: nat) -> Option<(LowerState, Seq<KernelOp>)> {
    match lookup_buffer_slot(s, i) {
        Some(slot) => Some((push_val(s, SymVal::BufferPtr(slot)), Seq::empty())),
        None => {
            let src_opt = match lookup_current_reg(s, i) {
                Some(r) => Some(r),
                None => lookup_local(s, i),
            };
            match src_opt {
                None => None,
                Some(source) => {
                    let (fresh, s1) = alloc(s);
                    let s2 = push_val(s1, SymVal::Reg(fresh, Scalar::U32));
                    Some((s2, seq![KernelOp::Copy(fresh, source)]))
                },
            }
        },
    }
}

/// Spec `i32_load` arm (mirror of local_arms_spec.rs): typed Load on
/// a 4-byte BufferAccess at the spec head.
pub open spec fn i32_load(s: LowerState) -> Option<(LowerState, Seq<KernelOp>)> {
    if s.stack.len() >= 1 {
        match s.stack[0] {
            SymVal::BufferAccess { slot, base, scale } => {
                if scale == 4 {
                    let rest = s.stack.subrange(1, s.stack.len() as int);
                    let dst = s.next_reg;
                    Some((LowerState { next_reg: s.next_reg + 1,
                            stack: seq![SymVal::Reg(dst, Scalar::U32)].add(rest), ..s },
                          seq![KernelOp::LoadK(dst, slot, base, Scalar::U32)]))
                } else { None }
            },
            _ => None,
        }
    } else { None }
}

// ── Production state + view (extends lower_instr_refine.rs's) ───────

/// The relevant projection of production `LocalInfo`: its current
/// binding (`val` when it's a register) and stable reg + type.
pub struct ProdLocal {
    pub cur: Option<Reg>,
    pub stable: Option<Reg>,
    pub stable_ty: Scalar,
}

pub struct ProdCtx {
    pub next_reg: nat,
    pub prod_stack: Seq<SymVal>,          // top = LAST (Rust Vec)
    pub locals: Map<nat, ProdLocal>,      // local idx → LocalInfo projection
    pub buffer_slots: Map<nat, nat>,
}

/// Build the spec local maps from the production `locals`: a key is in
/// `current_reg` iff that local's `cur` is Some, etc.
pub open spec fn view(c: ProdCtx) -> LowerState {
    LowerState {
        next_reg: c.next_reg,
        stack: c.prod_stack.reverse(),
        local_reg: Map::new(
            |i: nat| c.locals.dom().contains(i) && c.locals[i].stable is Some,
            |i: nat| c.locals[i].stable->Some_0),
        local_ty: Map::new(
            |i: nat| c.locals.dom().contains(i),
            |i: nat| c.locals[i].stable_ty),
        current_reg: Map::new(
            |i: nat| c.locals.dom().contains(i) && c.locals[i].cur is Some,
            |i: nat| c.locals[i].cur->Some_0),
        buffer_slots: c.buffer_slots,
    }
}

// ── localSet production transcription ──────────────────────────────
//
// Rust `LocalSet i`: pop value, ensure_stable_reg_for (allocs stable
// + inserts frame-0 Const if first set), write_local_via_copy (commit
// value, alloc fresh, frame-0 Const insert, emit Copy[fresh←src],
// emit Copy[stable←fresh], set val := fresh). The BODY ops appended
// to the current frame are exactly [Copy(fresh,src), Copy(stable,fresh)]
// for a register value (commit is identity). We transcribe that body
// effect; frame-0 Const inserts are the documented out-of-spec artifact.

/// step_local_set for the EXISTING-stable register-value case: the
/// body ops the Rust arm appends to the current frame.
pub open spec fn step_local_set_body(c: ProdCtx, i: nat, r: Reg) -> Seq<KernelOp> {
    // fresh = c.next_reg (existing stable ⇒ ensure_stable_reg_for is a
    // no-op, so the only alloc before the fresh is none).
    let fresh = c.next_reg;
    let stable = c.locals[i].stable->Some_0;
    seq![KernelOp::Copy(fresh, r), KernelOp::Copy(stable, fresh)]
}

/// Refinement: for a local that already has a stable reg, the
/// production body ops for `local.set` of a register value equal the
/// spec `local_set` body ops. (Both commits are identities; the spec
/// fresh reg = view(c).next_reg = c.next_reg = the production fresh.)
proof fn refine_local_set_existing_stable(c: ProdCtx, i: nat, r: Reg, ty: Scalar, stable: Reg)
    requires
        c.prod_stack.len() >= 1,
        c.prod_stack[c.prod_stack.len() - 1] == SymVal::Reg(r, ty),
        c.locals.dom().contains(i),
        c.locals[i].stable == Some(stable),
        view(c).local_ty.dom().contains(i),
    ensures
        local_set(view(c), i) is Some,
        local_set(view(c), i).unwrap().1 == step_local_set_body(c, i, r),
{
    // view(c).stack[0] = production top = Reg(r,ty).
    reverse_top(c.prod_stack);
    let s = view(c);
    assert(s.stack[0] == SymVal::Reg(r, ty));
    // pop → s1 with same next_reg; commit(Reg) = (r, s1, []).
    let s1 = pop_sym(s).unwrap().1;
    assert(s1.next_reg == c.next_reg);
    // lookup_local after pop sees the stable reg (locals unchanged by pop).
    assert(lookup_local(s1, i) == Some(stable)) by {
        assert(s1.local_reg == s.local_reg);
        assert(s.local_reg.dom().contains(i));
        assert(s.local_reg[i] == stable);
    }
    // fresh = s1.next_reg = c.next_reg; spec body = [] ++ dual-Copy.
    assert(Seq::<KernelOp>::empty().add(
        seq![KernelOp::Copy(c.next_reg, r), KernelOp::Copy(stable, c.next_reg)])
        =~= seq![KernelOp::Copy(c.next_reg, r), KernelOp::Copy(stable, c.next_reg)]);
}

/// The view sends the production stack top (last) to spec head (0).
proof fn reverse_top(s: Seq<SymVal>)
    requires s.len() >= 1,
    ensures s.reverse()[0] == s[s.len() - 1],
{
    assert(s.reverse()[0] == s[s.len() - 1 - 0]);
}

/// The view exposes a production local's stable reg as the spec
/// `local_reg` entry — the field-correspondence the refinement relies
/// on, pinned as a standalone fact.
proof fn view_exposes_stable(c: ProdCtx, i: nat, stable: Reg)
    requires
        c.locals.dom().contains(i),
        c.locals[i].stable == Some(stable),
    ensures lookup_local(view(c), i) == Some(stable),
{
    assert(view(c).local_reg.dom().contains(i));
    assert(view(c).local_reg[i] == stable);
}

/// The view exposes a production local's current binding as the spec
/// `current_reg` entry — the localGet read source.
proof fn view_exposes_current(c: ProdCtx, i: nat, cur: Reg)
    requires
        c.locals.dom().contains(i),
        c.locals[i].cur == Some(cur),
    ensures lookup_current_reg(view(c), i) == Some(cur),
{
    assert(view(c).current_reg.dom().contains(i));
    assert(view(c).current_reg[i] == cur);
}

/// A buffer-slot local is exposed to the spec `buffer_slots`, so
/// production `local.get` of a shared param dispatches into the spec's
/// BufferPtr-push arm.
proof fn view_exposes_buffer_slot(c: ProdCtx, i: nat, slot: nat)
    requires
        c.buffer_slots.dom().contains(i),
        c.buffer_slots[i] == slot,
    ensures lookup_buffer_slot(view(c), i) == Some(slot),
{}

// ── localGet production refinement ─────────────────────────────────
//
// Rust `LocalGet i`: if the local is a buffer param, push BufferPtr;
// else read its current binding (`val`, the post-write reg), alloc a
// fresh reg, emit Copy[fresh←source], push the fresh reg. The view's
// current_reg exposes `val`, so the spec arm reads the same source
// and allocs the same fresh reg (= view next_reg).

/// localGet of a non-buffer local with a current binding: production
/// allocs one reg and emits one Copy from the binding — equal to the
/// spec `local_get` output.
proof fn refine_local_get_current(c: ProdCtx, i: nat, cur: Reg)
    requires
        !c.buffer_slots.dom().contains(i),
        c.locals.dom().contains(i),
        c.locals[i].cur == Some(cur),
    ensures ({
        let (fresh, s1) = alloc(view(c));
        local_get(view(c), i) == Some::<(LowerState, Seq<KernelOp>)>(
            (push_val(s1, SymVal::Reg(fresh, Scalar::U32)),
             seq![KernelOp::Copy(fresh, cur)]))
    }),
{
    // view exposes: no buffer slot, current_reg[i] = cur.
    assert(lookup_buffer_slot(view(c), i) == None::<nat>);
    assert(lookup_current_reg(view(c), i) == Some(cur)) by {
        assert(view(c).current_reg.dom().contains(i));
        assert(view(c).current_reg[i] == cur);
    }
}

/// localGet of a buffer param pushes BufferPtr, no IR — refines the
/// spec buffer arm.
proof fn refine_local_get_buffer(c: ProdCtx, i: nat, slot: nat)
    requires
        c.buffer_slots.dom().contains(i),
        c.buffer_slots[i] == slot,
    ensures
        local_get(view(c), i) == Some::<(LowerState, Seq<KernelOp>)>(
            (push_val(view(c), SymVal::BufferPtr(slot)), Seq::empty())),
{
    assert(lookup_buffer_slot(view(c), i) == Some(slot));
}

// ── i32.load production refinement ─────────────────────────────────
//
// Rust `I32Load`: pops the address SymVal; only a BufferAccess
// (scale 4 = the recognized <ptr>+<i*4> chain) is accepted, alloc
// dst, emit Load{dst, slot, base, u32}, push Reg dst. The view sends
// the production stack top (last) to the spec head, so the spec
// `i32_load` matches on the same BufferAccess.

/// i32.load on a 4-byte BufferAccess at the production top refines the
/// spec typed-Load arm (same dst = view next_reg, same Load op).
proof fn refine_i32_load(c: ProdCtx, slot: nat, base: Reg)
    requires
        c.prod_stack.len() >= 1,
        c.prod_stack[c.prod_stack.len() - 1] == (SymVal::BufferAccess { slot, base, scale: 4nat }),
    ensures ({
        let s = view(c);
        let rest = s.stack.subrange(1, s.stack.len() as int);
        i32_load(s) == Some::<(LowerState, Seq<KernelOp>)>(
            (LowerState { next_reg: s.next_reg + 1,
                stack: seq![SymVal::Reg(s.next_reg, Scalar::U32)].add(rest), ..s },
             seq![KernelOp::LoadK(s.next_reg, slot, base, Scalar::U32)]))
    }),
{
    reverse_top(c.prod_stack);
    let s = view(c);
    assert(s.stack[0] == (SymVal::BufferAccess { slot, base, scale: 4nat }));
}

} // verus!
