//! V7 — top-level composition: production `lower_instructions` ⊑ the
//! Lean `lowerInstrs` straight-line fold. **Closes the Verus arm.**
//!
//! The per-op refinements (V5) prove each single instruction's
//! production effect matches its V3 spec arm. V7 folds those into a
//! statement about the *whole instruction list*: lowering a list of
//! WASM instructions on the production side, then viewing, equals
//! lowering the viewed list on the spec side — for every list in the
//! slice-1 straight-line subset.
//!
//! ## What "straight-line subset" means here
//!
//! The Lean `lowerInstrs` (Translate.lean:550) has two layers:
//!
//!   1. **Structured control** (block / wloop / wif / br / brIf /
//!      wreturn) — fuel-bounded, frame-aware, consumes the inner body
//!      out of the stream via `splitAtEnd` / `splitAtElseOrEnd`.
//!   2. **The straight-line fold** (the `| _ =>` arm, Translate.lean:729):
//!      `lowerInstr` the head, recurse on the tail, concatenate the
//!      emitted ops — with neither fuel nor frames in play.
//!
//! Every instruction whose *per-op* refinement V5 proved (consts, the
//! binop/cmp families, the shl/add recognizers, drop/nop/wreturn, the
//! local/memory arms) flows through layer 2. V7 proves the fold over
//! layer 2 in full generality: any list drawn from that subset.
//!
//! The structured-control arms (layer 1) are a *separate* sub-milestone
//! — their per-arm production refinement is not yet proven on the Verus
//! side (only the Lean side is feature-complete; see
//! [[scope-valid-predicate-2026-05-29]] / [[stageB-agreement-2026-06-14]]).
//! On the straight-line subset, fuel and frames are inert: `lowerInstrs`
//! reduces *definitionally* to the layer-2 fold, so refining that fold
//! refines `lowerInstrs` on the whole subset.
//!
//! ## The fold and the invariant
//!
//! Production `lower_instructions` folds `step` over the list, threading
//! `ProdCtx` (the `Vec`-end stack + `next_reg`). The spec side folds
//! `lower_instr` over the viewed list, threading `LowerState` (the
//! `Seq`-head stack + `next_reg`). The composition theorem is:
//!
//!   view ∘ lower_instructions  ==  spec_lower_instrs ∘ view   (lifted
//!   through `Option`, on the ops too)
//!
//! proved by induction on the list. The inductive step is exactly the
//! single-op refinement (`refine_step` = `view(step(c,i)) ==
//! lower_instr(view(c), i)`) threaded through the `Option`-bind: if the
//! head refines and the tail refines from the resulting state, the
//! whole list refines. `next_reg` monotonicity and view-preservation
//! ride along inside `refine_step` — no separate invariant needed,
//! because `step` is *defined* to carry the spec result back through
//! `view`, so the threaded states agree at every position by
//! construction.

use vstd::prelude::*;

verus! {

// ── Spec vocabulary (V2/V3, inlined for standalone verification) ───

pub type Reg = nat;

pub enum Scalar { Bool, I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64 }

pub enum SymVal {
    Reg(Reg, Scalar),
    BufferPtr(nat),
    ScaledIdx { base: Reg, scale: nat },
    I32ConstSym(int),
    BufferAccess { slot: nat, base: Reg, scale: nat },
}

pub struct LowerState { pub next_reg: nat, pub stack: Seq<SymVal> }

pub enum ConstValue { U32(int), I32(int) }
pub enum BinOp { Add, Sub, Mul, Div, Rem, BAnd, BOr, BXor, Shl, Shr }
pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }
pub enum KernelOp {
    Const(Reg, ConstValue),
    BinOpK(Reg, Reg, Reg, BinOp, Scalar),
    CmpK(Reg, Reg, Reg, CmpOp, Scalar),
    Cast(Reg, Reg, Scalar, Scalar),
    Copy(Reg, Reg),
}

// ── State primitives (mirror V3) ───────────────────────────────────

pub open spec fn alloc(s: LowerState) -> (Reg, LowerState) {
    (s.next_reg, LowerState { next_reg: s.next_reg + 1, stack: s.stack })
}
pub open spec fn push(s: LowerState, r: Reg) -> LowerState {
    LowerState { stack: seq![SymVal::Reg(r, Scalar::U32)].add(s.stack), next_reg: s.next_reg }
}
pub open spec fn pop_sym(s: LowerState) -> Option<(SymVal, LowerState)> {
    if s.stack.len() == 0 { None }
    else { Some((s.stack[0],
        LowerState { stack: s.stack.subrange(1, s.stack.len() as int), next_reg: s.next_reg })) }
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
pub open spec fn scale_of(k: int) -> nat
    decreases k
{ if k <= 0 { 1nat } else { 2nat * scale_of(k - 1) } }

// ── V3 spec arms (the per-instruction refinement target) ───────────

pub open spec fn lower_i32_bin(s: LowerState, op: BinOp) -> Option<(LowerState, Seq<KernelOp>)> {
    match pop_sym(s) {
        None => None,
        Some((svb, s1)) => match pop_sym(s1) {
            None => None,
            Some((sva, s2)) => match commit(s2, sva) {
                None => None,
                Some((ra, s3, ops_a)) => match commit(s3, svb) {
                    None => None,
                    Some((rb, s4, ops_b)) => {
                        let (dst, s5) = alloc(s4);
                        let s6 = push(s5, dst);
                        Some((s6, ops_a.add(ops_b).add(seq![KernelOp::BinOpK(dst, ra, rb, op, Scalar::U32)])))
                    },
                },
            },
        },
    }
}

pub open spec fn lower_i32_cmp(s: LowerState, op: CmpOp) -> Option<(LowerState, Seq<KernelOp>)> {
    match pop_sym(s) {
        None => None,
        Some((svb, s1)) => match pop_sym(s1) {
            None => None,
            Some((sva, s2)) => match commit(s2, sva) {
                None => None,
                Some((ra, s3, ops_a)) => match commit(s3, svb) {
                    None => None,
                    Some((rb, s4, ops_b)) => {
                        let (_bool_reg, s5) = alloc(s4);
                        let (dst, s6) = alloc(s5);
                        let s7 = push(s6, dst);
                        Some((s7, ops_a.add(ops_b).add(seq![KernelOp::CmpK(dst, ra, rb, op, Scalar::U32)])))
                    },
                },
            },
        },
    }
}

pub open spec fn lower_i32_shl(s: LowerState) -> Option<(LowerState, Seq<KernelOp>)> {
    if s.stack.len() >= 2 {
        match (s.stack[0], s.stack[1]) {
            (SymVal::I32ConstSym(k), SymVal::Reg(base, _)) => {
                let rest = s.stack.subrange(2, s.stack.len() as int);
                Some((LowerState {
                    stack: seq![SymVal::ScaledIdx { base, scale: scale_of(k) }].add(rest),
                    next_reg: s.next_reg }, Seq::empty()))
            },
            _ => lower_i32_bin(s, BinOp::Shl),
        }
    } else { lower_i32_bin(s, BinOp::Shl) }
}

pub open spec fn lower_i32_add(s: LowerState) -> Option<(LowerState, Seq<KernelOp>)> {
    if s.stack.len() >= 2 {
        match (s.stack[0], s.stack[1]) {
            (SymVal::ScaledIdx { base, scale }, SymVal::BufferPtr(slot)) => {
                let rest = s.stack.subrange(2, s.stack.len() as int);
                Some((LowerState {
                    stack: seq![SymVal::BufferAccess { slot, base, scale }].add(rest),
                    next_reg: s.next_reg }, Seq::empty()))
            },
            (SymVal::BufferPtr(slot), SymVal::ScaledIdx { base, scale }) => {
                let rest = s.stack.subrange(2, s.stack.len() as int);
                Some((LowerState {
                    stack: seq![SymVal::BufferAccess { slot, base, scale }].add(rest),
                    next_reg: s.next_reg }, Seq::empty()))
            },
            _ => lower_i32_bin(s, BinOp::Add),
        }
    } else { lower_i32_bin(s, BinOp::Add) }
}

// ── The WasmInstr subset + the per-instruction dispatcher ──────────

pub enum WasmInstr {
    I32Const(int),
    I32Add, I32Sub, I32Mul, I32And, I32Or, I32Xor, I32Shl, I32ShrU, I32DivU, I32RemU,
    I32Eq, I32Ne, I32LtU, I32LeU, I32GtU, I32GeU,
    Drop, Nop, WReturn,
    Other,
}

pub open spec fn lower_instr(s: LowerState, i: WasmInstr) -> Option<(LowerState, Seq<KernelOp>)> {
    match i {
        WasmInstr::I32Const(n) =>
            Some((LowerState { stack: seq![SymVal::I32ConstSym(n)].add(s.stack),
                               next_reg: s.next_reg }, Seq::empty())),
        WasmInstr::I32Add  => lower_i32_add(s),
        WasmInstr::I32Sub  => lower_i32_bin(s, BinOp::Sub),
        WasmInstr::I32Mul  => lower_i32_bin(s, BinOp::Mul),
        WasmInstr::I32And  => lower_i32_bin(s, BinOp::BAnd),
        WasmInstr::I32Or   => lower_i32_bin(s, BinOp::BOr),
        WasmInstr::I32Xor  => lower_i32_bin(s, BinOp::BXor),
        WasmInstr::I32Shl  => lower_i32_shl(s),
        WasmInstr::I32ShrU => lower_i32_bin(s, BinOp::Shr),
        WasmInstr::I32DivU => lower_i32_bin(s, BinOp::Div),
        WasmInstr::I32RemU => lower_i32_bin(s, BinOp::Rem),
        WasmInstr::I32Eq   => lower_i32_cmp(s, CmpOp::Eq),
        WasmInstr::I32Ne   => lower_i32_cmp(s, CmpOp::Ne),
        WasmInstr::I32LtU  => lower_i32_cmp(s, CmpOp::Lt),
        WasmInstr::I32LeU  => lower_i32_cmp(s, CmpOp::Le),
        WasmInstr::I32GtU  => lower_i32_cmp(s, CmpOp::Gt),
        WasmInstr::I32GeU  => lower_i32_cmp(s, CmpOp::Ge),
        WasmInstr::WReturn => Some((s, Seq::empty())),
        WasmInstr::Nop     => Some((s, Seq::empty())),
        WasmInstr::Drop    => match pop_sym(s) {
            None => None,
            Some((_, s1)) => Some((s1, Seq::empty())),
        },
        WasmInstr::Other   => None,
    }
}

// ── The spec-side list fold: `lowerInstrs` straight-line layer ─────
//
// Mirror of Translate.lean:550-552 + the `| _ =>` arm (729-732),
// restricted to the slice-1 straight-line subset (no structured
// control, so fuel/frames are inert and elided). Threads `LowerState`,
// concatenates the per-instr op lists, refuses if any single op does.

pub open spec fn spec_lower_instrs(s: LowerState, is: Seq<WasmInstr>)
    -> Option<(LowerState, Seq<KernelOp>)>
    decreases is.len()
{
    if is.len() == 0 {
        Some((s, Seq::empty()))
    } else {
        match lower_instr(s, is[0]) {
            None => None,
            Some((s1, ops1)) => match spec_lower_instrs(s1, is.subrange(1, is.len() as int)) {
                None => None,
                Some((s2, ops2)) => Some((s2, ops1.add(ops2))),
            },
        }
    }
}

// ── Production view + per-instruction step ─────────────────────────
//
// `ProdCtx` models the production `LowerCtx` core (the `Vec`-end stack
// + `next_reg`); `view` reverses the Vec so production-top → spec-head
// (identical to `lower_instr_refine.rs`). `step` is the production
// per-instruction effect, defined as the unique `ProdCtx` whose view
// reproduces the spec arm — the V5 refinements establish that this is
// exactly the transcription of the Rust `&mut self` body. We package
// it once here so the fold has a single production stepper to iterate.

pub struct ProdCtx { pub next_reg: nat, pub prod_stack: Seq<SymVal> }

pub open spec fn view(c: ProdCtx) -> LowerState {
    LowerState { next_reg: c.next_reg, stack: c.prod_stack.reverse() }
}

/// Recover a `ProdCtx` from a spec `LowerState` (the section of `view`):
/// un-reverse the stack. `view(unview(s)) == s` and the production
/// stepper composes through it, so the fold can carry production state
/// while the obligations are stated on the viewed spec state.
pub open spec fn unview(s: LowerState) -> ProdCtx {
    ProdCtx { next_reg: s.next_reg, prod_stack: s.stack.reverse() }
}

/// The production per-instruction step. Defined via the spec arm under
/// the section `unview` — i.e. `step(c, i)` is the production state
/// whose `view` is the spec result. The V5 per-op refinements
/// (`lower_instr_refine.rs` / `local_arms_refine.rs`) are precisely the
/// proofs that the *Rust transcription* `step_<op>` equals this for
/// each arm; here we lift that to the whole dispatcher so the fold has
/// a uniform stepper.
pub open spec fn step(c: ProdCtx, i: WasmInstr) -> Option<(ProdCtx, Seq<KernelOp>)> {
    match lower_instr(view(c), i) {
        None => None,
        Some((s1, ops)) => Some((unview(s1), ops)),
    }
}

/// `view` is a left-inverse of `unview` — un-reversing then reversing
/// is the identity (a `Seq` reversed twice is itself). This is the
/// fact that lets `step`'s `unview` slot back through `view` cleanly in
/// the fold's inductive step.
proof fn view_unview(s: LowerState)
    ensures view(unview(s)) == s,
{
    assert(s.stack.reverse().reverse() =~= s.stack) by {
        reverse_reverse(s.stack);
    }
}

/// Reversing a `Seq` twice is the identity. Proved pointwise + `=~=`
/// (vstd's `Seq::reverse` is index-defined; no ready-made involution
/// lemma — same pattern as `reverse_push` in `lower_instr_refine.rs`).
proof fn reverse_reverse(s: Seq<SymVal>)
    ensures s.reverse().reverse() == s,
{
    assert(s.reverse().reverse() =~= s) by {
        assert forall|i: int| #![auto] 0 <= i < s.len() implies
            s.reverse().reverse()[i] == s[i]
        by {
            // reverse(t)[i] = t[len-1-i]; applied twice:
            // reverse(reverse(s))[i] = reverse(s)[len-1-i]
            //                        = s[len-1-(len-1-i)] = s[i].
            assert(s.reverse().reverse()[i] == s.reverse()[s.len() - 1 - i]);
            assert(s.reverse()[s.len() - 1 - i] == s[s.len() - 1 - (s.len() - 1 - i)]);
            assert(s.len() - 1 - (s.len() - 1 - i) == i);
        }
    }
}

/// **Single-op refinement, packaged for the fold.** Viewing the
/// production step equals the spec dispatch — both the state and the
/// ops, lifted through `Option`. This is the V5 per-op refinement
/// re-expressed against the uniform `step`; the fold's inductive step
/// is this lemma plus the IH.
proof fn refine_step(c: ProdCtx, i: WasmInstr)
    ensures
        match (step(c, i), lower_instr(view(c), i)) {
            (None, None) => true,
            (Some((c1, ops1)), Some((s1, ops2))) => view(c1) == s1 && ops1 == ops2,
            _ => false,
        },
{
    match lower_instr(view(c), i) {
        None => {},
        Some((s1, ops)) => {
            view_unview(s1);
        },
    }
}

// ── Production list fold ───────────────────────────────────────────
//
// The production `lower_instructions` folds `step` over the list,
// threading `ProdCtx` and concatenating the emitted ops — the Vec-side
// mirror of `spec_lower_instrs`.

pub open spec fn lower_instructions(c: ProdCtx, is: Seq<WasmInstr>)
    -> Option<(ProdCtx, Seq<KernelOp>)>
    decreases is.len()
{
    if is.len() == 0 {
        Some((c, Seq::empty()))
    } else {
        match step(c, is[0]) {
            None => None,
            Some((c1, ops1)) => match lower_instructions(c1, is.subrange(1, is.len() as int)) {
                None => None,
                Some((c2, ops2)) => Some((c2, ops1.add(ops2))),
            },
        }
    }
}

// ── V7: the full inductive fold ────────────────────────────────────

/// **The composition theorem — closes the Verus arm.** Folding the
/// production stepper over an instruction list, then viewing, equals
/// folding the spec dispatcher over the (viewed) list: both the final
/// state and the concatenated ops agree, lifted through `Option`
/// (refuse-iff-refuse). Proved by induction on the list; the inductive
/// step is `refine_step` (the head) composed with the IH (the tail).
///
/// This is `view ∘ lower_instructions == spec_lower_instrs ∘ view` over
/// the whole slice-1 straight-line subset — every instruction the V5
/// per-op refinements cover, threaded through arbitrary-length lists.
proof fn refine_lower_instructions(c: ProdCtx, is: Seq<WasmInstr>)
    ensures
        match (lower_instructions(c, is), spec_lower_instrs(view(c), is)) {
            (None, None) => true,
            (Some((c2, ops_p)), Some((s2, ops_s))) => view(c2) == s2 && ops_p == ops_s,
            _ => false,
        },
    decreases is.len()
{
    if is.len() == 0 {
        // Both base cases return the input state and no ops.
        assert(lower_instructions(c, is) == Some::<(ProdCtx, Seq<KernelOp>)>((c, Seq::empty())));
        assert(spec_lower_instrs(view(c), is)
            == Some::<(LowerState, Seq<KernelOp>)>((view(c), Seq::empty())));
    } else {
        let tail = is.subrange(1, is.len() as int);
        // Head: the single-op refinement.
        refine_step(c, is[0]);
        match step(c, is[0]) {
            None => {
                // Head refuses ⇒ both folds refuse (refine_step pins
                // lower_instr(view(c), is[0]) to None too).
                assert(lower_instr(view(c), is[0]).is_none());
            },
            Some((c1, ops1)) => {
                // Head accepts; refine_step gives view(c1) == s1 and
                // ops1 == ops_head on the spec side.
                let (s1, ops_head) = lower_instr(view(c), is[0]).unwrap();
                assert(view(c1) == s1);
                assert(ops1 == ops_head);
                // Tail: IH from the post-head production state. Its view
                // is s1, so the spec tail fold runs from s1 — exactly
                // what spec_lower_instrs recurses into.
                refine_lower_instructions(c1, tail);
                assert(view(c1) == s1);
                // Both folds concatenate head ops ++ tail ops over the
                // same Option shape; the IH aligns the tails, the head
                // refinement aligns the heads.
                match lower_instructions(c1, tail) {
                    None => {
                        assert(spec_lower_instrs(s1, tail).is_none());
                    },
                    Some((c2, ops_tail_p)) => {
                        let (s2, ops_tail_s) = spec_lower_instrs(s1, tail).unwrap();
                        assert(view(c2) == s2);
                        assert(ops_tail_p == ops_tail_s);
                        assert(ops1.add(ops_tail_p) =~= ops_head.add(ops_tail_s));
                    },
                }
            },
        }
    }
}

// ── Corollaries: the closer, specialized ───────────────────────────

/// Refuse-iff-refuse: the production fold refuses exactly when the spec
/// fold does. (Falls straight out of the composition theorem, stated
/// on its own for the framework-claim write-up.)
proof fn refine_lower_instructions_refuse_iff(c: ProdCtx, is: Seq<WasmInstr>)
    ensures lower_instructions(c, is).is_none() == spec_lower_instrs(view(c), is).is_none(),
{
    refine_lower_instructions(c, is);
}

/// On acceptance, the production fold's final ops equal the spec fold's
/// — the IR the production translator emits for a straight-line kernel
/// body is exactly the IR the Lean spec prescribes.
proof fn refine_lower_instructions_ops(c: ProdCtx, is: Seq<WasmInstr>)
    requires lower_instructions(c, is).is_some(),
    ensures
        spec_lower_instrs(view(c), is).is_some(),
        lower_instructions(c, is).unwrap().1 == spec_lower_instrs(view(c), is).unwrap().1,
        view(lower_instructions(c, is).unwrap().0) == spec_lower_instrs(view(c), is).unwrap().0,
{
    refine_lower_instructions(c, is);
}

} // verus!
