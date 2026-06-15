//! V5 (local/memory arms) — extended LowerState + the local-binding
//! lowering arms, mirroring Lean `Translate.lowerInstr`'s
//! `localGet` / `localSet` / `localTee`.
//!
//! The arithmetic refinements (lower_instr_refine.rs) work over the
//! LowerState *core* — `next_reg` + `stack`. The local arms need the
//! four binding maps the Lean spec carries:
//!
//!   localReg    : local idx → stable register (the merge anchor)
//!   localTy     : local idx → IR scalar type
//!   currentReg  : local idx → the per-frame post-write binding
//!   bufferSlots : local idx → buffer slot (for `#[quanta::shared]`)
//!
//! Lean stores these as assoc-lists with `find?` lookup and
//! filter-then-prepend setters; we model them as `Map<nat, _>` (same
//! lookup/insert semantics, cleaner in the spec world). The
//! definitional lemmas pin the exact output shape of each arm — the
//! dual-Copy pattern in particular — which is what the production
//! refinement (the `view` + `step` correspondence from
//! lower_instr_refine.rs) composes against.
//!
//! Scope here: the spec arms + their shape lemmas. `localGet` for a
//! buffer-typed local pushes `BufferPtr` (the memory-op producer);
//! the non-buffer `localGet`, `localSet`, and `localTee` carry the
//! stable-reg / dual-Copy machinery. `i32Load`/`i32Store` consume the
//! `BufferAccess` shape and are mirrored alongside.

use vstd::prelude::*;

verus! {

// ── Spec vocabulary (extended with the local maps) ─────────────────

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
    Load(Reg, nat, Reg, Scalar),      // dst, field/slot, index, ty
    Store(nat, Reg, Reg, Scalar),     // field/slot, index, src, ty
}

/// LowerState with the four local-binding maps (the Lean assoc-lists
/// modeled as `Map`s).
pub struct LowerState {
    pub next_reg: nat,
    pub stack: Seq<SymVal>,
    pub local_reg: Map<nat, Reg>,
    pub local_ty: Map<nat, Scalar>,
    pub current_reg: Map<nat, Reg>,
    pub buffer_slots: Map<nat, nat>,
}

// ── Map helpers mirroring the Lean lookup/set fns ──────────────────

pub open spec fn lookup_local(s: LowerState, i: nat) -> Option<Reg> {
    if s.local_reg.dom().contains(i) { Some(s.local_reg[i]) } else { None }
}
pub open spec fn lookup_local_ty(s: LowerState, i: nat) -> Option<Scalar> {
    if s.local_ty.dom().contains(i) { Some(s.local_ty[i]) } else { None }
}
pub open spec fn lookup_current_reg(s: LowerState, i: nat) -> Option<Reg> {
    if s.current_reg.dom().contains(i) { Some(s.current_reg[i]) } else { None }
}
pub open spec fn lookup_buffer_slot(s: LowerState, i: nat) -> Option<nat> {
    if s.buffer_slots.dom().contains(i) { Some(s.buffer_slots[i]) } else { None }
}

/// `setLocalReg`: bind local `i` to stable reg `r` of type `ty`
/// (Lean filter-then-prepend = Map insert on both maps).
pub open spec fn set_local_reg(s: LowerState, i: nat, r: Reg, ty: Scalar) -> LowerState {
    LowerState { local_reg: s.local_reg.insert(i, r),
                 local_ty: s.local_ty.insert(i, ty), ..s }
}
/// `setCurrentReg`: update the per-frame binding.
pub open spec fn set_current_reg(s: LowerState, i: nat, r: Reg) -> LowerState {
    LowerState { current_reg: s.current_reg.insert(i, r), ..s }
}

// ── State primitives ───────────────────────────────────────────────

pub open spec fn alloc(s: LowerState) -> (Reg, LowerState) {
    (s.next_reg, LowerState { next_reg: s.next_reg + 1, ..s })
}
pub open spec fn push_val(s: LowerState, v: SymVal) -> LowerState {
    LowerState { stack: seq![v].add(s.stack), ..s }
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

/// Zero literal for a scalar (mirror of Lean `LowerState.zeroConst`):
/// signed-int → I32(0), else U32(0). The frame-0 zero-init constant.
pub open spec fn zero_const(ty: Scalar) -> ConstValue {
    match ty {
        Scalar::I8 | Scalar::I16 | Scalar::I32 | Scalar::I64 => ConstValue::I32(0),
        _ => ConstValue::U32(0),
    }
}

// ── localGet (Lean lowerInstr `.localGet` arm) ─────────────────────
//
// Buffer-typed local → push `BufferPtr slot`, no reg, no IR.
// Otherwise: read from currentReg first, else localReg; alloc a
// fresh reg; emit `Copy { fresh, source }`; push fresh. The Copy
// breaks aliasing between the stack and the local's stable reg.

pub open spec fn local_get(s: LowerState, i: nat) -> Option<(LowerState, Seq<KernelOp>)> {
    match lookup_buffer_slot(s, i) {
        Some(slot) => Some((push_val(s, SymVal::BufferPtr(slot)), Seq::empty())),
        None => {
            // source = currentReg[i] orElse localReg[i]
            let src_opt = match lookup_current_reg(s, i) {
                Some(r) => Some(r),
                None => lookup_local(s, i),
            };
            match src_opt {
                None => None,  // uninitialized local — refuse
                Some(source) => {
                    let (fresh, s1) = alloc(s);
                    let s2 = push_val(s1, SymVal::Reg(fresh, Scalar::U32));
                    Some((s2, seq![KernelOp::Copy(fresh, source)]))
                },
            }
        },
    }
}

// ── localSet (post-V8 Lean lowerInstr `.localSet` arm) ─────────────
//
// popSym + commit the value, then the zero-init + dual-Copy:
//   1. fresh reg (new per-set binding)
//   2. Const { fresh, zeroConst ty }   (frame-0 zero-init, now inline)
//   3. Copy  { fresh, src }
//   4. Copy  { stable, fresh }    (keep the stable anchor in sync)
//   5. currentReg[i] := fresh
// stable reg: reused if the local already has one, else freshly
// allocated. ty defaults to U32. The Const is a dead write to `fresh`
// (overwritten by step 3 before any read) — V8 divergence #1, folded
// in inline. Production routes it to the frame-0 head (placement-only
// residual, pinned in local_arms_refine.rs).

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
                        Some((s4, ops_commit.add(seq![KernelOp::Const(fresh, zero_const(ty)),
                                                       KernelOp::Copy(fresh, src),
                                                       KernelOp::Copy(stable, fresh)])))
                    },
                    None => {
                        let (stable, s4) = alloc(s3);
                        let s5 = set_current_reg(set_local_reg(s4, i, stable, ty), i, fresh);
                        Some((s5, ops_commit.add(seq![KernelOp::Const(fresh, zero_const(ty)),
                                                       KernelOp::Copy(fresh, src),
                                                       KernelOp::Copy(stable, fresh)])))
                    },
                }
            },
        },
    }
}

// ── i32.load / i32.store (consume the BufferAccess shape) ──────────
//
// load: BufferAccess{slot,base,scale=4} on top → alloc dst, emit
//   Load { dst, slot, base, u32 }, push Reg dst. Any other shape → None.
// store: pop value (commit), pop BufferAccess → emit Store. (Modeled
//   at the shape level: the address must be a BufferAccess.)

pub open spec fn i32_load(s: LowerState) -> Option<(LowerState, Seq<KernelOp>)> {
    if s.stack.len() >= 1 {
        match s.stack[0] {
            SymVal::BufferAccess { slot, base, scale } => {
                if scale == 4 {
                    let rest = s.stack.subrange(1, s.stack.len() as int);
                    let dst = s.next_reg;
                    Some((LowerState { next_reg: s.next_reg + 1,
                            stack: seq![SymVal::Reg(dst, Scalar::U32)].add(rest), ..s },
                          seq![KernelOp::Load(dst, slot, base, Scalar::U32)]))
                } else { None }
            },
            _ => None,
        }
    } else { None }
}

// ── Definitional lemmas ────────────────────────────────────────────

/// localGet of a buffer local pushes BufferPtr and emits nothing.
proof fn local_get_buffer_pushes_ptr(s: LowerState, i: nat, slot: nat)
    requires lookup_buffer_slot(s, i) == Some(slot),
    ensures
        local_get(s, i) == Some::<(LowerState, Seq<KernelOp>)>(
            (push_val(s, SymVal::BufferPtr(slot)), Seq::empty())),
{}

/// localGet of a non-buffer local with a current-frame binding
/// allocs one reg and emits a single Copy from that binding.
proof fn local_get_current_binding(s: LowerState, i: nat, cur: Reg)
    requires
        lookup_buffer_slot(s, i) == None::<nat>,
        lookup_current_reg(s, i) == Some(cur),
    ensures ({
        let (fresh, s1) = alloc(s);
        local_get(s, i) == Some::<(LowerState, Seq<KernelOp>)>(
            (push_val(s1, SymVal::Reg(fresh, Scalar::U32)),
             seq![KernelOp::Copy(fresh, cur)]))
    }),
{}

/// localGet falls back to the stable reg when there's no current
/// binding.
proof fn local_get_stable_fallback(s: LowerState, i: nat, stable: Reg)
    requires
        lookup_buffer_slot(s, i) == None::<nat>,
        lookup_current_reg(s, i) == None::<Reg>,
        lookup_local(s, i) == Some(stable),
    ensures ({
        let (fresh, s1) = alloc(s);
        local_get(s, i) == Some::<(LowerState, Seq<KernelOp>)>(
            (push_val(s1, SymVal::Reg(fresh, Scalar::U32)),
             seq![KernelOp::Copy(fresh, stable)]))
    }),
{}

/// localGet of an uninitialized local refuses.
proof fn local_get_uninit_refuses(s: LowerState, i: nat)
    requires
        lookup_buffer_slot(s, i) == None::<nat>,
        lookup_current_reg(s, i) == None::<Reg>,
        lookup_local(s, i) == None::<Reg>,
    ensures local_get(s, i).is_none(),
{}

/// localSet of a register value into a local WITH a stable reg:
/// no commit ops, then the zero-init `Const(fresh, zeroConst ty)`
/// followed by the dual-Copy [fresh←src, stable←fresh]. fresh =
/// s1.next_reg; currentReg[i] = fresh, stable reg unchanged. The
/// local's recorded type is the `ty` used for the zero-init.
proof fn local_set_reg_existing_stable(s: LowerState, i: nat, r: Reg, ty: Scalar, stable: Reg)
    requires
        s.stack.len() >= 1,
        s.stack[0] == SymVal::Reg(r, ty),
        lookup_local_ty(s, i) is Some,
        // after the pop, the local still has its stable reg + recorded type
        ({ let s1 = pop_sym(s).unwrap().1; lookup_local(s1, i) == Some(stable) }),
        ({ let s1 = pop_sym(s).unwrap().1; lookup_local_ty(s1, i) == Some(ty) }),
    ensures ({
        let s1 = pop_sym(s).unwrap().1;
        local_set(s, i).unwrap().1
            == seq![KernelOp::Const(s1.next_reg, zero_const(ty)),
                    KernelOp::Copy(s1.next_reg, r),
                    KernelOp::Copy(stable, s1.next_reg)]
    }),
{
    let s1 = pop_sym(s).unwrap().1;
    // commit of a register is identity → ops_commit = [] → the result
    // ops are exactly [Const, dual-Copy]. fresh = s1.next_reg (no commit alloc).
    assert(commit(s1, SymVal::Reg(r, ty)) == Some::<(Reg, LowerState, Seq<KernelOp>)>(
        (r, s1, Seq::<KernelOp>::empty())));
    assert(Seq::<KernelOp>::empty().add(
        seq![KernelOp::Const(s1.next_reg, zero_const(ty)),
             KernelOp::Copy(s1.next_reg, r), KernelOp::Copy(stable, s1.next_reg)])
        =~= seq![KernelOp::Const(s1.next_reg, zero_const(ty)),
                 KernelOp::Copy(s1.next_reg, r), KernelOp::Copy(stable, s1.next_reg)]);
}

/// localSet of a register into a FRESH local (no prior stable reg)
/// allocates two regs: the per-set `fresh` then the local's `stable`.
/// next_reg therefore bumps by 2.
proof fn local_set_reg_fresh_local(s: LowerState, i: nat, r: Reg, ty: Scalar)
    requires
        s.stack.len() >= 1,
        s.stack[0] == SymVal::Reg(r, ty),
        ({ let s1 = pop_sym(s).unwrap().1; lookup_local(s1, i) == None::<Reg> }),
    ensures
        local_set(s, i).unwrap().0.next_reg == pop_sym(s).unwrap().1.next_reg + 2,
{
    let s1 = pop_sym(s).unwrap().1;
    assert(commit(s1, SymVal::Reg(r, ty)) == Some::<(Reg, LowerState, Seq<KernelOp>)>(
        (r, s1, Seq::<KernelOp>::empty())));
}

/// i32.load on a 4-byte BufferAccess allocs one reg and emits one
/// typed Load against the slot.
proof fn load_emits_typed_load(s: LowerState, slot: nat, base: Reg)
    requires
        s.stack.len() >= 1,
        s.stack[0] == (SymVal::BufferAccess { slot, base, scale: 4nat }),
    ensures ({
        let rest = s.stack.subrange(1, s.stack.len() as int);
        i32_load(s) == Some::<(LowerState, Seq<KernelOp>)>(
            (LowerState { next_reg: s.next_reg + 1,
                stack: seq![SymVal::Reg(s.next_reg, Scalar::U32)].add(rest), ..s },
             seq![KernelOp::Load(s.next_reg, slot, base, Scalar::U32)]))
    }),
{}

/// i32.load refuses a non-4-byte or non-BufferAccess address — the
/// only producers of the accepted shape are the buffer-pattern arms,
/// matching production's `UnsupportedOp` on any other address.
proof fn load_refuses_bad_address(s: LowerState, r: Reg, ty: Scalar)
    requires
        s.stack.len() >= 1,
        s.stack[0] == SymVal::Reg(r, ty),
    ensures i32_load(s).is_none(),
{}

} // verus!
