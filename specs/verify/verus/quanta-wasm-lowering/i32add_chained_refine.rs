//! i32Add chained-address arithmetic — production beyond the Lean
//! slice-1 `BufferAccess` fast-path.
//!
//! The Lean spec's `lowerI32Add` recognizes exactly one buffer pattern:
//! `BufferPtr(slot) + ScaledIdx{base,scale}` (either order) folds to a
//! `BufferAccess{slot,base,scale}` with **no IR**. The production
//! `RawInstr::I32Add` arm (`lower.rs:1083-1215`) folds three *additional*
//! chained-address shapes that rustc emits when it has precomputed part
//! of a byte offset and leaves the rest runtime — each **emitting IR**:
//!
//!   1. **same-scale add** — `BufferAccess{slot,base,scale} +
//!      ScaledIdx{b2,s2}` with `scale == s2`: combine the indices.
//!      Emit `BinOp::Add(dst, base, b2)`; push `BufferAccess{slot, dst,
//!      scale}`. (lower.rs:1129-1139)
//!   2. **rescale add** — same shape but `scale > s2`, `scale % s2 ==
//!      0`, `scale/s2` a power of two: rescale the BufferAccess base to
//!      the smaller scale. Emit `Const(shift, log2(scale/s2))`,
//!      `Shl(scaled, base, shift)`, `Add(dst, scaled, b2)`; push
//!      `BufferAccess{slot, dst, s2}`. (lower.rs:1140-1167)
//!   3. **const-offset add** — `BufferAccess{slot,base,scale} +
//!      I32Const(c)` with `c % scale == 0`: fold the constant element
//!      offset. Emit `Const(off, c/scale)`, `Add(dst, base, off)`; push
//!      `BufferAccess{slot, dst, scale}`. (lower.rs:1177-1199)
//!
//! These three folds are now **CLOSED in the Lean spec (V8-#3)**:
//! `Translate.lean`'s `lowerI32Add` has matching same-scale, rescale,
//! and const-offset arms (either operand order), and
//! `Preservation.lean` proves each one's encoding correctness end to
//! end (`preservation_i32Add_chained_{sameScale,constOff,rescale}`):
//! the emitted `Add` / `Shl`+`Add` / `Const`+`Add` make the merged
//! `bufferAccess` encode the WASM `addr + idx` by the distributivity /
//! shift-as-multiply / divisibility facts, with no-overflow discharged
//! by per-arm preconditions. The structural lowering invariants
//! (scope-validity, well-scopedness, nextReg-monotonicity,
//! bufferSlots) are also proven for the new arms.
//!
//! This file remains as the **production transcription** — it pins each
//! arm's exact register/op shape (`step_i32_add_chained_*` effect equals
//! the stated state + ops), the same status the straight-line
//! `step_<op>` models carry. It is no longer a recorded *gap*: the Lean
//! arm it would refine against now exists and is proven.

use vstd::prelude::*;

verus! {

// ── Vocabulary (V2/V3, inlined) ────────────────────────────────────

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
pub enum BinOp { Add, Sub, Mul, Div, Rem, BAnd, BOr, BXor, Shl, Shr }
pub enum KernelOp {
    Const(Reg, ConstValue),
    BinOpK(Reg, Reg, Reg, BinOp, Scalar),
}

/// Production context: the `Vec`-end stack + `next_reg` (same as the
/// V5/V7 `ProdCtx`). The chained arms pop the top two, emit, push one.
pub struct ProdCtx { pub next_reg: nat, pub prod_stack: Seq<SymVal> }

// ── log2 of a power-of-two scale (the `trailing_zeros` production uses) ──

pub open spec fn log2(n: nat) -> nat
    decreases n
{
    if n <= 1 { 0nat } else { 1nat + log2(n / 2) }
}

// ── Case 1: same-scale add ─────────────────────────────────────────
//
// Production (lower.rs:1129-1139): pop b, pop a; on
// (BufferAccess{slot,base,scale}, ScaledIdx{b2,scale}) in either order
// with equal scale → alloc dst, emit Add(dst, base, b2), push
// BufferAccess{slot, dst, scale}. Transcribed onto ProdCtx (top = last).

pub open spec fn step_i32_add_same_scale(
    c: ProdCtx, slot: nat, base: Reg, scale: nat, b2: Reg) -> (ProdCtx, Seq<KernelOp>)
{
    let dst = c.next_reg;
    let rest = c.prod_stack.subrange(0, c.prod_stack.len() - 2);
    (ProdCtx {
        next_reg: c.next_reg + 1,
        prod_stack: rest.push(SymVal::BufferAccess { slot, base: dst, scale }),
     },
     seq![KernelOp::BinOpK(dst, base, b2, BinOp::Add, Scalar::U32)])
}

/// **Shape lemma — same-scale add.** Allocates one reg, emits a single
/// `Add` combining the two indices, and pushes the merged
/// `BufferAccess` (same slot, new base = the added reg, same scale).
/// `next_reg` bumps by exactly 1.
proof fn add_same_scale_shape(c: ProdCtx, slot: nat, base: Reg, scale: nat, b2: Reg)
    requires c.prod_stack.len() >= 2,
    ensures ({
        let (c1, ops) = step_i32_add_same_scale(c, slot, base, scale, b2);
        &&& c1.next_reg == c.next_reg + 1
        &&& ops == seq![KernelOp::BinOpK(c.next_reg, base, b2, BinOp::Add, Scalar::U32)]
        &&& c1.prod_stack.len() == c.prod_stack.len() - 1
        &&& c1.prod_stack.last() == SymVal::BufferAccess { slot, base: c.next_reg, scale }
    }),
{
    let (c1, _ops) = step_i32_add_same_scale(c, slot, base, scale, b2);
    let rest = c.prod_stack.subrange(0, c.prod_stack.len() - 2);
    assert(c1.prod_stack == rest.push(SymVal::BufferAccess { slot, base: c.next_reg, scale }));
    assert(c1.prod_stack.len() == rest.len() + 1);
    assert(c1.prod_stack.last() == SymVal::BufferAccess { slot, base: c.next_reg, scale });
}

// ── Case 2: rescale add (larger BufferAccess scale) ────────────────
//
// Production (lower.rs:1140-1167): scale > s2, scale % s2 == 0,
// scale/s2 a power of two. shift_amt = log2(scale/s2); emit
// Const(shift, shift_amt), Shl(scaled, base, shift), Add(dst, scaled,
// b2); push BufferAccess{slot, dst, s2}. Three allocs, three ops.

pub open spec fn step_i32_add_rescale(
    c: ProdCtx, slot: nat, base: Reg, scale: nat, b2: Reg, s2: nat) -> (ProdCtx, Seq<KernelOp>)
{
    let shift_reg = c.next_reg;
    let scaled = c.next_reg + 1;
    let dst = c.next_reg + 2;
    let shift_amt = log2(scale / s2);
    let rest = c.prod_stack.subrange(0, c.prod_stack.len() - 2);
    (ProdCtx {
        next_reg: c.next_reg + 3,
        prod_stack: rest.push(SymVal::BufferAccess { slot, base: dst, scale: s2 }),
     },
     seq![
        KernelOp::Const(shift_reg, ConstValue::U32(shift_amt as int)),
        KernelOp::BinOpK(scaled, base, shift_reg, BinOp::Shl, Scalar::U32),
        KernelOp::BinOpK(dst, scaled, b2, BinOp::Add, Scalar::U32),
     ])
}

/// **Shape lemma — rescale add.** Allocates three regs (shift, scaled,
/// dst), emits Const(shift=log2(scale/s2)) ; Shl(scaled, base, shift) ;
/// Add(dst, scaled, b2), and pushes `BufferAccess{slot, dst, s2}` at
/// the *smaller* scale. `next_reg` bumps by 3.
proof fn add_rescale_shape(c: ProdCtx, slot: nat, base: Reg, scale: nat, b2: Reg, s2: nat)
    requires c.prod_stack.len() >= 2, s2 > 0,
    ensures ({
        let (c1, ops) = step_i32_add_rescale(c, slot, base, scale, b2, s2);
        &&& c1.next_reg == c.next_reg + 3
        &&& ops.len() == 3
        &&& ops[0] == KernelOp::Const(c.next_reg, ConstValue::U32(log2(scale / s2) as int))
        &&& ops[1] == KernelOp::BinOpK(c.next_reg + 1, base, c.next_reg, BinOp::Shl, Scalar::U32)
        &&& ops[2] == KernelOp::BinOpK(c.next_reg + 2, c.next_reg + 1, b2, BinOp::Add, Scalar::U32)
        &&& c1.prod_stack.last() == SymVal::BufferAccess { slot, base: c.next_reg + 2, scale: s2 }
    }),
{
    let (c1, _ops) = step_i32_add_rescale(c, slot, base, scale, b2, s2);
    let rest = c.prod_stack.subrange(0, c.prod_stack.len() - 2);
    assert(c1.prod_stack == rest.push(SymVal::BufferAccess { slot, base: c.next_reg + 2, scale: s2 }));
}

// ── Case 3: const-offset add ───────────────────────────────────────
//
// Production (lower.rs:1177-1199): (BufferAccess{slot,base,scale},
// I32Const(c_val)) in either order with c_val % scale == 0. off =
// c_val/scale; emit Const(off_reg, off) [tagged I32], Add(dst, base,
// off_reg); push BufferAccess{slot, dst, scale}. Two allocs, two ops.
// Note the I32 const tag (the V6-recorded const-tag divergence again).

pub open spec fn step_i32_add_const_off(
    c: ProdCtx, slot: nat, base: Reg, scale: nat, c_val: int) -> (ProdCtx, Seq<KernelOp>)
{
    let off_reg = c.next_reg;
    let dst = c.next_reg + 1;
    let off = c_val / (scale as int);
    let rest = c.prod_stack.subrange(0, c.prod_stack.len() - 2);
    (ProdCtx {
        next_reg: c.next_reg + 2,
        prod_stack: rest.push(SymVal::BufferAccess { slot, base: dst, scale }),
     },
     seq![
        KernelOp::Const(off_reg, ConstValue::I32(off)),
        KernelOp::BinOpK(dst, base, off_reg, BinOp::Add, Scalar::U32),
     ])
}

/// **Shape lemma — const-offset add.** Allocates two regs, emits
/// `Const(off = c/scale)` (tagged **I32** — the same const-tag
/// divergence V6 recorded for `commit`) then `Add(dst, base, off)`, and
/// pushes `BufferAccess{slot, dst, scale}`. `next_reg` bumps by 2.
proof fn add_const_off_shape(c: ProdCtx, slot: nat, base: Reg, scale: nat, c_val: int)
    requires c.prod_stack.len() >= 2, scale > 0,
    ensures ({
        let (c1, ops) = step_i32_add_const_off(c, slot, base, scale, c_val);
        &&& c1.next_reg == c.next_reg + 2
        &&& ops == seq![
                KernelOp::Const(c.next_reg, ConstValue::I32(c_val / (scale as int))),
                KernelOp::BinOpK(c.next_reg + 1, base, c.next_reg, BinOp::Add, Scalar::U32)]
        &&& c1.prod_stack.last() == SymVal::BufferAccess { slot, base: c.next_reg + 1, scale }
    }),
{
    let (c1, _ops) = step_i32_add_const_off(c, slot, base, scale, c_val);
    let rest = c.prod_stack.subrange(0, c.prod_stack.len() - 2);
    assert(c1.prod_stack == rest.push(SymVal::BufferAccess { slot, base: c.next_reg + 1, scale }));
}

// ── All three preserve the BufferAccess slot and shrink the stack ──

/// Every chained-address arm keeps the buffer `slot` and reduces the
/// stack by one (pop two address pieces, push one merged
/// `BufferAccess`). This is the invariant that lets a chain of these
/// compose (`out + block_off + pos_off` = two nested adds) — each step
/// leaves a single `BufferAccess` on top for the next `i32.add` (or the
/// terminating `load`/`store`) to consume.
proof fn chained_preserves_slot_and_shrinks(c: ProdCtx, slot: nat, base: Reg, scale: nat, b2: Reg, s2: nat, c_val: int)
    requires c.prod_stack.len() >= 2, s2 > 0, scale > 0,
    ensures
        ({ let (c1, _) = step_i32_add_same_scale(c, slot, base, scale, b2);
           c1.prod_stack.len() == c.prod_stack.len() - 1
           && (c1.prod_stack.last() is BufferAccess) }),
        ({ let (c1, _) = step_i32_add_rescale(c, slot, base, scale, b2, s2);
           c1.prod_stack.len() == c.prod_stack.len() - 1
           && (c1.prod_stack.last() is BufferAccess) }),
        ({ let (c1, _) = step_i32_add_const_off(c, slot, base, scale, c_val);
           c1.prod_stack.len() == c.prod_stack.len() - 1
           && (c1.prod_stack.last() is BufferAccess) }),
{
    let rest = c.prod_stack.subrange(0, c.prod_stack.len() - 2);
    assert(rest.len() == c.prod_stack.len() - 2);
}

} // verus!
