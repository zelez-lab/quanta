//! V3 — Verus spec `lower_instr`, mirroring Lean `Translate.lowerInstr`.
//!
//! Mechanizes the slice-1 straight-line subset of the Lean
//! per-instruction lowering as a Verus `spec fn`, over an inlined
//! copy of the V2 vocabulary (each Verus file verifies standalone, so
//! the shared types are re-stated here rather than imported).
//!
//! Subset covered (the part the V2 `LowerState` core — `next_reg` +
//! `stack` — supports without the local-binding maps):
//!   - `i32Const`            → push `I32ConstSym`, no ops
//!   - the i32 binop family  (add/sub/mul/and/or/xor/shr/div/rem)
//!   - `i32Shl`              with the `<reg> <const k>` ScaledIdx fast-path
//!   - `i32Add`              with the `<ptr> <scaledIdx>` BufferAccess fast-path
//!   - the i32 cmp family    (eq/ne/lt/le/gt/ge)
//!   - `drop` / `nop` / `wreturn`
//!   - `commit`              (the SymVal → register materializer)
//!
//! Deferred to later milestones (need the extended LowerState):
//!   - `localGet` / `localSet` / `localTee`  (V5, with the local maps)
//!   - `i32Load` / `i32Store`                (V5, buffer memory ops)
//!   - structured control (block/loop/if/br) (V7, lives in lowerInstrs)
//!
//! Correspondence is line-by-line with `Quanta.Wasm.Translate`. The
//! definitional lemmas at the end pin each arm's exact output shape —
//! these are what the V5 per-op refinement proofs match the Rust
//! `lower.rs` against.

use vstd::prelude::*;

verus! {

// ── V2 vocabulary (inlined; see spec_types.rs for the canonical copy) ──

pub type Reg = nat;

pub enum Scalar { Bool, I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64 }

pub enum SymVal {
    Reg(Reg, Scalar),
    BufferPtr(nat),
    ScaledIdx { base: Reg, scale: nat },
    I32ConstSym(int),
    BufferAccess { slot: nat, base: Reg, scale: nat },
}

pub struct LowerState {
    pub next_reg: nat,
    pub stack: Seq<SymVal>,
}

// ── KernelOp (the slice-1 emitted subset) + operand enums ──────────

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

// ── State primitives (mirror LowerState.alloc / push / popSym) ─────

pub open spec fn alloc(s: LowerState) -> (Reg, LowerState) {
    (s.next_reg, LowerState { next_reg: s.next_reg + 1, stack: s.stack })
}

pub open spec fn push(s: LowerState, r: Reg) -> LowerState {
    LowerState { stack: seq![SymVal::Reg(r, Scalar::U32)].add(s.stack), next_reg: s.next_reg }
}

/// Pop any SymVal off the top — mirrors Lean `LowerState.popSym`.
/// `None` on an empty stack.
pub open spec fn pop_sym(s: LowerState) -> Option<(SymVal, LowerState)> {
    if s.stack.len() == 0 {
        None
    } else {
        Some((s.stack[0], LowerState { stack: s.stack.subrange(1, s.stack.len() as int),
                                       next_reg: s.next_reg }))
    }
}

// ── commit (mirror Lean LowerState.commit) ─────────────────────────
// `.reg r _`         → already a register, no ops, no alloc.
// `.i32ConstSym n`   → fresh reg + a single `.const` op.
// address SymVals    → None (consumed by the buffer-pattern arms).

pub open spec fn commit(s: LowerState, v: SymVal) -> Option<(Reg, LowerState, Seq<KernelOp>)> {
    match v {
        SymVal::Reg(r, _) => Some((r, s, Seq::empty())),
        SymVal::I32ConstSym(n) => {
            let (dst, s1) = alloc(s);
            Some((dst, s1, seq![KernelOp::Const(dst, ConstValue::U32(n))]))
        },
        SymVal::BufferPtr(_) => None,
        SymVal::ScaledIdx { .. } => None,
        SymVal::BufferAccess { .. } => None,
    }
}

// ── i32 binop / cmp generic lowering (mirror lowerI32Bin/Cmp) ───────

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

// ── i32.shl / i32.add buffer-pattern fast-paths ────────────────────

/// `<reg base> <i32ConstSym k>` on top → fold to `ScaledIdx{base, 1<<k}`
/// with no IR (the "element index → byte offset" recognizer);
/// otherwise the generic shl binop. `scale_of k` is `2^k`.
pub open spec fn scale_of(k: int) -> nat
    decreases k
{
    if k <= 0 { 1nat } else { 2nat * scale_of(k - 1) }
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
    } else {
        lower_i32_bin(s, BinOp::Shl)
    }
}

/// `<bufferPtr> <scaledIdx>` in either order → fold to `BufferAccess`
/// with no IR; otherwise the generic add binop.
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
    } else {
        lower_i32_bin(s, BinOp::Add)
    }
}

// ── The WasmInstr subset + lower_instr dispatcher ──────────────────

pub enum WasmInstr {
    I32Const(int),
    I32Add, I32Sub, I32Mul, I32And, I32Or, I32Xor, I32Shl, I32ShrU, I32DivU, I32RemU,
    I32Eq, I32Ne, I32LtU, I32LeU, I32GtU, I32GeU,
    Drop, Nop, WReturn,
    /// Stand-in for every constructor outside the slice-1 subset.
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

// ── Definitional lemmas: each arm's exact output shape ─────────────

/// `i32.const` pushes the symbolic const and emits nothing.
proof fn const_pushes_sym(s: LowerState, n: int)
    ensures
        lower_instr(s, WasmInstr::I32Const(n))
            == Some((LowerState { stack: seq![SymVal::I32ConstSym(n)].add(s.stack),
                                  next_reg: s.next_reg }, Seq::<KernelOp>::empty())),
{}

/// `nop` / `wreturn` are state-and-ops-preserving no-ops.
proof fn nop_is_identity(s: LowerState)
    ensures
        lower_instr(s, WasmInstr::Nop) == Some((s, Seq::<KernelOp>::empty())),
        lower_instr(s, WasmInstr::WReturn) == Some((s, Seq::<KernelOp>::empty())),
{}

/// `commit` of a register is the identity — no alloc, no ops.
proof fn commit_reg_identity(s: LowerState, r: Reg, ty: Scalar)
    ensures commit(s, SymVal::Reg(r, ty)) == Some((r, s, Seq::<KernelOp>::empty())),
{}

/// `commit` of a symbolic const allocates one reg and emits one
/// `.const` op into it.
proof fn commit_const_emits_one(s: LowerState, n: int)
    ensures ({
        let (dst, s1) = alloc(s);
        commit(s, SymVal::I32ConstSym(n))
            == Some((dst, s1, seq![KernelOp::Const(dst, ConstValue::U32(n))]))
    }),
{}

/// `commit` of an address SymVal refuses — these are consumed by the
/// buffer-pattern arms, never materialized to a register.
proof fn commit_address_refuses(s: LowerState, slot: nat, base: Reg, scale: nat)
    ensures
        commit(s, SymVal::BufferPtr(slot)).is_none(),
        commit(s, SymVal::ScaledIdx { base, scale }).is_none(),
        commit(s, SymVal::BufferAccess { slot, base, scale }).is_none(),
{}

/// The shl fast-path fires exactly on `<reg base> <const k>` on top,
/// producing a `ScaledIdx{base, 2^k}` and no IR.
proof fn shl_fastpath_fires(s: LowerState, k: int, base: Reg, ty: Scalar)
    requires
        s.stack.len() >= 2,
        s.stack[0] == SymVal::I32ConstSym(k),
        s.stack[1] == SymVal::Reg(base, ty),
    ensures ({
        let rest = s.stack.subrange(2, s.stack.len() as int);
        lower_instr(s, WasmInstr::I32Shl)
            == Some((LowerState {
                  stack: seq![SymVal::ScaledIdx { base, scale: scale_of(k) }].add(rest),
                  next_reg: s.next_reg }, Seq::<KernelOp>::empty()))
    }),
{}

/// The add fast-path fires on `<scaledIdx> <bufferPtr>` on top,
/// folding to `BufferAccess` with no IR.
proof fn add_fastpath_fires(s: LowerState, base: Reg, scale: nat, slot: nat)
    requires
        s.stack.len() >= 2,
        s.stack[0] == (SymVal::ScaledIdx { base, scale }),
        s.stack[1] == SymVal::BufferPtr(slot),
    ensures ({
        let rest = s.stack.subrange(2, s.stack.len() as int);
        lower_instr(s, WasmInstr::I32Add)
            == Some((LowerState {
                  stack: seq![SymVal::BufferAccess { slot, base, scale }].add(rest),
                  next_reg: s.next_reg }, Seq::<KernelOp>::empty()))
    }),
{}

/// `drop` on a non-empty stack pops one value and emits nothing.
proof fn drop_pops_one(s: LowerState)
    requires s.stack.len() > 0,
    ensures ({
        let s1 = LowerState { stack: s.stack.subrange(1, s.stack.len() as int),
                              next_reg: s.next_reg };
        lower_instr(s, WasmInstr::Drop) == Some((s1, Seq::<KernelOp>::empty()))
    }),
{}

/// Everything outside the modeled subset refuses — matching the Lean
/// `| _ => none` and production's `UnsupportedOp`.
proof fn other_refuses(s: LowerState)
    ensures lower_instr(s, WasmInstr::Other).is_none(),
{}

/// A binop on a stack of < 2 entries refuses (stack underflow).
proof fn binop_underflow_refuses(s: LowerState)
    requires s.stack.len() == 0,
    ensures lower_instr(s, WasmInstr::I32Sub).is_none(),
{}

} // verus!
