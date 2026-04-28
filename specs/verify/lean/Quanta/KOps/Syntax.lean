/-
# KernelOps syntax — minimal Lean view

Step **E.2a** of the source-preservation track. Lean inductive view
of `quanta_ir::KernelOp` (`crates/quanta-ir/src/types.rs`). The
shape mirrors the Rust enum 1:1; we only model variants the proc
macro currently emits (the texture / cooperative-MMA / mesh-shader
variants are wired in their respective drivers but not yet on the
source-preservation chain — they enter as Lean theorems extend in
parallel with their feature work).

The big-step semantics in `Quanta.KOps.Semantics` (E.2b) consumes
this view; the translator in `Quanta.KRust.Translate` (E.3) targets
it. Together those two close the diagram

```
   KRust.Stmt  ⇓_KRust  Value
        │                  │
   translate              ≡   ← E.4 per-rule lemmas
        │                  │
   KOps.Stmt   ⇓_KOps   Value
```

Naming convention: each constructor matches `KernelOp::Variant`
verbatim so the translator E.3 reads as `KRust.Expr.Binary _ _ →
[KOp.binOp _ _ _ _]` and the proof obligations stay legible.
-/

namespace Quanta.KOps

-- ════════════════════════════════════════════════════════════════════
-- Scalar types — same alphabet as KRust (kept as its own view here
-- so the KOps module can be imported independently of KRust).
-- ════════════════════════════════════════════════════════════════════

inductive Scalar where
  | bool
  | i8 | i16 | i32 | i64
  | u8 | u16 | u32 | u64
  | f16 | f32 | f64
  deriving Repr, DecidableEq

-- ════════════════════════════════════════════════════════════════════
-- Register and constant types
-- ════════════════════════════════════════════════════════════════════

/-- SSA-style virtual register. Mirrors `quanta_ir::Reg(pub u32)`. -/
abbrev Reg : Type := Nat

/-- Constant value embedded in the IR. Matches `ConstValue`. -/
inductive ConstValue where
  | bool (b : Bool)
  | i32  (n : Int)
  | u32  (n : UInt32)
  | f32  (bits : UInt32)        -- IEEE-754 bit pattern
  deriving Repr, DecidableEq

-- ════════════════════════════════════════════════════════════════════
-- Operators — same set as KRust.BinOp / UnaryOp / CmpOp
-- ════════════════════════════════════════════════════════════════════

inductive BinOp where
  | add | sub | mul | div | rem
  | bAnd | bOr | bXor | shl | shr
  | satAdd | satSub
  deriving Repr, DecidableEq

inductive UnaryOp where
  | neg | bNot | logNot
  deriving Repr, DecidableEq

inductive CmpOp where
  | eq | ne | lt | le | gt | ge
  deriving Repr, DecidableEq

-- ════════════════════════════════════════════════════════════════════
-- KernelOp — the Lean view of the IR opcode set
-- ════════════════════════════════════════════════════════════════════
--
-- One constructor per `quanta_ir::KernelOp` variant the proc macro
-- emits today. Branch / Loop are recursive — the body lists nest.

mutual
inductive KernelOp where
  /-- `Const { dst, value }` — load `value` into register `dst`. -/
  | const     (dst : Reg) (value : ConstValue)
  /-- `BinOp { dst, a, b, op, ty }`. -/
  | binOp     (dst : Reg) (a b : Reg) (op : BinOp) (ty : Scalar)
  /-- `UnaryOp { dst, a, op, ty }`. -/
  | unaryOp   (dst : Reg) (a : Reg) (op : UnaryOp) (ty : Scalar)
  /-- `Cmp { dst, a, b, op, ty }` — comparison ops produce bool. -/
  | cmp       (dst : Reg) (a b : Reg) (op : CmpOp) (ty : Scalar)
  /-- `Cast { dst, src, from, to }`. -/
  | cast      (dst : Reg) (src : Reg) (fromTy : Scalar) (toTy : Scalar)
  /-- `Bitcast { dst, src, from, to }`. -/
  | bitcast   (dst : Reg) (src : Reg) (fromTy : Scalar) (toTy : Scalar)
  /-- `Copy { dst, src }` — register copy. -/
  | copy      (dst : Reg) (src : Reg)
  /-- `Load { dst, field, index, ty }` — read `field[index]` into `dst`. -/
  | load      (dst : Reg) (field : Nat) (index : Reg) (ty : Scalar)
  /-- `Store { field, index, src, ty }`. -/
  | store     (field : Nat) (index : Reg) (src : Reg) (ty : Scalar)
  /-- `Branch { cond, then_ops, else_ops }`. -/
  | branch    (cond : Reg) (thenOps elseOps : List KernelOp)
  /-- `Loop { body }`. -/
  | loopOp    (body : List KernelOp)
  /-- `Break`. -/
  | breakOp
  /-- `QuarkId { dst }` — populated from dispatch context. -/
  | quarkId   (dst : Reg)
  /-- `ProtonId { dst }`. -/
  | protonId  (dst : Reg)
  /-- `NucleusId { dst }`. -/
  | nucleusId (dst : Reg)
  /-- `ProtonSize { dst }` — workgroup-size scalar. -/
  | protonSize (dst : Reg)
  /-- `QuarkCount { dst }` — total quark count. -/
  | quarkCount (dst : Reg)
  /-- `Barrier`. -/
  | barrier
  deriving Repr
end

end Quanta.KOps
