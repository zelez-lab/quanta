/-
# KRust → KernelOps translator

Step **E.3** of the source-preservation track. Mirrors
`crates/quanta-macros/src/parse/{expr,stmt}.rs` arm-for-arm. Each
`KRust.Expr` / `KRust.Stmt` constructor maps to a fixed `List
KOps.KernelOp` shape; the function is structurally recursive and
threads an `EmitCtx` (next-register counter + variable→register
map + parameter→slot map) just like the proc macro's Rust-side
`EmitCtx`.

The per-rule preservation theorems in `Quanta.KRust.Preservation`
(E.4) will state `evalKRust ≡ evalKOps ∘ translate` over closed
expressions and statements; the shape of this translator is the
mirror of the proc macro, so any case the proc macro handles
appears as a constructor here, and any case the macro currently
rejects (atomic / wave / texture intrinsics not yet wired into
KRust's eval) returns `none`.

Conventions:
- `EmitCtx.nextReg` is bumped monotonically; consumers never
  reuse a register for two different values.
- Variable rebinding (`assignVar`) flushes the old `(name, reg)`
  entry and inserts a fresh one — same shape `Env.bind` uses on
  the source side.
- `translateExpr` returns `(resultReg, scalarTy, ctx')`; the
  caller threads `ctx'` into subsequent translations.
- `translateStmt` returns just `ctx'`.

A future B″-style commit will collapse this and the Rust proc
macro into a *single* parsed AST that emits both views — that's
E.6, the route-(a) dual emitter. Until then, this module is
hand-aligned with the Rust source; the per-rule preservation
theorems give us the discipline to catch drift.
-/

import Quanta.KRust.Syntax
import Quanta.KOps.Syntax

namespace Quanta.KRust

open Quanta.KOps (KernelOp Reg)

-- ════════════════════════════════════════════════════════════════════
-- Translation context
-- ════════════════════════════════════════════════════════════════════

/-- Translation state. Mirrors `crates/quanta-macros/src/parse.rs::EmitCtx`. -/
structure EmitCtx where
  /-- Next free register; register IDs are bumped monotonically. -/
  nextReg : Nat
  /-- Variable name → register currently holding its value. -/
  vars    : List (Ident × Reg)
  /-- Parameter name → slot index. Populated up front from the
      kernel's `Param` list before translation walks the body. -/
  params  : List (Ident × Nat)
  /-- Accumulator of emitted ops, in source order. -/
  ops     : List KernelOp
  deriving Repr

def EmitCtx.empty (params : List (Ident × Nat)) : EmitCtx :=
  { nextReg := 0, vars := [], params := params, ops := [] }

private def EmitCtx.fresh (c : EmitCtx) : Reg × EmitCtx :=
  (c.nextReg, { c with nextReg := c.nextReg + 1 })

private def EmitCtx.emit (c : EmitCtx) (op : KernelOp) : EmitCtx :=
  { c with ops := c.ops ++ [op] }

private def EmitCtx.lookupVar (c : EmitCtx) (name : Ident) : Option Reg :=
  c.vars.find? (fun p => p.fst = name) |>.map Prod.snd

private def EmitCtx.bindVar (c : EmitCtx) (name : Ident) (r : Reg) : EmitCtx :=
  { c with vars := (name, r) :: c.vars.filter (fun p => p.fst ≠ name) }

private def EmitCtx.lookupParam (c : EmitCtx) (name : Ident) : Option Nat :=
  c.params.find? (fun p => p.fst = name) |>.map Prod.snd

-- ════════════════════════════════════════════════════════════════════
-- Operator translation
-- ════════════════════════════════════════════════════════════════════

/-- KRust binary operator → KernelOps binary or compare op. The
    proc macro splits these into two functions (`syn_binop_to_ir`
    and `syn_binop_to_cmp`); the Lean dispatcher returns a sum so
    `translateExpr` can decide between `KernelOp.binOp` and
    `KernelOp.cmp`. -/
inductive BinDispatch where
  | bin (op : KOps.BinOp)
  | cmp (op : KOps.CmpOp)
  | logical (logAnd : Bool)   -- `&&` / `||` — implemented via branch-eval
  deriving Repr

def dispatchBinOp : BinOp → BinDispatch
  | .add    => .bin .add
  | .sub    => .bin .sub
  | .mul    => .bin .mul
  | .div    => .bin .div
  | .rem    => .bin .rem
  | .shl    => .bin .shl
  | .shr    => .bin .shr
  | .bAnd   => .bin .bAnd
  | .bOr    => .bin .bOr
  | .bXor   => .bin .bXor
  | .logAnd => .logical true
  | .logOr  => .logical false
  | .eq     => .cmp .eq
  | .ne     => .cmp .ne
  | .lt     => .cmp .lt
  | .le     => .cmp .le
  | .gt     => .cmp .gt
  | .ge     => .cmp .ge

def dispatchUnaryOp : UnaryOp → KOps.UnaryOp
  | .neg    => .neg
  | .logNot => .logNot
  | .bNot   => .bNot

/-- KRust scalar → KOps scalar. Both views share the same alphabet
    structurally; this is just the namespace bridge. -/
def dispatchScalar : Scalar → KOps.Scalar
  | .bool => .bool
  | .i8   => .i8 | .i16 => .i16 | .i32 => .i32 | .i64 => .i64
  | .u8   => .u8 | .u16 => .u16 | .u32 => .u32 | .u64 => .u64
  | .f16  => .f16 | .f32 => .f32 | .f64 => .f64

-- ════════════════════════════════════════════════════════════════════
-- Literals — pin a `ConstValue` from a `Lit`
-- ════════════════════════════════════════════════════════════════════

def constOfLit : Lit → Option (KOps.ConstValue × KOps.Scalar)
  | .bool b           => some (.bool b, .bool)
  | .int n .i32       => some (.i32 n, .i32)
  | .int n .u32       =>
      if n < 0 then none
      else some (.u32 (UInt32.ofNat n.toNat), .u32)
  | .int _ _          => none   -- 8/16/64-bit not yet
  | .float w f .f32   =>
      -- Same fixed-point encoding `KRust.Semantics.evalLit` uses;
      -- the per-rule theorem then has both sides agreeing on bits.
      some (.f32 (UInt32.ofNat ((w.toNat * 1000) + f.toNat % 1000)), .f32)
  | .float _ _ _      => none

-- ════════════════════════════════════════════════════════════════════
-- Mutual translator
-- ════════════════════════════════════════════════════════════════════
--
-- `translateExpr` returns the destination register, the result
-- type, and the updated context (with the emitted ops appended to
-- `ctx.ops`). `translateStmt` and `translateStmts` thread the
-- same context.

mutual
partial def translateExpr (ctx : EmitCtx) : Expr → Option (Reg × KOps.Scalar × EmitCtx)
  | .lit l => do
      let (cv, ty) ← constOfLit l
      let (dst, ctx1) := ctx.fresh
      let ctx2 := ctx1.emit (KernelOp.const dst cv)
      pure (dst, ty, ctx2)
  | .path n => do
      let r ← ctx.lookupVar n
      -- A path read is a register reference; the type is recovered
      -- from the binding (which we don't track at this layer for
      -- simplicity — set to `u32` and let the per-rule theorem fix
      -- this once the typed binding map lands in E.4).
      pure (r, .u32, ctx)
  | .binary op a b => do
      let (ra, _, ctx1) ← translateExpr ctx a
      let (rb, _, ctx2) ← translateExpr ctx1 b
      let (dst, ctx3) := ctx2.fresh
      match dispatchBinOp op with
      | .bin bop =>
          pure (dst, .u32, ctx3.emit (KernelOp.binOp dst ra rb bop .u32))
      | .cmp cop =>
          pure (dst, .bool, ctx3.emit (KernelOp.cmp dst ra rb cop .u32))
      | .logical _ =>
          -- `&&` / `||` short-circuit; the proc macro lowers them
          -- to `branch` blocks. Not yet modelled here — return
          -- `none` so per-rule preservation skips them and the
          -- caller knows this is the open obligation.
          none
  | .unary op a => do
      let (ra, _, ctx1) ← translateExpr ctx a
      let (dst, ctx2) := ctx1.fresh
      pure (dst, .u32, ctx2.emit (KernelOp.unaryOp dst ra (dispatchUnaryOp op) .u32))
  | .index arr i => do
      let slot ← ctx.lookupParam arr
      let (ri, _, ctx1) ← translateExpr ctx i
      let (dst, ctx2) := ctx1.fresh
      pure (dst, .u32, ctx2.emit (KernelOp.load dst slot ri .u32))
  | .indexRef _ _ =>
      -- `&arr[i]` only appears as an argument to atomic ops; the
      -- atomic call site folds the load into the atomic op rather
      -- than emitting a separate KOp. Until atomicCall lands, this
      -- traps so per-rule preservation skips it.
      none
  | .fieldRef _ _ =>
      -- Macro flattens these to plain index ops before translate;
      -- a `fieldRef` reaching here is a translator-context bug.
      none
  | .cast e ty => do
      let (rs, _, ctx1) ← translateExpr ctx e
      let (dst, ctx2) := ctx1.fresh
      pure (dst, dispatchScalar ty,
            ctx2.emit (KernelOp.cast dst rs .u32 (dispatchScalar ty)))
  | .ifE c t e => do
      let (rc, _, ctx1) ← translateExpr ctx c
      -- Translate both arms into separate sub-contexts so we can
      -- harvest just the ops (the outer ctx accumulates everything
      -- via the `branch` op below).
      let thenCtx0 : EmitCtx := { ctx1 with ops := [] }
      let (rt, _, thenCtx) ← translateExpr thenCtx0 t
      let elseCtx0 : EmitCtx := { thenCtx with ops := [] }
      let (re, _, elseCtx) ← translateExpr elseCtx0 e
      let (dst, ctx2) := elseCtx.fresh
      let ctx3 := { ctx2 with ops := ctx1.ops }
      let thenOps := thenCtx.ops ++ [KernelOp.copy dst rt]
      let elseOps := elseCtx.ops ++ [KernelOp.copy dst re]
      pure (dst, .u32, ctx3.emit (KernelOp.branch rc thenOps elseOps))
  | .blockE body tail => do
      let ctx1 ← translateStmts ctx body
      translateExpr ctx1 tail
  | .mathCall _ _ | .waveCall _ _ | .atomicCall _ _ | .identityCall _ | .method _ _ _ =>
      -- Open obligations — per-rule preservation theorems for these
      -- cases land in E.4 once `KRust.Semantics`'s mathCall arm
      -- wires through to `Quanta.Semantics.Cpu`'s f32 library.
      none

partial def translateStmt (ctx : EmitCtx) : Stmt → Option EmitCtx
  | .letDecl name _ rhs => do
      let (r, _, ctx1) ← translateExpr ctx rhs
      pure (ctx1.bindVar name r)
  | .letTuple _ _ _ => none
  | .exprS e => do
      let (_, _, ctx1) ← translateExpr ctx e
      pure ctx1
  | .assignVar name rhs => do
      let (r, _, ctx1) ← translateExpr ctx rhs
      pure (ctx1.bindVar name r)
  | .assignIdx arr idx rhs => do
      let slot ← ctx.lookupParam arr
      let (ri, _, ctx1) ← translateExpr ctx idx
      let (rr, _, ctx2) ← translateExpr ctx1 rhs
      pure (ctx2.emit (KernelOp.store slot ri rr .u32))
  | .ifS c thenS elseS =>
      match translateExpr ctx c with
      | none => none
      | some (rc, _, ctx1) =>
          let thenCtx0 : EmitCtx := { ctx1 with ops := [] }
          match (translateStmts thenCtx0 thenS : Option EmitCtx) with
          | none => none
          | some thenCtxS =>
              let elseCtx0 : EmitCtx := { thenCtxS with ops := [] }
              match (translateStmts elseCtx0 elseS : Option EmitCtx) with
              | none => none
              | some elseCtxS =>
                  let ctx2 : EmitCtx := { elseCtxS with ops := ctx1.ops }
                  some (ctx2.emit (KernelOp.branch rc thenCtxS.ops elseCtxS.ops))
  | .forRange name lo hi body =>
      -- `for i in lo..hi { body }` lowers to:
      --   let i = lo;
      --   loop { if !(i < hi) break; body; i = i + 1; }
      match translateExpr ctx lo with
      | none => none
      | some (rlo, _, ctx1) =>
          match translateExpr ctx1 hi with
          | none => none
          | some (rhi, _, ctx2) =>
              let (ri, ctx3) := ctx2.fresh
              let ctx4 := ctx3.emit (KernelOp.copy ri rlo)
              let ctx5 := ctx4.bindVar name ri
              let bodyCtx0 : EmitCtx := { ctx5 with ops := [] }
              match (translateStmts bodyCtx0 body : Option EmitCtx) with
              | none => none
              | some bodyCtxF =>
                  let (rcmp, ctx6) := bodyCtxF.fresh
                  let (rone, ctx7) := ctx6.fresh
                  let (rinc, ctx8) := ctx7.fresh
                  let header :=
                    [ KernelOp.cmp rcmp ri rhi .ge .u32
                    , KernelOp.branch rcmp [KernelOp.breakOp] []
                    ]
                  let trailer :=
                    [ KernelOp.const rone (.u32 1)
                    , KernelOp.binOp rinc ri rone .add .u32
                    , KernelOp.copy ri rinc
                    ]
                  let loopBody := header ++ bodyCtxF.ops ++ trailer
                  let ctx9 : EmitCtx := { ctx8 with ops := ctx5.ops }
                  some (ctx9.emit (KernelOp.loopOp loopBody))
  | .whileS cond body =>
      let condCtx0 : EmitCtx := { ctx with ops := [] }
      match translateExpr condCtx0 cond with
      | none => none
      | some (rc, _, condCtx) =>
          let bodyCtx0 : EmitCtx := { condCtx with ops := [] }
          match (translateStmts bodyCtx0 body : Option EmitCtx) with
          | none => none
          | some bodyCtxW =>
              let header :=
                condCtx.ops ++
                [ KernelOp.unaryOp rc rc .logNot .bool
                , KernelOp.branch rc [KernelOp.breakOp] []
                ]
              let loopBody := header ++ bodyCtxW.ops
              let ctx2 : EmitCtx := { bodyCtxW with ops := ctx.ops }
              some (ctx2.emit (KernelOp.loopOp loopBody))
  | .loopS body =>
      let bodyCtx0 : EmitCtx := { ctx with ops := [] }
      match (translateStmts bodyCtx0 body : Option EmitCtx) with
      | none => none
      | some bodyCtxL =>
          let ctx1 : EmitCtx := { bodyCtxL with ops := ctx.ops }
          some (ctx1.emit (KernelOp.loopOp bodyCtxL.ops))
  | .breakS =>
      pure (ctx.emit KernelOp.breakOp)
  | .callS _ _ => none

partial def translateStmts (ctx : EmitCtx) : List Stmt → Option EmitCtx
  | []      => some ctx
  | st :: rest => do
      let ctx1 ← translateStmt ctx st
      translateStmts ctx1 rest
end

end Quanta.KRust
