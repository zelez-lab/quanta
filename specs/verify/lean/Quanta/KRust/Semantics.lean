/-
# Kernel Rust big-step semantics

Step **E.1b** of the source-preservation track. Defines a big-step
operational semantics over `KRust.Expr` and `KRust.Stmt` (from
`Quanta.KRust.Syntax`). The semantics is a function rather than a
relation so it composes with Lean's `decide` / `native_decide` for
closed examples; loops carry a fuel parameter to stay terminating.

Conventions match `crates/quanta-ir/src/driver/cpu/eval.rs` (the
reference executor) — Rust's wrapping arithmetic, division-by-zero
returns 0, signed cast follows `as`. The shared primitive library
in `Quanta.Semantics.Cpu` is the source of truth for those choices;
this module just lifts them to syntax-level evaluation.

Status (2026-04-28):
- Value type, Env, Heap, eval skeleton.
- BinOp / UnaryOp dispatchers wired into `Quanta.Semantics.Cpu`.
- Math / wave / atomic intrinsics: stubs that delegate to the same
  primitives where present, return `none` (trap) where the CPU
  reference doesn't define them yet.
- Loops carry a `Nat` fuel; the eval is total *given fuel*.

Restrictions kept light intentionally:
- Initial Value alphabet is `bool | i32 | u32 | f32` (not yet f16,
  64-bit, or signed-byte). The translator E.3 only needs whatever
  the proc macro currently emits; we widen alongside as needed.
- Struct-ref params land as plain field-keyed buffers (no nested
  struct semantics). The proc macro flattens them at parse time;
  the Lean view sees the post-flattening shape.
-/

import Quanta.KRust.Syntax
import Quanta.Semantics.Cpu

namespace Quanta.KRust

open Quanta.Semantics.Cpu

-- ════════════════════════════════════════════════════════════════════
-- Values + state
-- ════════════════════════════════════════════════════════════════════

/-- The runtime value alphabet. Trapped / undefined results
    surface as `none` from the eval functions, not as a constructor
    here — keeps `Value.eq?` decidable. -/
inductive Value where
  | vBool (b : Bool)
  | vI32  (n : Int)
  | vU32  (n : UInt32)
  | vF32  (bits : UInt32)   -- IEEE-754 bits, evaluated via Cpu.eval_f32_*
  deriving Repr, DecidableEq

/-- A buffer slot keyed by parameter name. The macro lowers struct-
    ref kernels to a flat slot list; this view honours that. -/
abbrev Slot : Type := Ident

/-- Environment: variable name → current value. -/
abbrev Env : Type := List (Ident × Value)

def Env.lookup (env : Env) (name : Ident) : Option Value :=
  env.find? (fun p => p.fst = name) |>.map Prod.snd

def Env.bind (env : Env) (name : Ident) (v : Value) : Env :=
  (name, v) :: env.filter (fun p => p.fst ≠ name)

/-- Heap: parameter slot + index → value. Models flat-buffer
    storage; the macro pre-resolves struct-ref `p.field` accesses
    onto a parameter slot, so the heap key is the slot identifier
    rather than `(struct, field)` pair. -/
abbrev Heap : Type := List ((Slot × Nat) × Value)

def Heap.lookup (h : Heap) (slot : Slot) (idx : Nat) : Option Value :=
  h.find? (fun p => p.fst = (slot, idx)) |>.map Prod.snd

def Heap.store (h : Heap) (slot : Slot) (idx : Nat) (v : Value) : Heap :=
  ((slot, idx), v) :: h.filter (fun p => p.fst ≠ (slot, idx))

/-- The triple `(env, heap, broke)` threaded through evaluation.
    `broke = true` means the most recent `breakS` fired and
    enclosing loops should stop iterating; resets at loop exit. -/
structure State where
  env  : Env
  heap : Heap
  broke : Bool := false
  deriving Repr

def State.reset_broke (s : State) : State := { s with broke := false }

-- ════════════════════════════════════════════════════════════════════
-- Literal eval
-- ════════════════════════════════════════════════════════════════════

/-- Convert a `KRust.Lit` to a runtime `Value`. Out-of-range
    literals trap (`none`). -/
def evalLit : Lit → Option Value
  | .bool b           => some (.vBool b)
  | .int n .i32       => some (.vI32 n)
  | .int n .u32       =>
      if n < 0 then none
      else some (.vU32 (UInt32.ofNat n.toNat))
  | .int _ _          => none   -- 8/16/64-bit not yet modelled
  | .float w f .f32   =>
      -- Coarse approximation — encode whole.frac as a fixed-point
      -- bit pattern. Sufficient for the per-rule preservation
      -- theorems (E.4) which thread literals as opaque values; the
      -- precise IEEE-754 interpretation is delegated to
      -- `Cpu.eval_f32_*` once the bits are constructed by the
      -- translator from a real `ConstValue::F32`.
      some (.vF32 (UInt32.ofNat ((w.toNat * 1000) + f.toNat % 1000)))
  | .float _ _ _      => none

-- ════════════════════════════════════════════════════════════════════
-- Type-driven binop dispatch
-- ════════════════════════════════════════════════════════════════════
--
-- A `KRust.BinOp` over two values commits to one of `eval_u32_*`,
-- `eval_i32_*`, or `eval_f32_*` depending on the runtime types.
-- Mismatch is a trap.

private def liftU32 (op : UInt32 → UInt32 → UInt32) (va vb : Value) : Option Value :=
  match va, vb with
  | .vU32 a, .vU32 b => some (.vU32 (op a b))
  | _, _ => none

private def liftI32 (op : Int → Int → Int) (va vb : Value) : Option Value :=
  match va, vb with
  | .vI32 a, .vI32 b => some (.vI32 (op a b))
  | _, _ => none

private def liftCmpU32 (p : UInt32 → UInt32 → Bool) (va vb : Value) : Option Value :=
  match va, vb with
  | .vU32 a, .vU32 b => some (.vBool (p a b))
  | _, _ => none

private def liftCmpI32 (p : Int → Int → Bool) (va vb : Value) : Option Value :=
  match va, vb with
  | .vI32 a, .vI32 b => some (.vBool (p a b))
  | _, _ => none

private def liftBoolBin (op : Bool → Bool → Bool) (va vb : Value) : Option Value :=
  match va, vb with
  | .vBool a, .vBool b => some (.vBool (op a b))
  | _, _ => none

/-- Big-step binary operator dispatch. The `Cpu` module supplies
    the wrapping-add, wrapping-sub, etc. primitives so the rules
    here track `eval.rs` exactly. Only u32 and i32 are wired today;
    the per-rule theorems (E.4) range only over types the proc
    macro produces. -/
def evalBinOp : BinOp → Value → Value → Option Value
  | .add   => fun va vb =>
      (liftU32 eval_u32_wrapping_add va vb).orElse (fun _ =>
        liftI32 (· + ·) va vb)
  | .sub   => fun va vb =>
      (liftU32 eval_u32_wrapping_sub va vb).orElse (fun _ =>
        liftI32 (· - ·) va vb)
  | .mul   => fun va vb =>
      (liftU32 eval_u32_wrapping_mul va vb).orElse (fun _ =>
        liftI32 (· * ·) va vb)
  | .div   => liftU32 eval_u32_div
  | .rem   => liftU32 eval_u32_rem
  | .bAnd  => liftU32 eval_u32_bitand
  | .bOr   => liftU32 eval_u32_bitor
  | .bXor  => liftU32 eval_u32_bitxor
  | .shl   => fun va vb =>
      match va, vb with
      | .vU32 a, .vU32 b => some (.vU32 (a <<< b))
      | _, _ => none
  | .shr   => fun va vb =>
      match va, vb with
      | .vU32 a, .vU32 b => some (.vU32 (a >>> b))
      | _, _ => none
  | .logAnd => liftBoolBin (· && ·)
  | .logOr  => liftBoolBin (· || ·)
  | .eq    => fun va vb =>
      (liftCmpU32 (· == ·) va vb).orElse (fun _ =>
      (liftCmpI32 (· == ·) va vb).orElse (fun _ =>
        liftBoolBin (· == ·) va vb))
  | .ne    => fun va vb =>
      (liftCmpU32 (· != ·) va vb).orElse (fun _ =>
      (liftCmpI32 (· != ·) va vb).orElse (fun _ =>
        liftBoolBin (· != ·) va vb))
  | .lt    => fun va vb =>
      (liftCmpU32 (· < ·) va vb).orElse (fun _ =>
        liftCmpI32 (· < ·) va vb)
  | .le    => fun va vb =>
      (liftCmpU32 (· <= ·) va vb).orElse (fun _ =>
        liftCmpI32 (· <= ·) va vb)
  | .gt    => fun va vb =>
      (liftCmpU32 (· > ·) va vb).orElse (fun _ =>
        liftCmpI32 (· > ·) va vb)
  | .ge    => fun va vb =>
      (liftCmpU32 (· >= ·) va vb).orElse (fun _ =>
        liftCmpI32 (· >= ·) va vb)

def evalUnaryOp : UnaryOp → Value → Option Value
  | .neg    => fun v => match v with
      | .vI32 n => some (.vI32 (-n))
      | _       => none
  | .logNot => fun v => match v with
      | .vBool b => some (.vBool (!b))
      | _        => none
  | .bNot   => fun v => match v with
      | .vU32 n => some (.vU32 (~~~ n))
      | _       => none

-- ════════════════════════════════════════════════════════════════════
-- Cast
-- ════════════════════════════════════════════════════════════════════
--
-- Mirrors Rust `as` semantics for the type pairs the macro emits.
-- Anything not yet wired traps.

def evalCast (v : Value) : Scalar → Option Value
  | .u32  => match v with
      | .vU32 n => some (.vU32 n)
      | .vI32 n => some (.vU32 (UInt32.ofNat n.toNat))
      | _       => none
  | .i32  => match v with
      | .vI32 n => some (.vI32 n)
      | .vU32 n => some (.vI32 n.toNat)
      | _       => none
  | .bool => match v with
      | .vBool b => some (.vBool b)
      | _        => none
  | _ => none

-- ════════════════════════════════════════════════════════════════════
-- Big-step eval
-- ════════════════════════════════════════════════════════════════════
--
-- `evalExpr` returns `(value, state)` — the state may have been
-- mutated by side-effecting sub-expressions inside `blockE` /
-- `ifE`. `evalStmt` returns `state` only.
--
-- A shared `Nat` fuel parameter bounds recursion so the function
-- terminates structurally on `(fuel, expr/stmt)`. Closed kernels
-- with no loops eval to completion at any fuel ≥ syntax depth.

mutual
def evalExpr (fuel : Nat) (s : State) : Expr → Option (Value × State)
  | .lit l => (evalLit l).map (fun v => (v, s))
  | .path n => (s.env.lookup n).map (fun v => (v, s))
  | .binary op a b => do
      let (va, s1) ← evalExpr fuel s a
      let (vb, s2) ← evalExpr fuel s1 b
      let v ← evalBinOp op va vb
      pure (v, s2)
  | .unary op a => do
      let (va, s1) ← evalExpr fuel s a
      let v ← evalUnaryOp op va
      pure (v, s1)
  | .index arr i => do
      let (vi, s1) ← evalExpr fuel s i
      match vi with
      | .vU32 n =>
          let v ← s1.heap.lookup arr n.toNat
          pure (v, s1)
      | .vI32 n =>
          if n < 0 then none
          else
            let v ← s1.heap.lookup arr n.toNat
            pure (v, s1)
      | _ => none
  | .indexRef arr i => do
      -- A reference to `arr[i]` evaluates as the same value the
      -- atomic op would observe; the `&` here is a marker for the
      -- macro, not a runtime distinction.
      let (vi, s1) ← evalExpr fuel s i
      match vi with
      | .vU32 n =>
          let v ← s1.heap.lookup arr n.toNat
          pure (v, s1)
      | _ => none
  | .fieldRef _base _field =>
      -- The macro flattens struct-ref accesses to slot lookups
      -- before reaching this point; if a `fieldRef` survives into
      -- the Lean view it's a translator bug, not a runtime fact.
      none
  | .cast e ty => do
      let (v, s1) ← evalExpr fuel s e
      let v' ← evalCast v ty
      pure (v', s1)
  | .ifE c t e => do
      let (vc, s1) ← evalExpr fuel s c
      match vc with
      | .vBool true  => evalExpr fuel s1 t
      | .vBool false => evalExpr fuel s1 e
      | _ => none
  | .blockE body tail => do
      let s1 ← evalStmts fuel s body
      evalExpr fuel s1 tail
  | .mathCall _ _    => none   -- E.1c will wire to Cpu.eval_f32_*
  | .waveCall _ _    => none   -- not modelled at this layer
  | .atomicCall _ _  => none   -- atomic semantics — separate module
  | .identityCall _  => none   -- thread-id values supplied by dispatch context (E.5)
  | .method _ _ _    => none   -- normalised away in E.3 — same target as mathCall
  termination_by e => (fuel, sizeOf e)

def evalStmt (fuel : Nat) (s : State) : Stmt → Option State
  | .letDecl name _ rhs => do
      let (v, s1) ← evalExpr fuel s rhs
      pure { s1 with env := s1.env.bind name v }
  | .letTuple _ _ _ =>
      -- Tuple destructuring — the macro lowers this to a sequence
      -- of single `letDecl`s in E.3; not modelled at this layer.
      none
  | .exprS e => do
      let (_, s1) ← evalExpr fuel s e
      pure s1
  | .assignVar name rhs => do
      let (v, s1) ← evalExpr fuel s rhs
      pure { s1 with env := s1.env.bind name v }
  | .assignIdx arr idx rhs => do
      let (vi, s1) ← evalExpr fuel s idx
      let (vr, s2) ← evalExpr fuel s1 rhs
      match vi with
      | .vU32 n => pure { s2 with heap := s2.heap.store arr n.toNat vr }
      | _ => none
  | .ifS c thenS elseS => do
      let (vc, s1) ← evalExpr fuel s c
      match vc with
      | .vBool true  => evalStmts fuel s1 thenS
      | .vBool false => evalStmts fuel s1 elseS
      | _ => none
  | .forRange name lo hi body =>
      match evalExpr fuel s lo, body with
      | none, _ => none
      | some (vlo, _), _ =>
          match evalExpr fuel s hi with
          | none => none
          | some (vhi, s2) =>
              match vlo, vhi with
              | .vU32 a, .vU32 b => evalForLoop fuel s2 name a.toNat b.toNat body
              | _, _ => none
  | .whileS cond body => evalWhileLoop fuel s cond body
  | .loopS body       => evalBareLoop fuel s body
  | .breakS => some { s with broke := true }
  | .callS _ _ => none   -- atomic / debug stmts — not at this layer
  termination_by st => (fuel, sizeOf st)

/-- For-loop iteration helper. Lifted out of `evalStmt`'s `let rec`
    so the mutual block has a single termination measure per
    function. Recurses on its own `fuel` parameter. -/
def evalForLoop : Nat → State → Ident → Nat → Nat → List Stmt → Option State
  | 0,     _,  _,    _, _, _    => none
  | f+1,   st, name, j, n, body =>
      if j ≥ n then some st.reset_broke
      else if st.broke then some st.reset_broke
      else
        let st' := { st with env := st.env.bind name (.vU32 (UInt32.ofNat j)) }
        match evalStmts f st' body with
        | none      => none
        | some st'' => evalForLoop f st'' name (j + 1) n body
  termination_by f _ => (f, 0)

/-- While-loop iteration helper. -/
def evalWhileLoop : Nat → State → Expr → List Stmt → Option State
  | 0,   _,  _,    _    => none
  | f+1, st, cond, body =>
      if st.broke then some st.reset_broke
      else
        match evalExpr f st cond with
        | some (.vBool false, st') => some st'
        | some (.vBool true, st')  =>
            match evalStmts f st' body with
            | none      => none
            | some st'' => evalWhileLoop f st'' cond body
        | _ => none
  termination_by f _ => (f, 0)

/-- Bare-loop iteration helper. -/
def evalBareLoop : Nat → State → List Stmt → Option State
  | 0,   _,  _    => none
  | f+1, st, body =>
      if st.broke then some st.reset_broke
      else
        match evalStmts f st body with
        | none     => none
        | some st' => evalBareLoop f st' body
  termination_by f _ => (f, 0)

def evalStmts (fuel : Nat) (s : State) : List Stmt → Option State
  | []      => some s
  | st :: rest => do
      let s1 ← evalStmt fuel s st
      if s1.broke then some s1
      else evalStmts fuel s1 rest
  termination_by xs => (fuel, sizeOf xs)
end

end Quanta.KRust
