/-
# KernelOps big-step semantics

Big-step ⇓ over `Quanta.KOps.KernelOp` and `List KernelOp`. Defines
the `Value` alphabet directly here — the legacy KRust source
preservation track (formerly the `Quanta.KRust.*` modules) was
deleted alongside its production translator in the WASM-route
cutover (2026-05-05). Step 059 will reintroduce a source-language
preservation theorem on top of the WASM operator subset, at which
point the Value alphabet may be shared again.

State shape:
- `RegFile` — `Reg → Value`. SSA-style: each register is written
  once during a kernel invocation, but the structure here is a
  flat association list updated by `Reg.write` (last-write-wins),
  matching how the executor actually behaves.
- `Heap` — `(Slot, idx) → Value`, modeling flat-buffer storage.
- `Dispatch` — thread-id values supplied by the runtime
  (`quark_id`, `proton_id`, …). Treated as a pure value tuple at
  this layer.

Loops carry a fuel parameter; the function is total given fuel.

Conventions match `crates/quanta-ir/src/driver/cpu/eval.rs` via the
shared primitives in `Quanta.Semantics.Cpu` — wrapping arithmetic,
div-by-zero returns 0.
-/

import Quanta.KOps.Syntax
import Quanta.Semantics.Cpu

namespace Quanta.KOps

open Quanta.Semantics.Cpu

-- ════════════════════════════════════════════════════════════════════
-- Value alphabet
-- ════════════════════════════════════════════════════════════════════

/-- The runtime value alphabet. Trapped / undefined results surface
    as `none` from the eval functions, not as a constructor here —
    keeps `Value.eq?` decidable. -/
inductive Value where
  | vBool (b : Bool)
  | vI32  (n : Int)
  | vU32  (n : UInt32)
  | vF32  (bits : UInt32)   -- IEEE-754 bits, evaluated via Cpu.eval_f32_*
  deriving Repr, DecidableEq

-- Convenience aliases so the dispatcher reads naturally.
@[inline] def vBool : Bool   → Value := Value.vBool
@[inline] def vI32  : Int    → Value := Value.vI32
@[inline] def vU32  : UInt32 → Value := Value.vU32
@[inline] def vF32  : UInt32 → Value := Value.vF32

-- ════════════════════════════════════════════════════════════════════
-- State (RegFile + Heap + dispatch context + break flag)
-- ════════════════════════════════════════════════════════════════════

/-- Register file: `Reg → Value`. Stored as a list keyed by `Reg`
    (a `Nat`); `regWrite` overwrites prior values for the same
    register, matching SSA-with-reassignment-tolerated semantics. -/
abbrev RegFile : Type := List (Reg × Value)

def regLookup (rf : RegFile) (r : Reg) : Option Value :=
  rf.find? (fun p => p.fst = r) |>.map Prod.snd

def regWrite (rf : RegFile) (r : Reg) (v : Value) : RegFile :=
  (r, v) :: rf.filter (fun p => p.fst ≠ r)

/-- Heap is field-slot keyed by parameter slot index — a `Nat` that
    matches the `KernelOp.load`/`store` `field` field. The same
    flat-buffer shape KRust uses, viewed via slot index instead of
    parameter name. The translator (E.3) carries a name → slot
    mapping so the two views see the same heap when projected
    through that mapping. -/
abbrev Heap : Type := List ((Nat × Nat) × Value)

def heapLookup (h : Heap) (slot idx : Nat) : Option Value :=
  h.find? (fun p => p.fst = (slot, idx)) |>.map Prod.snd

def heapStore (h : Heap) (slot idx : Nat) (v : Value) : Heap :=
  ((slot, idx), v) :: h.filter (fun p => p.fst ≠ (slot, idx))

/-- Dispatch context — the per-thread identity values. Every kernel
    runs at a position in the dispatch grid; these are reads-only
    from the kernel's perspective. -/
structure Dispatch where
  quarkId    : UInt32
  protonId   : UInt32
  nucleusId  : UInt32
  protonSize : UInt32
  quarkCount : UInt32
  deriving Repr

structure State where
  rf       : RegFile
  heap     : Heap
  dispatch : Dispatch
  broke    : Bool := false
  deriving Repr

def State.reset_broke (s : State) : State := { s with broke := false }

-- ════════════════════════════════════════════════════════════════════
-- Const + dispatch wirings
-- ════════════════════════════════════════════════════════════════════

def evalConst : ConstValue → Value
  | .bool b => vBool b
  | .i32  n => vI32 n
  | .u32  n => vU32 n
  | .f32  b => vF32 b

-- ════════════════════════════════════════════════════════════════════
-- BinOp / UnaryOp / Cmp dispatch — shares the lifters with KRust
-- ════════════════════════════════════════════════════════════════════
--
-- Reuses the same numeric primitives `Quanta.Semantics.Cpu` exposes.
-- The operator alphabet differs slightly (KOps adds `satAdd`/`satSub`)
-- so we list those separately; everything else routes through the
-- exact same `eval_u32_*` / `eval_i32_*` calls KRust uses.

private def liftU32 (op : UInt32 → UInt32 → UInt32) : Value → Value → Option Value
  | .vU32 a, .vU32 b => some (vU32 (op a b))
  | _, _ => none

private def liftI32 (op : Int → Int → Int) : Value → Value → Option Value
  | .vI32 a, .vI32 b => some (vI32 (op a b))
  | _, _ => none

private def liftCmpU32 (p : UInt32 → UInt32 → Bool) : Value → Value → Option Value
  | .vU32 a, .vU32 b => some (vBool (p a b))
  | _, _ => none

private def liftCmpI32 (p : Int → Int → Bool) : Value → Value → Option Value
  | .vI32 a, .vI32 b => some (vBool (p a b))
  | _, _ => none

private def liftBoolBin (op : Bool → Bool → Bool) : Value → Value → Option Value
  | .vBool a, .vBool b => some (vBool (op a b))
  | _, _ => none

def evalBinOp : BinOp → Value → Value → Option Value
  | .add  => fun va vb =>
      (liftU32 eval_u32_wrapping_add va vb).orElse (fun _ =>
        liftI32 (· + ·) va vb)
  | .sub  => fun va vb =>
      (liftU32 eval_u32_wrapping_sub va vb).orElse (fun _ =>
        liftI32 (· - ·) va vb)
  | .mul  => fun va vb =>
      (liftU32 eval_u32_wrapping_mul va vb).orElse (fun _ =>
        liftI32 (· * ·) va vb)
  | .div  => liftU32 eval_u32_div
  | .rem  => liftU32 eval_u32_rem
  | .bAnd => liftU32 eval_u32_bitand
  | .bOr  => liftU32 eval_u32_bitor
  | .bXor => liftU32 eval_u32_bitxor
  | .shl  => fun va vb =>
      match va, vb with
      | .vU32 a, .vU32 b => some (vU32 (a <<< b))
      | _, _ => none
  | .shr  => fun va vb =>
      match va, vb with
      | .vU32 a, .vU32 b => some (vU32 (a >>> b))
      | _, _ => none
  -- KOps-only saturating ops; not yet in KRust.
  | .satAdd => fun va vb =>
      match va, vb with
      | .vU32 a, .vU32 b =>
          let s := a.toNat + b.toNat
          some (vU32 (UInt32.ofNat (min s 0xFFFFFFFF)))
      | _, _ => none
  | .satSub => fun va vb =>
      match va, vb with
      | .vU32 a, .vU32 b =>
          if a < b then some (vU32 0) else some (vU32 (a - b))
      | _, _ => none

def evalUnaryOp : UnaryOp → Value → Option Value
  | .neg    => fun v => match v with
      | .vI32 n => some (vI32 (-n))
      | _       => none
  | .logNot => fun v => match v with
      | .vBool b => some (vBool (!b))
      | _        => none
  | .bNot   => fun v => match v with
      | .vU32 n => some (vU32 (~~~ n))
      | _       => none

def evalCmpOp : CmpOp → Value → Value → Option Value
  | .eq => fun va vb =>
      (liftCmpU32 (· == ·) va vb).orElse (fun _ =>
      (liftCmpI32 (· == ·) va vb).orElse (fun _ =>
        liftBoolBin (· == ·) va vb))
  | .ne => fun va vb =>
      (liftCmpU32 (· != ·) va vb).orElse (fun _ =>
      (liftCmpI32 (· != ·) va vb).orElse (fun _ =>
        liftBoolBin (· != ·) va vb))
  | .lt => fun va vb =>
      (liftCmpU32 (· < ·) va vb).orElse (fun _ =>
        liftCmpI32 (· < ·) va vb)
  | .le => fun va vb =>
      (liftCmpU32 (· <= ·) va vb).orElse (fun _ =>
        liftCmpI32 (· <= ·) va vb)
  | .gt => fun va vb =>
      (liftCmpU32 (· > ·) va vb).orElse (fun _ =>
        liftCmpI32 (· > ·) va vb)
  | .ge => fun va vb =>
      (liftCmpU32 (· >= ·) va vb).orElse (fun _ =>
        liftCmpI32 (· >= ·) va vb)

/-- Cast follows Rust `as`: same alphabet as `KRust.evalCast`,
    just keyed on the *target* `Scalar` since the source type is
    inferable from the runtime value. -/
def evalCast (v : Value) : Scalar → Option Value
  | .u32  => match v with
      | .vU32 n  => some (vU32 n)
      | .vI32 n  => some (vU32 (UInt32.ofNat n.toNat))
      -- Bool→U32: WASM encodes booleans as i32 0/1. The WASM-route
      -- translator emits `Cmp ; Cast bool→u32` so the cmp result re-
      -- enters the u32 arithmetic alphabet immediately. The Rust
      -- `bool as u32` lowering matches: `false → 0`, `true → 1`.
      | .vBool b => some (vU32 (if b then 1 else 0))
      | _        => none
  | .i32  => match v with
      | .vI32 n  => some (vI32 n)
      | .vU32 n  => some (vI32 n.toNat)
      -- Bool→I32 added for symmetry with Bool→U32; same numeric mapping.
      | .vBool b => some (vI32 (if b then 1 else 0))
      | _        => none
  | .bool => match v with
      | .vBool b => some (vBool b)
      | _        => none
  | _ => none

/-- Bitcast preserves the bit pattern; only u32 ↔ f32 wired here. -/
def evalBitcast (v : Value) : Scalar → Option Value
  | .f32 => match v with
      | .vU32 n => some (vF32 n)
      | _       => none
  | .u32 => match v with
      | .vF32 n => some (vU32 n)
      | _       => none
  | _ => none

-- ════════════════════════════════════════════════════════════════════
-- Big-step eval
-- ════════════════════════════════════════════════════════════════════

mutual
def evalOp (fuel : Nat) (s : State) : KernelOp → Option State
  | .const dst c =>
      pure { s with rf := regWrite s.rf dst (evalConst c) }
  | .binOp dst a b op _ty => do
      let va ← regLookup s.rf a
      let vb ← regLookup s.rf b
      let v ← evalBinOp op va vb
      pure { s with rf := regWrite s.rf dst v }
  | .unaryOp dst a op _ty => do
      let va ← regLookup s.rf a
      let v ← evalUnaryOp op va
      pure { s with rf := regWrite s.rf dst v }
  | .cmp dst a b op _ty => do
      let va ← regLookup s.rf a
      let vb ← regLookup s.rf b
      let v ← evalCmpOp op va vb
      pure { s with rf := regWrite s.rf dst v }
  | .cast dst src _fromTy to => do
      let v ← regLookup s.rf src
      let v' ← evalCast v to
      pure { s with rf := regWrite s.rf dst v' }
  | .bitcast dst src _fromTy to => do
      let v ← regLookup s.rf src
      let v' ← evalBitcast v to
      pure { s with rf := regWrite s.rf dst v' }
  | .copy dst src => do
      let v ← regLookup s.rf src
      pure { s with rf := regWrite s.rf dst v }
  | .load dst field idx _ty => do
      let vi ← regLookup s.rf idx
      match vi with
      | .vU32 n =>
          let v ← heapLookup s.heap field n.toNat
          pure { s with rf := regWrite s.rf dst v }
      | _ => none
  | .store field idx src _ty => do
      let vi ← regLookup s.rf idx
      let vs ← regLookup s.rf src
      match vi with
      | .vU32 n => pure { s with heap := heapStore s.heap field n.toNat vs }
      | _ => none
  | .branch cond thenOps elseOps => do
      let vc ← regLookup s.rf cond
      match vc with
      | .vBool true  => evalOps fuel s thenOps
      | .vBool false => evalOps fuel s elseOps
      | _ => none
  | .loopOp body =>
      let rec opLoop (f : Nat) (st : State) : Option State :=
        match f with
        | 0 => none
        | f+1 =>
            if st.broke then some st.reset_broke
            else
              match evalOps fuel st body with
              | none => none
              | some st' => opLoop f st'
      opLoop fuel s
  | .breakOp =>
      pure { s with broke := true }
  | .quarkId   dst => pure { s with rf := regWrite s.rf dst (vU32 s.dispatch.quarkId) }
  | .protonId  dst => pure { s with rf := regWrite s.rf dst (vU32 s.dispatch.protonId) }
  | .nucleusId dst => pure { s with rf := regWrite s.rf dst (vU32 s.dispatch.nucleusId) }
  | .protonSize dst => pure { s with rf := regWrite s.rf dst (vU32 s.dispatch.protonSize) }
  | .quarkCount dst => pure { s with rf := regWrite s.rf dst (vU32 s.dispatch.quarkCount) }
  | .barrier =>
      -- Barrier is a synchronization point in the parallel
      -- dispatch model; sequential CPU evaluation treats it as a
      -- no-op. The race-freedom track (Level 2, separate from E)
      -- pulls the parallel composition apart and gives barrier
      -- the happens-before semantics it deserves.
      pure s

def evalOps (fuel : Nat) (s : State) : List KernelOp → Option State
  | [] => some s
  | op :: rest => do
      let s1 ← evalOp fuel s op
      if s1.broke then some s1
      else evalOps fuel s1 rest
end

end Quanta.KOps
