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

/-- A `heapStore` at `(slot, idx)` makes the lookup at the same key
    return the stored value. Direct from the head-of-list structure. -/
@[simp] theorem heapLookup_heapStore_self
    (h : Heap) (slot idx : Nat) (v : Value) :
    heapLookup (heapStore h slot idx v) slot idx = some v := by
  unfold heapLookup heapStore
  simp [List.find?]

/-- For an association list keyed by `Nat × Nat`, dropping all entries
    matching one key (via `filter ≠ target`) doesn't perturb the
    `find?` result for any other key. -/
private theorem find?_filter_ne_target
    (h : Heap) (target search : Nat × Nat) (h_ne : search ≠ target) :
    (h.filter (fun p => p.fst ≠ target)).find? (fun p => decide (p.fst = search)) =
    h.find? (fun p => decide (p.fst = search)) := by
  induction h with
  | nil => rfl
  | cons p ps ih =>
    by_cases hp_target : p.fst = target
    · -- Filter drops p; original-list find? skips p (since p.fst = target ≠ search).
      have hp_ne_search : ¬ p.fst = search := by
        intro heq; exact h_ne (heq.symm.trans hp_target)
      have h_filter_drop : List.filter (fun q => decide (q.fst ≠ target)) (p :: ps) =
                           List.filter (fun q => decide (q.fst ≠ target)) ps := by
        simp [List.filter, hp_target]
      rw [h_filter_drop]
      -- Reduce the RHS via List.find?_cons + decide-false on the head check.
      conv => rhs; rw [List.find?_cons]
      have h_search_false : decide (p.fst = search) = false := decide_eq_false hp_ne_search
      simp only [h_search_false, Bool.false_eq_true, ite_false]
      exact ih
    · -- Filter keeps p.
      have h_filter_keep : List.filter (fun q => decide (q.fst ≠ target)) (p :: ps) =
                           p :: List.filter (fun q => decide (q.fst ≠ target)) ps := by
        simp [List.filter, hp_target]
      rw [h_filter_keep]
      conv => lhs; rw [List.find?_cons]
      conv => rhs; rw [List.find?_cons]
      by_cases hp_search : p.fst = search
      · -- Head matches search; both `if`s take the true branch.
        have h_search_true : decide (p.fst = search) = true := decide_eq_true hp_search
        simp only [h_search_true, ite_true]
      · -- Head doesn't match; both `if`s take the false branch; recurse.
        have h_search_false : decide (p.fst = search) = false := decide_eq_false hp_search
        simp only [h_search_false, Bool.false_eq_true, ite_false]
        exact ih

/-- `heapStore` at `(slot, idx)` doesn't affect the lookup at any
    other key. Cons-of-store doesn't match, filter preserves the
    target key's entry by hypothesis. -/
theorem heapLookup_heapStore_other
    (h : Heap) (slot idx : Nat) (v : Value)
    (slot' idx' : Nat) (h_ne : (slot', idx') ≠ (slot, idx)) :
    heapLookup (heapStore h slot idx v) slot' idx' = heapLookup h slot' idx' := by
  unfold heapLookup heapStore
  -- Strip cons head: ((slot, idx), v) doesn't match (slot', idx') by h_ne.
  have h_head_ne_decide : decide (((slot, idx) : Nat × Nat) = (slot', idx')) = false := by
    apply decide_eq_false
    intro heq; exact h_ne heq.symm
  show Option.map Prod.snd (List.find? (fun p => decide (p.fst = (slot', idx')))
        (((slot, idx), v) :: h.filter (fun p => p.fst ≠ (slot, idx)))) =
       Option.map Prod.snd (List.find? (fun p => decide (p.fst = (slot', idx'))) h)
  rw [List.find?_cons]
  simp only [h_head_ne_decide, ite_false]
  congr 1
  exact find?_filter_ne_target h (slot, idx) (slot', idx') h_ne

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

def liftU32 (op : UInt32 → UInt32 → UInt32) : Value → Value → Option Value
  | .vU32 a, .vU32 b => some (vU32 (op a b))
  | _, _ => none

/-- `liftU32` is `none` whenever an operand is `vI32` — so an `orElse`
    chain falls through to the mixed lane. -/
@[simp] theorem liftU32_i32_left (op : UInt32 → UInt32 → UInt32) (a : Int) (v : Value) :
    liftU32 op (vI32 a) v = none := by cases v <;> simp [liftU32, vI32]
@[simp] theorem liftU32_i32_right (op : UInt32 → UInt32 → UInt32) (a : UInt32) (b : Int) :
    liftU32 op (vU32 a) (vI32 b) = none := by simp [liftU32, vU32, vI32]

/-- The 32-bit pattern an integer-typed value carries, as a `UInt32`:
    `vU32 n ↦ n`, `vI32 z ↦ UInt32.ofNat z.toNat` (the `vI32`↔`vU32`
    cast image — the `vI32` holds the non-negative bit pattern). `none`
    for non-integer values. The reinterpret used to give MIXED-tag
    integer binops well-defined bit-level semantics. -/
@[inline] def asU32Bits : Value → Option UInt32
  | .vU32 n => some n
  | .vI32 z => some (UInt32.ofNat z.toNat)
  | _       => none

/-- Round-trip: a `vI32` carrying the `Nat` bit pattern of a `UInt32`
    reads back as that `UInt32` through `asU32Bits` (the `vI32`↔`vU32`
    bit coherence). Used by the chained-address-add preservation, where
    the mixed evaluator tags a combined index `vI32 ↑(x.toNat)`. -/
@[simp] theorem asU32Bits_vI32_ofNat_toNat (x : UInt32) :
    asU32Bits (vI32 ↑(x.toNat)) = some x := by
  simp only [asU32Bits, vI32, Int.toNat_ofNat, UInt32.ofNat_toNat]

/-- Mixed-operand integer binop: when the operands carry different
    integer tags (one `vU32`, one `vI32`), reinterpret both to the
    common 32-bit pattern, apply the `UInt32` op, and tag the result
    `vI32 (·.toNat)` — mirroring production's `ty = I32` choice for a
    mixed binop. The same-`vU32` case is left to `liftU32` (returns
    `none` here), so wiring this as an `orElse` fallback keeps the
    existing `vU32,vU32` / `vI32,vI32` arms byte-identical. -/
def liftMixedI32 (op : UInt32 → UInt32 → UInt32) : Value → Value → Option Value
  | va, vb =>
    match va, vb with
    | .vU32 _, .vU32 _ => none
    | _, _ =>
      match asU32Bits va, asU32Bits vb with
      | some a, some b => some (vI32 ((op a b).toNat))
      | _, _ => none

/-- The three mixed/i32 lanes of `liftMixedI32`, as standalone facts the
    binop preservation instantiations reuse uniformly. -/
@[simp] theorem liftMixedI32_i32_i32 (op : UInt32 → UInt32 → UInt32) (a b : UInt32) :
    liftMixedI32 op (vI32 (a.toNat)) (vI32 (b.toNat)) = some (vI32 ((op a b).toNat)) := by
  simp [liftMixedI32, asU32Bits, vI32]
@[simp] theorem liftMixedI32_u32_i32 (op : UInt32 → UInt32 → UInt32) (a b : UInt32) :
    liftMixedI32 op (vU32 a) (vI32 (b.toNat)) = some (vI32 ((op a b).toNat)) := by
  simp [liftMixedI32, asU32Bits, vI32, vU32]
@[simp] theorem liftMixedI32_i32_u32 (op : UInt32 → UInt32 → UInt32) (a b : UInt32) :
    liftMixedI32 op (vI32 (a.toNat)) (vU32 b) = some (vI32 ((op a b).toNat)) := by
  simp [liftMixedI32, asU32Bits, vI32, vU32]

def liftCmpU32 (p : UInt32 → UInt32 → Bool) : Value → Value → Option Value
  | .vU32 a, .vU32 b => some (vBool (p a b))
  | _, _ => none

/-- Mixed/i32 comparison lane: when an operand carries a different
    integer tag than `vU32`, reinterpret BOTH to the common 32-bit
    pattern (`asU32Bits`) and apply the UNSIGNED predicate `p` —
    matching wasm's `i32.{lt,le,gt,ge}_u` / `eq` / `ne`, which compare
    the bit pattern. Inert on the `vU32,vU32` case (handled by
    `liftCmpU32`), so it wires as an `orElse` fallback. -/
def liftCmpMixed (p : UInt32 → UInt32 → Bool) : Value → Value → Option Value
  | va, vb =>
    match va, vb with
    | .vU32 _, .vU32 _ => none
    | _, _ =>
      match asU32Bits va, asU32Bits vb with
      | some a, some b => some (vBool (p a b))
      | _, _ => none

@[simp] theorem liftCmpMixed_i32_i32 (p : UInt32 → UInt32 → Bool) (a b : UInt32) :
    liftCmpMixed p (vI32 (a.toNat)) (vI32 (b.toNat)) = some (vBool (p a b)) := by
  simp [liftCmpMixed, asU32Bits, vI32]
@[simp] theorem liftCmpMixed_u32_i32 (p : UInt32 → UInt32 → Bool) (a b : UInt32) :
    liftCmpMixed p (vU32 a) (vI32 (b.toNat)) = some (vBool (p a b)) := by
  simp [liftCmpMixed, asU32Bits, vI32, vU32]
@[simp] theorem liftCmpMixed_i32_u32 (p : UInt32 → UInt32 → Bool) (a b : UInt32) :
    liftCmpMixed p (vI32 (a.toNat)) (vU32 b) = some (vBool (p a b)) := by
  simp [liftCmpMixed, asU32Bits, vI32, vU32]
@[simp] theorem liftCmpU32_i32_left (p : UInt32 → UInt32 → Bool) (a : Int) (v : Value) :
    liftCmpU32 p (vI32 a) v = none := by cases v <;> simp [liftCmpU32, vI32]
@[simp] theorem liftCmpU32_i32_right (p : UInt32 → UInt32 → Bool) (a : UInt32) (b : Int) :
    liftCmpU32 p (vU32 a) (vI32 b) = none := by simp [liftCmpU32, vU32, vI32]

def liftBoolBin (op : Bool → Bool → Bool) : Value → Value → Option Value
  | .vBool a, .vBool b => some (vBool (op a b))
  | _, _ => none

@[simp] theorem liftBoolBin_i32_left (op : Bool → Bool → Bool) (a : Int) (v : Value) :
    liftBoolBin op (vI32 a) v = none := by cases v <;> simp [liftBoolBin, vI32]
@[simp] theorem liftBoolBin_i32_right (op : Bool → Bool → Bool) (a : UInt32) (b : Int) :
    liftBoolBin op (vU32 a) (vI32 b) = none := by simp [liftBoolBin, vU32, vI32]

def evalBinOp : BinOp → Value → Value → Option Value
  -- u32 lane first; every non-(u32,u32) integer combination (i32+i32 or
  -- mixed) routes through `liftMixedI32`, which reinterprets to the
  -- common 32-bit pattern and applies the WRAPPING op — matching wasm
  -- i32 semantics. (The old `liftI32` arm did unbounded `Int`
  -- arithmetic, which is wrong for wrapping; it was only reachable once
  -- i32-typed values exist, i.e. post-V8-#2, and is dropped here.)
  | .add  => fun va vb =>
      (liftU32 eval_u32_wrapping_add va vb).orElse (fun _ =>
        liftMixedI32 eval_u32_wrapping_add va vb)
  | .sub  => fun va vb =>
      (liftU32 eval_u32_wrapping_sub va vb).orElse (fun _ =>
        liftMixedI32 eval_u32_wrapping_sub va vb)
  | .mul  => fun va vb =>
      (liftU32 eval_u32_wrapping_mul va vb).orElse (fun _ =>
        liftMixedI32 eval_u32_wrapping_mul va vb)
  | .div  => fun va vb =>
      (liftU32 eval_u32_div va vb).orElse (fun _ => liftMixedI32 eval_u32_div va vb)
  | .rem  => fun va vb =>
      (liftU32 eval_u32_rem va vb).orElse (fun _ => liftMixedI32 eval_u32_rem va vb)
  | .bAnd => fun va vb =>
      (liftU32 eval_u32_bitand va vb).orElse (fun _ => liftMixedI32 (· &&& ·) va vb)
  | .bOr  => fun va vb =>
      (liftU32 eval_u32_bitor va vb).orElse (fun _ => liftMixedI32 (· ||| ·) va vb)
  | .bXor => fun va vb =>
      (liftU32 eval_u32_bitxor va vb).orElse (fun _ => liftMixedI32 (· ^^^ ·) va vb)
  | .shl  => fun va vb =>
      match va, vb with
      | .vU32 a, .vU32 b => some (vU32 (a <<< b))
      | _, _ => liftMixedI32 (· <<< ·) va vb
  | .shr  => fun va vb =>
      match va, vb with
      | .vU32 a, .vU32 b => some (vU32 (a >>> b))
      | _, _ => liftMixedI32 (· >>> ·) va vb
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

-- ════════════════════════════════════════════════════════════════════
-- Mixed-lane correctness (V8-#2)
--
-- When a binop sees a `vU32 a` and a `vI32 (b.toNat)` operand (the
-- shape an i32-tagged const adds to a u32-tagged reg), the result
-- carries the SAME 32-bit pattern the pure-`vU32` lane would produce —
-- the `vI32` operand reinterprets to its bits `b`, the op runs on
-- `(a, b)`, and the result re-enters the `vI32` lane carrying those
-- bits. These lemmas are the bridge the binop preservation uses to
-- show a mixed binop refines the wasm bit-pattern semantics.
-- ════════════════════════════════════════════════════════════════════

/-- `asU32Bits` recovers the bit pattern from a `vI32` carrying a
    UInt32's `toNat` (the canonical i32-reg encoding). -/
@[simp] theorem asU32Bits_vI32_ofBits (b : UInt32) :
    asU32Bits (vI32 (b.toNat)) = some b := by
  simp [asU32Bits, vI32]

@[simp] theorem asU32Bits_vU32 (a : UInt32) :
    asU32Bits (vU32 a) = some a := by
  simp [asU32Bits, vU32]

/-- The mixed `.add` lane: `vU32 a` + `vI32 (b.toNat)` evaluates to
    `vI32 ((a + b).toNat)` — the wrapping-add bit pattern, tagged i32.
    This is the exact result an `i32.add` of a u32-reg and an
    i32-const produces, and the value the `.reg dst .i32` encoding of
    `wI32 (a + b)` expects (`regHoldsU32Bits` at the bits `a + b`). -/
theorem evalBinOp_add_mixed_u32_i32 (a b : UInt32) :
    evalBinOp .add (vU32 a) (vI32 (b.toNat))
      = some (vI32 ((eval_u32_wrapping_add a b).toNat)) := by
  simp only [evalBinOp, liftU32, vU32, vI32, Option.orElse]
  -- u32 lane: liftU32 on (vU32, vI32) is none; i32 lane: none; mixed fires.
  simp [liftMixedI32, asU32Bits, eval_u32_wrapping_add, vI32, vU32]

/-- Symmetric: `vI32 (a.toNat)` + `vU32 b` (i32-const as the FIRST
    operand). Same bit pattern. -/
theorem evalBinOp_add_mixed_i32_u32 (a b : UInt32) :
    evalBinOp .add (vI32 (a.toNat)) (vU32 b)
      = some (vI32 ((eval_u32_wrapping_add a b).toNat)) := by
  simp only [evalBinOp, liftU32, vU32, vI32, Option.orElse]
  simp [liftMixedI32, asU32Bits, eval_u32_wrapping_add, vI32, vU32]

/-- Both-i32 `.add` (two i32 consts): wrapping bit pattern, tagged i32. -/
theorem evalBinOp_add_i32_i32 (a b : UInt32) :
    evalBinOp .add (vI32 (a.toNat)) (vI32 (b.toNat))
      = some (vI32 ((eval_u32_wrapping_add a b).toNat)) := by
  simp [evalBinOp, liftU32, liftMixedI32, asU32Bits, eval_u32_wrapping_add, vI32, vU32,
        Option.orElse]

/-- Bit-level `.add` over arbitrary integer-tagged operands: from the
    32-bit patterns `a`/`b` of two values (`asU32Bits`), `evalBinOp .add`
    succeeds and the RESULT carries the wrapping-add pattern `a + b`
    (regardless of the two operand tags — u32+u32 keeps `vU32`, any mix
    keeps `vI32`, but `asU32Bits` of the result is `a + b` either way).
    The address-fold preservation reads only the bit pattern of the
    combined index, so this tag-agnostic form is exactly what it needs. -/
theorem evalBinOp_add_asU32Bits {va vb : Value} {a b : UInt32}
    (ha : asU32Bits va = some a) (hb : asU32Bits vb = some b) :
    ∃ vr, evalBinOp .add va vb = some vr ∧
      (vr = vU32 (a + b) ∨ vr = vI32 ↑((a + b).toNat)) := by
  -- Case on the two value tags. Only integer values have `asU32Bits`.
  cases va with
  | vU32 na =>
    cases vb with
    | vU32 nb =>
      simp only [asU32Bits, Option.some.injEq] at ha hb; subst ha; subst hb
      exact ⟨vU32 (na + nb), by
        simp [evalBinOp, liftU32, vU32, eval_u32_wrapping_add, Option.orElse], Or.inl rfl⟩
    | vI32 nb =>
      simp only [asU32Bits, Option.some.injEq] at ha hb; subst ha; subst hb
      refine ⟨vI32 ((na + UInt32.ofNat nb.toNat).toNat), ?_, Or.inr rfl⟩
      simp [evalBinOp, liftU32, liftMixedI32, asU32Bits, eval_u32_wrapping_add,
            vU32, vI32, Option.orElse]
    | vBool _ => simp [asU32Bits] at hb
    | vF32 _ => simp [asU32Bits] at hb
  | vI32 na =>
    cases vb with
    | vU32 nb =>
      simp only [asU32Bits, Option.some.injEq] at ha hb; subst ha; subst hb
      refine ⟨vI32 ((UInt32.ofNat na.toNat + nb).toNat), ?_, Or.inr rfl⟩
      simp [evalBinOp, liftU32, liftMixedI32, asU32Bits, eval_u32_wrapping_add,
            vU32, vI32, Option.orElse]
    | vI32 nb =>
      simp only [asU32Bits, Option.some.injEq] at ha hb; subst ha; subst hb
      refine ⟨vI32 ((UInt32.ofNat na.toNat + UInt32.ofNat nb.toNat).toNat), ?_, Or.inr rfl⟩
      simp [evalBinOp, liftU32, liftMixedI32, asU32Bits, eval_u32_wrapping_add,
            vI32, Option.orElse]
    | vBool _ => simp [asU32Bits] at hb
    | vF32 _ => simp [asU32Bits] at hb
  | vBool _ => simp [asU32Bits] at ha
  | vF32 _ => simp [asU32Bits] at ha

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
  -- u32 lane first, then bool (for `eq`/`ne` on `vBool`), then the
  -- mixed/i32 lane (unsigned compare on the common bit pattern). The
  -- old signed `liftCmpI32` arm is dropped — the lowered slice only
  -- emits UNSIGNED i32 comparisons (`i32.lt_u` etc.), and a signed Int
  -- compare on i32-typed values would be wrong for those.
  | .eq => fun va vb =>
      (liftCmpU32 (· == ·) va vb).orElse (fun _ =>
      (liftBoolBin (· == ·) va vb).orElse (fun _ =>
        liftCmpMixed (· == ·) va vb))
  | .ne => fun va vb =>
      (liftCmpU32 (· != ·) va vb).orElse (fun _ =>
      (liftBoolBin (· != ·) va vb).orElse (fun _ =>
        liftCmpMixed (· != ·) va vb))
  | .lt => fun va vb =>
      (liftCmpU32 (· < ·) va vb).orElse (fun _ => liftCmpMixed (· < ·) va vb)
  | .le => fun va vb =>
      (liftCmpU32 (· <= ·) va vb).orElse (fun _ => liftCmpMixed (· <= ·) va vb)
  | .gt => fun va vb =>
      (liftCmpU32 (· > ·) va vb).orElse (fun _ => liftCmpMixed (· > ·) va vb)
  | .ge => fun va vb =>
      (liftCmpU32 (· >= ·) va vb).orElse (fun _ => liftCmpMixed (· >= ·) va vb)

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
      -- U32→Bool: WASM `br_if` reads i32 cond as bool (any non-zero
      -- value is true). The WASM-route brIf lowering inserts a
      -- `.cast cond_bool cond .u32 .bool` between the comparison's
      -- u32 result and the `.branch`'s bool input. Total on vU32.
      | .vU32 n  => some (vBool (n ≠ 0))
      -- An i32-tagged condition (e.g. `br_if (i32.const k)`) carries the
      -- same 32-bit pattern; non-zero ⇒ true. (V8-#2.)
      | .vI32 z  => some (vBool (z.toNat ≠ 0))
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
      -- An i32-tagged index reg carries the same byte offset as a u32
      -- one (`z.toNat` is the bit pattern); production's pointer
      -- arithmetic can tag the index either way. (V8-#2.)
      | .vI32 z =>
          let v ← heapLookup s.heap field z.toNat
          pure { s with rf := regWrite s.rf dst v }
      | _ => none
  | .store field idx src _ty => do
      let vi ← regLookup s.rf idx
      let vs ← regLookup s.rf src
      -- The stored value is normalized to its 32-bit pattern as `vU32`:
      -- a flat-buffer cell holds bits, and production may tag the stored
      -- value i32 or u32 (same bits). Keeps the heap u32-shaped so
      -- `HeapRefines` / typed loads see a consistent alphabet. (V8-#2.)
      let bits ← asU32Bits vs
      match vi with
      | .vU32 n => pure { s with heap := heapStore s.heap field n.toNat (vU32 bits) }
      | .vI32 z => pure { s with heap := heapStore s.heap field z.toNat (vU32 bits) }
      | _ => none
  | .branch cond thenOps elseOps => do
      let vc ← regLookup s.rf cond
      match vc with
      | .vBool true  => evalOps fuel s thenOps
      | .vBool false => evalOps fuel s elseOps
      | _ => none
  | .loopOp body => opLoop fuel body fuel s
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

/-- Body-iteration loop for `.loopOp`. Iterates `evalOps fuel st
    body` until either `st.broke = true` (clear flag, return cleaned
    state) or the iteration counter `f` runs out (return `none`).

    Lifted from a nested `let rec` inside `evalOp .loopOp`'s arm so
    that fuel-monotonicity lemmas about loop iteration can be stated
    and proven externally. The `evalOp .loopOp body` arm just
    delegates: `opLoop fuel body fuel s`.

    Two fuel dimensions:
    * `fuel` (outer): passed to each `evalOps fuel st body` call,
      bounding the depth of structured control inside the body.
    * `f` (iteration counter): bounds the total number of loop
      iterations. Decremented per iteration. -/
def opLoop (fuel : Nat) (body : List KernelOp) :
    Nat → State → Option State
  | 0,     _  => none
  | f + 1, st =>
      if st.broke then some st.reset_broke
      else
        match evalOps fuel st body with
        | none => none
        | some st' => opLoop fuel body f st'
end

end Quanta.KOps
