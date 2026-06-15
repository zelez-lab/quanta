/-
# WASM → KernelOps preservation theorems (step 059, slices 1-4-cascade)

For every WASM instruction `i` in the lowered subset, executing `i`
on a WASM state `ws` and executing the lowered ops `lowerInstr s i`
on a KOps state `kst` from refinement-equivalent starting points
produces refinement-equivalent ending states.

Refinement structure:
* **Stack** (`StackRefines`) — the WASM stack and the symbolic stack
  zip element-wise; each WASM value encodes via a `SymVal`. The
  encoding is non-False only for `(.wI32 n, .reg r .u32)`; richer
  SymVal shapes (`bufferPtr`, `scaledIdx`, `bufferAccess`,
  `i32ConstSym`) are reserved for the buffer-pattern recognition
  arms in slice-4 step 7+ and are consumed inline before any value
  consumer fires.
* **Locals** (`LocalsRefines`) — every WASM local with a stable
  register encodes through that register, lifted as `.reg r .u32`.
* **Freshness** (`Fresh`) — every register currently held (any reg
  referenced by any stack SymVal, plus every local stable reg) is
  strictly less than `nextReg`. Load-bearing: `lowerInstr` always
  allocates `nextReg` and bumps it.
* **AliasFree** — no local's stable_reg appears anywhere in the
  symbolic stack's reg projection. `localGet` / `localTee` allocate
  fresh regs and Copy to break aliasing.
* **InjectiveLocals** — distinct local indices map to distinct
  stable_regs.
* **HeapRefines** — every in-bounds `(slot, idx)` in the KOps heap
  matches the 4-byte WASM-memory slice at the layout-derived
  address. Slice-3 ops don't touch memory or heap, so each per-op
  preservation theorem propagates the input `R.heapRefines` through
  unchanged. Slice-4 buffer-pattern arms are the first consumers.

## What ships now (slices 1 + 2 + 3 + slice-4 stack-type cascade
   + popSym/commit unification + slice-4 step 7 translator arms
   + slice-4 step 8 buffer-pattern preservation end-to-end)

* The full refinement bundle (`Refines` = stack + locals + freshness +
  alias-free + injective-locals + heap). `WasmValue.encodes` now takes
  `BufferLayout` with three address-SymVal arms:
  - `wI32 n ↔ .bufferPtr slot` when `n.toNat = layout.startAddr slot`.
  - `wI32 n ↔ .scaledIdx base scale` when `∃ b, regLookup rf base =
    some (vU32 b) ∧ n.toNat = b.toNat * scale`.
  - `wI32 n ↔ .bufferAccess slot base scale` when `∃ b, regLookup rf
    base = some (vU32 b) ∧ n.toNat = layout.startAddr slot + b.toNat
    * scale`. The Nat-equation form refuses overflow on the address
    arithmetic.
* Register-file lemmas: `regLookup_regWrite_self`,
  `regLookup_regWrite_of_ne`, `regLookup_preserved_of_fresh`.
* Encoding-lifting lemmas: `encodes_preserved_of_fresh`,
  `encodes_preserved_of_disjoint`, `encodes_preserved_of_lookup_eq`,
  plus inversion lemmas (`encodes_wI32_reg_inv`, `encodes_reg_shape`,
  `encodes_i32ConstSym_inv`, `encodes_bufferAccess_wI32_inv`).
* `commit_correct` + five sibling helpers.
* `evalOps_append` for chaining op-list evaluations past broke flags.
* Shape lemmas: `binI32_some_shape`, `cmpI32_some_shape`,
  `lowerI32Bin_some_shape`, `lowerI32Cmp_some_shape`.
* Generic binop / cmp preservation theorems (per-state `h_l`).
* Closed per-instruction theorems:
  - `preservation_nop`, `preservation_return`, `preservation_i32Const`
  - `preservation_localGet` (precondition: not buffer slot)
  - `preservation_localGet_bufferSlot` (buffer-typed; `HeapRefines`
    not yet consumed — bufferPtr push only)
  - `preservation_localSet`, `preservation_localTee`
  - `preservation_i32{Add,Shl}` (precondition: stack not buffer-pattern)
  - `preservation_i32{Sub,Mul,And,Or,Xor,ShrU,DivU,RemU}`
  - `preservation_i32{Eq,Ne,LtU,LeU,GtU,GeU}`
  - `preservation_i32Shl_bufferPattern` (folded `scaledIdx`)
  - `preservation_i32Add_bufferPattern_{scaledFirst,ptrFirst}`
    (folded `bufferAccess`)
  - `preservation_i32Load` (folded typed Load — first use of
    `HeapRefines`)
  - `preservation_i32Store` (folded typed Store — uses two
    `WasmMem.store_load_*` TCB axioms in `Quanta.Wasm.Semantics`
    plus the new `heapLookup_heapStore_{self,other}` helpers in
    `Quanta.KOps.Semantics`)

That's **28 closed preservation theorems**, 0 sorries, 2 new TCB
axioms (WasmMem byte-load/store roundtrip — narrow, capturing
well-known WASM spec compliance). The entire buffer-pattern
recognition chain (`localGet` → `i32.shl` → `i32.add` → `i32.load`
/ `i32.store`) is preserved end-to-end.

## What's next

**Slice 5** — control flow: frame reflection in `LowerState`;
proofs for `block`, `loop`, `if`/`else`, `br`, `br_if`, plus the
non-trivial `wreturn` (non-empty stack).

**Slice 6** — calls + intrinsics.

**Slice 7** — top-level composition.

## Invariant used by every per-instr theorem

The Translate pass guarantees `s'.nextReg ≥ s.nextReg` (it only
bumps, never resets). Every freshly-allocated register is exactly
`s.nextReg`, which by `Fresh s` is strictly larger than every
register the old state holds. So *any* write into the freshly-
allocated register preserves the readback of every register the
prior `Refines` instance constrains. That's the structural shape of
the per-instr proofs.
-/

import Quanta.Wasm.Syntax
import Quanta.Wasm.Semantics
import Quanta.Wasm.Translate
import Quanta.KOps.Syntax
import Quanta.KOps.Semantics

namespace Quanta.Wasm

open Quanta.KOps (KernelOp Reg evalOps regLookup regWrite)
open Quanta.Semantics.Cpu

-- ════════════════════════════════════════════════════════════════════
-- BufferLayout — mapping from heap slot to WASM-memory byte address
--
-- Each `#[quanta::shared]` buffer parameter occupies a contiguous
-- region of WASM linear memory; `BufferLayout` records the byte
-- address where each slot starts and the slot's element count. The
-- element type is always u32 in the slice-3 surface (4 bytes each);
-- richer element types lift in a later slice.
-- ════════════════════════════════════════════════════════════════════

structure BufferLayout where
  /-- Byte address in WASM linear memory where slot `s`'s data starts. -/
  startAddr : Nat → Nat
  /-- Number of u32 elements in slot `s`. -/
  length    : Nat → Nat

/-- Heap refinement: every in-bounds `(slot, idx)` pair in the KOps
    heap matches the 4-byte WASM-memory slice at the layout-derived
    address. Slice-3 ops don't touch `ws.mem` or `kst.heap`, so each
    per-op preservation theorem just propagates the input
    `HeapRefines` to the output. The slice-4 buffer-pattern arms
    (`i32.load`/`i32.store` consuming a `bufferAccess`) are the first
    consumers that USE this clause to bridge the byte-level WASM
    memory to the typed KOps heap. -/
def HeapRefines (mem : WasmMem) (heap : Quanta.KOps.Heap) (layout : BufferLayout) : Prop :=
  ∀ slot idx, idx < layout.length slot →
    ∃ n : UInt32,
      Quanta.KOps.heapLookup heap slot idx = some (Quanta.KOps.Value.vU32 n) ∧
      mem.load_u32 (layout.startAddr slot + idx * 4) = some n

-- ════════════════════════════════════════════════════════════════════
-- Refinement relation
-- ════════════════════════════════════════════════════════════════════

/-- A WASM value is encoded by a SymVal stack slot if any of:
    * the slot is `.reg r .u32` and the regfile holds the matching
      `vU32` at that register, or
    * the slot is `.i32ConstSym m` and the WASM value is the matching
      `wI32 (UInt32.ofNat m.toNat)` — purely symbolic, no regfile
      dependency, or
    * the slot is `.bufferPtr slot` and the WASM value is the i32
      byte-pointer to that slot's start in linear memory
      (`layout.startAddr slot`). The translator's buffer-typed
      `localGet` produces this; buffer-pattern arms consume it.

    Other SymVal shapes (`scaledIdx`, `bufferAccess`) are addresses
    that depend on register lookups; their encoding arms land
    alongside the corresponding consumer proofs.

    The `layout` parameter ties WASM byte-pointer values to KOps heap
    slots. Every preservation theorem fixes a layout across input and
    output (`layout` is static in the lowered program). -/
def WasmValue.encodes
    (v : WasmValue) (layout : BufferLayout)
    (rf : Quanta.KOps.RegFile) (sv : SymVal) : Prop :=
  match v, sv with
  | .wI32 n, .reg r .u32      => regLookup rf r = some (Quanta.KOps.Value.vU32 n)
  | .wI32 n, .i32ConstSym m   => n = UInt32.ofNat m.toNat
  | .wI32 n, .bufferPtr slot  => n.toNat = layout.startAddr slot
  -- `scaledIdx base scale` represents the byte offset
  -- `(lookup base) * scale`. The Nat-equation form refuses overflow:
  -- `n.toNat = b.toNat * scale` forces `b.toNat * scale < 2^32`.
  -- Future kernel-entry composition theorem will discharge that.
  | .wI32 n, .scaledIdx base scale =>
      ∃ b : UInt32, regLookup rf base = some (Quanta.KOps.Value.vU32 b) ∧
                    n.toNat = b.toNat * scale
  -- `bufferAccess slot base scale` represents the absolute address
  -- `layout.startAddr slot + (lookup base) * scale`. Same Nat-
  -- equation form refuses overflow on the address arithmetic.
  | .wI32 n, .bufferAccess slot base scale =>
      ∃ b : UInt32, regLookup rf base = some (Quanta.KOps.Value.vU32 b) ∧
                    n.toNat = layout.startAddr slot + b.toNat * scale
  | _, _                      => False

/-- Stack refinement: WASM stack and symbolic stack zip element-wise
    through `WasmValue.encodes`. Length-aligned, top-aligned. -/
def StackRefines (layout : BufferLayout)
    (ws : List WasmValue) (svs : List SymVal) (rf : Quanta.KOps.RegFile) : Prop :=
  ws.length = svs.length ∧
  ∀ i, ∀ v, ws.get? i = some v → ∃ sv, svs.get? i = some sv ∧ v.encodes layout rf sv

/-- Locals refinement: every local with a stable register encodes
    through that register, lifted into the symbolic alphabet as
    `.reg r .u32`. Locals not in `localReg` are unconstrained.

    Note: post the `currentReg` field addition, `localReg` plays the
    role of production's `stable_reg`. Every `localSet` keeps it in
    sync via a parallel `Copy { stable, fresh }` op so this
    refinement remains an invariant of the lowered IR. The new
    `CurrentRegRefines` predicate (below) imposes the same shape on
    the per-frame `currentReg` map. -/
def LocalsRefines (layout : BufferLayout)
    (locs : List WasmValue) (lreg : List (Nat × Reg)) (rf : Quanta.KOps.RegFile) : Prop :=
  ∀ i r, lreg.find? (fun p => p.fst = i) = some (i, r) →
    ∀ v, locs.get? i = some v → v.encodes layout rf (SymVal.reg r .u32)

/-- Per-frame current-binding refinement: every local with a
    `currentReg` entry encodes its WASM value through that register
    too. Same shape as `LocalsRefines` — both predicates hold
    simultaneously after a `localSet`, because the lowering emits
    `[.copy fresh src, .copy stable fresh]` keeping both regs in
    lockstep until the frame-close fixup resets `currentReg`.

    Locals NOT in `currentReg` are unconstrained — readers fall back
    to `localReg` (the stable merge anchor). -/
def CurrentRegRefines (layout : BufferLayout)
    (locs : List WasmValue) (creg : List (Nat × Reg)) (rf : Quanta.KOps.RegFile) : Prop :=
  ∀ i r, creg.find? (fun p => p.fst = i) = some (i, r) →
    ∀ v, locs.get? i = some v → v.encodes layout rf (SymVal.reg r .u32)

/-- Freshness invariant: every register the lowering currently holds
    (any reg referenced by any stack SymVal, plus every local stable
    reg) is strictly less than `nextReg`. The currentReg map's
    freshness is captured separately by `FreshCurrent` (Stage 3)
    so the existing `.left / .right` projections on Fresh remain
    stable across the refactor. -/
def Fresh (s : LowerState) : Prop :=
  (∀ sv ∈ s.stack, ∀ r ∈ sv.regs, r < s.nextReg) ∧
  (∀ ir ∈ s.localReg, ir.snd < s.nextReg)

/-- Stage 3 freshness clause for `currentReg`: every per-frame
    current-binding register is strictly less than `nextReg`.
    Stored as a separate field of `Refines` (the 8th field, after
    currentReg) so the existing `Fresh` projections remain
    backward-compatible. -/
def FreshCurrent (s : LowerState) : Prop :=
  ∀ ir ∈ s.currentReg, ir.snd < s.nextReg

/-- Alias-free invariant: no local's stable register appears anywhere
    in the symbolic stack's reg projection. The Lean translator's
    `localGet`/`localTee` emit Copy ops to fresh registers precisely
    to maintain this — so a subsequent `localSet` writing to a
    stable_reg can't clobber a stack-aliased copy of the old value. -/
def AliasFree (s : LowerState) : Prop :=
  ∀ ir ∈ s.localReg, ∀ sv ∈ s.stack, ir.snd ∉ sv.regs

/-- Injective locals: distinct local indices map to distinct stable
    registers. Maintained by always allocating a fresh `s.nextReg` for
    a brand-new local entry, and never aliasing an existing entry.

    Note: this does NOT extend to `currentReg` because currentReg can
    be reset (entries removed at frame close), and a localSet inside
    a frame ALWAYS allocates a fresh reg above all existing regs —
    so naturally injective. The localReg invariant survives all of
    that. -/
def InjectiveLocals (s : LowerState) : Prop :=
  ∀ p q, p ∈ s.localReg → q ∈ s.localReg → p.fst = q.fst ∨ p.snd ≠ q.snd

/-- Stage 3 disjointness invariant: every register in `currentReg`
    is distinct from every register in `localReg` for non-matching
    indices. Concretely: a `currentReg` entry `(k, r_cur)` and a
    `localReg` entry `(j, q)` with `k ≠ j` satisfy `r_cur ≠ q`.

    Structural truth: the translator only ever adds to `currentReg`
    via `setCurrentReg i fresh` immediately after `s.alloc`, so the
    fresh register is strictly greater than every other live register
    (including all stable_regs). Stable regs for OTHER locals were
    allocated earlier with smaller indices.

    For matching indices (k = j), the currentReg and localReg
    registers MAY differ — currentReg holds the per-frame current
    binding (fresh per localSet) while localReg holds the merge
    anchor (allocated on first set, reused thereafter). -/
def CurrentLocalDisjoint (s : LowerState) : Prop :=
  ∀ (p : Nat × Reg) (q : Nat × Reg),
    p ∈ s.currentReg → q ∈ s.localReg → p.fst ≠ q.fst → p.snd ≠ q.snd

/-- Bundle. The `layout : BufferLayout` parameter is the shared side-
    channel that relates WASM linear memory to the KOps heap; each
    theorem fixes `layout` across input and output (the layout is
    static in the lowered program).

    Stage 3 integration: `currentReg` is the 7th field. Stage 1+2
    added the field to `LowerState` and the `CurrentRegRefines`
    predicate; Stage 3 binds them into the main `Refines` bundle
    because `localGet` now consults `currentReg` first (so its
    correctness depends on the per-frame current binding's encoding).
    Frame-entry states have `currentReg = []`, satisfying the
    invariant trivially. -/
structure Refines (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
                  (layout : BufferLayout) : Prop where
  stk          : StackRefines layout ws.stack s.stack kst.rf
  locs         : LocalsRefines layout ws.locals s.localReg kst.rf
  fresh        : Fresh s
  aliasFree    : AliasFree s
  injLocals    : InjectiveLocals s
  heapRefines  : HeapRefines ws.mem kst.heap layout
  currentReg   : CurrentRegRefines layout ws.locals s.currentReg kst.rf
  freshCurrent : FreshCurrent s
  curLocDisj   : CurrentLocalDisjoint s

-- ════════════════════════════════════════════════════════════════════
-- evalOps composition lemma
--
-- `lowerI32Bin` (and any future op whose lowering chains multiple
-- sub-ops via `commit`) emits an op-list of the form
-- `opsA ++ opsB ++ [op_main]`. `evalOps` short-circuits on `broke`
-- between ops, so chaining `evalOps 0 kst opsA = some kst1` then
-- `evalOps 0 kst1 opsB = some kst2` requires both intermediate states
-- to have `broke = false`. The lemma below packages that as a
-- one-step rewrite.
-- ════════════════════════════════════════════════════════════════════

theorem evalOps_append {fuel : Nat} {s : Quanta.KOps.State}
    {l1 l2 : List KernelOp} {s1 : Quanta.KOps.State}
    (h : evalOps fuel s l1 = some s1) (h_ok : s1.broke = false) :
    evalOps fuel s (l1 ++ l2) = evalOps fuel s1 l2 := by
  induction l1 generalizing s with
  | nil =>
    simp only [evalOps] at h
    rw [List.nil_append, ← (Option.some.injEq _ _).mp h]
  | cons op rest ih =>
    simp only [List.cons_append, evalOps] at h ⊢
    rcases ho : Quanta.KOps.evalOp fuel s op with _ | s_after
    · simp [ho] at h
    · simp only [ho, Option.some_bind, bind, Option.bind] at h ⊢
      by_cases hbroke : s_after.broke = true
      · simp only [if_pos hbroke] at h
        -- h : some s_after = some s1, but s_after.broke = true and s1.broke = false: contradiction.
        have h_eq : s_after = s1 := (Option.some.injEq _ _).mp h
        rw [h_eq] at hbroke
        rw [hbroke] at h_ok
        cases h_ok
      · have hbroke' : s_after.broke = false := by
          cases hb : s_after.broke
          · rfl
          · exact (hbroke hb).elim
        simp only [if_neg hbroke, hbroke'] at h ⊢
        -- Goal: evalOps fuel s_after (rest ++ l2) = evalOps fuel s1 l2.
        -- IH gives this from h : evalOps fuel s_after rest = some s1.
        exact ih h

-- ════════════════════════════════════════════════════════════════════
-- Register-file lemmas
-- ════════════════════════════════════════════════════════════════════

/-- Reading the freshly-written register reads back the value we
    wrote. Closes in one `simp` step. -/
@[simp] theorem regLookup_regWrite_self (rf : Quanta.KOps.RegFile) (r : Reg) (v : Quanta.KOps.Value) :
    regLookup (regWrite rf r v) r = some v := by
  simp [regLookup, regWrite]

/-- Auxiliary: when looking up key `r'` after the filter+find?
    fusion that `simp` produces, the extra ≠-r conjunct is redundant
    given `r' ≠ r` (any element matching key=r' automatically has
    key ≠ r). -/
private theorem find?_pred_eq
    (rf : List (Reg × Quanta.KOps.Value)) (r r' : Reg) (h : r' ≠ r) :
    rf.find? (fun a => !decide (a.fst = r) && decide (a.fst = r')) =
    rf.find? (fun p => decide (p.fst = r')) := by
  induction rf with
  | nil => rfl
  | cons p ps ih =>
    by_cases hpr' : p.fst = r'
    · -- Head matches r'. Then it can't match r (because r' ≠ r).
      have hpr_ne_r : ¬ (p.fst = r) := fun heq => h (hpr'.symm.trans heq)
      have h_ne : ¬ (r' = r) := h
      simp [List.find?, hpr', hpr_ne_r, h_ne]
    · -- Head doesn't match r'. Both sides fall through; IH closes.
      simp [List.find?, hpr', ih]

/-- Writing register `r` doesn't disturb lookups of any other
    register. -/
theorem regLookup_regWrite_of_ne (rf : Quanta.KOps.RegFile) (r r' : Reg) (v : Quanta.KOps.Value)
    (h : r' ≠ r) :
    regLookup (regWrite rf r v) r' = regLookup rf r' := by
  unfold regLookup regWrite
  simp only [List.find?]
  have h_head : ¬ (r = r') := fun heq => h heq.symm
  simp [h_head]
  -- After the simp, `simp` has fused (filter ∘ find?) into a single
  -- `find?` over a conjunctive predicate. The auxiliary lemma shows
  -- that conjunctive predicate is equivalent (under `r' ≠ r`) to the
  -- original key=r' check.
  congr 1
  exact find?_pred_eq rf r r' h

/-- Overwriting the same register collapses: a write of `v1` to `r`
    after a write of `v0` to `r` equals just writing `v1`. The inner
    write's `(r, v0)` head is dropped by the outer write's
    `filter (· ≠ r)`, and `filter` is idempotent on the tail. The
    load-bearing fact behind dead-write absorption (the frame-0
    zero-init: `Const fresh 0` immediately overwritten by
    `Copy fresh src`). -/
theorem regWrite_regWrite_self
    (rf : Quanta.KOps.RegFile) (r : Reg) (v0 v1 : Quanta.KOps.Value) :
    regWrite (regWrite rf r v0) r v1 = regWrite rf r v1 := by
  unfold regWrite
  -- Goal: (r,v1) :: ((r,v0) :: rf.filter(≠r)).filter(≠r)
  --     = (r,v1) :: rf.filter(≠r)
  congr 1
  -- The filter of the cons: the head (r,v0) has fst = r, so the
  -- predicate (·.fst ≠ r) is false on it ⇒ it's dropped; the tail
  -- rf.filter(≠r) re-filtered by (≠r) is itself (idempotence).
  rw [List.filter_cons]
  have h_head : (decide (((r, v0) : Reg × Quanta.KOps.Value).fst ≠ r)) = false := by
    simp
  rw [h_head]
  simp only [Bool.false_eq_true, if_false]
  -- filter idempotence on rf.filter(≠r).
  rw [List.filter_filter]
  congr 1
  funext p
  simp [Bool.and_self]

/-- Dead-write absorption: a `Const f z` immediately followed by a
    `Copy f src` (writing the same register `f`, reading a *different*
    register `src ≠ f`) evaluates the same as the `Copy f src` alone —
    the const's write to `f` is overwritten before any read, and the
    copy's read of `src` is undisturbed because `src ≠ f`. Stated at
    the `evalOps` level so the localSet / localTee preservation proofs
    absorb the frame-0 zero-init op with a single rewrite, leaving
    their `Refines` reasoning (about the final state) untouched.

    `h_ne : src ≠ f` holds in every caller: the copied source is a
    committed operand (`src < s.nextReg`) while `f` is the freshly
    allocated per-set register (`f = s.nextReg`), so `src < f`.

    `h_ok : s.broke = false` lets the `evalOps` cons reach the copy;
    the const step preserves `broke`. -/
theorem evalOps_const_copy_absorb
    {fuel : Nat} {s : Quanta.KOps.State} {f src : Reg}
    {z : Quanta.KOps.ConstValue} {rest : List KernelOp}
    (h_ne : src ≠ f) (h_ok : s.broke = false) :
    evalOps fuel s (.const f z :: .copy f src :: rest)
      = evalOps fuel s (.copy f src :: rest) := by
  -- Reads of `src` agree between s.rf and the const-progressed regfile
  -- (regWrite s.rf f _) because src ≠ f.
  have h_read_eq : regLookup (regWrite s.rf f (Quanta.KOps.evalConst z)) src
      = regLookup s.rf src := regLookup_regWrite_of_ne _ f src _ h_ne
  -- Unfold both sides fully: the const head writes f then (broke kept)
  -- runs the copy; the copy reads src and writes f.
  simp only [evalOps, Quanta.KOps.evalOp, Option.pure_def,
             Option.bind_eq_bind, Option.some_bind, h_ok,
             if_neg (by decide : ¬ (false = true))]
  -- Now: do (copy from sz) on LHS vs do (copy from s) on RHS, with
  -- the const write only affecting f. Case-split on the src read.
  rw [h_read_eq]
  rcases hsrc : regLookup s.rf src with _ | vsrc
  · -- Both copies fail to read ⇒ both `none`.
    simp only [Option.none_bind]
  · -- Both copies write f ↦ vsrc; collapse the double write to f.
    simp only [Option.some_bind, h_ok, if_neg (by decide : ¬ (false = true))]
    rw [regWrite_regWrite_self s.rf f (Quanta.KOps.evalConst z) vsrc]

/-- For any register strictly below `nextReg` and any fresh write to
    `nextReg`, the lookup is preserved. Convenient corollary for the
    per-instr lemmas. -/
theorem regLookup_preserved_of_fresh
    {rf : Quanta.KOps.RegFile} {nr r : Reg} {v : Quanta.KOps.Value}
    (h : r < nr) :
    regLookup (regWrite rf nr v) r = regLookup rf r :=
  regLookup_regWrite_of_ne rf nr r v (Nat.ne_of_lt h)

/-- Encoding is preserved under a fresh-register write, provided every
    reg referenced by the SymVal is strictly below the freshly-written
    register. The single load-bearing lemma every fresh-write per-op
    preservation proof uses to thread `R.stk` / `R.locs` past a
    `regWrite kst.rf s.nextReg _`. -/
theorem WasmValue.encodes_preserved_of_fresh
    {v : WasmValue} {layout : BufferLayout} {rf : Quanta.KOps.RegFile} {sv : SymVal}
    {nr : Reg} {newval : Quanta.KOps.Value}
    (h_lt : ∀ r ∈ sv.regs, r < nr)
    (h : v.encodes layout rf sv) :
    v.encodes layout (regWrite rf nr newval) sv := by
  match v, sv, h with
  | .wI32 n, .reg r .u32, h =>
    have h' : regLookup rf r = some (Quanta.KOps.Value.vU32 n) := h
    have hr_lt : r < nr := h_lt r (by simp [SymVal.regs])
    show regLookup (regWrite rf nr newval) r = some (Quanta.KOps.Value.vU32 n)
    rw [regLookup_preserved_of_fresh hr_lt]
    exact h'
  | .wI32 _, .i32ConstSym _, h =>
    -- i32ConstSym encoding is regfile-independent.
    exact h
  | .wI32 _, .bufferPtr _, h =>
    -- bufferPtr encoding (n.toNat = layout.startAddr slot) is
    -- regfile-independent.
    exact h
  | .wI32 n, .scaledIdx base scale, h =>
    -- Existential ⟨b, regLookup rf base = some (vU32 b), n.toNat = b.toNat * scale⟩.
    -- The base reg lookup lifts past the fresh write because `base ∈ sv.regs`.
    obtain ⟨b, h_lookup, h_eq⟩ := h
    refine ⟨b, ?_, h_eq⟩
    have hb_lt : base < nr := h_lt base (by simp [SymVal.regs])
    rw [regLookup_preserved_of_fresh hb_lt]
    exact h_lookup
  | .wI32 n, .bufferAccess slot base scale, h =>
    obtain ⟨b, h_lookup, h_eq⟩ := h
    refine ⟨b, ?_, h_eq⟩
    have hb_lt : base < nr := h_lt base (by simp [SymVal.regs])
    rw [regLookup_preserved_of_fresh hb_lt]
    exact h_lookup

/-- Encoding is preserved under any register write disjoint from the
    SymVal's reg projection. The general-form companion to
    `encodes_preserved_of_fresh` used by `localSet` / `localTee`
    preservation, where the write target is an existing stable_reg
    (not strictly above all held regs) but is disjoint from the
    stack's regs by `AliasFree`. -/
theorem WasmValue.encodes_preserved_of_disjoint
    {v : WasmValue} {layout : BufferLayout} {rf : Quanta.KOps.RegFile} {sv : SymVal}
    {dst : Reg} {newval : Quanta.KOps.Value}
    (h_disj : dst ∉ sv.regs)
    (h : v.encodes layout rf sv) :
    v.encodes layout (regWrite rf dst newval) sv := by
  match v, sv, h with
  | .wI32 n, .reg r .u32, h =>
    have h' : regLookup rf r = some (Quanta.KOps.Value.vU32 n) := h
    have hr_ne : r ≠ dst := by
      intro h_eq
      apply h_disj
      simp [SymVal.regs, h_eq]
    show regLookup (regWrite rf dst newval) r = some (Quanta.KOps.Value.vU32 n)
    rw [regLookup_regWrite_of_ne rf dst r newval hr_ne]
    exact h'
  | .wI32 _, .i32ConstSym _, h =>
    exact h
  | .wI32 _, .bufferPtr _, h =>
    exact h
  | .wI32 _, .scaledIdx base _, h =>
    obtain ⟨b, h_lookup, h_eq⟩ := h
    refine ⟨b, ?_, h_eq⟩
    have hb_ne : base ≠ dst := by
      intro h_eq2
      apply h_disj
      simp [SymVal.regs, h_eq2]
    rw [regLookup_regWrite_of_ne rf dst base newval hb_ne]
    exact h_lookup
  | .wI32 _, .bufferAccess _ base _, h =>
    obtain ⟨b, h_lookup, h_eq⟩ := h
    refine ⟨b, ?_, h_eq⟩
    have hb_ne : base ≠ dst := by
      intro h_eq2
      apply h_disj
      simp [SymVal.regs, h_eq2]
    rw [regLookup_regWrite_of_ne rf dst base newval hb_ne]
    exact h_lookup

/-- Lift `CurrentRegRefines` past a fresh regWrite at `r` when
    `r` is above every register currently bound in `currentReg`.
    Mirrors `LocalsRefines_preserved_fresh`'s pattern for the
    stable-reg map. Used after `alloc` + `regWrite` in proofs where
    the new state's `kst.rf` differs from the old by a single
    write at a fresh register. -/
theorem CurrentRegRefines_preserved_fresh
    {layout : BufferLayout} {locs : List WasmValue}
    {creg : List (Nat × Reg)} {rf : Quanta.KOps.RegFile}
    (h : CurrentRegRefines layout locs creg rf)
    {dst : Reg} (h_fresh : ∀ ir ∈ creg, ir.snd < dst) (v : Quanta.KOps.Value) :
    CurrentRegRefines layout locs creg (regWrite rf dst v) := by
  intro i r_cur hfind v_w hloc
  have henc := h i r_cur hfind v_w hloc
  have hpair : (i, r_cur) ∈ creg := List.mem_of_find?_eq_some hfind
  have h_r_cur_lt : r_cur < dst := h_fresh (i, r_cur) hpair
  apply WasmValue.encodes_preserved_of_fresh _ henc
  intro r_in hr_in
  simp [SymVal.regs] at hr_in
  rw [hr_in]
  exact h_r_cur_lt

/-- Encoding is preserved under any regfile transition that agrees on
    the SymVal's regs. The full-strength companion to
    `encodes_preserved_of_fresh` / `encodes_preserved_of_disjoint`,
    used when a sequence of ops (e.g. `commit svb`'s `opsB`) writes
    only to fresh registers — we collapse that into a pointwise
    `regLookup` equality on the regs of any older SymVal. -/
theorem WasmValue.encodes_preserved_of_lookup_eq
    {v : WasmValue} {layout : BufferLayout}
    {rf rf' : Quanta.KOps.RegFile} {sv : SymVal}
    (h_lookup : ∀ r ∈ sv.regs, regLookup rf' r = regLookup rf r)
    (h : v.encodes layout rf sv) :
    v.encodes layout rf' sv := by
  match v, sv, h with
  | .wI32 n, .reg r .u32, h =>
    have h' : regLookup rf r = some (Quanta.KOps.Value.vU32 n) := h
    have h_eq := h_lookup r (by simp [SymVal.regs])
    show regLookup rf' r = some (Quanta.KOps.Value.vU32 n)
    rw [h_eq]; exact h'
  | .wI32 _, .i32ConstSym _, h => exact h
  | .wI32 _, .bufferPtr _, h => exact h
  | .wI32 _, .scaledIdx base _, h =>
    obtain ⟨b, h_lup, h_eq⟩ := h
    refine ⟨b, ?_, h_eq⟩
    have h_lup_eq := h_lookup base (by simp [SymVal.regs])
    rw [h_lup_eq]; exact h_lup
  | .wI32 _, .bufferAccess _ base _, h =>
    obtain ⟨b, h_lup, h_eq⟩ := h
    refine ⟨b, ?_, h_eq⟩
    have h_lup_eq := h_lookup base (by simp [SymVal.regs])
    rw [h_lup_eq]; exact h_lup

/-- Inverting a `wI32`-encoding-via-`.reg`: forces the scalar type to
    `.u32` and exposes the underlying regfile lookup. Used by the
    `localSet` / `localTee` proofs to extract the encoding constraint
    after `R.stk.right 0` returns a `.reg src tysrc` SymVal. -/
theorem WasmValue.encodes_wI32_reg_inv
    {n : UInt32} {layout : BufferLayout} {rf : Quanta.KOps.RegFile}
    {r : Reg} {ty : Quanta.KOps.Scalar}
    (h : (WasmValue.wI32 n).encodes layout rf (.reg r ty)) :
    ty = .u32 ∧ regLookup rf r = some (Quanta.KOps.Value.vU32 n) := by
  match ty, h with
  | .u32, h => exact ⟨rfl, h⟩

/-- Stronger inversion: from `v.encodes rf (.reg r ty)` *non-False*,
    deduce `v = wI32 n` AND `ty = .u32` AND the regfile lookup. The
    only `WasmValue.encodes` arm with non-False content matches the
    `(wI32, reg _ .u32)` shape; every other case is `False`, so a
    proof of the non-False predicate forces the value/type shape. -/
theorem WasmValue.encodes_reg_shape
    {v : WasmValue} {layout : BufferLayout} {rf : Quanta.KOps.RegFile}
    {r : Reg} {ty : Quanta.KOps.Scalar}
    (h : v.encodes layout rf (.reg r ty)) :
    ∃ n, v = .wI32 n ∧ ty = .u32 ∧ regLookup rf r = some (Quanta.KOps.Value.vU32 n) := by
  match v, ty, h with
  | .wI32 n, .u32, h => exact ⟨n, rfl, rfl, h⟩

/-- Inversion for `i32ConstSym` encoding. -/
theorem WasmValue.encodes_i32ConstSym_inv
    {v : WasmValue} {layout : BufferLayout} {rf : Quanta.KOps.RegFile} {n : Int}
    (h : v.encodes layout rf (.i32ConstSym n)) :
    v = .wI32 (UInt32.ofNat n.toNat) := by
  match v, h with
  | .wI32 m, h =>
    have h' : m = UInt32.ofNat n.toNat := h
    rw [h']

open Quanta.KOps (vU32) in
/-- `commit` correctness. Materializing a `SymVal` to a real `Reg`
    preserves both the encoding (the resulting reg encodes the same
    `wI32` value the SymVal did) and the `Refines` bundle (any state
    bumps are fresh, so existing stack/locals encodings lift through
    `encodes_preserved_of_fresh`).

    Used by every per-op preservation theorem whose translator arm
    consumes a popped `SymVal`: rather than case-splitting on the
    SymVal shape, the theorem applies `commit_correct` to thread the
    encoding into a real reg + propagate the `Refines` bundle.

    Also exposes three structural facts every chained-commit caller
    needs — `kst'` agrees with `kst` on every reg strictly below
    `s.nextReg`, `s'.nextReg ≥ s.nextReg`, and the materialized reg
    `r` is `< s'.nextReg`. The first lets a second commit's encoding
    hypothesis lift through the first commit's regfile changes via
    `encodes_preserved_of_lookup_eq`. The third is what justifies
    re-using `r` as a binop operand after the second commit, since
    the freshness invariant on `s'` only constrains regs strictly
    less than `s'.nextReg`.

    The `h_regs_lt` precondition is naturally satisfied by every
    caller: the popped SymVal comes from the stack, and `R.fresh.left`
    bounds every reg in any stack SymVal by `s.nextReg`. -/
theorem commit_correct
    {ws : WasmState} {s : LowerState} {kst : Quanta.KOps.State}
    {layout : BufferLayout} (R : Refines ws s kst layout)
    {sv : SymVal} (h_regs_lt : ∀ r ∈ sv.regs, r < s.nextReg)
    {r : Reg} {s' : LowerState} {ops : List KernelOp}
    (h_commit : s.commit sv = some (r, s', ops))
    {v_w : WasmValue} (h_enc : v_w.encodes layout kst.rf sv) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧
            Refines ws s' kst' layout ∧
            v_w.encodes layout kst'.rf (.reg r .u32) ∧
            (∀ r', r' < s.nextReg → regLookup kst'.rf r' = regLookup kst.rf r') ∧
            s.nextReg ≤ s'.nextReg ∧
            r < s'.nextReg := by
  match sv, h_commit with
  | .reg rsv tysv, h_commit =>
    -- commit returns (rsv, s, []).
    simp [LowerState.commit] at h_commit
    obtain ⟨hr, hs', hops⟩ := h_commit
    have hrsv_lt : rsv < s.nextReg := h_regs_lt rsv (by simp [SymVal.regs])
    refine ⟨kst, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · subst hops; simp [evalOps]
    · subst hs'; exact R
    · subst hr
      obtain ⟨n, hv_eq, htysv, h_lookup⟩ := WasmValue.encodes_reg_shape h_enc
      subst hv_eq htysv
      -- Goal: (wI32 n).encodes kst.rf (.reg rsv .u32) = regLookup kst.rf rsv = some (vU32 n).
      simpa [WasmValue.encodes] using h_lookup
    · intro r' _; rfl
    · subst hs'; exact Nat.le_refl _
    · subst hs' hr; exact hrsv_lt
  | .i32ConstSym n, h_commit =>
    simp [LowerState.commit, LowerState.alloc] at h_commit
    obtain ⟨hr, hs', hops⟩ := h_commit
    refine ⟨{ kst with rf := regWrite kst.rf s.nextReg
                          (vU32 (UInt32.ofNat n.toNat)) }, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · subst hops
      simp [evalOps, Quanta.KOps.evalOp, Quanta.KOps.evalConst]
    · subst hs'
      refine ⟨?_, ?_, ?_, R.aliasFree, R.injLocals, R.heapRefines, ?_, ?_, R.curLocDisj⟩
      · -- StackRefines: stack unchanged; lift each entry past the fresh write.
        refine ⟨R.stk.left, ?_⟩
        intro i v hv
        obtain ⟨svi, hsvi_get, henc⟩ := R.stk.right i v hv
        refine ⟨svi, hsvi_get, ?_⟩
        have hsvi_in : svi ∈ s.stack := List.mem_of_get? hsvi_get
        apply WasmValue.encodes_preserved_of_fresh _ henc
        intro r' hr'
        exact R.fresh.left svi hsvi_in r' hr'
      · -- LocalsRefines: localReg unchanged; lift past the fresh write.
        intro i r' hfind v hv
        have hpair : (i, r') ∈ s.localReg := List.mem_of_find?_eq_some hfind
        have hr'_lt : r' < s.nextReg := R.fresh.right (i, r') hpair
        have henc := R.locs i r' hfind v hv
        apply WasmValue.encodes_preserved_of_fresh _ henc
        intro r'' hr''_in
        simp [SymVal.regs] at hr''_in
        subst hr''_in; exact hr'_lt
      · -- Fresh: nextReg bumps by 1; stack/locals unchanged.
        refine ⟨?_, ?_⟩
        · intro sv' hsv' r'' hr''
          exact Nat.lt_succ_of_lt (R.fresh.left sv' hsv' r'' hr'')
        · intro ir hir
          exact Nat.lt_succ_of_lt (R.fresh.right ir hir)
      · -- CurrentRegRefines: currentReg unchanged; lift past fresh write.
        exact CurrentRegRefines_preserved_fresh R.currentReg R.freshCurrent _
      · -- FreshCurrent: nextReg bumps by 1; currentReg unchanged.
        intro ir hir
        exact Nat.lt_succ_of_lt (R.freshCurrent ir hir)
    · -- v_w encodes via .reg s.nextReg .u32 in the new regfile.
      subst hr
      have hv_eq := WasmValue.encodes_i32ConstSym_inv h_enc
      subst hv_eq
      simp [WasmValue.encodes, regLookup_regWrite_self]
      rfl
    · -- Lookups below s.nextReg are preserved through the single fresh write.
      intro r' hr'_lt
      exact regLookup_preserved_of_fresh hr'_lt
    · subst hs'; exact Nat.le_succ _
    · subst hs' hr; exact Nat.lt_succ_self _
  | .bufferPtr _,        h_commit => simp [LowerState.commit] at h_commit
  | .scaledIdx _ _,      h_commit => simp [LowerState.commit] at h_commit
  | .bufferAccess _ _ _, h_commit => simp [LowerState.commit] at h_commit

/-- `commit` only bumps `nextReg`; `stack`, `localReg`, `localTy`
    are preserved. The `.reg` arm is identity, the `.i32ConstSym` arm
    only calls `alloc` (which bumps `nextReg`). The address SymVals
    refuse, so unreachable. -/
theorem commit_only_bumps_nextReg {s : LowerState} {sv : SymVal}
    {r : Reg} {s' : LowerState} {ops : List KernelOp}
    (h : s.commit sv = some (r, s', ops)) :
    s' = { s with nextReg := s'.nextReg } := by
  match sv, h with
  | .reg _ _, h =>
    simp [LowerState.commit] at h
    obtain ⟨_, hs', _⟩ := h
    rw [← hs']
  | .i32ConstSym _, h =>
    simp [LowerState.commit, LowerState.alloc] at h
    obtain ⟨_, hs', _⟩ := h
    rw [← hs']

/-- `commit` preserves the stack. Corollary of
    `commit_only_bumps_nextReg`. -/
theorem commit_preserves_stack {s : LowerState} {sv : SymVal}
    {r : Reg} {s' : LowerState} {ops : List KernelOp}
    (h : s.commit sv = some (r, s', ops)) :
    s'.stack = s.stack := by
  rw [commit_only_bumps_nextReg h]

/-- `commit` preserves both local maps. Corollary of
    `commit_only_bumps_nextReg`. -/
theorem commit_preserves_locals {s : LowerState} {sv : SymVal}
    {r : Reg} {s' : LowerState} {ops : List KernelOp}
    (h : s.commit sv = some (r, s', ops)) :
    s'.localReg = s.localReg ∧ s'.localTy = s.localTy := by
  rw [commit_only_bumps_nextReg h]
  exact ⟨rfl, rfl⟩

/-- `commit` preserves the `bufferSlots` map. Corollary of
    `commit_only_bumps_nextReg`. Used by the shape lemmas
    (`lowerI32Bin_some_shape`, `lowerI32Cmp_some_shape`) to discharge
    the `bufferSlots` field of the post-state literal. -/
theorem commit_preserves_bufferSlots {s : LowerState} {sv : SymVal}
    {r : Reg} {s' : LowerState} {ops : List KernelOp}
    (h : s.commit sv = some (r, s', ops)) :
    s'.bufferSlots = s.bufferSlots := by
  rw [commit_only_bumps_nextReg h]

/-- `commit` preserves the `currentReg` map (added when the
    structure grew the per-frame current-binding field). Same
    rationale as the bufferSlots variant — commit only bumps
    nextReg. -/
theorem commit_preserves_currentReg {s : LowerState} {sv : SymVal}
    {r : Reg} {s' : LowerState} {ops : List KernelOp}
    (h : s.commit sv = some (r, s', ops)) :
    s'.currentReg = s.currentReg := by
  rw [commit_only_bumps_nextReg h]

/-- The KOps `evalOps` of a commit's emitted op list preserves the
    `broke` flag — `.reg` emits no ops, `.i32ConstSym` emits a single
    `.const` write that doesn't touch `broke`. The address SymVals
    refuse, so unreachable. Used to chain `evalOps_append` across
    multiple commits in a binop preservation proof — each
    intermediate state inherits the input's `broke = false`. -/
theorem commit_preserves_broke {s : LowerState} {sv : SymVal}
    {r : Reg} {s' : LowerState} {ops : List KernelOp}
    (h_commit : s.commit sv = some (r, s', ops))
    {kst kst' : Quanta.KOps.State}
    (h_eval : evalOps 0 kst ops = some kst') :
    kst'.broke = kst.broke := by
  match sv, h_commit with
  | .reg _ _, h_commit =>
    simp [LowerState.commit] at h_commit
    obtain ⟨_, _, hops⟩ := h_commit
    rw [hops] at h_eval
    simp [evalOps] at h_eval
    rw [← h_eval]
  | .i32ConstSym _, h_commit =>
    simp [LowerState.commit, LowerState.alloc] at h_commit
    obtain ⟨_, _, hops⟩ := h_commit
    -- simp leaves `hops` in `[const …] = ops` form (RHS = ops); rw ←
    -- substitutes `ops` with the literal list.
    rw [← hops] at h_eval
    simp [evalOps, Quanta.KOps.evalOp] at h_eval
    rw [← h_eval]

/-- For an association list keyed by `Nat`, `find?` over the
    post-`setLocalReg` list (which prepends a new entry and filters
    out any older ones with the same key) behaves like `find?` on
    the original list for queries `k ≠ i`. The `filter` only drops
    entries with `p.fst = i`, which by `k ≠ i` are irrelevant. Used
    by the `LocalsRefines` arm of `preservation_localSet` to fold the
    new-state `find?` back into the prior `R.locs` instance. -/
private theorem find?_filter_fst_ne_of_ne {α : Type}
    (xs : List (Nat × α)) (k i : Nat) (hki : k ≠ i) :
    (xs.filter (fun p => !decide (p.fst = i))).find? (fun p => decide (p.fst = k)) =
    xs.find? (fun p => decide (p.fst = k)) := by
  -- `List.find?_filter` fuses filter+find? into a single find? with a
  -- conjunctive predicate. The conjunctive predicate is pointwise
  -- equivalent to `a.fst = k` because `a.fst = k → a.fst ≠ i` under
  -- `hki : k ≠ i`.
  rw [List.find?_filter]
  congr 1
  funext a
  by_cases hak : a.fst = k
  · have hai : a.fst ≠ i := fun heq => hki (hak.symm.trans heq)
    simp [hai, hak, hki]
  · simp [hak]

/-- Combined helper: `find?` over `(i, a) :: filter (≠ i) xs` for query
    `k ≠ i` reduces to `find?` over the original `xs`. Bundles the head-
    drop (cons doesn't match) and filter-collapse together so the proof
    of `preservation_localSet` can discharge the `LocalsRefines k ≠ i`
    case in a single rewrite. -/
private theorem find?_setLocalReg_ne {α : Type}
    (xs : List (Nat × α)) (i k : Nat) (a : α) (hki : k ≠ i) :
    ((i, a) :: xs.filter (fun p => !decide (p.fst = i))).find?
        (fun p => decide (p.fst = k)) =
    xs.find? (fun p => decide (p.fst = k)) := by
  rw [List.find?_cons]
  have h_dec : decide ((i, a).fst = k) = false := by
    apply decide_eq_false; exact fun h => hki h.symm
  rw [h_dec]
  exact find?_filter_fst_ne_of_ne xs k i hki

-- ════════════════════════════════════════════════════════════════════
-- Per-instruction preservation — slice 1 closed proofs
-- ════════════════════════════════════════════════════════════════════

/-- `nop` preservation. Both sides leave state untouched and emit
    nothing. -/
theorem preservation_nop (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws .nop = some ws')
    (hl : lowerInstr s .nop = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  simp [evalInstr] at hw
  simp [lowerInstr] at hl
  obtain ⟨hs_eq, hops_eq⟩ := hl
  refine ⟨kst, ?_, ?_⟩
  · subst hops_eq
    simp [evalOps]
  · subst hw hs_eq
    exact R

/-- `wreturn` preservation. Lowering emits no ops; WASM sets
    `halted := true`. The KOps register file and the lowering's
    locals/stack are untouched, so the refinement bundle survives —
    the WASM state's `halted` field isn't constrained by `Refines`. -/
theorem preservation_return (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws .wreturn = some ws')
    (hl : lowerInstr s .wreturn = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  simp [evalInstr] at hw
  simp [lowerInstr] at hl
  obtain ⟨hs_eq, hops_eq⟩ := hl
  refine ⟨kst, ?_, ?_⟩
  · subst hops_eq
    simp [evalOps]
  · subst hs_eq
    refine ⟨?_, ?_, R.fresh, R.aliasFree, R.injLocals, ?_, ?_, R.freshCurrent, R.curLocDisj⟩
    · have : ws'.stack = ws.stack := by rw [← hw]
      rw [this]; exact R.stk
    · have : ws'.locals = ws.locals := by rw [← hw]
      rw [this]; exact R.locs
    · -- HeapRefines: ws'.mem = ws.mem (return doesn't touch memory).
      have : ws'.mem = ws.mem := by rw [← hw]
      rw [this]; exact R.heapRefines
    · -- CurrentRegRefines: ws'.locals = ws.locals, s.currentReg unchanged.
      have : ws'.locals = ws.locals := by rw [← hw]
      rw [this]; exact R.currentReg

/-- `drop` preservation. Both sides pop one value; lowering emits no
    IR. KOps state untouched. -/
theorem preservation_drop (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws .drop = some ws')
    (hl : lowerInstr s .drop = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- Both sides require a non-empty stack.
  rcases hws_stack : ws.stack with _ | ⟨v_w, rest_ws⟩
  · -- WASM pop fails → hw says some.
    simp [evalInstr, WasmState.pop, hws_stack] at hw
  rcases hls_stack : s.stack with _ | ⟨sva, lrest⟩
  · -- Symbolic pop fails → hl says some.
    simp [lowerInstr, LowerState.popSym, hls_stack] at hl
  -- Both succeed: extract the mid-states.
  have h_ws_eq : ws' = { ws with stack := rest_ws } := by
    simp [evalInstr, WasmState.pop, hws_stack] at hw
    exact hw.symm
  have h_s_eq : s' = { nextReg := s.nextReg, stack := lrest,
                       localReg := s.localReg, localTy := s.localTy,
                       bufferSlots := s.bufferSlots, currentReg := s.currentReg } ∧ ops = [] := by
    simp [lowerInstr, LowerState.popSym, hls_stack] at hl
    exact ⟨hl.1.symm, hl.2⟩
  refine ⟨kst, ?_, ?_⟩
  · rw [h_s_eq.2]; simp [evalOps]
  · -- Refines after the pop. R.stk lifts via index shift; R.locs +
    -- R.heapRefines untouched; R.fresh + R.aliasFree restrict to a
    -- suffix of the original stack.
    rw [h_ws_eq, h_s_eq.1]
    have h_rest_lrest_len : rest_ws.length = lrest.length := by
      have hl_orig := R.stk.left
      rw [hws_stack, hls_stack] at hl_orig
      simpa using hl_orig
    refine ⟨⟨h_rest_lrest_len, ?_⟩, R.locs, ?_, ?_, R.injLocals, R.heapRefines,
            R.currentReg, R.freshCurrent, R.curLocDisj⟩
    · intro k v hv
      have hrest_get : ws.stack.get? (k + 1) = some v := by
        rw [hws_stack]; simpa using hv
      obtain ⟨svk, hsvk_get, henc⟩ := R.stk.right (k + 1) v hrest_get
      have hlrest_get : lrest.get? k = some svk := by
        have h2 : s.stack.get? (k + 1) = some svk := hsvk_get
        rw [hls_stack] at h2; simpa using h2
      exact ⟨svk, by simpa using hlrest_get, henc⟩
    · refine ⟨?_, R.fresh.right⟩
      intro sv hsv r hr
      have hsv_in : sv ∈ s.stack := by
        rw [hls_stack]; exact List.mem_cons_of_mem _ hsv
      exact R.fresh.left sv hsv_in r hr
    · intro ir hir sv hsv
      have hsv_in : sv ∈ s.stack := by
        rw [hls_stack]; exact List.mem_cons_of_mem _ hsv
      exact R.aliasFree ir hir sv hsv_in

-- ════════════════════════════════════════════════════════════════════
-- Slice 2: i32 constants + local reads
-- ════════════════════════════════════════════════════════════════════

open Quanta.KOps (vU32) in
/-- `i32.const n` preservation. Lowering allocates `s.nextReg`, emits
    `.const s.nextReg (.u32 …)`, pushes the register; WASM pushes
    `wI32 (UInt32.ofNat n.toNat)`. The fresh write doesn't disturb any
    register the prior `Refines` constrained, because every such
    register is `< s.nextReg` by `Fresh`. -/
theorem preservation_i32Const (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout)
    (n : Int)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws (.i32Const n) = some ws')
    (hl : lowerInstr s (.i32Const n) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  simp [evalInstr, WasmState.push] at hw
  simp [lowerInstr] at hl
  obtain ⟨hs_eq, hops_eq⟩ := hl
  -- New shape: ops = [] (no IR emitted), s'.stack = .i32ConstSym n :: s.stack,
  -- s'.nextReg unchanged. kst' = kst (regfile untouched).
  refine ⟨kst, ?_, ?_⟩
  · subst hops_eq; simp [evalOps]
  · subst hw hs_eq
    refine ⟨?_, ?_, ?_, ?_, ?_, R.heapRefines, R.currentReg, R.freshCurrent, R.curLocDisj⟩
    · -- StackRefines: top is .i32ConstSym n encoding wI32 (UInt32.ofNat n.toNat);
      -- below entries are unchanged from the old stack with kst.rf unchanged.
      refine ⟨by simp [R.stk.left], ?_⟩
      intro i v hv
      cases i with
      | zero =>
        simp at hv
        refine ⟨SymVal.i32ConstSym n, by simp, ?_⟩
        subst hv
        simp [WasmValue.encodes]
      | succ k =>
        have hwsk : ws.stack.get? k = some v := by simpa using hv
        obtain ⟨svk, hsvk_get, henc⟩ := R.stk.right k v hwsk
        exact ⟨svk, by simpa using hsvk_get, henc⟩
    · -- LocalsRefines: regfile unchanged, localReg unchanged.
      exact R.locs
    · -- Fresh: nextReg unchanged; new top is .i32ConstSym n with regs = [].
      refine ⟨?_, R.fresh.right⟩
      intro sv hsv r' hr'
      simp at hsv
      rcases hsv with h_eq | h_in
      · subst h_eq; simp [SymVal.regs] at hr'
      · exact R.fresh.left sv h_in r' hr'
    · -- AliasFree: new top has empty regs ⇒ trivially disjoint.
      intro ir hir sv hsv
      simp at hsv
      rcases hsv with h_eq | h_in
      · subst h_eq; simp [SymVal.regs]
      · exact R.aliasFree ir hir sv h_in
    · -- InjectiveLocals: localReg unchanged.
      exact R.injLocals

open Quanta.KOps (vU32) in
/-- Refines bundle for the post-localGet state, given the source
    register is in localReg (currentReg miss). Used by both the
    currentReg-miss arm of `preservation_localGet` and (after a
    register-equality bridge) the currentReg-hit arm. -/
theorem localGet_post_refines_via_localReg
    {ws : WasmState} {s : LowerState} {kst : Quanta.KOps.State}
    {layout : BufferLayout} (R : Refines ws s kst layout)
    {i : Nat} {v : WasmValue} {nv : UInt32} {entry_snd : Reg}
    (hloc : ws.locals.get? i = some v)
    (_hfind' : s.localReg.find? (fun p => p.fst = i) = some (i, entry_snd))
    (_h_lookup : regLookup kst.rf entry_snd = some (Quanta.KOps.Value.vU32 nv))
    (h_v_eq : v = .wI32 nv := by trivial) :
    Refines { ws with stack := v :: ws.stack }
            { nextReg := s.nextReg + 1,
              stack := SymVal.reg s.nextReg .u32 :: s.stack,
              localReg := s.localReg, localTy := s.localTy,
              bufferSlots := s.bufferSlots, currentReg := s.currentReg }
            { kst with rf := regWrite kst.rf s.nextReg
                                (Quanta.KOps.Value.vU32 nv) }
            layout := by
  subst h_v_eq
  refine ⟨?_, ?_, ?_, ?_, ?_, R.heapRefines, ?_, ?_, R.curLocDisj⟩
  · -- StackRefines
    refine ⟨by simp [R.stk.left], ?_⟩
    intro j vj hvj
    cases j with
    | zero =>
      simp at hvj
      refine ⟨SymVal.reg s.nextReg .u32, by simp, ?_⟩
      subst hvj
      simp [WasmValue.encodes]
    | succ k =>
      have hwsk : ws.stack.get? k = some vj := by simpa using hvj
      obtain ⟨svk, hsvk, henc⟩ := R.stk.right k vj hwsk
      refine ⟨svk, by simpa using hsvk, ?_⟩
      have hsvk_in : svk ∈ s.stack := List.mem_of_get? hsvk
      apply WasmValue.encodes_preserved_of_fresh _ henc
      intro r' hr'_in
      exact R.fresh.left svk hsvk_in r' hr'_in
  · -- LocalsRefines
    intro k r hk_find vk hvk
    have hpair : (k, r) ∈ s.localReg := List.mem_of_find?_eq_some hk_find
    have hr_lt : r < s.nextReg := R.fresh.right (k, r) hpair
    have henc' := R.locs k r hk_find vk hvk
    apply WasmValue.encodes_preserved_of_fresh _ henc'
    intro r' hr'_in
    simp [SymVal.regs] at hr'_in
    subst hr'_in; exact hr_lt
  · -- Fresh
    refine ⟨?_, ?_⟩
    · intro sv hsv r' hr'
      simp at hsv
      rcases hsv with h_eq | h_in
      · subst h_eq
        simp [SymVal.regs] at hr'
        subst hr'; exact Nat.lt_succ_self _
      · exact Nat.lt_succ_of_lt (R.fresh.left sv h_in r' hr')
    · intro ir hir
      exact Nat.lt_succ_of_lt (R.fresh.right ir hir)
  · -- AliasFree
    intro ir hir sv hsv
    have hir_lt : ir.snd < s.nextReg := R.fresh.right ir hir
    simp at hsv
    rcases hsv with h_eq | h_in
    · subst h_eq
      simp [SymVal.regs]
      exact Nat.ne_of_lt hir_lt
    · exact R.aliasFree ir hir sv h_in
  · -- InjectiveLocals
    exact R.injLocals
  · -- CurrentRegRefines: currentReg unchanged; lift past fresh write.
    exact CurrentRegRefines_preserved_fresh R.currentReg R.freshCurrent _
  · -- FreshCurrent: nextReg bumps by 1; currentReg unchanged.
    intro ir hir
    exact Nat.lt_succ_of_lt (R.freshCurrent ir hir)

open Quanta.KOps (vU32) in
/-- Companion to `localGet_post_refines_via_localReg` for the
    currentReg-hit arm. Structurally identical except the source-reg
    encoding witness comes from `R.currentReg` instead of `R.locs`. -/
theorem localGet_post_refines_via_currentReg
    {ws : WasmState} {s : LowerState} {kst : Quanta.KOps.State}
    {layout : BufferLayout} (R : Refines ws s kst layout)
    {i : Nat} {v : WasmValue} {nv : UInt32} {cur_snd : Reg}
    (hloc : ws.locals.get? i = some v)
    (_hfind' : s.currentReg.find? (fun p => p.fst = i) = some (i, cur_snd))
    (_h_lookup : regLookup kst.rf cur_snd = some (Quanta.KOps.Value.vU32 nv))
    (h_v_eq : v = .wI32 nv := by trivial) :
    Refines { ws with stack := v :: ws.stack }
            { nextReg := s.nextReg + 1,
              stack := SymVal.reg s.nextReg .u32 :: s.stack,
              localReg := s.localReg, localTy := s.localTy,
              bufferSlots := s.bufferSlots, currentReg := s.currentReg }
            { kst with rf := regWrite kst.rf s.nextReg
                                (Quanta.KOps.Value.vU32 nv) }
            layout := by
  subst h_v_eq
  refine ⟨?_, ?_, ?_, ?_, ?_, R.heapRefines, ?_, ?_, R.curLocDisj⟩
  · refine ⟨by simp [R.stk.left], ?_⟩
    intro j vj hvj
    cases j with
    | zero =>
      simp at hvj
      refine ⟨SymVal.reg s.nextReg .u32, by simp, ?_⟩
      subst hvj
      simp [WasmValue.encodes]
    | succ k =>
      have hwsk : ws.stack.get? k = some vj := by simpa using hvj
      obtain ⟨svk, hsvk, henc⟩ := R.stk.right k vj hwsk
      refine ⟨svk, by simpa using hsvk, ?_⟩
      have hsvk_in : svk ∈ s.stack := List.mem_of_get? hsvk
      apply WasmValue.encodes_preserved_of_fresh _ henc
      intro r' hr'_in
      exact R.fresh.left svk hsvk_in r' hr'_in
  · intro k r hk_find vk hvk
    have hpair : (k, r) ∈ s.localReg := List.mem_of_find?_eq_some hk_find
    have hr_lt : r < s.nextReg := R.fresh.right (k, r) hpair
    have henc' := R.locs k r hk_find vk hvk
    apply WasmValue.encodes_preserved_of_fresh _ henc'
    intro r' hr'_in
    simp [SymVal.regs] at hr'_in
    subst hr'_in; exact hr_lt
  · refine ⟨?_, ?_⟩
    · intro sv hsv r' hr'
      simp at hsv
      rcases hsv with h_eq | h_in
      · subst h_eq
        simp [SymVal.regs] at hr'
        subst hr'; exact Nat.lt_succ_self _
      · exact Nat.lt_succ_of_lt (R.fresh.left sv h_in r' hr')
    · intro ir hir
      exact Nat.lt_succ_of_lt (R.fresh.right ir hir)
  · intro ir hir sv hsv
    have hir_lt : ir.snd < s.nextReg := R.fresh.right ir hir
    simp at hsv
    rcases hsv with h_eq | h_in
    · subst h_eq
      simp [SymVal.regs]
      exact Nat.ne_of_lt hir_lt
    · exact R.aliasFree ir hir sv h_in
  · exact R.injLocals
  · exact CurrentRegRefines_preserved_fresh R.currentReg R.freshCurrent _
  · intro ir hir
    exact Nat.lt_succ_of_lt (R.freshCurrent ir hir)

open Quanta.KOps (vU32) in
/-- `local.get i` preservation. Stage 3: source = currentReg-then-localReg.
    Lowering allocates a fresh register, emits Copy from the source,
    pushes the fresh reg. WASM pushes `locals[i]`. -/
theorem preservation_localGet (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout)
    (i : Nat)
    (h_no_buf : s.lookupBufferSlot i = none)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws (.localGet i) = some ws')
    (hl : lowerInstr s (.localGet i) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  simp only [evalInstr, WasmState.getLocal, WasmState.push,
             Option.bind_eq_bind, Option.bind, pure] at hw
  match hloc : ws.locals.get? i, hw with
  | some v, hw =>
    -- Stage 3 lowering: source = (s.lookupCurrentReg i).orElse (s.lookupLocal i)
    -- Then alloc fresh, push fresh, emit [.copy fresh source].
    simp only [lowerInstr, LowerState.lookupLocal, LowerState.lookupCurrentReg,
               LowerState.alloc, LowerState.push, Option.bind_eq_bind,
               Option.bind, pure, h_no_buf] at hl
    -- Resolve source via the orElse: either currentReg has (i, r_cur) or
    -- localReg has (i, r_stable). If neither, hl reduces to none.
    rcases hcurfind : s.currentReg.find? (fun p => p.fst = i) with _ | cur_entry
    · -- currentReg miss → fall back to localReg.
      simp [hcurfind, Option.orElse] at hl
      rcases hregfind : s.localReg.find? (fun p => p.fst = i) with _ | entry
      · simp [hregfind] at hl
      · simp [hregfind] at hl
        obtain ⟨hs_eq, hops_eq⟩ := hl
        simp at hw
        have hki : entry.fst = i := by
          have := List.find?_some hregfind
          simpa using this
        have hfind' : s.localReg.find? (fun p => p.fst = i) = some (i, entry.snd) := by
          rcases entry with ⟨ek, er⟩
          simp at hki; subst hki
          exact hregfind
        have henc_local := R.locs i entry.snd hfind' v hloc
        cases v with
        | wI32 nv =>
          simp only [WasmValue.encodes] at henc_local
          refine ⟨{ kst with rf := regWrite kst.rf s.nextReg
                                  (Quanta.KOps.Value.vU32 nv) }, ?_, ?_⟩
          · subst hops_eq
            simp [evalOps, Quanta.KOps.evalOp, henc_local]
          · subst hs_eq; subst hw
            exact localGet_post_refines_via_localReg R hloc hfind' henc_local
        | _ =>
          unfold WasmValue.encodes at henc_local
          exact henc_local.elim
    · -- currentReg hit → source is cur_entry.snd.
      simp [hcurfind, Option.orElse] at hl
      obtain ⟨hs_eq, hops_eq⟩ := hl
      simp at hw
      have hki : cur_entry.fst = i := by
        have := List.find?_some hcurfind
        simpa using this
      have hfind' : s.currentReg.find? (fun p => p.fst = i)
                      = some (i, cur_entry.snd) := by
        rcases cur_entry with ⟨ek, er⟩
        simp at hki; subst hki
        exact hcurfind
      have henc_cur := R.currentReg i cur_entry.snd hfind' v hloc
      cases v with
      | wI32 nv =>
        simp only [WasmValue.encodes] at henc_cur
        refine ⟨{ kst with rf := regWrite kst.rf s.nextReg
                                (Quanta.KOps.Value.vU32 nv) }, ?_, ?_⟩
        · subst hops_eq
          simp [evalOps, Quanta.KOps.evalOp, henc_cur]
        · subst hs_eq; subst hw
          exact localGet_post_refines_via_currentReg R hloc hfind' henc_cur
      | _ =>
        unfold WasmValue.encodes at henc_cur
        exact henc_cur.elim
  | none, hw => simp [hloc] at hw

/-- `local.get i` preservation for a buffer-typed local. The
    translator's buffer-slot fast-path pushes `SymVal.bufferPtr slot`
    symbolically — no register is allocated, no IR is emitted. The
    WASM-side push of `locals[i]` (an i32 byte-pointer to the buffer)
    encodes via the new `bufferPtr` arm of `WasmValue.encodes`,
    provided the WASM value at `locals[i]` matches the layout's
    `startAddr` for `slot`.

    The `h_loc_buf` precondition captures that match. It is the per-
    call obligation a future top-level composition theorem will
    discharge from the kernel-entry layout (every buffer-typed
    parameter local is initialized to its slot's start address). -/
theorem preservation_localGet_bufferSlot
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout)
    (i : Nat) (slot : Nat)
    (h_buf : s.lookupBufferSlot i = some slot)
    (h_loc_buf : ∀ v, ws.locals.get? i = some v →
      ∃ n : UInt32, v = .wI32 n ∧ n.toNat = layout.startAddr slot)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws (.localGet i) = some ws')
    (hl : lowerInstr s (.localGet i) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  simp only [evalInstr, WasmState.getLocal, WasmState.push,
             Option.bind_eq_bind, Option.bind, pure] at hw
  match hloc : ws.locals.get? i, hw with
  | some v, hw =>
    simp only [lowerInstr, LowerState.pushSym,
               Option.bind_eq_bind, Option.bind, pure, h_buf] at hl
    -- After simp, hl : some ({s with stack := .bufferPtr slot :: s.stack}, []) = some (s', ops).
    -- Extract via Option.some.injEq + Prod.mk.injEq.
    obtain ⟨hs_eq, hops_eq⟩ :=
      Prod.mk.injEq _ _ _ _ |>.mp ((Option.some.injEq _ _).mp hl)
    simp at hw
    obtain ⟨n, hv_eq, h_n_eq⟩ := h_loc_buf v hloc
    -- ops = [], so kst' = kst.
    refine ⟨kst, ?_, ?_⟩
    · rw [← hops_eq]; simp [evalOps]
    · rw [← hs_eq]; subst hw
      refine ⟨?_, R.locs, ?_, ?_, R.injLocals, R.heapRefines,
              R.currentReg, R.freshCurrent, R.curLocDisj⟩
      · -- StackRefines: top = wI32 n encodes via .bufferPtr slot, tail by R.stk.
        refine ⟨by simp [R.stk.left], ?_⟩
        intro j vj hvj
        cases j with
        | zero =>
          simp at hvj
          refine ⟨SymVal.bufferPtr slot, by simp, ?_⟩
          subst hvj
          subst hv_eq
          show (WasmValue.wI32 n).encodes layout kst.rf (.bufferPtr slot)
          simp [WasmValue.encodes, h_n_eq]
        | succ k =>
          have hwsk : ws.stack.get? k = some vj := by simpa using hvj
          obtain ⟨svk, hsvk, henc⟩ := R.stk.right k vj hwsk
          refine ⟨svk, by simpa using hsvk, henc⟩
      · -- Fresh: nextReg unchanged; new top is .bufferPtr with regs = [].
        refine ⟨?_, R.fresh.right⟩
        intro sv hsv r' hr'
        simp at hsv
        rcases hsv with h_eq | h_in
        · subst h_eq; simp [SymVal.regs] at hr'
        · exact R.fresh.left sv h_in r' hr'
      · -- AliasFree: new top has empty regs.
        intro ir hir sv hsv
        simp at hsv
        rcases hsv with h_eq | h_in
        · subst h_eq; simp [SymVal.regs]
        · exact R.aliasFree ir hir sv h_in
  | none, hw => simp [hloc] at hw

-- ════════════════════════════════════════════════════════════════════
-- Slice 3 follow-up: local.set / local.tee preservation
--
-- The helper lemmas above (`WasmValue.encodes_preserved_of_disjoint`,
-- `WasmValue.encodes_wI32_reg_inv`) are the proof-foundation pieces
-- every localSet/localTee preservation proof needs. The full theorems
-- themselves are ~200-300 LoC each and stay queued for the next
-- slice-3 session — the cleanups here (translator simplified to
-- `getD .u32` instead of `getDM`, helper lemmas in place) make that
-- session significantly tractable. A `find?_filter_keep_of_ne`
-- variant will land alongside those proofs (it's a list-list lemma,
-- not load-bearing on the cascade).
-- ════════════════════════════════════════════════════════════════════

-- ════════════════════════════════════════════════════════════════════
-- Slice 3: i32 binop archetype
--
-- Helpers to extract the two-pop shape from successful `binI32` /
-- `lowerI32Bin` runs, then a single `preservation_i32Bin` lemma
-- parameterized by the op closes the whole 10-op family.
-- ════════════════════════════════════════════════════════════════════

/-- Successful `binI32` runs imply the WASM stack had two `wI32`
    values on top, and the resulting state has the op result
    pushed on the rest. -/
theorem binI32_some_shape {op : UInt32 → UInt32 → UInt32} {s s' : WasmState}
    (h : binI32 op s = some s') :
    ∃ av bv rest, s.stack = .wI32 bv :: .wI32 av :: rest ∧
                  s' = { s with stack := .wI32 (op av bv) :: rest } := by
  unfold binI32 at h
  -- Stack must be at least two-deep.
  rcases hs : s.stack with _ | ⟨b, _ | ⟨a, rest⟩⟩
  · simp [hs, WasmState.pop] at h
  · simp [hs, WasmState.pop] at h
  · -- Both top values must be wI32.
    cases b with
    | wI32 bv =>
      cases a with
      | wI32 av =>
        refine ⟨av, bv, rest, rfl, ?_⟩
        simp [hs, WasmState.pop, WasmState.push] at h
        exact h.symm
      | wI64 _ => simp [hs, WasmState.pop] at h
      | wF32 _ => simp [hs, WasmState.pop] at h
      | wF64 _ => simp [hs, WasmState.pop] at h
    | wI64 _ => cases a <;> simp [hs, WasmState.pop] at h
    | wF32 _ => cases a <;> simp [hs, WasmState.pop] at h
    | wF64 _ => cases a <;> simp [hs, WasmState.pop] at h

/-- Successful `cmpI32` runs imply the same shape but with a 0/1
    bool-encoded as `wI32` on top. -/
theorem cmpI32_some_shape {p : UInt32 → UInt32 → Bool} {s s' : WasmState}
    (h : cmpI32 p s = some s') :
    ∃ av bv rest, s.stack = .wI32 bv :: .wI32 av :: rest ∧
                  s' = { s with stack := .wI32 (if p av bv then 1 else 0) :: rest } := by
  unfold cmpI32 at h
  rcases hs : s.stack with _ | ⟨b, _ | ⟨a, rest⟩⟩
  · simp [hs, WasmState.pop] at h
  · simp [hs, WasmState.pop] at h
  · cases b with
    | wI32 bv =>
      cases a with
      | wI32 av =>
        refine ⟨av, bv, rest, rfl, ?_⟩
        simp [hs, WasmState.pop, WasmState.push] at h
        exact h.symm
      | wI64 _ => simp [hs, WasmState.pop] at h
      | wF32 _ => simp [hs, WasmState.pop] at h
      | wF64 _ => simp [hs, WasmState.pop] at h
    | wI64 _ => cases a <;> simp [hs, WasmState.pop] at h
    | wF32 _ => cases a <;> simp [hs, WasmState.pop] at h
    | wF64 _ => cases a <;> simp [hs, WasmState.pop] at h

/-- Successful `lowerI32Bin` runs split into a deterministic 6-step
    chain: `popSym` × 2 to extract two SymVals from the top of the
    stack, `commit` × 2 to materialize them into real regs (with any
    materialization ops emitted into `opsA` / `opsB`), then `alloc`
    + `push` for the destination register.

    The shape lemma exposes the two intermediate states `s3` (post-
    first-commit) and `s4` (post-second-commit) so the preservation
    proof can apply `commit_correct` once per operand. The
    intermediate states inherit `s.localReg` / `s.localTy` and have
    `lrest` as their stack — derived via `commit_preserves_stack` /
    `commit_preserves_locals`. -/
theorem lowerI32Bin_some_shape {bop : Quanta.KOps.BinOp} {s s' : LowerState}
    {ops : List KernelOp} (h : lowerI32Bin s bop = some (s', ops)) :
    ∃ svb sva lrest ra s3 opsA rb s4 opsB,
      s.stack = svb :: sva :: lrest ∧
      ({ s with stack := lrest } : LowerState).commit sva = some (ra, s3, opsA) ∧
      s3.commit svb = some (rb, s4, opsB) ∧
      s4.stack = lrest ∧
      s4.localReg = s.localReg ∧ s4.localTy = s.localTy ∧
      s.nextReg ≤ s4.nextReg ∧
      s' = { nextReg := s4.nextReg + 1,
             stack := SymVal.reg s4.nextReg .u32 :: lrest,
             localReg := s.localReg,
             localTy := s.localTy,
             bufferSlots := s.bufferSlots, currentReg := s.currentReg } ∧
      ops = opsA ++ opsB ++ [.binOp s4.nextReg ra rb bop .u32] := by
  unfold lowerI32Bin at h
  rcases hs : s.stack with _ | ⟨svb, _ | ⟨sva, lrest⟩⟩
  · simp [hs, LowerState.popSym] at h
  · simp [hs, LowerState.popSym] at h
  ·
    -- Both popSyms succeed; s_pop = { s with stack := lrest }.
    simp only [hs, LowerState.popSym, Option.bind_eq_bind, Option.some_bind] at h
    -- Branch on commit sva success.
    rcases hca : ({s with stack := lrest} : LowerState).commit sva
        with _ | ⟨ra, s3, opsA⟩
    · simp [hca] at h
    ·
      simp only [hca, Option.some_bind] at h
      -- Branch on commit svb success.
      rcases hcb : s3.commit svb with _ | ⟨rb, s4, opsB⟩
      · simp [hcb] at h
      ·
        simp only [hcb, Option.some_bind, LowerState.alloc, LowerState.push] at h
        -- h : some ({...post-state...}, opsA ++ opsB ++ [...]) = some (s', ops)
        obtain ⟨hs_eq, hops_eq⟩ := Prod.mk.injEq _ _ _ _ |>.mp ((Option.some.injEq _ _).mp h)
        -- Derive stack/locals/nextReg facts about s3, s4.
        have h_locA := commit_preserves_locals hca
        have h_stkA := commit_preserves_stack hca
        have h_locB := commit_preserves_locals hcb
        have h_stkB := commit_preserves_stack hcb
        have h_s4_stack : s4.stack = lrest := by rw [h_stkB, h_stkA]
        have h_s4_lr   : s4.localReg = s.localReg := by rw [h_locB.1, h_locA.1]
        have h_s4_lt   : s4.localTy  = s.localTy  := by rw [h_locB.2, h_locA.2]
        have h_s4_bs   : s4.bufferSlots = s.bufferSlots := by
          rw [commit_preserves_bufferSlots hcb, commit_preserves_bufferSlots hca]
        have h_s4_cr   : s4.currentReg = s.currentReg := by
          rw [commit_preserves_currentReg hcb, commit_preserves_currentReg hca]
        have h_s3_nr   : s.nextReg ≤ s3.nextReg := by
          -- commit returns either s (no bump) or s.alloc.snd (bump by 1).
          match sva, hca with
          | .reg _ _, hca =>
            simp [LowerState.commit] at hca
            obtain ⟨_, hs3, _⟩ := hca; rw [← hs3]; exact Nat.le_refl _
          | .i32ConstSym _, hca =>
            simp [LowerState.commit, LowerState.alloc] at hca
            obtain ⟨_, hs3, _⟩ := hca; rw [← hs3]; exact Nat.le_succ _
        have h_s4_nr : s3.nextReg ≤ s4.nextReg := by
          match svb, hcb with
          | .reg _ _, hcb =>
            simp [LowerState.commit] at hcb
            obtain ⟨_, hs4, _⟩ := hcb; rw [← hs4]; exact Nat.le_refl _
          | .i32ConstSym _, hcb =>
            simp [LowerState.commit, LowerState.alloc] at hcb
            obtain ⟨_, hs4, _⟩ := hcb; rw [← hs4]; exact Nat.le_succ _
        refine ⟨svb, sva, lrest, ra, s3, opsA, rb, s4, opsB,
                rfl, hca, hcb, h_s4_stack, h_s4_lr, h_s4_lt,
                Nat.le_trans h_s3_nr h_s4_nr, ?_, hops_eq.symm⟩
        rw [← hs_eq]
        rw [h_s4_stack, h_s4_lr, h_s4_lt, h_s4_bs, h_s4_cr]

/-- Shape for `lowerI32Cmp`. Same `popSym + commit` chain as
    `lowerI32Bin_some_shape`, but the final emission is the two-op
    `cmp + cast` pair (vBool at `s4.nextReg`, vU32 at
    `s4.nextReg + 1`) and the pushed slot points at the cast's
    destination. -/
theorem lowerI32Cmp_some_shape {cop : Quanta.KOps.CmpOp} {s s' : LowerState}
    {ops : List KernelOp} (h : lowerI32Cmp s cop = some (s', ops)) :
    ∃ svb sva lrest ra s3 opsA rb s4 opsB,
      s.stack = svb :: sva :: lrest ∧
      ({ s with stack := lrest } : LowerState).commit sva = some (ra, s3, opsA) ∧
      s3.commit svb = some (rb, s4, opsB) ∧
      s4.stack = lrest ∧
      s4.localReg = s.localReg ∧ s4.localTy = s.localTy ∧
      s.nextReg ≤ s4.nextReg ∧
      s' = { nextReg := s4.nextReg + 2,
             stack := SymVal.reg (s4.nextReg + 1) .u32 :: lrest,
             localReg := s.localReg,
             localTy := s.localTy,
             bufferSlots := s.bufferSlots, currentReg := s.currentReg } ∧
      ops = opsA ++ opsB ++ [.cmp s4.nextReg ra rb cop .bool,
                              .cast (s4.nextReg + 1) s4.nextReg .bool .u32] := by
  unfold lowerI32Cmp at h
  rcases hs : s.stack with _ | ⟨svb, _ | ⟨sva, lrest⟩⟩
  · simp [hs, LowerState.popSym] at h
  · simp [hs, LowerState.popSym] at h
  ·
    simp only [hs, LowerState.popSym, Option.bind_eq_bind, Option.some_bind] at h
    rcases hca : ({s with stack := lrest} : LowerState).commit sva
        with _ | ⟨ra, s3, opsA⟩
    · simp [hca] at h
    ·
      simp only [hca, Option.some_bind] at h
      rcases hcb : s3.commit svb with _ | ⟨rb, s4, opsB⟩
      · simp [hcb] at h
      ·
        simp only [hcb, Option.some_bind, LowerState.alloc, LowerState.push] at h
        obtain ⟨hs_eq, hops_eq⟩ := Prod.mk.injEq _ _ _ _ |>.mp ((Option.some.injEq _ _).mp h)
        have h_locA := commit_preserves_locals hca
        have h_stkA := commit_preserves_stack hca
        have h_locB := commit_preserves_locals hcb
        have h_stkB := commit_preserves_stack hcb
        have h_s4_stack : s4.stack = lrest := by rw [h_stkB, h_stkA]
        have h_s4_lr   : s4.localReg = s.localReg := by rw [h_locB.1, h_locA.1]
        have h_s4_lt   : s4.localTy  = s.localTy  := by rw [h_locB.2, h_locA.2]
        have h_s4_bs   : s4.bufferSlots = s.bufferSlots := by
          rw [commit_preserves_bufferSlots hcb, commit_preserves_bufferSlots hca]
        have h_s4_cr   : s4.currentReg = s.currentReg := by
          rw [commit_preserves_currentReg hcb, commit_preserves_currentReg hca]
        have h_s3_nr   : s.nextReg ≤ s3.nextReg := by
          match sva, hca with
          | .reg _ _, hca =>
            simp [LowerState.commit] at hca
            obtain ⟨_, hs3, _⟩ := hca; rw [← hs3]; exact Nat.le_refl _
          | .i32ConstSym _, hca =>
            simp [LowerState.commit, LowerState.alloc] at hca
            obtain ⟨_, hs3, _⟩ := hca; rw [← hs3]; exact Nat.le_succ _
        have h_s4_nr : s3.nextReg ≤ s4.nextReg := by
          match svb, hcb with
          | .reg _ _, hcb =>
            simp [LowerState.commit] at hcb
            obtain ⟨_, hs4, _⟩ := hcb; rw [← hs4]; exact Nat.le_refl _
          | .i32ConstSym _, hcb =>
            simp [LowerState.commit, LowerState.alloc] at hcb
            obtain ⟨_, hs4, _⟩ := hcb; rw [← hs4]; exact Nat.le_succ _
        refine ⟨svb, sva, lrest, ra, s3, opsA, rb, s4, opsB,
                rfl, hca, hcb, h_s4_stack, h_s4_lr, h_s4_lt,
                Nat.le_trans h_s3_nr h_s4_nr, ?_, hops_eq.symm⟩
        rw [← hs_eq]
        rw [h_s4_stack, h_s4_lr, h_s4_lt, h_s4_bs, h_s4_cr]

-- ════════════════════════════════════════════════════════════════════
-- Generic i32 binop preservation (instantiates for the 10-op family)
-- ════════════════════════════════════════════════════════════════════

open Quanta.KOps (vU32) in
/-- Generic preservation for any WASM i32 binop the lowering pass
    handles. Takes:
    * `instr`  — the WASM instruction.
    * `op_w`   — the u32 → u32 → u32 op the WASM semantics dispatches.
    * `op_k`   — the matching `KOps.BinOp` the lowering emits.
    * `h_w`    — `evalInstr s instr = binI32 op_w s` (by rfl per-arm).
    * `h_l`    — `lowerInstr s instr = lowerI32Bin s op_k` (by rfl).
    * `h_agree` — the KOps eval matches the WASM eval on u32 values.

    Each of the 10 i32-binop preservation theorems below is one line:
    instantiate with `rfl rfl (by intro …; rfl)`.

    The lowering now consumes operands via `popSym + commit` (not raw
    `pop`), so the emitted op list is `opsA ++ opsB ++ [binOp]` where
    `opsA` / `opsB` are the (possibly empty) materialization ops from
    each `commit` — matching production's pull-based const folding.
    The proof applies `commit_correct` once per popped operand,
    threading encodings through the regfile evolution. The
    `kst.broke = false` precondition is required (was implicit before
    when only one op was emitted) because `evalOps_append` short-
    circuits on `broke` between ops. -/
theorem preservation_i32Bin_generic
    (instr : WasmInstr) (op_w : UInt32 → UInt32 → UInt32)
    (op_k : Quanta.KOps.BinOp)
    (h_w : ∀ s, evalInstr s instr = binI32 op_w s)
    (h_agree : ∀ av bv,
       Quanta.KOps.evalBinOp op_k (vU32 av) (vU32 bv) = some (vU32 (op_w av bv)))
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout) (h_kst_ok : kst.broke = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (h_l : lowerInstr s instr = lowerI32Bin s op_k)
    (hw : evalInstr ws instr = some ws')
    (hl : lowerInstr s instr = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  rw [h_w] at hw
  rw [h_l] at hl
  obtain ⟨av, bv, rest, hwstack, hws_eq⟩ := binI32_some_shape hw
  obtain ⟨svb, sva, lrest, ra, s3, opsA, rb, s4, opsB, hlstack, hca, hcb,
          h_s4_stack, h_s4_lr, h_s4_lt, h_s_le_s4, hs_eq, hops_eq⟩ :=
    lowerI32Bin_some_shape hl
  -- Operand encodings in kst.rf — extracted from R.stk before any commit.
  have h_enc_svb : (WasmValue.wI32 bv).encodes layout kst.rf svb := by
    have hb := R.stk.right 0 (.wI32 bv) (by rw [hwstack]; simp)
    obtain ⟨sv0, hsv0_get, henc⟩ := hb
    have hs0 : s.stack.get? 0 = some svb := by rw [hlstack]; simp
    rw [hs0] at hsv0_get
    have h_eq : svb = sv0 := (Option.some.injEq _ _).mp hsv0_get
    rw [h_eq]; exact henc
  have h_enc_sva : (WasmValue.wI32 av).encodes layout kst.rf sva := by
    have ha := R.stk.right 1 (.wI32 av) (by rw [hwstack]; simp)
    obtain ⟨sv1, hsv1_get, henc⟩ := ha
    have hs1 : s.stack.get? 1 = some sva := by rw [hlstack]; simp
    rw [hs1] at hsv1_get
    have h_eq : sva = sv1 := (Option.some.injEq _ _).mp hsv1_get
    rw [h_eq]; exact henc
  -- Membership of svb, sva in s.stack — used to extract Fresh / AliasFree facts.
  have h_svb_in : svb ∈ s.stack := by rw [hlstack]; simp
  have h_sva_in : sva ∈ s.stack := by rw [hlstack]; simp
  -- Fresh-bound on each operand's regs.
  have h_svb_lt : ∀ r ∈ svb.regs, r < s.nextReg :=
    fun r hr => R.fresh.left svb h_svb_in r hr
  have h_sva_lt : ∀ r ∈ sva.regs, r < s.nextReg :=
    fun r hr => R.fresh.left sva h_sva_in r hr
  -- Construct Refines for the popped state s_pop = { s with stack := lrest }
  -- on ws_pop = { ws with stack := rest }. Each clause weakens trivially.
  have h_rest_lrest_len : rest.length = lrest.length := by
    have hl_orig := R.stk.left
    rw [hwstack, hlstack] at hl_orig
    simpa using hl_orig
  let s_pop : LowerState :=
    { nextReg := s.nextReg, stack := lrest,
      localReg := s.localReg, localTy := s.localTy,
      bufferSlots := s.bufferSlots, currentReg := s.currentReg }
  let ws_pop : WasmState :=
    { ws with stack := rest }
  have R_pop : Refines ws_pop s_pop kst layout := by
    refine ⟨⟨h_rest_lrest_len, ?_⟩, R.locs, ?_, ?_, R.injLocals, R.heapRefines,
            R.currentReg, R.freshCurrent, R.curLocDisj⟩
    · -- StackRefines on the (rest, lrest) suffix — shift indices by 2 and reuse R.stk.
      intro i v hv
      have hrest_get : ws.stack.get? (i + 2) = some v := by
        rw [hwstack]; simpa using hv
      obtain ⟨svi, hsvi_get, henc⟩ := R.stk.right (i + 2) v hrest_get
      have hlrest_get : lrest.get? i = some svi := by
        have h2 : s.stack.get? (i + 2) = some svi := hsvi_get
        rw [hlstack] at h2; simpa using h2
      exact ⟨svi, by simpa using hlrest_get, henc⟩
    · -- Fresh: s_pop.stack ⊆ s.stack and same locals.
      refine ⟨?_, R.fresh.right⟩
      intro sv hsv r hr
      have hsv_in : sv ∈ s.stack := by rw [hlstack]; exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hsv)
      exact R.fresh.left sv hsv_in r hr
    · -- AliasFree: same projection on the lrest suffix.
      intro ir hir sv hsv
      have hsv_in : sv ∈ s.stack := by rw [hlstack]; exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hsv)
      exact R.aliasFree ir hir sv hsv_in
  -- First commit: materialize sva → ra. Emits opsA, evolves kst → kst1.
  obtain ⟨kst1, h_evalA, R1, h_enc_ra1, h_lookupA, h_s_le_s3, h_ra_lt_s3⟩ :=
    commit_correct R_pop h_sva_lt hca h_enc_sva
  -- After opsA, broke is preserved (commit emits only `.const` ops, which
  -- write the regfile and inherit `broke`).
  have h_kst1_ok : kst1.broke = false := by
    rw [commit_preserves_broke hca h_evalA]; exact h_kst_ok
  -- Encoding of svb at kst1.rf — lift through opsA via lookup-preservation.
  have h_enc_svb1 : (WasmValue.wI32 bv).encodes layout kst1.rf svb :=
    WasmValue.encodes_preserved_of_lookup_eq
      (fun r hr => h_lookupA r (h_svb_lt r hr)) h_enc_svb
  -- Fresh-bound on svb at s3.nextReg.
  have h_svb_lt_s3 : ∀ r ∈ svb.regs, r < s3.nextReg :=
    fun r hr => Nat.lt_of_lt_of_le (h_svb_lt r hr) h_s_le_s3
  -- Second commit: materialize svb → rb on R1. Emits opsB, evolves kst1 → kst2.
  obtain ⟨kst2, h_evalB, R2, h_enc_rb2, h_lookupB, h_s3_le_s4, h_rb_lt_s4⟩ :=
    commit_correct R1 h_svb_lt_s3 hcb h_enc_svb1
  have h_kst2_ok : kst2.broke = false := by
    rw [commit_preserves_broke hcb h_evalB]; exact h_kst1_ok
  -- Lift ra's encoding from kst1 to kst2 via the second commit's lookup-preservation.
  have h_enc_ra2 : (WasmValue.wI32 av).encodes layout kst2.rf (.reg ra .u32) :=
    WasmValue.encodes_preserved_of_lookup_eq
      (fun r hr => by
        simp [SymVal.regs] at hr
        rw [hr]
        exact h_lookupB ra h_ra_lt_s3)
      h_enc_ra1
  -- Extract reg lookups in kst2 from the encodings (used by the binOp eval).
  have h_lookup_ra : regLookup kst2.rf ra = some (vU32 av) := h_enc_ra2
  have h_lookup_rb : regLookup kst2.rf rb = some (vU32 bv) := h_enc_rb2
  -- Final kst'. Note: `broke := false` rather than `kst2.broke` so the
  -- post-binOp simp output (which substitutes h_kst2_ok) matches without
  -- an extra rewrite. heap and dispatch carry through from kst2.
  refine ⟨{ rf := regWrite kst2.rf s4.nextReg (vU32 (op_w av bv)),
            heap := kst2.heap, dispatch := kst2.dispatch, broke := false }, ?_, ?_⟩
  · -- evalOps 0 kst (opsA ++ opsB ++ [binOp …]) = some kst3.
    subst hops_eq
    -- Glue the three sub-evaluations via evalOps_append.
    rw [show opsA ++ opsB ++ [KernelOp.binOp s4.nextReg ra rb op_k Quanta.KOps.Scalar.u32]
          = opsA ++ (opsB ++ [KernelOp.binOp s4.nextReg ra rb op_k Quanta.KOps.Scalar.u32]) from
        by rw [List.append_assoc]]
    rw [evalOps_append h_evalA h_kst1_ok]
    rw [evalOps_append h_evalB h_kst2_ok]
    -- Now reduce the single-op `evalOps 0 kst2 [binOp …]`.
    simp [evalOps, Quanta.KOps.evalOp, h_lookup_ra, h_lookup_rb, h_agree, h_kst2_ok]
  · -- Refines ws' s' kst3 layout.
    subst hs_eq; subst hws_eq
    refine ⟨?_, ?_, ?_, ?_, ?_, R2.heapRefines, ?_, ?_, ?_⟩
    · -- StackRefines on (wI32 (op_w av bv) :: rest, .reg s4.nextReg .u32 :: lrest).
      refine ⟨?_, ?_⟩
      · -- Length.
        simp; exact h_rest_lrest_len
      · intro j v hv
        cases j with
        | zero =>
          simp at hv
          refine ⟨SymVal.reg s4.nextReg .u32, by simp, ?_⟩
          subst hv
          show regLookup (regWrite kst2.rf s4.nextReg (vU32 (op_w av bv))) s4.nextReg
                 = some (vU32 (op_w av bv))
          simp [regLookup_regWrite_self]
        | succ k =>
          -- Re-extract via R2.stk.right at index k. ws_pop.stack = rest, s4.stack = lrest.
          have hk : ws_pop.stack.get? k = some v := by
            show rest.get? k = some v
            simpa using hv
          obtain ⟨svk, hsvk_get, henc⟩ := R2.stk.right k v hk
          -- s4.stack = lrest, so unfolding yields s4.stack.get? k = some svk.
          have h_s4_get : s4.stack.get? k = some svk := hsvk_get
          rw [h_s4_stack] at h_s4_get
          refine ⟨svk, by simpa using h_s4_get, ?_⟩
          have hsvk_in : svk ∈ s4.stack := List.mem_of_get? hsvk_get
          apply WasmValue.encodes_preserved_of_fresh _ henc
          intro r' hr'
          exact R2.fresh.left svk hsvk_in r' hr'
    · -- LocalsRefines: localReg unchanged through commits + binOp.
      intro i r hfind v hv
      rw [← h_s4_lr] at hfind
      have hpair : (i, r) ∈ s4.localReg := List.mem_of_find?_eq_some hfind
      have hr_lt : r < s4.nextReg := R2.fresh.right (i, r) hpair
      have henc := R2.locs i r hfind v hv
      apply WasmValue.encodes_preserved_of_fresh _ henc
      intro r' hr'_in
      simp [SymVal.regs] at hr'_in
      subst hr'_in; exact hr_lt
    · -- Fresh on s' — top is .reg s4.nextReg .u32, lrest ⊆ s4.stack, locals from s4.
      refine ⟨?_, ?_⟩
      · intro sv hsv r' hr'
        simp at hsv
        rcases hsv with h_eq | h_in
        · subst h_eq
          simp [SymVal.regs] at hr'
          subst hr'; exact Nat.lt_succ_self _
        · have hsv_in_s4 : sv ∈ s4.stack := by rw [h_s4_stack]; exact h_in
          exact Nat.lt_succ_of_lt (R2.fresh.left sv hsv_in_s4 r' hr')
      · intro ir hir
        rw [← h_s4_lr] at hir
        exact Nat.lt_succ_of_lt (R2.fresh.right ir hir)
    · -- AliasFree on s'.
      intro ir hir sv hsv
      rw [← h_s4_lr] at hir
      have hir_lt : ir.snd < s4.nextReg := R2.fresh.right ir hir
      simp at hsv
      rcases hsv with h_eq | h_in
      · subst h_eq
        simp [SymVal.regs]
        exact Nat.ne_of_lt hir_lt
      · have hsv_in_s4 : sv ∈ s4.stack := by rw [h_s4_stack]; exact h_in
        exact R2.aliasFree ir hir sv hsv_in_s4
    · -- InjectiveLocals: localReg unchanged through commits and binOp.
      intro p q hp hq
      rw [← h_s4_lr] at hp hq
      exact R2.injLocals p q hp hq
    · -- CurrentRegRefines: ws.locals unchanged (binop doesn't touch
      -- locals); s'.currentReg = s.currentReg (commit preserves
      -- currentReg, both commits). Lift R2.currentReg past the
      -- fresh write at s4.nextReg.
      have h_s4_cur : s4.currentReg = s.currentReg := by
        rw [commit_preserves_currentReg hcb, commit_preserves_currentReg hca]
      have h_lift := CurrentRegRefines_preserved_fresh R2.currentReg
        R2.freshCurrent (vU32 (op_w av bv))
      show CurrentRegRefines layout ws.locals s.currentReg
            (regWrite kst2.rf s4.nextReg (vU32 (op_w av bv)))
      rw [← h_s4_cur]
      have h_locs_eq : ws.locals = ws_pop.locals := rfl
      rw [h_locs_eq]
      exact h_lift
    · -- FreshCurrent: s'.nextReg = s4.nextReg + 1, s'.currentReg = s.currentReg.
      intro ir hir
      have h_s4_cur : s4.currentReg = s.currentReg := by
        rw [commit_preserves_currentReg hcb, commit_preserves_currentReg hca]
      rw [← h_s4_cur] at hir
      exact Nat.lt_succ_of_lt (R2.freshCurrent ir hir)
    · -- CurrentLocalDisjoint: s'.currentReg / s'.localReg same as s4's; lift R2.curLocDisj.
      intro p q hp hq hpq
      have h_s4_cur : s4.currentReg = s.currentReg := by
        rw [commit_preserves_currentReg hcb, commit_preserves_currentReg hca]
      rw [← h_s4_cur] at hp
      rw [← h_s4_lr] at hq
      exact R2.curLocDisj p q hp hq hpq

-- ── Per-op specializations (10 binops) ─────────────────────────────────
--
-- Each per-op preservation is a thin wrapper around
-- `preservation_i32Bin_generic`. For the 8 ops without a buffer-pattern
-- fast-path (Sub, Mul, And, Or, Xor, ShrU, DivU, RemU), the wrapper
-- supplies `h_l := rfl` — `lowerInstr s .i32Foo = lowerI32Bin s .Foo`
-- holds definitionally for any `s`. For `Add` and `Shl`, the lowering
-- arm dispatches into `lowerI32Add` / `lowerI32Shl` whose buffer-
-- pattern fast-paths return a folded `bufferAccess` / `scaledIdx`
-- without emitting IR — the wrapper takes a `h_no_buf` precondition
-- that excludes the buffer-pattern stack shape, and derives the
-- `h_l` equation from it. The folded-path preservation lands with
-- `HeapRefines` consumers in slice-4 step 8.

theorem preservation_i32Sub
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout) (h_kst_ok : kst.broke = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws .i32Sub = some ws')
    (hl : lowerInstr s .i32Sub = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_i32Bin_generic .i32Sub eval_u32_wrapping_sub .sub
    (fun _ => rfl) (by intro av bv; rfl)
    ws s kst layout R h_kst_ok ws' s' ops rfl hw hl

theorem preservation_i32Mul
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout) (h_kst_ok : kst.broke = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws .i32Mul = some ws')
    (hl : lowerInstr s .i32Mul = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_i32Bin_generic .i32Mul eval_u32_wrapping_mul .mul
    (fun _ => rfl) (by intro av bv; rfl)
    ws s kst layout R h_kst_ok ws' s' ops rfl hw hl

theorem preservation_i32And
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout) (h_kst_ok : kst.broke = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws .i32And = some ws')
    (hl : lowerInstr s .i32And = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_i32Bin_generic .i32And eval_u32_bitand .bAnd
    (fun _ => rfl) (by intro av bv; rfl)
    ws s kst layout R h_kst_ok ws' s' ops rfl hw hl

theorem preservation_i32Or
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout) (h_kst_ok : kst.broke = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws .i32Or = some ws')
    (hl : lowerInstr s .i32Or = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_i32Bin_generic .i32Or eval_u32_bitor .bOr
    (fun _ => rfl) (by intro av bv; rfl)
    ws s kst layout R h_kst_ok ws' s' ops rfl hw hl

theorem preservation_i32Xor
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout) (h_kst_ok : kst.broke = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws .i32Xor = some ws')
    (hl : lowerInstr s .i32Xor = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_i32Bin_generic .i32Xor eval_u32_bitxor .bXor
    (fun _ => rfl) (by intro av bv; rfl)
    ws s kst layout R h_kst_ok ws' s' ops rfl hw hl

theorem preservation_i32ShrU
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout) (h_kst_ok : kst.broke = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws .i32ShrU = some ws')
    (hl : lowerInstr s .i32ShrU = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_i32Bin_generic .i32ShrU (fun a b => a >>> b) .shr
    (fun _ => rfl) (by intro av bv; rfl)
    ws s kst layout R h_kst_ok ws' s' ops rfl hw hl

theorem preservation_i32DivU
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout) (h_kst_ok : kst.broke = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws .i32DivU = some ws')
    (hl : lowerInstr s .i32DivU = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_i32Bin_generic .i32DivU eval_u32_div .div
    (fun _ => rfl) (by intro av bv; rfl)
    ws s kst layout R h_kst_ok ws' s' ops rfl hw hl

theorem preservation_i32RemU
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout) (h_kst_ok : kst.broke = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws .i32RemU = some ws')
    (hl : lowerInstr s .i32RemU = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_i32Bin_generic .i32RemU eval_u32_rem .rem
    (fun _ => rfl) (by intro av bv; rfl)
    ws s kst layout R h_kst_ok ws' s' ops rfl hw hl

/-- The buffer-pattern stack shape `lowerI32Add` recognizes:
    `<scaledIdx> :: <bufferPtr> :: _` or its order-flipped twin. When
    excluded by `h_no_buf`, `lowerI32Add s = lowerI32Bin s .add` and
    the generic binop preservation closes the proof. The folded-path
    preservation (push `bufferAccess` symbolically) lands in
    slice-4 step 8 alongside `i32.load`/`i32.store` consumers. -/
theorem preservation_i32Add
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout) (h_kst_ok : kst.broke = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (h_no_buf : ∀ slot base scale rest,
      s.stack ≠ .scaledIdx base scale :: .bufferPtr slot :: rest ∧
      s.stack ≠ .bufferPtr slot :: .scaledIdx base scale :: rest)
    (hw : evalInstr ws .i32Add = some ws')
    (hl : lowerInstr s .i32Add = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  have h_l : lowerInstr s .i32Add = lowerI32Bin s .add := by
    show lowerI32Add s = lowerI32Bin s .add
    unfold lowerI32Add
    split
    next base scale slot rest hs => exact absurd hs (h_no_buf slot base scale rest).left
    next slot base scale rest hs => exact absurd hs (h_no_buf slot base scale rest).right
    next => rfl
  exact preservation_i32Bin_generic .i32Add eval_u32_wrapping_add .add
    (fun _ => rfl) (by intro av bv; rfl)
    ws s kst layout R h_kst_ok ws' s' ops h_l hw hl

/-- The buffer-pattern stack shape `lowerI32Shl` recognizes:
    `<i32ConstSym k> :: <reg base _> :: _`. When excluded by
    `h_no_buf`, `lowerI32Shl s = lowerI32Bin s .shl` and the generic
    binop preservation closes the proof. The folded-path preservation
    (push `scaledIdx` symbolically) lands in slice-4 step 8. -/
theorem preservation_i32Shl
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout) (h_kst_ok : kst.broke = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (h_no_buf : ∀ k base ty rest,
      s.stack ≠ .i32ConstSym k :: .reg base ty :: rest)
    (hw : evalInstr ws .i32Shl = some ws')
    (hl : lowerInstr s .i32Shl = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  have h_l : lowerInstr s .i32Shl = lowerI32Bin s .shl := by
    show lowerI32Shl s = lowerI32Bin s .shl
    unfold lowerI32Shl
    split
    next k base ty rest hs => exact absurd hs (h_no_buf k base ty rest)
    next => rfl
  exact preservation_i32Bin_generic .i32Shl (fun a b => a <<< b) .shl
    (fun _ => rfl) (by intro av bv; rfl)
    ws s kst layout R h_kst_ok ws' s' ops h_l hw hl

/-- Folded `i32.shl` preservation: when the popped stack matches
    `<i32ConstSym k> :: <reg base ty> :: rest`, the lowering emits no
    IR and pushes `SymVal.scaledIdx base (1 <<< k.toNat)`. The new
    top encodes the WASM shift result `wI32 (a <<< (UInt32.ofNat
    k.toNat))` via the `scaledIdx` arm, witnessed by `b := a` (the
    register base's value).

    The `h_shift_eq` precondition captures both "shift amount in
    range" (`k.toNat < 32`) and "no overflow" (`a.toNat * 2^k.toNat
    < 2^32`) — when either fails, the WASM `<<<` truncates and the
    encoding's `n.toNat = b.toNat * scale` Nat-equation breaks.
    Future kernel-entry composition theorem will discharge it from
    layout bounds (typical kernels have `scale ≤ 8` and indices
    `< 2^29`). -/
theorem preservation_i32Shl_bufferPattern
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout)
    (k : Int) (base : Reg) (ty : Quanta.KOps.Scalar) (rest : List SymVal)
    (h_stack : s.stack = .i32ConstSym k :: .reg base ty :: rest)
    (h_shift_eq : ∀ a : UInt32,
       regLookup kst.rf base = some (Quanta.KOps.Value.vU32 a) →
       (a <<< (UInt32.ofNat k.toNat)).toNat = a.toNat * (1 <<< k.toNat))
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws .i32Shl = some ws')
    (hl : lowerInstr s .i32Shl = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- 1. Reduce hl using h_stack: the buffer-pattern arm fires.
  have hl_reduced : lowerInstr s .i32Shl =
      some ({ s with stack := .scaledIdx base (1 <<< k.toNat) :: rest }, []) := by
    show lowerI32Shl s = _
    unfold lowerI32Shl
    rw [h_stack]
  rw [hl_reduced] at hl
  obtain ⟨hs_eq, hops_eq⟩ :=
    Prod.mk.injEq _ _ _ _ |>.mp ((Option.some.injEq _ _).mp hl)
  -- 2. Reduce hw via binI32 shape lemma.
  have hw' : binI32 (fun a b => a <<< b) ws = some ws' := hw
  obtain ⟨a_w, b_w, ws_rest, h_ws_stack, h_ws_eq⟩ := binI32_some_shape hw'
  -- 3. From R.stk, derive a_w / b_w identities.
  have h_stk0 := R.stk.right 0 (.wI32 b_w) (by rw [h_ws_stack]; simp)
  obtain ⟨sv0, hsv0_get, henc0⟩ := h_stk0
  have hs0_eq : s.stack.get? 0 = some (.i32ConstSym k) := by rw [h_stack]; simp
  rw [hs0_eq] at hsv0_get
  have hsv0_eq : sv0 = .i32ConstSym k :=
    ((Option.some.injEq _ _).mp hsv0_get).symm
  rw [hsv0_eq] at henc0
  -- `encodes_i32ConstSym_inv` returns `wI32 b_w = wI32 (UInt32.ofNat k.toNat)`;
  -- strip the wI32 constructor.
  have h_b_eq : b_w = UInt32.ofNat k.toNat := by
    have := WasmValue.encodes_i32ConstSym_inv henc0
    exact WasmValue.wI32.injEq _ _ |>.mp this
  have h_stk1 := R.stk.right 1 (.wI32 a_w) (by rw [h_ws_stack]; simp)
  obtain ⟨sv1, hsv1_get, henc1⟩ := h_stk1
  have hs1_eq : s.stack.get? 1 = some (.reg base ty) := by rw [h_stack]; simp
  rw [hs1_eq] at hsv1_get
  have hsv1_eq : sv1 = .reg base ty :=
    ((Option.some.injEq _ _).mp hsv1_get).symm
  rw [hsv1_eq] at henc1
  obtain ⟨_h_ty_eq, h_lookup_a⟩ := WasmValue.encodes_wI32_reg_inv henc1
  -- 4. ops = [] → kst' = kst.
  refine ⟨kst, ?_, ?_⟩
  · rw [← hops_eq]; simp [evalOps]
  · rw [← hs_eq]; subst h_ws_eq
    refine ⟨?_, R.locs, ?_, ?_, R.injLocals, R.heapRefines,
            R.currentReg, R.freshCurrent, R.curLocDisj⟩
    · -- StackRefines: top encodes via .scaledIdx; tail unchanged.
      refine ⟨?_, ?_⟩
      · -- Length: ws_rest.length = rest.length, derived from R.stk.left.
        have hlen := R.stk.left
        rw [h_stack, h_ws_stack] at hlen
        simp at hlen
        simp; exact hlen
      · intro j vj hvj
        cases j with
        | zero =>
          simp at hvj; subst hvj
          refine ⟨.scaledIdx base (1 <<< k.toNat), by simp, ?_⟩
          show (WasmValue.wI32 (a_w <<< b_w)).encodes layout kst.rf
                 (.scaledIdx base (1 <<< k.toNat))
          rw [h_b_eq]
          refine ⟨a_w, h_lookup_a, h_shift_eq a_w h_lookup_a⟩
        | succ j' =>
          have hwsk : ws.stack.get? (j' + 2) = some vj := by
            rw [h_ws_stack]; simpa using hvj
          obtain ⟨svk, hsvk, henck⟩ := R.stk.right (j' + 2) vj hwsk
          have hsk : s.stack.get? (j' + 2) = some svk := hsvk
          rw [h_stack] at hsk
          simp at hsk
          refine ⟨svk, by simpa using hsk, henck⟩
    · -- Fresh: new top has regs = [base], existing in s.stack via h_stack.
      refine ⟨?_, R.fresh.right⟩
      intro sv hsv r' hr'
      simp at hsv
      rcases hsv with h_eq | h_in
      · subst h_eq
        simp [SymVal.regs] at hr'
        -- hr' : r' = base. Rewrite the goal's r' to base, then bound base.
        rw [hr']
        have hbase_in_stack : (.reg base ty : SymVal) ∈ s.stack := by
          rw [h_stack]; simp
        exact R.fresh.left _ hbase_in_stack base (by simp [SymVal.regs])
      · have hsv_in_s : sv ∈ s.stack := by
          rw [h_stack]; exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ h_in)
        exact R.fresh.left sv hsv_in_s r' hr'
    · -- AliasFree: new top's reg is base; reuse R.aliasFree on the original .reg base ty entry.
      intro ir hir sv hsv
      simp at hsv
      rcases hsv with h_eq | h_in
      · subst h_eq
        have hbase_in_stack : (.reg base ty : SymVal) ∈ s.stack := by
          rw [h_stack]; simp
        have hb_disj := R.aliasFree ir hir _ hbase_in_stack
        simp [SymVal.regs] at hb_disj ⊢
        exact hb_disj
      · have hsv_in_s : sv ∈ s.stack := by
          rw [h_stack]; exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ h_in)
        exact R.aliasFree ir hir sv hsv_in_s

/-- Folded `i32.add` preservation, scaled-first order: when the popped
    stack matches `<scaledIdx base scale> :: <bufferPtr slot> :: rest`,
    the lowering emits no IR and pushes
    `SymVal.bufferAccess slot base scale`. The new top encodes the
    WASM add result via the `bufferAccess` arm.

    The `h_addr_eq` precondition captures no-overflow on the address
    arithmetic — the WASM UInt32 add must equal the corresponding Nat
    add of `layout.startAddr slot + b.toNat * scale`. Future
    kernel-entry composition theorem will discharge it from layout
    bounds (typical kernels have addresses `< 2^31`). -/
theorem preservation_i32Add_bufferPattern_scaledFirst
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout)
    (slot : Nat) (base : Reg) (scale : Nat) (rest : List SymVal)
    (h_stack : s.stack = .scaledIdx base scale :: .bufferPtr slot :: rest)
    (h_addr_eq : ∀ a b_ptr : UInt32, ∀ b : UInt32,
       regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
       a.toNat = b.toNat * scale →
       b_ptr.toNat = layout.startAddr slot →
       (b_ptr + a).toNat = layout.startAddr slot + b.toNat * scale)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws .i32Add = some ws')
    (hl : lowerInstr s .i32Add = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- 1. Reduce hl using h_stack: scaled-first arm fires.
  have hl_reduced : lowerInstr s .i32Add =
      some ({ s with stack := .bufferAccess slot base scale :: rest }, []) := by
    show lowerI32Add s = _
    unfold lowerI32Add
    rw [h_stack]
  rw [hl_reduced] at hl
  obtain ⟨hs_eq, hops_eq⟩ :=
    Prod.mk.injEq _ _ _ _ |>.mp ((Option.some.injEq _ _).mp hl)
  -- 2. Reduce hw via binI32 shape.
  have hw' : binI32 eval_u32_wrapping_add ws = some ws' := hw
  obtain ⟨a_w, b_w, ws_rest, h_ws_stack, h_ws_eq⟩ := binI32_some_shape hw'
  -- 3. From R.stk, derive a_w / b_w identities.
  -- ws.stack[0] = wI32 b_w ↔ s.stack[0] = .scaledIdx base scale.
  have h_stk0 := R.stk.right 0 (.wI32 b_w) (by rw [h_ws_stack]; simp)
  obtain ⟨sv0, hsv0_get, henc0⟩ := h_stk0
  have hs0_eq : s.stack.get? 0 = some (.scaledIdx base scale) := by
    rw [h_stack]; simp
  rw [hs0_eq] at hsv0_get
  have hsv0_eq : sv0 = .scaledIdx base scale :=
    ((Option.some.injEq _ _).mp hsv0_get).symm
  rw [hsv0_eq] at henc0
  -- henc0 : ∃ b, regLookup kst.rf base = some (vU32 b) ∧ b_w.toNat = b.toNat * scale.
  obtain ⟨b, h_lookup_b, h_bw_eq⟩ := henc0
  -- ws.stack[1] = wI32 a_w ↔ s.stack[1] = .bufferPtr slot.
  have h_stk1 := R.stk.right 1 (.wI32 a_w) (by rw [h_ws_stack]; simp)
  obtain ⟨sv1, hsv1_get, henc1⟩ := h_stk1
  have hs1_eq : s.stack.get? 1 = some (.bufferPtr slot) := by
    rw [h_stack]; simp
  rw [hs1_eq] at hsv1_get
  have hsv1_eq : sv1 = .bufferPtr slot :=
    ((Option.some.injEq _ _).mp hsv1_get).symm
  rw [hsv1_eq] at henc1
  -- henc1 : a_w.toNat = layout.startAddr slot.
  have h_aw_eq : a_w.toNat = layout.startAddr slot := henc1
  -- 4. ops = [] → kst' = kst.
  refine ⟨kst, ?_, ?_⟩
  · rw [← hops_eq]; simp [evalOps]
  · rw [← hs_eq]; subst h_ws_eq
    refine ⟨?_, R.locs, ?_, ?_, R.injLocals, R.heapRefines,
            R.currentReg, R.freshCurrent, R.curLocDisj⟩
    · -- StackRefines: top encodes via .bufferAccess; tail unchanged.
      refine ⟨?_, ?_⟩
      · have hlen := R.stk.left
        rw [h_stack, h_ws_stack] at hlen
        simp at hlen
        simp; exact hlen
      · intro j vj hvj
        cases j with
        | zero =>
          simp at hvj; subst hvj
          refine ⟨.bufferAccess slot base scale, by simp, ?_⟩
          show (WasmValue.wI32 (eval_u32_wrapping_add a_w b_w)).encodes layout kst.rf
                 (.bufferAccess slot base scale)
          -- eval_u32_wrapping_add av bv := av + bv.
          show ∃ b', regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b') ∧
                 (eval_u32_wrapping_add a_w b_w).toNat =
                   layout.startAddr slot + b'.toNat * scale
          refine ⟨b, h_lookup_b, ?_⟩
          show (a_w + b_w).toNat = layout.startAddr slot + b.toNat * scale
          exact h_addr_eq b_w a_w b h_lookup_b h_bw_eq h_aw_eq
        | succ j' =>
          have hwsk : ws.stack.get? (j' + 2) = some vj := by
            rw [h_ws_stack]; simpa using hvj
          obtain ⟨svk, hsvk, henck⟩ := R.stk.right (j' + 2) vj hwsk
          have hsk : s.stack.get? (j' + 2) = some svk := hsvk
          rw [h_stack] at hsk
          simp at hsk
          refine ⟨svk, by simpa using hsk, henck⟩
    · -- Fresh: new top has regs = [base], existing in s.stack via h_stack.
      refine ⟨?_, R.fresh.right⟩
      intro sv hsv r' hr'
      simp at hsv
      rcases hsv with h_eq | h_in
      · subst h_eq
        simp [SymVal.regs] at hr'
        rw [hr']
        have hbase_in_stack : (.scaledIdx base scale : SymVal) ∈ s.stack := by
          rw [h_stack]; simp
        exact R.fresh.left _ hbase_in_stack base (by simp [SymVal.regs])
      · have hsv_in_s : sv ∈ s.stack := by
          rw [h_stack]; exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ h_in)
        exact R.fresh.left sv hsv_in_s r' hr'
    · -- AliasFree: new top's reg is base; reuse R.aliasFree.
      intro ir hir sv hsv
      simp at hsv
      rcases hsv with h_eq | h_in
      · subst h_eq
        have hbase_in_stack : (.scaledIdx base scale : SymVal) ∈ s.stack := by
          rw [h_stack]; simp
        have hb_disj := R.aliasFree ir hir _ hbase_in_stack
        simp [SymVal.regs] at hb_disj ⊢
        exact hb_disj
      · have hsv_in_s : sv ∈ s.stack := by
          rw [h_stack]; exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ h_in)
        exact R.aliasFree ir hir sv hsv_in_s

/-- Folded `i32.add` preservation, ptr-first order: when the popped
    stack matches `<bufferPtr slot> :: <scaledIdx base scale> :: rest`,
    the lowering emits no IR and pushes `bufferAccess`. Same shape as
    the scaled-first variant; only the WASM add direction flips
    (`a + b` vs `b + a`, both equal by commutativity). -/
theorem preservation_i32Add_bufferPattern_ptrFirst
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout)
    (slot : Nat) (base : Reg) (scale : Nat) (rest : List SymVal)
    (h_stack : s.stack = .bufferPtr slot :: .scaledIdx base scale :: rest)
    (h_addr_eq : ∀ a b_ptr : UInt32, ∀ b : UInt32,
       regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
       a.toNat = b.toNat * scale →
       b_ptr.toNat = layout.startAddr slot →
       (a + b_ptr).toNat = layout.startAddr slot + b.toNat * scale)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws .i32Add = some ws')
    (hl : lowerInstr s .i32Add = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  have hl_reduced : lowerInstr s .i32Add =
      some ({ s with stack := .bufferAccess slot base scale :: rest }, []) := by
    show lowerI32Add s = _
    unfold lowerI32Add
    rw [h_stack]
  rw [hl_reduced] at hl
  obtain ⟨hs_eq, hops_eq⟩ :=
    Prod.mk.injEq _ _ _ _ |>.mp ((Option.some.injEq _ _).mp hl)
  have hw' : binI32 eval_u32_wrapping_add ws = some ws' := hw
  obtain ⟨a_w, b_w, ws_rest, h_ws_stack, h_ws_eq⟩ := binI32_some_shape hw'
  have h_stk0 := R.stk.right 0 (.wI32 b_w) (by rw [h_ws_stack]; simp)
  obtain ⟨sv0, hsv0_get, henc0⟩ := h_stk0
  have hs0_eq : s.stack.get? 0 = some (.bufferPtr slot) := by rw [h_stack]; simp
  rw [hs0_eq] at hsv0_get
  have hsv0_eq : sv0 = .bufferPtr slot :=
    ((Option.some.injEq _ _).mp hsv0_get).symm
  rw [hsv0_eq] at henc0
  -- henc0 : b_w.toNat = layout.startAddr slot.
  have h_bw_eq : b_w.toNat = layout.startAddr slot := henc0
  have h_stk1 := R.stk.right 1 (.wI32 a_w) (by rw [h_ws_stack]; simp)
  obtain ⟨sv1, hsv1_get, henc1⟩ := h_stk1
  have hs1_eq : s.stack.get? 1 = some (.scaledIdx base scale) := by rw [h_stack]; simp
  rw [hs1_eq] at hsv1_get
  have hsv1_eq : sv1 = .scaledIdx base scale :=
    ((Option.some.injEq _ _).mp hsv1_get).symm
  rw [hsv1_eq] at henc1
  obtain ⟨b, h_lookup_b, h_aw_eq⟩ := henc1
  refine ⟨kst, ?_, ?_⟩
  · rw [← hops_eq]; simp [evalOps]
  · rw [← hs_eq]; subst h_ws_eq
    refine ⟨?_, R.locs, ?_, ?_, R.injLocals, R.heapRefines, R.currentReg, R.freshCurrent, R.curLocDisj⟩
    · refine ⟨?_, ?_⟩
      · have hlen := R.stk.left
        rw [h_stack, h_ws_stack] at hlen
        simp at hlen
        simp; exact hlen
      · intro j vj hvj
        cases j with
        | zero =>
          simp at hvj; subst hvj
          refine ⟨.bufferAccess slot base scale, by simp, ?_⟩
          show ∃ b', regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b') ∧
                 (eval_u32_wrapping_add a_w b_w).toNat =
                   layout.startAddr slot + b'.toNat * scale
          refine ⟨b, h_lookup_b, ?_⟩
          show (a_w + b_w).toNat = layout.startAddr slot + b.toNat * scale
          exact h_addr_eq a_w b_w b h_lookup_b h_aw_eq h_bw_eq
        | succ j' =>
          have hwsk : ws.stack.get? (j' + 2) = some vj := by
            rw [h_ws_stack]; simpa using hvj
          obtain ⟨svk, hsvk, henck⟩ := R.stk.right (j' + 2) vj hwsk
          have hsk : s.stack.get? (j' + 2) = some svk := hsvk
          rw [h_stack] at hsk
          simp at hsk
          refine ⟨svk, by simpa using hsk, henck⟩
    · refine ⟨?_, R.fresh.right⟩
      intro sv hsv r' hr'
      simp at hsv
      rcases hsv with h_eq | h_in
      · subst h_eq
        simp [SymVal.regs] at hr'
        rw [hr']
        have hbase_in_stack : (.scaledIdx base scale : SymVal) ∈ s.stack := by
          rw [h_stack]; simp
        exact R.fresh.left _ hbase_in_stack base (by simp [SymVal.regs])
      · have hsv_in_s : sv ∈ s.stack := by
          rw [h_stack]; exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ h_in)
        exact R.fresh.left sv hsv_in_s r' hr'
    · intro ir hir sv hsv
      simp at hsv
      rcases hsv with h_eq | h_in
      · subst h_eq
        have hbase_in_stack : (.scaledIdx base scale : SymVal) ∈ s.stack := by
          rw [h_stack]; simp
        have hb_disj := R.aliasFree ir hir _ hbase_in_stack
        simp [SymVal.regs] at hb_disj ⊢
        exact hb_disj
      · have hsv_in_s : sv ∈ s.stack := by
          rw [h_stack]; exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ h_in)
        exact R.aliasFree ir hir sv hsv_in_s

/-- Inversion for `bufferAccess` encoding: the encoded WASM value
    must be `wI32` (every other constructor encodes to `False`). -/
theorem WasmValue.encodes_bufferAccess_wI32_inv
    {v : WasmValue} {layout : BufferLayout} {rf : Quanta.KOps.RegFile}
    {slot : Nat} {base : Reg} {scale : Nat}
    (h : v.encodes layout rf (.bufferAccess slot base scale)) :
    ∃ n : UInt32, v = .wI32 n := by
  cases v with
  | wI32 n => exact ⟨n, rfl⟩
  | wI64 _ => simp [WasmValue.encodes] at h
  | wF32 _ => simp [WasmValue.encodes] at h
  | wF64 _ => simp [WasmValue.encodes] at h

/-- `i32.load` preservation against a recognized `bufferAccess` shape.
    The lowering allocates a fresh register and emits
    `KernelOp.load dst slot base .u32`. KOps eval reads
    `heapLookup heap slot b.toNat` (where `b = regLookup base`); WASM
    eval reads `mem.load_u32 (addr.toNat + offset)`. `HeapRefines`
    bridges them: when `b.toNat < layout.length slot`, both reads
    return the same `vU32 n`.

    Preconditions:
    * `h_offset` — the WASM `offset` immediate is zero. Production
      ignores `offset` in the buffer-access arm because rustc folds
      memory offsets into the byte-offset arithmetic before the
      `i32.load`; this proof faithfully assumes that fold.
    * `h_in_bounds` — for any `b` matching the base reg's value,
      `b.toNat < layout.length slot`. This bounds the heap/memory
      access; outside the layout's length, `HeapRefines` doesn't
      relate the two views. Future kernel-entry composition theorem
      will discharge it from the kernel's array bounds. -/
theorem preservation_i32Load
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout)
    (slot : Nat) (base : Reg) (rest : List SymVal)
    (offset align : Nat)
    (h_stack : s.stack = .bufferAccess slot base 4 :: rest)
    (h_offset : offset = 0)
    (h_in_bounds : ∀ b : UInt32,
       regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
       b.toNat < layout.length slot)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws (.i32Load offset align) = some ws')
    (hl : lowerInstr s (.i32Load offset align) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- 1. Reduce hl using h_stack.
  have hl_reduced : lowerInstr s (.i32Load offset align) =
      some ({ s with nextReg := s.nextReg + 1,
                     stack := .reg s.nextReg .u32 :: rest },
             [.load s.nextReg slot base .u32]) := by
    show lowerI32Load s = _
    unfold lowerI32Load
    rw [h_stack]
    rfl
  rw [hl_reduced] at hl
  obtain ⟨hs_eq, hops_eq⟩ :=
    Prod.mk.injEq _ _ _ _ |>.mp ((Option.some.injEq _ _).mp hl)
  -- 2. Reduce hw via loadI32.
  have hw' : loadI32 ws offset = some ws' := hw
  unfold loadI32 at hw'
  -- ws.stack must be vaddr :: ws_rest.
  rcases hws : ws.stack with _ | ⟨vaddr, ws_rest⟩
  · simp [hws, WasmState.pop] at hw'
  simp [hws, WasmState.pop] at hw'
  -- 3. Derive bufferAccess encoding for vaddr from R.stk; vaddr must be wI32.
  have h_stk0 := R.stk.right 0 vaddr (by rw [hws]; simp)
  obtain ⟨sv0, hsv0_get, henc0⟩ := h_stk0
  have hs0_eq : s.stack.get? 0 = some (.bufferAccess slot base 4) := by
    rw [h_stack]; simp
  rw [hs0_eq] at hsv0_get
  have hsv0_eq : sv0 = .bufferAccess slot base 4 :=
    ((Option.some.injEq _ _).mp hsv0_get).symm
  rw [hsv0_eq] at henc0
  -- vaddr must be wI32 by encodes_bufferAccess_wI32_inv.
  obtain ⟨addr_w, h_vaddr_eq⟩ := WasmValue.encodes_bufferAccess_wI32_inv henc0
  subst h_vaddr_eq
  -- Now henc0 : (wI32 addr_w).encodes layout kst.rf (.bufferAccess slot base 4)
  --          = ∃ b, regLookup kst.rf base = some (vU32 b) ∧ addr_w.toNat = layout.startAddr slot + b.toNat * 4.
  obtain ⟨b, h_lookup_b, h_addr_eq⟩ := henc0
  -- 4. Reduce hw' fully using addr_w being wI32.
  simp at hw'
  rcases hmem : (ws.mem.load_u32 (addr_w.toNat + offset)) with _ | n
  · simp [hmem] at hw'
  simp [hmem, WasmState.push] at hw'
  -- hw' : ws' = { ws with stack := wI32 n :: ws_rest }.
  -- 5. From HeapRefines + in-bounds, get the heap/mem agreement.
  obtain ⟨nh, h_heap_lookup, h_mem_load⟩ :=
    R.heapRefines slot b.toNat (h_in_bounds b h_lookup_b)
  have h_addr_total : addr_w.toNat + offset = layout.startAddr slot + b.toNat * 4 := by
    rw [h_offset, h_addr_eq, Nat.add_zero]
  have h_mem_eq : ws.mem.load_u32 (addr_w.toNat + offset) = some nh := by
    rw [h_addr_total]; exact h_mem_load
  have h_n_eq : n = nh := by
    rw [h_mem_eq] at hmem
    exact ((Option.some.injEq _ _).mp hmem).symm
  -- 6. Compute kst' via the load op. Use vU32 nh (the heap's value);
  -- the StackRefines proof bridges to wI32 n via h_n_eq.
  refine ⟨{ kst with rf := regWrite kst.rf s.nextReg (Quanta.KOps.Value.vU32 nh) }, ?_, ?_⟩
  · rw [← hops_eq]
    simp [evalOps, Quanta.KOps.evalOp, h_lookup_b, h_heap_lookup]
  · rw [← hs_eq]; rw [← hw']
    refine ⟨?_, ?_, ?_, ?_, R.injLocals, ?_, ?_, ?_, ?_⟩
    · -- StackRefines: top wI32 n ↔ .reg s.nextReg .u32; tail past fresh write.
      refine ⟨?_, ?_⟩
      · have hlen := R.stk.left
        rw [h_stack, hws] at hlen
        simp at hlen
        simp; exact hlen
      · intro j vj hvj
        cases j with
        | zero =>
          simp at hvj; subst hvj
          refine ⟨.reg s.nextReg .u32, by simp, ?_⟩
          show regLookup (regWrite kst.rf s.nextReg (Quanta.KOps.Value.vU32 nh)) s.nextReg
                 = some (Quanta.KOps.Value.vU32 n)
          rw [h_n_eq]
          simp [regLookup_regWrite_self]
        | succ j' =>
          have hwsk : ws.stack.get? (j' + 1) = some vj := by
            rw [hws]; simpa using hvj
          obtain ⟨svk, hsvk, henck⟩ := R.stk.right (j' + 1) vj hwsk
          have hsk : s.stack.get? (j' + 1) = some svk := hsvk
          rw [h_stack] at hsk
          simp at hsk
          refine ⟨svk, by simpa using hsk, ?_⟩
          have hsvk_in : svk ∈ s.stack := List.mem_of_get? hsvk
          apply WasmValue.encodes_preserved_of_fresh _ henck
          intro r' hr'
          exact R.fresh.left svk hsvk_in r' hr'
    · -- LocalsRefines: localReg unchanged; lift past fresh write.
      intro k r hfind v hv
      have hpair : (k, r) ∈ s.localReg := List.mem_of_find?_eq_some hfind
      have hr_lt : r < s.nextReg := R.fresh.right (k, r) hpair
      have henc := R.locs k r hfind v hv
      apply WasmValue.encodes_preserved_of_fresh _ henc
      intro r' hr'_in
      simp [SymVal.regs] at hr'_in
      subst hr'_in; exact hr_lt
    · -- Fresh: nextReg bumps by 1; new top is .reg s.nextReg .u32; rest ⊆ s.stack.
      refine ⟨?_, ?_⟩
      · intro sv hsv r' hr'
        simp at hsv
        rcases hsv with h_eq | h_in
        · subst h_eq
          simp [SymVal.regs] at hr'
          subst hr'; exact Nat.lt_succ_self _
        · have hsv_in : sv ∈ s.stack := by
            rw [h_stack]; exact List.mem_cons_of_mem _ h_in
          exact Nat.lt_succ_of_lt (R.fresh.left sv hsv_in r' hr')
      · intro ir hir
        exact Nat.lt_succ_of_lt (R.fresh.right ir hir)
    · -- AliasFree: new top has regs = [s.nextReg]; fresh ≠ any stable_reg.
      intro ir hir sv hsv
      have hir_lt : ir.snd < s.nextReg := R.fresh.right ir hir
      simp at hsv
      rcases hsv with h_eq | h_in
      · subst h_eq
        simp [SymVal.regs]
        exact Nat.ne_of_lt hir_lt
      · have hsv_in : sv ∈ s.stack := by
          rw [h_stack]; exact List.mem_cons_of_mem _ h_in
        exact R.aliasFree ir hir sv hsv_in
    · -- HeapRefines: heap unchanged; mem unchanged.
      exact R.heapRefines
    · -- CurrentRegRefines: currentReg unchanged; lift past fresh write.
      exact CurrentRegRefines_preserved_fresh R.currentReg R.freshCurrent _
    · -- FreshCurrent: nextReg bumps by 1; currentReg unchanged.
      intro ir hir
      exact Nat.lt_succ_of_lt (R.freshCurrent ir hir)
    · -- CurrentLocalDisjoint: currentReg / localReg unchanged.
      exact R.curLocDisj

open Quanta.KOps (vU32) in
/-- `i32.store` preservation against a recognized `bufferAccess`
    shape. Lowering: `popSym` val + addr, `commit` val (handles
    `.reg` / `.i32ConstSym` source shapes), emit
    `KernelOp.store slot base src .u32`. KOps eval writes
    `vU32 val_w` to `(slot, b.toNat)` in the heap; WASM eval writes
    the same `val_w` as 4 little-endian bytes to `mem` at
    `addr_w.toNat + offset`.

    The new `HeapRefines` clause uses the `WasmMem.store_load_*`
    TCB axioms — at the written `(slot, b.toNat)` entry,
    `store_load_same` gives the byte view back; for every other
    in-bounds `(slot', idx')`, `store_load_disjoint` (with the
    layout no-overlap precondition) lifts the old `HeapRefines`,
    while `heapLookup_heapStore_other` lifts the heap projection.

    Preconditions:
    * `kst.broke = false` — required for `evalOps_append` chaining.
    * `h_offset = 0` — production drops the WASM offset.
    * `h_in_bounds : b.toNat < layout.length slot` — store hits a real entry.
    * `h_layout_no_overlap` — for any other in-bounds `(slot', idx')`,
      its 4-byte mem range is disjoint from the store's. Future
      kernel-entry composition theorem will discharge it from the
      layout's inherent non-overlap. -/
theorem preservation_i32Store
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout) (h_kst_ok : kst.broke = false)
    (sv_val : SymVal) (slot : Nat) (base : Reg) (rest : List SymVal)
    (offset align : Nat)
    (h_stack : s.stack = sv_val :: .bufferAccess slot base 4 :: rest)
    (h_offset : offset = 0)
    (h_in_bounds : ∀ b : UInt32,
       regLookup kst.rf base = some (vU32 b) →
       b.toNat < layout.length slot)
    (h_layout_no_overlap : ∀ b : UInt32,
       regLookup kst.rf base = some (vU32 b) →
       ∀ slot' idx',
         idx' < layout.length slot' →
         (slot', idx') ≠ (slot, b.toNat) →
         layout.startAddr slot + b.toNat * 4 + 4 ≤ layout.startAddr slot' + idx' * 4 ∨
         layout.startAddr slot' + idx' * 4 + 4 ≤ layout.startAddr slot + b.toNat * 4)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws (.i32Store offset align) = some ws')
    (hl : lowerInstr s (.i32Store offset align) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- 1. Reduce hl: lowerI32Store via the buffer-access arm.
  have hl' : lowerI32Store s = some (s', ops) := hl
  unfold lowerI32Store at hl'
  simp only [LowerState.popSym, h_stack, Option.bind_eq_bind, Option.some_bind] at hl'
  -- After two popSyms, the popped state has stack := rest.
  -- commit sv_val on it returns (src, s3, opsCommit) (or none if sv_val is an address SymVal).
  rcases hca : LowerState.commit { s with stack := rest } sv_val
      with _ | ⟨src, s3, opsCommit⟩
  · simp [hca] at hl'
  simp [hca] at hl'
  -- After simp, hl' is the stripped conjunction.
  obtain ⟨hs_eq, hops_eq⟩ := hl'
  -- 2. Reduce hw: storeI32 ws offset.
  have hw' : storeI32 ws offset = some ws' := hw
  unfold storeI32 at hw'
  rcases hws : ws.stack with _ | ⟨vval, _ | ⟨vaddr, ws_rest⟩⟩
  · simp [hws, WasmState.pop] at hw'
  · simp [hws, WasmState.pop] at hw'
  simp [hws, WasmState.pop] at hw'
  -- 3. Derive sv_val/bufferAccess encodings from R.stk.
  have h_stk0 := R.stk.right 0 vval (by rw [hws]; simp)
  obtain ⟨sv0, hsv0_get, henc_val⟩ := h_stk0
  have hs0_eq : s.stack.get? 0 = some sv_val := by rw [h_stack]; simp
  rw [hs0_eq] at hsv0_get
  have hsv0_eq : sv0 = sv_val := ((Option.some.injEq _ _).mp hsv0_get).symm
  rw [hsv0_eq] at henc_val
  have h_stk1 := R.stk.right 1 vaddr (by rw [hws]; simp)
  obtain ⟨sv1, hsv1_get, henc_addr⟩ := h_stk1
  have hs1_eq : s.stack.get? 1 = some (.bufferAccess slot base 4) := by rw [h_stack]; simp
  rw [hs1_eq] at hsv1_get
  have hsv1_eq : sv1 = .bufferAccess slot base 4 :=
    ((Option.some.injEq _ _).mp hsv1_get).symm
  rw [hsv1_eq] at henc_addr
  obtain ⟨addr_w, h_vaddr_eq⟩ := WasmValue.encodes_bufferAccess_wI32_inv henc_addr
  subst h_vaddr_eq
  obtain ⟨b, h_lookup_b, h_addr_eq⟩ := henc_addr
  -- 4. Continue reducing hw' (vval must be wI32).
  cases vval with
  | wI32 val_w =>
    simp at hw'
    rcases hmem : ws.mem.store_u32 (addr_w.toNat + offset) val_w with _ | new_mem
    · simp [hmem] at hw'
    simp [hmem] at hw'
    -- hw' : ws' = { ws with mem := new_mem }.
    -- 5. Apply commit_correct: build s_pop / ws_pop / R_pop, then call.
    have h_ws_rest_rest_len : ws_rest.length = rest.length := by
      have hl_orig := R.stk.left
      rw [hws, h_stack] at hl_orig
      simpa using hl_orig
    let s_pop : LowerState := { s with stack := rest }
    let ws_pop : WasmState := { ws with stack := ws_rest }
    have R_pop : Refines ws_pop s_pop kst layout := by
      refine ⟨⟨h_ws_rest_rest_len, ?_⟩, R.locs, ?_, ?_, R.injLocals, R.heapRefines, R.currentReg, R.freshCurrent, R.curLocDisj⟩
      · intro i v hv
        have hrest_get : ws.stack.get? (i + 2) = some v := by
          rw [hws]; simpa using hv
        obtain ⟨svi, hsvi_get, henc⟩ := R.stk.right (i + 2) v hrest_get
        have hlrest_get : rest.get? i = some svi := by
          have h2 : s.stack.get? (i + 2) = some svi := hsvi_get
          rw [h_stack] at h2; simpa using h2
        exact ⟨svi, by simpa using hlrest_get, henc⟩
      · refine ⟨?_, R.fresh.right⟩
        intro sv hsv r hr
        have hsv_in : sv ∈ s.stack := by
          rw [h_stack]; exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hsv)
        exact R.fresh.left sv hsv_in r hr
      · intro ir hir sv hsv
        have hsv_in : sv ∈ s.stack := by
          rw [h_stack]; exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hsv)
        exact R.aliasFree ir hir sv hsv_in
    have h_sv_val_lt : ∀ r ∈ sv_val.regs, r < s.nextReg := by
      intro r hr
      have h_in : sv_val ∈ s.stack := by rw [h_stack]; simp
      exact R.fresh.left sv_val h_in r hr
    have hca' : s_pop.commit sv_val = some (src, s3, opsCommit) := hca
    obtain ⟨kst1, h_evalCommit, R_commit, h_enc_src1, h_lookup_lo, h_s_le_s3, h_src_lt_s3⟩ :=
      commit_correct R_pop h_sv_val_lt hca' henc_val
    have h_kst1_ok : kst1.broke = false := by
      rw [commit_preserves_broke hca' h_evalCommit]; exact h_kst_ok
    -- 6. KOps .store needs lookups for base (in kst1.rf) and src (= val_w).
    have h_base_lt : base < s.nextReg :=
      R.fresh.left (.bufferAccess slot base 4) (by rw [h_stack]; simp) base (by simp [SymVal.regs])
    have h_lookup_b1 : regLookup kst1.rf base = some (vU32 b) := by
      rw [h_lookup_lo base h_base_lt]; exact h_lookup_b
    have h_lookup_src : regLookup kst1.rf src = some (vU32 val_w) := h_enc_src1
    -- 7. Compute kst' via the .store op.
    refine ⟨{ kst1 with heap := Quanta.KOps.heapStore kst1.heap slot b.toNat (vU32 val_w) }, ?_, ?_⟩
    · rw [← hops_eq]
      rw [evalOps_append h_evalCommit h_kst1_ok]
      simp [evalOps, Quanta.KOps.evalOp, h_lookup_b1, h_lookup_src, h_kst1_ok,
            Quanta.KOps.vU32]
    · -- Refines ws' s' kst' layout.
      rw [← hs_eq]; rw [← hw']
      have h_in_bounds_b : b.toNat < layout.length slot := h_in_bounds b h_lookup_b
      have h_no_ovr_b := h_layout_no_overlap b h_lookup_b
      have h_addr_eq_total : addr_w.toNat + offset = layout.startAddr slot + b.toNat * 4 := by
        rw [h_offset, h_addr_eq, Nat.add_zero]
      have h_s3_stack : s3.stack = rest := commit_preserves_stack hca'
      have h_s3_loc : s3.localReg = s_pop.localReg := (commit_preserves_locals hca').1
      -- kst1.heap = kst.heap: commit only writes the regfile.
      have h_heap_eq : kst1.heap = kst.heap := by
        cases sv_val with
        | reg _ _ =>
          have hopss : opsCommit = [] := by
            have := hca'
            simp [LowerState.commit] at this
            exact this.2.2
          rw [hopss] at h_evalCommit
          simp [evalOps] at h_evalCommit
          rw [← h_evalCommit]
        | i32ConstSym n =>
          -- commit emits [.const s_pop.nextReg ...]; src = s_pop.nextReg.
          have hcommit_shape := hca'
          simp [LowerState.commit, LowerState.alloc] at hcommit_shape
          obtain ⟨hsrc_eq, _, hopss⟩ := hcommit_shape
          -- hsrc_eq : s_pop.nextReg = src; hopss : [.const s_pop.nextReg ...] = opsCommit.
          rw [← hopss] at h_evalCommit
          simp [evalOps, Quanta.KOps.evalOp, Quanta.KOps.evalConst] at h_evalCommit
          rw [← h_evalCommit]
        | bufferPtr _ => simp [LowerState.commit] at hca'
        | scaledIdx _ _ => simp [LowerState.commit] at hca'
        | bufferAccess _ _ _ => simp [LowerState.commit] at hca'
      refine ⟨?_, ?_, ?_, ?_, R_commit.injLocals, ?_,
              R_commit.currentReg, R_commit.freshCurrent, R_commit.curLocDisj⟩
      · -- StackRefines: ws_rest matches rest under kst1.rf.
        refine ⟨?_, ?_⟩
        · rw [h_s3_stack]; exact h_ws_rest_rest_len
        · intro j v hv
          have hwsk : ws_pop.stack.get? j = some v := hv
          obtain ⟨svk, hsvk_get, henck⟩ := R_commit.stk.right j v hwsk
          exact ⟨svk, hsvk_get, henck⟩
      · -- LocalsRefines: pass through R_commit.locs (s3.localReg = s_pop.localReg).
        intro k r hfind v hv
        exact R_commit.locs k r hfind v hv
      · -- Fresh: s'.stack = s3.stack, localReg unchanged.
        refine ⟨?_, ?_⟩
        · intro sv hsv r' hr'
          exact R_commit.fresh.left sv hsv r' hr'
        · intro ir hir
          exact R_commit.fresh.right ir hir
      · -- AliasFree: s'.stack = s3.stack, localReg unchanged.
        intro ir hir sv hsv
        exact R_commit.aliasFree ir hir sv hsv
      · -- HeapRefines: split on (slot', idx') = (slot, b.toNat).
        intro slot' idx' h_idx'_lt
        by_cases h_eq_target : (slot', idx') = (slot, b.toNat)
        · -- Target entry: rw the equation and use heapStore_self + store_load_same.
          obtain ⟨h_slot_eq, h_idx_eq⟩ := Prod.mk.injEq _ _ _ _ |>.mp h_eq_target
          refine ⟨val_w, ?_, ?_⟩
          · rw [h_slot_eq, h_idx_eq, Quanta.KOps.heapLookup_heapStore_self]
            rfl
          · rw [h_slot_eq, h_idx_eq, ← h_addr_eq_total]
            exact WasmMem.store_load_same _ _ _ _ hmem
        · -- Disjoint entry.
          obtain ⟨n_old, h_heap_old, h_mem_old⟩ := R.heapRefines slot' idx' h_idx'_lt
          refine ⟨n_old, ?_, ?_⟩
          · -- heapLookup new_heap slot' idx' = heapLookup kst1.heap slot' idx' (heapStore_other)
            --                                = heapLookup kst.heap slot' idx' (h_heap_eq)
            --                                = some (vU32 n_old) (h_heap_old).
            rw [Quanta.KOps.heapLookup_heapStore_other _ _ _ _ _ _ h_eq_target,
                h_heap_eq]
            exact h_heap_old
          · have h_disj := h_no_ovr_b slot' idx' h_idx'_lt h_eq_target
            have h_disj_total : addr_w.toNat + offset + 4 ≤ layout.startAddr slot' + idx' * 4 ∨
                                layout.startAddr slot' + idx' * 4 + 4 ≤ addr_w.toNat + offset := by
              rw [h_addr_eq_total]; exact h_disj
            rw [WasmMem.store_load_disjoint _ _ _ _ hmem _ h_disj_total]
            exact h_mem_old
  | wI64 _ => simp at hw'
  | wF32 _ => simp at hw'
  | wF64 _ => simp at hw'

-- ════════════════════════════════════════════════════════════════════
-- Generic i32 comparison preservation (instantiates for the 6-op family)
--
-- Mirrors `preservation_i32Bin_generic` but the lowering emits TWO
-- ops — `.cmp` (vBool result at `s.nextReg`) followed by
-- `.cast .bool .u32` (vU32 0/1 at `s.nextReg + 1`). The two-op
-- `evalOps` requires `kst.broke = false` so the inter-op short-circuit
-- doesn't fire; binops only emit one op so they didn't need this.
-- ════════════════════════════════════════════════════════════════════

open Quanta.KOps (vU32) in
/-- Generic preservation for any WASM i32 comparison the lowering pass
    handles. Takes:
    * `instr`    — the WASM instruction.
    * `p_w`      — the u32 → u32 → Bool predicate the WASM semantics dispatches.
    * `op_k`     — the matching `KOps.CmpOp` the lowering emits.
    * `h_w`      — `evalInstr s instr = cmpI32 p_w s` (by rfl per-arm).
    * `h_l`      — `lowerInstr s instr = lowerI32Cmp s op_k` (by rfl).
    * `h_agree`  — KOps `evalCmpOp op_k (vU32 av) (vU32 bv) = some (vBool (p_w av bv))`.

    Each of the 6 i32-cmp preservation theorems below is one line.

    Same `popSym + commit` chain as `preservation_i32Bin_generic`,
    plus a final 2-op `cmp + cast` emission whose result reg
    `s4.nextReg + 1` is what the new stack top points at. -/
theorem preservation_i32Cmp_generic
    (instr : WasmInstr) (p_w : UInt32 → UInt32 → Bool)
    (op_k : Quanta.KOps.CmpOp)
    (h_w : ∀ s, evalInstr s instr = cmpI32 p_w s)
    (h_l : ∀ s, lowerInstr s instr = lowerI32Cmp s op_k)
    (h_agree : ∀ av bv,
       Quanta.KOps.evalCmpOp op_k (vU32 av) (vU32 bv)
         = some (Quanta.KOps.Value.vBool (p_w av bv)))
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout) (h_kst_ok : kst.broke = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws instr = some ws')
    (hl : lowerInstr s instr = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  rw [h_w] at hw
  rw [h_l] at hl
  obtain ⟨av, bv, rest, hwstack, hws_eq⟩ := cmpI32_some_shape hw
  obtain ⟨svb, sva, lrest, ra, s3, opsA, rb, s4, opsB, hlstack, hca, hcb,
          h_s4_stack, h_s4_lr, h_s4_lt, h_s_le_s4, hs_eq, hops_eq⟩ :=
    lowerI32Cmp_some_shape hl
  -- Operand encodings in kst.rf.
  have h_enc_svb : (WasmValue.wI32 bv).encodes layout kst.rf svb := by
    have hb := R.stk.right 0 (.wI32 bv) (by rw [hwstack]; simp)
    obtain ⟨sv0, hsv0_get, henc⟩ := hb
    have hs0 : s.stack.get? 0 = some svb := by rw [hlstack]; simp
    rw [hs0] at hsv0_get
    have h_eq : svb = sv0 := (Option.some.injEq _ _).mp hsv0_get
    rw [h_eq]; exact henc
  have h_enc_sva : (WasmValue.wI32 av).encodes layout kst.rf sva := by
    have ha := R.stk.right 1 (.wI32 av) (by rw [hwstack]; simp)
    obtain ⟨sv1, hsv1_get, henc⟩ := ha
    have hs1 : s.stack.get? 1 = some sva := by rw [hlstack]; simp
    rw [hs1] at hsv1_get
    have h_eq : sva = sv1 := (Option.some.injEq _ _).mp hsv1_get
    rw [h_eq]; exact henc
  -- Stack-membership and Fresh-bound on each operand.
  have h_svb_in : svb ∈ s.stack := by rw [hlstack]; simp
  have h_sva_in : sva ∈ s.stack := by rw [hlstack]; simp
  have h_svb_lt : ∀ r ∈ svb.regs, r < s.nextReg :=
    fun r hr => R.fresh.left svb h_svb_in r hr
  have h_sva_lt : ∀ r ∈ sva.regs, r < s.nextReg :=
    fun r hr => R.fresh.left sva h_sva_in r hr
  -- Length agreement on the popped suffix.
  have h_rest_lrest_len : rest.length = lrest.length := by
    have hl_orig := R.stk.left
    rw [hwstack, hlstack] at hl_orig
    simpa using hl_orig
  -- Refines bundle for the popped state s_pop / ws_pop.
  let s_pop : LowerState :=
    { nextReg := s.nextReg, stack := lrest,
      localReg := s.localReg, localTy := s.localTy,
      bufferSlots := s.bufferSlots, currentReg := s.currentReg }
  let ws_pop : WasmState :=
    { ws with stack := rest }
  have R_pop : Refines ws_pop s_pop kst layout := by
    refine ⟨⟨h_rest_lrest_len, ?_⟩, R.locs, ?_, ?_, R.injLocals, R.heapRefines, R.currentReg, R.freshCurrent, R.curLocDisj⟩
    · intro i v hv
      have hrest_get : ws.stack.get? (i + 2) = some v := by
        rw [hwstack]; simpa using hv
      obtain ⟨svi, hsvi_get, henc⟩ := R.stk.right (i + 2) v hrest_get
      have hlrest_get : lrest.get? i = some svi := by
        have h2 : s.stack.get? (i + 2) = some svi := hsvi_get
        rw [hlstack] at h2; simpa using h2
      exact ⟨svi, by simpa using hlrest_get, henc⟩
    · refine ⟨?_, R.fresh.right⟩
      intro sv hsv r hr
      have hsv_in : sv ∈ s.stack := by
        rw [hlstack]; exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hsv)
      exact R.fresh.left sv hsv_in r hr
    · intro ir hir sv hsv
      have hsv_in : sv ∈ s.stack := by
        rw [hlstack]; exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hsv)
      exact R.aliasFree ir hir sv hsv_in
  -- First commit: sva → ra. Emits opsA, evolves kst → kst1.
  obtain ⟨kst1, h_evalA, R1, h_enc_ra1, h_lookupA, h_s_le_s3, h_ra_lt_s3⟩ :=
    commit_correct R_pop h_sva_lt hca h_enc_sva
  have h_kst1_ok : kst1.broke = false := by
    rw [commit_preserves_broke hca h_evalA]; exact h_kst_ok
  have h_enc_svb1 : (WasmValue.wI32 bv).encodes layout kst1.rf svb :=
    WasmValue.encodes_preserved_of_lookup_eq
      (fun r hr => h_lookupA r (h_svb_lt r hr)) h_enc_svb
  have h_svb_lt_s3 : ∀ r ∈ svb.regs, r < s3.nextReg :=
    fun r hr => Nat.lt_of_lt_of_le (h_svb_lt r hr) h_s_le_s3
  -- Second commit: svb → rb on R1. Emits opsB, evolves kst1 → kst2.
  obtain ⟨kst2, h_evalB, R2, h_enc_rb2, h_lookupB, h_s3_le_s4, h_rb_lt_s4⟩ :=
    commit_correct R1 h_svb_lt_s3 hcb h_enc_svb1
  have h_kst2_ok : kst2.broke = false := by
    rw [commit_preserves_broke hcb h_evalB]; exact h_kst1_ok
  have h_enc_ra2 : (WasmValue.wI32 av).encodes layout kst2.rf (.reg ra .u32) :=
    WasmValue.encodes_preserved_of_lookup_eq
      (fun r hr => by
        simp [SymVal.regs] at hr
        rw [hr]
        exact h_lookupB ra h_ra_lt_s3)
      h_enc_ra1
  have h_lookup_ra : regLookup kst2.rf ra = some (vU32 av) := h_enc_ra2
  have h_lookup_rb : regLookup kst2.rf rb = some (vU32 bv) := h_enc_rb2
  -- Final kst' has both writes applied (vBool at s4.nextReg, vU32 at s4.nextReg+1).
  -- broke := false matches the simp output (h_kst2_ok normalizes kst2.broke).
  refine ⟨{ rf := regWrite (regWrite kst2.rf s4.nextReg
                              (Quanta.KOps.Value.vBool (p_w av bv)))
                            (s4.nextReg + 1)
                            (vU32 (if p_w av bv then 1 else 0)),
            heap := kst2.heap, dispatch := kst2.dispatch, broke := false }, ?_, ?_⟩
  · -- evalOps 0 kst (opsA ++ opsB ++ [cmp …, cast …]) = some kst3.
    subst hops_eq
    rw [show opsA ++ opsB ++ [KernelOp.cmp s4.nextReg ra rb op_k Quanta.KOps.Scalar.bool,
                              KernelOp.cast (s4.nextReg + 1) s4.nextReg
                                Quanta.KOps.Scalar.bool Quanta.KOps.Scalar.u32]
          = opsA ++ (opsB ++ [KernelOp.cmp s4.nextReg ra rb op_k Quanta.KOps.Scalar.bool,
                              KernelOp.cast (s4.nextReg + 1) s4.nextReg
                                Quanta.KOps.Scalar.bool Quanta.KOps.Scalar.u32]) from
        by rw [List.append_assoc]]
    rw [evalOps_append h_evalA h_kst1_ok]
    rw [evalOps_append h_evalB h_kst2_ok]
    -- Now reduce the 2-op `evalOps 0 kst2 [cmp …, cast …]`.
    simp [evalOps, Quanta.KOps.evalOp, h_lookup_ra, h_lookup_rb, h_agree,
          regLookup_regWrite_self, Quanta.KOps.evalCast, h_kst2_ok]
  · -- Refines ws' s' kst3 layout.
    subst hs_eq; subst hws_eq
    -- h_lift: any encoding whose SymVal-regs are < s4.nextReg lifts past the two writes.
    let h_lift : ∀ (sv : SymVal) (v : WasmValue),
        (∀ r ∈ sv.regs, r < s4.nextReg) →
        v.encodes layout kst2.rf sv →
        v.encodes layout (regWrite (regWrite kst2.rf s4.nextReg
                              (Quanta.KOps.Value.vBool (p_w av bv)))
                            (s4.nextReg + 1)
                            (Quanta.KOps.Value.vU32 (if p_w av bv then 1 else 0))) sv :=
      fun sv v h_lt henc =>
        WasmValue.encodes_preserved_of_fresh
          (fun r hr => Nat.lt_succ_of_lt (h_lt r hr))
          (WasmValue.encodes_preserved_of_fresh h_lt henc)
    refine ⟨?_, ?_, ?_, ?_, ?_, R2.heapRefines, ?_, ?_, ?_⟩
    · -- StackRefines on (wI32 (if p_w av bv then 1 else 0) :: rest, .reg (s4.nextReg+1) .u32 :: lrest).
      refine ⟨?_, ?_⟩
      · simp; exact h_rest_lrest_len
      · intro j v hv
        cases j with
        | zero =>
          simp at hv
          refine ⟨SymVal.reg (s4.nextReg + 1) .u32, by simp, ?_⟩
          subst hv
          show regLookup
                 (regWrite (regWrite kst2.rf s4.nextReg
                              (Quanta.KOps.Value.vBool (p_w av bv)))
                            (s4.nextReg + 1)
                            (vU32 (if p_w av bv then 1 else 0)))
                 (s4.nextReg + 1) = some (vU32 (if p_w av bv then 1 else 0))
          simp [regLookup_regWrite_self]
        | succ k =>
          have hk : ws_pop.stack.get? k = some v := by
            show rest.get? k = some v
            simpa using hv
          obtain ⟨svk, hsvk_get, henc⟩ := R2.stk.right k v hk
          have h_s4_get : s4.stack.get? k = some svk := hsvk_get
          rw [h_s4_stack] at h_s4_get
          refine ⟨svk, by simpa using h_s4_get, ?_⟩
          have hsvk_in : svk ∈ s4.stack := List.mem_of_get? hsvk_get
          exact h_lift svk v (fun r hr => R2.fresh.left svk hsvk_in r hr) henc
    · -- LocalsRefines: localReg unchanged through commits + cmp+cast.
      intro i r hfind v hv
      rw [← h_s4_lr] at hfind
      have hpair : (i, r) ∈ s4.localReg := List.mem_of_find?_eq_some hfind
      have hr_lt : r < s4.nextReg := R2.fresh.right (i, r) hpair
      have henc := R2.locs i r hfind v hv
      apply h_lift _ _ _ henc
      intro r' hr'_in
      simp [SymVal.regs] at hr'_in
      subst hr'_in; exact hr_lt
    · -- Fresh on s' — top reg is s4.nextReg + 1, lrest ⊆ s4.stack, locals from s4.
      refine ⟨?_, ?_⟩
      · intro sv hsv r' hr'
        simp at hsv
        rcases hsv with h_eq | h_in
        · subst h_eq
          simp [SymVal.regs] at hr'
          subst hr'; exact Nat.lt_succ_self _
        · have hsv_in_s4 : sv ∈ s4.stack := by rw [h_s4_stack]; exact h_in
          have : r' < s4.nextReg := R2.fresh.left sv hsv_in_s4 r' hr'
          exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt this)
      · intro ir hir
        rw [← h_s4_lr] at hir
        have : ir.snd < s4.nextReg := R2.fresh.right ir hir
        exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt this)
    · -- AliasFree on s'.
      intro ir hir sv hsv
      rw [← h_s4_lr] at hir
      have hir_lt : ir.snd < s4.nextReg := R2.fresh.right ir hir
      simp at hsv
      rcases hsv with h_eq | h_in
      · subst h_eq
        simp [SymVal.regs]
        exact Nat.ne_of_lt (Nat.lt_succ_of_lt hir_lt)
      · have hsv_in_s4 : sv ∈ s4.stack := by rw [h_s4_stack]; exact h_in
        exact R2.aliasFree ir hir sv hsv_in_s4
    · -- InjectiveLocals: localReg unchanged through commits.
      intro p q hp hq
      rw [← h_s4_lr] at hp hq
      exact R2.injLocals p q hp hq
    · -- CurrentRegRefines: s4.currentReg = s.currentReg (commits
      -- preserve), then lift past TWO fresh writes (at s4.nextReg
      -- and s4.nextReg + 1).
      have h_s4_cur : s4.currentReg = s.currentReg := by
        rw [commit_preserves_currentReg hcb, commit_preserves_currentReg hca]
      have h_lift1 := CurrentRegRefines_preserved_fresh R2.currentReg
        R2.freshCurrent (Quanta.KOps.Value.vBool (p_w av bv))
      have h_freshCurrent_bump1 : ∀ ir ∈ s4.currentReg, ir.snd < s4.nextReg + 1 :=
        fun ir hir => Nat.lt_succ_of_lt (R2.freshCurrent ir hir)
      have h_lift2 := CurrentRegRefines_preserved_fresh h_lift1
        h_freshCurrent_bump1 (Quanta.KOps.Value.vU32 (if p_w av bv then 1 else 0))
      show CurrentRegRefines layout ws.locals s.currentReg _
      rw [← h_s4_cur]
      exact h_lift2
    · -- FreshCurrent: nextReg bumps by 2; currentReg = s.currentReg.
      intro ir hir
      have h_s4_cur : s4.currentReg = s.currentReg := by
        rw [commit_preserves_currentReg hcb, commit_preserves_currentReg hca]
      rw [← h_s4_cur] at hir
      exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt (R2.freshCurrent ir hir))
    · -- CurrentLocalDisjoint: currentReg/localReg unchanged through commits and twin writes.
      intro p q hp hq hpq
      have h_s4_cur : s4.currentReg = s.currentReg := by
        rw [commit_preserves_currentReg hcb, commit_preserves_currentReg hca]
      rw [← h_s4_cur] at hp
      rw [← h_s4_lr] at hq
      exact R2.curLocDisj p q hp hq hpq

-- ── Per-op specializations (6 cmps) ────────────────────────────────────

def preservation_i32Eq :=
  preservation_i32Cmp_generic .i32Eq (· == ·) .eq
    (fun _ => rfl) (fun _ => rfl)
    (by intro av bv; rfl)

def preservation_i32Ne :=
  preservation_i32Cmp_generic .i32Ne (· != ·) .ne
    (fun _ => rfl) (fun _ => rfl)
    (by intro av bv; rfl)

def preservation_i32LtU :=
  preservation_i32Cmp_generic .i32LtU (· < ·) .lt
    (fun _ => rfl) (fun _ => rfl)
    (by intro av bv; rfl)

def preservation_i32LeU :=
  preservation_i32Cmp_generic .i32LeU (· <= ·) .le
    (fun _ => rfl) (fun _ => rfl)
    (by intro av bv; rfl)

def preservation_i32GtU :=
  preservation_i32Cmp_generic .i32GtU (· > ·) .gt
    (fun _ => rfl) (fun _ => rfl)
    (by intro av bv; rfl)

def preservation_i32GeU :=
  preservation_i32Cmp_generic .i32GeU (· >= ·) .ge
    (fun _ => rfl) (fun _ => rfl)
    (by intro av bv; rfl)

-- ════════════════════════════════════════════════════════════════════
-- Slice 3 follow-up: localSet preservation
--
-- Lowering pops `src` from the symbolic stack, looks up local `i`'s
-- stable register (allocating a fresh one on first write), and emits a
-- single `Copy { dst, src }`. WASM pops `v_w` and writes
-- `locals[i] := v_w`. Two outer cases on `lookupLocal i`:
--   * `some entry` (existing dst) — `dst = entry.snd` is already in
--     `s.localReg`; AliasFree gives `dst ∉ stack regs`, InjLocals
--     gives `dst ≠ r` for distinct local entries.
--   * `none` (fresh dst) — `dst = s.nextReg`; freshness gives `dst >
--     every stack reg`, so the regWrite at dst preserves all prior
--     stack/local encodings.
-- ════════════════════════════════════════════════════════════════════

open Quanta.KOps (vU32) in
/-- `local.set i` preservation. Stage 3 dual-Copy emission:
    `opsCommit ++ [.copy fresh src, .copy stable fresh]` plus
    `setCurrentReg i fresh`.

    Proof structure: commit (via commit_correct) materializes the
    popped SymVal into `src`. The two Copy ops compose into two
    regWrites: at `fresh` (= s2.nextReg) and at `stable` (the
    local's merge anchor — either reused or freshly allocated).
    The post-state Refines tracks both writes via repeated
    encodes_preserved_of_fresh / _of_disjoint lifts. -/
theorem preservation_localSet (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout) (h_kst_ok : kst.broke = false)
    (i : Nat)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws (.localSet i) = some ws')
    (hl : lowerInstr s (.localSet i) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- WASM side: pop v_w from ws.stack, then setLocal i v_w.
  simp only [evalInstr, WasmState.pop,
             Option.bind_eq_bind, Option.bind, pure] at hw
  rcases hws_stack : ws.stack with _ | ⟨v_w, rest⟩
  · simp [hws_stack] at hw
  simp only [hws_stack, WasmState.setLocal] at hw
  by_cases hbound : i < List.length ws.locals
  case neg => simp [if_neg hbound] at hw
  simp only [if_pos hbound] at hw
  have hws'_eq : ws' = { locals := ws.locals.set i v_w, stack := rest,
                          mem := ws.mem, halted := ws.halted,
                          branchTarget := ws.branchTarget } :=
    ((Option.some.injEq _ _).mp hw).symm
  subst hws'_eq
  -- Lean side: popSym first (always succeeds for non-empty stack), then commit.
  unfold lowerInstr at hl
  rcases hls_stack : s.stack with _ | ⟨sva, lrest⟩
  · simp [hls_stack, LowerState.popSym] at hl
  simp only [hls_stack, LowerState.popSym, Option.bind_eq_bind, Option.some_bind] at hl
  rcases hca : ({s with stack := lrest} : LowerState).commit sva
      with _ | ⟨src, s2, opsCommit⟩
  · simp [hca] at hl
  simp only [hca, Option.some_bind] at hl
  -- v_w must be wI32 (encoded on stack).
  have hv_enc : v_w.encodes layout kst.rf sva := by
    have hb := R.stk.right 0 v_w (by rw [hws_stack]; simp)
    obtain ⟨sv0, hsv0_get, henc⟩ := hb
    have hs0 : s.stack.get? 0 = some sva := by rw [hls_stack]; simp
    rw [hs0] at hsv0_get
    have h_eq : sva = sv0 := (Option.some.injEq _ _).mp hsv0_get
    rw [h_eq]; exact henc
  obtain ⟨n_w, hv_w_eq⟩ : ∃ n_w, v_w = WasmValue.wI32 n_w := by
    cases v_w with
    | wI32 n_w => exact ⟨n_w, rfl⟩
    | wI64 _ => cases sva <;> simp [WasmValue.encodes] at hv_enc
    | wF32 _ => cases sva <;> simp [WasmValue.encodes] at hv_enc
    | wF64 _ => cases sva <;> simp [WasmValue.encodes] at hv_enc
  subst hv_w_eq
  -- sva is on s.stack, its regs are < s.nextReg.
  have h_sva_in : sva ∈ s.stack := by rw [hls_stack]; simp
  have h_sva_lt : ∀ r ∈ sva.regs, r < s.nextReg :=
    fun r hr => R.fresh.left sva h_sva_in r hr
  -- Build R_pop : Refines for the popped state.
  have h_rest_lrest_len : rest.length = lrest.length := by
    have hl_orig := R.stk.left
    rw [hws_stack, hls_stack] at hl_orig
    simpa using hl_orig
  let s_pop : LowerState :=
    { nextReg := s.nextReg, stack := lrest,
      localReg := s.localReg, localTy := s.localTy,
      bufferSlots := s.bufferSlots, currentReg := s.currentReg }
  let ws_pop : WasmState := { ws with stack := rest }
  have R_pop : Refines ws_pop s_pop kst layout := by
    refine ⟨⟨h_rest_lrest_len, ?_⟩, R.locs, ?_, ?_, R.injLocals,
            R.heapRefines, R.currentReg, R.freshCurrent, R.curLocDisj⟩
    · intro k v hv
      have hrest_get : ws.stack.get? (k + 1) = some v := by
        rw [hws_stack]; simpa using hv
      obtain ⟨svk, hsvk_get, henc⟩ := R.stk.right (k + 1) v hrest_get
      have hlrest_get : lrest.get? k = some svk := by
        have h2 : s.stack.get? (k + 1) = some svk := hsvk_get
        rw [hls_stack] at h2; simpa using h2
      exact ⟨svk, by simpa using hlrest_get, henc⟩
    · refine ⟨?_, R.fresh.right⟩
      intro sv hsv r hr
      have hsv_in : sv ∈ s.stack := by
        rw [hls_stack]; exact List.mem_cons_of_mem _ hsv
      exact R.fresh.left sv hsv_in r hr
    · intro ir hir sv hsv
      have hsv_in : sv ∈ s.stack := by
        rw [hls_stack]; exact List.mem_cons_of_mem _ hsv
      exact R.aliasFree ir hir sv hsv_in
  -- commit gives us src + Refines on s2.
  obtain ⟨kst1, h_evalC, R1, h_enc_src1, h_lookupC, _h_s_le_s2, h_src_lt_s2⟩ :=
    commit_correct R_pop h_sva_lt hca hv_enc
  have h_kst1_ok : kst1.broke = false := by
    rw [commit_preserves_broke hca h_evalC]; exact h_kst_ok
  have h_src_lookup : regLookup kst1.rf src = some (Quanta.KOps.Value.vU32 n_w) :=
    h_enc_src1
  have h_s2_lr : s2.localReg = s.localReg := (commit_preserves_locals hca).1
  have h_s2_lt : s2.localTy  = s.localTy  := (commit_preserves_locals hca).2
  have h_s2_stack : s2.stack = lrest := commit_preserves_stack hca
  have h_s2_cr : s2.currentReg = s.currentReg := by
    have h := commit_preserves_currentReg hca
    -- h : s2.currentReg = ({s with stack := lrest}).currentReg = s.currentReg
    exact h
  -- Now branch on lookupLocal of s3 := { s2 with nextReg := s2.nextReg + 1 } (post-alloc fresh).
  -- alloc returns (s2.nextReg, { s2 with nextReg := s2.nextReg + 1 }).
  -- lookupLocal s3 i looks at s3.localReg = s2.localReg = s.localReg.
  simp only [LowerState.lookupLocal, LowerState.lookupLocalTy, LowerState.alloc,
             LowerState.setLocalReg, LowerState.setCurrentReg,
             Option.bind_eq_bind, Option.bind, pure] at hl
  rw [h_s2_lt, h_s2_lr] at hl
  -- The fresh reg is `s2.nextReg`. The dual-Copy ops are
  --   [.copy s2.nextReg src, .copy stable s2.nextReg]
  -- evaluated against kst1 → kst_after_fresh → kst'.
  let fresh : Reg := s2.nextReg
  -- Encoding of source value at kst1's regWrite-progressed regfile.
  -- After .copy fresh src, the regfile has fresh ↦ vU32 n_w. After
  -- .copy stable fresh, also stable ↦ vU32 n_w.
  rcases hreg_find : s.localReg.find? (fun p => p.fst = i) with _ | entry
  -- Case B: first write, allocate new stable too. lookupLocal returns none.
  · simp [hreg_find] at hl
    obtain ⟨hs_eq, hops_eq⟩ := hl
    -- stable = s2.nextReg + 1 (allocated after fresh).
    let stable : Reg := s2.nextReg + 1
    -- kst_after_fresh = kst1 with rf := regWrite kst1.rf fresh (vU32 n_w).
    let kst_after_fresh : Quanta.KOps.State :=
      { kst1 with rf := regWrite kst1.rf fresh (vU32 n_w) }
    -- kst' = kst_after_fresh with rf := regWrite kst_after_fresh.rf stable (vU32 n_w).
    let kst' : Quanta.KOps.State :=
      { kst_after_fresh with rf := regWrite kst_after_fresh.rf stable (vU32 n_w) }
    refine ⟨kst', ?_, ?_⟩
    · -- evalOps 0 kst (opsCommit ++ [zinit, .copy fresh src, .copy stable fresh]) = some kst'.
      subst hops_eq
      -- Frame-0 zero-init absorption: the leading `Const fresh _` is
      -- overwritten by the next `Copy fresh src` (and src ≠ fresh since
      -- src < s2.nextReg = fresh), so it drops out before the existing
      -- dual-Copy reasoning. Peel `opsCommit` (→ kst1, broke = false),
      -- absorb the const, glue back.
      have h_src_ne_fresh : src ≠ fresh :=
        Nat.ne_of_lt h_src_lt_s2
      rw [evalOps_append h_evalC h_kst1_ok]
      rw [evalOps_const_copy_absorb h_src_ne_fresh h_kst1_ok]
      rw [← evalOps_append h_evalC h_kst1_ok]
      -- Step 1: evalOps over opsCommit gives kst1 (h_evalC).
      -- Step 2: evalOps over [.copy fresh src] from kst1 gives kst_after_fresh.
      have h_evalC1 : evalOps 0 kst1 [.copy fresh src] = some kst_after_fresh := by
        simp only [evalOps, Quanta.KOps.evalOp]
        rw [h_src_lookup]
        simp only [Option.bind_eq_bind, Option.bind, Option.some_bind, pure, h_kst1_ok]
        rw [if_neg (by decide : ¬ (false = true))]
        congr 1
        show ({ kst1 with rf := regWrite kst1.rf fresh (vU32 n_w),
                          broke := false } : Quanta.KOps.State) = kst_after_fresh
        rw [show (false : Bool) = kst1.broke from h_kst1_ok.symm]
      have h_kaf_ok : kst_after_fresh.broke = false := h_kst1_ok
      -- Step 3: evalOps over [.copy stable fresh] from kst_after_fresh gives kst'.
      have h_lookup_fresh :
          regLookup kst_after_fresh.rf fresh = some (vU32 n_w) := by
        show regLookup (regWrite kst1.rf fresh (vU32 n_w)) fresh = _
        rw [regLookup_regWrite_self]
      have h_evalC2 : evalOps 0 kst_after_fresh [.copy stable fresh] = some kst' := by
        simp only [evalOps, Quanta.KOps.evalOp]
        rw [h_lookup_fresh]
        simp only [Option.bind_eq_bind, Option.bind, Option.some_bind, pure, h_kaf_ok]
        rw [if_neg (by decide : ¬ (false = true))]
        congr 1
        show ({ kst_after_fresh with
                  rf := regWrite kst_after_fresh.rf stable (vU32 n_w),
                  broke := false } : Quanta.KOps.State) = kst'
        rw [show (false : Bool) = kst_after_fresh.broke from h_kaf_ok.symm]
      -- Compose: evalOps 0 kst (opsCommit ++ [copy fresh src, copy stable fresh]) = kst'.
      rw [show (opsCommit ++ [KernelOp.copy fresh src, KernelOp.copy stable fresh])
            = opsCommit ++ [KernelOp.copy fresh src] ++ [KernelOp.copy stable fresh] from by
            simp [List.append_assoc]]
      rw [evalOps_append (evalOps_append h_evalC h_kst1_ok ▸ h_evalC1) h_kaf_ok]
      exact h_evalC2
    · -- Refines ws' s' kst' layout.
      subst hs_eq
      -- kst'.rf = regWrite (regWrite kst1.rf fresh v) stable v where v = vU32 n_w.
      -- s'.localReg = (i, stable) :: filter(≠i) s.localReg (since hreg_find = none).
      -- s'.currentReg = (i, fresh) :: filter(≠i) s2.currentReg = ... :: filter(≠i) s.currentReg.
      -- s'.nextReg = s2.nextReg + 1 + 1.
      have h_stable_gt_fresh : fresh < stable := Nat.lt_succ_self _
      have h_fresh_ge_s2 : s2.nextReg ≤ fresh := Nat.le_refl _
      have h_stable_ge_s2 : s2.nextReg ≤ stable := Nat.le_succ _
      -- Lookup helpers in kst'.rf.
      have h_lookup_stable_kst' : regLookup kst'.rf stable = some (vU32 n_w) := by
        show regLookup (regWrite kst_after_fresh.rf stable (vU32 n_w)) stable = _
        rw [regLookup_regWrite_self]
      have h_lookup_fresh_kaf :
          regLookup kst_after_fresh.rf fresh = some (vU32 n_w) := by
        show regLookup (regWrite kst1.rf fresh (vU32 n_w)) fresh = _
        rw [regLookup_regWrite_self]
      have h_lookup_fresh_kst' : regLookup kst'.rf fresh = some (vU32 n_w) := by
        show regLookup (regWrite kst_after_fresh.rf stable (vU32 n_w)) fresh = _
        rw [regLookup_regWrite_of_ne _ stable fresh _ (Nat.ne_of_lt h_stable_gt_fresh)]
        exact h_lookup_fresh_kaf
      refine ⟨?_, ?_, ?_, ?_, ?_, R1.heapRefines, ?_, ?_, ?_⟩
      · -- StackRefines on ws'.stack = rest, s'.stack = s2.stack = lrest.
        refine ⟨?_, ?_⟩
        · show rest.length = s2.stack.length
          rw [h_s2_stack]; simpa using h_rest_lrest_len
        · intro j v hv
          have hk : ws_pop.stack.get? j = some v := by
            show rest.get? j = some v; simpa using hv
          obtain ⟨svj, hsvj_get, henc⟩ := R1.stk.right j v hk
          refine ⟨svj, by simpa using hsvj_get, ?_⟩
          have hsvj_in : svj ∈ s2.stack := List.mem_of_get? hsvj_get
          -- Lift past both regWrites — all regs in svj are < s2.nextReg ≤ fresh, stable.
          apply WasmValue.encodes_preserved_of_fresh _ _
          · intro r hr
            have hr_s2 : r < s2.nextReg := R1.fresh.left svj hsvj_in r hr
            exact Nat.lt_of_lt_of_le hr_s2 h_stable_ge_s2
          · apply WasmValue.encodes_preserved_of_fresh _ henc
            intro r hr
            exact R1.fresh.left svj hsvj_in r hr
      · -- LocalsRefines on s'.localReg = (i, stable) :: filter(≠i) s.localReg.
        intro k r hfind v hv
        by_cases hki : k = i
        · subst hki
          change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((k, stable) :: List.filter (fun p => !decide (p.fst = k)) s.localReg)
                 = some (k, r) at hfind
          change (ws.locals.set k (WasmValue.wI32 n_w)).get? k = some v at hv
          rw [List.find?_cons] at hfind
          simp only [show decide ((k, stable).fst = k) = true from by simp] at hfind
          injection hfind with h_pair
          have hr_eq : stable = r := (Prod.ext_iff.mp h_pair).2
          subst hr_eq
          have hv_eq : v = WasmValue.wI32 n_w := by
            have hget : (ws.locals.set k (.wI32 n_w)).get? k =
                        some (WasmValue.wI32 n_w) := by
              rw [List.get?_eq_getElem?]
              exact List.getElem?_set_self (by simpa using hbound)
            rw [hget] at hv
            exact ((Option.some.injEq _ _).mp hv).symm
          subst hv_eq
          simp [WasmValue.encodes]; exact h_lookup_stable_kst'
        · change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((i, stable) :: List.filter (fun p => !decide (p.fst = i)) s.localReg)
                 = some (k, r) at hfind
          rw [find?_setLocalReg_ne _ i k _ hki] at hfind
          have hv_old : ws.locals.get? k = some v := by
            rw [List.get?_eq_getElem?] at hv ⊢
            rw [List.getElem?_set_ne (Ne.symm hki)] at hv
            exact hv
          have hfind_s2 : s2.localReg.find? (fun p => p.fst = k) = some (k, r) := by
            rw [h_s2_lr]; exact hfind
          have henc := R1.locs k r hfind_s2 v hv_old
          have hr_lt : r < s2.nextReg := by
            have hpair : (k, r) ∈ s2.localReg :=
              List.mem_of_find?_eq_some hfind_s2
            exact R1.fresh.right (k, r) hpair
          -- Lift past two regWrites at fresh, stable — both ≥ s2.nextReg > r.
          apply WasmValue.encodes_preserved_of_fresh _ _
          · intro r' hr'
            simp [SymVal.regs] at hr'
            subst hr'
            exact Nat.lt_of_lt_of_le hr_lt h_stable_ge_s2
          · apply WasmValue.encodes_preserved_of_fresh _ henc
            intro r' hr'
            simp [SymVal.regs] at hr'
            subst hr'
            exact hr_lt
      · -- Fresh.
        refine ⟨?_, ?_⟩
        · intro sv hsv r hr
          have hsv_in_s2 : sv ∈ s2.stack := hsv
          have h := R1.fresh.left sv hsv_in_s2 r hr
          -- r < s2.nextReg < s2.nextReg + 1 < s2.nextReg + 1 + 1.
          exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt h)
        · intro ir hir
          simp at hir
          rcases hir with h_eq | ⟨h_in, _⟩
          · subst h_eq; exact Nat.lt_succ_self _
          · have hin_s2 : ir ∈ s2.localReg := by rw [h_s2_lr]; exact h_in
            have h := R1.fresh.right ir hin_s2
            exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt h)
      · -- AliasFree on s'.localReg = (i, stable) :: filter(≠i) s.localReg.
        intro ir hir sv hsv
        have hsv_in_s2 : sv ∈ s2.stack := hsv
        simp at hir
        rcases hir with h_eq | ⟨h_in, _⟩
        · subst h_eq
          intro hcontra
          have h_lt : stable < s2.nextReg :=
            R1.fresh.left sv hsv_in_s2 stable hcontra
          -- But stable = s2.nextReg + 1 > s2.nextReg, contradiction.
          show False
          have : s2.nextReg + 1 < s2.nextReg := h_lt
          omega
        · have hin_s2 : ir ∈ s2.localReg := by rw [h_s2_lr]; exact h_in
          exact R1.aliasFree ir hin_s2 sv hsv_in_s2
      · -- InjectiveLocals.
        intro p q hp hq
        simp at hp hq
        rcases hp with hp_eq | ⟨hp_in, hp_ne⟩ <;>
        rcases hq with hq_eq | ⟨hq_in, hq_ne⟩
        · subst hp_eq; subst hq_eq; left; rfl
        · right
          subst hp_eq
          have hin_s2 : q ∈ s2.localReg := by rw [h_s2_lr]; exact hq_in
          have h_lt : q.snd < s2.nextReg := R1.fresh.right q hin_s2
          show stable ≠ q.snd
          exact Ne.symm (Nat.ne_of_lt (Nat.lt_of_lt_of_le h_lt h_stable_ge_s2))
        · right
          subst hq_eq
          have hin_s2 : p ∈ s2.localReg := by rw [h_s2_lr]; exact hp_in
          have h_lt : p.snd < s2.nextReg := R1.fresh.right p hin_s2
          show p.snd ≠ stable
          exact Nat.ne_of_lt (Nat.lt_of_lt_of_le h_lt h_stable_ge_s2)
        · have hpin_s2 : p ∈ s2.localReg := by rw [h_s2_lr]; exact hp_in
          have hqin_s2 : q ∈ s2.localReg := by rw [h_s2_lr]; exact hq_in
          exact R1.injLocals p q hpin_s2 hqin_s2
      · -- CurrentRegRefines on s'.currentReg = (i, fresh) :: filter(≠i) s2.currentReg.
        intro k r_cur hfind v hv
        by_cases hki : k = i
        · subst hki
          -- s'.currentReg head is (k, fresh).
          change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((k, fresh) :: List.filter (fun p => !decide (p.fst = k)) s2.currentReg)
                 = some (k, r_cur) at hfind
          change (ws.locals.set k (WasmValue.wI32 n_w)).get? k = some v at hv
          rw [List.find?_cons] at hfind
          simp only [show decide ((k, fresh).fst = k) = true from by simp] at hfind
          injection hfind with h_pair
          have hr_eq : fresh = r_cur := (Prod.ext_iff.mp h_pair).2
          subst hr_eq
          have hv_eq : v = WasmValue.wI32 n_w := by
            have hget : (ws.locals.set k (.wI32 n_w)).get? k =
                        some (WasmValue.wI32 n_w) := by
              rw [List.get?_eq_getElem?]
              exact List.getElem?_set_self (by simpa using hbound)
            rw [hget] at hv
            exact ((Option.some.injEq _ _).mp hv).symm
          subst hv_eq
          simp [WasmValue.encodes]; exact h_lookup_fresh_kst'
        · -- Other indices fall back to filter — but s2.currentReg = s.currentReg.
          change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((i, fresh) :: List.filter (fun p => !decide (p.fst = i)) s2.currentReg)
                 = some (k, r_cur) at hfind
          rw [find?_setLocalReg_ne _ i k _ hki] at hfind
          -- hfind : s2.currentReg.find? ... = some (k, r_cur)
          have hv_old : ws.locals.get? k = some v := by
            rw [List.get?_eq_getElem?] at hv ⊢
            rw [List.getElem?_set_ne (Ne.symm hki)] at hv
            exact hv
          have henc := R1.currentReg k r_cur hfind v hv_old
          -- R1.currentReg gives encodes against kst1.rf.
          -- Need encodes against kst'.rf = regWrite (regWrite kst1.rf fresh _) stable _.
          -- The reg `r_cur` was in s2.currentReg = s.currentReg, so its value is < s.nextReg.
          have hpair : (k, r_cur) ∈ s2.currentReg :=
            List.mem_of_find?_eq_some hfind
          have hpair_s : (k, r_cur) ∈ s.currentReg := by rw [← h_s2_cr]; exact hpair
          have hr_cur_lt_s : r_cur < s.nextReg := R.freshCurrent (k, r_cur) hpair_s
          have hr_cur_lt_s2 : r_cur < s2.nextReg :=
            Nat.lt_of_lt_of_le hr_cur_lt_s _h_s_le_s2
          -- Lift past two fresh regWrites at fresh = s2.nextReg, stable = s2.nextReg + 1.
          apply WasmValue.encodes_preserved_of_fresh _ _
          · intro r' hr'
            simp [SymVal.regs] at hr'
            subst hr'
            -- Goal: r_cur < stable = s2.nextReg + 1.
            exact Nat.lt_of_lt_of_le hr_cur_lt_s2 h_stable_ge_s2
          · apply WasmValue.encodes_preserved_of_fresh _ henc
            intro r' hr'
            simp [SymVal.regs] at hr'
            subst hr'
            -- Goal: r_cur < fresh = s2.nextReg.
            exact hr_cur_lt_s2
      · -- FreshCurrent: every (i', r') in s'.currentReg has r' < s'.nextReg.
        intro ir hir
        simp at hir
        rcases hir with h_eq | ⟨h_in, _⟩
        · subst h_eq
          show fresh < s2.nextReg + 1 + 1
          exact Nat.lt_succ_of_lt (Nat.lt_succ_self _)
        · -- ir was in s2.currentReg (= s.currentReg); use R.freshCurrent.
          have hin_s : ir ∈ s.currentReg := by rw [← h_s2_cr]; exact h_in
          have h := R.freshCurrent ir hin_s
          have h_s_s2 : s.nextReg ≤ s2.nextReg := _h_s_le_s2
          exact Nat.lt_of_lt_of_le h (by omega : s.nextReg ≤ s2.nextReg + 1 + 1)
      · -- CurrentLocalDisjoint on s'.currentReg = (i, fresh) :: ..., s'.localReg = (i, stable) :: ...
        intro p q hp hq hpq
        simp at hp hq
        -- p ∈ currentReg ⇒ p = (i, fresh) ∨ (p ∈ filter ... s2.currentReg ∧ p.fst ≠ i).
        -- q ∈ localReg   ⇒ q = (i, stable) ∨ (q ∈ filter ... s.localReg ∧ q.fst ≠ i).
        rcases hp with hp_eq | ⟨hp_in, hp_ne⟩ <;>
        rcases hq with hq_eq | ⟨hq_in, hq_ne⟩
        · -- p = (i, fresh), q = (i, stable): both fst = i, contradiction with hpq.
          subst hp_eq; subst hq_eq; exact absurd rfl hpq
        · -- p = (i, fresh), q ∈ s.localReg with q.fst ≠ i.
          subst hp_eq
          -- Need fresh ≠ q.snd. q.snd < s.nextReg ≤ s2.nextReg = fresh.
          have hq_lt : q.snd < s.nextReg := R.fresh.right q hq_in
          have hq_lt_s2 : q.snd < s2.nextReg :=
            Nat.lt_of_lt_of_le hq_lt _h_s_le_s2
          exact Ne.symm (Nat.ne_of_lt hq_lt_s2)
        · -- p ∈ s2.currentReg with p.fst ≠ i, q = (i, stable).
          subst hq_eq
          -- stable = s2.nextReg + 1; p.snd < s.nextReg (via R.freshCurrent) < stable.
          have hp_in_s : p ∈ s.currentReg := by rw [← h_s2_cr]; exact hp_in
          have hp_lt : p.snd < s.nextReg := R.freshCurrent p hp_in_s
          have hp_lt_s2 : p.snd < s2.nextReg := Nat.lt_of_lt_of_le hp_lt _h_s_le_s2
          have hp_lt_stable : p.snd < stable :=
            Nat.lt_of_lt_of_le hp_lt_s2 h_stable_ge_s2
          exact Nat.ne_of_lt hp_lt_stable
        · -- p ∈ filter s2.currentReg, q ∈ filter s.localReg, both with fst ≠ i.
          have hp_in_s : p ∈ s.currentReg := by rw [← h_s2_cr]; exact hp_in
          exact R.curLocDisj p q hp_in_s hq_in hpq
  -- Case A: existing entry. lookupLocal returns some stable_old.
  · simp [hreg_find] at hl
    obtain ⟨hs_eq, hops_eq⟩ := hl
    have hentry_fst : entry.fst = i := by
      have := List.find?_some hreg_find
      simpa using this
    let stable_old : Reg := entry.snd
    -- kst_after_fresh = kst1 with rf := regWrite kst1.rf fresh (vU32 n_w).
    let kst_after_fresh_A : Quanta.KOps.State :=
      { kst1 with rf := regWrite kst1.rf fresh (vU32 n_w) }
    let kst'_A : Quanta.KOps.State :=
      { kst_after_fresh_A with rf := regWrite kst_after_fresh_A.rf stable_old (vU32 n_w) }
    have hentry_in : entry ∈ s.localReg :=
      List.mem_of_find?_eq_some hreg_find
    have hentry_in_s2 : entry ∈ s2.localReg := by rw [h_s2_lr]; exact hentry_in
    have hentry_pair : (i, stable_old) ∈ s.localReg := by
      have h_eq : entry = (i, stable_old) := by
        show entry = (i, entry.snd)
        rcases entry with ⟨ek, er⟩
        simp at hentry_fst
        simp [hentry_fst]
      rw [← h_eq]; exact hentry_in
    have hentry_pair_s2 : (i, stable_old) ∈ s2.localReg := by
      rw [h_s2_lr]; exact hentry_pair
    have h_stable_old_lt_s2 : stable_old < s2.nextReg :=
      R1.fresh.right entry hentry_in_s2
    have h_stable_old_ne_fresh : stable_old ≠ fresh :=
      Nat.ne_of_lt h_stable_old_lt_s2
    refine ⟨kst'_A, ?_, ?_⟩
    · -- evalOps closes via same dual-Copy as case B (mod stable_old vs s2.nextReg+1).
      subst hops_eq
      -- Absorb the frame-0 zero-init (Const fresh _ overwritten by Copy fresh src).
      have h_src_ne_fresh : src ≠ fresh := Nat.ne_of_lt h_src_lt_s2
      rw [evalOps_append h_evalC h_kst1_ok,
          evalOps_const_copy_absorb h_src_ne_fresh h_kst1_ok,
          ← evalOps_append h_evalC h_kst1_ok]
      have h_evalC1 : evalOps 0 kst1 [.copy fresh src] = some kst_after_fresh_A := by
        simp only [evalOps, Quanta.KOps.evalOp]
        rw [h_src_lookup]
        simp only [Option.bind_eq_bind, Option.bind, Option.some_bind, pure, h_kst1_ok]
        rw [if_neg (by decide : ¬ (false = true))]
        congr 1
        show ({ kst1 with rf := regWrite kst1.rf fresh (vU32 n_w),
                          broke := false } : Quanta.KOps.State) = kst_after_fresh_A
        rw [show (false : Bool) = kst1.broke from h_kst1_ok.symm]
      have h_kaf_ok : kst_after_fresh_A.broke = false := h_kst1_ok
      have h_lookup_fresh_A :
          regLookup kst_after_fresh_A.rf fresh = some (vU32 n_w) := by
        show regLookup (regWrite kst1.rf fresh (vU32 n_w)) fresh = _
        rw [regLookup_regWrite_self]
      have h_evalC2 : evalOps 0 kst_after_fresh_A [.copy stable_old fresh] = some kst'_A := by
        simp only [evalOps, Quanta.KOps.evalOp]
        rw [h_lookup_fresh_A]
        simp only [Option.bind_eq_bind, Option.bind, Option.some_bind, pure, h_kaf_ok]
        rw [if_neg (by decide : ¬ (false = true))]
        congr 1
        show ({ kst_after_fresh_A with
                  rf := regWrite kst_after_fresh_A.rf stable_old (vU32 n_w),
                  broke := false } : Quanta.KOps.State) = kst'_A
        rw [show (false : Bool) = kst_after_fresh_A.broke from h_kaf_ok.symm]
      rw [show (opsCommit ++ [KernelOp.copy fresh src, KernelOp.copy stable_old fresh])
            = opsCommit ++ [KernelOp.copy fresh src] ++ [KernelOp.copy stable_old fresh] from by
            simp [List.append_assoc]]
      rw [evalOps_append (evalOps_append h_evalC h_kst1_ok ▸ h_evalC1) h_kaf_ok]
      exact h_evalC2
    · -- Refines on the post-state.
      subst hs_eq
      have h_lookup_stable_kst'A : regLookup kst'_A.rf stable_old = some (vU32 n_w) := by
        show regLookup (regWrite kst_after_fresh_A.rf stable_old (vU32 n_w)) stable_old = _
        rw [regLookup_regWrite_self]
      refine ⟨?_, ?_, ?_, ?_, ?_, R1.heapRefines, ?_, ?_, ?_⟩
      · -- StackRefines. Lift past write at fresh (≥ s2.nextReg) AND stable_old.
        -- For stable_old: by aliasFree on s2.localReg, no stack reg = stable_old.
        refine ⟨?_, ?_⟩
        · show rest.length = s2.stack.length
          rw [h_s2_stack]; simpa using h_rest_lrest_len
        · intro j v hv
          have hk : ws_pop.stack.get? j = some v := by
            show rest.get? j = some v; simpa using hv
          obtain ⟨svj, hsvj_get, henc⟩ := R1.stk.right j v hk
          refine ⟨svj, by simpa using hsvj_get, ?_⟩
          have hsvj_in : svj ∈ s2.stack := List.mem_of_get? hsvj_get
          -- Lift past write at stable_old (no alias by aliasFree on entry).
          apply WasmValue.encodes_preserved_of_disjoint _ _
          · exact R1.aliasFree entry hentry_in_s2 svj hsvj_in
          -- Lift past write at fresh (fresh > all stack regs).
          apply WasmValue.encodes_preserved_of_fresh _ henc
          intro r hr
          exact R1.fresh.left svj hsvj_in r hr
      · -- LocalsRefines. s'.localReg = (i, stable_old) :: filter(≠i) s.localReg.
        intro k r hfind v hv
        by_cases hki : k = i
        · subst hki
          change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((k, stable_old) :: List.filter (fun p => !decide (p.fst = k)) s.localReg)
                 = some (k, r) at hfind
          change (ws.locals.set k (WasmValue.wI32 n_w)).get? k = some v at hv
          rw [List.find?_cons] at hfind
          simp only [show decide ((k, stable_old).fst = k) = true from by simp] at hfind
          injection hfind with h_pair
          have hr_eq : stable_old = r := (Prod.ext_iff.mp h_pair).2
          subst hr_eq
          have hv_eq : v = WasmValue.wI32 n_w := by
            have hget : (ws.locals.set k (.wI32 n_w)).get? k =
                        some (WasmValue.wI32 n_w) := by
              rw [List.get?_eq_getElem?]
              exact List.getElem?_set_self (by simpa using hbound)
            rw [hget] at hv
            exact ((Option.some.injEq _ _).mp hv).symm
          subst hv_eq
          simp [WasmValue.encodes]; exact h_lookup_stable_kst'A
        · change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((i, stable_old) :: List.filter (fun p => !decide (p.fst = i)) s.localReg)
                 = some (k, r) at hfind
          rw [find?_setLocalReg_ne _ i k _ hki] at hfind
          have hv_old : ws.locals.get? k = some v := by
            rw [List.get?_eq_getElem?] at hv ⊢
            rw [List.getElem?_set_ne (Ne.symm hki)] at hv
            exact hv
          have hfind_s2 : s2.localReg.find? (fun p => p.fst = k) = some (k, r) := by
            rw [h_s2_lr]; exact hfind
          have henc := R1.locs k r hfind_s2 v hv_old
          have hkr_in_s2 : (k, r) ∈ s2.localReg :=
            List.mem_of_find?_eq_some hfind_s2
          have hr_lt : r < s2.nextReg := R1.fresh.right (k, r) hkr_in_s2
          have hr_ne_stable_old : r ≠ stable_old := by
            have := R1.injLocals (k, r) (i, stable_old) hkr_in_s2 hentry_pair_s2
            rcases this with h_keq | h_rne
            · exact absurd h_keq hki
            · exact h_rne
          -- Lift past writes at fresh (= s2.nextReg) and stable_old (≠ r).
          apply WasmValue.encodes_preserved_of_disjoint _ _
          · simp [SymVal.regs]; exact Ne.symm hr_ne_stable_old
          apply WasmValue.encodes_preserved_of_fresh _ henc
          intro r' hr'
          simp [SymVal.regs] at hr'
          subst hr'
          exact hr_lt
      · -- Fresh. nextReg = s2.nextReg + 1.
        refine ⟨?_, ?_⟩
        · intro sv hsv r hr
          have hsv_in_s2 : sv ∈ s2.stack := hsv
          exact Nat.lt_succ_of_lt (R1.fresh.left sv hsv_in_s2 r hr)
        · intro ir hir
          simp at hir
          rcases hir with h_eq | ⟨h_in, _⟩
          · subst h_eq
            -- (i, stable_old).snd = stable_old < s2.nextReg < s2.nextReg + 1.
            exact Nat.lt_succ_of_lt h_stable_old_lt_s2
          · have hin_s2 : ir ∈ s2.localReg := by rw [h_s2_lr]; exact h_in
            exact Nat.lt_succ_of_lt (R1.fresh.right ir hin_s2)
      · -- AliasFree. localReg = (i, stable_old) :: filter(≠i) s.localReg.
        intro ir hir sv hsv
        have hsv_in_s2 : sv ∈ s2.stack := hsv
        simp at hir
        rcases hir with h_eq | ⟨h_in, _⟩
        · subst h_eq
          exact R1.aliasFree entry hentry_in_s2 sv hsv_in_s2
        · have hin_s2 : ir ∈ s2.localReg := by rw [h_s2_lr]; exact h_in
          exact R1.aliasFree ir hin_s2 sv hsv_in_s2
      · -- InjectiveLocals. Head (i, stable_old) — distinct from filtered entries by injLocals.
        intro p q hp hq
        simp at hp hq
        rcases hp with hp_eq | ⟨hp_in, hp_ne⟩ <;>
        rcases hq with hq_eq | ⟨hq_in, hq_ne⟩
        · subst hp_eq; subst hq_eq; left; rfl
        · right
          subst hp_eq
          have hin_s2 : q ∈ s2.localReg := by rw [h_s2_lr]; exact hq_in
          have h_old := R1.injLocals q (i, stable_old) hin_s2 hentry_pair_s2
          rcases h_old with h_keq | h_rne
          · exact absurd h_keq hq_ne
          · exact h_rne.symm
        · right
          subst hq_eq
          have hin_s2 : p ∈ s2.localReg := by rw [h_s2_lr]; exact hp_in
          have h_old := R1.injLocals p (i, stable_old) hin_s2 hentry_pair_s2
          rcases h_old with h_keq | h_rne
          · exact absurd h_keq hp_ne
          · exact h_rne
        · have hpin_s2 : p ∈ s2.localReg := by rw [h_s2_lr]; exact hp_in
          have hqin_s2 : q ∈ s2.localReg := by rw [h_s2_lr]; exact hq_in
          exact R1.injLocals p q hpin_s2 hqin_s2
      · -- CurrentRegRefines. s'.currentReg = (i, fresh) :: filter(≠i) s2.currentReg.
        intro k r_cur hfind v hv
        by_cases hki : k = i
        · subst hki
          change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((k, fresh) :: List.filter (fun p => !decide (p.fst = k)) s2.currentReg)
                 = some (k, r_cur) at hfind
          change (ws.locals.set k (WasmValue.wI32 n_w)).get? k = some v at hv
          rw [List.find?_cons] at hfind
          simp only [show decide ((k, fresh).fst = k) = true from by simp] at hfind
          injection hfind with h_pair
          have hr_eq : fresh = r_cur := (Prod.ext_iff.mp h_pair).2
          subst hr_eq
          have hv_eq : v = WasmValue.wI32 n_w := by
            have hget : (ws.locals.set k (.wI32 n_w)).get? k =
                        some (WasmValue.wI32 n_w) := by
              rw [List.get?_eq_getElem?]
              exact List.getElem?_set_self (by simpa using hbound)
            rw [hget] at hv
            exact ((Option.some.injEq _ _).mp hv).symm
          subst hv_eq
          have h_lookup_fresh_A_kst' : regLookup kst'_A.rf fresh = some (vU32 n_w) := by
            show regLookup (regWrite kst_after_fresh_A.rf stable_old (vU32 n_w)) fresh = _
            rw [regLookup_regWrite_of_ne _ stable_old fresh _ (Ne.symm h_stable_old_ne_fresh)]
            show regLookup (regWrite kst1.rf fresh (vU32 n_w)) fresh = _
            rw [regLookup_regWrite_self]
          simp [WasmValue.encodes]; exact h_lookup_fresh_A_kst'
        · -- Off-i: s2.currentReg = s.currentReg, fall back via R1.currentReg.
          change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((i, fresh) :: List.filter (fun p => !decide (p.fst = i)) s2.currentReg)
                 = some (k, r_cur) at hfind
          rw [find?_setLocalReg_ne _ i k _ hki] at hfind
          have hv_old : ws.locals.get? k = some v := by
            rw [List.get?_eq_getElem?] at hv ⊢
            rw [List.getElem?_set_ne (Ne.symm hki)] at hv
            exact hv
          have henc := R1.currentReg k r_cur hfind v hv_old
          have hpair_cur : (k, r_cur) ∈ s2.currentReg :=
            List.mem_of_find?_eq_some hfind
          have hpair_s_cur : (k, r_cur) ∈ s.currentReg := by rw [← h_s2_cr]; exact hpair_cur
          have hr_cur_lt_s : r_cur < s.nextReg := R.freshCurrent (k, r_cur) hpair_s_cur
          have hr_cur_lt_s2 : r_cur < s2.nextReg :=
            Nat.lt_of_lt_of_le hr_cur_lt_s _h_s_le_s2
          -- Lift past writes at fresh (= s2.nextReg) and stable_old (< s2.nextReg, may = r_cur).
          -- For stable_old: is r_cur = stable_old? injLocals says (k, r_cur)
          -- and (i, stable_old) both in localReg s2 → either k=i (excluded) or
          -- r_cur ≠ stable_old.
          -- r_cur ≠ stable_old via CurrentLocalDisjoint applied to s
          -- (lifted to s2 via h_s2_cr / h_s2_lr).
          have hpair_cur : (k, r_cur) ∈ s2.currentReg :=
            List.mem_of_find?_eq_some hfind
          have hpair_cur_s : (k, r_cur) ∈ s.currentReg := by
            rw [← h_s2_cr]; exact hpair_cur
          have hentry_pair_s : (i, stable_old) ∈ s.localReg := hentry_pair
          have hr_cur_ne_stable : r_cur ≠ stable_old :=
            R.curLocDisj (k, r_cur) (i, stable_old) hpair_cur_s hentry_pair_s hki
          -- r_cur < s.nextReg ≤ s2.nextReg = fresh, so r_cur ≠ fresh.
          have hpair_s_cur : (k, r_cur) ∈ s.currentReg := hpair_cur_s
          have hr_cur_lt_s : r_cur < s.nextReg := R.freshCurrent (k, r_cur) hpair_s_cur
          have hr_cur_lt_s2 : r_cur < s2.nextReg :=
            Nat.lt_of_lt_of_le hr_cur_lt_s _h_s_le_s2
          have hr_cur_ne_fresh : r_cur ≠ fresh := Nat.ne_of_lt hr_cur_lt_s2
          -- Goal: encodes layout kst'_A.rf .reg r_cur .u32.
          -- kst'_A.rf = regWrite (regWrite kst1.rf fresh _) stable_old _.
          -- Lift via two disjoint regWrites.
          apply WasmValue.encodes_preserved_of_disjoint _ _
          · simp [SymVal.regs]; exact Ne.symm hr_cur_ne_stable
          apply WasmValue.encodes_preserved_of_disjoint _ henc
          simp [SymVal.regs]; exact Ne.symm hr_cur_ne_fresh
      · -- FreshCurrent: every (k, r) in s'.currentReg has r < s'.nextReg.
        intro ir hir
        simp at hir
        rcases hir with h_eq | ⟨h_in, _⟩
        · subst h_eq
          -- (i, fresh).snd = fresh = s2.nextReg < s2.nextReg + 1.
          exact Nat.lt_succ_self _
        · have hin_s : ir ∈ s.currentReg := by rw [← h_s2_cr]; exact h_in
          have h := R.freshCurrent ir hin_s
          have h_s_s2 : s.nextReg ≤ s2.nextReg := _h_s_le_s2
          exact Nat.lt_of_lt_of_le h (by omega : s.nextReg ≤ s2.nextReg + 1)
      · -- CurrentLocalDisjoint on s'.currentReg = (i, fresh) :: ..., s'.localReg = (i, stable_old) :: ...
        intro p q hp hq hpq
        simp at hp hq
        rcases hp with hp_eq | ⟨hp_in, hp_ne⟩ <;>
        rcases hq with hq_eq | ⟨hq_in, hq_ne⟩
        · subst hp_eq; subst hq_eq; exact absurd rfl hpq
        · -- p = (i, fresh), q ∈ filter s.localReg with q.fst ≠ i.
          subst hp_eq
          have hq_lt : q.snd < s.nextReg := R.fresh.right q hq_in
          have hq_lt_s2 : q.snd < s2.nextReg :=
            Nat.lt_of_lt_of_le hq_lt _h_s_le_s2
          exact Ne.symm (Nat.ne_of_lt hq_lt_s2)
        · -- p ∈ filter s2.currentReg with p.fst ≠ i, q = (i, stable_old).
          subst hq_eq
          have hp_in_s : p ∈ s.currentReg := by rw [← h_s2_cr]; exact hp_in
          exact R.curLocDisj p (i, stable_old) hp_in_s hentry_pair hp_ne
        · -- p ∈ filter s2.currentReg, q ∈ filter s.localReg, both fst ≠ i.
          have hp_in_s : p ∈ s.currentReg := by rw [← h_s2_cr]; exact hp_in
          exact R.curLocDisj p q hp_in_s hq_in hpq

-- ════════════════════════════════════════════════════════════════════
-- Slice 3 follow-up: localTee preservation
--
-- `local.tee i` = `local.set i` then re-push. The translator emits two
-- ops: `.copy dst src` (mirrors localSet's stable-write) followed by
-- `.copy post_fresh dst` (mirrors localGet's alias-breaking copy).
-- Two-op sequence ⇒ `kst.broke = false` precondition required.
-- The post-tee top of the stack is `.reg post_fresh .u32`, encoding
-- the same `wI32 n_w` value the popped `src` register held.
--
-- Same existing-vs-fresh dst split as localSet, with these tweaks:
--   * Two regWrites in the produced `kst'` (at `dst`, then at
--     `post_fresh`).
--   * `Refines.stk` adds the new top entry encoding `wI32 n_w`.
--   * `ws'.stack = v_w :: rest` (top preserved, unlike localSet).
-- ════════════════════════════════════════════════════════════════════

open Quanta.KOps (vU32) in
/-- `local.tee i` preservation. Stage 3 3-Copy emission:
    `opsCommit ++ [.copy fresh src, .copy stable fresh, .copy post_fresh fresh]`
    plus pushes `.reg post_fresh .u32` onto the symbolic stack and
    `setCurrentReg i fresh`.

    Mirrors `preservation_localSet`'s structure with two additions:
    (1) an extra regWrite at `post_fresh`, (2) ws'.stack = wI32 :: rest
    matched by s'.stack = .reg post_fresh .u32 :: s2.stack. -/
theorem preservation_localTee (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout) (h_kst_ok : kst.broke = false)
    (i : Nat)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws (.localTee i) = some ws')
    (hl : lowerInstr s (.localTee i) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- WASM side: pop v_w, setLocal i v_w, push v_w back.
  simp only [evalInstr, WasmState.pop, WasmState.push,
             Option.bind_eq_bind, Option.bind, pure] at hw
  rcases hws_stack : ws.stack with _ | ⟨v_w, rest⟩
  · simp [hws_stack] at hw
  simp only [hws_stack, WasmState.setLocal] at hw
  by_cases hbound : i < List.length ws.locals
  case neg => simp [if_neg hbound] at hw
  simp only [if_pos hbound] at hw
  have hws'_eq : ws' = { locals := ws.locals.set i v_w, stack := v_w :: rest,
                          mem := ws.mem, halted := ws.halted,
                          branchTarget := ws.branchTarget } :=
    ((Option.some.injEq _ _).mp hw).symm
  subst hws'_eq
  -- Lean side: popSym + commit.
  unfold lowerInstr at hl
  rcases hls_stack : s.stack with _ | ⟨sva, lrest⟩
  · simp [hls_stack, LowerState.popSym] at hl
  simp only [hls_stack, LowerState.popSym, Option.bind_eq_bind, Option.some_bind] at hl
  rcases hca : ({s with stack := lrest} : LowerState).commit sva
      with _ | ⟨src, s2, opsCommit⟩
  · simp [hca] at hl
  simp only [hca, Option.some_bind] at hl
  -- v_w must be wI32.
  have hv_enc : v_w.encodes layout kst.rf sva := by
    have hb := R.stk.right 0 v_w (by rw [hws_stack]; simp)
    obtain ⟨sv0, hsv0_get, henc⟩ := hb
    have hs0 : s.stack.get? 0 = some sva := by rw [hls_stack]; simp
    rw [hs0] at hsv0_get
    have h_eq : sva = sv0 := (Option.some.injEq _ _).mp hsv0_get
    rw [h_eq]; exact henc
  obtain ⟨n_w, hv_w_eq⟩ : ∃ n_w, v_w = WasmValue.wI32 n_w := by
    cases v_w with
    | wI32 n_w => exact ⟨n_w, rfl⟩
    | wI64 _ => cases sva <;> simp [WasmValue.encodes] at hv_enc
    | wF32 _ => cases sva <;> simp [WasmValue.encodes] at hv_enc
    | wF64 _ => cases sva <;> simp [WasmValue.encodes] at hv_enc
  subst hv_w_eq
  have h_sva_in : sva ∈ s.stack := by rw [hls_stack]; simp
  have h_sva_lt : ∀ r ∈ sva.regs, r < s.nextReg :=
    fun r hr => R.fresh.left sva h_sva_in r hr
  have h_rest_lrest_len : rest.length = lrest.length := by
    have hl_orig := R.stk.left
    rw [hws_stack, hls_stack] at hl_orig
    simpa using hl_orig
  let s_pop : LowerState :=
    { nextReg := s.nextReg, stack := lrest,
      localReg := s.localReg, localTy := s.localTy,
      bufferSlots := s.bufferSlots, currentReg := s.currentReg }
  let ws_pop : WasmState := { ws with stack := rest }
  have R_pop : Refines ws_pop s_pop kst layout := by
    refine ⟨⟨h_rest_lrest_len, ?_⟩, R.locs, ?_, ?_, R.injLocals,
            R.heapRefines, R.currentReg, R.freshCurrent, R.curLocDisj⟩
    · intro k v hv
      have hrest_get : ws.stack.get? (k + 1) = some v := by
        rw [hws_stack]; simpa using hv
      obtain ⟨svk, hsvk_get, henc⟩ := R.stk.right (k + 1) v hrest_get
      have hlrest_get : lrest.get? k = some svk := by
        have h2 : s.stack.get? (k + 1) = some svk := hsvk_get
        rw [hls_stack] at h2; simpa using h2
      exact ⟨svk, by simpa using hlrest_get, henc⟩
    · refine ⟨?_, R.fresh.right⟩
      intro sv hsv r hr
      have hsv_in : sv ∈ s.stack := by
        rw [hls_stack]; exact List.mem_cons_of_mem _ hsv
      exact R.fresh.left sv hsv_in r hr
    · intro ir hir sv hsv
      have hsv_in : sv ∈ s.stack := by
        rw [hls_stack]; exact List.mem_cons_of_mem _ hsv
      exact R.aliasFree ir hir sv hsv_in
  obtain ⟨kst1, h_evalC, R1, h_enc_src1, h_lookupC, _h_s_le_s2, h_src_lt_s2⟩ :=
    commit_correct R_pop h_sva_lt hca hv_enc
  have h_kst1_ok : kst1.broke = false := by
    rw [commit_preserves_broke hca h_evalC]; exact h_kst_ok
  have h_src_lookup : regLookup kst1.rf src = some (Quanta.KOps.Value.vU32 n_w) :=
    h_enc_src1
  have h_s2_lr : s2.localReg = s.localReg := (commit_preserves_locals hca).1
  have h_s2_lt : s2.localTy  = s.localTy  := (commit_preserves_locals hca).2
  have h_s2_stack : s2.stack = lrest := commit_preserves_stack hca
  have h_s2_cr : s2.currentReg = s.currentReg := by
    have h := commit_preserves_currentReg hca
    exact h
  -- Branch on lookupLocal post-alloc.
  simp only [LowerState.lookupLocal, LowerState.lookupLocalTy, LowerState.alloc,
             LowerState.setLocalReg, LowerState.setCurrentReg, LowerState.push,
             Option.bind_eq_bind, Option.bind, pure] at hl
  rw [h_s2_lt, h_s2_lr] at hl
  let fresh : Reg := s2.nextReg
  rcases hreg_find : s.localReg.find? (fun p => p.fst = i) with _ | entry
  -- Case B: first write. stable = fresh + 1, post_fresh = stable + 1.
  · simp [hreg_find] at hl
    obtain ⟨hs_eq, hops_eq⟩ := hl
    let stable : Reg := s2.nextReg + 1
    let post_fresh : Reg := s2.nextReg + 1 + 1
    let kst_after_fresh : Quanta.KOps.State :=
      { kst1 with rf := regWrite kst1.rf fresh (vU32 n_w) }
    let kst_after_stable : Quanta.KOps.State :=
      { kst_after_fresh with rf := regWrite kst_after_fresh.rf stable (vU32 n_w) }
    let kst' : Quanta.KOps.State :=
      { kst_after_stable with rf := regWrite kst_after_stable.rf post_fresh (vU32 n_w) }
    refine ⟨kst', ?_, ?_⟩
    · -- evalOps closes via three Copy steps.
      subst hops_eq
      -- Absorb the frame-0 zero-init (Const fresh _ overwritten by Copy fresh src).
      have h_src_ne_fresh : src ≠ fresh := Nat.ne_of_lt h_src_lt_s2
      rw [evalOps_append h_evalC h_kst1_ok,
          evalOps_const_copy_absorb h_src_ne_fresh h_kst1_ok,
          ← evalOps_append h_evalC h_kst1_ok]
      have h_evalC1 : evalOps 0 kst1 [.copy fresh src] = some kst_after_fresh := by
        simp only [evalOps, Quanta.KOps.evalOp]
        rw [h_src_lookup]
        simp only [Option.bind_eq_bind, Option.bind, Option.some_bind, pure, h_kst1_ok]
        rw [if_neg (by decide : ¬ (false = true))]
        congr 1
        show ({ kst1 with rf := regWrite kst1.rf fresh (vU32 n_w),
                          broke := false } : Quanta.KOps.State) = kst_after_fresh
        rw [show (false : Bool) = kst1.broke from h_kst1_ok.symm]
      have h_kaf_ok : kst_after_fresh.broke = false := h_kst1_ok
      have h_lookup_fresh_kaf :
          regLookup kst_after_fresh.rf fresh = some (vU32 n_w) := by
        show regLookup (regWrite kst1.rf fresh (vU32 n_w)) fresh = _
        rw [regLookup_regWrite_self]
      have h_evalC2 : evalOps 0 kst_after_fresh [.copy stable fresh] = some kst_after_stable := by
        simp only [evalOps, Quanta.KOps.evalOp]
        rw [h_lookup_fresh_kaf]
        simp only [Option.bind_eq_bind, Option.bind, Option.some_bind, pure, h_kaf_ok]
        rw [if_neg (by decide : ¬ (false = true))]
        congr 1
        show ({ kst_after_fresh with
                  rf := regWrite kst_after_fresh.rf stable (vU32 n_w),
                  broke := false } : Quanta.KOps.State) = kst_after_stable
        rw [show (false : Bool) = kst_after_fresh.broke from h_kaf_ok.symm]
      have h_kas_ok : kst_after_stable.broke = false := h_kaf_ok
      have h_lookup_fresh_kas :
          regLookup kst_after_stable.rf fresh = some (vU32 n_w) := by
        show regLookup (regWrite kst_after_fresh.rf stable (vU32 n_w)) fresh = _
        rw [regLookup_regWrite_of_ne _ stable fresh _
              (Nat.ne_of_lt (Nat.lt_succ_self _))]
        exact h_lookup_fresh_kaf
      have h_evalC3 :
          evalOps 0 kst_after_stable [.copy post_fresh fresh] = some kst' := by
        simp only [evalOps, Quanta.KOps.evalOp]
        rw [h_lookup_fresh_kas]
        simp only [Option.bind_eq_bind, Option.bind, Option.some_bind, pure, h_kas_ok]
        rw [if_neg (by decide : ¬ (false = true))]
        congr 1
        show ({ kst_after_stable with
                  rf := regWrite kst_after_stable.rf post_fresh (vU32 n_w),
                  broke := false } : Quanta.KOps.State) = kst'
        rw [show (false : Bool) = kst_after_stable.broke from h_kas_ok.symm]
      rw [show (opsCommit ++ [KernelOp.copy fresh src,
                              KernelOp.copy stable fresh,
                              KernelOp.copy post_fresh fresh])
            = opsCommit ++ [KernelOp.copy fresh src]
                       ++ [KernelOp.copy stable fresh]
                       ++ [KernelOp.copy post_fresh fresh] from by
            simp [List.append_assoc]]
      rw [evalOps_append
            (evalOps_append (evalOps_append h_evalC h_kst1_ok ▸ h_evalC1) h_kaf_ok ▸ h_evalC2)
            h_kas_ok]
      exact h_evalC3
    · -- Refines on the post-state.
      subst hs_eq
      have h_stable_gt_fresh : fresh < stable := Nat.lt_succ_self _
      have h_post_fresh_gt_stable : stable < post_fresh := Nat.lt_succ_self _
      have h_post_fresh_gt_fresh : fresh < post_fresh :=
        Nat.lt_trans h_stable_gt_fresh h_post_fresh_gt_stable
      have h_fresh_ge_s2 : s2.nextReg ≤ fresh := Nat.le_refl _
      have h_stable_ge_s2 : s2.nextReg ≤ stable := Nat.le_succ _
      have h_post_fresh_ge_s2 : s2.nextReg ≤ post_fresh :=
        Nat.le_of_lt (Nat.lt_of_le_of_lt h_stable_ge_s2 h_post_fresh_gt_stable)
      -- Lookup helpers in kst'.rf.
      have h_lookup_post_fresh_kst' :
          regLookup kst'.rf post_fresh = some (vU32 n_w) := by
        show regLookup (regWrite kst_after_stable.rf post_fresh (vU32 n_w)) post_fresh = _
        rw [regLookup_regWrite_self]
      have h_lookup_stable_kst' :
          regLookup kst'.rf stable = some (vU32 n_w) := by
        show regLookup (regWrite kst_after_stable.rf post_fresh (vU32 n_w)) stable = _
        rw [regLookup_regWrite_of_ne _ post_fresh stable _
              (Nat.ne_of_lt h_post_fresh_gt_stable)]
        show regLookup (regWrite kst_after_fresh.rf stable (vU32 n_w)) stable = _
        rw [regLookup_regWrite_self]
      have h_lookup_fresh_kst' :
          regLookup kst'.rf fresh = some (vU32 n_w) := by
        show regLookup (regWrite kst_after_stable.rf post_fresh (vU32 n_w)) fresh = _
        rw [regLookup_regWrite_of_ne _ post_fresh fresh _
              (Nat.ne_of_lt h_post_fresh_gt_fresh)]
        show regLookup (regWrite kst_after_fresh.rf stable (vU32 n_w)) fresh = _
        rw [regLookup_regWrite_of_ne _ stable fresh _
              (Nat.ne_of_lt h_stable_gt_fresh)]
        show regLookup (regWrite kst1.rf fresh (vU32 n_w)) fresh = _
        rw [regLookup_regWrite_self]
      refine ⟨?_, ?_, ?_, ?_, ?_, R1.heapRefines, ?_, ?_, ?_⟩
      · -- StackRefines: ws'.stack = wI32 n_w :: rest;
        -- s'.stack = .reg post_fresh .u32 :: s2.stack.
        refine ⟨?_, ?_⟩
        · show (WasmValue.wI32 n_w :: rest).length
              = (SymVal.reg post_fresh .u32 :: s2.stack).length
          rw [h_s2_stack]; simpa using h_rest_lrest_len
        · intro j v hv
          cases j with
          | zero =>
            simp at hv
            refine ⟨SymVal.reg post_fresh .u32, ?_, ?_⟩
            · show (SymVal.reg post_fresh .u32 :: s2.stack).get? 0
                = some (SymVal.reg post_fresh .u32)
              rfl
            subst hv
            simp [WasmValue.encodes]; exact h_lookup_post_fresh_kst'
          | succ k =>
            have hk : ws_pop.stack.get? k = some v := by
              show rest.get? k = some v; simpa using hv
            obtain ⟨svk, hsvk_get, henc⟩ := R1.stk.right k v hk
            refine ⟨svk, by simpa using hsvk_get, ?_⟩
            have hsvk_in : svk ∈ s2.stack := List.mem_of_get? hsvk_get
            -- Lift past three writes at fresh, stable, post_fresh — all ≥ s2.nextReg.
            apply WasmValue.encodes_preserved_of_fresh _ _
            · intro r hr
              have hr_s2 : r < s2.nextReg := R1.fresh.left svk hsvk_in r hr
              exact Nat.lt_of_lt_of_le hr_s2 h_post_fresh_ge_s2
            apply WasmValue.encodes_preserved_of_fresh _ _
            · intro r hr
              have hr_s2 : r < s2.nextReg := R1.fresh.left svk hsvk_in r hr
              exact Nat.lt_of_lt_of_le hr_s2 h_stable_ge_s2
            apply WasmValue.encodes_preserved_of_fresh _ henc
            intro r hr
            exact R1.fresh.left svk hsvk_in r hr
      · -- LocalsRefines on s'.localReg = (i, stable) :: filter(≠i) s.localReg.
        intro k r hfind v hv
        by_cases hki : k = i
        · subst hki
          change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((k, stable) :: List.filter (fun p => !decide (p.fst = k)) s.localReg)
                 = some (k, r) at hfind
          change (ws.locals.set k (WasmValue.wI32 n_w)).get? k = some v at hv
          rw [List.find?_cons] at hfind
          simp only [show decide ((k, stable).fst = k) = true from by simp] at hfind
          injection hfind with h_pair
          have hr_eq : stable = r := (Prod.ext_iff.mp h_pair).2
          subst hr_eq
          have hv_eq : v = WasmValue.wI32 n_w := by
            have hget : (ws.locals.set k (.wI32 n_w)).get? k =
                        some (WasmValue.wI32 n_w) := by
              rw [List.get?_eq_getElem?]
              exact List.getElem?_set_self (by simpa using hbound)
            rw [hget] at hv
            exact ((Option.some.injEq _ _).mp hv).symm
          subst hv_eq
          simp [WasmValue.encodes]; exact h_lookup_stable_kst'
        · change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((i, stable) :: List.filter (fun p => !decide (p.fst = i)) s.localReg)
                 = some (k, r) at hfind
          rw [find?_setLocalReg_ne _ i k _ hki] at hfind
          have hv_old : ws.locals.get? k = some v := by
            rw [List.get?_eq_getElem?] at hv ⊢
            rw [List.getElem?_set_ne (Ne.symm hki)] at hv
            exact hv
          have hfind_s2 : s2.localReg.find? (fun p => p.fst = k) = some (k, r) := by
            rw [h_s2_lr]; exact hfind
          have henc := R1.locs k r hfind_s2 v hv_old
          have hr_lt : r < s2.nextReg := by
            have hpair : (k, r) ∈ s2.localReg :=
              List.mem_of_find?_eq_some hfind_s2
            exact R1.fresh.right (k, r) hpair
          -- Lift past three regWrites at fresh, stable, post_fresh — all ≥ s2.nextReg > r.
          apply WasmValue.encodes_preserved_of_fresh _ _
          · intro r' hr'
            simp [SymVal.regs] at hr'
            subst hr'
            exact Nat.lt_of_lt_of_le hr_lt h_post_fresh_ge_s2
          apply WasmValue.encodes_preserved_of_fresh _ _
          · intro r' hr'
            simp [SymVal.regs] at hr'
            subst hr'
            exact Nat.lt_of_lt_of_le hr_lt h_stable_ge_s2
          apply WasmValue.encodes_preserved_of_fresh _ henc
          intro r' hr'
          simp [SymVal.regs] at hr'
          subst hr'
          exact hr_lt
      · -- Fresh: nextReg = s2.nextReg + 3.
        refine ⟨?_, ?_⟩
        · intro sv hsv r hr
          simp at hsv
          rcases hsv with h_eq | h_in
          · subst h_eq
            simp [SymVal.regs] at hr
            subst hr
            -- r = post_fresh = s2.nextReg + 1 + 1 < s2.nextReg + 1 + 1 + 1.
            exact Nat.lt_succ_self _
          · have hsv_in_s2 : sv ∈ s2.stack := h_in
            have h := R1.fresh.left sv hsv_in_s2 r hr
            exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt (Nat.lt_succ_of_lt h))
        · intro ir hir
          simp at hir
          rcases hir with h_eq | ⟨h_in, _⟩
          · subst h_eq
            -- (i, stable).snd = stable = s2.nextReg + 1 < s2.nextReg + 1 + 1 + 1.
            exact Nat.lt_succ_of_lt (Nat.lt_succ_self _)
          · have hin_s2 : ir ∈ s2.localReg := by rw [h_s2_lr]; exact h_in
            have h := R1.fresh.right ir hin_s2
            exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt (Nat.lt_succ_of_lt h))
      · -- AliasFree: localReg = (i, stable) :: filter(≠i) s.localReg.
        intro ir hir sv hsv
        simp at hir hsv
        rcases hir with hir_eq | ⟨hir_in, _⟩ <;>
        rcases hsv with hsv_eq | hsv_in
        · subst hir_eq; subst hsv_eq
          -- stable ∉ (.reg post_fresh .u32).regs = {post_fresh}; stable ≠ post_fresh.
          simp [SymVal.regs, Nat.ne_of_lt h_post_fresh_gt_stable]
        · subst hir_eq
          have hsv_in_s2 : sv ∈ s2.stack := hsv_in
          intro hcontra
          -- stable ∈ sv.regs ⇒ stable < s2.nextReg, but stable = s2.nextReg + 1.
          have h_lt : stable < s2.nextReg :=
            R1.fresh.left sv hsv_in_s2 stable hcontra
          show False
          have : s2.nextReg + 1 < s2.nextReg := h_lt
          omega
        · subst hsv_eq
          have hin_s2 : ir ∈ s2.localReg := by rw [h_s2_lr]; exact hir_in
          have ir_lt : ir.snd < s2.nextReg := R1.fresh.right ir hin_s2
          simp [SymVal.regs]
          -- ir.snd ≠ post_fresh; ir.snd < s2.nextReg < post_fresh.
          exact Nat.ne_of_lt (Nat.lt_of_lt_of_le ir_lt h_post_fresh_ge_s2)
        · have hsv_in_s2 : sv ∈ s2.stack := hsv_in
          have hin_s2 : ir ∈ s2.localReg := by rw [h_s2_lr]; exact hir_in
          exact R1.aliasFree ir hin_s2 sv hsv_in_s2
      · -- InjectiveLocals.
        intro p q hp hq
        simp at hp hq
        rcases hp with hp_eq | ⟨hp_in, hp_ne⟩ <;>
        rcases hq with hq_eq | ⟨hq_in, hq_ne⟩
        · subst hp_eq; subst hq_eq; left; rfl
        · right
          subst hp_eq
          have hin_s2 : q ∈ s2.localReg := by rw [h_s2_lr]; exact hq_in
          have h_lt : q.snd < s2.nextReg := R1.fresh.right q hin_s2
          show stable ≠ q.snd
          exact Ne.symm (Nat.ne_of_lt (Nat.lt_of_lt_of_le h_lt h_stable_ge_s2))
        · right
          subst hq_eq
          have hin_s2 : p ∈ s2.localReg := by rw [h_s2_lr]; exact hp_in
          have h_lt : p.snd < s2.nextReg := R1.fresh.right p hin_s2
          show p.snd ≠ stable
          exact Nat.ne_of_lt (Nat.lt_of_lt_of_le h_lt h_stable_ge_s2)
        · have hpin_s2 : p ∈ s2.localReg := by rw [h_s2_lr]; exact hp_in
          have hqin_s2 : q ∈ s2.localReg := by rw [h_s2_lr]; exact hq_in
          exact R1.injLocals p q hpin_s2 hqin_s2
      · -- CurrentRegRefines: s'.currentReg = (i, fresh) :: filter(≠i) s2.currentReg.
        intro k r_cur hfind v hv
        by_cases hki : k = i
        · subst hki
          change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((k, fresh) :: List.filter (fun p => !decide (p.fst = k)) s2.currentReg)
                 = some (k, r_cur) at hfind
          change (ws.locals.set k (WasmValue.wI32 n_w)).get? k = some v at hv
          rw [List.find?_cons] at hfind
          simp only [show decide ((k, fresh).fst = k) = true from by simp] at hfind
          injection hfind with h_pair
          have hr_eq : fresh = r_cur := (Prod.ext_iff.mp h_pair).2
          subst hr_eq
          have hv_eq : v = WasmValue.wI32 n_w := by
            have hget : (ws.locals.set k (.wI32 n_w)).get? k =
                        some (WasmValue.wI32 n_w) := by
              rw [List.get?_eq_getElem?]
              exact List.getElem?_set_self (by simpa using hbound)
            rw [hget] at hv
            exact ((Option.some.injEq _ _).mp hv).symm
          subst hv_eq
          simp [WasmValue.encodes]; exact h_lookup_fresh_kst'
        · -- Off-i: lift R1.currentReg past 3 writes.
          change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((i, fresh) :: List.filter (fun p => !decide (p.fst = i)) s2.currentReg)
                 = some (k, r_cur) at hfind
          rw [find?_setLocalReg_ne _ i k _ hki] at hfind
          have hv_old : ws.locals.get? k = some v := by
            rw [List.get?_eq_getElem?] at hv ⊢
            rw [List.getElem?_set_ne (Ne.symm hki)] at hv
            exact hv
          have henc := R1.currentReg k r_cur hfind v hv_old
          have hpair_cur : (k, r_cur) ∈ s2.currentReg :=
            List.mem_of_find?_eq_some hfind
          have hpair_s_cur : (k, r_cur) ∈ s.currentReg := by rw [← h_s2_cr]; exact hpair_cur
          have hr_cur_lt_s : r_cur < s.nextReg := R.freshCurrent (k, r_cur) hpair_s_cur
          have hr_cur_lt_s2 : r_cur < s2.nextReg :=
            Nat.lt_of_lt_of_le hr_cur_lt_s _h_s_le_s2
          -- Lift past three regWrites at fresh, stable, post_fresh (all ≥ s2.nextReg).
          apply WasmValue.encodes_preserved_of_fresh _ _
          · intro r' hr'
            simp [SymVal.regs] at hr'
            subst hr'
            exact Nat.lt_of_lt_of_le hr_cur_lt_s2 h_post_fresh_ge_s2
          apply WasmValue.encodes_preserved_of_fresh _ _
          · intro r' hr'
            simp [SymVal.regs] at hr'
            subst hr'
            exact Nat.lt_of_lt_of_le hr_cur_lt_s2 h_stable_ge_s2
          apply WasmValue.encodes_preserved_of_fresh _ henc
          intro r' hr'
          simp [SymVal.regs] at hr'
          subst hr'
          exact hr_cur_lt_s2
      · -- FreshCurrent: every (k, r) in s'.currentReg has r < s'.nextReg.
        intro ir hir
        simp at hir
        rcases hir with h_eq | ⟨h_in, _⟩
        · subst h_eq
          -- (i, fresh).snd = fresh = s2.nextReg < s2.nextReg + 3.
          show fresh < s2.nextReg + 1 + 1 + 1
          exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt (Nat.lt_succ_self _))
        · have hin_s : ir ∈ s.currentReg := by rw [← h_s2_cr]; exact h_in
          have h := R.freshCurrent ir hin_s
          have h_s_s2 : s.nextReg ≤ s2.nextReg := _h_s_le_s2
          exact Nat.lt_of_lt_of_le h
            (by omega : s.nextReg ≤ s2.nextReg + 1 + 1 + 1)
      · -- CurrentLocalDisjoint on s'.currentReg = (i, fresh) :: …,
        -- s'.localReg = (i, stable) :: ….
        intro p q hp hq hpq
        simp at hp hq
        rcases hp with hp_eq | ⟨hp_in, hp_ne⟩ <;>
        rcases hq with hq_eq | ⟨hq_in, hq_ne⟩
        · subst hp_eq; subst hq_eq; exact absurd rfl hpq
        · -- p = (i, fresh), q ∈ filter s.localReg with q.fst ≠ i.
          subst hp_eq
          have hq_lt : q.snd < s.nextReg := R.fresh.right q hq_in
          have hq_lt_s2 : q.snd < s2.nextReg :=
            Nat.lt_of_lt_of_le hq_lt _h_s_le_s2
          exact Ne.symm (Nat.ne_of_lt hq_lt_s2)
        · -- p ∈ filter s2.currentReg with p.fst ≠ i, q = (i, stable).
          subst hq_eq
          have hp_in_s : p ∈ s.currentReg := by rw [← h_s2_cr]; exact hp_in
          have hp_lt : p.snd < s.nextReg := R.freshCurrent p hp_in_s
          have hp_lt_stable : p.snd < stable := by
            have hp_lt_s2 : p.snd < s2.nextReg :=
              Nat.lt_of_lt_of_le hp_lt _h_s_le_s2
            exact Nat.lt_of_lt_of_le hp_lt_s2 h_stable_ge_s2
          exact Nat.ne_of_lt hp_lt_stable
        · have hp_in_s : p ∈ s.currentReg := by rw [← h_s2_cr]; exact hp_in
          exact R.curLocDisj p q hp_in_s hq_in hpq
  -- Case A: existing entry. stable_old = entry.snd, post_fresh = fresh + 1.
  · simp [hreg_find] at hl
    obtain ⟨hs_eq, hops_eq⟩ := hl
    have hentry_fst : entry.fst = i := by
      have := List.find?_some hreg_find
      simpa using this
    let stable_old : Reg := entry.snd
    let post_fresh_A : Reg := s2.nextReg + 1
    let kst_after_fresh_A : Quanta.KOps.State :=
      { kst1 with rf := regWrite kst1.rf fresh (vU32 n_w) }
    let kst_after_stable_A : Quanta.KOps.State :=
      { kst_after_fresh_A with
          rf := regWrite kst_after_fresh_A.rf stable_old (vU32 n_w) }
    let kst'_A : Quanta.KOps.State :=
      { kst_after_stable_A with
          rf := regWrite kst_after_stable_A.rf post_fresh_A (vU32 n_w) }
    have hentry_in : entry ∈ s.localReg :=
      List.mem_of_find?_eq_some hreg_find
    have hentry_in_s2 : entry ∈ s2.localReg := by rw [h_s2_lr]; exact hentry_in
    have hentry_pair : (i, stable_old) ∈ s.localReg := by
      have h_eq : entry = (i, stable_old) := by
        show entry = (i, entry.snd)
        rcases entry with ⟨ek, er⟩
        simp at hentry_fst
        simp [hentry_fst]
      rw [← h_eq]; exact hentry_in
    have hentry_pair_s2 : (i, stable_old) ∈ s2.localReg := by
      rw [h_s2_lr]; exact hentry_pair
    have h_stable_old_lt_s2 : stable_old < s2.nextReg :=
      R1.fresh.right entry hentry_in_s2
    have h_stable_old_ne_fresh : stable_old ≠ fresh :=
      Nat.ne_of_lt h_stable_old_lt_s2
    have h_stable_old_ne_post_fresh : stable_old ≠ post_fresh_A := by
      show stable_old ≠ s2.nextReg + 1
      have : stable_old < s2.nextReg + 1 := Nat.lt_succ_of_lt h_stable_old_lt_s2
      exact Nat.ne_of_lt this
    have h_post_fresh_A_gt_fresh : fresh < post_fresh_A := Nat.lt_succ_self _
    refine ⟨kst'_A, ?_, ?_⟩
    · -- evalOps closes via three Copy steps, similar to case B but
      -- with stable_old instead of stable.
      subst hops_eq
      -- Absorb the frame-0 zero-init (Const fresh _ overwritten by Copy fresh src).
      have h_src_ne_fresh : src ≠ fresh := Nat.ne_of_lt h_src_lt_s2
      rw [evalOps_append h_evalC h_kst1_ok,
          evalOps_const_copy_absorb h_src_ne_fresh h_kst1_ok,
          ← evalOps_append h_evalC h_kst1_ok]
      have h_evalC1 : evalOps 0 kst1 [.copy fresh src] = some kst_after_fresh_A := by
        simp only [evalOps, Quanta.KOps.evalOp]
        rw [h_src_lookup]
        simp only [Option.bind_eq_bind, Option.bind, Option.some_bind, pure, h_kst1_ok]
        rw [if_neg (by decide : ¬ (false = true))]
        congr 1
        show ({ kst1 with rf := regWrite kst1.rf fresh (vU32 n_w),
                          broke := false } : Quanta.KOps.State) = kst_after_fresh_A
        rw [show (false : Bool) = kst1.broke from h_kst1_ok.symm]
      have h_kaf_ok : kst_after_fresh_A.broke = false := h_kst1_ok
      have h_lookup_fresh_kaf :
          regLookup kst_after_fresh_A.rf fresh = some (vU32 n_w) := by
        show regLookup (regWrite kst1.rf fresh (vU32 n_w)) fresh = _
        rw [regLookup_regWrite_self]
      have h_evalC2 :
          evalOps 0 kst_after_fresh_A [.copy stable_old fresh] = some kst_after_stable_A := by
        simp only [evalOps, Quanta.KOps.evalOp]
        rw [h_lookup_fresh_kaf]
        simp only [Option.bind_eq_bind, Option.bind, Option.some_bind, pure, h_kaf_ok]
        rw [if_neg (by decide : ¬ (false = true))]
        congr 1
        show ({ kst_after_fresh_A with
                  rf := regWrite kst_after_fresh_A.rf stable_old (vU32 n_w),
                  broke := false } : Quanta.KOps.State) = kst_after_stable_A
        rw [show (false : Bool) = kst_after_fresh_A.broke from h_kaf_ok.symm]
      have h_kas_ok : kst_after_stable_A.broke = false := h_kaf_ok
      have h_lookup_fresh_kas :
          regLookup kst_after_stable_A.rf fresh = some (vU32 n_w) := by
        show regLookup (regWrite kst_after_fresh_A.rf stable_old (vU32 n_w)) fresh = _
        rw [regLookup_regWrite_of_ne _ stable_old fresh _
              (Ne.symm h_stable_old_ne_fresh)]
        exact h_lookup_fresh_kaf
      have h_evalC3 :
          evalOps 0 kst_after_stable_A [.copy post_fresh_A fresh] = some kst'_A := by
        simp only [evalOps, Quanta.KOps.evalOp]
        rw [h_lookup_fresh_kas]
        simp only [Option.bind_eq_bind, Option.bind, Option.some_bind, pure, h_kas_ok]
        rw [if_neg (by decide : ¬ (false = true))]
        congr 1
        show ({ kst_after_stable_A with
                  rf := regWrite kst_after_stable_A.rf post_fresh_A (vU32 n_w),
                  broke := false } : Quanta.KOps.State) = kst'_A
        rw [show (false : Bool) = kst_after_stable_A.broke from h_kas_ok.symm]
      rw [show (opsCommit ++ [KernelOp.copy fresh src,
                              KernelOp.copy stable_old fresh,
                              KernelOp.copy post_fresh_A fresh])
            = opsCommit ++ [KernelOp.copy fresh src]
                       ++ [KernelOp.copy stable_old fresh]
                       ++ [KernelOp.copy post_fresh_A fresh] from by
            simp [List.append_assoc]]
      rw [evalOps_append
            (evalOps_append (evalOps_append h_evalC h_kst1_ok ▸ h_evalC1) h_kaf_ok ▸ h_evalC2)
            h_kas_ok]
      exact h_evalC3
    · -- Refines on the post-state.
      subst hs_eq
      have h_lookup_post_fresh_kst'_A :
          regLookup kst'_A.rf post_fresh_A = some (vU32 n_w) := by
        show regLookup (regWrite kst_after_stable_A.rf post_fresh_A (vU32 n_w)) post_fresh_A = _
        rw [regLookup_regWrite_self]
      have h_lookup_stable_kst'_A :
          regLookup kst'_A.rf stable_old = some (vU32 n_w) := by
        show regLookup (regWrite kst_after_stable_A.rf post_fresh_A (vU32 n_w)) stable_old = _
        rw [regLookup_regWrite_of_ne _ post_fresh_A stable_old _
              h_stable_old_ne_post_fresh]
        show regLookup (regWrite kst_after_fresh_A.rf stable_old (vU32 n_w)) stable_old = _
        rw [regLookup_regWrite_self]
      have h_lookup_fresh_kst'_A :
          regLookup kst'_A.rf fresh = some (vU32 n_w) := by
        show regLookup (regWrite kst_after_stable_A.rf post_fresh_A (vU32 n_w)) fresh = _
        rw [regLookup_regWrite_of_ne _ post_fresh_A fresh _
              (Nat.ne_of_lt h_post_fresh_A_gt_fresh)]
        show regLookup (regWrite kst_after_fresh_A.rf stable_old (vU32 n_w)) fresh = _
        rw [regLookup_regWrite_of_ne _ stable_old fresh _
              (Ne.symm h_stable_old_ne_fresh)]
        show regLookup (regWrite kst1.rf fresh (vU32 n_w)) fresh = _
        rw [regLookup_regWrite_self]
      refine ⟨?_, ?_, ?_, ?_, ?_, R1.heapRefines, ?_, ?_, ?_⟩
      · -- StackRefines: top wI32 n_w ↔ .reg post_fresh_A .u32; tail lift.
        refine ⟨?_, ?_⟩
        · show (WasmValue.wI32 n_w :: rest).length
              = (SymVal.reg post_fresh_A .u32 :: s2.stack).length
          rw [h_s2_stack]; simpa using h_rest_lrest_len
        · intro j v hv
          cases j with
          | zero =>
            simp at hv
            refine ⟨SymVal.reg post_fresh_A .u32, ?_, ?_⟩
            · show (SymVal.reg post_fresh_A .u32 :: s2.stack).get? 0
                  = some (SymVal.reg post_fresh_A .u32)
              rfl
            subst hv
            simp [WasmValue.encodes]; exact h_lookup_post_fresh_kst'_A
          | succ k =>
            have hk : ws_pop.stack.get? k = some v := by
              show rest.get? k = some v; simpa using hv
            obtain ⟨svk, hsvk_get, henc⟩ := R1.stk.right k v hk
            refine ⟨svk, by simpa using hsvk_get, ?_⟩
            have hsvk_in : svk ∈ s2.stack := List.mem_of_get? hsvk_get
            -- Lift past write at stable_old (no alias by aliasFree on entry),
            -- then past two fresh writes at fresh, post_fresh_A.
            apply WasmValue.encodes_preserved_of_fresh _ _
            · -- post_fresh_A: r < s2.nextReg ≤ s2.nextReg + 1 = post_fresh_A.
              intro r hr
              have hr_s2 : r < s2.nextReg := R1.fresh.left svk hsvk_in r hr
              exact Nat.lt_succ_of_lt hr_s2
            apply WasmValue.encodes_preserved_of_disjoint _ _
            · -- stable_old: alias-free on entry rules out r = stable_old.
              exact R1.aliasFree entry hentry_in_s2 svk hsvk_in
            apply WasmValue.encodes_preserved_of_fresh _ henc
            intro r hr
            exact R1.fresh.left svk hsvk_in r hr
      · -- LocalsRefines. s'.localReg = (i, stable_old) :: filter(≠i) s.localReg.
        intro k r hfind v hv
        by_cases hki : k = i
        · subst hki
          change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((k, stable_old) :: List.filter (fun p => !decide (p.fst = k)) s.localReg)
                 = some (k, r) at hfind
          change (ws.locals.set k (WasmValue.wI32 n_w)).get? k = some v at hv
          rw [List.find?_cons] at hfind
          simp only [show decide ((k, stable_old).fst = k) = true from by simp] at hfind
          injection hfind with h_pair
          have hr_eq : stable_old = r := (Prod.ext_iff.mp h_pair).2
          subst hr_eq
          have hv_eq : v = WasmValue.wI32 n_w := by
            have hget : (ws.locals.set k (.wI32 n_w)).get? k =
                        some (WasmValue.wI32 n_w) := by
              rw [List.get?_eq_getElem?]
              exact List.getElem?_set_self (by simpa using hbound)
            rw [hget] at hv
            exact ((Option.some.injEq _ _).mp hv).symm
          subst hv_eq
          simp [WasmValue.encodes]; exact h_lookup_stable_kst'_A
        · change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((i, stable_old) :: List.filter (fun p => !decide (p.fst = i)) s.localReg)
                 = some (k, r) at hfind
          rw [find?_setLocalReg_ne _ i k _ hki] at hfind
          have hv_old : ws.locals.get? k = some v := by
            rw [List.get?_eq_getElem?] at hv ⊢
            rw [List.getElem?_set_ne (Ne.symm hki)] at hv
            exact hv
          have hfind_s2 : s2.localReg.find? (fun p => p.fst = k) = some (k, r) := by
            rw [h_s2_lr]; exact hfind
          have henc := R1.locs k r hfind_s2 v hv_old
          have hkr_in_s2 : (k, r) ∈ s2.localReg :=
            List.mem_of_find?_eq_some hfind_s2
          have hr_lt : r < s2.nextReg := R1.fresh.right (k, r) hkr_in_s2
          have hr_ne_stable_old : r ≠ stable_old := by
            have := R1.injLocals (k, r) (i, stable_old) hkr_in_s2 hentry_pair_s2
            rcases this with h_keq | h_rne
            · exact absurd h_keq hki
            · exact h_rne
          -- Lift past three writes: post_fresh_A (= s2.nextReg+1 > r),
          -- stable_old (≠ r by injLocals), fresh (= s2.nextReg > r).
          apply WasmValue.encodes_preserved_of_fresh _ _
          · intro r' hr'
            simp [SymVal.regs] at hr'
            subst hr'
            -- post_fresh_A = s2.nextReg + 1 > r.
            exact Nat.lt_succ_of_lt hr_lt
          apply WasmValue.encodes_preserved_of_disjoint _ _
          · simp [SymVal.regs]; exact Ne.symm hr_ne_stable_old
          apply WasmValue.encodes_preserved_of_fresh _ henc
          intro r' hr'
          simp [SymVal.regs] at hr'
          subst hr'
          exact hr_lt
      · -- Fresh: nextReg bumps by 2 (fresh + post_fresh_A).
        refine ⟨?_, ?_⟩
        · intro sv hsv r hr
          simp at hsv
          rcases hsv with h_eq | h_in
          · subst h_eq
            simp [SymVal.regs] at hr
            subst hr
            -- r = post_fresh_A = s2.nextReg + 1 < s2.nextReg + 1 + 1.
            exact Nat.lt_succ_self _
          · have hsv_in_s2 : sv ∈ s2.stack := h_in
            have h := R1.fresh.left sv hsv_in_s2 r hr
            exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt h)
        · intro ir hir
          simp at hir
          rcases hir with h_eq | ⟨h_in, _⟩
          · subst h_eq
            -- (i, stable_old).snd = stable_old < s2.nextReg < s2.nextReg + 1 + 1.
            exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt h_stable_old_lt_s2)
          · have hin_s2 : ir ∈ s2.localReg := by rw [h_s2_lr]; exact h_in
            have h := R1.fresh.right ir hin_s2
            exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt h)
      · -- AliasFree. localReg = (i, stable_old) :: filter(≠i) s.localReg.
        intro ir hir sv hsv
        simp at hir hsv
        rcases hir with hir_eq | ⟨hir_in, _⟩ <;>
        rcases hsv with hsv_eq | hsv_in
        · subst hir_eq; subst hsv_eq
          -- (i, stable_old).snd = stable_old, sv.regs = [post_fresh_A].
          show stable_old ∉ (SymVal.reg post_fresh_A .u32).regs
          simp [SymVal.regs]
          exact h_stable_old_ne_post_fresh
        · subst hir_eq
          have hsv_in_s2 : sv ∈ s2.stack := hsv_in
          -- stable_old ∉ sv.regs by aliasFree on entry.
          exact R1.aliasFree entry hentry_in_s2 sv hsv_in_s2
        · subst hsv_eq
          have hin_s2 : ir ∈ s2.localReg := by rw [h_s2_lr]; exact hir_in
          have ir_lt : ir.snd < s2.nextReg := R1.fresh.right ir hin_s2
          simp [SymVal.regs]
          exact Nat.ne_of_lt (Nat.lt_succ_of_lt ir_lt)
        · have hsv_in_s2 : sv ∈ s2.stack := hsv_in
          have hin_s2 : ir ∈ s2.localReg := by rw [h_s2_lr]; exact hir_in
          exact R1.aliasFree ir hin_s2 sv hsv_in_s2
      · -- InjectiveLocals.
        intro p q hp hq
        simp at hp hq
        rcases hp with hp_eq | ⟨hp_in, hp_ne⟩ <;>
        rcases hq with hq_eq | ⟨hq_in, hq_ne⟩
        · subst hp_eq; subst hq_eq; left; rfl
        · right
          subst hp_eq
          have hin_s2 : q ∈ s2.localReg := by rw [h_s2_lr]; exact hq_in
          have h_old := R1.injLocals q (i, stable_old) hin_s2 hentry_pair_s2
          rcases h_old with h_keq | h_rne
          · exact absurd h_keq hq_ne
          · exact h_rne.symm
        · right
          subst hq_eq
          have hin_s2 : p ∈ s2.localReg := by rw [h_s2_lr]; exact hp_in
          have h_old := R1.injLocals p (i, stable_old) hin_s2 hentry_pair_s2
          rcases h_old with h_keq | h_rne
          · exact absurd h_keq hp_ne
          · exact h_rne
        · have hpin_s2 : p ∈ s2.localReg := by rw [h_s2_lr]; exact hp_in
          have hqin_s2 : q ∈ s2.localReg := by rw [h_s2_lr]; exact hq_in
          exact R1.injLocals p q hpin_s2 hqin_s2
      · -- CurrentRegRefines. s'.currentReg = (i, fresh) :: filter(≠i) s2.currentReg.
        intro k r_cur hfind v hv
        by_cases hki : k = i
        · subst hki
          change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((k, fresh) :: List.filter (fun p => !decide (p.fst = k)) s2.currentReg)
                 = some (k, r_cur) at hfind
          change (ws.locals.set k (WasmValue.wI32 n_w)).get? k = some v at hv
          rw [List.find?_cons] at hfind
          simp only [show decide ((k, fresh).fst = k) = true from by simp] at hfind
          injection hfind with h_pair
          have hr_eq : fresh = r_cur := (Prod.ext_iff.mp h_pair).2
          subst hr_eq
          have hv_eq : v = WasmValue.wI32 n_w := by
            have hget : (ws.locals.set k (.wI32 n_w)).get? k =
                        some (WasmValue.wI32 n_w) := by
              rw [List.get?_eq_getElem?]
              exact List.getElem?_set_self (by simpa using hbound)
            rw [hget] at hv
            exact ((Option.some.injEq _ _).mp hv).symm
          subst hv_eq
          simp [WasmValue.encodes]; exact h_lookup_fresh_kst'_A
        · -- Off-i: lift R1.currentReg via curLocDisj (for stable_old) + freshness (fresh, post_fresh_A).
          change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((i, fresh) :: List.filter (fun p => !decide (p.fst = i)) s2.currentReg)
                 = some (k, r_cur) at hfind
          rw [find?_setLocalReg_ne _ i k _ hki] at hfind
          have hv_old : ws.locals.get? k = some v := by
            rw [List.get?_eq_getElem?] at hv ⊢
            rw [List.getElem?_set_ne (Ne.symm hki)] at hv
            exact hv
          have henc := R1.currentReg k r_cur hfind v hv_old
          have hpair_cur : (k, r_cur) ∈ s2.currentReg :=
            List.mem_of_find?_eq_some hfind
          have hpair_s_cur : (k, r_cur) ∈ s.currentReg := by rw [← h_s2_cr]; exact hpair_cur
          have hr_cur_lt_s : r_cur < s.nextReg := R.freshCurrent (k, r_cur) hpair_s_cur
          have hr_cur_lt_s2 : r_cur < s2.nextReg :=
            Nat.lt_of_lt_of_le hr_cur_lt_s _h_s_le_s2
          have hr_cur_ne_stable : r_cur ≠ stable_old :=
            R.curLocDisj (k, r_cur) (i, stable_old) hpair_s_cur hentry_pair hki
          -- Lift past three writes: post_fresh_A, stable_old, fresh.
          apply WasmValue.encodes_preserved_of_fresh _ _
          · intro r' hr'
            simp [SymVal.regs] at hr'
            subst hr'
            -- post_fresh_A = s2.nextReg + 1 > r_cur.
            exact Nat.lt_succ_of_lt hr_cur_lt_s2
          apply WasmValue.encodes_preserved_of_disjoint _ _
          · simp [SymVal.regs]; exact Ne.symm hr_cur_ne_stable
          apply WasmValue.encodes_preserved_of_fresh _ henc
          intro r' hr'
          simp [SymVal.regs] at hr'
          subst hr'
          exact hr_cur_lt_s2
      · -- FreshCurrent: (i, fresh) head + filter.
        intro ir hir
        simp at hir
        rcases hir with h_eq | ⟨h_in, _⟩
        · subst h_eq
          -- fresh = s2.nextReg < s2.nextReg + 1 + 1.
          exact Nat.lt_succ_of_lt (Nat.lt_succ_self _)
        · have hin_s : ir ∈ s.currentReg := by rw [← h_s2_cr]; exact h_in
          have h := R.freshCurrent ir hin_s
          have h_s_s2 : s.nextReg ≤ s2.nextReg := _h_s_le_s2
          exact Nat.lt_of_lt_of_le h
            (by omega : s.nextReg ≤ s2.nextReg + 1 + 1)
      · -- CurrentLocalDisjoint on (i, fresh) :: ..., (i, stable_old) :: ...
        intro p q hp hq hpq
        simp at hp hq
        rcases hp with hp_eq | ⟨hp_in, hp_ne⟩ <;>
        rcases hq with hq_eq | ⟨hq_in, hq_ne⟩
        · subst hp_eq; subst hq_eq; exact absurd rfl hpq
        · subst hp_eq
          have hq_lt : q.snd < s.nextReg := R.fresh.right q hq_in
          have hq_lt_s2 : q.snd < s2.nextReg :=
            Nat.lt_of_lt_of_le hq_lt _h_s_le_s2
          exact Ne.symm (Nat.ne_of_lt hq_lt_s2)
        · subst hq_eq
          have hp_in_s : p ∈ s.currentReg := by rw [← h_s2_cr]; exact hp_in
          exact R.curLocDisj p (i, stable_old) hp_in_s hentry_pair hp_ne
        · have hp_in_s : p ∈ s.currentReg := by rw [← h_s2_cr]; exact hp_in
          exact R.curLocDisj p q hp_in_s hq_in hpq

-- ════════════════════════════════════════════════════════════════════
-- Slice 3 follow-up status
--
-- The `Refines.injLocals` invariant is in place, bringing the bundle
-- to the 5 clauses needed for `localSet` / `localTee` preservation:
--
--   * stk        — stack encoding refinement
--   * locs       — locals encoding refinement
--   * fresh      — every held register < nextReg
--   * aliasFree  — no stable_reg appears on the symbolic stack
--   * injLocals  — distinct local indices map to distinct stable_regs
--
-- All 22 closed per-instr theorems (nop, return, i32Const, localGet,
-- localSet, localTee, 10 binops, 6 cmps) produce a Refines bundle
-- with all 5 clauses. Slice 3 is fully closed.
--
-- Slice 4 — what's next:
--   * Buffer-pattern recognition arms in `lowerInstr` (steps 7-8 of
--     the original plan): bufferPtr / scaledIdx / bufferAccess SymVals
--     consumed by the next i32.load/i32.store into a typed Load/Store.
--   * `HeapRefines` clause added to `Refines` — the first non-stack-
--     non-locals refinement clause; will cascade through every
--     existing per-op proof (each adds a `heapRefines := R.heapRefines`
--     line).
--   * One archetype memory-op preservation proof for the bufferAccess
--     consumer.
-- ════════════════════════════════════════════════════════════════════

end Quanta.Wasm
