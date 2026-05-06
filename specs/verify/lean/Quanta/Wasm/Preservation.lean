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

## What ships now (slices 1 + 2 + 3 + slice-4 stack-type cascade)

* The full refinement bundle (`Refines` = stack + locals + freshness +
  alias-free).
* Register-file lemmas: `regLookup_regWrite_self`,
  `regLookup_regWrite_of_ne` (closed via `find?_pred_eq` induction),
  `regLookup_preserved_of_fresh`.
* Shape-extraction helpers: `binI32_some_shape`, `cmpI32_some_shape`,
  `lowerI32Bin_some_shape`, `lowerI32Cmp_some_shape`.
* Generic binop preservation `preservation_i32Bin_generic`, plus 10
  specializations covering the entire i32-binop family.
* Closed per-instruction theorems:
  - `preservation_nop`
  - `preservation_return`
  - `preservation_i32Const`
  - `preservation_localGet`
  - `preservation_localSet`
  - `preservation_localTee`
  - `preservation_i32{Add,Sub,Mul,And,Or,Xor,Shl,ShrU,DivU,RemU}`
  - `preservation_i32{Eq,Ne,LtU,LeU,GtU,GeU}`

That's **22 closed preservation theorems**. Slice 3 is now fully
closed — every i32 instruction in the lowered subset has a
preservation proof. The seven archetypes — empty-emit
no-state-change (`nop`), empty-emit halted-flag (`wreturn`),
single-op fresh-write (`i32Const`), no-op stack-push (`localGet`),
the **two-pop one-fresh-write binop** archetype (`i32Bin`), the
**two-pop two-op cmp+cast** archetype (`i32Cmp`, requires
`kst.broke = false`), the **single-op stable-write** archetype
(`localSet`), and the **two-op stable+fresh-write** archetype
(`localTee`, requires `kst.broke = false`) — cover the entire
slice-3 surface.

## What's next

**Slice 3 fully closed.** Remaining work is slice-4 (buffer-pattern
arms + HeapRefines) and beyond:
* alias-free invariant is now baked into
  `Refines.aliasFree`, and the Lean translator's `localGet`/`localTee`
  allocate fresh registers + Copy to break aliasing. The remaining
  gap is an `InjectiveLocals` invariant: distinct local indices map
  to distinct stable_regs. Without it, `localSet i` writing to
  stable_reg(i) could clobber the encoding of stable_reg(j) for
  j ≠ i. Add the invariant to `Refines` and prove preservation
  (mostly trivial — only `localSet`/`localTee` mutate `localReg`,
  and they always allocate fresh stable_regs when introducing a new
  key).

**Slice 4 — stack-type cascade (THIS COMMIT)**: `LowerState.stack`
is now `List SymVal`, `WasmValue.encodes` consumes a `SymVal`,
`Fresh` / `AliasFree` flatten through `SymVal.regs`, and the load-
bearing `WasmValue.encodes_preserved_of_fresh` lemma threads encoding
past every `regWrite kst.rf s.nextReg _`. Every existing per-op
proof now produces a `Refines` bundle parameterized by the new
SymVal-indexed stack. The buffer-pattern recognition arms
(`bufferPtr + scaledIdx → bufferAccess → typed Load/Store`) and a
`HeapRefines` clause are still future work — slice 4 steps 7-8 in
the original plan.

The cascade was expected to produce a clean delta because every
per-op proof was already structured to thread `R.fresh.left` /
`R.aliasFree` over the stack's reg projection. The single `regs`
helper added to `SymVal` collapses the projection into a list of
regs, and `WasmValue.encodes_preserved_of_fresh` collapses the
fresh-write preservation reasoning that previously inlined into
each proof's `cases v` ladder.

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
-- Refinement relation
-- ════════════════════════════════════════════════════════════════════

/-- A WASM value is encoded by a SymVal stack slot if the slot is a
    plain `.reg r .u32` and the regfile holds the matching `vU32` at
    that register. Other SymVal shapes (`bufferPtr`, `scaledIdx`,
    `i32ConstSym`, `bufferAccess`) only appear in the transient
    sequence of buffer-pattern recognition (slice-4 step 7); they are
    consumed inline before any value consumer fires, so they never
    need to satisfy a value refinement. -/
def WasmValue.encodes (v : WasmValue) (rf : Quanta.KOps.RegFile) (sv : SymVal) : Prop :=
  match v, sv with
  | .wI32 n, .reg r .u32 => regLookup rf r = some (Quanta.KOps.Value.vU32 n)
  | _, _ => False

/-- Stack refinement: WASM stack and symbolic stack zip element-wise
    through `WasmValue.encodes`. Length-aligned, top-aligned. -/
def StackRefines (ws : List WasmValue) (svs : List SymVal) (rf : Quanta.KOps.RegFile) : Prop :=
  ws.length = svs.length ∧
  ∀ i, ∀ v, ws.get? i = some v → ∃ sv, svs.get? i = some sv ∧ v.encodes rf sv

/-- Locals refinement: every local with a stable register encodes
    through that register, lifted into the symbolic alphabet as
    `.reg r .u32`. Locals not in `localReg` are unconstrained. -/
def LocalsRefines (locs : List WasmValue) (lreg : List (Nat × Reg)) (rf : Quanta.KOps.RegFile) : Prop :=
  ∀ i r, lreg.find? (fun p => p.fst = i) = some (i, r) →
    ∀ v, locs.get? i = some v → v.encodes rf (SymVal.reg r .u32)

/-- Freshness invariant: every register the lowering currently holds
    (any reg referenced by any stack SymVal, plus every local stable
    reg) is strictly less than `nextReg`. -/
def Fresh (s : LowerState) : Prop :=
  (∀ sv ∈ s.stack, ∀ r ∈ sv.regs, r < s.nextReg) ∧
  (∀ ir ∈ s.localReg, ir.snd < s.nextReg)

/-- Alias-free invariant: no local's stable register appears anywhere
    in the symbolic stack's reg projection. The Lean translator's
    `localGet`/`localTee` emit Copy ops to fresh registers precisely
    to maintain this — so a subsequent `localSet` writing to a
    stable_reg can't clobber a stack-aliased copy of the old value. -/
def AliasFree (s : LowerState) : Prop :=
  ∀ ir ∈ s.localReg, ∀ sv ∈ s.stack, ir.snd ∉ sv.regs

/-- Injective locals: distinct local indices map to distinct stable
    registers. Maintained by always allocating a fresh `s.nextReg` for
    a brand-new local entry, and never aliasing an existing entry. -/
def InjectiveLocals (s : LowerState) : Prop :=
  ∀ p q, p ∈ s.localReg → q ∈ s.localReg → p.fst = q.fst ∨ p.snd ≠ q.snd

/-- Bundle. -/
structure Refines (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State) : Prop where
  stk         : StackRefines ws.stack s.stack kst.rf
  locs        : LocalsRefines ws.locals s.localReg kst.rf
  fresh       : Fresh s
  aliasFree   : AliasFree s
  injLocals   : InjectiveLocals s

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
    {v : WasmValue} {rf : Quanta.KOps.RegFile} {sv : SymVal}
    {nr : Reg} {newval : Quanta.KOps.Value}
    (h_lt : ∀ r ∈ sv.regs, r < nr)
    (h : v.encodes rf sv) :
    v.encodes (regWrite rf nr newval) sv := by
  match v, sv, h with
  | .wI32 n, .reg r .u32, h =>
    have h' : regLookup rf r = some (Quanta.KOps.Value.vU32 n) := h
    have hr_lt : r < nr := h_lt r (by simp [SymVal.regs])
    show regLookup (regWrite rf nr newval) r = some (Quanta.KOps.Value.vU32 n)
    rw [regLookup_preserved_of_fresh hr_lt]
    exact h'

/-- Encoding is preserved under any register write disjoint from the
    SymVal's reg projection. The general-form companion to
    `encodes_preserved_of_fresh` used by `localSet` / `localTee`
    preservation, where the write target is an existing stable_reg
    (not strictly above all held regs) but is disjoint from the
    stack's regs by `AliasFree`. -/
theorem WasmValue.encodes_preserved_of_disjoint
    {v : WasmValue} {rf : Quanta.KOps.RegFile} {sv : SymVal}
    {dst : Reg} {newval : Quanta.KOps.Value}
    (h_disj : dst ∉ sv.regs)
    (h : v.encodes rf sv) :
    v.encodes (regWrite rf dst newval) sv := by
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

/-- Inverting a `wI32`-encoding-via-`.reg`: forces the scalar type to
    `.u32` and exposes the underlying regfile lookup. Used by the
    `localSet` / `localTee` proofs to extract the encoding constraint
    after `R.stk.right 0` returns a `.reg src tysrc` SymVal. -/
theorem WasmValue.encodes_wI32_reg_inv
    {n : UInt32} {rf : Quanta.KOps.RegFile} {r : Reg} {ty : Quanta.KOps.Scalar}
    (h : (WasmValue.wI32 n).encodes rf (.reg r ty)) :
    ty = .u32 ∧ regLookup rf r = some (Quanta.KOps.Value.vU32 n) := by
  match ty, h with
  | .u32, h => exact ⟨rfl, h⟩

/-- Stronger inversion: from `v.encodes rf (.reg r ty)` *non-False*,
    deduce `v = wI32 n` AND `ty = .u32` AND the regfile lookup. The
    only `WasmValue.encodes` arm with non-False content matches the
    `(wI32, reg _ .u32)` shape; every other case is `False`, so a
    proof of the non-False predicate forces the value/type shape. -/
theorem WasmValue.encodes_reg_shape
    {v : WasmValue} {rf : Quanta.KOps.RegFile} {r : Reg} {ty : Quanta.KOps.Scalar}
    (h : v.encodes rf (.reg r ty)) :
    ∃ n, v = .wI32 n ∧ ty = .u32 ∧ regLookup rf r = some (Quanta.KOps.Value.vU32 n) := by
  match v, ty, h with
  | .wI32 n, .u32, h => exact ⟨n, rfl, rfl, h⟩

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
    (R : Refines ws s kst)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws .nop = some ws')
    (hl : lowerInstr s .nop = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' := by
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
    (R : Refines ws s kst)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws .wreturn = some ws')
    (hl : lowerInstr s .wreturn = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' := by
  simp [evalInstr] at hw
  simp [lowerInstr] at hl
  obtain ⟨hs_eq, hops_eq⟩ := hl
  refine ⟨kst, ?_, ?_⟩
  · subst hops_eq
    simp [evalOps]
  · subst hs_eq
    refine ⟨?_, ?_, R.fresh, R.aliasFree, R.injLocals⟩
    · have : ws'.stack = ws.stack := by rw [← hw]
      rw [this]; exact R.stk
    · have : ws'.locals = ws.locals := by rw [← hw]
      rw [this]; exact R.locs

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
    (R : Refines ws s kst)
    (n : Int)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws (.i32Const n) = some ws')
    (hl : lowerInstr s (.i32Const n) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' := by
  simp [evalInstr, WasmState.push] at hw
  simp [lowerInstr, freshAndPush, LowerState.alloc, LowerState.push] at hl
  obtain ⟨hs_eq, hops_eq⟩ := hl
  refine ⟨{ kst with rf := regWrite kst.rf s.nextReg (vU32 (UInt32.ofNat n.toNat)) }, ?_, ?_⟩
  · subst hops_eq
    simp [evalOps, Quanta.KOps.evalOp, Quanta.KOps.evalConst]
  · subst hw hs_eq
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · -- Stack refinement.
      refine ⟨by simp [R.stk.left], ?_⟩
      intro i v hv
      cases i with
      | zero =>
        -- Top: WASM v = wI32 (UInt32.ofNat n.toNat), new top sv = .reg s.nextReg .u32.
        simp at hv
        refine ⟨SymVal.reg s.nextReg .u32, by simp, ?_⟩
        subst hv
        simp [WasmValue.encodes]
        rfl
      | succ k =>
        -- Below the top: prior stack survives the fresh write.
        have hwsk : ws.stack.get? k = some v := by simpa using hv
        obtain ⟨svk, hsvk_get, henc⟩ := R.stk.right k v hwsk
        refine ⟨svk, by simpa using hsvk_get, ?_⟩
        have hsvk_in : svk ∈ s.stack := List.mem_of_get? hsvk_get
        apply WasmValue.encodes_preserved_of_fresh _ henc
        intro r hr_in
        exact R.fresh.left svk hsvk_in r hr_in
    · -- Locals refinement: stable regs are < s.nextReg by Fresh.right.
      intro i r hfind v hv
      have hpair : (i, r) ∈ s.localReg := List.mem_of_find?_eq_some hfind
      have hr_lt : r < s.nextReg := R.fresh.right (i, r) hpair
      have henc := R.locs i r hfind v hv
      apply WasmValue.encodes_preserved_of_fresh _ henc
      intro r' hr'_in
      simp [SymVal.regs] at hr'_in
      subst hr'_in; exact hr_lt
    · -- Freshness: nextReg bumps to nextReg + 1; new top is .reg s.nextReg .u32.
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
    · -- AliasFree: localReg unchanged; new stack adds .reg s.nextReg .u32,
      -- whose regs = [s.nextReg]. Every stable_reg < s.nextReg by Fresh,
      -- so no collision; for old stack entries, IH AliasFree applies.
      intro ir hir sv hsv
      have hir_lt : ir.snd < s.nextReg := R.fresh.right ir hir
      simp at hsv
      rcases hsv with h_eq | h_in
      · subst h_eq
        simp [SymVal.regs]
        exact Nat.ne_of_lt hir_lt
      · exact R.aliasFree ir hir sv h_in
    · -- InjectiveLocals: localReg unchanged.
      exact R.injLocals

open Quanta.KOps (vU32) in
/-- `local.get i` preservation. Lowering allocates a fresh register
    `s.nextReg`, emits `Copy { dst := s.nextReg, src := stable_reg }`,
    pushes the fresh reg. WASM pushes `locals[i]`. The local's stable
    register encodes its value via `R.locs`, the Copy propagates that
    encoding to the fresh reg, and freshness keeps every prior reg
    readable. -/
theorem preservation_localGet (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (R : Refines ws s kst)
    (i : Nat)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws (.localGet i) = some ws')
    (hl : lowerInstr s (.localGet i) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' := by
  simp only [evalInstr, WasmState.getLocal, WasmState.push,
             Option.bind_eq_bind, Option.bind, pure] at hw
  match hloc : ws.locals.get? i, hw with
  | some v, hw =>
    simp only [lowerInstr, LowerState.lookupLocal, LowerState.alloc,
               LowerState.push, Option.bind_eq_bind, Option.bind, pure] at hl
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
      -- Local i must be `wI32 nv` (slice-1 encoding).
      have henc_local := R.locs i entry.snd hfind' v hloc
      cases v with
      | wI32 nv =>
        simp only [WasmValue.encodes] at henc_local
        -- henc_local : regLookup kst.rf entry.snd = some (Value.vU32 nv)
        refine ⟨{ kst with rf := regWrite kst.rf s.nextReg (Quanta.KOps.Value.vU32 nv) }, ?_, ?_⟩
        · subst hops_eq
          simp [evalOps, Quanta.KOps.evalOp, henc_local]
        · subst hs_eq; subst hw
          refine ⟨?_, ?_, ?_, ?_, ?_⟩
          · -- Stack refinement.
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
          · -- Locals refinement: regfile changed only at fresh reg.
            intro k r hk_find vk hvk
            have hpair : (k, r) ∈ s.localReg := List.mem_of_find?_eq_some hk_find
            have hr_lt : r < s.nextReg := R.fresh.right (k, r) hpair
            have henc' := R.locs k r hk_find vk hvk
            apply WasmValue.encodes_preserved_of_fresh _ henc'
            intro r' hr'_in
            simp [SymVal.regs] at hr'_in
            subst hr'_in; exact hr_lt
          · -- Freshness: new top is .reg s.nextReg .u32; old refs ≤ s.nextReg.
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
          · -- AliasFree: localReg unchanged, new stack adds .reg s.nextReg .u32
            -- whose regs = [s.nextReg], fresh ≠ any stable_reg.
            intro ir hir sv hsv
            have hir_lt : ir.snd < s.nextReg := R.fresh.right ir hir
            simp at hsv
            rcases hsv with h_eq | h_in
            · subst h_eq
              simp [SymVal.regs]
              exact Nat.ne_of_lt hir_lt
            · exact R.aliasFree ir hir sv h_in
          · -- InjectiveLocals: localReg unchanged.
            exact R.injLocals
      | _ =>
        unfold WasmValue.encodes at henc_local
        exact henc_local.elim
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

/-- Successful `lowerI32Bin` runs imply the symbolic stack had two
    `.reg _ _` slots on top (pop refuses other shapes), and the
    resulting state allocated a fresh register, pushed it boxed as
    `.reg s.nextReg .u32`, and emitted a single `binOp`. -/
theorem lowerI32Bin_some_shape {bop : Quanta.KOps.BinOp} {s s' : LowerState}
    {ops : List KernelOp} (h : lowerI32Bin s bop = some (s', ops)) :
    ∃ ra rb tya tyb lrest,
      s.stack = SymVal.reg rb tyb :: SymVal.reg ra tya :: lrest ∧
      s' = { nextReg := s.nextReg + 1,
             stack := SymVal.reg s.nextReg .u32 :: lrest,
             localReg := s.localReg,
             localTy := s.localTy } ∧
      ops = [.binOp s.nextReg ra rb bop .u32] := by
  unfold lowerI32Bin at h
  rcases hs : s.stack with _ | ⟨svb, srest⟩
  · simp [hs, LowerState.pop] at h
  · cases svb with
    | reg rb tyb =>
      rcases hsr : srest with _ | ⟨sva, lrest⟩
      · simp [hs, hsr, LowerState.pop] at h
      · cases sva with
        | reg ra tya =>
          simp [hs, hsr, LowerState.pop, LowerState.alloc, LowerState.push] at h
          obtain ⟨hs', hops'⟩ := h
          refine ⟨ra, rb, tya, tyb, lrest, rfl, ?_, hops'.symm⟩
          exact hs'.symm
        | bufferPtr _          => simp [hs, hsr, LowerState.pop] at h
        | scaledIdx _ _        => simp [hs, hsr, LowerState.pop] at h
        | i32ConstSym _        => simp [hs, hsr, LowerState.pop] at h
        | bufferAccess _ _ _   => simp [hs, hsr, LowerState.pop] at h
    | bufferPtr _          => simp [hs, LowerState.pop] at h
    | scaledIdx _ _        => simp [hs, LowerState.pop] at h
    | i32ConstSym _        => simp [hs, LowerState.pop] at h
    | bufferAccess _ _ _   => simp [hs, LowerState.pop] at h

/-- Shape for `lowerI32Cmp`. The lowering emits TWO ops — `cmp` writing
    a vBool to `s.nextReg`, then `cast` lifting that vBool back into
    the u32 alphabet at `s.nextReg + 1`. The pushed slot is the cast's
    destination, typed `.u32` (matching every other value-producing
    arm). -/
theorem lowerI32Cmp_some_shape {cop : Quanta.KOps.CmpOp} {s s' : LowerState}
    {ops : List KernelOp} (h : lowerI32Cmp s cop = some (s', ops)) :
    ∃ ra rb tya tyb lrest,
      s.stack = SymVal.reg rb tyb :: SymVal.reg ra tya :: lrest ∧
      s' = { nextReg := s.nextReg + 2,
             stack := SymVal.reg (s.nextReg + 1) .u32 :: lrest,
             localReg := s.localReg,
             localTy := s.localTy } ∧
      ops = [.cmp s.nextReg ra rb cop .bool,
             .cast (s.nextReg + 1) s.nextReg .bool .u32] := by
  unfold lowerI32Cmp at h
  rcases hs : s.stack with _ | ⟨svb, srest⟩
  · simp [hs, LowerState.pop] at h
  · cases svb with
    | reg rb tyb =>
      rcases hsr : srest with _ | ⟨sva, lrest⟩
      · simp [hs, hsr, LowerState.pop] at h
      · cases sva with
        | reg ra tya =>
          simp [hs, hsr, LowerState.pop, LowerState.alloc, LowerState.push] at h
          obtain ⟨hs', hops'⟩ := h
          refine ⟨ra, rb, tya, tyb, lrest, rfl, ?_, hops'.symm⟩
          -- s' fields agree after two `alloc`s and one `push`.
          rw [← hs']
        | bufferPtr _          => simp [hs, hsr, LowerState.pop] at h
        | scaledIdx _ _        => simp [hs, hsr, LowerState.pop] at h
        | i32ConstSym _        => simp [hs, hsr, LowerState.pop] at h
        | bufferAccess _ _ _   => simp [hs, hsr, LowerState.pop] at h
    | bufferPtr _          => simp [hs, LowerState.pop] at h
    | scaledIdx _ _        => simp [hs, LowerState.pop] at h
    | i32ConstSym _        => simp [hs, LowerState.pop] at h
    | bufferAccess _ _ _   => simp [hs, LowerState.pop] at h

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
    instantiate with `rfl rfl (by intro …; rfl)`. -/
theorem preservation_i32Bin_generic
    (instr : WasmInstr) (op_w : UInt32 → UInt32 → UInt32)
    (op_k : Quanta.KOps.BinOp)
    (h_w : ∀ s, evalInstr s instr = binI32 op_w s)
    (h_l : ∀ s, lowerInstr s instr = lowerI32Bin s op_k)
    (h_agree : ∀ av bv,
       Quanta.KOps.evalBinOp op_k (vU32 av) (vU32 bv) = some (vU32 (op_w av bv)))
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (R : Refines ws s kst)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws instr = some ws')
    (hl : lowerInstr s instr = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' := by
  rw [h_w] at hw
  rw [h_l] at hl
  obtain ⟨av, bv, rest, hwstack, hws_eq⟩ := binI32_some_shape hw
  obtain ⟨ra, rb, tya, tyb, lrest, hlstack, hs_eq, hops_eq⟩ := lowerI32Bin_some_shape hl
  -- Extract reg encodings from R.stk applied at indices 0, 1.
  -- The encodings expose `regLookup … rb / ra = some (vU32 …)` only
  -- when the SymVal is `.reg _ .u32`; the shape-extraction helper
  -- handed us `.reg rb tyb / .reg ra tya`, but `WasmValue.encodes`
  -- being non-False forces tyb = tya = .u32.
  have hrb_enc : regLookup kst.rf rb = some (vU32 bv) := by
    have hb := R.stk.right 0 (.wI32 bv) (by rw [hwstack]; simp)
    obtain ⟨sv0, hsv0_get, henc⟩ := hb
    have hs0 : s.stack.get? 0 = some (SymVal.reg rb tyb) := by rw [hlstack]; simp
    rw [hs0] at hsv0_get
    have h_eq : SymVal.reg rb tyb = sv0 := (Option.some.injEq _ _).mp hsv0_get
    subst h_eq
    cases tyb <;> simp only [WasmValue.encodes] at henc <;>
      first | exact henc | exact henc.elim
  have hra_enc : regLookup kst.rf ra = some (vU32 av) := by
    have ha := R.stk.right 1 (.wI32 av) (by rw [hwstack]; simp)
    obtain ⟨sv1, hsv1_get, henc⟩ := ha
    have hs1 : s.stack.get? 1 = some (SymVal.reg ra tya) := by rw [hlstack]; simp
    rw [hs1] at hsv1_get
    have h_eq : SymVal.reg ra tya = sv1 := (Option.some.injEq _ _).mp hsv1_get
    subst h_eq
    cases tya <;> simp only [WasmValue.encodes] at henc <;>
      first | exact henc | exact henc.elim
  -- Build the final kst'.
  refine ⟨{ kst with rf := regWrite kst.rf s.nextReg (vU32 (op_w av bv)) }, ?_, ?_⟩
  · subst hops_eq
    simp [evalOps, Quanta.KOps.evalOp, hra_enc, hrb_enc, h_agree]
  · subst hs_eq; subst hws_eq
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · -- Stack refinement.
      refine ⟨?_, ?_⟩
      · -- Length: rest.length = lrest.length (from old R.stk on the
        -- 2-deep stacks).
        have hl_orig := R.stk.left
        rw [hwstack, hlstack] at hl_orig
        simpa using hl_orig
      · intro j v hv
        cases j with
        | zero =>
          simp at hv
          refine ⟨SymVal.reg s.nextReg .u32, by simp, ?_⟩
          subst hv
          simp [WasmValue.encodes]
          rfl
        | succ k =>
          have hrest_get : ws.stack.get? (k + 2) = some v := by
            rw [hwstack]; simpa using hv
          obtain ⟨svk, hsvk_get, henc⟩ := R.stk.right (k + 2) v hrest_get
          have hlrest_get : lrest.get? k = some svk := by
            have h2 : s.stack.get? (k + 2) = some svk := hsvk_get
            rw [hlstack] at h2; simpa using h2
          refine ⟨svk, by simpa using hlrest_get, ?_⟩
          have hsvk_in : svk ∈ s.stack := List.mem_of_get? hsvk_get
          apply WasmValue.encodes_preserved_of_fresh _ henc
          intro r' hr'_in
          exact R.fresh.left svk hsvk_in r' hr'_in
    · -- Locals refinement.
      intro i r hfind v hv
      have hpair : (i, r) ∈ s.localReg := List.mem_of_find?_eq_some hfind
      have hr_lt : r < s.nextReg := R.fresh.right (i, r) hpair
      have henc := R.locs i r hfind v hv
      apply WasmValue.encodes_preserved_of_fresh _ henc
      intro r' hr'_in
      simp [SymVal.regs] at hr'_in
      subst hr'_in; exact hr_lt
    · -- Freshness: new top is .reg s.nextReg .u32; lrest ⊆ s.stack.
      refine ⟨?_, ?_⟩
      · intro sv hsv r' hr'
        simp at hsv
        rcases hsv with h_eq | h_in
        · subst h_eq
          simp [SymVal.regs] at hr'
          subst hr'; exact Nat.lt_succ_self _
        · have hsv_in : sv ∈ s.stack := by rw [hlstack]; simp; right; right; exact h_in
          exact Nat.lt_succ_of_lt (R.fresh.left sv hsv_in r' hr')
      · intro ir hir
        exact Nat.lt_succ_of_lt (R.fresh.right ir hir)
    · -- AliasFree: localReg unchanged, new stack drops top 2 + adds
      -- .reg s.nextReg .u32. For the new top, regs = [s.nextReg],
      -- fresh ≠ any stable_reg. For lrest entries, IH AliasFree on
      -- s.stack ⊇ lrest applies.
      intro ir hir sv hsv
      have hir_lt : ir.snd < s.nextReg := R.fresh.right ir hir
      simp at hsv
      rcases hsv with h_eq | h_in
      · subst h_eq
        simp [SymVal.regs]
        exact Nat.ne_of_lt hir_lt
      · have hsv_in : sv ∈ s.stack := by rw [hlstack]; simp; right; right; exact h_in
        exact R.aliasFree ir hir sv hsv_in
    · -- InjectiveLocals: localReg unchanged.
      exact R.injLocals

-- ── Per-op specializations (10 binops) ─────────────────────────────────

def preservation_i32Add :=
  preservation_i32Bin_generic .i32Add eval_u32_wrapping_add .add
    (fun _ => rfl) (fun _ => rfl)
    (by intro av bv; rfl)

def preservation_i32Sub :=
  preservation_i32Bin_generic .i32Sub eval_u32_wrapping_sub .sub
    (fun _ => rfl) (fun _ => rfl)
    (by intro av bv; rfl)

def preservation_i32Mul :=
  preservation_i32Bin_generic .i32Mul eval_u32_wrapping_mul .mul
    (fun _ => rfl) (fun _ => rfl)
    (by intro av bv; rfl)

def preservation_i32And :=
  preservation_i32Bin_generic .i32And eval_u32_bitand .bAnd
    (fun _ => rfl) (fun _ => rfl)
    (by intro av bv; rfl)

def preservation_i32Or :=
  preservation_i32Bin_generic .i32Or eval_u32_bitor .bOr
    (fun _ => rfl) (fun _ => rfl)
    (by intro av bv; rfl)

def preservation_i32Xor :=
  preservation_i32Bin_generic .i32Xor eval_u32_bitxor .bXor
    (fun _ => rfl) (fun _ => rfl)
    (by intro av bv; rfl)

def preservation_i32Shl :=
  preservation_i32Bin_generic .i32Shl (fun a b => a <<< b) .shl
    (fun _ => rfl) (fun _ => rfl)
    (by intro av bv; rfl)

def preservation_i32ShrU :=
  preservation_i32Bin_generic .i32ShrU (fun a b => a >>> b) .shr
    (fun _ => rfl) (fun _ => rfl)
    (by intro av bv; rfl)

def preservation_i32DivU :=
  preservation_i32Bin_generic .i32DivU eval_u32_div .div
    (fun _ => rfl) (fun _ => rfl)
    (by intro av bv; rfl)

def preservation_i32RemU :=
  preservation_i32Bin_generic .i32RemU eval_u32_rem .rem
    (fun _ => rfl) (fun _ => rfl)
    (by intro av bv; rfl)

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

    Each of the 6 i32-cmp preservation theorems below is one line. -/
theorem preservation_i32Cmp_generic
    (instr : WasmInstr) (p_w : UInt32 → UInt32 → Bool)
    (op_k : Quanta.KOps.CmpOp)
    (h_w : ∀ s, evalInstr s instr = cmpI32 p_w s)
    (h_l : ∀ s, lowerInstr s instr = lowerI32Cmp s op_k)
    (h_agree : ∀ av bv,
       Quanta.KOps.evalCmpOp op_k (vU32 av) (vU32 bv)
         = some (Quanta.KOps.Value.vBool (p_w av bv)))
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (R : Refines ws s kst) (h_kst_ok : kst.broke = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws instr = some ws')
    (hl : lowerInstr s instr = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' := by
  rw [h_w] at hw
  rw [h_l] at hl
  obtain ⟨av, bv, rest, hwstack, hws_eq⟩ := cmpI32_some_shape hw
  obtain ⟨ra, rb, tya, tyb, lrest, hlstack, hs_eq, hops_eq⟩ := lowerI32Cmp_some_shape hl
  -- Operand encodings — same shape ladder as the binop generic.
  have hrb_enc : regLookup kst.rf rb = some (vU32 bv) := by
    have hb := R.stk.right 0 (.wI32 bv) (by rw [hwstack]; simp)
    obtain ⟨sv0, hsv0_get, henc⟩ := hb
    have hs0 : s.stack.get? 0 = some (SymVal.reg rb tyb) := by rw [hlstack]; simp
    rw [hs0] at hsv0_get
    have h_eq : SymVal.reg rb tyb = sv0 := (Option.some.injEq _ _).mp hsv0_get
    subst h_eq
    cases tyb <;> simp only [WasmValue.encodes] at henc <;>
      first | exact henc | exact henc.elim
  have hra_enc : regLookup kst.rf ra = some (vU32 av) := by
    have ha := R.stk.right 1 (.wI32 av) (by rw [hwstack]; simp)
    obtain ⟨sv1, hsv1_get, henc⟩ := ha
    have hs1 : s.stack.get? 1 = some (SymVal.reg ra tya) := by rw [hlstack]; simp
    rw [hs1] at hsv1_get
    have h_eq : SymVal.reg ra tya = sv1 := (Option.some.injEq _ _).mp hsv1_get
    subst h_eq
    cases tya <;> simp only [WasmValue.encodes] at henc <;>
      first | exact henc | exact henc.elim
  -- Final kst' has both writes applied: vBool at nextReg (transient),
  -- vU32 (if p_w av bv then 1 else 0) at nextReg + 1.
  refine ⟨{ kst with rf :=
              regWrite (regWrite kst.rf s.nextReg
                          (Quanta.KOps.Value.vBool (p_w av bv)))
                        (s.nextReg + 1)
                        (vU32 (if p_w av bv then 1 else 0)) }, ?_, ?_⟩
  · subst hops_eq
    simp [evalOps, Quanta.KOps.evalOp, hra_enc, hrb_enc, h_agree,
          regLookup_regWrite_self, Quanta.KOps.evalCast, h_kst_ok]
  · subst hs_eq; subst hws_eq
    -- Helper: lift any encoding past the two fresh writes (nextReg
    -- and nextReg+1), provided the encoding's regs are all < nextReg.
    -- Stated as a universally-quantified function so each clause can
    -- discharge its preservation obligation in one application.
    let h_lift : ∀ (sv : SymVal) (v : WasmValue),
        (∀ r ∈ sv.regs, r < s.nextReg) →
        v.encodes kst.rf sv →
        v.encodes (regWrite (regWrite kst.rf s.nextReg
                              (Quanta.KOps.Value.vBool (p_w av bv)))
                            (s.nextReg + 1)
                            (Quanta.KOps.Value.vU32 (if p_w av bv then 1 else 0))) sv :=
      fun sv v h_lt henc =>
        WasmValue.encodes_preserved_of_fresh
          (fun r hr => Nat.lt_succ_of_lt (h_lt r hr))
          (WasmValue.encodes_preserved_of_fresh h_lt henc)
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · -- Stack refinement.
      refine ⟨?_, ?_⟩
      · -- Length: rest.length = lrest.length (from old R.stk on the
        -- 2-deep stacks).
        have hl_orig := R.stk.left
        rw [hwstack, hlstack] at hl_orig
        simpa using hl_orig
      · intro j v hv
        cases j with
        | zero =>
          -- Top: WASM v = wI32 (if p_w av bv then 1 else 0); SymVal = .reg (nextReg+1) .u32.
          simp at hv
          refine ⟨SymVal.reg (s.nextReg + 1) .u32, by simp, ?_⟩
          subst hv
          simp [WasmValue.encodes, regLookup_regWrite_self]
          rfl
        | succ k =>
          -- Below the top: lift IH StackRefines past the two writes.
          have hrest_get : ws.stack.get? (k + 2) = some v := by
            rw [hwstack]; simpa using hv
          obtain ⟨svk, hsvk_get, henc⟩ := R.stk.right (k + 2) v hrest_get
          have hlrest_get : lrest.get? k = some svk := by
            have h2 : s.stack.get? (k + 2) = some svk := hsvk_get
            rw [hlstack] at h2; simpa using h2
          refine ⟨svk, by simpa using hlrest_get, ?_⟩
          have hsvk_in : svk ∈ s.stack := List.mem_of_get? hsvk_get
          exact h_lift svk v (fun r hr => R.fresh.left svk hsvk_in r hr) henc
    · -- Locals refinement.
      intro i r hfind v hv
      have hpair : (i, r) ∈ s.localReg := List.mem_of_find?_eq_some hfind
      have hr_lt : r < s.nextReg := R.fresh.right (i, r) hpair
      have henc := R.locs i r hfind v hv
      apply h_lift _ _ _ henc
      intro r' hr'_in
      simp [SymVal.regs] at hr'_in
      subst hr'_in; exact hr_lt
    · -- Freshness: nextReg bumps by 2; new top reg = nextReg + 1; lrest ⊆ s.stack.
      refine ⟨?_, ?_⟩
      · intro sv hsv r' hr'
        simp at hsv
        rcases hsv with h_eq | h_in
        · subst h_eq
          simp [SymVal.regs] at hr'
          subst hr'
          exact Nat.lt_succ_self _
        · have hsv_in : sv ∈ s.stack := by
            rw [hlstack]; simp; right; right; exact h_in
          have : r' < s.nextReg := R.fresh.left sv hsv_in r' hr'
          exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt this)
      · intro ir hir
        have : ir.snd < s.nextReg := R.fresh.right ir hir
        exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt this)
    · -- AliasFree: localReg unchanged; new stack drops top 2 + adds
      -- .reg (nextReg+1) .u32. Stable_regs are < nextReg < nextReg+1.
      intro ir hir sv hsv
      have hir_lt : ir.snd < s.nextReg := R.fresh.right ir hir
      simp at hsv
      rcases hsv with h_eq | h_in
      · subst h_eq
        simp [SymVal.regs]
        exact Nat.ne_of_lt (Nat.lt_succ_of_lt hir_lt)
      · have hsv_in : sv ∈ s.stack := by
          rw [hlstack]; simp; right; right; exact h_in
        exact R.aliasFree ir hir sv hsv_in
    · -- InjectiveLocals: localReg unchanged.
      exact R.injLocals

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
/-- `local.set i` preservation. Single-op emission (`.copy dst src`)
    so the inter-op short-circuit doesn't fire and we don't need
    `kst.broke = false`. -/
theorem preservation_localSet (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (R : Refines ws s kst)
    (i : Nat)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws (.localSet i) = some ws')
    (hl : lowerInstr s (.localSet i) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' := by
  -- WASM side: pop v_w from ws.stack, then setLocal i v_w.
  simp only [evalInstr, WasmState.pop,
             Option.bind_eq_bind, Option.bind, pure] at hw
  rcases hws_stack : ws.stack with _ | ⟨v_w, rest⟩
  · simp [hws_stack] at hw
  simp only [hws_stack, WasmState.setLocal] at hw
  -- Bound check on i: must hold or hw : `none = some ws'` is False.
  by_cases hbound : i < List.length ws.locals
  case neg => simp [if_neg hbound] at hw
  -- Pos case continues here.
  simp only [if_pos hbound] at hw
  -- Extract ws' = the record literal so ws'.locals etc. unfold cleanly.
  have hws'_eq : ws' = { locals := ws.locals.set i v_w, stack := rest,
                          mem := ws.mem, halted := ws.halted } :=
    ((Option.some.injEq _ _).mp hw).symm
  subst hws'_eq
  -- Lean side: pop src; cases on the popped SymVal then on lookupLocal.
  simp only [lowerInstr, LowerState.pop, LowerState.lookupLocal,
             LowerState.lookupLocalTy, LowerState.alloc, LowerState.setLocalReg,
             LowerState.push, Option.bind_eq_bind, Option.bind, pure] at hl
  rcases hls_stack : s.stack with _ | ⟨svsrc, lrest⟩
  · simp [hls_stack] at hl
  cases svsrc with
  | bufferPtr _          => simp [hls_stack] at hl
  | scaledIdx _ _        => simp [hls_stack] at hl
  | i32ConstSym _        => simp [hls_stack] at hl
  | bufferAccess _ _ _   => simp [hls_stack] at hl
  | reg src tysrc =>
    simp only [hls_stack] at hl
    -- v_w's encoding via R.stk.right 0.
    have hv_enc_raw : v_w.encodes kst.rf (.reg src tysrc) := by
      have h := R.stk.right 0 v_w (by rw [hws_stack]; simp)
      obtain ⟨sv0, hsv0_get, henc⟩ := h
      have hs0 : s.stack.get? 0 = some (.reg src tysrc) := by rw [hls_stack]; simp
      rw [hs0] at hsv0_get
      have h_eq : SymVal.reg src tysrc = sv0 := (Option.some.injEq _ _).mp hsv0_get
      rwa [← h_eq] at henc
    obtain ⟨n_w, hv_w_eq, htysrc, hsrc_lookup⟩ :=
      WasmValue.encodes_reg_shape hv_enc_raw
    subst hv_w_eq htysrc
    -- Branch on lookupLocal i — the translator's two-arm match.
    rcases hreg_find : s.localReg.find? (fun p => p.fst = i) with _ | entry
    -- Case B: fresh dst.
    · simp [hreg_find] at hl
      obtain ⟨hs_eq, hops_eq⟩ := hl
      -- dst = s.nextReg; s'.nextReg = s.nextReg + 1; ops = [.copy s.nextReg src].
      refine ⟨{ kst with rf := regWrite kst.rf s.nextReg
                            (Quanta.KOps.Value.vU32 n_w) }, ?_, ?_⟩
      · subst hops_eq
        simp [evalOps, Quanta.KOps.evalOp, hsrc_lookup]
      · subst hs_eq
        refine ⟨?_, ?_, ?_, ?_, ?_⟩
        · -- StackRefines: ws'.stack = rest, s'.stack = lrest. Lift each
          -- entry past the fresh write at s.nextReg.
          refine ⟨?_, ?_⟩
          · have hl_orig := R.stk.left
            rw [hws_stack, hls_stack] at hl_orig
            simpa using hl_orig
          · intro j v hv
            have hrest_get : ws.stack.get? (j + 1) = some v := by
              rw [hws_stack]; simpa using hv
            obtain ⟨svj, hsvj_get, henc⟩ := R.stk.right (j + 1) v hrest_get
            have hlrest_get : lrest.get? j = some svj := by
              have h2 : s.stack.get? (j + 1) = some svj := hsvj_get
              rw [hls_stack] at h2; simpa using h2
            refine ⟨svj, hlrest_get, ?_⟩
            have hsvj_in : svj ∈ s.stack := List.mem_of_get? hsvj_get
            apply WasmValue.encodes_preserved_of_fresh _ henc
            intro r hr
            exact R.fresh.left svj hsvj_in r hr
        · -- LocalsRefines for s'.localReg = (i, s.nextReg) :: filter (≠ i) ...
          intro k r hfind v hv
          -- Decompose s'.localReg.find? on the (k = i) vs (k ≠ i) split.
          by_cases hki : k = i
          · -- k = i: head matches, r = s.nextReg, v = locals.set i (.wI32 n_w) k = wI32 n_w.
            subst hki
            -- Normalize record-accessor + locals.set form via dsimp.
            change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                     ((k, s.nextReg) :: List.filter (fun p => !decide (p.fst = k)) s.localReg)
                   = some (k, r) at hfind
            change (ws.locals.set k (WasmValue.wI32 n_w)).get? k = some v at hv
            -- Reduce hfind: head matches predicate so find? returns some head.
            rw [List.find?_cons] at hfind
            simp only [show decide ((k, s.nextReg).fst = k) = true from by simp] at hfind
            -- hfind : some (k, s.nextReg) = some (k, r). Extract r = s.nextReg.
            injection hfind with h_pair
            have hr_eq : s.nextReg = r := (Prod.ext_iff.mp h_pair).2
            subst hr_eq
            -- v = wI32 n_w from the set at index k.
            have hv_eq : v = WasmValue.wI32 n_w := by
              have hget : (ws.locals.set k (.wI32 n_w)).get? k =
                          some (WasmValue.wI32 n_w) := by
                rw [List.get?_eq_getElem?]
                exact List.getElem?_set_self (by simpa using hbound)
              rw [hget] at hv
              exact ((Option.some.injEq _ _).mp hv).symm
            subst hv_eq
            simp [WasmValue.encodes, regLookup_regWrite_self]
          · -- k ≠ i: head doesn't match (head.fst = i ≠ k); recurse via filter,
            -- then collapse the filter via the helper.
            -- Normalize the record-accessor in hfind first.
            change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                     ((i, s.nextReg) :: List.filter (fun p => !decide (p.fst = i)) s.localReg)
                   = some (k, r) at hfind
            rw [find?_setLocalReg_ne _ i k _ hki] at hfind
            -- Now hfind : s.localReg.find? (... = k) = some (k, r). Use R.locs.
            have hv_old : ws.locals.get? k = some v := by
              rw [List.get?_eq_getElem?] at hv ⊢
              rw [List.getElem?_set_ne (Ne.symm hki)] at hv
              exact hv
            have henc := R.locs k r hfind v hv_old
            -- Lift past the fresh write.
            have hr_lt : r < s.nextReg := by
              have hpair : (k, r) ∈ s.localReg :=
                List.mem_of_find?_eq_some hfind
              exact R.fresh.right (k, r) hpair
            apply WasmValue.encodes_preserved_of_fresh _ henc
            intro r' hr'_in
            simp [SymVal.regs] at hr'_in
            subst hr'_in
            exact hr_lt
        · -- Fresh: nextReg = s.nextReg + 1.
          refine ⟨?_, ?_⟩
          · -- Stack regs (lrest ⊆ s.stack) all < s.nextReg < s.nextReg + 1.
            intro sv hsv r hr
            have hsv_in : sv ∈ s.stack := by
              rw [hls_stack]; exact List.mem_cons_of_mem _ hsv
            exact Nat.lt_succ_of_lt (R.fresh.left sv hsv_in r hr)
          · -- Local regs: (i, s.nextReg) :: filter ... All < s.nextReg + 1.
            intro ir hir
            simp at hir
            rcases hir with h_eq | ⟨h_in, _⟩
            · -- (i, s.nextReg). snd = s.nextReg < s.nextReg + 1.
              subst h_eq
              exact Nat.lt_succ_self _
            · -- Filtered entry — already < s.nextReg.
              exact Nat.lt_succ_of_lt (R.fresh.right ir h_in)
        · -- AliasFree: stable regs ∉ lrest's stack regs.
          intro ir hir sv hsv
          have hsv_in : sv ∈ s.stack := by
            rw [hls_stack]; exact List.mem_cons_of_mem _ hsv
          simp at hir
          rcases hir with h_eq | ⟨h_in, _⟩
          · -- ir = (i, s.nextReg). s.nextReg ∉ sv.regs by Fresh.
            subst h_eq
            intro hcontra
            have : s.nextReg < s.nextReg :=
              R.fresh.left sv hsv_in s.nextReg hcontra
            exact Nat.lt_irrefl _ this
          · -- Filtered entry — was in s.localReg.
            exact R.aliasFree ir h_in sv hsv_in
        · -- InjectiveLocals: head (i, s.nextReg) is fresh; filter preserves injectivity.
          intro p q hp hq
          simp at hp hq
          rcases hp with hp_eq | ⟨hp_in, hp_ne⟩ <;>
          rcases hq with hq_eq | ⟨hq_in, hq_ne⟩
          · -- Both head: same pair, fst equal.
            subst hp_eq; subst hq_eq; left; rfl
          · -- p head, q filtered. fst's differ ⇒ snds differ.
            right
            subst hp_eq
            have : q.snd < s.nextReg := R.fresh.right q hq_in
            exact (Nat.ne_of_lt this).symm
          · right
            subst hq_eq
            have : p.snd < s.nextReg := R.fresh.right p hp_in
            exact Nat.ne_of_lt this
          · -- Both filtered. By R.injLocals.
            exact R.injLocals p q hp_in hq_in
    -- Case A: existing dst = entry.snd.
    · simp [hreg_find] at hl
      obtain ⟨hs_eq, hops_eq⟩ := hl
      -- entry = (i, dst) where dst = entry.snd, by find?_some.
      have hentry_fst : entry.fst = i := by
        have := List.find?_some hreg_find
        simpa using this
      -- s'.nextReg = s.nextReg; ops = [.copy entry.snd src].
      refine ⟨{ kst with rf := regWrite kst.rf entry.snd
                            (Quanta.KOps.Value.vU32 n_w) }, ?_, ?_⟩
      · subst hops_eq
        simp [evalOps, Quanta.KOps.evalOp, hsrc_lookup]
      · subst hs_eq
        -- Pre-compute: entry ∈ s.localReg, dst < s.nextReg by Fresh,
        -- dst ∉ stack regs by AliasFree.
        have hentry_in : entry ∈ s.localReg :=
          List.mem_of_find?_eq_some hreg_find
        have hentry_pair : (i, entry.snd) ∈ s.localReg := by
          have : entry = (i, entry.snd) := by
            rcases entry with ⟨ek, er⟩
            simp at hentry_fst
            simp [hentry_fst]
          rw [← this]; exact hentry_in
        have hdst_lt : entry.snd < s.nextReg := R.fresh.right entry hentry_in
        refine ⟨?_, ?_, ?_, ?_, ?_⟩
        · -- StackRefines: lift past regWrite at entry.snd (a stable_reg, disjoint by AliasFree).
          refine ⟨?_, ?_⟩
          · have hl_orig := R.stk.left
            rw [hws_stack, hls_stack] at hl_orig
            simpa using hl_orig
          · intro j v hv
            have hrest_get : ws.stack.get? (j + 1) = some v := by
              rw [hws_stack]; simpa using hv
            obtain ⟨svj, hsvj_get, henc⟩ := R.stk.right (j + 1) v hrest_get
            have hlrest_get : lrest.get? j = some svj := by
              have h2 : s.stack.get? (j + 1) = some svj := hsvj_get
              rw [hls_stack] at h2; simpa using h2
            refine ⟨svj, hlrest_get, ?_⟩
            have hsvj_in : svj ∈ s.stack := List.mem_of_get? hsvj_get
            -- entry.snd ∉ svj.regs by AliasFree.
            have h_disj : entry.snd ∉ svj.regs :=
              R.aliasFree entry hentry_in svj hsvj_in
            exact WasmValue.encodes_preserved_of_disjoint h_disj henc
        · -- LocalsRefines for s'.localReg = (i, entry.snd) :: filter (≠ i) s.localReg.
          intro k r hfind v hv
          by_cases hki : k = i
          · subst hki
            change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                     ((k, entry.snd) :: List.filter (fun p => !decide (p.fst = k)) s.localReg)
                   = some (k, r) at hfind
            change (ws.locals.set k (WasmValue.wI32 n_w)).get? k = some v at hv
            rw [List.find?_cons] at hfind
            simp only [show decide ((k, entry.snd).fst = k) = true from by simp] at hfind
            injection hfind with h_pair
            have hr_eq : entry.snd = r := (Prod.ext_iff.mp h_pair).2
            subst hr_eq
            have hv_eq : v = WasmValue.wI32 n_w := by
              have hget : (ws.locals.set k (.wI32 n_w)).get? k =
                          some (WasmValue.wI32 n_w) := by
                rw [List.get?_eq_getElem?]
                exact List.getElem?_set_self (by simpa using hbound)
              rw [hget] at hv
              exact ((Option.some.injEq _ _).mp hv).symm
            subst hv_eq
            simp [WasmValue.encodes, regLookup_regWrite_self]
          · -- k ≠ i: same head-drop + filter-collapse as Case B.
            change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                     ((i, entry.snd) :: List.filter (fun p => !decide (p.fst = i)) s.localReg)
                   = some (k, r) at hfind
            rw [find?_setLocalReg_ne _ i k _ hki] at hfind
            have hv_old : ws.locals.get? k = some v := by
              rw [List.get?_eq_getElem?] at hv ⊢
              rw [List.getElem?_set_ne (Ne.symm hki)] at hv
              exact hv
            have henc := R.locs k r hfind v hv_old
            have hkr_in : (k, r) ∈ s.localReg :=
              List.mem_of_find?_eq_some hfind
            -- r ≠ entry.snd by InjectiveLocals on (i, entry.snd) and (k, r).
            have hr_ne : r ≠ entry.snd := by
              have := R.injLocals (k, r) (i, entry.snd) hkr_in hentry_pair
              rcases this with h_keq | h_rne
              · exact absurd h_keq hki
              · -- h_rne : (k, r).snd ≠ (i, entry.snd).snd, i.e. r ≠ entry.snd
                exact h_rne
            apply WasmValue.encodes_preserved_of_disjoint _ henc
            simp [SymVal.regs]
            exact hr_ne.symm
        · -- Fresh: nextReg unchanged.
          refine ⟨?_, ?_⟩
          · intro sv hsv r hr
            have hsv_in : sv ∈ s.stack := by
              rw [hls_stack]; exact List.mem_cons_of_mem _ hsv
            exact R.fresh.left sv hsv_in r hr
          · intro ir hir
            simp at hir
            rcases hir with h_eq | ⟨h_in, _⟩
            · -- ir = (i, entry.snd). entry.snd < s.nextReg.
              subst h_eq; exact hdst_lt
            · exact R.fresh.right ir h_in
        · -- AliasFree: same as Case B but entry.snd ∉ stack regs by old AliasFree.
          intro ir hir sv hsv
          have hsv_in : sv ∈ s.stack := by
            rw [hls_stack]; exact List.mem_cons_of_mem _ hsv
          simp at hir
          rcases hir with h_eq | ⟨h_in, _⟩
          · subst h_eq
            exact R.aliasFree entry hentry_in sv hsv_in
          · exact R.aliasFree ir h_in sv hsv_in
        · -- InjectiveLocals: head + filtered carries the (i, entry.snd) pair which
          -- existed in s.localReg, so old InjLocals applies for the cross terms.
          intro p q hp hq
          simp at hp hq
          rcases hp with hp_eq | ⟨hp_in, hp_ne⟩ <;>
          rcases hq with hq_eq | ⟨hq_in, hq_ne⟩
          · subst hp_eq; subst hq_eq; left; rfl
          · -- p = (i, entry.snd), q ∈ filter (q.fst ≠ i, q ∈ s.localReg).
            right
            subst hp_eq
            have h_old := R.injLocals q (i, entry.snd) hq_in hentry_pair
            rcases h_old with h_keq | h_rne
            · exact absurd h_keq hq_ne
            · exact h_rne.symm
          · right
            subst hq_eq
            have h_old := R.injLocals p (i, entry.snd) hp_in hentry_pair
            rcases h_old with h_keq | h_rne
            · exact absurd h_keq hp_ne
            · exact h_rne
          · exact R.injLocals p q hp_in hq_in

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
/-- `local.tee i` preservation. -/
theorem preservation_localTee (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (R : Refines ws s kst) (h_kst_ok : kst.broke = false)
    (i : Nat)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws (.localTee i) = some ws')
    (hl : lowerInstr s (.localTee i) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' := by
  -- WASM side: pop v_w, setLocal i v_w (on the popped state), push v_w back.
  simp only [evalInstr, WasmState.pop, WasmState.push,
             Option.bind_eq_bind, Option.bind, pure] at hw
  rcases hws_stack : ws.stack with _ | ⟨v_w, rest⟩
  · simp [hws_stack] at hw
  simp only [hws_stack, WasmState.setLocal] at hw
  by_cases hbound : i < List.length ws.locals
  case neg => simp [if_neg hbound] at hw
  simp only [if_pos hbound] at hw
  have hws'_eq : ws' = { locals := ws.locals.set i v_w, stack := v_w :: rest,
                          mem := ws.mem, halted := ws.halted } :=
    ((Option.some.injEq _ _).mp hw).symm
  subst hws'_eq
  -- Lean side.
  simp only [lowerInstr, LowerState.pop, LowerState.lookupLocal,
             LowerState.lookupLocalTy, LowerState.alloc, LowerState.setLocalReg,
             LowerState.push, Option.bind_eq_bind, Option.bind, pure] at hl
  rcases hls_stack : s.stack with _ | ⟨svsrc, lrest⟩
  · simp [hls_stack] at hl
  cases svsrc with
  | bufferPtr _          => simp [hls_stack] at hl
  | scaledIdx _ _        => simp [hls_stack] at hl
  | i32ConstSym _        => simp [hls_stack] at hl
  | bufferAccess _ _ _   => simp [hls_stack] at hl
  | reg src tysrc =>
    simp only [hls_stack] at hl
    have hv_enc_raw : v_w.encodes kst.rf (.reg src tysrc) := by
      have h := R.stk.right 0 v_w (by rw [hws_stack]; simp)
      obtain ⟨sv0, hsv0_get, henc⟩ := h
      have hs0 : s.stack.get? 0 = some (.reg src tysrc) := by rw [hls_stack]; simp
      rw [hs0] at hsv0_get
      have h_eq : SymVal.reg src tysrc = sv0 := (Option.some.injEq _ _).mp hsv0_get
      rwa [← h_eq] at henc
    obtain ⟨n_w, hv_w_eq, htysrc, hsrc_lookup⟩ :=
      WasmValue.encodes_reg_shape hv_enc_raw
    subst hv_w_eq htysrc
    rcases hreg_find : s.localReg.find? (fun p => p.fst = i) with _ | entry
    -- Case B: fresh dst = s.nextReg, post_fresh = s.nextReg + 1.
    · simp [hreg_find] at hl
      obtain ⟨hs_eq, hops_eq⟩ := hl
      refine ⟨{ kst with rf :=
                  regWrite (regWrite kst.rf s.nextReg
                              (Quanta.KOps.Value.vU32 n_w))
                            (s.nextReg + 1)
                            (Quanta.KOps.Value.vU32 n_w) }, ?_, ?_⟩
      · subst hops_eq
        simp [evalOps, Quanta.KOps.evalOp, hsrc_lookup, regLookup_regWrite_self,
              h_kst_ok]
      · subst hs_eq
        refine ⟨?_, ?_, ?_, ?_, ?_⟩
        · -- StackRefines.
          refine ⟨?_, ?_⟩
          · have hl_orig := R.stk.left
            rw [hws_stack, hls_stack] at hl_orig
            simpa using hl_orig
          · intro j v hv
            cases j with
            | zero =>
              simp at hv
              refine ⟨SymVal.reg (s.nextReg + 1) .u32, by simp, ?_⟩
              subst hv
              simp [WasmValue.encodes, regLookup_regWrite_self]
            | succ k =>
              have hrest_get : ws.stack.get? (k + 1) = some v := by
                rw [hws_stack]; simpa using hv
              obtain ⟨svk, hsvk_get, henc⟩ := R.stk.right (k + 1) v hrest_get
              have hlrest_get : lrest.get? k = some svk := by
                have h2 : s.stack.get? (k + 1) = some svk := hsvk_get
                rw [hls_stack] at h2; simpa using h2
              refine ⟨svk, hlrest_get, ?_⟩
              have hsvk_in : svk ∈ s.stack := List.mem_of_get? hsvk_get
              have h_lt : ∀ r ∈ svk.regs, r < s.nextReg :=
                fun r hr => R.fresh.left svk hsvk_in r hr
              exact WasmValue.encodes_preserved_of_fresh
                (fun r hr => Nat.lt_succ_of_lt (h_lt r hr))
                (WasmValue.encodes_preserved_of_fresh h_lt henc)
        · -- LocalsRefines.
          intro k r hfind v hv
          by_cases hki : k = i
          · subst hki
            change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                     ((k, s.nextReg) :: List.filter (fun p => !decide (p.fst = k)) s.localReg)
                   = some (k, r) at hfind
            change (ws.locals.set k (WasmValue.wI32 n_w)).get? k = some v at hv
            rw [List.find?_cons] at hfind
            simp only [show decide ((k, s.nextReg).fst = k) = true from by simp] at hfind
            injection hfind with h_pair
            have hr_eq : s.nextReg = r := (Prod.ext_iff.mp h_pair).2
            subst hr_eq
            have hv_eq : v = WasmValue.wI32 n_w := by
              have hget : (ws.locals.set k (.wI32 n_w)).get? k =
                          some (WasmValue.wI32 n_w) := by
                rw [List.get?_eq_getElem?]
                exact List.getElem?_set_self (by simpa using hbound)
              rw [hget] at hv
              exact ((Option.some.injEq _ _).mp hv).symm
            subst hv_eq
            -- Encoding via reg s.nextReg in kst'.rf = regWrite² ...
            -- s.nextReg ≠ s.nextReg + 1, so the second regWrite preserves the lookup.
            simp [WasmValue.encodes]
            rw [regLookup_regWrite_of_ne _ _ _ _
                  (Nat.ne_of_lt (Nat.lt_succ_self _))]
            simp [regLookup_regWrite_self]
          · change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                     ((i, s.nextReg) :: List.filter (fun p => !decide (p.fst = i)) s.localReg)
                   = some (k, r) at hfind
            rw [find?_setLocalReg_ne _ i k _ hki] at hfind
            have hv_old : ws.locals.get? k = some v := by
              rw [List.get?_eq_getElem?] at hv ⊢
              rw [List.getElem?_set_ne (Ne.symm hki)] at hv
              exact hv
            have henc := R.locs k r hfind v hv_old
            have hkr_in : (k, r) ∈ s.localReg := List.mem_of_find?_eq_some hfind
            have hr_lt : r < s.nextReg := R.fresh.right (k, r) hkr_in
            have h_lt : ∀ r' ∈ (SymVal.reg r .u32).regs, r' < s.nextReg := by
              intro r' hr'_in
              simp [SymVal.regs] at hr'_in
              subst hr'_in; exact hr_lt
            exact WasmValue.encodes_preserved_of_fresh
              (fun r' hr' => Nat.lt_succ_of_lt (h_lt r' hr'))
              (WasmValue.encodes_preserved_of_fresh h_lt henc)
        · -- Fresh: nextReg = s.nextReg + 2.
          refine ⟨?_, ?_⟩
          · intro sv hsv r hr
            simp at hsv
            rcases hsv with h_eq | h_in
            · subst h_eq
              simp [SymVal.regs] at hr
              subst hr
              exact Nat.lt_succ_self _
            · have hsv_in : sv ∈ s.stack := by
                rw [hls_stack]; exact List.mem_cons_of_mem _ h_in
              have : r < s.nextReg := R.fresh.left sv hsv_in r hr
              exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt this)
          · intro ir hir
            simp at hir
            rcases hir with h_eq | ⟨h_in, _⟩
            · subst h_eq
              exact Nat.lt_succ_of_lt (Nat.lt_succ_self _)
            · exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt (R.fresh.right ir h_in))
        · -- AliasFree.
          intro ir hir sv hsv
          simp at hir hsv
          rcases hir with hir_eq | ⟨hir_in, _⟩ <;>
          rcases hsv with hsv_eq | hsv_in
          · subst hir_eq; subst hsv_eq
            simp [SymVal.regs]
          · subst hir_eq
            have hsv_in_s : sv ∈ s.stack := by
              rw [hls_stack]; exact List.mem_cons_of_mem _ hsv_in
            intro hcontra
            have : s.nextReg < s.nextReg :=
              R.fresh.left sv hsv_in_s s.nextReg hcontra
            exact Nat.lt_irrefl _ this
          · subst hsv_eq
            have ir_lt : ir.snd < s.nextReg := R.fresh.right ir hir_in
            simp [SymVal.regs]
            exact Nat.ne_of_lt (Nat.lt_succ_of_lt ir_lt)
          · have hsv_in_s : sv ∈ s.stack := by
              rw [hls_stack]; exact List.mem_cons_of_mem _ hsv_in
            exact R.aliasFree ir hir_in sv hsv_in_s
        · -- InjectiveLocals.
          intro p q hp hq
          simp at hp hq
          rcases hp with hp_eq | ⟨hp_in, hp_ne⟩ <;>
          rcases hq with hq_eq | ⟨hq_in, hq_ne⟩
          · subst hp_eq; subst hq_eq; left; rfl
          · right
            subst hp_eq
            have : q.snd < s.nextReg := R.fresh.right q hq_in
            exact (Nat.ne_of_lt this).symm
          · right
            subst hq_eq
            have : p.snd < s.nextReg := R.fresh.right p hp_in
            exact Nat.ne_of_lt this
          · exact R.injLocals p q hp_in hq_in
    -- Case A: existing dst = entry.snd, post_fresh = s.nextReg.
    · simp [hreg_find] at hl
      obtain ⟨hs_eq, hops_eq⟩ := hl
      have hentry_fst : entry.fst = i := by
        have := List.find?_some hreg_find
        simpa using this
      refine ⟨{ kst with rf :=
                  regWrite (regWrite kst.rf entry.snd
                              (Quanta.KOps.Value.vU32 n_w))
                            s.nextReg
                            (Quanta.KOps.Value.vU32 n_w) }, ?_, ?_⟩
      · subst hops_eq
        simp [evalOps, Quanta.KOps.evalOp, hsrc_lookup, regLookup_regWrite_self,
              h_kst_ok]
      · subst hs_eq
        have hentry_in : entry ∈ s.localReg := List.mem_of_find?_eq_some hreg_find
        have hentry_pair : (i, entry.snd) ∈ s.localReg := by
          have : entry = (i, entry.snd) := by
            rcases entry with ⟨ek, er⟩
            simp at hentry_fst; simp [hentry_fst]
          rw [← this]; exact hentry_in
        have hdst_lt : entry.snd < s.nextReg := R.fresh.right entry hentry_in
        refine ⟨?_, ?_, ?_, ?_, ?_⟩
        · -- StackRefines.
          refine ⟨?_, ?_⟩
          · have hl_orig := R.stk.left
            rw [hws_stack, hls_stack] at hl_orig
            simpa using hl_orig
          · intro j v hv
            cases j with
            | zero =>
              simp at hv
              refine ⟨SymVal.reg s.nextReg .u32, by simp, ?_⟩
              subst hv
              simp [WasmValue.encodes, regLookup_regWrite_self]
            | succ k =>
              have hrest_get : ws.stack.get? (k + 1) = some v := by
                rw [hws_stack]; simpa using hv
              obtain ⟨svk, hsvk_get, henc⟩ := R.stk.right (k + 1) v hrest_get
              have hlrest_get : lrest.get? k = some svk := by
                have h2 : s.stack.get? (k + 1) = some svk := hsvk_get
                rw [hls_stack] at h2; simpa using h2
              refine ⟨svk, hlrest_get, ?_⟩
              have hsvk_in : svk ∈ s.stack := List.mem_of_get? hsvk_get
              -- First write at entry.snd: AliasFree gives entry.snd ∉ svk.regs.
              -- Second write at s.nextReg: freshness gives svk's regs < s.nextReg.
              have h_disj : entry.snd ∉ svk.regs :=
                R.aliasFree entry hentry_in svk hsvk_in
              have h_lt : ∀ r ∈ svk.regs, r < s.nextReg :=
                fun r hr => R.fresh.left svk hsvk_in r hr
              exact WasmValue.encodes_preserved_of_fresh h_lt
                (WasmValue.encodes_preserved_of_disjoint h_disj henc)
        · -- LocalsRefines.
          intro k r hfind v hv
          by_cases hki : k = i
          · subst hki
            change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                     ((k, entry.snd) :: List.filter (fun p => !decide (p.fst = k)) s.localReg)
                   = some (k, r) at hfind
            change (ws.locals.set k (WasmValue.wI32 n_w)).get? k = some v at hv
            rw [List.find?_cons] at hfind
            simp only [show decide ((k, entry.snd).fst = k) = true from by simp] at hfind
            injection hfind with h_pair
            have hr_eq : entry.snd = r := (Prod.ext_iff.mp h_pair).2
            subst hr_eq
            have hv_eq : v = WasmValue.wI32 n_w := by
              have hget : (ws.locals.set k (.wI32 n_w)).get? k =
                          some (WasmValue.wI32 n_w) := by
                rw [List.get?_eq_getElem?]
                exact List.getElem?_set_self (by simpa using hbound)
              rw [hget] at hv
              exact ((Option.some.injEq _ _).mp hv).symm
            subst hv_eq
            -- Encoding via reg entry.snd; entry.snd ≠ s.nextReg (entry.snd < s.nextReg).
            simp [WasmValue.encodes]
            rw [regLookup_regWrite_of_ne _ _ _ _ (Nat.ne_of_lt hdst_lt)]
            simp [regLookup_regWrite_self]
          · change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                     ((i, entry.snd) :: List.filter (fun p => !decide (p.fst = i)) s.localReg)
                   = some (k, r) at hfind
            rw [find?_setLocalReg_ne _ i k _ hki] at hfind
            have hv_old : ws.locals.get? k = some v := by
              rw [List.get?_eq_getElem?] at hv ⊢
              rw [List.getElem?_set_ne (Ne.symm hki)] at hv
              exact hv
            have henc := R.locs k r hfind v hv_old
            have hkr_in : (k, r) ∈ s.localReg := List.mem_of_find?_eq_some hfind
            have hr_ne : r ≠ entry.snd := by
              have := R.injLocals (k, r) (i, entry.snd) hkr_in hentry_pair
              rcases this with h_keq | h_rne
              · exact absurd h_keq hki
              · exact h_rne
            have hr_lt : r < s.nextReg := R.fresh.right (k, r) hkr_in
            have h_disj : entry.snd ∉ (SymVal.reg r .u32).regs := by
              simp [SymVal.regs]; exact hr_ne.symm
            have h_lt : ∀ r' ∈ (SymVal.reg r .u32).regs, r' < s.nextReg := by
              intro r' hr'_in
              simp [SymVal.regs] at hr'_in
              subst hr'_in; exact hr_lt
            exact WasmValue.encodes_preserved_of_fresh h_lt
              (WasmValue.encodes_preserved_of_disjoint h_disj henc)
        · -- Fresh: nextReg = s.nextReg + 1.
          refine ⟨?_, ?_⟩
          · intro sv hsv r hr
            simp at hsv
            rcases hsv with h_eq | h_in
            · subst h_eq
              simp [SymVal.regs] at hr
              subst hr
              exact Nat.lt_succ_self _
            · have hsv_in : sv ∈ s.stack := by
                rw [hls_stack]; exact List.mem_cons_of_mem _ h_in
              exact Nat.lt_succ_of_lt (R.fresh.left sv hsv_in r hr)
          · intro ir hir
            simp at hir
            rcases hir with h_eq | ⟨h_in, _⟩
            · subst h_eq; exact Nat.lt_succ_of_lt hdst_lt
            · exact Nat.lt_succ_of_lt (R.fresh.right ir h_in)
        · -- AliasFree.
          intro ir hir sv hsv
          simp at hir hsv
          rcases hir with hir_eq | ⟨hir_in, _⟩ <;>
          rcases hsv with hsv_eq | hsv_in
          · subst hir_eq; subst hsv_eq
            simp [SymVal.regs]
            exact Nat.ne_of_lt hdst_lt
          · subst hir_eq
            have hsv_in_s : sv ∈ s.stack := by
              rw [hls_stack]; exact List.mem_cons_of_mem _ hsv_in
            exact R.aliasFree entry hentry_in sv hsv_in_s
          · subst hsv_eq
            simp [SymVal.regs]
            exact Nat.ne_of_lt (R.fresh.right ir hir_in)
          · have hsv_in_s : sv ∈ s.stack := by
              rw [hls_stack]; exact List.mem_cons_of_mem _ hsv_in
            exact R.aliasFree ir hir_in sv hsv_in_s
        · -- InjectiveLocals.
          intro p q hp hq
          simp at hp hq
          rcases hp with hp_eq | ⟨hp_in, hp_ne⟩ <;>
          rcases hq with hq_eq | ⟨hq_in, hq_ne⟩
          · subst hp_eq; subst hq_eq; left; rfl
          · right
            subst hp_eq
            have h_old := R.injLocals q (i, entry.snd) hq_in hentry_pair
            rcases h_old with h_keq | h_rne
            · exact absurd h_keq hq_ne
            · exact h_rne.symm
          · right
            subst hq_eq
            have h_old := R.injLocals p (i, entry.snd) hp_in hentry_pair
            rcases h_old with h_keq | h_rne
            · exact absurd h_keq hp_ne
            · exact h_rne
          · exact R.injLocals p q hp_in hq_in

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
