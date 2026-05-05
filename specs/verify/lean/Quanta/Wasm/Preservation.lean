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
  - `preservation_i32{Add,Sub,Mul,And,Or,Xor,Shl,ShrU,DivU,RemU}`

That's **14 closed preservation theorems**. The four archetypes —
empty-emit no-state-change (`nop`), empty-emit halted-flag (`wreturn`),
single-op fresh-write (`i32Const`), no-op stack-push (`localGet`),
plus the **two-pop one-fresh-write binop** archetype (`i32Bin`) — now
all have at least one closed instance. Every remaining op outside
slices 4+ rides on these archetypes.

## What's next

**Slice 3 follow-ups still open** (deferred to slice 4 entry):
* i32-comparison family — needs a generalized `WasmValue.encodes`
  that recognizes `wI32 0/1` ↔ `vBool b`. WASM's i32-encoded booleans
  don't match KOps's native `Value.vBool` shape; production handles
  this via a Bool→U32 cast in `commit` when a bool-typed reg flows
  into arithmetic. The Lean port should mirror that.
* `localSet`/`localTee` — alias-free invariant is now baked into
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

/-- Same shape for `lowerI32Cmp`. -/
theorem lowerI32Cmp_some_shape {cop : Quanta.KOps.CmpOp} {s s' : LowerState}
    {ops : List KernelOp} (h : lowerI32Cmp s cop = some (s', ops)) :
    ∃ ra rb tya tyb lrest,
      s.stack = SymVal.reg rb tyb :: SymVal.reg ra tya :: lrest ∧
      s' = { nextReg := s.nextReg + 1,
             stack := SymVal.reg s.nextReg .u32 :: lrest,
             localReg := s.localReg,
             localTy := s.localTy } ∧
      ops = [.cmp s.nextReg ra rb cop .u32] := by
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
          exact hs'.symm
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
-- Slice 3 follow-up status
--
-- The `Refines.injLocals` invariant is now in place (this commit),
-- bringing the bundle to the 5 clauses needed for `localSet` /
-- `localTee` preservation:
--
--   * stk        — stack encoding refinement
--   * locs       — locals encoding refinement
--   * fresh      — every held register < nextReg
--   * aliasFree  — no stable_reg appears on the symbolic stack
--   * injLocals  — distinct local indices map to distinct stable_regs
--
-- All 14 closed per-instr theorems (nop, return, i32Const, localGet,
-- 10 binops) now produce a Refines bundle with all 5 clauses.
--
-- The `preservation_localSet` and `preservation_localTee` theorems
-- themselves are deferred to slice 4 entry. The blocker is only proof
-- volume — each requires a 2-case split (existing vs fresh dst), each
-- case proves all 5 Refines clauses, with the LocalsRefines clause
-- splitting again on `k = i` vs `k ≠ i` for the per-local encoding
-- obligation. Total ~200-300 lines per theorem of straightforward
-- but voluminous case analysis. The invariants needed (above) are all
-- present; the proof is mechanical from here.
-- ════════════════════════════════════════════════════════════════════

end Quanta.Wasm
