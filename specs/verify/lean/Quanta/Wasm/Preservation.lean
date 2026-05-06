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

/-- A WASM value is encoded by a SymVal stack slot if either:
    * the slot is `.reg r .u32` and the regfile holds the matching
      `vU32` at that register, or
    * the slot is `.i32ConstSym m` and the WASM value is the matching
      `wI32 (UInt32.ofNat m.toNat)` — purely symbolic, no regfile
      dependency. The translator pushes `i32ConstSym` for every
      `i32.const`; the buffer-pattern arms (or a `commit` materializer)
      consume it before any reg-typed consumer fires.

    Other SymVal shapes (`bufferPtr`, `scaledIdx`, `bufferAccess`)
    are addresses, not values — their encoding stays `False`. They
    are consumed inline by the buffer-pattern arms in slice-4 step 7+
    and never satisfy a value refinement. -/
def WasmValue.encodes (v : WasmValue) (rf : Quanta.KOps.RegFile) (sv : SymVal) : Prop :=
  match v, sv with
  | .wI32 n, .reg r .u32      => regLookup rf r = some (Quanta.KOps.Value.vU32 n)
  | .wI32 n, .i32ConstSym m   => n = UInt32.ofNat m.toNat
  | _, _                      => False

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

/-- Bundle. The `layout : BufferLayout` parameter is the shared side-
    channel that relates WASM linear memory to the KOps heap; each
    theorem fixes `layout` across input and output (the layout is
    static in the lowered program). -/
structure Refines (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
                  (layout : BufferLayout) : Prop where
  stk         : StackRefines ws.stack s.stack kst.rf
  locs        : LocalsRefines ws.locals s.localReg kst.rf
  fresh       : Fresh s
  aliasFree   : AliasFree s
  injLocals   : InjectiveLocals s
  heapRefines : HeapRefines ws.mem kst.heap layout

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
  | .wI32 _, .i32ConstSym _, h =>
    -- i32ConstSym encoding is regfile-independent.
    exact h

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
  | .wI32 _, .i32ConstSym _, h =>
    exact h

/-- Encoding is preserved under any regfile transition that agrees on
    the SymVal's regs. The full-strength companion to
    `encodes_preserved_of_fresh` / `encodes_preserved_of_disjoint`,
    used when a sequence of ops (e.g. `commit svb`'s `opsB`) writes
    only to fresh registers — we collapse that into a pointwise
    `regLookup` equality on the regs of any older SymVal. -/
theorem WasmValue.encodes_preserved_of_lookup_eq
    {v : WasmValue} {rf rf' : Quanta.KOps.RegFile} {sv : SymVal}
    (h_lookup : ∀ r ∈ sv.regs, regLookup rf' r = regLookup rf r)
    (h : v.encodes rf sv) :
    v.encodes rf' sv := by
  match v, sv, h with
  | .wI32 n, .reg r .u32, h =>
    have h' : regLookup rf r = some (Quanta.KOps.Value.vU32 n) := h
    have h_eq := h_lookup r (by simp [SymVal.regs])
    show regLookup rf' r = some (Quanta.KOps.Value.vU32 n)
    rw [h_eq]; exact h'
  | .wI32 _, .i32ConstSym _, h => exact h

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

/-- Inversion for `i32ConstSym` encoding. -/
theorem WasmValue.encodes_i32ConstSym_inv
    {v : WasmValue} {rf : Quanta.KOps.RegFile} {n : Int}
    (h : v.encodes rf (.i32ConstSym n)) :
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
    {v_w : WasmValue} (h_enc : v_w.encodes kst.rf sv) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧
            Refines ws s' kst' layout ∧
            v_w.encodes kst'.rf (.reg r .u32) ∧
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
      refine ⟨?_, ?_, ?_, R.aliasFree, R.injLocals, R.heapRefines⟩
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
    refine ⟨?_, ?_, R.fresh, R.aliasFree, R.injLocals, ?_⟩
    · have : ws'.stack = ws.stack := by rw [← hw]
      rw [this]; exact R.stk
    · have : ws'.locals = ws.locals := by rw [← hw]
      rw [this]; exact R.locs
    · -- HeapRefines: ws'.mem = ws.mem (return doesn't touch memory).
      have : ws'.mem = ws.mem := by rw [← hw]
      rw [this]; exact R.heapRefines

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
    refine ⟨?_, ?_, ?_, ?_, ?_, R.heapRefines⟩
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
/-- `local.get i` preservation. Lowering allocates a fresh register
    `s.nextReg`, emits `Copy { dst := s.nextReg, src := stable_reg }`,
    pushes the fresh reg. WASM pushes `locals[i]`. The local's stable
    register encodes its value via `R.locs`, the Copy propagates that
    encoding to the fresh reg, and freshness keeps every prior reg
    readable. -/
theorem preservation_localGet (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout)
    (i : Nat)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws (.localGet i) = some ws')
    (hl : lowerInstr s (.localGet i) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
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
          refine ⟨?_, ?_, ?_, ?_, ?_, R.heapRefines⟩
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
             localTy := s.localTy } ∧
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
        -- Goal: { s4 with nextReg := s4.nextReg + 1, stack := SymVal.reg s4.nextReg .u32 :: s4.stack }
        --     = { nextReg := s4.nextReg + 1, stack := SymVal.reg s4.nextReg .u32 :: lrest,
        --         localReg := s.localReg, localTy := s.localTy }
        rw [h_s4_stack, h_s4_lr, h_s4_lt]

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
             localTy := s.localTy } ∧
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
        rw [h_s4_stack, h_s4_lr, h_s4_lt]

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
    (h_l : ∀ s, lowerInstr s instr = lowerI32Bin s op_k)
    (h_agree : ∀ av bv,
       Quanta.KOps.evalBinOp op_k (vU32 av) (vU32 bv) = some (vU32 (op_w av bv)))
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout) (R : Refines ws s kst layout) (h_kst_ok : kst.broke = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
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
  have h_enc_svb : (WasmValue.wI32 bv).encodes kst.rf svb := by
    have hb := R.stk.right 0 (.wI32 bv) (by rw [hwstack]; simp)
    obtain ⟨sv0, hsv0_get, henc⟩ := hb
    have hs0 : s.stack.get? 0 = some svb := by rw [hlstack]; simp
    rw [hs0] at hsv0_get
    have h_eq : svb = sv0 := (Option.some.injEq _ _).mp hsv0_get
    rw [h_eq]; exact henc
  have h_enc_sva : (WasmValue.wI32 av).encodes kst.rf sva := by
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
      localReg := s.localReg, localTy := s.localTy }
  let ws_pop : WasmState :=
    { ws with stack := rest }
  have R_pop : Refines ws_pop s_pop kst layout := by
    refine ⟨⟨h_rest_lrest_len, ?_⟩, R.locs, ?_, ?_, R.injLocals, R.heapRefines⟩
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
  have h_enc_svb1 : (WasmValue.wI32 bv).encodes kst1.rf svb :=
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
  have h_enc_ra2 : (WasmValue.wI32 av).encodes kst2.rf (.reg ra .u32) :=
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
    refine ⟨?_, ?_, ?_, ?_, ?_, R2.heapRefines⟩
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
  have h_enc_svb : (WasmValue.wI32 bv).encodes kst.rf svb := by
    have hb := R.stk.right 0 (.wI32 bv) (by rw [hwstack]; simp)
    obtain ⟨sv0, hsv0_get, henc⟩ := hb
    have hs0 : s.stack.get? 0 = some svb := by rw [hlstack]; simp
    rw [hs0] at hsv0_get
    have h_eq : svb = sv0 := (Option.some.injEq _ _).mp hsv0_get
    rw [h_eq]; exact henc
  have h_enc_sva : (WasmValue.wI32 av).encodes kst.rf sva := by
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
      localReg := s.localReg, localTy := s.localTy }
  let ws_pop : WasmState :=
    { ws with stack := rest }
  have R_pop : Refines ws_pop s_pop kst layout := by
    refine ⟨⟨h_rest_lrest_len, ?_⟩, R.locs, ?_, ?_, R.injLocals, R.heapRefines⟩
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
  have h_enc_svb1 : (WasmValue.wI32 bv).encodes kst1.rf svb :=
    WasmValue.encodes_preserved_of_lookup_eq
      (fun r hr => h_lookupA r (h_svb_lt r hr)) h_enc_svb
  have h_svb_lt_s3 : ∀ r ∈ svb.regs, r < s3.nextReg :=
    fun r hr => Nat.lt_of_lt_of_le (h_svb_lt r hr) h_s_le_s3
  -- Second commit: svb → rb on R1. Emits opsB, evolves kst1 → kst2.
  obtain ⟨kst2, h_evalB, R2, h_enc_rb2, h_lookupB, h_s3_le_s4, h_rb_lt_s4⟩ :=
    commit_correct R1 h_svb_lt_s3 hcb h_enc_svb1
  have h_kst2_ok : kst2.broke = false := by
    rw [commit_preserves_broke hcb h_evalB]; exact h_kst1_ok
  have h_enc_ra2 : (WasmValue.wI32 av).encodes kst2.rf (.reg ra .u32) :=
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
        v.encodes kst2.rf sv →
        v.encodes (regWrite (regWrite kst2.rf s4.nextReg
                              (Quanta.KOps.Value.vBool (p_w av bv)))
                            (s4.nextReg + 1)
                            (Quanta.KOps.Value.vU32 (if p_w av bv then 1 else 0))) sv :=
      fun sv v h_lt henc =>
        WasmValue.encodes_preserved_of_fresh
          (fun r hr => Nat.lt_succ_of_lt (h_lt r hr))
          (WasmValue.encodes_preserved_of_fresh h_lt henc)
    refine ⟨?_, ?_, ?_, ?_, ?_, R2.heapRefines⟩
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
                          mem := ws.mem, halted := ws.halted } :=
    ((Option.some.injEq _ _).mp hw).symm
  subst hws'_eq
  -- Lean side: popSym first (always succeeds for non-empty stack), then commit
  -- (refuses buffer SymVals; succeeds on `.reg` and `.i32ConstSym`).
  unfold lowerInstr at hl
  rcases hls_stack : s.stack with _ | ⟨sva, lrest⟩
  · simp [hls_stack, LowerState.popSym] at hl
  simp only [hls_stack, LowerState.popSym, Option.bind_eq_bind, Option.some_bind] at hl
  -- Branch on commit success.
  rcases hca : ({s with stack := lrest} : LowerState).commit sva
      with _ | ⟨src, s2, opsCommit⟩
  · simp [hca] at hl
  simp only [hca, Option.some_bind] at hl
  -- v_w must be wI32 (encoding non-False on stack); extract n_w.
  have hv_enc : v_w.encodes kst.rf sva := by
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
  -- sva is on s.stack, so its regs are < s.nextReg.
  have h_sva_in : sva ∈ s.stack := by rw [hls_stack]; simp
  have h_sva_lt : ∀ r ∈ sva.regs, r < s.nextReg :=
    fun r hr => R.fresh.left sva h_sva_in r hr
  -- Build R_pop : Refines ws_pop s_pop kst layout for the popped state.
  have h_rest_lrest_len : rest.length = lrest.length := by
    have hl_orig := R.stk.left
    rw [hws_stack, hls_stack] at hl_orig
    simpa using hl_orig
  let s_pop : LowerState :=
    { nextReg := s.nextReg, stack := lrest,
      localReg := s.localReg, localTy := s.localTy }
  let ws_pop : WasmState := { ws with stack := rest }
  have R_pop : Refines ws_pop s_pop kst layout := by
    refine ⟨⟨h_rest_lrest_len, ?_⟩, R.locs, ?_, ?_, R.injLocals, R.heapRefines⟩
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
  -- Apply commit_correct: get kst1, R1 (Refines on s2), src register, lookup-preservation.
  obtain ⟨kst1, h_evalC, R1, h_enc_src1, h_lookupC, _h_s_le_s2, h_src_lt_s2⟩ :=
    commit_correct R_pop h_sva_lt hca hv_enc
  have h_kst1_ok : kst1.broke = false := by
    rw [commit_preserves_broke hca h_evalC]; exact h_kst_ok
  have h_src_lookup : regLookup kst1.rf src = some (Quanta.KOps.Value.vU32 n_w) :=
    h_enc_src1
  -- s2's localReg / localTy / stack relate to s via the commit-only-bumps lemmas.
  have h_s2_lr : s2.localReg = s.localReg := (commit_preserves_locals hca).1
  have h_s2_lt : s2.localTy  = s.localTy  := (commit_preserves_locals hca).2
  have h_s2_stack : s2.stack = lrest := commit_preserves_stack hca
  -- Continue with the lookupLocal branch on s2 (= s.localReg by h_s2_lr).
  simp only [LowerState.lookupLocal, LowerState.lookupLocalTy, LowerState.alloc,
             LowerState.setLocalReg, LowerState.push, Option.bind_eq_bind, Option.bind,
             pure] at hl
  rw [h_s2_lt, h_s2_lr] at hl
  rcases hreg_find : s.localReg.find? (fun p => p.fst = i) with _ | entry
  -- Case B: fresh dst = s2.nextReg.
  · simp [hreg_find] at hl
    obtain ⟨hs_eq, hops_eq⟩ := hl
    -- Final kst' applies one more regWrite at s2.nextReg.
    refine ⟨{ kst1 with rf := regWrite kst1.rf s2.nextReg
                          (Quanta.KOps.Value.vU32 n_w) }, ?_, ?_⟩
    · subst hops_eq
      rw [evalOps_append h_evalC h_kst1_ok]
      simp [evalOps, Quanta.KOps.evalOp, h_src_lookup]
    · subst hs_eq
      refine ⟨?_, ?_, ?_, ?_, ?_, R1.heapRefines⟩
      · -- StackRefines on (rest, lrest), lifted past the regWrite at s2.nextReg.
        refine ⟨?_, ?_⟩
        · -- Length: rest.length = s2.stack.length (= lrest.length).
          rw [h_s2_stack]; simpa using h_rest_lrest_len
        · intro j v hv
          have hk : ws_pop.stack.get? j = some v := by
            show rest.get? j = some v; simpa using hv
          obtain ⟨svj, hsvj_get, henc⟩ := R1.stk.right j v hk
          refine ⟨svj, by simpa using hsvj_get, ?_⟩
          have hsvj_in : svj ∈ s2.stack := List.mem_of_get? hsvj_get
          apply WasmValue.encodes_preserved_of_fresh _ henc
          intro r hr
          exact R1.fresh.left svj hsvj_in r hr
      · -- LocalsRefines on s'.localReg = (i, s2.nextReg) :: filter (≠ i) s.localReg.
        intro k r hfind v hv
        by_cases hki : k = i
        · subst hki
          change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((k, s2.nextReg) :: List.filter (fun p => !decide (p.fst = k)) s.localReg)
                 = some (k, r) at hfind
          change (ws.locals.set k (WasmValue.wI32 n_w)).get? k = some v at hv
          rw [List.find?_cons] at hfind
          simp only [show decide ((k, s2.nextReg).fst = k) = true from by simp] at hfind
          injection hfind with h_pair
          have hr_eq : s2.nextReg = r := (Prod.ext_iff.mp h_pair).2
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
        · change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((i, s2.nextReg) :: List.filter (fun p => !decide (p.fst = i)) s.localReg)
                 = some (k, r) at hfind
          rw [find?_setLocalReg_ne _ i k _ hki] at hfind
          have hv_old : ws.locals.get? k = some v := by
            rw [List.get?_eq_getElem?] at hv ⊢
            rw [List.getElem?_set_ne (Ne.symm hki)] at hv
            exact hv
          -- Translate via R1.locs (its localReg = s2.localReg = s.localReg via h_s2_lr).
          have hfind_s2 : s2.localReg.find? (fun p => p.fst = k) = some (k, r) := by
            rw [h_s2_lr]; exact hfind
          have henc := R1.locs k r hfind_s2 v hv_old
          have hr_lt : r < s2.nextReg := by
            have hpair : (k, r) ∈ s2.localReg :=
              List.mem_of_find?_eq_some hfind_s2
            exact R1.fresh.right (k, r) hpair
          apply WasmValue.encodes_preserved_of_fresh _ henc
          intro r' hr'_in
          simp [SymVal.regs] at hr'_in
          subst hr'_in
          exact hr_lt
      · -- Fresh: nextReg = s2.nextReg + 1.
        refine ⟨?_, ?_⟩
        · intro sv hsv r hr
          have hsv_in_s2 : sv ∈ s2.stack := hsv
          exact Nat.lt_succ_of_lt (R1.fresh.left sv hsv_in_s2 r hr)
        · intro ir hir
          simp at hir
          rcases hir with h_eq | ⟨h_in, _⟩
          · subst h_eq; exact Nat.lt_succ_self _
          · have hin_s2 : ir ∈ s2.localReg := by rw [h_s2_lr]; exact h_in
            exact Nat.lt_succ_of_lt (R1.fresh.right ir hin_s2)
      · -- AliasFree.
        intro ir hir sv hsv
        have hsv_in_s2 : sv ∈ s2.stack := hsv
        simp at hir
        rcases hir with h_eq | ⟨h_in, _⟩
        · subst h_eq
          intro hcontra
          have : s2.nextReg < s2.nextReg :=
            R1.fresh.left sv hsv_in_s2 s2.nextReg hcontra
          exact Nat.lt_irrefl _ this
        · have hin_s2 : ir ∈ s2.localReg := by rw [h_s2_lr]; exact h_in
          exact R1.aliasFree ir hin_s2 sv hsv_in_s2
      · -- InjectiveLocals: head fresh; filter preserves R1.injLocals.
        intro p q hp hq
        simp at hp hq
        rcases hp with hp_eq | ⟨hp_in, hp_ne⟩ <;>
        rcases hq with hq_eq | ⟨hq_in, hq_ne⟩
        · subst hp_eq; subst hq_eq; left; rfl
        · right
          subst hp_eq
          have hin_s2 : q ∈ s2.localReg := by rw [h_s2_lr]; exact hq_in
          have : q.snd < s2.nextReg := R1.fresh.right q hin_s2
          exact (Nat.ne_of_lt this).symm
        · right
          subst hq_eq
          have hin_s2 : p ∈ s2.localReg := by rw [h_s2_lr]; exact hp_in
          have : p.snd < s2.nextReg := R1.fresh.right p hin_s2
          exact Nat.ne_of_lt this
        · have hpin_s2 : p ∈ s2.localReg := by rw [h_s2_lr]; exact hp_in
          have hqin_s2 : q ∈ s2.localReg := by rw [h_s2_lr]; exact hq_in
          exact R1.injLocals p q hpin_s2 hqin_s2
  -- Case A: existing dst = entry.snd.
  · simp [hreg_find] at hl
    obtain ⟨hs_eq, hops_eq⟩ := hl
    have hentry_fst : entry.fst = i := by
      have := List.find?_some hreg_find
      simpa using this
    refine ⟨{ kst1 with rf := regWrite kst1.rf entry.snd
                          (Quanta.KOps.Value.vU32 n_w) }, ?_, ?_⟩
    · subst hops_eq
      rw [evalOps_append h_evalC h_kst1_ok]
      simp [evalOps, Quanta.KOps.evalOp, h_src_lookup]
    · subst hs_eq
      have hentry_in : entry ∈ s.localReg :=
        List.mem_of_find?_eq_some hreg_find
      have hentry_in_s2 : entry ∈ s2.localReg := by rw [h_s2_lr]; exact hentry_in
      have hentry_pair : (i, entry.snd) ∈ s.localReg := by
        have : entry = (i, entry.snd) := by
          rcases entry with ⟨ek, er⟩
          simp at hentry_fst
          simp [hentry_fst]
        rw [← this]; exact hentry_in
      have hentry_pair_s2 : (i, entry.snd) ∈ s2.localReg := by
        rw [h_s2_lr]; exact hentry_pair
      have hdst_lt : entry.snd < s2.nextReg := R1.fresh.right entry hentry_in_s2
      refine ⟨?_, ?_, ?_, ?_, ?_, R1.heapRefines⟩
      · -- StackRefines: lift past regWrite at entry.snd (a stable_reg, disjoint by AliasFree).
        refine ⟨?_, ?_⟩
        · -- Length: rest.length = s2.stack.length (= lrest.length).
          rw [h_s2_stack]; simpa using h_rest_lrest_len
        · intro j v hv
          have hk : ws_pop.stack.get? j = some v := by
            show rest.get? j = some v; simpa using hv
          obtain ⟨svj, hsvj_get, henc⟩ := R1.stk.right j v hk
          refine ⟨svj, by simpa using hsvj_get, ?_⟩
          have hsvj_in : svj ∈ s2.stack := List.mem_of_get? hsvj_get
          have h_disj : entry.snd ∉ svj.regs :=
            R1.aliasFree entry hentry_in_s2 svj hsvj_in
          exact WasmValue.encodes_preserved_of_disjoint h_disj henc
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
          simp [WasmValue.encodes, regLookup_regWrite_self]
        · change List.find? (fun p : Nat × Reg => decide (p.fst = k))
                   ((i, entry.snd) :: List.filter (fun p => !decide (p.fst = i)) s.localReg)
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
          have hr_ne : r ≠ entry.snd := by
            have := R1.injLocals (k, r) (i, entry.snd) hkr_in_s2 hentry_pair_s2
            rcases this with h_keq | h_rne
            · exact absurd h_keq hki
            · exact h_rne
          apply WasmValue.encodes_preserved_of_disjoint _ henc
          simp [SymVal.regs]
          exact hr_ne.symm
      · -- Fresh: nextReg unchanged at s2.nextReg.
        refine ⟨?_, ?_⟩
        · intro sv hsv r hr
          have hsv_in_s2 : sv ∈ s2.stack := hsv
          exact R1.fresh.left sv hsv_in_s2 r hr
        · intro ir hir
          simp at hir
          rcases hir with h_eq | ⟨h_in, _⟩
          · subst h_eq; exact hdst_lt
          · have hin_s2 : ir ∈ s2.localReg := by rw [h_s2_lr]; exact h_in
            exact R1.fresh.right ir hin_s2
      · -- AliasFree.
        intro ir hir sv hsv
        have hsv_in_s2 : sv ∈ s2.stack := hsv
        simp at hir
        rcases hir with h_eq | ⟨h_in, _⟩
        · subst h_eq
          exact R1.aliasFree entry hentry_in_s2 sv hsv_in_s2
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
          have h_old := R1.injLocals q (i, entry.snd) hin_s2 hentry_pair_s2
          rcases h_old with h_keq | h_rne
          · exact absurd h_keq hq_ne
          · exact h_rne.symm
        · right
          subst hq_eq
          have hin_s2 : p ∈ s2.localReg := by rw [h_s2_lr]; exact hp_in
          have h_old := R1.injLocals p (i, entry.snd) hin_s2 hentry_pair_s2
          rcases h_old with h_keq | h_rne
          · exact absurd h_keq hp_ne
          · exact h_rne
        · have hpin_s2 : p ∈ s2.localReg := by rw [h_s2_lr]; exact hp_in
          have hqin_s2 : q ∈ s2.localReg := by rw [h_s2_lr]; exact hq_in
          exact R1.injLocals p q hpin_s2 hqin_s2

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
    (layout : BufferLayout) (R : Refines ws s kst layout) (h_kst_ok : kst.broke = false)
    (i : Nat)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstr ws (.localTee i) = some ws')
    (hl : lowerInstr s (.localTee i) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
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
        refine ⟨?_, ?_, ?_, ?_, ?_, R.heapRefines⟩
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
        refine ⟨?_, ?_, ?_, ?_, ?_, R.heapRefines⟩
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
