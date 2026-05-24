/-
# Bridge-augmented per-op cons preservation (step 059 L8.5)

The per-op cons theorems in `Quanta.Wasm.PreservationList` (33 closed
theorems as of `1c5cdec`) prove:

  ∃ kst' F, evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout

L8.5 (the bridging invariant `body_branchTarget_implies_IR_broke`)
needs two additional output clauses on every per-op cons theorem:

  (∀ d, ws'.branchTarget = some d → kst'.broke = true)  -- the bridge
  (ws'.branchTarget = none → kst'.broke = false)         -- the inverse

This module follows the two-layer API (`l8_5_scoping.md` §5 R1
mitigation, §8 step 3+4): every existing theorem stays untouched; this
module adds a `_bridge` variant per theorem that

1. takes a stronger IH-on-rest carrying the same two clauses, and
2. produces the two clauses in its conclusion.

For non-control ops (head doesn't touch `branchTarget` / `broke` /
`halted`), the bridge clauses on `ws'` come straight from the bridge
IH applied to the recursion on `rest` — no need to invoke the
non-bridge theorem at all. The cons-default reductions on the WASM
and lowering sides are duplicated here to keep this module
independent of `PreservationList`'s private helpers.

Status (this commit): `cons_nop_bridge` ships first — establishes the
direct-from-IH pattern for non-control ops. Sessions 2+ fill the
remaining 32 per-op theorems.
-/

import Quanta.Wasm.PreservationList

namespace Quanta.Wasm

open Quanta.KOps (KernelOp evalOps regLookup)
open Quanta.Semantics.Cpu

-- ════════════════════════════════════════════════════════════════════
-- Bundled bridge output predicate
-- ════════════════════════════════════════════════════════════════════

/-- The two correspondence clauses linking `ws'.branchTarget` to
    `kst'.broke`. Used both as the IH-on-rest's strengthened
    conclusion and as the new theorem's strengthened conclusion. -/
@[reducible] def BridgeClauses
    (ws' : WasmState) (kst' : Quanta.KOps.State) : Prop :=
  (∀ d, ws'.branchTarget = some d → kst'.broke = true) ∧
  (ws'.branchTarget = none → kst'.broke = false)

-- ════════════════════════════════════════════════════════════════════
-- `nop :: rest` — bridge variant
-- ════════════════════════════════════════════════════════════════════

/-- Bridge-augmented `cons_nop`. `nop` is the simplest case: both
    sides reduce to the recursion on `rest` with the input state
    unchanged, so the bridge IH applied to `rest` yields the full
    conclusion (existence, Refines, and both bridge clauses). -/
theorem preservation_evalInstrs_cons_nop_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest_bridge : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        (_h_kst_no_broke_mid : kst_mid.broke = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout ∧
        BridgeClauses ws'_mid kst'_mid)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.nop :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.nop :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  -- Reduce lowering side to the recursion on `rest` (replicates the
  -- non-bridge proof's hl' step, including the `bind pure` collapse
  -- that the private helper `cons_default_lowerInstrs_collapse_empty_head`
  -- handles in `PreservationList.lean`).
  have hl' : lowerInstrs fuel frames s rest = some (s', ops) := by
    rw [lowerInstrs_cons_default fuel frames s .nop rest rfl] at hl
    simp only [lowerInstr, Option.bind_eq_bind, Option.some_bind,
               List.nil_append] at hl
    cases h_eq : lowerInstrs fuel frames s rest with
    | none => rw [h_eq] at hl; simp only [Option.none_bind] at hl; exact hl
    | some pair =>
        rw [h_eq] at hl
        rcases pair with ⟨s_out, ops_out⟩
        simp only [Option.some_bind, pure] at hl
        exact hl
  -- Reduce eval side to the recursion on `rest`.
  have hw' : evalInstrs fuel ws rest = some ws' := by
    rw [evalInstrs_cons_default fuel ws .nop rest h_no_branch h_no_halt rfl] at hw
    simp only [evalInstr] at hw
    exact hw
  -- The bridge IH directly produces the full conclusion.
  exact preservation_rest_bridge R h_no_branch h_no_halt h_kst_no_broke hw' hl'

-- ════════════════════════════════════════════════════════════════════
-- `i32Const n :: rest` — bridge variant
-- ════════════════════════════════════════════════════════════════════

/-- Bridge-augmented `cons_i32Const`. Pushes `(.wI32 n)` on the WASM
    stack and `(.i32ConstSym n)` on the lowering stack; emits no IR.
    `branchTarget` / `halted` / `broke` all unchanged, so the bridge
    IH applied at the mid-state (with new R_mid built explicitly via
    the same plumbing as the non-bridge proof) discharges the full
    conclusion. -/
theorem preservation_evalInstrs_cons_i32Const_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (n : Int) (rest : List WasmInstr)
    (preservation_rest_bridge : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        (_h_kst_no_broke_mid : kst_mid.broke = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout ∧
        BridgeClauses ws'_mid kst'_mid)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Const n :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Const n :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  let s_mid : LowerState := s.pushSym (.i32ConstSym n)
  let ws_mid : WasmState := ws.push (.wI32 (UInt32.ofNat n.toNat))
  have hl' : lowerInstrs fuel frames s_mid rest = some (s', ops) := by
    rw [lowerInstrs_cons_default fuel frames s (.i32Const n) rest rfl] at hl
    simp only [lowerInstr, Option.bind_eq_bind, Option.some_bind,
               List.nil_append] at hl
    -- After simp, `hl` references the unfolded record form of
    -- `s.pushSym (.i32ConstSym n)`. Cases on that exact term to
    -- match the bind chain in `hl`.
    show lowerInstrs fuel frames (s.pushSym (.i32ConstSym n)) rest = some (s', ops)
    have h_form : (s.pushSym (.i32ConstSym n) : LowerState) =
        { nextReg := s.nextReg, stack := SymVal.i32ConstSym n :: s.stack,
          localReg := s.localReg, localTy := s.localTy,
          bufferSlots := s.bufferSlots } := rfl
    rw [h_form]
    cases h_eq : lowerInstrs fuel frames
        { nextReg := s.nextReg, stack := SymVal.i32ConstSym n :: s.stack,
          localReg := s.localReg, localTy := s.localTy,
          bufferSlots := s.bufferSlots } rest with
    | none => rw [h_eq] at hl; simp only [Option.none_bind] at hl; exact hl
    | some pair =>
        rw [h_eq] at hl
        rcases pair with ⟨s_out, ops_out⟩
        simp only [Option.some_bind, pure] at hl
        -- hl : (s_out, ops_out) = (s', ops). Lift to the option form.
        rw [hl]
  have hw' : evalInstrs fuel ws_mid rest = some ws' := by
    rw [evalInstrs_cons_default fuel ws (.i32Const n) rest h_no_branch h_no_halt rfl] at hw
    simp only [evalInstr] at hw
    show evalInstrs fuel (ws.push (.wI32 (UInt32.ofNat n.toNat))) rest = some ws'
    exact hw
  -- Build `Refines ws_mid s_mid kst layout` (same plumbing as the
  -- non-bridge proof in `PreservationList.lean`).
  have R_mid : Refines ws_mid s_mid kst layout := by
    refine ⟨?_, ?_, ?_, ?_, ?_, R.heapRefines⟩
    · refine ⟨by simp [ws_mid, s_mid, WasmState.push, LowerState.pushSym, R.stk.left], ?_⟩
      intro i v hv
      cases i with
      | zero =>
        simp [ws_mid, WasmState.push] at hv
        refine ⟨SymVal.i32ConstSym n, by simp [s_mid, LowerState.pushSym], ?_⟩
        subst hv
        simp [WasmValue.encodes]
      | succ k =>
        have hwsk : ws.stack.get? k = some v := by
          simpa [ws_mid, WasmState.push] using hv
        obtain ⟨svk, hsvk_get, henc⟩ := R.stk.right k v hwsk
        refine ⟨svk, ?_, henc⟩
        simpa [s_mid, LowerState.pushSym] using hsvk_get
    · simpa [s_mid, LowerState.pushSym] using R.locs
    · refine ⟨?_, ?_⟩
      · intro sv hsv r' hr'
        simp [s_mid, LowerState.pushSym] at hsv
        rcases hsv with h_eq | h_in
        · subst h_eq; simp [SymVal.regs] at hr'
        · exact R.fresh.left sv h_in r' hr'
      · simpa [s_mid, LowerState.pushSym] using R.fresh.right
    · intro ir hir sv hsv
      simp [s_mid, LowerState.pushSym] at hsv ⊢
      rcases hsv with h_eq | h_in
      · subst h_eq; simp [SymVal.regs]
      · exact R.aliasFree ir (by simpa [s_mid, LowerState.pushSym] using hir) sv h_in
    · simpa [s_mid, LowerState.pushSym] using R.injLocals
  have h_no_branch_mid : ws_mid.branchTarget = none := by
    simp [ws_mid, WasmState.push, h_no_branch]
  have h_no_halt_mid : ws_mid.halted = false := by
    simp [ws_mid, WasmState.push, h_no_halt]
  exact preservation_rest_bridge R_mid h_no_branch_mid h_no_halt_mid
    h_kst_no_broke hw' hl'

end Quanta.Wasm
