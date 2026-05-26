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
          bufferSlots := s.bufferSlots, currentReg := s.currentReg } := rfl
    rw [h_form]
    cases h_eq : lowerInstrs fuel frames
        { nextReg := s.nextReg, stack := SymVal.i32ConstSym n :: s.stack,
          localReg := s.localReg, localTy := s.localTy,
          bufferSlots := s.bufferSlots, currentReg := s.currentReg } rest with
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
    refine ⟨?_, ?_, ?_, ?_, ?_, R.heapRefines, R.currentReg, R.freshCurrent⟩
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

-- ════════════════════════════════════════════════════════════════════
-- Bridge variant of `cons_compose_shallow`
-- ════════════════════════════════════════════════════════════════════

/-- Bridge-augmented `preservation_evalInstrs_cons_compose_shallow`.
    The head ops are non-control-flow (loopFree, no `.branch`), so
    `evalOps F kst (ops_head ++ ops_rest) = evalOps F kst_mid
    ops_rest` after consuming the head. Any bridge clauses on the
    rest's output state lift directly to the composed result. -/
theorem preservation_evalInstrs_cons_compose_shallow_bridge
    {F : Nat} {kst kst_mid : Quanta.KOps.State}
    {ops_head ops_rest : List KernelOp}
    {ws' : WasmState} {s' : LowerState}
    {layout : BufferLayout}
    (h_lf : loopFree ops_head = true)
    (h_head : evalOps 0 kst ops_head = some kst_mid)
    (h_no_broke : kst_mid.broke = false)
    (h_rest : ∃ kst', evalOps F kst_mid ops_rest = some kst'
                       ∧ Refines ws' s' kst' layout
                       ∧ BridgeClauses ws' kst') :
    ∃ kst', evalOps F kst (ops_head ++ ops_rest) = some kst'
              ∧ Refines ws' s' kst' layout
              ∧ BridgeClauses ws' kst' := by
  obtain ⟨kst', h_eval', R', h_bridge⟩ := h_rest
  refine ⟨kst', ?_, R', h_bridge⟩
  exact evalOps_append_loopFreeDeep_head (loopFree_implies_deep _ h_lf)
    h_head h_no_broke h_eval'

-- ════════════════════════════════════════════════════════════════════
-- `localGet i :: rest` (non-buffer path) — bridge variant
-- ════════════════════════════════════════════════════════════════════

/-- Bridge-augmented `cons_localGet` (non-buffer path). Head emits
    `[.copy fresh stable]`, which (a) is shallow-loopFree, (b)
    preserves the `broke` flag, (c) doesn't touch `branchTarget` on
    the WASM side. The bridge clauses on the final state come from
    the IH-bridge applied to `rest`; the head's effect is purely
    register data movement. -/
theorem preservation_evalInstrs_cons_localGet_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (i : Nat) (h_no_buf : s.lookupBufferSlot i = none)
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
    (hw : evalInstrs fuel ws (.localGet i :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.localGet i :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  -- Stage 3 cascade: body deferred (bridge variants depend on
  -- cons_local* which are currently sorry-deferred).
  sorry

-- ════════════════════════════════════════════════════════════════════
-- `localSet i :: rest` — bridge variant
-- ════════════════════════════════════════════════════════════════════

/-- Bridge-augmented `cons_localSet`. Head emits a loopFreeNoBreak
    `popSym + commit` op-list; mid-state preconditions discharge
    because `localSet` only touches locals/stack (not
    branchTarget / halted / broke). -/
theorem preservation_evalInstrs_cons_localSet_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (i : Nat)
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
    (hw : evalInstrs fuel ws (.localSet i :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.localSet i :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  -- Stage 3 cascade: body deferred (bridge variants depend on
  -- cons_local* which are currently sorry-deferred).
  sorry

-- ════════════════════════════════════════════════════════════════════
-- `localTee i :: rest` — bridge variant
-- ════════════════════════════════════════════════════════════════════

/-- Bridge-augmented `cons_localTee`. Same shape as `localSet` —
    head emits loopFreeNoBreak commit + two `.copy` ops, mid-state
    discharge identical. -/
theorem preservation_evalInstrs_cons_localTee_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (i : Nat)
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
    (hw : evalInstrs fuel ws (.localTee i :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.localTee i :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  -- Stage 3 cascade: body deferred (bridge variants depend on
  -- cons_local* which are currently sorry-deferred).
  sorry

-- ════════════════════════════════════════════════════════════════════
-- `drop :: rest` — bridge variant
-- ════════════════════════════════════════════════════════════════════

/-- Bridge-augmented `cons_drop`. Head emits no IR; both sides pop
    one stack value. branchTarget / halted / broke all unchanged. -/
theorem preservation_evalInstrs_cons_drop_bridge
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
    (hw : evalInstrs fuel ws (.drop :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.drop :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  rcases hws_stack : ws.stack with _ | ⟨v_w, rest_ws⟩
  · rw [evalInstrs_cons_default fuel ws .drop rest h_no_branch h_no_halt rfl] at hw
    have h_ev : evalInstr ws .drop = none := by
      show (do let (_, s1) ← ws.pop; pure s1) = none
      simp [WasmState.pop, hws_stack]
    rw [h_ev] at hw
    simp at hw
  rcases hls_stack : s.stack with _ | ⟨sva, lrest⟩
  · rw [lowerInstrs_cons_default fuel frames s .drop rest rfl] at hl
    have h_lw : lowerInstr s .drop = none := by
      show (do let (_, s1) ← s.popSym; pure (s1, ([] : List KernelOp))) = none
      simp [LowerState.popSym, hls_stack]
    rw [h_lw] at hl
    simp at hl
  let ws_mid : WasmState := { ws with stack := rest_ws }
  let s_mid : LowerState :=
    { nextReg := s.nextReg, stack := lrest,
      localReg := s.localReg, localTy := s.localTy,
      bufferSlots := s.bufferSlots, currentReg := s.currentReg }
  have hl' : lowerInstrs fuel frames s_mid rest = some (s', ops) := by
    rw [lowerInstrs_cons_default fuel frames s .drop rest rfl] at hl
    have h_lw : lowerInstr s .drop = some (s_mid, []) := by
      show (do let (_, s1) ← s.popSym; pure (s1, ([] : List KernelOp))) = some (s_mid, [])
      unfold LowerState.popSym
      rw [hls_stack]
      rfl
    rw [h_lw] at hl
    simp only [Option.bind_eq_bind, Option.some_bind, List.nil_append] at hl
    show lowerInstrs fuel frames s_mid rest = some (s', ops)
    cases h_eq : lowerInstrs fuel frames s_mid rest with
    | none => rw [h_eq] at hl; simp only [Option.none_bind] at hl; exact hl
    | some pair =>
        rw [h_eq] at hl
        rcases pair with ⟨s_out, ops_out⟩
        simp only [Option.some_bind, pure] at hl
        rw [hl]
  have hw' : evalInstrs fuel ws_mid rest = some ws' := by
    rw [evalInstrs_cons_default fuel ws .drop rest h_no_branch h_no_halt rfl] at hw
    have h_ev : evalInstr ws .drop = some ws_mid := by
      show (do let (_, s1) ← ws.pop; pure s1) = some ws_mid
      unfold WasmState.pop
      rw [hws_stack]
      rfl
    rw [h_ev] at hw
    simp only at hw
    exact hw
  have h_rest_lrest_len : rest_ws.length = lrest.length := by
    have hl_orig := R.stk.left
    rw [hws_stack, hls_stack] at hl_orig
    simpa using hl_orig
  have R_mid : Refines ws_mid s_mid kst layout := by
    refine ⟨⟨h_rest_lrest_len, ?_⟩, R.locs, ?_, ?_, R.injLocals, R.heapRefines, R.currentReg, R.freshCurrent⟩
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
  have h_mid_no_branch : ws_mid.branchTarget = none := by
    simp [ws_mid, h_no_branch]
  have h_mid_no_halt : ws_mid.halted = false := by
    simp [ws_mid, h_no_halt]
  exact preservation_rest_bridge R_mid h_mid_no_branch h_mid_no_halt h_kst_no_broke hw' hl'

-- ════════════════════════════════════════════════════════════════════
-- Generic i32-binop cons bridge variant
-- ════════════════════════════════════════════════════════════════════

/-- Bridge-augmented `cons_i32Bin_generic`. Replays the non-bridge
    generic with the bridge IH; head ops are `opsA ++ opsB ++
    [.binOp]` (all loopFreeNoBreak), so the cons-compose-bridge
    threads the bridge clauses through unchanged. -/
theorem preservation_evalInstrs_cons_i32Bin_generic_bridge
    (instr : WasmInstr) (op_w : UInt32 → UInt32 → UInt32)
    (op_k : Quanta.KOps.BinOp)
    (h_w : ∀ s, evalInstr s instr = binI32 op_w s)
    (h_agree : ∀ av bv,
       Quanta.KOps.evalBinOp op_k (Quanta.KOps.Value.vU32 av)
         (Quanta.KOps.Value.vU32 bv) =
         some (Quanta.KOps.Value.vU32 (op_w av bv)))
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (h_l_eq : lowerInstr s instr = lowerI32Bin s op_k)
    (h_ns_lower : isStructuredLower instr = false)
    (h_ns_eval : isStructuredEval instr = false)
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
    (hw : evalInstrs fuel ws (instr :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (instr :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  rw [lowerInstrs_cons_default fuel frames s instr rest h_ns_lower] at hl
  cases h_head : lowerInstr s instr with
  | none => rw [h_head] at hl; simp at hl
  | some head_pair =>
      rcases head_pair with ⟨s_after, ops_head⟩
      rw [h_head] at hl
      simp only [Option.bind_eq_bind, Option.some_bind] at hl
      cases h_post : lowerInstrs fuel frames s_after rest with
      | none => simp [h_post] at hl
      | some post_pair =>
          rcases post_pair with ⟨s_post, postOps⟩
          simp [h_post] at hl
          rcases hl with ⟨h_s_eq, h_ops_eq⟩
          rw [evalInstrs_cons_default fuel ws instr rest h_no_branch h_no_halt h_ns_eval] at hw
          cases h_eval_head : evalInstr ws instr with
          | none => rw [h_eval_head] at hw; simp at hw
          | some ws_after =>
              rw [h_eval_head] at hw
              simp only at hw
              obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
                preservation_i32Bin_generic instr op_w op_k h_w h_agree
                  ws s kst layout R h_kst_no_broke
                  ws_after s_after ops_head h_l_eq
                  h_eval_head h_head
              rw [h_l_eq] at h_head
              obtain ⟨_svb, _sva, _lrest, ra, _s3, opsA, rb, s4, opsB,
                      _h_stk, hca, hcb, _h_s4_stk, _h_s4_lr, _h_s4_lt,
                      _h_nr_le, _h_s_eq_shape, h_ops_head_eq⟩ :=
                lowerI32Bin_some_shape h_head
              have h_lf_opsA : loopFreeNoBreak opsA = true :=
                commit_emits_loopFreeNoBreak hca
              have h_lf_opsB : loopFreeNoBreak opsB = true :=
                commit_emits_loopFreeNoBreak hcb
              have h_lf_binOp :
                  loopFreeNoBreak [KernelOp.binOp s4.nextReg ra rb op_k .u32] = true := rfl
              have h_lf_head : loopFreeNoBreak ops_head = true := by
                rw [h_ops_head_eq]
                simp [loopFreeNoBreak_append, h_lf_opsA, h_lf_opsB, h_lf_binOp]
              have h_lf_head_shallow : loopFree ops_head = true :=
                loopFreeNoBreak_implies_loopFree h_lf_head
              have h_mid_broke : kst_mid.broke = false :=
                evalOps_loopFreeNoBreak_preserves_broke
                  h_lf_head h_kst_no_broke h_kst_eval
              have h_mid_no_branch : ws_after.branchTarget = none := by
                rw [h_w] at h_eval_head
                obtain ⟨_, _, _, _, h_ws_eq⟩ := binI32_some_shape h_eval_head
                rw [h_ws_eq]; simp [h_no_branch]
              have h_mid_no_halt : ws_after.halted = false := by
                rw [h_w] at h_eval_head
                obtain ⟨_, _, _, _, h_ws_eq⟩ := binI32_some_shape h_eval_head
                rw [h_ws_eq]; simp [h_no_halt]
              obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest, h_bridge_rest⟩ :=
                preservation_rest_bridge R_mid h_mid_no_branch h_mid_no_halt
                  h_mid_broke hw h_post
              obtain ⟨kst'', h_eval'', R'', h_bridge''⟩ :=
                preservation_evalInstrs_cons_compose_shallow_bridge
                  (F := F_rest) h_lf_head_shallow h_kst_eval h_mid_broke
                  ⟨kst'_mid, h_eval_rest, R_rest, h_bridge_rest⟩
              refine ⟨kst'', F_rest, ?_, ?_, ?_⟩
              · rw [← h_ops_eq]; exact h_eval''
              · rw [← h_s_eq]; exact R''
              · exact h_bridge''

-- ════════════════════════════════════════════════════════════════════
-- Generic i32-cmp cons bridge variant
-- ════════════════════════════════════════════════════════════════════

/-- Bridge-augmented `cons_i32Cmp_generic`. Same shape as the binop
    generic; head ops are `opsA ++ opsB ++ [.cmp, .cast]`. -/
theorem preservation_evalInstrs_cons_i32Cmp_generic_bridge
    (instr : WasmInstr) (p_w : UInt32 → UInt32 → Bool)
    (op_k : Quanta.KOps.CmpOp)
    (h_w : ∀ s, evalInstr s instr = cmpI32 p_w s)
    (h_l : ∀ s, lowerInstr s instr = lowerI32Cmp s op_k)
    (h_agree : ∀ av bv,
       Quanta.KOps.evalCmpOp op_k (Quanta.KOps.Value.vU32 av)
         (Quanta.KOps.Value.vU32 bv)
         = some (Quanta.KOps.Value.vBool (p_w av bv)))
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (h_ns_lower : isStructuredLower instr = false)
    (h_ns_eval : isStructuredEval instr = false)
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
    (hw : evalInstrs fuel ws (instr :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (instr :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  rw [lowerInstrs_cons_default fuel frames s instr rest h_ns_lower] at hl
  cases h_head : lowerInstr s instr with
  | none => rw [h_head] at hl; simp at hl
  | some head_pair =>
      rcases head_pair with ⟨s_after, ops_head⟩
      rw [h_head] at hl
      simp only [Option.bind_eq_bind, Option.some_bind] at hl
      cases h_post : lowerInstrs fuel frames s_after rest with
      | none => simp [h_post] at hl
      | some post_pair =>
          rcases post_pair with ⟨s_post, postOps⟩
          simp [h_post] at hl
          rcases hl with ⟨h_s_eq, h_ops_eq⟩
          rw [evalInstrs_cons_default fuel ws instr rest h_no_branch h_no_halt h_ns_eval] at hw
          cases h_eval_head : evalInstr ws instr with
          | none => rw [h_eval_head] at hw; simp at hw
          | some ws_after =>
              rw [h_eval_head] at hw
              simp only at hw
              obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
                preservation_i32Cmp_generic instr p_w op_k h_w h_l h_agree
                  ws s kst layout R h_kst_no_broke
                  ws_after s_after ops_head
                  h_eval_head h_head
              rw [h_l s] at h_head
              obtain ⟨_svb, _sva, _lrest, ra, _s3, opsA, rb, s4, opsB,
                      _h_stk, hca, hcb, _h_s4_stk, _h_s4_lr, _h_s4_lt,
                      _h_nr_le, _h_s_eq_shape, h_ops_head_eq⟩ :=
                lowerI32Cmp_some_shape h_head
              have h_lf_opsA : loopFreeNoBreak opsA = true :=
                commit_emits_loopFreeNoBreak hca
              have h_lf_opsB : loopFreeNoBreak opsB = true :=
                commit_emits_loopFreeNoBreak hcb
              have h_lf_tail :
                  loopFreeNoBreak [KernelOp.cmp s4.nextReg ra rb op_k .bool,
                                    KernelOp.cast (s4.nextReg + 1) s4.nextReg .bool .u32]
                    = true := rfl
              have h_lf_head : loopFreeNoBreak ops_head = true := by
                rw [h_ops_head_eq]
                simp [loopFreeNoBreak_append, h_lf_opsA, h_lf_opsB, h_lf_tail]
              have h_lf_head_shallow : loopFree ops_head = true :=
                loopFreeNoBreak_implies_loopFree h_lf_head
              have h_mid_broke : kst_mid.broke = false :=
                evalOps_loopFreeNoBreak_preserves_broke
                  h_lf_head h_kst_no_broke h_kst_eval
              have h_mid_no_branch : ws_after.branchTarget = none := by
                rw [h_w] at h_eval_head
                obtain ⟨_, _, _, _, h_ws_eq⟩ := cmpI32_some_shape h_eval_head
                rw [h_ws_eq]; simp [h_no_branch]
              have h_mid_no_halt : ws_after.halted = false := by
                rw [h_w] at h_eval_head
                obtain ⟨_, _, _, _, h_ws_eq⟩ := cmpI32_some_shape h_eval_head
                rw [h_ws_eq]; simp [h_no_halt]
              obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest, h_bridge_rest⟩ :=
                preservation_rest_bridge R_mid h_mid_no_branch h_mid_no_halt
                  h_mid_broke hw h_post
              obtain ⟨kst'', h_eval'', R'', h_bridge''⟩ :=
                preservation_evalInstrs_cons_compose_shallow_bridge
                  (F := F_rest) h_lf_head_shallow h_kst_eval h_mid_broke
                  ⟨kst'_mid, h_eval_rest, R_rest, h_bridge_rest⟩
              refine ⟨kst'', F_rest, ?_, ?_, ?_⟩
              · rw [← h_ops_eq]; exact h_eval''
              · rw [← h_s_eq]; exact R''
              · exact h_bridge''

-- ════════════════════════════════════════════════════════════════════
-- i32 binop bridge wrappers (Add / Sub / Mul / And / Or / Xor /
-- ShrU / DivU / RemU) — thin delegations to the generic bridge
-- ════════════════════════════════════════════════════════════════════

/-- Shared signature alias for the i32-binop bridge IH (cuts the
    repetition in the 10 wrappers below to a syntactic minimum). -/
@[reducible] def I32BinIHBridge
    (fuel : Nat) (frames : List FrameKind) (layout : BufferLayout)
    (rest : List WasmInstr) : Prop :=
  ∀ {ws_mid : WasmState} {s_mid : LowerState}
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
      BridgeClauses ws'_mid kst'_mid

/-- `i32Add :: rest` bridge (non-buffer path). -/
theorem preservation_evalInstrs_cons_i32Add_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (h_no_buf : ∀ slot base scale rest,
      s.stack ≠ .scaledIdx base scale :: .bufferPtr slot :: rest ∧
      s.stack ≠ .bufferPtr slot :: .scaledIdx base scale :: rest)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32BinIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Add :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Add :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  have h_l_eq : lowerInstr s .i32Add = lowerI32Bin s .add := by
    show lowerI32Add s = lowerI32Bin s .add
    unfold lowerI32Add
    split
    next base scale slot rest hs => exact absurd hs (h_no_buf slot base scale rest).left
    next slot base scale rest hs => exact absurd hs (h_no_buf slot base scale rest).right
    next => rfl
  exact preservation_evalInstrs_cons_i32Bin_generic_bridge
    .i32Add eval_u32_wrapping_add .add
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke h_l_eq rfl rfl rest
    preservation_rest_bridge ws' s' ops hw hl

/-- `i32Sub :: rest` bridge. -/
theorem preservation_evalInstrs_cons_i32Sub_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32BinIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Sub :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Sub :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  preservation_evalInstrs_cons_i32Bin_generic_bridge
    .i32Sub eval_u32_wrapping_sub .sub
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rfl rest
    preservation_rest_bridge ws' s' ops hw hl

/-- `i32Mul :: rest` bridge. -/
theorem preservation_evalInstrs_cons_i32Mul_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32BinIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Mul :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Mul :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  preservation_evalInstrs_cons_i32Bin_generic_bridge
    .i32Mul eval_u32_wrapping_mul .mul
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rfl rest
    preservation_rest_bridge ws' s' ops hw hl

/-- `i32And :: rest` bridge. -/
theorem preservation_evalInstrs_cons_i32And_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32BinIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32And :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32And :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  preservation_evalInstrs_cons_i32Bin_generic_bridge
    .i32And eval_u32_bitand .bAnd
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rfl rest
    preservation_rest_bridge ws' s' ops hw hl

/-- `i32Or :: rest` bridge. -/
theorem preservation_evalInstrs_cons_i32Or_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32BinIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Or :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Or :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  preservation_evalInstrs_cons_i32Bin_generic_bridge
    .i32Or eval_u32_bitor .bOr
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rfl rest
    preservation_rest_bridge ws' s' ops hw hl

/-- `i32Xor :: rest` bridge. -/
theorem preservation_evalInstrs_cons_i32Xor_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32BinIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Xor :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Xor :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  preservation_evalInstrs_cons_i32Bin_generic_bridge
    .i32Xor eval_u32_bitxor .bXor
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rfl rest
    preservation_rest_bridge ws' s' ops hw hl

/-- `i32ShrU :: rest` bridge. -/
theorem preservation_evalInstrs_cons_i32ShrU_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32BinIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32ShrU :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32ShrU :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  preservation_evalInstrs_cons_i32Bin_generic_bridge
    .i32ShrU (fun a b => a >>> b) .shr
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rfl rest
    preservation_rest_bridge ws' s' ops hw hl

/-- `i32DivU :: rest` bridge. -/
theorem preservation_evalInstrs_cons_i32DivU_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32BinIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32DivU :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32DivU :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  preservation_evalInstrs_cons_i32Bin_generic_bridge
    .i32DivU eval_u32_div .div
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rfl rest
    preservation_rest_bridge ws' s' ops hw hl

/-- `i32RemU :: rest` bridge. -/
theorem preservation_evalInstrs_cons_i32RemU_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32BinIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32RemU :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32RemU :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  preservation_evalInstrs_cons_i32Bin_generic_bridge
    .i32RemU eval_u32_rem .rem
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rfl rest
    preservation_rest_bridge ws' s' ops hw hl

-- ════════════════════════════════════════════════════════════════════
-- i32 cmp bridge wrappers (Eq / Ne / LtU / LeU / GtU / GeU)
-- ════════════════════════════════════════════════════════════════════

/-- Shared signature alias for the i32-cmp bridge IH. -/
@[reducible] def I32CmpIHBridge
    (fuel : Nat) (frames : List FrameKind) (layout : BufferLayout)
    (rest : List WasmInstr) : Prop :=
  I32BinIHBridge fuel frames layout rest

/-- `i32Eq :: rest` bridge. -/
theorem preservation_evalInstrs_cons_i32Eq_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32CmpIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Eq :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Eq :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  preservation_evalInstrs_cons_i32Cmp_generic_bridge
    .i32Eq (· == ·) .eq
    (fun _ => rfl) (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rest
    preservation_rest_bridge ws' s' ops hw hl

/-- `i32Ne :: rest` bridge. -/
theorem preservation_evalInstrs_cons_i32Ne_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32CmpIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Ne :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Ne :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  preservation_evalInstrs_cons_i32Cmp_generic_bridge
    .i32Ne (· != ·) .ne
    (fun _ => rfl) (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rest
    preservation_rest_bridge ws' s' ops hw hl

/-- `i32LtU :: rest` bridge. -/
theorem preservation_evalInstrs_cons_i32LtU_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32CmpIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32LtU :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32LtU :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  preservation_evalInstrs_cons_i32Cmp_generic_bridge
    .i32LtU (· < ·) .lt
    (fun _ => rfl) (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rest
    preservation_rest_bridge ws' s' ops hw hl

/-- `i32LeU :: rest` bridge. -/
theorem preservation_evalInstrs_cons_i32LeU_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32CmpIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32LeU :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32LeU :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  preservation_evalInstrs_cons_i32Cmp_generic_bridge
    .i32LeU (· <= ·) .le
    (fun _ => rfl) (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rest
    preservation_rest_bridge ws' s' ops hw hl

/-- `i32GtU :: rest` bridge. -/
theorem preservation_evalInstrs_cons_i32GtU_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32CmpIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32GtU :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32GtU :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  preservation_evalInstrs_cons_i32Cmp_generic_bridge
    .i32GtU (· > ·) .gt
    (fun _ => rfl) (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rest
    preservation_rest_bridge ws' s' ops hw hl

/-- `i32GeU :: rest` bridge. -/
theorem preservation_evalInstrs_cons_i32GeU_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32CmpIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32GeU :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32GeU :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  preservation_evalInstrs_cons_i32Cmp_generic_bridge
    .i32GeU (· >= ·) .ge
    (fun _ => rfl) (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rest
    preservation_rest_bridge ws' s' ops hw hl

/-- `localGet i :: rest` bridge (buffer-slot path). Head emits no
    IR — `lowerInstr` returns `(s.pushSym (.bufferPtr slot), [])`.
    kst_mid = kst because evalOps on the empty list is identity. -/
theorem preservation_evalInstrs_cons_localGet_bufferSlot_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (i : Nat) (slot : Nat)
    (h_buf : s.lookupBufferSlot i = some slot)
    (h_loc_buf : ∀ v, ws.locals.get? i = some v →
      ∃ n : UInt32, v = .wI32 n ∧ n.toNat = layout.startAddr slot)
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
    (hw : evalInstrs fuel ws (.localGet i :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.localGet i :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  -- Stage 3 cascade: body deferred (bridge variants depend on
  -- cons_local* which are currently sorry-deferred).
  sorry

theorem preservation_evalInstrs_cons_i32Shl_bufferPattern_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (k : Int) (base : Quanta.KOps.Reg) (ty : Quanta.KOps.Scalar)
    (lstk_rest : List SymVal)
    (h_stack : s.stack = .i32ConstSym k :: .reg base ty :: lstk_rest)
    (h_shift_eq : ∀ a : UInt32,
       regLookup kst.rf base = some (Quanta.KOps.Value.vU32 a) →
       (a <<< (UInt32.ofNat k.toNat)).toNat = a.toNat * (1 <<< k.toNat))
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32BinIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Shl :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Shl :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  rw [lowerInstrs_cons_default fuel frames s .i32Shl rest rfl] at hl
  have h_head : lowerInstr s .i32Shl =
      some ({ s with stack := .scaledIdx base (1 <<< k.toNat) :: lstk_rest }, []) := by
    show lowerI32Shl s = _
    unfold lowerI32Shl
    rw [h_stack]
  rw [h_head] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl
  cases h_post : lowerInstrs fuel frames
                  ({ s with stack := .scaledIdx base (1 <<< k.toNat) :: lstk_rest })
                  rest with
  | none => simp [h_post] at hl
  | some post_pair =>
      rcases post_pair with ⟨s_post, postOps⟩
      simp [h_post] at hl
      rcases hl with ⟨h_s_eq, h_ops_eq⟩
      rw [evalInstrs_cons_default fuel ws .i32Shl rest h_no_branch h_no_halt rfl] at hw
      cases h_eval_head : evalInstr ws .i32Shl with
      | none => rw [h_eval_head] at hw; simp at hw
      | some ws_after =>
          rw [h_eval_head] at hw
          simp only at hw
          obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
            preservation_i32Shl_bufferPattern ws s kst layout R k base ty lstk_rest
              h_stack h_shift_eq
              ws_after _ [] h_eval_head h_head
          have h_kst_mid_eq : kst_mid = kst := by
            simp [evalOps] at h_kst_eval; exact h_kst_eval.symm
          have h_mid_broke : kst_mid.broke = false := by
            rw [h_kst_mid_eq]; exact h_kst_no_broke
          have h_w : evalInstr ws .i32Shl = binI32 (· <<< ·) ws := rfl
          rw [h_w] at h_eval_head
          obtain ⟨_, _, _, _, h_ws_eq⟩ := binI32_some_shape h_eval_head
          have h_mid_no_branch : ws_after.branchTarget = none := by
            rw [h_ws_eq]; simp [h_no_branch]
          have h_mid_no_halt : ws_after.halted = false := by
            rw [h_ws_eq]; simp [h_no_halt]
          obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest, h_bridge_rest⟩ :=
            preservation_rest_bridge R_mid h_mid_no_branch h_mid_no_halt h_mid_broke hw h_post
          refine ⟨kst'_mid, F_rest, ?_, ?_, ?_⟩
          · rw [← h_ops_eq]
            rw [h_kst_mid_eq] at h_eval_rest
            exact h_eval_rest
          · rw [← h_s_eq]; exact R_rest
          · exact h_bridge_rest

/-- `i32Add :: rest` bridge (buffer-pattern fold, scaledIdx-first
    arm). Head emits no IR — symbolic stack rewritten to
    `.bufferAccess slot base scale`. -/
theorem preservation_evalInstrs_cons_i32Add_bufferPattern_scaledFirst_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (slot : Nat) (base : Quanta.KOps.Reg) (scale : Nat) (lstk_rest : List SymVal)
    (h_stack : s.stack = .scaledIdx base scale :: .bufferPtr slot :: lstk_rest)
    (h_addr_eq : ∀ a b_ptr : UInt32, ∀ b : UInt32,
       regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
       a.toNat = b.toNat * scale →
       b_ptr.toNat = layout.startAddr slot →
       (b_ptr + a).toNat = layout.startAddr slot + b.toNat * scale)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32BinIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Add :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Add :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  rw [lowerInstrs_cons_default fuel frames s .i32Add rest rfl] at hl
  have h_head : lowerInstr s .i32Add =
      some ({ s with stack := .bufferAccess slot base scale :: lstk_rest }, []) := by
    show lowerI32Add s = _
    unfold lowerI32Add
    rw [h_stack]
  rw [h_head] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl
  cases h_post : lowerInstrs fuel frames
                  ({ s with stack := .bufferAccess slot base scale :: lstk_rest })
                  rest with
  | none => simp [h_post] at hl
  | some post_pair =>
      rcases post_pair with ⟨s_post, postOps⟩
      simp [h_post] at hl
      rcases hl with ⟨h_s_eq, h_ops_eq⟩
      rw [evalInstrs_cons_default fuel ws .i32Add rest h_no_branch h_no_halt rfl] at hw
      cases h_eval_head : evalInstr ws .i32Add with
      | none => rw [h_eval_head] at hw; simp at hw
      | some ws_after =>
          rw [h_eval_head] at hw
          simp only at hw
          obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
            preservation_i32Add_bufferPattern_scaledFirst ws s kst layout R
              slot base scale lstk_rest h_stack h_addr_eq
              ws_after _ [] h_eval_head h_head
          have h_kst_mid_eq : kst_mid = kst := by
            simp [evalOps] at h_kst_eval; exact h_kst_eval.symm
          have h_mid_broke : kst_mid.broke = false := by
            rw [h_kst_mid_eq]; exact h_kst_no_broke
          have h_w : evalInstr ws .i32Add = binI32 eval_u32_wrapping_add ws := rfl
          rw [h_w] at h_eval_head
          obtain ⟨_, _, _, _, h_ws_eq⟩ := binI32_some_shape h_eval_head
          have h_mid_no_branch : ws_after.branchTarget = none := by
            rw [h_ws_eq]; simp [h_no_branch]
          have h_mid_no_halt : ws_after.halted = false := by
            rw [h_ws_eq]; simp [h_no_halt]
          obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest, h_bridge_rest⟩ :=
            preservation_rest_bridge R_mid h_mid_no_branch h_mid_no_halt h_mid_broke hw h_post
          refine ⟨kst'_mid, F_rest, ?_, ?_, ?_⟩
          · rw [← h_ops_eq]
            rw [h_kst_mid_eq] at h_eval_rest
            exact h_eval_rest
          · rw [← h_s_eq]; exact R_rest
          · exact h_bridge_rest

/-- `i32Add :: rest` bridge (buffer-pattern fold, bufferPtr-first
    arm). Same shape as the scaledFirst variant; addr-eq hypothesis
    accommodates the reversed addend order. -/
theorem preservation_evalInstrs_cons_i32Add_bufferPattern_ptrFirst_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (slot : Nat) (base : Quanta.KOps.Reg) (scale : Nat) (lstk_rest : List SymVal)
    (h_stack : s.stack = .bufferPtr slot :: .scaledIdx base scale :: lstk_rest)
    (h_addr_eq : ∀ a b_ptr : UInt32, ∀ b : UInt32,
       regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
       a.toNat = b.toNat * scale →
       b_ptr.toNat = layout.startAddr slot →
       (a + b_ptr).toNat = layout.startAddr slot + b.toNat * scale)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32BinIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Add :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Add :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  rw [lowerInstrs_cons_default fuel frames s .i32Add rest rfl] at hl
  have h_head : lowerInstr s .i32Add =
      some ({ s with stack := .bufferAccess slot base scale :: lstk_rest }, []) := by
    show lowerI32Add s = _
    unfold lowerI32Add
    rw [h_stack]
  rw [h_head] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl
  cases h_post : lowerInstrs fuel frames
                  ({ s with stack := .bufferAccess slot base scale :: lstk_rest })
                  rest with
  | none => simp [h_post] at hl
  | some post_pair =>
      rcases post_pair with ⟨s_post, postOps⟩
      simp [h_post] at hl
      rcases hl with ⟨h_s_eq, h_ops_eq⟩
      rw [evalInstrs_cons_default fuel ws .i32Add rest h_no_branch h_no_halt rfl] at hw
      cases h_eval_head : evalInstr ws .i32Add with
      | none => rw [h_eval_head] at hw; simp at hw
      | some ws_after =>
          rw [h_eval_head] at hw
          simp only at hw
          obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
            preservation_i32Add_bufferPattern_ptrFirst ws s kst layout R
              slot base scale lstk_rest h_stack h_addr_eq
              ws_after _ [] h_eval_head h_head
          have h_kst_mid_eq : kst_mid = kst := by
            simp [evalOps] at h_kst_eval; exact h_kst_eval.symm
          have h_mid_broke : kst_mid.broke = false := by
            rw [h_kst_mid_eq]; exact h_kst_no_broke
          have h_w : evalInstr ws .i32Add = binI32 eval_u32_wrapping_add ws := rfl
          rw [h_w] at h_eval_head
          obtain ⟨_, _, _, _, h_ws_eq⟩ := binI32_some_shape h_eval_head
          have h_mid_no_branch : ws_after.branchTarget = none := by
            rw [h_ws_eq]; simp [h_no_branch]
          have h_mid_no_halt : ws_after.halted = false := by
            rw [h_ws_eq]; simp [h_no_halt]
          obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest, h_bridge_rest⟩ :=
            preservation_rest_bridge R_mid h_mid_no_branch h_mid_no_halt h_mid_broke hw h_post
          refine ⟨kst'_mid, F_rest, ?_, ?_, ?_⟩
          · rw [← h_ops_eq]
            rw [h_kst_mid_eq] at h_eval_rest
            exact h_eval_rest
          · rw [← h_s_eq]; exact R_rest
          · exact h_bridge_rest

/-- `i32Shl :: rest` bridge (non-buffer path). Same fallthrough as
    the non-bridge wrapper: `h_no_buf` excludes the
    `<i32ConstSym k> :: <reg base ty> :: rest` fold so lowerInstr
    reduces to `lowerI32Bin s .shl`. -/
theorem preservation_evalInstrs_cons_i32Shl_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (h_no_buf : ∀ k base ty rest,
      s.stack ≠ .i32ConstSym k :: .reg base ty :: rest)
    (rest : List WasmInstr)
    (preservation_rest_bridge : I32BinIHBridge fuel frames layout rest)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Shl :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Shl :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  have h_l_eq : lowerInstr s .i32Shl = lowerI32Bin s .shl := by
    show lowerI32Shl s = lowerI32Bin s .shl
    unfold lowerI32Shl
    split
    next k base ty rest hs => exact absurd hs (h_no_buf k base ty rest)
    next => rfl
  exact preservation_evalInstrs_cons_i32Bin_generic_bridge
    .i32Shl (fun a b => a <<< b) .shl
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke h_l_eq rfl rfl rest
    preservation_rest_bridge ws' s' ops hw hl

/-- `i32Load offset align :: rest` bridge (buffer-access path).
    Head emits a single `.load` op; mid-state preserved via the
    standard loadI32 stack-only mutation. -/
theorem preservation_evalInstrs_cons_i32Load_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (slot : Nat) (base : Quanta.KOps.Reg) (lstk_rest : List SymVal)
    (offset align : Nat)
    (h_stack : s.stack = .bufferAccess slot base 4 :: lstk_rest)
    (h_offset : offset = 0)
    (h_in_bounds : ∀ b : UInt32,
       regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
       b.toNat < layout.length slot)
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
    (hw : evalInstrs fuel ws (.i32Load offset align :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Load offset align :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  rw [lowerInstrs_cons_default fuel frames s (.i32Load offset align) rest rfl] at hl
  have h_head : lowerInstr s (.i32Load offset align) =
      some ({ s with nextReg := s.nextReg + 1,
                     stack := .reg s.nextReg .u32 :: lstk_rest },
             [.load s.nextReg slot base .u32]) := by
    show lowerI32Load s = _
    unfold lowerI32Load
    rw [h_stack]
    rfl
  rw [h_head] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl
  cases h_post : lowerInstrs fuel frames
                  ({ s with nextReg := s.nextReg + 1,
                            stack := .reg s.nextReg .u32 :: lstk_rest })
                  rest with
  | none => simp [h_post] at hl
  | some post_pair =>
      rcases post_pair with ⟨s_post, postOps⟩
      simp [h_post] at hl
      rcases hl with ⟨h_s_eq, h_ops_eq⟩
      rw [evalInstrs_cons_default fuel ws (.i32Load offset align) rest h_no_branch h_no_halt rfl] at hw
      cases h_eval_head : evalInstr ws (.i32Load offset align) with
      | none => rw [h_eval_head] at hw; simp at hw
      | some ws_after =>
          rw [h_eval_head] at hw
          simp only at hw
          obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
            preservation_i32Load ws s kst layout R slot base lstk_rest offset align
              h_stack h_offset h_in_bounds
              ws_after _ _ h_eval_head h_head
          have h_lf_head : loopFreeNoBreak [KernelOp.load s.nextReg slot base .u32] = true :=
            rfl
          have h_lf_head_shallow : loopFree [KernelOp.load s.nextReg slot base .u32] = true :=
            loopFreeNoBreak_implies_loopFree h_lf_head
          have h_mid_broke : kst_mid.broke = false :=
            evalOps_loopFreeNoBreak_preserves_broke
              h_lf_head h_kst_no_broke h_kst_eval
          have h_w : evalInstr ws (.i32Load offset align) = loadI32 ws offset := rfl
          rw [h_w] at h_eval_head
          have h_mid_no_branch : ws_after.branchTarget = none := by
            unfold loadI32 at h_eval_head
            rcases hws : ws.stack with _ | ⟨vaddr, ws_rest⟩
            · simp [hws, WasmState.pop] at h_eval_head
            · simp [hws, WasmState.pop] at h_eval_head
              cases vaddr with
              | wI32 addr_w =>
                  simp at h_eval_head
                  rcases hmem : ws.mem.load_u32 (addr_w.toNat + offset) with _ | n
                  · simp [hmem] at h_eval_head
                  · simp [hmem, WasmState.push] at h_eval_head
                    rw [← h_eval_head]; simp [h_no_branch]
              | wI64 _ => simp at h_eval_head
              | wF32 _ => simp at h_eval_head
              | wF64 _ => simp at h_eval_head
          have h_mid_no_halt : ws_after.halted = false := by
            unfold loadI32 at h_eval_head
            rcases hws : ws.stack with _ | ⟨vaddr, ws_rest⟩
            · simp [hws, WasmState.pop] at h_eval_head
            · simp [hws, WasmState.pop] at h_eval_head
              cases vaddr with
              | wI32 addr_w =>
                  simp at h_eval_head
                  rcases hmem : ws.mem.load_u32 (addr_w.toNat + offset) with _ | n
                  · simp [hmem] at h_eval_head
                  · simp [hmem, WasmState.push] at h_eval_head
                    rw [← h_eval_head]; simp [h_no_halt]
              | wI64 _ => simp at h_eval_head
              | wF32 _ => simp at h_eval_head
              | wF64 _ => simp at h_eval_head
          obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest, h_bridge_rest⟩ :=
            preservation_rest_bridge R_mid h_mid_no_branch h_mid_no_halt h_mid_broke hw h_post
          obtain ⟨kst'', h_eval'', R'', h_bridge''⟩ :=
            preservation_evalInstrs_cons_compose_shallow_bridge
              (F := F_rest) h_lf_head_shallow h_kst_eval h_mid_broke
              ⟨kst'_mid, h_eval_rest, R_rest, h_bridge_rest⟩
          refine ⟨kst'', F_rest, ?_, ?_, ?_⟩
          · rw [← h_ops_eq]; exact h_eval''
          · rw [← h_s_eq]; exact R''
          · exact h_bridge''

/-- `i32Store offset align :: rest` bridge (buffer-access path).
    Head emits `opsCommit ++ [.store ...]`. -/
theorem preservation_evalInstrs_cons_i32Store_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (sv_val : SymVal) (slot : Nat) (base : Quanta.KOps.Reg) (lstk_rest : List SymVal)
    (offset align : Nat)
    (h_stack : s.stack = sv_val :: .bufferAccess slot base 4 :: lstk_rest)
    (h_offset : offset = 0)
    (h_in_bounds : ∀ b : UInt32,
       regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
       b.toNat < layout.length slot)
    (h_layout_no_overlap : ∀ b : UInt32,
       regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
       ∀ slot' idx',
         idx' < layout.length slot' →
         (slot', idx') ≠ (slot, b.toNat) →
         layout.startAddr slot + b.toNat * 4 + 4 ≤ layout.startAddr slot' + idx' * 4 ∨
         layout.startAddr slot' + idx' * 4 + 4 ≤ layout.startAddr slot + b.toNat * 4)
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
    (hw : evalInstrs fuel ws (.i32Store offset align :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Store offset align :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  rw [lowerInstrs_cons_default fuel frames s (.i32Store offset align) rest rfl] at hl
  let s_pop : LowerState :=
    { nextReg := s.nextReg, stack := lstk_rest,
      localReg := s.localReg, localTy := s.localTy,
      bufferSlots := s.bufferSlots, currentReg := s.currentReg }
  cases hca : s_pop.commit sv_val with
  | none =>
      have h_lw : lowerInstr s (.i32Store offset align) = none := by
        show lowerI32Store s = none
        unfold lowerI32Store
        simp only [LowerState.popSym, h_stack, Option.bind_eq_bind, Option.some_bind]
        rw [show ({ nextReg := s.nextReg, stack := lstk_rest,
                    localReg := s.localReg, localTy := s.localTy,
                    bufferSlots := s.bufferSlots, currentReg := s.currentReg } : LowerState).commit sv_val
                = s_pop.commit sv_val from rfl]
        rw [hca]
        rfl
      rw [h_lw] at hl
      simp at hl
  | some commit_pair =>
      rcases commit_pair with ⟨src, s3, opsCommit⟩
      let s_after : LowerState := s3
      let ops_head : List KernelOp := opsCommit ++ [.store slot base src .u32]
      have h_head : lowerInstr s (.i32Store offset align) = some (s_after, ops_head) := by
        show lowerI32Store s = some (s_after, ops_head)
        unfold lowerI32Store
        simp only [LowerState.popSym, h_stack, Option.bind_eq_bind, Option.some_bind]
        rw [show ({ nextReg := s.nextReg, stack := lstk_rest,
                    localReg := s.localReg, localTy := s.localTy,
                    bufferSlots := s.bufferSlots, currentReg := s.currentReg } : LowerState).commit sv_val
                = s_pop.commit sv_val from rfl]
        rw [hca]
        rfl
      rw [h_head] at hl
      simp only [Option.bind_eq_bind, Option.some_bind] at hl
      cases h_post : lowerInstrs fuel frames s_after rest with
      | none => simp [h_post] at hl
      | some post_pair =>
          rcases post_pair with ⟨s_post, postOps⟩
          simp [h_post] at hl
          rcases hl with ⟨h_s_eq, h_ops_eq⟩
          rw [evalInstrs_cons_default fuel ws (.i32Store offset align) rest h_no_branch h_no_halt rfl] at hw
          cases h_eval_head : evalInstr ws (.i32Store offset align) with
          | none => rw [h_eval_head] at hw; simp at hw
          | some ws_after =>
              rw [h_eval_head] at hw
              simp only at hw
              obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
                preservation_i32Store ws s kst layout R h_kst_no_broke
                  sv_val slot base lstk_rest offset align
                  h_stack h_offset h_in_bounds h_layout_no_overlap
                  ws_after s_after ops_head h_eval_head h_head
              have h_lf_commit : loopFreeNoBreak opsCommit = true :=
                commit_emits_loopFreeNoBreak hca
              have h_lf_store :
                  loopFreeNoBreak [KernelOp.store slot base src .u32] = true := rfl
              have h_lf_head : loopFreeNoBreak ops_head = true := by
                show loopFreeNoBreak (opsCommit ++ [KernelOp.store slot base src .u32]) = true
                simp [loopFreeNoBreak_append, h_lf_commit, h_lf_store]
              have h_lf_head_shallow : loopFree ops_head = true :=
                loopFreeNoBreak_implies_loopFree h_lf_head
              have h_mid_broke : kst_mid.broke = false :=
                evalOps_loopFreeNoBreak_preserves_broke
                  h_lf_head h_kst_no_broke h_kst_eval
              have h_w : evalInstr ws (.i32Store offset align) = storeI32 ws offset := rfl
              rw [h_w] at h_eval_head
              have h_ws_after_shape : ∃ ws_rest m',
                  ws_after = { ws with stack := ws_rest, mem := m' } := by
                unfold storeI32 at h_eval_head
                rcases hws : ws.stack with _ | ⟨vval, _ | ⟨vaddr, ws_rest⟩⟩
                · simp [hws, WasmState.pop] at h_eval_head
                · simp [hws, WasmState.pop] at h_eval_head
                · simp [hws, WasmState.pop] at h_eval_head
                  cases vaddr with
                  | wI32 addr_w =>
                      cases vval with
                      | wI32 v_w =>
                          simp at h_eval_head
                          rcases hmem : ws.mem.store_u32 (addr_w.toNat + offset) v_w with _ | m'
                          · simp [hmem] at h_eval_head
                          · simp [hmem] at h_eval_head
                            refine ⟨ws_rest, m', ?_⟩
                            rw [← h_eval_head]
                      | wI64 _ => simp at h_eval_head
                      | wF32 _ => simp at h_eval_head
                      | wF64 _ => simp at h_eval_head
                  | wI64 _ => simp at h_eval_head
                  | wF32 _ => simp at h_eval_head
                  | wF64 _ => simp at h_eval_head
              obtain ⟨_, _, h_ws_after_eq⟩ := h_ws_after_shape
              have h_mid_no_branch : ws_after.branchTarget = none := by
                rw [h_ws_after_eq]; simp [h_no_branch]
              have h_mid_no_halt : ws_after.halted = false := by
                rw [h_ws_after_eq]; simp [h_no_halt]
              obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest, h_bridge_rest⟩ :=
                preservation_rest_bridge R_mid h_mid_no_branch h_mid_no_halt h_mid_broke hw h_post
              obtain ⟨kst'', h_eval'', R'', h_bridge''⟩ :=
                preservation_evalInstrs_cons_compose_shallow_bridge
                  (F := F_rest) h_lf_head_shallow h_kst_eval h_mid_broke
                  ⟨kst'_mid, h_eval_rest, R_rest, h_bridge_rest⟩
              refine ⟨kst'', F_rest, ?_, ?_, ?_⟩
              · rw [← h_ops_eq]; exact h_eval''
              · rw [← h_s_eq]; exact R''
              · exact h_bridge''

-- ════════════════════════════════════════════════════════════════════
-- Control-flow bridges: br / brIf / wreturn
--
-- These differ from the non-control bridge variants. Each control-
-- flow theorem produces an exact characterization of (ws', kst') —
-- the post-state shape is fully determined by the arm. The downstream
-- bridging invariant (`body_branchTarget_implies_IR_broke`, the
-- mutual-block centerpiece) consumes these explicit shapes per arm.
--
-- The control-flow theorems below all output:
--   - the existence + Refines (same as non-bridge)
--   - an exact ws' = ws_post characterization
--   - an exact kst'.broke = b characterization (where b depends on arm)
--
-- This is **not** the same shape as `BridgeClauses` (which assumes a
-- non-control "ws' passthrough" semantics). The bridging invariant's
-- proof case-analyzes on the head instruction kind to use either
-- BridgeClauses (non-control) or these explicit shapes (control).
-- ════════════════════════════════════════════════════════════════════

/-- Bridge-augmented `preservation_br_loop_zero`. Produces exact
    post-state: `ws' = { ws with branchTarget := some 0 }`, `kst' = kst`.
    Bridge consequence: ws'.branchTarget = some 0 ∧ kst'.broke = false.
    The naive bridge "branchTarget set ⇒ broke = true" does NOT hold —
    `br 0` to enclosing Loop signals iteration-continue, not exit. -/
theorem preservation_br_loop_zero_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (_h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (h_target : frames.get? 0 = some .loopK)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.br 0 :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.br 0 :: rest) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧
            Refines ws' s' kst' layout ∧
            ws' = { ws with branchTarget := some 0 } ∧
            kst' = kst := by
  -- Delegate body of proof verbatim to the non-bridge theorem, then
  -- add the exact-shape clauses by recomputing ws_post / kst_post.
  obtain ⟨kst', h_ev, h_R⟩ :=
    preservation_br_loop_zero fuel frames ws s kst layout R h_no_branch h_no_halt
      rest h_target ws' s' ops hw hl
  -- Re-derive the exact post-state shape directly (mirrors the non-
  -- bridge proof's `ws_post` / `kst' = kst` derivation).
  have h_lower : lowerInstrs fuel frames s (.br 0 :: rest) = some (s, []) := by
    simp only [lowerInstrs, h_target, ↓reduceIte]
  rw [h_lower] at hl
  have hl' : (s, ([] : List KernelOp)) = (s', ops) := (Option.some.injEq _ _).mp hl
  have hops_eq : ([] : List KernelOp) = ops := congrArg Prod.snd hl'
  have h_cond : (ws.halted || ws.branchTarget.isSome) = false := by
    rw [h_no_halt, h_no_branch]; rfl
  let ws_post : WasmState := { ws with branchTarget := some 0 }
  have h_post_branch : ws_post.branchTarget = some 0 := rfl
  have h_evalInstr : evalInstr ws (.br 0) = some ws_post := rfl
  have h_step : evalInstrs fuel ws (.br 0 :: rest)
                  = evalInstrs fuel ws_post rest := by
    conv => lhs; unfold evalInstrs
    rw [h_cond]
    simp [h_evalInstr]
  rw [h_step] at hw
  rw [evalInstrs_branchTarget_some fuel ws_post rest 0 h_post_branch] at hw
  have hws'_eq : ws' = ws_post := ((Option.some.injEq _ _).mp hw).symm
  -- kst' from h_ev (which uses `evalOps 0 kst ops = some kst'`). Since
  -- ops = [] (from hops_eq), evalOps 0 kst [] = some kst, so kst' = kst.
  have hkst_eq : kst' = kst := by
    rw [← hops_eq] at h_ev
    simp [evalOps] at h_ev
    exact h_ev.symm
  exact ⟨kst', h_ev, h_R, hws'_eq, hkst_eq⟩

/-- Bridge-augmented `preservation_br_break_nonLoop`. Produces exact
    post-state: `ws' = { ws with branchTarget := some depth }`,
    `kst' = { kst with broke := true }`. -/
theorem preservation_br_break_nonLoop_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (depth : Nat) (rest : List WasmInstr)
    (kind : FrameKind) (h_kind_ne_loop : kind ≠ .loopK)
    (h_target : frames.get? depth = some kind)
    (h_loop_above : hasLoopAbove frames depth = true)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.br depth :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.br depth :: rest) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧
            Refines ws' s' kst' layout ∧
            ws' = { ws with branchTarget := some depth } ∧
            kst' = { kst with broke := true } := by
  obtain ⟨kst', h_ev, h_R⟩ :=
    preservation_br_break_nonLoop fuel frames ws s kst layout R h_no_branch h_no_halt
      depth rest kind h_kind_ne_loop h_target h_loop_above ws' s' ops hw hl
  -- Re-derive exact shape.
  have h_lower : lowerInstrs fuel frames s (.br depth :: rest)
                  = some (s, [KernelOp.breakOp]) := by
    cases kind with
    | block => simp only [lowerInstrs, h_target, h_loop_above, ↓reduceIte]
    | wif   => simp only [lowerInstrs, h_target, h_loop_above, ↓reduceIte]
    | loopK => exact (h_kind_ne_loop rfl).elim
  rw [h_lower] at hl
  have hl' : (s, [KernelOp.breakOp]) = (s', ops) :=
    (Option.some.injEq _ _).mp hl
  have hops_eq : [KernelOp.breakOp] = ops := congrArg Prod.snd hl'
  have h_cond : (ws.halted || ws.branchTarget.isSome) = false := by
    rw [h_no_halt, h_no_branch]; rfl
  let ws_post : WasmState := { ws with branchTarget := some depth }
  have h_post_branch : ws_post.branchTarget = some depth := rfl
  have h_evalInstr : evalInstr ws (.br depth) = some ws_post := rfl
  have h_step : evalInstrs fuel ws (.br depth :: rest)
                  = evalInstrs fuel ws_post rest := by
    conv => lhs; unfold evalInstrs
    rw [h_cond]
    simp [h_evalInstr]
  rw [h_step] at hw
  rw [evalInstrs_branchTarget_some fuel ws_post rest depth h_post_branch] at hw
  have hws'_eq : ws' = ws_post := ((Option.some.injEq _ _).mp hw).symm
  have hkst_eq : kst' = { kst with broke := true } := by
    rw [← hops_eq] at h_ev
    simp [evalOps, Quanta.KOps.evalOp] at h_ev
    exact h_ev.symm
  exact ⟨kst', h_ev, h_R, hws'_eq, hkst_eq⟩

/-- Bridge-augmented `preservation_br_loop_outer_break`. Produces
    exact post-state: same shape as `br_break_nonLoop_bridge`. -/
theorem preservation_br_loop_outer_break_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (depth : Nat) (rest : List WasmInstr)
    (h_depth_pos : depth ≠ 0)
    (h_target : frames.get? depth = some .loopK)
    (h_loop_above : hasLoopAbove frames depth = true)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.br depth :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.br depth :: rest) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧
            Refines ws' s' kst' layout ∧
            ws' = { ws with branchTarget := some depth } ∧
            kst' = { kst with broke := true } := by
  obtain ⟨kst', h_ev, h_R⟩ :=
    preservation_br_loop_outer_break fuel frames ws s kst layout R h_no_branch h_no_halt
      depth rest h_depth_pos h_target h_loop_above ws' s' ops hw hl
  have h_lower : lowerInstrs fuel frames s (.br depth :: rest)
                  = some (s, [KernelOp.breakOp]) := by
    simp only [lowerInstrs, h_target, h_depth_pos, ↓reduceIte, h_loop_above]
  rw [h_lower] at hl
  have hl' : (s, [KernelOp.breakOp]) = (s', ops) :=
    (Option.some.injEq _ _).mp hl
  have hops_eq : [KernelOp.breakOp] = ops := congrArg Prod.snd hl'
  have h_cond : (ws.halted || ws.branchTarget.isSome) = false := by
    rw [h_no_halt, h_no_branch]; rfl
  let ws_post : WasmState := { ws with branchTarget := some depth }
  have h_post_branch : ws_post.branchTarget = some depth := rfl
  have h_evalInstr : evalInstr ws (.br depth) = some ws_post := rfl
  have h_step : evalInstrs fuel ws (.br depth :: rest)
                  = evalInstrs fuel ws_post rest := by
    conv => lhs; unfold evalInstrs
    rw [h_cond]
    simp [h_evalInstr]
  rw [h_step] at hw
  rw [evalInstrs_branchTarget_some fuel ws_post rest depth h_post_branch] at hw
  have hws'_eq : ws' = ws_post := ((Option.some.injEq _ _).mp hw).symm
  have hkst_eq : kst' = { kst with broke := true } := by
    rw [← hops_eq] at h_ev
    simp [evalOps, Quanta.KOps.evalOp] at h_ev
    exact h_ev.symm
  exact ⟨kst', h_ev, h_R, hws'_eq, hkst_eq⟩

/-- Bridge-augmented `preservation_evalInstrs_cons_wreturn`. Produces
    exact post-state: `ws' = { ws with halted := true }`, `kst' = kst`.
    Note: wreturn does NOT set broke on the IR side — the bridge
    relies on the surrounding context propagating `ws'.halted = true`
    directly (not via broke). -/
theorem preservation_evalInstrs_cons_wreturn_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.wreturn :: []) = some ws')
    (hl : lowerInstrs fuel frames s (.wreturn :: []) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧
            Refines ws' s' kst' layout ∧
            ws' = { ws with halted := true } ∧
            kst' = kst := by
  obtain ⟨kst', h_ev, h_R⟩ :=
    preservation_evalInstrs_cons_wreturn fuel frames ws s kst layout R h_no_branch h_no_halt
      ws' s' ops hw hl
  rw [lowerInstrs_cons_default fuel frames s .wreturn []
      (by simp [isStructuredLower])] at hl
  simp only [lowerInstr, Option.bind_eq_bind, Option.some_bind,
             List.nil_append, lowerInstrs, pure, Option.some.injEq,
             Prod.mk.injEq] at hl
  obtain ⟨h_s_eq, h_ops_eq⟩ := hl
  let ws_post : WasmState := { ws with halted := true }
  have h_post_halted : ws_post.halted = true := rfl
  have h_evalInstr : evalInstr ws .wreturn = some ws_post := rfl
  rw [evalInstrs_cons_default fuel ws .wreturn [] h_no_branch h_no_halt
      (by simp [isStructuredEval])] at hw
  rw [h_evalInstr] at hw
  simp only at hw
  rw [evalInstrs_halted_true fuel ws_post [] h_post_halted] at hw
  have hws'_eq : ws' = ws_post := ((Option.some.injEq _ _).mp hw).symm
  have hkst_eq : kst' = kst := by
    rw [← h_ops_eq] at h_ev
    simp [evalOps] at h_ev
    exact h_ev.symm
  exact ⟨kst', h_ev, h_R, hws'_eq, hkst_eq⟩

-- ════════════════════════════════════════════════════════════════════
-- brIf bridges (all have `rest = []` precondition, per the existing
-- non-bridge theorems' L6 design — `brif_design.md` §2A).
--
-- The bridge clause depends on the WASM-side cond:
-- * cond = 0 (false): WASM falls through (branchTarget = none)
-- * cond ≠ 0 (true):  WASM sets branchTarget = some depth
--
-- And on the lowering arm:
-- * loop_self  (depth=0, target=enclosing Loop):
--     cond=0 → IR broke=true (exit loop); cond≠0 → IR no broke (continue)
-- * loop_outer_no_inner (depth>0, target=outer Loop, no inner loop):
--     cond=0 → IR no effect; cond≠0 → IR no broke (just branchTarget)
-- * loop_break_inner (depth>0, target+loopAbove):
--     cond=0 → IR no effect; cond≠0 → IR broke=true + branchTarget
--
-- The bridge variant outputs the (cond, broke, branchTarget) triple
-- as an existential — the downstream bridging invariant proof
-- case-splits on cond to discharge the appropriate clause.
-- ════════════════════════════════════════════════════════════════════

/-- Bridge-augmented `cons_brIf_loop_self`. Exposes the popped cond
    `c` and the WASM-side post-state shape (the existing non-bridge
    theorem already produces the IR-side facts). The `kst'.broke`
    characterization is left for the downstream bridging invariant
    proof to derive by re-running the non-bridge proof inline (which
    threads the cond-cases through the cast + branch ops). -/
theorem preservation_evalInstrs_cons_brIf_loop_self_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (h_target : frames.get? 0 = some .loopK)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.brIf 0 :: []) = some ws')
    (hl : lowerInstrs fuel frames s (.brIf 0 :: []) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      ∃ c : UInt32, ∃ rest_w : List WasmValue,
        ws.stack = .wI32 c :: rest_w ∧
        ((c = 0 ∧ ws' = { ws with stack := rest_w }) ∨
         (c ≠ 0 ∧ ws' = { ws with stack := rest_w,
                                   branchTarget := some 0 })) := by
  obtain ⟨kst', F, h_ev, h_R⟩ :=
    preservation_evalInstrs_cons_brIf_loop_self fuel frames ws s kst layout R
      h_no_branch h_no_halt h_kst_no_broke h_target ws' s' ops hw hl
  rw [evalInstrs_cons_default fuel ws (.brIf 0) [] h_no_branch h_no_halt
      (by simp [isStructuredEval])] at hw
  cases h_eval_head : evalInstr ws (.brIf 0) with
  | none => rw [h_eval_head] at hw; simp at hw
  | some ws_post =>
    rw [h_eval_head] at hw
    simp only at hw
    have h_eval_nil : evalInstrs fuel ws_post [] = some ws_post := by
      simp [evalInstrs]
    rw [h_eval_nil] at hw
    have h_ws'_eq : ws' = ws_post := ((Option.some.injEq _ _).mp hw).symm
    obtain ⟨c, rest_w, h_ws_stack, h_branch⟩ := evalInstr_brIf_shape_pub h_eval_head
    refine ⟨kst', F, h_ev, h_R, c, rest_w, h_ws_stack, ?_⟩
    rcases h_branch with ⟨hc, h_eq⟩ | ⟨hc, h_eq⟩
    · left; exact ⟨hc, by rw [h_ws'_eq]; exact h_eq⟩
    · right; exact ⟨hc, by rw [h_ws'_eq]; exact h_eq⟩

/-- Bridge-augmented `cons_brIf_loop_outer_no_inner`. Exposes the
    popped cond + WASM post-state. -/
theorem preservation_evalInstrs_cons_brIf_loop_outer_no_inner_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (depth : Nat) (h_depth_pos : depth ≠ 0)
    (h_target : frames.get? depth = some .loopK)
    (h_no_loop_above : hasLoopAbove frames depth = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.brIf depth :: []) = some ws')
    (hl : lowerInstrs fuel frames s (.brIf depth :: []) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      ∃ c : UInt32, ∃ rest_w : List WasmValue,
        ws.stack = .wI32 c :: rest_w ∧
        ((c = 0 ∧ ws' = { ws with stack := rest_w }) ∨
         (c ≠ 0 ∧ ws' = { ws with stack := rest_w,
                                   branchTarget := some depth })) := by
  obtain ⟨kst', F, h_ev, h_R⟩ :=
    preservation_evalInstrs_cons_brIf_loop_outer_no_inner fuel frames ws s kst layout R
      h_no_branch h_no_halt h_kst_no_broke depth h_depth_pos h_target h_no_loop_above
      ws' s' ops hw hl
  rw [evalInstrs_cons_default fuel ws (.brIf depth) [] h_no_branch h_no_halt
      (by simp [isStructuredEval])] at hw
  cases h_eval_head : evalInstr ws (.brIf depth) with
  | none => rw [h_eval_head] at hw; simp at hw
  | some ws_post =>
    rw [h_eval_head] at hw
    simp only at hw
    have h_eval_nil : evalInstrs fuel ws_post [] = some ws_post := by
      simp [evalInstrs]
    rw [h_eval_nil] at hw
    have h_ws'_eq : ws' = ws_post := ((Option.some.injEq _ _).mp hw).symm
    obtain ⟨c, rest_w, h_ws_stack, h_branch⟩ := evalInstr_brIf_shape_pub h_eval_head
    refine ⟨kst', F, h_ev, h_R, c, rest_w, h_ws_stack, ?_⟩
    rcases h_branch with ⟨hc, h_eq⟩ | ⟨hc, h_eq⟩
    · left; exact ⟨hc, by rw [h_ws'_eq]; exact h_eq⟩
    · right; exact ⟨hc, by rw [h_ws'_eq]; exact h_eq⟩

/-- Bridge-augmented `cons_brIf_loop_break_inner`. Same shape;
    handles arms 2 + 4. -/
theorem preservation_evalInstrs_cons_brIf_loop_break_inner_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (depth : Nat)
    (kind : FrameKind)
    (h_depth_pos_or_nonloop :
      (depth ≠ 0 ∧ kind = .loopK) ∨ kind ≠ .loopK)
    (h_target : frames.get? depth = some kind)
    (h_loop_above : hasLoopAbove frames depth = true)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.brIf depth :: []) = some ws')
    (hl : lowerInstrs fuel frames s (.brIf depth :: []) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      ∃ c : UInt32, ∃ rest_w : List WasmValue,
        ws.stack = .wI32 c :: rest_w ∧
        ((c = 0 ∧ ws' = { ws with stack := rest_w }) ∨
         (c ≠ 0 ∧ ws' = { ws with stack := rest_w,
                                   branchTarget := some depth })) := by
  obtain ⟨kst', F, h_ev, h_R⟩ :=
    preservation_evalInstrs_cons_brIf_loop_break_inner fuel frames ws s kst layout R
      h_no_branch h_no_halt h_kst_no_broke depth kind h_depth_pos_or_nonloop
      h_target h_loop_above ws' s' ops hw hl
  rw [evalInstrs_cons_default fuel ws (.brIf depth) [] h_no_branch h_no_halt
      (by simp [isStructuredEval])] at hw
  cases h_eval_head : evalInstr ws (.brIf depth) with
  | none => rw [h_eval_head] at hw; simp at hw
  | some ws_post =>
    rw [h_eval_head] at hw
    simp only at hw
    have h_eval_nil : evalInstrs fuel ws_post [] = some ws_post := by
      simp [evalInstrs]
    rw [h_eval_nil] at hw
    have h_ws'_eq : ws' = ws_post := ((Option.some.injEq _ _).mp hw).symm
    obtain ⟨c, rest_w, h_ws_stack, h_branch⟩ := evalInstr_brIf_shape_pub h_eval_head
    refine ⟨kst', F, h_ev, h_R, c, rest_w, h_ws_stack, ?_⟩
    rcases h_branch with ⟨hc, h_eq⟩ | ⟨hc, h_eq⟩
    · left; exact ⟨hc, by rw [h_ws'_eq]; exact h_eq⟩
    · right; exact ⟨hc, by rw [h_ws'_eq]; exact h_eq⟩

-- ════════════════════════════════════════════════════════════════════
-- L8.1 — cons_block preservation (fall-through bodies, Path A)
--
-- Scope: bodies whose iteration completes with `branchTarget = none`
-- on the WASM side AND `kst.broke = false` on the IR side. This
-- covers blocks containing only straight-line code, nested wif
-- without inner escapes, or nested wloop that completes normally.
--
-- The general claim (bodies that propagate branchTarget = some d
-- out of the block boundary, or bodies that exit via br to outer
-- frames) is deferred to a follow-up session that proves the
-- bridging invariant `body_branchTarget_implies_IR_broke` via the
-- mutual block (per `l8_5_scoping.md` §4b).
--
-- The body's "fall-through" property is supplied as an explicit
-- hypothesis `body_falls_through`; downstream (the framework
-- theorem) discharges it via a syntactic `WellFormedKernel`
-- predicate that rules out inner escapes (see scoping doc §5 R1).
-- ════════════════════════════════════════════════════════════════════

/-- `block _ :: rest` preservation under the fall-through body
    precondition. Body runs to completion with no branchTarget /
    halted set on the WASM side and no broke on the IR side; then
    post (the rest after the matching `wend`) runs via the IH. -/
theorem preservation_evalInstrs_cons_block_fallthrough
    (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (bt : Nat) (rest : List WasmInstr)
    -- The lowering uses `fuel = bt + 1`; body lowering recurses
    -- with `bt`. We split off `bt` to make the structured-control
    -- decrement explicit.
    (body post : List WasmInstr)
    (h_split : splitAtEnd rest = some (body, post))
    -- Body's recursive bridge result (the IH-on-body the caller
    -- supplies). Note `frames` extends with `.block` here.
    (body_preserves : ∀ {ws_b : WasmState} {s_b : LowerState}
        {kst_b : Quanta.KOps.State}
        (_R_b : Refines ws_b s_b kst_b layout)
        (_h_nb_b : ws_b.branchTarget = none)
        (_h_nh_b : ws_b.halted = false)
        (_h_nbk_b : kst_b.broke = false)
        {ws'_b : WasmState} {s'_b : LowerState} {bodyOps : List KernelOp}
        (_hw_b : evalInstrs bt ws_b body = some ws'_b)
        (_hl_b : lowerInstrs bt (.block :: frames) s_b body = some (s'_b, bodyOps)),
      ∃ (kst'_b : Quanta.KOps.State) (F : Nat),
        evalOps F kst_b bodyOps = some kst'_b ∧
        Refines ws'_b s'_b kst'_b layout ∧
        BridgeClauses ws'_b kst'_b)
    -- Fall-through hypothesis: body's post-state has no branchTarget,
    -- no halt, no broke. Downstream caller discharges this.
    (body_falls_through : ∀ {ws_b : WasmState} {s_b : LowerState}
        {kst_b : Quanta.KOps.State} {ws'_b : WasmState} {s'_b : LowerState}
        {bodyOps : List KernelOp}
        (_R_b : Refines ws_b s_b kst_b layout)
        (_h_nb_b : ws_b.branchTarget = none)
        (_h_nh_b : ws_b.halted = false)
        (_h_nbk_b : kst_b.broke = false)
        (_hw_b : evalInstrs bt ws_b body = some ws'_b)
        (_hl_b : lowerInstrs bt (.block :: frames) s_b body = some (s'_b, bodyOps)),
      ws'_b.branchTarget = none ∧ ws'_b.halted = false)
    (post_preserves : ∀ {ws_p : WasmState} {s_p : LowerState}
        {kst_p : Quanta.KOps.State}
        (_R_p : Refines ws_p s_p kst_p layout)
        (_h_nb_p : ws_p.branchTarget = none)
        (_h_nh_p : ws_p.halted = false)
        (_h_nbk_p : kst_p.broke = false)
        {ws'_p : WasmState} {s'_p : LowerState} {postOps : List KernelOp}
        (_hw_p : evalInstrs bt ws_p post = some ws'_p)
        (_hl_p : lowerInstrs bt frames s_p post = some (s'_p, postOps)),
      ∃ (kst'_p : Quanta.KOps.State) (F : Nat),
        evalOps F kst_p postOps = some kst'_p ∧
        Refines ws'_p s'_p kst'_p layout ∧
        BridgeClauses ws'_p kst'_p)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs (bt + 1) ws (.block 0 :: rest) = some ws')
    (hl : lowerInstrs (bt + 1) frames s (.block 0 :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  -- Unfold the lowerInstrs block arm.
  simp only [lowerInstrs] at hl
  rw [h_split] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl
  -- hl: do { (s1, innerOps) ← lowerInstrs bt (.block :: frames) s body
  --        ; (s2, postOps) ← lowerInstrs bt frames s1 post
  --        ; pure (s2, innerOps ++ postOps) } = some (s', ops)
  cases h_lb : lowerInstrs bt (.block :: frames) s body with
  | none => simp [h_lb] at hl
  | some body_pair =>
    rcases body_pair with ⟨s1, innerOps⟩
    simp [h_lb] at hl
    cases h_lp : lowerInstrs bt frames s1 post with
    | none => simp [h_lp] at hl
    | some post_pair =>
      rcases post_pair with ⟨s2, postOps⟩
      simp [h_lp] at hl
      rcases hl with ⟨h_s_eq, h_ops_eq⟩
      -- Eval side: block arm.
      simp only [evalInstrs] at hw
      have h_cond : (ws.halted || ws.branchTarget.isSome) = false := by
        rw [h_no_halt, h_no_branch]; rfl
      rw [h_cond] at hw
      simp only [Bool.false_eq_true, ↓reduceIte] at hw
      rw [h_split] at hw
      simp only at hw
      cases h_eb : evalInstrs bt ws body with
      | none => simp [h_eb] at hw
      | some ws_after_body =>
        simp [h_eb] at hw
        -- ws_after_body is the body's post-state.
        -- Apply the fall-through hypothesis to learn it has no
        -- branchTarget / halted.
        obtain ⟨kst_after_body, F_body, h_eb_kst, R_body, _h_body_bridge⟩ :=
          body_preserves R h_no_branch h_no_halt h_kst_no_broke h_eb h_lb
        obtain ⟨h_bft_none, h_bft_nh⟩ :=
          body_falls_through R h_no_branch h_no_halt h_kst_no_broke h_eb h_lb
        -- branchTarget = none arm of block's eval.
        rw [h_bft_none] at hw
        simp only at hw
        -- hw: evalInstrs bt ws_after_body post = some ws'.
        -- Need kst_after_body.broke = false. Comes from
        -- BridgeClauses with ws_after_body.branchTarget = none.
        have h_body_broke : kst_after_body.broke = false := by
          have ⟨_, h_nb_implies_no_broke⟩ := _h_body_bridge
          exact h_nb_implies_no_broke h_bft_none
        -- Apply post_preserves on ws_after_body / s1 / kst_after_body.
        obtain ⟨kst', F_post, h_ev_post, R_post, h_bridge_post⟩ :=
          post_preserves R_body h_bft_none h_bft_nh h_body_broke hw h_lp
        -- Compose: ops = innerOps ++ postOps. evalOps the concat.
        -- Need to lift (innerOps, postOps) composition. Use
        -- evalOps_append (broke-aware): runs innerOps to kst_after_body
        -- (broke=false), then postOps from there to kst'.
        refine ⟨kst', max F_body F_post, ?_, ?_, h_bridge_post⟩
        · rw [← h_ops_eq]
          -- evalOps (max F_body F_post) kst (innerOps ++ postOps)
          --   = some kst', via evalOps_append + fuel monotonicity.
          have h_body_max : evalOps (max F_body F_post) kst innerOps
                              = some kst_after_body := by
            have h := evalOps_fuel_mono (Nat.le_max_left F_body F_post)
                        h_eb_kst
            exact h
          have h_post_max : evalOps (max F_body F_post) kst_after_body postOps
                              = some kst' := by
            have h := evalOps_fuel_mono (Nat.le_max_right F_body F_post)
                        h_ev_post
            exact h
          exact evalOps_append h_body_max h_body_broke
                |>.trans h_post_max
        · rw [← h_s_eq]; exact R_post

-- ════════════════════════════════════════════════════════════════════
-- L8.2 — cons_wif preservation (no-else, fall-through, Path A)
--
-- The wif lowering's `localReg` snapshot/restore (Translate.lean,
-- post-this-session commit) unblocks the cons_wif proof. The two
-- bodies are both lowered from a `localReg` snapshot taken at
-- If-entry; the post-frame state restores the same snapshot. So
-- the eval-side `Refines ws s kst` propagates cleanly across the
-- branch the eval picks, and the unselected branch's lowering
-- doesn't corrupt the post-state's locals view.
--
-- Scope below: empty elseBody (canonical Rust `if cond { ... }` —
-- no `else` clause). The thenBody is fall-through (post-state has
-- `branchTarget = none`, `halted = false`, `broke = false`).
--
-- Full wif (non-empty thenBody / elseBody) requires careful
-- handling of:
--
--  1. The Refines lift across the unselected branch's lowering —
--     UNBLOCKED by the snapshot/restore in Translate.lean
--     (commit a045ead).
--  2. thenBody mutating locals — STILL BLOCKED. Spec's localSet
--     emits `[.copy stable src]` (single Copy). Production emits
--     `[.copy fresh src, .copy stable fresh]` so stable_reg
--     always holds the latest value. Without the dual Copy, if
--     thenBody runs `localSet i`, post-frame reads see the OLD
--     value at the snapshot's register — Refines fails on
--     `LocalsRefines`. Stage 3 of the wasm_local_renaming port
--     unblocks this; deferred until cons_wloop needs it.
--
-- Below: minimal cons_wif activation for the empty-thenBody +
-- empty-elseBody case. Validates the structured-control composition
-- end-to-end at trivial body shapes. Larger fall-through bodies
-- without localSet are the next extension.
-- ════════════════════════════════════════════════════════════════════

/-- `wif _ :: rest` preservation, both bodies empty. The wif
    degenerates to "pop cond + run post"; useful as a smoke test
    of the structured-control composition machinery before tackling
    the general fall-through case. -/
theorem preservation_evalInstrs_cons_wif_trivialBodies
    (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (bt : Nat) (rest : List WasmInstr)
    (post : List WasmInstr)
    (h_split : splitAtElseOrEnd rest = some ([], [], post))
    (post_preserves : ∀ {ws_p : WasmState} {s_p : LowerState}
        {kst_p : Quanta.KOps.State}
        (_R_p : Refines ws_p s_p kst_p layout)
        (_h_nb_p : ws_p.branchTarget = none)
        (_h_nh_p : ws_p.halted = false)
        (_h_nbk_p : kst_p.broke = false)
        {ws'_p : WasmState} {s'_p : LowerState} {postOps : List KernelOp}
        (_hw_p : evalInstrs bt ws_p post = some ws'_p)
        (_hl_p : lowerInstrs bt frames s_p post = some (s'_p, postOps)),
      ∃ (kst'_p : Quanta.KOps.State) (F : Nat),
        evalOps F kst_p postOps = some kst'_p ∧
        Refines ws'_p s'_p kst'_p layout ∧
        BridgeClauses ws'_p kst'_p)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs (bt + 1) ws (.wif 0 :: rest) = some ws')
    (hl : lowerInstrs (bt + 1) frames s (.wif 0 :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  -- Unfold the lowerInstrs wif arm.
  simp only [lowerInstrs] at hl
  rw [h_split] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl
  cases hpop : s.popSym with
  | none => simp [hpop] at hl
  | some pop_pair =>
    rcases pop_pair with ⟨svCond, s0⟩
    simp [hpop] at hl
    cases hcommit : s0.commit svCond with
    | none => simp [hcommit] at hl
    | some commit_triple =>
      rcases commit_triple with ⟨cond, s1, opsCommit⟩
      simp [hcommit] at hl
      simp only [LowerState.alloc] at hl
      -- With both bodies empty, lowerInstrs ... [] = some (input, []).
      -- s_cast = { s1 with nextReg := s1.nextReg + 1 }.
      -- Reduce both inner lowerInstrs calls.
      have h_then_nil :
          lowerInstrs bt (.wif :: frames)
            ({ s1 with nextReg := s1.nextReg + 1 } : LowerState) []
            = some ({ s1 with nextReg := s1.nextReg + 1 }, []) := by
        simp [lowerInstrs]
      rw [h_then_nil] at hl
      simp only [Option.some_bind] at hl
      -- After then, restore localReg/localTy → equals s_cast since they
      -- were unchanged. The restored state is still s_cast in shape.
      have h_else_nil :
          lowerInstrs bt (.wif :: frames)
            ({ ({ s1 with nextReg := s1.nextReg + 1 } : LowerState) with
                  localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                  localTy  := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy } : LowerState) []
            = some ({ ({ s1 with nextReg := s1.nextReg + 1 } : LowerState) with
                       localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                       localTy  := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy }, []) := by
        simp [lowerInstrs]
      rw [h_else_nil] at hl
      simp only [Option.some_bind] at hl
      -- Now `hl` is: pure on post's lowering output.
      -- s3_restored simplifies. Let s_cast' = the state passed to post.
      cases hlp : lowerInstrs bt frames
          ({ ({ ({ s1 with nextReg := s1.nextReg + 1 } : LowerState) with
                   localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                   localTy  := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy } : LowerState) with
                localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                localTy  := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy } : LowerState)
          post with
      | none => simp [hlp] at hl
      | some post_pair =>
        rcases post_pair with ⟨s4, postOps⟩
        simp [hlp] at hl
        rcases hl with ⟨h_s_eq, h_ops_eq⟩
        -- Eval side: unfold wif.
        simp only [evalInstrs] at hw
        have h_cond : (ws.halted || ws.branchTarget.isSome) = false := by
          rw [h_no_halt, h_no_branch]; rfl
        rw [h_cond] at hw
        simp only [Bool.false_eq_true, ↓reduceIte] at hw
        cases hpop_w : ws.pop with
        | none => simp [hpop_w] at hw
        | some pop_w =>
          rcases pop_w with ⟨vc, ws0⟩
          simp [hpop_w] at hw
          rw [h_split] at hw
          simp only at hw
          cases vc with
          | wI32 c =>
            simp only at hw
            -- body picked is [] either way (c=0 or c≠0). evalInstrs bt ws0 [] = some ws0.
            have h_ws0_nb : ws0.branchTarget = none := by
              rw [WasmState.pop] at hpop_w
              rcases hst : ws.stack with _ | ⟨v0, rst⟩
              · rw [hst] at hpop_w; simp at hpop_w
              · rw [hst] at hpop_w
                simp at hpop_w
                obtain ⟨_, hws0⟩ := hpop_w
                rw [← hws0]
                simp [h_no_branch]
            have h_ws0_nh : ws0.halted = false := by
              rw [WasmState.pop] at hpop_w
              rcases hst : ws.stack with _ | ⟨v0, rst⟩
              · rw [hst] at hpop_w; simp at hpop_w
              · rw [hst] at hpop_w
                simp at hpop_w
                obtain ⟨_, hws0⟩ := hpop_w
                rw [← hws0]
                simp [h_no_halt]
            -- Pop semantics: ws.pop = some (.wI32 c, ws0) ⇒ ws.stack = .wI32 c :: ws0.stack.
            have h_pop_facts : ws.stack = .wI32 c :: ws0.stack ∧
                              ws0 = { ws with stack := ws0.stack } := by
              rw [WasmState.pop] at hpop_w
              rcases hst : ws.stack with _ | ⟨v0, rest⟩
              · rw [hst] at hpop_w; simp at hpop_w
              · rw [hst] at hpop_w
                simp at hpop_w
                obtain ⟨hv0, hws0⟩ := hpop_w
                subst hv0
                -- ws.stack = v0 :: rest in goal; ws0 = { ws with stack := rest }
                have h_ws0_stack : ws0.stack = rest := by rw [← hws0]
                refine ⟨?_, ?_⟩
                · rw [h_ws0_stack]
                · rw [h_ws0_stack]; exact hws0.symm
            obtain ⟨h_ws_stack, h_ws0_eq⟩ := h_pop_facts
            -- Whichever body picked, evalInstrs bt ws0 [] = some ws0.
            have h_body_eval : evalInstrs bt ws0 (if c = 0 then ([] : List WasmInstr) else []) = some ws0 := by
              split <;> simp [evalInstrs]
            rw [h_body_eval] at hw
            simp only at hw
            -- ws0.branchTarget = none → run post.
            rw [h_ws0_nb] at hw
            simp only at hw
            -- hw: evalInstrs bt ws0 post = some ws'.
            -- IR side: opsCommit (kst → kst1) + cast (kst1 → kst_cast) +
            -- branch reads cond_bool = vBool (!decide (c=0)); both arms = []
            -- so branch is a no-op (kst_cast → kst_cast). Then postOps from
            -- kst_cast → kst'.
            obtain ⟨kst1, h_evalCommit, h_kst1_ok, h_lookup, _R_post,
                    _, _, _, _, _, _⟩ :=
              brIf_cond_pop_commit_correct_pub R h_ws_stack hpop hcommit h_kst_no_broke
            let cond_bool : Quanta.KOps.Reg := s1.nextReg
            let kst_cast : Quanta.KOps.State :=
              { kst1 with rf := Quanta.KOps.regWrite kst1.rf cond_bool
                                  (Quanta.KOps.vBool (!decide (c = 0))) }
            have h_kst_cast_broke : kst_cast.broke = false := h_kst1_ok
            -- Build Refines ws0 s_cast kst_cast layout.
            have h_s1_stack : s1.stack = s0.stack := commit_preserves_stack hcommit
            have h_s1_eq : ({ s1 with stack := s0.stack } : LowerState) = s1 := by
              cases s1 with
              | mk nr st lr lt bs cr =>
                simp at h_s1_stack
                rw [h_s1_stack]
            have R_at_s1 : Refines ws0 s1 kst1 layout := by
              rw [h_ws0_eq, ← h_s1_eq]
              exact _R_post
            -- Build Refines ws0 s_cast kst_cast layout (lift past the cast's
            -- fresh write at cond_bool = s1.nextReg).
            have R_at_cast : Refines ws0
                { s1 with nextReg := s1.nextReg + 1 } kst_cast layout := by
              refine ⟨?_, ?_, ?_, ?_, R_at_s1.injLocals, R_at_s1.heapRefines, ?_, ?_⟩
              · -- StackRefines.
                refine ⟨?_, ?_⟩
                · show ws0.stack.length = s1.stack.length
                  exact R_at_s1.stk.left
                · intro i v hv
                  obtain ⟨svi, hsv_get, henc⟩ := R_at_s1.stk.right i v hv
                  have hsv_in : svi ∈ s1.stack := List.mem_of_get? hsv_get
                  refine ⟨svi, hsv_get, ?_⟩
                  apply WasmValue.encodes_preserved_of_fresh _ henc
                  intro r hr
                  exact R_at_s1.fresh.left svi hsv_in r hr
              · -- LocalsRefines.
                intro i r hfind v hv
                have henc := R_at_s1.locs i r hfind v hv
                have hr_lt : r < s1.nextReg := by
                  have hpair : (i, r) ∈ s1.localReg := List.mem_of_find?_eq_some hfind
                  exact R_at_s1.fresh.right (i, r) hpair
                apply WasmValue.encodes_preserved_of_fresh _ henc
                intro r' hr'
                simp [SymVal.regs] at hr'
                subst hr'; exact hr_lt
              · -- Fresh.
                refine ⟨?_, ?_⟩
                · intro sv hsv r hr
                  show r < s1.nextReg + 1
                  exact Nat.lt_succ_of_lt (R_at_s1.fresh.left sv hsv r hr)
                · intro ir hir
                  show ir.snd < s1.nextReg + 1
                  exact Nat.lt_succ_of_lt (R_at_s1.fresh.right ir hir)
              · -- AliasFree.
                intro ir hir sv hsv
                exact R_at_s1.aliasFree ir hir sv hsv
              · -- CurrentRegRefines: s_cast.currentReg = s1.currentReg; lift past fresh write.
                show CurrentRegRefines layout _ s1.currentReg _
                exact CurrentRegRefines_preserved_fresh R_at_s1.currentReg R_at_s1.freshCurrent _
              · -- FreshCurrent: nextReg bumps by 1.
                intro ir hir
                show ir.snd < s1.nextReg + 1
                exact Nat.lt_succ_of_lt (R_at_s1.freshCurrent ir hir)
            -- s3_restored simplifies to s_cast (both restores idempotent on
            -- already-snapshot-matching state since bodies are empty).
            -- The state passed to post is structurally s_cast.
            -- Apply post_preserves.
            -- First normalize the messy nested structure-update to s_cast.
            have h_s_cast_form :
                ({ ({ ({ s1 with nextReg := s1.nextReg + 1 } : LowerState) with
                         localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                         localTy  := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy } : LowerState) with
                      localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                      localTy  := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy } : LowerState)
                = ({ s1 with nextReg := s1.nextReg + 1 } : LowerState) := by
              cases s1 with
              | mk nr st lr lt bs cr => rfl
            rw [h_s_cast_form] at hlp
            obtain ⟨kst', F_p, h_ev_p, R_p, h_bridge_p⟩ :=
              post_preserves R_at_cast h_ws0_nb h_ws0_nh h_kst_cast_broke hw hlp
            -- Compose: opsCommit (→ kst1) + cast (→ kst_cast) + branch[[],[]] (→ kst_cast) + postOps (→ kst').
            let F : Nat := max F_p 1
            refine ⟨kst', F, ?_, ?_, h_bridge_p⟩
            · rw [← h_ops_eq]
              -- evalOps F kst (opsCommit ++ [cast, branch] ++ postOps)
              -- = evalOps F kst1 ([cast, branch] ++ postOps)  via evalOps_append
              -- = evalOps F kst_cast (branch :: postOps)  via cast eval
              -- = evalOps F kst_cast postOps  via branch eval (both arms = [])
              -- = some kst'  via post_preserves
              have h1 : evalOps F kst opsCommit = some kst1 :=
                evalOps_fuel_mono (Nat.zero_le _) h_evalCommit
              have h_post_max : evalOps F kst_cast postOps = some kst' :=
                evalOps_fuel_mono (Nat.le_max_left _ _) h_ev_p
              have h_cast_max : Quanta.KOps.evalOp F kst1
                  (KernelOp.cast cond_bool cond Quanta.KOps.Scalar.u32 Quanta.KOps.Scalar.bool)
                  = some kst_cast := by
                simp [Quanta.KOps.evalOp, h_lookup, Quanta.KOps.evalCast, kst_cast]
              -- branch reads cond_bool (= vBool _), picks one arm; both arms = [].
              -- evalOps F kst_cast [] = some kst_cast.
              have h_lookup_cast :
                  Quanta.KOps.regLookup kst_cast.rf cond_bool
                    = some (Quanta.KOps.Value.vBool (!decide (c = 0))) := by
                show Quanta.KOps.regLookup
                       (Quanta.KOps.regWrite kst1.rf cond_bool
                         (Quanta.KOps.vBool (!decide (c = 0)))) cond_bool
                     = _
                exact regLookup_regWrite_self _ _ _
              have h_branch_evals_to :
                  Quanta.KOps.evalOp F kst_cast
                    (KernelOp.branch cond_bool [] []) = some kst_cast := by
                simp [Quanta.KOps.evalOp, h_lookup_cast, Quanta.KOps.evalOps]
                -- match on vBool b — both true and false arms evaluate to
                -- some kst_cast (since both elseOps and thenOps are []).
                cases h : !decide (c = 0) <;> rfl
              -- Now compose: cast + branch + postOps.
              have h_cast_branch_post :
                  Quanta.KOps.evalOps F kst1
                    (KernelOp.cast cond_bool cond .u32 .bool
                      :: KernelOp.branch cond_bool [] [] :: postOps)
                    = some kst' := by
                rw [Quanta.KOps.evalOps]
                rw [h_cast_max]
                simp [h_kst_cast_broke]
                rw [Quanta.KOps.evalOps]
                rw [h_branch_evals_to]
                simp [h_kst_cast_broke, h_post_max]
              exact (evalOps_append h1 h_kst1_ok).trans h_cast_branch_post
            · rw [← h_s_eq]; exact R_p
          | wI64 _ => simp at hw
          | wF32 _ => simp at hw
          | wF64 _ => simp at hw

/-- `wif _ :: rest` preservation, non-trivial fall-through thenBody,
    empty elseBody, thenBody doesn't mutate locals OR the stack
    (stack-effect zero — the canonical type-`[] → []` Rust pattern).

    Generalizes `cons_wif_trivialBodies` to non-empty thenBody. The
    extra hypothesis `then_falls_through` now records the
    stack-effect-zero property: both `s'_b.stack = s_b.stack`
    (lowering) AND `ws'_b.stack = ws_b.stack` (eval). These are what
    let the post-state's Refines lift cleanly across the snapshot/
    restore boundary. -/
theorem preservation_evalInstrs_cons_wif_noElse_fallthrough_noLocalSet
    (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (bt : Nat) (rest : List WasmInstr)
    (thenBody post : List WasmInstr)
    (h_split : splitAtElseOrEnd rest = some (thenBody, [], post))
    (then_preserves : ∀ {ws_b : WasmState} {s_b : LowerState}
        {kst_b : Quanta.KOps.State}
        (_R_b : Refines ws_b s_b kst_b layout)
        (_h_nb_b : ws_b.branchTarget = none)
        (_h_nh_b : ws_b.halted = false)
        (_h_nbk_b : kst_b.broke = false)
        {ws'_b : WasmState} {s'_b : LowerState} {bodyOps : List KernelOp}
        (_hw_b : evalInstrs bt ws_b thenBody = some ws'_b)
        (_hl_b : lowerInstrs bt (.wif :: frames) s_b thenBody = some (s'_b, bodyOps)),
      ∃ (kst'_b : Quanta.KOps.State) (F : Nat),
        evalOps F kst_b bodyOps = some kst'_b ∧
        Refines ws'_b s'_b kst'_b layout ∧
        BridgeClauses ws'_b kst'_b)
    -- Eval-side fall-through certificate (depends on actual evaluation).
    (then_falls_through : ∀ {ws_b : WasmState} {s_b : LowerState}
        {kst_b : Quanta.KOps.State} {ws'_b : WasmState} {s'_b : LowerState}
        {bodyOps : List KernelOp}
        (_R_b : Refines ws_b s_b kst_b layout)
        (_h_nb_b : ws_b.branchTarget = none)
        (_h_nh_b : ws_b.halted = false)
        (_h_nbk_b : kst_b.broke = false)
        (_hw_b : evalInstrs bt ws_b thenBody = some ws'_b)
        (_hl_b : lowerInstrs bt (.wif :: frames) s_b thenBody = some (s'_b, bodyOps)),
      ws'_b.branchTarget = none ∧ ws'_b.halted = false ∧
      ws'_b.locals = ws_b.locals ∧ ws'_b.stack = ws_b.stack ∧
      ws'_b.mem = ws_b.mem ∧
      s'_b.localReg = s_b.localReg ∧ s'_b.localTy = s_b.localTy ∧
      s'_b.stack = s_b.stack ∧ s'_b.bufferSlots = s_b.bufferSlots)
    -- Lowering-only structural invariants (no eval dependency).
    -- Needed for the c=0 case where eval doesn't run thenBody, but
    -- the lowering output s2's shape still matters for the post-state
    -- Refines lift.
    (then_lowering_preserves : ∀ {s_b s'_b : LowerState} {bodyOps : List KernelOp},
        lowerInstrs bt (.wif :: frames) s_b thenBody = some (s'_b, bodyOps) →
        s'_b.localReg = s_b.localReg ∧ s'_b.localTy = s_b.localTy ∧
        s'_b.stack = s_b.stack ∧ s'_b.bufferSlots = s_b.bufferSlots ∧
        s'_b.currentReg = s_b.currentReg ∧
        s_b.nextReg ≤ s'_b.nextReg)
    (post_preserves : ∀ {ws_p : WasmState} {s_p : LowerState}
        {kst_p : Quanta.KOps.State}
        (_R_p : Refines ws_p s_p kst_p layout)
        (_h_nb_p : ws_p.branchTarget = none)
        (_h_nh_p : ws_p.halted = false)
        (_h_nbk_p : kst_p.broke = false)
        {ws'_p : WasmState} {s'_p : LowerState} {postOps : List KernelOp}
        (_hw_p : evalInstrs bt ws_p post = some ws'_p)
        (_hl_p : lowerInstrs bt frames s_p post = some (s'_p, postOps)),
      ∃ (kst'_p : Quanta.KOps.State) (F : Nat),
        evalOps F kst_p postOps = some kst'_p ∧
        Refines ws'_p s'_p kst'_p layout ∧
        BridgeClauses ws'_p kst'_p)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs (bt + 1) ws (.wif 0 :: rest) = some ws')
    (hl : lowerInstrs (bt + 1) frames s (.wif 0 :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  -- Unfold lowerInstrs wif arm.
  simp only [lowerInstrs] at hl
  rw [h_split] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl
  cases hpop : s.popSym with
  | none => simp [hpop] at hl
  | some pop_pair =>
    rcases pop_pair with ⟨svCond, s0⟩
    simp [hpop] at hl
    cases hcommit : s0.commit svCond with
    | none => simp [hcommit] at hl
    | some commit_triple =>
      rcases commit_triple with ⟨cond, s1, opsCommit⟩
      simp [hcommit] at hl
      simp only [LowerState.alloc] at hl
      cases hlt : lowerInstrs bt (.wif :: frames)
          ({ s1 with nextReg := s1.nextReg + 1 } : LowerState) thenBody with
      | none => simp [hlt] at hl
      | some then_pair =>
        rcases then_pair with ⟨s2, thenOps⟩
        simp [hlt] at hl
        have h_else_nil : ∀ s_in : LowerState,
            lowerInstrs bt (.wif :: frames) s_in [] = some (s_in, []) := by
          intro s_in; simp [lowerInstrs]
        rw [h_else_nil] at hl
        simp only [Option.some_bind] at hl
        cases hlp : lowerInstrs bt frames
            ({ ({ s2 with
                    localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                    localTy  := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy,
                    currentReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).currentReg } : LowerState) with
                  localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                  localTy  := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy,
                  currentReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).currentReg } : LowerState)
            post with
        | none => simp [hlp] at hl
        | some post_pair =>
          rcases post_pair with ⟨s4, postOps⟩
          simp [hlp] at hl
          rcases hl with ⟨h_s_eq, h_ops_eq⟩
          -- Eval side: same plumbing as trivialBodies.
          simp only [evalInstrs] at hw
          have h_cond : (ws.halted || ws.branchTarget.isSome) = false := by
            rw [h_no_halt, h_no_branch]; rfl
          rw [h_cond] at hw
          simp only [Bool.false_eq_true, ↓reduceIte] at hw
          cases hpop_w : ws.pop with
          | none => simp [hpop_w] at hw
          | some pop_w =>
            rcases pop_w with ⟨vc, ws0⟩
            simp [hpop_w] at hw
            rw [h_split] at hw
            simp only at hw
            cases vc with
            | wI32 c =>
              simp only at hw
              have h_pop_facts : ws.stack = .wI32 c :: ws0.stack ∧
                                ws0 = { ws with stack := ws0.stack } := by
                rw [WasmState.pop] at hpop_w
                rcases hst : ws.stack with _ | ⟨v0, rest⟩
                · rw [hst] at hpop_w; simp at hpop_w
                · rw [hst] at hpop_w
                  simp at hpop_w
                  obtain ⟨hv0, hws0⟩ := hpop_w
                  subst hv0
                  have h_ws0_stack : ws0.stack = rest := by rw [← hws0]
                  refine ⟨?_, ?_⟩
                  · rw [h_ws0_stack]
                  · rw [h_ws0_stack]; exact hws0.symm
              obtain ⟨h_ws_stack, h_ws0_eq⟩ := h_pop_facts
              have h_ws0_nb : ws0.branchTarget = none := by
                rw [h_ws0_eq]; simp [h_no_branch]
              have h_ws0_nh : ws0.halted = false := by
                rw [h_ws0_eq]; simp [h_no_halt]
              obtain ⟨kst1, h_evalCommit, h_kst1_ok, h_lookup, _R_post,
                      _, _, _, _, _, _⟩ :=
                brIf_cond_pop_commit_correct_pub R h_ws_stack hpop hcommit h_kst_no_broke
              let cond_bool : Quanta.KOps.Reg := s1.nextReg
              let kst_cast : Quanta.KOps.State :=
                { kst1 with rf := Quanta.KOps.regWrite kst1.rf cond_bool
                                    (Quanta.KOps.vBool (!decide (c = 0))) }
              have h_kst_cast_broke : kst_cast.broke = false := h_kst1_ok
              have h_s1_stack : s1.stack = s0.stack := commit_preserves_stack hcommit
              have h_s1_eq : ({ s1 with stack := s0.stack } : LowerState) = s1 := by
                cases s1 with
                | mk nr st lr lt bs cr =>
                  simp at h_s1_stack
                  rw [h_s1_stack]
              have R_at_s1 : Refines ws0 s1 kst1 layout := by
                rw [h_ws0_eq, ← h_s1_eq]
                exact _R_post
              have R_at_cast : Refines ws0
                  { s1 with nextReg := s1.nextReg + 1 } kst_cast layout := by
                refine ⟨?_, ?_, ?_, ?_, R_at_s1.injLocals, R_at_s1.heapRefines, ?_, ?_⟩
                · refine ⟨?_, ?_⟩
                  · show ws0.stack.length = s1.stack.length
                    exact R_at_s1.stk.left
                  · intro i v hv
                    obtain ⟨svi, hsv_get, henc⟩ := R_at_s1.stk.right i v hv
                    have hsv_in : svi ∈ s1.stack := List.mem_of_get? hsv_get
                    refine ⟨svi, hsv_get, ?_⟩
                    apply WasmValue.encodes_preserved_of_fresh _ henc
                    intro r hr
                    exact R_at_s1.fresh.left svi hsv_in r hr
                · intro i r hfind v hv
                  have henc := R_at_s1.locs i r hfind v hv
                  have hr_lt : r < s1.nextReg := by
                    have hpair : (i, r) ∈ s1.localReg := List.mem_of_find?_eq_some hfind
                    exact R_at_s1.fresh.right (i, r) hpair
                  apply WasmValue.encodes_preserved_of_fresh _ henc
                  intro r' hr'
                  simp [SymVal.regs] at hr'
                  subst hr'; exact hr_lt
                · refine ⟨?_, ?_⟩
                  · intro sv hsv r hr
                    show r < s1.nextReg + 1
                    exact Nat.lt_succ_of_lt (R_at_s1.fresh.left sv hsv r hr)
                  · intro ir hir
                    show ir.snd < s1.nextReg + 1
                    exact Nat.lt_succ_of_lt (R_at_s1.fresh.right ir hir)
                · intro ir hir sv hsv
                  exact R_at_s1.aliasFree ir hir sv hsv
                · -- CurrentRegRefines: s_cast.currentReg = s1.currentReg; lift past cast write.
                  show CurrentRegRefines layout _ s1.currentReg _
                  exact CurrentRegRefines_preserved_fresh R_at_s1.currentReg R_at_s1.freshCurrent _
                · -- FreshCurrent: nextReg bumps by 1.
                  intro ir hir
                  show ir.snd < s1.nextReg + 1
                  exact Nat.lt_succ_of_lt (R_at_s1.freshCurrent ir hir)
              -- s_cast normalization: the literal record in hlp is
              -- { s2 with localReg := s_cast.localReg, localTy := s_cast.localTy,
              --          currentReg := s_cast.currentReg } twice over (Stage 3).
              -- Reduce to a single update.
              have h_s3r_form :
                  ({ ({ s2 with
                          localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                          localTy  := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy,
                          currentReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).currentReg } : LowerState) with
                        localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                        localTy  := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy,
                        currentReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).currentReg } : LowerState)
                  = { s2 with localReg := s1.localReg, localTy := s1.localTy,
                              currentReg := s1.currentReg } := by
                cases s2 with
                | mk nr st lr lt bs cr => rfl
              rw [h_s3r_form] at hlp
              by_cases hc : c = 0
              · -- c = 0: WASM picks elseBody = []. ws_after_body = ws0.
                -- IR: branch picks elseOps = [] → kst_cast. Then postOps → kst'.
                simp only [hc, ↓reduceIte] at hw
                simp only [evalInstrs] at hw
                rw [h_ws0_nb] at hw
                simp only at hw
                -- Use the lowering-only invariants to learn s2's shape.
                obtain ⟨h_s2_lr, h_s2_lt, h_s2_stack, h_s2_bs, h_s2_cr, h_s2_nr⟩ :=
                  then_lowering_preserves hlt
                -- The state passed to post equals s2 (after restore is idempotent).
                have h_s2_restored_eq :
                    ({ s2 with localReg := s1.localReg,
                                localTy := s1.localTy,
                                currentReg := s1.currentReg } : LowerState) = s2 := by
                  have h_lr_eq : s1.localReg =
                      ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg := rfl
                  have h_lt_eq : s1.localTy =
                      ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy := rfl
                  have h_cr_eq : s1.currentReg =
                      ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).currentReg := rfl
                  rw [h_lr_eq, h_lt_eq, h_cr_eq, ← h_s2_lr, ← h_s2_lt, ← h_s2_cr]
                rw [h_s2_restored_eq] at hlp
                -- Build Refines ws0 s2 kst_cast layout by lifting R_at_cast.
                -- s2 differs from s_cast = { s1 with nextReg := s1.nextReg + 1 }
                -- in: nextReg (≥), bufferSlots (same), localReg (same),
                --     localTy (same), stack (same).
                -- s2.stack = s1.stack (from lowering-preserves transitivity:
                -- s_cast.stack = s1.stack already, and s2.stack = s_cast.stack).
                have h_s2_stack_eq_s1 : s2.stack = s1.stack := by
                  -- h_s2_stack: s2.stack = s_cast.stack. s_cast.stack = s1.stack.
                  rw [h_s2_stack]
                have h_s2_bs_eq_s1 : s2.bufferSlots = s1.bufferSlots := by
                  rw [h_s2_bs]
                have R_at_s2 : Refines ws0 s2 kst_cast layout := by
                  -- Same kst_cast as R_at_cast — no regfile change.
                  -- s2 differs from s_cast only in (possibly) nextReg
                  -- and bufferSlots, but bufferSlots is preserved by
                  -- then_lowering_preserves; stack/localReg/localTy
                  -- also preserved. Constructor-by-constructor lift.
                  refine ⟨?_, ?_, ?_, ?_, ?_, R_at_cast.heapRefines, R_at_cast.currentReg, R_at_cast.freshCurrent⟩
                  · refine ⟨?_, ?_⟩
                    · show ws0.stack.length = s2.stack.length
                      rw [h_s2_stack_eq_s1]
                      exact R_at_cast.stk.left
                    · intro i v hv
                      obtain ⟨svi, hsv_get, henc⟩ := R_at_cast.stk.right i v hv
                      have hsv_get_s2 : s2.stack.get? i = some svi := by
                        rw [h_s2_stack_eq_s1]; exact hsv_get
                      exact ⟨svi, hsv_get_s2, henc⟩
                  · -- LocalsRefines.
                    intro i r hfind v hv
                    have hfind_cast : ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg.find?
                        (fun p => p.fst = i) = some (i, r) := by
                      rw [show ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg = s2.localReg
                          from h_s2_lr.symm]
                      exact hfind
                    exact R_at_cast.locs i r hfind_cast v hv
                  · -- Fresh: regs bounded by s2.nextReg ≥ s1.nextReg + 1.
                    refine ⟨?_, ?_⟩
                    · intro sv hsv r hr
                      have hsv_in_cast : sv ∈
                          ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).stack := by
                        rw [show ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).stack = s2.stack
                            from h_s2_stack.symm]
                        exact hsv
                      have h_r_lt : r < s1.nextReg + 1 :=
                        R_at_cast.fresh.left sv hsv_in_cast r hr
                      exact Nat.lt_of_lt_of_le h_r_lt h_s2_nr
                    · intro ir hir
                      have hir_cast : ir ∈
                          ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg := by
                        rw [show ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg = s2.localReg
                            from h_s2_lr.symm]
                        exact hir
                      have h_r_lt : ir.snd < s1.nextReg + 1 :=
                        R_at_cast.fresh.right ir hir_cast
                      exact Nat.lt_of_lt_of_le h_r_lt h_s2_nr
                  · -- AliasFree.
                    intro ir hir sv hsv
                    have hir_cast : ir ∈
                        ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg := by
                      rw [show ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg = s2.localReg
                          from h_s2_lr.symm]
                      exact hir
                    have hsv_in_cast : sv ∈
                        ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).stack := by
                      rw [show ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).stack = s2.stack
                          from h_s2_stack.symm]
                      exact hsv
                    exact R_at_cast.aliasFree ir hir_cast sv hsv_in_cast
                  · -- InjectiveLocals.
                    intro p q hp hq
                    have hp_cast : p ∈
                        ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg := by
                      rw [show ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg = s2.localReg
                          from h_s2_lr.symm]
                      exact hp
                    have hq_cast : q ∈
                        ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg := by
                      rw [show ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg = s2.localReg
                          from h_s2_lr.symm]
                      exact hq
                    exact R_at_cast.injLocals p q hp_cast hq_cast
                obtain ⟨kst', F_p, h_ev_p, R_p, h_bridge_p⟩ :=
                  post_preserves R_at_s2 h_ws0_nb h_ws0_nh h_kst_cast_broke hw hlp
                -- IR composition: opsCommit (→ kst1) + cast (→ kst_cast) +
                -- branch picks elseOps = [] (cond_bool = vBool false) → kst_cast.
                -- postOps from kst_cast → kst'.
                let F : Nat := max F_p 1
                refine ⟨kst', F, ?_, ?_, h_bridge_p⟩
                · rw [← h_ops_eq]
                  have h1 : evalOps F kst opsCommit = some kst1 :=
                    evalOps_fuel_mono (Nat.zero_le _) h_evalCommit
                  have h_post_max : evalOps F kst_cast postOps = some kst' :=
                    evalOps_fuel_mono (Nat.le_max_left _ _) h_ev_p
                  have h_cast_max : Quanta.KOps.evalOp F kst1
                      (KernelOp.cast cond_bool cond Quanta.KOps.Scalar.u32 Quanta.KOps.Scalar.bool)
                      = some kst_cast := by
                    simp [Quanta.KOps.evalOp, h_lookup, Quanta.KOps.evalCast, kst_cast]
                  have h_lookup_cast :
                      Quanta.KOps.regLookup kst_cast.rf cond_bool
                        = some (Quanta.KOps.Value.vBool (!decide (c = 0))) := by
                    show Quanta.KOps.regLookup
                           (Quanta.KOps.regWrite kst1.rf cond_bool
                             (Quanta.KOps.vBool (!decide (c = 0)))) cond_bool
                         = _
                    exact regLookup_regWrite_self _ _ _
                  -- branch reads vBool false (c=0), picks elseOps = []
                  -- → evalOps F kst_cast [] = some kst_cast.
                  have h_branch_evals_to :
                      Quanta.KOps.evalOp F kst_cast
                        (KernelOp.branch cond_bool thenOps []) = some kst_cast := by
                    simp [Quanta.KOps.evalOp, h_lookup_cast, hc, Quanta.KOps.evalOps]
                  have h_cast_branch_post :
                      Quanta.KOps.evalOps F kst1
                        (KernelOp.cast cond_bool cond .u32 .bool
                          :: KernelOp.branch cond_bool thenOps [] :: postOps)
                        = some kst' := by
                    rw [Quanta.KOps.evalOps]
                    rw [h_cast_max]
                    simp [h_kst_cast_broke]
                    rw [Quanta.KOps.evalOps]
                    rw [h_branch_evals_to]
                    simp [h_kst_cast_broke, h_post_max]
                  exact (evalOps_append h1 h_kst1_ok).trans h_cast_branch_post
                · rw [← h_s_eq]; exact R_p
              · -- c ≠ 0: WASM picks thenBody. Apply then_preserves + then_falls_through.
                simp only [hc, ↓reduceIte] at hw
                cases h_eb : evalInstrs bt ws0 thenBody with
                | none => simp [h_eb] at hw
                | some ws_ab =>
                  simp [h_eb] at hw
                  -- Apply then_preserves at R_at_cast.
                  obtain ⟨kst_ab, F_b, h_ev_b, R_b, h_bridge_b⟩ :=
                    then_preserves R_at_cast h_ws0_nb h_ws0_nh h_kst_cast_broke
                      h_eb hlt
                  obtain ⟨h_ab_nb, h_ab_nh, h_ab_locals, h_ab_stack, h_ab_mem,
                          h_s2_lr, h_s2_lt, h_s2_stack, h_s2_bs⟩ :=
                    then_falls_through R_at_cast h_ws0_nb h_ws0_nh h_kst_cast_broke
                      h_eb hlt
                  rw [h_ab_nb] at hw
                  simp only at hw
                  -- hw: evalInstrs bt ws_ab post = some ws'.
                  have h_ab_broke : kst_ab.broke = false := h_bridge_b.right h_ab_nb
                  -- Build Refines ws_ab { s2 with localReg := s1.localReg,
                  --   localTy := s1.localTy } kst_ab layout. From R_b and
                  -- then_falls_through, s2.localReg = s_cast.localReg = s1.localReg,
                  -- so the restored state equals s2. Refines lifts directly.
                  have h_s2_restored_eq :
                      ({ s2 with localReg := s1.localReg,
                                  localTy := s1.localTy } : LowerState) = s2 := by
                    have h_lr_eq : s1.localReg =
                        ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg := rfl
                    have h_lt_eq : s1.localTy =
                        ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy := rfl
                    rw [h_lr_eq, h_lt_eq, ← h_s2_lr, ← h_s2_lt]
                  rw [h_s2_restored_eq] at hlp
                  obtain ⟨kst', F_p, h_ev_p, R_p, h_bridge_p⟩ :=
                    post_preserves R_b h_ab_nb h_ab_nh h_ab_broke hw hlp
                  -- Compose IR: opsCommit (→ kst1) + cast (→ kst_cast) +
                  -- branch picks thenOps (cond_bool = vBool true since c ≠ 0)
                  -- → kst_ab. Then [] elseOps unused; postOps from kst_ab → kst'.
                  let F : Nat := max (max F_b F_p) 1
                  refine ⟨kst', F, ?_, ?_, h_bridge_p⟩
                  · rw [← h_ops_eq]
                    have h1 : evalOps F kst opsCommit = some kst1 :=
                      evalOps_fuel_mono (Nat.zero_le _) h_evalCommit
                    have h_then_max : evalOps F kst_cast thenOps = some kst_ab :=
                      evalOps_fuel_mono
                        (Nat.le_trans (Nat.le_max_left _ _) (Nat.le_max_left _ _)) h_ev_b
                    have h_post_max : evalOps F kst_ab postOps = some kst' :=
                      evalOps_fuel_mono
                        (Nat.le_trans (Nat.le_max_right _ _) (Nat.le_max_left _ _)) h_ev_p
                    have h_cast_max : Quanta.KOps.evalOp F kst1
                        (KernelOp.cast cond_bool cond Quanta.KOps.Scalar.u32 Quanta.KOps.Scalar.bool)
                        = some kst_cast := by
                      simp [Quanta.KOps.evalOp, h_lookup, Quanta.KOps.evalCast, kst_cast]
                    have h_lookup_cast :
                        Quanta.KOps.regLookup kst_cast.rf cond_bool
                          = some (Quanta.KOps.Value.vBool (!decide (c = 0))) := by
                      show Quanta.KOps.regLookup
                             (Quanta.KOps.regWrite kst1.rf cond_bool
                               (Quanta.KOps.vBool (!decide (c = 0)))) cond_bool
                           = _
                      exact regLookup_regWrite_self _ _ _
                    have h_branch_evals_to :
                        Quanta.KOps.evalOp F kst_cast
                          (KernelOp.branch cond_bool thenOps []) = some kst_ab := by
                      simp [Quanta.KOps.evalOp, h_lookup_cast, hc, h_then_max]
                    have h_cast_branch_post :
                        Quanta.KOps.evalOps F kst1
                          (KernelOp.cast cond_bool cond .u32 .bool
                            :: KernelOp.branch cond_bool thenOps [] :: postOps)
                          = some kst' := by
                      rw [Quanta.KOps.evalOps]
                      rw [h_cast_max]
                      simp [h_kst_cast_broke]
                      rw [Quanta.KOps.evalOps]
                      rw [h_branch_evals_to]
                      simp [h_ab_broke, h_post_max]
                    exact (evalOps_append h1 h_kst1_ok).trans h_cast_branch_post
                  · rw [← h_s_eq]; exact R_p
            | wI64 _ => simp at hw
            | wF32 _ => simp at hw
            | wF64 _ => simp at hw

/-- `wif _ :: rest` preservation, both bodies non-empty + fall-
    through + no-localSet. Full symmetric variant — closes the
    canonical Rust `if cond { ... } else { ... }` shape under
    the no-mutation restriction.

    Five caller-supplied IHs:
    - `then_preserves` / `else_preserves`: standard recursive
      bridge IHs for thenBody / elseBody
    - `then_falls_through` / `else_falls_through`: eval-side
      fall-through (branchTarget = none, halted = false,
      stack-effect zero, no localSet)
    - `then_lowering_preserves` / `else_lowering_preserves`:
      lowering-only structural invariants — needed for the cond
      arm where eval doesn't run the corresponding body but the
      lowering output state's shape still matters for the
      post-state Refines lift. -/
theorem preservation_evalInstrs_cons_wif_fallthrough_noLocalSet
    (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (bt : Nat) (rest : List WasmInstr)
    (thenBody elseBody post : List WasmInstr)
    (h_split : splitAtElseOrEnd rest = some (thenBody, elseBody, post))
    (then_preserves : ∀ {ws_b : WasmState} {s_b : LowerState}
        {kst_b : Quanta.KOps.State}
        (_R_b : Refines ws_b s_b kst_b layout)
        (_h_nb_b : ws_b.branchTarget = none)
        (_h_nh_b : ws_b.halted = false)
        (_h_nbk_b : kst_b.broke = false)
        {ws'_b : WasmState} {s'_b : LowerState} {bodyOps : List KernelOp}
        (_hw_b : evalInstrs bt ws_b thenBody = some ws'_b)
        (_hl_b : lowerInstrs bt (.wif :: frames) s_b thenBody = some (s'_b, bodyOps)),
      ∃ (kst'_b : Quanta.KOps.State) (F : Nat),
        evalOps F kst_b bodyOps = some kst'_b ∧
        Refines ws'_b s'_b kst'_b layout ∧
        BridgeClauses ws'_b kst'_b)
    (then_falls_through : ∀ {ws_b : WasmState} {s_b : LowerState}
        {kst_b : Quanta.KOps.State} {ws'_b : WasmState} {s'_b : LowerState}
        {bodyOps : List KernelOp}
        (_R_b : Refines ws_b s_b kst_b layout)
        (_h_nb_b : ws_b.branchTarget = none)
        (_h_nh_b : ws_b.halted = false)
        (_h_nbk_b : kst_b.broke = false)
        (_hw_b : evalInstrs bt ws_b thenBody = some ws'_b)
        (_hl_b : lowerInstrs bt (.wif :: frames) s_b thenBody = some (s'_b, bodyOps)),
      ws'_b.branchTarget = none ∧ ws'_b.halted = false ∧
      ws'_b.locals = ws_b.locals ∧ ws'_b.stack = ws_b.stack ∧
      ws'_b.mem = ws_b.mem ∧
      s'_b.localReg = s_b.localReg ∧ s'_b.localTy = s_b.localTy ∧
      s'_b.stack = s_b.stack ∧ s'_b.bufferSlots = s_b.bufferSlots)
    (then_lowering_preserves : ∀ {s_b s'_b : LowerState} {bodyOps : List KernelOp},
        lowerInstrs bt (.wif :: frames) s_b thenBody = some (s'_b, bodyOps) →
        s'_b.localReg = s_b.localReg ∧ s'_b.localTy = s_b.localTy ∧
        s'_b.stack = s_b.stack ∧ s'_b.bufferSlots = s_b.bufferSlots ∧
        s_b.nextReg ≤ s'_b.nextReg)
    (else_preserves : ∀ {ws_b : WasmState} {s_b : LowerState}
        {kst_b : Quanta.KOps.State}
        (_R_b : Refines ws_b s_b kst_b layout)
        (_h_nb_b : ws_b.branchTarget = none)
        (_h_nh_b : ws_b.halted = false)
        (_h_nbk_b : kst_b.broke = false)
        {ws'_b : WasmState} {s'_b : LowerState} {bodyOps : List KernelOp}
        (_hw_b : evalInstrs bt ws_b elseBody = some ws'_b)
        (_hl_b : lowerInstrs bt (.wif :: frames) s_b elseBody = some (s'_b, bodyOps)),
      ∃ (kst'_b : Quanta.KOps.State) (F : Nat),
        evalOps F kst_b bodyOps = some kst'_b ∧
        Refines ws'_b s'_b kst'_b layout ∧
        BridgeClauses ws'_b kst'_b)
    (else_falls_through : ∀ {ws_b : WasmState} {s_b : LowerState}
        {kst_b : Quanta.KOps.State} {ws'_b : WasmState} {s'_b : LowerState}
        {bodyOps : List KernelOp}
        (_R_b : Refines ws_b s_b kst_b layout)
        (_h_nb_b : ws_b.branchTarget = none)
        (_h_nh_b : ws_b.halted = false)
        (_h_nbk_b : kst_b.broke = false)
        (_hw_b : evalInstrs bt ws_b elseBody = some ws'_b)
        (_hl_b : lowerInstrs bt (.wif :: frames) s_b elseBody = some (s'_b, bodyOps)),
      ws'_b.branchTarget = none ∧ ws'_b.halted = false ∧
      ws'_b.locals = ws_b.locals ∧ ws'_b.stack = ws_b.stack ∧
      ws'_b.mem = ws_b.mem ∧
      s'_b.localReg = s_b.localReg ∧ s'_b.localTy = s_b.localTy ∧
      s'_b.stack = s_b.stack ∧ s'_b.bufferSlots = s_b.bufferSlots)
    (else_lowering_preserves : ∀ {s_b s'_b : LowerState} {bodyOps : List KernelOp},
        lowerInstrs bt (.wif :: frames) s_b elseBody = some (s'_b, bodyOps) →
        s'_b.localReg = s_b.localReg ∧ s'_b.localTy = s_b.localTy ∧
        s'_b.stack = s_b.stack ∧ s'_b.bufferSlots = s_b.bufferSlots ∧
        s_b.nextReg ≤ s'_b.nextReg)
    (post_preserves : ∀ {ws_p : WasmState} {s_p : LowerState}
        {kst_p : Quanta.KOps.State}
        (_R_p : Refines ws_p s_p kst_p layout)
        (_h_nb_p : ws_p.branchTarget = none)
        (_h_nh_p : ws_p.halted = false)
        (_h_nbk_p : kst_p.broke = false)
        {ws'_p : WasmState} {s'_p : LowerState} {postOps : List KernelOp}
        (_hw_p : evalInstrs bt ws_p post = some ws'_p)
        (_hl_p : lowerInstrs bt frames s_p post = some (s'_p, postOps)),
      ∃ (kst'_p : Quanta.KOps.State) (F : Nat),
        evalOps F kst_p postOps = some kst'_p ∧
        Refines ws'_p s'_p kst'_p layout ∧
        BridgeClauses ws'_p kst'_p)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs (bt + 1) ws (.wif 0 :: rest) = some ws')
    (hl : lowerInstrs (bt + 1) frames s (.wif 0 :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  -- Unfold lowerInstrs wif arm.
  simp only [lowerInstrs] at hl
  rw [h_split] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl
  cases hpop : s.popSym with
  | none => simp [hpop] at hl
  | some pop_pair =>
    rcases pop_pair with ⟨svCond, s0⟩
    simp [hpop] at hl
    cases hcommit : s0.commit svCond with
    | none => simp [hcommit] at hl
    | some commit_triple =>
      rcases commit_triple with ⟨cond, s1, opsCommit⟩
      simp [hcommit] at hl
      simp only [LowerState.alloc] at hl
      cases hlt : lowerInstrs bt (.wif :: frames)
          ({ s1 with nextReg := s1.nextReg + 1 } : LowerState) thenBody with
      | none => simp [hlt] at hl
      | some then_pair =>
        rcases then_pair with ⟨s2, thenOps⟩
        simp [hlt] at hl
        -- After thenBody, restore localReg/localTy. By then_lowering_preserves
        -- on s2, the restore is idempotent on those fields (s2.localReg already
        -- equals s_cast.localReg). The restored state thus equals s2.
        -- Case on elseBody's lowering.
        obtain ⟨h_s2_lr, h_s2_lt, h_s2_stack, h_s2_bs, h_s2_nr⟩ :=
          then_lowering_preserves hlt
        have h_s2_restored_eq :
            ({ s2 with localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                       localTy  := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy } : LowerState)
            = s2 := by
          have h_lr_eq : ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg = s2.localReg :=
            h_s2_lr.symm
          have h_lt_eq : ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy = s2.localTy :=
            h_s2_lt.symm
          rw [h_lr_eq, h_lt_eq]
        rw [h_s2_restored_eq] at hl
        cases hle : lowerInstrs bt (.wif :: frames) s2 elseBody with
        | none => simp [hle] at hl
        | some else_pair =>
          rcases else_pair with ⟨s3, elseOps⟩
          simp [hle] at hl
          -- After elseBody, restore again to s_cast snapshot.
          obtain ⟨h_s3_lr, h_s3_lt, h_s3_stack, h_s3_bs, h_s3_nr⟩ :=
            else_lowering_preserves hle
          -- s3.localReg = s2.localReg = s_cast.localReg, so the restore is idempotent.
          have h_s3_restored_eq :
              ({ s3 with localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                         localTy  := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy } : LowerState)
              = s3 := by
            have h_s3_lr_s1 : s3.localReg = ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg := by
              rw [h_s3_lr, h_s2_lr]
            have h_s3_lt_s1 : s3.localTy = ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy := by
              rw [h_s3_lt, h_s2_lt]
            rw [← h_s3_lr_s1, ← h_s3_lt_s1]
          rw [h_s3_restored_eq] at hl
          cases hlp : lowerInstrs bt frames s3 post with
          | none => simp [hlp] at hl
          | some post_pair =>
            rcases post_pair with ⟨s4, postOps⟩
            simp [hlp] at hl
            rcases hl with ⟨h_s_eq, h_ops_eq⟩
            -- Eval side: same plumbing through pop.
            simp only [evalInstrs] at hw
            have h_cond : (ws.halted || ws.branchTarget.isSome) = false := by
              rw [h_no_halt, h_no_branch]; rfl
            rw [h_cond] at hw
            simp only [Bool.false_eq_true, ↓reduceIte] at hw
            cases hpop_w : ws.pop with
            | none => simp [hpop_w] at hw
            | some pop_w =>
              rcases pop_w with ⟨vc, ws0⟩
              simp [hpop_w] at hw
              rw [h_split] at hw
              simp only at hw
              cases vc with
              | wI32 c =>
                simp only at hw
                have h_pop_facts : ws.stack = .wI32 c :: ws0.stack ∧
                                  ws0 = { ws with stack := ws0.stack } := by
                  rw [WasmState.pop] at hpop_w
                  rcases hst : ws.stack with _ | ⟨v0, rest⟩
                  · rw [hst] at hpop_w; simp at hpop_w
                  · rw [hst] at hpop_w
                    simp at hpop_w
                    obtain ⟨hv0, hws0⟩ := hpop_w
                    subst hv0
                    have h_ws0_stack : ws0.stack = rest := by rw [← hws0]
                    refine ⟨?_, ?_⟩
                    · rw [h_ws0_stack]
                    · rw [h_ws0_stack]; exact hws0.symm
                obtain ⟨h_ws_stack, h_ws0_eq⟩ := h_pop_facts
                have h_ws0_nb : ws0.branchTarget = none := by
                  rw [h_ws0_eq]; simp [h_no_branch]
                have h_ws0_nh : ws0.halted = false := by
                  rw [h_ws0_eq]; simp [h_no_halt]
                obtain ⟨kst1, h_evalCommit, h_kst1_ok, h_lookup, _R_post,
                        _, _, _, _, _, _⟩ :=
                  brIf_cond_pop_commit_correct_pub R h_ws_stack hpop hcommit h_kst_no_broke
                let cond_bool : Quanta.KOps.Reg := s1.nextReg
                let kst_cast : Quanta.KOps.State :=
                  { kst1 with rf := Quanta.KOps.regWrite kst1.rf cond_bool
                                      (Quanta.KOps.vBool (!decide (c = 0))) }
                have h_kst_cast_broke : kst_cast.broke = false := h_kst1_ok
                have h_s1_stack : s1.stack = s0.stack := commit_preserves_stack hcommit
                have h_s1_eq : ({ s1 with stack := s0.stack } : LowerState) = s1 := by
                  cases s1 with
                  | mk nr st lr lt bs cr =>
                    simp at h_s1_stack
                    rw [h_s1_stack]
                have R_at_s1 : Refines ws0 s1 kst1 layout := by
                  rw [h_ws0_eq, ← h_s1_eq]
                  exact _R_post
                have R_at_cast : Refines ws0
                    { s1 with nextReg := s1.nextReg + 1 } kst_cast layout := by
                  refine ⟨?_, ?_, ?_, ?_, R_at_s1.injLocals, R_at_s1.heapRefines⟩
                  · refine ⟨?_, ?_⟩
                    · show ws0.stack.length = s1.stack.length
                      exact R_at_s1.stk.left
                    · intro i v hv
                      obtain ⟨svi, hsv_get, henc⟩ := R_at_s1.stk.right i v hv
                      have hsv_in : svi ∈ s1.stack := List.mem_of_get? hsv_get
                      refine ⟨svi, hsv_get, ?_⟩
                      apply WasmValue.encodes_preserved_of_fresh _ henc
                      intro r hr
                      exact R_at_s1.fresh.left svi hsv_in r hr
                  · intro i r hfind v hv
                    have henc := R_at_s1.locs i r hfind v hv
                    have hr_lt : r < s1.nextReg := by
                      have hpair : (i, r) ∈ s1.localReg := List.mem_of_find?_eq_some hfind
                      exact R_at_s1.fresh.right (i, r) hpair
                    apply WasmValue.encodes_preserved_of_fresh _ henc
                    intro r' hr'
                    simp [SymVal.regs] at hr'
                    subst hr'; exact hr_lt
                  · refine ⟨?_, ?_⟩
                    · intro sv hsv r hr
                      show r < s1.nextReg + 1
                      exact Nat.lt_succ_of_lt (R_at_s1.fresh.left sv hsv r hr)
                    · intro ir hir
                      show ir.snd < s1.nextReg + 1
                      exact Nat.lt_succ_of_lt (R_at_s1.fresh.right ir hir)
                  · intro ir hir sv hsv
                    exact R_at_s1.aliasFree ir hir sv hsv
                -- Lifting helper: given Refines ws_target s_x kst_cast,
                -- with s_x's stack = s_cast.stack, localReg = s_cast.localReg,
                -- localTy = s_cast.localTy, bufferSlots = s_cast.bufferSlots,
                -- and s_cast.nextReg ≤ s_x.nextReg, lift R_at_cast to s_x.
                -- Used for both c=0 (target s3) and c≠0 (target s3 from s2).
                have R_lift : ∀ (s_x : LowerState),
                    s_x.stack = ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).stack →
                    s_x.localReg = ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg →
                    ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).nextReg ≤ s_x.nextReg →
                    Refines ws0 s_x kst_cast layout := by
                  intro s_x h_stk h_lr h_nr
                  refine ⟨?_, ?_, ?_, ?_, ?_, R_at_cast.heapRefines, R_at_cast.currentReg, R_at_cast.freshCurrent⟩
                  · refine ⟨?_, ?_⟩
                    · show ws0.stack.length = s_x.stack.length
                      rw [h_stk]; exact R_at_cast.stk.left
                    · intro i v hv
                      obtain ⟨svi, hsv_get, henc⟩ := R_at_cast.stk.right i v hv
                      have hsv_get_x : s_x.stack.get? i = some svi := by
                        rw [h_stk]; exact hsv_get
                      exact ⟨svi, hsv_get_x, henc⟩
                  · intro i r hfind v hv
                    have hfind_cast :
                        ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg.find?
                          (fun p => p.fst = i) = some (i, r) := by
                      rw [← h_lr]; exact hfind
                    exact R_at_cast.locs i r hfind_cast v hv
                  · refine ⟨?_, ?_⟩
                    · intro sv hsv r hr
                      have hsv_cast : sv ∈
                          ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).stack := by
                        rw [← h_stk]; exact hsv
                      exact Nat.lt_of_lt_of_le (R_at_cast.fresh.left sv hsv_cast r hr) h_nr
                    · intro ir hir
                      have hir_cast : ir ∈
                          ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg := by
                        rw [← h_lr]; exact hir
                      exact Nat.lt_of_lt_of_le (R_at_cast.fresh.right ir hir_cast) h_nr
                  · intro ir hir sv hsv
                    have hir_cast : ir ∈
                        ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg := by
                      rw [← h_lr]; exact hir
                    have hsv_cast : sv ∈
                        ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).stack := by
                      rw [← h_stk]; exact hsv
                    exact R_at_cast.aliasFree ir hir_cast sv hsv_cast
                  · intro p q hp hq
                    have hp_cast : p ∈
                        ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg := by
                      rw [← h_lr]; exact hp
                    have hq_cast : q ∈
                        ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg := by
                      rw [← h_lr]; exact hq
                    exact R_at_cast.injLocals p q hp_cast hq_cast
                -- Combined preservation facts for s3 (relative to s_cast).
                have h_s3_stack_cast :
                    s3.stack = ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).stack := by
                  rw [h_s3_stack, h_s2_stack]
                have h_s3_lr_cast :
                    s3.localReg = ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg := by
                  rw [h_s3_lr, h_s2_lr]
                have h_s3_nr_cast :
                    ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).nextReg ≤ s3.nextReg :=
                  Nat.le_trans h_s2_nr h_s3_nr
                by_cases hc : c = 0
                · -- c = 0: WASM picks elseBody, eval runs it.
                  simp only [hc, ↓reduceIte] at hw
                  cases h_eb : evalInstrs bt ws0 elseBody with
                  | none => simp [h_eb] at hw
                  | some ws_ab =>
                    simp [h_eb] at hw
                    -- Need Refines ws0 s2 kst_cast layout for else_preserves.
                    -- s2 has localReg = s_cast.localReg (from then_lowering),
                    -- stack = s_cast.stack, nextReg ≥ s_cast.nextReg. Lift R_at_cast.
                    have R_at_s2 : Refines ws0 s2 kst_cast layout := by
                      apply R_lift s2 h_s2_stack h_s2_lr h_s2_nr
                    obtain ⟨kst_ab, F_b, h_ev_b, R_b, h_bridge_b⟩ :=
                      else_preserves R_at_s2 h_ws0_nb h_ws0_nh h_kst_cast_broke
                        h_eb hle
                    obtain ⟨h_ab_nb, h_ab_nh, _, _, _, _, _, _, _⟩ :=
                      else_falls_through R_at_s2 h_ws0_nb h_ws0_nh h_kst_cast_broke
                        h_eb hle
                    rw [h_ab_nb] at hw
                    simp only at hw
                    -- hw: evalInstrs bt ws_ab post = some ws'.
                    have h_ab_broke : kst_ab.broke = false := h_bridge_b.right h_ab_nb
                    obtain ⟨kst', F_p, h_ev_p, R_p, h_bridge_p⟩ :=
                      post_preserves R_b h_ab_nb h_ab_nh h_ab_broke hw hlp
                    -- IR composition.
                    let F : Nat := max (max F_b F_p) 1
                    refine ⟨kst', F, ?_, ?_, h_bridge_p⟩
                    · rw [← h_ops_eq]
                      have h1 : evalOps F kst opsCommit = some kst1 :=
                        evalOps_fuel_mono (Nat.zero_le _) h_evalCommit
                      have h_else_max : evalOps F kst_cast elseOps = some kst_ab :=
                        evalOps_fuel_mono
                          (Nat.le_trans (Nat.le_max_left _ _) (Nat.le_max_left _ _)) h_ev_b
                      have h_post_max : evalOps F kst_ab postOps = some kst' :=
                        evalOps_fuel_mono
                          (Nat.le_trans (Nat.le_max_right _ _) (Nat.le_max_left _ _)) h_ev_p
                      have h_cast_max : Quanta.KOps.evalOp F kst1
                          (KernelOp.cast cond_bool cond Quanta.KOps.Scalar.u32 Quanta.KOps.Scalar.bool)
                          = some kst_cast := by
                        simp [Quanta.KOps.evalOp, h_lookup, Quanta.KOps.evalCast, kst_cast]
                      have h_lookup_cast :
                          Quanta.KOps.regLookup kst_cast.rf cond_bool
                            = some (Quanta.KOps.Value.vBool (!decide (c = 0))) := by
                        show Quanta.KOps.regLookup
                               (Quanta.KOps.regWrite kst1.rf cond_bool
                                 (Quanta.KOps.vBool (!decide (c = 0)))) cond_bool
                             = _
                        exact regLookup_regWrite_self _ _ _
                      -- c=0 → cond_bool = vBool false → branch picks elseOps.
                      have h_branch_evals_to :
                          Quanta.KOps.evalOp F kst_cast
                            (KernelOp.branch cond_bool thenOps elseOps) = some kst_ab := by
                        simp [Quanta.KOps.evalOp, h_lookup_cast, hc, h_else_max]
                      have h_cast_branch_post :
                          Quanta.KOps.evalOps F kst1
                            (KernelOp.cast cond_bool cond .u32 .bool
                              :: KernelOp.branch cond_bool thenOps elseOps :: postOps)
                            = some kst' := by
                        rw [Quanta.KOps.evalOps]
                        rw [h_cast_max]
                        simp [h_kst_cast_broke]
                        rw [Quanta.KOps.evalOps]
                        rw [h_branch_evals_to]
                        simp [h_ab_broke, h_post_max]
                      exact (evalOps_append h1 h_kst1_ok).trans h_cast_branch_post
                    · rw [← h_s_eq]; exact R_p
                · -- c ≠ 0: WASM picks thenBody, eval runs it.
                  simp only [hc, ↓reduceIte] at hw
                  cases h_eb : evalInstrs bt ws0 thenBody with
                  | none => simp [h_eb] at hw
                  | some ws_ab =>
                    simp [h_eb] at hw
                    obtain ⟨kst_ab, F_b, h_ev_b, R_b, h_bridge_b⟩ :=
                      then_preserves R_at_cast h_ws0_nb h_ws0_nh h_kst_cast_broke
                        h_eb hlt
                    obtain ⟨h_ab_nb, h_ab_nh, _, _, _, _, _, _, _⟩ :=
                      then_falls_through R_at_cast h_ws0_nb h_ws0_nh h_kst_cast_broke
                        h_eb hlt
                    rw [h_ab_nb] at hw
                    simp only at hw
                    have h_ab_broke : kst_ab.broke = false := h_bridge_b.right h_ab_nb
                    -- R_b is at s2. Lift to s3 via else_lowering_preserves
                    -- (same reg-frame except bumped nextReg, same stack/locals).
                    have R_b_at_s3 : Refines ws_ab s3 kst_ab layout := by
                      refine ⟨?_, ?_, ?_, ?_, ?_, R_b.heapRefines⟩
                      · refine ⟨?_, ?_⟩
                        · show ws_ab.stack.length = s3.stack.length
                          rw [h_s3_stack]
                          exact R_b.stk.left
                        · intro i v hv
                          obtain ⟨svi, hsv_get, henc⟩ := R_b.stk.right i v hv
                          have hsv_get_s3 : s3.stack.get? i = some svi := by
                            rw [h_s3_stack]; exact hsv_get
                          exact ⟨svi, hsv_get_s3, henc⟩
                      · intro i r hfind v hv
                        have hfind_s2 :
                            s2.localReg.find? (fun p => p.fst = i) = some (i, r) := by
                          rw [← h_s3_lr]; exact hfind
                        exact R_b.locs i r hfind_s2 v hv
                      · refine ⟨?_, ?_⟩
                        · intro sv hsv r hr
                          have hsv_s2 : sv ∈ s2.stack := by rw [← h_s3_stack]; exact hsv
                          exact Nat.lt_of_lt_of_le (R_b.fresh.left sv hsv_s2 r hr) h_s3_nr
                        · intro ir hir
                          have hir_s2 : ir ∈ s2.localReg := by rw [← h_s3_lr]; exact hir
                          exact Nat.lt_of_lt_of_le (R_b.fresh.right ir hir_s2) h_s3_nr
                      · intro ir hir sv hsv
                        have hir_s2 : ir ∈ s2.localReg := by rw [← h_s3_lr]; exact hir
                        have hsv_s2 : sv ∈ s2.stack := by rw [← h_s3_stack]; exact hsv
                        exact R_b.aliasFree ir hir_s2 sv hsv_s2
                      · intro p q hp hq
                        have hp_s2 : p ∈ s2.localReg := by rw [← h_s3_lr]; exact hp
                        have hq_s2 : q ∈ s2.localReg := by rw [← h_s3_lr]; exact hq
                        exact R_b.injLocals p q hp_s2 hq_s2
                    obtain ⟨kst', F_p, h_ev_p, R_p, h_bridge_p⟩ :=
                      post_preserves R_b_at_s3 h_ab_nb h_ab_nh h_ab_broke hw hlp
                    -- IR composition.
                    let F : Nat := max (max F_b F_p) 1
                    refine ⟨kst', F, ?_, ?_, h_bridge_p⟩
                    · rw [← h_ops_eq]
                      have h1 : evalOps F kst opsCommit = some kst1 :=
                        evalOps_fuel_mono (Nat.zero_le _) h_evalCommit
                      have h_then_max : evalOps F kst_cast thenOps = some kst_ab :=
                        evalOps_fuel_mono
                          (Nat.le_trans (Nat.le_max_left _ _) (Nat.le_max_left _ _)) h_ev_b
                      have h_post_max : evalOps F kst_ab postOps = some kst' :=
                        evalOps_fuel_mono
                          (Nat.le_trans (Nat.le_max_right _ _) (Nat.le_max_left _ _)) h_ev_p
                      have h_cast_max : Quanta.KOps.evalOp F kst1
                          (KernelOp.cast cond_bool cond Quanta.KOps.Scalar.u32 Quanta.KOps.Scalar.bool)
                          = some kst_cast := by
                        simp [Quanta.KOps.evalOp, h_lookup, Quanta.KOps.evalCast, kst_cast]
                      have h_lookup_cast :
                          Quanta.KOps.regLookup kst_cast.rf cond_bool
                            = some (Quanta.KOps.Value.vBool (!decide (c = 0))) := by
                        show Quanta.KOps.regLookup
                               (Quanta.KOps.regWrite kst1.rf cond_bool
                                 (Quanta.KOps.vBool (!decide (c = 0)))) cond_bool
                             = _
                        exact regLookup_regWrite_self _ _ _
                      have h_branch_evals_to :
                          Quanta.KOps.evalOp F kst_cast
                            (KernelOp.branch cond_bool thenOps elseOps) = some kst_ab := by
                        simp [Quanta.KOps.evalOp, h_lookup_cast, hc, h_then_max]
                      have h_cast_branch_post :
                          Quanta.KOps.evalOps F kst1
                            (KernelOp.cast cond_bool cond .u32 .bool
                              :: KernelOp.branch cond_bool thenOps elseOps :: postOps)
                            = some kst' := by
                        rw [Quanta.KOps.evalOps]
                        rw [h_cast_max]
                        simp [h_kst_cast_broke]
                        rw [Quanta.KOps.evalOps]
                        rw [h_branch_evals_to]
                        simp [h_ab_broke, h_post_max]
                      exact (evalOps_append h1 h_kst1_ok).trans h_cast_branch_post
                    · rw [← h_s_eq]; exact R_p
              | wI64 _ => simp at hw
              | wF32 _ => simp at hw
              | wF64 _ => simp at hw

-- ════════════════════════════════════════════════════════════════════
-- L8.5 §4b — bridging invariant foundations
--
-- The framework_preservation_straightLine theorem's BridgeClauses
-- output already establishes the bridging invariant for non-control
-- bodies (the universal direction: branchTarget = none → broke =
-- false). cons_wloop needs the INVERSE bridge: in a brIf_loop_self
-- body's exit iteration (cond = 0 falls through), WASM
-- branchTarget = none BUT IR broke = true.
--
-- The foundational lemma below establishes the IR side of this
-- inversion: if body's IR yields broke = true after one run, opLoop
-- exits in one iteration. This decouples the iteration mechanism
-- from the body's specific shape — any body whose evalOps sets
-- broke=true on the first iter triggers a one-iter loop exit.
-- ════════════════════════════════════════════════════════════════════

/-- N-iteration exit lemma for `opLoop`: if running the body's IR
    produces `n` consecutive states `st = st₀, st₁, …, st_n` where
    each `evalOps fuel st_i body = some st_{i+1}`, `st_i.broke =
    false` for `i < n`, and `st_n.broke = true`, then opLoop runs
    body exactly `n` times then exits (returning `st_n.reset_broke`).

    Requires the iteration counter `f` to be at least `n + 1`:
    `n` to run body each time, plus 1 to observe broke=true on
    the (n+1)-th iteration. -/
theorem opLoop_n_iter_exit
    {fuel : Nat} {body : List KernelOp}
    {n : Nat} (states : Fin (n + 1) → Quanta.KOps.State)
    (h_step : ∀ i : Fin n,
        evalOps fuel (states i.castSucc) body
          = some (states i.succ))
    (h_no_broke : ∀ i : Fin n, (states i.castSucc).broke = false)
    (h_final_broke : (states (Fin.last n)).broke = true)
    {f : Nat} (h_f : f ≥ n + 1) :
    Quanta.KOps.opLoop fuel body f (states 0) = some (states (Fin.last n)).reset_broke := by
  induction n generalizing f with
  | zero =>
      have h_broke_0 : (states 0).broke = true := h_final_broke
      have h_f_pos : f ≥ 1 := by omega
      obtain ⟨k, hk⟩ : ∃ k, f = k + 1 := ⟨f - 1, by omega⟩
      rw [hk]
      rw [Quanta.KOps.opLoop]
      simp [h_broke_0]
  | succ n IH =>
      have h_broke_0 : (states 0).broke = false := h_no_broke 0
      have h_step_0 : Quanta.KOps.evalOps fuel (states 0) body
                        = some (states ⟨1, by omega⟩) := h_step 0
      have h_f_ge_1 : f ≥ 1 := by omega
      obtain ⟨k, hk⟩ : ∃ k, f = k + 1 := ⟨f - 1, by omega⟩
      rw [hk]
      rw [Quanta.KOps.opLoop]
      simp [h_broke_0]
      rw [h_step_0]
      simp only
      have h_k_bound : k ≥ n + 1 := by omega
      -- Apply IH on shifted states.
      have h_ih := IH (fun i => states ⟨i.val + 1, by omega⟩)
        (fun i => h_step ⟨i.val + 1, by omega⟩)
        (fun i => h_no_broke ⟨i.val + 1, by omega⟩)
        h_final_broke h_k_bound
      -- h_ih has shape opLoop fuel body k (states ⟨1, _⟩)
      --   = some (states ⟨n+1, _⟩).reset_broke. Match the goal.
      exact h_ih

open Quanta.KOps (opLoop State) in
/-- One-iteration exit lemma for `opLoop`: if running the body's
    IR for `evalOps fuel st body = some st_next` with
    `st_next.broke = true`, the loop runs body exactly once,
    then exits (returning the broke-reset state).

    Requires the iteration counter `f` to be at least 2: one to
    run body, one to observe broke=true on the next iteration. -/
theorem opLoop_one_iter_exit
    {fuel : Nat} {body : List KernelOp}
    {st st_next : State}
    (h_pre_broke : st.broke = false)
    (h_body : Quanta.KOps.evalOps fuel st body = some st_next)
    (h_post_broke : st_next.broke = true)
    {f : Nat} (h_f : f ≥ 2) :
    opLoop fuel body f st = some st_next.reset_broke := by
  -- f ≥ 2 means f = f₀ + 2.
  rcases Nat.exists_eq_add_of_le h_f with ⟨k, hk⟩
  -- hk : f = 2 + k. Rewrite as (k + 1) + 1.
  have h_f_eq : f = (k + 1) + 1 := by omega
  rw [h_f_eq]
  rw [Quanta.KOps.opLoop]
  simp [h_pre_broke]
  rw [h_body]
  simp only
  -- Now goal: opLoop fuel body (k + 1) st_next = some st_next.reset_broke
  rw [Quanta.KOps.opLoop]
  simp [h_post_broke]

/-- WASM-side `iterLoop` N-iteration exit lemma. Given a sequence
    of body-output states where the first N-1 set `branchTarget =
    some 0` (continue) and the last sets `branchTarget = none`
    (exit), and `post` evaluates from the exit state to ws_final,
    iterLoop returns `some ws_final`.

    `entries i` is the entry state to iteration i (so entries 0 is
    iterLoop's starting state, entries (i+1) is the previous
    iteration's body output with branchTarget cleared).

    Requires the iteration counter `f` to be at least `n + 1`:
    `n - 1` continues + the exit iteration + the post-eval. The
    `+ 1` slack is because the last iteration is the exit and
    consumes one fuel unit. -/
theorem iterLoop_n_iter_exit
    {fuel : Nat}
    {body post : List WasmInstr}
    {n : Nat}
    (entries : Fin (n + 1) → WasmState)
    (bodyOuts : Fin (n + 1) → WasmState)
    (h_step : ∀ i : Fin (n + 1),
        evalInstrs fuel (entries i) body = some (bodyOuts i))
    (h_continue : ∀ i : Fin n,
        (bodyOuts i.castSucc).branchTarget = some 0 ∧
        entries i.succ = { bodyOuts i.castSucc with branchTarget := none })
    (h_exit : (bodyOuts (Fin.last n)).branchTarget = none)
    {ws_final : WasmState}
    (h_post : evalInstrs fuel (bodyOuts (Fin.last n)) post = some ws_final)
    {f : Nat} (h_f : f ≥ n + 1) :
    evalInstrs.iterLoop fuel body post f (entries 0) = some ws_final := by
  induction n generalizing f with
  | zero =>
      -- 0 continues, just the exit iteration: bodyOuts (Fin.last 0)
      -- has branchTarget = none, post runs from it.
      have h_f_ge_1 : f ≥ 1 := by omega
      obtain ⟨k, hk⟩ : ∃ k, f = k + 1 := ⟨f - 1, by omega⟩
      rw [hk]
      unfold evalInstrs.iterLoop
      rw [h_step 0]
      simp only
      -- Reduce Fin.last 0 to 0 in h_exit / h_post.
      have h_bt : (bodyOuts 0).branchTarget = none := h_exit
      have h_post' : evalInstrs fuel (bodyOuts 0) post = some ws_final := h_post
      rw [h_bt]
      exact h_post'
  | succ n IH =>
      -- iter 0 continues; then iter 1..n + exit.
      have h_f_ge_1 : f ≥ 1 := by omega
      obtain ⟨k, hk⟩ : ∃ k, f = k + 1 := ⟨f - 1, by omega⟩
      rw [hk]
      unfold evalInstrs.iterLoop
      rw [h_step 0]
      simp only
      have h_cont_0 := h_continue 0
      -- h_cont_0.left : (bodyOuts (Fin.castSucc 0)).branchTarget = some 0
      -- Reduce Fin.castSucc 0 to 0 first.
      have h_bt_0 : (bodyOuts 0).branchTarget = some 0 := h_cont_0.left
      rw [h_bt_0]
      simp only
      -- Now: iterLoop fuel body post k {bodyOuts 0 with branchTarget := none}.
      have h_entries_1 : ({ bodyOuts (0 : Fin (n + 1 + 1))
                                with branchTarget := none } : WasmState)
                            = entries ⟨1, by omega⟩ := h_cont_0.right.symm
      rw [h_entries_1]
      have h_k_bound : k ≥ n + 1 := by omega
      exact IH
        (fun i => entries ⟨i.val + 1, by omega⟩)
        (fun i => bodyOuts ⟨i.val + 1, by omega⟩)
        (fun i => h_step ⟨i.val + 1, by omega⟩)
        (fun i => h_continue ⟨i.val + 1, by omega⟩)
        h_exit
        h_post
        h_k_bound

/-- Inverse of `iterLoop_n_iter_exit`: given the trace AND a fact
    that iterLoop returns some ws', derives that post-eval from the
    exit body-out state yields ws'. Used by cons_wloop_nIterExit to
    extract the post-eval hypothesis from hw. -/
theorem iterLoop_n_iter_exit_post_eval
    {fuel : Nat}
    {body post : List WasmInstr}
    {n : Nat}
    (entries : Fin (n + 1) → WasmState)
    (bodyOuts : Fin (n + 1) → WasmState)
    (h_step : ∀ i : Fin (n + 1),
        evalInstrs fuel (entries i) body = some (bodyOuts i))
    (h_continue : ∀ i : Fin n,
        (bodyOuts i.castSucc).branchTarget = some 0 ∧
        entries i.succ = { bodyOuts i.castSucc with branchTarget := none })
    (h_exit : (bodyOuts (Fin.last n)).branchTarget = none)
    {f : Nat} (h_f : f ≥ n + 1)
    {ws_final : WasmState}
    (h_iter : evalInstrs.iterLoop fuel body post f (entries 0) = some ws_final) :
    evalInstrs fuel (bodyOuts (Fin.last n)) post = some ws_final := by
  induction n generalizing f with
  | zero =>
      have h_f_ge_1 : f ≥ 1 := by omega
      obtain ⟨k, hk⟩ : ∃ k, f = k + 1 := ⟨f - 1, by omega⟩
      rw [hk] at h_iter
      unfold evalInstrs.iterLoop at h_iter
      rw [h_step 0] at h_iter
      simp only at h_iter
      have h_bt : (bodyOuts 0).branchTarget = none := h_exit
      rw [h_bt] at h_iter
      exact h_iter
  | succ n IH =>
      have h_f_ge_1 : f ≥ 1 := by omega
      obtain ⟨k, hk⟩ : ∃ k, f = k + 1 := ⟨f - 1, by omega⟩
      rw [hk] at h_iter
      unfold evalInstrs.iterLoop at h_iter
      rw [h_step 0] at h_iter
      simp only at h_iter
      have h_cont_0 := h_continue 0
      have h_bt_0 : (bodyOuts 0).branchTarget = some 0 := h_cont_0.left
      rw [h_bt_0] at h_iter
      simp only at h_iter
      have h_entries_1 : ({ bodyOuts (0 : Fin (n + 1 + 1))
                                with branchTarget := none } : WasmState)
                            = entries ⟨1, by omega⟩ := h_cont_0.right.symm
      rw [h_entries_1] at h_iter
      have h_k_bound : k ≥ n + 1 := by omega
      exact IH
        (fun i => entries ⟨i.val + 1, by omega⟩)
        (fun i => bodyOuts ⟨i.val + 1, by omega⟩)
        (fun i => h_step ⟨i.val + 1, by omega⟩)
        (fun i => h_continue ⟨i.val + 1, by omega⟩)
        h_exit
        h_k_bound
        h_iter

/-- `wloop _ :: rest` preservation, single-iteration-exit case.
    The body runs exactly once on both sides:
    - WASM: body's eval ends with branchTarget = none (fall-
      through after the exit-cond brIf), halted = false. iterLoop
      then runs post.
    - IR: body's IR ends with broke = true (the exit arm of the
      brIf_loop_self emits .breakOp). opLoop sees broke=true on
      the next iter and returns reset_broke state. post runs.

    Caller supplies `body_preserves` (the standard recursive
    bridge IH) AND a stronger
    `body_exits_with_broke_true` IH that asserts the body's IR
    leaves kst'.broke = true (which is what the brIf_loop_self_bridge
    discharges for `[.i32Const 0; .brIf 0]`-style bodies).

    Restrictions: body doesn't mutate locals (no localSet), body's
    lowering preserves structural invariants. Inherits the same
    well-formedness package as cons_wif. -/
theorem preservation_evalInstrs_cons_wloop_singleIterExit
    (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (bt : Nat) (rest : List WasmInstr)
    (body post : List WasmInstr)
    (h_split : splitAtEnd rest = some (body, post))
    -- Body's recursive preservation IH (under .loopK frame).
    -- Note: this loop variant intentionally DROPS the BridgeClauses
    -- requirement from the standard cons-bridge shape — for the
    -- single-iter-exit case, the body's WASM falls through
    -- (branchTarget = none) while the IR exits via broke=true, which
    -- violates the standard BridgeClauses.right clause. The bridge
    -- mechanism here lives at the wloop-frame boundary instead: the
    -- post-loop bridge clauses come from `post_preserves` against the
    -- broke-reset state.
    (body_preserves : ∀ {ws_b : WasmState} {s_b : LowerState}
        {kst_b : Quanta.KOps.State}
        (_R_b : Refines ws_b s_b kst_b layout)
        (_h_nb_b : ws_b.branchTarget = none)
        (_h_nh_b : ws_b.halted = false)
        (_h_nbk_b : kst_b.broke = false)
        {ws'_b : WasmState} {s'_b : LowerState} {bodyOps : List KernelOp}
        (_hw_b : evalInstrs bt ws_b body = some ws'_b)
        (_hl_b : lowerInstrs bt (.loopK :: frames) s_b body = some (s'_b, bodyOps)),
      ∃ (kst'_b : Quanta.KOps.State) (F : Nat),
        evalOps F kst_b bodyOps = some kst'_b ∧
        Refines ws'_b s'_b kst'_b layout)
    -- Eval-side exit hypothesis: body ends with branchTarget=none,
    -- halted=false on the eval side. Stack/locals also fall-through.
    (body_falls_through : ∀ {ws_b : WasmState} {s_b : LowerState}
        {kst_b : Quanta.KOps.State} {ws'_b : WasmState} {s'_b : LowerState}
        {bodyOps : List KernelOp}
        (_R_b : Refines ws_b s_b kst_b layout)
        (_h_nb_b : ws_b.branchTarget = none)
        (_h_nh_b : ws_b.halted = false)
        (_h_nbk_b : kst_b.broke = false)
        (_hw_b : evalInstrs bt ws_b body = some ws'_b)
        (_hl_b : lowerInstrs bt (.loopK :: frames) s_b body = some (s'_b, bodyOps)),
      ws'_b.branchTarget = none ∧ ws'_b.halted = false ∧
      ws'_b.locals = ws_b.locals ∧ ws'_b.stack = ws_b.stack ∧
      ws'_b.mem = ws_b.mem ∧
      s'_b.localReg = s_b.localReg ∧ s'_b.localTy = s_b.localTy ∧
      s'_b.stack = s_b.stack ∧ s'_b.bufferSlots = s_b.bufferSlots)
    -- IR-side exit hypothesis: body's IR sets broke=true.
    -- Inverted bridge — what brIf_loop_self_bridge produces for
    -- the exit arm.
    (body_exits_with_broke : ∀ {ws_b : WasmState} {s_b : LowerState}
        {kst_b : Quanta.KOps.State} {ws'_b : WasmState} {s'_b : LowerState}
        {bodyOps : List KernelOp} {kst'_b : Quanta.KOps.State} {F_b : Nat}
        (_R_b : Refines ws_b s_b kst_b layout)
        (_h_nb_b : ws_b.branchTarget = none)
        (_h_nh_b : ws_b.halted = false)
        (_h_nbk_b : kst_b.broke = false)
        (_hw_b : evalInstrs bt ws_b body = some ws'_b)
        (_hl_b : lowerInstrs bt (.loopK :: frames) s_b body = some (s'_b, bodyOps))
        (_h_ev_b : evalOps F_b kst_b bodyOps = some kst'_b),
      kst'_b.broke = true)
    -- Lowering-only structural invariants. Stage 3: includes
    -- currentReg invariance because the wloop body's lowering now
    -- propagates currentReg through localSet/Tee's setCurrentReg
    -- updates, and the post-loop restore needs s'_b.currentReg
    -- to equal s_b.currentReg.
    (body_lowering_preserves : ∀ {s_b s'_b : LowerState} {bodyOps : List KernelOp},
        lowerInstrs bt (.loopK :: frames) s_b body = some (s'_b, bodyOps) →
        s'_b.localReg = s_b.localReg ∧ s'_b.localTy = s_b.localTy ∧
        s'_b.stack = s_b.stack ∧ s'_b.bufferSlots = s_b.bufferSlots ∧
        s'_b.currentReg = s_b.currentReg ∧
        s_b.nextReg ≤ s'_b.nextReg)
    -- Post-IH.
    (post_preserves : ∀ {ws_p : WasmState} {s_p : LowerState}
        {kst_p : Quanta.KOps.State}
        (_R_p : Refines ws_p s_p kst_p layout)
        (_h_nb_p : ws_p.branchTarget = none)
        (_h_nh_p : ws_p.halted = false)
        (_h_nbk_p : kst_p.broke = false)
        {ws'_p : WasmState} {s'_p : LowerState} {postOps : List KernelOp}
        (_hw_p : evalInstrs bt ws_p post = some ws'_p)
        (_hl_p : lowerInstrs bt frames s_p post = some (s'_p, postOps)),
      ∃ (kst'_p : Quanta.KOps.State) (F : Nat),
        evalOps F kst_p postOps = some kst'_p ∧
        Refines ws'_p s'_p kst'_p layout ∧
        BridgeClauses ws'_p kst'_p)
    -- Fuel ≥ 2 so opLoop_one_iter_exit applies.
    (h_fuel_ge_2 : bt ≥ 2)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs (bt + 1) ws (.wloop 0 :: rest) = some ws')
    (hl : lowerInstrs (bt + 1) frames s (.wloop 0 :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  -- Unfold the lowering's wloop arm.
  simp only [lowerInstrs] at hl
  rw [h_split] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl
  cases h_lb : lowerInstrs bt (.loopK :: frames) s body with
  | none => simp [h_lb] at hl
  | some body_pair =>
    rcases body_pair with ⟨s1, bodyOps⟩
    simp [h_lb] at hl
    -- s1_restored = { s1 with localReg := s.localReg, localTy := s.localTy }.
    -- body_lowering_preserves on h_lb says s1.localReg = s.localReg
    -- and s1.localTy = s.localTy, so s1_restored = s1.
    obtain ⟨h_s1_lr, h_s1_lt, h_s1_stack, h_s1_bs, h_s1_cr, h_s1_nr⟩ :=
      body_lowering_preserves h_lb
    have h_s1_restored_eq :
        ({ s1 with localReg := s.localReg, localTy := s.localTy,
                    currentReg := s.currentReg } : LowerState) = s1 := by
      have h_lr_eq : s.localReg = s1.localReg := h_s1_lr.symm
      have h_lt_eq : s.localTy = s1.localTy := h_s1_lt.symm
      have h_cr_eq : s.currentReg = s1.currentReg := h_s1_cr.symm
      rw [h_lr_eq, h_lt_eq, h_cr_eq]
    rw [h_s1_restored_eq] at hl
    cases h_lp : lowerInstrs bt frames s1 post with
    | none => simp [h_lp] at hl
    | some post_pair =>
      rcases post_pair with ⟨s2, postOps⟩
      simp [h_lp] at hl
      rcases hl with ⟨h_s_eq, h_ops_eq⟩
      -- Eval side: wloop arm.
      simp only [evalInstrs] at hw
      have h_cond : (ws.halted || ws.branchTarget.isSome) = false := by
        rw [h_no_halt, h_no_branch]; rfl
      rw [h_cond] at hw
      simp only [Bool.false_eq_true, ↓reduceIte] at hw
      rw [h_split] at hw
      simp only at hw
      -- iterLoop bt ws → first iteration: evalInstrs bt ws body.
      -- We extract this from hw by unfolding iterLoop one step.
      -- hw : iterLoop bt ws = some ws'.
      -- Since bt = bt' + 1 (we'd need bt ≥ 1 to unfold), reduce.
      -- Note: bt ≥ 2 (from h_fuel_ge_2). Express bt = bt' + 1 to
      -- unfold iterLoop, but keep bt in all IH-facing positions.
      match h_bt_match : bt, h_fuel_ge_2 with
      | 0, h_le => exact absurd h_le (by decide)
      | bt' + 1, _ =>
        -- Unfold iterLoop one step in hw.
        unfold evalInstrs.iterLoop at hw
        cases h_eb : evalInstrs (bt' + 1) ws body with
        | none =>
          rw [h_eb] at hw
          simp at hw
        | some ws_after_body =>
          rw [h_eb] at hw
          -- ws_after_body.branchTarget = none from body_falls_through.
          obtain ⟨kst_after_body, F_b, h_ev_b, R_b⟩ :=
            body_preserves R h_no_branch h_no_halt h_kst_no_broke h_eb h_lb
          obtain ⟨h_ab_nb, h_ab_nh, _h_ab_locs, _h_ab_stk, _h_ab_mem,
                  _h_s1_lr', _h_s1_lt', _h_s1_stk', _h_s1_bs'⟩ :=
            body_falls_through R h_no_branch h_no_halt h_kst_no_broke h_eb h_lb
          have h_ab_broke : kst_after_body.broke = true :=
            body_exits_with_broke R h_no_branch h_no_halt h_kst_no_broke h_eb h_lb h_ev_b
          -- Reduce the outer `match some ws_after_body` and then
          -- match on ws_after_body.branchTarget = none.
          simp only at hw
          rw [h_ab_nb] at hw
          simp only at hw
          -- hw : evalInstrs (bt'+1) ws_after_body post = some ws'.
          -- IR side: opLoop runs body once via opLoop_one_iter_exit
          --   → returns kst_after_body.reset_broke.
          -- For opLoop's F ≥ 2 requirement, use F_b's body fuel for the inner
          -- evalOps and a separate iteration counter ≥ 2.
          -- The actual `.loopOp bodyOps` evaluator uses `fuel` for both knobs
          -- (opLoop fuel body fuel s). Pick a sufficiently large F.
          --
          -- Apply post_preserves to get post's output.
          -- We need Refines ws_after_body s1 kst_after_body.reset_broke layout.
          -- R_b gives Refines ws_after_body s1 kst_after_body layout.
          -- reset_broke only changes the broke flag (which Refines doesn't see).
          have R_ab_reset :
              Refines ws_after_body s1 kst_after_body.reset_broke layout := by
            refine ⟨R_b.stk, R_b.locs, R_b.fresh, R_b.aliasFree,
                    R_b.injLocals, R_b.heapRefines⟩
          have h_reset_broke : kst_after_body.reset_broke.broke = false := by
            simp [Quanta.KOps.State.reset_broke]
          obtain ⟨kst', F_p, h_ev_p, R_p, h_bridge_p⟩ :=
            post_preserves R_ab_reset h_ab_nb h_ab_nh h_reset_broke hw h_lp
          -- Compose: ops = [.loopOp bodyOps] ++ postOps.
          -- evalOps for this: evalOp .loopOp = opLoop fuel bodyOps fuel kst →
          --   = some kst_after_body.reset_broke (via opLoop_one_iter_exit).
          --   Then postOps from there → some kst'.
          let F : Nat := max (max F_b F_p) 2
          have h_F_ge_2 : F ≥ 2 := by
            simp [F]
            omega
          have h_F_ge_Fb : F_b ≤ F := by
            simp [F]; omega
          have h_F_ge_Fp : F_p ≤ F := by
            simp [F]; omega
          refine ⟨kst', F, ?_, ?_, h_bridge_p⟩
          · rw [← h_ops_eq]
            -- evalOps F kst ([.loopOp bodyOps] ++ postOps).
            -- First step: evalOp F kst (.loopOp bodyOps) = opLoop F bodyOps F kst.
            -- Apply opLoop_one_iter_exit with F as the iteration counter.
            have h_ev_b_F : evalOps F kst bodyOps = some kst_after_body :=
              evalOps_fuel_mono h_F_ge_Fb h_ev_b
            have h_op_loop :
                Quanta.KOps.opLoop F bodyOps F kst = some kst_after_body.reset_broke :=
              opLoop_one_iter_exit h_kst_no_broke h_ev_b_F h_ab_broke h_F_ge_2
            have h_ev_p_F : evalOps F kst_after_body.reset_broke postOps = some kst' :=
              evalOps_fuel_mono h_F_ge_Fp h_ev_p
            -- Compose via evalOps_append.
            have h_loop_op :
                Quanta.KOps.evalOp F kst (.loopOp bodyOps)
                  = some kst_after_body.reset_broke := by
              simp [Quanta.KOps.evalOp]
              exact h_op_loop
            have h_full :
                Quanta.KOps.evalOps F kst
                  (KernelOp.loopOp bodyOps :: postOps)
                  = some kst' := by
              rw [Quanta.KOps.evalOps]
              rw [h_loop_op]
              simp [h_reset_broke, h_ev_p_F]
            exact h_full
          · rw [← h_s_eq]; exact R_p

-- ════════════════════════════════════════════════════════════════════
-- L8.3 general — cons_wloop_nIterExit
--
-- Generalization of cons_wloop_singleIterExit to N iterations.
-- The wloop runs body exactly N+1 times: N continues (each setting
-- WASM branchTarget = some 0 and keeping IR broke = false) followed
-- by one exit (WASM branchTarget = none and IR broke = true).
-- Composes opLoop_n_iter_exit (IR side) with iterLoop_n_iter_exit
-- (WASM side) via per-iteration Refines preservation.
--
-- Caller supplies:
--   - WASM-side entry / body-out state sequences
--   - IR-side state sequence
--   - Per-iteration body preservation IHs proving each iter's
--     transition matches across WASM/IR
--   - Standard post and lowering IHs
--
-- The cons_wloop_singleIterExit case corresponds to n = 0 here
-- (0 continue iterations, just the exit body run).
-- ════════════════════════════════════════════════════════════════════

/-- `wloop _ :: rest` preservation, n-iteration exit case.
    The wloop runs body exactly (n + 1) times: n continues + 1 exit.

    Caller supplies the iteration trace as state sequences and
    per-iteration preservation evidence. This generalizes the
    `cons_wloop_singleIterExit` (which handles n = 0). -/
theorem preservation_evalInstrs_cons_wloop_nIterExit
    (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (_R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (bt : Nat) (rest : List WasmInstr)
    (body post : List WasmInstr)
    (h_split : splitAtEnd rest = some (body, post))
    (n : Nat)
    -- WASM-side iteration trace: entry states + body-output states.
    -- entries 0 = ws (loop entry); entries (i+1) = bodyOuts i with
    -- branchTarget cleared (continue semantics).
    (wasmEntries wasmBodyOuts : Fin (n + 1) → WasmState)
    (h_wasm_start : wasmEntries 0 = ws)
    (h_wasm_step : ∀ i : Fin (n + 1),
        evalInstrs bt (wasmEntries i) body = some (wasmBodyOuts i))
    (h_wasm_continue : ∀ i : Fin n,
        (wasmBodyOuts i.castSucc).branchTarget = some 0 ∧
        wasmEntries i.succ
          = { wasmBodyOuts i.castSucc with branchTarget := none })
    (h_wasm_exit : (wasmBodyOuts (Fin.last n)).branchTarget = none ∧
                   (wasmBodyOuts (Fin.last n)).halted = false)
    -- IR-side iteration trace: lowering state s1 (after body lowering)
    -- + IR body-output state sequence kstStates.
    (s1 : LowerState) (bodyOps : List KernelOp)
    (h_lb : lowerInstrs bt (.loopK :: frames) s body = some (s1, bodyOps))
    (kstStates : Fin (n + 2) → Quanta.KOps.State)
    (h_kst_start : kstStates 0 = kst)
    (F_b : Nat)
    (h_ir_step : ∀ i : Fin (n + 1),
        evalOps F_b (kstStates i.castSucc) bodyOps
          = some (kstStates i.succ))
    (h_ir_continue : ∀ i : Fin n,
        (kstStates i.castSucc.succ).broke = false)
    (h_ir_exit : (kstStates (Fin.last (n + 1))).broke = true)
    -- Per-iteration Refines preservation across body iterations.
    (h_per_iter_refines : ∀ i : Fin (n + 1),
        Refines (wasmBodyOuts i) s1 (kstStates i.succ) layout)
    -- Post-loop bridge.
    (s2 : LowerState) (postOps : List KernelOp)
    (h_lp : lowerInstrs bt frames s1 post = some (s2, postOps))
    (post_preserves : ∀ {ws_p : WasmState} {s_p : LowerState}
        {kst_p : Quanta.KOps.State}
        (_R_p : Refines ws_p s_p kst_p layout)
        (_h_nb_p : ws_p.branchTarget = none)
        (_h_nh_p : ws_p.halted = false)
        (_h_nbk_p : kst_p.broke = false)
        {ws'_p : WasmState} {s'_p : LowerState} {postOps' : List KernelOp}
        (_hw_p : evalInstrs bt ws_p post = some ws'_p)
        (_hl_p : lowerInstrs bt frames s_p post = some (s'_p, postOps')),
      ∃ (kst'_p : Quanta.KOps.State) (F : Nat),
        evalOps F kst_p postOps' = some kst'_p ∧
        Refines ws'_p s'_p kst'_p layout ∧
        BridgeClauses ws'_p kst'_p)
    -- Lowering-only structural invariants for body.
    (h_body_lowering : s1.localReg = s.localReg ∧ s1.localTy = s.localTy ∧
                       s1.stack = s.stack ∧ s1.bufferSlots = s.bufferSlots ∧
                       s.nextReg ≤ s1.nextReg)
    -- Fuel constraints: bt ≥ n + 2 (iterLoop needs n+1 continues +
    -- exit; opLoop needs n + 2 iter check + body runs).
    (h_fuel_bound : bt ≥ n + 2)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs (bt + 1) ws (.wloop 0 :: rest) = some ws')
    (hl : lowerInstrs (bt + 1) frames s (.wloop 0 :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  -- Unfold lowering's wloop arm: matches up s1, bodyOps, s2, postOps via h_lb, h_lp.
  simp only [lowerInstrs] at hl
  rw [h_split] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl
  rw [h_lb] at hl
  simp only [Option.some_bind] at hl
  obtain ⟨h_lr, h_lt, h_stk_eq, h_bs, h_nr⟩ := h_body_lowering
  have h_s1_restored_eq :
      ({ s1 with localReg := s.localReg, localTy := s.localTy } : LowerState) = s1 := by
    have h_lr_eq : s.localReg = s1.localReg := h_lr.symm
    have h_lt_eq : s.localTy = s1.localTy := h_lt.symm
    rw [h_lr_eq, h_lt_eq]
  rw [h_s1_restored_eq] at hl
  rw [h_lp] at hl
  simp only [Option.some_bind, Option.some.injEq, Prod.mk.injEq, pure] at hl
  -- hl should now be: s' = s2 ∧ ops = [.loopOp bodyOps] ++ postOps
  obtain ⟨h_s_eq, h_ops_eq⟩ := hl
  -- Eval side: wloop arm. Decompose into iterLoop on ws.
  simp only [evalInstrs] at hw
  have h_cond : (ws.halted || ws.branchTarget.isSome) = false := by
    rw [h_no_halt, h_no_branch]; rfl
  rw [h_cond] at hw
  simp only [Bool.false_eq_true, ↓reduceIte] at hw
  rw [h_split] at hw
  simp only at hw
  -- hw : iterLoop bt body post bt ws = some ws'.
  -- Apply iterLoop_n_iter_exit. We need:
  --   evalInstrs (bt+1) (wasmBodyOuts (Fin.last n)) post = some ws'
  -- which comes from post_preserves applied to the exit state.
  -- First derive post-eval via post_preserves.
  have h_exit_nb : (wasmBodyOuts (Fin.last n)).branchTarget = none := h_wasm_exit.left
  have h_exit_nh : (wasmBodyOuts (Fin.last n)).halted = false := h_wasm_exit.right
  -- Build Refines on the exit state.
  have R_exit : Refines (wasmBodyOuts (Fin.last n)) s1
                          (kstStates (Fin.last (n + 1))) layout := by
    have h_succ_eq : (Fin.last n).succ = Fin.last (n + 1) := rfl
    have := h_per_iter_refines (Fin.last n)
    rw [h_succ_eq] at this
    exact this
  -- The IR exit state has broke = true; reset_broke clears it.
  have h_kst_exit_broke : (kstStates (Fin.last (n + 1))).broke = true := h_ir_exit
  -- Use post_preserves on the broke-reset state.
  have R_reset : Refines (wasmBodyOuts (Fin.last n)) s1
                          (kstStates (Fin.last (n + 1))).reset_broke layout := by
    refine ⟨R_exit.stk, R_exit.locs, R_exit.fresh, R_exit.aliasFree,
            R_exit.injLocals, R_exit.heapRefines⟩
  have h_reset_nbk : ((kstStates (Fin.last (n + 1))).reset_broke).broke = false := by
    simp [Quanta.KOps.State.reset_broke]
  -- Establish hw_post = evalInstrs bt (wasmBodyOuts last) post = some ws'.
  -- iterLoop_n_iter_exit reduces hw to evalInstrs bt (wasmBodyOuts last) post = some ws'.
  have h_iter_match : evalInstrs.iterLoop bt body post bt ws
                        = some ws' := hw
  -- Substitute ws = wasmEntries 0 in h_iter_match.
  rw [← h_wasm_start] at h_iter_match
  have h_post_eval : evalInstrs bt (wasmBodyOuts (Fin.last n)) post = some ws' := by
    have h_bt_bound : bt ≥ n + 1 := by omega
    exact iterLoop_n_iter_exit_post_eval wasmEntries wasmBodyOuts
      h_wasm_step h_wasm_continue h_exit_nb h_bt_bound h_iter_match
  -- Now apply post_preserves.
  obtain ⟨kst'_p, F_p, h_ev_p, R_p, h_bridge_p⟩ :=
    post_preserves R_reset h_exit_nb h_exit_nh h_reset_nbk h_post_eval h_lp
  -- Build the IR-side composition.
  let F : Nat := max (max F_b F_p) (n + 2)
  have h_F_ge_Fb : F_b ≤ F := by simp [F]; omega
  have h_F_ge_Fp : F_p ≤ F := by simp [F]; omega
  have h_F_ge_n_plus_2 : F ≥ n + 2 := by simp [F]; omega
  refine ⟨kst'_p, F, ?_, ?_, h_bridge_p⟩
  · rw [← h_ops_eq]
    -- evalOps F kst ([.loopOp bodyOps] ++ postOps).
    -- First: evalOp F kst (.loopOp bodyOps) = opLoop F bodyOps F kst.
    -- Use opLoop_n_iter_exit on the bumped-fuel state sequence.
    have h_ir_step_F : ∀ i : Fin (n + 1),
        evalOps F (kstStates i.castSucc) bodyOps = some (kstStates i.succ) := by
      intro i
      exact evalOps_fuel_mono h_F_ge_Fb (h_ir_step i)
    -- opLoop_n_iter_exit: need kstStates 0 = kst, but kstStates 0 = kst (from h_kst_start).
    have h_no_broke_seq : ∀ i : Fin (n + 1), (kstStates i.castSucc).broke = false := by
      intro i
      match i, i.isLt with
      | ⟨0, _⟩, _ =>
          show (kstStates 0).broke = false
          rw [h_kst_start]; exact h_kst_no_broke
      | ⟨k + 1, h_lt⟩, _ =>
          show (kstStates ⟨k + 1, by omega⟩).broke = false
          have h := h_ir_continue ⟨k, by omega⟩
          -- ⟨k, _⟩.castSucc.succ = ⟨k + 1, _⟩ (definitional).
          exact h
    have h_op_loop :
        Quanta.KOps.opLoop F bodyOps F (kstStates 0)
          = some (kstStates (Fin.last (n + 1))).reset_broke := by
      apply opLoop_n_iter_exit kstStates h_ir_step_F h_no_broke_seq h_ir_exit
      omega
    rw [h_kst_start] at h_op_loop
    have h_ev_p_F : evalOps F (kstStates (Fin.last (n + 1))).reset_broke postOps
                      = some kst'_p := evalOps_fuel_mono h_F_ge_Fp h_ev_p
    have h_loop_op :
        Quanta.KOps.evalOp F kst (.loopOp bodyOps)
          = some (kstStates (Fin.last (n + 1))).reset_broke := by
      simp [Quanta.KOps.evalOp]
      exact h_op_loop
    have h_full :
        Quanta.KOps.evalOps F kst
          (KernelOp.loopOp bodyOps :: postOps) = some kst'_p := by
      rw [Quanta.KOps.evalOps]
      rw [h_loop_op]
      simp [h_reset_nbk, h_ev_p_F]
    exact h_full
  · rw [← h_s_eq]; exact R_p

-- ════════════════════════════════════════════════════════════════════
-- Concrete body lowering preservation for [.i32Const 0, .brIf 0]
--
-- The simplest concrete wloop body — pushes 0, then exits via brIf 0
-- (cond=0 falls through and IR-side .branch picks the [.breakOp] arm).
-- Closes the `body_lowering_preserves` IH of L10v6 for this body.
-- ════════════════════════════════════════════════════════════════════

/-- `body_lowering_preserves` discharge for body = `[.i32Const 0, .brIf 0]`
    under a `.loopK :: frames` frame stack with `bt ≥ 1` fuel.
    Stack effect is zero (push from i32Const, pop from brIf cancel);
    locals + bufferSlots unchanged; nextReg grows by 2 (one for the
    const's fresh reg, one for cond_bool). -/
theorem const0_brIf0_lowering_preserves
    {bt : Nat} (frames : List FrameKind)
    {s_b s'_b : LowerState} {bodyOps : List KernelOp}
    (h_lb : lowerInstrs bt (.loopK :: frames) s_b
              [.i32Const 0, .brIf 0] = some (s'_b, bodyOps)) :
    s'_b.localReg = s_b.localReg ∧ s'_b.localTy = s_b.localTy ∧
    s'_b.stack = s_b.stack ∧ s'_b.bufferSlots = s_b.bufferSlots ∧
    s_b.nextReg ≤ s'_b.nextReg := by
  -- Simp the lowering all the way down. The dispatch for i32Const goes
  -- through the default `_` arm; brIf 0 at .loopK :: frames takes the
  -- depth=0 path. popSym + commit + alloc + recurse-on-[] all reduce.
  -- frames.get? 0 reduces explicitly.
  have h_get0 : (FrameKind.loopK :: frames).get? 0 = some .loopK := rfl
  simp only [lowerInstrs, lowerInstr, LowerState.popSym, LowerState.commit,
             LowerState.alloc, Option.bind_eq_bind, Option.some_bind, pure,
             Option.some.injEq, Prod.mk.injEq, List.append_nil,
             h_get0, ↓reduceIte, if_true] at h_lb
  obtain ⟨h_s_eq, _⟩ := h_lb
  subst h_s_eq
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rfl
  · rfl
  · rfl
  · rfl
  · show s_b.nextReg ≤ s_b.nextReg + 1 + 1
    omega

/-- `body_falls_through` discharge for body = `[.i32Const 0, .brIf 0]`.
    WASM eval pushes wI32 0 then pops it (brIf 0 with cond=0 falls
    through) — net stack effect zero, no branch, no halt, locals/mem
    unchanged. Combines with the lowering-only invariants from
    `const0_brIf0_lowering_preserves`. -/
theorem const0_brIf0_falls_through
    {bt : Nat} (frames : List FrameKind)
    {ws_b : WasmState} {s_b : LowerState}
    {ws'_b : WasmState} {s'_b : LowerState}
    {bodyOps : List KernelOp}
    (h_no_branch : ws_b.branchTarget = none)
    (h_no_halt : ws_b.halted = false)
    (hw_b : evalInstrs bt ws_b [.i32Const 0, .brIf 0] = some ws'_b)
    (h_lb : lowerInstrs bt (.loopK :: frames) s_b
              [.i32Const 0, .brIf 0] = some (s'_b, bodyOps)) :
    ws'_b.branchTarget = none ∧ ws'_b.halted = false ∧
    ws'_b.locals = ws_b.locals ∧ ws'_b.stack = ws_b.stack ∧
    ws'_b.mem = ws_b.mem ∧
    s'_b.localReg = s_b.localReg ∧ s'_b.localTy = s_b.localTy ∧
    s'_b.stack = s_b.stack ∧ s'_b.bufferSlots = s_b.bufferSlots := by
  -- Lowering side: pull from const0_brIf0_lowering_preserves.
  obtain ⟨h_lr, h_lt, h_st, h_bs, _h_nr⟩ :=
    const0_brIf0_lowering_preserves frames h_lb
  -- Eval side: unfold the cons evaluation step by step.
  rw [evalInstrs_cons_default bt ws_b (.i32Const 0) [.brIf 0]
      h_no_branch h_no_halt (by simp [isStructuredEval])] at hw_b
  simp only [evalInstr, WasmState.push, Int.toNat] at hw_b
  -- The push state inherits ws_b.branchTarget = none and ws_b.halted = false.
  have h_push_nb :
      ({ ws_b with stack := .wI32 (UInt32.ofNat 0) :: ws_b.stack }
        : WasmState).branchTarget = none := h_no_branch
  have h_push_nh :
      ({ ws_b with stack := .wI32 (UInt32.ofNat 0) :: ws_b.stack }
        : WasmState).halted = false := h_no_halt
  rw [evalInstrs_cons_default bt _ (.brIf 0) [] h_push_nb h_push_nh
      (by simp [isStructuredEval])] at hw_b
  simp only [evalInstr, WasmState.pop, Option.bind_eq_bind,
             Option.some_bind, evalInstrs] at hw_b
  -- hw_b should now read: `some <restored ws_b copy> = some ws'_b` (modulo decide).
  simp at hw_b
  -- ws'_b is the restored copy: same locals/stack/mem/halted/branchTarget as ws_b.
  -- Use the WasmState extensionality from hw_b.
  refine ⟨?_, ?_, ?_, ?_, ?_, h_lr, h_lt, h_st, h_bs⟩
  all_goals (
    first
    | (rw [← hw_b]; exact h_no_branch)
    | (rw [← hw_b]; exact h_no_halt)
    | (rw [← hw_b])
  )

/-- The concrete `bodyOps` shape that `[.i32Const 0, .brIf 0]` lowers
    to under `.loopK :: frames` and `bt ≥ 1`. Used to discharge
    `body_exits_with_broke` by unfolding the IR-side evalOps. -/
theorem const0_brIf0_lowering_shape
    {bt : Nat} (frames : List FrameKind)
    {s_b s'_b : LowerState} {bodyOps : List KernelOp}
    (h_lb : lowerInstrs bt (.loopK :: frames) s_b
              [.i32Const 0, .brIf 0] = some (s'_b, bodyOps)) :
    bodyOps =
      [KernelOp.const s_b.nextReg (.u32 (UInt32.ofNat 0)),
       KernelOp.cast (s_b.nextReg + 1) s_b.nextReg .u32 .bool,
       KernelOp.branch (s_b.nextReg + 1) [] [KernelOp.breakOp]] := by
  have h_get0 : (FrameKind.loopK :: frames).get? 0 = some .loopK := rfl
  simp only [lowerInstrs, lowerInstr, LowerState.popSym, LowerState.commit,
             LowerState.alloc, Option.bind_eq_bind, Option.some_bind, pure,
             Option.some.injEq, Prod.mk.injEq, List.append_nil,
             h_get0, ↓reduceIte, if_true] at h_lb
  obtain ⟨_, h_ops_eq⟩ := h_lb
  rw [← h_ops_eq]
  simp [Int.toNat]

/-- `body_exits_with_broke` discharge for body = `[.i32Const 0, .brIf 0]`.
    The IR pipeline: const writes vU32 0 → cast → vBool false → branch
    picks elseOps [.breakOp] → broke := true. Final kst'_b.broke = true
    regardless of fuel (as long as evalOps succeeds, the .breakOp ran). -/
theorem const0_brIf0_exits_with_broke
    {bt : Nat} (frames : List FrameKind)
    {s_b : LowerState}
    {kst_b : Quanta.KOps.State} {s'_b : LowerState}
    {bodyOps : List KernelOp} {kst'_b : Quanta.KOps.State} {F_b : Nat}
    (h_kst_no_broke : kst_b.broke = false)
    (h_lb : lowerInstrs bt (.loopK :: frames) s_b
              [.i32Const 0, .brIf 0] = some (s'_b, bodyOps))
    (h_ev_b : evalOps F_b kst_b bodyOps = some kst'_b) :
    kst'_b.broke = true := by
  -- Extract the concrete bodyOps shape.
  have h_shape := const0_brIf0_lowering_shape frames h_lb
  subst h_shape
  -- Step through evalOps on [const, cast, branch].
  simp only [Quanta.KOps.evalOps, Quanta.KOps.evalOp,
             Quanta.KOps.evalConst, Quanta.KOps.evalCast,
             Option.bind_eq_bind, Option.some_bind, pure,
             regLookup_regWrite_self,
             h_kst_no_broke, Bool.false_eq_true, if_false] at h_ev_b
  -- Reduce vU32 / vBool aliases so the match arms fire.
  simp only [Quanta.KOps.vU32, Quanta.KOps.vBool] at h_ev_b
  simp at h_ev_b
  rw [← h_ev_b]

/-- `body_preserves` discharge for body = `[.i32Const 0, .brIf 0]`.
    Composes: (1) the i32Const step which pushes `.wI32 0` /
    `.i32ConstSym 0` on both sides with no IR emitted; (2) the brIf 0
    step via `cons_brIf_loop_self_bridge`. The cons-bridge produces
    the existential + Refines; this discharger uses the bridge variant
    (which exposes the popped cond) to thread through, but does NOT
    need to maintain BridgeClauses on the body's output — the
    cons_wloop_singleIterExit IH skips that requirement because the
    body's IR-broke=true conflicts with the standard bridge clauses. -/
theorem const0_brIf0_body_preserves
    {bt : Nat} (frames : List FrameKind)
    {ws_b : WasmState} {s_b : LowerState}
    {kst_b : Quanta.KOps.State}
    (layout : BufferLayout)
    (R_b : Refines ws_b s_b kst_b layout)
    (h_nb_b : ws_b.branchTarget = none)
    (h_nh_b : ws_b.halted = false)
    (h_nbk_b : kst_b.broke = false)
    {ws'_b : WasmState} {s'_b : LowerState} {bodyOps : List KernelOp}
    (hw_b : evalInstrs bt ws_b [.i32Const 0, .brIf 0] = some ws'_b)
    (hl_b : lowerInstrs bt (.loopK :: frames) s_b
              [.i32Const 0, .brIf 0] = some (s'_b, bodyOps)) :
    ∃ (kst'_b : Quanta.KOps.State) (F : Nat),
      evalOps F kst_b bodyOps = some kst'_b ∧
      Refines ws'_b s'_b kst'_b layout := by
  -- Use preservation_evalInstrs_cons_i32Const (non-bridge variant —
  -- it doesn't require BridgeClauses on the rest's output). Its
  -- preservation_rest IH gets discharged by cons_brIf_loop_self
  -- against rest = [.brIf 0] starting from the i32Const-pushed
  -- mid-state.
  have rest_preserves :
      ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        (_h_kst_no_broke_mid : kst_mid.broke = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs bt ws_mid [.brIf 0] = some ws'_mid)
        (_hl_mid : lowerInstrs bt (.loopK :: frames) s_mid [.brIf 0]
                     = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout := by
    intro ws_mid s_mid kst_mid R_mid h_nb_m h_nh_m h_nbk_m
          ws'_mid s'_mid postOps hw_mid hl_mid
    exact preservation_evalInstrs_cons_brIf_loop_self
      bt (.loopK :: frames) ws_mid s_mid kst_mid layout R_mid
      h_nb_m h_nh_m h_nbk_m rfl
      ws'_mid s'_mid postOps hw_mid hl_mid
  exact preservation_evalInstrs_cons_i32Const
    bt (.loopK :: frames) ws_b s_b kst_b layout R_b
    h_nb_b h_nh_b h_nbk_b
    0 [.brIf 0] rest_preserves
    ws'_b s'_b bodyOps hw_b hl_b

-- ════════════════════════════════════════════════════════════════════
-- L8.3 cons_wloop — INVESTIGATION RESULTS
--
-- cons_wloop's general claim depends on the iteration bridge: each
-- body iteration must have a known correspondence between WASM
-- branchTarget and IR broke. The investigation surfaced several
-- structural obstacles to a "fall-through" Path A approach analogous
-- to cons_block/cons_wif:
--
-- 1. Empty body: WASM iterLoop exits immediately (body's
--    branchTarget = none triggers the post-arm); IR's opLoop sees
--    body's broke=false and re-iterates forever. Cannot prove
--    preservation — the two semantics diverge.
--
-- 2. Fall-through body (no br/wreturn): same problem. WASM exits
--    after one iter; IR loops forever.
--
-- 3. Body ends with wreturn: WASM halts (st'.halted=true), iterLoop
--    matches branchTarget=none → runs post → post is short-circuited
--    by halted. Final: ws' = body_state with halted=true. IR: wreturn
--    emits no IR, body's broke=false → IR loops forever. Mismatch.
--
-- 4. Body ends with `br_if 0 c=0_unconditional` (e.g., `[i32Const 0,
--    brIf 0]`): WASM falls through after the brIf (cond=0), iterLoop
--    exits, runs post. IR: brIf_loop_self emits a `.branch cond_bool
--    [] [.breakOp]` — cond_bool=false → picks elseOps=[.breakOp] →
--    broke=true. opLoop sees broke after one iter → exits, runs post.
--    THIS CASE WORKS. But it's a degenerate (single-iter-only) wloop
--    and exercises the iteration bridge in its simplest form.
--
-- 5. Body ends with `br_if 0 c=runtime_value`: WASM iterates while
--    c≠0, exits when c=0. IR: brIf_loop_self emits the cast+branch;
--    on c≠0 sets branchTarget=some 0, WASM clears it and iterates;
--    on c=0 falls through, IR sets broke=true → exits. Each
--    iteration's bridge is given by brIf_loop_self_bridge. The
--    general claim composes iteration counts.
--
-- The general claim (case 5) requires the iteration bridge invariant
-- — genuinely novel content from §4b of the L8.5 endgame doc. Estimated
-- 2-3 sessions of focused work.
--
-- Case 4 (degenerate single-iter) is achievable in one session if a
-- caller-supplied "exits-on-first-iter" hypothesis is acceptable.
-- This commit ships the investigation note as the cons_wloop
-- placeholder; the actual proof lands when the bridging invariant
-- arrives.
-- ════════════════════════════════════════════════════════════════════

-- ════════════════════════════════════════════════════════════════════
-- L10 — framework_preservation (straight-line kernels)
--
-- Top-level composition theorem. Scope: kernels with no structured
-- control (block / wloop / wif / br / brIf / wreturn). Pure
-- straight-line code over the per-op bridged alphabet.
--
-- `StraightLineInstr` admits the constructor set that the per-op
-- cons bridge variants in this module cover, with side conditions
-- normalised so the framework theorem can dispatch cleanly:
--
--  * Stack-pure / locals-pure: nop, drop, localGet (both arms),
--    localSet, localTee
--  * Numeric: i32Const, the 10 i32 binops (Add / Sub / Mul / And /
--    Or / Xor / ShrU / DivU / RemU / Shl-nonbuffer), the 6 unsigned
--    cmps (Eq / Ne / LtU / LeU / GtU / GeU)
--  * Memory: i32Load, i32Store (buffer-access path)
--
-- Excluded: every form of control transfer (the body's framework
-- claim depends on the bridging invariant, not yet landed). For
-- kernels that use loops, the next-tier framework theorem will
-- compose cons_wloop + the bridging invariant.
-- ════════════════════════════════════════════════════════════════════

/-- The kernel-body alphabet admitted by the L10 v0.1 framework
    theorem. Restricted to per-op bridge variants whose signatures
    have no side conditions beyond the standard
    (R, h_no_branch, h_no_halt, h_kst_no_broke, rest, IH) tuple.
    Excluded: every form of structured control, buffer-pattern
    folds (i32Add, i32Shl), buffer-typed localGet, and the heavy
    memory ops (i32Load, i32Store — they take h_stack / h_offset /
    h_in_bounds packages). Those ship as separate non-framework
    theorems pending a richer well-formedness predicate. -/
def StraightLineInstr : WasmInstr → Prop
  | .nop                  => True
  | .drop                 => True
  | .i32Const _           => True
  | .localGet _           => True
  | .localSet _           => True
  | .localTee _           => True
  | .i32Add               => True
  | .i32Sub               => True
  | .i32Mul               => True
  | .i32And               => True
  | .i32Or                => True
  | .i32Xor               => True
  | .i32Shl               => True
  | .i32ShrU              => True
  | .i32DivU              => True
  | .i32RemU              => True
  | .i32Eq                => True
  | .i32Ne                => True
  | .i32LtU               => True
  | .i32LeU               => True
  | .i32GtU               => True
  | .i32GeU               => True
  | .i32Load offset _     => offset = 0
  | .i32Store offset _    => offset = 0
  | _                     => False

/-- Kernel-input well-formedness for buffer-typed locals.
    Captures the invariant that every local registered as a
    `#[quanta::shared]` buffer-pointer param holds a `wI32` whose
    value equals the buffer slot's `startAddr`.

    For the framework theorem this is taken as a universally-
    quantified hypothesis over every mid-state during evaluation —
    a kernel-input well-formedness assumption discharged by the
    downstream caller. Well-typed kernels satisfy this structurally
    (the lowering refuses to commit non-buffer SymVals into buffer-
    typed locals, so localSet/localTee on buffer-typed locals
    fails at lowering time and the framework's `hl` precondition
    rules out the bad case). -/
def BufferLocalsWellFormed
    (layout : BufferLayout) (ws : WasmState) (s : LowerState) : Prop :=
  ∀ i slot, s.lookupBufferSlot i = some slot →
    ∀ v, ws.locals.get? i = some v →
      ∃ n : UInt32, v = .wI32 n ∧ n.toNat = layout.startAddr slot

/-- Kernel-wide assumption that the lowering state's symbolic
    stack never matches a buffer-fold pattern at the moment an
    i32Add or i32Shl is about to lower. Concretely:

    For i32Add: `s.stack` doesn't have shape
      `.scaledIdx _ _ :: .bufferPtr _ :: _` or
      `.bufferPtr _ :: .scaledIdx _ _ :: _` at the top.
    For i32Shl: `s.stack` doesn't have shape
      `.i32ConstSym _ :: .reg _ _ :: _` at the top.

    Discharged downstream by kernels that don't use buffer
    pointers as arithmetic operands (the common case — buffer
    pointers are read by localGet then consumed by i32Load/Store
    immediately via the buffer-pattern fast-paths). Kernels that
    exercise the buffer-fold paths are out of the
    `framework_preservation_straightLine` scope. -/
def NoBufferPatternStack (s : LowerState) : Prop :=
  (∀ slot base scale rest,
      s.stack ≠ .scaledIdx base scale :: .bufferPtr slot :: rest ∧
      s.stack ≠ .bufferPtr slot :: .scaledIdx base scale :: rest) ∧
  (∀ k base ty rest,
      s.stack ≠ .i32ConstSym k :: .reg base ty :: rest)

/-- Kernel-wide assumption: whenever `s.stack` has a `bufferAccess`
    at the top, the base register's runtime value is in-bounds for
    the slot. Discharged downstream from per-kernel knowledge of
    how addresses are constructed (the buffer-pattern fast-path
    folds typically guarantee this by construction). -/
def LoadAddressesInBounds
    (layout : BufferLayout)
    (s : LowerState) (kst : Quanta.KOps.State) : Prop :=
  ∀ slot base lstk_rest,
    s.stack = .bufferAccess slot base 4 :: lstk_rest →
    ∀ b : UInt32,
      regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
      b.toNat < layout.length slot

/-- Kernel-wide assumption: when an i32Store is about to lower,
    the base register's value is in-bounds for the slot. Same
    shape as `LoadAddressesInBounds` but keyed on the i32Store
    stack pattern (sv_val :: bufferAccess :: lstk_rest). -/
def StoreAddressInBounds
    (layout : BufferLayout)
    (s : LowerState) (kst : Quanta.KOps.State) : Prop :=
  ∀ sv_val slot base lstk_rest,
    s.stack = sv_val :: .bufferAccess slot base 4 :: lstk_rest →
    ∀ b : UInt32,
      regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
      b.toNat < layout.length slot

/-- Kernel-wide assumption: the buffer layout has no slot overlap.
    Required by `i32Store` to ensure that writing to one slot
    doesn't corrupt another. Universal over reachable states at
    the i32Store site (the predicate is per-(s_x, kst_x) because
    the relevant base address comes from kst_x.rf). -/
def StoreLayoutNoOverlap
    (layout : BufferLayout)
    (s : LowerState) (kst : Quanta.KOps.State) : Prop :=
  ∀ sv_val slot base lstk_rest,
    s.stack = sv_val :: .bufferAccess slot base 4 :: lstk_rest →
    ∀ b : UInt32,
      regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
      ∀ slot' idx',
        idx' < layout.length slot' →
        (slot', idx') ≠ (slot, b.toNat) →
        layout.startAddr slot + b.toNat * 4 + 4 ≤ layout.startAddr slot' + idx' * 4 ∨
        layout.startAddr slot' + idx' * 4 + 4 ≤ layout.startAddr slot + b.toNat * 4

/-- `instrs : List WasmInstr` is straight-line if every element is. -/
def StraightLineInstrs : List WasmInstr → Prop
  | []        => True
  | i :: rest => StraightLineInstr i ∧ StraightLineInstrs rest

/-- Syntactic predicate: an instruction whose lowering is IR-empty
    and stack-pure (no state change beyond a no-op on the
    LowerState). Admitting an op here requires proving the four
    peel-step lemmas (defined later in the file). -/
def IsIrEmptyOp : WasmInstr → Prop
  | .nop => True
  | _    => False

/-- List of IR-empty / stack-pure instructions. Admitted as a
    wloop-body prefix. -/
def IsIrEmptyPrefix : List WasmInstr → Prop
  | [] => True
  | i :: rest => IsIrEmptyOp i ∧ IsIrEmptyPrefix rest

/-- Body shape admitted by the L10v7 framework's wloop arm:
    a list of IR-empty prefix instructions followed by the
    single-iter-exit suffix `[.i32Const 0, .brIf 0]`. -/
def WloopBodyShape (body : List WasmInstr) : Prop :=
  ∃ pref : List WasmInstr,
    IsIrEmptyPrefix pref ∧ body = pref ++ [.i32Const 0, .brIf 0]

/-- Kernel-body well-formedness for the L10v7 framework.
    Admits a flat sequence of `StraightLineInstr` ops interleaved
    with `wloop 0 :: <body> ++ [.wend]` segments where the body
    matches `WloopBodyShape`.

    Defined as `Type`-valued (not Prop) so a `depth` measure can be
    extracted for the framework theorem's fuel bound. -/
inductive KernelInstrs : List WasmInstr → Type
  | empty : KernelInstrs []
  | sl_cons {i : WasmInstr} {rest : List WasmInstr} :
      StraightLineInstr i →
      KernelInstrs rest →
      KernelInstrs (i :: rest)
  | wloop_cons {rest body post : List WasmInstr} :
      splitAtEnd rest = some (body, post) →
      WloopBodyShape body →
      KernelInstrs post →
      KernelInstrs (.wloop 0 :: rest)

/-- Maximum wloop nesting depth in a KernelInstrs proof.
    Used as a fuel bound for the L10v7 framework theorem: outer fuel
    must be ≥ 2 + depth so the IH recursion through nested wloops
    has enough fuel at every level. -/
def KernelInstrs.depth : ∀ {instrs : List WasmInstr},
    KernelInstrs instrs → Nat
  | _, .empty => 0
  | _, .sl_cons _ rest_wf => rest_wf.depth
  | _, .wloop_cons _ _ post_wf => 1 + post_wf.depth

/-- L10 — framework_preservation for straight-line kernels.

    Composes the per-op bridge variants for every constructor in
    `StraightLineInstr`. Inducts on `instrs`; the cons case
    dispatches on the head constructor and applies the matching
    `cons_<op>_bridge` theorem, threading the recursive IH.

    Conclusion: existence of a `kst'` matching `ws'` under the
    refinement, plus the bridge clauses (which for straight-line
    kernels are vacuously true since branchTarget never gets set
    and broke never gets toggled). -/
theorem framework_preservation_straightLine
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    -- Kernel-input hypothesis: at every mid-state reachable
    -- during the kernel's evaluation, buffer-typed locals hold
    -- their slot start address. Discharged downstream by a
    -- syntactic well-typedness check (no localSet on buffer
    -- locals) plus a kernel-input precondition. Abstracted
    -- over (ws_x, s_x) so the recursion can pass it through
    -- without proving bufferSlot preservation per-op.
    (h_buf_locals : ∀ (ws_x : WasmState) (s_x : LowerState),
        BufferLocalsWellFormed layout ws_x s_x)
    -- Kernel-wide hypothesis: the symbolic stack never matches
    -- the buffer-fold patterns at any reachable state. Universal
    -- over s_x for the same reason as h_buf_locals.
    (h_no_buf_stack : ∀ (s_x : LowerState), NoBufferPatternStack s_x)
    -- Kernel-wide hypothesis: at any reachable (s_x, kst_x), if the
    -- symbolic stack has bufferAccess at the top, the base register
    -- in kst_x's regfile holds an in-bounds index for the slot.
    -- Universal over reachable states.
    (h_load_bounds : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        LoadAddressesInBounds layout s_x kst_x)
    -- Kernel-wide hypothesis: at any reachable (s_x, kst_x), the
    -- store-pattern stack has its base reg in-bounds.
    (h_store_bounds : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        StoreAddressInBounds layout s_x kst_x)
    -- Kernel-wide hypothesis: the buffer layout has no overlapping
    -- slots. Needed for i32Store to ensure stores don't corrupt
    -- other slots.
    (h_store_layout : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        StoreLayoutNoOverlap layout s_x kst_x)
    (instrs : List WasmInstr)
    (h_wf : StraightLineInstrs instrs)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws instrs = some ws')
    (hl : lowerInstrs fuel frames s instrs = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  induction instrs generalizing ws s kst ws' s' ops with
  | nil =>
    -- Empty body: evalInstrs and lowerInstrs both return the input.
    simp only [evalInstrs] at hw
    simp only [lowerInstrs] at hl
    have h_ws_eq : ws' = ws := ((Option.some.injEq _ _).mp hw).symm
    have h_pair : (s, ([] : List KernelOp)) = (s', ops) :=
      (Option.some.injEq _ _).mp hl
    have h_s_eq : s = s' := congrArg Prod.fst h_pair
    have h_ops_eq : ([] : List KernelOp) = ops := congrArg Prod.snd h_pair
    refine ⟨kst, 0, ?_, ?_, ?_⟩
    · rw [← h_ops_eq]; simp [evalOps]
    · rw [h_ws_eq, ← h_s_eq]; exact R
    · refine ⟨?_, ?_⟩
      · intro d hd
        rw [h_ws_eq] at hd
        rw [h_no_branch] at hd
        exact (Option.noConfusion hd)
      · intro _
        exact h_kst_no_broke
  | cons i rest IH =>
    -- Recursive case: dispatch on i. Build the rest-IH from the
    -- list-level IH (paramerterised over the mid-state).
    obtain ⟨h_wf_head, h_wf_rest⟩ := h_wf
    have preservation_rest_bridge :
        ∀ {ws_mid : WasmState} {s_mid : LowerState}
          {kst_mid : Quanta.KOps.State}
          (_R_mid : Refines ws_mid s_mid kst_mid layout)
          (_h_nb_mid : ws_mid.branchTarget = none)
          (_h_nh_mid : ws_mid.halted = false)
          (_h_nbk_mid : kst_mid.broke = false)
          {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
          (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
          (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
        ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
          evalOps F kst_mid postOps = some kst'_mid ∧
          Refines ws'_mid s'_mid kst'_mid layout ∧
          BridgeClauses ws'_mid kst'_mid := by
      intro ws_mid s_mid kst_mid R_mid h_nb_mid h_nh_mid h_nbk_mid
            ws'_mid s'_mid postOps hw_mid hl_mid
      exact IH (ws := ws_mid) (s := s_mid) (kst := kst_mid)
        R_mid h_nb_mid h_nh_mid h_nbk_mid h_wf_rest
        ws'_mid s'_mid postOps hw_mid hl_mid
    -- Dispatch on head constructor.
    cases i with
    | nop =>
        exact preservation_evalInstrs_cons_nop_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          rest preservation_rest_bridge ws' s' ops hw hl
    | drop =>
        exact preservation_evalInstrs_cons_drop_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          rest preservation_rest_bridge ws' s' ops hw hl
    | i32Const n =>
        exact preservation_evalInstrs_cons_i32Const_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          n rest preservation_rest_bridge ws' s' ops hw hl
    | localGet idx =>
        -- Dispatch on s.lookupBufferSlot idx.
        cases h_lookup : s.lookupBufferSlot idx with
        | none =>
            exact preservation_evalInstrs_cons_localGet_bridge
              fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
              idx h_lookup rest preservation_rest_bridge ws' s' ops hw hl
        | some slot =>
            have h_loc_buf := h_buf_locals ws s idx slot h_lookup
            exact preservation_evalInstrs_cons_localGet_bufferSlot_bridge
              fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
              idx slot h_lookup h_loc_buf
              rest preservation_rest_bridge ws' s' ops hw hl
    | localSet idx =>
        exact preservation_evalInstrs_cons_localSet_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          idx rest preservation_rest_bridge ws' s' ops hw hl
    | localTee idx =>
        exact preservation_evalInstrs_cons_localTee_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          idx rest preservation_rest_bridge ws' s' ops hw hl
    | i32Add =>
        exact preservation_evalInstrs_cons_i32Add_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          (h_no_buf_stack s).left
          rest preservation_rest_bridge ws' s' ops hw hl
    | i32Shl =>
        exact preservation_evalInstrs_cons_i32Shl_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          (h_no_buf_stack s).right
          rest preservation_rest_bridge ws' s' ops hw hl
    | i32Sub =>
        exact preservation_evalInstrs_cons_i32Sub_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          rest preservation_rest_bridge ws' s' ops hw hl
    | i32Mul =>
        exact preservation_evalInstrs_cons_i32Mul_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          rest preservation_rest_bridge ws' s' ops hw hl
    | i32And =>
        exact preservation_evalInstrs_cons_i32And_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          rest preservation_rest_bridge ws' s' ops hw hl
    | i32Or =>
        exact preservation_evalInstrs_cons_i32Or_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          rest preservation_rest_bridge ws' s' ops hw hl
    | i32Xor =>
        exact preservation_evalInstrs_cons_i32Xor_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          rest preservation_rest_bridge ws' s' ops hw hl
    | i32ShrU =>
        exact preservation_evalInstrs_cons_i32ShrU_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          rest preservation_rest_bridge ws' s' ops hw hl
    | i32DivU =>
        exact preservation_evalInstrs_cons_i32DivU_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          rest preservation_rest_bridge ws' s' ops hw hl
    | i32RemU =>
        exact preservation_evalInstrs_cons_i32RemU_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          rest preservation_rest_bridge ws' s' ops hw hl
    | i32Eq =>
        exact preservation_evalInstrs_cons_i32Eq_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          rest preservation_rest_bridge ws' s' ops hw hl
    | i32Ne =>
        exact preservation_evalInstrs_cons_i32Ne_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          rest preservation_rest_bridge ws' s' ops hw hl
    | i32LtU =>
        exact preservation_evalInstrs_cons_i32LtU_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          rest preservation_rest_bridge ws' s' ops hw hl
    | i32LeU =>
        exact preservation_evalInstrs_cons_i32LeU_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          rest preservation_rest_bridge ws' s' ops hw hl
    | i32GtU =>
        exact preservation_evalInstrs_cons_i32GtU_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          rest preservation_rest_bridge ws' s' ops hw hl
    | i32GeU =>
        exact preservation_evalInstrs_cons_i32GeU_bridge
          fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
          rest preservation_rest_bridge ws' s' ops hw hl
    | i32Load offset align =>
        -- StraightLineInstr enforces offset = 0.
        have h_offset : offset = 0 := h_wf_head
        -- Extract the bufferAccess shape via lowerI32Load. The
        -- lowering succeeded (hl), so s.stack must have form
        -- `.bufferAccess slot base 4 :: lstk_rest`. Case on s.stack.
        rcases h_stk : s.stack with _ | ⟨sv, lstk⟩
        · -- Empty stack: lowerI32Load returns none → contradicts hl.
          simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
        cases sv with
        | reg _ _           =>
            simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
        | i32ConstSym _     =>
            simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
        | bufferPtr _       =>
            simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
        | scaledIdx _ _     =>
            simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
        | bufferAccess slot base scale =>
            match scale with
            | 4 =>
                have h_load_bnd :
                    ∀ b : UInt32,
                      regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
                      b.toNat < layout.length slot :=
                  h_load_bounds s kst slot base lstk h_stk
                exact preservation_evalInstrs_cons_i32Load_bridge
                  fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
                  slot base lstk offset align h_stk h_offset h_load_bnd
                  rest preservation_rest_bridge ws' s' ops hw hl
            | 0 =>
                simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
            | 1 =>
                simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
            | 2 =>
                simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
            | 3 =>
                simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
            | n + 5 =>
                simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
    | i32Store offset align =>
        have h_offset : offset = 0 := h_wf_head
        -- s.stack = sv_val :: .bufferAccess slot base 4 :: lstk_rest.
        rcases h_stk : s.stack with _ | ⟨sv_val, lstk1⟩
        · simp [lowerInstrs, lowerInstr, lowerI32Store, LowerState.popSym, h_stk] at hl
        rcases h_stk2 : lstk1 with _ | ⟨sv2, lstk_rest⟩
        · simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                LowerState.popSym, h_stk2] at hl
        cases sv2 with
        | reg _ _ =>
            simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                  LowerState.popSym, h_stk2] at hl
        | i32ConstSym _ =>
            simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                  LowerState.popSym, h_stk2] at hl
        | bufferPtr _ =>
            simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                  LowerState.popSym, h_stk2] at hl
        | scaledIdx _ _ =>
            simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                  LowerState.popSym, h_stk2] at hl
        | bufferAccess slot base scale =>
            match scale with
            | 4 =>
                have h_full_stk :
                    s.stack = sv_val :: .bufferAccess slot base 4 :: lstk_rest := by
                  rw [h_stk, h_stk2]
                have h_in_bounds := h_store_bounds s kst sv_val slot base lstk_rest
                  h_full_stk
                have h_no_overlap := h_store_layout s kst sv_val slot base lstk_rest
                  h_full_stk
                exact preservation_evalInstrs_cons_i32Store_bridge
                  fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
                  sv_val slot base lstk_rest offset align
                  h_full_stk h_offset h_in_bounds h_no_overlap
                  rest preservation_rest_bridge ws' s' ops hw hl
            | 0 =>
                simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                      LowerState.popSym, h_stk2] at hl
            | 1 =>
                simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                      LowerState.popSym, h_stk2] at hl
            | 2 =>
                simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                      LowerState.popSym, h_stk2] at hl
            | 3 =>
                simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                      LowerState.popSym, h_stk2] at hl
            | n + 5 =>
                simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                      LowerState.popSym, h_stk2] at hl
    -- All other constructors are excluded by StraightLineInstr.
    -- The `h_wf_head : StraightLineInstr i = True` is `False` for
    -- these, contradicting the hypothesis.
    | i64Const _ => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Const _ => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f64Const _ => exact absurd h_wf_head (by simp [StraightLineInstr])
    | i32DivS => exact absurd h_wf_head (by simp [StraightLineInstr])
    | i32RemS => exact absurd h_wf_head (by simp [StraightLineInstr])
    | i32ShrS => exact absurd h_wf_head (by simp [StraightLineInstr])
    | i32LtS => exact absurd h_wf_head (by simp [StraightLineInstr])
    | i32GtS => exact absurd h_wf_head (by simp [StraightLineInstr])
    | i32LeS => exact absurd h_wf_head (by simp [StraightLineInstr])
    | i32GeS => exact absurd h_wf_head (by simp [StraightLineInstr])
    | i32Eqz => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Add => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Sub => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Mul => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Div => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Eq => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Ne => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Lt => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Gt => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Le => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Ge => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Neg => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Abs => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Sqrt => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Min => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Max => exact absurd h_wf_head (by simp [StraightLineInstr])
    | i32WrapI64 => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32ConvertI32S => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32ConvertI32U => exact absurd h_wf_head (by simp [StraightLineInstr])
    | i32TruncF32S => exact absurd h_wf_head (by simp [StraightLineInstr])
    | i32TruncF32U => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32ReinterpretI32 => exact absurd h_wf_head (by simp [StraightLineInstr])
    | i32ReinterpretF32 => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Load _ _ => exact absurd h_wf_head (by simp [StraightLineInstr])
    | f32Store _ _ => exact absurd h_wf_head (by simp [StraightLineInstr])
    | i32Load8U _ _ => exact absurd h_wf_head (by simp [StraightLineInstr])
    | i32Load8S _ _ => exact absurd h_wf_head (by simp [StraightLineInstr])
    | i32Store8 _ _ => exact absurd h_wf_head (by simp [StraightLineInstr])
    | block _ => exact absurd h_wf_head (by simp [StraightLineInstr])
    | wloop _ => exact absurd h_wf_head (by simp [StraightLineInstr])
    | wif _ => exact absurd h_wf_head (by simp [StraightLineInstr])
    | welse => exact absurd h_wf_head (by simp [StraightLineInstr])
    | wend => exact absurd h_wf_head (by simp [StraightLineInstr])
    | br _ => exact absurd h_wf_head (by simp [StraightLineInstr])
    | brIf _ => exact absurd h_wf_head (by simp [StraightLineInstr])
    | wreturn => exact absurd h_wf_head (by simp [StraightLineInstr])
    | call _ => exact absurd h_wf_head (by simp [StraightLineInstr])
    | wselect => exact absurd h_wf_head (by simp [StraightLineInstr])
    | unreachable => exact absurd h_wf_head (by simp [StraightLineInstr])
    | unsupported _ => exact absurd h_wf_head (by simp [StraightLineInstr])

-- ════════════════════════════════════════════════════════════════════
-- L10v6 — kernel framework admitting a wloop head
--
-- Composes `cons_wloop_singleIterExit` (head) with
-- `framework_preservation_straightLine` (post). Caller supplies the
-- four body IHs (the same shape cons_wloop_singleIterExit requires)
-- and the five kernel-wide well-formedness predicates the
-- straight-line framework needs.
--
-- Kernel shape: `.wloop 0 :: rest` where `splitAtEnd rest = some
-- (body, post)` and `post` is `StraightLineInstrs`.
-- ════════════════════════════════════════════════════════════════════

/-- Framework variant admitting a single wloop head whose body exits
    on its first iteration via `broke = true` (the case covered by
    `cons_wloop_singleIterExit`). Post is straight-line; its IH is
    discharged via `framework_preservation_straightLine`. -/
theorem framework_preservation_wloopThenStraightLine
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    -- Kernel-wide hypotheses for the straight-line post.
    (h_buf_locals : ∀ (ws_x : WasmState) (s_x : LowerState),
        BufferLocalsWellFormed layout ws_x s_x)
    (h_no_buf_stack : ∀ (s_x : LowerState), NoBufferPatternStack s_x)
    (h_load_bounds : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        LoadAddressesInBounds layout s_x kst_x)
    (h_store_bounds : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        StoreAddressInBounds layout s_x kst_x)
    (h_store_layout : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        StoreLayoutNoOverlap layout s_x kst_x)
    -- Wloop body + post split.
    (rest : List WasmInstr)
    (body post : List WasmInstr)
    (h_split : splitAtEnd rest = some (body, post))
    (h_post_wf : StraightLineInstrs post)
    -- Body IHs (same shape as cons_wloop_singleIterExit).
    (body_preserves : ∀ {ws_b : WasmState} {s_b : LowerState}
        {kst_b : Quanta.KOps.State}
        (_R_b : Refines ws_b s_b kst_b layout)
        (_h_nb_b : ws_b.branchTarget = none)
        (_h_nh_b : ws_b.halted = false)
        (_h_nbk_b : kst_b.broke = false)
        {ws'_b : WasmState} {s'_b : LowerState} {bodyOps : List KernelOp}
        (_hw_b : evalInstrs fuel ws_b body = some ws'_b)
        (_hl_b : lowerInstrs fuel (.loopK :: frames) s_b body = some (s'_b, bodyOps)),
      ∃ (kst'_b : Quanta.KOps.State) (F : Nat),
        evalOps F kst_b bodyOps = some kst'_b ∧
        Refines ws'_b s'_b kst'_b layout)
    (body_falls_through : ∀ {ws_b : WasmState} {s_b : LowerState}
        {kst_b : Quanta.KOps.State} {ws'_b : WasmState} {s'_b : LowerState}
        {bodyOps : List KernelOp}
        (_R_b : Refines ws_b s_b kst_b layout)
        (_h_nb_b : ws_b.branchTarget = none)
        (_h_nh_b : ws_b.halted = false)
        (_h_nbk_b : kst_b.broke = false)
        (_hw_b : evalInstrs fuel ws_b body = some ws'_b)
        (_hl_b : lowerInstrs fuel (.loopK :: frames) s_b body = some (s'_b, bodyOps)),
      ws'_b.branchTarget = none ∧ ws'_b.halted = false ∧
      ws'_b.locals = ws_b.locals ∧ ws'_b.stack = ws_b.stack ∧
      ws'_b.mem = ws_b.mem ∧
      s'_b.localReg = s_b.localReg ∧ s'_b.localTy = s_b.localTy ∧
      s'_b.stack = s_b.stack ∧ s'_b.bufferSlots = s_b.bufferSlots)
    (body_exits_with_broke : ∀ {ws_b : WasmState} {s_b : LowerState}
        {kst_b : Quanta.KOps.State} {ws'_b : WasmState} {s'_b : LowerState}
        {bodyOps : List KernelOp} {kst'_b : Quanta.KOps.State} {F_b : Nat}
        (_R_b : Refines ws_b s_b kst_b layout)
        (_h_nb_b : ws_b.branchTarget = none)
        (_h_nh_b : ws_b.halted = false)
        (_h_nbk_b : kst_b.broke = false)
        (_hw_b : evalInstrs fuel ws_b body = some ws'_b)
        (_hl_b : lowerInstrs fuel (.loopK :: frames) s_b body = some (s'_b, bodyOps))
        (_h_ev_b : evalOps F_b kst_b bodyOps = some kst'_b),
      kst'_b.broke = true)
    (body_lowering_preserves : ∀ {s_b s'_b : LowerState} {bodyOps : List KernelOp},
        lowerInstrs fuel (.loopK :: frames) s_b body = some (s'_b, bodyOps) →
        s'_b.localReg = s_b.localReg ∧ s'_b.localTy = s_b.localTy ∧
        s'_b.stack = s_b.stack ∧ s'_b.bufferSlots = s_b.bufferSlots ∧
        s'_b.currentReg = s_b.currentReg ∧
        s_b.nextReg ≤ s'_b.nextReg)
    (h_fuel_ge_2 : fuel ≥ 2)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs (fuel + 1) ws (.wloop 0 :: rest) = some ws')
    (hl : lowerInstrs (fuel + 1) frames s (.wloop 0 :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  -- Build the post_preserves IH for cons_wloop_singleIterExit by
  -- delegating to framework_preservation_straightLine.
  have post_preserves :
      ∀ {ws_p : WasmState} {s_p : LowerState}
        {kst_p : Quanta.KOps.State}
        (_R_p : Refines ws_p s_p kst_p layout)
        (_h_nb_p : ws_p.branchTarget = none)
        (_h_nh_p : ws_p.halted = false)
        (_h_nbk_p : kst_p.broke = false)
        {ws'_p : WasmState} {s'_p : LowerState} {postOps : List KernelOp}
        (_hw_p : evalInstrs fuel ws_p post = some ws'_p)
        (_hl_p : lowerInstrs fuel frames s_p post = some (s'_p, postOps)),
      ∃ (kst'_p : Quanta.KOps.State) (F : Nat),
        evalOps F kst_p postOps = some kst'_p ∧
        Refines ws'_p s'_p kst'_p layout ∧
        BridgeClauses ws'_p kst'_p := by
    intro ws_p s_p kst_p R_p h_nb_p h_nh_p h_nbk_p
          ws'_p s'_p postOps hw_p hl_p
    exact framework_preservation_straightLine
      fuel frames ws_p s_p kst_p layout R_p h_nb_p h_nh_p h_nbk_p
      h_buf_locals h_no_buf_stack h_load_bounds h_store_bounds h_store_layout
      post h_post_wf ws'_p s'_p postOps hw_p hl_p
  -- Apply cons_wloop_singleIterExit with the discharged post IH.
  exact preservation_evalInstrs_cons_wloop_singleIterExit
    frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
    fuel rest body post h_split
    body_preserves body_falls_through body_exits_with_broke
    body_lowering_preserves
    post_preserves h_fuel_ge_2
    ws' s' ops hw hl

-- ════════════════════════════════════════════════════════════════════
-- First end-to-end wloop kernel preservation theorem
--
-- Kernel shape: `.wloop 0 :: [.i32Const 0, .brIf 0] ++ [.wend] ++ post`
-- where `post` is straight-line. The wloop body is the simplest
-- single-iteration exit pattern. All four body IHs are discharged
-- inline from the concrete-body lemmas; the post comes from
-- `framework_preservation_straightLine` via the L10v6 wrapper.
-- ════════════════════════════════════════════════════════════════════

/-- First end-to-end concrete wloop kernel preservation theorem.
    No caller body hypotheses — only the five kernel-wide
    well-formedness predicates the straight-line framework needs.
    The wloop body is hard-coded to `[.i32Const 0, .brIf 0]`,
    exercising the iteration-bridge mechanism on the simplest
    single-iter-exit shape. -/
theorem framework_preservation_const0_brIf0_wloop_then_straightLine
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (h_buf_locals : ∀ (ws_x : WasmState) (s_x : LowerState),
        BufferLocalsWellFormed layout ws_x s_x)
    (h_no_buf_stack : ∀ (s_x : LowerState), NoBufferPatternStack s_x)
    (h_load_bounds : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        LoadAddressesInBounds layout s_x kst_x)
    (h_store_bounds : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        StoreAddressInBounds layout s_x kst_x)
    (h_store_layout : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        StoreLayoutNoOverlap layout s_x kst_x)
    (post : List WasmInstr)
    (h_post_wf : StraightLineInstrs post)
    (h_fuel_ge_2 : fuel ≥ 2)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs (fuel + 1) ws
            (.wloop 0 :: ([.i32Const 0, .brIf 0] ++ [.wend] ++ post))
            = some ws')
    (hl : lowerInstrs (fuel + 1) frames s
            (.wloop 0 :: ([.i32Const 0, .brIf 0] ++ [.wend] ++ post))
            = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  -- splitAtEnd on [.i32Const 0, .brIf 0, .wend] ++ post yields
  -- (body = [.i32Const 0, .brIf 0], post).
  have h_split : splitAtEnd ([.i32Const 0, .brIf 0] ++ [.wend] ++ post)
      = some ([.i32Const 0, .brIf 0], post) := by
    -- splitAtEnd splits on .wend; the canonical case for our kernel
    -- shape returns the expected pair. Verified by direct evaluation.
    show splitAtEnd ([WasmInstr.i32Const 0, WasmInstr.brIf 0, WasmInstr.wend] ++ post)
      = some ([WasmInstr.i32Const 0, WasmInstr.brIf 0], post)
    simp only [List.append_assoc, List.cons_append, List.nil_append]
    -- Unfold splitAtEnd, then walkUntilCloser three times (once per
    -- instruction before .wend).
    unfold splitAtEnd
    simp only [walkUntilCloser]
    -- Each walkUntilCloser step unfolds the head match.
    simp
  exact framework_preservation_wloopThenStraightLine
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
    h_buf_locals h_no_buf_stack h_load_bounds h_store_bounds h_store_layout
    ([.i32Const 0, .brIf 0] ++ [.wend] ++ post)
    [.i32Const 0, .brIf 0] post h_split h_post_wf
    -- body_preserves
    (fun {ws_b s_b kst_b} R_b h_nb h_nh h_nbk {ws'_b s'_b bodyOps} hw_b hl_b =>
      const0_brIf0_body_preserves frames layout R_b h_nb h_nh h_nbk hw_b hl_b)
    -- body_falls_through
    (fun {ws_b s_b kst_b ws'_b s'_b bodyOps} _R_b h_nb h_nh _h_nbk hw_b hl_b =>
      const0_brIf0_falls_through frames h_nb h_nh hw_b hl_b)
    -- body_exits_with_broke
    (fun {ws_b s_b kst_b ws'_b s'_b bodyOps kst'_b F_b} _R_b _h_nb _h_nh h_nbk _hw_b hl_b h_ev_b =>
      const0_brIf0_exits_with_broke frames h_nbk hl_b h_ev_b)
    -- body_lowering_preserves
    (fun {s_b s'_b bodyOps} hl_b =>
      const0_brIf0_lowering_preserves frames hl_b)
    h_fuel_ge_2 ws' s' ops hw hl

-- ════════════════════════════════════════════════════════════════════
-- Second concrete wloop body — `[.nop, .i32Const 0, .brIf 0]`
--
-- Adds a leading no-op. Validates that the framework + the iteration-
-- bridge mechanism compose with arbitrary stack-pure / IR-empty
-- prefix instructions in the body. Same lowering output as
-- [.i32Const 0, .brIf 0] (nop emits no IR).
-- ════════════════════════════════════════════════════════════════════

/-- `body_lowering_preserves` discharge for body = `[.nop, .i32Const 0,
    .brIf 0]`. nop is stack-pure / IR-empty, so the lowering result
    matches [.i32Const 0, .brIf 0]. -/
theorem nop_const0_brIf0_lowering_preserves
    {bt : Nat} (frames : List FrameKind)
    {s_b s'_b : LowerState} {bodyOps : List KernelOp}
    (h_lb : lowerInstrs bt (.loopK :: frames) s_b
              [.nop, .i32Const 0, .brIf 0] = some (s'_b, bodyOps)) :
    s'_b.localReg = s_b.localReg ∧ s'_b.localTy = s_b.localTy ∧
    s'_b.stack = s_b.stack ∧ s'_b.bufferSlots = s_b.bufferSlots ∧
    s_b.nextReg ≤ s'_b.nextReg := by
  -- Peel the nop step — it lowers to (s_b, []) so the rest reduces
  -- directly to lowerInstrs on [.i32Const 0, .brIf 0] from s_b.
  have h_lb' : lowerInstrs bt (.loopK :: frames) s_b
                 [.i32Const 0, .brIf 0] = some (s'_b, bodyOps) := by
    rw [lowerInstrs_cons_default bt (.loopK :: frames) s_b .nop
        [.i32Const 0, .brIf 0] rfl] at h_lb
    -- After the rw, h_lb has shape `(lowerInstr s_b .nop).bind ...`
    -- which reduces via `lowerInstr` unfolding + `Option.bind` on
    -- `some (s_b, [])`. Drop the inner bind: the only remaining outer
    -- bind comes from the `pure (s2, ops1 ++ ops2)` wrapper.
    simp only [lowerInstr, Option.bind_eq_bind, Option.some_bind,
               pure, List.nil_append] at h_lb
    -- h_lb now reads `(lowerInstrs bt (.loopK :: frames) s_b
    --   [.i32Const 0, .brIf 0]).bind (fun discr => some (discr.fst, discr.snd))
    -- = some (s'_b, bodyOps)`. Case on the inner result.
    cases h_inner : lowerInstrs bt (.loopK :: frames) s_b
                       [.i32Const 0, .brIf 0] with
    | none =>
        rw [h_inner] at h_lb
        simp at h_lb
    | some pair =>
        rcases pair with ⟨s'', bops⟩
        rw [h_inner] at h_lb
        simp at h_lb
        obtain ⟨h_s, h_ops⟩ := h_lb
        rw [h_s, h_ops]
  exact const0_brIf0_lowering_preserves frames h_lb'

/-- Helper: peel a leading nop from a body's lowering output, exposing
    the underlying `[.i32Const 0, .brIf 0]` lowering. -/
private theorem peel_nop_lowering
    {bt : Nat} (frames : List FrameKind)
    {s_b s'_b : LowerState} {bodyOps : List KernelOp}
    (h_lb : lowerInstrs bt (.loopK :: frames) s_b
              [.nop, .i32Const 0, .brIf 0] = some (s'_b, bodyOps)) :
    lowerInstrs bt (.loopK :: frames) s_b
      [.i32Const 0, .brIf 0] = some (s'_b, bodyOps) := by
  rw [lowerInstrs_cons_default bt (.loopK :: frames) s_b .nop
      [.i32Const 0, .brIf 0] rfl] at h_lb
  simp only [lowerInstr, Option.bind_eq_bind, Option.some_bind,
             pure, List.nil_append] at h_lb
  cases h_inner : lowerInstrs bt (.loopK :: frames) s_b
                     [.i32Const 0, .brIf 0] with
  | none => rw [h_inner] at h_lb; simp at h_lb
  | some pair =>
      rcases pair with ⟨s'', bops⟩
      rw [h_inner] at h_lb
      simp at h_lb
      obtain ⟨h_s, h_ops⟩ := h_lb
      rw [← h_s, ← h_ops]

/-- Helper: peel a leading nop from a body's eval output, exposing
    the underlying `[.i32Const 0, .brIf 0]` eval. -/
private theorem peel_nop_eval
    {bt : Nat}
    {ws_b ws'_b : WasmState}
    (h_no_branch : ws_b.branchTarget = none)
    (h_no_halt : ws_b.halted = false)
    (hw_b : evalInstrs bt ws_b [.nop, .i32Const 0, .brIf 0] = some ws'_b) :
    evalInstrs bt ws_b [.i32Const 0, .brIf 0] = some ws'_b := by
  rw [evalInstrs_cons_default bt ws_b .nop [.i32Const 0, .brIf 0]
      h_no_branch h_no_halt (by simp [isStructuredEval])] at hw_b
  simp only [evalInstr] at hw_b
  exact hw_b

/-- `body_falls_through` for body = `[.nop, .i32Const 0, .brIf 0]`. -/
theorem nop_const0_brIf0_falls_through
    {bt : Nat} (frames : List FrameKind)
    {ws_b : WasmState} {s_b : LowerState}
    {ws'_b : WasmState} {s'_b : LowerState}
    {bodyOps : List KernelOp}
    (h_no_branch : ws_b.branchTarget = none)
    (h_no_halt : ws_b.halted = false)
    (hw_b : evalInstrs bt ws_b [.nop, .i32Const 0, .brIf 0] = some ws'_b)
    (h_lb : lowerInstrs bt (.loopK :: frames) s_b
              [.nop, .i32Const 0, .brIf 0] = some (s'_b, bodyOps)) :
    ws'_b.branchTarget = none ∧ ws'_b.halted = false ∧
    ws'_b.locals = ws_b.locals ∧ ws'_b.stack = ws_b.stack ∧
    ws'_b.mem = ws_b.mem ∧
    s'_b.localReg = s_b.localReg ∧ s'_b.localTy = s_b.localTy ∧
    s'_b.stack = s_b.stack ∧ s'_b.bufferSlots = s_b.bufferSlots := by
  exact const0_brIf0_falls_through frames h_no_branch h_no_halt
    (peel_nop_eval h_no_branch h_no_halt hw_b)
    (peel_nop_lowering frames h_lb)

/-- `body_exits_with_broke` for body = `[.nop, .i32Const 0, .brIf 0]`. -/
theorem nop_const0_brIf0_exits_with_broke
    {bt : Nat} (frames : List FrameKind)
    {s_b : LowerState}
    {kst_b : Quanta.KOps.State} {s'_b : LowerState}
    {bodyOps : List KernelOp} {kst'_b : Quanta.KOps.State} {F_b : Nat}
    (h_kst_no_broke : kst_b.broke = false)
    (h_lb : lowerInstrs bt (.loopK :: frames) s_b
              [.nop, .i32Const 0, .brIf 0] = some (s'_b, bodyOps))
    (h_ev_b : evalOps F_b kst_b bodyOps = some kst'_b) :
    kst'_b.broke = true := by
  exact const0_brIf0_exits_with_broke frames h_kst_no_broke
    (peel_nop_lowering frames h_lb) h_ev_b

/-- `body_preserves` for body = `[.nop, .i32Const 0, .brIf 0]`. -/
theorem nop_const0_brIf0_body_preserves
    {bt : Nat} (frames : List FrameKind)
    {ws_b : WasmState} {s_b : LowerState}
    {kst_b : Quanta.KOps.State}
    (layout : BufferLayout)
    (R_b : Refines ws_b s_b kst_b layout)
    (h_nb_b : ws_b.branchTarget = none)
    (h_nh_b : ws_b.halted = false)
    (h_nbk_b : kst_b.broke = false)
    {ws'_b : WasmState} {s'_b : LowerState} {bodyOps : List KernelOp}
    (hw_b : evalInstrs bt ws_b [.nop, .i32Const 0, .brIf 0] = some ws'_b)
    (hl_b : lowerInstrs bt (.loopK :: frames) s_b
              [.nop, .i32Const 0, .brIf 0] = some (s'_b, bodyOps)) :
    ∃ (kst'_b : Quanta.KOps.State) (F : Nat),
      evalOps F kst_b bodyOps = some kst'_b ∧
      Refines ws'_b s'_b kst'_b layout := by
  exact const0_brIf0_body_preserves frames layout R_b h_nb_b h_nh_b h_nbk_b
    (peel_nop_eval h_nb_b h_nh_b hw_b)
    (peel_nop_lowering frames hl_b)

/-- End-to-end concrete wloop kernel preservation for the
    nop-prefixed body `[.nop, .i32Const 0, .brIf 0]`. Wraps
    `framework_preservation_wloopThenStraightLine` with the four
    nop-prefixed-body discharges, validating that wloop bodies
    composed with leading stack-pure/IR-empty prefix instructions
    proceed identically. -/
theorem framework_preservation_nop_const0_brIf0_wloop_then_straightLine
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (h_buf_locals : ∀ (ws_x : WasmState) (s_x : LowerState),
        BufferLocalsWellFormed layout ws_x s_x)
    (h_no_buf_stack : ∀ (s_x : LowerState), NoBufferPatternStack s_x)
    (h_load_bounds : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        LoadAddressesInBounds layout s_x kst_x)
    (h_store_bounds : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        StoreAddressInBounds layout s_x kst_x)
    (h_store_layout : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        StoreLayoutNoOverlap layout s_x kst_x)
    (post : List WasmInstr)
    (h_post_wf : StraightLineInstrs post)
    (h_fuel_ge_2 : fuel ≥ 2)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs (fuel + 1) ws
            (.wloop 0 :: ([.nop, .i32Const 0, .brIf 0] ++ [.wend] ++ post))
            = some ws')
    (hl : lowerInstrs (fuel + 1) frames s
            (.wloop 0 :: ([.nop, .i32Const 0, .brIf 0] ++ [.wend] ++ post))
            = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  have h_split : splitAtEnd ([.nop, .i32Const 0, .brIf 0] ++ [.wend] ++ post)
      = some ([.nop, .i32Const 0, .brIf 0], post) := by
    show splitAtEnd ([WasmInstr.nop, WasmInstr.i32Const 0, WasmInstr.brIf 0,
                      WasmInstr.wend] ++ post)
      = some ([WasmInstr.nop, WasmInstr.i32Const 0, WasmInstr.brIf 0], post)
    simp only [List.append_assoc, List.cons_append, List.nil_append]
    unfold splitAtEnd
    simp only [walkUntilCloser]
    simp
  exact framework_preservation_wloopThenStraightLine
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
    h_buf_locals h_no_buf_stack h_load_bounds h_store_bounds h_store_layout
    ([.nop, .i32Const 0, .brIf 0] ++ [.wend] ++ post)
    [.nop, .i32Const 0, .brIf 0] post h_split h_post_wf
    (fun {ws_b s_b kst_b} R_b h_nb h_nh h_nbk {ws'_b s'_b bodyOps} hw_b hl_b =>
      nop_const0_brIf0_body_preserves frames layout R_b h_nb h_nh h_nbk hw_b hl_b)
    (fun {ws_b s_b kst_b ws'_b s'_b bodyOps} _R_b h_nb h_nh _h_nbk hw_b hl_b =>
      nop_const0_brIf0_falls_through frames h_nb h_nh hw_b hl_b)
    (fun {ws_b s_b kst_b ws'_b s'_b bodyOps kst'_b F_b} _R_b _h_nb _h_nh h_nbk _hw_b hl_b h_ev_b =>
      nop_const0_brIf0_exits_with_broke frames h_nbk hl_b h_ev_b)
    (fun {s_b s'_b bodyOps} hl_b =>
      nop_const0_brIf0_lowering_preserves frames hl_b)
    h_fuel_ge_2 ws' s' ops hw hl

-- ════════════════════════════════════════════════════════════════════
-- Parametric IR-empty prefix lifting
--
-- Generalizes the peel-delegate pattern: for any list `prefix` of
-- "IR-empty stack-pure" instructions, the four body IHs for
-- `prefix ++ core` reduce to the four body IHs for `core`.
-- IsIrEmptyOp / IsIrEmptyPrefix are defined earlier (next to the
-- straight-line predicates) so KernelInstrs / WloopBodyShape can
-- reference them. Currently admits `.nop`; extensible by adding
-- constructors here and proving the corresponding peel-step
-- lemmas. The wloop kernel builder admits
-- `prefix ++ [.i32Const 0, .brIf 0]` for any well-formed prefix
-- in one shot.
-- ════════════════════════════════════════════════════════════════════

/-- Parametric peel for IR-empty prefix on the lowering side. -/
theorem peel_irEmptyPrefix_lowering
    {bt : Nat} (frames : List FrameKind)
    {pref rest : List WasmInstr}
    (h_prefix : IsIrEmptyPrefix pref)
    {s_b s'_b : LowerState} {bodyOps : List KernelOp}
    (h_lb : lowerInstrs bt frames s_b (pref ++ rest) = some (s'_b, bodyOps)) :
    lowerInstrs bt frames s_b rest = some (s'_b, bodyOps) := by
  induction pref with
  | nil =>
      simp only [List.nil_append] at h_lb
      exact h_lb
  | cons i pre_rest IH =>
      obtain ⟨h_i, h_pre⟩ := h_prefix
      -- Pattern match on i — only .nop is admitted.
      cases i with
      | nop =>
          rw [List.cons_append] at h_lb
          rw [lowerInstrs_cons_default bt frames s_b .nop (pre_rest ++ rest) rfl] at h_lb
          simp only [lowerInstr, Option.bind_eq_bind, Option.some_bind,
                     pure, List.nil_append] at h_lb
          -- h_lb is `(lowerInstrs bt frames s_b (pre_rest ++ rest)).bind ...`.
          cases h_inner : lowerInstrs bt frames s_b (pre_rest ++ rest) with
          | none => rw [h_inner] at h_lb; simp at h_lb
          | some pair =>
              rcases pair with ⟨s'', bops⟩
              rw [h_inner] at h_lb
              simp at h_lb
              obtain ⟨h_s, h_ops⟩ := h_lb
              subst h_s
              subst h_ops
              exact IH h_pre h_inner
      | _ =>
          -- All other constructors fail IsIrEmptyOp.
          all_goals (exfalso; exact h_i)

/-- Parametric peel for IR-empty prefix on the eval side. -/
theorem peel_irEmptyPrefix_eval
    {bt : Nat}
    {pref rest : List WasmInstr}
    (h_prefix : IsIrEmptyPrefix pref)
    {ws_b ws'_b : WasmState}
    (h_no_branch : ws_b.branchTarget = none)
    (h_no_halt : ws_b.halted = false)
    (hw_b : evalInstrs bt ws_b (pref ++ rest) = some ws'_b) :
    evalInstrs bt ws_b rest = some ws'_b := by
  induction pref generalizing ws_b with
  | nil =>
      simp only [List.nil_append] at hw_b
      exact hw_b
  | cons i pre_rest IH =>
      obtain ⟨h_i, h_pre⟩ := h_prefix
      cases i with
      | nop =>
          rw [List.cons_append] at hw_b
          rw [evalInstrs_cons_default bt ws_b .nop (pre_rest ++ rest)
              h_no_branch h_no_halt (by simp [isStructuredEval])] at hw_b
          simp only [evalInstr] at hw_b
          -- After nop: state unchanged. Apply IH on pre_rest.
          exact IH h_pre h_no_branch h_no_halt hw_b
      | _ =>
          all_goals (exfalso; exact h_i)

/-- Parametric `body_lowering_preserves` for body =
    `pref ++ [.i32Const 0, .brIf 0]` with `IsIrEmptyPrefix pref`. -/
theorem irEmptyPrefix_const0_brIf0_lowering_preserves
    {bt : Nat} (frames : List FrameKind)
    {pref : List WasmInstr} (h_prefix : IsIrEmptyPrefix pref)
    {s_b s'_b : LowerState} {bodyOps : List KernelOp}
    (h_lb : lowerInstrs bt (.loopK :: frames) s_b
              (pref ++ [.i32Const 0, .brIf 0]) = some (s'_b, bodyOps)) :
    s'_b.localReg = s_b.localReg ∧ s'_b.localTy = s_b.localTy ∧
    s'_b.stack = s_b.stack ∧ s'_b.bufferSlots = s_b.bufferSlots ∧
    s_b.nextReg ≤ s'_b.nextReg :=
  const0_brIf0_lowering_preserves frames
    (peel_irEmptyPrefix_lowering (.loopK :: frames) h_prefix h_lb)

/-- Parametric `body_falls_through` for body =
    `pref ++ [.i32Const 0, .brIf 0]` with `IsIrEmptyPrefix pref`. -/
theorem irEmptyPrefix_const0_brIf0_falls_through
    {bt : Nat} (frames : List FrameKind)
    {pref : List WasmInstr} (h_prefix : IsIrEmptyPrefix pref)
    {ws_b : WasmState} {s_b : LowerState}
    {ws'_b : WasmState} {s'_b : LowerState}
    {bodyOps : List KernelOp}
    (h_no_branch : ws_b.branchTarget = none)
    (h_no_halt : ws_b.halted = false)
    (hw_b : evalInstrs bt ws_b (pref ++ [.i32Const 0, .brIf 0])
              = some ws'_b)
    (h_lb : lowerInstrs bt (.loopK :: frames) s_b
              (pref ++ [.i32Const 0, .brIf 0]) = some (s'_b, bodyOps)) :
    ws'_b.branchTarget = none ∧ ws'_b.halted = false ∧
    ws'_b.locals = ws_b.locals ∧ ws'_b.stack = ws_b.stack ∧
    ws'_b.mem = ws_b.mem ∧
    s'_b.localReg = s_b.localReg ∧ s'_b.localTy = s_b.localTy ∧
    s'_b.stack = s_b.stack ∧ s'_b.bufferSlots = s_b.bufferSlots :=
  const0_brIf0_falls_through frames h_no_branch h_no_halt
    (peel_irEmptyPrefix_eval h_prefix h_no_branch h_no_halt hw_b)
    (peel_irEmptyPrefix_lowering (.loopK :: frames) h_prefix h_lb)

/-- Parametric `body_exits_with_broke` for body =
    `pref ++ [.i32Const 0, .brIf 0]` with `IsIrEmptyPrefix pref`. -/
theorem irEmptyPrefix_const0_brIf0_exits_with_broke
    {bt : Nat} (frames : List FrameKind)
    {pref : List WasmInstr} (h_prefix : IsIrEmptyPrefix pref)
    {s_b : LowerState}
    {kst_b : Quanta.KOps.State} {s'_b : LowerState}
    {bodyOps : List KernelOp} {kst'_b : Quanta.KOps.State} {F_b : Nat}
    (h_kst_no_broke : kst_b.broke = false)
    (h_lb : lowerInstrs bt (.loopK :: frames) s_b
              (pref ++ [.i32Const 0, .brIf 0]) = some (s'_b, bodyOps))
    (h_ev_b : evalOps F_b kst_b bodyOps = some kst'_b) :
    kst'_b.broke = true :=
  const0_brIf0_exits_with_broke frames h_kst_no_broke
    (peel_irEmptyPrefix_lowering (.loopK :: frames) h_prefix h_lb)
    h_ev_b

/-- Parametric `body_preserves` for body =
    `pref ++ [.i32Const 0, .brIf 0]` with `IsIrEmptyPrefix pref`. -/
theorem irEmptyPrefix_const0_brIf0_body_preserves
    {bt : Nat} (frames : List FrameKind)
    {pref : List WasmInstr} (h_prefix : IsIrEmptyPrefix pref)
    {ws_b : WasmState} {s_b : LowerState}
    {kst_b : Quanta.KOps.State}
    (layout : BufferLayout)
    (R_b : Refines ws_b s_b kst_b layout)
    (h_nb_b : ws_b.branchTarget = none)
    (h_nh_b : ws_b.halted = false)
    (h_nbk_b : kst_b.broke = false)
    {ws'_b : WasmState} {s'_b : LowerState} {bodyOps : List KernelOp}
    (hw_b : evalInstrs bt ws_b (pref ++ [.i32Const 0, .brIf 0])
              = some ws'_b)
    (hl_b : lowerInstrs bt (.loopK :: frames) s_b
              (pref ++ [.i32Const 0, .brIf 0]) = some (s'_b, bodyOps)) :
    ∃ (kst'_b : Quanta.KOps.State) (F : Nat),
      evalOps F kst_b bodyOps = some kst'_b ∧
      Refines ws'_b s'_b kst'_b layout :=
  const0_brIf0_body_preserves frames layout R_b h_nb_b h_nh_b h_nbk_b
    (peel_irEmptyPrefix_eval h_prefix h_nb_b h_nh_b hw_b)
    (peel_irEmptyPrefix_lowering (.loopK :: frames) h_prefix hl_b)

/-- Generalized `walkUntilCloser` acc-shift invariant: appending
    `extra` to the right of the initial acc PREPENDS `extra.reverse`
    to the taken-list output. The walker traverses input identically
    — extras only join the acc-passes-through-reverse path:
    `(eventually-built-acc ++ extra).reverse = extra.reverse ++ taken`.
    The cons-into-acc step preserves the `acc ++ extra` form:
    `(i :: acc) ++ extra = i :: (acc ++ extra)`. -/
private theorem walkUntilCloser_acc_shift
    (l : List WasmInstr) :
    ∀ (n : Nat) (acc extra : List WasmInstr)
      {taken : List WasmInstr} {marker : WasmInstr} {rest : List WasmInstr},
      walkUntilCloser l n acc = some (taken, marker, rest) →
      walkUntilCloser l n (acc ++ extra)
        = some (extra.reverse ++ taken, marker, rest) := by
  induction l with
  | nil =>
      intro n acc extra taken marker rest h_walk
      simp [walkUntilCloser] at h_walk
  | cons i tail IH =>
      intro n acc extra taken marker rest h_walk
      -- Handle the two depth-0 terminating cases first
      -- (.wend at depth 0; .welse at depth 0) before the catch-all.
      cases h_i_eq : i with
      | wend =>
          subst h_i_eq
          match n with
          | 0 =>
              -- walkUntilCloser (.wend :: tail) 0 acc = some (acc.reverse, .wend, tail).
              simp only [walkUntilCloser, Option.some.injEq, Prod.mk.injEq] at h_walk
              obtain ⟨h_t, h_m, h_r⟩ := h_walk
              subst h_t; subst h_m; subst h_r
              -- Goal: walkUntilCloser (.wend :: tail) 0 (acc ++ extra)
              --   = some (extra.reverse ++ acc.reverse, .wend, tail).
              -- (acc ++ extra).reverse = extra.reverse ++ acc.reverse.
              simp [walkUntilCloser, List.reverse_append]
          | k + 1 =>
              -- Catch-all: recurse on tail with n' = (k+1) - 1 = k and
              -- acc' = .wend :: acc.
              simp only [walkUntilCloser] at h_walk
              simp only [walkUntilCloser]
              -- Goal: walkUntilCloser tail k ((.wend :: acc) ++ extra) =
              --       some (extra.reverse ++ taken, marker, rest).
              -- Apply IH at acc' = .wend :: acc.
              show walkUntilCloser tail k (.wend :: (acc ++ extra))
                = some (extra.reverse ++ taken, marker, rest)
              have h_app : (.wend :: acc : List WasmInstr) ++ extra
                              = .wend :: (acc ++ extra) := by simp
              rw [← h_app]
              exact IH k _ extra h_walk
      | welse =>
          subst h_i_eq
          match n with
          | 0 =>
              simp only [walkUntilCloser, Option.some.injEq, Prod.mk.injEq] at h_walk
              obtain ⟨h_t, h_m, h_r⟩ := h_walk
              subst h_t; subst h_m; subst h_r
              simp [walkUntilCloser, List.reverse_append]
          | k + 1 =>
              simp only [walkUntilCloser] at h_walk
              simp only [walkUntilCloser]
              show walkUntilCloser tail (k + 1) (.welse :: (acc ++ extra))
                = some (extra.reverse ++ taken, marker, rest)
              have h_app : (.welse :: acc : List WasmInstr) ++ extra
                              = .welse :: (acc ++ extra) := by simp
              rw [← h_app]
              exact IH (k + 1) _ extra h_walk
      | block bn =>
          subst h_i_eq
          simp only [walkUntilCloser] at h_walk
          simp only [walkUntilCloser]
          show walkUntilCloser tail (n + 1) (.block bn :: (acc ++ extra))
            = some (extra.reverse ++ taken, marker, rest)
          have h_app : (.block bn :: acc : List WasmInstr) ++ extra
                          = .block bn :: (acc ++ extra) := by simp
          rw [← h_app]
          exact IH (n + 1) _ extra h_walk
      | wloop ln =>
          subst h_i_eq
          simp only [walkUntilCloser] at h_walk
          simp only [walkUntilCloser]
          show walkUntilCloser tail (n + 1) (.wloop ln :: (acc ++ extra))
            = some (extra.reverse ++ taken, marker, rest)
          have h_app : (.wloop ln :: acc : List WasmInstr) ++ extra
                          = .wloop ln :: (acc ++ extra) := by simp
          rw [← h_app]
          exact IH (n + 1) _ extra h_walk
      | wif fn =>
          subst h_i_eq
          simp only [walkUntilCloser] at h_walk
          simp only [walkUntilCloser]
          show walkUntilCloser tail (n + 1) (.wif fn :: (acc ++ extra))
            = some (extra.reverse ++ taken, marker, rest)
          have h_app : (.wif fn :: acc : List WasmInstr) ++ extra
                          = .wif fn :: (acc ++ extra) := by simp
          rw [← h_app]
          exact IH (n + 1) _ extra h_walk
      | _ =>
          all_goals (
            subst h_i_eq
            simp only [walkUntilCloser] at h_walk
            simp only [walkUntilCloser]
            (first
              | rfl
              | (have h_app : ∀ (j : WasmInstr),
                    (j :: acc : List WasmInstr) ++ extra
                      = j :: (acc ++ extra) := by intro _; simp
                 rw [← h_app]
                 exact IH _ _ extra h_walk)))

/-- Walking through an IR-empty prefix (e.g., a list of nops) at any
    depth `n` extends acc by `pref.reverse` and leaves both depth and
    walker behavior unchanged: the walk reduces to a walk on the tail
    with the prefix-extended accumulator. -/
private theorem walkUntilCloser_irEmptyPrefix
    {pref : List WasmInstr} (h_prefix : IsIrEmptyPrefix pref) :
    ∀ (tail : List WasmInstr) (n : Nat) (acc : List WasmInstr),
      walkUntilCloser (pref ++ tail) n acc
        = walkUntilCloser tail n (pref.reverse ++ acc) := by
  induction pref with
  | nil =>
      intro tail n acc
      simp [walkUntilCloser]
  | cons i pre_rest IH =>
      intro tail n acc
      obtain ⟨h_i, h_pre⟩ := h_prefix
      cases i with
      | nop =>
          simp only [List.cons_append, walkUntilCloser]
          rw [IH h_pre]
          simp
      | _ =>
          all_goals (exfalso; exact h_i)

/-- splitAtEnd on `pref ++ [.i32Const 0, .brIf 0, .wend] ++ post`
    correctly extracts `(pref ++ [.i32Const 0, .brIf 0], post)` for
    any `IsIrEmptyPrefix pref`. Combines walkUntilCloser_irEmptyPrefix
    (peel the prefix) with the literal walk through
    [.i32Const 0, .brIf 0, .wend]. -/
theorem splitAtEnd_irEmptyPrefix_const0_brIf0
    {pref : List WasmInstr} (h_prefix : IsIrEmptyPrefix pref)
    (post : List WasmInstr) :
    splitAtEnd (pref ++ [.i32Const 0, .brIf 0, .wend] ++ post)
      = some (pref ++ [.i32Const 0, .brIf 0], post) := by
  unfold splitAtEnd
  have h_eq : pref ++ [WasmInstr.i32Const 0, WasmInstr.brIf 0, WasmInstr.wend]
                ++ post
            = pref ++ ([WasmInstr.i32Const 0, WasmInstr.brIf 0, WasmInstr.wend]
                ++ post) := by
    simp [List.append_assoc]
  rw [h_eq]
  rw [walkUntilCloser_irEmptyPrefix h_prefix]
  -- walkUntilCloser ([i32Const 0, brIf 0, wend] ++ post) 0 (pref.reverse ++ []).
  simp only [List.append_nil, List.cons_append, List.nil_append, walkUntilCloser]
  -- After three steps: returns some (final_acc.reverse, .wend, post)
  -- where final_acc = .brIf 0 :: .i32Const 0 :: pref.reverse.
  -- final_acc.reverse = pref ++ [.i32Const 0, .brIf 0].
  simp [List.reverse_cons, List.reverse_append, List.reverse_reverse]

/-- Parametric end-to-end wloop kernel preservation for body =
    `pref ++ [.i32Const 0, .brIf 0]` with `IsIrEmptyPrefix pref`.
    Subsumes both concrete catalog entries (`pref = []` gives the
    const0_brIf0 case; `pref = [.nop]` gives the nop_const0_brIf0
    case). Discharges all four wloop-body IHs via the parametric
    irEmptyPrefix_* IHs and the splitAtEnd_irEmptyPrefix_const0_brIf0
    splitter lemma. -/
theorem framework_preservation_irEmptyPrefix_const0_brIf0_wloop_then_straightLine
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (h_buf_locals : ∀ (ws_x : WasmState) (s_x : LowerState),
        BufferLocalsWellFormed layout ws_x s_x)
    (h_no_buf_stack : ∀ (s_x : LowerState), NoBufferPatternStack s_x)
    (h_load_bounds : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        LoadAddressesInBounds layout s_x kst_x)
    (h_store_bounds : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        StoreAddressInBounds layout s_x kst_x)
    (h_store_layout : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        StoreLayoutNoOverlap layout s_x kst_x)
    (pref : List WasmInstr) (h_prefix : IsIrEmptyPrefix pref)
    (post : List WasmInstr)
    (h_post_wf : StraightLineInstrs post)
    (h_fuel_ge_2 : fuel ≥ 2)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs (fuel + 1) ws
            (.wloop 0 :: ((pref ++ [.i32Const 0, .brIf 0]) ++ [.wend] ++ post))
            = some ws')
    (hl : lowerInstrs (fuel + 1) frames s
            (.wloop 0 :: ((pref ++ [.i32Const 0, .brIf 0]) ++ [.wend] ++ post))
            = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  have h_split : splitAtEnd ((pref ++ [.i32Const 0, .brIf 0])
                              ++ [.wend] ++ post)
      = some (pref ++ [.i32Const 0, .brIf 0], post) := by
    have h_eq : (pref ++ [.i32Const 0, .brIf 0]) ++ [.wend] ++ post
              = pref ++ [.i32Const 0, .brIf 0, .wend] ++ post := by
      simp [List.append_assoc]
    rw [h_eq]
    exact splitAtEnd_irEmptyPrefix_const0_brIf0 h_prefix post
  exact framework_preservation_wloopThenStraightLine
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
    h_buf_locals h_no_buf_stack h_load_bounds h_store_bounds h_store_layout
    ((pref ++ [.i32Const 0, .brIf 0]) ++ [.wend] ++ post)
    (pref ++ [.i32Const 0, .brIf 0]) post h_split h_post_wf
    (fun {ws_b s_b kst_b} R_b h_nb h_nh h_nbk {ws'_b s'_b bodyOps} hw_b hl_b =>
      irEmptyPrefix_const0_brIf0_body_preserves frames h_prefix layout R_b
        h_nb h_nh h_nbk hw_b hl_b)
    (fun {ws_b s_b kst_b ws'_b s'_b bodyOps} _R_b h_nb h_nh _h_nbk hw_b hl_b =>
      irEmptyPrefix_const0_brIf0_falls_through frames h_prefix h_nb h_nh
        hw_b hl_b)
    (fun {ws_b s_b kst_b ws'_b s'_b bodyOps kst'_b F_b} _R_b _h_nb _h_nh h_nbk _hw_b hl_b h_ev_b =>
      irEmptyPrefix_const0_brIf0_exits_with_broke frames h_prefix h_nbk
        hl_b h_ev_b)
    (fun {s_b s'_b bodyOps} hl_b =>
      irEmptyPrefix_const0_brIf0_lowering_preserves frames h_prefix hl_b)
    h_fuel_ge_2 ws' s' ops hw hl

-- ════════════════════════════════════════════════════════════════════
-- Concrete wloop kernel theorem catalog
--
-- The following end-to-end kernel preservation theorems are shipped
-- (each with no caller body hypotheses, only the 5 standard
-- kernel-wide well-formedness predicates):
--
-- - framework_preservation_const0_brIf0_wloop_then_straightLine:
--     body = [.i32Const 0, .brIf 0]
-- - framework_preservation_nop_const0_brIf0_wloop_then_straightLine:
--     body = [.nop, .i32Const 0, .brIf 0]
-- - framework_preservation_irEmptyPrefix_const0_brIf0_wloop_then_straightLine:
--     body = pref ++ [.i32Const 0, .brIf 0] for any
--     IsIrEmptyPrefix pref (list of nops). Subsumes the two
--     concrete entries above.
-- - framework_preservation_kernel (L10v7): admits any KernelInstrs
--     body (straight-line ops interleaved with arbitrarily nested
--     wloop segments matching WloopBodyShape). Subsumes all of the
--     above. Fuel constraint is depth-aware.
--
-- All variants produce the same lowered IR (`[.const r0, .cast r1 r0,
-- .branch r1 [] [.breakOp]]`) since IR-empty prefix ops emit no
-- KernelOps. The parametric variant validates the peel-delegate
-- pattern at full generality over IR-empty prefixes.
--
-- All bodies covered have exit semantics: WASM falls through after
-- the brIf 0 (cond=0); IR sets broke=true via the .breakOp arm.
-- opLoop sees broke after one iter and exits to the post-loop
-- straight-line code.
-- ════════════════════════════════════════════════════════════════════

-- ════════════════════════════════════════════════════════════════════
-- L10v7 — framework_preservation_kernel
--
-- Top-level composition theorem admitting `KernelInstrs`: a kernel
-- body composed of `StraightLineInstr` ops and `wloop 0` segments
-- (with bodies matching `WloopBodyShape`). Fuel constraint is
-- depth-aware: `fuel ≥ 2 + KernelInstrs.depth h_wf`, so nested
-- wloops get enough fuel at every recursion level.
--
-- Inducts on `KernelInstrs instrs` (not on the list directly), so
-- the wloop arm gets the splitAtEnd witness and the post IH from
-- the same constructor. Dispatches:
--  * `empty`: nil case (input states pass through).
--  * `sl_cons`: applies the straight-line framework on `[i]` then
--    composes with the recursive IH on rest. Simpler: just route
--    through framework_preservation_straightLine which already
--    handles the per-op dispatch for any straight-line head.
--  * `wloop_cons`: applies cons_wloop_singleIterExit with body IHs
--    from WloopBodyShape + the framework's own IH on post.
-- ════════════════════════════════════════════════════════════════════

/-- L10v7 — framework_preservation_kernel.
    Admits any `KernelInstrs` kernel body. -/
theorem framework_preservation_kernel
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (h_buf_locals : ∀ (ws_x : WasmState) (s_x : LowerState),
        BufferLocalsWellFormed layout ws_x s_x)
    (h_no_buf_stack : ∀ (s_x : LowerState), NoBufferPatternStack s_x)
    (h_load_bounds : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        LoadAddressesInBounds layout s_x kst_x)
    (h_store_bounds : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        StoreAddressInBounds layout s_x kst_x)
    (h_store_layout : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        StoreLayoutNoOverlap layout s_x kst_x)
    (instrs : List WasmInstr)
    (h_wf : KernelInstrs instrs)
    (h_fuel : fuel ≥ 2 + h_wf.depth)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs (fuel + 1) ws instrs = some ws')
    (hl : lowerInstrs (fuel + 1) frames s instrs = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  induction h_wf generalizing fuel ws s kst ws' s' ops with
  | empty =>
      -- evalInstrs and lowerInstrs both return input on [].
      simp only [evalInstrs] at hw
      simp only [lowerInstrs] at hl
      have h_ws_eq : ws' = ws := ((Option.some.injEq _ _).mp hw).symm
      have h_pair : (s, ([] : List KernelOp)) = (s', ops) :=
        (Option.some.injEq _ _).mp hl
      have h_s_eq : s = s' := congrArg Prod.fst h_pair
      have h_ops_eq : ([] : List KernelOp) = ops := congrArg Prod.snd h_pair
      refine ⟨kst, 0, ?_, ?_, ?_⟩
      · rw [← h_ops_eq]; simp [evalOps]
      · rw [h_ws_eq, ← h_s_eq]; exact R
      · refine ⟨?_, ?_⟩
        · intro d hd
          rw [h_ws_eq] at hd
          rw [h_no_branch] at hd
          exact (Option.noConfusion hd)
        · intro _
          exact h_kst_no_broke
  | @sl_cons i rest h_sl _h_rest_wf IH =>
      -- Build the rest IH in the bridge shape that per-op cons bridges
      -- expect, then dispatch on the head constructor via the same
      -- pattern as framework_preservation_straightLine's cons case.
      have preservation_rest_bridge :
          ∀ {ws_mid : WasmState} {s_mid : LowerState}
            {kst_mid : Quanta.KOps.State}
            (_R_mid : Refines ws_mid s_mid kst_mid layout)
            (_h_nb_mid : ws_mid.branchTarget = none)
            (_h_nh_mid : ws_mid.halted = false)
            (_h_nbk_mid : kst_mid.broke = false)
            {ws'_mid : WasmState} {s'_mid : LowerState}
            {postOps : List KernelOp}
            (_hw_mid : evalInstrs (fuel + 1) ws_mid rest = some ws'_mid)
            (_hl_mid : lowerInstrs (fuel + 1) frames s_mid rest
                          = some (s'_mid, postOps)),
          ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
            evalOps F kst_mid postOps = some kst'_mid ∧
            Refines ws'_mid s'_mid kst'_mid layout ∧
            BridgeClauses ws'_mid kst'_mid := by
        intro ws_mid s_mid kst_mid R_mid h_nb_m h_nh_m h_nbk_m
              ws'_mid s'_mid postOps hw_mid hl_mid
        -- sl_cons preserves depth (depth h_wf_rest = depth h_wf for
        -- sl_cons), so h_fuel passes through to IH unchanged.
        exact IH (fuel := fuel) (ws := ws_mid) (s := s_mid) (kst := kst_mid)
          R_mid h_nb_m h_nh_m h_nbk_m h_fuel ws'_mid s'_mid postOps hw_mid hl_mid
      -- Dispatch on head constructor.
      cases i with
      | nop =>
          exact preservation_evalInstrs_cons_nop_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            rest preservation_rest_bridge ws' s' ops hw hl
      | drop =>
          exact preservation_evalInstrs_cons_drop_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            rest preservation_rest_bridge ws' s' ops hw hl
      | i32Const n =>
          exact preservation_evalInstrs_cons_i32Const_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            n rest preservation_rest_bridge ws' s' ops hw hl
      | localGet idx =>
          cases h_lookup : s.lookupBufferSlot idx with
          | none =>
              exact preservation_evalInstrs_cons_localGet_bridge
                (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
                idx h_lookup rest preservation_rest_bridge ws' s' ops hw hl
          | some slot =>
              have h_loc_buf := h_buf_locals ws s idx slot h_lookup
              exact preservation_evalInstrs_cons_localGet_bufferSlot_bridge
                (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
                idx slot h_lookup h_loc_buf
                rest preservation_rest_bridge ws' s' ops hw hl
      | localSet idx =>
          exact preservation_evalInstrs_cons_localSet_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            idx rest preservation_rest_bridge ws' s' ops hw hl
      | localTee idx =>
          exact preservation_evalInstrs_cons_localTee_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            idx rest preservation_rest_bridge ws' s' ops hw hl
      | i32Add =>
          exact preservation_evalInstrs_cons_i32Add_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            (h_no_buf_stack s).left
            rest preservation_rest_bridge ws' s' ops hw hl
      | i32Shl =>
          exact preservation_evalInstrs_cons_i32Shl_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            (h_no_buf_stack s).right
            rest preservation_rest_bridge ws' s' ops hw hl
      | i32Sub =>
          exact preservation_evalInstrs_cons_i32Sub_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            rest preservation_rest_bridge ws' s' ops hw hl
      | i32Mul =>
          exact preservation_evalInstrs_cons_i32Mul_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            rest preservation_rest_bridge ws' s' ops hw hl
      | i32And =>
          exact preservation_evalInstrs_cons_i32And_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            rest preservation_rest_bridge ws' s' ops hw hl
      | i32Or =>
          exact preservation_evalInstrs_cons_i32Or_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            rest preservation_rest_bridge ws' s' ops hw hl
      | i32Xor =>
          exact preservation_evalInstrs_cons_i32Xor_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            rest preservation_rest_bridge ws' s' ops hw hl
      | i32ShrU =>
          exact preservation_evalInstrs_cons_i32ShrU_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            rest preservation_rest_bridge ws' s' ops hw hl
      | i32DivU =>
          exact preservation_evalInstrs_cons_i32DivU_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            rest preservation_rest_bridge ws' s' ops hw hl
      | i32RemU =>
          exact preservation_evalInstrs_cons_i32RemU_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            rest preservation_rest_bridge ws' s' ops hw hl
      | i32Eq =>
          exact preservation_evalInstrs_cons_i32Eq_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            rest preservation_rest_bridge ws' s' ops hw hl
      | i32Ne =>
          exact preservation_evalInstrs_cons_i32Ne_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            rest preservation_rest_bridge ws' s' ops hw hl
      | i32LtU =>
          exact preservation_evalInstrs_cons_i32LtU_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            rest preservation_rest_bridge ws' s' ops hw hl
      | i32LeU =>
          exact preservation_evalInstrs_cons_i32LeU_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            rest preservation_rest_bridge ws' s' ops hw hl
      | i32GtU =>
          exact preservation_evalInstrs_cons_i32GtU_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            rest preservation_rest_bridge ws' s' ops hw hl
      | i32GeU =>
          exact preservation_evalInstrs_cons_i32GeU_bridge
            (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
            rest preservation_rest_bridge ws' s' ops hw hl
      | i32Load offset align =>
          have h_offset : offset = 0 := h_sl
          rcases h_stk : s.stack with _ | ⟨sv, lstk⟩
          · simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
          cases sv with
          | reg _ _ =>
              simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
          | i32ConstSym _ =>
              simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
          | bufferPtr _ =>
              simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
          | scaledIdx _ _ =>
              simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
          | bufferAccess slot base scale =>
              match scale with
              | 4 =>
                  have h_load_bnd :
                      ∀ b : UInt32,
                        regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
                        b.toNat < layout.length slot :=
                    h_load_bounds s kst slot base lstk h_stk
                  exact preservation_evalInstrs_cons_i32Load_bridge
                    (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
                    slot base lstk offset align h_stk h_offset h_load_bnd
                    rest preservation_rest_bridge ws' s' ops hw hl
              | 0 => simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
              | 1 => simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
              | 2 => simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
              | 3 => simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
              | n + 5 => simp [lowerInstrs, lowerInstr, lowerI32Load, h_stk] at hl
      | i32Store offset align =>
          have h_offset : offset = 0 := h_sl
          rcases h_stk : s.stack with _ | ⟨sv_val, lstk1⟩
          · simp [lowerInstrs, lowerInstr, lowerI32Store, LowerState.popSym, h_stk] at hl
          rcases h_stk2 : lstk1 with _ | ⟨sv2, lstk_rest⟩
          · simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                  LowerState.popSym, h_stk2] at hl
          cases sv2 with
          | reg _ _ =>
              simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                    LowerState.popSym, h_stk2] at hl
          | i32ConstSym _ =>
              simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                    LowerState.popSym, h_stk2] at hl
          | bufferPtr _ =>
              simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                    LowerState.popSym, h_stk2] at hl
          | scaledIdx _ _ =>
              simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                    LowerState.popSym, h_stk2] at hl
          | bufferAccess slot base scale =>
              match scale with
              | 4 =>
                  have h_full_stk :
                      s.stack = sv_val :: .bufferAccess slot base 4 :: lstk_rest := by
                    rw [h_stk, h_stk2]
                  have h_in_bounds := h_store_bounds s kst sv_val slot base lstk_rest
                    h_full_stk
                  have h_no_overlap := h_store_layout s kst sv_val slot base lstk_rest
                    h_full_stk
                  exact preservation_evalInstrs_cons_i32Store_bridge
                    (fuel + 1) frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
                    sv_val slot base lstk_rest offset align
                    h_full_stk h_offset h_in_bounds h_no_overlap
                    rest preservation_rest_bridge ws' s' ops hw hl
              | 0 => simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                          LowerState.popSym, h_stk2] at hl
              | 1 => simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                          LowerState.popSym, h_stk2] at hl
              | 2 => simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                          LowerState.popSym, h_stk2] at hl
              | 3 => simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                          LowerState.popSym, h_stk2] at hl
              | n + 5 => simp [lowerInstrs, lowerInstr, lowerI32Store, h_stk,
                              LowerState.popSym, h_stk2] at hl
      -- Excluded constructors (StraightLineInstr is False for them).
      | i64Const _ => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Const _ => exact absurd h_sl (by simp [StraightLineInstr])
      | f64Const _ => exact absurd h_sl (by simp [StraightLineInstr])
      | i32DivS => exact absurd h_sl (by simp [StraightLineInstr])
      | i32RemS => exact absurd h_sl (by simp [StraightLineInstr])
      | i32ShrS => exact absurd h_sl (by simp [StraightLineInstr])
      | i32LtS => exact absurd h_sl (by simp [StraightLineInstr])
      | i32GtS => exact absurd h_sl (by simp [StraightLineInstr])
      | i32LeS => exact absurd h_sl (by simp [StraightLineInstr])
      | i32GeS => exact absurd h_sl (by simp [StraightLineInstr])
      | i32Eqz => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Add => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Sub => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Mul => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Div => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Eq => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Ne => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Lt => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Gt => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Le => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Ge => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Neg => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Abs => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Sqrt => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Min => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Max => exact absurd h_sl (by simp [StraightLineInstr])
      | i32WrapI64 => exact absurd h_sl (by simp [StraightLineInstr])
      | f32ConvertI32S => exact absurd h_sl (by simp [StraightLineInstr])
      | f32ConvertI32U => exact absurd h_sl (by simp [StraightLineInstr])
      | i32TruncF32S => exact absurd h_sl (by simp [StraightLineInstr])
      | i32TruncF32U => exact absurd h_sl (by simp [StraightLineInstr])
      | f32ReinterpretI32 => exact absurd h_sl (by simp [StraightLineInstr])
      | i32ReinterpretF32 => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Load _ _ => exact absurd h_sl (by simp [StraightLineInstr])
      | f32Store _ _ => exact absurd h_sl (by simp [StraightLineInstr])
      | i32Load8U _ _ => exact absurd h_sl (by simp [StraightLineInstr])
      | i32Load8S _ _ => exact absurd h_sl (by simp [StraightLineInstr])
      | i32Store8 _ _ => exact absurd h_sl (by simp [StraightLineInstr])
      | block _ => exact absurd h_sl (by simp [StraightLineInstr])
      | wloop _ => exact absurd h_sl (by simp [StraightLineInstr])
      | wif _ => exact absurd h_sl (by simp [StraightLineInstr])
      | welse => exact absurd h_sl (by simp [StraightLineInstr])
      | wend => exact absurd h_sl (by simp [StraightLineInstr])
      | br _ => exact absurd h_sl (by simp [StraightLineInstr])
      | brIf _ => exact absurd h_sl (by simp [StraightLineInstr])
      | wreturn => exact absurd h_sl (by simp [StraightLineInstr])
      | call _ => exact absurd h_sl (by simp [StraightLineInstr])
      | wselect => exact absurd h_sl (by simp [StraightLineInstr])
      | unreachable => exact absurd h_sl (by simp [StraightLineInstr])
      | unsupported _ => exact absurd h_sl (by simp [StraightLineInstr])
  | @wloop_cons rest body post h_split h_body h_post_wf IH =>
      -- Head is .wloop 0; body matches WloopBodyShape (pref ++
      -- [.i32Const 0, .brIf 0]), post is KernelInstrs.
      -- Depth-aware fuel: depth (wloop_cons) = 1 + depth post,
      -- so h_fuel : fuel ≥ 2 + (1 + depth post) = 3 + depth post.
      -- For cons_wloop_singleIterExit: bt = fuel, need bt ≥ 2. ✓
      -- For IH on post at inner fuel `fuel`: need
      --   fuel ≥ 2 + depth post. We have fuel ≥ 3 + depth post. ✓
      -- Pre-compute the depth equality before destructuring h_body
      -- (the rfl proof needs h_body in scope).
      have h_depth : (KernelInstrs.wloop_cons h_split h_body h_post_wf).depth
                        = 1 + h_post_wf.depth := rfl
      rw [h_depth] at h_fuel
      have h_fuel_inner : fuel ≥ 2 + h_post_wf.depth := by omega
      obtain ⟨pref, h_prefix, h_body_eq⟩ := h_body
      have h_fuel_ge_2 : fuel ≥ 2 := by
        have : 2 + h_post_wf.depth ≥ 2 := by omega
        omega
      have post_preserves :
          ∀ {ws_p : WasmState} {s_p : LowerState}
            {kst_p : Quanta.KOps.State}
            (_R_p : Refines ws_p s_p kst_p layout)
            (_h_nb_p : ws_p.branchTarget = none)
            (_h_nh_p : ws_p.halted = false)
            (_h_nbk_p : kst_p.broke = false)
            {ws'_p : WasmState} {s'_p : LowerState}
            {postOps : List KernelOp}
            (_hw_p : evalInstrs fuel ws_p post = some ws'_p)
            (_hl_p : lowerInstrs fuel frames s_p post = some (s'_p, postOps)),
          ∃ (kst'_p : Quanta.KOps.State) (F : Nat),
            evalOps F kst_p postOps = some kst'_p ∧
            Refines ws'_p s'_p kst'_p layout ∧
            BridgeClauses ws'_p kst'_p := by
        intro ws_p s_p kst_p R_p h_nb_p h_nh_p h_nbk_p
              ws'_p s'_p postOps hw_p hl_p
        -- IH on post needs evalInstrs (fuel' + 1) form. Use
        -- fuel' = fuel - 1; h_fuel_inner gives fuel - 1 ≥ 1 + depth.
        -- IH needs h_fuel : (fuel - 1) ≥ 2 + h_post_wf.depth.
        -- We have fuel ≥ 2 + h_post_wf.depth, so fuel - 1 ≥ 1 + depth.
        -- That's NOT ≥ 2 + depth.
        --
        -- The right framing: IH's fuel is the parameter `fuel'`, and
        -- the eval is at `fuel' + 1`. We have hw_p at fuel `fuel`.
        -- Set fuel' = fuel - 1, so fuel' + 1 = fuel. IH needs
        -- fuel' ≥ 2 + depth post, i.e., fuel - 1 ≥ 2 + depth post.
        -- h_fuel_inner gives fuel ≥ 2 + depth post, so fuel - 1 ≥
        -- 1 + depth post. Still off by 1.
        --
        -- Need h_fuel ≥ 3 + depth post (stronger). Use h_fuel_inner
        -- + 1 (we already have h_fuel : fuel ≥ 3 + depth post from
        -- the original wloop_cons constraint).
        -- h_fuel was rewritten earlier to fuel ≥ 2 + (1 + h_post_wf.depth).
        have h_fuel_for_ih : fuel - 1 ≥ 2 + h_post_wf.depth := by omega
        have h_fuel_eq : fuel = (fuel - 1) + 1 := by
          have : fuel ≥ 1 := by omega
          omega
        rw [h_fuel_eq] at hw_p hl_p
        exact IH (fuel := fuel - 1) (ws := ws_p) (s := s_p) (kst := kst_p)
          R_p h_nb_p h_nh_p h_nbk_p h_fuel_for_ih
          ws'_p s'_p postOps hw_p hl_p
      subst h_body_eq
      exact preservation_evalInstrs_cons_wloop_singleIterExit
        frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
        fuel rest (pref ++ [.i32Const 0, .brIf 0]) post h_split
        (fun {ws_b s_b kst_b} R_b h_nb h_nh h_nbk {ws'_b s'_b bodyOps} hw_b hl_b =>
          irEmptyPrefix_const0_brIf0_body_preserves frames h_prefix layout R_b
            h_nb h_nh h_nbk hw_b hl_b)
        (fun {ws_b s_b kst_b ws'_b s'_b bodyOps} _R_b h_nb h_nh _h_nbk hw_b hl_b =>
          irEmptyPrefix_const0_brIf0_falls_through frames h_prefix h_nb h_nh
            hw_b hl_b)
        (fun {ws_b s_b kst_b ws'_b s'_b bodyOps kst'_b F_b} _R_b _h_nb _h_nh h_nbk _hw_b hl_b h_ev_b =>
          irEmptyPrefix_const0_brIf0_exits_with_broke frames h_prefix h_nbk
            hl_b h_ev_b)
        (fun {s_b s'_b bodyOps} hl_b =>
          irEmptyPrefix_const0_brIf0_lowering_preserves frames h_prefix hl_b)
        post_preserves h_fuel_ge_2
        ws' s' ops hw hl

end Quanta.Wasm
