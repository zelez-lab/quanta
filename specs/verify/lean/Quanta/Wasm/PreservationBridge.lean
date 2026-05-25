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
  -- Same reductions as the non-bridge proof — copied here because
  -- the case-split structure is what threads through the bridge.
  rw [lowerInstrs_cons_default fuel frames s (.localGet i) rest rfl] at hl
  cases h_stable : s.lookupLocal i with
  | none =>
      simp only [lowerInstr, h_no_buf, h_stable, Option.bind_eq_bind,
                 Option.some_bind, Option.none_bind, LowerState.alloc,
                 LowerState.push] at hl
      exact (Option.noConfusion hl)
  | some stable =>
      let s_after : LowerState :=
        { s with nextReg := s.nextReg + 1,
                 stack := SymVal.reg s.nextReg .u32 :: s.stack }
      let ops_head : List KernelOp := [.copy s.nextReg stable]
      have hl_head : lowerInstr s (.localGet i) = some (s_after, ops_head) := by
        show (match s.lookupBufferSlot i with
              | some slot => some (s.pushSym (.bufferPtr slot), [])
              | none => do
                  let stable ← s.lookupLocal i
                  let (fresh, s1) := s.alloc
                  let s2 := s1.push fresh
                  pure (s2, [.copy fresh stable])) = some (s_after, ops_head)
        rw [h_no_buf, h_stable]
        rfl
      rw [hl_head] at hl
      simp only [Option.bind_eq_bind, Option.some_bind] at hl
      cases h_post : lowerInstrs fuel frames s_after rest with
      | none => simp [h_post] at hl
      | some post_pair =>
          rcases post_pair with ⟨s_post, postOps⟩
          simp [h_post] at hl
          rcases hl with ⟨h_s_eq, h_ops_eq⟩
          rw [evalInstrs_cons_default fuel ws (.localGet i) rest h_no_branch h_no_halt rfl] at hw
          cases h_loc : ws.getLocal i with
          | none =>
              have hw_step : evalInstr ws (.localGet i) = none := by
                show (do let v ← ws.getLocal i; pure (ws.push v)) = none
                rw [h_loc]; rfl
              rw [hw_step] at hw
              simp at hw
          | some v =>
              let ws_after : WasmState := ws.push v
              have hw_step : evalInstr ws (.localGet i) = some ws_after := by
                show (do let v ← ws.getLocal i; pure (ws.push v)) = some ws_after
                rw [h_loc]
                rfl
              rw [hw_step] at hw
              simp only at hw
              obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
                preservation_localGet ws s kst layout R i h_no_buf
                  ws_after s_after ops_head
                  hw_step hl_head
              have h_mid_broke : kst_mid.broke = false := by
                have := evalOps_copy_singleton_preserves_broke h_kst_eval
                rw [this]; exact h_kst_no_broke
              have h_mid_no_branch : ws_after.branchTarget = none := by
                simp [ws_after, WasmState.push, h_no_branch]
              have h_mid_no_halt : ws_after.halted = false := by
                simp [ws_after, WasmState.push, h_no_halt]
              -- Bridge IH on `rest` yields existence + Refines + bridge clauses.
              obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest, h_bridge_rest⟩ :=
                preservation_rest_bridge R_mid h_mid_no_branch h_mid_no_halt
                  h_mid_broke hw h_post
              -- Bridge-aware cons-compose (head is shallow loopFree).
              have h_lf : loopFree ops_head = true := by
                simp [loopFree, loopFreeOp, ops_head]
              obtain ⟨kst'', h_eval'', R'', h_bridge''⟩ :=
                preservation_evalInstrs_cons_compose_shallow_bridge
                  (F := F_rest) h_lf h_kst_eval h_mid_broke
                  ⟨kst'_mid, h_eval_rest, R_rest, h_bridge_rest⟩
              refine ⟨kst'', F_rest, ?_, ?_, ?_⟩
              · rw [← h_ops_eq]; exact h_eval''
              · rw [← h_s_eq]; exact R''
              · exact h_bridge''

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
  rw [lowerInstrs_cons_default fuel frames s (.localSet i) rest rfl] at hl
  cases h_head : lowerInstr s (.localSet i) with
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
          rw [evalInstrs_cons_default fuel ws (.localSet i) rest h_no_branch h_no_halt rfl] at hw
          cases h_eval_head : evalInstr ws (.localSet i) with
          | none => rw [h_eval_head] at hw; simp at hw
          | some ws_after =>
              rw [h_eval_head] at hw
              simp only at hw
              obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
                preservation_localSet ws s kst layout R h_kst_no_broke i
                  ws_after s_after ops_head
                  h_eval_head h_head
              have h_lf_head : loopFreeNoBreak ops_head = true :=
                lowerInstr_localSet_emits_loopFreeNoBreak_pub h_head
              have h_lf_head_shallow : loopFree ops_head = true :=
                loopFreeNoBreak_implies_loopFree h_lf_head
              have h_mid_broke : kst_mid.broke = false :=
                evalOps_loopFreeNoBreak_preserves_broke
                  h_lf_head h_kst_no_broke h_kst_eval
              have h_mid_no_branch : ws_after.branchTarget = none := by
                simp only [evalInstr, WasmState.pop, WasmState.setLocal,
                           Option.bind_eq_bind, Option.bind, pure] at h_eval_head
                rcases hws : ws.stack with _ | ⟨v_w, rest_ws⟩
                · simp [hws] at h_eval_head
                · simp only [hws] at h_eval_head
                  by_cases hbnd : i < ws.locals.length
                  · simp only [if_pos hbnd] at h_eval_head
                    have := ((Option.some.injEq _ _).mp h_eval_head).symm
                    rw [this]; simp [h_no_branch]
                  · simp only [if_neg hbnd] at h_eval_head
                    simp at h_eval_head
              have h_mid_no_halt : ws_after.halted = false := by
                simp only [evalInstr, WasmState.pop, WasmState.setLocal,
                           Option.bind_eq_bind, Option.bind, pure] at h_eval_head
                rcases hws : ws.stack with _ | ⟨v_w, rest_ws⟩
                · simp [hws] at h_eval_head
                · simp only [hws] at h_eval_head
                  by_cases hbnd : i < ws.locals.length
                  · simp only [if_pos hbnd] at h_eval_head
                    have := ((Option.some.injEq _ _).mp h_eval_head).symm
                    rw [this]; simp [h_no_halt]
                  · simp only [if_neg hbnd] at h_eval_head
                    simp at h_eval_head
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
  rw [lowerInstrs_cons_default fuel frames s (.localTee i) rest rfl] at hl
  cases h_head : lowerInstr s (.localTee i) with
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
          rw [evalInstrs_cons_default fuel ws (.localTee i) rest h_no_branch h_no_halt rfl] at hw
          cases h_eval_head : evalInstr ws (.localTee i) with
          | none => rw [h_eval_head] at hw; simp at hw
          | some ws_after =>
              rw [h_eval_head] at hw
              simp only at hw
              obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
                preservation_localTee ws s kst layout R h_kst_no_broke i
                  ws_after s_after ops_head
                  h_eval_head h_head
              have h_lf_head : loopFreeNoBreak ops_head = true :=
                lowerInstr_localTee_emits_loopFreeNoBreak_pub h_head
              have h_lf_head_shallow : loopFree ops_head = true :=
                loopFreeNoBreak_implies_loopFree h_lf_head
              have h_mid_broke : kst_mid.broke = false :=
                evalOps_loopFreeNoBreak_preserves_broke
                  h_lf_head h_kst_no_broke h_kst_eval
              have h_mid_no_branch : ws_after.branchTarget = none := by
                simp only [evalInstr, WasmState.pop, WasmState.push, WasmState.setLocal,
                           Option.bind_eq_bind, Option.bind, pure] at h_eval_head
                rcases hws : ws.stack with _ | ⟨v_w, rest_ws⟩
                · simp [hws] at h_eval_head
                · simp only [hws] at h_eval_head
                  by_cases hbnd : i < ws.locals.length
                  · simp only [if_pos hbnd] at h_eval_head
                    have := ((Option.some.injEq _ _).mp h_eval_head).symm
                    rw [this]; simp [h_no_branch]
                  · simp only [if_neg hbnd] at h_eval_head
                    simp at h_eval_head
              have h_mid_no_halt : ws_after.halted = false := by
                simp only [evalInstr, WasmState.pop, WasmState.push, WasmState.setLocal,
                           Option.bind_eq_bind, Option.bind, pure] at h_eval_head
                rcases hws : ws.stack with _ | ⟨v_w, rest_ws⟩
                · simp [hws] at h_eval_head
                · simp only [hws] at h_eval_head
                  by_cases hbnd : i < ws.locals.length
                  · simp only [if_pos hbnd] at h_eval_head
                    have := ((Option.some.injEq _ _).mp h_eval_head).symm
                    rw [this]; simp [h_no_halt]
                  · simp only [if_neg hbnd] at h_eval_head
                    simp at h_eval_head
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
  rw [lowerInstrs_cons_default fuel frames s (.localGet i) rest rfl] at hl
  have h_head : lowerInstr s (.localGet i)
                  = some (s.pushSym (.bufferPtr slot), []) := by
    show (match s.lookupBufferSlot i with
          | some slot => some (s.pushSym (.bufferPtr slot), [])
          | none =>
              (s.lookupLocal i).bind (fun stable =>
                let (fresh, s1) := s.alloc
                let s2 := s1.push fresh
                some (s2, [KernelOp.copy fresh stable])))
              = some (s.pushSym (.bufferPtr slot), [])
    rw [h_buf]
  rw [h_head] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl
  cases h_post : lowerInstrs fuel frames (s.pushSym (.bufferPtr slot)) rest with
  | none => simp [h_post] at hl
  | some post_pair =>
      rcases post_pair with ⟨s_post, postOps⟩
      simp [h_post] at hl
      rcases hl with ⟨h_s_eq, h_ops_eq⟩
      rw [evalInstrs_cons_default fuel ws (.localGet i) rest h_no_branch h_no_halt rfl] at hw
      cases h_eval_head : evalInstr ws (.localGet i) with
      | none => rw [h_eval_head] at hw; simp at hw
      | some ws_after =>
          rw [h_eval_head] at hw
          simp only at hw
          obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
            preservation_localGet_bufferSlot ws s kst layout R i slot h_buf h_loc_buf
              ws_after (s.pushSym (.bufferPtr slot)) []
              h_eval_head h_head
          have h_kst_mid_eq : kst_mid = kst := by
            simp [evalOps] at h_kst_eval
            exact h_kst_eval.symm
          have h_ws_after_shape : ∃ v, ws_after = ws.push v := by
            cases hloc : ws.getLocal i with
            | none =>
                have h_ev : evalInstr ws (.localGet i) = none := by
                  show (do let v ← ws.getLocal i; pure (ws.push v)) = none
                  rw [hloc]; rfl
                rw [h_ev] at h_eval_head; exact (Option.noConfusion h_eval_head)
            | some v =>
                refine ⟨v, ?_⟩
                have h_ev : evalInstr ws (.localGet i) = some (ws.push v) := by
                  show (do let v ← ws.getLocal i; pure (ws.push v)) = some (ws.push v)
                  rw [hloc]; rfl
                rw [h_ev] at h_eval_head
                exact ((Option.some.injEq _ _).mp h_eval_head).symm
          obtain ⟨v_pushed, h_ws_after_eq⟩ := h_ws_after_shape
          have h_mid_no_branch : ws_after.branchTarget = none := by
            rw [h_ws_after_eq]; simp [WasmState.push, h_no_branch]
          have h_mid_no_halt : ws_after.halted = false := by
            rw [h_ws_after_eq]; simp [WasmState.push, h_no_halt]
          have h_mid_broke : kst_mid.broke = false := by
            rw [h_kst_mid_eq]; exact h_kst_no_broke
          obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest, h_bridge_rest⟩ :=
            preservation_rest_bridge R_mid h_mid_no_branch h_mid_no_halt h_mid_broke hw h_post
          refine ⟨kst'_mid, F_rest, ?_, ?_, ?_⟩
          · rw [← h_ops_eq]
            rw [h_kst_mid_eq] at h_eval_rest
            exact h_eval_rest
          · rw [← h_s_eq]; exact R_rest
          · exact h_bridge_rest

/-- `i32Shl :: rest` bridge (buffer-pattern fold path). Head emits
    no IR — symbolic stack rewritten to `.scaledIdx base (1<<<k)`. -/
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
-- L8.2 / L8.3 — cons_wif and cons_wloop preservation (DEFERRED)
--
-- The wif/wloop lowerings thread lower-state through bodies
-- sequentially. For wif:
--
--     (cond_bool, s_cast) := s1.alloc
--     (s2, thenOps) ← lowerInstrs f (.wif::frames) s_cast thenBody
--     (s3, elseOps) ← lowerInstrs f (.wif::frames) s2 elseBody
--     (s4, postOps) ← lowerInstrs f frames s3 post
--
-- But eval runs ONE body based on cond, then post. For c = 0,
-- the Refines IH for elseBody needs Refines at `s2` (post-thenBody),
-- not at `s_cast`.
--
-- This requires a Refines-freshness-monotonicity lemma on
-- `lowerInstrs` that says (informally): lowering only bumps
-- `nextReg` and adds register allocations, leaving stack/locals/
-- bufferSlots compatible with the input Refines.
--
-- Estimated effort for the lemma + cons_wif + cons_wloop + the
-- bridging invariant: 2-4 sessions. Tracked in tasks #5/#7/#8.
-- ════════════════════════════════════════════════════════════════════

end Quanta.Wasm
