/-
# rustc's `while` — one iteration of `block { loop { pref; br_if 1; body2; br 0 } }`

The body of the loop rustc emits for `while cond { body }` is

    pref ++ [.brIf 1] ++ body2 ++ [.br 0]

lowered (Stage B, `lowerInstrsP`) under the frames `.loopK :: .block :: frames`
to

    prefOps ++ opsCommit ++ [cast cb c .u32 .bool,
                             branch cb [const flag (.bool true), breakOp] []]
            ++ body2Ops

with one exit-flag record left pending for the loop close. This file
proves what one run of that body does, in the two shapes the block-level
N-iteration theorem consumes:

* **continue** (`c = 0`): the whole body runs, WASM leaves
  `branchTarget = some 0` (the `br 0`), the IR ends with `broke = false`,
  and the body-out refines the post-body lowering state `s1`;
* **exit** (`c ≠ 0`): WASM leaves `branchTarget = some 1` and skips
  `body2 ++ [br 0]`; the IR sets the flag, breaks, and skips `body2Ops`;
  the body-out refines the loop-CLOSE state `{ s1 with currentReg := [] }`
  — under the side condition that `body2` rebinds no local's stable
  register or label (its sets only refresh registers that exist at the
  exit site), since on the exit path none of `body2`'s registers were
  written.

Pieces: the lowering shape (`lowerInstrsP_brIf1_exit`,
`blockWhileBody_lowerP`), three `Refines` transfers, and
`blockWhileBody_iteration`.
-/

import Quanta.Wasm.PreservationWhile
import Quanta.Wasm.FlagSemantics
import Quanta.Wasm.PreservationWhileExit
import Quanta.Wasm.TranslatePendingAgree

namespace Quanta.Wasm

open Quanta.KOps (KernelOp evalOps regLookup regWrite vBool)

-- ════════════════════════════════════════════════════════════════════
-- `Refines` transfers
-- ════════════════════════════════════════════════════════════════════

/-- `Refines` never reads `branchTarget`. -/
theorem Refines.set_branch
    {ws : WasmState} {s : LowerState} {kst : Quanta.KOps.State} {layout : BufferLayout}
    (R : Refines ws s kst layout) (bt : Option Nat) :
    Refines { ws with branchTarget := bt } s kst layout :=
  ⟨R.stk, R.locs, R.fresh, R.aliasFree, R.injLocals, R.heapRefines, R.currentReg,
   R.freshCurrent, R.curLocDisj⟩

/-- A larger `nextReg` keeps every freshness bound. -/
theorem Refines.bump_nextReg
    {ws : WasmState} {s : LowerState} {kst : Quanta.KOps.State} {layout : BufferLayout}
    (R : Refines ws s kst layout) {n : Nat} (h : s.nextReg ≤ n) :
    Refines ws { s with nextReg := n } kst layout := by
  refine ⟨R.stk, R.locs, ?_, R.aliasFree, R.injLocals, R.heapRefines, R.currentReg, ?_,
          R.curLocDisj⟩
  · refine ⟨fun sv hsv r hr => ?_, fun ir hir => ?_⟩
    · exact Nat.lt_of_lt_of_le (R.fresh.1 sv hsv r hr) h
    · exact Nat.lt_of_lt_of_le (R.fresh.2 ir hir) h
  · intro ir hir
    exact Nat.lt_of_lt_of_le (R.freshCurrent ir hir) h

/-- `Refines` carried to the close state of a lowering state `s'` that
    agrees with `s` on the stack and the stable layer and has at least
    its `nextReg`: the per-frame bindings are dropped (vacuous
    invariants), everything else is `s`'s. -/
theorem Refines.to_close
    {ws : WasmState} {s s' : LowerState} {kst : Quanta.KOps.State} {layout : BufferLayout}
    (R : Refines ws s kst layout)
    (h_stk : s'.stack = s.stack) (h_lr : s'.localReg = s.localReg)
    (h_lt : s'.localTy = s.localTy) (h_bs : s'.bufferSlots = s.bufferSlots)
    (h_nr : s.nextReg ≤ s'.nextReg) :
    Refines ws { s' with currentReg := [] } kst layout := by
  have R1 : Refines ws { s with nextReg := s'.nextReg } kst layout := R.bump_nextReg h_nr
  have h_eq : ({ s' with currentReg := [] } : LowerState)
      = { ({ s with nextReg := s'.nextReg } : LowerState) with currentReg := [] } := by
    cases s'; cases s
    simp only at h_stk h_lr h_lt h_bs
    simp [h_stk, h_lr, h_lt, h_bs]
  rw [h_eq]
  exact R1.clear_current

/-- A list evaluated from a state with a pending branch is returned as
    it is. -/
theorem evalInstrs_of_branch_set {fuel : Nat} {st : WasmState} {d : Nat}
    (h : st.branchTarget = some d) (l : List WasmInstr) :
    evalInstrs fuel st l = some st := by
  cases l with
  | nil => simp [evalInstrs]
  | cons i rest =>
      unfold evalInstrs
      simp [h]

-- ════════════════════════════════════════════════════════════════════
-- The lowering shape
-- ════════════════════════════════════════════════════════════════════

/-- The Stage-B arm for `br_if 1` inside a loop inside a block: pop and
    commit the condition, allocate the bool register and the flag, emit
    the exit site, lower the rest sequentially, and leave the exit-flag
    record (one level, no frames to skip) on top of the rest's pending. -/
theorem lowerInstrsP_brIf1_exit
    (fuel : Nat) (frames : List FrameKind) (s : LowerState) (p : List PendingWrap)
    (rest : List WasmInstr) :
    lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s, p⟩ (.brIf 1 :: rest) =
      (do
        let (svCond, s0) ← s.popSym
        let (cond, s1, opsCommit) ← s0.commit svCond
        let (cond_bool, s_cast) := s1.alloc
        let (flag, s_flag) := s_cast.alloc
        let (s2, restOps) ← lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s_flag, p⟩ rest
        pure (⟨s2.base, ({ levels := 1, cond := flag, flag := true, skip := 0 } : PendingWrap)
                          :: s2.pending⟩,
              opsCommit ++ [.cast cond_bool cond .u32 .bool,
                            .branch cond_bool [.const flag (.bool true), .breakOp] []]
                        ++ restOps)) := by
  simp only [lowerInstrsP]
  rcases hpop : s.popSym with _ | ⟨svCond, s0⟩
  · rfl
  simp only [Option.bind_eq_bind, Option.some_bind]
  rcases hcommit : s0.commit svCond with _ | ⟨cond, s1, opsCommit⟩
  · rfl
  simp only [Option.some_bind]
  have h_idx : List.findIdx (fun x => decide (x = FrameKind.loopK)) [FrameKind.loopK] = 0 := by
    decide
  simp [hasLoopAbove, loopsAbove, exitFlagEntry, loopIndex, LowerState.alloc, h_idx]

/-- `br 0` at the end of a loop body: no ops, nothing pending. -/
theorem lowerInstrsP_br0_loop (fuel : Nat) (frames : List FrameKind)
    (s : LowerState) (p : List PendingWrap) :
    lowerInstrsP fuel (.loopK :: frames) ⟨s, p⟩ [.br 0] = some (⟨s, p⟩, []) := by
  simp [lowerInstrsP]

/-- The lowering of one iteration's body, taken apart: the straight-line
    prefix (plain Stage A), the popped and committed condition, the bool
    register `s_c.nextReg` and the flag `s_c.nextReg + 1`, `body2` (plain
    Stage A, from the state after both allocations), and the `br 0` that
    emits nothing. One exit-flag record is left pending. -/
theorem blockWhileBody_lowerP
    {fuel : Nat} {frames : List FrameKind} {pref body2 : List WasmInstr}
    (h_pref : StraightLineInstrs pref) (h_body2 : StraightLineInstrs body2)
    {s : LowerState} {sp' : LowerStateP} {ops : List KernelOp}
    (hl : lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s, []⟩
            (pref ++ [.brIf 1] ++ body2 ++ [.br 0]) = some (sp', ops)) :
    ∃ (s_m s0 s_c s_b : LowerState) (svCond : SymVal) (cond : Quanta.KOps.Reg)
      (opsPref opsCommit opsBody2 : List KernelOp),
      lowerInstrs fuel (.loopK :: .block :: frames) s pref = some (s_m, opsPref) ∧
      s_m.popSym = some (svCond, s0) ∧
      s0.commit svCond = some (cond, s_c, opsCommit) ∧
      lowerInstrs fuel (.loopK :: .block :: frames) { s_c with nextReg := s_c.nextReg + 2 } body2
        = some (s_b, opsBody2) ∧
      sp' = ⟨s_b, [{ levels := 1, cond := s_c.nextReg + 1, flag := true, skip := 0 }]⟩ ∧
      ops = opsPref ++ opsCommit
              ++ [.cast s_c.nextReg cond .u32 .bool,
                  .branch s_c.nextReg [.const (s_c.nextReg + 1) (.bool true), .breakOp] []]
              ++ opsBody2 := by
  have h_list : pref ++ [.brIf 1] ++ body2 ++ [.br 0]
      = pref ++ (.brIf 1 :: (body2 ++ [.br 0])) := by simp
  rw [h_list] at hl
  obtain ⟨s_m, opsPref, ops2, hl_pref, hl_rest, h_ops⟩ :=
    lowerInstrsP_straightLine_append h_pref hl
  rw [lowerInstrsP_brIf1_exit] at hl_rest
  rcases h_pop : s_m.popSym with _ | ⟨svCond, s0⟩
  · rw [h_pop] at hl_rest; simp at hl_rest
  rw [h_pop] at hl_rest
  simp only [Option.bind_eq_bind, Option.some_bind] at hl_rest
  rcases h_commit : s0.commit svCond with _ | ⟨cond, s_c, opsCommit⟩
  · rw [h_commit] at hl_rest; simp at hl_rest
  rw [h_commit] at hl_rest
  simp only [Option.some_bind, LowerState.alloc] at hl_rest
  rcases h_body : lowerInstrsP fuel (.loopK :: .block :: frames)
      ⟨{ s_c with nextReg := s_c.nextReg + 1 + 1 }, []⟩ (body2 ++ [.br 0]) with _ | ⟨sp_b, restOps⟩
  · rw [h_body] at hl_rest; simp at hl_rest
  rw [h_body] at hl_rest
  simp only [Option.some_bind, pure, Option.some.injEq, Prod.mk.injEq] at hl_rest
  obtain ⟨h_sp, h_ops2⟩ := hl_rest
  obtain ⟨s_b, opsBody2, opsBr, hl_body2, hl_br, h_rest_ops⟩ :=
    lowerInstrsP_straightLine_append h_body2 h_body
  rw [lowerInstrsP_br0_loop] at hl_br
  simp only [Option.some.injEq, Prod.mk.injEq] at hl_br
  obtain ⟨h_spb, h_opsBr⟩ := hl_br
  subst h_spb; subst h_opsBr
  refine ⟨s_m, s0, s_c, s_b, svCond, cond, opsPref, opsCommit, opsBody2, hl_pref, h_pop,
          h_commit, ?_, ?_, ?_⟩
  · simpa using hl_body2
  · rw [← h_sp]
  · rw [h_ops, ← h_ops2, h_rest_ops]; simp

-- ════════════════════════════════════════════════════════════════════
-- One iteration
-- ════════════════════════════════════════════════════════════════════

/-- One run of rustc's loop body. Exposes the exit site's lowering state
    `s_flag` (after the condition was committed and both registers were
    allocated) and the flag register, and gives the two outcomes:

    * continue — `c = 0`: WASM ends at `branchTarget = some 0`, the IR at
      `broke = false`, the body-out refines the post-body state
      `sp'.base`;
    * exit — `c ≠ 0`: WASM ends at `branchTarget = some 1` with
      `body2 ++ [br 0]` untouched, the IR at `broke = true` with the flag
      set and `body2Ops` skipped, the body-out refines the exit site's
      state `s_flag`.

    The caller carries the exit-side refinement to the loop's close state
    with `Refines.to_close` and the body frame of `body2` (stack and
    buffer slots restored, stable layer unchanged under the side
    condition). -/
theorem blockWhileBody_iteration
    (fuel : Nat) (frames : List FrameKind)
    (pref body2 : List WasmInstr)
    (h_pref : StraightLineInstrs pref) (h_body2 : StraightLineInstrs body2)
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
    (ws' : WasmState) (sp' : LowerStateP) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (pref ++ [.brIf 1] ++ body2 ++ [.br 0]) = some ws')
    (hl : lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s, []⟩
            (pref ++ [.brIf 1] ++ body2 ++ [.br 0]) = some (sp', ops)) :
    ∃ (s_flag : LowerState) (flag : Quanta.KOps.Reg) (opsBody2 : List KernelOp)
      (kst' : Quanta.KOps.State) (F : Nat),
      sp'.pending = [{ levels := 1, cond := flag, flag := true, skip := 0 }] ∧
      s.nextReg ≤ flag ∧ flag < s_flag.nextReg ∧
      lowerInstrs fuel (.loopK :: .block :: frames) s_flag body2 = some (sp'.base, opsBody2) ∧
      evalOps F kst ops = some kst' ∧
      ws'.halted = false ∧
      ((ws'.branchTarget = some 0 ∧ kst'.broke = false ∧
          Refines ws' sp'.base kst' layout) ∨
       (ws'.branchTarget = some 1 ∧ kst'.broke = true ∧
          Refines ws' s_flag kst' layout ∧
          regLookup kst'.rf flag = some (vBool true))) := by
  obtain ⟨s_m, s0, s_c, s_b, svCond, cond, opsPref, opsCommit, opsBody2,
          hl_pref, h_pop, h_commit, hl_body2, h_sp, h_ops⟩ :=
    blockWhileBody_lowerP h_pref h_body2 hl
  subst h_sp
  subst h_ops
  -- WASM: the prefix, then the brIf.
  have h_list : pref ++ [.brIf 1] ++ body2 ++ [.br 0]
      = pref ++ (.brIf 1 :: (body2 ++ [.br 0])) := by simp
  rw [h_list] at hw
  obtain ⟨ws_m, hw_pref, h_mb, h_mh, hw_rest⟩ :=
    evalInstrs_straightLine_append h_pref h_no_branch h_no_halt hw
  -- IR: the prefix.
  obtain ⟨kst_m, F1, h_ev1, R_m, h_bridge_m⟩ :=
    framework_preservation_straightLine fuel (.loopK :: .block :: frames) ws s kst layout R
      h_no_branch h_no_halt h_kst_no_broke h_buf_locals h_no_buf_stack
      h_load_bounds h_store_bounds h_store_layout pref h_pref ws_m s_m opsPref
      hw_pref hl_pref
  have h_kst_m_ok : kst_m.broke = false := h_bridge_m.right h_mb
  have h_nr_m : s.nextReg ≤ s_m.nextReg := by
    have := lowerInstrs_nextReg_mono _ _ _ _ hl_pref
    exact this
  -- The brIf on the WASM side: pops `c`.
  rw [evalInstrs_cons_default fuel ws_m (WasmInstr.brIf 1) (body2 ++ [WasmInstr.br 0]) h_mb h_mh rfl]
    at hw_rest
  cases he : evalInstr ws_m (.brIf 1) with
  | none => rw [he] at hw_rest; simp at hw_rest
  | some ws1 =>
  rw [he] at hw_rest
  simp only at hw_rest
  obtain ⟨c, rest_w, h_stack, h_branch⟩ := evalInstr_brIf_shape_pub he
  -- The commit on the IR side.
  obtain ⟨kst1, h_ev_commit, h_kst1_ok, h_lookup, R1, h_nr_c, h_cond_lt, _, _, _, _⟩ :=
    brIf_cond_pop_commit_correct_pub R_m h_stack h_pop h_commit h_kst_m_ok
  have h_sc_stack : ({ s_c with stack := s0.stack } : LowerState) = s_c := by
    have := commit_preserves_stack h_commit
    cases s_c; simp only at this; simp [this]
  rw [h_sc_stack] at R1
  -- The cast: `cb := (c ≠ 0)`.
  have h_c_toNat : decide (c.toNat = 0) = decide (c = 0) := by
    by_cases hc : c = 0
    · subst hc; rfl
    · have : c.toNat ≠ 0 := by
        intro h
        exact hc (by simpa using congrArg UInt32.ofNat h)
      simp [hc, this]
  let kst_cast : Quanta.KOps.State :=
    { kst1 with rf := regWrite kst1.rf s_c.nextReg (vBool (!decide (c = 0))) }
  have h_cast : Quanta.KOps.evalOp F1 kst1
      (KernelOp.cast s_c.nextReg cond Quanta.KOps.Scalar.u32 Quanta.KOps.Scalar.bool)
      = some kst_cast := by
    rcases h_lookup with h_u32 | h_i32
    · simp [Quanta.KOps.evalOp, h_u32, Quanta.KOps.evalCast, kst_cast]
    · simp [Quanta.KOps.evalOp, h_i32, Quanta.KOps.evalCast, kst_cast, h_c_toNat]
  have h_kst_cast_ok : kst_cast.broke = false := h_kst1_ok
  have h_cb : regLookup kst_cast.rf s_c.nextReg = some (vBool (!decide (c = 0))) :=
    regLookup_regWrite_self _ _ _
  -- Refines at the exit site's lowering state.
  have R_cast : Refines { ws_m with stack := rest_w } s_c kst_cast layout :=
    R1.regWrite_fresh (Nat.le_refl _) _
  have R_flag : Refines { ws_m with stack := rest_w }
      { s_c with nextReg := s_c.nextReg + 2 } kst_cast layout :=
    R_cast.bump_nextReg (by omega)
  -- Shared prefix of the IR run: prefix ops, commit ops, the cast.
  have h_ev_to_cast : ∀ F, F1 ≤ F →
      evalOps F kst (opsPref ++ opsCommit
        ++ [KernelOp.cast s_c.nextReg cond .u32 .bool,
            KernelOp.branch s_c.nextReg [.const (s_c.nextReg + 1) (.bool true), .breakOp] []]
        ++ opsBody2)
      = evalOps F kst_cast
          (KernelOp.branch s_c.nextReg [.const (s_c.nextReg + 1) (.bool true), .breakOp] []
            :: opsBody2) := by
    intro F hF
    rw [List.append_assoc, List.append_assoc,
        evalOps_append (evalOps_fuel_mono hF h_ev1) h_kst_m_ok,
        evalOps_append (evalOps_fuel_mono (Nat.zero_le F) h_ev_commit) h_kst1_ok]
    simp only [List.cons_append, List.nil_append]
    have h_cast_F : Quanta.KOps.evalOp F kst1
        (KernelOp.cast s_c.nextReg cond Quanta.KOps.Scalar.u32 Quanta.KOps.Scalar.bool)
        = some kst_cast := by
      rcases h_lookup with h_u32 | h_i32
      · simp [Quanta.KOps.evalOp, h_u32, Quanta.KOps.evalCast, kst_cast]
      · simp [Quanta.KOps.evalOp, h_i32, Quanta.KOps.evalCast, kst_cast, h_c_toNat]
    rw [evalOps_cons_continue h_cast_F h_kst_cast_ok]
  rcases h_branch with ⟨hc, h_ws1⟩ | ⟨hc, h_ws1⟩
  · -- continue: the condition is zero, the site falls through, body2 and
    -- br 0 run.
    subst h_ws1
    have h_cb_false : regLookup kst_cast.rf s_c.nextReg = some (vBool false) := by
      rw [h_cb]; simp [hc]
    have h_ws1_nb : ({ ws_m with stack := rest_w } : WasmState).branchTarget = none := h_mb
    have h_ws1_nh : ({ ws_m with stack := rest_w } : WasmState).halted = false := h_mh
    obtain ⟨ws_b, hw_body2, h_bb, h_bh, hw_br⟩ :=
      evalInstrs_straightLine_append h_body2 h_ws1_nb h_ws1_nh hw_rest
    rw [br0_in_loop_continues h_bb h_bh, Option.some.injEq] at hw_br
    subst hw_br
    obtain ⟨kst_b, F2, h_ev_b, R_b, h_bridge_b⟩ :=
      framework_preservation_straightLine fuel (.loopK :: .block :: frames)
        { ws_m with stack := rest_w } { s_c with nextReg := s_c.nextReg + 2 } kst_cast layout
        R_flag h_ws1_nb h_ws1_nh h_kst_cast_ok h_buf_locals h_no_buf_stack
        h_load_bounds h_store_bounds h_store_layout body2 h_body2 ws_b s_b opsBody2
        hw_body2 hl_body2
    refine ⟨{ s_c with nextReg := s_c.nextReg + 2 }, s_c.nextReg + 1, opsBody2, kst_b,
            max F1 F2, rfl, by omega, by simp, hl_body2, ?_, h_bh, Or.inl ⟨rfl, ?_, ?_⟩⟩
    · rw [h_ev_to_cast (max F1 F2) (Nat.le_max_left _ _),
          evalOps_exitSite_falls_through h_kst_cast_ok h_cb_false]
      exact evalOps_fuel_mono (Nat.le_max_right _ _) h_ev_b
    · exact h_bridge_b.right h_bb
    · exact R_b.set_branch _
  · -- exit: the condition is non-zero, the site sets the flag and breaks;
    -- body2 and br 0 are skipped on both sides.
    subst h_ws1
    have h_cb_true : regLookup kst_cast.rf s_c.nextReg = some (vBool true) := by
      rw [h_cb]; simp [hc]
    rw [evalInstrs_of_branch_set rfl, Option.some.injEq] at hw_rest
    subst hw_rest
    refine ⟨{ s_c with nextReg := s_c.nextReg + 2 }, s_c.nextReg + 1, opsBody2,
            { kst_cast with rf := regWrite kst_cast.rf (s_c.nextReg + 1) (vBool true),
                            broke := true },
            F1, rfl, by omega, by simp, hl_body2, ?_, h_mh, Or.inr ⟨rfl, rfl, ?_, ?_⟩⟩
    · rw [h_ev_to_cast F1 (Nat.le_refl _), evalOps_exitSite_fires h_cb_true]
    · have R_w : Refines { ws_m with stack := rest_w }
          { s_c with nextReg := s_c.nextReg + 1 } kst_cast layout :=
        R_cast.bump_nextReg (by omega)
      have R_w2 := (R_w.regWrite_fresh (r := s_c.nextReg + 1) (Nat.le_refl _) (vBool true))
      have R_w3 : Refines { ws_m with stack := rest_w } { s_c with nextReg := s_c.nextReg + 2 }
          { kst_cast with rf := regWrite kst_cast.rf (s_c.nextReg + 1) (vBool true) } layout :=
        R_w2.bump_nextReg (by simp)
      exact (R_w3.set_broke true).set_branch _
    · exact regLookup_regWrite_self _ _ _

-- ════════════════════════════════════════════════════════════════════
-- The block-level N-iteration composition
-- ════════════════════════════════════════════════════════════════════

/-- The Stage-B lowering of rustc's `while` skeleton, taken apart: the
    loop body from the loop-entry state (with the exit-flag record it
    leaves), the post from the loop-close state, and the kernel ops
    `const flag false; loopOp bodyOps; branch flag [] []` ahead of the
    post's. The block's own tail is empty in this shape — the body
    always continues with `br 0` or leaves through `br_if 1`, so there
    is nothing between the loop's `end` and the block's. -/
theorem blockWhile_lowerP
    {f : Nat} {frames : List FrameKind} {pref body2 post : List WasmInstr}
    (h_pref : StraightLineInstrs pref) (h_body2 : StraightLineInstrs body2)
    {s : LowerState} {s' : LowerState} {ops : List KernelOp}
    (hl : lowerInstrsP (f + 2) frames ⟨s, []⟩
            (.block 0 :: .wloop 0 :: (pref ++ [.brIf 1] ++ body2 ++ [.br 0])
               ++ [.wend] ++ [] ++ [.wend] ++ post)
          = some (⟨s', []⟩, ops)) :
    ∃ (s1 : LowerState) (flag : Quanta.KOps.Reg) (bodyOps postOps : List KernelOp),
      lowerInstrsP f (.loopK :: .block :: frames) ⟨{ s with currentReg := [] }, []⟩
          (pref ++ [.brIf 1] ++ body2 ++ [.br 0])
        = some (⟨s1, [{ levels := 1, cond := flag, flag := true, skip := 0 }]⟩, bodyOps) ∧
      lowerInstrsP (f + 1) frames ⟨{ s1 with currentReg := [] }, []⟩ post
        = some (⟨s', []⟩, postOps) ∧
      ops = [.const flag (.bool false), .loopOp bodyOps, .branch flag [] []] ++ postOps := by
  have h_ns : NoStructured (pref ++ [.brIf 1] ++ body2 ++ [.br 0]) :=
    blockWhileBody_noStructured h_pref h_body2
  have h_split_b := splitAtEnd_wloop_noStructured h_ns (tail := []) (by trivial) post
  have h_split_l := splitAtEnd_noStructured h_ns []
  simp only [lowerInstrsP, List.cons_append, List.append_assoc, List.nil_append,
             List.append_nil, List.singleton_append] at hl h_split_b h_split_l
  rw [h_split_b] at hl
  simp only [Option.bind_eq_bind] at hl
  -- The block body: the wloop arm.
  rcases hb : lowerInstrsP (f + 1) (.block :: frames) ⟨s, []⟩
      (.wloop 0 :: (pref ++ .brIf 1 :: (body2 ++ [.br 0, .wend]))) with _ | ⟨spb, innerOps⟩
  · rw [hb] at hl; simp at hl
  rw [hb] at hl
  simp only [Option.some_bind] at hl
  simp only [lowerInstrsP] at hb
  rw [h_split_l] at hb
  simp only [Option.bind_eq_bind] at hb
  rcases hlb : lowerInstrsP f (.loopK :: .block :: frames) ⟨{ s with currentReg := [] }, []⟩
      (pref ++ .brIf 1 :: (body2 ++ [.br 0])) with _ | ⟨spl, bodyOps⟩
  · rw [hlb] at hb; simp at hb
  rw [hlb] at hb
  simp only [Option.some_bind] at hb
  have hlb' : lowerInstrsP f (.loopK :: .block :: frames) ⟨{ s with currentReg := [] }, []⟩
      (pref ++ [.brIf 1] ++ body2 ++ [.br 0]) = some (spl, bodyOps) := by
    simpa using hlb
  obtain ⟨s_m, s0, s_c, s_b, svCond, cond, opsPref, opsCommit, opsBody2,
          _, _, _, _, h_spl, _⟩ := blockWhileBody_lowerP h_pref h_body2 hlb'
  subst h_spl
  -- The loop close: one flag record, consumed here; the empty tail.
  simp only [hasPlainActive, List.any_cons, List.any_nil, Bool.not_true, Bool.and_false,
             Bool.or_false, Bool.false_eq_true, ↓reduceIte, lowerInstrsP, Option.bind_eq_bind,
             Option.some_bind, pure, stepPending, List.filterMap_cons, List.filterMap_nil,
             closeDecls, applyWraps, activeWraps, List.filter_cons, List.filter_nil,
             List.foldr_cons, List.foldr_nil] at hb
  simp at hb
  obtain ⟨h_spb, h_inner⟩ := hb
  subst h_spb
  -- The block close: the post from the close state, nothing pending.
  simp only at hl
  rcases hlp : lowerInstrsP (f + 1) frames ⟨{ s_b with currentReg := [] }, []⟩ post
      with _ | ⟨spp, postOps⟩
  · rw [hlp] at hl; simp at hl
  rw [hlp] at hl
  simp only [Option.some_bind, pure, stepPending_nil, closeDecls_nil, applyWraps_nil,
             List.append_nil, List.nil_append, Option.some.injEq, Prod.mk.injEq] at hl
  obtain ⟨h_spp, h_ops⟩ := hl
  subst h_inner
  refine ⟨s_b, s_c.nextReg + 1, bodyOps, postOps, hlb', ?_, ?_⟩
  · rw [hlp]
    cases spp
    simp only [LowerStateP.mk.injEq] at h_spp
    obtain ⟨h1, h2⟩ := h_spp
    subst h1
    subst h2
    rfl
  · rw [← h_ops]

/-- The N-iteration composition at the block: `n` continues then the
    `br_if 1` exit. The IR runs the flag's declaration, the loop op
    (`opLoop_n_iter_exit` on the body-state sequence), the no-op wrap
    of the empty block tail (the flag reads `true` — the exit set it),
    then the post from the loop-close state; WASM consumes the exit's
    target at the block and runs the post from the cleared exit state. -/
theorem preservation_blockWhile_nIterExit
    (f : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (h_kst_no_broke : kst.broke = false)
    (pref body2 post : List WasmInstr)
    (h_pref : StraightLineInstrs pref) (h_body2 : StraightLineInstrs body2)
    -- WASM-side iteration trace.
    (n : Nat) (entries bodyOuts : Fin (n + 1) → WasmState)
    (h_exit : (bodyOuts (Fin.last n)).branchTarget = some 1)
    (h_exit_nh : (bodyOuts (Fin.last n)).halted = false)
    (ws' : WasmState)
    (h_post_eval : evalInstrs (f + 1) { bodyOuts (Fin.last n) with branchTarget := none } post
                     = some ws')
    -- IR-side iteration trace, from the state after the flag's declaration.
    (s1 : LowerState) (flag : Quanta.KOps.Reg) (bodyOps : List KernelOp)
    (h_lb : lowerInstrsP f (.loopK :: .block :: frames) ⟨{ s with currentReg := [] }, []⟩
              (pref ++ [.brIf 1] ++ body2 ++ [.br 0])
            = some (⟨s1, [{ levels := 1, cond := flag, flag := true, skip := 0 }]⟩, bodyOps))
    (kstStates : Fin (n + 2) → Quanta.KOps.State)
    (h_kst_start : kstStates 0 = { kst with rf := regWrite kst.rf flag (vBool false) })
    (F_b : Nat)
    (h_ir_step : ∀ i : Fin (n + 1),
        evalOps F_b (kstStates i.castSucc) bodyOps = some (kstStates i.succ))
    (h_ir_continue : ∀ i : Fin n, (kstStates i.castSucc.succ).broke = false)
    (h_ir_exit : (kstStates (Fin.last (n + 1))).broke = true)
    (h_exit_refines : Refines (bodyOuts (Fin.last n)) { s1 with currentReg := [] }
                        (kstStates (Fin.last (n + 1))) layout)
    (h_flag_exit : regLookup (kstStates (Fin.last (n + 1))).rf flag = some (vBool true))
    -- Post-loop bridge, from the loop-close state.
    (post_preserves : ∀ {ws_p : WasmState} {kst_p : Quanta.KOps.State}
        (_R_p : Refines ws_p { s1 with currentReg := [] } kst_p layout)
        (_h_nb_p : ws_p.branchTarget = none)
        (_h_nh_p : ws_p.halted = false)
        (_h_nbk_p : kst_p.broke = false)
        {ws'_p : WasmState} {s'_p : LowerState} {postOps' : List KernelOp}
        (_hw_p : evalInstrs (f + 1) ws_p post = some ws'_p)
        (_hl_p : lowerInstrsP (f + 1) frames ⟨{ s1 with currentReg := [] }, []⟩ post
                   = some (⟨s'_p, []⟩, postOps')),
      ∃ (kst'_p : Quanta.KOps.State) (F : Nat),
        evalOps F kst_p postOps' = some kst'_p ∧
        Refines ws'_p s'_p kst'_p layout ∧
        BridgeClauses ws'_p kst'_p)
    (s' : LowerState) (ops : List KernelOp)
    (hl : lowerInstrsP (f + 2) frames ⟨s, []⟩
            (.block 0 :: .wloop 0 :: (pref ++ [.brIf 1] ++ body2 ++ [.br 0])
               ++ [.wend] ++ [] ++ [.wend] ++ post)
          = some (⟨s', []⟩, ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  obtain ⟨s1', flag', bodyOps', postOps, h_lb', hlp, h_ops⟩ :=
    blockWhile_lowerP h_pref h_body2 hl
  rw [h_lb] at h_lb'
  simp only [Option.some.injEq, Prod.mk.injEq, LowerStateP.mk.injEq, List.cons.injEq,
             PendingWrap.mk.injEq, and_true, true_and] at h_lb'
  obtain ⟨⟨h_s1, h_flag⟩, h_bodyOps⟩ := h_lb'
  subst h_s1; subst h_flag; subst h_bodyOps
  subst h_ops
  -- The post, from the exit state with its target cleared.
  have h_exit_nb : ({ bodyOuts (Fin.last n) with branchTarget := none } : WasmState).branchTarget
      = none := rfl
  obtain ⟨kst', F_p, h_ev_p, R_p, h_bridge_p⟩ :=
    post_preserves (h_exit_refines.reset_broke.clear_branch) h_exit_nb h_exit_nh
      (State.reset_broke_broke _) h_post_eval hlp
  let F : Nat := max (max F_b F_p) (n + 2)
  have h_F_ge_Fb : F_b ≤ F := by simp [F]; omega
  have h_F_ge_Fp : F_p ≤ F := by simp [F]; omega
  have h_F_ge_n : F ≥ n + 2 := by simp [F]; omega
  refine ⟨kst', F, ?_, R_p, h_bridge_p⟩
  -- The declaration.
  rw [List.cons_append, evalOps_decl h_kst_no_broke, ← h_kst_start]
  -- The loop op.
  have h_ir_step_F : ∀ i : Fin (n + 1),
      evalOps F (kstStates i.castSucc) bodyOps = some (kstStates i.succ) :=
    fun i => evalOps_fuel_mono h_F_ge_Fb (h_ir_step i)
  have h_no_broke_seq : ∀ i : Fin (n + 1), (kstStates i.castSucc).broke = false := by
    intro i
    match i, i.isLt with
    | ⟨0, _⟩, _ =>
        show (kstStates 0).broke = false
        rw [h_kst_start]; exact h_kst_no_broke
    | ⟨k + 1, h_lt⟩, _ =>
        show (kstStates ⟨k + 1, by omega⟩).broke = false
        exact h_ir_continue ⟨k, by omega⟩
  have h_loop : Quanta.KOps.evalOp F (kstStates 0) (.loopOp bodyOps)
      = some (kstStates (Fin.last (n + 1))).reset_broke := by
    simp only [Quanta.KOps.evalOp]
    exact opLoop_n_iter_exit kstStates h_ir_step_F h_no_broke_seq h_ir_exit (by omega)
  rw [List.cons_append, evalOps_cons_continue h_loop (State.reset_broke_broke _)]
  -- The no-op wrap: the flag reads `true`.
  have h_flag' : regLookup (kstStates (Fin.last (n + 1))).reset_broke.rf flag
      = some (vBool true) := by
    rw [State.reset_broke_rf]; exact h_flag_exit
  rw [List.cons_append, List.nil_append,
      evalOps_tailWrap_skips (State.reset_broke_broke _) h_flag']
  exact evalOps_fuel_mono h_F_ge_Fp h_ev_p

end Quanta.Wasm
