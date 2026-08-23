/-
# The backedge + exit-tail loop (L15)

rustc's other loop: `block { loop { pref; br_if 0; br 1 } } post` — the
body computes the continue condition, the `br_if 0` is the backedge
(continue while the condition holds), and the unconditional `br 1` on
its fall-through path is the exit, crossing the loop to the block.
Production records the backedge on the Loop frame and wraps the tail
at loop close: `Branch { cond, then: [], else: [flag := true, Break] }`
(`crates/gpu/quanta-wasm-lowering/tests/lower_backedge_exit_tail.rs`);
the model's Stage-B `brIf 0` arm nests the tail in the else arm the
same way. Semantically it is the rotated do-while: the same WASM
trace as `wloop 0 { pref; br_if 0 }`, lowered through the exit flag.

This file proves the body's lemmas — decomposition, WASM continue-or-
exit, one IR iteration, the N-iteration trace — and plugs them into
the body-agnostic block composition `preservation_blockLoop_nIterExit`.
The apex arm lives in `PreservationKernelWhile`.
-/

import Quanta.Wasm.PreservationBlockWhile

namespace Quanta.Wasm

open Quanta.KOps (KernelOp evalOps evalOp regLookup regWrite vBool State)

-- ════════════════════════════════════════════════════════════════════
-- The lowering shape
-- ════════════════════════════════════════════════════════════════════

/-- The Stage-B arm for `br_if 0` inside a loop inside a block: pop and
    commit the condition, allocate the bool register, lower the rest,
    and nest it as the else arm of the backedge branch (with the
    end-of-body Break rule on the tail). -/
theorem lowerInstrsP_brIf0_backedge
    (fuel : Nat) (frames : List FrameKind) (s : LowerState) (p : List PendingWrap)
    (rest : List WasmInstr) :
    lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s, p⟩ (.brIf 0 :: rest) =
      (do
        let (svCond, s0) ← s.popSym
        let (cond, s1, opsCommit) ← s0.commit svCond
        let (cond_bool, s_cast) := s1.alloc
        let (s2, postOps) ← lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s_cast, p⟩ rest
        pure (s2,
          opsCommit
          ++ [.cast cond_bool cond .u32 .bool,
              .branch cond_bool []
                (postOps ++ backedgeEndBreak (tailReenters 0 rest) postOps)])) := by
  simp only [lowerInstrsP]
  rcases hpop : s.popSym with _ | ⟨svCond, s0⟩
  · rfl
  simp only [Option.bind_eq_bind, Option.some_bind]
  rcases hcommit : s0.commit svCond with _ | ⟨cond, s1, opsCommit⟩
  · rfl
  simp [LowerState.alloc]

/-- The unconditional `br 1` inside a loop inside a block, alone: the
    exit-flag route — allocate the flag, set it, break; one exit-flag
    record left pending. -/
theorem lowerInstrsP_br1_exit
    (fuel : Nat) (frames : List FrameKind) (s : LowerState) (p : List PendingWrap) :
    lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s, p⟩ [.br 1] =
      some (⟨(s.alloc).2,
             ({ levels := 1, cond := (s.alloc).1, flag := true, skip := 0 } : PendingWrap) :: p⟩,
            [.const (s.alloc).1 (.bool true), .breakOp]) := by
  have h_idx : List.findIdx (fun x => decide (x = FrameKind.loopK)) [FrameKind.loopK] = 0 := by
    decide
  simp [lowerInstrsP, hasLoopAbove, loopsAbove, exitFlagEntry, loopIndex, LowerState.alloc, h_idx]

/-- The body is marker-free. -/
theorem backedgeTailBody_noStructured {pref : List WasmInstr}
    (h_pref : StraightLineInstrs pref) :
    NoStructured (pref ++ [.brIf 0] ++ [.br 1]) :=
  noStructured_append
    (noStructured_append (straightLine_noStructured_list h_pref) ⟨trivial, trivial⟩)
    ⟨trivial, trivial⟩

/-- The body's lowering, taken apart: the prefix (plain Stage A), the
    popped and committed condition, the bool register `s_c.nextReg`,
    the flag `s_c.nextReg + 1`, and the backedge branch carrying the
    exit in its else arm. One exit-flag record is left pending. -/
theorem backedgeTailBody_lowerP
    {fuel : Nat} {frames : List FrameKind} {pref : List WasmInstr}
    (h_pref : StraightLineInstrs pref)
    {s : LowerState} {sp' : LowerStateP} {ops : List KernelOp}
    (hl : lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s, []⟩
            (pref ++ [.brIf 0] ++ [.br 1]) = some (sp', ops)) :
    ∃ (s_m s0 s_c : LowerState) (svCond : SymVal) (cond : Quanta.KOps.Reg)
      (opsPref opsCommit : List KernelOp),
      lowerInstrs fuel (.loopK :: .block :: frames) s pref = some (s_m, opsPref) ∧
      s_m.popSym = some (svCond, s0) ∧
      s0.commit svCond = some (cond, s_c, opsCommit) ∧
      sp' = ⟨{ s_c with nextReg := s_c.nextReg + 2 },
             [{ levels := 1, cond := s_c.nextReg + 1, flag := true, skip := 0 }]⟩ ∧
      ops = opsPref ++ opsCommit
              ++ [.cast s_c.nextReg cond .u32 .bool,
                  .branch s_c.nextReg []
                    [.const (s_c.nextReg + 1) (.bool true), .breakOp]] := by
  have h_list : pref ++ [.brIf 0] ++ [.br 1] = pref ++ (.brIf 0 :: [.br 1]) := by simp
  rw [h_list] at hl
  obtain ⟨s_m, opsPref, ops2, hl_pref, hl_rest, h_ops⟩ :=
    lowerInstrsP_straightLine_append h_pref hl
  rw [lowerInstrsP_brIf0_backedge] at hl_rest
  rcases h_pop : s_m.popSym with _ | ⟨svCond, s0⟩
  · rw [h_pop] at hl_rest; simp at hl_rest
  rw [h_pop] at hl_rest
  simp only [Option.bind_eq_bind, Option.some_bind] at hl_rest
  rcases h_commit : s0.commit svCond with _ | ⟨cond, s_c, opsCommit⟩
  · rw [h_commit] at hl_rest; simp at hl_rest
  rw [h_commit] at hl_rest
  simp only [Option.some_bind, LowerState.alloc] at hl_rest
  rw [lowerInstrsP_br1_exit] at hl_rest
  simp only [LowerState.alloc, Option.some_bind, pure, Option.some.injEq, Prod.mk.injEq,
             tailReenters, backedgeEndBreak, endsInBreak, List.append_nil] at hl_rest
  obtain ⟨h_sp, h_ops2⟩ := hl_rest
  refine ⟨s_m, s0, s_c, svCond, cond, opsPref, opsCommit, hl_pref, h_pop, h_commit, ?_, ?_⟩
  · rw [← h_sp]
  · rw [h_ops, ← h_ops2]
    simp [List.append_assoc]

-- ════════════════════════════════════════════════════════════════════
-- WASM: the body continues or exits
-- ════════════════════════════════════════════════════════════════════

/-- `br_if 0` in a loop: pops the condition; fires the backedge
    (`some 0`) when it is non-zero, falls through otherwise. -/
theorem brIf0_in_loop_backedge
    {fuel : Nat} {ws : WasmState} {c : UInt32} {rest_w : List WasmValue}
    (rest : List WasmInstr)
    (h_nb : ws.branchTarget = none) (h_nh : ws.halted = false)
    (h_stack : ws.stack = .wI32 c :: rest_w) :
    evalInstrs fuel ws (.brIf 0 :: rest)
      = (if c = 0 then evalInstrs fuel { ws with stack := rest_w } rest
         else some { ws with stack := rest_w, branchTarget := some 0 }) := by
  rw [evalInstrs_cons_default fuel ws (.brIf 0) rest h_nb h_nh rfl]
  simp only [evalInstr, WasmState.pop, h_stack, Option.bind_eq_bind, Option.some_bind]
  by_cases hc : c = 0
  · simp only [hc, ↓reduceIte]
  · simp only [hc, ↓reduceIte]
    exact evalInstrs_of_branch_set rfl rest

/-- `br 1` in a loop: the exit target. -/
theorem br1_in_loop_exits
    {fuel : Nat} {ws : WasmState}
    (h_nb : ws.branchTarget = none) (h_nh : ws.halted = false) :
    evalInstrs fuel ws [.br 1] = some { ws with branchTarget := some 1 } := by
  rw [evalInstrs_cons_default fuel ws (.br 1) [] h_nb h_nh rfl]
  simp only [evalInstr, evalInstrs]

/-- The body `pref ++ [br_if 0] ++ [br 1]` with straight-line `pref`
    continues (`some 0`, the backedge) or exits (`some 1`, the tail). -/
theorem backedgeTailBody_continuesOrExits1
    {fuel : Nat} {pref : List WasmInstr}
    (h_pref : StraightLineInstrs pref) :
    BodyContinuesOrExits1 fuel (pref ++ [.brIf 0] ++ [.br 1]) := by
  intro st st' h_nb h_nh hw
  simp only [List.append_assoc, List.cons_append, List.nil_append] at hw
  obtain ⟨ws_m, _, h_mb, h_mh, hw_br⟩ :=
    evalInstrs_straightLine_append h_pref h_nb h_nh hw
  rw [evalInstrs_cons_default fuel ws_m (.brIf 0) [WasmInstr.br 1] h_mb h_mh rfl] at hw_br
  cases he : evalInstr ws_m (.brIf 0) with
  | none => rw [he] at hw_br; exact Option.noConfusion hw_br
  | some ws1 =>
      rw [he] at hw_br
      simp only at hw_br
      obtain ⟨c, rest_w, _, h_branch⟩ := evalInstr_brIf_shape_pub he
      rcases h_branch with ⟨_, h_ws1⟩ | ⟨_, h_ws1⟩
      · -- fell through: `br 1` exits.
        subst h_ws1
        rw [br1_in_loop_exits (ws := { ws_m with stack := rest_w }) h_mb h_mh,
            Option.some.injEq] at hw_br
        subst hw_br
        exact ⟨h_mh, Or.inr rfl⟩
      · -- the backedge fired: the rest is skipped.
        subst h_ws1
        rw [evalInstrs_of_branch_set rfl, Option.some.injEq] at hw_br
        subst hw_br
        exact ⟨h_mh, Or.inl rfl⟩

-- ════════════════════════════════════════════════════════════════════
-- IR: the backedge site
-- ════════════════════════════════════════════════════════════════════

/-- The backedge site CONTINUES: the condition reads `true`, the empty
    then-arm runs, and evaluation goes on with `rest` unchanged. -/
theorem evalOps_backedgeSite_continues {F : Nat} {kst : State} {cb flag : Quanta.KOps.Reg}
    {rest : List KernelOp}
    (h_cb : regLookup kst.rf cb = some (vBool true)) (h_br : kst.broke = false) :
    evalOps F kst (.branch cb [] [.const flag (.bool true), .breakOp] :: rest)
      = evalOps F kst rest := by
  have h_site : evalOp F kst (.branch cb [] [.const flag (.bool true), .breakOp]) = some kst := by
    rw [evalOp_branch_true h_cb]
    simp only [evalOps]
  exact evalOps_cons_continue h_site h_br

/-- The backedge site EXITS: the condition reads `false`, the else arm
    sets the flag and breaks, `rest` is skipped. -/
theorem evalOps_backedgeSite_exits {F : Nat} {kst : State} {cb flag : Quanta.KOps.Reg}
    {rest : List KernelOp}
    (h_cb : regLookup kst.rf cb = some (vBool false)) :
    evalOps F kst (.branch cb [] [.const flag (.bool true), .breakOp] :: rest)
      = some { kst with rf := regWrite kst.rf flag (vBool true), broke := true } := by
  have h_site : evalOp F kst (.branch cb [] [.const flag (.bool true), .breakOp])
      = some { kst with rf := regWrite kst.rf flag (vBool true), broke := true } := by
    rw [evalOp_branch_false h_cb, evalOps_setFlag_break]
  exact evalOps_cons_stop h_site rfl

-- ════════════════════════════════════════════════════════════════════
-- One iteration
-- ════════════════════════════════════════════════════════════════════

/-- One run of the body against one run of its lowering. Both outcomes
    refine the post-body state `sp'.base` (the whole body ran either
    way — the site sits at its end):
    * continue — `c ≠ 0`: the backedge fires, WASM ends at
      `branchTarget = some 0`, the IR at `broke = false`;
    * exit — `c = 0`: the tail runs, WASM ends at `some 1`, the IR at
      `broke = true` with the flag set. -/
theorem backedgeTailBody_iteration
    (fuel : Nat) (frames : List FrameKind) (pref : List WasmInstr)
    (h_pref : StraightLineInstrs pref)
    (ws : WasmState) (s : LowerState) (kst : State) (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (h_buf_locals : ∀ (ws_x : WasmState) (s_x : LowerState),
        BufferLocalsWellFormed layout ws_x s_x)
    (h_no_buf_stack : ∀ (s_x : LowerState), NoBufferPatternStack s_x)
    (h_load_bounds : ∀ (s_x : LowerState) (kst_x : State),
        LoadAddressesInBounds layout s_x kst_x)
    (h_store_bounds : ∀ (s_x : LowerState) (kst_x : State),
        StoreAddressInBounds layout s_x kst_x)
    (h_store_layout : ∀ (s_x : LowerState) (kst_x : State),
        StoreLayoutNoOverlap layout s_x kst_x)
    (ws' : WasmState) (sp' : LowerStateP) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (pref ++ [.brIf 0] ++ [.br 1]) = some ws')
    (hl : lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s, []⟩
            (pref ++ [.brIf 0] ++ [.br 1]) = some (sp', ops)) :
    ∃ (flag : Quanta.KOps.Reg) (kst' : State) (F : Nat),
      sp'.pending = [{ levels := 1, cond := flag, flag := true, skip := 0 }] ∧
      s.nextReg ≤ flag ∧ flag < sp'.base.nextReg ∧
      evalOps F kst ops = some kst' ∧
      ws'.halted = false ∧
      ((ws'.branchTarget = some 0 ∧ kst'.broke = false ∧
          Refines ws' sp'.base kst' layout) ∨
       (ws'.branchTarget = some 1 ∧ kst'.broke = true ∧
          Refines ws' sp'.base kst' layout ∧
          regLookup kst'.rf flag = some (vBool true))) := by
  obtain ⟨s_m, s0, s_c, svCond, cond, opsPref, opsCommit,
          hl_pref, h_pop, h_commit, h_sp, h_ops⟩ :=
    backedgeTailBody_lowerP h_pref hl
  subst h_sp
  subst h_ops
  -- WASM: the prefix, then the brIf.
  have h_list : pref ++ [.brIf 0] ++ [.br 1] = pref ++ (.brIf 0 :: [.br 1]) := by simp
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
  rw [evalInstrs_cons_default fuel ws_m (WasmInstr.brIf 0) [WasmInstr.br 1] h_mb h_mh rfl]
    at hw_rest
  cases he : evalInstr ws_m (.brIf 0) with
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
  let kst_cast : State :=
    { kst1 with rf := regWrite kst1.rf s_c.nextReg (vBool (!decide (c = 0))) }
  have h_cast : ∀ F, evalOp F kst1
      (KernelOp.cast s_c.nextReg cond Quanta.KOps.Scalar.u32 Quanta.KOps.Scalar.bool)
      = some kst_cast := by
    intro F
    rcases h_lookup with h_u32 | h_i32
    · simp [evalOp, h_u32, Quanta.KOps.evalCast, kst_cast]
    · simp [evalOp, h_i32, Quanta.KOps.evalCast, kst_cast, h_c_toNat]
  have h_kst_cast_ok : kst_cast.broke = false := h_kst1_ok
  have h_cb : regLookup kst_cast.rf s_c.nextReg = some (vBool (!decide (c = 0))) :=
    regLookup_regWrite_self _ _ _
  -- Refines at the post-body state.
  have R_cast : Refines { ws_m with stack := rest_w } s_c kst_cast layout :=
    R1.regWrite_fresh (Nat.le_refl _) _
  have R_body : Refines { ws_m with stack := rest_w }
      { s_c with nextReg := s_c.nextReg + 2 } kst_cast layout :=
    R_cast.bump_nextReg (by omega)
  -- The IR run up to the site.
  have h_ev_to_site : ∀ F, F1 ≤ F →
      evalOps F kst (opsPref ++ opsCommit
        ++ [KernelOp.cast s_c.nextReg cond .u32 .bool,
            KernelOp.branch s_c.nextReg [] [.const (s_c.nextReg + 1) (.bool true), .breakOp]])
      = evalOps F kst_cast
          [KernelOp.branch s_c.nextReg [] [.const (s_c.nextReg + 1) (.bool true), .breakOp]] := by
    intro F hF
    rw [List.append_assoc,
        evalOps_append (evalOps_fuel_mono hF h_ev1) h_kst_m_ok,
        evalOps_append (evalOps_fuel_mono (Nat.zero_le F) h_ev_commit) h_kst1_ok]
    rw [evalOps_cons_continue (h_cast F) h_kst_cast_ok]
  rcases h_branch with ⟨hc, h_ws1⟩ | ⟨hc, h_ws1⟩
  · -- exit: the condition is zero, the backedge does not fire, `br 1`.
    subst h_ws1
    have h_cb_false : regLookup kst_cast.rf s_c.nextReg = some (vBool false) := by
      rw [h_cb]; simp [hc]
    rw [br1_in_loop_exits (ws := { ws_m with stack := rest_w }) h_mb h_mh,
        Option.some.injEq] at hw_rest
    subst hw_rest
    let kst_exit : State :=
      { kst_cast with rf := regWrite kst_cast.rf (s_c.nextReg + 1) (vBool true), broke := true }
    refine ⟨s_c.nextReg + 1, kst_exit, F1, rfl, by omega, by simp, ?_, h_mh,
            Or.inr ⟨rfl, rfl, ?_, ?_⟩⟩
    · rw [h_ev_to_site F1 (Nat.le_refl _)]
      exact evalOps_backedgeSite_exits h_cb_false
    · exact ((R_cast.regWrite_fresh_set_broke (by omega) _ _).bump_nextReg (by omega)).set_branch _
    · exact regLookup_regWrite_self _ _ _
  · -- continue: the backedge fires, the tail is skipped.
    subst h_ws1
    have h_cb_true : regLookup kst_cast.rf s_c.nextReg = some (vBool true) := by
      rw [h_cb]; simp [hc]
    rw [evalInstrs_of_branch_set rfl, Option.some.injEq] at hw_rest
    subst hw_rest
    refine ⟨s_c.nextReg + 1, kst_cast, F1, rfl, by omega, by simp, ?_, h_mh,
            Or.inl ⟨rfl, h_kst_cast_ok, R_body.set_branch _⟩⟩
    rw [h_ev_to_site F1 (Nat.le_refl _),
        evalOps_backedgeSite_continues h_cb_true h_kst_cast_ok]
    simp [evalOps]

-- ════════════════════════════════════════════════════════════════════
-- Frames and the N-iteration trace
-- ════════════════════════════════════════════════════════════════════

/-- The body's lowering frame: the stack is restored (the prefix pushes
    the condition, the `br_if` pops it), buffer slots are kept, the
    stable layer only grows, `nextReg` only grows. -/
theorem backedgeTailBody_frames
    {fuel : Nat} {frames : List FrameKind} {pref : List WasmInstr}
    (h_pref : StraightLineInstrs pref) (h_ht_pref : stackHeight 0 pref = some 1)
    {s s1 : LowerState} {flag : Quanta.KOps.Reg} {bodyOps : List KernelOp}
    (h_lb : lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s, []⟩ (pref ++ [.brIf 0] ++ [.br 1])
      = some (⟨s1, [{ levels := 1, cond := flag, flag := true, skip := 0 }]⟩, bodyOps)) :
    s1.stack = s.stack ∧ s1.bufferSlots = s.bufferSlots ∧
      s.nextReg ≤ s1.nextReg ∧ LocalsExtend s s1 := by
  obtain ⟨s_m, s0, s_c, svCond, cond, opsPref, opsCommit,
          hl_pref, h_pop, h_commit, h_sp, _⟩ :=
    backedgeTailBody_lowerP h_pref h_lb
  simp only [LowerStateP.mk.injEq, List.cons.injEq, PendingWrap.mk.injEq, and_true,
             true_and] at h_sp
  obtain ⟨h_s1, _⟩ := h_sp
  subst h_s1
  obtain ⟨⟨_, h_len, h_drop, h_bs, h_nr⟩, h_ext⟩ :=
    lowerInstrs_bodyFrame_from h_pref h_ht_pref (StackFrame.refl s) (LocalsExtend.refl s) hl_pref
  have h_s0_stack : s0.stack = s.stack := by
    unfold LowerState.popSym at h_pop
    rcases hs : s_m.stack with _ | ⟨sv, rs⟩
    · rw [hs] at h_pop; simp at h_pop
    · rw [hs] at h_pop; simp at h_pop
      obtain ⟨_, h_eq⟩ := h_pop
      rw [← h_eq]
      simp only
      rw [hs] at h_drop
      simpa using h_drop
  have h_pop_nr := LowerState.popSym_nextReg h_pop
  have h_pop_lr := LowerState.popSym_localReg h_pop
  have h_pop_bs := LowerState.popSym_preserves_bufferSlots h_pop
  have h_c_stk := LowerState.commit_stack h_commit
  have h_c_lr := LowerState.commit_localReg h_commit
  have h_c_bs := LowerState.commit_preserves_bufferSlots h_commit
  have h_c_nr := LowerState.commit_nextReg_mono h_commit
  refine ⟨?_, ?_, ?_, ?_⟩
  · show s_c.stack = s.stack
    rw [h_c_stk, h_s0_stack]
  · show s_c.bufferSlots = s.bufferSlots
    rw [h_c_bs, h_pop_bs, h_bs]
  · show s.nextReg ≤ s_c.nextReg + 2
    omega
  · exact h_ext.trans (LocalsExtend.of_eq (show s_c.localReg = s_m.localReg by
      rw [h_c_lr, h_pop_lr]))

/-- The IR runs the lowered body once per WASM iteration: one fuel for
    all runs, `broke = false` after every continue, `broke = true` after
    the exit, the exit body-out refining the loop-close state, the flag
    reading `true` there. Side condition: the body keeps every label
    (`s1.localTy = s.localTy`) — the rotated do-while has no exit site
    before the body's writes, so no register condition arises. -/
theorem backedgeTailBody_ir_trace
    (fuel : Nat) (frames : List FrameKind) (pref : List WasmInstr)
    (h_pref : StraightLineInstrs pref) (h_ht_pref : stackHeight 0 pref = some 1)
    (s s1 : LowerState) (flag : Quanta.KOps.Reg) (bodyOps : List KernelOp)
    (h_lb : lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s, []⟩ (pref ++ [.brIf 0] ++ [.br 1])
            = some (⟨s1, [{ levels := 1, cond := flag, flag := true, skip := 0 }]⟩, bodyOps))
    (h_cr : s.currentReg = [])
    (h_lt : s1.localTy = s.localTy)
    (layout : BufferLayout)
    (h_buf_locals : ∀ (ws_x : WasmState) (s_x : LowerState),
        BufferLocalsWellFormed layout ws_x s_x)
    (h_no_buf_stack : ∀ (s_x : LowerState), NoBufferPatternStack s_x)
    (h_load_bounds : ∀ (s_x : LowerState) (kst_x : State),
        LoadAddressesInBounds layout s_x kst_x)
    (h_store_bounds : ∀ (s_x : LowerState) (kst_x : State),
        StoreAddressInBounds layout s_x kst_x)
    (h_store_layout : ∀ (s_x : LowerState) (kst_x : State),
        StoreLayoutNoOverlap layout s_x kst_x)
    (n : Nat) (entries bodyOuts : Fin (n + 1) → WasmState)
    (h_step : ∀ i : Fin (n + 1),
        evalInstrs fuel (entries i) (pref ++ [.brIf 0] ++ [.br 1]) = some (bodyOuts i))
    (h_cont : ∀ i : Fin n,
        (bodyOuts i.castSucc).branchTarget = some 0 ∧
        entries i.succ = { bodyOuts i.castSucc with branchTarget := none })
    (h_exit : (bodyOuts (Fin.last n)).branchTarget = some 1)
    (kst : State)
    (R : Refines (entries 0) s kst layout)
    (h_e0_nb : (entries 0).branchTarget = none)
    (h_e0_nh : (entries 0).halted = false)
    (h_kst : kst.broke = false) :
    ∃ (kstStates : Fin (n + 2) → State) (F_b : Nat),
      kstStates 0 = kst ∧
      (∀ i : Fin (n + 1),
        evalOps F_b (kstStates i.castSucc) bodyOps = some (kstStates i.succ)) ∧
      (∀ i : Fin n, (kstStates i.castSucc.succ).broke = false) ∧
      (kstStates (Fin.last (n + 1))).broke = true ∧
      Refines (bodyOuts (Fin.last n)) { s1 with currentReg := [] }
        (kstStates (Fin.last (n + 1))) layout ∧
      regLookup (kstStates (Fin.last (n + 1))).rf flag = some (vBool true) ∧
      (∀ i : Fin (n + 1), (bodyOuts i).halted = false) := by
  -- One iteration, with the flag pinned to the lowering's, and the
  -- frames around it.
  obtain ⟨h_stk1, h_bs1, h_nr1, h_ext1⟩ := backedgeTailBody_frames h_pref h_ht_pref h_lb
  have iter : ∀ (ws_i : WasmState) (kst_i : State) (ws_o : WasmState),
      Refines ws_i s kst_i layout → ws_i.branchTarget = none → ws_i.halted = false →
      kst_i.broke = false →
      evalInstrs fuel ws_i (pref ++ [.brIf 0] ++ [.br 1]) = some ws_o →
      ∃ (kst' : State) (F : Nat),
        evalOps F kst_i bodyOps = some kst' ∧ ws_o.halted = false ∧
        ((ws_o.branchTarget = some 0 ∧ kst'.broke = false ∧ Refines ws_o s kst' layout) ∨
         (ws_o.branchTarget = some 1 ∧ kst'.broke = true ∧
            Refines ws_o { s1 with currentReg := [] } kst' layout ∧
            regLookup kst'.rf flag = some (vBool true))) := by
    intro ws_i kst_i ws_o R_i h_nb h_nh h_ok hw
    obtain ⟨flag', kst', F, h_pend, _, _, h_ev, h_nh_o, h_out⟩ :=
      backedgeTailBody_iteration fuel frames pref h_pref ws_i s kst_i layout R_i
        h_nb h_nh h_ok h_buf_locals h_no_buf_stack h_load_bounds h_store_bounds h_store_layout
        ws_o _ bodyOps hw h_lb
    simp only [List.cons.injEq, PendingWrap.mk.injEq, and_true, true_and] at h_pend
    subst h_pend
    refine ⟨kst', F, h_ev, h_nh_o, ?_⟩
    rcases h_out with ⟨h_bt, h_b, R_o⟩ | ⟨h_bt, h_b, R_o, h_fl⟩
    · left
      exact ⟨h_bt, h_b, Refines.retarget R_o R_i h_stk1 h_ext1 h_lt h_cr⟩
    · right
      exact ⟨h_bt, h_b, R_o.to_close rfl rfl rfl rfl (Nat.le_refl _), h_fl⟩
  induction n generalizing kst with
  | zero =>
      obtain ⟨kst1, F, h_ev, h_nh1, h_out⟩ :=
        iter (entries 0) kst (bodyOuts 0) R h_e0_nb h_e0_nh h_kst (h_step 0)
      have h_exit0 : (bodyOuts 0).branchTarget = some 1 := h_exit
      obtain ⟨h_broke, R_close, h_fl⟩ :
          kst1.broke = true ∧ Refines (bodyOuts 0) { s1 with currentReg := [] } kst1 layout ∧
          regLookup kst1.rf flag = some (vBool true) := by
        rcases h_out with ⟨h_bt, _⟩ | ⟨_, h_b, R_c, h_fl⟩
        · rw [h_exit0] at h_bt; simp at h_bt
        · exact ⟨h_b, R_c, h_fl⟩
      refine ⟨fun i => if i.val = 0 then kst else kst1, F, by simp, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · intro i
        have h0 : i = 0 := Fin.ext (by omega)
        subst h0
        simpa using h_ev
      · intro i; exact absurd i.isLt (by simp)
      · simpa using h_broke
      · simpa using R_close
      · simpa using h_fl
      · intro i
        have h0 : i = 0 := Fin.ext (by omega)
        subst h0
        exact h_nh1
  | succ n IH =>
      obtain ⟨kst1, F0, h_ev0, h_nh1, h_out0⟩ :=
        iter (entries 0) kst (bodyOuts 0) R h_e0_nb h_e0_nh h_kst (h_step 0)
      obtain ⟨h_bt0, h_e1⟩ := h_cont 0
      have h_bt0' : (bodyOuts 0).branchTarget = some 0 := h_bt0
      obtain ⟨h_kst1_ok, R1⟩ : kst1.broke = false ∧ Refines (bodyOuts 0) s kst1 layout := by
        rcases h_out0 with ⟨_, h_b, R_o⟩ | ⟨h_bt, _⟩
        · exact ⟨h_b, R_o⟩
        · rw [h_bt] at h_bt0'; simp at h_bt0'
      have h_e1' : entries 1 = { bodyOuts 0 with branchTarget := none } := h_e1
      obtain ⟨seq', F', h_s0', h_step', h_cont', h_exit', R_close', h_fl', h_nh'⟩ :=
        IH (fun i => entries i.succ) (fun i => bodyOuts i.succ)
          (fun i => h_step i.succ)
          (fun i => by
            obtain ⟨h1, h2⟩ := h_cont i.succ
            exact ⟨h1, h2⟩)
          h_exit kst1
          (by
            show Refines (entries 1) s kst1 layout
            rw [h_e1']
            exact R1.clear_branch)
          (by show (entries 1).branchTarget = none; rw [h_e1'])
          (by show (entries 1).halted = false; rw [h_e1']; exact h_nh1)
          h_kst1_ok
      refine ⟨fun i => if h : i.val = 0 then kst else seq' ⟨i.val - 1, by omega⟩,
              max F0 F', by simp, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · intro i
        by_cases h : i.val = 0
        · have h_cs : (i.castSucc).val = 0 := by simp [h]
          have h_sc : (i.succ).val = 1 := by simp [h]
          simp only [h_cs, h_sc, ↓reduceDIte, Nat.one_ne_zero, Nat.sub_self]
          have : (⟨0, by omega⟩ : Fin (n + 2)) = 0 := rfl
          rw [this, h_s0']
          exact evalOps_fuel_mono (Nat.le_max_left _ _) h_ev0
        · have h_cs : (i.castSucc).val = i.val := by simp
          have h_sc : (i.succ).val = i.val + 1 := by simp
          have h_ne : i.val + 1 ≠ 0 := by omega
          simp only [h_cs, h_sc, h, ↓reduceDIte, h_ne, Nat.add_sub_cancel]
          have h_st := h_step' ⟨i.val - 1, by omega⟩
          simp only [Fin.castSucc_mk, Fin.succ_mk] at h_st
          have h_idx : i.val - 1 + 1 = i.val := by omega
          have h_fin : (⟨i.val - 1 + 1, by omega⟩ : Fin (n + 2)) = ⟨i.val, by omega⟩ :=
            Fin.ext h_idx
          rw [h_fin] at h_st
          exact evalOps_fuel_mono (Nat.le_max_right _ _) h_st
      · intro i
        by_cases h : i.val = 0
        · have h_v : (i.castSucc.succ).val = 1 := by simp [h]
          simp only [h_v, ↓reduceDIte, Nat.one_ne_zero, Nat.sub_self]
          have : (⟨0, by omega⟩ : Fin (n + 2)) = 0 := rfl
          rw [this, h_s0']; exact h_kst1_ok
        · have h_v : (i.castSucc.succ).val = i.val + 1 := by simp
          have h_ne : i.val + 1 ≠ 0 := by omega
          simp only [h_v, ↓reduceDIte, h_ne, Nat.add_sub_cancel]
          have h_c := h_cont' ⟨i.val - 1, by omega⟩
          simp only [Fin.castSucc_mk, Fin.succ_mk] at h_c
          have h_idx : i.val - 1 + 1 = i.val := by omega
          have h_fin : (⟨i.val - 1 + 1, by omega⟩ : Fin (n + 2)) = ⟨i.val, by omega⟩ :=
            Fin.ext h_idx
          rw [h_fin] at h_c
          exact h_c
      · have h_last : (Fin.last (n + 2)).val = n + 2 := by simp
        simp only [h_last, Nat.succ_ne_zero, ↓reduceDIte]
        have : (⟨n + 2 - 1, by omega⟩ : Fin (n + 2)) = Fin.last (n + 1) := Fin.ext (by simp)
        rw [this]; exact h_exit'
      · have h_last : (Fin.last (n + 2)).val = n + 2 := by simp
        simp only [h_last, Nat.succ_ne_zero, ↓reduceDIte]
        have : (⟨n + 2 - 1, by omega⟩ : Fin (n + 2)) = Fin.last (n + 1) := Fin.ext (by simp)
        rw [this]
        have h_bo : bodyOuts (Fin.last (n + 1))
            = (fun i : Fin (n + 1) => bodyOuts i.succ) (Fin.last n) := rfl
        rw [h_bo]; exact R_close'
      · have h_last : (Fin.last (n + 2)).val = n + 2 := by simp
        simp only [h_last, Nat.succ_ne_zero, ↓reduceDIte]
        have : (⟨n + 2 - 1, by omega⟩ : Fin (n + 2)) = Fin.last (n + 1) := Fin.ext (by simp)
        rw [this]; exact h_fl'
      · intro i
        by_cases h : i.val = 0
        · have h_i : i = 0 := Fin.ext h
          subst h_i; exact h_nh1
        · have h_n := h_nh' ⟨i.val - 1, by omega⟩
          simp only [Fin.succ_mk] at h_n
          have h_idx : i.val - 1 + 1 = i.val := by omega
          have h_fin : (⟨i.val - 1 + 1, by omega⟩ : Fin (n + 2)) = i := Fin.ext h_idx
          rw [h_fin] at h_n
          exact h_n

-- ════════════════════════════════════════════════════════════════════
-- The whole segment
-- ════════════════════════════════════════════════════════════════════

/-- The `block { loop { pref; br_if 0; br 1 } } post` segment, taken
    apart: the body's lowering (one exit-flag record pending), the
    post's from the loop-close state, and the kernel ops — the flag's
    declaration, the loop op, the no-op wrap, the post. -/
theorem blockBackedgeTail_lowerP
    {f : Nat} {frames : List FrameKind} {pref post : List WasmInstr}
    (h_pref : StraightLineInstrs pref)
    {s s' : LowerState} {ops : List KernelOp}
    (hl : lowerInstrsP (f + 2) frames ⟨s, []⟩
            (.block 0 :: .wloop 0 :: (pref ++ [.brIf 0] ++ [.br 1])
               ++ [.wend] ++ [] ++ [.wend] ++ post)
          = some (⟨s', []⟩, ops)) :
    ∃ (s1 : LowerState) (flag : Quanta.KOps.Reg) (bodyOps postOps : List KernelOp),
      lowerInstrsP f (.loopK :: .block :: frames) ⟨{ s with currentReg := [] }, []⟩
          (pref ++ [.brIf 0] ++ [.br 1])
        = some (⟨s1, [{ levels := 1, cond := flag, flag := true, skip := 0 }]⟩, bodyOps) ∧
      lowerInstrsP (f + 1) frames ⟨{ s1 with currentReg := [] }, []⟩ post
        = some (⟨s', []⟩, postOps) ∧
      ops = [.const flag (.bool false), .loopOp bodyOps, .branch flag [] []] ++ postOps := by
  have h_ns : NoStructured (pref ++ [.brIf 0] ++ [.br 1]) := backedgeTailBody_noStructured h_pref
  -- Reach the body's lowering through the block and loop opens.
  have h_split_b := splitAtEnd_wloop_noStructured h_ns (tail := []) (by trivial) post
  have h_split_l := splitAtEnd_noStructured h_ns []
  have hl' := hl
  simp only [lowerInstrsP, List.cons_append, List.append_assoc, List.nil_append,
             List.append_nil, List.singleton_append] at hl' h_split_b h_split_l
  rw [h_split_b] at hl'
  simp only [Option.bind_eq_bind] at hl'
  rcases hb : lowerInstrsP (f + 1) (.block :: frames) ⟨s, []⟩
      (.wloop 0 :: (pref ++ .brIf 0 :: [.br 1, .wend])) with _ | ⟨spb, innerOps⟩
  · rw [hb] at hl'; simp at hl'
  simp only [lowerInstrsP] at hb
  rw [h_split_l] at hb
  simp only [Option.bind_eq_bind] at hb
  rcases hlb : lowerInstrsP f (.loopK :: .block :: frames) ⟨{ s with currentReg := [] }, []⟩
      (pref ++ .brIf 0 :: [.br 1]) with _ | ⟨spl, bodyOps⟩
  · rw [hlb] at hb; simp at hb
  have hlb' : lowerInstrsP f (.loopK :: .block :: frames) ⟨{ s with currentReg := [] }, []⟩
      (pref ++ [.brIf 0] ++ [.br 1]) = some (spl, bodyOps) := by
    simpa using hlb
  obtain ⟨_, _, s_c, _, _, _, _, _, _, _, h_spl, _⟩ := backedgeTailBody_lowerP h_pref hlb'
  subst h_spl
  obtain ⟨postOps, hlp, h_ops⟩ := blockLoop_lowerP h_ns hlb' hl
  exact ⟨_, _, bodyOps, postOps, hlb', hlp, h_ops⟩

/-- The flag is fresh for the entry and below the post-body `nextReg`. -/
theorem backedgeTailBody_flag_fresh
    {fuel : Nat} {frames : List FrameKind} {pref : List WasmInstr}
    (h_pref : StraightLineInstrs pref)
    {s s1 : LowerState} {flag : Quanta.KOps.Reg} {bodyOps : List KernelOp}
    (h_lb : lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s, []⟩ (pref ++ [.brIf 0] ++ [.br 1])
      = some (⟨s1, [{ levels := 1, cond := flag, flag := true, skip := 0 }]⟩, bodyOps)) :
    s.nextReg ≤ flag ∧ flag < s1.nextReg := by
  obtain ⟨s_m, s0, s_c, svCond, cond, opsPref, opsCommit,
          hl_pref, h_pop, h_commit, h_sp, _⟩ :=
    backedgeTailBody_lowerP h_pref h_lb
  simp only [LowerStateP.mk.injEq, List.cons.injEq, PendingWrap.mk.injEq, and_true,
             true_and] at h_sp
  obtain ⟨h_s1, h_flag⟩ := h_sp
  subst h_s1; subst h_flag
  have h_nr_m : s.nextReg ≤ s_m.nextReg := lowerInstrs_nextReg_mono _ _ _ _ hl_pref
  have h_pop_nr := LowerState.popSym_nextReg h_pop
  have h_c_nr := LowerState.commit_nextReg_mono h_commit
  constructor
  · show s.nextReg ≤ s_c.nextReg + 1
    omega
  · show s_c.nextReg + 1 < s_c.nextReg + 2
    omega

end Quanta.Wasm
