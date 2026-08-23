/-
# The apex over the pending-wrap translator: straight-line code, `do … while`, and rustc's `while`

`KernelInstrsW2` is `KernelInstrsW` plus rustc's `while`:

    block { loop { pref; br_if 1; body2; br 0 } } post

lowered by `lowerInstrsP` through the exit flag. The apex
`framework_preservation_kernel_while2` is stated over `lowerInstrsP` —
the first of the three arms is Stage A's own (`TranslatePendingAgree`),
the second reaches the Stage-A loop theorem through the decomposed
lowering (`…nIterExit_core`), the third is `PreservationBlockWhile`.
Its side condition, `KernelInstrsW2.stable`, is the label-stability
condition of the earlier apex, plus, for the new shape, that `body2`
rebinds no local's stable register or label past the exit site.
`KernelInstrsW` embeds (`KernelInstrsW.toW2`), so the earlier apex is a
corollary.
-/

import Quanta.Wasm.PreservationBlockWhile

namespace Quanta.Wasm

open Quanta.KOps (KernelOp evalOps regLookup regWrite vBool)

-- ════════════════════════════════════════════════════════════════════
-- The kernel shape
-- ════════════════════════════════════════════════════════════════════

/-- Kernel-body well-formedness: straight-line instructions, `wloop 0`
    segments with a `WhileBody`, and rustc's `while` — a block around a
    loop whose body is a straight-line prefix computing the condition,
    the `br_if 1` exit, a balanced straight-line body and the `br 0`
    continue. The block's tail (between the loop's `end` and the
    block's) is empty: the body never falls through. -/
inductive KernelInstrsW2 : List WasmInstr → Type
  | empty : KernelInstrsW2 []
  | sl_cons {i : WasmInstr} {rest : List WasmInstr} :
      StraightLineInstr i →
      KernelInstrsW2 rest →
      KernelInstrsW2 (i :: rest)
  | while_cons {rest body post : List WasmInstr} :
      splitAtEnd rest = some (body, post) →
      WhileBody body →
      KernelInstrsW2 post →
      KernelInstrsW2 (.wloop 0 :: rest)
  | block_while_cons {pref body2 post : List WasmInstr} :
      StraightLineInstrs pref →
      StraightLineInstrs body2 →
      stackHeight 0 pref = some 1 →
      stackHeight 0 body2 = some 0 →
      KernelInstrsW2 post →
      KernelInstrsW2 (.block 0 :: .wloop 0 :: (pref ++ [.brIf 1] ++ body2 ++ [.br 0])
                        ++ [.wend] ++ [] ++ [.wend] ++ post)

/-- Loop nesting depth — the fuel measure. A `while` counts one: its
    block and its loop spend the two units the bound already grants. -/
def KernelInstrsW2.depth : ∀ {instrs : List WasmInstr}, KernelInstrsW2 instrs → Nat
  | _, .empty => 0
  | _, .sl_cons _ rest_wf => rest_wf.depth
  | _, .while_cons _ _ post_wf => 1 + post_wf.depth
  | _, .block_while_cons _ _ _ _ post_wf => 1 + post_wf.depth

/-- The side conditions, lowered from the state the kernel actually
    reaches (the same `LoopsTypeStable` on the shapes the earlier apex
    has; see its docstring for why the relation needs it). For rustc's
    `while`: the body keeps every label, and past the exit site it
    rebinds no local's stable register or label — on the exit path
    `body2`'s registers were never written. -/
def KernelInstrsW2.stable : ∀ {instrs : List WasmInstr},
    KernelInstrsW2 instrs → Nat → List FrameKind → LowerState → Prop
  | _, .empty, _, _, _ => True
  | _, @sl_cons i _ _ rest_wf, fuel, frames, s =>
      ∀ s1 ops, lowerInstr s i = some (s1, ops) → rest_wf.stable fuel frames s1
  | _, @while_cons _ body _ _ _ post_wf, fuel, frames, s =>
      match fuel with
      | 0 => True
      | f + 1 =>
          ∀ s1 bodyOps,
            lowerInstrs f (.loopK :: frames) { s with currentReg := [] } body
              = some (s1, bodyOps) →
            s1.localTy = s.localTy ∧
            post_wf.stable f frames { s1 with currentReg := [] }
  | _, @block_while_cons pref body2 _ _ _ _ _ post_wf, fuel, frames, s =>
      match fuel with
      | 0 => True
      | 1 => True
      | f + 2 =>
          ∀ s_site p1 ops1,
            lowerInstrsP f (.loopK :: .block :: frames) ⟨{ s with currentReg := [] }, []⟩
                (pref ++ [.brIf 1]) = some (⟨s_site, p1⟩, ops1) →
            ∀ s1 p2 bodyOps,
              lowerInstrsP f (.loopK :: .block :: frames) ⟨{ s with currentReg := [] }, []⟩
                  (pref ++ [.brIf 1] ++ body2 ++ [.br 0]) = some (⟨s1, p2⟩, bodyOps) →
              s1.localTy = s.localTy ∧
              s1.localReg = s_site.localReg ∧ s1.localTy = s_site.localTy ∧
              post_wf.stable (f + 1) frames { s1 with currentReg := [] }

-- ════════════════════════════════════════════════════════════════════
-- The site's lowering, out of the body's
-- ════════════════════════════════════════════════════════════════════

/-- From the body's Stage-B lowering, the exit site's: the prefix and
    the `br_if` alone, ending at the state `body2` is lowered from; and
    the flag is fresh for the entry state. -/
theorem blockWhileBody_site_lowerP
    {fuel : Nat} {frames : List FrameKind} {pref body2 : List WasmInstr}
    (h_pref : StraightLineInstrs pref) (h_body2 : StraightLineInstrs body2)
    {s s1 : LowerState} {flag : Quanta.KOps.Reg} {bodyOps : List KernelOp}
    (h_lb : lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s, []⟩
              (pref ++ [.brIf 1] ++ body2 ++ [.br 0])
            = some (⟨s1, [{ levels := 1, cond := flag, flag := true, skip := 0 }]⟩, bodyOps)) :
    ∃ (s_site : LowerState) (opsSite opsBody2 : List KernelOp),
      lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s, []⟩ (pref ++ [.brIf 1])
        = some (⟨s_site, [{ levels := 1, cond := flag, flag := true, skip := 0 }]⟩, opsSite) ∧
      lowerInstrs fuel (.loopK :: .block :: frames) s_site body2 = some (s1, opsBody2) ∧
      s.nextReg ≤ flag := by
  obtain ⟨s_m, s0, s_c, s_b, svCond, cond, opsPref, opsCommit, opsBody2,
          hl_pref, h_pop, h_commit, hl_body2, h_sp, _⟩ :=
    blockWhileBody_lowerP h_pref h_body2 h_lb
  simp only [LowerStateP.mk.injEq, List.cons.injEq, PendingWrap.mk.injEq, and_true,
             true_and] at h_sp
  obtain ⟨h_s1, h_flag⟩ := h_sp
  subst h_s1; subst h_flag
  have h_nr_m : s.nextReg ≤ s_m.nextReg := by
    have := lowerInstrs_nextReg_mono _ _ _ _ hl_pref
    exact this
  have h_pop_nr := LowerState.popSym_nextReg h_pop
  have h_c_nr := LowerState.commit_nextReg_mono h_commit
  refine ⟨{ s_c with nextReg := s_c.nextReg + 2 },
          opsPref ++ (opsCommit
            ++ [KernelOp.cast s_c.nextReg cond .u32 .bool,
                KernelOp.branch s_c.nextReg
                  [.const (s_c.nextReg + 1) (.bool true), .breakOp] []]),
          opsBody2, ?_, hl_body2, by omega⟩
  refine lowerInstrsP_straightLine_compose h_pref hl_pref ?_
  rw [lowerInstrsP_brIf1_exit, h_pop]
  simp only [Option.bind_eq_bind, Option.some_bind, h_commit, LowerState.alloc, lowerInstrsP,
             pure, List.append_nil]

-- ════════════════════════════════════════════════════════════════════
-- The apex
-- ════════════════════════════════════════════════════════════════════

/-- The apex over the pending-wrap translator. -/
theorem framework_preservation_kernel_while2
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
    (h_wf : KernelInstrsW2 instrs)
    (h_fuel : fuel ≥ 2 + h_wf.depth)
    (h_st : h_wf.stable (fuel + 1) frames s)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs (fuel + 1) ws instrs = some ws')
    (hl : lowerInstrsP (fuel + 1) frames ⟨s, []⟩ instrs = some (⟨s', []⟩, ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  induction h_wf generalizing fuel ws s kst ws' s' ops with
  | empty =>
      simp only [evalInstrs, Option.some.injEq] at hw
      simp only [lowerInstrsP, Option.some.injEq, Prod.mk.injEq, LowerStateP.mk.injEq,
                 and_true] at hl
      obtain ⟨h_s, h_ops⟩ := hl
      subst hw; subst h_s; subst h_ops
      refine ⟨kst, 0, by simp [evalOps], R, ?_, ?_⟩
      · intro d hd; rw [h_no_branch] at hd; exact Option.noConfusion hd
      · intro _; exact h_kst_no_broke
  | @sl_cons i rest h_sl _h_rest_wf IH =>
      have h_sl1 : StraightLineInstrs [i] := ⟨h_sl, trivial⟩
      obtain ⟨ws_m, hw_i, h_mb, h_mh, hw_rest⟩ :=
        evalInstrs_straightLine_append (pref := [i]) (rest := rest) h_sl1
          h_no_branch h_no_halt hw
      obtain ⟨s_m, ops1, ops2, hl_i, hl_rest, h_ops⟩ :=
        lowerInstrsP_straightLine_append (pref := [i]) (rest := rest) h_sl1 hl
      subst h_ops
      obtain ⟨kst_m, F1, h_ev1, R_m, h_bridge_m⟩ :=
        framework_preservation_straightLine (fuel + 1) frames ws s kst layout R
          h_no_branch h_no_halt h_kst_no_broke h_buf_locals h_no_buf_stack
          h_load_bounds h_store_bounds h_store_layout [i] h_sl1 ws_m s_m ops1
          hw_i hl_i
      have h_kst_m_ok : kst_m.broke = false := h_bridge_m.right h_mb
      have h_st_rest : _h_rest_wf.stable (fuel + 1) frames s_m := by
        simp only [KernelInstrsW2.stable] at h_st
        rw [lowerInstrs_cons_default (fuel + 1) frames s i []
            (straightLine_not_structured_lower h_sl)] at hl_i
        cases hli : lowerInstr s i with
        | none => rw [hli] at hl_i; simp at hl_i
        | some p1 =>
            rw [hli] at hl_i
            obtain ⟨s1, ops_i⟩ := p1
            simp only [lowerInstrs, Option.bind_eq_bind, Option.some_bind, pure,
                       Option.some.injEq, Prod.mk.injEq] at hl_i
            obtain ⟨h_s, _⟩ := hl_i
            subst h_s
            exact h_st s1 ops_i hli
      obtain ⟨kst', F2, h_ev2, R', h_bridge'⟩ :=
        IH fuel ws_m s_m kst_m R_m h_mb h_mh h_kst_m_ok h_fuel h_st_rest ws' s' ops2
          hw_rest hl_rest
      refine ⟨kst', max F1 F2, ?_, R', h_bridge'⟩
      exact evalOps_append_fuel_mono_head (Nat.le_max_left F1 F2) h_ev1 h_kst_m_ok
        (evalOps_fuel_mono (Nat.le_max_right F1 F2) h_ev2)
  | @while_cons rest body post h_split h_body h_post_wf IH =>
      have h_depth : (KernelInstrsW2.while_cons h_split h_body h_post_wf).depth
                        = 1 + h_post_wf.depth := rfl
      rw [h_depth] at h_fuel
      obtain ⟨pref, h_sl, h_ht, h_body_eq⟩ := h_body
      subst h_body_eq
      -- The Stage-B lowering of the segment: the body is plain Stage A,
      -- nothing pends at the loop close, the post is Stage B.
      simp only [lowerInstrsP, h_split] at hl
      cases hlb : lowerInstrsP fuel (.loopK :: frames) ⟨{ s with currentReg := [] }, []⟩
          (pref ++ [.brIf 0]) with
      | none => rw [hlb] at hl; simp at hl
      | some q1 =>
      rw [hlb] at hl
      obtain ⟨⟨s1, p1⟩, bodyOps⟩ := q1
      obtain ⟨s_m, ops1, ops2, hA_pref, hB_br, h_bodyOps⟩ :=
        lowerInstrsP_straightLine_append h_sl hlb
      have h_p1 : p1 = [] := (lowerInstrsP_brIf0_loop_to_lowerInstrs rfl hB_br).1
      have hA_br : lowerInstrs fuel (.loopK :: frames) s_m [.brIf 0] = some (s1, ops2) :=
        (lowerInstrsP_brIf0_loop_to_lowerInstrs rfl hB_br).2
      subst h_p1
      subst h_bodyOps
      have h_lb : lowerInstrs fuel (.loopK :: frames) { s with currentReg := [] }
          (pref ++ [.brIf 0]) = some (s1, ops1 ++ ops2) :=
        lowerInstrs_straightLine_compose h_sl hA_pref hA_br
      simp only [Option.bind_eq_bind, Option.some_bind, hasPlainActive_nil,
                 Bool.false_eq_true, ↓reduceIte] at hl
      cases hlp : lowerInstrsP fuel frames ⟨{ s1 with currentReg := [] }, []⟩ post with
      | none => rw [hlp] at hl; simp at hl
      | some q2 =>
      rw [hlp] at hl
      obtain ⟨⟨s2, p2⟩, postOps⟩ := q2
      simp only [Option.some_bind, pure, stepPending_nil, closeDecls_nil, applyWraps_nil,
                 List.append_nil, List.nil_append, Option.some.injEq, Prod.mk.injEq,
                 LowerStateP.mk.injEq] at hl
      obtain ⟨⟨h_s, h_p2⟩, h_ops⟩ := hl
      subst h_p2
      -- The side condition at this loop.
      have h_st' := h_st
      simp only [KernelInstrsW2.stable] at h_st'
      obtain ⟨h_lt, h_st_post⟩ := h_st' s1 (ops1 ++ ops2) h_lb
      -- The WASM trace.
      have hw_iter : evalInstrs.iterLoop fuel (pref ++ [.brIf 0]) post fuel ws = some ws' := by
        have hw2 := hw
        simp only [evalInstrs] at hw2
        have h_cond : (ws.halted || ws.branchTarget.isSome) = false := by
          rw [h_no_halt, h_no_branch]; rfl
        rw [h_cond] at hw2
        simp only [Bool.false_eq_true, ↓reduceIte] at hw2
        rw [h_split] at hw2
        exact hw2
      obtain ⟨n, entries, bodyOuts, h_e0, h_step, h_cont, h_exit_bt, _, h_bound⟩ :=
        iterLoop_trace_of_eval
          (fun _ _ h_nb h_ev => whileBody_branches_at_most_zero h_sl h_nb h_ev)
          h_no_branch hw_iter
      -- Post IH in the core's `post_preserves` shape, at fuel `(fuel - 1) + 1`.
      have post_preserves :
          ∀ {ws_p : WasmState} {kst_p : Quanta.KOps.State}
            (_R_p : Refines ws_p { s1 with currentReg := [] } kst_p layout)
            (_h_nb_p : ws_p.branchTarget = none)
            (_h_nh_p : ws_p.halted = false)
            (_h_nbk_p : kst_p.broke = false)
            {ws'_p : WasmState}
            (_hw_p : evalInstrs fuel ws_p post = some ws'_p),
          ∃ (kst'_p : Quanta.KOps.State) (F : Nat),
            evalOps F kst_p postOps = some kst'_p ∧
            Refines ws'_p s2 kst'_p layout ∧
            BridgeClauses ws'_p kst'_p := by
        intro ws_p kst_p R_p h_nb_p h_nh_p h_nbk_p ws'_p hw_p
        have h_fuel_for_ih : fuel - 1 ≥ 2 + h_post_wf.depth := by omega
        have h_fuel_eq : fuel = (fuel - 1) + 1 := by omega
        rw [h_fuel_eq] at hw_p hlp h_st_post
        exact IH (fuel - 1) ws_p _ kst_p R_p h_nb_p h_nh_p h_nbk_p h_fuel_for_ih h_st_post
          ws'_p s2 postOps hw_p hlp
      -- The IR trace, from the loop-entry state.
      have R0 : Refines (entries 0) { s with currentReg := [] } kst layout := by
        rw [h_e0]; exact R.clear_current
      obtain ⟨kstStates, F_b, h_kst_start, h_ir_step, h_ir_cont, h_ir_exit, h_ref, h_nh⟩ :=
        whileBody_ir_trace fuel frames pref h_sl h_ht { s with currentReg := [] } s1
          (ops1 ++ ops2) h_lb h_lt rfl layout
          h_buf_locals h_no_buf_stack h_load_bounds h_store_bounds h_store_layout
          n entries bodyOuts h_step h_cont h_exit_bt kst R0
          (by rw [h_e0]; exact h_no_branch) (by rw [h_e0]; exact h_no_halt) h_kst_no_broke
      exact preservation_evalInstrs_cons_wloop_nIterExit_core
        frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
        fuel rest (pref ++ [.brIf 0]) post h_split n entries bodyOuts h_e0 h_step h_cont
        ⟨h_exit_bt, h_nh (Fin.last n)⟩ s1 (ops1 ++ ops2) kstStates h_kst_start F_b
        h_ir_step h_ir_cont h_ir_exit h_ref s2 postOps post_preserves
        h_bound ws' s' ops hw h_s h_ops
  | @block_while_cons pref body2 post h_pref h_body2 h_ht_pref h_ht_body2 h_post_wf IH =>
      have h_depth : (KernelInstrsW2.block_while_cons h_pref h_body2 h_ht_pref h_ht_body2
                        h_post_wf).depth = 1 + h_post_wf.depth := rfl
      rw [h_depth] at h_fuel
      -- Fuel: the block spends one unit, the loop the other.
      obtain ⟨f, h_f⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by omega⟩
      subst h_f
      -- The WASM trace out of the skeleton.
      obtain ⟨n, entries, bodyOuts, h_e0, h_step, h_cont, h_nh_all, h_exit, h_post_eval,
              h_bound⟩ :=
        evalInstrs_block_wloop_trace_of_eval h_no_branch h_no_halt
          (blockWhileBody_noStructured h_pref h_body2) (by trivial)
          (blockWhileBody_continuesOrExits1 h_pref h_body2) hw
      -- The lowering, taken apart.
      obtain ⟨s1, flag, bodyOps, postOps, h_lb, hlp, h_ops⟩ :=
        blockWhile_lowerP h_pref h_body2 hl
      obtain ⟨s_site, opsSite, opsBody2, hl_site, hl_body2, h_flag_fresh⟩ :=
        blockWhileBody_site_lowerP h_pref h_body2 h_lb
      -- The side condition at this loop.
      have h_st' := h_st
      simp only [KernelInstrsW2.stable] at h_st'
      obtain ⟨h_lt, h_lr_site, h_lt_site, h_st_post⟩ :=
        h_st' s_site _ opsSite hl_site s1 _ bodyOps h_lb
      -- The IR trace, from the state after the flag's declaration.
      have R0 : Refines (entries 0) { s with currentReg := [] }
          { kst with rf := regWrite kst.rf flag (vBool false) } layout := by
        rw [h_e0]
        exact R.clear_current.regWrite_fresh h_flag_fresh _
      obtain ⟨kstStates, F_b, h_kst_start, h_ir_step, h_ir_cont, h_ir_exit, h_exit_ref,
              h_flag_exit, _⟩ :=
        blockWhileBody_ir_trace f frames pref body2 h_pref h_body2 h_ht_pref h_ht_body2
          { s with currentReg := [] } s1 s_site flag bodyOps opsSite h_lb hl_site rfl h_lt
          h_lr_site h_lt_site layout
          h_buf_locals h_no_buf_stack h_load_bounds h_store_bounds h_store_layout
          n entries bodyOuts h_step h_cont h_exit _ R0
          (by rw [h_e0]; exact h_no_branch) (by rw [h_e0]; exact h_no_halt) h_kst_no_broke
      -- Post IH in the `post_preserves` shape, at fuel `f + 1`.
      have post_preserves :
          ∀ {ws_p : WasmState} {kst_p : Quanta.KOps.State}
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
            BridgeClauses ws'_p kst'_p := by
        intro ws_p kst_p R_p h_nb_p h_nh_p h_nbk_p ws'_p s'_p postOps' hw_p hl_p
        have h_fuel_for_ih : f ≥ 2 + h_post_wf.depth := by omega
        exact IH f ws_p _ kst_p R_p h_nb_p h_nh_p h_nbk_p h_fuel_for_ih h_st_post
          ws'_p s'_p postOps' hw_p hl_p
      exact preservation_blockWhile_nIterExit f frames s kst layout h_kst_no_broke
        pref body2 post h_pref h_body2 n bodyOuts (h_nh_all (Fin.last n)) ws' h_post_eval
        s1 flag bodyOps h_lb kstStates h_kst_start F_b h_ir_step h_ir_cont h_ir_exit
        h_exit_ref h_flag_exit post_preserves s' ops hl

-- ════════════════════════════════════════════════════════════════════
-- Subsumption and witnesses
-- ════════════════════════════════════════════════════════════════════

/-- Every `KernelInstrsW` kernel is a `KernelInstrsW2` kernel, at the
    same depth. -/
def KernelInstrsW.toW2 : ∀ {instrs : List WasmInstr}, KernelInstrsW instrs → KernelInstrsW2 instrs
  | _, .empty => .empty
  | _, .sl_cons h rest => .sl_cons h rest.toW2
  | _, .while_cons h_split h_body post => .while_cons h_split h_body post.toW2

theorem KernelInstrsW.toW2_depth : ∀ {instrs : List WasmInstr} (h : KernelInstrsW instrs),
    h.toW2.depth = h.depth
  | _, .empty => rfl
  | _, .sl_cons _ rest => by
      simp [KernelInstrsW.toW2, KernelInstrsW2.depth, KernelInstrsW.depth, rest.toW2_depth]
  | _, .while_cons _ _ post => by
      simp [KernelInstrsW.toW2, KernelInstrsW2.depth, KernelInstrsW.depth, post.toW2_depth]

/-- The earlier apex's side condition is this one on the embedded kernel. -/
theorem KernelInstrsW.stable_toW2 : ∀ {instrs : List WasmInstr} (h : KernelInstrsW instrs)
    (fuel : Nat) (frames : List FrameKind) (s : LowerState),
    LoopsTypeStable fuel frames s instrs → h.toW2.stable fuel frames s
  | _, .empty, _, _, _, _ => trivial
  | _, .sl_cons h_sl rest, fuel, frames, s, h_ts => by
      rw [LoopsTypeStable_cons_straightLine h_sl] at h_ts
      intro s1 ops hli
      exact rest.stable_toW2 fuel frames s1 (h_ts s1 ops hli)
  | _, .while_cons h_split h_body post, fuel, frames, s, h_ts => by
      cases fuel with
      | zero => trivial
      | succ f =>
          simp only [LoopsTypeStable, h_split] at h_ts
          intro s1 bodyOps h_lb
          obtain ⟨h_lt, h_post⟩ := h_ts s1 bodyOps h_lb
          exact ⟨h_lt, post.stable_toW2 f frames _ h_post⟩

/-- The earlier apex, as a corollary: embed the kernel, lift its
    lowering to Stage B (forward agreement), and carry its side
    condition over. -/
theorem framework_preservation_kernel_while_of_W2
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
    (h_wf : KernelInstrsW instrs)
    (h_fuel : fuel ≥ 2 + h_wf.depth)
    (h_ts : LoopsTypeStable (fuel + 1) frames s instrs)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs (fuel + 1) ws instrs = some ws')
    (hl : lowerInstrs (fuel + 1) frames s instrs = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  framework_preservation_kernel_while2 fuel frames ws s kst layout R h_no_branch h_no_halt
    h_kst_no_broke h_buf_locals h_no_buf_stack h_load_bounds h_store_bounds h_store_layout
    instrs h_wf.toW2 (by rw [KernelInstrsW.toW2_depth]; exact h_fuel)
    (h_wf.stable_toW2 (fuel + 1) frames s h_ts) ws' s' ops hw
    (lowerInstrsP_agrees_with_lowerInstrs (fuel + 1) frames s instrs hl)

/-- rustc's `i = 0; while i < n { i += 1 }` — the kernel of
    `crates/gpu/quanta-wasm-lowering/tests/lower_while_exit_flag.rs` and
    of the `while_exit_flag` pins — typechecks as a `KernelInstrsW2`. -/
example : KernelInstrsW2
    [.i32Const 0, .localSet 2,
     .block 0, .wloop 0,
       .localGet 2, .localGet 1, .i32GeU, .brIf 1,
       .localGet 2, .i32Const 1, .i32Add, .localSet 2,
       .br 0,
     .wend, .wend] :=
  .sl_cons trivial (.sl_cons trivial
    (.block_while_cons (pref := [.localGet 2, .localGet 1, .i32GeU])
      (body2 := [.localGet 2, .i32Const 1, .i32Add, .localSet 2]) (post := [])
      (by simp [StraightLineInstrs, StraightLineInstr])
      (by simp [StraightLineInstrs, StraightLineInstr])
      rfl rfl .empty))

/-- Its depth is one. -/
example : (KernelInstrsW2.block_while_cons (pref := [.localGet 1]) (body2 := []) (post := [])
    ⟨trivial, trivial⟩ trivial rfl rfl .empty).depth = 1 := rfl

end Quanta.Wasm
