/-
# L16 — the if-close merge (`wif` with local sets and stores)

The general `wif` cons-composer: the arms may set locals and store to
buffers — the guard patterns real kernel bodies are made of
(`if c { x = …; out[i] = … }`). The `…noLocalSet` composers in
`PreservationBridge` pinned the arms with state equalities; this file
keeps only what the merge itself needs, so the hypotheses are
dischargeable for seeded straight-line arms (L14's
`lowerInstrs_localReg_seeded` gives the stable-layer identity WITH
localSets present).

The merge (`currentReg := []` at the frame close, production's
`merge_locals_post_frame`) is refinement-preserving because `Refines`
carries the stable layer separately (`locs : LocalsRefines`) and the
dual-Copy discipline keeps it current in either branch; the untaken
branch's lowering contributes only a `nextReg` advance, and its
per-set bindings — registers the IR never wrote — are exactly what
the close erases.
-/

import Quanta.Wasm.PreservationBridge

namespace Quanta.Wasm

open Quanta.KOps (KernelOp evalOps regLookup)
open Quanta.Semantics.Cpu

/-- `popSym` only pops the symbolic stack; both local maps ride. -/
theorem popSym_locals {s s' : LowerState} {sv : SymVal}
    (h : s.popSym = some (sv, s')) :
    s'.localReg = s.localReg ∧ s'.localTy = s.localTy := by
  unfold LowerState.popSym at h
  rcases hs : s.stack with _ | ⟨sv', rs⟩
  · rw [hs] at h; simp at h
  · rw [hs] at h; simp at h
    obtain ⟨_, hs_eq⟩ := h
    rw [← hs_eq]
    exact ⟨rfl, rfl⟩

/-- `wif _ :: rest` preservation with the if-close merge — L16.
    The general composer: the arms may SET LOCALS and STORE TO
    BUFFERS. Where `…_fallthrough_noLocalSet` pinned the arms with
    state equalities (`ws'.locals = ws.locals`, `mem`, `currentReg`),
    this theorem keeps only what the merge needs:

    - `then/else_preserves`: the recursive bridge IHs (unchanged);
    - `then/else_exits_clean`: the arm's evaluation falls through
      (`branchTarget = none`, `halted = false`) — provable for any
      straight-line arm;
    - `then/else_lowering_frame`: the arm's lowering keeps
      `localReg` / `localTy` / `stack` / `bufferSlots` and advances
      `nextReg` — under L14 seeding these hold WITH localSets
      (`lowerInstrs_localReg_seeded`); `currentReg` is deliberately
      unconstrained (per-set bindings appear and die at the close).

    Soundness of the close (`currentReg := []`) is the dual-Copy
    discipline: `locs : LocalsRefines` rides the stable layer, which
    either branch keeps current, so dropping the per-frame layer
    preserves `Refines` (`clear_current`) and the untaken branch
    contributes only a `nextReg` advance. -/
theorem preservation_evalInstrs_cons_wif_merge
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
    (then_exits_clean : ∀ {ws_b : WasmState} {s_b : LowerState}
        {kst_b : Quanta.KOps.State} {ws'_b : WasmState} {s'_b : LowerState}
        {bodyOps : List KernelOp}
        (_R_b : Refines ws_b s_b kst_b layout)
        (_h_nb_b : ws_b.branchTarget = none)
        (_h_nh_b : ws_b.halted = false)
        (_h_nbk_b : kst_b.broke = false)
        (_hw_b : evalInstrs bt ws_b thenBody = some ws'_b)
        (_hl_b : lowerInstrs bt (.wif :: frames) s_b thenBody = some (s'_b, bodyOps)),
      ws'_b.branchTarget = none ∧ ws'_b.halted = false)
    (then_lowering_frame : ∀ {s_b s'_b : LowerState} {bodyOps : List KernelOp},
        s_b.localReg = s.localReg → s_b.localTy = s.localTy →
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
    (else_exits_clean : ∀ {ws_b : WasmState} {s_b : LowerState}
        {kst_b : Quanta.KOps.State} {ws'_b : WasmState} {s'_b : LowerState}
        {bodyOps : List KernelOp}
        (_R_b : Refines ws_b s_b kst_b layout)
        (_h_nb_b : ws_b.branchTarget = none)
        (_h_nh_b : ws_b.halted = false)
        (_h_nbk_b : kst_b.broke = false)
        (_hw_b : evalInstrs bt ws_b elseBody = some ws'_b)
        (_hl_b : lowerInstrs bt (.wif :: frames) s_b elseBody = some (s'_b, bodyOps)),
      ws'_b.branchTarget = none ∧ ws'_b.halted = false)
    (else_lowering_frame : ∀ {s_b s'_b : LowerState} {bodyOps : List KernelOp},
        s_b.localReg = s.localReg → s_b.localTy = s.localTy →
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
        -- After thenBody, restore localReg/localTy. By then_lowering_frame
        -- on s2, the restore is idempotent on those fields (s2.localReg already
        -- equals s_cast.localReg). The restored state thus equals s2.
        -- Case on elseBody's lowering.
        have h_s0_locals : s0.localReg = s.localReg ∧ s0.localTy = s.localTy :=
          popSym_locals hpop
        have h_s1_locals := commit_preserves_locals hcommit
        have h_cast_lr :
            ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg = s.localReg := by
          show s1.localReg = s.localReg
          rw [h_s1_locals.1, h_s0_locals.1]
        have h_cast_lt :
            ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy = s.localTy := by
          show s1.localTy = s.localTy
          rw [h_s1_locals.2, h_s0_locals.2]
        obtain ⟨h_s2_lr, h_s2_lt, h_s2_stack, h_s2_bs, h_s2_nr⟩ :=
          then_lowering_frame h_cast_lr h_cast_lt hlt
        -- The restored state is NOT s2 in general (thenBody's per-set
        -- currentReg bindings die here; localReg/localTy reset to the
        -- entry snapshot). Keep the record and thread it.
        cases hle : lowerInstrs bt (.wif :: frames)
            ({ s2 with
                localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                localTy := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy,
                currentReg :=
                  ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).currentReg } : LowerState)
            elseBody with
        | none => simp [hle] at hl
        | some else_pair =>
          rcases else_pair with ⟨s3, elseOps⟩
          simp [hle] at hl
          -- After elseBody, restore again to s_cast snapshot.
          have h_s2R_lr :
              ({ s2 with
                  localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                  localTy := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy,
                  currentReg :=
                    ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).currentReg }
                : LowerState).localReg = s.localReg := h_cast_lr
          have h_s2R_lt :
              ({ s2 with
                  localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                  localTy := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy,
                  currentReg :=
                    ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).currentReg }
                : LowerState).localTy = s.localTy := h_cast_lt
          obtain ⟨h_s3_lr, h_s3_lt, h_s3_stack, h_s3_bs, h_s3_nr⟩ :=
            else_lowering_frame h_s2R_lr h_s2R_lt hle
          cases hlp : lowerInstrs bt frames { s3 with currentReg := [] } post with
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
                  refine ⟨?_, ?_, ?_, ?_, R_at_s1.injLocals, R_at_s1.heapRefines, ?_, ?_, ?_⟩
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
                    show CurrentRegRefines layout _ s1.currentReg _ _
                    exact CurrentRegRefines_preserved_fresh R_at_s1.currentReg R_at_s1.freshCurrent _
                  · -- FreshCurrent: s_cast.nextReg = s1.nextReg + 1; currentReg unchanged.
                    intro ir hir
                    exact Nat.lt_succ_of_lt (R_at_s1.freshCurrent ir hir)
                  · -- CurrentLocalDisjoint: currentReg/localReg unchanged from s1.
                    exact R_at_s1.curLocDisj
                -- Lifting helper: given Refines ws_target s_x kst_cast,
                -- with s_x's stack = s_cast.stack, localReg = s_cast.localReg,
                -- localTy = s_cast.localTy, bufferSlots = s_cast.bufferSlots,
                -- and s_cast.nextReg ≤ s_x.nextReg, lift R_at_cast to s_x.
                -- Used for both c=0 (target s3) and c≠0 (target s3 from s2).
                have R_lift : ∀ (s_x : LowerState),
                    s_x.stack = ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).stack →
                    s_x.localReg = ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg →
                    s_x.currentReg = ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).currentReg →
                    s_x.localTy = ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy →
                    ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).nextReg ≤ s_x.nextReg →
                    Refines ws0 s_x kst_cast layout := by
                  intro s_x h_stk h_lr h_cr h_lt h_nr
                  refine ⟨?_, ?_, ?_, ?_, ?_, R_at_cast.heapRefines, ?_, ?_, ?_⟩
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
                    rw [show (localTyOf s_x.localTy i)
                          = localTyOf ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy i
                        from by rw [h_lt]]
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
                  · -- currentReg: s_x.currentReg = s_cast.currentReg (h_cr).
                    show CurrentRegRefines layout ws0.locals s_x.currentReg s_x.localTy kst_cast.rf
                    rw [h_cr, h_lt]
                    exact R_at_cast.currentReg
                  · -- freshCurrent: ir.snd < s_cast.nextReg ≤ s_x.nextReg.
                    intro ir hir
                    have hir_cast : ir ∈
                        ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).currentReg := by
                      rw [← h_cr]; exact hir
                    exact Nat.lt_of_lt_of_le (R_at_cast.freshCurrent ir hir_cast) h_nr
                  · -- curLocDisj: currentReg/localReg same as s_cast via h_cr / h_lr.
                    intro p q hp hq hpq
                    rw [h_cr] at hp
                    rw [h_lr] at hq
                    exact R_at_cast.curLocDisj p q hp hq hpq
                -- Combined facts: the restore resets localReg/localTy to the
                -- entry snapshot (projection-rfl on the record), so the else
                -- lowering's frame equalities compose through it — and back
                -- to s2 via then_lowering_frame.
                have h_s3_lr2 : s3.localReg = s2.localReg := by
                  rw [h_s3_lr]; exact h_s2_lr.symm
                have h_s3_lt2 : s3.localTy = s2.localTy := by
                  rw [h_s3_lt]; exact h_s2_lt.symm
                have h_s3_stack2 : s3.stack = s2.stack := h_s3_stack
                have h_s3_nr2 : s2.nextReg ≤ s3.nextReg := h_s3_nr
                by_cases hc : c = 0
                · -- c = 0: WASM picks elseBody, eval runs it.
                  simp only [hc, ↓reduceIte] at hw
                  cases h_eb : evalInstrs bt ws0 elseBody with
                  | none => simp [h_eb] at hw
                  | some ws_ab =>
                    simp [h_eb] at hw
                    -- Refines at the RESTORED record: its localReg/localTy/
                    -- currentReg are the entry snapshot's by construction
                    -- (rfl), its stack is s2's (untouched by the restore,
                    -- = s_cast's by then_lowering_frame), and its nextReg
                    -- is s2's. Lift R_at_cast.
                    have R_at_s2R : Refines ws0
                        ({ s2 with
                            localReg :=
                              ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                            localTy :=
                              ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy,
                            currentReg :=
                              ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).currentReg }
                          : LowerState)
                        kst_cast layout := by
                      refine R_lift _ ?_ ?_ ?_ ?_ ?_
                      · exact h_s2_stack
                      · rfl
                      · rfl
                      · rfl
                      · exact h_s2_nr
                    obtain ⟨kst_ab, F_b, h_ev_b, R_b, h_bridge_b⟩ :=
                      else_preserves R_at_s2R h_ws0_nb h_ws0_nh h_kst_cast_broke
                        h_eb hle
                    obtain ⟨h_ab_nb, h_ab_nh⟩ :=
                      else_exits_clean R_at_s2R h_ws0_nb h_ws0_nh h_kst_cast_broke
                        h_eb hle
                    rw [h_ab_nb] at hw
                    simp only at hw
                    -- hw: evalInstrs bt ws_ab post = some ws'.
                    have h_ab_broke : kst_ab.broke = false := h_bridge_b.right h_ab_nb
                    obtain ⟨kst', F_p, h_ev_p, R_p, h_bridge_p⟩ :=
                      post_preserves R_b.clear_current h_ab_nb h_ab_nh h_ab_broke hw hlp
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
                        have h_c_toNat : decide (c.toNat = 0) = decide (c = 0) := by
                          by_cases hc : c = 0
                          · subst hc; rfl
                          · have hh : c.toNat ≠ 0 := by
                              intro h; exact hc (by simpa using congrArg UInt32.ofNat h)
                            simp [hc, hh]
                        rcases h_lookup with h_u32 | h_i32
                        · simp [Quanta.KOps.evalOp, h_u32, Quanta.KOps.evalCast, kst_cast]
                        · simp [Quanta.KOps.evalOp, h_i32, Quanta.KOps.evalCast, kst_cast, h_c_toNat]
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
                    obtain ⟨h_ab_nb, h_ab_nh⟩ :=
                      then_exits_clean R_at_cast h_ws0_nb h_ws0_nh h_kst_cast_broke
                        h_eb hlt
                    rw [h_ab_nb] at hw
                    simp only at hw
                    have h_ab_broke : kst_ab.broke = false := h_bridge_b.right h_ab_nb
                    -- R_b is at s2 (thenBody's output). Lift DIRECTLY to the
                    -- closed state { s3 with currentReg := [] }: its stable
                    -- layer is s2's (restore + else frame equalities), its
                    -- per-frame layer is empty (the currentReg-family clauses
                    -- are vacuous — this is where the else arm's per-set
                    -- bindings, which kst never wrote, are erased), and its
                    -- nextReg only advanced (Fresh is monotone).
                    have R_b_at_s3c : Refines ws_ab
                        ({ s3 with currentReg := [] } : LowerState) kst_ab layout := by
                      refine ⟨?_, ?_, ?_, ?_, ?_, R_b.heapRefines, ?_, ?_, ?_⟩
                      · refine ⟨?_, ?_⟩
                        · show ws_ab.stack.length = s3.stack.length
                          rw [h_s3_stack2]
                          exact R_b.stk.left
                        · intro i v hv
                          obtain ⟨svi, hsv_get, henc⟩ := R_b.stk.right i v hv
                          have hsv_get_s3 : s3.stack.get? i = some svi := by
                            rw [h_s3_stack2]; exact hsv_get
                          exact ⟨svi, hsv_get_s3, henc⟩
                      · intro i r hfind v hv
                        have hfind_s2 :
                            s2.localReg.find? (fun p => p.fst = i) = some (i, r) := by
                          rw [← h_s3_lr2]; exact hfind
                        show v.encodes layout kst_ab.rf (SymVal.reg r (localTyOf s3.localTy i))
                        rw [show (localTyOf s3.localTy i) = localTyOf s2.localTy i from by rw [h_s3_lt2]]
                        exact R_b.locs i r hfind_s2 v hv
                      · refine ⟨?_, ?_⟩
                        · intro sv hsv r hr
                          have hsv_s2 : sv ∈ s2.stack := by rw [← h_s3_stack2]; exact hsv
                          exact Nat.lt_of_lt_of_le (R_b.fresh.left sv hsv_s2 r hr) h_s3_nr2
                        · intro ir hir
                          have hir_s2 : ir ∈ s2.localReg := by rw [← h_s3_lr2]; exact hir
                          exact Nat.lt_of_lt_of_le (R_b.fresh.right ir hir_s2) h_s3_nr2
                      · intro ir hir sv hsv
                        have hir_s2 : ir ∈ s2.localReg := by rw [← h_s3_lr2]; exact hir
                        have hsv_s2 : sv ∈ s2.stack := by rw [← h_s3_stack2]; exact hsv
                        exact R_b.aliasFree ir hir_s2 sv hsv_s2
                      · intro p q hp hq
                        have hp_s2 : p ∈ s2.localReg := by rw [← h_s3_lr2]; exact hp
                        have hq_s2 : q ∈ s2.localReg := by rw [← h_s3_lr2]; exact hq
                        exact R_b.injLocals p q hp_s2 hq_s2
                      · -- currentReg: the close cleared the per-frame layer.
                        intro i r hfind
                        simp at hfind
                      · -- freshCurrent: vacuous over [].
                        intro ir hir
                        simp at hir
                      · -- curLocDisj: vacuous over [].
                        intro p q hp
                        simp at hp
                    obtain ⟨kst', F_p, h_ev_p, R_p, h_bridge_p⟩ :=
                      post_preserves R_b_at_s3c h_ab_nb h_ab_nh h_ab_broke hw hlp
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
                        have h_c_toNat : decide (c.toNat = 0) = decide (c = 0) := by
                          by_cases hc : c = 0
                          · subst hc; rfl
                          · have hh : c.toNat ≠ 0 := by
                              intro h; exact hc (by simpa using congrArg UInt32.ofNat h)
                            simp [hc, hh]
                        rcases h_lookup with h_u32 | h_i32
                        · simp [Quanta.KOps.evalOp, h_u32, Quanta.KOps.evalCast, kst_cast]
                        · simp [Quanta.KOps.evalOp, h_i32, Quanta.KOps.evalCast, kst_cast, h_c_toNat]
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

end Quanta.Wasm
