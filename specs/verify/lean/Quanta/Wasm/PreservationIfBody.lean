/-
# L16 (3a) — loop bodies with balanced `wif` frames: the split lemmas

`preservation_blockLoop_nIterExit` demands `NoStructured body` only so
the `splitAtEnd` scanner finds the loop's own `wend`. A *balanced*
`wif … (welse …)? wend` keeps the scanner correct — it enters the
frame at depth `n+1` and leaves it back at `n`, never tripping the
depth-0 stop arms. `IfBalanced` names that body class (marker-free
arms; nested frames are the depth-2 rung's business), and the two
split lemmas below are the `_noStructured` pair generalized to it.
-/

import Quanta.Wasm.PreservationIfMerge
import Quanta.Wasm.PreservationWhileExit
import Quanta.Wasm.PreservationBlockWhile

namespace Quanta.Wasm

open Quanta.KOps (KernelOp evalOps regLookup regWrite vBool)

/-- A stream of marker-free instructions and balanced single-level
    `wif` frames (marker-free arms, with or without `welse`). The
    body class the loop machinery can scan across: every opener's
    closer is inside the stream. -/
inductive IfBalanced : List WasmInstr → Prop
  | empty : IfBalanced []
  | sl_cons {i : WasmInstr} {rest : List WasmInstr} :
      NoStructuredInstr i → IfBalanced rest → IfBalanced (i :: rest)
  | wif_else_cons {bt : Nat} {thenB elseB post : List WasmInstr} :
      NoStructured thenB → NoStructured elseB → IfBalanced post →
      IfBalanced (.wif bt :: (thenB ++ .welse :: elseB ++ .wend :: post))
  | wif_noelse_cons {bt : Nat} {thenB post : List WasmInstr} :
      NoStructured thenB → IfBalanced post →
      IfBalanced (.wif bt :: (thenB ++ .wend :: post))

/-- Marker-free streams are trivially balanced. -/
theorem IfBalanced.of_noStructured : ∀ {l : List WasmInstr},
    NoStructured l → IfBalanced l
  | [], _ => .empty
  | _ :: _, h => .sl_cons h.1 (IfBalanced.of_noStructured h.2)

/-- One walker step over an `wif`: the depth bumps. -/
theorem walkUntilCloser_wif (bt : Nat) (rest : List WasmInstr)
    (n : Nat) (acc : List WasmInstr) :
    walkUntilCloser (.wif bt :: rest) n acc
      = walkUntilCloser rest (n + 1) (.wif bt :: acc) := rfl

/-- One walker step over a `welse` strictly above the stop level. -/
theorem walkUntilCloser_welse_succ (rest : List WasmInstr)
    (n : Nat) (acc : List WasmInstr) :
    walkUntilCloser (.welse :: rest) (n + 1) acc
      = walkUntilCloser rest (n + 1) (.welse :: acc) := rfl

/-- One walker step over a `wend` strictly above the stop level: the
    depth restores. -/
theorem walkUntilCloser_wend_succ (rest : List WasmInstr)
    (n : Nat) (acc : List WasmInstr) :
    walkUntilCloser (.wend :: rest) (n + 1) acc
      = walkUntilCloser rest n (.wend :: acc) := rfl

/-- The walker crosses a balanced stream at any depth, accumulating it
    in reverse — the `_noStructured` walker lemma generalized: a `wif`
    bumps the depth, its markers live strictly above the stop level,
    and its `wend` restores the entry depth. -/
theorem walkUntilCloser_append_ifBalanced
    {l : List WasmInstr} (h : IfBalanced l)
    (rest : List WasmInstr) (n : Nat) (acc : List WasmInstr) :
    walkUntilCloser (l ++ rest) n acc = walkUntilCloser rest n (l.reverse ++ acc) := by
  induction h generalizing n acc with
  | empty => rfl
  | sl_cons h_i _ IH =>
      rw [List.cons_append, walkUntilCloser_cons_noStructured h_i, IH]
      simp
  | @wif_else_cons bt thenB elseB post h_then h_else _ IH =>
      simp only [List.cons_append, List.append_assoc]
      rw [walkUntilCloser_wif, walkUntilCloser_append_noStructured h_then,
          walkUntilCloser_welse_succ, walkUntilCloser_append_noStructured h_else,
          walkUntilCloser_wend_succ, IH]
      simp [List.reverse_append]
  | @wif_noelse_cons bt thenB post h_then _ IH =>
      simp only [List.cons_append, List.append_assoc]
      rw [walkUntilCloser_wif, walkUntilCloser_append_noStructured h_then,
          walkUntilCloser_wend_succ, IH]
      simp [List.reverse_append]

/-- A balanced body closed by `wend` splits exactly there
    (`splitAtEnd_noStructured` generalized). -/
theorem splitAtEnd_ifBalanced
    {l : List WasmInstr} (h : IfBalanced l) (t : List WasmInstr) :
    splitAtEnd (l ++ [.wend] ++ t) = some (l, t) := by
  rw [List.append_assoc, List.singleton_append]
  simp only [splitAtEnd, walkUntilCloser_append_ifBalanced h, walkUntilCloser,
             List.append_nil, List.reverse_reverse, Option.bind_eq_bind, Option.some_bind]

/-- The block body `wloop 0 :: loopBody ++ [wend] ++ tail` splits at
    the block's own `wend`, with balanced loop body and tail
    (`splitAtEnd_wloop_noStructured` generalized). -/
theorem splitAtEnd_wloop_ifBalanced
    {loopBody tail : List WasmInstr}
    (h_body : IfBalanced loopBody) (h_tail : IfBalanced tail)
    (post : List WasmInstr) :
    splitAtEnd (.wloop 0 :: loopBody ++ [.wend] ++ tail ++ [.wend] ++ post)
      = some (.wloop 0 :: loopBody ++ [.wend] ++ tail, post) := by
  simp only [List.cons_append, List.append_assoc, List.singleton_append]
  simp only [splitAtEnd, walkUntilCloser, walkUntilCloser_append_ifBalanced h_body,
             walkUntilCloser_append_ifBalanced h_tail, Option.bind_eq_bind,
             Option.some_bind, List.reverse_append, List.reverse_cons, List.reverse_reverse,
             List.reverse_nil, List.nil_append, List.append_nil, List.cons_append,
             List.append_assoc]

/-- Balanced streams compose. -/
theorem IfBalanced.append : ∀ {a b : List WasmInstr},
    IfBalanced a → IfBalanced b → IfBalanced (a ++ b) := by
  intro a b ha hb
  induction ha with
  | empty => exact hb
  | sl_cons h_i _ IH => exact .sl_cons h_i IH
  | @wif_else_cons bt thenB elseB post h_then h_else _ IH =>
      have h := IfBalanced.wif_else_cons (bt := bt) h_then h_else IH
      simpa [List.cons_append, List.append_assoc] using h
  | @wif_noelse_cons bt thenB post h_then _ IH =>
      have h := IfBalanced.wif_noelse_cons (bt := bt) h_then IH
      simpa [List.cons_append, List.append_assoc] using h

/-- `blockLoop_lowerP` with the body class relaxed to `IfBalanced` —
    the `NoStructured` hypothesis fed only the two split lemmas, and
    those now hold for balanced bodies. Everything else is the same
    walk down the Stage-B lowering: the loop close consumes the single
    exit-flag record, the block close wraps the empty tail, the post
    follows. -/
theorem blockLoop_lowerP_ifBalanced
    {f : Nat} {frames : List FrameKind} {body post : List WasmInstr}
    (h_bal : IfBalanced body)
    {s s1 : LowerState} {flag : Quanta.KOps.Reg} {bodyOps : List KernelOp}
    (h_lb : lowerInstrsP f (.loopK :: .block :: frames) ⟨{ s with currentReg := [] }, []⟩ body
        = some (⟨s1, [{ levels := 1, cond := flag, flag := true, skip := 0 }]⟩, bodyOps))
    {s' : LowerState} {ops : List KernelOp}
    (hl : lowerInstrsP (f + 2) frames ⟨s, []⟩
            (.block 0 :: .wloop 0 :: body ++ [.wend] ++ [] ++ [.wend] ++ post)
          = some (⟨s', []⟩, ops)) :
    ∃ (postOps : List KernelOp),
      lowerInstrsP (f + 1) frames ⟨{ s1 with currentReg := [] }, []⟩ post
        = some (⟨s', []⟩, postOps) ∧
      ops = [.const flag (.bool false), .loopOp bodyOps, .branch flag [] []] ++ postOps := by
  have h_split_b := splitAtEnd_wloop_ifBalanced h_bal (tail := []) .empty post
  have h_split_l := splitAtEnd_ifBalanced h_bal []
  simp only [lowerInstrsP, List.cons_append, List.append_assoc, List.nil_append,
             List.append_nil, List.singleton_append] at hl h_split_b h_split_l
  rw [h_split_b] at hl
  simp only [Option.bind_eq_bind] at hl
  -- The block body: the wloop arm.
  rcases hb : lowerInstrsP (f + 1) (.block :: frames) ⟨s, []⟩
      (.wloop 0 :: (body ++ [.wend])) with _ | ⟨spb, innerOps⟩
  · rw [hb] at hl; simp at hl
  rw [hb] at hl
  simp only [Option.some_bind] at hl
  simp only [lowerInstrsP] at hb
  rw [h_split_l] at hb
  simp only [Option.bind_eq_bind] at hb
  rw [h_lb] at hb
  simp only [Option.some_bind] at hb
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
  rcases hlp : lowerInstrsP (f + 1) frames ⟨{ s1 with currentReg := [] }, []⟩ post
      with _ | ⟨spp, postOps⟩
  · rw [hlp] at hl; simp at hl
  rw [hlp] at hl
  simp only [Option.some_bind, pure, stepPending_nil, closeDecls_nil, applyWraps_nil,
             List.append_nil, List.nil_append, Option.some.injEq, Prod.mk.injEq] at hl
  obtain ⟨h_spp, h_ops⟩ := hl
  subst h_inner
  refine ⟨postOps, ?_, ?_⟩
  · cases spp
    simp only [LowerStateP.mk.injEq] at h_spp
    obtain ⟨h1, h2⟩ := h_spp
    subst h1
    subst h2
    rfl
  · rw [← h_ops]

/-- `splitAtElseOrEnd` on a marker-free then-arm with an else clause:
    the walker stops at the frame's own `welse`, then at its `wend`. -/
theorem splitAtElseOrEnd_noStructured_else
    {thenB elseB : List WasmInstr}
    (h_then : NoStructured thenB) (h_else : NoStructured elseB)
    (t : List WasmInstr) :
    splitAtElseOrEnd (thenB ++ .welse :: elseB ++ .wend :: t)
      = some (thenB, elseB, t) := by
  unfold splitAtElseOrEnd
  simp only [List.cons_append, List.append_assoc]
  rw [walkUntilCloser_append_noStructured h_then]
  simp only [walkUntilCloser, List.append_nil, List.reverse_reverse,
             Option.bind_eq_bind, Option.some_bind]
  rw [walkUntilCloser_append_noStructured h_else]
  simp only [walkUntilCloser, List.append_nil, List.reverse_reverse,
             Option.bind_eq_bind, Option.some_bind]

/-- `splitAtElseOrEnd` on a marker-free arm with no else clause. -/
theorem splitAtElseOrEnd_noStructured_noelse
    {thenB : List WasmInstr} (h_then : NoStructured thenB)
    (t : List WasmInstr) :
    splitAtElseOrEnd (thenB ++ .wend :: t) = some (thenB, [], t) := by
  unfold splitAtElseOrEnd
  simp only [List.cons_append, List.append_assoc]
  rw [walkUntilCloser_append_noStructured h_then]
  simp only [walkUntilCloser, List.append_nil, List.reverse_reverse,
             Option.bind_eq_bind, Option.some_bind]

/-- The incoming pending list is a passenger on a `wif` segment: the
    arm reads only `base` (its sub-lowerings all start from fresh
    pending lists) and prepends `s.pending` to the result unchanged. -/
theorem lowerInstrsP_wif_pending_ride
    {fuel : Nat} {frames : List FrameKind} {base : LowerState}
    {p : List PendingWrap} {bt : Nat} {rest : List WasmInstr} :
    lowerInstrsP fuel frames ⟨base, p⟩ (.wif bt :: rest)
      = (lowerInstrsP fuel frames ⟨base, []⟩ (.wif bt :: rest)).map
          (fun q => (⟨q.1.base, p ++ q.1.pending⟩, q.2)) := by
  cases fuel with
  | zero => simp [lowerInstrsP]
  | succ f =>
      simp only [lowerInstrsP]
      cases h_split : splitAtElseOrEnd rest with
      | none => simp
      | some q =>
          obtain ⟨thenB, elseB, post⟩ := q
          simp only [Option.bind_eq_bind]
          cases h_pop : base.popSym with
          | none => simp
          | some q0 =>
              obtain ⟨svCond, s0⟩ := q0
              simp only [Option.some_bind]
              cases h_commit : s0.commit svCond with
              | none => simp
              | some q1 =>
                  obtain ⟨cond, s1, opsCommit⟩ := q1
                  simp only [Option.some_bind]
                  cases h_then : lowerInstrsP f (.wif :: frames)
                      ⟨({ s1 with nextReg := s1.nextReg + 1 } : LowerState), []⟩ thenB with
                  | none => simp [LowerState.alloc, h_then]
                  | some q2 =>
                      obtain ⟨s2, thenOps⟩ := q2
                      simp only [LowerState.alloc, h_then, Option.some_bind]
                      cases h_guard2 : s2.pending.any (·.skip = 0) with
                      | true => simp [h_guard2]
                      | false =>
                          simp only [h_guard2, Bool.false_eq_true, ↓reduceIte]
                          cases h_else : lowerInstrsP f (.wif :: frames)
                              ⟨({ s2.base with
                                    localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                                    localTy := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy,
                                    currentReg :=
                                      ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).currentReg }
                                  : LowerState), []⟩ elseB with
                          | none => simp [h_else]
                          | some q3 =>
                              obtain ⟨s3, elseOps⟩ := q3
                              simp only [h_else, Option.some_bind]
                              cases h_guard3 : s3.pending.any (·.skip = 0) with
                              | true => simp [h_guard3]
                              | false =>
                                  simp only [h_guard3, Bool.false_eq_true, ↓reduceIte]
                                  cases h_post : lowerInstrsP f frames
                                      ⟨({ s3.base with currentReg := [] } : LowerState), []⟩ post with
                                  | none => simp [h_post]
                                  | some q4 =>
                                      obtain ⟨s4, postOps⟩ := q4
                                      simp [h_post, List.append_assoc]

/-- The bridge-grade body class: straight-line instructions and
    balanced `wif` frames with STRAIGHT-LINE arms (the composer's
    requirement; `IfBalanced` keeps the weaker marker-free arms for
    the scanner side). -/
inductive IfBodyStraight : List WasmInstr → Prop
  | empty : IfBodyStraight []
  | sl_cons {i : WasmInstr} {rest : List WasmInstr} :
      StraightLineInstr i → IfBodyStraight rest → IfBodyStraight (i :: rest)
  | wif_else_cons {bt : Nat} {thenB elseB post : List WasmInstr} :
      StraightLineInstrs thenB → StraightLineInstrs elseB → IfBodyStraight post →
      IfBodyStraight (.wif bt :: (thenB ++ .welse :: elseB ++ .wend :: post))
  | wif_noelse_cons {bt : Nat} {thenB post : List WasmInstr} :
      StraightLineInstrs thenB → IfBodyStraight post →
      IfBodyStraight (.wif bt :: (thenB ++ .wend :: post))

/-- Bridge-grade bodies are scanner-grade. -/
theorem IfBodyStraight.ifBalanced : ∀ {l : List WasmInstr},
    IfBodyStraight l → IfBalanced l := by
  intro l h
  induction h with
  | empty => exact .empty
  | sl_cons h_i _ IH => exact .sl_cons (straightLine_noStructured h_i) IH
  | wif_else_cons h_then h_else _ IH =>
      exact .wif_else_cons (straightLine_noStructured_list h_then)
        (straightLine_noStructured_list h_else) IH
  | wif_noelse_cons h_then _ IH =>
      exact .wif_noelse_cons (straightLine_noStructured_list h_then) IH

/-- A bridge-grade body followed by the loop backedge, under Stage B:
    the body lowers exactly as Stage A does — each `wif` frame's
    sub-pendings are born empty and die at its close, the passenger
    list rides untouched — and `br 0` (the continue) contributes no
    IR and no pending. The segment's whole P-run IS the body's A-run. -/
theorem lowerInstrsP_ifBodyStraight_br0
    {frames : List FrameKind}
    (h_fr : frames.get? 0 = some .loopK)
    {body : List WasmInstr} (h_b : IfBodyStraight body)
    {fuel : Nat} {s : LowerState} {p : List PendingWrap}
    {sp' : LowerStateP} {ops : List KernelOp}
    (hl : lowerInstrsP fuel frames ⟨s, p⟩ (body ++ [.br 0]) = some (sp', ops)) :
    ∃ s_m : LowerState,
      lowerInstrs fuel frames s body = some (s_m, ops) ∧ sp' = ⟨s_m, p⟩ := by
  induction h_b generalizing fuel s p sp' ops with
  | empty =>
      rw [List.nil_append] at hl
      simp only [lowerInstrsP, h_fr, ↓reduceIte, Option.some.injEq,
                 Prod.mk.injEq] at hl
      refine ⟨s, ?_, hl.1.symm⟩
      simp [lowerInstrs, ← hl.2]
  | @sl_cons i rest' h_i _ IH =>
      rw [List.cons_append] at hl
      rw [lowerInstrsP_cons_default _ _ _ _ _ _ h_i] at hl
      cases hli : lowerInstr s i with
      | none => rw [hli] at hl; simp at hl
      | some q1 =>
          rw [hli] at hl
          obtain ⟨s1, ops_i⟩ := q1
          simp only [Option.bind_eq_bind, Option.some_bind] at hl
          cases hlr : lowerInstrsP fuel frames ⟨s1, p⟩ (rest' ++ [.br 0]) with
          | none => rw [hlr] at hl; simp at hl
          | some q2 =>
              rw [hlr] at hl
              obtain ⟨sp2, ops_r⟩ := q2
              simp only [Option.some_bind, pure, Option.some.injEq, Prod.mk.injEq] at hl
              obtain ⟨h_sp, h_ops⟩ := hl
              subst h_sp
              obtain ⟨s_m, hA, h_sp2⟩ := IH hlr
              refine ⟨s_m, ?_, h_sp2⟩
              rw [lowerInstrs_cons_default fuel frames s i rest'
                    (straightLine_not_structured_lower h_i), hli]
              simp only [Option.bind_eq_bind, Option.some_bind, hA]
              simp [← h_ops]
  | @wif_else_cons bt thenB elseB post h_then h_else _ IH =>
      -- The P-wif arm needs fuel; at 0 it refuses.
      cases fuel with
      | zero => simp [List.cons_append, lowerInstrsP] at hl
      | succ f =>
          have h_split := splitAtElseOrEnd_noStructured_else
            (straightLine_noStructured_list h_then)
            (straightLine_noStructured_list h_else) (post ++ [.br 0])
          simp only [List.cons_append, List.append_assoc, List.singleton_append] at hl h_split ⊢
          simp only [lowerInstrsP, h_split, Option.bind_eq_bind] at hl
          cases h_pop : s.popSym with
          | none => simp [h_pop] at hl
          | some q0 =>
              obtain ⟨svCond, s0⟩ := q0
              simp only [h_pop, Option.some_bind] at hl
              cases h_commit : s0.commit svCond with
              | none => simp [h_commit] at hl
              | some q1 =>
                  obtain ⟨cond, s1, opsCommit⟩ := q1
                  simp only [h_commit, Option.some_bind, LowerState.alloc] at hl
                  cases h_thenP : lowerInstrsP f (.wif :: frames)
                      ⟨({ s1 with nextReg := s1.nextReg + 1 } : LowerState), []⟩ thenB with
                  | none => rw [h_thenP] at hl; simp at hl
                  | some q2 =>
                      obtain ⟨sp2, thenOps⟩ := q2
                      rw [h_thenP] at hl
                      simp only [Option.some_bind] at hl
                      -- The then arm is straight-line: its P-run is its A-run
                      -- with an empty pending output.
                      obtain ⟨s2, ops_t, ops_nil, hA_then, h_rest_t, h_ops_t⟩ :=
                        lowerInstrsP_straightLine_append h_then
                          (rest := []) (by rw [List.append_nil]; exact h_thenP)
                      simp only [lowerInstrsP, Option.some.injEq, Prod.mk.injEq] at h_rest_t
                      obtain ⟨h_sp2, h_ops_nil⟩ := h_rest_t
                      subst h_sp2
                      subst h_ops_nil
                      rw [List.append_nil] at h_ops_t
                      subst h_ops_t
                      simp only [List.any_nil, Bool.false_eq_true, ↓reduceIte] at hl
                      cases h_elseP : lowerInstrsP f (.wif :: frames)
                          ⟨({ s2 with
                                localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                                localTy := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy,
                                currentReg :=
                                  ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).currentReg }
                              : LowerState), []⟩ elseB with
                      | none => rw [h_elseP] at hl; simp at hl
                      | some q3 =>
                          obtain ⟨sp3, elseOps⟩ := q3
                          rw [h_elseP] at hl
                          simp only [Option.some_bind] at hl
                          obtain ⟨s3, ops_e, ops_nil3, hA_else, h_rest_e, h_ops_e⟩ :=
                            lowerInstrsP_straightLine_append h_else
                              (rest := []) (by rw [List.append_nil]; exact h_elseP)
                          simp only [lowerInstrsP, Option.some.injEq, Prod.mk.injEq] at h_rest_e
                          obtain ⟨h_sp3, h_ops_nil3⟩ := h_rest_e
                          subst h_sp3
                          subst h_ops_nil3
                          rw [List.append_nil] at h_ops_e
                          subst h_ops_e
                          simp only [List.any_nil, Bool.false_eq_true, ↓reduceIte] at hl
                          cases h_postP : lowerInstrsP f frames
                              ⟨({ s3 with currentReg := [] } : LowerState), []⟩
                              (post ++ [.br 0]) with
                          | none => rw [h_postP] at hl; simp at hl
                          | some q4 =>
                              obtain ⟨sp4, postOps⟩ := q4
                              rw [h_postP] at hl
                              simp only [Option.some_bind, pure, Option.some.injEq,
                                         Prod.mk.injEq] at hl
                              obtain ⟨h_sp, h_ops⟩ := hl
                              obtain ⟨s_m, hA_post, h_sp4⟩ := IH h_postP
                              subst h_sp4
                              refine ⟨s_m, ?_, ?_⟩
                              · -- Assemble the Stage-A wif arm from the pieces,
                                -- restated over the reduced record literals the
                                -- goal's projections produce (defeq `exact`,
                                -- then syntactic `rw`).
                                have h_splitA := splitAtElseOrEnd_noStructured_else
                                  (straightLine_noStructured_list h_then)
                                  (straightLine_noStructured_list h_else) post
                                simp only [List.cons_append, List.append_assoc,
                                           List.singleton_append] at h_splitA ⊢
                                have hA_then' : lowerInstrs f (.wif :: frames)
                                    ({ nextReg := s1.nextReg + 1, stack := s1.stack,
                                       localReg := s1.localReg, localTy := s1.localTy,
                                       bufferSlots := s1.bufferSlots,
                                       currentReg := s1.currentReg } : LowerState) thenB
                                    = some (s2, thenOps) := hA_then
                                have hA_else' : lowerInstrs f (.wif :: frames)
                                    ({ nextReg := s2.nextReg, stack := s2.stack,
                                       localReg := s1.localReg, localTy := s1.localTy,
                                       bufferSlots := s2.bufferSlots,
                                       currentReg := s1.currentReg } : LowerState) elseB
                                    = some (s3, elseOps) := hA_else
                                have hA_post' : lowerInstrs f frames
                                    ({ nextReg := s3.nextReg, stack := s3.stack,
                                       localReg := s3.localReg, localTy := s3.localTy,
                                       bufferSlots := s3.bufferSlots,
                                       currentReg := [] } : LowerState) post
                                    = some (s_m, postOps) := hA_post
                                simp only [lowerInstrs]
                                rw [h_splitA]
                                simp only [Option.bind_eq_bind, h_pop, Option.some_bind,
                                           h_commit, LowerState.alloc]
                                rw [hA_then']
                                simp only [Option.some_bind]
                                rw [hA_else']
                                simp only [Option.some_bind]
                                rw [hA_post']
                                simp only [Option.some_bind, pure, Option.some.injEq,
                                           Prod.mk.injEq]
                                exact ⟨trivial, h_ops⟩
                              · rw [← h_sp]
                                simp
  | @wif_noelse_cons bt thenB post h_then _ IH =>
      cases fuel with
      | zero => simp [List.cons_append, lowerInstrsP] at hl
      | succ f =>
          have h_split := splitAtElseOrEnd_noStructured_noelse
            (straightLine_noStructured_list h_then) (post ++ [.br 0])
          simp only [List.cons_append, List.append_assoc, List.singleton_append] at hl h_split ⊢
          simp only [lowerInstrsP, h_split, Option.bind_eq_bind] at hl
          cases h_pop : s.popSym with
          | none => simp [h_pop] at hl
          | some q0 =>
              obtain ⟨svCond, s0⟩ := q0
              simp only [h_pop, Option.some_bind] at hl
              cases h_commit : s0.commit svCond with
              | none => simp [h_commit] at hl
              | some q1 =>
                  obtain ⟨cond, s1, opsCommit⟩ := q1
                  simp only [h_commit, Option.some_bind, LowerState.alloc] at hl
                  cases h_thenP : lowerInstrsP f (.wif :: frames)
                      ⟨({ s1 with nextReg := s1.nextReg + 1 } : LowerState), []⟩ thenB with
                  | none => rw [h_thenP] at hl; simp at hl
                  | some q2 =>
                      obtain ⟨sp2, thenOps⟩ := q2
                      rw [h_thenP] at hl
                      simp only [Option.some_bind] at hl
                      obtain ⟨s2, ops_t, ops_nil, hA_then, h_rest_t, h_ops_t⟩ :=
                        lowerInstrsP_straightLine_append h_then
                          (rest := []) (by rw [List.append_nil]; exact h_thenP)
                      simp only [lowerInstrsP, Option.some.injEq, Prod.mk.injEq] at h_rest_t
                      obtain ⟨h_sp2, h_ops_nil⟩ := h_rest_t
                      subst h_sp2
                      subst h_ops_nil
                      rw [List.append_nil] at h_ops_t
                      subst h_ops_t
                      -- The empty else arm reduces in place: its P-run is
                      -- `some (⟨restored, []⟩, [])`, the guard is false.
                      simp only [List.any_nil, Bool.false_eq_true, ↓reduceIte,
                                 lowerInstrsP, Option.some_bind] at hl
                      cases h_postP : lowerInstrsP f frames
                              ⟨({ s2 with
                                    localReg := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localReg,
                                    localTy := ({ s1 with nextReg := s1.nextReg + 1 } : LowerState).localTy,
                                    currentReg := [] } : LowerState), []⟩
                              (post ++ [.br 0]) with
                          | none => rw [h_postP] at hl; simp at hl
                          | some q4 =>
                              obtain ⟨sp4, postOps⟩ := q4
                              rw [h_postP] at hl
                              simp only [Option.some_bind, pure, Option.some.injEq,
                                         Prod.mk.injEq] at hl
                              obtain ⟨h_sp, h_ops⟩ := hl
                              obtain ⟨s_m, hA_post, h_sp4⟩ := IH h_postP
                              subst h_sp4
                              refine ⟨s_m, ?_, ?_⟩
                              · have h_splitA := splitAtElseOrEnd_noStructured_noelse
                                  (straightLine_noStructured_list h_then) post
                                simp only [List.cons_append, List.append_assoc,
                                           List.singleton_append] at h_splitA ⊢
                                have hA_then' : lowerInstrs f (.wif :: frames)
                                    ({ nextReg := s1.nextReg + 1, stack := s1.stack,
                                       localReg := s1.localReg, localTy := s1.localTy,
                                       bufferSlots := s1.bufferSlots,
                                       currentReg := s1.currentReg } : LowerState) thenB
                                    = some (s2, thenOps) := hA_then
                                have hA_post' : lowerInstrs f frames
                                    ({ nextReg := s2.nextReg, stack := s2.stack,
                                       localReg := s1.localReg, localTy := s1.localTy,
                                       bufferSlots := s2.bufferSlots,
                                       currentReg := [] } : LowerState) post
                                    = some (s_m, postOps) := hA_post
                                simp only [lowerInstrs]
                                rw [h_splitA]
                                simp only [Option.bind_eq_bind, h_pop, Option.some_bind,
                                           h_commit, LowerState.alloc, lowerInstrs]
                                rw [hA_then']
                                simp only [Option.some_bind]
                                rw [hA_post']
                                simp only [Option.some_bind, pure, Option.some.injEq,
                                           Prod.mk.injEq]
                                exact ⟨trivial, h_ops⟩
                              · rw [← h_sp]
                                simp

end Quanta.Wasm
