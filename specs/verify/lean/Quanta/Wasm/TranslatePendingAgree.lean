/-
# The pending-wrap translator agrees with the plain one on loop kernels

`TranslatePending.lean` proves one direction of Stage-A/Stage-B
agreement: whenever `lowerInstrs` accepts, `lowerInstrsP` with empty
pending in accepts with the same state and ops and empty pending out
(`lowerInstrsP_agrees_with_lowerInstrs`). The while-loop apex
`framework_preservation_kernel_while` (PreservationWhile.lean) is
stated over `lowerInstrs`; restating it over `lowerInstrsP` needs the
CONVERSE on the kernels it admits (`KernelInstrsW`: straight-line
instructions and `wloop 0` segments whose body is `pref ++ [.brIf 0]`).

Pieces:

1. `lowerInstrsP_cons_default` — a straight-line head takes the
   default arm of `lowerInstrsP`, pending untouched.
2. `lowerInstrsP_straightLine_append` — the straight-line part of a
   Stage-B lowering IS a Stage-A lowering; the split at the prefix
   boundary hands the rest on with the same pending.
3. `lowerInstrsP_brIf0_loop_empty_tail` — the body-closing `brIf 0`
   arm, Stage-B shape, pending carried through unchanged.
4. `lowerInstrsP_to_lowerInstrs_kernelW` — the converse on
   `KernelInstrsW` kernels, by induction on the kernel shape.
5. `lowerInstrsP_iff_lowerInstrs_kernelW` — both directions.
6. `framework_preservation_kernel_while_P` — the apex over
   `lowerInstrsP`.
-/

import Quanta.Wasm.TranslatePending
import Quanta.Wasm.PreservationWhile

namespace Quanta.Wasm

open Quanta.KOps (KernelOp evalOps)

-- ════════════════════════════════════════════════════════════════════
-- Straight-line heads: the Stage-B default arm.
-- ════════════════════════════════════════════════════════════════════

/-- `lowerInstrsP` on a straight-line head delegates to `lowerInstr`
    on the base state and recurses on the rest with the same pending
    list — the Stage-B analogue of `lowerInstrs_cons_default`. -/
theorem lowerInstrsP_cons_default
    (fuel : Nat) (frames : List FrameKind) (s : LowerState) (p : List PendingWrap)
    (i : WasmInstr) (rest : List WasmInstr)
    (h_sl : StraightLineInstr i) :
    lowerInstrsP fuel frames ⟨s, p⟩ (i :: rest) =
      (do
        let (s1, ops1) ← lowerInstr s i
        let (s2, ops2) ← lowerInstrsP fuel frames ⟨s1, p⟩ rest
        pure (s2, ops1 ++ ops2)) := by
  cases i
  all_goals try simp [StraightLineInstr] at h_sl
  all_goals simp only [lowerInstrsP]

/-- Stage-B lowering of a straight-line prefix followed by anything
    splits at the prefix boundary, and the prefix part is a PLAIN
    Stage-A lowering: the pending machinery is inert on straight-line
    instructions, so the rest starts from the prefix's end state with
    the pending list it was given. -/
theorem lowerInstrsP_straightLine_append
    {fuel : Nat} {frames : List FrameKind} {pref rest : List WasmInstr}
    (h_sl : StraightLineInstrs pref)
    {s : LowerState} {p : List PendingWrap} {sp' : LowerStateP} {ops : List KernelOp}
    (hl : lowerInstrsP fuel frames ⟨s, p⟩ (pref ++ rest) = some (sp', ops)) :
    ∃ (s_m : LowerState) (ops1 ops2 : List KernelOp),
      lowerInstrs fuel frames s pref = some (s_m, ops1) ∧
      lowerInstrsP fuel frames ⟨s_m, p⟩ rest = some (sp', ops2) ∧
      ops = ops1 ++ ops2 := by
  induction pref generalizing s ops with
  | nil =>
      refine ⟨s, [], ops, ?_, ?_, rfl⟩
      · simp [lowerInstrs]
      · simpa using hl
  | cons i pref' IH =>
      obtain ⟨h_i, h_pref'⟩ := h_sl
      rw [List.cons_append] at hl
      rw [lowerInstrsP_cons_default fuel frames s p i (pref' ++ rest) h_i] at hl
      cases hli : lowerInstr s i with
      | none => rw [hli] at hl; simp at hl
      | some q1 =>
          rw [hli] at hl
          obtain ⟨s1, ops_i⟩ := q1
          simp only [Option.bind_eq_bind, Option.some_bind] at hl
          cases hlr : lowerInstrsP fuel frames ⟨s1, p⟩ (pref' ++ rest) with
          | none => rw [hlr] at hl; simp at hl
          | some q2 =>
              rw [hlr] at hl
              obtain ⟨sp2, ops_r⟩ := q2
              simp only [Option.some_bind, pure, Option.some.injEq, Prod.mk.injEq] at hl
              obtain ⟨h_s, h_ops⟩ := hl
              subst h_s
              subst h_ops
              obtain ⟨s_m, ops1, ops2, h_p, h_r, h_eq⟩ := IH h_pref' hlr
              refine ⟨s_m, ops_i ++ ops1, ops2, ?_, h_r, ?_⟩
              · rw [lowerInstrs_cons_default fuel frames s i pref'
                    (straightLine_not_structured_lower h_i)]
                rw [hli]
                simp only [Option.bind_eq_bind, Option.some_bind]
                rw [h_p]
                rfl
              · rw [h_eq, List.append_assoc]

/-- Converse of `lowerInstrs_straightLine_append`: a Stage-A lowering
    of a straight-line prefix followed by one of the rest is a lowering
    of the concatenation, ops concatenated. Reassembles the while body
    `pref ++ [.brIf 0]` after its two halves are matched to Stage A. -/
theorem lowerInstrs_straightLine_compose
    {fuel : Nat} {frames : List FrameKind} {pref rest : List WasmInstr}
    (h_sl : StraightLineInstrs pref)
    {s s_m s' : LowerState} {ops1 ops2 : List KernelOp}
    (h_pref : lowerInstrs fuel frames s pref = some (s_m, ops1))
    (h_rest : lowerInstrs fuel frames s_m rest = some (s', ops2)) :
    lowerInstrs fuel frames s (pref ++ rest) = some (s', ops1 ++ ops2) := by
  induction pref generalizing s ops1 with
  | nil =>
      simp only [lowerInstrs, Option.some.injEq, Prod.mk.injEq] at h_pref
      obtain ⟨h_s, h_ops⟩ := h_pref
      subst h_s
      subst h_ops
      simpa using h_rest
  | cons i pref' IH =>
      obtain ⟨h_i, h_pref'⟩ := h_sl
      rw [lowerInstrs_cons_default fuel frames s i pref'
          (straightLine_not_structured_lower h_i)] at h_pref
      rw [List.cons_append, lowerInstrs_cons_default fuel frames s i (pref' ++ rest)
          (straightLine_not_structured_lower h_i)]
      cases hli : lowerInstr s i with
      | none => rw [hli] at h_pref; simp at h_pref
      | some q1 =>
          rw [hli] at h_pref
          obtain ⟨s1, ops_i⟩ := q1
          simp only [Option.bind_eq_bind, Option.some_bind] at h_pref
          cases hlr : lowerInstrs fuel frames s1 pref' with
          | none => rw [hlr] at h_pref; simp at h_pref
          | some q2 =>
              rw [hlr] at h_pref
              obtain ⟨s2, ops_r⟩ := q2
              simp only [Option.some_bind, pure, Option.some.injEq, Prod.mk.injEq] at h_pref
              obtain ⟨h_s, h_ops⟩ := h_pref
              subst h_s
              subst h_ops
              simp only [Option.bind_eq_bind, Option.some_bind]
              rw [IH h_pref' hlr]
              simp [List.append_assoc]

-- ════════════════════════════════════════════════════════════════════
-- The body-closing `brIf 0`: the Stage-B shape.
-- ════════════════════════════════════════════════════════════════════

/-- Empty-tail reduction of the Stage-B depth-0 loop-backedge arm —
    the analogue of `lowerInstrs_brIf0_loop_empty_tail`: the same
    pop/commit/cast/branch do-block, with the pending list carried
    through unchanged. -/
theorem lowerInstrsP_brIf0_loop_empty_tail
    (fuel : Nat) (frames : List FrameKind) (s : LowerState) (p : List PendingWrap)
    (h_target : frames.get? 0 = some .loopK) :
    lowerInstrsP fuel frames ⟨s, p⟩ [.brIf 0] =
      (do
        let (svCond, s0) ← s.popSym
        let (cond, s1, opsCommit) ← s0.commit svCond
        let (cond_bool, s_cast) := s1.alloc
        pure (⟨s_cast, p⟩,
          opsCommit ++ [.cast cond_bool cond .u32 .bool,
                        .branch cond_bool [] [.breakOp]])) := by
  simp only [lowerInstrsP]
  rcases hpop : s.popSym with _ | ⟨svCond, s0⟩
  · rfl
  simp only [Option.bind_eq_bind, Option.some_bind]
  rcases hcommit : s0.commit svCond with _ | ⟨cond, s1, opsCommit⟩
  · rfl
  simp only [Option.some_bind]
  rw [h_target]
  simp [lowerInstrsP, LowerState.alloc, backedgeEndBreak, endsInBreak]

/-- A Stage-B success on the body-closing `brIf 0` is a Stage-A
    success with the same base state and ops, the pending list
    untouched — both empty-tail reductions are the same do-block. -/
theorem lowerInstrsP_brIf0_loop_to_lowerInstrs
    {fuel : Nat} {frames : List FrameKind} {s : LowerState} {p : List PendingWrap}
    {sp' : LowerStateP} {ops : List KernelOp}
    (h_target : frames.get? 0 = some .loopK)
    (hl : lowerInstrsP fuel frames ⟨s, p⟩ [.brIf 0] = some (sp', ops)) :
    sp'.pending = p ∧ lowerInstrs fuel frames s [.brIf 0] = some (sp'.base, ops) := by
  rw [lowerInstrsP_brIf0_loop_empty_tail fuel frames s p h_target] at hl
  rw [lowerInstrs_brIf0_loop_empty_tail fuel frames s h_target]
  rcases hpop : s.popSym with _ | ⟨svCond, s0⟩
  · rw [hpop] at hl; simp at hl
  rw [hpop] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl ⊢
  rcases hcommit : s0.commit svCond with _ | ⟨cond, s1, opsCommit⟩
  · rw [hcommit] at hl; simp at hl
  rw [hcommit] at hl
  simp only [Option.some_bind, LowerState.alloc, pure, Option.some.injEq, Prod.mk.injEq] at hl ⊢
  obtain ⟨h_s, h_ops⟩ := hl
  subst h_s
  subst h_ops
  exact ⟨rfl, rfl, rfl⟩

-- ════════════════════════════════════════════════════════════════════
-- The converse on loop kernels.
-- ════════════════════════════════════════════════════════════════════

/-- On a `KernelInstrsW` kernel, a Stage-B success from empty pending
    ends with empty pending and is a Stage-A success on the same
    state and ops. Induction on the kernel shape: a straight-line head
    is the default arm on both sides; a `wloop 0` segment's body is
    `pref ++ [.brIf 0]`, whose Stage-B lowering is Stage A's piece by
    piece, and with nothing pending at the loop close the Stage-B
    `.wloop` arm is Stage A's. -/
theorem lowerInstrsP_to_lowerInstrs_kernelW
    {instrs : List WasmInstr} (h_wf : KernelInstrsW instrs) :
    ∀ (fuel : Nat) (frames : List FrameKind) (s : LowerState)
      (sp' : LowerStateP) (ops : List KernelOp),
      lowerInstrsP fuel frames ⟨s, []⟩ instrs = some (sp', ops) →
      sp'.pending = [] ∧ lowerInstrs fuel frames s instrs = some (sp'.base, ops) := by
  induction h_wf with
  | empty =>
      intro fuel frames s sp' ops hl
      simp only [lowerInstrsP, Option.some.injEq, Prod.mk.injEq] at hl
      obtain ⟨h_s, h_ops⟩ := hl
      subst h_s
      subst h_ops
      exact ⟨rfl, by simp only [lowerInstrs]⟩
  | @sl_cons i rest h_sl _h_rest_wf IH =>
      intro fuel frames s sp' ops hl
      rw [lowerInstrsP_cons_default fuel frames s [] i rest h_sl] at hl
      rw [lowerInstrs_cons_default fuel frames s i rest
          (straightLine_not_structured_lower h_sl)]
      cases hli : lowerInstr s i with
      | none => rw [hli] at hl; simp at hl
      | some q1 =>
          rw [hli] at hl
          obtain ⟨s1, ops_i⟩ := q1
          simp only [Option.bind_eq_bind, Option.some_bind] at hl ⊢
          cases hlr : lowerInstrsP fuel frames ⟨s1, []⟩ rest with
          | none => rw [hlr] at hl; simp at hl
          | some q2 =>
              rw [hlr] at hl
              obtain ⟨sp2, ops_r⟩ := q2
              simp only [Option.some_bind, pure, Option.some.injEq, Prod.mk.injEq] at hl
              obtain ⟨h_s, h_ops⟩ := hl
              subst h_s
              subst h_ops
              obtain ⟨h_pend, h_rest⟩ := IH fuel frames s1 sp2 ops_r hlr
              refine ⟨h_pend, ?_⟩
              rw [h_rest]
              rfl
  | @while_cons rest body post h_split h_body _h_post_wf IH =>
      intro fuel frames s sp' ops hl
      obtain ⟨pref, h_sl, _h_ht, h_body_eq⟩ := h_body
      subst h_body_eq
      cases fuel with
      | zero => simp [lowerInstrsP] at hl
      | succ f =>
          simp only [lowerInstrsP, h_split] at hl
          -- The body: straight-line prefix (Stage A) then the closing
          -- `brIf 0` (Stage A again, pending still empty).
          cases hlb : lowerInstrsP f (.loopK :: frames) ⟨{ s with currentReg := [] }, []⟩
              (pref ++ [.brIf 0]) with
          | none => rw [hlb] at hl; simp at hl
          | some q1 =>
              rw [hlb] at hl
              obtain ⟨⟨s1, p1⟩, bodyOps⟩ := q1
              obtain ⟨s_m, ops1, ops2, hA_pref, hB_br, h_bodyOps⟩ :=
                lowerInstrsP_straightLine_append h_sl hlb
              have h_p1 : p1 = [] :=
                (lowerInstrsP_brIf0_loop_to_lowerInstrs rfl hB_br).1
              have hA_br : lowerInstrs f (.loopK :: frames) s_m [.brIf 0] = some (s1, ops2) :=
                (lowerInstrsP_brIf0_loop_to_lowerInstrs rfl hB_br).2
              subst h_p1
              subst h_bodyOps
              have hA_body : lowerInstrs f (.loopK :: frames) { s with currentReg := [] }
                  (pref ++ [.brIf 0]) = some (s1, ops1 ++ ops2) :=
                lowerInstrs_straightLine_compose h_sl hA_pref hA_br
              simp only [Option.bind_eq_bind, Option.some_bind, hasPlainActive_nil,
                         Bool.false_eq_true, ↓reduceIte, ne_eq, not_true_eq_false, if_false] at hl
              -- The post, by the IH.
              cases hlp : lowerInstrsP f frames ⟨{ s1 with currentReg := [] }, []⟩ post with
              | none => rw [hlp] at hl; simp at hl
              | some q2 =>
                  rw [hlp] at hl
                  obtain ⟨⟨s2, p2⟩, postOps⟩ := q2
                  have h_p2 : p2 = [] := (IH f frames _ _ _ hlp).1
                  have hA_post : lowerInstrs f frames { s1 with currentReg := [] } post
                      = some (s2, postOps) := (IH f frames _ _ _ hlp).2
                  subst h_p2
                  simp only [Option.some_bind, pure, stepPending_nil, closeDecls_nil,
                             applyWraps_nil, List.append_nil, List.nil_append,
                             Option.some.injEq, Prod.mk.injEq] at hl
                  obtain ⟨h_s, h_ops⟩ := hl
                  subst h_s
                  subst h_ops
                  refine ⟨rfl, ?_⟩
                  simp only [lowerInstrs, h_split, hA_body, Option.bind_eq_bind,
                             Option.some_bind, hA_post, pure]

/-- Both directions on a `KernelInstrsW` kernel: Stage B from and to
    empty pending is Stage A. -/
theorem lowerInstrsP_iff_lowerInstrs_kernelW
    {instrs : List WasmInstr} (h_wf : KernelInstrsW instrs)
    (fuel : Nat) (frames : List FrameKind) (s s' : LowerState) (ops : List KernelOp) :
    lowerInstrsP fuel frames ⟨s, []⟩ instrs = some (⟨s', []⟩, ops) ↔
    lowerInstrs fuel frames s instrs = some (s', ops) := by
  constructor
  · intro hp
    exact (lowerInstrsP_to_lowerInstrs_kernelW h_wf fuel frames s ⟨s', []⟩ ops hp).2
  · intro ha
    exact lowerInstrsP_agrees_with_lowerInstrs fuel frames s instrs ha

-- ════════════════════════════════════════════════════════════════════
-- The while-loop apex over the pending-wrap translator.
-- ════════════════════════════════════════════════════════════════════

/-- `framework_preservation_kernel_while` restated over `lowerInstrsP`:
    on a `KernelInstrsW` kernel the two translators agree, so the
    Stage-A apex applies to the Stage-B lowering verbatim. -/
theorem framework_preservation_kernel_while_P
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
    (hl : lowerInstrsP (fuel + 1) frames ⟨s, []⟩ instrs = some (⟨s', []⟩, ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  framework_preservation_kernel_while fuel frames ws s kst layout R h_no_branch h_no_halt
    h_kst_no_broke h_buf_locals h_no_buf_stack h_load_bounds h_store_bounds h_store_layout
    instrs h_wf h_fuel h_ts ws' s' ops hw
    ((lowerInstrsP_iff_lowerInstrs_kernelW h_wf (fuel + 1) frames s s' ops).mp hl)

end Quanta.Wasm
