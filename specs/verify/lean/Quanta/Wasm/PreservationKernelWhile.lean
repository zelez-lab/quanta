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
import Quanta.Wasm.PreservationBackedgeTail
import Quanta.Wasm.SeededLocals

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
  | backedge_tail_cons {pref post : List WasmInstr} :
      StraightLineInstrs pref →
      stackHeight 0 pref = some 1 →
      KernelInstrsW2 post →
      KernelInstrsW2 (.block 0 :: .wloop 0 :: (pref ++ [.brIf 0] ++ [.br 1])
                        ++ [.wend] ++ [] ++ [.wend] ++ post)

/-- Loop nesting depth — the fuel measure. A `while` counts one: its
    block and its loop spend the two units the bound already grants. -/
def KernelInstrsW2.depth : ∀ {instrs : List WasmInstr}, KernelInstrsW2 instrs → Nat
  | _, .empty => 0
  | _, .sl_cons _ rest_wf => rest_wf.depth
  | _, .while_cons _ _ post_wf => 1 + post_wf.depth
  | _, .block_while_cons _ _ _ _ post_wf => 1 + post_wf.depth
  | _, .backedge_tail_cons _ _ post_wf => 1 + post_wf.depth

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
  | _, @backedge_tail_cons pref _ _ _ post_wf, fuel, frames, s =>
      match fuel with
      | 0 => True
      | 1 => True
      | f + 2 =>
          ∀ s1 p2 bodyOps,
            lowerInstrsP f (.loopK :: .block :: frames) ⟨{ s with currentReg := [] }, []⟩
                (pref ++ [.brIf 0] ++ [.br 1]) = some (⟨s1, p2⟩, bodyOps) →
            s1.localTy = s.localTy ∧
            post_wf.stable (f + 1) frames { s1 with currentReg := [] }

-- ════════════════════════════════════════════════════════════════════
-- Seeding discharges the exit-side registers condition
-- ════════════════════════════════════════════════════════════════════

/-- Under seeding, the rustc-while arm's `localReg` conjuncts hold
    outright: prefix, exit site and `body2` all leave the stable layer
    list-identical, so the state at the body's end IS the site's map —
    no matter how many (bound) locals `body2` rebinds. -/
theorem blockWhile_localReg_of_seeded
    {fuel : Nat} {frames : List FrameKind} {pref body2 : List WasmInstr}
    (h_pref : StraightLineInstrs pref) (h_body2 : StraightLineInstrs body2)
    {s s_site s1 : LowerState} {p1 p2 : List PendingWrap}
    {ops1 bodyOps : List KernelOp}
    (hnd : KeysNodup s.localReg)
    (h_seed_pref : LocalsSeeded s pref) (h_seed_body2 : LocalsSeeded s body2)
    (hl_site : lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s, []⟩ (pref ++ [.brIf 1])
        = some (⟨s_site, p1⟩, ops1))
    (h_lb : lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s, []⟩
        (pref ++ [.brIf 1] ++ body2 ++ [.br 0]) = some (⟨s1, p2⟩, bodyOps)) :
    s1.localReg = s_site.localReg ∧ s_site.localReg = s.localReg := by
  -- Split both lowerings at the straight-line prefix.
  obtain ⟨s_m, opsP, opsR, h_low_pref, h_brIf, _⟩ :=
    lowerInstrsP_straightLine_append h_pref hl_site
  have h_lb' : lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s, []⟩
      (pref ++ (.brIf 1 :: (body2 ++ [.br 0]))) = some (⟨s1, p2⟩, bodyOps) := by
    have h_shape : pref ++ [.brIf 1] ++ body2 ++ [.br 0]
        = pref ++ (.brIf 1 :: (body2 ++ [.br 0])) := by
      simp [List.append_assoc]
    rw [← h_shape]
    exact h_lb
  obtain ⟨s_m2, opsP2, opsR2, h_low_pref2, h_rest2, _⟩ :=
    lowerInstrsP_straightLine_append h_pref h_lb'
  -- Determinism: the prefix lowering is one function call.
  rw [h_low_pref] at h_low_pref2
  have h_m2 : s_m = s_m2 := by
    have h_pair := (Option.some.injEq _ _).mp h_low_pref2
    exact ((Prod.mk.injEq _ _ _ _).mp h_pair).1
  rw [← h_m2] at h_rest2
  -- The prefix leaves the stable layer alone.
  have h_pref_lr : s_m.localReg = s.localReg :=
    lowerInstrs_localReg_seeded h_pref hnd h_seed_pref h_low_pref
  -- Unfold the br_if arm in both continuations.
  have h_singleton : ([.brIf 1] : List WasmInstr) = .brIf 1 :: [] := rfl
  rw [h_singleton, lowerInstrsP_brIf1_exit] at h_brIf
  rw [lowerInstrsP_brIf1_exit] at h_rest2
  -- The pop and the commit are the same in both.
  rcases hpop : s_m.popSym with _ | ⟨svCond, s0⟩
  · rw [hpop] at h_brIf; simp at h_brIf
  rw [hpop] at h_brIf h_rest2
  simp only [Option.bind_eq_bind, Option.some_bind] at h_brIf h_rest2
  rcases hcommit : s0.commit svCond with _ | ⟨cond, s_c, opsCommit⟩
  · rw [hcommit] at h_brIf; simp at h_brIf
  rw [hcommit] at h_brIf h_rest2
  simp only [Option.some_bind, LowerState.alloc] at h_brIf h_rest2
  -- The flag state's stable layer is the prefix end's.
  have h_c_lr : s_c.localReg = s.localReg := by
    rw [LowerState.commit_localReg hcommit, LowerState.popSym_localReg hpop, h_pref_lr]
  -- Site continuation: rest = [], the lowering is the flag state.
  simp only [lowerInstrsP, Option.some_bind, pure, Option.some.injEq, Prod.mk.injEq,
             LowerStateP.mk.injEq] at h_brIf
  obtain ⟨⟨h_site_eq, _⟩, _⟩ := h_brIf
  -- Body continuation: split at body2, close with the inert `br 0`.
  rcases hrest : lowerInstrsP fuel (.loopK :: .block :: frames)
      ⟨{ s_c with nextReg := s_c.nextReg + 1 + 1 }, []⟩ (body2 ++ [.br 0])
      with _ | ⟨sp_b, restOps⟩
  · rw [hrest] at h_rest2; simp at h_rest2
  rw [hrest] at h_rest2
  simp only [Option.some_bind, pure, Option.some.injEq, Prod.mk.injEq,
             LowerStateP.mk.injEq] at h_rest2
  obtain ⟨⟨h_s1_eq, _⟩, _⟩ := h_rest2
  obtain ⟨s_b, opsB2, opsBr, h_low_b2, h_br0, _⟩ :=
    lowerInstrsP_straightLine_append h_body2 hrest
  rw [lowerInstrsP_br0_loop] at h_br0
  have h_spb : sp_b = ⟨s_b, []⟩ := by
    have := (Option.some.injEq _ _).mp h_br0
    exact ((Prod.mk.injEq _ _ _ _).mp this).1.symm
  -- body2 leaves the stable layer alone, from the flag state.
  have h_flag_lr : ({ s_c with nextReg := s_c.nextReg + 1 + 1 } : LowerState).localReg
      = s.localReg := h_c_lr
  have h_b2_lr : s_b.localReg = s.localReg := by
    have := lowerInstrs_localReg_seeded h_body2
      (show KeysNodup ({ s_c with nextReg := s_c.nextReg + 1 + 1 } : LowerState).localReg by
        rw [h_flag_lr]; exact hnd)
      (h_seed_body2.of_localReg_eq h_flag_lr) h_low_b2
    rw [this, h_flag_lr]
  constructor
  · rw [← h_s1_eq, h_spb]
    show s_b.localReg = s_site.localReg
    rw [h_b2_lr, ← h_site_eq]
    exact h_c_lr.symm
  · rw [← h_site_eq]
    exact h_c_lr

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
-- Label-only side conditions + the seeded bridge
-- ════════════════════════════════════════════════════════════════════

/-- The label-only side conditions: `stable` minus the `localReg`
    conjunct the seeded discharge provides. What remains is the
    label-stability face (`localTy`). -/
def KernelInstrsW2.labelStable : ∀ {instrs : List WasmInstr},
    KernelInstrsW2 instrs → Nat → List FrameKind → LowerState → Prop
  | _, .empty, _, _, _ => True
  | _, @sl_cons i _ _ rest_wf, fuel, frames, s =>
      ∀ s1 ops, lowerInstr s i = some (s1, ops) → rest_wf.labelStable fuel frames s1
  | _, @while_cons _ body _ _ _ post_wf, fuel, frames, s =>
      match fuel with
      | 0 => True
      | f + 1 =>
          ∀ s1 bodyOps,
            lowerInstrs f (.loopK :: frames) { s with currentReg := [] } body
              = some (s1, bodyOps) →
            s1.localTy = s.localTy ∧
            post_wf.labelStable f frames { s1 with currentReg := [] }
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
              s1.localTy = s.localTy ∧ s1.localTy = s_site.localTy ∧
              post_wf.labelStable (f + 1) frames { s1 with currentReg := [] }
  | _, @backedge_tail_cons pref _ _ _ post_wf, fuel, frames, s =>
      match fuel with
      | 0 => True
      | 1 => True
      | f + 2 =>
          ∀ s1 p2 bodyOps,
            lowerInstrsP f (.loopK :: .block :: frames) ⟨{ s with currentReg := [] }, []⟩
                (pref ++ [.brIf 0] ++ [.br 1]) = some (⟨s1, p2⟩, bodyOps) →
            s1.localTy = s.localTy ∧
            post_wf.labelStable (f + 1) frames { s1 with currentReg := [] }

/-- Under seeding, the backedge+tail body leaves the stable layer
    list-identical: prefix (straight-line, seeded), then the pop, the
    commit and the two allocations of the site. -/
theorem backedgeTail_localReg_of_seeded
    {fuel : Nat} {frames : List FrameKind} {pref : List WasmInstr}
    (h_pref : StraightLineInstrs pref)
    {s s1 : LowerState} {p2 : List PendingWrap} {bodyOps : List KernelOp}
    (hnd : KeysNodup s.localReg) (h_seed : LocalsSeeded s pref)
    (h_lb : lowerInstrsP fuel (.loopK :: .block :: frames) ⟨s, []⟩
        (pref ++ [.brIf 0] ++ [.br 1]) = some (⟨s1, p2⟩, bodyOps)) :
    s1.localReg = s.localReg := by
  obtain ⟨s_m, s0, s_c, svCond, cond, opsPref, opsCommit,
          hl_pref, h_pop, h_commit, h_sp, _⟩ :=
    backedgeTailBody_lowerP h_pref h_lb
  simp only [LowerStateP.mk.injEq] at h_sp
  obtain ⟨h_s1, _⟩ := h_sp
  subst h_s1
  show s_c.localReg = s.localReg
  rw [LowerState.commit_localReg h_commit, LowerState.popSym_localReg h_pop]
  exact lowerInstrs_localReg_seeded h_pref hnd h_seed hl_pref

/-- Seeding turns the label-only conditions into the full `stable`: the
    rustc-while arm's `localReg` conjunct holds outright, and the
    seeding+uniqueness invariants ride the (unchanged) stable layer
    into every recursive position. -/
theorem KernelInstrsW2.stable_of_seeded :
    ∀ {instrs : List WasmInstr} (wf : KernelInstrsW2 instrs)
      (fuel : Nat) (frames : List FrameKind) (s : LowerState),
    wf.labelStable fuel frames s →
    KeysNodup s.localReg →
    LocalsSeeded s instrs →
    wf.stable fuel frames s := by
  intro instrs wf
  induction wf with
  | empty => intro fuel frames s _ _ _; trivial
  | @sl_cons i rest h_i rest_wf IH =>
      intro fuel frames s h_lab hnd h_seed
      intro s1 ops hl
      have h_lr : s1.localReg = s.localReg :=
        lowerInstr_localReg_seeded h_i hnd h_seed.head hl
      exact IH fuel frames s1 (h_lab s1 ops hl)
        (by rw [h_lr]; exact hnd) (h_seed.tail.of_localReg_eq h_lr)
  | @while_cons rest body post h_split h_body post_wf IH =>
      intro fuel frames s h_lab hnd h_seed
      cases fuel with
      | zero => trivial
      | succ f =>
          intro s1 bodyOps hl
          obtain ⟨h_lt, h_post_lab⟩ := h_lab s1 bodyOps hl
          refine ⟨h_lt, ?_⟩
          obtain ⟨pref', h_sl', h_ht', h_beq⟩ := h_body
          rw [h_beq] at hl
          have h_shape := splitAtEnd_append h_split
          have h_wl : writtenLocals (WasmInstr.wloop 0 :: rest)
              = writtenLocals pref' ++ writtenLocals post := by
            rw [writtenLocals_cons, ← h_shape, h_beq]
            simp
          have h_seed_pref : LocalsSeeded ({ s with currentReg := [] } : LowerState) pref' := by
            refine LocalsSeeded.of_subset (b := .wloop 0 :: rest) ?_
              (h_seed.of_localReg_eq rfl)
            intro j hj
            rw [h_wl]
            exact List.mem_append_left _ hj
          have h_lr : s1.localReg = s.localReg :=
            whileBody_localReg_of_seeded (s := { s with currentReg := [] })
              h_sl' hnd h_seed_pref hl
          have h_seed_post : LocalsSeeded ({ s1 with currentReg := [] } : LowerState) post := by
            refine LocalsSeeded.of_subset (b := .wloop 0 :: rest) ?_
              (h_seed.of_localReg_eq (show ({ s1 with currentReg := [] } : LowerState).localReg
                = s.localReg from h_lr))
            intro j hj
            rw [h_wl]
            exact List.mem_append_right _ hj
          exact IH f frames { s1 with currentReg := [] } h_post_lab
            (show KeysNodup ({ s1 with currentReg := [] } : LowerState).localReg by
              show KeysNodup s1.localReg; rw [h_lr]; exact hnd) h_seed_post
  | @block_while_cons pref body2 post h_pref h_body2 h_ht_pref h_ht_body2 post_wf IH =>
      intro fuel frames s h_lab hnd h_seed
      cases fuel with
      | zero => trivial
      | succ f0 =>
      cases f0 with
      | zero => trivial
      | succ f =>
          intro s_site p1 ops1 hl_site s1 p2 bodyOps h_lb
          obtain ⟨h_lt, h_lt_site, h_post_lab⟩ :=
            h_lab s_site p1 ops1 hl_site s1 p2 bodyOps h_lb
          have h_wl : writtenLocals (WasmInstr.block 0 :: .wloop 0 ::
                (pref ++ [.brIf 1] ++ body2 ++ [.br 0]) ++ [.wend] ++ [] ++ [.wend] ++ post)
              = writtenLocals pref ++ writtenLocals body2 ++ writtenLocals post := by
            simp
          have h_seed_shape : ∀ j, j ∈ writtenLocals pref ∨ j ∈ writtenLocals body2 ∨
              j ∈ writtenLocals post →
              (s.lookupLocal j).isSome := by
            intro j hj
            apply h_seed
            rw [h_wl]
            rcases hj with h | h | h
            · exact List.mem_append_left _ (List.mem_append_left _ h)
            · exact List.mem_append_left _ (List.mem_append_right _ h)
            · exact List.mem_append_right _ h
          have h_seed_pref : LocalsSeeded ({ s with currentReg := [] } : LowerState) pref :=
            fun j hj => h_seed_shape j (Or.inl hj)
          have h_seed_body2 : LocalsSeeded ({ s with currentReg := [] } : LowerState) body2 :=
            fun j hj => h_seed_shape j (Or.inr (Or.inl hj))
          obtain ⟨h_lr_site_eq, h_site_s⟩ :=
            blockWhile_localReg_of_seeded (s := { s with currentReg := [] })
              h_pref h_body2 hnd h_seed_pref h_seed_body2 hl_site h_lb
          refine ⟨h_lt, h_lr_site_eq, h_lt_site, ?_⟩
          have h_lr : s1.localReg = s.localReg := by rw [h_lr_site_eq, h_site_s]
          have h_seed_post : LocalsSeeded ({ s1 with currentReg := [] } : LowerState) post := by
            intro j hj
            have h_s := h_seed_shape j (Or.inr (Or.inr hj))
            rw [lookupLocal_find?] at h_s ⊢
            show ((s1.localReg.find? (fun p => p.fst = j)).map Prod.snd).isSome
            rw [h_lr]
            exact h_s
          exact IH (f + 1) frames { s1 with currentReg := [] } h_post_lab
            (show KeysNodup ({ s1 with currentReg := [] } : LowerState).localReg by
              show KeysNodup s1.localReg; rw [h_lr]; exact hnd) h_seed_post
  | @backedge_tail_cons pref post h_pref h_ht_pref post_wf IH =>
      intro fuel frames s h_lab hnd h_seed
      cases fuel with
      | zero => trivial
      | succ f0 =>
      cases f0 with
      | zero => trivial
      | succ f =>
          intro s1 p2 bodyOps h_lb
          obtain ⟨h_lt, h_post_lab⟩ := h_lab s1 p2 bodyOps h_lb
          have h_wl : writtenLocals (WasmInstr.block 0 :: .wloop 0 ::
                (pref ++ [.brIf 0] ++ [.br 1]) ++ [.wend] ++ [] ++ [.wend] ++ post)
              = writtenLocals pref ++ writtenLocals post := by
            simp
          have h_seed_pref : LocalsSeeded ({ s with currentReg := [] } : LowerState) pref := by
            intro j hj
            apply h_seed
            rw [h_wl]
            exact List.mem_append_left _ hj
          have h_lr : s1.localReg = s.localReg :=
            backedgeTail_localReg_of_seeded (s := { s with currentReg := [] })
              h_pref hnd h_seed_pref h_lb
          refine ⟨h_lt, ?_⟩
          have h_seed_post : LocalsSeeded ({ s1 with currentReg := [] } : LowerState) post := by
            intro j hj
            have h_s : (s.lookupLocal j).isSome := by
              apply h_seed
              rw [h_wl]
              exact List.mem_append_right _ hj
            rw [lookupLocal_find?] at h_s ⊢
            show ((s1.localReg.find? (fun p => p.fst = j)).map Prod.snd).isSome
            rw [h_lr]
            exact h_s
          exact IH (f + 1) frames { s1 with currentReg := [] } h_post_lab
            (show KeysNodup ({ s1 with currentReg := [] } : LowerState).localReg by
              show KeysNodup s1.localReg; rw [h_lr]; exact hnd) h_seed_post

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
  | @backedge_tail_cons pref post h_pref h_ht_pref h_post_wf IH =>
      have h_depth : (KernelInstrsW2.backedge_tail_cons h_pref h_ht_pref h_post_wf).depth
          = 1 + h_post_wf.depth := rfl
      rw [h_depth] at h_fuel
      obtain ⟨f, h_f⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by omega⟩
      subst h_f
      -- The WASM trace out of the skeleton.
      obtain ⟨n, entries, bodyOuts, h_e0, h_step, h_cont, h_nh_all, h_exit, h_post_eval,
              h_bound⟩ :=
        evalInstrs_block_wloop_trace_of_eval h_no_branch h_no_halt
          (backedgeTailBody_noStructured h_pref) (by trivial)
          (backedgeTailBody_continuesOrExits1 h_pref) hw
      -- The lowering, taken apart.
      obtain ⟨s1, flag, bodyOps, postOps, h_lb, hlp, h_ops⟩ :=
        blockBackedgeTail_lowerP h_pref hl
      obtain ⟨h_flag_fresh, _⟩ := backedgeTailBody_flag_fresh h_pref h_lb
      -- The side condition at this loop.
      have h_st' := h_st
      simp only [KernelInstrsW2.stable] at h_st'
      obtain ⟨h_lt, h_st_post⟩ := h_st' s1 _ bodyOps h_lb
      -- The IR trace, from the state after the flag's declaration.
      have R0 : Refines (entries 0) { s with currentReg := [] }
          { kst with rf := regWrite kst.rf flag (vBool false) } layout := by
        rw [h_e0]
        exact R.clear_current.regWrite_fresh h_flag_fresh _
      obtain ⟨kstStates, F_b, h_kst_start, h_ir_step, h_ir_cont, h_ir_exit, h_exit_ref,
              h_flag_exit, _⟩ :=
        backedgeTailBody_ir_trace f frames pref h_pref h_ht_pref
          { s with currentReg := [] } s1 flag bodyOps h_lb rfl h_lt layout
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
      exact preservation_blockLoop_nIterExit f frames s kst layout h_kst_no_broke
        (pref ++ [.brIf 0] ++ [.br 1]) post (backedgeTailBody_noStructured h_pref)
        n bodyOuts (h_nh_all (Fin.last n)) ws' h_post_eval
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

/-- The apex under seeding: the entry state binds every local the
    kernel writes (production's function-entry pre-allocation), so the
    side conditions reduce to the label-only face — `body2` may rebind
    any number of locals past the exit site. -/
theorem framework_preservation_kernel_while2_seeded
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
    (h_lab : h_wf.labelStable (fuel + 1) frames s)
    (hnd : KeysNodup s.localReg)
    (h_seed : LocalsSeeded s instrs)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs (fuel + 1) ws instrs = some ws')
    (hl : lowerInstrsP (fuel + 1) frames ⟨s, []⟩ instrs = some (⟨s', []⟩, ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  framework_preservation_kernel_while2 fuel frames ws s kst layout R h_no_branch
    h_no_halt h_kst_no_broke h_buf_locals h_no_buf_stack h_load_bounds h_store_bounds
    h_store_layout instrs h_wf h_fuel
    (h_wf.stable_of_seeded (fuel + 1) frames s h_lab hnd h_seed)
    ws' s' ops hw hl

/-- A TWO-local while — `i = 0; acc = 0; while i < n { acc += i; i += 1 }`
    — typechecks, and its seeding hypothesis is a concrete map: under the
    old list-level side condition the `local.set 3` in `body2` PERMUTED
    the assoc list whenever local 2 sat at its head, so this kernel was
    out of reach; upsert + seeding admit it. -/
example : KernelInstrsW2
    [.i32Const 0, .localSet 2, .i32Const 0, .localSet 3,
     .block 0, .wloop 0,
       .localGet 2, .localGet 1, .i32GeU, .brIf 1,
       .localGet 3, .localGet 2, .i32Add, .localSet 3,
       .localGet 2, .i32Const 1, .i32Add, .localSet 2,
       .br 0,
     .wend, .wend] :=
  .sl_cons trivial (.sl_cons trivial (.sl_cons trivial (.sl_cons trivial
    (.block_while_cons (pref := [.localGet 2, .localGet 1, .i32GeU])
      (body2 := [.localGet 3, .localGet 2, .i32Add, .localSet 3,
                 .localGet 2, .i32Const 1, .i32Add, .localSet 2]) (post := [])
      (by simp [StraightLineInstrs, StraightLineInstr])
      (by simp [StraightLineInstrs, StraightLineInstr])
      rfl rfl .empty))))

/-- Its written locals are exactly the two counters, so any entry state
    binding locals 2 and 3 is seeded. -/
example :
    writtenLocals
      [.i32Const 0, .localSet 2, .i32Const 0, .localSet 3,
       .block 0, .wloop 0,
         .localGet 2, .localGet 1, .i32GeU, .brIf 1,
         .localGet 3, .localGet 2, .i32Add, .localSet 3,
         .localGet 2, .i32Const 1, .i32Add, .localSet 2,
         .br 0,
       .wend, .wend] = [2, 3, 3, 2] := by
  simp [writtenLocals, writtenLocalsInstr]

/-- The function-level apex: the entry seed stream, then the kernel.
    Every hypothesis is checkable at the function boundary — the
    declared locals (all `.u32`, WASM zero-initialises them) cover the
    kernel's written locals, the entry state is a function entry
    (empty stack, empty per-frame map, duplicate-free params), and the
    label-only conditions hold at the seeded state. The op stream is
    the seed `Const`s followed by the kernel's — production's function
    body modulo the hoisted per-write declarations
    (`lower_entry_seed.rs`). -/
theorem framework_preservation_kernel_fn
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
    (decls : List (Nat × Quanta.KOps.Scalar))
    (h_tys : ∀ p ∈ decls, p.snd = Quanta.KOps.Scalar.u32)
    (h_zero : ∀ p ∈ decls, ws.locals.get? p.fst = some (.wI32 0))
    (h_stack : s.stack = []) (h_creg : s.currentReg = [])
    (hnd : KeysNodup s.localReg)
    (instrs : List WasmInstr)
    (h_wf : KernelInstrsW2 instrs)
    (h_fuel : fuel ≥ 2 + h_wf.depth)
    (h_lab : h_wf.labelStable (fuel + 1) frames (seedLocals decls s).1)
    (h_written : ∀ j ∈ writtenLocals instrs,
        (∃ ty, (j, ty) ∈ decls) ∨ (s.lookupLocal j).isSome)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs (fuel + 1) ws instrs = some ws')
    (hl : lowerInstrsP (fuel + 1) frames ⟨(seedLocals decls s).1, []⟩ instrs
        = some (⟨s', []⟩, ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ((seedLocals decls s).2 ++ ops) = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  obtain ⟨kst_seed, h_seed_eval, R_seed, h_b_seed, _⟩ :=
    seedLocals_refines decls ws s kst layout R h_tys h_zero h_stack h_creg
      h_kst_no_broke
  obtain ⟨kst', F, h_eval, R', h_bridge⟩ :=
    framework_preservation_kernel_while2_seeded fuel frames ws (seedLocals decls s).1
      kst_seed layout R_seed h_no_branch h_no_halt h_b_seed
      h_buf_locals h_no_buf_stack h_load_bounds h_store_bounds h_store_layout
      instrs h_wf h_fuel h_lab
      (seedLocals_keysNodup decls s hnd)
      (seedLocals_seeded decls s instrs h_written)
      ws' s' ops hw hl
  refine ⟨kst', F, ?_, R', h_bridge⟩
  rw [evalOps_append (h_seed_eval F) h_b_seed]
  exact h_eval

-- ════════════════════════════════════════════════════════════════════
-- A computable discharge of the label condition
-- ════════════════════════════════════════════════════════════════════

/-- `labelStable`, computed: run the very lowerings the condition
    quantifies over and compare the labels. A lowering that fails makes
    the condition vacuous, so the check says `true` there. -/
def KernelInstrsW2.labelStableCheck : ∀ {instrs : List WasmInstr},
    KernelInstrsW2 instrs → Nat → List FrameKind → LowerState → Bool
  | _, .empty, _, _, _ => true
  | _, @sl_cons i _ _ rest_wf, fuel, frames, s =>
      match lowerInstr s i with
      | none => true
      | some (s1, _) => rest_wf.labelStableCheck fuel frames s1
  | _, @while_cons _ body _ _ _ post_wf, fuel, frames, s =>
      match fuel with
      | 0 => true
      | f + 1 =>
          match lowerInstrs f (.loopK :: frames) { s with currentReg := [] } body with
          | none => true
          | some (s1, _) =>
              decide (s1.localTy = s.localTy) &&
              post_wf.labelStableCheck f frames { s1 with currentReg := [] }
  | _, @block_while_cons pref body2 _ _ _ _ _ post_wf, fuel, frames, s =>
      match fuel with
      | 0 => true
      | 1 => true
      | f + 2 =>
          match lowerInstrsP f (.loopK :: .block :: frames) ⟨{ s with currentReg := [] }, []⟩
                  (pref ++ [.brIf 1]),
                lowerInstrsP f (.loopK :: .block :: frames) ⟨{ s with currentReg := [] }, []⟩
                  (pref ++ [.brIf 1] ++ body2 ++ [.br 0]) with
          | some (⟨s_site, _⟩, _), some (⟨s1, _⟩, _) =>
              decide (s1.localTy = s.localTy) && decide (s1.localTy = s_site.localTy) &&
              post_wf.labelStableCheck (f + 1) frames { s1 with currentReg := [] }
          | _, _ => true
  | _, @backedge_tail_cons pref _ _ _ post_wf, fuel, frames, s =>
      match fuel with
      | 0 => true
      | 1 => true
      | f + 2 =>
          match lowerInstrsP f (.loopK :: .block :: frames) ⟨{ s with currentReg := [] }, []⟩
                  (pref ++ [.brIf 0] ++ [.br 1]) with
          | some (⟨s1, _⟩, _) =>
              decide (s1.localTy = s.localTy) &&
              post_wf.labelStableCheck (f + 1) frames { s1 with currentReg := [] }
          | none => true

/-- The check is sound: the lowering is a function, so the state the
    condition quantifies over is the one the check computed. -/
theorem KernelInstrsW2.labelStable_of_check :
    ∀ {instrs : List WasmInstr} (wf : KernelInstrsW2 instrs)
      (fuel : Nat) (frames : List FrameKind) (s : LowerState),
    wf.labelStableCheck fuel frames s = true → wf.labelStable fuel frames s := by
  intro instrs wf
  induction wf with
  | empty => intro _ _ _ _; trivial
  | @sl_cons i rest h_i rest_wf IH =>
      intro fuel frames s h
      intro s1 ops hl
      simp only [KernelInstrsW2.labelStableCheck, hl] at h
      exact IH fuel frames s1 h
  | @while_cons rest body post h_split h_body post_wf IH =>
      intro fuel frames s h
      cases fuel with
      | zero => trivial
      | succ f =>
          intro s1 bodyOps hl
          simp only [KernelInstrsW2.labelStableCheck, hl, Bool.and_eq_true,
                     decide_eq_true_eq] at h
          exact ⟨h.1, IH f frames _ h.2⟩
  | @block_while_cons pref body2 post h_pref h_body2 h_ht_pref h_ht_body2 post_wf IH =>
      intro fuel frames s h
      cases fuel with
      | zero => trivial
      | succ f0 =>
      cases f0 with
      | zero => trivial
      | succ f =>
          intro s_site p1 ops1 hl_site s1 p2 bodyOps h_lb
          simp only [KernelInstrsW2.labelStableCheck, hl_site, h_lb, Bool.and_eq_true,
                     decide_eq_true_eq] at h
          exact ⟨h.1.1, h.1.2, IH (f + 1) frames _ h.2⟩
  | @backedge_tail_cons pref post h_pref h_ht_pref post_wf IH =>
      intro fuel frames s h
      cases fuel with
      | zero => trivial
      | succ f0 =>
      cases f0 with
      | zero => trivial
      | succ f =>
          intro s1 p2 bodyOps h_lb
          simp only [KernelInstrsW2.labelStableCheck, h_lb, Bool.and_eq_true,
                     decide_eq_true_eq] at h
          exact ⟨h.1, IH (f + 1) frames _ h.2⟩

-- ════════════════════════════════════════════════════════════════════
-- End to end on a concrete kernel
-- ════════════════════════════════════════════════════════════════════

/-- The two-local witness as data. -/
def sumKernel : List WasmInstr :=
  [.i32Const 0, .localSet 2, .i32Const 0, .localSet 3,
   .block 0, .wloop 0,
     .localGet 2, .localGet 1, .i32GeU, .brIf 1,
     .localGet 3, .localGet 2, .i32Add, .localSet 3,
     .localGet 2, .i32Const 1, .i32Add, .localSet 2,
     .br 0,
   .wend, .wend]

def sumKernelWf : KernelInstrsW2 sumKernel :=
  .sl_cons trivial (.sl_cons trivial (.sl_cons trivial (.sl_cons trivial
    (.block_while_cons (pref := [.localGet 2, .localGet 1, .i32GeU])
      (body2 := [.localGet 3, .localGet 2, .i32Add, .localSet 3,
                 .localGet 2, .i32Const 1, .i32Add, .localSet 2]) (post := [])
      (by simp [StraightLineInstrs, StraightLineInstr])
      (by simp [StraightLineInstrs, StraightLineInstr])
      rfl rfl .empty))))

/-- Its function entry: the scalar param `n` at register 0; locals 2
    and 3 are the declared ones. -/
def sumEntry : LowerState :=
  { LowerState.empty with nextReg := 1, localReg := [(1, 0)], localTy := [(1, .u32)] }

def sumDecls : List (Nat × Quanta.KOps.Scalar) := [(2, .u32), (3, .u32)]

/-- The label condition of the witness holds — by running the
    lowerings. -/
example : sumKernelWf.labelStableCheck 4 [] (seedLocals sumDecls sumEntry).1 = true := by
  native_decide

/-- The function-level apex INSTANTIATES on the witness: every
    syntactic hypothesis is discharged by computation (the label
    check, the declared-local coverage, the entry shape); what remains
    is the semantic setting — a refined entry with zeroed locals, the
    buffer bundles, and the two evaluations. -/
example
    (ws : WasmState) (kst : Quanta.KOps.State) (layout : BufferLayout)
    (R : Refines ws sumEntry kst layout)
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
    (h_zero2 : ws.locals.get? 2 = some (.wI32 0))
    (h_zero3 : ws.locals.get? 3 = some (.wI32 0))
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs 4 ws sumKernel = some ws')
    (hl : lowerInstrsP 4 [] ⟨(seedLocals sumDecls sumEntry).1, []⟩ sumKernel
        = some (⟨s', []⟩, ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ((seedLocals sumDecls sumEntry).2 ++ ops) = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' :=
  framework_preservation_kernel_fn 3 [] ws sumEntry kst layout R h_no_branch h_no_halt
    h_kst_no_broke h_buf_locals h_no_buf_stack h_load_bounds h_store_bounds h_store_layout
    sumDecls
    (by intro p hp; simp [sumDecls] at hp; rcases hp with rfl | rfl <;> rfl)
    (by intro p hp; simp [sumDecls] at hp; rcases hp with rfl | rfl
        · exact h_zero2
        · exact h_zero3)
    rfl rfl (by show ((sumEntry.localReg).map Prod.fst).Nodup; simp [sumEntry])
    sumKernel sumKernelWf (by decide)
    (KernelInstrsW2.labelStable_of_check _ _ _ _ (by native_decide))
    (by intro j hj
        have h_all : ∀ j ∈ writtenLocals sumKernel,
            (j, Quanta.KOps.Scalar.u32) ∈ sumDecls := by native_decide
        exact Or.inl ⟨.u32, h_all j hj⟩)
    ws' s' ops hw hl

/-- The backedge + exit-tail witness: `i = 0; loop { i += 1; if i < n
    { continue } else { break } }` — rustc's `block { loop { …; br_if 0;
    br 1 } }`. The body computes the continue condition AFTER its
    writes, so the rotated shape carries no register condition at all. -/
def tailKernel : List WasmInstr :=
  [.i32Const 0, .localSet 2,
   .block 0, .wloop 0,
     .localGet 2, .i32Const 1, .i32Add, .localSet 2,
     .localGet 2, .localGet 1, .i32LtU,
     .brIf 0,
     .br 1,
   .wend, .wend]

def tailKernelWf : KernelInstrsW2 tailKernel :=
  .sl_cons trivial (.sl_cons trivial
    (.backedge_tail_cons
      (pref := [.localGet 2, .i32Const 1, .i32Add, .localSet 2,
                .localGet 2, .localGet 1, .i32LtU]) (post := [])
      (by simp [StraightLineInstrs, StraightLineInstr])
      rfl .empty))

/-- Its depth is one. -/
example : tailKernelWf.depth = 1 := rfl

/-- Its label condition holds — by running the lowerings from the
    seeded entry (local 2 declared). -/
example : tailKernelWf.labelStableCheck 4 [] (seedLocals [(2, .u32)] sumEntry).1 = true := by
  native_decide

/-- `KernelOp` is a nested inductive; the pin compares through `Repr`. -/
private def tailPinEq {α : Type} [Repr α] (a b : α) : Bool :=
  toString (repr a) == toString (repr b)

/-- The model's ops for the witness, op for op: the seeded entry binds
    local 2 at register 1; the kernel's const init, then the flag's
    declaration, the loop op — prefix, commit, cast, the backedge
    branch with the exit (flag := true, Break) in its else arm — and the
    no-op wrap of the empty block tail. The shape
    `lower_backedge_exit_tail.rs` pins on production. -/
example :
    tailPinEq
      (lowerInstrsP 4 [] ⟨(seedLocals [(2, .u32)] sumEntry).1, []⟩ tailKernel)
      (some (⟨{ LowerState.empty with nextReg := 14,
                                      localReg := [(2, 1), (1, 0)],
                                      localTy := [(2, .i32), (1, .u32)] }, []⟩,
        [.const 2 (.i32 0), .const 3 (.i32 0), .copy 3 2, .copy 1 3,
         .const 13 (.bool false),
         .loopOp
           [.copy 4 1, .const 5 (.i32 1), .binOp 6 4 5 .add .i32,
            .const 7 (.i32 0), .copy 7 6, .copy 1 7,
            .copy 8 7, .copy 9 0, .cmp 10 8 9 .lt .bool,
            .cast 11 10 .bool .u32, .cast 12 11 .u32 .bool,
            .branch 12 [] [.const 13 (.bool true), .breakOp]],
         .branch 13 [] []])) = true := by
  native_decide

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
