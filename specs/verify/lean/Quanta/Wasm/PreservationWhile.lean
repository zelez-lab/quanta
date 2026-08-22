/-
# L11 — while-loops with straight-line bodies

The L10v7 apex admits `wloop 0` segments whose body is IR-empty and exits
on its first iteration. This file is the composition that admits a real
while-loop: a straight-line body computing a condition, closed by
`brIf 0`, running any number of iterations.

Three pieces, composed at the end:

1. `iterLoop_trace_of_eval` — from the wloop arm's `iterLoop` returning
   `some ws'`, extract the iteration trace the N-iteration theorem
   (`preservation_evalInstrs_cons_wloop_nIterExit`) takes as input:
   entry states, body-out states, the continue/exit facts, and the
   iteration bound.
2. (next) per-iteration evidence for a `pref ++ [.brIf 0]` body.
3. (next) the `KernelInstrs` constructor and the apex arm.

See `roadmap/in_progress/059_source_to_ir_proof/L11_while_loops.md`.
-/

import Quanta.Wasm.PreservationBridge
import Quanta.Wasm.PreservationInduction
import Quanta.Wasm.LowerScopeValid

namespace Quanta.Wasm

open Quanta.KOps (KernelOp evalOps)

/-- A body whose runs never branch past the enclosing loop: every body
    evaluation leaves `branchTarget` either cleared (fall through = loop
    exit) or `some 0` (continue). Straight-line prefixes closed by
    `brIf 0` satisfy this — `brIf 0` is the only branch and it targets
    depth 0 — which is what keeps the `some (n+1)` arm of `iterLoop`
    unreachable. -/
def BodyBranchesAtMostZero (fuel : Nat) (body : List WasmInstr) : Prop :=
  ∀ (st st' : WasmState), st.branchTarget = none →
    evalInstrs fuel st body = some st' →
    st'.branchTarget = none ∨ st'.branchTarget = some 0

/-- Piece 1 — the iteration trace exists whenever `iterLoop` returns.

    Inducts on the iteration counter `f`, peeling one body run per
    step. `n` is the number of continues; the body runs `n + 1` times;
    the trace is presented exactly as `cons_wloop_nIterExit` consumes it
    (`entries 0` is the loop entry, `entries (i+1)` is body-out `i` with
    the continue target cleared), plus `n + 1 ≤ f`, which is the bound
    `iterLoop_n_iter_exit` needs to run it back. -/
theorem iterLoop_trace_of_eval
    {fuel : Nat} {body post : List WasmInstr}
    (h_body : BodyBranchesAtMostZero fuel body)
    {f : Nat} {st0 ws' : WasmState}
    (h_nb : st0.branchTarget = none)
    (h_iter : evalInstrs.iterLoop fuel body post f st0 = some ws') :
    ∃ (n : Nat) (entries bodyOuts : Fin (n + 1) → WasmState),
      entries 0 = st0 ∧
      (∀ i : Fin (n + 1), evalInstrs fuel (entries i) body = some (bodyOuts i)) ∧
      (∀ i : Fin n,
        (bodyOuts i.castSucc).branchTarget = some 0 ∧
        entries i.succ = { bodyOuts i.castSucc with branchTarget := none }) ∧
      (bodyOuts (Fin.last n)).branchTarget = none ∧
      evalInstrs fuel (bodyOuts (Fin.last n)) post = some ws' ∧
      n + 1 ≤ f := by
  induction f generalizing st0 with
  | zero =>
      unfold evalInstrs.iterLoop at h_iter
      simp at h_iter
  | succ k IH =>
      unfold evalInstrs.iterLoop at h_iter
      cases h_step0 : evalInstrs fuel st0 body with
      | none => rw [h_step0] at h_iter; simp at h_iter
      | some st1 =>
        rw [h_step0] at h_iter
        simp only at h_iter
        cases h_bt : st1.branchTarget with
        | none =>
            -- Exit on the first body run: n = 0.
            rw [h_bt] at h_iter
            simp only at h_iter
            refine ⟨0, fun _ => st0, fun _ => st1, rfl, ?_, ?_, ?_, ?_, ?_⟩
            · intro i; exact h_step0
            · intro i; exact absurd i.isLt (by simp)
            · exact h_bt
            · exact h_iter
            · omega
        | some d =>
            cases d with
            | zero =>
                -- Continue: the rest of the trace comes from the IH on k.
                rw [h_bt] at h_iter
                simp only at h_iter
                obtain ⟨n', entries', bodyOuts', h_e0, h_step', h_cont', h_exit', h_post', h_bound'⟩ :=
                  IH rfl h_iter
                refine ⟨n' + 1,
                  fun i => if h : i.val = 0 then st0 else entries' ⟨i.val - 1, by omega⟩,
                  fun i => if h : i.val = 0 then st1 else bodyOuts' ⟨i.val - 1, by omega⟩,
                  by simp, ?_, ?_, ?_, ?_, ?_⟩
                · intro i
                  by_cases h : i.val = 0
                  · simp only [h, ↓reduceDIte]; exact h_step0
                  · simp only [h, ↓reduceDIte]; exact h_step' _
                · intro i
                  by_cases h : i.val = 0
                  · -- First continue: body-out 0 = st1 with branchTarget some 0;
                    -- entry 1 = entries' 0 = st1 cleared.
                    have h_cs : (i.castSucc).val = 0 := by simp [h]
                    have h_sc : (i.succ).val = 1 := by simp [h]
                    simp only [h_cs, h_sc, ↓reduceDIte, Nat.one_ne_zero, Nat.sub_self]
                    refine ⟨h_bt, ?_⟩
                    have : (⟨0, by omega⟩ : Fin (n' + 1)) = 0 := rfl
                    rw [this, h_e0]
                  · have h_cs : (i.castSucc).val = i.val := by simp
                    have h_sc : (i.succ).val = i.val + 1 := by simp
                    have h_ne : i.val + 1 ≠ 0 := by omega
                    simp only [h_cs, h_sc, h, ↓reduceDIte, h_ne, Nat.add_sub_cancel]
                    obtain ⟨hc1, hc2⟩ := h_cont' ⟨i.val - 1, by omega⟩
                    simp only [Fin.castSucc_mk, Fin.succ_mk] at hc1 hc2
                    have h_idx : i.val - 1 + 1 = i.val := by omega
                    refine ⟨hc1, ?_⟩
                    have h_fin : (⟨i.val - 1 + 1, by omega⟩ : Fin (n' + 1)) = ⟨i.val, by omega⟩ :=
                      Fin.ext h_idx
                    rw [h_fin] at hc2
                    exact hc2
                · have h_last : (Fin.last (n' + 1)).val = n' + 1 := by simp
                  simp only [h_last, Nat.succ_ne_zero, ↓reduceDIte, Nat.add_sub_cancel]
                  have : (⟨n', by omega⟩ : Fin (n' + 1)) = Fin.last n' := rfl
                  rw [this]; exact h_exit'
                · have h_last : (Fin.last (n' + 1)).val = n' + 1 := by simp
                  simp only [h_last, Nat.succ_ne_zero, ↓reduceDIte, Nat.add_sub_cancel]
                  have : (⟨n', by omega⟩ : Fin (n' + 1)) = Fin.last n' := rfl
                  rw [this]; exact h_post'
                · omega
            | succ m =>
                -- A branch past the loop — excluded by the body shape.
                exfalso
                rcases h_body st0 st1 h_nb h_step0 with h0 | h0 <;> rw [h_bt] at h0 <;> simp at h0


-- ════════════════════════════════════════════════════════════════════
-- Piece 2a — straight-line prefixes keep the control flags and split
-- off the front of a list on both sides.
-- ════════════════════════════════════════════════════════════════════

/-- `pop` only touches the stack. -/
theorem pop_some_flags {s s1 : WasmState} {v : WasmValue}
    (h : s.pop = some (v, s1)) :
    s1.branchTarget = s.branchTarget ∧ s1.halted = s.halted := by
  unfold WasmState.pop at h
  cases hs : s.stack with
  | nil => rw [hs] at h; simp at h
  | cons x rs =>
      rw [hs] at h
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨_, h2⟩ := h
      rw [← h2]
      exact ⟨rfl, rfl⟩

/-- `setLocal` only touches the locals. -/
theorem setLocal_some_flags {s s1 : WasmState} {i : Nat} {v : WasmValue}
    (h : s.setLocal i v = some s1) :
    s1.branchTarget = s.branchTarget ∧ s1.halted = s.halted := by
  unfold WasmState.setLocal at h
  split at h
  · simp only [Option.some.injEq] at h
    rw [← h]; exact ⟨rfl, rfl⟩
  · simp at h

/-- Every `StraightLineInstr` leaves `branchTarget` and `halted` as it
    found them: none of them branch, return, or trap in a way that
    sets a flag (a stuck step is `none`, not a flag). -/
theorem evalInstr_straightLine_flags
    {s s' : WasmState} {i : WasmInstr}
    (h_sl : StraightLineInstr i)
    (h : evalInstr s i = some s') :
    s'.branchTarget = s.branchTarget ∧ s'.halted = s.halted := by
  cases i with
  | nop =>
      simp [evalInstr] at h
      refine ⟨?_, ?_⟩ <;> rw [← h]
  | drop =>
      simp only [evalInstr] at h
      cases hp : s.pop with
      | none => rw [hp] at h; simp at h
      | some vs =>
          rw [hp] at h
          obtain ⟨v, s1⟩ := vs
          simp at h
          rw [← h]
          exact pop_some_flags hp
  | i32Const n =>
      simp [evalInstr, WasmState.push] at h
      refine ⟨?_, ?_⟩ <;> rw [← h]
  | localGet idx =>
      simp only [evalInstr] at h
      cases hl : s.getLocal idx with
      | none => rw [hl] at h; simp at h
      | some v =>
          rw [hl] at h
          simp [WasmState.push] at h
          refine ⟨?_, ?_⟩ <;> rw [← h]
  | localSet idx =>
      simp only [evalInstr] at h
      cases hp : s.pop with
      | none => rw [hp] at h; simp at h
      | some vs =>
          rw [hp] at h
          obtain ⟨v, s1⟩ := vs
          simp at h
          obtain ⟨h1, h2⟩ := pop_some_flags hp
          obtain ⟨h3, h4⟩ := setLocal_some_flags h
          exact ⟨h3.trans h1, h4.trans h2⟩
  | localTee idx =>
      simp only [evalInstr] at h
      cases hp : s.pop with
      | none => rw [hp] at h; simp at h
      | some vs =>
          rw [hp] at h
          obtain ⟨v, s1⟩ := vs
          simp at h
          cases hs : s1.setLocal idx v with
          | none => rw [hs] at h; simp at h
          | some s2 =>
              rw [hs] at h
              simp [WasmState.push] at h
              obtain ⟨h1, h2⟩ := pop_some_flags hp
              obtain ⟨h3, h4⟩ := setLocal_some_flags hs
              rw [← h]
              exact ⟨h3.trans h1, h4.trans h2⟩
  | i32Add => exact binI32_preserves_branchTarget h
  | i32Sub => exact binI32_preserves_branchTarget h
  | i32Mul => exact binI32_preserves_branchTarget h
  | i32And => exact binI32_preserves_branchTarget h
  | i32Or => exact binI32_preserves_branchTarget h
  | i32Xor => exact binI32_preserves_branchTarget h
  | i32Shl => exact binI32_preserves_branchTarget h
  | i32ShrU => exact binI32_preserves_branchTarget h
  | i32DivU => exact binI32_preserves_branchTarget h
  | i32RemU => exact binI32_preserves_branchTarget h
  | i32Eq => exact cmpI32_preserves_branchTarget h
  | i32Ne => exact cmpI32_preserves_branchTarget h
  | i32LtU => exact cmpI32_preserves_branchTarget h
  | i32LeU => exact cmpI32_preserves_branchTarget h
  | i32GtU => exact cmpI32_preserves_branchTarget h
  | i32GeU => exact cmpI32_preserves_branchTarget h
  | i32Load offset align =>
      simp only [evalInstr, loadI32] at h
      cases hp : s.pop with
      | none => rw [hp] at h; simp at h
      | some vs =>
          rw [hp] at h
          obtain ⟨va, s1⟩ := vs
          simp at h
          obtain ⟨h1, h2⟩ := pop_some_flags hp
          cases va with
          | wI32 a =>
              simp at h
              cases hm : s1.mem.load_u32 (a.toNat + offset) with
              | none => rw [hm] at h; simp at h
              | some v =>
                  rw [hm] at h
                  simp [WasmState.push] at h
                  rw [← h]
                  exact ⟨h1, h2⟩
          | _ => simp at h
  | i32Store offset align =>
      simp only [evalInstr, storeI32] at h
      cases hp : s.pop with
      | none => rw [hp] at h; simp at h
      | some vs =>
          rw [hp] at h
          obtain ⟨vv, s1⟩ := vs
          simp at h
          obtain ⟨h1, h2⟩ := pop_some_flags hp
          cases hp2 : s1.pop with
          | none => rw [hp2] at h; simp at h
          | some vs2 =>
              rw [hp2] at h
              obtain ⟨va, s2⟩ := vs2
              simp at h
              obtain ⟨h3, h4⟩ := pop_some_flags hp2
              cases va with
              | wI32 a =>
                  cases vv with
                  | wI32 v =>
                      simp at h
                      cases hm : s2.mem.store_u32 (a.toNat + offset) v with
                      | none => rw [hm] at h; simp at h
                      | some m' =>
                          rw [hm] at h
                          simp at h
                          rw [← h]
                          exact ⟨h3.trans h1, h4.trans h2⟩
                  | _ => simp at h
              | _ => simp at h
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
  | _ => exact absurd h_sl (by simp [StraightLineInstr])

/-- A `StraightLineInstr` takes the default (non-structured) arm of
    `evalInstrs` and `lowerInstrs`. -/
theorem straightLine_not_structured_eval {i : WasmInstr} (h : StraightLineInstr i) :
    isStructuredEval i = false := by
  cases i <;> first | rfl | exact absurd h (by simp [StraightLineInstr])

theorem straightLine_not_structured_lower {i : WasmInstr} (h : StraightLineInstr i) :
    isStructuredLower i = false := by
  cases i <;> first | rfl | exact absurd h (by simp [StraightLineInstr])

/-- Evaluation of a straight-line prefix followed by anything splits
    at the prefix boundary, and the boundary state keeps the clean
    flags the prefix started with. -/
theorem evalInstrs_straightLine_append
    {fuel : Nat} {pref rest : List WasmInstr}
    (h_sl : StraightLineInstrs pref)
    {ws ws' : WasmState}
    (h_nb : ws.branchTarget = none) (h_nh : ws.halted = false)
    (hw : evalInstrs fuel ws (pref ++ rest) = some ws') :
    ∃ ws_m : WasmState,
      evalInstrs fuel ws pref = some ws_m ∧
      ws_m.branchTarget = none ∧ ws_m.halted = false ∧
      evalInstrs fuel ws_m rest = some ws' := by
  induction pref generalizing ws with
  | nil =>
      refine ⟨ws, ?_, h_nb, h_nh, ?_⟩
      · simp [evalInstrs]
      · simpa using hw
  | cons i pref' IH =>
      obtain ⟨h_i, h_pref'⟩ := h_sl
      rw [List.cons_append] at hw
      rw [evalInstrs_cons_default fuel ws i (pref' ++ rest) h_nb h_nh
          (straightLine_not_structured_eval h_i)] at hw
      cases he : evalInstr ws i with
      | none => rw [he] at hw; simp at hw
      | some ws1 =>
          rw [he] at hw
          simp only at hw
          obtain ⟨hb1, hh1⟩ := evalInstr_straightLine_flags h_i he
          rw [h_nb] at hb1
          rw [h_nh] at hh1
          obtain ⟨ws_m, h_pref_eval, h_mb, h_mh, h_rest⟩ := IH h_pref' hb1 hh1 hw
          refine ⟨ws_m, ?_, h_mb, h_mh, h_rest⟩
          rw [evalInstrs_cons_default fuel ws i pref' h_nb h_nh
              (straightLine_not_structured_eval h_i)]
          rw [he]
          exact h_pref_eval

/-- Lowering of a straight-line prefix followed by anything splits at
    the prefix boundary; the emitted ops concatenate. -/
theorem lowerInstrs_straightLine_append
    {fuel : Nat} {frames : List FrameKind} {pref rest : List WasmInstr}
    (h_sl : StraightLineInstrs pref)
    {s s' : LowerState} {ops : List KernelOp}
    (hl : lowerInstrs fuel frames s (pref ++ rest) = some (s', ops)) :
    ∃ (s_m : LowerState) (ops1 ops2 : List KernelOp),
      lowerInstrs fuel frames s pref = some (s_m, ops1) ∧
      lowerInstrs fuel frames s_m rest = some (s', ops2) ∧
      ops = ops1 ++ ops2 := by
  induction pref generalizing s ops with
  | nil =>
      refine ⟨s, [], ops, ?_, ?_, rfl⟩
      · simp [lowerInstrs]
      · simpa using hl
  | cons i pref' IH =>
      obtain ⟨h_i, h_pref'⟩ := h_sl
      rw [List.cons_append] at hl
      rw [lowerInstrs_cons_default fuel frames s i (pref' ++ rest)
          (straightLine_not_structured_lower h_i)] at hl
      cases hli : lowerInstr s i with
      | none => rw [hli] at hl; simp at hl
      | some p1 =>
          rw [hli] at hl
          obtain ⟨s1, ops_i⟩ := p1
          simp only [Option.bind_eq_bind, Option.some_bind] at hl
          cases hlr : lowerInstrs fuel frames s1 (pref' ++ rest) with
          | none => rw [hlr] at hl; simp at hl
          | some p2 =>
              rw [hlr] at hl
              obtain ⟨s2, ops_r⟩ := p2
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


-- ════════════════════════════════════════════════════════════════════
-- Piece 2b — one iteration of a `pref ++ [.brIf 0]` body.
-- ════════════════════════════════════════════════════════════════════

/-- What the lowered `brIf 0` (inside a loop, empty tail) does to the IR
    `broke` flag: it is exactly the WASM exit decision. The ops are
    `opsCommit ++ [cast cond_bool cond u32→bool, branch cond_bool [] [breakOp]]`
    (`lowerInstrs_brIf0_loop_empty_tail`); the commit leaves the popped
    condition in `cond` (`brIf_cond_pop_commit_correct_pub`), the cast
    makes it a bool, and the branch either does nothing (continue) or
    runs `breakOp` (exit). -/
theorem brIf0_loop_ops_broke
    {fuel : Nat} {frames : List FrameKind}
    {ws : WasmState} {s : LowerState} {kst : Quanta.KOps.State}
    {layout : BufferLayout}
    (R : Refines ws s kst layout)
    (h_kst_ok : kst.broke = false)
    {c : UInt32} {rest_w : List WasmValue}
    (h_stack : ws.stack = .wI32 c :: rest_w)
    {s' : LowerState} {ops : List KernelOp}
    (hl : lowerInstrs fuel (.loopK :: frames) s [.brIf 0] = some (s', ops))
    {F : Nat} {kst' : Quanta.KOps.State}
    (h_ev : evalOps F kst ops = some kst') :
    kst'.broke = decide (c = 0) := by
  rw [lowerInstrs_brIf0_loop_empty_tail fuel (.loopK :: frames) s rfl] at hl
  rcases h_pop : s.popSym with _ | ⟨svCond, s0⟩
  · rw [h_pop] at hl; simp at hl
  rw [h_pop] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl
  rcases h_commit : s0.commit svCond with _ | ⟨cond, s1, opsCommit⟩
  · rw [h_commit] at hl; simp at hl
  rw [h_commit] at hl
  simp only [Option.some_bind, LowerState.alloc, pure, Option.some.injEq,
             Prod.mk.injEq] at hl
  obtain ⟨_, h_ops⟩ := hl
  subst h_ops
  obtain ⟨kst1, h_ev1, h_kst1_ok, h_lookup, _R1, _, _, _, _, _, _⟩ :=
    brIf_cond_pop_commit_correct_pub R h_stack h_pop h_commit h_kst_ok
  -- Run the commit ops at fuel F (mono from 0), then the two tail ops.
  have h_ev1_F : evalOps F kst opsCommit = some kst1 :=
    evalOps_fuel_mono (Nat.zero_le F) h_ev1
  rw [evalOps_append h_ev1_F h_kst1_ok] at h_ev
  -- The cast writes `vBool (c ≠ 0)` into `s1.nextReg`.
  have h_c_toNat : decide (c.toNat = 0) = decide (c = 0) := by
    by_cases hc : c = 0
    · subst hc; rfl
    · have : c.toNat ≠ 0 := by
        intro h
        exact hc (by simpa using congrArg UInt32.ofNat h)
      simp [hc, this]
  let kst2 : Quanta.KOps.State :=
    { kst1 with rf := Quanta.KOps.regWrite kst1.rf s1.nextReg
                        (Quanta.KOps.vBool (!decide (c = 0))) }
  have h_cast : Quanta.KOps.evalOp F kst1
      (KernelOp.cast s1.nextReg cond Quanta.KOps.Scalar.u32 Quanta.KOps.Scalar.bool)
      = some kst2 := by
    rcases h_lookup with h_u32 | h_i32
    · simp [Quanta.KOps.evalOp, h_u32, Quanta.KOps.evalCast, kst2]
    · simp [Quanta.KOps.evalOp, h_i32, Quanta.KOps.evalCast, kst2, h_c_toNat]
  simp only [Quanta.KOps.evalOps, Option.bind_eq_bind, h_cast, Option.some_bind] at h_ev
  -- After the cast, broke is still false; the branch reads the bool.
  have h_kst2_ok : kst2.broke = false := h_kst1_ok
  simp only [h_kst2_ok, Bool.false_eq_true, ↓reduceIte] at h_ev
  have h_lookup2 : Quanta.KOps.regLookup kst2.rf s1.nextReg
      = some (Quanta.KOps.vBool (!decide (c = 0))) :=
    regLookup_regWrite_self _ _ _
  simp only [Quanta.KOps.evalOps, Quanta.KOps.evalOp, Option.bind_eq_bind,
             h_lookup2, Option.some_bind] at h_ev
  by_cases hc : c = 0
  · -- exit: else arm `[breakOp]` runs, broke := true
    simp [hc, pure, Quanta.KOps.vBool] at h_ev
    rw [← h_ev]; simp [hc]
  · -- continue: then arm `[]`, state unchanged, broke = false
    simp [hc, Quanta.KOps.vBool] at h_ev
    rw [← h_ev, h_kst2_ok]; simp [hc]

/-- Piece 2b — one iteration of a while body `pref ++ [.brIf 0]`
    (straight-line `pref` computing the condition): the IR runs the
    lowered body to a state refining the WASM body-out, and the IR
    `broke` flag is the WASM exit decision — `branchTarget = some 0`
    (continue) ⇔ `broke = false`, `branchTarget = none` (exit) ⇔
    `broke = true`. This is the per-iteration evidence
    `cons_wloop_nIterExit` consumes. -/
theorem whileBody_iteration
    (fuel : Nat) (frames : List FrameKind)
    (pref : List WasmInstr) (h_pref : StraightLineInstrs pref)
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
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (pref ++ [.brIf 0]) = some ws')
    (hl : lowerInstrs fuel (.loopK :: frames) s (pref ++ [.brIf 0]) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      ws'.halted = false ∧
      ((ws'.branchTarget = some 0 ∧ kst'.broke = false) ∨
       (ws'.branchTarget = none ∧ kst'.broke = true)) := by
  -- Split both sides at the prefix boundary.
  obtain ⟨ws_m, hw_pref, h_mb, h_mh, hw_br⟩ :=
    evalInstrs_straightLine_append h_pref h_no_branch h_no_halt hw
  obtain ⟨s_m, ops1, ops2, hl_pref, hl_br, h_ops⟩ :=
    lowerInstrs_straightLine_append h_pref hl
  subst h_ops
  -- The straight-line prefix.
  obtain ⟨kst_m, F1, h_ev1, R_m, h_bridge_m⟩ :=
    framework_preservation_straightLine fuel (.loopK :: frames) ws s kst layout R
      h_no_branch h_no_halt h_kst_no_broke h_buf_locals h_no_buf_stack
      h_load_bounds h_store_bounds h_store_layout pref h_pref ws_m s_m ops1
      hw_pref hl_pref
  have h_kst_m_ok : kst_m.broke = false := h_bridge_m.right h_mb
  -- The brIf: Refines after, plus the popped condition and the outcome.
  obtain ⟨kst', F2, h_ev2, R', c, rest_w, h_stack, h_out⟩ :=
    preservation_evalInstrs_cons_brIf_loop_self_bridge fuel (.loopK :: frames)
      ws_m s_m kst_m layout R_m h_mb h_mh h_kst_m_ok rfl ws' s' ops2 hw_br hl_br
  have h_broke : kst'.broke = decide (c = 0) :=
    brIf0_loop_ops_broke R_m h_kst_m_ok h_stack hl_br h_ev2
  -- One fuel for both halves.
  refine ⟨kst', max F1 F2, ?_, R', ?_, ?_⟩
  · exact evalOps_append_fuel_mono_head (Nat.le_max_left F1 F2) h_ev1 h_kst_m_ok
      (evalOps_fuel_mono (Nat.le_max_right F1 F2) h_ev2)
  · rcases h_out with ⟨_, h_eq⟩ | ⟨_, h_eq⟩ <;> rw [h_eq] <;> exact h_mh
  · rcases h_out with ⟨hc, h_eq⟩ | ⟨hc, h_eq⟩
    · right
      refine ⟨by rw [h_eq]; exact h_mb, ?_⟩
      rw [h_broke, hc]; rfl
    · left
      refine ⟨by rw [h_eq], ?_⟩
      rw [h_broke]; simp [hc]


-- ════════════════════════════════════════════════════════════════════
-- Piece 3a — the lowering frame of a straight-line, non-local-writing
-- instruction: pops `k` symbolic slots, pushes `p`, leaves the rest of
-- the symbolic stack and every non-stack field alone (`nextReg` grows).
--
-- Why this matters for loops: the body is lowered ONCE but runs every
-- iteration, so the N-iteration theorem needs the lowering state after
-- the body to agree with the entry state on everything `Refines` reads
-- (stack, locals, bindings). A body that only reads locals and carries
-- its loop state through memory has exactly that property; a body
-- that writes a local does not (`localSet` rebinds `currentReg`), and
-- the wloop arm's snapshot/restore is where L12 has to do more work.
-- ════════════════════════════════════════════════════════════════════

/-- WASM's stack typing of the straight-line subset: `(pops, pushes)`. -/
def stackEffect : WasmInstr → Nat × Nat
  | .nop          => (0, 0)
  | .drop         => (1, 0)
  | .i32Const _   => (0, 1)
  | .localGet _   => (0, 1)
  | .localSet _   => (1, 0)
  | .localTee _   => (1, 1)
  | .i32Load _ _  => (1, 1)
  | .i32Store _ _ => (2, 0)
  | .i32Add | .i32Sub | .i32Mul | .i32And | .i32Or | .i32Xor
  | .i32Shl | .i32ShrU | .i32DivU | .i32RemU
  | .i32Eq | .i32Ne | .i32LtU | .i32LeU | .i32GtU | .i32GeU => (2, 1)
  | _ => (0, 0)

/-- An instruction that rebinds no local. -/
def NoLocalWrite : WasmInstr → Prop
  | .localSet _ => False
  | .localTee _ => False
  | _           => True

def NoLocalWrites : List WasmInstr → Prop
  | []        => True
  | i :: rest => NoLocalWrite i ∧ NoLocalWrites rest

/-- The symbolic-stack height after `instrs`, starting from `h`; `none`
    if some instruction would pop below the starting level. -/
def stackHeight : Nat → List WasmInstr → Option Nat
  | h, []        => some h
  | h, i :: rest =>
      let (k, p) := stackEffect i
      if k ≤ h then stackHeight (h - k + p) rest else none

/-- `s'` is `s` after popping `k` symbolic slots and pushing `p`. -/
def LowerFrame (s s' : LowerState) (k p : Nat) : Prop :=
  k ≤ s.stack.length ∧
  s'.stack.length = p + (s.stack.length - k) ∧
  s'.stack.drop p = s.stack.drop k ∧
  s'.localReg = s.localReg ∧ s'.localTy = s.localTy ∧
  s'.currentReg = s.currentReg ∧ s'.bufferSlots = s.bufferSlots ∧
  s.nextReg ≤ s'.nextReg

/-- A buffer-pattern arm: two slots rewritten into one, fields kept,
    `nextReg` bumped. The literal on the right is the shape `subst`
    leaves after `simp only [LowerState.alloc]`. -/
theorem LowerFrame.rewrite2_1 {s : LowerState} {a b c : SymVal} {rest : List SymVal}
    {nr : Nat} (hs : s.stack = a :: b :: rest) (h_nr : s.nextReg ≤ nr) :
    LowerFrame s { nextReg := nr, stack := c :: rest, localReg := s.localReg,
                   localTy := s.localTy, bufferSlots := s.bufferSlots,
                   currentReg := s.currentReg } 2 1 := by
  refine ⟨?_, ?_, ?_, rfl, rfl, rfl, rfl, h_nr⟩ <;> simp [hs] <;> omega

theorem LowerFrame.rewrite1_1 {s : LowerState} {a c : SymVal} {rest : List SymVal}
    {nr : Nat} (hs : s.stack = a :: rest) (h_nr : s.nextReg ≤ nr) :
    LowerFrame s { nextReg := nr, stack := c :: rest, localReg := s.localReg,
                   localTy := s.localTy, bufferSlots := s.bufferSlots,
                   currentReg := s.currentReg } 1 1 := by
  refine ⟨?_, ?_, ?_, rfl, rfl, rfl, rfl, h_nr⟩ <;> simp [hs] <;> omega

theorem LowerState.commit_localTy {s s' : LowerState} {sv : SymVal}
    {r : Quanta.KOps.Reg} {ops : List KernelOp}
    (h : s.commit sv = some (r, s', ops)) : s'.localTy = s.localTy := by
  unfold LowerState.commit at h
  cases sv with
  | reg r' _ => simp at h; obtain ⟨_, h_s_eq, _⟩ := h; rw [← h_s_eq]
  | i32ConstSym n => simp [LowerState.alloc] at h; obtain ⟨_, h_s_eq, _⟩ := h; rw [← h_s_eq]
  | bufferPtr _ => simp at h
  | scaledIdx _ _ => simp at h
  | bufferAccess _ _ _ => simp at h

/-- The generic binop frame, from `lowerI32Bin_some_shape`. -/
theorem lowerI32Bin_frame {s s' : LowerState} {bop : Quanta.KOps.BinOp}
    {ops : List KernelOp} (h : lowerI32Bin s bop = some (s', ops)) :
    LowerFrame s s' 2 1 := by
  obtain ⟨svb, sva, lrest, ra, s3, opsA, rb, s4, opsB, hs, _, _, _, _, _, h_nr, h_s', _⟩ :=
    lowerI32Bin_some_shape h
  subst h_s'
  refine ⟨?_, ?_, ?_, rfl, rfl, rfl, rfl, ?_⟩ <;> simp [hs] <;> omega

/-- The comparison frame, from `lowerI32Cmp_some_shape`. -/
theorem lowerI32Cmp_frame {s s' : LowerState} {cop : Quanta.KOps.CmpOp}
    {ops : List KernelOp} (h : lowerI32Cmp s cop = some (s', ops)) :
    LowerFrame s s' 2 1 := by
  obtain ⟨svb, sva, lrest, ra, s3, opsA, rb, s4, opsB, hs, _, _, _, _, _, h_nr, h_s', _⟩ :=
    lowerI32Cmp_some_shape h
  subst h_s'
  refine ⟨?_, ?_, ?_, rfl, rfl, rfl, rfl, ?_⟩ <;> simp [hs] <;> omega

/-- `lowerI32Add`: every buffer-pattern arm rewrites two slots into one
    and allocates; the fall-through is the generic binop. -/
theorem lowerI32Add_frame {s s' : LowerState} {ops : List KernelOp}
    (h : lowerI32Add s = some (s', ops)) : LowerFrame s s' 2 1 := by
  unfold lowerI32Add at h
  simp only [LowerState.alloc] at h
  repeat' split at h
  all_goals
    first
    | exact lowerI32Bin_frame h
    | (obtain ⟨h_s_eq, _⟩ :=
        Prod.mk.injEq _ _ _ _ |>.mp ((Option.some.injEq _ _).mp h)
       subst h_s_eq
       exact LowerFrame.rewrite2_1 (by assumption) (by omega))

theorem lowerI32Shl_frame {s s' : LowerState} {ops : List KernelOp}
    (h : lowerI32Shl s = some (s', ops)) : LowerFrame s s' 2 1 := by
  unfold lowerI32Shl at h
  split at h
  · obtain ⟨h_s_eq, _⟩ :=
      Prod.mk.injEq _ _ _ _ |>.mp ((Option.some.injEq _ _).mp h)
    subst h_s_eq
    exact LowerFrame.rewrite2_1 (by assumption) (Nat.le_refl _)
  · exact lowerI32Bin_frame h

theorem lowerI32Load_frame {s s' : LowerState} {ops : List KernelOp}
    (h : lowerI32Load s = some (s', ops)) : LowerFrame s s' 1 1 := by
  unfold lowerI32Load at h
  simp only [LowerState.alloc] at h
  split at h
  · obtain ⟨h_s_eq, _⟩ :=
      Prod.mk.injEq _ _ _ _ |>.mp ((Option.some.injEq _ _).mp h)
    subst h_s_eq
    exact LowerFrame.rewrite1_1 (by assumption) (by omega)
  · exact Option.noConfusion h

theorem lowerI32Store_frame {s s' : LowerState} {ops : List KernelOp}
    (h : lowerI32Store s = some (s', ops)) : LowerFrame s s' 2 0 := by
  unfold lowerI32Store at h
  rcases hs : s.stack with _ | ⟨sv_val, _ | ⟨sv_addr, lrest⟩⟩
  · simp [hs, LowerState.popSym] at h
  · simp [hs, LowerState.popSym] at h
  · simp only [hs, LowerState.popSym, Option.bind_eq_bind, Option.some_bind] at h
    rcases hc : ({ s with stack := lrest } : LowerState).commit sv_val
        with _ | ⟨src, s3, opsCommit⟩
    · simp [hc] at h
    · simp only [hc, Option.some_bind] at h
      split at h
      · simp only [pure, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨h_s_eq, _⟩ := h
        subst h_s_eq
        have h_stk := LowerState.commit_stack hc
        have h_lr := LowerState.commit_localReg hc
        have h_lt := LowerState.commit_localTy hc
        have h_cr := LowerState.commit_currentReg hc
        have h_bs := LowerState.commit_preserves_bufferSlots hc
        have h_nr := LowerState.commit_nextReg_mono hc
        simp only at h_stk h_lr h_lt h_cr h_bs h_nr
        refine ⟨?_, ?_, ?_, h_lr, h_lt, h_cr, h_bs, h_nr⟩ <;> simp [hs, h_stk]
      · exact Option.noConfusion h

/-- The frame of every straight-line, non-local-writing instruction. -/
theorem lowerInstr_frame {s s' : LowerState} {i : WasmInstr} {ops : List KernelOp}
    (h_sl : StraightLineInstr i) (h_nw : NoLocalWrite i)
    (h : lowerInstr s i = some (s', ops)) :
    LowerFrame s s' (stackEffect i).1 (stackEffect i).2 := by
  cases i with
  | nop =>
      simp only [lowerInstr, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨h_s_eq, _⟩ := h; subst h_s_eq
      exact ⟨by simp [stackEffect], by simp [stackEffect], by simp [stackEffect],
             rfl, rfl, rfl, rfl, Nat.le_refl _⟩
  | drop =>
      simp only [lowerInstr] at h
      rcases hs : s.stack with _ | ⟨sv, rs⟩
      · simp [hs, LowerState.popSym] at h
      · simp only [hs, LowerState.popSym, Option.bind_eq_bind, Option.some_bind, pure,
                   Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨h_s_eq, _⟩ := h; subst h_s_eq
        refine ⟨?_, ?_, ?_, rfl, rfl, rfl, rfl, Nat.le_refl _⟩ <;> simp [hs, stackEffect] <;> omega
  | i32Const n =>
      simp only [lowerInstr, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨h_s_eq, _⟩ := h; subst h_s_eq
      refine ⟨?_, ?_, ?_, rfl, rfl, rfl, rfl, Nat.le_refl _⟩ <;> simp [stackEffect] <;> omega
  | localGet i =>
      simp only [lowerInstr] at h
      rcases hbuf : s.lookupBufferSlot i with _ | slot
      · rw [hbuf] at h
        simp only [Option.bind_eq_bind] at h
        rcases hsrc : ((s.lookupCurrentReg i).orElse (fun _ => s.lookupLocal i)) with _ | src
        · simp [hsrc] at h
        · simp only [hsrc, Option.some_bind, LowerState.alloc, LowerState.pushSym, pure,
                     Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨h_s_eq, _⟩ := h; subst h_s_eq
          refine ⟨?_, ?_, ?_, rfl, rfl, rfl, rfl, ?_⟩ <;> simp [stackEffect] <;> omega
      · rw [hbuf] at h
        simp only [LowerState.pushSym, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨h_s_eq, _⟩ := h; subst h_s_eq
        refine ⟨?_, ?_, ?_, rfl, rfl, rfl, rfl, Nat.le_refl _⟩ <;> simp [stackEffect] <;> omega
  | localSet _ => exact absurd h_nw (by simp [NoLocalWrite])
  | localTee _ => exact absurd h_nw (by simp [NoLocalWrite])
  | i32Add => exact lowerI32Add_frame h
  | i32Sub => exact lowerI32Bin_frame h
  | i32Mul => exact lowerI32Bin_frame h
  | i32And => exact lowerI32Bin_frame h
  | i32Or => exact lowerI32Bin_frame h
  | i32Xor => exact lowerI32Bin_frame h
  | i32Shl => exact lowerI32Shl_frame h
  | i32ShrU => exact lowerI32Bin_frame h
  | i32DivU => exact lowerI32Bin_frame h
  | i32RemU => exact lowerI32Bin_frame h
  | i32Eq => exact lowerI32Cmp_frame h
  | i32Ne => exact lowerI32Cmp_frame h
  | i32LtU => exact lowerI32Cmp_frame h
  | i32LeU => exact lowerI32Cmp_frame h
  | i32GtU => exact lowerI32Cmp_frame h
  | i32GeU => exact lowerI32Cmp_frame h
  | i32Load _ _ => exact lowerI32Load_frame h
  | i32Store _ _ => exact lowerI32Store_frame h
  | _ => exact absurd h_sl (by simp [StraightLineInstr])


-- ════════════════════════════════════════════════════════════════════
-- Piece 3b — the while-body shape and what its lowering leaves behind.
-- ════════════════════════════════════════════════════════════════════

/-- The L11 body shape: a straight-line prefix that writes no local and
    computes exactly one value on top of the entry stack (the continue
    condition), closed by `brIf 0`. Loop-carried state lives in memory
    (`i32.store`/`i32.load` through buffer locals). -/
def WhileBody (body : List WasmInstr) : Prop :=
  ∃ pref : List WasmInstr,
    StraightLineInstrs pref ∧ NoLocalWrites pref ∧
    stackHeight 0 pref = some 1 ∧
    body = pref ++ [.brIf 0]

theorem LowerFrame.refl (s : LowerState) : LowerFrame s s 0 0 :=
  ⟨Nat.zero_le _, by simp, rfl, rfl, rfl, rfl, rfl, Nat.le_refl _⟩

/-- Frames compose along the height fold. -/
theorem LowerFrame.trans {s1 s2 s3 : LowerState} {h0 h1 k p : Nat}
    (f1 : LowerFrame s1 s2 h0 h1) (f2 : LowerFrame s2 s3 k p) (h_k : k ≤ h1) :
    LowerFrame s1 s3 h0 (h1 - k + p) := by
  obtain ⟨a1, a2, a3, a4, a5, a6, a7, a8⟩ := f1
  obtain ⟨b1, b2, b3, b4, b5, b6, b7, b8⟩ := f2
  refine ⟨a1, ?_, ?_, b4.trans a4, b5.trans a5, b6.trans a6, b7.trans a7, Nat.le_trans a8 b8⟩
  · omega
  · have h_eq : h1 - k + p = p + (h1 - k) := by omega
    have h_k' : k + (h1 - k) = h1 := by omega
    rw [h_eq, ← List.drop_drop, b3, List.drop_drop, h_k', a3]

/-- The lowering of a straight-line, non-local-writing list follows the
    height fold, measured from an anchor state `s0` the list never pops
    into. -/
theorem lowerInstrs_frame_from
    {fuel : Nat} {frames : List FrameKind} {pref : List WasmInstr}
    (h_sl : StraightLineInstrs pref) (h_nw : NoLocalWrites pref)
    {s0 : LowerState} {h0 h1 : Nat} (h_ht : stackHeight h0 pref = some h1)
    {s s' : LowerState} {ops : List KernelOp}
    (f0 : LowerFrame s0 s 0 h0)
    (hl : lowerInstrs fuel frames s pref = some (s', ops)) :
    LowerFrame s0 s' 0 h1 := by
  induction pref generalizing s h0 ops with
  | nil =>
      simp only [stackHeight, Option.some.injEq] at h_ht
      simp only [lowerInstrs, Option.some.injEq, Prod.mk.injEq] at hl
      obtain ⟨h_s, _⟩ := hl
      subst h_s; subst h_ht
      exact f0
  | cons i pref' IH =>
      obtain ⟨h_i, h_pref'⟩ := h_sl
      obtain ⟨h_nw_i, h_nw'⟩ := h_nw
      simp only [stackHeight] at h_ht
      split at h_ht
      · rename_i h_k
        rw [lowerInstrs_cons_default fuel frames s i pref'
            (straightLine_not_structured_lower h_i)] at hl
        cases hli : lowerInstr s i with
        | none => rw [hli] at hl; simp at hl
        | some p1 =>
            rw [hli] at hl
            obtain ⟨s1, ops_i⟩ := p1
            simp only [Option.bind_eq_bind, Option.some_bind] at hl
            cases hlr : lowerInstrs fuel frames s1 pref' with
            | none => rw [hlr] at hl; simp at hl
            | some p2 =>
                rw [hlr] at hl
                obtain ⟨s2, ops_r⟩ := p2
                simp only [Option.some_bind, pure, Option.some.injEq, Prod.mk.injEq] at hl
                obtain ⟨h_s, _⟩ := hl
                subst h_s
                have f1 := lowerInstr_frame h_i h_nw_i hli
                exact IH h_pref' h_nw' h_ht (LowerFrame.trans f0 f1 h_k) hlr
      · exact Option.noConfusion h_ht

/-- What the lowering of a `WhileBody` leaves: the entry stack, locals
    and bindings, a grown `nextReg` — the `h_body_lowering` clause of
    the N-iteration theorem. -/
theorem whileBody_lowering_frame
    {fuel : Nat} {frames : List FrameKind}
    {pref : List WasmInstr}
    (h_sl : StraightLineInstrs pref) (h_nw : NoLocalWrites pref)
    (h_ht : stackHeight 0 pref = some 1)
    {s s' : LowerState} {ops : List KernelOp}
    (hl : lowerInstrs fuel (.loopK :: frames) s (pref ++ [.brIf 0]) = some (s', ops)) :
    s'.localReg = s.localReg ∧ s'.localTy = s.localTy ∧
    s'.stack = s.stack ∧ s'.bufferSlots = s.bufferSlots ∧
    s'.currentReg = s.currentReg ∧
    s.nextReg ≤ s'.nextReg := by
  obtain ⟨s_m, ops1, ops2, hl_pref, hl_br, _⟩ :=
    lowerInstrs_straightLine_append h_sl hl
  obtain ⟨_, h_len, h_drop, h_lr, h_lt, h_cr, h_bs, h_nr⟩ :=
    lowerInstrs_frame_from h_sl h_nw h_ht (LowerFrame.refl s) hl_pref
  -- s_m.stack = cond :: s.stack
  rw [lowerInstrs_brIf0_loop_empty_tail fuel (.loopK :: frames) s_m rfl] at hl_br
  rcases h_pop : s_m.popSym with _ | ⟨svCond, s0⟩
  · rw [h_pop] at hl_br; simp at hl_br
  rw [h_pop] at hl_br
  simp only [Option.bind_eq_bind, Option.some_bind] at hl_br
  rcases h_commit : s0.commit svCond with _ | ⟨cond, s1, opsCommit⟩
  · rw [h_commit] at hl_br; simp at hl_br
  rw [h_commit] at hl_br
  simp only [Option.some_bind, LowerState.alloc, pure, Option.some.injEq,
             Prod.mk.injEq] at hl_br
  obtain ⟨h_s', _⟩ := hl_br
  subst h_s'
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
  have h_pop_cr := LowerState.popSym_currentReg h_pop
  have h_pop_lt : s0.localTy = s_m.localTy := by
    unfold LowerState.popSym at h_pop
    rcases hs : s_m.stack with _ | ⟨sv, rs⟩
    · rw [hs] at h_pop; simp at h_pop
    · rw [hs] at h_pop; simp at h_pop
      obtain ⟨_, h_eq⟩ := h_pop
      rw [← h_eq]
  have h_pop_bs := LowerState.popSym_preserves_bufferSlots h_pop
  have h_c_stk := LowerState.commit_stack h_commit
  have h_c_lr := LowerState.commit_localReg h_commit
  have h_c_lt := LowerState.commit_localTy h_commit
  have h_c_cr := LowerState.commit_currentReg h_commit
  have h_c_bs := LowerState.commit_preserves_bufferSlots h_commit
  have h_c_nr := LowerState.commit_nextReg_mono h_commit
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> simp only
  · rw [h_c_lr, h_pop_lr, h_lr]
  · rw [h_c_lt, h_pop_lt, h_lt]
  · rw [h_c_stk, h_s0_stack]
  · rw [h_c_bs, h_pop_bs, h_bs]
  · rw [h_c_cr, h_pop_cr, h_cr]
  · omega

/-- A `WhileBody` never branches past its loop: from an entry with no
    pending branch, the straight-line prefix keeps the flags clean and
    `brIf 0` sets at most `some 0`. (A halted entry is returned as is —
    its `branchTarget` is the clean one it came in with.) -/
theorem whileBody_branches_at_most_zero
    {fuel : Nat} {pref : List WasmInstr} (h_sl : StraightLineInstrs pref)
    {st st' : WasmState} (h_nb : st.branchTarget = none)
    (hw : evalInstrs fuel st (pref ++ [.brIf 0]) = some st') :
    st'.branchTarget = none ∨ st'.branchTarget = some 0 := by
  by_cases h_halt : st.halted = true
  · -- Every non-empty list returns the state unchanged when halted.
    have h_eq : st' = st := by
      rcases pref with _ | ⟨i, pref'⟩
      · simp only [List.nil_append, evalInstrs, h_halt, Bool.true_or, ↓reduceIte,
                   Option.some.injEq] at hw
        exact hw.symm
      · rw [List.cons_append] at hw
        unfold evalInstrs at hw
        simp only [h_halt, Bool.true_or, ↓reduceIte, Option.some.injEq] at hw
        exact hw.symm
    left; rw [h_eq]; exact h_nb
  · have h_nh : st.halted = false := by simpa using h_halt
    obtain ⟨ws_m, _, h_mb, h_mh, hw_br⟩ :=
      evalInstrs_straightLine_append h_sl h_nb h_nh hw
    rw [evalInstrs_cons_default fuel ws_m (.brIf 0) [] h_mb h_mh rfl] at hw_br
    cases he : evalInstr ws_m (.brIf 0) with
    | none => rw [he] at hw_br; simp at hw_br
    | some ws1 =>
        rw [he] at hw_br
        simp only [evalInstrs, Option.some.injEq] at hw_br
        subst hw_br
        simp only [evalInstr, Option.bind_eq_bind] at he
        rcases h_pop : ws_m.pop with _ | ⟨vc, ws2⟩
        · rw [h_pop] at he; simp at he
        · rw [h_pop] at he
          simp only [Option.some_bind] at he
          cases vc with
          | wI32 c =>
              have h_ws2_bt : ws2.branchTarget = ws_m.branchTarget := by
                unfold WasmState.pop at h_pop
                rcases hs : ws_m.stack with _ | ⟨v, rs⟩
                · rw [hs] at h_pop; simp at h_pop
                · rw [hs] at h_pop; simp at h_pop
                  obtain ⟨_, h_eq⟩ := h_pop
                  rw [← h_eq]
              by_cases hc : c = 0
              · simp only [hc, ↓reduceIte, Option.some.injEq] at he
                subst he
                left; rw [h_ws2_bt, h_mb]
              · simp only [hc, ↓reduceIte, Option.some.injEq] at he
                subst he
                right; rfl
          | _ => simp at he


-- ════════════════════════════════════════════════════════════════════
-- Piece 3c — the IR-side iteration trace.
-- ════════════════════════════════════════════════════════════════════

/-- `Refines` against a lowering state that agrees with `s` on every
    field it reads (the `WhileBody` frame) — the structural invariants
    (`Fresh`, `AliasFree`, …) are `s`'s own and come from any `Refines`
    at `s`. This is how the body-out refinement at `s1` becomes the next
    iteration's entry refinement at `s`, the state the body was lowered
    from. -/
theorem Refines.retarget
    {ws ws0 : WasmState} {s s1 : LowerState} {kst kst0 : Quanta.KOps.State}
    {layout : BufferLayout}
    (R1 : Refines ws s1 kst layout) (R0 : Refines ws0 s kst0 layout)
    (h_stk : s1.stack = s.stack) (h_lr : s1.localReg = s.localReg)
    (h_lt : s1.localTy = s.localTy) (h_cr : s1.currentReg = s.currentReg) :
    Refines ws s kst layout := by
  refine ⟨?_, ?_, R0.fresh, R0.aliasFree, R0.injLocals, R1.heapRefines, ?_,
          R0.freshCurrent, R0.curLocDisj⟩
  · have := R1.stk; rw [h_stk] at this; exact this
  · have := R1.locs; rw [h_lr, h_lt] at this; exact this
  · have := R1.currentReg; rw [h_cr, h_lt] at this; exact this

/-- `Refines` does not read `branchTarget`. -/
theorem Refines.clear_branch
    {ws : WasmState} {s : LowerState} {kst : Quanta.KOps.State} {layout : BufferLayout}
    (R : Refines ws s kst layout) :
    Refines { ws with branchTarget := none } s kst layout :=
  ⟨R.stk, R.locs, R.fresh, R.aliasFree, R.injLocals, R.heapRefines, R.currentReg,
   R.freshCurrent, R.curLocDisj⟩

/-- The IR runs the lowered `WhileBody` once per WASM iteration: given
    the WASM trace, there is an IR state sequence with one fuel for all
    runs, `broke = false` after every continue, `broke = true` after the
    exit, and body-out refinement at every iteration. -/
theorem whileBody_ir_trace
    (fuel : Nat) (frames : List FrameKind)
    (pref : List WasmInstr)
    (h_sl : StraightLineInstrs pref) (h_nw : NoLocalWrites pref)
    (h_ht : stackHeight 0 pref = some 1)
    (s s1 : LowerState) (bodyOps : List KernelOp)
    (h_lb : lowerInstrs fuel (.loopK :: frames) s (pref ++ [.brIf 0]) = some (s1, bodyOps))
    (layout : BufferLayout)
    (h_buf_locals : ∀ (ws_x : WasmState) (s_x : LowerState),
        BufferLocalsWellFormed layout ws_x s_x)
    (h_no_buf_stack : ∀ (s_x : LowerState), NoBufferPatternStack s_x)
    (h_load_bounds : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        LoadAddressesInBounds layout s_x kst_x)
    (h_store_bounds : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        StoreAddressInBounds layout s_x kst_x)
    (h_store_layout : ∀ (s_x : LowerState) (kst_x : Quanta.KOps.State),
        StoreLayoutNoOverlap layout s_x kst_x)
    (n : Nat) (entries bodyOuts : Fin (n + 1) → WasmState)
    (h_step : ∀ i : Fin (n + 1),
        evalInstrs fuel (entries i) (pref ++ [.brIf 0]) = some (bodyOuts i))
    (h_cont : ∀ i : Fin n,
        (bodyOuts i.castSucc).branchTarget = some 0 ∧
        entries i.succ = { bodyOuts i.castSucc with branchTarget := none })
    (h_exit : (bodyOuts (Fin.last n)).branchTarget = none)
    (kst : Quanta.KOps.State)
    (R : Refines (entries 0) s kst layout)
    (h_e0_nb : (entries 0).branchTarget = none)
    (h_e0_nh : (entries 0).halted = false)
    (h_kst : kst.broke = false) :
    ∃ (kstStates : Fin (n + 2) → Quanta.KOps.State) (F_b : Nat),
      kstStates 0 = kst ∧
      (∀ i : Fin (n + 1),
        evalOps F_b (kstStates i.castSucc) bodyOps = some (kstStates i.succ)) ∧
      (∀ i : Fin n, (kstStates i.castSucc.succ).broke = false) ∧
      (kstStates (Fin.last (n + 1))).broke = true ∧
      (∀ i : Fin (n + 1), Refines (bodyOuts i) s1 (kstStates i.succ) layout) ∧
      (∀ i : Fin (n + 1), (bodyOuts i).halted = false) := by
  obtain ⟨h_lr, h_lt, h_stk, _, h_cr, _⟩ := whileBody_lowering_frame h_sl h_nw h_ht h_lb
  induction n generalizing kst with
  | zero =>
      obtain ⟨kst1, F, h_ev, R1, h_nh1, h_out⟩ :=
        whileBody_iteration fuel frames pref h_sl (entries 0) s kst layout R
          h_e0_nb h_e0_nh h_kst h_buf_locals h_no_buf_stack h_load_bounds
          h_store_bounds h_store_layout (bodyOuts 0) s1 bodyOps (h_step 0) h_lb
      have h_broke : kst1.broke = true := by
        have h_exit0 : (bodyOuts 0).branchTarget = none := h_exit
        rcases h_out with ⟨h_bt, _⟩ | ⟨_, h_b⟩
        · rw [h_exit0] at h_bt; exact Option.noConfusion h_bt
        · exact h_b
      refine ⟨fun i => if i.val = 0 then kst else kst1, F, by simp, ?_, ?_, ?_, ?_, ?_⟩
      · intro i
        have h0 : i = 0 := Fin.ext (by omega)
        subst h0
        simpa using h_ev
      · intro i; exact absurd i.isLt (by simp)
      · simpa using h_broke
      · intro i
        have h0 : i = 0 := Fin.ext (by omega)
        subst h0
        simpa using R1
      · intro i
        have h0 : i = 0 := Fin.ext (by omega)
        subst h0
        exact h_nh1
  | succ n IH =>
      -- First iteration.
      obtain ⟨kst1, F0, h_ev0, R1, h_nh1, h_out0⟩ :=
        whileBody_iteration fuel frames pref h_sl (entries 0) s kst layout R
          h_e0_nb h_e0_nh h_kst h_buf_locals h_no_buf_stack h_load_bounds
          h_store_bounds h_store_layout (bodyOuts 0) s1 bodyOps (h_step 0) h_lb
      obtain ⟨h_bt0, h_e1⟩ := h_cont 0
      have h_bt0' : (bodyOuts 0).branchTarget = some 0 := h_bt0
      have h_kst1_ok : kst1.broke = false := by
        rcases h_out0 with ⟨_, h_b⟩ | ⟨h_bt, _⟩
        · exact h_b
        · rw [h_bt] at h_bt0'; exact Option.noConfusion h_bt0'
      -- The shifted trace.
      have h_e1' : entries 1 = { bodyOuts 0 with branchTarget := none } := h_e1
      obtain ⟨seq', F', h_s0', h_step', h_cont', h_exit', h_ref', h_nh'⟩ :=
        IH (fun i => entries i.succ) (fun i => bodyOuts i.succ)
          (fun i => h_step i.succ)
          (fun i => by
            obtain ⟨h1, h2⟩ := h_cont i.succ
            exact ⟨h1, h2⟩)
          h_exit kst1
          (by
            show Refines (entries 1) s kst1 layout
            rw [h_e1']
            exact (Refines.retarget R1 R h_stk h_lr h_lt h_cr).clear_branch)
          (by show (entries 1).branchTarget = none; rw [h_e1'])
          (by show (entries 1).halted = false; rw [h_e1']; exact h_nh1)
          h_kst1_ok
      refine ⟨fun i => if h : i.val = 0 then kst else seq' ⟨i.val - 1, by omega⟩,
              max F0 F', by simp, ?_, ?_, ?_, ?_, ?_⟩
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
      · intro i
        by_cases h : i.val = 0
        · have h_sc : (i.succ).val = 1 := by simp [h]
          simp only [h_sc, ↓reduceDIte, Nat.one_ne_zero, Nat.sub_self]
          have h_i : i = 0 := Fin.ext h
          subst h_i
          have : (⟨0, by omega⟩ : Fin (n + 2)) = 0 := rfl
          rw [this, h_s0']; exact R1
        · have h_sc : (i.succ).val = i.val + 1 := by simp
          have h_ne : i.val + 1 ≠ 0 := by omega
          simp only [h_sc, ↓reduceDIte, h_ne, Nat.add_sub_cancel]
          have h_r := h_ref' ⟨i.val - 1, by omega⟩
          simp only [Fin.succ_mk] at h_r
          have h_idx : i.val - 1 + 1 = i.val := by omega
          have h_fin : (⟨i.val - 1 + 1, by omega⟩ : Fin (n + 2)) = ⟨i.val, by omega⟩ :=
            Fin.ext h_idx
          rw [h_fin] at h_r
          have h_i : (⟨i.val, i.isLt⟩ : Fin (n + 2)) = i := Fin.ext rfl
          rw [h_i] at h_r
          exact h_r
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
-- Piece 3d — the kernel shape and the apex.
-- ════════════════════════════════════════════════════════════════════

/-- Kernel-body well-formedness for L11: straight-line ops interleaved
    with `wloop 0 :: <body> ++ [.wend]` segments whose body is a
    `WhileBody`. Subsumes L10v7's `KernelInstrs` (`KernelInstrs.toW`). -/
inductive KernelInstrsW : List WasmInstr → Type
  | empty : KernelInstrsW []
  | sl_cons {i : WasmInstr} {rest : List WasmInstr} :
      StraightLineInstr i →
      KernelInstrsW rest →
      KernelInstrsW (i :: rest)
  | while_cons {rest body post : List WasmInstr} :
      splitAtEnd rest = some (body, post) →
      WhileBody body →
      KernelInstrsW post →
      KernelInstrsW (.wloop 0 :: rest)

/-- Loop nesting depth — the fuel measure, as for `KernelInstrs`. -/
def KernelInstrsW.depth : ∀ {instrs : List WasmInstr}, KernelInstrsW instrs → Nat
  | _, .empty => 0
  | _, .sl_cons _ rest_wf => rest_wf.depth
  | _, .while_cons _ _ post_wf => 1 + post_wf.depth

/-- L11 apex — `framework_preservation_kernel_while`.
    Admits any `KernelInstrsW` kernel body; the while segments run any
    number of iterations the WASM fuel allows. -/
theorem framework_preservation_kernel_while
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
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs (fuel + 1) ws instrs = some ws')
    (hl : lowerInstrs (fuel + 1) frames s instrs = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  induction h_wf generalizing fuel ws s kst ws' s' ops with
  | empty =>
      simp only [evalInstrs, Option.some.injEq] at hw
      simp only [lowerInstrs, Option.some.injEq, Prod.mk.injEq] at hl
      obtain ⟨h_s, h_ops⟩ := hl
      subst hw; subst h_s; subst h_ops
      refine ⟨kst, 0, by simp [evalOps], R, ?_, ?_⟩
      · intro d hd; rw [h_no_branch] at hd; exact Option.noConfusion hd
      · intro _; exact h_kst_no_broke
  | @sl_cons i rest h_sl _h_rest_wf IH =>
      -- `[i] ++ rest`: the straight-line framework on the head, the IH
      -- on the tail, composed under one fuel.
      have h_sl1 : StraightLineInstrs [i] := ⟨h_sl, trivial⟩
      obtain ⟨ws_m, hw_i, h_mb, h_mh, hw_rest⟩ :=
        evalInstrs_straightLine_append (pref := [i]) (rest := rest) h_sl1
          h_no_branch h_no_halt hw
      obtain ⟨s_m, ops1, ops2, hl_i, hl_rest, h_ops⟩ :=
        lowerInstrs_straightLine_append (pref := [i]) (rest := rest) h_sl1 hl
      subst h_ops
      obtain ⟨kst_m, F1, h_ev1, R_m, h_bridge_m⟩ :=
        framework_preservation_straightLine (fuel + 1) frames ws s kst layout R
          h_no_branch h_no_halt h_kst_no_broke h_buf_locals h_no_buf_stack
          h_load_bounds h_store_bounds h_store_layout [i] h_sl1 ws_m s_m ops1
          hw_i hl_i
      have h_kst_m_ok : kst_m.broke = false := h_bridge_m.right h_mb
      obtain ⟨kst', F2, h_ev2, R', h_bridge'⟩ :=
        IH fuel ws_m s_m kst_m R_m h_mb h_mh h_kst_m_ok h_fuel ws' s' ops2 hw_rest hl_rest
      refine ⟨kst', max F1 F2, ?_, R', h_bridge'⟩
      exact evalOps_append_fuel_mono_head (Nat.le_max_left F1 F2) h_ev1 h_kst_m_ok
        (evalOps_fuel_mono (Nat.le_max_right F1 F2) h_ev2)
  | @while_cons rest body post h_split h_body h_post_wf IH =>
      have h_depth : (KernelInstrsW.while_cons h_split h_body h_post_wf).depth
                        = 1 + h_post_wf.depth := rfl
      rw [h_depth] at h_fuel
      obtain ⟨pref, h_sl, h_nw, h_ht, h_body_eq⟩ := h_body
      subst h_body_eq
      -- Post IH in the `post_preserves` shape, at fuel `fuel = (fuel - 1) + 1`.
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
        intro ws_p s_p kst_p R_p h_nb_p h_nh_p h_nbk_p ws'_p s'_p postOps hw_p hl_p
        have h_fuel_for_ih : fuel - 1 ≥ 2 + h_post_wf.depth := by omega
        have h_fuel_eq : fuel = (fuel - 1) + 1 := by omega
        rw [h_fuel_eq] at hw_p hl_p
        exact IH (fuel - 1) ws_p s_p kst_p R_p h_nb_p h_nh_p h_nbk_p h_fuel_for_ih
          ws'_p s'_p postOps hw_p hl_p
      -- The WASM trace from the wloop arm.
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
      -- The body and post lowerings, out of the wloop arm.
      obtain ⟨s1, bodyOps, s2, postOps, h_lb, h_lp⟩ :
          ∃ (s1 : LowerState) (bodyOps : List KernelOp) (s2 : LowerState)
            (postOps : List KernelOp),
            lowerInstrs fuel (.loopK :: frames) s (pref ++ [.brIf 0]) = some (s1, bodyOps) ∧
            lowerInstrs fuel frames s1 post = some (s2, postOps) := by
        have hl2 := hl
        simp only [lowerInstrs] at hl2
        rw [h_split] at hl2
        simp only [Option.bind_eq_bind, Option.some_bind] at hl2
        cases h_lb : lowerInstrs fuel (.loopK :: frames) s (pref ++ [.brIf 0]) with
        | none => rw [h_lb] at hl2; simp at hl2
        | some p1 =>
            rw [h_lb] at hl2
            obtain ⟨s1, bodyOps⟩ := p1
            simp only [Option.some_bind] at hl2
            obtain ⟨h_lr, h_lt, _, _, h_cr, _⟩ := whileBody_lowering_frame h_sl h_nw h_ht h_lb
            have h_restored :
                ({ s1 with localReg := s.localReg, localTy := s.localTy,
                           currentReg := s.currentReg } : LowerState) = s1 := by
              rw [← h_lr, ← h_lt, ← h_cr]
            rw [h_restored] at hl2
            cases h_lp : lowerInstrs fuel frames s1 post with
            | none => rw [h_lp] at hl2; simp at hl2
            | some p2 =>
                obtain ⟨s2, postOps⟩ := p2
                exact ⟨s1, bodyOps, s2, postOps, rfl, h_lp⟩
      -- The IR trace.
      have R0 : Refines (entries 0) s kst layout := by rw [h_e0]; exact R
      obtain ⟨kstStates, F_b, h_kst_start, h_ir_step, h_ir_cont, h_ir_exit, h_ref, h_nh⟩ :=
        whileBody_ir_trace fuel frames pref h_sl h_nw h_ht s s1 bodyOps h_lb layout
          h_buf_locals h_no_buf_stack h_load_bounds h_store_bounds h_store_layout
          n entries bodyOuts h_step h_cont h_exit_bt kst R0
          (by rw [h_e0]; exact h_no_branch) (by rw [h_e0]; exact h_no_halt) h_kst_no_broke
      exact preservation_evalInstrs_cons_wloop_nIterExit
        frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke
        fuel rest (pref ++ [.brIf 0]) post h_split n entries bodyOuts h_e0 h_step h_cont
        ⟨h_exit_bt, h_nh (Fin.last n)⟩ s1 bodyOps h_lb kstStates h_kst_start F_b
        h_ir_step h_ir_cont h_ir_exit h_ref s2 postOps h_lp post_preserves
        (whileBody_lowering_frame h_sl h_nw h_ht h_lb) h_bound ws' s' ops hw hl


-- ════════════════════════════════════════════════════════════════════
-- Subsumption of L10v7 and compile-time witnesses.
-- ════════════════════════════════════════════════════════════════════

/-- A list of nops is straight-line and writes no local. -/
theorem irEmptyPrefix_straightLine {pref : List WasmInstr} (h : IsIrEmptyPrefix pref) :
    StraightLineInstrs pref ∧ NoLocalWrites pref := by
  induction pref with
  | nil => exact ⟨trivial, trivial⟩
  | cons i rest IH =>
      obtain ⟨h_i, h_rest⟩ := h
      obtain ⟨h1, h2⟩ := IH h_rest
      cases i <;> simp [IsIrEmptyOp] at h_i
      exact ⟨⟨trivial, h1⟩, ⟨trivial, h2⟩⟩

/-- Nops do not move the stack height. -/
theorem irEmptyPrefix_stackHeight {pref : List WasmInstr} (h : IsIrEmptyPrefix pref) (k : Nat) :
    stackHeight k pref = some k := by
  induction pref generalizing k with
  | nil => rfl
  | cons i rest IH =>
      obtain ⟨h_i, h_rest⟩ := h
      cases i <;> simp [IsIrEmptyOp] at h_i
      simp [stackHeight, stackEffect, IH h_rest]

/-- L10v7's body shape is a `WhileBody`: the nop prefix plus `i32Const 0`
    is a straight-line, non-writing prefix of height 1. -/
theorem WloopBodyShape.toWhile {body : List WasmInstr} (h : WloopBodyShape body) :
    WhileBody body := by
  obtain ⟨pref, h_pref, h_eq⟩ := h
  obtain ⟨h_sl, h_nw⟩ := irEmptyPrefix_straightLine h_pref
  refine ⟨pref ++ [.i32Const 0], ?_, ?_, ?_, by rw [h_eq, List.append_assoc]; rfl⟩
  · clear h_nw h_eq
    induction pref with
    | nil => exact ⟨trivial, trivial⟩
    | cons i rest IH =>
        obtain ⟨h_i, h_rest⟩ := h_sl
        exact ⟨h_i, IH h_pref.right h_rest⟩
  · clear h_sl h_eq
    induction pref with
    | nil => exact ⟨trivial, trivial⟩
    | cons i rest IH =>
        obtain ⟨h_i, h_rest⟩ := h_nw
        exact ⟨h_i, IH h_pref.right h_rest⟩
  · clear h_sl h_nw h_eq
    induction pref with
    | nil => rfl
    | cons i rest IH =>
        obtain ⟨h_i, h_rest⟩ := h_pref
        cases i <;> simp [IsIrEmptyOp] at h_i
        simp only [List.cons_append, stackHeight, stackEffect]
        simpa using IH h_rest

/-- Every L10v7 kernel is an L11 kernel, at the same depth. -/
def KernelInstrs.toW : ∀ {instrs : List WasmInstr}, KernelInstrs instrs → KernelInstrsW instrs
  | _, .empty => .empty
  | _, .sl_cons h rest => .sl_cons h rest.toW
  | _, .wloop_cons h_split h_body post => .while_cons h_split h_body.toWhile post.toW

theorem KernelInstrs.toW_depth : ∀ {instrs : List WasmInstr} (h : KernelInstrs instrs),
    h.toW.depth = h.depth
  | _, .empty => rfl
  | _, .sl_cons _ rest => by simp [KernelInstrs.toW, KernelInstrsW.depth, KernelInstrs.depth, rest.toW_depth]
  | _, .wloop_cons _ _ post => by simp [KernelInstrs.toW, KernelInstrsW.depth, KernelInstrs.depth, post.toW_depth]

/-- The canonical memory-carried while loop —
    `loop { *p += 1; if *p < n then continue }` over a buffer local:

        wloop 0
          local.get p; local.get p; i32.load; i32.const 1; i32.add; i32.store
          local.get p; i32.load; local.get n; i32.lt_u
          br_if 0
        wend

    (`p` a `#[quanta::shared]` buffer local at index 0 and `n` a plain
    local at index 1; the `i32.shl`-by-2 that makes the byte address is
    what a `[u32]` access compiles to and is folded by the buffer arms,
    written here as the lowering sees it after `local.get` of a buffer
    local: the `bufferPtr` is the address.) The witness typechecks at
    definition time — the shape is admitted by the apex. -/
example : KernelInstrsW
    [.wloop 0,
       .localGet 0, .localGet 0, .i32Load 0 0, .i32Const 1, .i32Add, .i32Store 0 0,
       .localGet 0, .i32Load 0 0, .localGet 1, .i32LtU,
       .brIf 0,
     .wend] :=
  .while_cons rfl
    ⟨[.localGet 0, .localGet 0, .i32Load 0 0, .i32Const 1, .i32Add, .i32Store 0 0,
      .localGet 0, .i32Load 0 0, .localGet 1, .i32LtU],
     by simp [StraightLineInstrs, StraightLineInstr],
     by simp [NoLocalWrites, NoLocalWrite],
     rfl, rfl⟩
    .empty

/-- A while loop followed by straight-line code, and the depth measure. -/
example : (KernelInstrsW.while_cons (rest := [.i32Const 0, .brIf 0, .wend, .nop]) rfl
    ⟨[.i32Const 0], ⟨trivial, trivial⟩, ⟨trivial, trivial⟩, rfl, rfl⟩
    (.sl_cons (i := .nop) trivial .empty)).depth = 1 := rfl

end Quanta.Wasm
