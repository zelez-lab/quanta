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

namespace Quanta.Wasm

open Quanta.KOps (KernelOp evalOps)

/-- A body whose runs never branch past the enclosing loop: every body
    evaluation leaves `branchTarget` either cleared (fall through = loop
    exit) or `some 0` (continue). Straight-line prefixes closed by
    `brIf 0` satisfy this — `brIf 0` is the only branch and it targets
    depth 0 — which is what keeps the `some (n+1)` arm of `iterLoop`
    unreachable. -/
def BodyBranchesAtMostZero (fuel : Nat) (body : List WasmInstr) : Prop :=
  ∀ (st st' : WasmState), evalInstrs fuel st body = some st' →
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
                  IH h_iter
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
                rcases h_body st0 st1 h_step0 with h0 | h0 <;> rw [h_bt] at h0 <;> simp at h0


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

end Quanta.Wasm
