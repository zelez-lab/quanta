/-
# The WASM side of a loop left by a conditional branch to its block

rustc lowers `while cond { body }` to

    block { loop { …pref…; br_if 1; …body…; br 0 } tail } post

The loop is never left by falling through: every iteration ends in
`br 0` (continue), and the exit is the `br_if 1` firing — a branch past
the loop, to the end of the enclosing block. On the `evalInstrs` side
that is the `some (n + 1)` arm of `iterLoop`: the body leaves
`branchTarget = some 1`, `iterLoop` returns the state with the target
decremented to `some 0` WITHOUT running the loop's tail, and the block
arm consumes the `some 0` and runs its post.

`PreservationWhile` has the fall-through-exit version of every piece
below (`BodyBranchesAtMostZero`, `iterLoop_trace_of_eval`, and in
`PreservationBridge` `iterLoop_n_iter_exit` / `_post_eval`). This file
is the `br_if 1` twin, in exactly the trace shape those consume:

1. `BodyContinuesOrExits1` — the body predicate, and
   `iterLoop_trace_of_eval_exit1` — the iteration trace out of a
   returning `iterLoop`.
2. `iterLoop_n_iter_exit1` — the converse replay.
3. The shape lemmas — `NoStructured` bodies split at their closing
   `wend` (`splitAtEnd_noStructured`) and `evalInstrs_block_wloop`
   reads rustc's `block { loop … } post` skeleton down to `iterLoop`
   and the block arm's dispatch, with the two directions the apex
   needs (`evalInstrs_block_wloop_exit1`,
   `evalInstrs_block_wloop_trace_of_eval`).
4. The per-instruction facts at the two branch sites
   (`brIf1_in_loop_sets_target`, `br0_in_loop_continues`) and the body
   predicate for the rustc shape (`blockWhileBody_continuesOrExits1`).
-/

import Quanta.Wasm.PreservationWhile

namespace Quanta.Wasm

open Quanta.KOps (KernelOp evalOps)

-- ════════════════════════════════════════════════════════════════════
-- Piece 1 — the body predicate and the iteration trace.
-- ════════════════════════════════════════════════════════════════════

/-- A body that never falls through and never halts: every run ends in
    `br 0` (continue, `branchTarget = some 0`) or in a fired `br_if 1`
    (exit past the loop, `branchTarget = some 1`). `halted = false` is
    carried so the next entry state is clean again — the trace rebuilds
    entries by clearing `branchTarget` only. -/
def BodyContinuesOrExits1 (fuel : Nat) (body : List WasmInstr) : Prop :=
  ∀ (st st' : WasmState), st.branchTarget = none → st.halted = false →
    evalInstrs fuel st body = some st' →
    st'.halted = false ∧ (st'.branchTarget = some 0 ∨ st'.branchTarget = some 1)

/-- The iteration trace exists whenever `iterLoop` returns on a body
    that continues or exits through `br_if 1`.

    Inducts on the iteration counter `f`, peeling one body run per step
    exactly as `iterLoop_trace_of_eval` does: the `some 0` arm prepends
    an iteration, the `some 1` arm is the exit (`iterLoop` hands back
    the body-out with the target decremented to `some 0`, and never
    runs the loop's tail), and `none` / `some (m + 2)` are refuted by
    the predicate. `n` is the number of continues; the body runs
    `n + 1` times; `n + 1 ≤ f` is what `iterLoop_n_iter_exit1` needs to
    run the trace back. -/
theorem iterLoop_trace_of_eval_exit1
    {fuel : Nat} {body post : List WasmInstr}
    (h_body : BodyContinuesOrExits1 fuel body)
    {f : Nat} {st0 ws' : WasmState}
    (h_nb : st0.branchTarget = none) (h_nh : st0.halted = false)
    (h_iter : evalInstrs.iterLoop fuel body post f st0 = some ws') :
    ∃ (n : Nat) (entries bodyOuts : Fin (n + 1) → WasmState),
      entries 0 = st0 ∧
      (∀ i : Fin (n + 1), evalInstrs fuel (entries i) body = some (bodyOuts i)) ∧
      (∀ i : Fin n,
        (bodyOuts i.castSucc).branchTarget = some 0 ∧
        entries i.succ = { bodyOuts i.castSucc with branchTarget := none }) ∧
      (∀ i : Fin (n + 1), (bodyOuts i).halted = false) ∧
      (bodyOuts (Fin.last n)).branchTarget = some 1 ∧
      ws' = { bodyOuts (Fin.last n) with branchTarget := some 0 } ∧
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
        obtain ⟨h_nh1, h_bt1⟩ := h_body st0 st1 h_nb h_nh h_step0
        cases h_bt : st1.branchTarget with
        | none =>
            exfalso
            rcases h_bt1 with h0 | h0 <;> rw [h_bt] at h0 <;> exact Option.noConfusion h0
        | some d =>
            cases d with
            | zero =>
                -- Continue: the rest of the trace comes from the IH on k.
                rw [h_bt] at h_iter
                simp only at h_iter
                obtain ⟨n', entries', bodyOuts', h_e0, h_step', h_cont', h_nh', h_exit',
                        h_ws', h_bound'⟩ :=
                  IH (st0 := { st1 with branchTarget := none }) rfl h_nh1 h_iter
                refine ⟨n' + 1,
                  fun i => if h : i.val = 0 then st0 else entries' ⟨i.val - 1, by omega⟩,
                  fun i => if h : i.val = 0 then st1 else bodyOuts' ⟨i.val - 1, by omega⟩,
                  by simp, ?_, ?_, ?_, ?_, ?_, ?_⟩
                · intro i
                  by_cases h : i.val = 0
                  · simp only [h, ↓reduceDIte]; exact h_step0
                  · simp only [h, ↓reduceDIte]; exact h_step' _
                · intro i
                  by_cases h : i.val = 0
                  · have h_cs : (i.castSucc).val = 0 := by simp [h]
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
                · intro i
                  by_cases h : i.val = 0
                  · simp only [h, ↓reduceDIte]; exact h_nh1
                  · simp only [h, ↓reduceDIte]; exact h_nh' _
                · have h_last : (Fin.last (n' + 1)).val = n' + 1 := by simp
                  simp only [h_last, Nat.succ_ne_zero, ↓reduceDIte, Nat.add_sub_cancel]
                  have : (⟨n', by omega⟩ : Fin (n' + 1)) = Fin.last n' := rfl
                  rw [this]; exact h_exit'
                · have h_last : (Fin.last (n' + 1)).val = n' + 1 := by simp
                  simp only [h_last, Nat.succ_ne_zero, ↓reduceDIte, Nat.add_sub_cancel]
                  have : (⟨n', by omega⟩ : Fin (n' + 1)) = Fin.last n' := rfl
                  rw [this]; exact h_ws'
                · omega
            | succ m =>
                cases m with
                | zero =>
                    -- Exit through `br_if 1`: n = 0, the tail never runs.
                    rw [h_bt] at h_iter
                    simp only [Option.some.injEq] at h_iter
                    refine ⟨0, fun _ => st0, fun _ => st1, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
                    · intro i; exact h_step0
                    · intro i; exact absurd i.isLt (by simp)
                    · intro i; exact h_nh1
                    · exact h_bt
                    · exact h_iter.symm
                    · omega
                | succ m' =>
                    -- A branch past the block — excluded by the body shape.
                    exfalso
                    rcases h_bt1 with h0 | h0 <;> rw [h_bt] at h0 <;> simp at h0


-- ════════════════════════════════════════════════════════════════════
-- Piece 2 — the converse replay.
-- ════════════════════════════════════════════════════════════════════

/-- Replay of a `br_if 1` trace through `iterLoop`: `n` continues, then
    the exit iteration leaves `branchTarget = some 1`, and `iterLoop`
    returns that body-out with the target decremented to `some 0` —
    the loop's tail is skipped. The shape of `iterLoop_n_iter_exit`;
    needs `f ≥ n + 1` iterations of fuel. -/
theorem iterLoop_n_iter_exit1
    {fuel : Nat}
    {body post : List WasmInstr}
    {n : Nat}
    (entries : Fin (n + 1) → WasmState)
    (bodyOuts : Fin (n + 1) → WasmState)
    (h_step : ∀ i : Fin (n + 1),
        evalInstrs fuel (entries i) body = some (bodyOuts i))
    (h_continue : ∀ i : Fin n,
        (bodyOuts i.castSucc).branchTarget = some 0 ∧
        entries i.succ = { bodyOuts i.castSucc with branchTarget := none })
    (h_exit : (bodyOuts (Fin.last n)).branchTarget = some 1)
    {f : Nat} (h_f : f ≥ n + 1) :
    evalInstrs.iterLoop fuel body post f (entries 0)
      = some { bodyOuts (Fin.last n) with branchTarget := some 0 } := by
  induction n generalizing f with
  | zero =>
      have h_f_ge_1 : f ≥ 1 := by omega
      obtain ⟨k, hk⟩ : ∃ k, f = k + 1 := ⟨f - 1, by omega⟩
      rw [hk]
      unfold evalInstrs.iterLoop
      rw [h_step 0]
      simp only
      have h_bt : (bodyOuts 0).branchTarget = some 1 := h_exit
      rw [h_bt]
      rfl
  | succ n IH =>
      have h_f_ge_1 : f ≥ 1 := by omega
      obtain ⟨k, hk⟩ : ∃ k, f = k + 1 := ⟨f - 1, by omega⟩
      rw [hk]
      unfold evalInstrs.iterLoop
      rw [h_step 0]
      simp only
      have h_cont_0 := h_continue 0
      have h_bt_0 : (bodyOuts 0).branchTarget = some 0 := h_cont_0.left
      rw [h_bt_0]
      simp only
      have h_entries_1 : ({ bodyOuts (0 : Fin (n + 1 + 1))
                                with branchTarget := none } : WasmState)
                            = entries ⟨1, by omega⟩ := h_cont_0.right.symm
      rw [h_entries_1]
      have h_k_bound : k ≥ n + 1 := by omega
      exact IH
        (fun i => entries ⟨i.val + 1, by omega⟩)
        (fun i => bodyOuts ⟨i.val + 1, by omega⟩)
        (fun i => h_step ⟨i.val + 1, by omega⟩)
        (fun i => h_continue ⟨i.val + 1, by omega⟩)
        h_exit
        h_k_bound


-- ════════════════════════════════════════════════════════════════════
-- Piece 3a — bodies without structured markers split at their `wend`.
-- ════════════════════════════════════════════════════════════════════

/-- An instruction `walkUntilCloser` neither counts nor stops on: not
    an opener (`block` / `loop` / `if`) and not a closer (`else` /
    `end`). Branches, returns and everything straight-line qualify. -/
def NoStructuredInstr : WasmInstr → Prop
  | .block _ => False
  | .wloop _ => False
  | .wif _   => False
  | .welse   => False
  | .wend    => False
  | _        => True

/-- A list with no structured marker at all — the bodies rustc's
    `while` shape puts between its markers. -/
def NoStructured : List WasmInstr → Prop
  | []        => True
  | i :: rest => NoStructuredInstr i ∧ NoStructured rest

/-- A straight-line instruction carries no structured marker. -/
theorem straightLine_noStructured {i : WasmInstr} (h : StraightLineInstr i) :
    NoStructuredInstr i := by
  cases i <;> first | trivial | exact absurd h (by simp [StraightLineInstr])

/-- A straight-line list carries no structured marker. -/
theorem straightLine_noStructured_list {l : List WasmInstr} (h : StraightLineInstrs l) :
    NoStructured l := by
  induction l with
  | nil => trivial
  | cons i rest IH =>
      obtain ⟨h_i, h_rest⟩ := h
      exact ⟨straightLine_noStructured h_i, IH h_rest⟩

/-- Marker-free lists concatenate. -/
theorem noStructured_append {l1 l2 : List WasmInstr}
    (h1 : NoStructured l1) (h2 : NoStructured l2) : NoStructured (l1 ++ l2) := by
  induction l1 with
  | nil => exact h2
  | cons i rest IH =>
      obtain ⟨h_i, h_rest⟩ := h1
      exact ⟨h_i, IH h_rest⟩

/-- The rustc body `pref ++ [br_if 1] ++ body2 ++ [br 0]` with
    straight-line `pref` and `body2` is marker-free: branches are not
    markers, so the loop's `wend` is the first closer the walker sees.
    This is what lets `evalInstrs_block_wloop` read the skeleton. -/
theorem blockWhileBody_noStructured
    {pref body2 : List WasmInstr}
    (h_pref : StraightLineInstrs pref) (h_body2 : StraightLineInstrs body2) :
    NoStructured (pref ++ [.brIf 1] ++ body2 ++ [.br 0]) :=
  noStructured_append
    (noStructured_append
      (noStructured_append (straightLine_noStructured_list h_pref) ⟨trivial, trivial⟩)
      (straightLine_noStructured_list h_body2))
    ⟨trivial, trivial⟩

/-- The walker steps over a marker-free instruction at the same depth,
    pushing it onto the accumulator. -/
theorem walkUntilCloser_cons_noStructured
    {i : WasmInstr} (h : NoStructuredInstr i)
    (rest : List WasmInstr) (n : Nat) (acc : List WasmInstr) :
    walkUntilCloser (i :: rest) n acc = walkUntilCloser rest n (i :: acc) := by
  cases i <;> first | rfl | exact absurd h (by simp [NoStructuredInstr])

/-- The walker steps over a marker-free list at the same depth,
    accumulating it in reverse. -/
theorem walkUntilCloser_append_noStructured
    {l : List WasmInstr} (h : NoStructured l)
    (rest : List WasmInstr) (n : Nat) (acc : List WasmInstr) :
    walkUntilCloser (l ++ rest) n acc = walkUntilCloser rest n (l.reverse ++ acc) := by
  induction l generalizing acc with
  | nil => rfl
  | cons i l' IH =>
      obtain ⟨h_i, h_l'⟩ := h
      rw [List.cons_append, walkUntilCloser_cons_noStructured h_i, IH h_l']
      simp

/-- A marker-free body closed by `wend` splits exactly there. -/
theorem splitAtEnd_noStructured
    {l : List WasmInstr} (h : NoStructured l) (t : List WasmInstr) :
    splitAtEnd (l ++ [.wend] ++ t) = some (l, t) := by
  rw [List.append_assoc, List.singleton_append]
  simp only [splitAtEnd, walkUntilCloser_append_noStructured h, walkUntilCloser,
             List.append_nil, List.reverse_reverse, Option.bind_eq_bind, Option.some_bind]


-- ════════════════════════════════════════════════════════════════════
-- Piece 3b — rustc's `block { loop … tail } post` skeleton, read down
-- to `iterLoop` and the block arm's dispatch.
-- ════════════════════════════════════════════════════════════════════

/-- The block body `wloop 0 :: loopBody ++ [wend] ++ tail` splits at
    the block's own `wend`: the walker enters the loop at depth 1,
    leaves it at the loop's `wend`, and stops at the next one. -/
theorem splitAtEnd_wloop_noStructured
    {loopBody tail : List WasmInstr}
    (h_body : NoStructured loopBody) (h_tail : NoStructured tail)
    (post : List WasmInstr) :
    splitAtEnd (.wloop 0 :: loopBody ++ [.wend] ++ tail ++ [.wend] ++ post)
      = some (.wloop 0 :: loopBody ++ [.wend] ++ tail, post) := by
  simp only [List.cons_append, List.append_assoc, List.singleton_append]
  simp only [splitAtEnd, walkUntilCloser, walkUntilCloser_append_noStructured h_body,
             walkUntilCloser_append_noStructured h_tail, Option.bind_eq_bind,
             Option.some_bind, List.reverse_append, List.reverse_cons, List.reverse_reverse,
             List.reverse_nil, List.nil_append, List.append_nil, List.cons_append,
             List.append_assoc]

/-- rustc's `while` skeleton under `evalInstrs`: with clean flags and
    marker-free `loopBody` / `tail`, the block arm opens on the loop,
    the loop arm runs `iterLoop` (body and tail at fuel `f`), and the
    block arm dispatches on the result's `branchTarget` — `none` falls
    through to `post`, `some 0` (a decremented `br_if 1`) clears and
    runs `post`, `some (k + 1)` propagates decremented. The post runs at
    fuel `f + 1`, one more than the loop body: the block arm spends one
    unit, the loop arm the other. -/
theorem evalInstrs_block_wloop
    {f : Nat} {ws : WasmState} {loopBody tail post : List WasmInstr}
    (h_nb : ws.branchTarget = none) (h_nh : ws.halted = false)
    (h_body : NoStructured loopBody) (h_tail : NoStructured tail) :
    evalInstrs (f + 2) ws (.block 0 :: .wloop 0 :: loopBody ++ [.wend] ++ tail ++ [.wend] ++ post)
      = match evalInstrs.iterLoop f loopBody tail f ws with
        | none => none
        | some out =>
            match out.branchTarget with
            | none => evalInstrs (f + 1) out post
            | some 0 => evalInstrs (f + 1) { out with branchTarget := none } post
            | some (k + 1) => some { out with branchTarget := some k } := by
  have h_cond : (ws.halted || ws.branchTarget.isSome) = false := by
    rw [h_nh, h_nb]; rfl
  have h_split_block := splitAtEnd_wloop_noStructured h_body h_tail post
  have h_split_loop := splitAtEnd_noStructured h_body tail
  simp only [List.cons_append, List.append_assoc, List.nil_append] at h_split_block h_split_loop ⊢
  -- The block arm.
  rw [evalInstrs.eq_def]
  simp only [h_cond, Bool.false_eq_true, ↓reduceIte, h_split_block]
  -- The loop arm, inside the block's body.
  rw [evalInstrs.eq_def (f + 1) ws]
  simp only [h_cond, Bool.false_eq_true, ↓reduceIte, h_split_loop]
  rfl

/-- The `br_if 1` trace, replayed through the skeleton: the kernel-level
    evaluation of `block { loop … } post` from the trace's entry is the
    post evaluated from the exit body-out with the target cleared. -/
theorem evalInstrs_block_wloop_exit1
    {f : Nat} {loopBody tail post : List WasmInstr}
    (h_body : NoStructured loopBody) (h_tail : NoStructured tail)
    {n : Nat}
    (entries : Fin (n + 1) → WasmState)
    (bodyOuts : Fin (n + 1) → WasmState)
    (h_nb : (entries 0).branchTarget = none) (h_nh : (entries 0).halted = false)
    (h_step : ∀ i : Fin (n + 1),
        evalInstrs f (entries i) loopBody = some (bodyOuts i))
    (h_continue : ∀ i : Fin n,
        (bodyOuts i.castSucc).branchTarget = some 0 ∧
        entries i.succ = { bodyOuts i.castSucc with branchTarget := none })
    (h_exit : (bodyOuts (Fin.last n)).branchTarget = some 1)
    (h_f : f ≥ n + 1) :
    evalInstrs (f + 2) (entries 0)
        (.block 0 :: .wloop 0 :: loopBody ++ [.wend] ++ tail ++ [.wend] ++ post)
      = evalInstrs (f + 1) { bodyOuts (Fin.last n) with branchTarget := none } post := by
  rw [evalInstrs_block_wloop h_nb h_nh h_body h_tail,
      iterLoop_n_iter_exit1 entries bodyOuts h_step h_continue h_exit h_f]

/-- From the kernel-level evaluation of the skeleton to the trace and
    the post: a returning `evalInstrs` on `block { loop … } post` with a
    body that continues or exits through `br_if 1` yields the iteration
    trace, the fuel bound, and the post evaluated from the exit body-out
    with the target cleared. -/
theorem evalInstrs_block_wloop_trace_of_eval
    {f : Nat} {ws ws' : WasmState} {loopBody tail post : List WasmInstr}
    (h_nb : ws.branchTarget = none) (h_nh : ws.halted = false)
    (h_body : NoStructured loopBody) (h_tail : NoStructured tail)
    (h_pred : BodyContinuesOrExits1 f loopBody)
    (hw : evalInstrs (f + 2) ws
            (.block 0 :: .wloop 0 :: loopBody ++ [.wend] ++ tail ++ [.wend] ++ post)
          = some ws') :
    ∃ (n : Nat) (entries bodyOuts : Fin (n + 1) → WasmState),
      entries 0 = ws ∧
      (∀ i : Fin (n + 1), evalInstrs f (entries i) loopBody = some (bodyOuts i)) ∧
      (∀ i : Fin n,
        (bodyOuts i.castSucc).branchTarget = some 0 ∧
        entries i.succ = { bodyOuts i.castSucc with branchTarget := none }) ∧
      (∀ i : Fin (n + 1), (bodyOuts i).halted = false) ∧
      (bodyOuts (Fin.last n)).branchTarget = some 1 ∧
      evalInstrs (f + 1) { bodyOuts (Fin.last n) with branchTarget := none } post = some ws' ∧
      n + 1 ≤ f := by
  rw [evalInstrs_block_wloop h_nb h_nh h_body h_tail] at hw
  cases h_iter : evalInstrs.iterLoop f loopBody tail f ws with
  | none => rw [h_iter] at hw; exact Option.noConfusion hw
  | some out =>
      rw [h_iter] at hw
      obtain ⟨n, entries, bodyOuts, h_e0, h_step, h_cont, h_nh', h_exit, h_out, h_bound⟩ :=
        iterLoop_trace_of_eval_exit1 h_pred h_nb h_nh h_iter
      refine ⟨n, entries, bodyOuts, h_e0, h_step, h_cont, h_nh', h_exit, ?_, h_bound⟩
      subst h_out
      simpa using hw


-- ════════════════════════════════════════════════════════════════════
-- Piece 4 — the two branch sites and the body predicate for the
-- rustc shape.
-- ════════════════════════════════════════════════════════════════════

/-- `br_if 1` at the exit site: pops the condition; zero falls through,
    non-zero arms a branch to depth 1 (past the loop, to the end of the
    enclosing block). The shape of `evalInstr_brIf_shape_pub` at depth
    1, as an equation. -/
theorem brIf1_in_loop_sets_target
    {fuel : Nat} {ws : WasmState} {c : UInt32} {rest_w : List WasmValue}
    (h_nb : ws.branchTarget = none) (h_nh : ws.halted = false)
    (h_stack : ws.stack = .wI32 c :: rest_w) :
    evalInstrs fuel ws [.brIf 1]
      = some (if c = 0 then { ws with stack := rest_w }
              else { ws with stack := rest_w, branchTarget := some 1 }) := by
  rw [evalInstrs_cons_default fuel ws (.brIf 1) [] h_nb h_nh rfl]
  simp only [evalInstr, WasmState.pop, h_stack, Option.bind_eq_bind, Option.some_bind]
  by_cases hc : c = 0
  · simp only [hc, ↓reduceIte, evalInstrs]
  · simp only [hc, ↓reduceIte, evalInstrs]

/-- `br 0` at the end of the body: arms the continue. -/
theorem br0_in_loop_continues
    {fuel : Nat} {ws : WasmState}
    (h_nb : ws.branchTarget = none) (h_nh : ws.halted = false) :
    evalInstrs fuel ws [.br 0] = some { ws with branchTarget := some 0 } := by
  rw [evalInstrs_cons_default fuel ws (.br 0) [] h_nb h_nh rfl]
  simp only [evalInstr, evalInstrs]

/-- The rustc body `pref ++ [br_if 1] ++ body2 ++ [br 0]` with
    straight-line `pref` and `body2` continues or exits through
    `br_if 1`: the prefix keeps the flags clean, the `br_if 1` either
    falls through (then `body2` keeps the flags and `br 0` arms the
    continue) or fires (then `body2 ++ [br 0]` is a no-op under the
    `branchTarget` short-circuit and the exit target is `some 1`).
    Either way the body never halts. -/
theorem blockWhileBody_continuesOrExits1
    {fuel : Nat} {pref body2 : List WasmInstr}
    (h_pref : StraightLineInstrs pref) (h_body2 : StraightLineInstrs body2) :
    BodyContinuesOrExits1 fuel (pref ++ [.brIf 1] ++ body2 ++ [.br 0]) := by
  intro st st' h_nb h_nh hw
  simp only [List.append_assoc, List.cons_append, List.nil_append] at hw
  obtain ⟨ws_m, _, h_mb, h_mh, hw_br⟩ :=
    evalInstrs_straightLine_append h_pref h_nb h_nh hw
  rw [evalInstrs_cons_default fuel ws_m (.brIf 1) (body2 ++ [WasmInstr.br 0]) h_mb h_mh rfl]
    at hw_br
  cases he : evalInstr ws_m (.brIf 1) with
  | none => rw [he] at hw_br; exact Option.noConfusion hw_br
  | some ws1 =>
      rw [he] at hw_br
      simp only at hw_br
      obtain ⟨c, rest, _, h_out⟩ := evalInstr_brIf_shape_pub he
      rcases h_out with ⟨_, h_eq⟩ | ⟨_, h_eq⟩
      · -- Fell through: `body2` keeps the flags, `br 0` arms the continue.
        subst h_eq
        obtain ⟨ws_m2, _, h_mb2, h_mh2, hw_br0⟩ :=
          evalInstrs_straightLine_append (ws := { ws_m with stack := rest }) h_body2 h_mb h_mh hw_br
        rw [br0_in_loop_continues h_mb2 h_mh2, Option.some.injEq] at hw_br0
        subst hw_br0
        exact ⟨h_mh2, Or.inl rfl⟩
      · -- Fired: the rest is a no-op under the short-circuit.
        subst h_eq
        rw [evalInstrs_branchTarget_some fuel _ _ 1 rfl, Option.some.injEq] at hw_br
        subst hw_br
        exact ⟨h_mh, Or.inr rfl⟩

end Quanta.Wasm
