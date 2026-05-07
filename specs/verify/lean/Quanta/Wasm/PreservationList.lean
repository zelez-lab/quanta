/-
# WASM → KernelOps list-level preservation theorems (step 059, slice 5c)

The 28 closed theorems in `Quanta.Wasm.Preservation` are **per-
instruction** — they pair `evalInstr s i` with `lowerInstr s i` for
a single `i`. Slices 5a + 5b moved control-flow handling out of the
per-op layer into `evalInstrs` / `lowerInstrs` (the structured-
control arms recurse on inner bodies extracted by
`Quanta.Wasm.Structured.splitAt{End,ElseOrEnd}`), so the matching
preservation theorems are inherently **list-level**.

The pattern is the same as the per-op layer:
* WASM evaluator: `evalInstrs fuel ws instrs = some ws'`.
* Lowering pass: `lowerInstrs fuel frames s instrs = some (s', ops)`.
* Refinement: `Refines ws s kst layout`.
* Branch / halt are not in `Refines`, so each theorem requires the
  pre-state to be `branchTarget = none ∧ halted = false`.
* Conclusion: `∃ kst' F, evalOps F kst ops = some kst' ∧ Refines
  ws' s' kst' layout`.

This module starts with the easy cases:
* The empty-list case (trivial — both sides return immediately).
* `br depth` to a Loop frame at depth 0 — emits no IR; the new
  `branchTarget` on `ws'` lifts through `Refines` because the
  refinement components don't see it.

More cases (block / wif / wloop / brIf, plus the cross-Loop break
arm of `br`) land as separate theorems below as proofs come online.
-/

import Quanta.Wasm.Preservation

namespace Quanta.Wasm

open Quanta.KOps (KernelOp evalOps)

-- ════════════════════════════════════════════════════════════════════
-- Short-circuit lemmas for `evalInstrs`
-- ════════════════════════════════════════════════════════════════════

/-- `evalInstrs` returns the state untouched on a branchTarget-set
    pre-state. Holds for any instruction list (empty or non-empty)
    because the head check fires before any `evalInstr` is called.
    Used wherever a `br` / `brIf` lowering arm leaves the WASM-side
    state with `branchTarget` set, then the surrounding context's
    `evalInstrs` continuation must produce the same final state. -/
theorem evalInstrs_branchTarget_some
    (fuel : Nat) (ws : WasmState) (instrs : List WasmInstr) (d : Nat)
    (h : ws.branchTarget = some d) :
    evalInstrs fuel ws instrs = some ws := by
  cases instrs with
  | nil => simp [evalInstrs]
  | cons i rest =>
    unfold evalInstrs
    simp [h]

-- ════════════════════════════════════════════════════════════════════
-- Empty-list case
-- ════════════════════════════════════════════════════════════════════

/-- Empty input: both `evalInstrs` and `lowerInstrs` return the
    state untouched and emit nothing. The `Refines` bundle survives
    by reflexivity. -/
theorem preservation_evalInstrs_nil
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws [] = some ws')
    (hl : lowerInstrs fuel frames s [] = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  simp [evalInstrs] at hw
  simp [lowerInstrs] at hl
  obtain ⟨hs_eq, hops_eq⟩ := hl
  refine ⟨kst, ?_, ?_⟩
  · subst hops_eq; simp [evalOps]
  · subst hw hs_eq; exact R

-- ════════════════════════════════════════════════════════════════════
-- `br depth` to a Loop frame at depth 0
-- ════════════════════════════════════════════════════════════════════

/-- `br 0` to a Loop frame: lowering emits no IR (structured-Loop
    auto-continues at fall-through); WASM-side sets
    `branchTarget := some 0`. The recursive `evalInstrs` call on
    `rest` short-circuits on `branchTarget.isSome`, so `ws'` is just
    `ws` with the new `branchTarget`. `Refines` is preserved
    because none of its components inspect `branchTarget`.

    Precondition `frames.get? 0 = some .loopK` is what the lowering
    arm matches; without it, lowering would either refuse or emit a
    different shape (Break / refusal). -/
theorem preservation_br_loop_zero
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (rest : List WasmInstr)
    (h_target : frames.get? 0 = some .loopK)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.br 0 :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.br 0 :: rest) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- Lowering side: br arm with depth=0, frames.get? 0 = some .loopK
  -- reduces to `(s, [])`.
  have h_lower : lowerInstrs fuel frames s (.br 0 :: rest) = some (s, []) := by
    simp only [lowerInstrs, h_target, ↓reduceIte]
  rw [h_lower] at hl
  have hl' : (s, ([] : List KernelOp)) = (s', ops) := (Option.some.injEq _ _).mp hl
  have hs_eq : s = s' := congrArg Prod.fst hl'
  have hops_eq : ([] : List KernelOp) = ops := congrArg Prod.snd hl'
  -- Eval side: br 0 sets branchTarget := some 0 on the head's
  -- evalInstr; the continuation on `rest` short-circuits via
  -- `evalInstrs_branchTarget_some`. We compute `ws'` step by step
  -- without rewriting `ws.halted` away (so the post-evalInstr state
  -- still reads `halted := ws.halted`, matching `ws_post`).
  have h_cond : (ws.halted || ws.branchTarget.isSome) = false := by
    rw [h_no_halt, h_no_branch]; rfl
  let ws_post : WasmState := { ws with branchTarget := some 0 }
  have h_post_branch : ws_post.branchTarget = some 0 := rfl
  have h_evalInstr : evalInstr ws (.br 0) = some ws_post := rfl
  have h_step : evalInstrs fuel ws (.br 0 :: rest)
                  = evalInstrs fuel ws_post rest := by
    conv => lhs; unfold evalInstrs
    rw [h_cond]
    simp [h_evalInstr]
  rw [h_step] at hw
  rw [evalInstrs_branchTarget_some fuel ws_post rest 0 h_post_branch] at hw
  have hws'_eq : ws' = ws_post := ((Option.some.injEq _ _).mp hw).symm
  refine ⟨kst, ?_, ?_⟩
  · rw [← hops_eq]; simp [evalOps]
  · rw [← hs_eq, hws'_eq]
    -- Goal: Refines ws_post s kst layout. ws_post differs from ws only
    -- in branchTarget, which Refines doesn't see.
    refine ⟨R.stk, R.locs, R.fresh, R.aliasFree, R.injLocals, R.heapRefines⟩

end Quanta.Wasm
