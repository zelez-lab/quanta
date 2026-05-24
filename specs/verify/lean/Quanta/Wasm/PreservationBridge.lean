/-
# Bridge-augmented per-op cons preservation (step 059 L8.5)

The per-op cons theorems in `Quanta.Wasm.PreservationList` (33 closed
theorems as of `1c5cdec`) prove:

  ∃ kst' F, evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout

L8.5 (the bridging invariant `body_branchTarget_implies_IR_broke`)
needs two additional output clauses on every per-op cons theorem:

  (∀ d, ws'.branchTarget = some d → kst'.broke = true)  -- the bridge
  (ws'.branchTarget = none → kst'.broke = false)         -- the inverse

This module follows the two-layer API (`l8_5_scoping.md` §5 R1
mitigation, §8 step 3+4): every existing theorem stays untouched; this
module adds a `_bridge` variant per theorem that

1. takes a stronger IH-on-rest carrying the same two clauses, and
2. produces the two clauses in its conclusion.

For non-control ops (head doesn't touch `branchTarget` / `broke` /
`halted`), the bridge clauses on `ws'` come straight from the bridge
IH applied to the recursion on `rest` — no need to invoke the
non-bridge theorem at all. The cons-default reductions on the WASM
and lowering sides are duplicated here to keep this module
independent of `PreservationList`'s private helpers.

Status (this commit): `cons_nop_bridge` ships first — establishes the
direct-from-IH pattern for non-control ops. Sessions 2+ fill the
remaining 32 per-op theorems.
-/

import Quanta.Wasm.PreservationList

namespace Quanta.Wasm

open Quanta.KOps (KernelOp evalOps regLookup)
open Quanta.Semantics.Cpu

-- ════════════════════════════════════════════════════════════════════
-- Bundled bridge output predicate
-- ════════════════════════════════════════════════════════════════════

/-- The two correspondence clauses linking `ws'.branchTarget` to
    `kst'.broke`. Used both as the IH-on-rest's strengthened
    conclusion and as the new theorem's strengthened conclusion. -/
@[reducible] def BridgeClauses
    (ws' : WasmState) (kst' : Quanta.KOps.State) : Prop :=
  (∀ d, ws'.branchTarget = some d → kst'.broke = true) ∧
  (ws'.branchTarget = none → kst'.broke = false)

-- ════════════════════════════════════════════════════════════════════
-- `nop :: rest` — bridge variant
-- ════════════════════════════════════════════════════════════════════

/-- Bridge-augmented `cons_nop`. `nop` is the simplest case: both
    sides reduce to the recursion on `rest` with the input state
    unchanged, so the bridge IH applied to `rest` yields the full
    conclusion (existence, Refines, and both bridge clauses). -/
theorem preservation_evalInstrs_cons_nop_bridge
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest_bridge : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        (_h_kst_no_broke_mid : kst_mid.broke = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout ∧
        BridgeClauses ws'_mid kst'_mid)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.nop :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.nop :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧
      Refines ws' s' kst' layout ∧
      BridgeClauses ws' kst' := by
  -- Reduce lowering side to the recursion on `rest` (replicates the
  -- non-bridge proof's hl' step, including the `bind pure` collapse
  -- that the private helper `cons_default_lowerInstrs_collapse_empty_head`
  -- handles in `PreservationList.lean`).
  have hl' : lowerInstrs fuel frames s rest = some (s', ops) := by
    rw [lowerInstrs_cons_default fuel frames s .nop rest rfl] at hl
    simp only [lowerInstr, Option.bind_eq_bind, Option.some_bind,
               List.nil_append] at hl
    cases h_eq : lowerInstrs fuel frames s rest with
    | none => rw [h_eq] at hl; simp only [Option.none_bind] at hl; exact hl
    | some pair =>
        rw [h_eq] at hl
        rcases pair with ⟨s_out, ops_out⟩
        simp only [Option.some_bind, pure] at hl
        exact hl
  -- Reduce eval side to the recursion on `rest`.
  have hw' : evalInstrs fuel ws rest = some ws' := by
    rw [evalInstrs_cons_default fuel ws .nop rest h_no_branch h_no_halt rfl] at hw
    simp only [evalInstr] at hw
    exact hw
  -- The bridge IH directly produces the full conclusion.
  exact preservation_rest_bridge R h_no_branch h_no_halt h_kst_no_broke hw' hl'

end Quanta.Wasm
