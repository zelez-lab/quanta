/-
# WASM → KernelOps strong-induction skeleton for list-level preservation

Each cons-case theorem in `Quanta.Wasm.PreservationList` takes its
`preservation_rest` (the inductive hypothesis on the tail) as an
explicit Pi-binder. This module discharges that binder by recursive
call for the **side-condition-free** instruction subset.

An instruction is "closed-shape" here when its preservation theorem
needs no precondition beyond `Refines`, `branchTarget = none`,
`halted = false`, and `kst.broke = false`. That covers:

* `nop` / `i32Const _` / `drop` — head ops empty, kst unchanged.
* Eight buffer-pattern-free `i32` binops — `i32Sub`, `i32Mul`,
  `i32And`, `i32Or`, `i32Xor`, `i32ShrU`, `i32DivU`, `i32RemU`.
  (`i32Add` and `i32Shl` need `h_no_buf`; excluded.)
* Six unsigned `i32` comparisons — `i32Eq`, `i32Ne`, `i32LtU`,
  `i32LeU`, `i32GtU`, `i32GeU`.

Total: 17 of the 30 cons-case theorems in `PreservationList`. The
remaining 13 carry per-instruction preconditions (`h_no_buf`,
`h_stack`, `h_in_bounds`, `h_layout_no_overlap`, …) that don't fit a
uniform predicate; their cons theorems remain individually applicable.

Each closed cons-case preserves `kst.broke = false` across its head
ops, so the recursion's broke-flag invariant propagates uniformly.
-/

import Quanta.Wasm.PreservationList

namespace Quanta.Wasm

open Quanta.KOps (KernelOp evalOps)
open Quanta.Semantics.Cpu

-- ════════════════════════════════════════════════════════════════════
-- closedInstr recognizer
-- ════════════════════════════════════════════════════════════════════

/-- Side-condition-free recognizer. Every instruction `i` with
    `closedInstr i = true` has a cons-case theorem in
    `PreservationList` whose only preconditions are the four
    standard ones (`Refines`, `branchTarget = none`,
    `halted = false`, `kst.broke = false`). -/
def closedInstr : WasmInstr → Bool
  | .nop          => true
  | .i32Const _   => true
  | .drop         => true
  | .i32Sub       => true
  | .i32Mul       => true
  | .i32And       => true
  | .i32Or        => true
  | .i32Xor       => true
  | .i32ShrU      => true
  | .i32DivU      => true
  | .i32RemU      => true
  | .i32Eq        => true
  | .i32Ne        => true
  | .i32LtU       => true
  | .i32LeU       => true
  | .i32GtU       => true
  | .i32GeU       => true
  | _             => false

/-- Closed-shape list: every element is `closedInstr`. -/
def closedInstrs : List WasmInstr → Bool
  | []          => true
  | i :: rest   => closedInstr i && closedInstrs rest

theorem closedInstrs_cons {i : WasmInstr} {rest : List WasmInstr} :
    closedInstrs (i :: rest) = (closedInstr i && closedInstrs rest) := rfl

theorem closedInstrs_head {i : WasmInstr} {rest : List WasmInstr}
    (h : closedInstrs (i :: rest) = true) : closedInstr i = true :=
  (Bool.and_eq_true _ _).mp h |>.left

theorem closedInstrs_tail {i : WasmInstr} {rest : List WasmInstr}
    (h : closedInstrs (i :: rest) = true) : closedInstrs rest = true :=
  (Bool.and_eq_true _ _).mp h |>.right

-- ════════════════════════════════════════════════════════════════════
-- preservation_evalInstrs_main — strong induction over instrs
-- ════════════════════════════════════════════════════════════════════

/-- **Strong-induction skeleton** for the closed-shape subset.

    For any `instrs : List WasmInstr` with `closedInstrs instrs =
    true`, the per-cons-case theorems compose recursively: the head
    dispatches to its dedicated `preservation_evalInstrs_cons_*`,
    and the tail's preservation is supplied by the recursive call
    as `preservation_rest`.

    Discharges the IH-on-rest Pi-binder that every cons theorem in
    `PreservationList` currently exposes. -/
theorem preservation_evalInstrs_main
    (fuel : Nat) (frames : List FrameKind)
    (instrs : List WasmInstr) (h_closed : closedInstrs instrs = true)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws instrs = some ws')
    (hl : lowerInstrs fuel frames s instrs = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  induction instrs generalizing ws s kst ws' s' ops with
  | nil =>
      obtain ⟨kst', h_eval, R'⟩ :=
        preservation_evalInstrs_nil fuel frames ws s kst layout R
          ws' s' ops hw hl
      exact ⟨kst', 0, h_eval, R'⟩
  | cons i rest ih =>
      have h_rest_closed : closedInstrs rest = true := closedInstrs_tail h_closed
      have h_head_closed : closedInstr i = true := closedInstrs_head h_closed
      -- The IH on `rest` becomes the cons theorems' `preservation_rest`.
      have preservation_rest :
          ∀ {ws_mid : WasmState} {s_mid : LowerState}
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
            Refines ws'_mid s'_mid kst'_mid layout :=
        fun {ws_mid s_mid kst_mid} R_mid h_nb_mid h_nh_mid h_kb_mid
            {ws'_mid s'_mid postOps} hw_mid hl_mid =>
          ih h_rest_closed ws_mid s_mid kst_mid R_mid h_nb_mid h_nh_mid h_kb_mid
            ws'_mid s'_mid postOps hw_mid hl_mid
      -- Dispatch on the head's closed-shape.
      cases i with
      | nop =>
          exact preservation_evalInstrs_cons_nop fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke rest preservation_rest
            ws' s' ops hw hl
      | i32Const n =>
          exact preservation_evalInstrs_cons_i32Const fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke n rest preservation_rest
            ws' s' ops hw hl
      | drop =>
          exact preservation_evalInstrs_cons_drop fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke rest preservation_rest
            ws' s' ops hw hl
      | i32Sub =>
          exact preservation_evalInstrs_cons_i32Sub fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke rest preservation_rest
            ws' s' ops hw hl
      | i32Mul =>
          exact preservation_evalInstrs_cons_i32Mul fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke rest preservation_rest
            ws' s' ops hw hl
      | i32And =>
          exact preservation_evalInstrs_cons_i32And fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke rest preservation_rest
            ws' s' ops hw hl
      | i32Or =>
          exact preservation_evalInstrs_cons_i32Or fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke rest preservation_rest
            ws' s' ops hw hl
      | i32Xor =>
          exact preservation_evalInstrs_cons_i32Xor fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke rest preservation_rest
            ws' s' ops hw hl
      | i32ShrU =>
          exact preservation_evalInstrs_cons_i32ShrU fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke rest preservation_rest
            ws' s' ops hw hl
      | i32DivU =>
          exact preservation_evalInstrs_cons_i32DivU fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke rest preservation_rest
            ws' s' ops hw hl
      | i32RemU =>
          exact preservation_evalInstrs_cons_i32RemU fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke rest preservation_rest
            ws' s' ops hw hl
      | i32Eq =>
          exact preservation_evalInstrs_cons_i32Eq fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke rest preservation_rest
            ws' s' ops hw hl
      | i32Ne =>
          exact preservation_evalInstrs_cons_i32Ne fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke rest preservation_rest
            ws' s' ops hw hl
      | i32LtU =>
          exact preservation_evalInstrs_cons_i32LtU fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke rest preservation_rest
            ws' s' ops hw hl
      | i32LeU =>
          exact preservation_evalInstrs_cons_i32LeU fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke rest preservation_rest
            ws' s' ops hw hl
      | i32GtU =>
          exact preservation_evalInstrs_cons_i32GtU fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke rest preservation_rest
            ws' s' ops hw hl
      | i32GeU =>
          exact preservation_evalInstrs_cons_i32GeU fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke rest preservation_rest
            ws' s' ops hw hl
      -- All non-closed-shape instructions: closedInstr returns false,
      -- so h_head_closed contradicts via `simp [closedInstr] at h_head_closed`.
      | localGet _   => simp [closedInstr] at h_head_closed
      | localSet _   => simp [closedInstr] at h_head_closed
      | localTee _   => simp [closedInstr] at h_head_closed
      | i64Const _   => simp [closedInstr] at h_head_closed
      | f32Const _   => simp [closedInstr] at h_head_closed
      | f64Const _   => simp [closedInstr] at h_head_closed
      | i32Add       => simp [closedInstr] at h_head_closed
      | i32DivS      => simp [closedInstr] at h_head_closed
      | i32RemS      => simp [closedInstr] at h_head_closed
      | i32Shl       => simp [closedInstr] at h_head_closed
      | i32ShrS      => simp [closedInstr] at h_head_closed
      | i32LtS       => simp [closedInstr] at h_head_closed
      | i32GtS       => simp [closedInstr] at h_head_closed
      | i32LeS       => simp [closedInstr] at h_head_closed
      | i32GeS       => simp [closedInstr] at h_head_closed
      | i32Eqz       => simp [closedInstr] at h_head_closed
      | f32Add       => simp [closedInstr] at h_head_closed
      | f32Sub       => simp [closedInstr] at h_head_closed
      | f32Mul       => simp [closedInstr] at h_head_closed
      | f32Div       => simp [closedInstr] at h_head_closed
      | f32Eq        => simp [closedInstr] at h_head_closed
      | f32Ne        => simp [closedInstr] at h_head_closed
      | f32Lt        => simp [closedInstr] at h_head_closed
      | f32Gt        => simp [closedInstr] at h_head_closed
      | f32Le        => simp [closedInstr] at h_head_closed
      | f32Ge        => simp [closedInstr] at h_head_closed
      | f32Neg       => simp [closedInstr] at h_head_closed
      | f32Abs       => simp [closedInstr] at h_head_closed
      | f32Sqrt      => simp [closedInstr] at h_head_closed
      | f32Min       => simp [closedInstr] at h_head_closed
      | f32Max       => simp [closedInstr] at h_head_closed
      | i32WrapI64   => simp [closedInstr] at h_head_closed
      | f32ConvertI32S => simp [closedInstr] at h_head_closed
      | f32ConvertI32U => simp [closedInstr] at h_head_closed
      | i32TruncF32S => simp [closedInstr] at h_head_closed
      | i32TruncF32U => simp [closedInstr] at h_head_closed
      | f32ReinterpretI32 => simp [closedInstr] at h_head_closed
      | i32ReinterpretF32 => simp [closedInstr] at h_head_closed
      | i32Load _ _    => simp [closedInstr] at h_head_closed
      | i32Store _ _   => simp [closedInstr] at h_head_closed
      | f32Load _ _    => simp [closedInstr] at h_head_closed
      | f32Store _ _   => simp [closedInstr] at h_head_closed
      | i32Load8U _ _  => simp [closedInstr] at h_head_closed
      | i32Load8S _ _  => simp [closedInstr] at h_head_closed
      | i32Store8 _ _  => simp [closedInstr] at h_head_closed
      | block _      => simp [closedInstr] at h_head_closed
      | wloop _      => simp [closedInstr] at h_head_closed
      | wif _        => simp [closedInstr] at h_head_closed
      | welse        => simp [closedInstr] at h_head_closed
      | wend         => simp [closedInstr] at h_head_closed
      | br _         => simp [closedInstr] at h_head_closed
      | brIf _       => simp [closedInstr] at h_head_closed
      | wreturn      => simp [closedInstr] at h_head_closed
      | call _       => simp [closedInstr] at h_head_closed
      | wselect      => simp [closedInstr] at h_head_closed
      | unreachable  => simp [closedInstr] at h_head_closed
      | unsupported _ => simp [closedInstr] at h_head_closed

end Quanta.Wasm
