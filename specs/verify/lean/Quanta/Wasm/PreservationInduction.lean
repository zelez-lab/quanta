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
* `localSet _` / `localTee _` — index-only; the cons theorems carry
  no extra logical precondition.
* Eight buffer-pattern-free `i32` binops — `i32Sub`, `i32Mul`,
  `i32And`, `i32Or`, `i32Xor`, `i32ShrU`, `i32DivU`, `i32RemU`.
  (`i32Add` and `i32Shl` need `h_no_buf`; excluded.)
* Six unsigned `i32` comparisons — `i32Eq`, `i32Ne`, `i32LtU`,
  `i32LeU`, `i32GtU`, `i32GeU`.

Total: 19 of the 30 cons-case theorems in `PreservationList`. The
remaining 11 carry per-instruction preconditions (`h_no_buf`,
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
    `halted = false`, `kst.broke = false`).

    Note: `localSet _` and `localTee _` are also closed in this
    sense — their cons theorems carry an index `i` as data but
    impose no extra logical precondition. -/
def closedInstr : WasmInstr → Bool
  | .nop          => true
  | .i32Const _   => true
  | .drop         => true
  | .localSet _   => true
  | .localTee _   => true
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
      | localSet idx =>
          exact preservation_evalInstrs_cons_localSet fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke idx rest preservation_rest
            ws' s' ops hw hl
      | localTee idx =>
          exact preservation_evalInstrs_cons_localTee fuel frames ws s kst layout R
            h_no_branch h_no_halt h_kst_no_broke idx rest preservation_rest
            ws' s' ops hw hl
      -- All non-closed-shape instructions: closedInstr returns false,
      -- so h_head_closed contradicts via `simp [closedInstr] at h_head_closed`.
      | localGet _   => simp [closedInstr] at h_head_closed
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

-- ════════════════════════════════════════════════════════════════════
-- L3.1: state-aware recognizer (closedInstrAt)
--
-- The state-free `closedInstr` recognizer rules out `localGet i`
-- entirely because the buffer-slot vs non-buffer-slot arm choice
-- depends on `s.lookupBufferSlot i`. The state-aware variant
-- `closedInstrAt s i` consults the LowerState to admit `localGet i`
-- when the local is non-buffer (`lookupBufferSlot = none`).
--
-- The recognizer's only state-dependent fact is `lookupBufferSlot`,
-- which reads `s.bufferSlots` only. The list-level invariant
-- `lowerInstrs_preserves_bufferSlots_default` (in `PreservationList`)
-- shows `bufferSlots` is preserved across every successful lowering
-- of a non-structured-control list. Combined, `closedInstrAt`'s
-- result is preserved across every closed-shape step — the lift
-- needed to thread the recognizer through a recursive proof.
--
-- This module ships the recognizer + lift lemmas + the bridge from
-- the existing state-free recognizer. The recursive proof that
-- *uses* the state-aware recognizer (extending the skeleton to
-- dispatch `localGet`, `i32Add`, `i32Shl` arms) is L3.2 / L3.3 —
-- requires either refactoring the cons theorems' `preservation_rest`
-- Pi-binder to expose mid-state `bufferSlots`-equality, or
-- reconstructing the chain at the skeleton level via the per-op
-- preservation theorems directly.
-- ════════════════════════════════════════════════════════════════════

/-- State-aware extension of `closedInstr`. Admits `localGet i` to
    the closed set when `s.lookupBufferSlot i = none` (the non-buffer
    arm of `lowerInstr`). All other arms delegate to `closedInstr`. -/
def closedInstrAt (s : LowerState) : WasmInstr → Bool
  | .localGet i =>
      match s.lookupBufferSlot i with
      | none   => true
      | some _ => false
  | other => closedInstr other

/-- State-aware closed-list. -/
def closedInstrsAt (s : LowerState) : List WasmInstr → Bool
  | []        => true
  | i :: rest => closedInstrAt s i && closedInstrsAt s rest

theorem closedInstrsAt_cons {s : LowerState} {i : WasmInstr} {rest : List WasmInstr} :
    closedInstrsAt s (i :: rest) = (closedInstrAt s i && closedInstrsAt s rest) := rfl

theorem closedInstrsAt_head {s : LowerState} {i : WasmInstr} {rest : List WasmInstr}
    (h : closedInstrsAt s (i :: rest) = true) : closedInstrAt s i = true :=
  (Bool.and_eq_true _ _).mp h |>.left

theorem closedInstrsAt_tail {s : LowerState} {i : WasmInstr} {rest : List WasmInstr}
    (h : closedInstrsAt s (i :: rest) = true) : closedInstrsAt s rest = true :=
  (Bool.and_eq_true _ _).mp h |>.right

/-- `closedInstrAt` is invariant along any `bufferSlots`-preserving
    step. Used to lift the recognizer's witness from the entry state
    to a mid-state after some closed-shape execution. -/
theorem closedInstrAt_of_bufferSlots_eq
    {s s' : LowerState} (h : s'.bufferSlots = s.bufferSlots) :
    ∀ i, closedInstrAt s' i = closedInstrAt s i := by
  intro i
  cases i with
  | localGet j =>
      show (match s'.lookupBufferSlot j with
            | none => true | some _ => false)
            = (match s.lookupBufferSlot j with
            | none => true | some _ => false)
      rw [LowerState.lookupBufferSlot_of_bufferSlots_eq j h]
  | _ => rfl

/-- List-level lift of `closedInstrAt_of_bufferSlots_eq`. -/
theorem closedInstrsAt_of_bufferSlots_eq
    {s s' : LowerState} (h : s'.bufferSlots = s.bufferSlots) :
    ∀ instrs, closedInstrsAt s' instrs = closedInstrsAt s instrs := by
  intro instrs
  induction instrs with
  | nil => rfl
  | cons i rest ih =>
      show (closedInstrAt s' i && closedInstrsAt s' rest)
            = (closedInstrAt s i && closedInstrsAt s rest)
      rw [closedInstrAt_of_bufferSlots_eq h, ih]

/-- Bridge: every `closedInstr`-recognized instruction is also
    `closedInstrAt`-recognized, for any state. Used to transport an
    existing `closedInstrs instrs = true` witness into the state-
    aware predicate, opening the recognizer up to extensions without
    breaking existing call sites. -/
theorem closedInstrAt_of_closedInstr
    {s : LowerState} {i : WasmInstr} (h : closedInstr i = true) :
    closedInstrAt s i = true := by
  cases i with
  | localGet _ => simp [closedInstr] at h
  | nop          => exact h
  | i32Const _   => exact h
  | drop         => exact h
  | localSet _   => exact h
  | localTee _   => exact h
  | i32Sub       => exact h
  | i32Mul       => exact h
  | i32And       => exact h
  | i32Or        => exact h
  | i32Xor       => exact h
  | i32ShrU      => exact h
  | i32DivU      => exact h
  | i32RemU      => exact h
  | i32Eq        => exact h
  | i32Ne        => exact h
  | i32LtU       => exact h
  | i32LeU       => exact h
  | i32GtU       => exact h
  | i32GeU       => exact h
  -- The remaining arms aren't in the closed set; `simp` on closedInstr
  -- yields False from `h`.
  | i64Const _   => simp [closedInstr] at h
  | f32Const _   => simp [closedInstr] at h
  | f64Const _   => simp [closedInstr] at h
  | i32Add       => simp [closedInstr] at h
  | i32DivS      => simp [closedInstr] at h
  | i32RemS      => simp [closedInstr] at h
  | i32Shl       => simp [closedInstr] at h
  | i32ShrS      => simp [closedInstr] at h
  | i32LtS       => simp [closedInstr] at h
  | i32GtS       => simp [closedInstr] at h
  | i32LeS       => simp [closedInstr] at h
  | i32GeS       => simp [closedInstr] at h
  | i32Eqz       => simp [closedInstr] at h
  | f32Add | f32Sub | f32Mul | f32Div =>
      all_goals simp [closedInstr] at h
  | f32Eq | f32Ne | f32Lt | f32Gt | f32Le | f32Ge =>
      all_goals simp [closedInstr] at h
  | f32Neg | f32Abs | f32Sqrt | f32Min | f32Max =>
      all_goals simp [closedInstr] at h
  | i32WrapI64 | f32ConvertI32S | f32ConvertI32U
  | i32TruncF32S | i32TruncF32U
  | f32ReinterpretI32 | i32ReinterpretF32 =>
      all_goals simp [closedInstr] at h
  | i32Load _ _ | i32Store _ _ | f32Load _ _ | f32Store _ _
  | i32Load8U _ _ | i32Load8S _ _ | i32Store8 _ _ =>
      all_goals simp [closedInstr] at h
  | block _ | wloop _ | wif _ | welse | wend =>
      all_goals simp [closedInstr] at h
  | br _ | brIf _ | wreturn =>
      all_goals simp [closedInstr] at h
  | call _ | wselect | unreachable | unsupported _ =>
      all_goals simp [closedInstr] at h

/-- List-level bridge. -/
theorem closedInstrsAt_of_closedInstrs
    {s : LowerState} {instrs : List WasmInstr}
    (h : closedInstrs instrs = true) :
    closedInstrsAt s instrs = true := by
  induction instrs with
  | nil => rfl
  | cons i rest ih =>
      have h_head : closedInstr i = true := closedInstrs_head h
      have h_rest : closedInstrs rest = true := closedInstrs_tail h
      show (closedInstrAt s i && closedInstrsAt s rest) = true
      rw [closedInstrAt_of_closedInstr h_head, ih h_rest]
      rfl

-- ════════════════════════════════════════════════════════════════════
-- L3.2: eval-side branch/halt preservation for closed-shape instrs
--
-- Every closed-shape `evalInstr` step (the 19 base ops + `localGet`)
-- leaves `ws.branchTarget` and `ws.halted` untouched. We extract a
-- common helper for the binI32 / cmpI32 family so the main case-
-- split stays manageable.
-- ════════════════════════════════════════════════════════════════════

/-- `binI32` only touches the operand stack; the rest of the state
    (locals, mem, branchTarget, halted) passes through. Proved via
    `binI32_some_shape`'s state-equation. -/
theorem binI32_preserves_branchTarget
    {s s' : WasmState} {op : UInt32 → UInt32 → UInt32}
    (h : binI32 op s = some s') :
    s'.branchTarget = s.branchTarget ∧ s'.halted = s.halted := by
  obtain ⟨_, _, _, _, h_s_eq⟩ := binI32_some_shape h
  refine ⟨?_, ?_⟩ <;> rw [h_s_eq]

/-- `cmpI32` mirrors `binI32_preserves_branchTarget`. -/
theorem cmpI32_preserves_branchTarget
    {s s' : WasmState} {p : UInt32 → UInt32 → Bool}
    (h : cmpI32 p s = some s') :
    s'.branchTarget = s.branchTarget ∧ s'.halted = s.halted := by
  obtain ⟨_, _, _, _, h_s_eq⟩ := cmpI32_some_shape h
  refine ⟨?_, ?_⟩ <;> rw [h_s_eq]

theorem evalInstr_closed_preserves_branchTarget
    {s s' : WasmState} {i : WasmInstr} {ls : LowerState}
    (h_closed : closedInstrAt ls i = true)
    (h : evalInstr s i = some s') :
    s'.branchTarget = s.branchTarget ∧ s'.halted = s.halted := by
  cases i with
  | nop =>
      simp [evalInstr] at h
      refine ⟨?_, ?_⟩ <;> rw [← h]
  | i32Const n =>
      simp [evalInstr, WasmState.push] at h
      refine ⟨?_, ?_⟩ <;> rw [← h]
  | localGet idx =>
      simp [evalInstr] at h
      rcases h_loc : s.getLocal idx with _ | v
      · simp [h_loc] at h
      · simp [h_loc, WasmState.push] at h
        refine ⟨?_, ?_⟩ <;> rw [← h]
  | localSet idx =>
      -- evalInstr ws .localSet idx unfolds to (pop ≫= setLocal).
      -- Both pop and setLocal preserve branchTarget/halted: pop only
      -- shortens stack, setLocal only updates locals.
      unfold evalInstr at h
      rcases h_pop : s.pop with _ | ⟨v, s1⟩
      · simp [h_pop] at h
      simp [h_pop] at h
      unfold WasmState.setLocal at h
      by_cases h_lt : idx < s1.locals.length
      · simp [h_lt] at h
        have h_s1_fields : s1.branchTarget = s.branchTarget ∧ s1.halted = s.halted := by
          unfold WasmState.pop at h_pop
          rcases hs : s.stack with _ | ⟨v', rs⟩
          · rw [hs] at h_pop; simp at h_pop
          rw [hs] at h_pop; simp at h_pop
          refine ⟨?_, ?_⟩ <;> (rw [← h_pop.2])
        refine ⟨?_, ?_⟩
        · rw [← h]; exact h_s1_fields.1
        · rw [← h]; exact h_s1_fields.2
      · simp [h_lt] at h
  | localTee idx =>
      -- evalInstr ws .localTee idx: peek top, set, push back. The
      -- final state is `srest.setLocal idx v |> .push v` which only
      -- mutates locals + stack.
      unfold evalInstr at h
      rcases h_peek : s.pop with _ | ⟨v, srest⟩
      · simp [h_peek] at h
      simp [h_peek] at h
      rcases h_set : srest.setLocal idx v with _ | s2
      · simp [h_set] at h
      simp [h_set, WasmState.push] at h
      have h_srest_fields : srest.branchTarget = s.branchTarget ∧ srest.halted = s.halted := by
        unfold WasmState.pop at h_peek
        rcases hs : s.stack with _ | ⟨v', rs⟩
        · rw [hs] at h_peek; simp at h_peek
        rw [hs] at h_peek; simp at h_peek
        refine ⟨?_, ?_⟩ <;> (rw [← h_peek.2])
      have h_s2_fields : s2.branchTarget = srest.branchTarget ∧ s2.halted = srest.halted := by
        unfold WasmState.setLocal at h_set
        by_cases h_lt : idx < srest.locals.length
        · simp [h_lt] at h_set
          refine ⟨?_, ?_⟩ <;> (rw [← h_set])
        · simp [h_lt] at h_set
      refine ⟨?_, ?_⟩
      · rw [← h]; show s2.branchTarget = s.branchTarget
        rw [h_s2_fields.1]; exact h_srest_fields.1
      · rw [← h]; show s2.halted = s.halted
        rw [h_s2_fields.2]; exact h_srest_fields.2
  | drop =>
      unfold evalInstr at h
      rcases h_pop : s.pop with _ | ⟨v, s1⟩
      · simp [h_pop] at h
      simp [h_pop] at h
      have h_s1_fields : s1.branchTarget = s.branchTarget ∧ s1.halted = s.halted := by
        unfold WasmState.pop at h_pop
        rcases hs : s.stack with _ | ⟨v', rs⟩
        · rw [hs] at h_pop; simp at h_pop
        rw [hs] at h_pop; simp at h_pop
        refine ⟨?_, ?_⟩ <;> (rw [← h_pop.2])
      refine ⟨?_, ?_⟩
      · rw [← h]; exact h_s1_fields.1
      · rw [← h]; exact h_s1_fields.2
  -- 8 buffer-pattern-free i32 binops.
  | i32Sub  => exact binI32_preserves_branchTarget (by simp [evalInstr] at h; exact h)
  | i32Mul  => exact binI32_preserves_branchTarget (by simp [evalInstr] at h; exact h)
  | i32And  => exact binI32_preserves_branchTarget (by simp [evalInstr] at h; exact h)
  | i32Or   => exact binI32_preserves_branchTarget (by simp [evalInstr] at h; exact h)
  | i32Xor  => exact binI32_preserves_branchTarget (by simp [evalInstr] at h; exact h)
  | i32ShrU => exact binI32_preserves_branchTarget (by simp [evalInstr] at h; exact h)
  | i32DivU => exact binI32_preserves_branchTarget (by simp [evalInstr] at h; exact h)
  | i32RemU => exact binI32_preserves_branchTarget (by simp [evalInstr] at h; exact h)
  -- 6 unsigned i32 cmps.
  | i32Eq   => exact cmpI32_preserves_branchTarget (by simp [evalInstr] at h; exact h)
  | i32Ne   => exact cmpI32_preserves_branchTarget (by simp [evalInstr] at h; exact h)
  | i32LtU  => exact cmpI32_preserves_branchTarget (by simp [evalInstr] at h; exact h)
  | i32LeU  => exact cmpI32_preserves_branchTarget (by simp [evalInstr] at h; exact h)
  | i32GtU  => exact cmpI32_preserves_branchTarget (by simp [evalInstr] at h; exact h)
  | i32GeU  => exact cmpI32_preserves_branchTarget (by simp [evalInstr] at h; exact h)
  -- Remaining: closedInstrAt = false contradicts h_closed.
  | i64Const _ => simp [closedInstrAt, closedInstr] at h_closed
  | f32Const _ => simp [closedInstrAt, closedInstr] at h_closed
  | f64Const _ => simp [closedInstrAt, closedInstr] at h_closed
  | i32Add => simp [closedInstrAt, closedInstr] at h_closed
  | i32DivS => simp [closedInstrAt, closedInstr] at h_closed
  | i32RemS => simp [closedInstrAt, closedInstr] at h_closed
  | i32Shl => simp [closedInstrAt, closedInstr] at h_closed
  | i32ShrS => simp [closedInstrAt, closedInstr] at h_closed
  | i32LtS => simp [closedInstrAt, closedInstr] at h_closed
  | i32GtS => simp [closedInstrAt, closedInstr] at h_closed
  | i32LeS => simp [closedInstrAt, closedInstr] at h_closed
  | i32GeS => simp [closedInstrAt, closedInstr] at h_closed
  | i32Eqz => simp [closedInstrAt, closedInstr] at h_closed
  | f32Add => simp [closedInstrAt, closedInstr] at h_closed
  | f32Sub => simp [closedInstrAt, closedInstr] at h_closed
  | f32Mul => simp [closedInstrAt, closedInstr] at h_closed
  | f32Div => simp [closedInstrAt, closedInstr] at h_closed
  | f32Eq => simp [closedInstrAt, closedInstr] at h_closed
  | f32Ne => simp [closedInstrAt, closedInstr] at h_closed
  | f32Lt => simp [closedInstrAt, closedInstr] at h_closed
  | f32Gt => simp [closedInstrAt, closedInstr] at h_closed
  | f32Le => simp [closedInstrAt, closedInstr] at h_closed
  | f32Ge => simp [closedInstrAt, closedInstr] at h_closed
  | f32Neg => simp [closedInstrAt, closedInstr] at h_closed
  | f32Abs => simp [closedInstrAt, closedInstr] at h_closed
  | f32Sqrt => simp [closedInstrAt, closedInstr] at h_closed
  | f32Min => simp [closedInstrAt, closedInstr] at h_closed
  | f32Max => simp [closedInstrAt, closedInstr] at h_closed
  | i32WrapI64 => simp [closedInstrAt, closedInstr] at h_closed
  | f32ConvertI32S => simp [closedInstrAt, closedInstr] at h_closed
  | f32ConvertI32U => simp [closedInstrAt, closedInstr] at h_closed
  | i32TruncF32S => simp [closedInstrAt, closedInstr] at h_closed
  | i32TruncF32U => simp [closedInstrAt, closedInstr] at h_closed
  | f32ReinterpretI32 => simp [closedInstrAt, closedInstr] at h_closed
  | i32ReinterpretF32 => simp [closedInstrAt, closedInstr] at h_closed
  | i32Load _ _ => simp [closedInstrAt, closedInstr] at h_closed
  | i32Store _ _ => simp [closedInstrAt, closedInstr] at h_closed
  | f32Load _ _ => simp [closedInstrAt, closedInstr] at h_closed
  | f32Store _ _ => simp [closedInstrAt, closedInstr] at h_closed
  | i32Load8U _ _ => simp [closedInstrAt, closedInstr] at h_closed
  | i32Load8S _ _ => simp [closedInstrAt, closedInstr] at h_closed
  | i32Store8 _ _ => simp [closedInstrAt, closedInstr] at h_closed
  | block _ => simp [closedInstrAt, closedInstr] at h_closed
  | wloop _ => simp [closedInstrAt, closedInstr] at h_closed
  | wif _ => simp [closedInstrAt, closedInstr] at h_closed
  | welse => simp [closedInstrAt, closedInstr] at h_closed
  | wend => simp [closedInstrAt, closedInstr] at h_closed
  | br _ => simp [closedInstrAt, closedInstr] at h_closed
  | brIf _ => simp [closedInstrAt, closedInstr] at h_closed
  | wreturn => simp [closedInstrAt, closedInstr] at h_closed
  | call _ => simp [closedInstrAt, closedInstr] at h_closed
  | wselect => simp [closedInstrAt, closedInstr] at h_closed
  | unreachable => simp [closedInstrAt, closedInstr] at h_closed
  | unsupported _ => simp [closedInstrAt, closedInstr] at h_closed

end Quanta.Wasm
