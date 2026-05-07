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
import Quanta.Wasm.PreservationFuel

namespace Quanta.Wasm

open Quanta.KOps (KernelOp evalOps)
open Quanta.Semantics.Cpu

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

-- ════════════════════════════════════════════════════════════════════
-- `br depth` with cross-Loop break (emits [.breakOp])
-- ════════════════════════════════════════════════════════════════════

/-- `br depth` to a non-Loop frame with a `loopK` between top and
    target: lowering emits `[KernelOp.breakOp]` so the cond register
    stays inside the surrounding Loop body. WASM-side semantics is
    the same as any `br`: set `branchTarget := some depth`, then
    short-circuit on `rest`.

    `Refines` is preserved because none of its components looks at
    either `WasmState.branchTarget` (set by br) or `KOps.State.broke`
    (set by `evalOp .breakOp`). -/
theorem preservation_br_break_nonLoop
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (depth : Nat) (rest : List WasmInstr)
    (kind : FrameKind) (h_kind_ne_loop : kind ≠ .loopK)
    (h_target : frames.get? depth = some kind)
    (h_loop_above : hasLoopAbove frames depth = true)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.br depth :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.br depth :: rest) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- Lowering side: br arm with `frames.get? depth = some kind` (kind ≠ loopK)
  -- and `hasLoopAbove = true` selects the `[.breakOp]` arm.
  have h_lower : lowerInstrs fuel frames s (.br depth :: rest)
                  = some (s, [KernelOp.breakOp]) := by
    cases kind with
    | block => simp only [lowerInstrs, h_target, h_loop_above, ↓reduceIte]
    | wif   => simp only [lowerInstrs, h_target, h_loop_above, ↓reduceIte]
    | loopK => exact (h_kind_ne_loop rfl).elim
  rw [h_lower] at hl
  have hl' : (s, [KernelOp.breakOp]) = (s', ops) :=
    (Option.some.injEq _ _).mp hl
  have hs_eq : s = s' := congrArg Prod.fst hl'
  have hops_eq : [KernelOp.breakOp] = ops := congrArg Prod.snd hl'
  -- Eval side: `br depth` step + short-circuit on rest.
  have h_cond : (ws.halted || ws.branchTarget.isSome) = false := by
    rw [h_no_halt, h_no_branch]; rfl
  let ws_post : WasmState := { ws with branchTarget := some depth }
  have h_post_branch : ws_post.branchTarget = some depth := rfl
  have h_evalInstr : evalInstr ws (.br depth) = some ws_post := rfl
  have h_step : evalInstrs fuel ws (.br depth :: rest)
                  = evalInstrs fuel ws_post rest := by
    conv => lhs; unfold evalInstrs
    rw [h_cond]
    simp [h_evalInstr]
  rw [h_step] at hw
  rw [evalInstrs_branchTarget_some fuel ws_post rest depth h_post_branch] at hw
  have hws'_eq : ws' = ws_post := ((Option.some.injEq _ _).mp hw).symm
  -- KOps side: `[.breakOp]` runs to `{ kst with broke := true }`.
  let kst_post : Quanta.KOps.State := { kst with broke := true }
  refine ⟨kst_post, ?_, ?_⟩
  · rw [← hops_eq]; simp [evalOps, Quanta.KOps.evalOp, kst_post]
  · rw [← hs_eq, hws'_eq]
    -- Refines lifts: Refines doesn't see branchTarget or broke. The
    -- regfile / heap / stack / locals all carry over from R.
    refine ⟨R.stk, R.locs, R.fresh, R.aliasFree, R.injLocals, R.heapRefines⟩

/-- `br depth` targeting an outer Loop frame (`depth ≠ 0`) with a
    Loop frame between top and target: lowering emits
    `[KernelOp.breakOp]` to escape the inner loop. Same shape as
    `preservation_br_break_nonLoop` but the target frame kind is
    `.loopK` (and `depth > 0` is required for the inner `if-else`
    to fall through to the `hasLoopAbove` check). -/
theorem preservation_br_loop_outer_break
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (depth : Nat) (rest : List WasmInstr)
    (h_depth_pos : depth ≠ 0)
    (h_target : frames.get? depth = some .loopK)
    (h_loop_above : hasLoopAbove frames depth = true)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.br depth :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.br depth :: rest) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  have h_lower : lowerInstrs fuel frames s (.br depth :: rest)
                  = some (s, [KernelOp.breakOp]) := by
    simp only [lowerInstrs, h_target, h_depth_pos, ↓reduceIte, h_loop_above]
  rw [h_lower] at hl
  have hl' : (s, [KernelOp.breakOp]) = (s', ops) :=
    (Option.some.injEq _ _).mp hl
  have hs_eq : s = s' := congrArg Prod.fst hl'
  have hops_eq : [KernelOp.breakOp] = ops := congrArg Prod.snd hl'
  have h_cond : (ws.halted || ws.branchTarget.isSome) = false := by
    rw [h_no_halt, h_no_branch]; rfl
  let ws_post : WasmState := { ws with branchTarget := some depth }
  have h_post_branch : ws_post.branchTarget = some depth := rfl
  have h_evalInstr : evalInstr ws (.br depth) = some ws_post := rfl
  have h_step : evalInstrs fuel ws (.br depth :: rest)
                  = evalInstrs fuel ws_post rest := by
    conv => lhs; unfold evalInstrs
    rw [h_cond]
    simp [h_evalInstr]
  rw [h_step] at hw
  rw [evalInstrs_branchTarget_some fuel ws_post rest depth h_post_branch] at hw
  have hws'_eq : ws' = ws_post := ((Option.some.injEq _ _).mp hw).symm
  let kst_post : Quanta.KOps.State := { kst with broke := true }
  refine ⟨kst_post, ?_, ?_⟩
  · rw [← hops_eq]; simp [evalOps, Quanta.KOps.evalOp, kst_post]
  · rw [← hs_eq, hws'_eq]
    refine ⟨R.stk, R.locs, R.fresh, R.aliasFree, R.injLocals, R.heapRefines⟩

-- ════════════════════════════════════════════════════════════════════
-- Helper: collapse the cons-default bind chain when the head's
-- per-op lowering returned `some (s, [])`.
--
-- After `rw [lowerInstrs_cons_default ...]` the goal contains a
-- residual `(lowerInstrs ... rest).bind (fun __discr => pure
-- (__discr.fst, __discr.snd))`. With Lean 4's structure-eta reduction
-- the `pure` argument is definitionally `pure __discr`, and
-- `m.bind pure = m`. The simp set below is what's needed to make
-- the bind chain disappear cleanly across both `nop` and `i32Const`
-- cases (and any future ops where the per-op `lowerInstr` returns
-- `(s, [])`).
-- ════════════════════════════════════════════════════════════════════

private theorem cons_default_lowerInstrs_collapse_empty_head
    {fuel : Nat} {frames : List FrameKind} {s s' : LowerState}
    {ops : List KernelOp} {rest : List WasmInstr}
    (h : (lowerInstrs fuel frames s rest).bind
            (fun __discr => pure (__discr.fst, __discr.snd))
          = some (s', ops)) :
    lowerInstrs fuel frames s rest = some (s', ops) := by
  cases h_eq : lowerInstrs fuel frames s rest with
  | none =>
      rw [h_eq] at h
      simp only [Option.none_bind] at h
      exact h
  | some pair =>
      rw [h_eq] at h
      rcases pair with ⟨s_out, ops_out⟩
      simp only [Option.some_bind, pure] at h
      exact h

-- ════════════════════════════════════════════════════════════════════
-- Non-control cons cases (head emits no IR)
--
-- The `nop` and `i32Const` cases are the simplest list-level cons
-- patterns: lowering's per-op step emits the empty op list, so the
-- aggregated list-level lowering is just the recursion on `rest`. The
-- WASM-side step also leaves the state unchanged (`nop`) or only
-- pushes a stack value with no register effects (`i32Const` materializes
-- only at commit time). Both reduce to "apply IH-on-rest with the same
-- mid-state R", which is the cleanest exercise of the cons-default
-- unfold lemmas in `PreservationFuel`.
--
-- The `IH-on-rest` is supplied as a Pi-type hypothesis. A future
-- strong-induction skeleton (`preservation_evalInstrs_main`) will
-- discharge that hypothesis by recursive call; for now the standalone
-- form lets us verify the cons-pattern infrastructure independently.
-- ════════════════════════════════════════════════════════════════════

/-- `nop :: rest` preservation. `nop` is a pure no-op on both sides:
    `lowerInstr s .nop = some (s, [])`, so `lowerInstrs fuel frames s
    (.nop :: rest)` reduces to `lowerInstrs fuel frames s rest`.
    `evalInstr ws .nop = some ws`, so `evalInstrs fuel ws (.nop ::
    rest)` reduces to `evalInstrs fuel ws rest`. The conclusion is
    just the IH-on-rest applied to the (unchanged) input state. -/
theorem preservation_evalInstrs_cons_nop
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.nop :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.nop :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- Lowering side: lowerInstrs fuel frames s (.nop :: rest)
  -- = lowerInstrs fuel frames s rest (since .nop emits no IR and
  -- preserves state).
  have hl' : lowerInstrs fuel frames s rest = some (s', ops) := by
    rw [lowerInstrs_cons_default fuel frames s .nop rest rfl] at hl
    simp only [lowerInstr, Option.bind_eq_bind, Option.some_bind,
               List.nil_append] at hl
    exact cons_default_lowerInstrs_collapse_empty_head hl
  -- Eval side: evalInstrs fuel ws (.nop :: rest) = evalInstrs fuel ws rest.
  have hw' : evalInstrs fuel ws rest = some ws' := by
    rw [evalInstrs_cons_default fuel ws .nop rest h_no_branch h_no_halt rfl] at hw
    simp only [evalInstr] at hw
    exact hw
  -- Apply the IH on `rest` with the unchanged state.
  exact preservation_rest R h_no_branch h_no_halt hw' hl'

/-- `i32Const n :: rest` preservation. `i32Const` emits no IR — it
    only pushes `SymVal.i32ConstSym n` onto the lowering stack, which
    encodes the WASM `wI32 (UInt32.ofNat n.toNat)` push. The lowered
    op list collapses to the lowering of `rest`, evaluated against
    `kst` (regfile untouched).

    The mid-state Refines is established via `preservation_i32Const`
    composed back into the cons-form: WASM stack gains `wI32 n`, lower
    stack gains `i32ConstSym n`, kst.rf unchanged, and Fresh /
    AliasFree / InjectiveLocals lift through the empty-regs-pushed
    SymVal. Then the IH on `rest` discharges the tail. -/
theorem preservation_evalInstrs_cons_i32Const
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (n : Int) (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Const n :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Const n :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- Lowering side: lowerInstr s (.i32Const n) = some (s.pushSym (.i32ConstSym n), [])
  -- so the cons-default arm reduces lowerInstrs to a recursive call on `rest`
  -- starting from `s.pushSym (.i32ConstSym n)`.
  let s_mid : LowerState := s.pushSym (.i32ConstSym n)
  let ws_mid : WasmState := ws.push (.wI32 (UInt32.ofNat n.toNat))
  have hl' : lowerInstrs fuel frames s_mid rest = some (s', ops) := by
    rw [lowerInstrs_cons_default fuel frames s (.i32Const n) rest rfl] at hl
    simp only [lowerInstr, Option.bind_eq_bind, Option.some_bind,
               List.nil_append] at hl
    -- After simp, hl is a bind chain on `lowerInstrs fuel frames {...} rest`
    -- where `{...}` is the unfolded `s_mid`. The helper expects the
    -- `s_mid` form, so unfold s_mid in the goal to match.
    show lowerInstrs fuel frames (s.pushSym (.i32ConstSym n)) rest = some (s', ops)
    exact cons_default_lowerInstrs_collapse_empty_head hl
  have hw' : evalInstrs fuel ws_mid rest = some ws' := by
    rw [evalInstrs_cons_default fuel ws (.i32Const n) rest h_no_branch h_no_halt rfl] at hw
    simp only [evalInstr] at hw
    show evalInstrs fuel (ws.push (.wI32 (UInt32.ofNat n.toNat))) rest = some ws'
    exact hw
  -- Build `Refines ws_mid s_mid kst layout` directly (regfile unchanged,
  -- new stack-top SymVal has empty regs so freshness / aliasFree lift).
  have R_mid : Refines ws_mid s_mid kst layout := by
    refine ⟨?_, ?_, ?_, ?_, ?_, R.heapRefines⟩
    · -- StackRefines: pushed entry is i32ConstSym n encoding wI32 n.
      refine ⟨by simp [ws_mid, s_mid, WasmState.push, LowerState.pushSym, R.stk.left], ?_⟩
      intro i v hv
      cases i with
      | zero =>
        simp [ws_mid, WasmState.push] at hv
        refine ⟨SymVal.i32ConstSym n, by simp [s_mid, LowerState.pushSym], ?_⟩
        subst hv
        simp [WasmValue.encodes]
      | succ k =>
        have hwsk : ws.stack.get? k = some v := by
          simpa [ws_mid, WasmState.push] using hv
        obtain ⟨svk, hsvk_get, henc⟩ := R.stk.right k v hwsk
        refine ⟨svk, ?_, henc⟩
        simpa [s_mid, LowerState.pushSym] using hsvk_get
    · -- LocalsRefines: localReg unchanged.
      simpa [s_mid, LowerState.pushSym] using R.locs
    · -- Fresh: nextReg unchanged; new top has empty SymVal.regs.
      refine ⟨?_, ?_⟩
      · intro sv hsv r' hr'
        simp [s_mid, LowerState.pushSym] at hsv
        rcases hsv with h_eq | h_in
        · subst h_eq; simp [SymVal.regs] at hr'
        · exact R.fresh.left sv h_in r' hr'
      · simpa [s_mid, LowerState.pushSym] using R.fresh.right
    · -- AliasFree: new top has empty regs → trivially disjoint.
      intro ir hir sv hsv
      simp [s_mid, LowerState.pushSym] at hsv ⊢
      rcases hsv with h_eq | h_in
      · subst h_eq; simp [SymVal.regs]
      · exact R.aliasFree ir (by simpa [s_mid, LowerState.pushSym] using hir) sv h_in
    · -- InjectiveLocals: localReg unchanged.
      simpa [s_mid, LowerState.pushSym] using R.injLocals
  have h_no_branch_mid : ws_mid.branchTarget = none := by
    simp [ws_mid, WasmState.push, h_no_branch]
  have h_no_halt_mid : ws_mid.halted = false := by
    simp [ws_mid, WasmState.push, h_no_halt]
  -- Apply IH on `rest` with the mid-state.
  exact preservation_rest R_mid h_no_branch_mid h_no_halt_mid hw' hl'

-- ════════════════════════════════════════════════════════════════════
-- localGet (non-buffer) cons case
--
-- First non-trivial cons-composer use: head emits a single `.copy
-- fresh stable` op. The proof structure is:
--   1. Unfold lowerInstrs / evalInstrs cons-default.
--   2. Apply preservation_localGet to get (kst_mid, h_eval, R_mid).
--   3. Derive kst_mid.broke = false via evalOps_copy_singleton_preserves_broke.
--   4. Apply IH-on-rest with R_mid.
--   5. Chain via preservation_evalInstrs_cons_compose_shallow.
--
-- The `kst.broke = false` precondition is what we need to discharge
-- the cons-composer's no-broke-mid-state requirement (`.copy`'s
-- semantics preserve broke, so kst_mid.broke = kst.broke).
-- ════════════════════════════════════════════════════════════════════

/-- `localGet i :: rest` preservation (non-buffer path). Head ops are
    `[.copy s.nextReg stable]` — a single non-control-flow op that
    preserves the `broke` flag. The cons-composer chains the head's
    per-op result with the IH-on-rest. -/
theorem preservation_evalInstrs_cons_localGet
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (i : Nat) (h_no_buf : s.lookupBufferSlot i = none)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.localGet i :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.localGet i :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- Unfold the lowering's cons-default arm.
  rw [lowerInstrs_cons_default fuel frames s (.localGet i) rest rfl] at hl
  -- Extract the per-op lowering result for `.localGet i`.
  -- lowerInstr s (.localGet i) for non-buffer path returns
  -- `(s_after, [.copy fresh stable])` where fresh = s.nextReg, stable
  -- comes from R.locs / R.injLocals via lookupLocal.
  -- We extract the head pair by case-splitting on lookupLocal.
  cases h_stable : s.lookupLocal i with
  | none =>
      -- lookupLocal failed → lowerInstr returns none → lowerInstrs returns none.
      simp only [lowerInstr, h_no_buf, h_stable, Option.bind_eq_bind,
                 Option.some_bind, Option.none_bind, LowerState.alloc,
                 LowerState.push] at hl
      exact (Option.noConfusion hl)
  | some stable =>
      -- Head pair: (s_after, ops_head) where s_after = ((s.alloc.snd).push s.nextReg)
      -- and ops_head = [.copy s.nextReg stable].
      let s_after : LowerState :=
        { s with nextReg := s.nextReg + 1,
                 stack := SymVal.reg s.nextReg .u32 :: s.stack }
      let ops_head : List KernelOp := [.copy s.nextReg stable]
      have hl_head : lowerInstr s (.localGet i) = some (s_after, ops_head) := by
        show (match s.lookupBufferSlot i with
              | some slot => some (s.pushSym (.bufferPtr slot), [])
              | none => do
                  let stable ← s.lookupLocal i
                  let (fresh, s1) := s.alloc
                  let s2 := s1.push fresh
                  pure (s2, [.copy fresh stable])) = some (s_after, ops_head)
        rw [h_no_buf, h_stable]
        rfl
      -- After cons-default unfold, hl is:
      -- (do let (s1, ops1) ← lowerInstr s (.localGet i); let (s2, ops2) ← lowerInstrs ... rest;
      --     pure (s2, ops1 ++ ops2)) = some (s', ops)
      -- Substitute hl_head and reduce.
      rw [hl_head] at hl
      simp only [Option.bind_eq_bind, Option.some_bind] at hl
      -- hl is now: (do let (s2, ops2) ← lowerInstrs fuel frames s_after rest;
      --                 pure (s2, ops_head ++ ops2)) = some (s', ops)
      -- Extract postOps from the rest's lowering.
      cases h_post : lowerInstrs fuel frames s_after rest with
      | none => simp [h_post] at hl
      | some post_pair =>
          rcases post_pair with ⟨s_post, postOps⟩
          simp [h_post] at hl
          -- hl : (s_post, ops_head ++ postOps) = (s', ops)
          rcases hl with ⟨h_s_eq, h_ops_eq⟩
          -- Now h_s_eq : s_post = s', h_ops_eq : ops_head ++ postOps = ops.
          -- Eval side: cons-default unfold + evalInstr (.localGet i) = some ws_after.
          rw [evalInstrs_cons_default fuel ws (.localGet i) rest h_no_branch h_no_halt rfl] at hw
          -- Match on locals.get? i — splitting via the structure of evalInstr ws (.localGet i).
          cases h_loc : ws.getLocal i with
          | none =>
              -- evalInstr returns none, the match branch returns none, hw : none = some ws'.
              have hw_step : evalInstr ws (.localGet i) = none := by
                show (do let v ← ws.getLocal i; pure (ws.push v)) = none
                rw [h_loc]; rfl
              rw [hw_step] at hw
              simp at hw
          | some v =>
              let ws_after : WasmState := ws.push v
              have hw_step : evalInstr ws (.localGet i) = some ws_after := by
                show (do let v ← ws.getLocal i; pure (ws.push v)) = some ws_after
                rw [h_loc]
                rfl
              rw [hw_step] at hw
              simp only at hw
              -- hw : evalInstrs fuel ws_after rest = some ws'.
              -- Now apply preservation_localGet to the head pair (single-instr level).
              obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
                preservation_localGet ws s kst layout R i h_no_buf
                  ws_after s_after ops_head
                  hw_step hl_head
              -- Derive kst_mid.broke = false via .copy preservation lemma.
              have h_mid_broke : kst_mid.broke = false := by
                have := evalOps_copy_singleton_preserves_broke h_kst_eval
                rw [this]; exact h_kst_no_broke
              -- Mid-state preconditions: branchTarget / halted unchanged through localGet.
              have h_mid_no_branch : ws_after.branchTarget = none := by
                simp [ws_after, WasmState.push, h_no_branch]
              have h_mid_no_halt : ws_after.halted = false := by
                simp [ws_after, WasmState.push, h_no_halt]
              -- Apply IH-on-rest. Returns `∃ kst'_mid F, ...` (double existential).
              obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest⟩ :=
                preservation_rest R_mid h_mid_no_branch h_mid_no_halt hw h_post
              -- Chain via cons-composer (shallow: ops_head = [.copy] is shallow loop-free).
              have h_lf : loopFree ops_head = true := by
                simp [loopFree, loopFreeOp, ops_head]
              have h_chained :
                  ∃ kst'', evalOps F_rest kst (ops_head ++ postOps) = some kst''
                    ∧ Refines ws' s_post kst'' layout :=
                preservation_evalInstrs_cons_compose_shallow
                  h_lf h_kst_eval h_mid_broke ⟨kst'_mid, h_eval_rest, R_rest⟩
              -- Bridge ops shape: ops = ops_head ++ postOps, s' = s_post.
              obtain ⟨kst'', h_eval'', R''⟩ := h_chained
              refine ⟨kst'', F_rest, ?_, ?_⟩
              · rw [← h_ops_eq]; exact h_eval''
              · rw [← h_s_eq]; exact R''

-- ════════════════════════════════════════════════════════════════════
-- i32 binary-op cons cases
--
-- Generic theorem `preservation_evalInstrs_cons_i32Bin_generic` factors
-- the cons proof for any `instr` whose lowering reduces to `lowerI32Bin
-- s op_k`. Head ops are `opsA ++ opsB ++ [.binOp …]`, where each commit
-- result is `[]` or `[.const …]`; all three sub-lists are
-- loopFreeNoBreak, so the generic broke-preservation helper discharges
-- the cons-composer's mid-state precondition.
--
-- For the 8 buffer-pattern-free ops (Sub/Mul/And/Or/Xor/ShrU/DivU/RemU),
-- `lowerInstr s instr = lowerI32Bin s op_k` is `rfl`. For i32Add (and
-- i32Shl, deferred), the buffer-pattern fast-path of `lowerI32Add` /
-- `lowerI32Shl` requires an `h_no_buf` precondition that excludes the
-- folded stack shape — the wrapper supplies that, derives the
-- equational `h_l`, and dispatches to the generic.
-- ════════════════════════════════════════════════════════════════════

/-- Generic cons preservation parametric over WASM instruction, KOps
    binop, agreement, and the equational lowering reduction
    `lowerInstr s instr = lowerI32Bin s op_k`. -/
theorem preservation_evalInstrs_cons_i32Bin_generic
    (instr : WasmInstr) (op_w : UInt32 → UInt32 → UInt32)
    (op_k : Quanta.KOps.BinOp)
    (h_w : ∀ s, evalInstr s instr = binI32 op_w s)
    (h_agree : ∀ av bv,
       Quanta.KOps.evalBinOp op_k (Quanta.KOps.Value.vU32 av)
         (Quanta.KOps.Value.vU32 bv) =
         some (Quanta.KOps.Value.vU32 (op_w av bv)))
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (h_l_eq : lowerInstr s instr = lowerI32Bin s op_k)
    (h_ns_lower : isStructuredLower instr = false)
    (h_ns_eval : isStructuredEval instr = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (instr :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (instr :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  rw [lowerInstrs_cons_default fuel frames s instr rest h_ns_lower] at hl
  cases h_head : lowerInstr s instr with
  | none =>
      rw [h_head] at hl
      simp at hl
  | some head_pair =>
      rcases head_pair with ⟨s_after, ops_head⟩
      rw [h_head] at hl
      simp only [Option.bind_eq_bind, Option.some_bind] at hl
      cases h_post : lowerInstrs fuel frames s_after rest with
      | none => simp [h_post] at hl
      | some post_pair =>
          rcases post_pair with ⟨s_post, postOps⟩
          simp [h_post] at hl
          rcases hl with ⟨h_s_eq, h_ops_eq⟩
          rw [evalInstrs_cons_default fuel ws instr rest h_no_branch h_no_halt h_ns_eval] at hw
          cases h_eval_head : evalInstr ws instr with
          | none =>
              rw [h_eval_head] at hw
              simp at hw
          | some ws_after =>
              rw [h_eval_head] at hw
              simp only at hw
              obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
                preservation_i32Bin_generic instr op_w op_k h_w h_agree
                  ws s kst layout R h_kst_no_broke
                  ws_after s_after ops_head h_l_eq
                  h_eval_head h_head
              rw [h_l_eq] at h_head
              obtain ⟨_svb, _sva, _lrest, ra, _s3, opsA, rb, s4, opsB,
                      _h_stk, hca, hcb, _h_s4_stk, _h_s4_lr, _h_s4_lt,
                      _h_nr_le, _h_s_eq_shape, h_ops_head_eq⟩ :=
                lowerI32Bin_some_shape h_head
              have h_lf_opsA : loopFreeNoBreak opsA = true :=
                commit_emits_loopFreeNoBreak hca
              have h_lf_opsB : loopFreeNoBreak opsB = true :=
                commit_emits_loopFreeNoBreak hcb
              have h_lf_binOp :
                  loopFreeNoBreak [KernelOp.binOp s4.nextReg ra rb op_k .u32] = true := rfl
              have h_lf_head : loopFreeNoBreak ops_head = true := by
                rw [h_ops_head_eq]
                simp [loopFreeNoBreak_append, h_lf_opsA, h_lf_opsB, h_lf_binOp]
              have h_lf_head_shallow : loopFree ops_head = true :=
                loopFreeNoBreak_implies_loopFree h_lf_head
              have h_mid_broke : kst_mid.broke = false :=
                evalOps_loopFreeNoBreak_preserves_broke
                  h_lf_head h_kst_no_broke h_kst_eval
              -- Mid-state branchTarget / halted: binI32 only mutates stack.
              have h_mid_no_branch : ws_after.branchTarget = none := by
                rw [h_w] at h_eval_head
                obtain ⟨_, _, _, _, h_ws_eq⟩ := binI32_some_shape h_eval_head
                rw [h_ws_eq]; simp [h_no_branch]
              have h_mid_no_halt : ws_after.halted = false := by
                rw [h_w] at h_eval_head
                obtain ⟨_, _, _, _, h_ws_eq⟩ := binI32_some_shape h_eval_head
                rw [h_ws_eq]; simp [h_no_halt]
              obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest⟩ :=
                preservation_rest R_mid h_mid_no_branch h_mid_no_halt hw h_post
              have h_chained :
                  ∃ kst'', evalOps F_rest kst (ops_head ++ postOps) = some kst''
                    ∧ Refines ws' s_post kst'' layout :=
                preservation_evalInstrs_cons_compose_shallow
                  h_lf_head_shallow h_kst_eval h_mid_broke
                  ⟨kst'_mid, h_eval_rest, R_rest⟩
              obtain ⟨kst'', h_eval'', R''⟩ := h_chained
              refine ⟨kst'', F_rest, ?_, ?_⟩
              · rw [← h_ops_eq]; exact h_eval''
              · rw [← h_s_eq]; exact R''

/-- `i32Add :: rest` (non-buffer path). Wraps the generic; supplies an
    equational `h_l` derived from the no-buffer-pattern guard. -/
theorem preservation_evalInstrs_cons_i32Add
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (h_no_buf : ∀ slot base scale rest,
      s.stack ≠ .scaledIdx base scale :: .bufferPtr slot :: rest ∧
      s.stack ≠ .bufferPtr slot :: .scaledIdx base scale :: rest)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Add :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Add :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  have h_l_eq : lowerInstr s .i32Add = lowerI32Bin s .add := by
    show lowerI32Add s = lowerI32Bin s .add
    unfold lowerI32Add
    split
    next base scale slot rest hs =>
        exact absurd hs (h_no_buf slot base scale rest).left
    next slot base scale rest hs =>
        exact absurd hs (h_no_buf slot base scale rest).right
    next => rfl
  exact preservation_evalInstrs_cons_i32Bin_generic
    .i32Add eval_u32_wrapping_add .add
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke h_l_eq rfl rfl rest
    preservation_rest ws' s' ops hw hl

/-- `i32Sub :: rest`. Lowering goes directly to `lowerI32Bin s .sub`. -/
theorem preservation_evalInstrs_cons_i32Sub
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Sub :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Sub :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_evalInstrs_cons_i32Bin_generic
    .i32Sub eval_u32_wrapping_sub .sub
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rfl rest
    preservation_rest ws' s' ops hw hl

/-- `i32Mul :: rest`. -/
theorem preservation_evalInstrs_cons_i32Mul
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Mul :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Mul :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_evalInstrs_cons_i32Bin_generic
    .i32Mul eval_u32_wrapping_mul .mul
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rfl rest
    preservation_rest ws' s' ops hw hl

/-- `i32And :: rest`. -/
theorem preservation_evalInstrs_cons_i32And
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32And :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32And :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_evalInstrs_cons_i32Bin_generic
    .i32And eval_u32_bitand .bAnd
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rfl rest
    preservation_rest ws' s' ops hw hl

/-- `i32Or :: rest`. -/
theorem preservation_evalInstrs_cons_i32Or
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Or :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Or :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_evalInstrs_cons_i32Bin_generic
    .i32Or eval_u32_bitor .bOr
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rfl rest
    preservation_rest ws' s' ops hw hl

/-- `i32Xor :: rest`. -/
theorem preservation_evalInstrs_cons_i32Xor
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Xor :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Xor :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_evalInstrs_cons_i32Bin_generic
    .i32Xor eval_u32_bitxor .bXor
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rfl rest
    preservation_rest ws' s' ops hw hl

/-- `i32ShrU :: rest`. -/
theorem preservation_evalInstrs_cons_i32ShrU
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32ShrU :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32ShrU :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_evalInstrs_cons_i32Bin_generic
    .i32ShrU (fun a b => a >>> b) .shr
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rfl rest
    preservation_rest ws' s' ops hw hl

/-- `i32DivU :: rest`. -/
theorem preservation_evalInstrs_cons_i32DivU
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32DivU :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32DivU :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_evalInstrs_cons_i32Bin_generic
    .i32DivU eval_u32_div .div
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rfl rest
    preservation_rest ws' s' ops hw hl

/-- `i32RemU :: rest`. -/
theorem preservation_evalInstrs_cons_i32RemU
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32RemU :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32RemU :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_evalInstrs_cons_i32Bin_generic
    .i32RemU eval_u32_rem .rem
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rfl rest
    preservation_rest ws' s' ops hw hl

-- ════════════════════════════════════════════════════════════════════
-- localSet cons case
--
-- Head ops are `opsCommit ++ [.copy dst src]`. The two arms of
-- `lookupLocal i` differ only in whether `dst` is the local's existing
-- stable register or a freshly-allocated one — both yield the same
-- `[.copy …]` op-shape, so the cons proof factors through the per-op
-- `preservation_localSet` and case-splits the ops-shape analysis once
-- on `lookupLocal`.
-- ════════════════════════════════════════════════════════════════════

/-- Helper: every successful `localSet` lowering emits a `loopFreeNoBreak`
    op-list. By case-split on `popSym`, `commit`, and `lookupLocal`. -/
private theorem lowerInstr_localSet_emits_loopFreeNoBreak
    {s s' : LowerState} {i : Nat} {ops : List KernelOp}
    (h : lowerInstr s (.localSet i) = some (s', ops)) :
    loopFreeNoBreak ops = true := by
  unfold lowerInstr at h
  rcases hs : s.stack with _ | ⟨sva, lrest⟩
  · simp [hs, LowerState.popSym] at h
  simp only [hs, LowerState.popSym, Option.bind_eq_bind, Option.some_bind] at h
  rcases hca : ({s with stack := lrest} : LowerState).commit sva
      with _ | ⟨src, s2, opsCommit⟩
  · simp [hca] at h
  simp only [hca, Option.some_bind] at h
  have h_lf_commit : loopFreeNoBreak opsCommit = true :=
    commit_emits_loopFreeNoBreak hca
  cases hlk : s2.lookupLocal i with
  | none =>
      simp [hlk, LowerState.alloc, LowerState.setLocalReg] at h
      obtain ⟨_, hops⟩ := h
      cases hops
      simp only [loopFreeNoBreak_append, h_lf_commit, Bool.true_and]
      rfl
  | some dst =>
      simp [hlk, LowerState.setLocalReg] at h
      obtain ⟨_, hops⟩ := h
      cases hops
      simp only [loopFreeNoBreak_append, h_lf_commit, Bool.true_and]
      rfl

/-- `localSet i :: rest` preservation. Head ops are `opsCommit ++
    [.copy dst src]` — both pieces are loop-free with no break, so the
    generic broke-preservation helper discharges the cons-composer's
    mid-state requirement. -/
theorem preservation_evalInstrs_cons_localSet
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (i : Nat)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.localSet i :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.localSet i :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  rw [lowerInstrs_cons_default fuel frames s (.localSet i) rest rfl] at hl
  cases h_head : lowerInstr s (.localSet i) with
  | none =>
      rw [h_head] at hl
      simp at hl
  | some head_pair =>
      rcases head_pair with ⟨s_after, ops_head⟩
      rw [h_head] at hl
      simp only [Option.bind_eq_bind, Option.some_bind] at hl
      cases h_post : lowerInstrs fuel frames s_after rest with
      | none => simp [h_post] at hl
      | some post_pair =>
          rcases post_pair with ⟨s_post, postOps⟩
          simp [h_post] at hl
          rcases hl with ⟨h_s_eq, h_ops_eq⟩
          rw [evalInstrs_cons_default fuel ws (.localSet i) rest h_no_branch h_no_halt rfl] at hw
          cases h_eval_head : evalInstr ws (.localSet i) with
          | none =>
              rw [h_eval_head] at hw
              simp at hw
          | some ws_after =>
              rw [h_eval_head] at hw
              simp only at hw
              obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
                preservation_localSet ws s kst layout R h_kst_no_broke i
                  ws_after s_after ops_head
                  h_eval_head h_head
              -- loopFreeNoBreak via the helper.
              have h_lf_head : loopFreeNoBreak ops_head = true :=
                lowerInstr_localSet_emits_loopFreeNoBreak h_head
              have h_lf_head_shallow : loopFree ops_head = true :=
                loopFreeNoBreak_implies_loopFree h_lf_head
              -- broke preservation through ops_head.
              have h_mid_broke : kst_mid.broke = false :=
                evalOps_loopFreeNoBreak_preserves_broke
                  h_lf_head h_kst_no_broke h_kst_eval
              -- Mid-state: localSet doesn't touch branchTarget / halted.
              -- evalInstr ws (.localSet i) only updates locals/stack, so
              -- ws_after.{branchTarget, halted} = ws.{branchTarget, halted}.
              have h_mid_no_branch : ws_after.branchTarget = none := by
                -- Reduce evalInstr to extract the post-state shape.
                simp only [evalInstr, WasmState.pop, WasmState.setLocal,
                           Option.bind_eq_bind, Option.bind, pure] at h_eval_head
                rcases hws : ws.stack with _ | ⟨v_w, rest_ws⟩
                · simp [hws] at h_eval_head
                · simp only [hws] at h_eval_head
                  by_cases hbnd : i < ws.locals.length
                  · simp only [if_pos hbnd] at h_eval_head
                    have := ((Option.some.injEq _ _).mp h_eval_head).symm
                    rw [this]; simp [h_no_branch]
                  · simp only [if_neg hbnd] at h_eval_head
                    simp at h_eval_head
              have h_mid_no_halt : ws_after.halted = false := by
                simp only [evalInstr, WasmState.pop, WasmState.setLocal,
                           Option.bind_eq_bind, Option.bind, pure] at h_eval_head
                rcases hws : ws.stack with _ | ⟨v_w, rest_ws⟩
                · simp [hws] at h_eval_head
                · simp only [hws] at h_eval_head
                  by_cases hbnd : i < ws.locals.length
                  · simp only [if_pos hbnd] at h_eval_head
                    have := ((Option.some.injEq _ _).mp h_eval_head).symm
                    rw [this]; simp [h_no_halt]
                  · simp only [if_neg hbnd] at h_eval_head
                    simp at h_eval_head
              obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest⟩ :=
                preservation_rest R_mid h_mid_no_branch h_mid_no_halt hw h_post
              have h_chained :
                  ∃ kst'', evalOps F_rest kst (ops_head ++ postOps) = some kst''
                    ∧ Refines ws' s_post kst'' layout :=
                preservation_evalInstrs_cons_compose_shallow
                  h_lf_head_shallow h_kst_eval h_mid_broke
                  ⟨kst'_mid, h_eval_rest, R_rest⟩
              obtain ⟨kst'', h_eval'', R''⟩ := h_chained
              refine ⟨kst'', F_rest, ?_, ?_⟩
              · rw [← h_ops_eq]; exact h_eval''
              · rw [← h_s_eq]; exact R''

-- ════════════════════════════════════════════════════════════════════
-- localTee cons case
--
-- Head ops are `opsCommit ++ [.copy dst src, .copy post_fresh dst]`
-- — same `popSym + commit` prefix as localSet, with an extra `.copy`
-- to break aliasing when the post-tee stack value is read back. Both
-- `lookupLocal` arms emit the same op-shape; only `dst` differs.
-- ════════════════════════════════════════════════════════════════════

private theorem lowerInstr_localTee_emits_loopFreeNoBreak
    {s s' : LowerState} {i : Nat} {ops : List KernelOp}
    (h : lowerInstr s (.localTee i) = some (s', ops)) :
    loopFreeNoBreak ops = true := by
  unfold lowerInstr at h
  rcases hs : s.stack with _ | ⟨sva, lrest⟩
  · simp [hs, LowerState.popSym] at h
  simp only [hs, LowerState.popSym, Option.bind_eq_bind, Option.some_bind] at h
  rcases hca : ({s with stack := lrest} : LowerState).commit sva
      with _ | ⟨src, s2, opsCommit⟩
  · simp [hca] at h
  simp only [hca, Option.some_bind] at h
  have h_lf_commit : loopFreeNoBreak opsCommit = true :=
    commit_emits_loopFreeNoBreak hca
  cases hlk : s2.lookupLocal i with
  | none =>
      simp [hlk, LowerState.alloc, LowerState.setLocalReg, LowerState.push] at h
      obtain ⟨_, hops⟩ := h
      cases hops
      simp only [loopFreeNoBreak_append, h_lf_commit, Bool.true_and]
      rfl
  | some dst =>
      simp [hlk, LowerState.setLocalReg, LowerState.alloc, LowerState.push] at h
      obtain ⟨_, hops⟩ := h
      cases hops
      simp only [loopFreeNoBreak_append, h_lf_commit, Bool.true_and]
      rfl

/-- `localTee i :: rest` preservation. Head ops are `opsCommit ++
    [.copy dst src, .copy post_fresh dst]` — same broke-preservation
    discharge pattern as localSet. -/
theorem preservation_evalInstrs_cons_localTee
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (i : Nat)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.localTee i :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.localTee i :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  rw [lowerInstrs_cons_default fuel frames s (.localTee i) rest rfl] at hl
  cases h_head : lowerInstr s (.localTee i) with
  | none =>
      rw [h_head] at hl
      simp at hl
  | some head_pair =>
      rcases head_pair with ⟨s_after, ops_head⟩
      rw [h_head] at hl
      simp only [Option.bind_eq_bind, Option.some_bind] at hl
      cases h_post : lowerInstrs fuel frames s_after rest with
      | none => simp [h_post] at hl
      | some post_pair =>
          rcases post_pair with ⟨s_post, postOps⟩
          simp [h_post] at hl
          rcases hl with ⟨h_s_eq, h_ops_eq⟩
          rw [evalInstrs_cons_default fuel ws (.localTee i) rest h_no_branch h_no_halt rfl] at hw
          cases h_eval_head : evalInstr ws (.localTee i) with
          | none =>
              rw [h_eval_head] at hw
              simp at hw
          | some ws_after =>
              rw [h_eval_head] at hw
              simp only at hw
              obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
                preservation_localTee ws s kst layout R h_kst_no_broke i
                  ws_after s_after ops_head
                  h_eval_head h_head
              have h_lf_head : loopFreeNoBreak ops_head = true :=
                lowerInstr_localTee_emits_loopFreeNoBreak h_head
              have h_lf_head_shallow : loopFree ops_head = true :=
                loopFreeNoBreak_implies_loopFree h_lf_head
              have h_mid_broke : kst_mid.broke = false :=
                evalOps_loopFreeNoBreak_preserves_broke
                  h_lf_head h_kst_no_broke h_kst_eval
              -- localTee preserves branchTarget / halted (same as localSet,
              -- with an extra push of the same v_w).
              have h_mid_no_branch : ws_after.branchTarget = none := by
                simp only [evalInstr, WasmState.pop, WasmState.push, WasmState.setLocal,
                           Option.bind_eq_bind, Option.bind, pure] at h_eval_head
                rcases hws : ws.stack with _ | ⟨v_w, rest_ws⟩
                · simp [hws] at h_eval_head
                · simp only [hws] at h_eval_head
                  by_cases hbnd : i < ws.locals.length
                  · simp only [if_pos hbnd] at h_eval_head
                    have := ((Option.some.injEq _ _).mp h_eval_head).symm
                    rw [this]; simp [h_no_branch]
                  · simp only [if_neg hbnd] at h_eval_head
                    simp at h_eval_head
              have h_mid_no_halt : ws_after.halted = false := by
                simp only [evalInstr, WasmState.pop, WasmState.push, WasmState.setLocal,
                           Option.bind_eq_bind, Option.bind, pure] at h_eval_head
                rcases hws : ws.stack with _ | ⟨v_w, rest_ws⟩
                · simp [hws] at h_eval_head
                · simp only [hws] at h_eval_head
                  by_cases hbnd : i < ws.locals.length
                  · simp only [if_pos hbnd] at h_eval_head
                    have := ((Option.some.injEq _ _).mp h_eval_head).symm
                    rw [this]; simp [h_no_halt]
                  · simp only [if_neg hbnd] at h_eval_head
                    simp at h_eval_head
              obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest⟩ :=
                preservation_rest R_mid h_mid_no_branch h_mid_no_halt hw h_post
              have h_chained :
                  ∃ kst'', evalOps F_rest kst (ops_head ++ postOps) = some kst''
                    ∧ Refines ws' s_post kst'' layout :=
                preservation_evalInstrs_cons_compose_shallow
                  h_lf_head_shallow h_kst_eval h_mid_broke
                  ⟨kst'_mid, h_eval_rest, R_rest⟩
              obtain ⟨kst'', h_eval'', R''⟩ := h_chained
              refine ⟨kst'', F_rest, ?_, ?_⟩
              · rw [← h_ops_eq]; exact h_eval''
              · rw [← h_s_eq]; exact R''

-- ════════════════════════════════════════════════════════════════════
-- drop cons case
--
-- `drop` emits no IR — head ops is `[]`. WASM pops one value from the
-- stack; lowering pops one SymVal from the symbolic stack. Refines
-- lifts via the same suffix-shift pattern used for the popped state in
-- `lowerI32Bin_some_shape` callers — Stack indices shift by 1, all
-- other components carry over.
-- ════════════════════════════════════════════════════════════════════

/-- `drop :: rest` preservation. Head ops empty; mid-state is the
    popped state on both sides. -/
theorem preservation_evalInstrs_cons_drop
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.drop :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.drop :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- Both sides require a non-empty stack; case-split.
  rcases hws_stack : ws.stack with _ | ⟨v_w, rest_ws⟩
  · -- WASM-side pop fails → evalInstrs returns none → contradiction with hw.
    rw [evalInstrs_cons_default fuel ws .drop rest h_no_branch h_no_halt rfl] at hw
    have h_ev : evalInstr ws .drop = none := by
      show (do let (_, s1) ← ws.pop; pure s1) = none
      simp [WasmState.pop, hws_stack]
    rw [h_ev] at hw
    simp at hw
  rcases hls_stack : s.stack with _ | ⟨sva, lrest⟩
  · -- Symbolic-side pop fails → lowerInstrs returns none → contradiction with hl.
    rw [lowerInstrs_cons_default fuel frames s .drop rest rfl] at hl
    have h_lw : lowerInstr s .drop = none := by
      show (do let (_, s1) ← s.popSym; pure (s1, ([] : List KernelOp))) = none
      simp [LowerState.popSym, hls_stack]
    rw [h_lw] at hl
    simp at hl
  -- Both pops succeed. Build mid-states.
  let ws_mid : WasmState := { ws with stack := rest_ws }
  let s_mid : LowerState :=
    { nextReg := s.nextReg, stack := lrest,
      localReg := s.localReg, localTy := s.localTy,
      bufferSlots := s.bufferSlots }
  -- Lowering side: lowerInstrs fuel frames s_mid rest = some (s', ops).
  have hl' : lowerInstrs fuel frames s_mid rest = some (s', ops) := by
    rw [lowerInstrs_cons_default fuel frames s .drop rest rfl] at hl
    have h_lw : lowerInstr s .drop = some (s_mid, []) := by
      show (do let (_, s1) ← s.popSym; pure (s1, ([] : List KernelOp))) = some (s_mid, [])
      unfold LowerState.popSym
      rw [hls_stack]
      rfl
    rw [h_lw] at hl
    simp only [Option.bind_eq_bind, Option.some_bind, List.nil_append] at hl
    show lowerInstrs fuel frames s_mid rest = some (s', ops)
    exact cons_default_lowerInstrs_collapse_empty_head hl
  -- Eval side: evalInstrs fuel ws_mid rest = some ws'.
  have hw' : evalInstrs fuel ws_mid rest = some ws' := by
    rw [evalInstrs_cons_default fuel ws .drop rest h_no_branch h_no_halt rfl] at hw
    have h_ev : evalInstr ws .drop = some ws_mid := by
      show (do let (_, s1) ← ws.pop; pure s1) = some ws_mid
      unfold WasmState.pop
      rw [hws_stack]
      rfl
    rw [h_ev] at hw
    simp only at hw
    exact hw
  -- Refines lift on the pop.
  have h_rest_lrest_len : rest_ws.length = lrest.length := by
    have hl_orig := R.stk.left
    rw [hws_stack, hls_stack] at hl_orig
    simpa using hl_orig
  have R_mid : Refines ws_mid s_mid kst layout := by
    refine ⟨⟨h_rest_lrest_len, ?_⟩, R.locs, ?_, ?_, R.injLocals, R.heapRefines⟩
    · -- StackRefines on suffixes (indices shift by 1).
      intro k v hv
      have hrest_get : ws.stack.get? (k + 1) = some v := by
        rw [hws_stack]; simpa using hv
      obtain ⟨svk, hsvk_get, henc⟩ := R.stk.right (k + 1) v hrest_get
      have hlrest_get : lrest.get? k = some svk := by
        have h2 : s.stack.get? (k + 1) = some svk := hsvk_get
        rw [hls_stack] at h2; simpa using h2
      exact ⟨svk, by simpa using hlrest_get, henc⟩
    · -- Fresh: s_mid.stack ⊆ s.stack (we removed sva).
      refine ⟨?_, R.fresh.right⟩
      intro sv hsv r hr
      have hsv_in : sv ∈ s.stack := by
        rw [hls_stack]; exact List.mem_cons_of_mem _ hsv
      exact R.fresh.left sv hsv_in r hr
    · -- AliasFree: same projection on the lrest suffix.
      intro ir hir sv hsv
      have hsv_in : sv ∈ s.stack := by
        rw [hls_stack]; exact List.mem_cons_of_mem _ hsv
      exact R.aliasFree ir hir sv hsv_in
  have h_mid_no_branch : ws_mid.branchTarget = none := by
    simp [ws_mid, h_no_branch]
  have h_mid_no_halt : ws_mid.halted = false := by
    simp [ws_mid, h_no_halt]
  exact preservation_rest R_mid h_mid_no_branch h_mid_no_halt hw' hl'

-- ════════════════════════════════════════════════════════════════════
-- i32 comparison cons cases
--
-- Same shape as the binop family: head emits `opsA ++ opsB ++ [.cmp,
-- .cast]`. Both `.cmp` and `.cast` are loop-free no-break, so the
-- broke-preservation discharge is identical. Six specializations:
-- Eq, Ne, LtU, LeU, GtU, GeU.
-- ════════════════════════════════════════════════════════════════════

/-- Generic cons preservation for any WASM instr whose lowering reduces
    to `lowerI32Cmp s op_k`. -/
theorem preservation_evalInstrs_cons_i32Cmp_generic
    (instr : WasmInstr) (p_w : UInt32 → UInt32 → Bool)
    (op_k : Quanta.KOps.CmpOp)
    (h_w : ∀ s, evalInstr s instr = cmpI32 p_w s)
    (h_l : ∀ s, lowerInstr s instr = lowerI32Cmp s op_k)
    (h_agree : ∀ av bv,
       Quanta.KOps.evalCmpOp op_k (Quanta.KOps.Value.vU32 av)
         (Quanta.KOps.Value.vU32 bv)
         = some (Quanta.KOps.Value.vBool (p_w av bv)))
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (h_ns_lower : isStructuredLower instr = false)
    (h_ns_eval : isStructuredEval instr = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (instr :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (instr :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  rw [lowerInstrs_cons_default fuel frames s instr rest h_ns_lower] at hl
  cases h_head : lowerInstr s instr with
  | none =>
      rw [h_head] at hl
      simp at hl
  | some head_pair =>
      rcases head_pair with ⟨s_after, ops_head⟩
      rw [h_head] at hl
      simp only [Option.bind_eq_bind, Option.some_bind] at hl
      cases h_post : lowerInstrs fuel frames s_after rest with
      | none => simp [h_post] at hl
      | some post_pair =>
          rcases post_pair with ⟨s_post, postOps⟩
          simp [h_post] at hl
          rcases hl with ⟨h_s_eq, h_ops_eq⟩
          rw [evalInstrs_cons_default fuel ws instr rest h_no_branch h_no_halt h_ns_eval] at hw
          cases h_eval_head : evalInstr ws instr with
          | none =>
              rw [h_eval_head] at hw
              simp at hw
          | some ws_after =>
              rw [h_eval_head] at hw
              simp only at hw
              obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
                preservation_i32Cmp_generic instr p_w op_k h_w h_l h_agree
                  ws s kst layout R h_kst_no_broke
                  ws_after s_after ops_head
                  h_eval_head h_head
              rw [h_l s] at h_head
              obtain ⟨_svb, _sva, _lrest, ra, _s3, opsA, rb, s4, opsB,
                      _h_stk, hca, hcb, _h_s4_stk, _h_s4_lr, _h_s4_lt,
                      _h_nr_le, _h_s_eq_shape, h_ops_head_eq⟩ :=
                lowerI32Cmp_some_shape h_head
              have h_lf_opsA : loopFreeNoBreak opsA = true :=
                commit_emits_loopFreeNoBreak hca
              have h_lf_opsB : loopFreeNoBreak opsB = true :=
                commit_emits_loopFreeNoBreak hcb
              have h_lf_tail :
                  loopFreeNoBreak [KernelOp.cmp s4.nextReg ra rb op_k .bool,
                                    KernelOp.cast (s4.nextReg + 1) s4.nextReg .bool .u32]
                    = true := rfl
              have h_lf_head : loopFreeNoBreak ops_head = true := by
                rw [h_ops_head_eq]
                simp [loopFreeNoBreak_append, h_lf_opsA, h_lf_opsB, h_lf_tail]
              have h_lf_head_shallow : loopFree ops_head = true :=
                loopFreeNoBreak_implies_loopFree h_lf_head
              have h_mid_broke : kst_mid.broke = false :=
                evalOps_loopFreeNoBreak_preserves_broke
                  h_lf_head h_kst_no_broke h_kst_eval
              have h_mid_no_branch : ws_after.branchTarget = none := by
                rw [h_w] at h_eval_head
                obtain ⟨_, _, _, _, h_ws_eq⟩ := cmpI32_some_shape h_eval_head
                rw [h_ws_eq]; simp [h_no_branch]
              have h_mid_no_halt : ws_after.halted = false := by
                rw [h_w] at h_eval_head
                obtain ⟨_, _, _, _, h_ws_eq⟩ := cmpI32_some_shape h_eval_head
                rw [h_ws_eq]; simp [h_no_halt]
              obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest⟩ :=
                preservation_rest R_mid h_mid_no_branch h_mid_no_halt hw h_post
              have h_chained :
                  ∃ kst'', evalOps F_rest kst (ops_head ++ postOps) = some kst''
                    ∧ Refines ws' s_post kst'' layout :=
                preservation_evalInstrs_cons_compose_shallow
                  h_lf_head_shallow h_kst_eval h_mid_broke
                  ⟨kst'_mid, h_eval_rest, R_rest⟩
              obtain ⟨kst'', h_eval'', R''⟩ := h_chained
              refine ⟨kst'', F_rest, ?_, ?_⟩
              · rw [← h_ops_eq]; exact h_eval''
              · rw [← h_s_eq]; exact R''

/-- `i32Eq :: rest`. -/
theorem preservation_evalInstrs_cons_i32Eq
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Eq :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Eq :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_evalInstrs_cons_i32Cmp_generic
    .i32Eq (· == ·) .eq
    (fun _ => rfl) (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rest
    preservation_rest ws' s' ops hw hl

/-- `i32Ne :: rest`. -/
theorem preservation_evalInstrs_cons_i32Ne
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Ne :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Ne :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_evalInstrs_cons_i32Cmp_generic
    .i32Ne (· != ·) .ne
    (fun _ => rfl) (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rest
    preservation_rest ws' s' ops hw hl

/-- `i32LtU :: rest`. -/
theorem preservation_evalInstrs_cons_i32LtU
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32LtU :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32LtU :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_evalInstrs_cons_i32Cmp_generic
    .i32LtU (· < ·) .lt
    (fun _ => rfl) (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rest
    preservation_rest ws' s' ops hw hl

/-- `i32LeU :: rest`. -/
theorem preservation_evalInstrs_cons_i32LeU
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32LeU :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32LeU :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_evalInstrs_cons_i32Cmp_generic
    .i32LeU (· <= ·) .le
    (fun _ => rfl) (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rest
    preservation_rest ws' s' ops hw hl

/-- `i32GtU :: rest`. -/
theorem preservation_evalInstrs_cons_i32GtU
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32GtU :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32GtU :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_evalInstrs_cons_i32Cmp_generic
    .i32GtU (· > ·) .gt
    (fun _ => rfl) (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rest
    preservation_rest ws' s' ops hw hl

/-- `i32GeU :: rest`. -/
theorem preservation_evalInstrs_cons_i32GeU
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32GeU :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32GeU :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout :=
  preservation_evalInstrs_cons_i32Cmp_generic
    .i32GeU (· >= ·) .ge
    (fun _ => rfl) (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke rfl rfl rest
    preservation_rest ws' s' ops hw hl

-- ════════════════════════════════════════════════════════════════════
-- localGet (buffer-slot path) cons case
--
-- When `lookupBufferSlot i = some slot`, lowering pushes a `.bufferPtr
-- slot` SymVal and emits NO IR. The mid-state's symbolic stack gains
-- the buffer pointer; the regfile is untouched. The semantic
-- precondition `h_loc_buf` (the WASM local at `i` encodes the buffer's
-- start address) is the same one the per-op theorem requires.
-- ════════════════════════════════════════════════════════════════════

theorem preservation_evalInstrs_cons_localGet_bufferSlot
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (i : Nat) (slot : Nat)
    (h_buf : s.lookupBufferSlot i = some slot)
    (h_loc_buf : ∀ v, ws.locals.get? i = some v →
      ∃ n : UInt32, v = .wI32 n ∧ n.toNat = layout.startAddr slot)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
        {kst_mid : Quanta.KOps.State}
        (_R_mid : Refines ws_mid s_mid kst_mid layout)
        (_h_no_branch_mid : ws_mid.branchTarget = none)
        (_h_no_halt_mid : ws_mid.halted = false)
        {ws'_mid : WasmState} {s'_mid : LowerState} {postOps : List KernelOp}
        (_hw_mid : evalInstrs fuel ws_mid rest = some ws'_mid)
        (_hl_mid : lowerInstrs fuel frames s_mid rest = some (s'_mid, postOps)),
      ∃ (kst'_mid : Quanta.KOps.State) (F : Nat),
        evalOps F kst_mid postOps = some kst'_mid ∧
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.localGet i :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.localGet i :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  rw [lowerInstrs_cons_default fuel frames s (.localGet i) rest rfl] at hl
  -- Buffer path: lowerInstr s (.localGet i) reduces to
  -- some (s.pushSym (.bufferPtr slot), []).
  have h_head : lowerInstr s (.localGet i)
                  = some (s.pushSym (.bufferPtr slot), []) := by
    show (match s.lookupBufferSlot i with
          | some slot => some (s.pushSym (.bufferPtr slot), [])
          | none =>
              (s.lookupLocal i).bind (fun stable =>
                let (fresh, s1) := s.alloc
                let s2 := s1.push fresh
                some (s2, [KernelOp.copy fresh stable])))
              = some (s.pushSym (.bufferPtr slot), [])
    rw [h_buf]
  rw [h_head] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl
  cases h_post : lowerInstrs fuel frames (s.pushSym (.bufferPtr slot)) rest with
  | none => simp [h_post] at hl
  | some post_pair =>
      rcases post_pair with ⟨s_post, postOps⟩
      simp [h_post] at hl
      rcases hl with ⟨h_s_eq, h_ops_eq⟩
      rw [evalInstrs_cons_default fuel ws (.localGet i) rest h_no_branch h_no_halt rfl] at hw
      cases h_eval_head : evalInstr ws (.localGet i) with
      | none =>
          rw [h_eval_head] at hw
          simp at hw
      | some ws_after =>
          rw [h_eval_head] at hw
          simp only at hw
          -- Apply per-op preservation_localGet_bufferSlot to derive R_mid.
          obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
            preservation_localGet_bufferSlot ws s kst layout R i slot h_buf h_loc_buf
              ws_after (s.pushSym (.bufferPtr slot)) []
              h_eval_head h_head
          -- ops_head = [] → kst_mid = kst (regfile / heap unchanged).
          have h_kst_mid_eq : kst_mid = kst := by
            simp [evalOps] at h_kst_eval
            exact h_kst_eval.symm
          -- Extract ws_after's shape once: ws_after = ws.push v for some v.
          have h_ws_after_shape : ∃ v, ws_after = ws.push v := by
            cases hloc : ws.getLocal i with
            | none =>
                have h_ev : evalInstr ws (.localGet i) = none := by
                  show (do let v ← ws.getLocal i; pure (ws.push v)) = none
                  rw [hloc]; rfl
                rw [h_ev] at h_eval_head; exact (Option.noConfusion h_eval_head)
            | some v =>
                refine ⟨v, ?_⟩
                have h_ev : evalInstr ws (.localGet i) = some (ws.push v) := by
                  show (do let v ← ws.getLocal i; pure (ws.push v)) = some (ws.push v)
                  rw [hloc]; rfl
                rw [h_ev] at h_eval_head
                exact ((Option.some.injEq _ _).mp h_eval_head).symm
          obtain ⟨v_pushed, h_ws_after_eq⟩ := h_ws_after_shape
          have h_mid_no_branch : ws_after.branchTarget = none := by
            rw [h_ws_after_eq]; simp [WasmState.push, h_no_branch]
          have h_mid_no_halt : ws_after.halted = false := by
            rw [h_ws_after_eq]; simp [WasmState.push, h_no_halt]
          obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest⟩ :=
            preservation_rest R_mid h_mid_no_branch h_mid_no_halt hw h_post
          refine ⟨kst'_mid, F_rest, ?_, ?_⟩
          · -- ops_head = [], so ops = postOps; kst_mid = kst.
            rw [← h_ops_eq]
            rw [h_kst_mid_eq] at h_eval_rest
            exact h_eval_rest
          · rw [← h_s_eq]; exact R_rest

end Quanta.Wasm
