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
import Quanta.Wasm.LowerInvariants

namespace Quanta.Wasm

open Quanta.KOps (KernelOp evalOps regLookup)
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
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
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
  exact preservation_rest R h_no_branch h_no_halt h_kst_no_broke hw' hl'

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
    (h_kst_no_broke : kst.broke = false)
    (n : Int) (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
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
  exact preservation_rest R_mid h_no_branch_mid h_no_halt_mid h_kst_no_broke hw' hl'

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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
                preservation_rest R_mid h_mid_no_branch h_mid_no_halt h_mid_broke hw h_post
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
                preservation_rest R_mid h_mid_no_branch h_mid_no_halt h_mid_broke hw h_post
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
                preservation_rest R_mid h_mid_no_branch h_mid_no_halt h_mid_broke hw h_post
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
                preservation_rest R_mid h_mid_no_branch h_mid_no_halt h_mid_broke hw h_post
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
    (h_kst_no_broke : kst.broke = false)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
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
  exact preservation_rest R_mid h_mid_no_branch h_mid_no_halt h_kst_no_broke hw' hl'

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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
                preservation_rest R_mid h_mid_no_branch h_mid_no_halt h_mid_broke hw h_post
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
    (h_kst_no_broke : kst.broke = false)
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
        (_h_kst_no_broke_mid : kst_mid.broke = false)
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
          have h_mid_broke : kst_mid.broke = false := by
            rw [h_kst_mid_eq]; exact h_kst_no_broke
          obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest⟩ :=
            preservation_rest R_mid h_mid_no_branch h_mid_no_halt h_mid_broke hw h_post
          refine ⟨kst'_mid, F_rest, ?_, ?_⟩
          · -- ops_head = [], so ops = postOps; kst_mid = kst.
            rw [← h_ops_eq]
            rw [h_kst_mid_eq] at h_eval_rest
            exact h_eval_rest
          · rw [← h_s_eq]; exact R_rest

-- ════════════════════════════════════════════════════════════════════
-- i32Shl (non-buffer path) cons case
--
-- Same shape as i32Add: when the popped stack doesn't match
-- `<i32ConstSym k> :: <reg base ty> :: rest`, lowerI32Shl falls
-- through to lowerI32Bin s .shl. h_no_buf supplies the equational
-- reduction; the binop generic does the rest.
-- ════════════════════════════════════════════════════════════════════

theorem preservation_evalInstrs_cons_i32Shl
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (h_no_buf : ∀ k base ty rest,
      s.stack ≠ .i32ConstSym k :: .reg base ty :: rest)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
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
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Shl :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Shl :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  have h_l_eq : lowerInstr s .i32Shl = lowerI32Bin s .shl := by
    show lowerI32Shl s = lowerI32Bin s .shl
    unfold lowerI32Shl
    split
    next k base ty rest hs => exact absurd hs (h_no_buf k base ty rest)
    next => rfl
  exact preservation_evalInstrs_cons_i32Bin_generic
    .i32Shl (fun a b => a <<< b) .shl
    (fun _ => rfl) (by intro av bv; rfl)
    fuel frames ws s kst layout R h_no_branch h_no_halt h_kst_no_broke h_l_eq rfl rfl rest
    preservation_rest ws' s' ops hw hl

-- ════════════════════════════════════════════════════════════════════
-- Buffer-pattern fold cons cases
--
-- i32Shl folds <reg base> :: <i32ConstSym k> into SymVal.scaledIdx.
-- i32Add folds <bufferPtr> :: <scaledIdx> (or scaled-first) into
-- SymVal.bufferAccess. All three are no-IR: head ops empty,
-- mid-state symbolic stack rewritten with the folded SymVal.
-- ════════════════════════════════════════════════════════════════════

theorem preservation_evalInstrs_cons_i32Shl_bufferPattern
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (k : Int) (base : Quanta.KOps.Reg) (ty : Quanta.KOps.Scalar)
    (lstk_rest : List SymVal)
    (h_stack : s.stack = .i32ConstSym k :: .reg base ty :: lstk_rest)
    (h_shift_eq : ∀ a : UInt32,
       regLookup kst.rf base = some (Quanta.KOps.Value.vU32 a) →
       (a <<< (UInt32.ofNat k.toNat)).toNat = a.toNat * (1 <<< k.toNat))
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
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
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Shl :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Shl :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  rw [lowerInstrs_cons_default fuel frames s .i32Shl rest rfl] at hl
  -- Per-op fold reduces lowerInstr to a no-IR push of scaledIdx.
  have h_head : lowerInstr s .i32Shl =
      some ({ s with stack := .scaledIdx base (1 <<< k.toNat) :: lstk_rest }, []) := by
    show lowerI32Shl s = _
    unfold lowerI32Shl
    rw [h_stack]
  rw [h_head] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl
  cases h_post : lowerInstrs fuel frames
                  ({ s with stack := .scaledIdx base (1 <<< k.toNat) :: lstk_rest })
                  rest with
  | none => simp [h_post] at hl
  | some post_pair =>
      rcases post_pair with ⟨s_post, postOps⟩
      simp [h_post] at hl
      rcases hl with ⟨h_s_eq, h_ops_eq⟩
      rw [evalInstrs_cons_default fuel ws .i32Shl rest h_no_branch h_no_halt rfl] at hw
      cases h_eval_head : evalInstr ws .i32Shl with
      | none =>
          rw [h_eval_head] at hw
          simp at hw
      | some ws_after =>
          rw [h_eval_head] at hw
          simp only at hw
          obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
            preservation_i32Shl_bufferPattern ws s kst layout R k base ty lstk_rest
              h_stack h_shift_eq
              ws_after _ [] h_eval_head h_head
          have h_kst_mid_eq : kst_mid = kst := by
            simp [evalOps] at h_kst_eval; exact h_kst_eval.symm
          have h_mid_broke : kst_mid.broke = false := by
            rw [h_kst_mid_eq]; exact h_kst_no_broke
          -- Mid-state branch/halt: binI32 preserves both.
          have h_w : evalInstr ws .i32Shl = binI32 (· <<< ·) ws := rfl
          rw [h_w] at h_eval_head
          obtain ⟨_, _, _, _, h_ws_eq⟩ := binI32_some_shape h_eval_head
          have h_mid_no_branch : ws_after.branchTarget = none := by
            rw [h_ws_eq]; simp [h_no_branch]
          have h_mid_no_halt : ws_after.halted = false := by
            rw [h_ws_eq]; simp [h_no_halt]
          obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest⟩ :=
            preservation_rest R_mid h_mid_no_branch h_mid_no_halt h_mid_broke hw h_post
          refine ⟨kst'_mid, F_rest, ?_, ?_⟩
          · rw [← h_ops_eq]
            rw [h_kst_mid_eq] at h_eval_rest
            exact h_eval_rest
          · rw [← h_s_eq]; exact R_rest

theorem preservation_evalInstrs_cons_i32Add_bufferPattern_scaledFirst
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (slot : Nat) (base : Quanta.KOps.Reg) (scale : Nat) (lstk_rest : List SymVal)
    (h_stack : s.stack = .scaledIdx base scale :: .bufferPtr slot :: lstk_rest)
    (h_addr_eq : ∀ a b_ptr : UInt32, ∀ b : UInt32,
       regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
       a.toNat = b.toNat * scale →
       b_ptr.toNat = layout.startAddr slot →
       (b_ptr + a).toNat = layout.startAddr slot + b.toNat * scale)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
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
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Add :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Add :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  rw [lowerInstrs_cons_default fuel frames s .i32Add rest rfl] at hl
  have h_head : lowerInstr s .i32Add =
      some ({ s with stack := .bufferAccess slot base scale :: lstk_rest }, []) := by
    show lowerI32Add s = _
    unfold lowerI32Add
    rw [h_stack]
  rw [h_head] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl
  cases h_post : lowerInstrs fuel frames
                  ({ s with stack := .bufferAccess slot base scale :: lstk_rest })
                  rest with
  | none => simp [h_post] at hl
  | some post_pair =>
      rcases post_pair with ⟨s_post, postOps⟩
      simp [h_post] at hl
      rcases hl with ⟨h_s_eq, h_ops_eq⟩
      rw [evalInstrs_cons_default fuel ws .i32Add rest h_no_branch h_no_halt rfl] at hw
      cases h_eval_head : evalInstr ws .i32Add with
      | none =>
          rw [h_eval_head] at hw
          simp at hw
      | some ws_after =>
          rw [h_eval_head] at hw
          simp only at hw
          obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
            preservation_i32Add_bufferPattern_scaledFirst ws s kst layout R
              slot base scale lstk_rest h_stack h_addr_eq
              ws_after _ [] h_eval_head h_head
          have h_kst_mid_eq : kst_mid = kst := by
            simp [evalOps] at h_kst_eval; exact h_kst_eval.symm
          have h_mid_broke : kst_mid.broke = false := by
            rw [h_kst_mid_eq]; exact h_kst_no_broke
          have h_w : evalInstr ws .i32Add = binI32 eval_u32_wrapping_add ws := rfl
          rw [h_w] at h_eval_head
          obtain ⟨_, _, _, _, h_ws_eq⟩ := binI32_some_shape h_eval_head
          have h_mid_no_branch : ws_after.branchTarget = none := by
            rw [h_ws_eq]; simp [h_no_branch]
          have h_mid_no_halt : ws_after.halted = false := by
            rw [h_ws_eq]; simp [h_no_halt]
          obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest⟩ :=
            preservation_rest R_mid h_mid_no_branch h_mid_no_halt h_mid_broke hw h_post
          refine ⟨kst'_mid, F_rest, ?_, ?_⟩
          · rw [← h_ops_eq]
            rw [h_kst_mid_eq] at h_eval_rest
            exact h_eval_rest
          · rw [← h_s_eq]; exact R_rest

theorem preservation_evalInstrs_cons_i32Add_bufferPattern_ptrFirst
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (slot : Nat) (base : Quanta.KOps.Reg) (scale : Nat) (lstk_rest : List SymVal)
    (h_stack : s.stack = .bufferPtr slot :: .scaledIdx base scale :: lstk_rest)
    (h_addr_eq : ∀ a b_ptr : UInt32, ∀ b : UInt32,
       regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
       a.toNat = b.toNat * scale →
       b_ptr.toNat = layout.startAddr slot →
       (a + b_ptr).toNat = layout.startAddr slot + b.toNat * scale)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
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
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Add :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Add :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  rw [lowerInstrs_cons_default fuel frames s .i32Add rest rfl] at hl
  have h_head : lowerInstr s .i32Add =
      some ({ s with stack := .bufferAccess slot base scale :: lstk_rest }, []) := by
    show lowerI32Add s = _
    unfold lowerI32Add
    rw [h_stack]
  rw [h_head] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl
  cases h_post : lowerInstrs fuel frames
                  ({ s with stack := .bufferAccess slot base scale :: lstk_rest })
                  rest with
  | none => simp [h_post] at hl
  | some post_pair =>
      rcases post_pair with ⟨s_post, postOps⟩
      simp [h_post] at hl
      rcases hl with ⟨h_s_eq, h_ops_eq⟩
      rw [evalInstrs_cons_default fuel ws .i32Add rest h_no_branch h_no_halt rfl] at hw
      cases h_eval_head : evalInstr ws .i32Add with
      | none =>
          rw [h_eval_head] at hw
          simp at hw
      | some ws_after =>
          rw [h_eval_head] at hw
          simp only at hw
          obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
            preservation_i32Add_bufferPattern_ptrFirst ws s kst layout R
              slot base scale lstk_rest h_stack h_addr_eq
              ws_after _ [] h_eval_head h_head
          have h_kst_mid_eq : kst_mid = kst := by
            simp [evalOps] at h_kst_eval; exact h_kst_eval.symm
          have h_mid_broke : kst_mid.broke = false := by
            rw [h_kst_mid_eq]; exact h_kst_no_broke
          have h_w : evalInstr ws .i32Add = binI32 eval_u32_wrapping_add ws := rfl
          rw [h_w] at h_eval_head
          obtain ⟨_, _, _, _, h_ws_eq⟩ := binI32_some_shape h_eval_head
          have h_mid_no_branch : ws_after.branchTarget = none := by
            rw [h_ws_eq]; simp [h_no_branch]
          have h_mid_no_halt : ws_after.halted = false := by
            rw [h_ws_eq]; simp [h_no_halt]
          obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest⟩ :=
            preservation_rest R_mid h_mid_no_branch h_mid_no_halt h_mid_broke hw h_post
          refine ⟨kst'_mid, F_rest, ?_, ?_⟩
          · rw [← h_ops_eq]
            rw [h_kst_mid_eq] at h_eval_rest
            exact h_eval_rest
          · rw [← h_s_eq]; exact R_rest

-- ════════════════════════════════════════════════════════════════════
-- i32Load / i32Store (buffer-access path) cons cases
--
-- i32Load emits a single `.load` op against the BufferAccess address.
-- i32Store emits opsCommit ++ [.store ...]. Both heads are loop-free
-- no-break; the generic broke-preservation helper handles the
-- cons-composer mid-state. Both use loadI32 / storeI32 on the WASM
-- side, which preserve branchTarget / halted.
-- ════════════════════════════════════════════════════════════════════

theorem preservation_evalInstrs_cons_i32Load
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (slot : Nat) (base : Quanta.KOps.Reg) (lstk_rest : List SymVal)
    (offset align : Nat)
    (h_stack : s.stack = .bufferAccess slot base 4 :: lstk_rest)
    (h_offset : offset = 0)
    (h_in_bounds : ∀ b : UInt32,
       regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
       b.toNat < layout.length slot)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
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
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Load offset align :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Load offset align :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  rw [lowerInstrs_cons_default fuel frames s (.i32Load offset align) rest rfl] at hl
  -- Per-op fold reduces lowerInstr to a single `.load` op.
  have h_head : lowerInstr s (.i32Load offset align) =
      some ({ s with nextReg := s.nextReg + 1,
                     stack := .reg s.nextReg .u32 :: lstk_rest },
             [.load s.nextReg slot base .u32]) := by
    show lowerI32Load s = _
    unfold lowerI32Load
    rw [h_stack]
    rfl
  rw [h_head] at hl
  simp only [Option.bind_eq_bind, Option.some_bind] at hl
  cases h_post : lowerInstrs fuel frames
                  ({ s with nextReg := s.nextReg + 1,
                            stack := .reg s.nextReg .u32 :: lstk_rest })
                  rest with
  | none => simp [h_post] at hl
  | some post_pair =>
      rcases post_pair with ⟨s_post, postOps⟩
      simp [h_post] at hl
      rcases hl with ⟨h_s_eq, h_ops_eq⟩
      rw [evalInstrs_cons_default fuel ws (.i32Load offset align) rest h_no_branch h_no_halt rfl] at hw
      cases h_eval_head : evalInstr ws (.i32Load offset align) with
      | none =>
          rw [h_eval_head] at hw
          simp at hw
      | some ws_after =>
          rw [h_eval_head] at hw
          simp only at hw
          obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
            preservation_i32Load ws s kst layout R slot base lstk_rest offset align
              h_stack h_offset h_in_bounds
              ws_after _ _ h_eval_head h_head
          -- Head ops = [.load ...] is loop-free no-break.
          have h_lf_head : loopFreeNoBreak [KernelOp.load s.nextReg slot base .u32] = true :=
            rfl
          have h_lf_head_shallow : loopFree [KernelOp.load s.nextReg slot base .u32] = true :=
            loopFreeNoBreak_implies_loopFree h_lf_head
          have h_mid_broke : kst_mid.broke = false :=
            evalOps_loopFreeNoBreak_preserves_broke
              h_lf_head h_kst_no_broke h_kst_eval
          -- Mid-state branch/halt: loadI32 only mutates stack.
          have h_w : evalInstr ws (.i32Load offset align) = loadI32 ws offset := rfl
          rw [h_w] at h_eval_head
          -- loadI32 success implies ws_after = { ws with stack := wI32 n :: ws_rest }
          -- for some n, ws_rest. Extract via the loadI32 unfold.
          have h_mid_no_branch : ws_after.branchTarget = none := by
            unfold loadI32 at h_eval_head
            rcases hws : ws.stack with _ | ⟨vaddr, ws_rest⟩
            · simp [hws, WasmState.pop] at h_eval_head
            · simp [hws, WasmState.pop] at h_eval_head
              cases vaddr with
              | wI32 addr_w =>
                  simp at h_eval_head
                  rcases hmem : ws.mem.load_u32 (addr_w.toNat + offset) with _ | n
                  · simp [hmem] at h_eval_head
                  · simp [hmem, WasmState.push] at h_eval_head
                    rw [← h_eval_head]; simp [h_no_branch]
              | wI64 _ => simp at h_eval_head
              | wF32 _ => simp at h_eval_head
              | wF64 _ => simp at h_eval_head
          have h_mid_no_halt : ws_after.halted = false := by
            unfold loadI32 at h_eval_head
            rcases hws : ws.stack with _ | ⟨vaddr, ws_rest⟩
            · simp [hws, WasmState.pop] at h_eval_head
            · simp [hws, WasmState.pop] at h_eval_head
              cases vaddr with
              | wI32 addr_w =>
                  simp at h_eval_head
                  rcases hmem : ws.mem.load_u32 (addr_w.toNat + offset) with _ | n
                  · simp [hmem] at h_eval_head
                  · simp [hmem, WasmState.push] at h_eval_head
                    rw [← h_eval_head]; simp [h_no_halt]
              | wI64 _ => simp at h_eval_head
              | wF32 _ => simp at h_eval_head
              | wF64 _ => simp at h_eval_head
          obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest⟩ :=
            preservation_rest R_mid h_mid_no_branch h_mid_no_halt h_mid_broke hw h_post
          have h_chained :
              ∃ kst'', evalOps F_rest kst
                          ([KernelOp.load s.nextReg slot base .u32] ++ postOps) = some kst''
                ∧ Refines ws' s_post kst'' layout :=
            preservation_evalInstrs_cons_compose_shallow
              h_lf_head_shallow h_kst_eval h_mid_broke
              ⟨kst'_mid, h_eval_rest, R_rest⟩
          obtain ⟨kst'', h_eval'', R''⟩ := h_chained
          refine ⟨kst'', F_rest, ?_, ?_⟩
          · rw [← h_ops_eq]; exact h_eval''
          · rw [← h_s_eq]; exact R''

theorem preservation_evalInstrs_cons_i32Store
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (sv_val : SymVal) (slot : Nat) (base : Quanta.KOps.Reg) (lstk_rest : List SymVal)
    (offset align : Nat)
    (h_stack : s.stack = sv_val :: .bufferAccess slot base 4 :: lstk_rest)
    (h_offset : offset = 0)
    (h_in_bounds : ∀ b : UInt32,
       regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
       b.toNat < layout.length slot)
    (h_layout_no_overlap : ∀ b : UInt32,
       regLookup kst.rf base = some (Quanta.KOps.Value.vU32 b) →
       ∀ slot' idx',
         idx' < layout.length slot' →
         (slot', idx') ≠ (slot, b.toNat) →
         layout.startAddr slot + b.toNat * 4 + 4 ≤ layout.startAddr slot' + idx' * 4 ∨
         layout.startAddr slot' + idx' * 4 + 4 ≤ layout.startAddr slot + b.toNat * 4)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
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
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.i32Store offset align :: rest) = some ws')
    (hl : lowerInstrs fuel frames s (.i32Store offset align :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  rw [lowerInstrs_cons_default fuel frames s (.i32Store offset align) rest rfl] at hl
  -- Extract the head pair via h_stack: lowerI32Store reduces to a commit + [.store ...].
  let s_pop : LowerState :=
    { nextReg := s.nextReg, stack := lstk_rest,
      localReg := s.localReg, localTy := s.localTy,
      bufferSlots := s.bufferSlots }
  cases hca : s_pop.commit sv_val with
  | none =>
      have h_lw : lowerInstr s (.i32Store offset align) = none := by
        show lowerI32Store s = none
        unfold lowerI32Store
        simp only [LowerState.popSym, h_stack, Option.bind_eq_bind, Option.some_bind]
        rw [show ({ nextReg := s.nextReg, stack := lstk_rest,
                    localReg := s.localReg, localTy := s.localTy,
                    bufferSlots := s.bufferSlots } : LowerState).commit sv_val
                = s_pop.commit sv_val from rfl]
        rw [hca]
        rfl
      rw [h_lw] at hl
      simp at hl
  | some commit_pair =>
      rcases commit_pair with ⟨src, s3, opsCommit⟩
      let s_after : LowerState := s3
      let ops_head : List KernelOp := opsCommit ++ [.store slot base src .u32]
      have h_head : lowerInstr s (.i32Store offset align) = some (s_after, ops_head) := by
        show lowerI32Store s = some (s_after, ops_head)
        unfold lowerI32Store
        simp only [LowerState.popSym, h_stack, Option.bind_eq_bind, Option.some_bind]
        rw [show ({ nextReg := s.nextReg, stack := lstk_rest,
                    localReg := s.localReg, localTy := s.localTy,
                    bufferSlots := s.bufferSlots } : LowerState).commit sv_val
                = s_pop.commit sv_val from rfl]
        rw [hca]
        rfl
      rw [h_head] at hl
      simp only [Option.bind_eq_bind, Option.some_bind] at hl
      cases h_post : lowerInstrs fuel frames s_after rest with
      | none => simp [h_post] at hl
      | some post_pair =>
          rcases post_pair with ⟨s_post, postOps⟩
          simp [h_post] at hl
          rcases hl with ⟨h_s_eq, h_ops_eq⟩
          rw [evalInstrs_cons_default fuel ws (.i32Store offset align) rest h_no_branch h_no_halt rfl] at hw
          cases h_eval_head : evalInstr ws (.i32Store offset align) with
          | none =>
              rw [h_eval_head] at hw
              simp at hw
          | some ws_after =>
              rw [h_eval_head] at hw
              simp only at hw
              obtain ⟨kst_mid, h_kst_eval, R_mid⟩ :=
                preservation_i32Store ws s kst layout R h_kst_no_broke
                  sv_val slot base lstk_rest offset align
                  h_stack h_offset h_in_bounds h_layout_no_overlap
                  ws_after s_after ops_head h_eval_head h_head
              -- loopFreeNoBreak on opsCommit (commit emits .const-only) + .store.
              have h_lf_commit : loopFreeNoBreak opsCommit = true :=
                commit_emits_loopFreeNoBreak hca
              have h_lf_store :
                  loopFreeNoBreak [KernelOp.store slot base src .u32] = true := rfl
              have h_lf_head : loopFreeNoBreak ops_head = true := by
                show loopFreeNoBreak (opsCommit ++ [KernelOp.store slot base src .u32]) = true
                simp [loopFreeNoBreak_append, h_lf_commit, h_lf_store]
              have h_lf_head_shallow : loopFree ops_head = true :=
                loopFreeNoBreak_implies_loopFree h_lf_head
              have h_mid_broke : kst_mid.broke = false :=
                evalOps_loopFreeNoBreak_preserves_broke
                  h_lf_head h_kst_no_broke h_kst_eval
              -- Mid-state branch/halt: storeI32 only mutates mem and pops.
              have h_w : evalInstr ws (.i32Store offset align) = storeI32 ws offset := rfl
              rw [h_w] at h_eval_head
              have h_ws_after_shape : ∃ ws_rest m',
                  ws_after = { ws with stack := ws_rest, mem := m' } := by
                unfold storeI32 at h_eval_head
                rcases hws : ws.stack with _ | ⟨vval, _ | ⟨vaddr, ws_rest⟩⟩
                · simp [hws, WasmState.pop] at h_eval_head
                · simp [hws, WasmState.pop] at h_eval_head
                · simp [hws, WasmState.pop] at h_eval_head
                  cases vaddr with
                  | wI32 addr_w =>
                      cases vval with
                      | wI32 v_w =>
                          simp at h_eval_head
                          rcases hmem : ws.mem.store_u32 (addr_w.toNat + offset) v_w with _ | m'
                          · simp [hmem] at h_eval_head
                          · simp [hmem] at h_eval_head
                            refine ⟨ws_rest, m', ?_⟩
                            rw [← h_eval_head]
                      | wI64 _ => simp at h_eval_head
                      | wF32 _ => simp at h_eval_head
                      | wF64 _ => simp at h_eval_head
                  | wI64 _ => simp at h_eval_head
                  | wF32 _ => simp at h_eval_head
                  | wF64 _ => simp at h_eval_head
              obtain ⟨_, _, h_ws_after_eq⟩ := h_ws_after_shape
              have h_mid_no_branch : ws_after.branchTarget = none := by
                rw [h_ws_after_eq]; simp [h_no_branch]
              have h_mid_no_halt : ws_after.halted = false := by
                rw [h_ws_after_eq]; simp [h_no_halt]
              obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest⟩ :=
                preservation_rest R_mid h_mid_no_branch h_mid_no_halt h_mid_broke hw h_post
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
-- L1a: 2-step buffer-pointer prelude chain
--
-- Composes the first two steps of the canonical buffer-access chain:
--   localGet bufSlotIdx :: localGet idxIdx :: rest
-- where `bufSlotIdx` reads a `#[quanta::shared]` buffer parameter
-- (lookupBufferSlot returns `some bSlot`) and `idxIdx` reads a plain
-- u32 index local (lookupBufferSlot returns `none`).
--
-- After the two steps the symbolic stack has
--   .reg s.nextReg .u32 :: .bufferPtr bSlot :: s.stack
-- and the regfile has `s.nextReg ↦ <value of idxIdx>` via the
-- emitted `.copy s.nextReg stable` op.
--
-- User-facing witnesses (the entry-state cleanliness contract):
--   h_buf            -- bufSlot binding agrees with layout
--   h_loc_buf        -- the bufSlot local stores the layout start addr
--   h_no_buf_idx     -- the idx local is non-buffer (at entry state s)
--
-- The `h_no_buf_idx` precondition is stated on `s`, but step 2's
-- `preservation_localGet` requires it on the post-step-1 state `s_1`.
-- Since `localGet bufSlot` does not modify `s.bufferSlots` (per
-- `lowerInstr_preserves_bufferSlots`), and `lookupBufferSlot` only
-- reads `bufferSlots` (per
-- `LowerState.lookupBufferSlot_of_bufferSlots_eq`), the witness lifts
-- through step 1 for free.
--
-- This is the L1a wedge of the full L1 buffer-access LOAD chain.
-- L1b extends through `i32Const k :: i32Shl :: rest` to land the
-- `.scaledIdx` SymVal on the stack; L1c extends through
-- `i32Add :: i32Load 0 align :: rest` to close the full chain.
-- ════════════════════════════════════════════════════════════════════

/-- L1a: the 2-step buffer-pointer address prelude. -/
theorem preservation_evalInstrs_chain_buffer_prelude_2step
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (bufSlotIdx : Nat) (bSlot : Nat)
    (h_buf : s.lookupBufferSlot bufSlotIdx = some bSlot)
    (h_loc_buf : ∀ v, ws.locals.get? bufSlotIdx = some v →
      ∃ n : UInt32, v = .wI32 n ∧ n.toNat = layout.startAddr bSlot)
    (idxIdx : Nat) (h_no_buf_idx : s.lookupBufferSlot idxIdx = none)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
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
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws
            (.localGet bufSlotIdx :: .localGet idxIdx :: rest) = some ws')
    (hl : lowerInstrs fuel frames s
            (.localGet bufSlotIdx :: .localGet idxIdx :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- Step 1 of the chain is `localGet bufSlotIdx`. We discharge it by
  -- delegating to `preservation_evalInstrs_cons_localGet_bufferSlot`
  -- and supplying the step 2 closure as its `preservation_rest`.
  --
  -- The step 2 closure has type access to `s_mid` only via R_mid +
  -- hl_mid (abstract). Since the cons theorem's conclusion doesn't
  -- expose the state shape, we cannot directly call
  -- `preservation_evalInstrs_cons_localGet` with `h_no_buf` —
  -- the closure needs to derive `s_mid.lookupBufferSlot idxIdx = none`
  -- from `h_no_buf_idx : s.lookupBufferSlot idxIdx = none` plus the
  -- fact that step 1 preserves `bufferSlots`.
  --
  -- The path that works: inline both steps at this level. Unfold the
  -- outer `lowerInstrs` to expose step 1's concrete output state s_1,
  -- apply `preservation_localGet_bufferSlot` to get R_1, then apply
  -- `LowerInvariants.lookupBufferSlot_of_bufferSlots_eq` to lift
  -- `h_no_buf_idx` to `s_1`, then proceed as a normal cons proof for
  -- step 2.
  rw [lowerInstrs_cons_default fuel frames s (.localGet bufSlotIdx) _ rfl] at hl
  -- Step 1's per-op head: lowerInstr s (.localGet bufSlotIdx) = some (s.pushSym (.bufferPtr bSlot), [])
  have h_head_1 : lowerInstr s (.localGet bufSlotIdx)
                    = some (s.pushSym (.bufferPtr bSlot), []) := by
    show (match s.lookupBufferSlot bufSlotIdx with
          | some slot => some (s.pushSym (.bufferPtr slot), [])
          | none =>
              (s.lookupLocal bufSlotIdx).bind (fun stable =>
                let (fresh, s1) := s.alloc
                let s2 := s1.push fresh
                some (s2, [KernelOp.copy fresh stable])))
              = some (s.pushSym (.bufferPtr bSlot), [])
    rw [h_buf]
  rw [h_head_1] at hl
  simp only [Option.bind_eq_bind, Option.some_bind, List.nil_append] at hl
  -- After step 1, hl reduces to lowerInstrs ... (s.pushSym (.bufferPtr bSlot)) (localGet idxIdx :: rest).
  -- The bind+pure eta needs the collapse helper.
  let s_1 : LowerState := s.pushSym (.bufferPtr bSlot)
  have hl_after_1 : lowerInstrs fuel frames s_1 (.localGet idxIdx :: rest) = some (s', ops) :=
    cons_default_lowerInstrs_collapse_empty_head hl
  -- Lift h_no_buf_idx to s_1.
  have h_bufs_eq : s_1.bufferSlots = s.bufferSlots := by
    simp [s_1, LowerState.pushSym]
  have h_no_buf_idx_1 : s_1.lookupBufferSlot idxIdx = none := by
    rw [LowerState.lookupBufferSlot_of_bufferSlots_eq idxIdx h_bufs_eq]
    exact h_no_buf_idx
  -- Unfold step 2.
  rw [lowerInstrs_cons_default fuel frames s_1 (.localGet idxIdx) _ rfl] at hl_after_1
  -- Extract step 2's per-op result.
  cases h_stable : s_1.lookupLocal idxIdx with
  | none =>
      -- lookupLocal failed → lowerInstr returns none → lowerInstrs returns none.
      simp only [lowerInstr, h_no_buf_idx_1, h_stable, Option.bind_eq_bind,
                 Option.some_bind, Option.none_bind, LowerState.alloc,
                 LowerState.push] at hl_after_1
      exact (Option.noConfusion hl_after_1)
  | some stable =>
      let s_2 : LowerState :=
        { s_1 with nextReg := s_1.nextReg + 1,
                   stack := SymVal.reg s_1.nextReg .u32 :: s_1.stack }
      let ops_head_2 : List KernelOp := [.copy s_1.nextReg stable]
      have hl_head_2 : lowerInstr s_1 (.localGet idxIdx) = some (s_2, ops_head_2) := by
        show (match s_1.lookupBufferSlot idxIdx with
              | some slot => some (s_1.pushSym (.bufferPtr slot), [])
              | none => do
                  let stable ← s_1.lookupLocal idxIdx
                  let (fresh, s1) := s_1.alloc
                  let s2 := s1.push fresh
                  pure (s2, [.copy fresh stable])) = some (s_2, ops_head_2)
        rw [h_no_buf_idx_1, h_stable]
        rfl
      rw [hl_head_2] at hl_after_1
      simp only [Option.bind_eq_bind, Option.some_bind] at hl_after_1
      cases h_post : lowerInstrs fuel frames s_2 rest with
      | none => simp [h_post] at hl_after_1
      | some post_pair =>
          rcases post_pair with ⟨s_post, postOps⟩
          simp [h_post] at hl_after_1
          rcases hl_after_1 with ⟨h_s_eq, h_ops_eq⟩
          -- Eval side: unfold both step 1 and step 2.
          rw [evalInstrs_cons_default fuel ws (.localGet bufSlotIdx) _
                h_no_branch h_no_halt rfl] at hw
          cases h_eval_1 : evalInstr ws (.localGet bufSlotIdx) with
          | none =>
              rw [h_eval_1] at hw
              simp at hw
          | some ws_1 =>
              rw [h_eval_1] at hw
              simp only at hw
              -- Apply per-op preservation_localGet_bufferSlot for step 1.
              obtain ⟨kst_1, h_kst_eval_1, R_1⟩ :=
                preservation_localGet_bufferSlot ws s kst layout R bufSlotIdx bSlot
                  h_buf h_loc_buf
                  ws_1 (s.pushSym (.bufferPtr bSlot)) []
                  h_eval_1 h_head_1
              -- Step 1 emits no IR, so kst_1 = kst.
              have h_kst_1_eq : kst_1 = kst := by
                simp [evalOps] at h_kst_eval_1
                exact h_kst_eval_1.symm
              -- ws_1 has the pushed wI32 on top.
              have h_ws_1_shape : ∃ v, ws_1 = ws.push v := by
                cases h_loc : ws.getLocal bufSlotIdx with
                | none =>
                    have h_ev : evalInstr ws (.localGet bufSlotIdx) = none := by
                      show (do let v ← ws.getLocal bufSlotIdx; pure (ws.push v)) = none
                      rw [h_loc]; rfl
                    rw [h_ev] at h_eval_1; exact (Option.noConfusion h_eval_1)
                | some v =>
                    refine ⟨v, ?_⟩
                    have h_ev : evalInstr ws (.localGet bufSlotIdx) = some (ws.push v) := by
                      show (do let v ← ws.getLocal bufSlotIdx; pure (ws.push v)) = some (ws.push v)
                      rw [h_loc]; rfl
                    rw [h_ev] at h_eval_1
                    exact ((Option.some.injEq _ _).mp h_eval_1).symm
              obtain ⟨v_1, h_ws_1_eq⟩ := h_ws_1_shape
              have h_1_no_branch : ws_1.branchTarget = none := by
                rw [h_ws_1_eq]; simp [WasmState.push, h_no_branch]
              have h_1_no_halt : ws_1.halted = false := by
                rw [h_ws_1_eq]; simp [WasmState.push, h_no_halt]
              -- Step 2's eval head.
              rw [evalInstrs_cons_default fuel ws_1 (.localGet idxIdx) _
                    h_1_no_branch h_1_no_halt rfl] at hw
              cases h_eval_2 : evalInstr ws_1 (.localGet idxIdx) with
              | none =>
                  rw [h_eval_2] at hw
                  simp at hw
              | some ws_2 =>
                  rw [h_eval_2] at hw
                  simp only at hw
                  -- Apply per-op preservation_localGet for step 2.
                  -- Refines bundle for s_1 = s.pushSym (.bufferPtr bSlot) comes from R_1.
                  obtain ⟨kst_2, h_kst_eval_2, R_2⟩ :=
                    preservation_localGet ws_1 s_1 kst_1 layout R_1 idxIdx h_no_buf_idx_1
                      ws_2 s_2 ops_head_2
                      h_eval_2 hl_head_2
                  -- Derive kst_2.broke = false via .copy preservation.
                  have h_2_broke : kst_2.broke = false := by
                    have := evalOps_copy_singleton_preserves_broke h_kst_eval_2
                    rw [this]; rw [h_kst_1_eq]; exact h_kst_no_broke
                  -- ws_2 = ws_1.push v_2 — branchTarget / halted preserved.
                  have h_ws_2_shape : ∃ v, ws_2 = ws_1.push v := by
                    cases h_loc : ws_1.getLocal idxIdx with
                    | none =>
                        have h_ev : evalInstr ws_1 (.localGet idxIdx) = none := by
                          show (do let v ← ws_1.getLocal idxIdx; pure (ws_1.push v)) = none
                          rw [h_loc]; rfl
                        rw [h_ev] at h_eval_2; exact (Option.noConfusion h_eval_2)
                    | some v =>
                        refine ⟨v, ?_⟩
                        have h_ev : evalInstr ws_1 (.localGet idxIdx) = some (ws_1.push v) := by
                          show (do let v ← ws_1.getLocal idxIdx; pure (ws_1.push v)) = some (ws_1.push v)
                          rw [h_loc]; rfl
                        rw [h_ev] at h_eval_2
                        exact ((Option.some.injEq _ _).mp h_eval_2).symm
                  obtain ⟨v_2, h_ws_2_eq⟩ := h_ws_2_shape
                  have h_2_no_branch : ws_2.branchTarget = none := by
                    rw [h_ws_2_eq]; simp [WasmState.push, h_1_no_branch]
                  have h_2_no_halt : ws_2.halted = false := by
                    rw [h_ws_2_eq]; simp [WasmState.push, h_1_no_halt]
                  -- Apply user's preservation_rest on the tail.
                  obtain ⟨kst'_mid, F_rest, h_eval_rest, R_rest⟩ :=
                    preservation_rest R_2 h_2_no_branch h_2_no_halt h_2_broke hw h_post
                  -- Now compose: step 1 emits [], step 2 emits [.copy ...], tail emits postOps.
                  -- Total ops = [] ++ [.copy ...] ++ postOps = [.copy ...] ++ postOps.
                  -- evalOps composition: kst → kst (via step 1) → kst_2 (via step 2) → kst'_mid.
                  -- step 1's evalOps 0 kst [] = some kst (kst_1 = kst).
                  -- step 2's evalOps F_2 kst_1 [.copy ...] = some kst_2.
                  -- tail's evalOps F_rest kst_2 postOps = some kst'_mid.
                  -- Chain via cons-composer (shallow) on the step-2 head + tail.
                  have h_lf_2 : loopFree ops_head_2 = true := by
                    simp [loopFree, loopFreeOp, ops_head_2]
                  -- Rewrite h_kst_eval_2 in terms of kst (since kst_1 = kst).
                  rw [h_kst_1_eq] at h_kst_eval_2
                  have h_chained :
                      ∃ kst'', evalOps F_rest kst (ops_head_2 ++ postOps) = some kst''
                        ∧ Refines ws' s_post kst'' layout :=
                    preservation_evalInstrs_cons_compose_shallow
                      h_lf_2 h_kst_eval_2 h_2_broke ⟨kst'_mid, h_eval_rest, R_rest⟩
                  obtain ⟨kst'', h_eval'', R''⟩ := h_chained
                  refine ⟨kst'', F_rest, ?_, ?_⟩
                  · -- ops = [] ++ ops_head_2 ++ postOps = ops_head_2 ++ postOps.
                    rw [← h_ops_eq]
                    show evalOps F_rest kst ([] ++ (ops_head_2 ++ postOps)) = some kst''
                    rw [List.nil_append]
                    exact h_eval''
                  · rw [← h_s_eq]; exact R''

-- ════════════════════════════════════════════════════════════════════
-- L1b: 4-step buffer-pointer + scaled-index prelude chain
--
-- Extends L1a (2-step buffer-pointer prelude) through `i32Const k ::
-- i32Shl :: rest`. After the chain the symbolic stack has
--   .scaledIdx s.nextReg (1 <<< k.toNat) :: .bufferPtr bSlot :: s.stack
-- still with no IR emitted by steps 3+4 — the buffer-pattern fold for
-- i32Shl recognizes the `.i32ConstSym k :: .reg base ty :: rest`
-- shape on the symbolic stack and rewrites the top two slots into
-- a single `.scaledIdx` without emitting IR.
--
-- User-facing witnesses extend L1a's set with `h_shift_eq`, the
-- UInt32 shift-as-multiplication identity. Stated as a pure UInt32
-- property here (no kst dependency) — the chain proof specializes
-- it to `regLookup kst_2.rf s.nextReg`.
--
-- The 4-step chain still emits only the single `.copy s.nextReg
-- stable` op that step 2 contributes; steps 1, 3, 4 are all no-IR.
-- ════════════════════════════════════════════════════════════════════

/-- L1b: the 4-step buffer-pointer + scaled-index prelude. -/
theorem preservation_evalInstrs_chain_buffer_prelude_4step
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (bufSlotIdx : Nat) (bSlot : Nat)
    (h_buf : s.lookupBufferSlot bufSlotIdx = some bSlot)
    (h_loc_buf : ∀ v, ws.locals.get? bufSlotIdx = some v →
      ∃ n : UInt32, v = .wI32 n ∧ n.toNat = layout.startAddr bSlot)
    (idxIdx : Nat) (h_no_buf_idx : s.lookupBufferSlot idxIdx = none)
    (k : Int)
    (h_shift_eq : ∀ a : UInt32,
       (a <<< (UInt32.ofNat k.toNat)).toNat = a.toNat * (1 <<< k.toNat))
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
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
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws
            (.localGet bufSlotIdx :: .localGet idxIdx ::
             .i32Const k :: .i32Shl :: rest) = some ws')
    (hl : lowerInstrs fuel frames s
            (.localGet bufSlotIdx :: .localGet idxIdx ::
             .i32Const k :: .i32Shl :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- ── Step 1: localGet bufSlotIdx (buffer arm) ───────────────────────
  rw [lowerInstrs_cons_default fuel frames s (.localGet bufSlotIdx) _ rfl] at hl
  have h_head_1 : lowerInstr s (.localGet bufSlotIdx)
                    = some (s.pushSym (.bufferPtr bSlot), []) := by
    show (match s.lookupBufferSlot bufSlotIdx with
          | some slot => some (s.pushSym (.bufferPtr slot), [])
          | none =>
              (s.lookupLocal bufSlotIdx).bind (fun stable =>
                let (fresh, s1) := s.alloc
                let s2 := s1.push fresh
                some (s2, [KernelOp.copy fresh stable])))
              = some (s.pushSym (.bufferPtr bSlot), [])
    rw [h_buf]
  rw [h_head_1] at hl
  simp only [Option.bind_eq_bind, Option.some_bind, List.nil_append] at hl
  let s_1 : LowerState := s.pushSym (.bufferPtr bSlot)
  have hl_after_1 : lowerInstrs fuel frames s_1
                      (.localGet idxIdx :: .i32Const k :: .i32Shl :: rest)
                    = some (s', ops) :=
    cons_default_lowerInstrs_collapse_empty_head hl
  -- Lift h_no_buf_idx to s_1.
  have h_bufs_eq_1 : s_1.bufferSlots = s.bufferSlots := by
    simp [s_1, LowerState.pushSym]
  have h_no_buf_idx_1 : s_1.lookupBufferSlot idxIdx = none := by
    rw [LowerState.lookupBufferSlot_of_bufferSlots_eq idxIdx h_bufs_eq_1]
    exact h_no_buf_idx
  -- ── Step 2: localGet idxIdx (non-buffer arm) ───────────────────────
  rw [lowerInstrs_cons_default fuel frames s_1 (.localGet idxIdx) _ rfl] at hl_after_1
  cases h_stable : s_1.lookupLocal idxIdx with
  | none =>
      simp only [lowerInstr, h_no_buf_idx_1, h_stable, Option.bind_eq_bind,
                 Option.some_bind, Option.none_bind, LowerState.alloc,
                 LowerState.push] at hl_after_1
      exact (Option.noConfusion hl_after_1)
  | some stable =>
      let s_2 : LowerState :=
        { s_1 with nextReg := s_1.nextReg + 1,
                   stack := SymVal.reg s_1.nextReg .u32 :: s_1.stack }
      let ops_head_2 : List KernelOp := [.copy s_1.nextReg stable]
      have hl_head_2 : lowerInstr s_1 (.localGet idxIdx) = some (s_2, ops_head_2) := by
        show (match s_1.lookupBufferSlot idxIdx with
              | some slot => some (s_1.pushSym (.bufferPtr slot), [])
              | none => do
                  let stable ← s_1.lookupLocal idxIdx
                  let (fresh, s1) := s_1.alloc
                  let s2 := s1.push fresh
                  pure (s2, [.copy fresh stable])) = some (s_2, ops_head_2)
        rw [h_no_buf_idx_1, h_stable]
        rfl
      rw [hl_head_2] at hl_after_1
      simp only [Option.bind_eq_bind, Option.some_bind] at hl_after_1
      -- ── Step 3: i32Const k ─────────────────────────────────────────
      -- lowerInstr s_2 (.i32Const k) = some (s_2.pushSym (.i32ConstSym k), [])
      rw [lowerInstrs_cons_default fuel frames s_2 (.i32Const k) _ rfl] at hl_after_1
      have h_head_3 : lowerInstr s_2 (.i32Const k) =
          some (s_2.pushSym (.i32ConstSym k), []) := by
        rfl
      rw [h_head_3] at hl_after_1
      simp only [Option.bind_eq_bind, Option.some_bind, List.nil_append] at hl_after_1
      -- After step 3, hl_after_1's body is a bind on `lowerInstrs fuel
      -- frames (s_2.pushSym (.i32ConstSym k)) (.i32Shl :: rest)` with
      -- pure (s2, ops_head_2 ++ ops2). Extract via case-split.
      let s_3 : LowerState := s_2.pushSym (.i32ConstSym k)
      cases h_step4 : lowerInstrs fuel frames s_3 (.i32Shl :: rest) with
      | none =>
          simp [s_3, h_step4] at hl_after_1
      | some step4_pair =>
          rcases step4_pair with ⟨s_post, postOps_inner⟩
          simp [s_3, h_step4] at hl_after_1
          rcases hl_after_1 with ⟨h_s_eq, h_ops_eq⟩
          -- ── Step 4: i32Shl (buffer-pattern arm) ────────────────────
          -- We now know lowerInstrs fuel frames s_3 (.i32Shl :: rest) = some (s_post, postOps_inner).
          -- Need to apply the cons theorem for i32Shl_bufferPattern.
          -- s_3.stack = .i32ConstSym k :: .reg s_1.nextReg .u32 :: .bufferPtr bSlot :: s.stack
          have h_s_3_stack : s_3.stack = .i32ConstSym k :: .reg s_1.nextReg .u32 ::
                                          .bufferPtr bSlot :: s.stack := by
            show (s_2.pushSym (.i32ConstSym k)).stack = _
            simp [s_2, s_1, LowerState.pushSym]
          -- ── Eval side: unfold all 4 steps ────────────────────────
          rw [evalInstrs_cons_default fuel ws (.localGet bufSlotIdx) _
                h_no_branch h_no_halt rfl] at hw
          cases h_eval_1 : evalInstr ws (.localGet bufSlotIdx) with
          | none =>
              rw [h_eval_1] at hw
              simp at hw
          | some ws_1 =>
              rw [h_eval_1] at hw
              simp only at hw
              obtain ⟨kst_1, h_kst_eval_1, R_1⟩ :=
                preservation_localGet_bufferSlot ws s kst layout R bufSlotIdx bSlot
                  h_buf h_loc_buf
                  ws_1 (s.pushSym (.bufferPtr bSlot)) []
                  h_eval_1 h_head_1
              have h_kst_1_eq : kst_1 = kst := by
                simp [evalOps] at h_kst_eval_1
                exact h_kst_eval_1.symm
              have h_ws_1_shape : ∃ v, ws_1 = ws.push v := by
                cases h_loc : ws.getLocal bufSlotIdx with
                | none =>
                    have h_ev : evalInstr ws (.localGet bufSlotIdx) = none := by
                      show (do let v ← ws.getLocal bufSlotIdx; pure (ws.push v)) = none
                      rw [h_loc]; rfl
                    rw [h_ev] at h_eval_1; exact (Option.noConfusion h_eval_1)
                | some v =>
                    refine ⟨v, ?_⟩
                    have h_ev : evalInstr ws (.localGet bufSlotIdx) = some (ws.push v) := by
                      show (do let v ← ws.getLocal bufSlotIdx; pure (ws.push v)) = some (ws.push v)
                      rw [h_loc]; rfl
                    rw [h_ev] at h_eval_1
                    exact ((Option.some.injEq _ _).mp h_eval_1).symm
              obtain ⟨v_1, h_ws_1_eq⟩ := h_ws_1_shape
              have h_1_no_branch : ws_1.branchTarget = none := by
                rw [h_ws_1_eq]; simp [WasmState.push, h_no_branch]
              have h_1_no_halt : ws_1.halted = false := by
                rw [h_ws_1_eq]; simp [WasmState.push, h_no_halt]
              -- Step 2's eval head.
              rw [evalInstrs_cons_default fuel ws_1 (.localGet idxIdx) _
                    h_1_no_branch h_1_no_halt rfl] at hw
              cases h_eval_2 : evalInstr ws_1 (.localGet idxIdx) with
              | none =>
                  rw [h_eval_2] at hw
                  simp at hw
              | some ws_2 =>
                  rw [h_eval_2] at hw
                  simp only at hw
                  obtain ⟨kst_2, h_kst_eval_2, R_2⟩ :=
                    preservation_localGet ws_1 s_1 kst_1 layout R_1 idxIdx h_no_buf_idx_1
                      ws_2 s_2 ops_head_2
                      h_eval_2 hl_head_2
                  have h_2_broke : kst_2.broke = false := by
                    have := evalOps_copy_singleton_preserves_broke h_kst_eval_2
                    rw [this]; rw [h_kst_1_eq]; exact h_kst_no_broke
                  have h_ws_2_shape : ∃ v, ws_2 = ws_1.push v := by
                    cases h_loc : ws_1.getLocal idxIdx with
                    | none =>
                        have h_ev : evalInstr ws_1 (.localGet idxIdx) = none := by
                          show (do let v ← ws_1.getLocal idxIdx; pure (ws_1.push v)) = none
                          rw [h_loc]; rfl
                        rw [h_ev] at h_eval_2; exact (Option.noConfusion h_eval_2)
                    | some v =>
                        refine ⟨v, ?_⟩
                        have h_ev : evalInstr ws_1 (.localGet idxIdx) = some (ws_1.push v) := by
                          show (do let v ← ws_1.getLocal idxIdx; pure (ws_1.push v)) = some (ws_1.push v)
                          rw [h_loc]; rfl
                        rw [h_ev] at h_eval_2
                        exact ((Option.some.injEq _ _).mp h_eval_2).symm
                  obtain ⟨v_2, h_ws_2_eq⟩ := h_ws_2_shape
                  have h_2_no_branch : ws_2.branchTarget = none := by
                    rw [h_ws_2_eq]; simp [WasmState.push, h_1_no_branch]
                  have h_2_no_halt : ws_2.halted = false := by
                    rw [h_ws_2_eq]; simp [WasmState.push, h_1_no_halt]
                  -- ── Step 3 eval: i32Const k ──────────────────────
                  rw [evalInstrs_cons_default fuel ws_2 (.i32Const k) _
                        h_2_no_branch h_2_no_halt rfl] at hw
                  -- evalInstr ws_2 (.i32Const k) = some (ws_2.push (.wI32 ...))
                  have h_eval_3 : evalInstr ws_2 (.i32Const k) =
                      some (ws_2.push (.wI32 (UInt32.ofNat k.toNat))) := rfl
                  rw [h_eval_3] at hw
                  simp only at hw
                  let ws_3 := ws_2.push (.wI32 (UInt32.ofNat k.toNat))
                  -- Apply per-op preservation_i32Const for step 3 to get R_3.
                  obtain ⟨kst_3, h_kst_eval_3, R_3⟩ :=
                    preservation_i32Const ws_2 s_2 kst_2 layout R_2 k
                      ws_3 (s_2.pushSym (.i32ConstSym k)) []
                      h_eval_3 h_head_3
                  have h_kst_3_eq : kst_3 = kst_2 := by
                    simp [evalOps] at h_kst_eval_3
                    exact h_kst_eval_3.symm
                  have h_3_no_branch : ws_3.branchTarget = none := by
                    simp [ws_3, WasmState.push, h_2_no_branch]
                  have h_3_no_halt : ws_3.halted = false := by
                    simp [ws_3, WasmState.push, h_2_no_halt]
                  have h_3_broke : kst_3.broke = false := by
                    rw [h_kst_3_eq]; exact h_2_broke
                  -- ── Step 4: i32Shl (buffer-pattern arm) ──────────
                  -- Apply the i32Shl_bufferPattern cons theorem with rest as the tail.
                  -- s_3 = s_2.pushSym (.i32ConstSym k), and h_s_3_stack establishes the stack shape.
                  -- We need to call the cons theorem with:
                  --   ws = ws_3, s = s_3, kst = kst_3, R = R_3
                  --   k = k, base = s_1.nextReg, ty = .u32, lstk_rest = .bufferPtr bSlot :: s.stack
                  --   h_stack = h_s_3_stack
                  --   h_shift_eq specialized to kst_3.rf s_1.nextReg
                  --   rest as the tail; preservation_rest as user's
                  -- The cons theorem closes the chain.
                  have h_shift_eq_specialized :
                      ∀ a : UInt32,
                        regLookup kst_3.rf s_1.nextReg = some (Quanta.KOps.Value.vU32 a) →
                        (a <<< (UInt32.ofNat k.toNat)).toNat = a.toNat * (1 <<< k.toNat) := by
                    intro a _
                    exact h_shift_eq a
                  -- Eval side: hw now has shape `evalInstrs fuel ws_3 (.i32Shl :: rest) = some ws'`.
                  -- Lower side: we need lowerInstrs fuel frames s_3 (.i32Shl :: rest) = some (s', ops')
                  -- with ops' matching the residue. We already have h_step4.
                  -- But the ops accumulated so far are ops_head_2; the cons theorem's output ops
                  -- will be its own head ops + postOps. The relationship:
                  --   ops = ops_head_2 ++ (cons-theorem output ops) = ops_head_2 ++ postOps_inner
                  obtain ⟨kst', F_chain, h_eval_chain, R_chain⟩ :=
                    preservation_evalInstrs_cons_i32Shl_bufferPattern
                      fuel frames ws_3 s_3 kst_3 layout R_3
                      h_3_no_branch h_3_no_halt h_3_broke
                      k s_1.nextReg .u32 (.bufferPtr bSlot :: s.stack)
                      h_s_3_stack
                      h_shift_eq_specialized
                      rest preservation_rest
                      ws' s_post postOps_inner
                      hw h_step4
                  -- Now compose: total kst-evolution = kst → kst (step 1, []) → kst_2 (step 2, [.copy])
                  -- → kst_2 (step 3, []) → kst' (step 4 + rest, postOps_inner).
                  -- Total ops = ops_head_2 ++ postOps_inner.
                  -- We need evalOps F_total kst (ops_head_2 ++ postOps_inner) = some kst'.
                  have h_lf_2 : loopFree ops_head_2 = true := by
                    simp [loopFree, loopFreeOp, ops_head_2]
                  -- h_kst_eval_2 : evalOps 0 kst_1 ops_head_2 = some kst_2.
                  -- Rewrite via kst_1 = kst.
                  rw [h_kst_1_eq] at h_kst_eval_2
                  -- h_eval_chain : evalOps F_chain kst_3 postOps_inner = some kst'.
                  -- Rewrite via kst_3 = kst_2.
                  rw [h_kst_3_eq] at h_eval_chain
                  -- Now compose: evalOps F_chain kst (ops_head_2 ++ postOps_inner) = some kst'.
                  have h_chained :
                      ∃ kst'', evalOps F_chain kst (ops_head_2 ++ postOps_inner) = some kst''
                        ∧ Refines ws' s_post kst'' layout :=
                    preservation_evalInstrs_cons_compose_shallow
                      h_lf_2 h_kst_eval_2 h_2_broke ⟨kst', h_eval_chain, R_chain⟩
                  obtain ⟨kst'', h_eval'', R''⟩ := h_chained
                  refine ⟨kst'', F_chain, ?_, ?_⟩
                  · rw [← h_ops_eq]
                    show evalOps F_chain kst ([] ++ (ops_head_2 ++ postOps_inner)) = some kst''
                    rw [List.nil_append]
                    exact h_eval''
                  · rw [← h_s_eq]; exact R''

-- ════════════════════════════════════════════════════════════════════
-- L1c: full 6-step buffer-access LOAD chain (closes L1)
--
-- Composes the canonical buffer-access load chain:
--   localGet bufSlot :: localGet idx :: i32Const k :: i32Shl ::
--   i32Add :: i32Load 0 align :: rest
-- After the 6-step chain the symbolic stack has
--   .reg (s.nextReg + 1) .u32 :: s.stack
-- and the emitted IR is
--   [.copy s.nextReg stable, .load (s.nextReg + 1) bSlot s.nextReg .u32]
--
-- The chain captures the canonical kernel idiom `buffer[i]` for u32
-- buffers (the load reads 4 bytes from `bSlot + i * 4`). The user
-- supplies `k = 2` (so `1 <<< k.toNat = 4` matches `lowerI32Load`'s
-- scale-4 requirement), plus arithmetic witnesses `h_shift_eq`,
-- `h_addr_eq` (UInt32 add as multiplication-by-stride) and the
-- load-bounds witness `h_in_bounds`.
--
-- Proof structure: full-inline. All 6 lowering unfolds + 6 eval
-- unfolds + 6 per-op preservation invocations are interleaved at
-- this level, with the user's `preservation_rest` applied on the
-- tail. The final ops are composed via two `evalOps_append`
-- applications (one for the step-2 `.copy`, one for the step-6
-- `.load`).
-- ════════════════════════════════════════════════════════════════════

/-- L1c: the full 6-step buffer-access LOAD chain. -/
theorem preservation_evalInstrs_chain_buffer_load_6step
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (bufSlotIdx : Nat) (bSlot : Nat)
    (h_buf : s.lookupBufferSlot bufSlotIdx = some bSlot)
    (h_loc_buf : ∀ v, ws.locals.get? bufSlotIdx = some v →
      ∃ n : UInt32, v = .wI32 n ∧ n.toNat = layout.startAddr bSlot)
    (idxIdx : Nat) (h_no_buf_idx : s.lookupBufferSlot idxIdx = none)
    (k : Int) (h_k_eq_2 : k = 2)
    (h_shift_eq : ∀ a : UInt32,
       (a <<< (UInt32.ofNat k.toNat)).toNat = a.toNat * (1 <<< k.toNat))
    (h_addr_eq : ∀ a b_ptr b : UInt32,
       a.toNat = b.toNat * (1 <<< k.toNat) →
       b_ptr.toNat = layout.startAddr bSlot →
       (b_ptr + a).toNat = layout.startAddr bSlot + b.toNat * (1 <<< k.toNat))
    (offset align : Nat) (h_offset : offset = 0)
    (h_in_bounds : ∀ b : UInt32, b.toNat < layout.length bSlot)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
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
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws
            (.localGet bufSlotIdx :: .localGet idxIdx ::
             .i32Const k :: .i32Shl :: .i32Add ::
             .i32Load offset align :: rest) = some ws')
    (hl : lowerInstrs fuel frames s
            (.localGet bufSlotIdx :: .localGet idxIdx ::
             .i32Const k :: .i32Shl :: .i32Add ::
             .i32Load offset align :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- ── Lowering side: unfold all 6 cons_default steps ─────────────────
  rw [lowerInstrs_cons_default fuel frames s (.localGet bufSlotIdx) _ rfl] at hl
  have h_head_1 : lowerInstr s (.localGet bufSlotIdx)
                    = some (s.pushSym (.bufferPtr bSlot), []) := by
    show (match s.lookupBufferSlot bufSlotIdx with
          | some slot => some (s.pushSym (.bufferPtr slot), [])
          | none =>
              (s.lookupLocal bufSlotIdx).bind (fun stable =>
                let (fresh, s1) := s.alloc
                let s2 := s1.push fresh
                some (s2, [KernelOp.copy fresh stable])))
              = some (s.pushSym (.bufferPtr bSlot), [])
    rw [h_buf]
  rw [h_head_1] at hl
  simp only [Option.bind_eq_bind, Option.some_bind, List.nil_append] at hl
  let s_1 : LowerState := s.pushSym (.bufferPtr bSlot)
  have h_bufs_eq_1 : s_1.bufferSlots = s.bufferSlots := by
    simp [s_1, LowerState.pushSym]
  have h_no_buf_idx_1 : s_1.lookupBufferSlot idxIdx = none := by
    rw [LowerState.lookupBufferSlot_of_bufferSlots_eq idxIdx h_bufs_eq_1]
    exact h_no_buf_idx
  rw [lowerInstrs_cons_default fuel frames s_1 (.localGet idxIdx) _ rfl] at hl
  cases h_stable : s_1.lookupLocal idxIdx with
  | none =>
      simp only [lowerInstr, h_no_buf_idx_1, h_stable, Option.bind_eq_bind,
                 Option.some_bind, Option.none_bind, LowerState.alloc,
                 LowerState.push] at hl
      exact (Option.noConfusion hl)
  | some stable =>
      let s_2 : LowerState :=
        { s_1 with nextReg := s_1.nextReg + 1,
                   stack := SymVal.reg s_1.nextReg .u32 :: s_1.stack }
      let ops_2 : List KernelOp := [.copy s_1.nextReg stable]
      have h_head_2 : lowerInstr s_1 (.localGet idxIdx) = some (s_2, ops_2) := by
        show (match s_1.lookupBufferSlot idxIdx with
              | some slot => some (s_1.pushSym (.bufferPtr slot), [])
              | none => do
                  let stable ← s_1.lookupLocal idxIdx
                  let (fresh, s1) := s_1.alloc
                  let s2 := s1.push fresh
                  pure (s2, [.copy fresh stable])) = some (s_2, ops_2)
        rw [h_no_buf_idx_1, h_stable]
        rfl
      rw [h_head_2] at hl
      simp only [Option.bind_eq_bind, Option.some_bind] at hl
      rw [lowerInstrs_cons_default fuel frames s_2 (.i32Const k) _ rfl] at hl
      have h_head_3 : lowerInstr s_2 (.i32Const k) =
          some (s_2.pushSym (.i32ConstSym k), []) := rfl
      rw [h_head_3] at hl
      simp only [Option.bind_eq_bind, Option.some_bind, List.nil_append] at hl
      let s_3 : LowerState := s_2.pushSym (.i32ConstSym k)
      have h_s_3_stack : s_3.stack = .i32ConstSym k :: .reg s_1.nextReg .u32 ::
                                      .bufferPtr bSlot :: s.stack := by
        show (s_2.pushSym (.i32ConstSym k)).stack = _
        simp [s_2, s_1, LowerState.pushSym]
      rw [lowerInstrs_cons_default fuel frames s_3 .i32Shl _ rfl] at hl
      let s_4 : LowerState :=
        { s_3 with stack := .scaledIdx s_1.nextReg (1 <<< k.toNat) ::
                             .bufferPtr bSlot :: s.stack }
      have h_head_4 : lowerInstr s_3 .i32Shl = some (s_4, []) := by
        show lowerI32Shl s_3 = some (s_4, [])
        unfold lowerI32Shl
        rw [h_s_3_stack]
      rw [h_head_4] at hl
      simp only [Option.bind_eq_bind, Option.some_bind, List.nil_append] at hl
      have h_s_4_stack : s_4.stack = .scaledIdx s_1.nextReg (1 <<< k.toNat) ::
                                       .bufferPtr bSlot :: s.stack := rfl
      rw [lowerInstrs_cons_default fuel frames s_4 .i32Add _ rfl] at hl
      let s_5 : LowerState :=
        { s_4 with stack := .bufferAccess bSlot s_1.nextReg (1 <<< k.toNat) :: s.stack }
      have h_head_5 : lowerInstr s_4 .i32Add = some (s_5, []) := by
        show lowerI32Add s_4 = some (s_5, [])
        unfold lowerI32Add
        rw [h_s_4_stack]
      rw [h_head_5] at hl
      simp only [Option.bind_eq_bind, Option.some_bind, List.nil_append] at hl
      have h_scale_4 : (1 <<< k.toNat) = 4 := by rw [h_k_eq_2]; rfl
      have h_s_5_stack : s_5.stack = .bufferAccess bSlot s_1.nextReg 4 :: s.stack := by
        show (.bufferAccess bSlot s_1.nextReg (1 <<< k.toNat) :: s.stack) = _
        rw [h_scale_4]
      rw [lowerInstrs_cons_default fuel frames s_5 (.i32Load offset align) _ rfl] at hl
      let s_6 : LowerState :=
        { s_5 with nextReg := s_5.nextReg + 1,
                   stack := .reg s_5.nextReg .u32 :: s.stack }
      let ops_6 : List KernelOp := [.load s_5.nextReg bSlot s_1.nextReg .u32]
      have h_head_6 : lowerInstr s_5 (.i32Load offset align) = some (s_6, ops_6) := by
        show lowerI32Load s_5 = some (s_6, ops_6)
        unfold lowerI32Load
        rw [h_s_5_stack]
        rfl
      rw [h_head_6] at hl
      simp only [Option.bind_eq_bind, Option.some_bind] at hl
      cases h_post : lowerInstrs fuel frames s_6 rest with
      | none => simp [h_post] at hl
      | some post_pair =>
          rcases post_pair with ⟨s_post, postOps⟩
          simp [h_post] at hl
          rcases hl with ⟨h_s_eq, h_ops_eq⟩
          -- ── Eval side: unfold all 6 cons_default steps ────────────
          rw [evalInstrs_cons_default fuel ws (.localGet bufSlotIdx) _
                h_no_branch h_no_halt rfl] at hw
          cases h_eval_1 : evalInstr ws (.localGet bufSlotIdx) with
          | none =>
              rw [h_eval_1] at hw
              simp at hw
          | some ws_1 =>
              rw [h_eval_1] at hw
              simp only at hw
              obtain ⟨kst_1, h_kst_eval_1, R_1⟩ :=
                preservation_localGet_bufferSlot ws s kst layout R bufSlotIdx bSlot
                  h_buf h_loc_buf
                  ws_1 s_1 []
                  h_eval_1 h_head_1
              have h_kst_1_eq : kst_1 = kst := by
                simp [evalOps] at h_kst_eval_1
                exact h_kst_eval_1.symm
              have h_ws_1_shape : ∃ v, ws_1 = ws.push v := by
                cases h_loc : ws.getLocal bufSlotIdx with
                | none =>
                    have h_ev : evalInstr ws (.localGet bufSlotIdx) = none := by
                      show (do let v ← ws.getLocal bufSlotIdx; pure (ws.push v)) = none
                      rw [h_loc]; rfl
                    rw [h_ev] at h_eval_1; exact (Option.noConfusion h_eval_1)
                | some v =>
                    refine ⟨v, ?_⟩
                    have h_ev : evalInstr ws (.localGet bufSlotIdx) = some (ws.push v) := by
                      show (do let v ← ws.getLocal bufSlotIdx; pure (ws.push v)) = some (ws.push v)
                      rw [h_loc]; rfl
                    rw [h_ev] at h_eval_1
                    exact ((Option.some.injEq _ _).mp h_eval_1).symm
              obtain ⟨v_1, h_ws_1_eq⟩ := h_ws_1_shape
              have h_1_no_branch : ws_1.branchTarget = none := by
                rw [h_ws_1_eq]; simp [WasmState.push, h_no_branch]
              have h_1_no_halt : ws_1.halted = false := by
                rw [h_ws_1_eq]; simp [WasmState.push, h_no_halt]
              rw [evalInstrs_cons_default fuel ws_1 (.localGet idxIdx) _
                    h_1_no_branch h_1_no_halt rfl] at hw
              cases h_eval_2 : evalInstr ws_1 (.localGet idxIdx) with
              | none =>
                  rw [h_eval_2] at hw
                  simp at hw
              | some ws_2 =>
                  rw [h_eval_2] at hw
                  simp only at hw
                  obtain ⟨kst_2, h_kst_eval_2, R_2⟩ :=
                    preservation_localGet ws_1 s_1 kst_1 layout R_1 idxIdx h_no_buf_idx_1
                      ws_2 s_2 ops_2
                      h_eval_2 h_head_2
                  have h_2_broke : kst_2.broke = false := by
                    have := evalOps_copy_singleton_preserves_broke h_kst_eval_2
                    rw [this]; rw [h_kst_1_eq]; exact h_kst_no_broke
                  have h_ws_2_shape : ∃ v, ws_2 = ws_1.push v := by
                    cases h_loc : ws_1.getLocal idxIdx with
                    | none =>
                        have h_ev : evalInstr ws_1 (.localGet idxIdx) = none := by
                          show (do let v ← ws_1.getLocal idxIdx; pure (ws_1.push v)) = none
                          rw [h_loc]; rfl
                        rw [h_ev] at h_eval_2; exact (Option.noConfusion h_eval_2)
                    | some v =>
                        refine ⟨v, ?_⟩
                        have h_ev : evalInstr ws_1 (.localGet idxIdx) = some (ws_1.push v) := by
                          show (do let v ← ws_1.getLocal idxIdx; pure (ws_1.push v)) = some (ws_1.push v)
                          rw [h_loc]; rfl
                        rw [h_ev] at h_eval_2
                        exact ((Option.some.injEq _ _).mp h_eval_2).symm
                  obtain ⟨v_2, h_ws_2_eq⟩ := h_ws_2_shape
                  have h_2_no_branch : ws_2.branchTarget = none := by
                    rw [h_ws_2_eq]; simp [WasmState.push, h_1_no_branch]
                  have h_2_no_halt : ws_2.halted = false := by
                    rw [h_ws_2_eq]; simp [WasmState.push, h_1_no_halt]
                  rw [evalInstrs_cons_default fuel ws_2 (.i32Const k) _
                        h_2_no_branch h_2_no_halt rfl] at hw
                  have h_eval_3 : evalInstr ws_2 (.i32Const k) =
                      some (ws_2.push (.wI32 (UInt32.ofNat k.toNat))) := rfl
                  rw [h_eval_3] at hw
                  simp only at hw
                  let ws_3 := ws_2.push (.wI32 (UInt32.ofNat k.toNat))
                  obtain ⟨kst_3, h_kst_eval_3, R_3⟩ :=
                    preservation_i32Const ws_2 s_2 kst_2 layout R_2 k
                      ws_3 s_3 []
                      h_eval_3 h_head_3
                  have h_kst_3_eq : kst_3 = kst_2 := by
                    simp [evalOps] at h_kst_eval_3
                    exact h_kst_eval_3.symm
                  have h_3_no_branch : ws_3.branchTarget = none := by
                    simp [ws_3, WasmState.push, h_2_no_branch]
                  have h_3_no_halt : ws_3.halted = false := by
                    simp [ws_3, WasmState.push, h_2_no_halt]
                  have h_3_broke : kst_3.broke = false := by
                    rw [h_kst_3_eq]; exact h_2_broke
                  rw [evalInstrs_cons_default fuel ws_3 .i32Shl _
                        h_3_no_branch h_3_no_halt rfl] at hw
                  cases h_eval_4 : evalInstr ws_3 .i32Shl with
                  | none =>
                      rw [h_eval_4] at hw
                      simp at hw
                  | some ws_4 =>
                      rw [h_eval_4] at hw
                      simp only at hw
                      have h_shift_eq_spec :
                          ∀ a : UInt32,
                            regLookup kst_3.rf s_1.nextReg = some (Quanta.KOps.Value.vU32 a) →
                            (a <<< (UInt32.ofNat k.toNat)).toNat = a.toNat * (1 <<< k.toNat) := by
                        intro a _; exact h_shift_eq a
                      obtain ⟨kst_4, h_kst_eval_4, R_4⟩ :=
                        preservation_i32Shl_bufferPattern ws_3 s_3 kst_3 layout R_3
                          k s_1.nextReg .u32 (.bufferPtr bSlot :: s.stack)
                          h_s_3_stack h_shift_eq_spec
                          ws_4 s_4 []
                          h_eval_4 h_head_4
                      have h_kst_4_eq : kst_4 = kst_3 := by
                        simp [evalOps] at h_kst_eval_4
                        exact h_kst_eval_4.symm
                      have h_ws_4_shape : ∃ ws_rest a_v b_v,
                          ws_4 = { ws_3 with stack := .wI32 (b_v <<< a_v) :: ws_rest } := by
                        have h_w_eq : evalInstr ws_3 .i32Shl =
                            binI32 (fun a b => a <<< b) ws_3 := rfl
                        rw [h_w_eq] at h_eval_4
                        obtain ⟨a, b, ws_rest, _, h_ws_4_eq⟩ := binI32_some_shape h_eval_4
                        exact ⟨ws_rest, b, a, h_ws_4_eq⟩
                      obtain ⟨ws_rest_4, a_v_4, b_v_4, h_ws_4_eq⟩ := h_ws_4_shape
                      have h_4_no_branch : ws_4.branchTarget = none := by
                        rw [h_ws_4_eq]; simp [h_3_no_branch]
                      have h_4_no_halt : ws_4.halted = false := by
                        rw [h_ws_4_eq]; simp [h_3_no_halt]
                      have h_4_broke : kst_4.broke = false := by
                        rw [h_kst_4_eq]; exact h_3_broke
                      rw [evalInstrs_cons_default fuel ws_4 .i32Add _
                            h_4_no_branch h_4_no_halt rfl] at hw
                      cases h_eval_5 : evalInstr ws_4 .i32Add with
                      | none =>
                          rw [h_eval_5] at hw
                          simp at hw
                      | some ws_5 =>
                          rw [h_eval_5] at hw
                          simp only at hw
                          have h_addr_eq_spec :
                              ∀ a b_ptr : UInt32, ∀ b : UInt32,
                                regLookup kst_4.rf s_1.nextReg = some (Quanta.KOps.Value.vU32 b) →
                                a.toNat = b.toNat * (1 <<< k.toNat) →
                                b_ptr.toNat = layout.startAddr bSlot →
                                (b_ptr + a).toNat = layout.startAddr bSlot + b.toNat * (1 <<< k.toNat) := by
                            intro a b_ptr b _ h_a h_bp
                            exact h_addr_eq a b_ptr b h_a h_bp
                          obtain ⟨kst_5, h_kst_eval_5, R_5⟩ :=
                            preservation_i32Add_bufferPattern_scaledFirst
                              ws_4 s_4 kst_4 layout R_4
                              bSlot s_1.nextReg (1 <<< k.toNat) s.stack
                              h_s_4_stack h_addr_eq_spec
                              ws_5 s_5 []
                              h_eval_5 h_head_5
                          have h_kst_5_eq : kst_5 = kst_4 := by
                            simp [evalOps] at h_kst_eval_5
                            exact h_kst_eval_5.symm
                          have h_ws_5_shape : ∃ ws_rest a_v b_v,
                              ws_5 = { ws_4 with stack := .wI32 (b_v + a_v) :: ws_rest } := by
                            have h_w_eq : evalInstr ws_4 .i32Add =
                                binI32 eval_u32_wrapping_add ws_4 := rfl
                            rw [h_w_eq] at h_eval_5
                            obtain ⟨a, b, ws_rest, _, h_ws_5_eq⟩ := binI32_some_shape h_eval_5
                            exact ⟨ws_rest, b, a, h_ws_5_eq⟩
                          obtain ⟨ws_rest_5, a_v_5, b_v_5, h_ws_5_eq⟩ := h_ws_5_shape
                          have h_5_no_branch : ws_5.branchTarget = none := by
                            rw [h_ws_5_eq]; simp [h_4_no_branch]
                          have h_5_no_halt : ws_5.halted = false := by
                            rw [h_ws_5_eq]; simp [h_4_no_halt]
                          have h_5_broke : kst_5.broke = false := by
                            rw [h_kst_5_eq]; exact h_4_broke
                          rw [evalInstrs_cons_default fuel ws_5 (.i32Load offset align) _
                                h_5_no_branch h_5_no_halt rfl] at hw
                          cases h_eval_6 : evalInstr ws_5 (.i32Load offset align) with
                          | none =>
                              rw [h_eval_6] at hw
                              simp at hw
                          | some ws_6 =>
                              rw [h_eval_6] at hw
                              simp only at hw
                              have h_in_bounds_spec :
                                  ∀ b : UInt32,
                                    regLookup kst_5.rf s_1.nextReg = some (Quanta.KOps.Value.vU32 b) →
                                    b.toNat < layout.length bSlot := by
                                intro b _; exact h_in_bounds b
                              obtain ⟨kst_6, h_kst_eval_6, R_6⟩ :=
                                preservation_i32Load ws_5 s_5 kst_5 layout R_5
                                  bSlot s_1.nextReg s.stack offset align
                                  h_s_5_stack h_offset h_in_bounds_spec
                                  ws_6 s_6 ops_6
                                  h_eval_6 h_head_6
                              have h_lf_6 : loopFreeNoBreak ops_6 = true := rfl
                              have h_6_broke : kst_6.broke = false :=
                                evalOps_loopFreeNoBreak_preserves_broke
                                  h_lf_6 h_5_broke h_kst_eval_6
                              have h_ws_6_shape : ∃ ws_rest n,
                                  ws_6 = { ws_5 with stack := .wI32 n :: ws_rest } := by
                                have h_w_eq : evalInstr ws_5 (.i32Load offset align) =
                                    loadI32 ws_5 offset := rfl
                                rw [h_w_eq] at h_eval_6
                                unfold loadI32 at h_eval_6
                                rcases hws : ws_5.stack with _ | ⟨vaddr, ws_rest⟩
                                · simp [hws, WasmState.pop] at h_eval_6
                                · simp [hws, WasmState.pop] at h_eval_6
                                  cases vaddr with
                                  | wI32 addr_w =>
                                      simp at h_eval_6
                                      rcases hmem : ws_5.mem.load_u32 (addr_w.toNat + offset)
                                        with _ | n
                                      · simp [hmem] at h_eval_6
                                      · simp [hmem, WasmState.push] at h_eval_6
                                        refine ⟨ws_rest, n, ?_⟩
                                        rw [← h_eval_6]
                                  | wI64 _ => simp at h_eval_6
                                  | wF32 _ => simp at h_eval_6
                                  | wF64 _ => simp at h_eval_6
                              obtain ⟨ws_rest_6, n_6, h_ws_6_eq⟩ := h_ws_6_shape
                              have h_6_no_branch : ws_6.branchTarget = none := by
                                rw [h_ws_6_eq]; simp [h_5_no_branch]
                              have h_6_no_halt : ws_6.halted = false := by
                                rw [h_ws_6_eq]; simp [h_5_no_halt]
                              obtain ⟨kst', F_rest, h_eval_rest, R_rest⟩ :=
                                preservation_rest R_6 h_6_no_branch h_6_no_halt h_6_broke
                                  hw h_post
                              -- Compose: total ops contributions are ops_2 ++ ops_6 ++ postOps
                              -- (since steps 1, 3, 4, 5 emit []). h_ops_eq gives the equation.
                              -- evalOps chain:
                              --   evalOps F_rest kst   ops_2     = some kst_2  (via h_kst_eval_2 + kst_1=kst)
                              --   evalOps F_rest kst_2 ops_6     = some kst_6  (via h_kst_eval_6 + kst chain)
                              --   evalOps F_rest kst_6 postOps   = some kst'   (via h_eval_rest)
                              -- Compose via evalOps_append twice.
                              rw [h_kst_1_eq] at h_kst_eval_2
                              rw [h_kst_5_eq, h_kst_4_eq, h_kst_3_eq] at h_kst_eval_6
                              -- Lift h_kst_eval_2 and h_kst_eval_6 to fuel F_rest.
                              -- ops_2 = [.copy ...] and ops_6 = [.load ...] are both
                              -- loopFreeNoBreak, hence fuel-irrelevant per
                              -- evalOps_loopFreeNoBreak_preserves_broke's evalOp arm.
                              -- Use the existing evalOps_loopFree_fuel_irrel helper.
                              have h_lf_2 : loopFree ops_2 = true := by
                                simp [loopFree, loopFreeOp, ops_2]
                              have h_lf_6_shallow : loopFree ops_6 = true :=
                                loopFreeNoBreak_implies_loopFree h_lf_6
                              have h_kst_eval_2_F :
                                  evalOps F_rest kst ops_2 = some kst_2 :=
                                evalOps_loopFree_fuel_irrel h_lf_2 h_kst_eval_2
                              have h_kst_eval_6_F :
                                  evalOps F_rest kst_2 ops_6 = some kst_6 :=
                                evalOps_loopFree_fuel_irrel h_lf_6_shallow h_kst_eval_6
                              have h_eval_a :
                                  evalOps F_rest kst (ops_2 ++ ops_6) = some kst_6 := by
                                rw [evalOps_append h_kst_eval_2_F h_2_broke]
                                exact h_kst_eval_6_F
                              have h_eval_b :
                                  evalOps F_rest kst (ops_2 ++ ops_6 ++ postOps) = some kst' := by
                                rw [evalOps_append h_eval_a h_6_broke]
                                exact h_eval_rest
                              refine ⟨kst', F_rest, ?_, ?_⟩
                              · rw [← h_ops_eq]; exact h_eval_b
                              · rw [← h_s_eq]; exact R_rest

-- ════════════════════════════════════════════════════════════════════
-- L2: full 7-step buffer-access STORE chain
--
-- Symmetric to L1c. Composes the canonical buffer-access store chain:
--   localGet bufSlotIdx :: localGet idxIdx :: i32Const k :: i32Shl ::
--   i32Add :: localGet valIdx :: i32Store offset align :: rest
-- After the 7-step chain the symbolic stack is just `s.stack` (the
-- store pops val + bufferAccess, leaving the original prefix).
-- The emitted IR is
--   [.copy s.nextReg stable_idx,
--    .copy (s.nextReg + 1) stable_val,
--    .store bSlot s.nextReg (s.nextReg + 1) .u32]
--
-- The chain captures the canonical kernel idiom `buffer[i] = v` for
-- u32 buffers (the store writes 4 bytes at `bSlot + i * 4`).
--
-- Step layout matches L1c through step 5; step 6 is another
-- localGet (the value), step 7 is the i32Store. The store's value
-- SymVal is `.reg (s.nextReg + 1) .u32` whose commit returns `(src,
-- s, [])` — no extra ops at step 7 beyond the .store itself.
--
-- User-facing witnesses (the entry-state cleanliness contract):
--   - h_buf / h_loc_buf: buffer-slot binding agrees with layout.
--   - h_no_buf_idx: the idx local is non-buffer at entry state.
--   - h_no_buf_val: the val local is non-buffer at entry state.
--   - h_k_eq_2 / h_shift_eq / h_addr_eq: the arithmetic of `i * 4`.
--   - h_offset = 0.
--   - h_in_bounds: every UInt32 index fits in the layout slot.
--   - h_layout_no_overlap: store target's 4-byte range is disjoint
--     from every other in-bounds (slot', idx') byte range.
--
-- ════════════════════════════════════════════════════════════════════

/-- L2: the full 7-step buffer-access STORE chain (closes L2). -/
theorem preservation_evalInstrs_chain_buffer_store_7step
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (bufSlotIdx : Nat) (bSlot : Nat)
    (h_buf : s.lookupBufferSlot bufSlotIdx = some bSlot)
    (h_loc_buf : ∀ v, ws.locals.get? bufSlotIdx = some v →
      ∃ n : UInt32, v = .wI32 n ∧ n.toNat = layout.startAddr bSlot)
    (idxIdx : Nat) (h_no_buf_idx : s.lookupBufferSlot idxIdx = none)
    (valIdx : Nat) (h_no_buf_val : s.lookupBufferSlot valIdx = none)
    (k : Int) (h_k_eq_2 : k = 2)
    (h_shift_eq : ∀ a : UInt32,
       (a <<< (UInt32.ofNat k.toNat)).toNat = a.toNat * (1 <<< k.toNat))
    (h_addr_eq : ∀ a b_ptr b : UInt32,
       a.toNat = b.toNat * (1 <<< k.toNat) →
       b_ptr.toNat = layout.startAddr bSlot →
       (b_ptr + a).toNat = layout.startAddr bSlot + b.toNat * (1 <<< k.toNat))
    (offset align : Nat) (h_offset : offset = 0)
    (h_in_bounds : ∀ b : UInt32, b.toNat < layout.length bSlot)
    (h_layout_no_overlap : ∀ b : UInt32,
       ∀ slot' idx',
         idx' < layout.length slot' →
         (slot', idx') ≠ (bSlot, b.toNat) →
         layout.startAddr bSlot + b.toNat * 4 + 4 ≤ layout.startAddr slot' + idx' * 4 ∨
         layout.startAddr slot' + idx' * 4 + 4 ≤ layout.startAddr bSlot + b.toNat * 4)
    (rest : List WasmInstr)
    (preservation_rest : ∀ {ws_mid : WasmState} {s_mid : LowerState}
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
        Refines ws'_mid s'_mid kst'_mid layout)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws
            (.localGet bufSlotIdx :: .localGet idxIdx ::
             .i32Const k :: .i32Shl :: .i32Add ::
             .localGet valIdx ::
             .i32Store offset align :: rest) = some ws')
    (hl : lowerInstrs fuel frames s
            (.localGet bufSlotIdx :: .localGet idxIdx ::
             .i32Const k :: .i32Shl :: .i32Add ::
             .localGet valIdx ::
             .i32Store offset align :: rest) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- ── Lowering side: unfold all 7 cons_default steps ─────────────────
  rw [lowerInstrs_cons_default fuel frames s (.localGet bufSlotIdx) _ rfl] at hl
  have h_head_1 : lowerInstr s (.localGet bufSlotIdx)
                    = some (s.pushSym (.bufferPtr bSlot), []) := by
    show (match s.lookupBufferSlot bufSlotIdx with
          | some slot => some (s.pushSym (.bufferPtr slot), [])
          | none =>
              (s.lookupLocal bufSlotIdx).bind (fun stable =>
                let (fresh, s1) := s.alloc
                let s2 := s1.push fresh
                some (s2, [KernelOp.copy fresh stable])))
              = some (s.pushSym (.bufferPtr bSlot), [])
    rw [h_buf]
  rw [h_head_1] at hl
  simp only [Option.bind_eq_bind, Option.some_bind, List.nil_append] at hl
  let s_1 : LowerState := s.pushSym (.bufferPtr bSlot)
  have h_bufs_eq_1 : s_1.bufferSlots = s.bufferSlots := by
    simp [s_1, LowerState.pushSym]
  have h_no_buf_idx_1 : s_1.lookupBufferSlot idxIdx = none := by
    rw [LowerState.lookupBufferSlot_of_bufferSlots_eq idxIdx h_bufs_eq_1]
    exact h_no_buf_idx
  rw [lowerInstrs_cons_default fuel frames s_1 (.localGet idxIdx) _ rfl] at hl
  cases h_stable : s_1.lookupLocal idxIdx with
  | none =>
      simp only [lowerInstr, h_no_buf_idx_1, h_stable, Option.bind_eq_bind,
                 Option.some_bind, Option.none_bind, LowerState.alloc,
                 LowerState.push] at hl
      exact (Option.noConfusion hl)
  | some stable =>
      let s_2 : LowerState :=
        { s_1 with nextReg := s_1.nextReg + 1,
                   stack := SymVal.reg s_1.nextReg .u32 :: s_1.stack }
      let ops_2 : List KernelOp := [.copy s_1.nextReg stable]
      have h_head_2 : lowerInstr s_1 (.localGet idxIdx) = some (s_2, ops_2) := by
        show (match s_1.lookupBufferSlot idxIdx with
              | some slot => some (s_1.pushSym (.bufferPtr slot), [])
              | none => do
                  let stable ← s_1.lookupLocal idxIdx
                  let (fresh, s1) := s_1.alloc
                  let s2 := s1.push fresh
                  pure (s2, [.copy fresh stable])) = some (s_2, ops_2)
        rw [h_no_buf_idx_1, h_stable]
        rfl
      rw [h_head_2] at hl
      simp only [Option.bind_eq_bind, Option.some_bind] at hl
      rw [lowerInstrs_cons_default fuel frames s_2 (.i32Const k) _ rfl] at hl
      have h_head_3 : lowerInstr s_2 (.i32Const k) =
          some (s_2.pushSym (.i32ConstSym k), []) := rfl
      rw [h_head_3] at hl
      simp only [Option.bind_eq_bind, Option.some_bind, List.nil_append] at hl
      let s_3 : LowerState := s_2.pushSym (.i32ConstSym k)
      have h_s_3_stack : s_3.stack = .i32ConstSym k :: .reg s_1.nextReg .u32 ::
                                      .bufferPtr bSlot :: s.stack := by
        show (s_2.pushSym (.i32ConstSym k)).stack = _
        simp [s_2, s_1, LowerState.pushSym]
      rw [lowerInstrs_cons_default fuel frames s_3 .i32Shl _ rfl] at hl
      let s_4 : LowerState :=
        { s_3 with stack := .scaledIdx s_1.nextReg (1 <<< k.toNat) ::
                             .bufferPtr bSlot :: s.stack }
      have h_head_4 : lowerInstr s_3 .i32Shl = some (s_4, []) := by
        show lowerI32Shl s_3 = some (s_4, [])
        unfold lowerI32Shl
        rw [h_s_3_stack]
      rw [h_head_4] at hl
      simp only [Option.bind_eq_bind, Option.some_bind, List.nil_append] at hl
      have h_s_4_stack : s_4.stack = .scaledIdx s_1.nextReg (1 <<< k.toNat) ::
                                       .bufferPtr bSlot :: s.stack := rfl
      rw [lowerInstrs_cons_default fuel frames s_4 .i32Add _ rfl] at hl
      let s_5 : LowerState :=
        { s_4 with stack := .bufferAccess bSlot s_1.nextReg (1 <<< k.toNat) :: s.stack }
      have h_head_5 : lowerInstr s_4 .i32Add = some (s_5, []) := by
        show lowerI32Add s_4 = some (s_5, [])
        unfold lowerI32Add
        rw [h_s_4_stack]
      rw [h_head_5] at hl
      simp only [Option.bind_eq_bind, Option.some_bind, List.nil_append] at hl
      have h_scale_4 : (1 <<< k.toNat) = 4 := by rw [h_k_eq_2]; rfl
      have h_s_5_stack : s_5.stack = .bufferAccess bSlot s_1.nextReg 4 :: s.stack := by
        show (.bufferAccess bSlot s_1.nextReg (1 <<< k.toNat) :: s.stack) = _
        rw [h_scale_4]
      -- Step 6: localGet valIdx. The val local is non-buffer (h_no_buf_val
      -- lifts through steps 1-5 because none of them modify bufferSlots).
      have h_bufs_eq_5 : s_5.bufferSlots = s.bufferSlots := by
        -- s_5 = s_4 with stack', s_4 = s_3 with stack', s_3 = s_2.pushSym,
        -- s_2 = { s_1 with nextReg, stack }, s_1 = s.pushSym. None modify
        -- bufferSlots.
        show s.bufferSlots = s.bufferSlots
        rfl
      have h_no_buf_val_5 : s_5.lookupBufferSlot valIdx = none := by
        rw [LowerState.lookupBufferSlot_of_bufferSlots_eq valIdx h_bufs_eq_5]
        exact h_no_buf_val
      rw [lowerInstrs_cons_default fuel frames s_5 (.localGet valIdx) _ rfl] at hl
      cases h_stable_v : s_5.lookupLocal valIdx with
      | none =>
          simp only [lowerInstr, h_no_buf_val_5, h_stable_v, Option.bind_eq_bind,
                     Option.some_bind, Option.none_bind, LowerState.alloc,
                     LowerState.push] at hl
          exact (Option.noConfusion hl)
      | some stable_v =>
          let s_6 : LowerState :=
            { s_5 with nextReg := s_5.nextReg + 1,
                       stack := SymVal.reg s_5.nextReg .u32 :: s_5.stack }
          let ops_6 : List KernelOp := [.copy s_5.nextReg stable_v]
          have h_head_6 : lowerInstr s_5 (.localGet valIdx) = some (s_6, ops_6) := by
            show (match s_5.lookupBufferSlot valIdx with
                  | some slot => some (s_5.pushSym (.bufferPtr slot), [])
                  | none => do
                      let stable ← s_5.lookupLocal valIdx
                      let (fresh, s1) := s_5.alloc
                      let s2 := s1.push fresh
                      pure (s2, [.copy fresh stable])) = some (s_6, ops_6)
            rw [h_no_buf_val_5, h_stable_v]
            rfl
          rw [h_head_6] at hl
          simp only [Option.bind_eq_bind, Option.some_bind] at hl
          -- s_5.nextReg = s_1.nextReg + 1 (steps 3,4,5 don't alloc).
          have h_s_5_nextReg : s_5.nextReg = s_1.nextReg + 1 := rfl
          -- s_6.stack: top is `.reg s_5.nextReg .u32` = `.reg (s_1.nextReg + 1) .u32`.
          have h_s_6_stack : s_6.stack =
              .reg s_5.nextReg .u32 :: .bufferAccess bSlot s_1.nextReg 4 :: s.stack := by
            show .reg s_5.nextReg .u32 :: s_5.stack = _
            rw [h_s_5_stack]
          -- Step 7: i32Store. lowerI32Store on s_6 reduces via the
          -- buffer-access arm. The val SymVal is .reg s_5.nextReg .u32,
          -- whose commit emits no extra ops.
          rw [lowerInstrs_cons_default fuel frames s_6 (.i32Store offset align) _ rfl] at hl
          let s_7 : LowerState := s_6.popSym.bind (fun p => p.2.popSym.map Prod.snd)
            |>.getD s_6
          -- The "expected" output: s_7 with stack = s.stack, ops_7 = [.store ...].
          -- We can compute s_7 directly from the definition of lowerI32Store +
          -- the .reg commit.
          let s_7_real : LowerState :=
            { s_6 with stack := s.stack }
          let ops_7 : List KernelOp := [.store bSlot s_1.nextReg s_5.nextReg .u32]
          have h_head_7 : lowerInstr s_6 (.i32Store offset align) = some (s_7_real, ops_7) := by
            show lowerI32Store s_6 = some (s_7_real, ops_7)
            unfold lowerI32Store
            simp only [LowerState.popSym, h_s_6_stack, Option.bind_eq_bind,
                       Option.some_bind, LowerState.commit, List.nil_append]
            rfl
          rw [h_head_7] at hl
          simp only [Option.bind_eq_bind, Option.some_bind] at hl
          cases h_post : lowerInstrs fuel frames s_7_real rest with
          | none => simp [h_post] at hl
          | some post_pair =>
              rcases post_pair with ⟨s_post, postOps⟩
              simp [h_post] at hl
              rcases hl with ⟨h_s_eq, h_ops_eq⟩
              -- ── Eval side: unfold all 7 cons_default steps ────────────
              rw [evalInstrs_cons_default fuel ws (.localGet bufSlotIdx) _
                    h_no_branch h_no_halt rfl] at hw
              cases h_eval_1 : evalInstr ws (.localGet bufSlotIdx) with
              | none =>
                  rw [h_eval_1] at hw
                  simp at hw
              | some ws_1 =>
                  rw [h_eval_1] at hw
                  simp only at hw
                  obtain ⟨kst_1, h_kst_eval_1, R_1⟩ :=
                    preservation_localGet_bufferSlot ws s kst layout R bufSlotIdx bSlot
                      h_buf h_loc_buf
                      ws_1 s_1 []
                      h_eval_1 h_head_1
                  have h_kst_1_eq : kst_1 = kst := by
                    simp [evalOps] at h_kst_eval_1
                    exact h_kst_eval_1.symm
                  have h_ws_1_shape : ∃ v, ws_1 = ws.push v := by
                    cases h_loc : ws.getLocal bufSlotIdx with
                    | none =>
                        have h_ev : evalInstr ws (.localGet bufSlotIdx) = none := by
                          show (do let v ← ws.getLocal bufSlotIdx; pure (ws.push v)) = none
                          rw [h_loc]; rfl
                        rw [h_ev] at h_eval_1; exact (Option.noConfusion h_eval_1)
                    | some v =>
                        refine ⟨v, ?_⟩
                        have h_ev : evalInstr ws (.localGet bufSlotIdx) = some (ws.push v) := by
                          show (do let v ← ws.getLocal bufSlotIdx; pure (ws.push v)) = some (ws.push v)
                          rw [h_loc]; rfl
                        rw [h_ev] at h_eval_1
                        exact ((Option.some.injEq _ _).mp h_eval_1).symm
                  obtain ⟨v_1, h_ws_1_eq⟩ := h_ws_1_shape
                  have h_1_no_branch : ws_1.branchTarget = none := by
                    rw [h_ws_1_eq]; simp [WasmState.push, h_no_branch]
                  have h_1_no_halt : ws_1.halted = false := by
                    rw [h_ws_1_eq]; simp [WasmState.push, h_no_halt]
                  rw [evalInstrs_cons_default fuel ws_1 (.localGet idxIdx) _
                        h_1_no_branch h_1_no_halt rfl] at hw
                  cases h_eval_2 : evalInstr ws_1 (.localGet idxIdx) with
                  | none =>
                      rw [h_eval_2] at hw
                      simp at hw
                  | some ws_2 =>
                      rw [h_eval_2] at hw
                      simp only at hw
                      obtain ⟨kst_2, h_kst_eval_2, R_2⟩ :=
                        preservation_localGet ws_1 s_1 kst_1 layout R_1 idxIdx h_no_buf_idx_1
                          ws_2 s_2 ops_2
                          h_eval_2 h_head_2
                      have h_2_broke : kst_2.broke = false := by
                        have := evalOps_copy_singleton_preserves_broke h_kst_eval_2
                        rw [this]; rw [h_kst_1_eq]; exact h_kst_no_broke
                      have h_ws_2_shape : ∃ v, ws_2 = ws_1.push v := by
                        cases h_loc : ws_1.getLocal idxIdx with
                        | none =>
                            have h_ev : evalInstr ws_1 (.localGet idxIdx) = none := by
                              show (do let v ← ws_1.getLocal idxIdx; pure (ws_1.push v)) = none
                              rw [h_loc]; rfl
                            rw [h_ev] at h_eval_2; exact (Option.noConfusion h_eval_2)
                        | some v =>
                            refine ⟨v, ?_⟩
                            have h_ev : evalInstr ws_1 (.localGet idxIdx) = some (ws_1.push v) := by
                              show (do let v ← ws_1.getLocal idxIdx; pure (ws_1.push v)) = some (ws_1.push v)
                              rw [h_loc]; rfl
                            rw [h_ev] at h_eval_2
                            exact ((Option.some.injEq _ _).mp h_eval_2).symm
                      obtain ⟨v_2, h_ws_2_eq⟩ := h_ws_2_shape
                      have h_2_no_branch : ws_2.branchTarget = none := by
                        rw [h_ws_2_eq]; simp [WasmState.push, h_1_no_branch]
                      have h_2_no_halt : ws_2.halted = false := by
                        rw [h_ws_2_eq]; simp [WasmState.push, h_1_no_halt]
                      rw [evalInstrs_cons_default fuel ws_2 (.i32Const k) _
                            h_2_no_branch h_2_no_halt rfl] at hw
                      have h_eval_3 : evalInstr ws_2 (.i32Const k) =
                          some (ws_2.push (.wI32 (UInt32.ofNat k.toNat))) := rfl
                      rw [h_eval_3] at hw
                      simp only at hw
                      let ws_3 := ws_2.push (.wI32 (UInt32.ofNat k.toNat))
                      obtain ⟨kst_3, h_kst_eval_3, R_3⟩ :=
                        preservation_i32Const ws_2 s_2 kst_2 layout R_2 k
                          ws_3 s_3 []
                          h_eval_3 h_head_3
                      have h_kst_3_eq : kst_3 = kst_2 := by
                        simp [evalOps] at h_kst_eval_3
                        exact h_kst_eval_3.symm
                      have h_3_no_branch : ws_3.branchTarget = none := by
                        simp [ws_3, WasmState.push, h_2_no_branch]
                      have h_3_no_halt : ws_3.halted = false := by
                        simp [ws_3, WasmState.push, h_2_no_halt]
                      have h_3_broke : kst_3.broke = false := by
                        rw [h_kst_3_eq]; exact h_2_broke
                      rw [evalInstrs_cons_default fuel ws_3 .i32Shl _
                            h_3_no_branch h_3_no_halt rfl] at hw
                      cases h_eval_4 : evalInstr ws_3 .i32Shl with
                      | none =>
                          rw [h_eval_4] at hw
                          simp at hw
                      | some ws_4 =>
                          rw [h_eval_4] at hw
                          simp only at hw
                          have h_shift_eq_spec :
                              ∀ a : UInt32,
                                regLookup kst_3.rf s_1.nextReg = some (Quanta.KOps.Value.vU32 a) →
                                (a <<< (UInt32.ofNat k.toNat)).toNat = a.toNat * (1 <<< k.toNat) := by
                            intro a _; exact h_shift_eq a
                          obtain ⟨kst_4, h_kst_eval_4, R_4⟩ :=
                            preservation_i32Shl_bufferPattern ws_3 s_3 kst_3 layout R_3
                              k s_1.nextReg .u32 (.bufferPtr bSlot :: s.stack)
                              h_s_3_stack h_shift_eq_spec
                              ws_4 s_4 []
                              h_eval_4 h_head_4
                          have h_kst_4_eq : kst_4 = kst_3 := by
                            simp [evalOps] at h_kst_eval_4
                            exact h_kst_eval_4.symm
                          have h_ws_4_shape : ∃ ws_rest a_v b_v,
                              ws_4 = { ws_3 with stack := .wI32 (b_v <<< a_v) :: ws_rest } := by
                            have h_w_eq : evalInstr ws_3 .i32Shl =
                                binI32 (fun a b => a <<< b) ws_3 := rfl
                            rw [h_w_eq] at h_eval_4
                            obtain ⟨a, b, ws_rest, _, h_ws_4_eq⟩ := binI32_some_shape h_eval_4
                            exact ⟨ws_rest, b, a, h_ws_4_eq⟩
                          obtain ⟨ws_rest_4, a_v_4, b_v_4, h_ws_4_eq⟩ := h_ws_4_shape
                          have h_4_no_branch : ws_4.branchTarget = none := by
                            rw [h_ws_4_eq]; simp [h_3_no_branch]
                          have h_4_no_halt : ws_4.halted = false := by
                            rw [h_ws_4_eq]; simp [h_3_no_halt]
                          have h_4_broke : kst_4.broke = false := by
                            rw [h_kst_4_eq]; exact h_3_broke
                          rw [evalInstrs_cons_default fuel ws_4 .i32Add _
                                h_4_no_branch h_4_no_halt rfl] at hw
                          cases h_eval_5 : evalInstr ws_4 .i32Add with
                          | none =>
                              rw [h_eval_5] at hw
                              simp at hw
                          | some ws_5 =>
                              rw [h_eval_5] at hw
                              simp only at hw
                              have h_addr_eq_spec :
                                  ∀ a b_ptr : UInt32, ∀ b : UInt32,
                                    regLookup kst_4.rf s_1.nextReg = some (Quanta.KOps.Value.vU32 b) →
                                    a.toNat = b.toNat * (1 <<< k.toNat) →
                                    b_ptr.toNat = layout.startAddr bSlot →
                                    (b_ptr + a).toNat = layout.startAddr bSlot + b.toNat * (1 <<< k.toNat) := by
                                intro a b_ptr b _ h_a h_bp
                                exact h_addr_eq a b_ptr b h_a h_bp
                              obtain ⟨kst_5, h_kst_eval_5, R_5⟩ :=
                                preservation_i32Add_bufferPattern_scaledFirst
                                  ws_4 s_4 kst_4 layout R_4
                                  bSlot s_1.nextReg (1 <<< k.toNat) s.stack
                                  h_s_4_stack h_addr_eq_spec
                                  ws_5 s_5 []
                                  h_eval_5 h_head_5
                              have h_kst_5_eq : kst_5 = kst_4 := by
                                simp [evalOps] at h_kst_eval_5
                                exact h_kst_eval_5.symm
                              have h_ws_5_shape : ∃ ws_rest a_v b_v,
                                  ws_5 = { ws_4 with stack := .wI32 (b_v + a_v) :: ws_rest } := by
                                have h_w_eq : evalInstr ws_4 .i32Add =
                                    binI32 eval_u32_wrapping_add ws_4 := rfl
                                rw [h_w_eq] at h_eval_5
                                obtain ⟨a, b, ws_rest, _, h_ws_5_eq⟩ := binI32_some_shape h_eval_5
                                exact ⟨ws_rest, b, a, h_ws_5_eq⟩
                              obtain ⟨ws_rest_5, a_v_5, b_v_5, h_ws_5_eq⟩ := h_ws_5_shape
                              have h_5_no_branch : ws_5.branchTarget = none := by
                                rw [h_ws_5_eq]; simp [h_4_no_branch]
                              have h_5_no_halt : ws_5.halted = false := by
                                rw [h_ws_5_eq]; simp [h_4_no_halt]
                              have h_5_broke : kst_5.broke = false := by
                                rw [h_kst_5_eq]; exact h_4_broke
                              -- Step 6: localGet valIdx.
                              rw [evalInstrs_cons_default fuel ws_5 (.localGet valIdx) _
                                    h_5_no_branch h_5_no_halt rfl] at hw
                              cases h_eval_6 : evalInstr ws_5 (.localGet valIdx) with
                              | none =>
                                  rw [h_eval_6] at hw
                                  simp at hw
                              | some ws_6 =>
                                  rw [h_eval_6] at hw
                                  simp only at hw
                                  obtain ⟨kst_6, h_kst_eval_6, R_6⟩ :=
                                    preservation_localGet ws_5 s_5 kst_5 layout R_5
                                      valIdx h_no_buf_val_5
                                      ws_6 s_6 ops_6
                                      h_eval_6 h_head_6
                                  have h_6_broke : kst_6.broke = false := by
                                    have := evalOps_copy_singleton_preserves_broke h_kst_eval_6
                                    rw [this]; exact h_5_broke
                                  have h_ws_6_shape : ∃ v, ws_6 = ws_5.push v := by
                                    cases h_loc : ws_5.getLocal valIdx with
                                    | none =>
                                        have h_ev : evalInstr ws_5 (.localGet valIdx) = none := by
                                          show (do let v ← ws_5.getLocal valIdx; pure (ws_5.push v)) = none
                                          rw [h_loc]; rfl
                                        rw [h_ev] at h_eval_6; exact (Option.noConfusion h_eval_6)
                                    | some v =>
                                        refine ⟨v, ?_⟩
                                        have h_ev : evalInstr ws_5 (.localGet valIdx) = some (ws_5.push v) := by
                                          show (do let v ← ws_5.getLocal valIdx; pure (ws_5.push v)) = some (ws_5.push v)
                                          rw [h_loc]; rfl
                                        rw [h_ev] at h_eval_6
                                        exact ((Option.some.injEq _ _).mp h_eval_6).symm
                                  obtain ⟨v_6, h_ws_6_eq⟩ := h_ws_6_shape
                                  have h_6_no_branch : ws_6.branchTarget = none := by
                                    rw [h_ws_6_eq]; simp [WasmState.push, h_5_no_branch]
                                  have h_6_no_halt : ws_6.halted = false := by
                                    rw [h_ws_6_eq]; simp [WasmState.push, h_5_no_halt]
                                  -- Step 7: i32Store.
                                  rw [evalInstrs_cons_default fuel ws_6 (.i32Store offset align) _
                                        h_6_no_branch h_6_no_halt rfl] at hw
                                  cases h_eval_7 : evalInstr ws_6 (.i32Store offset align) with
                                  | none =>
                                      rw [h_eval_7] at hw
                                      simp at hw
                                  | some ws_7 =>
                                      rw [h_eval_7] at hw
                                      simp only at hw
                                      have h_in_bounds_spec :
                                          ∀ b : UInt32,
                                            regLookup kst_6.rf s_1.nextReg = some (Quanta.KOps.Value.vU32 b) →
                                            b.toNat < layout.length bSlot := by
                                        intro b _; exact h_in_bounds b
                                      have h_no_ovr_spec :
                                          ∀ b : UInt32,
                                            regLookup kst_6.rf s_1.nextReg = some (Quanta.KOps.Value.vU32 b) →
                                            ∀ slot' idx',
                                              idx' < layout.length slot' →
                                              (slot', idx') ≠ (bSlot, b.toNat) →
                                              layout.startAddr bSlot + b.toNat * 4 + 4 ≤
                                                layout.startAddr slot' + idx' * 4 ∨
                                              layout.startAddr slot' + idx' * 4 + 4 ≤
                                                layout.startAddr bSlot + b.toNat * 4 := by
                                        intro b _ slot' idx' h_lt h_ne
                                        exact h_layout_no_overlap b slot' idx' h_lt h_ne
                                      obtain ⟨kst_7, h_kst_eval_7, R_7⟩ :=
                                        preservation_i32Store ws_6 s_6 kst_6 layout R_6 h_6_broke
                                          (.reg s_5.nextReg .u32) bSlot s_1.nextReg s.stack
                                          offset align
                                          h_s_6_stack h_offset h_in_bounds_spec h_no_ovr_spec
                                          ws_7 s_7_real
                                          (([] : List KernelOp) ++
                                            [.store bSlot s_1.nextReg s_5.nextReg .u32])
                                          h_eval_7 (by
                                            -- Goal: lowerInstr s_6 (.i32Store offset align) =
                                            --   some (s_7_real, [] ++ [.store ...]).
                                            -- Our h_head_7 is the same with ops_7 = [.store ...].
                                            rw [h_head_7]; rfl)
                                      have h_lf_7 : loopFreeNoBreak ops_7 = true := rfl
                                      have h_7_broke : kst_7.broke = false := by
                                        have h_eval_eq : evalOps 0 kst_6 ops_7 = some kst_7 := by
                                          have := h_kst_eval_7
                                          simp at this
                                          exact this
                                        exact evalOps_loopFreeNoBreak_preserves_broke
                                          h_lf_7 h_6_broke h_eval_eq
                                      -- Mid-state ws_7 branch/halt: storeI32 only mutates mem
                                      -- and pops two values.
                                      have h_ws_7_shape : ∃ ws_rest m',
                                          ws_7 = { ws_6 with stack := ws_rest, mem := m' } := by
                                        have h_w_eq : evalInstr ws_6 (.i32Store offset align) =
                                            storeI32 ws_6 offset := rfl
                                        rw [h_w_eq] at h_eval_7
                                        unfold storeI32 at h_eval_7
                                        rcases hws : ws_6.stack with _ | ⟨vval, _ | ⟨vaddr, ws_rest⟩⟩
                                        · simp [hws, WasmState.pop] at h_eval_7
                                        · simp [hws, WasmState.pop] at h_eval_7
                                        · simp [hws, WasmState.pop] at h_eval_7
                                          cases vaddr with
                                          | wI32 addr_w =>
                                              cases vval with
                                              | wI32 v_w =>
                                                  simp at h_eval_7
                                                  rcases hmem :
                                                      ws_6.mem.store_u32 (addr_w.toNat + offset) v_w
                                                      with _ | m'
                                                  · simp [hmem] at h_eval_7
                                                  · simp [hmem] at h_eval_7
                                                    refine ⟨ws_rest, m', ?_⟩
                                                    rw [← h_eval_7]
                                              | wI64 _ => simp at h_eval_7
                                              | wF32 _ => simp at h_eval_7
                                              | wF64 _ => simp at h_eval_7
                                          | wI64 _ => simp at h_eval_7
                                          | wF32 _ => simp at h_eval_7
                                          | wF64 _ => simp at h_eval_7
                                      obtain ⟨ws_rest_7, m'_7, h_ws_7_eq⟩ := h_ws_7_shape
                                      have h_7_no_branch : ws_7.branchTarget = none := by
                                        rw [h_ws_7_eq]; simp [h_6_no_branch]
                                      have h_7_no_halt : ws_7.halted = false := by
                                        rw [h_ws_7_eq]; simp [h_6_no_halt]
                                      obtain ⟨kst', F_rest, h_eval_rest, R_rest⟩ :=
                                        preservation_rest R_7 h_7_no_branch h_7_no_halt h_7_broke
                                          hw h_post
                                      -- Compose: total ops = ops_2 ++ ops_6 ++ ops_7 ++ postOps.
                                      -- Steps 1,3,4,5 emit []. h_ops_eq gives the equation.
                                      rw [h_kst_1_eq] at h_kst_eval_2
                                      rw [h_kst_5_eq, h_kst_4_eq, h_kst_3_eq] at h_kst_eval_6
                                      -- Get evalOps F_rest forms via fuel-irrel.
                                      have h_lf_2 : loopFree ops_2 = true := by
                                        simp [loopFree, loopFreeOp, ops_2]
                                      have h_lf_6 : loopFree ops_6 = true := by
                                        simp [loopFree, loopFreeOp, ops_6]
                                      have h_lf_7_shallow : loopFree ops_7 = true := by
                                        simp [loopFree, loopFreeOp, ops_7]
                                      have h_kst_eval_2_F :
                                          evalOps F_rest kst ops_2 = some kst_2 :=
                                        evalOps_loopFree_fuel_irrel h_lf_2 h_kst_eval_2
                                      have h_kst_eval_6_F :
                                          evalOps F_rest kst_2 ops_6 = some kst_6 :=
                                        evalOps_loopFree_fuel_irrel h_lf_6 h_kst_eval_6
                                      -- h_kst_eval_7 returns evalOps 0; lift to F_rest.
                                      have h_kst_eval_7_0 :
                                          evalOps 0 kst_6 ops_7 = some kst_7 := by
                                        have := h_kst_eval_7
                                        simp at this
                                        exact this
                                      have h_kst_eval_7_F :
                                          evalOps F_rest kst_6 ops_7 = some kst_7 :=
                                        evalOps_loopFree_fuel_irrel h_lf_7_shallow h_kst_eval_7_0
                                      have h_eval_a :
                                          evalOps F_rest kst (ops_2 ++ ops_6) = some kst_6 := by
                                        rw [evalOps_append h_kst_eval_2_F h_2_broke]
                                        exact h_kst_eval_6_F
                                      have h_eval_b :
                                          evalOps F_rest kst (ops_2 ++ ops_6 ++ ops_7) = some kst_7 := by
                                        rw [evalOps_append h_eval_a h_6_broke]
                                        exact h_kst_eval_7_F
                                      have h_eval_c :
                                          evalOps F_rest kst (ops_2 ++ ops_6 ++ ops_7 ++ postOps)
                                              = some kst' := by
                                        rw [evalOps_append h_eval_b h_7_broke]
                                        exact h_eval_rest
                                      refine ⟨kst', F_rest, ?_, ?_⟩
                                      · rw [← h_ops_eq]; exact h_eval_c
                                      · rw [← h_s_eq]; exact R_rest

-- ════════════════════════════════════════════════════════════════════
-- L3.1 foundation — list-level bufferSlots-preservation invariant
--
-- Lifts the per-step `lowerInstr_preserves_bufferSlots` to a
-- list-level statement: every successful `lowerInstrs` over a
-- non-structured-control instruction list preserves
-- `s.bufferSlots`.
--
-- The hypothesis `h_no_struct : ∀ i ∈ instrs, isStructuredLower i =
-- false` rules out `block`/`wloop`/`wif` (whose recursion through
-- inner bodies would need separate bookkeeping). The `br`/`brIf`
-- arms are also excluded — they don't go through the default arm
-- and have their own bufferSlots-preservation but proving it
-- requires unfolding the structured-recursion case-split. For the
-- closedInstr-recognized subset the precondition is automatic.
--
-- This invariant is the architectural prerequisite for state-aware
-- recognizer extensions: any predicate on `LowerState` that only
-- reads `bufferSlots` lifts uniformly across an `lowerInstrs`
-- execution on a closed-shape list. Used by `closedInstrAt` /
-- `closedInstrsAt` in `PreservationInduction`.
-- ════════════════════════════════════════════════════════════════════

theorem lowerInstrs_preserves_bufferSlots_default
    {fuel : Nat} {frames : List FrameKind}
    {s s' : LowerState} {ops : List KernelOp} {instrs : List WasmInstr}
    (h_no_struct : ∀ i ∈ instrs, isStructuredLower i = false)
    (h : lowerInstrs fuel frames s instrs = some (s', ops)) :
    s'.bufferSlots = s.bufferSlots := by
  induction instrs generalizing s s' ops with
  | nil =>
      -- lowerInstrs ... s [] = some (s, [])
      simp [lowerInstrs] at h
      rw [← h.1]
  | cons i rest ih =>
      have h_ns_head : isStructuredLower i = false := h_no_struct i (by simp)
      have h_ns_rest : ∀ j ∈ rest, isStructuredLower j = false := by
        intro j hj; exact h_no_struct j (List.mem_cons_of_mem _ hj)
      rw [lowerInstrs_cons_default fuel frames s i rest h_ns_head] at h
      cases h_head : lowerInstr s i with
      | none => simp [h_head] at h
      | some pair =>
          rcases pair with ⟨s_mid, ops_head⟩
          simp [h_head] at h
          cases h_tail : lowerInstrs fuel frames s_mid rest with
          | none => simp [h_tail] at h
          | some tail_pair =>
              rcases tail_pair with ⟨s_post, ops_tail⟩
              simp [h_tail] at h
              rcases h with ⟨h_s_eq, _⟩
              have h_bufs_head := lowerInstr_preserves_bufferSlots h_head
              have h_bufs_tail := ih h_ns_rest h_tail
              rw [← h_s_eq, h_bufs_tail, h_bufs_head]

-- ════════════════════════════════════════════════════════════════════
-- Per-op loopFreeNoBreak emit lemmas
--
-- One per closed-shape instruction whose `lowerInstr` lands in a
-- non-trivial op list (constants / nop / drop emit []; binops emit
-- opsCommit ++ opsCommit ++ [.binOp]; cmps emit opsCommit ++
-- opsCommit ++ [.cmp, .cast]). Each result is loopFreeNoBreak.
--
-- localSet / localTee already have private helpers above; we re-
-- expose them for the L3.2 skeleton consumer.
-- ════════════════════════════════════════════════════════════════════

/-- Re-exposed: `lowerInstr s (.localSet i) = some (s', ops)` emits a
    loopFreeNoBreak op list. Public wrapper around the private
    helper higher up. -/
theorem lowerInstr_localSet_emits_loopFreeNoBreak_pub
    {s s' : LowerState} {i : Nat} {ops : List KernelOp}
    (h : lowerInstr s (.localSet i) = some (s', ops)) :
    loopFreeNoBreak ops = true :=
  lowerInstr_localSet_emits_loopFreeNoBreak h

/-- Re-exposed `localTee` variant. -/
theorem lowerInstr_localTee_emits_loopFreeNoBreak_pub
    {s s' : LowerState} {i : Nat} {ops : List KernelOp}
    (h : lowerInstr s (.localTee i) = some (s', ops)) :
    loopFreeNoBreak ops = true :=
  lowerInstr_localTee_emits_loopFreeNoBreak h

/-- `lowerI32Bin` emits `opsA ++ opsB ++ [.binOp ...]` where each
    `opsA` / `opsB` comes from `commit` (so loopFreeNoBreak). The
    trailing `.binOp` is loop-free and not break. Whole list is
    loopFreeNoBreak. -/
theorem lowerI32Bin_emits_loopFreeNoBreak
    {s s' : LowerState} {op : Quanta.KOps.BinOp} {ops : List KernelOp}
    (h : lowerI32Bin s op = some (s', ops)) :
    loopFreeNoBreak ops = true := by
  unfold lowerI32Bin at h
  rcases hb : s.popSym with _ | ⟨svb, s1⟩
  · simp [hb] at h
  simp only [hb, Option.bind_eq_bind, Option.some_bind] at h
  rcases ha : s1.popSym with _ | ⟨sva, s2⟩
  · simp [ha] at h
  simp only [ha, Option.some_bind] at h
  rcases hca : s2.commit sva with _ | ⟨ra, s3, opsA⟩
  · simp [hca] at h
  simp only [hca, Option.some_bind] at h
  rcases hcb : s3.commit svb with _ | ⟨rb, s4, opsB⟩
  · simp [hcb] at h
  simp only [hcb, Option.some_bind] at h
  simp [LowerState.alloc, LowerState.push] at h
  rcases h with ⟨_, hops⟩
  rw [← hops]
  have h_lf_a : loopFreeNoBreak opsA = true := commit_emits_loopFreeNoBreak hca
  have h_lf_b : loopFreeNoBreak opsB = true := commit_emits_loopFreeNoBreak hcb
  simp only [loopFreeNoBreak_append, h_lf_a, h_lf_b, Bool.true_and]
  rfl

/-- `lowerI32Cmp` emits `opsA ++ opsB ++ [.cmp, .cast]` — same shape
    as binop but with one more trailing op. -/
theorem lowerI32Cmp_emits_loopFreeNoBreak
    {s s' : LowerState} {op : Quanta.KOps.CmpOp} {ops : List KernelOp}
    (h : lowerI32Cmp s op = some (s', ops)) :
    loopFreeNoBreak ops = true := by
  unfold lowerI32Cmp at h
  rcases hb : s.popSym with _ | ⟨svb, s1⟩
  · simp [hb] at h
  simp only [hb, Option.bind_eq_bind, Option.some_bind] at h
  rcases ha : s1.popSym with _ | ⟨sva, s2⟩
  · simp [ha] at h
  simp only [ha, Option.some_bind] at h
  rcases hca : s2.commit sva with _ | ⟨ra, s3, opsA⟩
  · simp [hca] at h
  simp only [hca, Option.some_bind] at h
  rcases hcb : s3.commit svb with _ | ⟨rb, s4, opsB⟩
  · simp [hcb] at h
  simp only [hcb, Option.some_bind] at h
  simp [LowerState.alloc, LowerState.push] at h
  rcases h with ⟨_, hops⟩
  rw [← hops]
  have h_lf_a : loopFreeNoBreak opsA = true := commit_emits_loopFreeNoBreak hca
  have h_lf_b : loopFreeNoBreak opsB = true := commit_emits_loopFreeNoBreak hcb
  simp only [loopFreeNoBreak_append, h_lf_a, h_lf_b, Bool.true_and]
  rfl

/-- `lowerInstr` on a non-buffer `localGet` emits `[.copy fresh stable]`,
    which is loopFreeNoBreak. The buffer arm emits `[]`. -/
theorem lowerInstr_localGet_emits_loopFreeNoBreak
    {s s' : LowerState} {i : Nat} {ops : List KernelOp}
    (h : lowerInstr s (.localGet i) = some (s', ops)) :
    loopFreeNoBreak ops = true := by
  unfold lowerInstr at h
  cases hb : s.lookupBufferSlot i with
  | some slot =>
      simp [hb, LowerState.pushSym] at h
      rcases h with ⟨_, hops⟩
      rw [hops]; rfl
  | none =>
      simp [hb] at h
      rcases hlk : s.lookupLocal i with _ | stable
      · simp [hlk] at h
      simp [hlk, LowerState.alloc, LowerState.push] at h
      rcases h with ⟨_, hops⟩
      rw [← hops]; rfl

-- ════════════════════════════════════════════════════════════════════
-- L6.3 — brIf preservation theorems
--
-- Three list-level cons theorems covering the four real brIf arms in
-- `lowerInstrs` under the preconditions from `brif_design.md` (L5):
--
-- * Arm 1 — `brIf 0` to enclosing Loop frame.
--   Lowering emits `opsCommit ++ [.cast cond_bool cond .u32 .bool,
--                                  .branch cond_bool [] [.breakOp]]`.
--   Precondition: `rest = []` (canonical loop-iterate pattern).
--   `cond = 0` falls through; `cond ≠ 0` sets branchTarget := some 0,
--   IR-side `.breakOp` sets `broke := true`. Both consumed by the
--   surrounding `.loopOp`.
--
-- * Arm 3 — `brIf depth>0` to outer Loop frame, no Loop above.
--   Lowering emits `opsCommit` only — KOps side relies on the natural
--   loop wrap-around. Precondition: `rest = []`.
--   `cond = 0` falls through; `cond ≠ 0` sets branchTarget := some depth.
--
-- * Arms 2 + 4 — `brIf depth` with Loop above (target may be Loop or
--   non-Loop). Lowering emits `opsCommit ++ [.cast …, .branch cond_bool
--   [.breakOp] []] ++ postOps`. No `rest = []` precondition — `cond = 0`
--   runs `postOps` on both sides; `cond ≠ 0` sets branchTarget on WASM
--   (`rest` short-circuits) and sets `broke := true` on KOps (`postOps`
--   short-circuits via `evalOps`'s cons-broke check).
--
-- All three accept any committable cond SymVal (`.reg r .u32` or
-- `.i32ConstSym n`) — the buffer-pattern SymVals refuse at `commit`,
-- which forces the `lowerInstrs` arm to return `none` and contradicts
-- the `hl` hypothesis. The proofs proceed by unfolding the brIf arm of
-- `lowerInstrs` directly (since `isStructuredLower (.brIf _) = true`
-- forbids the `lowerInstrs_cons_default` route).
-- ════════════════════════════════════════════════════════════════════

/-- Helper: a successful brIf-evalInstr step shape. `evalInstr ws
    (.brIf depth)` pops a `wI32 c` and either sets `branchTarget :=
    some depth` (when `c ≠ 0`) or falls through (when `c = 0`). -/
private theorem evalInstr_brIf_shape
    {ws ws' : WasmState} {depth : Nat}
    (h : evalInstr ws (.brIf depth) = some ws') :
    ∃ (c : UInt32) (rest : List WasmValue),
      ws.stack = .wI32 c :: rest ∧
      ((c = 0 ∧ ws' = { ws with stack := rest }) ∨
       (c ≠ 0 ∧ ws' = { ws with stack := rest, branchTarget := some depth })) := by
  simp only [evalInstr] at h
  rcases hpop : ws.pop with _ | ⟨v, ws1⟩
  · simp [hpop] at h
  simp only [hpop, Option.some_bind] at h
  match v, h with
  | .wI32 c, h =>
    -- ws.pop = some (.wI32 c, ws1) ⇒ ws.stack = .wI32 c :: ws1.stack.
    rw [WasmState.pop] at hpop
    rcases hst : ws.stack with _ | ⟨v0, rest⟩
    · rw [hst] at hpop; simp at hpop
    rw [hst] at hpop
    simp at hpop
    obtain ⟨hv0, hws1⟩ := hpop
    -- After the rcases, `ws.stack` has been folded to `v0 :: rest` in
    -- the goal. Use `hv0 : v0 = .wI32 c` to close the head equality.
    subst hv0
    refine ⟨c, rest, rfl, ?_⟩
    by_cases hc : c = 0
    · left
      refine ⟨hc, ?_⟩
      simp [hc] at h
      rw [← h, ← hws1]
    · right
      refine ⟨hc, ?_⟩
      simp [hc] at h
      rw [← h, ← hws1]

/-- Helper: when `commit` succeeds on a popped cond SymVal, the
    intermediate `Refines` for `(ws.pop'd, s.popSym'd then committed)`
    holds, plus the `wI32 c` value encodes via `.reg cond .u32` in the
    post-commit regfile.

    Bundles: pop the top, run commit, give back the after-state Refines
    so the caller doesn't have to rebuild it from scratch. -/
private theorem brIf_cond_pop_commit_correct
    {ws : WasmState} {s : LowerState} {kst : Quanta.KOps.State}
    {layout : BufferLayout} (R : Refines ws s kst layout)
    {c : UInt32} {rest_w : List WasmValue}
    (h_ws_stack : ws.stack = .wI32 c :: rest_w)
    {sv_cond : SymVal} {s0 : LowerState}
    (h_pop : s.popSym = some (sv_cond, s0))
    {cond : Quanta.KOps.Reg} {s1 : LowerState} {opsCommit : List KernelOp}
    (h_commit : s0.commit sv_cond = some (cond, s1, opsCommit))
    (h_kst_ok : kst.broke = false) :
    ∃ kst1, evalOps 0 kst opsCommit = some kst1 ∧
            kst1.broke = false ∧
            regLookup kst1.rf cond = some (Quanta.KOps.Value.vU32 c) ∧
            -- Refines with the popped suffix as the stack on both sides.
            Refines { ws with stack := rest_w }
                    { s1 with stack := s0.stack } kst1 layout ∧
            s.nextReg ≤ s1.nextReg ∧
            cond < s1.nextReg ∧
            -- Side stipulations: bufferSlots and locals untouched.
            s1.localReg = s.localReg ∧
            s1.localTy = s.localTy ∧
            s1.bufferSlots = s.bufferSlots ∧
            s0.stack = s.stack.tail := by
  -- Unfold popSym: s.stack must be `sv_cond :: lrest` for some lrest.
  -- Pin that shape via case analysis up front.
  have h_pop' : s.popSym = some (sv_cond, s0) := h_pop
  rw [LowerState.popSym] at h_pop'
  rcases hst : s.stack with _ | ⟨svH, lrest⟩
  · rw [hst] at h_pop'; simp at h_pop'
  rw [hst] at h_pop'; simp at h_pop'
  obtain ⟨hsv_eq, h_s0_eq⟩ := h_pop'
  -- svH = sv_cond, but keep them distinct to avoid `subst` rewriting
  -- the rest of the goal away from `sv_cond`.
  have h_svH_eq : svH = sv_cond := hsv_eq
  -- h_s0_eq gives s0 explicitly.
  have h_s0_shape : s0 =
      { nextReg := s.nextReg, stack := lrest, localReg := s.localReg,
        localTy := s.localTy, bufferSlots := s.bufferSlots } := h_s0_eq.symm
  -- Encoding of wI32 c via sv_cond in kst.rf.
  have h_enc_cond_pre : (WasmValue.wI32 c).encodes layout kst.rf sv_cond := by
    have hget : ws.stack.get? 0 = some (.wI32 c) := by
      rw [h_ws_stack]; simp
    obtain ⟨sv0', hsv0_get, henc⟩ := R.stk.right 0 (.wI32 c) hget
    have h_s_stack_head : s.stack.get? 0 = some sv_cond := by
      rw [hst]; simp [h_svH_eq]
    rw [h_s_stack_head] at hsv0_get
    have h_eq : sv_cond = sv0' := (Option.some.injEq _ _).mp hsv0_get
    rw [h_eq]; exact henc
  -- ws_pop is the popped suffix WasmState.
  let ws_pop : WasmState := { ws with stack := rest_w }
  have h_len_pop : rest_w.length = lrest.length := by
    have h_len := R.stk.left
    rw [h_ws_stack, hst] at h_len
    simpa using h_len
  have h_sv_cond_in_stack : sv_cond ∈ s.stack := by
    rw [hst, ← h_svH_eq]; simp
  have h_s0_regs_lt : ∀ r ∈ sv_cond.regs, r < s0.nextReg := by
    intro r hr
    rw [h_s0_shape]
    exact R.fresh.left sv_cond h_sv_cond_in_stack r hr
  have h_lrest_sub_stack : ∀ sv ∈ lrest, sv ∈ s.stack := by
    intro sv hsv
    rw [hst]; exact List.mem_cons_of_mem _ hsv
  have R_pop : Refines ws_pop s0 kst layout := by
    refine ⟨⟨?_, ?_⟩, ?_, ?_, ?_, ?_, R.heapRefines⟩
    · rw [h_s0_shape]; exact h_len_pop
    · intro i v hv
      have h_rest_get : ws.stack.get? (i + 1) = some v := by
        rw [h_ws_stack]; simpa using hv
      obtain ⟨sv_i, hsv_get, henc⟩ := R.stk.right (i + 1) v h_rest_get
      have h_s_get : s.stack.get? (i + 1) = some sv_i := hsv_get
      rw [hst] at h_s_get
      simp at h_s_get
      have h_s0_get : s0.stack.get? i = some sv_i := by
        rw [h_s0_shape]; simpa using h_s_get
      exact ⟨sv_i, h_s0_get, henc⟩
    · rw [h_s0_shape]; exact R.locs
    · refine ⟨?_, ?_⟩
      · intro sv hsv r hr
        rw [h_s0_shape] at hsv ⊢
        simp at hsv
        exact R.fresh.left sv (h_lrest_sub_stack sv hsv) r hr
      · intro ir hir
        rw [h_s0_shape] at hir ⊢
        simp at hir
        exact R.fresh.right ir hir
    · intro ir hir sv hsv
      rw [h_s0_shape] at hir hsv
      simp at hir hsv
      exact R.aliasFree ir hir sv (h_lrest_sub_stack sv hsv)
    · rw [h_s0_shape]; exact R.injLocals
  -- Apply commit_correct.
  obtain ⟨kst1, h_evalCommit, R1, h_enc_cond_post, _h_lookup_below, h_le, h_cond_lt⟩ :=
    commit_correct R_pop h_s0_regs_lt h_commit h_enc_cond_pre
  have h_kst1_ok : kst1.broke = false := by
    rw [commit_preserves_broke h_commit h_evalCommit]; exact h_kst_ok
  have h_lookup : regLookup kst1.rf cond = some (Quanta.KOps.Value.vU32 c) :=
    h_enc_cond_post
  have h_local_eq : s1.localReg = s0.localReg ∧ s1.localTy = s0.localTy :=
    commit_preserves_locals h_commit
  have h_bs_eq : s1.bufferSlots = s0.bufferSlots :=
    commit_preserves_bufferSlots h_commit
  have h_s0_loc_eq : s0.localReg = s.localReg ∧ s0.localTy = s.localTy := by
    rw [h_s0_shape]; exact ⟨rfl, rfl⟩
  have h_s0_bs_eq : s0.bufferSlots = s.bufferSlots := by rw [h_s0_shape]
  have h_s_le_s0 : s.nextReg ≤ s0.nextReg := by rw [h_s0_shape]; exact Nat.le_refl _
  have h_s_le_s1 : s.nextReg ≤ s1.nextReg := Nat.le_trans h_s_le_s0 h_le
  -- After the prior `rcases hst : s.stack`, `s.stack` may be folded to
  -- either form. Discharge against `(svH :: lrest).tail = lrest`.
  have h_s0_stack_eq : s0.stack = (svH :: lrest).tail := by
    rw [h_s0_shape]; simp
  have h_s1_stack : s1.stack = s0.stack :=
    commit_preserves_stack h_commit
  refine ⟨kst1, h_evalCommit, h_kst1_ok, h_lookup, ?_, h_s_le_s1, h_cond_lt,
          ?_, ?_, ?_, h_s0_stack_eq⟩
  · have h_eq : ({ s1 with stack := s0.stack } : LowerState) = s1 := by
      cases s1 with
      | mk nr st lr lt bs =>
        simp at h_s1_stack
        rw [h_s1_stack]
    rw [h_eq]; exact R1
  · rw [h_local_eq.1, h_s0_loc_eq.1]
  · rw [h_local_eq.2, h_s0_loc_eq.2]
  · rw [h_bs_eq, h_s0_bs_eq]

-- ════════════════════════════════════════════════════════════════════
-- Arm 3 — brIf depth>0 to outer Loop, no Loop above, rest = []
--
-- Lowering: `opsCommit ++ postOps` where postOps comes from the
-- recursive `lowerInstrs fuel frames s1 rest`. With `rest = []`,
-- postOps = []; final op list is exactly `opsCommit`.
--
-- WASM semantics: `evalInstr ws (.brIf depth)` pops `wI32 c`. If
-- `c = 0`, fall through (recursive eval on `[]` returns the popped
-- state). If `c ≠ 0`, set `branchTarget := some depth` (recursive
-- eval on `[]` returns the post-pop state with branchTarget set).
--
-- IR side: only the commit ops run (loopFreeNoBreak). The cond
-- register is allocated but never read.
-- ════════════════════════════════════════════════════════════════════

theorem preservation_evalInstrs_cons_brIf_loop_outer_no_inner
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (depth : Nat) (h_depth_pos : depth ≠ 0)
    (h_target : frames.get? depth = some .loopK)
    (h_no_loop_above : hasLoopAbove frames depth = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.brIf depth :: []) = some ws')
    (hl : lowerInstrs fuel frames s (.brIf depth :: []) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- Unfold the brIf arm of lowerInstrs.
  simp only [lowerInstrs] at hl
  rcases hpop : s.popSym with _ | ⟨sv_cond, s0⟩
  · simp [hpop] at hl
  simp only [hpop, Option.bind_eq_bind, Option.some_bind] at hl
  rcases hcommit : s0.commit sv_cond with _ | ⟨cond, s1, opsCommit⟩
  · simp [hcommit] at hl
  simp only [hcommit, Option.some_bind] at hl
  rw [h_target] at hl
  simp only [h_depth_pos, ↓reduceIte, h_no_loop_above, Bool.false_eq_true,
             ↓reduceIte] at hl
  -- After the arm match, `hl` reduces to:
  --   `pure (s1, opsCommit ++ []) = some (s', ops)` because
  --   `lowerInstrs fuel frames s1 []` already collapses inside `simp`.
  simp only [pure, Option.some.injEq, Prod.mk.injEq, List.append_nil] at hl
  obtain ⟨h_s_eq, h_ops_eq⟩ := hl
  -- Eval side: brIf step + recursive eval on [].
  rw [evalInstrs_cons_default fuel ws (.brIf depth) [] h_no_branch h_no_halt
      (by simp [isStructuredEval])] at hw
  cases h_eval_head : evalInstr ws (.brIf depth) with
  | none => rw [h_eval_head] at hw; simp at hw
  | some ws_post =>
    rw [h_eval_head] at hw
    simp only at hw
    -- evalInstrs fuel ws_post [] = some ws_post.
    have h_eval_nil : evalInstrs fuel ws_post [] = some ws_post := by
      simp [evalInstrs]
    rw [h_eval_nil] at hw
    have h_ws'_eq : ws' = ws_post := ((Option.some.injEq _ _).mp hw).symm
    -- Get shape of ws_post from evalInstr_brIf_shape.
    obtain ⟨c, rest_w, h_ws_stack, h_branch⟩ := evalInstr_brIf_shape h_eval_head
    -- Apply commit-correct helper.
    obtain ⟨kst1, h_evalCommit, h_kst1_ok, _h_lookup, R_post, _h_le, _h_cond_lt,
            h_lr_eq, h_lt_eq, h_bs_eq2, _h_s0_stack⟩ :=
      brIf_cond_pop_commit_correct R h_ws_stack hpop hcommit h_kst_no_broke
    -- The final lowered state s' = s1, ops = opsCommit.
    -- Show that R_post (which is for `{ws with stack := rest_w}` and
    -- `{s1 with stack := s0.stack}`) refines ws_post / s1 / kst1.
    refine ⟨kst1, 0, ?_, ?_⟩
    · -- evalOps 0 kst opsCommit = some kst1 via h_evalCommit (already
    --   at fuel 0). ops = opsCommit by h_ops_eq.
      rw [← h_ops_eq]
      exact h_evalCommit
    · -- Refines ws_post s' kst1 layout. R_post has stack := rest_w
      -- on ws-side and stack := s0.stack on lower-side.
      -- s' = s1 by h_s_eq; s1.stack = s0.stack by commit_preserves_stack.
      rw [← h_s_eq, h_ws'_eq]
      have h_s1_stack : s1.stack = s0.stack := commit_preserves_stack hcommit
      -- Both branches of h_branch lead to a Refines for ws_post with
      -- stack = rest_w. The branchTarget differs but Refines ignores it.
      rcases h_branch with ⟨_, h_ws_post_eq⟩ | ⟨_, h_ws_post_eq⟩
      · -- cond = 0: ws_post = { ws with stack := rest_w }.
        rw [h_ws_post_eq]
        have h_s1_eq : ({ s1 with stack := s0.stack } : LowerState) = s1 := by
          cases s1 with
          | mk nr st lr lt bs =>
            simp at h_s1_stack
            rw [h_s1_stack]
        rw [← h_s1_eq]; exact R_post
      · -- cond ≠ 0: ws_post = { ws with stack := rest_w,
        --                       branchTarget := some depth }.
        rw [h_ws_post_eq]
        -- R_post is for { ws with stack := rest_w }. Adding branchTarget
        -- doesn't affect Refines (none of its fields look at branchTarget).
        have h_s1_eq : ({ s1 with stack := s0.stack } : LowerState) = s1 := by
          cases s1 with
          | mk nr st lr lt bs =>
            simp at h_s1_stack
            rw [h_s1_stack]
        rw [← h_s1_eq]
        refine ⟨R_post.stk, R_post.locs, R_post.fresh, R_post.aliasFree,
                R_post.injLocals, R_post.heapRefines⟩

-- ════════════════════════════════════════════════════════════════════
-- Arm 1 — brIf 0 to enclosing Loop, rest = []
--
-- Lowering: `opsCommit ++ [.cast cond_bool cond .u32 .bool,
--                          .branch cond_bool [] [.breakOp]]`.
-- (`postOps = []` because `rest = []`.)
--
-- WASM semantics: brIf 0 pops cond. If cond = 0 falls through; if
-- cond ≠ 0 sets `branchTarget := some 0`. With `rest = []`, the
-- recursive evalInstrs returns the post-pop state immediately.
--
-- IR side: the cast computes vBool (c ≠ 0) into cond_bool. The
-- `.branch` op then picks `[]` (if cond_bool = false) or `[.breakOp]`
-- (if cond_bool = true). The `[.breakOp]` arm sets `broke := true`;
-- both arms terminate.
--
-- The two sides agree:
-- * cond = 0 → both fall through. `broke` stays false (R-compatible).
-- * cond ≠ 0 → WASM sets branchTarget; IR sets `broke`. Neither field
--   is observed by `Refines`, so post-state refines.
-- ════════════════════════════════════════════════════════════════════

theorem preservation_evalInstrs_cons_brIf_loop_self
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (h_target : frames.get? 0 = some .loopK)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.brIf 0 :: []) = some ws')
    (hl : lowerInstrs fuel frames s (.brIf 0 :: []) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- Unfold the brIf arm of lowerInstrs.
  simp only [lowerInstrs] at hl
  rcases hpop : s.popSym with _ | ⟨sv_cond, s0⟩
  · simp [hpop] at hl
  simp only [hpop, Option.bind_eq_bind, Option.some_bind] at hl
  rcases hcommit : s0.commit sv_cond with _ | ⟨cond, s1, opsCommit⟩
  · simp [hcommit] at hl
  simp only [hcommit, Option.some_bind] at hl
  rw [h_target] at hl
  -- depth = 0 arm taken (the `if depth = 0 then …` branch).
  simp only [↓reduceIte] at hl
  -- Now `hl` references s1.alloc (cond_bool, s_cast) and the inner
  -- lowerInstrs on []. Reduce explicitly.
  let cond_bool : Quanta.KOps.Reg := s1.nextReg
  let s_cast : LowerState := { s1 with nextReg := s1.nextReg + 1 }
  -- After simp [LowerState.alloc] the inner pair becomes
  -- `(cond_bool, s_cast)` definitionally; `lowerInstrs fuel frames
  -- s_cast []` reduces to `some (s_cast, [])`.
  simp only [LowerState.alloc, pure, Option.some_bind, Option.bind_eq_bind,
             Option.some.injEq, Prod.mk.injEq, List.append_nil] at hl
  -- `hl` is now `(s_cast, opsCommit ++ [.cast cond_bool cond .u32 .bool,
  --                                       .branch cond_bool [] [.breakOp]])
  --             = (s', ops)`.
  -- Unfold the lowerInstrs nil call left over from the bind.
  rcases hl with ⟨h_s_eq, h_ops_eq⟩
  -- Eval side: brIf step + recursive eval on [].
  rw [evalInstrs_cons_default fuel ws (.brIf 0) [] h_no_branch h_no_halt
      (by simp [isStructuredEval])] at hw
  cases h_eval_head : evalInstr ws (.brIf 0) with
  | none => rw [h_eval_head] at hw; simp at hw
  | some ws_post =>
    rw [h_eval_head] at hw
    simp only at hw
    have h_eval_nil : evalInstrs fuel ws_post [] = some ws_post := by
      simp [evalInstrs]
    rw [h_eval_nil] at hw
    have h_ws'_eq : ws' = ws_post := ((Option.some.injEq _ _).mp hw).symm
    -- Shape of ws_post.
    obtain ⟨c, rest_w, h_ws_stack, h_branch⟩ := evalInstr_brIf_shape h_eval_head
    -- Apply commit-correct helper.
    obtain ⟨kst1, h_evalCommit, h_kst1_ok, h_lookup, R_post, _h_le, h_cond_lt,
            h_lr_eq, h_lt_eq, h_bs_eq2, _h_s0_stack⟩ :=
      brIf_cond_pop_commit_correct R h_ws_stack hpop hcommit h_kst_no_broke
    -- The cast op runs: writes cond_bool := vBool (c ≠ 0). The branch
    -- then reads cond_bool. We need to evaluate the full IR.
    -- Stack property: `cond_bool = s1.nextReg`. After cast, the regfile
    -- adds a vBool at cond_bool. Then .branch reads vBool and picks the
    -- appropriate arm.
    have h_s1_stack : s1.stack = s0.stack := commit_preserves_stack hcommit
    -- Build kst2: regfile after the cast. Use `!decide (c = 0)` (the
    -- normalized form `simp` produces from `decide (c ≠ 0)`).
    let kst2 : Quanta.KOps.State :=
      { kst1 with rf := Quanta.KOps.regWrite kst1.rf cond_bool
                          (Quanta.KOps.vBool (!decide (c = 0))) }
    -- evalOp on the cast: regLookup cond = vU32 c, evalCast .bool
    -- vU32 c = some (vBool (c ≠ 0)), write into cond_bool.
    have h_evalCast : Quanta.KOps.evalOp 0 kst1
        (KernelOp.cast cond_bool cond Quanta.KOps.Scalar.u32 Quanta.KOps.Scalar.bool)
        = some kst2 := by
      simp [Quanta.KOps.evalOp, h_lookup, Quanta.KOps.evalCast]
      rfl
    have h_kst2_broke : kst2.broke = false := by
      show kst1.broke = false
      exact h_kst1_ok
    -- The Refines bundle needs to lift past the cast's regWrite at
    -- s1.nextReg (= cond_bool). Build a helper that does this lift
    -- once over any (ws_post-shape with stack rest_w, s_cast, kst_post)
    -- where kst_post.rf agrees with kst2.rf on all old registers and
    -- kst_post.heap = kst1.heap.
    have h_s1_stack : s1.stack = s0.stack := commit_preserves_stack hcommit
    have h_s1_eq_struct :
        ({ s1 with stack := s0.stack } : LowerState) = s1 := by
      cases s1 with
      | mk nr st lr lt bs =>
        simp at h_s1_stack
        rw [h_s1_stack]
    -- The s1-shape Refines lifts to s_cast (nextReg bumped by 1).
    have R_post' : Refines { ws with stack := rest_w } s1 kst1 layout := by
      rw [← h_s1_eq_struct]; exact R_post
    -- Lift across cast's fresh write at cond_bool = s1.nextReg.
    have R_cast : Refines { ws with stack := rest_w } s_cast kst2 layout := by
      refine ⟨?_, ?_, ?_, ?_, R_post'.injLocals, R_post'.heapRefines⟩
      · -- StackRefines.
        refine ⟨?_, ?_⟩
        · show rest_w.length = s_cast.stack.length
          show rest_w.length = s1.stack.length
          exact R_post'.stk.left
        · intro i v hv
          have hv_pop : ({ ws with stack := rest_w } : WasmState).stack.get? i
                          = some v := hv
          obtain ⟨svi, hsv_get, henc⟩ := R_post'.stk.right i v hv_pop
          have hsv_in : svi ∈ s1.stack := List.mem_of_get? hsv_get
          refine ⟨svi, hsv_get, ?_⟩
          apply WasmValue.encodes_preserved_of_fresh _ henc
          intro r hr
          exact R_post'.fresh.left svi hsv_in r hr
      · -- LocalsRefines.
        intro i r hfind v hv
        have henc := R_post'.locs i r hfind v hv
        have hr_lt : r < s1.nextReg := by
          have hpair : (i, r) ∈ s1.localReg := List.mem_of_find?_eq_some hfind
          exact R_post'.fresh.right (i, r) hpair
        apply WasmValue.encodes_preserved_of_fresh _ henc
        intro r' hr'
        simp [SymVal.regs] at hr'
        subst hr'; exact hr_lt
      · -- Fresh.
        refine ⟨?_, ?_⟩
        · intro sv hsv r hr
          show r < s1.nextReg + 1
          exact Nat.lt_succ_of_lt (R_post'.fresh.left sv hsv r hr)
        · intro ir hir
          show ir.snd < s1.nextReg + 1
          exact Nat.lt_succ_of_lt (R_post'.fresh.right ir hir)
      · -- AliasFree.
        intro ir hir sv hsv
        exact R_post'.aliasFree ir hir sv hsv
    -- After the .branch, broke may flip to true. Build the
    -- broke-augmented refines: lift R_cast to (kst_brk).
    --
    -- Mapping per arm 1 (br_if 0 to enclosing Loop):
    -- * WASM cond=0 → branchTarget stays `none`; ws_post = pop only.
    --   KOps `.branch cond_bool=false` picks elseOps `[.breakOp]` →
    --   broke := true (loop exits).
    -- * WASM cond≠0 → branchTarget := some 0; ws_post sets it.
    --   KOps `.branch cond_bool=true` picks thenOps `[]` → no broke
    --   (loop continues).
    let kst_brk : Quanta.KOps.State := { kst2 with broke := true }
    have R_brk : Refines { ws with stack := rest_w }
                          s_cast kst_brk layout := by
      refine ⟨R_cast.stk, R_cast.locs, R_cast.fresh, R_cast.aliasFree,
              R_cast.injLocals, R_cast.heapRefines⟩
    -- Now provide kst' via case-split on c = 0.
    by_cases hc : c = 0
    · -- cond = 0: branch picks elseOps [.breakOp] → kst_brk;
      --            ws_post has stack := rest_w, no branchTarget.
      rcases h_branch with ⟨_, h_ws_post_eq⟩ | ⟨hc_ne, _⟩
      · refine ⟨kst_brk, 0, ?_, ?_⟩
        · -- evalOps 0 kst (opsCommit ++ [cast, branch ...]) = some kst_brk.
          rw [← h_ops_eq]
          rw [evalOps_append h_evalCommit h_kst1_ok]
          show Quanta.KOps.evalOps 0 kst1
            [KernelOp.cast cond_bool cond Quanta.KOps.Scalar.u32
              Quanta.KOps.Scalar.bool,
              KernelOp.branch cond_bool [] [KernelOp.breakOp]] = some kst_brk
          rw [Quanta.KOps.evalOps.eq_def]
          simp only []
          rw [h_evalCast]
          simp [h_kst2_broke]
          rw [Quanta.KOps.evalOps.eq_def]
          simp only []
          have h_branch_eval :
              Quanta.KOps.evalOp 0 kst2
                (KernelOp.branch cond_bool [] [KernelOp.breakOp]) = some kst_brk := by
            -- branch reads cond_bool = vBool (!decide (c = 0)) = vBool false
            -- (since hc : c = 0). Picks elseOps = [.breakOp]. evalOps on
            -- [.breakOp] writes broke := true and returns.
            show Quanta.KOps.evalOp 0 kst2
              (KernelOp.branch cond_bool [] [KernelOp.breakOp]) = _
            simp [Quanta.KOps.evalOp, kst2,
                  regLookup_regWrite_self, hc, kst_brk,
                  Quanta.KOps.evalOps]
            rfl
          rw [h_branch_eval]
          rfl
        · -- Refines.
          rw [← h_s_eq, h_ws'_eq, h_ws_post_eq]
          exact R_brk
      · exact absurd hc hc_ne
    · -- cond ≠ 0: branch picks thenOps [] → kst2;
      --            ws_post sets branchTarget := some 0.
      rcases h_branch with ⟨hc_eq, _⟩ | ⟨_, h_ws_post_eq⟩
      · exact absurd hc_eq hc
      · refine ⟨kst2, 0, ?_, ?_⟩
        · rw [← h_ops_eq]
          rw [evalOps_append h_evalCommit h_kst1_ok]
          show Quanta.KOps.evalOps 0 kst1
            [KernelOp.cast cond_bool cond Quanta.KOps.Scalar.u32
              Quanta.KOps.Scalar.bool,
              KernelOp.branch cond_bool [] [KernelOp.breakOp]] = some kst2
          rw [Quanta.KOps.evalOps.eq_def]
          simp only []
          rw [h_evalCast]
          simp [h_kst2_broke]
          rw [Quanta.KOps.evalOps.eq_def]
          simp only []
          have h_branch_eval :
              Quanta.KOps.evalOp 0 kst2
                (KernelOp.branch cond_bool [] [KernelOp.breakOp])
                = some kst2 := by
            -- branch reads cond_bool = vBool true (since hc : c ≠ 0).
            -- Picks thenOps = []. evalOps on [] returns kst2.
            show Quanta.KOps.evalOp 0 kst2
              (KernelOp.branch cond_bool [] [KernelOp.breakOp]) = _
            simp [Quanta.KOps.evalOp, kst2,
                  regLookup_regWrite_self, hc,
                  Quanta.KOps.evalOps]
            rfl
          rw [h_branch_eval]
          simp [h_kst2_broke, Quanta.KOps.evalOps]
        · rw [← h_s_eq, h_ws'_eq, h_ws_post_eq]
          -- Refines ignores branchTarget; R_cast suffices.
          refine ⟨R_cast.stk, R_cast.locs, R_cast.fresh, R_cast.aliasFree,
                  R_cast.injLocals, R_cast.heapRefines⟩

-- ════════════════════════════════════════════════════════════════════
-- Arms 2 + 4 — brIf with Loop above (any target kind), rest = []
--
-- Lowering with `rest = []`: `opsCommit ++ [.cast cond_bool cond .u32
--                                            .bool, .branch cond_bool
--                                            [.breakOp] []]`.
-- (`postOps = []` because the lowering's recursion on `[]` returns
-- `(s_cast, [])`.)
--
-- WASM semantics:
-- * cond = 0 → fall through; recursive eval on `[]` returns post-pop.
-- * cond ≠ 0 → branchTarget := some depth; recursive eval on `[]`
--              returns post-pop with branchTarget set.
--
-- IR side:
-- * cond_bool = false → branch picks elseOps `[]` → kst_cast unchanged.
-- * cond_bool = true  → branch picks thenOps `[.breakOp]` → broke := true.
--
-- Note: a more general theorem (no `rest = []` precondition) requires
-- bookkeeping for the lowering's recursion through `rest` after a WASM-
-- side short-circuit; the canonical rustc-emitted pattern places brIf
-- at the end of the loop body anyway, so this restriction matches
-- practice. (L6 design — see `brif_design.md` §2.)
-- ════════════════════════════════════════════════════════════════════

theorem preservation_evalInstrs_cons_brIf_loop_break_inner
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (depth : Nat)
    (kind : FrameKind)
    (h_depth_pos_or_nonloop :
      (depth ≠ 0 ∧ kind = .loopK) ∨ kind ≠ .loopK)
    (h_target : frames.get? depth = some kind)
    (h_loop_above : hasLoopAbove frames depth = true)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.brIf depth :: []) = some ws')
    (hl : lowerInstrs fuel frames s (.brIf depth :: []) = some (s', ops)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- Unfold the brIf arm of lowerInstrs.
  simp only [lowerInstrs] at hl
  rcases hpop : s.popSym with _ | ⟨sv_cond, s0⟩
  · simp [hpop] at hl
  simp only [hpop, Option.bind_eq_bind, Option.some_bind] at hl
  rcases hcommit : s0.commit sv_cond with _ | ⟨cond, s1, opsCommit⟩
  · simp [hcommit] at hl
  simp only [hcommit, Option.some_bind] at hl
  rw [h_target] at hl
  -- The arm match reduces according to `kind`. Both `loopK with depth ≠ 0`
  -- and `block/wif` arms produce the same emission shape. With `rest = []`,
  -- the recursive `lowerInstrs fuel frames _ []` collapses to `(_ , [])`,
  -- and the postOps suffix is empty.
  have hl_reduced :
      ({ s1 with nextReg := s1.nextReg + 1 } = s') ∧
      (opsCommit ++
        [KernelOp.cast s1.nextReg cond Quanta.KOps.Scalar.u32
          Quanta.KOps.Scalar.bool,
         KernelOp.branch s1.nextReg [KernelOp.breakOp] []] = ops) := by
    rcases h_depth_pos_or_nonloop with ⟨h_dp, h_kind_eq⟩ | h_kind_ne
    · subst h_kind_eq
      simp only [h_dp, ↓reduceIte, h_loop_above, lowerInstrs, pure,
                 Option.bind_eq_bind, Option.some_bind, Option.some.injEq,
                 Prod.mk.injEq, List.append_nil, LowerState.alloc] at hl
      exact hl
    · cases kind with
      | loopK => exact (h_kind_ne rfl).elim
      | block =>
        simp only [h_loop_above, ↓reduceIte, lowerInstrs, pure,
                   Option.bind_eq_bind, Option.some_bind, Option.some.injEq,
                   Prod.mk.injEq, List.append_nil, LowerState.alloc] at hl
        exact hl
      | wif =>
        simp only [h_loop_above, ↓reduceIte, lowerInstrs, pure,
                   Option.bind_eq_bind, Option.some_bind, Option.some.injEq,
                   Prod.mk.injEq, List.append_nil, LowerState.alloc] at hl
        exact hl
  obtain ⟨h_s_eq, h_ops_eq⟩ := hl_reduced
  -- s_cast (shorthand) for the post-cast-alloc state.
  let cond_bool : Quanta.KOps.Reg := s1.nextReg
  let s_cast : LowerState := { s1 with nextReg := s1.nextReg + 1 }
  -- Eval side: brIf + recursive eval on [].
  rw [evalInstrs_cons_default fuel ws (.brIf depth) [] h_no_branch h_no_halt
      (by simp [isStructuredEval])] at hw
  cases h_eval_head : evalInstr ws (.brIf depth) with
  | none => rw [h_eval_head] at hw; simp at hw
  | some ws_mid =>
    rw [h_eval_head] at hw
    simp only at hw
    -- Get shape of ws_mid.
    obtain ⟨c, rest_w, h_ws_stack, h_branch⟩ := evalInstr_brIf_shape h_eval_head
    -- Apply commit-correct helper.
    obtain ⟨kst1, h_evalCommit, h_kst1_ok, h_lookup, R_post, _h_le, _h_cond_lt,
            _h_lr_eq, _h_lt_eq, _h_bs_eq2, _h_s0_stack⟩ :=
      brIf_cond_pop_commit_correct R h_ws_stack hpop hcommit h_kst_no_broke
    have h_s1_stack : s1.stack = s0.stack := commit_preserves_stack hcommit
    have h_s1_eq_struct :
        ({ s1 with stack := s0.stack } : LowerState) = s1 := by
      cases s1 with
      | mk nr st lr lt bs =>
        simp at h_s1_stack
        rw [h_s1_stack]
    have R_post' : Refines { ws with stack := rest_w } s1 kst1 layout := by
      rw [← h_s1_eq_struct]; exact R_post
    -- Build kst2 (post-cast).
    let kst2 : Quanta.KOps.State :=
      { kst1 with rf := Quanta.KOps.regWrite kst1.rf cond_bool
                          (Quanta.KOps.vBool (!decide (c = 0))) }
    have h_evalCast : Quanta.KOps.evalOp 0 kst1
        (KernelOp.cast cond_bool cond Quanta.KOps.Scalar.u32 Quanta.KOps.Scalar.bool)
        = some kst2 := by
      simp [Quanta.KOps.evalOp, h_lookup, Quanta.KOps.evalCast]
      rfl
    have h_kst2_broke : kst2.broke = false := by
      show kst1.broke = false
      exact h_kst1_ok
    -- Build R_cast.
    have R_cast : Refines { ws with stack := rest_w } s_cast kst2 layout := by
      refine ⟨?_, ?_, ?_, ?_, R_post'.injLocals, R_post'.heapRefines⟩
      · refine ⟨R_post'.stk.left, ?_⟩
        intro i v hv
        have hv_pop : ({ ws with stack := rest_w } : WasmState).stack.get? i
                        = some v := hv
        obtain ⟨svi, hsv_get, henc⟩ := R_post'.stk.right i v hv_pop
        have hsv_in : svi ∈ s1.stack := List.mem_of_get? hsv_get
        refine ⟨svi, hsv_get, ?_⟩
        apply WasmValue.encodes_preserved_of_fresh _ henc
        intro r hr
        exact R_post'.fresh.left svi hsv_in r hr
      · intro i r hfind v hv
        have henc := R_post'.locs i r hfind v hv
        have hr_lt : r < s1.nextReg := by
          have hpair : (i, r) ∈ s1.localReg := List.mem_of_find?_eq_some hfind
          exact R_post'.fresh.right (i, r) hpair
        apply WasmValue.encodes_preserved_of_fresh _ henc
        intro r' hr'
        simp [SymVal.regs] at hr'
        subst hr'; exact hr_lt
      · refine ⟨?_, ?_⟩
        · intro sv hsv r hr
          show r < s1.nextReg + 1
          exact Nat.lt_succ_of_lt (R_post'.fresh.left sv hsv r hr)
        · intro ir hir
          show ir.snd < s1.nextReg + 1
          exact Nat.lt_succ_of_lt (R_post'.fresh.right ir hir)
      · intro ir hir sv hsv
        exact R_post'.aliasFree ir hir sv hsv
    -- Eval-side: rest = [] means recursive eval returns ws_mid immediately.
    have h_eval_nil : evalInstrs fuel ws_mid [] = some ws_mid := by
      simp [evalInstrs]
    rw [h_eval_nil] at hw
    have h_ws'_eq : ws' = ws_mid := ((Option.some.injEq _ _).mp hw).symm
    -- The branch's elseOps for arm 2/4 is []; thenOps is [.breakOp].
    let kst_brk : Quanta.KOps.State := { kst2 with broke := true }
    have R_brk : Refines { ws with stack := rest_w } s_cast kst_brk layout := by
      refine ⟨R_cast.stk, R_cast.locs, R_cast.fresh, R_cast.aliasFree,
              R_cast.injLocals, R_cast.heapRefines⟩
    -- Two cases on c = 0.
    by_cases hc : c = 0
    · -- cond = 0: WASM falls through (ws_mid stack popped, no branchTarget).
      --           IR: branch picks elseOps = [] → kst2 (no broke).
      rcases h_branch with ⟨_, h_ws_mid_eq⟩ | ⟨hc_ne, _⟩
      · refine ⟨kst2, 0, ?_, ?_⟩
        · rw [← h_ops_eq]
          rw [evalOps_append h_evalCommit h_kst1_ok]
          show Quanta.KOps.evalOps 0 kst1
            [KernelOp.cast cond_bool cond Quanta.KOps.Scalar.u32
              Quanta.KOps.Scalar.bool,
              KernelOp.branch cond_bool [KernelOp.breakOp] []] = some kst2
          rw [Quanta.KOps.evalOps.eq_def]
          simp only []
          rw [h_evalCast]
          simp [h_kst2_broke]
          rw [Quanta.KOps.evalOps.eq_def]
          simp only []
          have h_branch_eval :
              Quanta.KOps.evalOp 0 kst2
                (KernelOp.branch cond_bool [KernelOp.breakOp] [])
                = some kst2 := by
            show Quanta.KOps.evalOp 0 kst2
              (KernelOp.branch cond_bool [KernelOp.breakOp] []) = _
            simp [Quanta.KOps.evalOp, kst2,
                  regLookup_regWrite_self, hc,
                  Quanta.KOps.evalOps]
            rfl
          rw [h_branch_eval]
          simp [h_kst2_broke, Quanta.KOps.evalOps]
        · rw [← h_s_eq, h_ws'_eq, h_ws_mid_eq]
          exact R_cast
      · exact absurd hc hc_ne
    · -- cond ≠ 0: WASM sets branchTarget := some depth.
      --           IR: branch picks thenOps = [.breakOp] → kst_brk.
      rcases h_branch with ⟨hc_eq, _⟩ | ⟨_, h_ws_mid_eq⟩
      · exact absurd hc_eq hc
      · refine ⟨kst_brk, 0, ?_, ?_⟩
        · rw [← h_ops_eq]
          rw [evalOps_append h_evalCommit h_kst1_ok]
          show Quanta.KOps.evalOps 0 kst1
            [KernelOp.cast cond_bool cond Quanta.KOps.Scalar.u32
              Quanta.KOps.Scalar.bool,
              KernelOp.branch cond_bool [KernelOp.breakOp] []] = some kst_brk
          rw [Quanta.KOps.evalOps.eq_def]
          simp only []
          rw [h_evalCast]
          simp [h_kst2_broke]
          rw [Quanta.KOps.evalOps.eq_def]
          simp only []
          have h_branch_eval :
              Quanta.KOps.evalOp 0 kst2
                (KernelOp.branch cond_bool [KernelOp.breakOp] []) = some kst_brk := by
            show Quanta.KOps.evalOp 0 kst2
              (KernelOp.branch cond_bool [KernelOp.breakOp] []) = _
            simp [Quanta.KOps.evalOp, kst2,
                  regLookup_regWrite_self, hc, kst_brk,
                  Quanta.KOps.evalOps]
            rfl
          rw [h_branch_eval]
          rfl
        · rw [← h_s_eq, h_ws'_eq, h_ws_mid_eq]
          refine ⟨R_brk.stk, R_brk.locs, R_brk.fresh, R_brk.aliasFree,
                  R_brk.injLocals, R_brk.heapRefines⟩

-- ════════════════════════════════════════════════════════════════════
-- L6.4 — wreturn preservation
--
-- `wreturn` halts the surrounding function. Lowering emits no IR
-- (`lowerInstr s .wreturn = some (s, [])`); the WASM-side `evalInstr`
-- sets `halted := true`, and the surrounding `evalInstrs` short-
-- circuits on `s.halted` before touching `rest`.
--
-- Structurally simpler than brIf: no cond pop, no commit/cast/branch.
-- The IR side runs nothing; the KOps state is untouched. `Refines`
-- lifts because none of its fields look at `halted`.
--
-- Same shape as `preservation_br_loop_zero` (the depth=0 br arm)
-- with `halted` instead of `branchTarget` as the propagation flag.
-- ════════════════════════════════════════════════════════════════════

/-- `evalInstrs` returns the state untouched on a halted pre-state.
    Symmetric to `evalInstrs_branchTarget_some`. Used by the wreturn
    preservation theorem to discharge the recursive `evalInstrs` call
    on `rest` after `wreturn` sets `halted`. -/
theorem evalInstrs_halted_true
    (fuel : Nat) (ws : WasmState) (instrs : List WasmInstr)
    (h : ws.halted = true) :
    evalInstrs fuel ws instrs = some ws := by
  cases instrs with
  | nil => simp [evalInstrs]
  | cons i rest =>
    unfold evalInstrs
    simp [h]

/-- `wreturn :: []` preservation. Lowering emits no IR; eval-side
    sets `halted := true` and the recursive eval on `[]` returns the
    post-halt state. `Refines` carries over from the input by reflexivity
    on every field (none of them inspects `halted`).

    The `rest = []` precondition matches the brIf arms' choice (L6.3):
    a more general statement would need to handle the lowering's
    recursion through `rest` after a WASM-side short-circuit, and the
    existing `preservation_rest` IH-on-rest infrastructure requires
    `halted = false` on the mid-state. The canonical rustc-emitted
    `return` pattern places the instruction at the end of a body
    anyway, so this restriction matches production. -/
theorem preservation_evalInstrs_cons_wreturn
    (fuel : Nat) (frames : List FrameKind)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (ws' : WasmState) (s' : LowerState) (ops : List KernelOp)
    (hw : evalInstrs fuel ws (.wreturn :: []) = some ws')
    (hl : lowerInstrs fuel frames s (.wreturn :: []) = some (s', ops)) :
    ∃ kst', evalOps 0 kst ops = some kst' ∧ Refines ws' s' kst' layout := by
  -- Lowering side: wreturn goes through the default arm (not structured).
  -- `lowerInstr s .wreturn = some (s, [])`, then the recursive
  -- `lowerInstrs fuel frames s []` returns `(s, [])`. Net: `(s, [])`.
  rw [lowerInstrs_cons_default fuel frames s .wreturn []
      (by simp [isStructuredLower])] at hl
  simp only [lowerInstr, Option.bind_eq_bind, Option.some_bind,
             List.nil_append, lowerInstrs, pure, Option.some.injEq,
             Prod.mk.injEq] at hl
  obtain ⟨h_s_eq, h_ops_eq⟩ := hl
  -- Eval side: brIf-style — `evalInstrs_cons_default` reduces to the
  -- recursive call on `[]`, but the head state has halted := true so
  -- `evalInstrs fuel ws_post []` returns ws_post.
  let ws_post : WasmState := { ws with halted := true }
  have h_post_halted : ws_post.halted = true := rfl
  have h_evalInstr : evalInstr ws .wreturn = some ws_post := rfl
  rw [evalInstrs_cons_default fuel ws .wreturn [] h_no_branch h_no_halt
      (by simp [isStructuredEval])] at hw
  rw [h_evalInstr] at hw
  simp only at hw
  rw [evalInstrs_halted_true fuel ws_post [] h_post_halted] at hw
  have hws'_eq : ws' = ws_post := ((Option.some.injEq _ _).mp hw).symm
  -- Final assembly: kst' = kst (lowering emits no IR), Refines lifts.
  refine ⟨kst, ?_, ?_⟩
  · rw [← h_ops_eq]; simp [evalOps]
  · rw [← h_s_eq, hws'_eq]
    -- Refines { ws with halted := true } s kst layout — none of the
    -- Refines fields look at halted.
    refine ⟨R.stk, R.locs, R.fresh, R.aliasFree, R.injLocals, R.heapRefines⟩

end Quanta.Wasm
