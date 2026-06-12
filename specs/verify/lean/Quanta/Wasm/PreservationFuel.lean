/-
# Fuel-irrelevance for loop-free op lists

The 28 per-instruction preservation theorems in `Quanta.Wasm.Preservation`
all conclude with `evalOps 0 kst ops = some kst'`. The `0` works because
the ops they emit are drawn from a strict subset of `KernelOp` that does
not include `.branch` or `.loopOp` — fuel is captured but never consumed.

The list-level slice (`PreservationList`) needs to splice these per-op
results into a list-level walk that may itself recurse with non-zero
fuel (because the structured-control arms emit `.branch` / `.loopOp`
and consume fuel). The bridge is the lemma here: for ops whose top-level
constructors avoid `.branch` and `.loopOp`, evaluation is fuel-irrelevant.

`loopFreeOp` is **shallow** — it inspects only the top-level KernelOp
constructor. That is sufficient because per-op-emitted ops never wrap
sub-op lists at all (only `.branch` and `.loopOp` carry sub-lists), so a
top-level rejection of those two constructors transitively forbids
nested loops/branches.
-/

import Quanta.Wasm.Preservation

namespace Quanta.Wasm

open Quanta.KOps (KernelOp State evalOp evalOps)

/-- True for every `KernelOp` constructor except `.branch` and `.loopOp`.
    Shallow check — ops with sub-op payloads (only `.branch`/`.loopOp`)
    are rejected at the top level, so the predicate transitively forbids
    nested fuel-consuming structure. -/
def loopFreeOp : KernelOp → Bool
  | .branch _ _ _ => false
  | .loopOp _ => false
  | _ => true

/-- Evaluation of a single non-control-flow op does not depend on fuel.
    Each non-`.branch` / non-`.loopOp` arm of `evalOp` ignores its `fuel`
    argument (it is captured by the closure but never read on the RHS),
    so evaluating with any two fuel values gives the same result. -/
theorem evalOp_loopFreeOp_fuel_eq
    (fuel fuel' : Nat) (s : State) (op : KernelOp)
    (h : loopFreeOp op = true) :
    evalOp fuel s op = evalOp fuel' s op := by
  cases op
  case branch _ _ _ => simp [loopFreeOp] at h
  case loopOp _     => simp [loopFreeOp] at h
  all_goals (simp only [Quanta.KOps.evalOp.eq_def])

/-- A list of `KernelOp`s is loop-free iff every element is. -/
def loopFree (ops : List KernelOp) : Bool :=
  ops.all loopFreeOp

@[simp] theorem loopFree_nil : loopFree [] = true := rfl

@[simp] theorem loopFree_cons (op : KernelOp) (rest : List KernelOp) :
    loopFree (op :: rest) = (loopFreeOp op && loopFree rest) := by
  simp [loopFree, List.all]

@[simp] theorem loopFree_append (xs ys : List KernelOp) :
    loopFree (xs ++ ys) = (loopFree xs && loopFree ys) := by
  induction xs with
  | nil => simp
  | cons x rest ih =>
    simp [loopFree_cons, ih, Bool.and_assoc]

/-- Fuel-irrelevance for `evalOps` on loop-free op lists.

    By induction on the list. Empty case is trivial. For `op :: rest`,
    `loopFree` flattens via `loopFree_cons` to `loopFreeOp op = true ∧
    loopFree rest = true`. The head step uses
    `evalOp_loopFreeOp_fuel_eq` to swap `fuel` for `fuel'` on the head
    op, the IH handles the tail, and the `broke`-short-circuit lifts
    through both. -/
theorem evalOps_loopFree_fuel_eq
    (fuel fuel' : Nat) (s : State) (ops : List KernelOp)
    (h : loopFree ops = true) :
    evalOps fuel s ops = evalOps fuel' s ops := by
  induction ops generalizing s with
  | nil => rw [Quanta.KOps.evalOps.eq_def, Quanta.KOps.evalOps.eq_def]
  | cons op rest ih =>
    rw [loopFree_cons] at h
    have h_op : loopFreeOp op = true := (Bool.and_eq_true _ _).mp h |>.left
    have h_rest : loopFree rest = true := (Bool.and_eq_true _ _).mp h |>.right
    -- Unfold both `evalOps` heads via `eq_def`.
    rw [Quanta.KOps.evalOps.eq_def, Quanta.KOps.evalOps.eq_def]
    -- Reduce the cons-match.
    simp only []
    -- Bridge the head call's fuel: evalOp on a loop-free op is fuel-eq.
    rw [evalOp_loopFreeOp_fuel_eq fuel fuel' s op h_op]
    -- Both sides share `evalOp fuel' s op` as their head bind. Case-
    -- split on its result. `cases ... : ...` substitutes into the
    -- goal, then we reduce the `Option.bind` head.
    cases h_eo : evalOp fuel' s op with
    | none => rfl
    | some s_mid =>
      by_cases hbr : s_mid.broke = true
      · simp [hbr]
      · have hbr' : s_mid.broke = false := by
          cases hb : s_mid.broke
          · rfl
          · exact (hbr hb).elim
        simp [hbr']
        exact ih s_mid h_rest

/-- Implication form of `evalOps_loopFree_fuel_eq`: from a successful
    eval at any fuel, conclude success at any other fuel. -/
theorem evalOps_loopFree_fuel_irrel
    {fuel fuel' : Nat} {s : State} {ops : List KernelOp} {s' : State}
    (h_lf : loopFree ops = true)
    (h : evalOps fuel s ops = some s') :
    evalOps fuel' s ops = some s' := by
  rw [evalOps_loopFree_fuel_eq fuel' fuel s ops h_lf]
  exact h

/-- Append/composition lemma tailored to the slice 5c list-level
    cons-case: the per-op theorems produce `evalOps 0 kst ops_head =
    some kst_mid` (fuel = 0); the IH on the tail typically lives at the
    surrounding fuel `F`. This helper bridges the fuel gap (via
    `evalOps_loopFree_fuel_irrel`) and then applies the
    already-existing `evalOps_append` from `Quanta.Wasm.Preservation`. -/
theorem evalOps_append_loopFree_head
    {F : Nat} {kst kst_mid kst' : State}
    {ops_head ops_rest : List KernelOp}
    (h_lf : loopFree ops_head = true)
    (h_head : evalOps 0 kst ops_head = some kst_mid)
    (h_no_broke : kst_mid.broke = false)
    (h_rest : evalOps F kst_mid ops_rest = some kst') :
    evalOps F kst (ops_head ++ ops_rest) = some kst' := by
  have h_head_F : evalOps F kst ops_head = some kst_mid :=
    evalOps_loopFree_fuel_irrel h_lf h_head
  rw [evalOps_append h_head_F h_no_broke]
  exact h_rest

-- ════════════════════════════════════════════════════════════════════
-- Deep loop-free predicate
--
-- The shallow `loopFreeOp` rejects `.branch` outright. That works for
-- the 28 per-op preservation theorems (whose lowered ops never include
-- `.branch`), but it is too restrictive for the structured-control
-- preservation arms — `brIf 0` lowers to `opsCommit ++ [.branch cond
-- [] [.breakOp]] ++ postOps`, where the `.branch` carries loop-free
-- sub-payloads (the `.breakOp` arm is itself loop-free).
--
-- `loopFreeOpDeep` recurses into `.branch` payloads but still rejects
-- `.loopOp` outright. Since `.loopOp` is the only constructor whose
-- evaluation actually consumes fuel (the `opLoop` iteration counter),
-- this is the maximal fuel-irrelevance class without invoking full
-- mutual fuel monotonicity.
-- ════════════════════════════════════════════════════════════════════

mutual
/-- Deep loop-free: a `KernelOp` is deep loop-free iff it is not
    `.loopOp` and (if it is `.branch`) both sub-payloads are deep
    loop-free. All non-control-flow constructors are trivially deep
    loop-free. -/
def loopFreeOpDeep : KernelOp → Bool
  | .loopOp _ => false
  | .branch _ thenOps elseOps => loopFreeDeep thenOps && loopFreeDeep elseOps
  | _ => true

/-- A list is deep loop-free iff every element is. -/
def loopFreeDeep : List KernelOp → Bool
  | [] => true
  | op :: rest => loopFreeOpDeep op && loopFreeDeep rest
end

@[simp] theorem loopFreeDeep_nil : loopFreeDeep [] = true := rfl

@[simp] theorem loopFreeDeep_cons (op : KernelOp) (rest : List KernelOp) :
    loopFreeDeep (op :: rest) = (loopFreeOpDeep op && loopFreeDeep rest) := rfl

@[simp] theorem loopFreeDeep_append (xs ys : List KernelOp) :
    loopFreeDeep (xs ++ ys) = (loopFreeDeep xs && loopFreeDeep ys) := by
  induction xs with
  | nil => simp
  | cons x rest ih =>
    simp [loopFreeDeep_cons, ih, Bool.and_assoc]

/-- Shallow loop-free implies deep: shallow rejects `.branch`/`.loopOp`
    outright, so `loopFreeOpDeep` evaluates trivially for every
    surviving constructor. -/
theorem loopFreeOp_implies_deep (op : KernelOp) (h : loopFreeOp op = true) :
    loopFreeOpDeep op = true := by
  cases op
  case branch _ _ _ => simp [loopFreeOp] at h
  case loopOp _     => simp [loopFreeOp] at h
  all_goals rfl

theorem loopFree_implies_deep (ops : List KernelOp) (h : loopFree ops = true) :
    loopFreeDeep ops = true := by
  induction ops with
  | nil => rfl
  | cons op rest ih =>
    rw [loopFree_cons] at h
    have h_op : loopFreeOp op = true := (Bool.and_eq_true _ _).mp h |>.left
    have h_rest : loopFree rest = true := (Bool.and_eq_true _ _).mp h |>.right
    rw [loopFreeDeep_cons]
    exact (Bool.and_eq_true _ _).mpr
      ⟨loopFreeOp_implies_deep op h_op, ih h_rest⟩

-- Fuel-irrelevance for deep loop-free op lists, mutually with the
-- single-op variant. The proof mirrors the shallow case but routes
-- through `.branch` sub-payloads recursively. `.loopOp` is still
-- forbidden, so no `opLoop` iteration counter to wrangle.
mutual
theorem evalOp_loopFreeOpDeep_fuel_eq
    (fuel fuel' : Nat) (s : State) (op : KernelOp)
    (h : loopFreeOpDeep op = true) :
    evalOp fuel s op = evalOp fuel' s op := by
  cases op
  case loopOp _ => simp [loopFreeOpDeep] at h
  case branch cond thenOps elseOps =>
    simp only [loopFreeOpDeep] at h
    have h_then : loopFreeDeep thenOps = true := (Bool.and_eq_true _ _).mp h |>.left
    have h_else : loopFreeDeep elseOps = true := (Bool.and_eq_true _ _).mp h |>.right
    rw [Quanta.KOps.evalOp.eq_def, Quanta.KOps.evalOp.eq_def]
    -- After unfolding, both sides bind on `regLookup s.rf cond` and
    -- match on the resulting Value. The `none`-lookup and non-bool
    -- arms close by `rfl`; the `vBool` arms need the mutual IH.
    cases hr : Quanta.KOps.regLookup s.rf cond with
    | none => simp [hr]
    | some v =>
      cases v with
      | vBool b =>
        cases b
        · simp [hr, evalOps_loopFreeDeep_fuel_eq fuel fuel' s elseOps h_else]
        · simp [hr, evalOps_loopFreeDeep_fuel_eq fuel fuel' s thenOps h_then]
      | _ => simp [hr]
  all_goals (rw [Quanta.KOps.evalOp.eq_def, Quanta.KOps.evalOp.eq_def])
termination_by sizeOf op

theorem evalOps_loopFreeDeep_fuel_eq
    (fuel fuel' : Nat) (s : State) (ops : List KernelOp)
    (h : loopFreeDeep ops = true) :
    evalOps fuel s ops = evalOps fuel' s ops := by
  cases ops with
  | nil => rw [Quanta.KOps.evalOps.eq_def, Quanta.KOps.evalOps.eq_def]
  | cons op rest =>
    rw [loopFreeDeep_cons] at h
    have h_op : loopFreeOpDeep op = true := (Bool.and_eq_true _ _).mp h |>.left
    have h_rest : loopFreeDeep rest = true := (Bool.and_eq_true _ _).mp h |>.right
    rw [Quanta.KOps.evalOps.eq_def, Quanta.KOps.evalOps.eq_def]
    simp only []
    rw [evalOp_loopFreeOpDeep_fuel_eq fuel fuel' s op h_op]
    cases h_eo : evalOp fuel' s op with
    | none => rfl
    | some s_mid =>
      by_cases hbr : s_mid.broke = true
      · simp [hbr]
      · have hbr' : s_mid.broke = false := by
          cases hb : s_mid.broke
          · rfl
          · exact (hbr hb).elim
        simp [hbr']
        exact evalOps_loopFreeDeep_fuel_eq fuel fuel' s_mid rest h_rest
termination_by sizeOf ops
end

/-- Implication form of `evalOps_loopFreeDeep_fuel_eq`. -/
theorem evalOps_loopFreeDeep_fuel_irrel
    {fuel fuel' : Nat} {s : State} {ops : List KernelOp} {s' : State}
    (h_lf : loopFreeDeep ops = true)
    (h : evalOps fuel s ops = some s') :
    evalOps fuel' s ops = some s' := by
  rw [evalOps_loopFreeDeep_fuel_eq fuel' fuel s ops h_lf]
  exact h

/-- Deep variant of `evalOps_append_loopFree_head`: bridges the fuel
    gap on the head op-list (which may contain `.branch` with
    loop-free sub-payloads) and chains via `evalOps_append`. -/
theorem evalOps_append_loopFreeDeep_head
    {F : Nat} {kst kst_mid kst' : State}
    {ops_head ops_rest : List KernelOp}
    (h_lf : loopFreeDeep ops_head = true)
    (h_head : evalOps 0 kst ops_head = some kst_mid)
    (h_no_broke : kst_mid.broke = false)
    (h_rest : evalOps F kst_mid ops_rest = some kst') :
    evalOps F kst (ops_head ++ ops_rest) = some kst' := by
  have h_head_F : evalOps F kst ops_head = some kst_mid :=
    evalOps_loopFreeDeep_fuel_irrel h_lf h_head
  rw [evalOps_append h_head_F h_no_broke]
  exact h_rest

-- ════════════════════════════════════════════════════════════════════
-- Cons-default unfold lemmas
--
-- `lowerInstrs` (5 structured arms: block / wloop / wif / br / brIf)
-- and `evalInstrs` (3 structured arms: block / wloop / wif — `br`/`brIf`
-- go through `evalInstr` and the surrounding `branchTarget` short-
-- circuit) both fall through a default arm for non-structured
-- instructions. The `isStructuredLower` and `isStructuredEval` Bool
-- predicates carve out the structured constructors; the lemmas below
-- expose the default-arm shape so the cons-composer can rewrite the
-- list-level call into a per-instruction call + recursion on the rest.
-- ════════════════════════════════════════════════════════════════════

/-- `WasmInstr` arms that take a structured arm in `lowerInstrs`. -/
def isStructuredLower : WasmInstr → Bool
  | .block _ | .wloop _ | .wif _ | .br _ | .brIf _ => true
  | .wreturn => true  -- own arm since the 2026-06-12 production
                      -- re-sync (Break / refusal by frame context)
  | _ => false

/-- `WasmInstr` arms that take a structured arm in `evalInstrs`. Note
    that `br` / `brIf` are NOT here — they go through `evalInstr` which
    sets `branchTarget`, and the surrounding `evalInstrs` short-
    circuits via the `branchTarget.isSome` check. -/
def isStructuredEval : WasmInstr → Bool
  | .block _ | .wloop _ | .wif _ => true
  | _ => false

/-- `lowerInstrs` on a non-structured head delegates to `lowerInstr`
    and recurses on the rest. -/
theorem lowerInstrs_cons_default
    (fuel : Nat) (frames : List FrameKind) (s : LowerState)
    (i : WasmInstr) (rest : List WasmInstr)
    (h_ns : isStructuredLower i = false) :
    lowerInstrs fuel frames s (i :: rest) =
      (do
        let (s1, ops1) ← lowerInstr s i
        let (s2, ops2) ← lowerInstrs fuel frames s1 rest
        pure (s2, ops1 ++ ops2)) := by
  cases i
  all_goals try simp [isStructuredLower] at h_ns
  all_goals (rw [lowerInstrs.eq_def])

/-- `evalInstrs` on a non-structured head with a clean pre-state
    (no halt, no pending branch) delegates to `evalInstr` and recurses
    on the rest. -/
theorem evalInstrs_cons_default
    (fuel : Nat) (ws : WasmState) (i : WasmInstr) (rest : List WasmInstr)
    (h_no_branch : ws.branchTarget = none) (h_no_halt : ws.halted = false)
    (h_ns : isStructuredEval i = false) :
    evalInstrs fuel ws (i :: rest) =
      (match evalInstr ws i with
        | none => none
        | some ws' => evalInstrs fuel ws' rest) := by
  cases i
  all_goals try simp [isStructuredEval] at h_ns
  all_goals
    (rw [evalInstrs.eq_def]
     have h_cond : (ws.halted || ws.branchTarget.isSome) = false := by
       rw [h_no_halt, h_no_branch]; rfl
     simp only [h_cond, Bool.false_eq_true, ↓reduceIte])
  all_goals rfl

-- ════════════════════════════════════════════════════════════════════
-- Cons-composer: bundle head + tail Refines existentials
--
-- The list-level preservation theorems for non-control-flow cons-cases
-- have the shape:
--   * lowering returns `ops_head ++ ops_rest` where `ops_rest` is a
--     recursive `lowerInstrs` call on the tail;
--   * the head's per-op preservation (or a structured-control proof)
--     yields a partial `evalOps 0 kst ops_head = some kst_mid` plus a
--     `Refines ws_mid s_mid kst_mid layout` for the post-head state;
--   * the IH on the tail, given that mid-state, yields the final
--     `∃ kst', evalOps F kst_mid ops_rest = some kst' ∧ Refines ws' s' kst' layout`.
--
-- The bridge is: chain via `evalOps_append_loopFreeDeep_head` (deep
-- variant covers `.branch` heads such as those `brIf` emits), then
-- repackage the existential.
-- ════════════════════════════════════════════════════════════════════

/-- Compose head + tail Refines existentials into a bundled `evalOps F`
    on `ops_head ++ ops_rest` and a final `Refines`. Uses the deep
    loop-free fuel bridge so callers may include `.branch` heads with
    loop-free sub-payloads (the exact shape `brIf` lowers to). -/
theorem preservation_evalInstrs_cons_compose
    {F : Nat} {kst kst_mid : Quanta.KOps.State}
    {ops_head ops_rest : List KernelOp}
    {ws' : WasmState} {s' : LowerState}
    {layout : BufferLayout}
    (h_lf : loopFreeDeep ops_head = true)
    (h_head : evalOps 0 kst ops_head = some kst_mid)
    (h_no_broke : kst_mid.broke = false)
    (h_rest : ∃ kst', evalOps F kst_mid ops_rest = some kst'
                       ∧ Refines ws' s' kst' layout) :
    ∃ kst', evalOps F kst (ops_head ++ ops_rest) = some kst'
              ∧ Refines ws' s' kst' layout := by
  obtain ⟨kst', h_eval', R'⟩ := h_rest
  refine ⟨kst', ?_, R'⟩
  exact evalOps_append_loopFreeDeep_head h_lf h_head h_no_broke h_eval'

/-- Shallow variant of `preservation_evalInstrs_cons_compose` — for
    cons-cases whose head ops are entirely non-control-flow (no
    `.branch` constructor). Avoids needing to chase `loopFreeDeep`
    structure on the head. -/
theorem preservation_evalInstrs_cons_compose_shallow
    {F : Nat} {kst kst_mid : Quanta.KOps.State}
    {ops_head ops_rest : List KernelOp}
    {ws' : WasmState} {s' : LowerState}
    {layout : BufferLayout}
    (h_lf : loopFree ops_head = true)
    (h_head : evalOps 0 kst ops_head = some kst_mid)
    (h_no_broke : kst_mid.broke = false)
    (h_rest : ∃ kst', evalOps F kst_mid ops_rest = some kst'
                       ∧ Refines ws' s' kst' layout) :
    ∃ kst', evalOps F kst (ops_head ++ ops_rest) = some kst'
              ∧ Refines ws' s' kst' layout :=
  preservation_evalInstrs_cons_compose
    (loopFree_implies_deep _ h_lf) h_head h_no_broke h_rest

-- ════════════════════════════════════════════════════════════════════
-- Single-op broke preservation for `.copy`
--
-- The cons-localGet preservation chain needs `kst_mid.broke = false`
-- after running the `[.copy fresh stable]` op-list. `.copy` only
-- updates the regfile (`{ s with rf := regWrite s.rf dst v }`), so
-- the broke flag passes through untouched. The lemma below extracts
-- that fact from a successful `evalOps` run.
-- ════════════════════════════════════════════════════════════════════

/-- `evalOp .copy` preserves the broke flag. -/
theorem evalOp_copy_preserves_broke
    {fuel : Nat} {s s' : State} {dst src : Quanta.KOps.Reg}
    (h : evalOp fuel s (.copy dst src) = some s') :
    s'.broke = s.broke := by
  rw [Quanta.KOps.evalOp.eq_def] at h
  cases h_v : Quanta.KOps.regLookup s.rf src with
  | none => simp [h_v] at h
  | some v =>
      simp [h_v] at h
      rw [← h]

/-- `evalOps fuel s [.copy dst src] = some s'` implies `s'.broke = s.broke`.
    Single-op specialization sufficient for the `localGet` cons preservation
    chain (which produces `[.copy fresh stable]` as its head ops). -/
theorem evalOps_copy_singleton_preserves_broke
    {fuel : Nat} {s s' : State} {dst src : Quanta.KOps.Reg}
    (h : evalOps fuel s [.copy dst src] = some s') :
    s'.broke = s.broke := by
  -- Unfold one cons step.
  rw [Quanta.KOps.evalOps.eq_def] at h
  -- Case-split on evalOp result.
  cases h_eo : evalOp fuel s (.copy dst src) with
  | none => simp [h_eo] at h
  | some s_mid =>
    have h_mid_broke : s_mid.broke = s.broke :=
      evalOp_copy_preserves_broke h_eo
    simp [h_eo] at h
    by_cases hbr : s_mid.broke = true
    · simp [hbr] at h
      rw [← h]
      exact h_mid_broke
    · have hbr' : s_mid.broke = false := Bool.eq_false_iff.mpr hbr
      simp [hbr'] at h
      -- h : evalOps fuel s_mid [] = some s'
      rw [Quanta.KOps.evalOps.eq_def] at h
      simp at h
      rw [← h]
      exact h_mid_broke

-- ════════════════════════════════════════════════════════════════════
-- Generic broke preservation for non-`.breakOp` shallow loop-free ops
--
-- Generalizes `evalOp_copy_preserves_broke` to every `KernelOp`
-- constructor that is `loopFreeOp` (excludes `.branch` / `.loopOp`)
-- AND not `.breakOp`. Inspecting the `evalOp` definition: every
-- surviving constructor either updates `rf` (most arithmetic, dispatch
-- IDs), updates `heap` (`.store`), or returns `s` unchanged
-- (`.barrier`). None of them touches `broke`, so any successful
-- evaluation has `s'.broke = s.broke`.
--
-- The list-level companion `evalOps_loopFree_no_break_preserves_broke`
-- chains this through the cons short-circuit on `broke`, which is
-- vacuous when every op preserves `broke = false`. Used by every
-- non-control-flow cons preservation case (i32Add, localSet, drop, …)
-- to discharge the `kst_mid.broke = false` precondition of the
-- cons-composer.
-- ════════════════════════════════════════════════════════════════════

/-- Single-op broke preservation. -/
theorem evalOp_loopFree_no_break_preserves_broke
    {fuel : Nat} {s s' : State} {op : KernelOp}
    (h_lf : loopFreeOp op = true)
    (h_no_break : op ≠ .breakOp)
    (h : evalOp fuel s op = some s') :
    s'.broke = s.broke := by
  cases op with
  | branch _ _ _ => simp [loopFreeOp] at h_lf
  | loopOp _     => simp [loopFreeOp] at h_lf
  | breakOp      => exact (h_no_break rfl).elim
  | const _ _ =>
      rw [Quanta.KOps.evalOp.eq_def] at h
      simp at h; rw [← h]
  | binOp dst a b op _ty =>
      cases ha : Quanta.KOps.regLookup s.rf a with
      | none => simp [Quanta.KOps.evalOp, ha] at h
      | some va =>
        cases hb : Quanta.KOps.regLookup s.rf b with
        | none => simp [Quanta.KOps.evalOp, ha, hb] at h
        | some vb =>
          cases he : Quanta.KOps.evalBinOp op va vb with
          | none => simp [Quanta.KOps.evalOp, ha, hb, he] at h
          | some v =>
            simp [Quanta.KOps.evalOp, ha, hb, he] at h
            rw [← h]
  | unaryOp dst a op _ty =>
      cases ha : Quanta.KOps.regLookup s.rf a with
      | none => simp [Quanta.KOps.evalOp, ha] at h
      | some va =>
        cases he : Quanta.KOps.evalUnaryOp op va with
        | none => simp [Quanta.KOps.evalOp, ha, he] at h
        | some v =>
          simp [Quanta.KOps.evalOp, ha, he] at h
          rw [← h]
  | cmp dst a b op _ty =>
      cases ha : Quanta.KOps.regLookup s.rf a with
      | none => simp [Quanta.KOps.evalOp, ha] at h
      | some va =>
        cases hb : Quanta.KOps.regLookup s.rf b with
        | none => simp [Quanta.KOps.evalOp, ha, hb] at h
        | some vb =>
          cases he : Quanta.KOps.evalCmpOp op va vb with
          | none => simp [Quanta.KOps.evalOp, ha, hb, he] at h
          | some v =>
            simp [Quanta.KOps.evalOp, ha, hb, he] at h
            rw [← h]
  | cast dst src _fromTy to =>
      cases ha : Quanta.KOps.regLookup s.rf src with
      | none => simp [Quanta.KOps.evalOp, ha] at h
      | some va =>
        cases he : Quanta.KOps.evalCast va to with
        | none => simp [Quanta.KOps.evalOp, ha, he] at h
        | some v =>
          simp [Quanta.KOps.evalOp, ha, he] at h
          rw [← h]
  | bitcast dst src _fromTy to =>
      cases ha : Quanta.KOps.regLookup s.rf src with
      | none => simp [Quanta.KOps.evalOp, ha] at h
      | some va =>
        cases he : Quanta.KOps.evalBitcast va to with
        | none => simp [Quanta.KOps.evalOp, ha, he] at h
        | some v =>
          simp [Quanta.KOps.evalOp, ha, he] at h
          rw [← h]
  | copy dst src =>
      exact evalOp_copy_preserves_broke h
  | load dst field idx _ty =>
      cases hi : Quanta.KOps.regLookup s.rf idx with
      | none => simp [Quanta.KOps.evalOp, hi] at h
      | some vi =>
        cases vi with
        | vBool _ => simp [Quanta.KOps.evalOp, hi] at h
        | vI32 _  => simp [Quanta.KOps.evalOp, hi] at h
        | vF32 _  => simp [Quanta.KOps.evalOp, hi] at h
        | vU32 n =>
          cases hl : Quanta.KOps.heapLookup s.heap field n.toNat with
          | none => simp [Quanta.KOps.evalOp, hi, hl] at h
          | some v =>
            simp [Quanta.KOps.evalOp, hi, hl] at h
            rw [← h]
  | store field idx src _ty =>
      cases hi : Quanta.KOps.regLookup s.rf idx with
      | none => simp [Quanta.KOps.evalOp, hi] at h
      | some vi =>
        cases hs2 : Quanta.KOps.regLookup s.rf src with
        | none => simp [Quanta.KOps.evalOp, hi, hs2] at h
        | some vs =>
          cases vi with
          | vBool _ => simp [Quanta.KOps.evalOp, hi, hs2] at h
          | vI32 _  => simp [Quanta.KOps.evalOp, hi, hs2] at h
          | vF32 _  => simp [Quanta.KOps.evalOp, hi, hs2] at h
          | vU32 n =>
            simp [Quanta.KOps.evalOp, hi, hs2] at h
            rw [← h]
  | quarkId _ =>
      rw [Quanta.KOps.evalOp.eq_def] at h
      simp at h; rw [← h]
  | protonId _ =>
      rw [Quanta.KOps.evalOp.eq_def] at h
      simp at h; rw [← h]
  | nucleusId _ =>
      rw [Quanta.KOps.evalOp.eq_def] at h
      simp at h; rw [← h]
  | protonSize _ =>
      rw [Quanta.KOps.evalOp.eq_def] at h
      simp at h; rw [← h]
  | quarkCount _ =>
      rw [Quanta.KOps.evalOp.eq_def] at h
      simp at h; rw [← h]
  | barrier =>
      rw [Quanta.KOps.evalOp.eq_def] at h
      simp at h; rw [← h]

/-- Boolean check for "is not the `.breakOp` constructor". Avoids
    needing `DecidableEq KernelOp` (which the `.branch`/`.loopOp`
    constructors carrying `List KernelOp` would force a recursive
    derivation for). -/
def isNotBreakOp : KernelOp → Bool
  | .breakOp => false
  | _        => true

/-- Predicate: every op in the list is `loopFreeOp` AND not `.breakOp`.
    All per-op-emitted ops fall into this class — only the structured-
    control arms emit `.branch` / `.loopOp`, and `.breakOp` only appears
    in the cross-Loop break of `br`. Defined recursively (rather than
    via `List.all`) to keep the cons-unfolding rfl-simple. -/
def loopFreeNoBreak : List KernelOp → Bool
  | []          => true
  | op :: rest  => loopFreeOp op && isNotBreakOp op && loopFreeNoBreak rest

@[simp] theorem loopFreeNoBreak_nil : loopFreeNoBreak [] = true := rfl

@[simp] theorem loopFreeNoBreak_cons (op : KernelOp) (rest : List KernelOp) :
    loopFreeNoBreak (op :: rest)
      = (loopFreeOp op && isNotBreakOp op && loopFreeNoBreak rest) := rfl

@[simp] theorem loopFreeNoBreak_append (xs ys : List KernelOp) :
    loopFreeNoBreak (xs ++ ys) = (loopFreeNoBreak xs && loopFreeNoBreak ys) := by
  induction xs with
  | nil => simp
  | cons x rest ih => simp [loopFreeNoBreak_cons, ih, Bool.and_assoc]

theorem loopFreeNoBreak_implies_loopFree {ops : List KernelOp}
    (h : loopFreeNoBreak ops = true) : loopFree ops = true := by
  induction ops with
  | nil => rfl
  | cons op rest ih =>
    simp only [loopFreeNoBreak_cons, Bool.and_eq_true] at h
    obtain ⟨⟨h_op, _⟩, h_rest⟩ := h
    rw [loopFree_cons]
    simp [h_op, ih h_rest]

/-- `isNotBreakOp op = true` ↔ `op ≠ .breakOp`. -/
theorem isNotBreakOp_iff_ne {op : KernelOp} :
    isNotBreakOp op = true ↔ op ≠ .breakOp := by
  cases op <;> simp [isNotBreakOp]

/-- List-level broke preservation. By induction on `ops`, using the
    single-op variant on the head and the `evalOps` cons short-circuit
    on `broke` (which is vacuous when the head preserves `broke = false`). -/
theorem evalOps_loopFreeNoBreak_preserves_broke
    {fuel : Nat} {s s' : State} {ops : List KernelOp}
    (h_lf : loopFreeNoBreak ops = true)
    (h_kst_ok : s.broke = false)
    (h : evalOps fuel s ops = some s') :
    s'.broke = false := by
  induction ops generalizing s with
  | nil =>
    rw [Quanta.KOps.evalOps.eq_def] at h
    simp at h
    rw [← h]; exact h_kst_ok
  | cons op rest ih =>
    simp only [loopFreeNoBreak_cons, Bool.and_eq_true] at h_lf
    obtain ⟨⟨h_op_lf, h_op_nb_bool⟩, h_rest⟩ := h_lf
    have h_op_nb : op ≠ .breakOp :=
      isNotBreakOp_iff_ne.mp h_op_nb_bool
    rw [Quanta.KOps.evalOps.eq_def] at h
    cases h_eo : evalOp fuel s op with
    | none => simp [h_eo] at h
    | some s_mid =>
      have h_mid_eq : s_mid.broke = s.broke :=
        evalOp_loopFree_no_break_preserves_broke h_op_lf h_op_nb h_eo
      have h_mid_ok : s_mid.broke = false := by rw [h_mid_eq]; exact h_kst_ok
      simp [h_eo, h_mid_ok] at h
      exact ih h_rest h_mid_ok h

/-- `commit` emits either `[]` (for `.reg`) or `[.const ...]` (for
    `.i32ConstSym`); both are loop-free with no `.breakOp`. The address
    SymVals (`bufferPtr`/`scaledIdx`/`bufferAccess`) make `commit`
    return `none`, so a `some` result is always one of the two
    well-shaped lists. -/
theorem commit_emits_loopFreeNoBreak
    {s : LowerState} {sv : SymVal} {r : Quanta.KOps.Reg}
    {s' : LowerState} {ops : List KernelOp}
    (h : s.commit sv = some (r, s', ops)) :
    loopFreeNoBreak ops = true := by
  match sv, h with
  | .reg _ _, h =>
    simp [LowerState.commit] at h
    obtain ⟨_, _, hops⟩ := h
    -- After `simp`, `hops : ops = []` (orientation may vary, so we
    -- pattern-match either direction).
    cases hops <;> rfl
  | .i32ConstSym _, h =>
    simp [LowerState.commit, LowerState.alloc] at h
    obtain ⟨_, _, hops⟩ := h
    cases hops <;> rfl

-- ════════════════════════════════════════════════════════════════════
-- L7.1 — opLoop iteration-counter monotonicity
--
-- `Quanta.KOps.opLoop fuel body f st` takes two fuel-like parameters:
-- the body fuel `fuel` (passed to `evalOps fuel st body` per
-- iteration) and the iteration counter `f` (bounds total iterations).
-- This module proves monotonicity in `f` alone, holding `fuel` fixed.
--
-- Body-fuel monotonicity (lifting `fuel`) is genuinely mutual with
-- `evalOp_fuel_mono` / `evalOps_fuel_mono` — that lives in L7.2.
-- Iteration-counter monotonicity, by contrast, is provable
-- standalone by induction on `f`:
--
-- * `f = 0` → opLoop returns none → contradicts the `some st'`
--   hypothesis. Vacuous.
-- * `f = f₀ + 1` → either `st.broke` (returns immediately, fuel
--   irrelevant) or recurses with `f₀`. Lift `f₀ ≤ f₀'` via the IH;
--   the body call (`evalOps fuel st body`) is unchanged.
--
-- Sufficient on its own for the wloop preservation theorem's
-- iteration-count lift (the surrounding evalOps's fuel is at least
-- as large as the one we used for the head).
-- ════════════════════════════════════════════════════════════════════

open Quanta.KOps (opLoop) in
/-- `opLoop` is monotone in the iteration counter `f`: any successful
    run at `f` succeeds at every `f' ≥ f` with the same result. -/
theorem opLoop_iter_mono
    {fuel : Nat} {body : List KernelOp} {f f' : Nat} {st st' : State}
    (h_le : f ≤ f')
    (h : opLoop fuel body f st = some st') :
    opLoop fuel body f' st = some st' := by
  induction f generalizing f' st with
  | zero =>
    -- opLoop fuel body 0 st = none, contradicting `h`.
    simp [Quanta.KOps.opLoop] at h
  | succ f₀ ih =>
    -- f' ≥ f₀ + 1 forces f' = f₀' + 1 for some f₀' ≥ f₀.
    rcases Nat.exists_eq_add_of_le h_le with ⟨k, hk⟩
    -- hk : f' = f₀ + 1 + k. Rewrite f' as (f₀ + k) + 1.
    have h_f'_eq : f' = (f₀ + k) + 1 := by omega
    rw [h_f'_eq]
    -- Unfold both sides' opLoop one step.
    rw [Quanta.KOps.opLoop] at h ⊢
    -- Split on st.broke.
    by_cases h_broke : st.broke = true
    · -- broke = true: returns some st.reset_broke regardless of fuel.
      simp [h_broke] at h ⊢
      exact h
    · have h_broke' : st.broke = false := by
        cases hb : st.broke
        · rfl
        · exact (h_broke hb).elim
      simp [h_broke'] at h ⊢
      -- Both reduce to `match evalOps fuel st body with ...`.
      cases h_body : Quanta.KOps.evalOps fuel st body with
      | none =>
        rw [h_body] at h; simp at h
      | some st_next =>
        rw [h_body] at h
        simp at h
        -- h : opLoop fuel body f₀ st_next = some st'
        -- Goal: opLoop fuel body (f₀ + k) st_next = some st'
        -- Apply IH with f₀ ≤ f₀ + k.
        have h_ih_le : f₀ ≤ f₀ + k := Nat.le_add_right _ _
        exact ih h_ih_le h

-- ════════════════════════════════════════════════════════════════════
-- L7.2 — fuel monotonicity for evalOp / evalOps
--
-- Strategy: avoid a three-way mutual termination dance by taking the
-- per-iteration body-lift hypothesis as a hypothesis. The mutual
-- block has just two members (evalOp_fuel_mono, evalOps_fuel_mono),
-- both decreasing on `sizeOf`. The `.loopOp` arm calls a *parametric*
-- helper `opLoop_body_fuel_mono_param` that takes the body lift
-- (provided by the mutual `evalOps_fuel_mono` IH on body, which is
-- structurally smaller than `.loopOp body`).
--
-- Combined with `opLoop_iter_mono` (L7.1), this gives the full
-- fuel-monotonicity surface needed by downstream wloop preservation.
-- ════════════════════════════════════════════════════════════════════

/-- Parametric body-fuel lift for `opLoop`: given that `body`'s
    evalOps is monotone in fuel (passed as a hypothesis), lifting the
    body fuel of `opLoop fuel body f st` to a larger value preserves
    successful runs. Proof by induction on `f` (iteration counter).

    The body-lift hypothesis is parametric so this lemma is standalone
    (no mutual dependency). The caller (`evalOp_fuel_mono .loopOp`)
    fills it via the mutual `evalOps_fuel_mono` IH on body. -/
theorem opLoop_body_fuel_mono_param
    {fuel fuel' : Nat} {body : List KernelOp} {f : Nat} {st st' : State}
    (h_body_lift : ∀ {st_in st_out : State},
      Quanta.KOps.evalOps fuel st_in body = some st_out →
      Quanta.KOps.evalOps fuel' st_in body = some st_out)
    (h : Quanta.KOps.opLoop fuel body f st = some st') :
    Quanta.KOps.opLoop fuel' body f st = some st' := by
  induction f generalizing st with
  | zero => simp [Quanta.KOps.opLoop] at h
  | succ f₀ ih =>
    rw [Quanta.KOps.opLoop] at h ⊢
    by_cases h_broke : st.broke = true
    · simp [h_broke] at h ⊢
      exact h
    · have h_broke' : st.broke = false := by
        cases hb : st.broke
        · rfl
        · exact (h_broke hb).elim
      simp [h_broke'] at h ⊢
      cases h_body : Quanta.KOps.evalOps fuel st body with
      | none => rw [h_body] at h; simp at h
      | some st_next =>
        rw [h_body] at h
        simp at h
        have h_body' : Quanta.KOps.evalOps fuel' st body = some st_next :=
          h_body_lift h_body
        rw [h_body']
        simp
        exact ih h

mutual
/-- `evalOp` is monotone in fuel. -/
theorem evalOp_fuel_mono
    {fuel fuel' : Nat} {s s' : State} {op : KernelOp}
    (h_le : fuel ≤ fuel')
    (h : evalOp fuel s op = some s') :
    evalOp fuel' s op = some s' := by
  cases op with
  | branch cond thenOps elseOps =>
    rw [Quanta.KOps.evalOp.eq_def] at h ⊢
    cases hr : Quanta.KOps.regLookup s.rf cond with
    | none => simp [hr] at h
    | some v =>
      cases v with
      | vBool b =>
        cases b
        · simp [hr] at h ⊢
          exact evalOps_fuel_mono h_le h
        · simp [hr] at h ⊢
          exact evalOps_fuel_mono h_le h
      | _ => simp [hr] at h
  | loopOp body =>
    -- evalOp fuel s (.loopOp body) = opLoop fuel body fuel s.
    -- Lift in two steps:
    --   (1) iteration-counter lift: opLoop fuel body fuel s →
    --       opLoop fuel body fuel' s (via opLoop_iter_mono, L7.1).
    --   (2) body-fuel lift: opLoop fuel body fuel' s →
    --       opLoop fuel' body fuel' s (via opLoop_body_fuel_mono_param
    --       with the per-iteration evalOps lift provided by the
    --       mutual evalOps_fuel_mono on body, which is smaller than
    --       .loopOp body).
    rw [Quanta.KOps.evalOp.eq_def] at h ⊢
    simp only [] at h ⊢
    -- h : opLoop fuel body fuel s = some s'
    have h_iter_lift :
        Quanta.KOps.opLoop fuel body fuel' s = some s' :=
      opLoop_iter_mono h_le h
    exact opLoop_body_fuel_mono_param
      (fun {_ _} => evalOps_fuel_mono h_le) h_iter_lift
  | _ =>
    all_goals (rw [Quanta.KOps.evalOp.eq_def] at h ⊢; exact h)
termination_by sizeOf op

/-- `evalOps` is monotone in fuel. -/
theorem evalOps_fuel_mono
    {fuel fuel' : Nat} {s s' : State} {ops : List KernelOp}
    (h_le : fuel ≤ fuel')
    (h : evalOps fuel s ops = some s') :
    evalOps fuel' s ops = some s' := by
  cases ops with
  | nil =>
    rw [Quanta.KOps.evalOps.eq_def] at h ⊢
    exact h
  | cons op rest =>
    rw [Quanta.KOps.evalOps.eq_def] at h ⊢
    simp only [] at h ⊢
    cases h_eo : evalOp fuel s op with
    | none => rw [h_eo] at h; simp at h
    | some s_mid =>
      rw [h_eo] at h
      simp at h
      have h_head' : evalOp fuel' s op = some s_mid :=
        evalOp_fuel_mono h_le h_eo
      rw [h_head']
      simp
      by_cases hbr : s_mid.broke = true
      · simp [hbr] at h ⊢
        exact h
      · have hbr' : s_mid.broke = false := by
          cases hb : s_mid.broke
          · rfl
          · exact (hbr hb).elim
        simp [hbr'] at h ⊢
        exact evalOps_fuel_mono h_le h
termination_by sizeOf ops
end

-- ════════════════════════════════════════════════════════════════════
-- L7.3 — cons-composer fuel-mono variant
--
-- Mirrors the existing `evalOps_append_loopFreeDeep_head` and
-- `preservation_evalInstrs_cons_compose` but bridges the fuel gap via
-- `evalOps_fuel_mono` (L7.2) instead of `evalOps_loopFreeDeep_fuel_irrel`.
-- This drops the `loopFreeDeep ops_head = true` precondition — `ops_head`
-- can now contain `.loopOp` (the exact shape `wloop` lowers to).
--
-- The signature also lets the caller supply a non-zero head fuel
-- `F_head` (with `F_head ≤ F`) rather than requiring the head to be
-- evaluable at fuel 0. wloop preservation will need this: the
-- `.loopOp body` head needs `≥ iteration_count` fuel, supplied by the
-- caller's surrounding `evalInstrs` fuel.
--
-- The existing `loopFreeDeep` / `loopFree` variants stay in place —
-- they keep being the cheap choice for non-loop cons-cases (the 35
-- closed cons-case theorems don't need to change).
-- ════════════════════════════════════════════════════════════════════

/-- Fuel-mono variant of `evalOps_append_loopFreeDeep_head`: bridges
    the fuel gap via `evalOps_fuel_mono` instead of fuel-irrel, so
    `ops_head` may contain `.loopOp`. -/
theorem evalOps_append_fuel_mono_head
    {F_head F : Nat} {kst kst_mid kst' : State}
    {ops_head ops_rest : List KernelOp}
    (h_le : F_head ≤ F)
    (h_head : evalOps F_head kst ops_head = some kst_mid)
    (h_no_broke : kst_mid.broke = false)
    (h_rest : evalOps F kst_mid ops_rest = some kst') :
    evalOps F kst (ops_head ++ ops_rest) = some kst' := by
  have h_head_F : evalOps F kst ops_head = some kst_mid :=
    evalOps_fuel_mono h_le h_head
  rw [evalOps_append h_head_F h_no_broke]
  exact h_rest

/-- Fuel-mono variant of `preservation_evalInstrs_cons_compose`:
    packages head + tail Refines existentials into a bundled
    `evalOps F` on `ops_head ++ ops_rest` and a final `Refines`,
    via `evalOps_fuel_mono`. Used by wloop preservation. -/
theorem preservation_evalInstrs_cons_compose_with_loop
    {F_head F : Nat} {kst kst_mid : Quanta.KOps.State}
    {ops_head ops_rest : List KernelOp}
    {ws' : WasmState} {s' : LowerState}
    {layout : BufferLayout}
    (h_le : F_head ≤ F)
    (h_head : evalOps F_head kst ops_head = some kst_mid)
    (h_no_broke : kst_mid.broke = false)
    (h_rest : ∃ kst', evalOps F kst_mid ops_rest = some kst'
                       ∧ Refines ws' s' kst' layout) :
    ∃ kst', evalOps F kst (ops_head ++ ops_rest) = some kst'
              ∧ Refines ws' s' kst' layout := by
  obtain ⟨kst', h_eval', R'⟩ := h_rest
  refine ⟨kst', ?_, R'⟩
  exact evalOps_append_fuel_mono_head h_le h_head h_no_broke h_eval'

end Quanta.Wasm
