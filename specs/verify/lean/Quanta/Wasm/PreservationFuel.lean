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

end Quanta.Wasm
