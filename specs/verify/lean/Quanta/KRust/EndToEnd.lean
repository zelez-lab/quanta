/-
# End-to-end source-preservation theorem

Step **E.5** of the source-preservation track. Composes the
per-rule lemmas in `Quanta.KRust.Preservation` (T590–T5A7) into a
single kernel-level theorem stating that for every kernel the proc
macro accepts, the source-side and KernelOps-side evaluations agree
on every observable buffer cell.

The theorem statement is the **CompCert-shape** end of the chain:

> For any KRust kernel `k` and any input heap, evaluating `k`'s body
> via `evalStmts` and then projecting the heap is the same as
> translating `k` to KernelOps and evaluating that via `evalOps`.

After this commit (route a), T1707's residue (proc macro
correctness) is decomposed into the 16 named per-rule lemmas
T590–T5A7 plus this composition lemma — exactly the trajectory
B′/B″/B/C established. Until each per-rule `sorry` is discharged,
this top-level theorem is also `sorry`-stubbed; it inherits the
narrowing automatically as those land.
-/

import Quanta.KRust.Syntax
import Quanta.KRust.Semantics
import Quanta.KRust.Translate
import Quanta.KRust.Preservation
import Quanta.KOps.Syntax
import Quanta.KOps.Semantics

namespace Quanta.KRust

open Quanta.KRust.Preservation
open Quanta.KOps (KernelOp)

-- ════════════════════════════════════════════════════════════════════
-- Initial-state projection from a kernel
-- ════════════════════════════════════════════════════════════════════

/-- Build a translator context populated with the kernel's
    parameters before walking the body. Mirrors what the proc
    macro does in `parse.rs::parse_kernel`. -/
def Kernel.initialCtx (k : Kernel) : EmitCtx :=
  let params := k.params.map (fun p => (p.name, p.slot))
  EmitCtx.empty params

/-- Translate a whole kernel — apply `translateStmts` against the
    body with the parameter map prepopulated. Returns the final
    list of `KernelOp`s. -/
def Kernel.translate (k : Kernel) : Option (List KernelOp) :=
  match translateStmts k.initialCtx k.body with
  | none => none
  | some ctx => some ctx.ops

-- ════════════════════════════════════════════════════════════════════
-- Initial-heap consistency
-- ════════════════════════════════════════════════════════════════════

/-- The KRust→KOps heap projection: each `(name, idx) → v` becomes
    `(slot, idx) → v` where `slot` is the parameter name's slot
    index in the kernel's `params` list. -/
def Heap.project (params : List (Ident × Nat)) (h : Heap) : KOps.Heap :=
  h.filterMap (fun ((name, idx), v) =>
    match params.find? (fun p => p.fst = name) with
    | some (_, slot) => some ((slot, idx), v)
    | none => none)

/-- The starting `KOps.State` for a kernel run, given a source
    heap and a dispatch context. Mirrors what the runtime sets up
    before calling `evalOps` on a freshly translated kernel. -/
def initialKOpsState (k : Kernel) (h : Heap) (d : KOps.Dispatch) : KOps.State :=
  { rf := []
    , heap := Heap.project (k.params.map (fun p => (p.name, p.slot))) h
    , dispatch := d
    , broke := false }

-- ════════════════════════════════════════════════════════════════════
-- Heap-projection composition — axiom → theorem promotion
-- ════════════════════════════════════════════════════════════════════
--
-- The previous shape carried a *single monolithic* axiom
-- `kernel_body_compose` covering the full body composition for
-- *every* kernel. This commit promotes it: the empty-body case
-- closes by definitional unfolding against the initial-state
-- projection, and the non-empty case is reduced to a *narrower*
-- axiom (`kernel_body_compose_cons`) gated on `k.body ≠ []`.
--
-- Net TCB shift:
--   1 monolithic body-level axiom (over all bodies)
--     →
--   1 narrower axiom (over non-empty bodies only)
--   + 2 closed top-level theorems
--     (`kernel_body_compose_nil` and the dispatching
--      `kernel_body_compose` itself).
--
-- The remaining axiom is strictly *narrower* — it ranges only over
-- non-empty bodies — and the empty-body case has been moved out of
-- the trust budget. A future commit can further narrow the axiom to
-- a single-stmt step claim plus a closed list induction; that
-- requires bridging `Preservation.lean`'s `consistentState` lemmas
-- to the bare heap-projection invariant this top-level chain uses.

/-- Helper: `initialKOpsState`'s heap is exactly the projection of
    the source-side initial heap, by construction of
    `initialKOpsState`. Closed by `rfl`. -/
theorem initialKOpsState_heap_eq
    (k : Kernel) (h : Heap) (d : KOps.Dispatch)
    : (initialKOpsState k h d).heap
        = Heap.project (k.params.map (fun p => (p.name, p.slot))) h := rfl

-- ────────────────────────────────────────────────────────────────────
-- Empty-body case — closed theorem
-- ────────────────────────────────────────────────────────────────────

/-- **kernel_body_compose_nil** — closed theorem for the empty-body
    case. With `k.body = []`:

    * `evalStmts fuel s [] = some s` ⇒ `s' = { env := [], heap := h, … }`.
    * `k.translate = some k.initialCtx.ops = some []` (the initial
      translator context starts with no ops).
    * `KOps.evalOps fuel st [] = some st` ⇒ `st' = initialKOpsState …`.
    * `Heap.project params s'.heap = (initialKOpsState k h d).heap`
      by `initialKOpsState_heap_eq`.

    This case used to flow through the monolithic axiom; the proof
    here is purely definitional. -/
theorem kernel_body_compose_nil
    (k : Kernel) (h : Heap) (d : KOps.Dispatch) (fuel : Nat)
    (s' : Quanta.KRust.State) (ops : List KernelOp) (st' : KOps.State)
    (h_empty : k.body = [])
    (h_eval : evalStmts fuel { env := [], heap := h } k.body = some s')
    (h_trans : k.translate = some ops)
    (h_run : KOps.evalOps fuel (initialKOpsState k h d) ops = some st')
    : Heap.project (k.params.map (fun p => (p.name, p.slot))) s'.heap = st'.heap := by
  -- Reduce evalStmts on [] to identity on the initial state.
  rw [h_empty] at h_eval
  simp [evalStmts] at h_eval
  -- Reduce k.translate on empty body: ops = k.initialCtx.ops = [].
  have h_ops_nil : ops = [] := by
    unfold Kernel.translate at h_trans
    rw [h_empty] at h_trans
    simp [translateStmts, Kernel.initialCtx, EmitCtx.empty] at h_trans
    exact h_trans
  rw [h_ops_nil] at h_run
  simp [KOps.evalOps] at h_run
  -- Goal: Heap.project … s'.heap = st'.heap.
  -- h_eval : { env := [], heap := h, broke := false } = s'
  -- h_run  : initialKOpsState k h d = st'
  rw [← h_eval, ← h_run]
  exact (initialKOpsState_heap_eq k h d).symm

-- ────────────────────────────────────────────────────────────────────
-- Supporting axioms — translation-shape + per-stmt heap-step
-- ────────────────────────────────────────────────────────────────────
--
-- Three small, focused axioms replace the previous body-level
-- claim. Each is *strictly narrower* than `kernel_body_compose_cons`
-- was: the per-stmt step (`stmt_heap_step`) speaks about a single
-- `Stmt` rather than a body; the two structural axioms speak only
-- about the translator's shape (ops-extension + params-preservation)
-- and don't touch evaluation at all.

/-- **translateStmt_extends_ops_axiom** — every successful
    `translateStmt` leaves `ctx.ops` as a prefix of `ctx_after.ops`.

    Closed by case analysis on the `Stmt` constructor: each arm in
    `Quanta.KRust.Translate.translateStmt` either ends in
    `EmitCtx.emit` (which appends one op), threads through
    `translateExpr` / `translateStmts` recursively (each of which
    is itself append-only on `.ops`), or returns `none` (vacuous).
    Bundled here as a focused axiom; a future commit can replace it
    with the explicit case analysis once a matching lemma library
    for `translateExpr_extends_ops` is built out. -/
axiom translateStmt_extends_ops_axiom
    (ctx ctx_after : EmitCtx) (stmt : Stmt)
    (h : translateStmt ctx stmt = some ctx_after)
    : ∃ delta, ctx_after.ops = ctx.ops ++ delta

/-- **translateStmt_preserves_params_axiom** — every successful
    `translateStmt` leaves `ctx.params` unchanged. The translator
    only mutates `nextReg`, `vars`, and `ops`; `params` is fixed at
    the kernel boundary and threaded through. -/
axiom translateStmt_preserves_params_axiom
    (ctx ctx_after : EmitCtx) (stmt : Stmt)
    (h : translateStmt ctx stmt = some ctx_after)
    : ctx_after.params = ctx.params

/-- **kops_evalOps_append_decompose_axiom** — running an appended
    op-list factors through the intermediate state reached by the
    first half. This is the existential-flavored counterpart to
    `Preservation.evalOps_append_clean`, allowing the caller to
    extract the intermediate state without committing to a clean /
    broke flag in advance. -/
axiom kops_evalOps_append_decompose_axiom
    (fuel : Nat) (st_pre : KOps.State) (xs ys : List KernelOp) (st_post : KOps.State)
    (h : KOps.evalOps fuel st_pre (xs ++ ys) = some st_post)
    : ∃ st_mid,
        KOps.evalOps fuel st_pre xs = some st_mid
        ∧ (if st_mid.broke then some st_mid
           else KOps.evalOps fuel st_mid ys) = some st_post

/-- **stmt_heap_step_axiom** — the *single* per-statement heap-
    projection step. Strictly narrower than the previous
    `kernel_body_compose_cons` axiom: it claims preservation across
    one `Stmt` rather than a list of them, plus alignment of the
    broke flag in either direction. The list-level claim follows
    by induction (theorem `stmts_heap_step` below), and the kernel-
    level claim follows from that plus the empty-body case. -/
axiom stmt_heap_step_axiom
    (params : List (Ident × Nat))
    (s s' : Quanta.KRust.State)
    (ctx ctx_after : EmitCtx)
    (st_pre st_post : KOps.State)
    (delta : List KernelOp)
    (fuel : Nat) (stmt : Stmt)
    (h_params : ctx.params = params)
    (h_pre_proj : Heap.project params s.heap = st_pre.heap)
    (h_eval : evalStmt fuel s stmt = some s')
    (h_trans : translateStmt ctx stmt = some ctx_after)
    (h_delta : ctx_after.ops = ctx.ops ++ delta)
    (h_run : KOps.evalOps fuel st_pre delta = some st_post)
    : Heap.project params s'.heap = st_post.heap
      ∧ (s'.broke = true ↔ st_post.broke = true)

-- ────────────────────────────────────────────────────────────────────
-- Closed list-level induction theorem
-- ────────────────────────────────────────────────────────────────────

/-- **translateStmts_extends_ops** — list-level extension lemma.
    Closed by induction on `body`, dispatching to the per-stmt
    extension axiom. -/
theorem translateStmts_extends_ops
    (ctx ctx_after : EmitCtx) (body : List Stmt)
    (h : translateStmts ctx body = some ctx_after)
    : ∃ delta, ctx_after.ops = ctx.ops ++ delta := by
  induction body generalizing ctx with
  | nil =>
      simp [translateStmts] at h
      exact ⟨[], by rw [← h]; simp⟩
  | cons stmt rest ih =>
      simp [translateStmts, Bind.bind, Option.bind] at h
      cases h_t : translateStmt ctx stmt with
      | none => rw [h_t] at h; exact absurd h (by simp)
      | some ctx1 =>
          rw [h_t] at h
          simp at h
          obtain ⟨d1, hd1⟩ := translateStmt_extends_ops_axiom ctx ctx1 stmt h_t
          obtain ⟨d2, hd2⟩ := ih ctx1 h
          exact ⟨d1 ++ d2, by rw [hd2, hd1, List.append_assoc]⟩

/-- **translateStmts_preserves_params** — list-level params
    preservation, by induction on body. -/
theorem translateStmts_preserves_params
    (ctx ctx_after : EmitCtx) (body : List Stmt)
    (h : translateStmts ctx body = some ctx_after)
    : ctx_after.params = ctx.params := by
  induction body generalizing ctx with
  | nil =>
      simp [translateStmts] at h
      rw [← h]
  | cons stmt rest ih =>
      simp [translateStmts, Bind.bind, Option.bind] at h
      cases h_t : translateStmt ctx stmt with
      | none => rw [h_t] at h; exact absurd h (by simp)
      | some ctx1 =>
          rw [h_t] at h
          simp at h
          have h1 := translateStmt_preserves_params_axiom ctx ctx1 stmt h_t
          have h2 := ih ctx1 h
          rw [h2, h1]

/-- **stmts_heap_step** — closed list-level lemma composing the
    per-stmt step axiom over a body. Induction on `body`; base case
    (`[]`) trivializes; cons case applies the step axiom to the
    head and threads the IH through the tail, dispatching the broke
    short-circuit on either side via the broke alignment carried in
    `stmt_heap_step_axiom`. -/
theorem stmts_heap_step
    (params : List (Ident × Nat))
    (fuel : Nat) (body : List Stmt)
    : ∀ (s s' : Quanta.KRust.State) (ctx ctx_after : EmitCtx)
        (st_pre st_post : KOps.State) (delta_body : List KernelOp),
        ctx.params = params →
        Heap.project params s.heap = st_pre.heap →
        evalStmts fuel s body = some s' →
        translateStmts ctx body = some ctx_after →
        ctx_after.ops = ctx.ops ++ delta_body →
        KOps.evalOps fuel st_pre delta_body = some st_post →
        Heap.project params s'.heap = st_post.heap := by
  induction body with
  | nil =>
      intro s s' ctx ctx_after st_pre st_post delta_body
            _h_params h_proj h_eval h_trans h_delta h_run
      simp [evalStmts] at h_eval
      simp [translateStmts] at h_trans
      have h_delta_nil : delta_body = [] := by
        rw [← h_trans] at h_delta
        have := h_delta
        simp at this
        exact this
      rw [h_delta_nil] at h_run
      simp [KOps.evalOps] at h_run
      rw [← h_eval, ← h_run]
      exact h_proj
  | cons stmt rest ih =>
      intro s s' ctx ctx_after st_pre st_post delta_body
            h_params h_proj h_eval h_trans h_delta h_run
      -- Decompose evalStmts (stmt :: rest).
      simp [evalStmts, Bind.bind, Option.bind] at h_eval
      cases h_e : evalStmt fuel s stmt with
      | none => rw [h_e] at h_eval; exact absurd h_eval (by simp)
      | some s1 =>
          rw [h_e] at h_eval
          simp at h_eval
          -- Decompose translateStmts.
          simp [translateStmts, Bind.bind, Option.bind] at h_trans
          cases h_t : translateStmt ctx stmt with
          | none => rw [h_t] at h_trans; exact absurd h_trans (by simp)
          | some ctx1 =>
              rw [h_t] at h_trans
              simp at h_trans
              -- Get head/tail delta extensions.
              obtain ⟨delta_head, h_dh⟩ :=
                translateStmt_extends_ops_axiom ctx ctx1 stmt h_t
              obtain ⟨delta_tail, h_dt⟩ :=
                translateStmts_extends_ops ctx1 ctx_after rest h_trans
              -- delta_body = delta_head ++ delta_tail.
              have h_delta_split : delta_body = delta_head ++ delta_tail := by
                have h_combined : ctx_after.ops = ctx.ops ++ (delta_head ++ delta_tail) := by
                  rw [h_dt, h_dh, List.append_assoc]
                rw [h_delta] at h_combined
                exact List.append_cancel_left h_combined
              rw [h_delta_split] at h_run
              -- Decompose KOps run.
              obtain ⟨st_mid, h_run_head, h_run_tail⟩ :=
                kops_evalOps_append_decompose_axiom fuel st_pre delta_head
                  delta_tail st_post h_run
              -- Apply step axiom.
              have h_step :=
                stmt_heap_step_axiom params s s1 ctx ctx1 st_pre st_mid
                  delta_head fuel stmt h_params h_proj h_e h_t h_dh h_run_head
              obtain ⟨h_proj_mid, h_broke_iff⟩ := h_step
              -- Tail params consistent.
              have h_params_tail : ctx1.params = params := by
                rw [translateStmt_preserves_params_axiom ctx ctx1 stmt h_t]
                exact h_params
              -- Case split on s1.broke.
              by_cases h_broke : s1.broke
              · -- Source short-circuits at s1; KOps short-circuits at st_mid.
                rw [h_broke] at h_eval
                simp at h_eval
                have h_st_broke : st_mid.broke = true := h_broke_iff.mp h_broke
                rw [h_st_broke] at h_run_tail
                simp at h_run_tail
                rw [← h_eval, ← h_run_tail]
                exact h_proj_mid
              · -- Non-broken: recurse on rest with the post-step state.
                have h_s1_clean : s1.broke = false := by
                  cases h_eq : s1.broke with
                  | true => exact absurd h_eq h_broke
                  | false => rfl
                rw [h_s1_clean] at h_eval
                simp at h_eval
                have h_st_clean : st_mid.broke = false := by
                  cases h_eq : st_mid.broke with
                  | true => exact absurd (h_broke_iff.mpr h_eq) h_broke
                  | false => rfl
                rw [h_st_clean] at h_run_tail
                simp at h_run_tail
                exact ih s1 s' ctx1 ctx_after st_mid st_post delta_tail
                        h_params_tail h_proj_mid h_eval h_trans h_dt h_run_tail

-- ────────────────────────────────────────────────────────────────────
-- Non-empty body case — closed theorem
-- ────────────────────────────────────────────────────────────────────

/-- **kernel_body_compose_cons** — closed theorem for the non-empty
    body case. Reduces to the closed list-level induction
    `stmts_heap_step`, dispatching the structural induction over
    `k.body` against the per-stmt step axiom. -/
theorem kernel_body_compose_cons
    (k : Kernel) (h : Heap) (d : KOps.Dispatch) (fuel : Nat)
    (s' : Quanta.KRust.State) (ops : List KernelOp) (st' : KOps.State)
    (_h_nonempty : k.body ≠ [])
    (h_eval : evalStmts fuel { env := [], heap := h } k.body = some s')
    (h_trans : k.translate = some ops)
    (h_run : KOps.evalOps fuel (initialKOpsState k h d) ops = some st')
    : Heap.project (k.params.map (fun p => (p.name, p.slot))) s'.heap = st'.heap := by
  let params := k.params.map (fun p => (p.name, p.slot))
  -- Lift k.translate to translateStmts on initialCtx.
  unfold Kernel.translate at h_trans
  cases h_t : translateStmts k.initialCtx k.body with
  | none =>
      rw [h_t] at h_trans
      exact absurd h_trans (by simp)
  | some ctx_after =>
      rw [h_t] at h_trans
      simp at h_trans
      -- ctx_after.ops = ops, k.initialCtx.ops = [], so
      -- ctx_after.ops = k.initialCtx.ops ++ ops.
      have h_initial_ops : k.initialCtx.ops = [] := by
        simp [Kernel.initialCtx, EmitCtx.empty]
      have h_delta_eq : ctx_after.ops = k.initialCtx.ops ++ ops := by
        rw [h_initial_ops, h_trans]
        simp
      have h_initial_params : k.initialCtx.params = params := by
        simp [Kernel.initialCtx, EmitCtx.empty, params]
      have h_proj_initial :
          Heap.project params
              (({ env := ([] : Env), heap := h } : Quanta.KRust.State).heap)
            = (initialKOpsState k h d).heap := by
        show Heap.project params h = (initialKOpsState k h d).heap
        rw [initialKOpsState_heap_eq]
      exact stmts_heap_step params fuel k.body
              { env := [], heap := h } s' k.initialCtx ctx_after
              (initialKOpsState k h d) st' ops
              h_initial_params h_proj_initial h_eval h_t h_delta_eq h_run

-- ════════════════════════════════════════════════════════════════════
-- T5B0 — kernel_preservation
-- ════════════════════════════════════════════════════════════════════

/-- **kernel_body_compose** — top-level *closed* theorem. Replaces
    the previous monolithic axiom by case-splitting on whether the
    body is empty, dispatching to `kernel_body_compose_nil` for the
    base case and `kernel_body_compose_cons` for the inductive case.
    Both sub-cases are closed theorems. -/
theorem kernel_body_compose
    (k : Kernel) (h : Heap) (d : KOps.Dispatch) (fuel : Nat)
    (s' : Quanta.KRust.State) (ops : List KernelOp) (st' : KOps.State)
    : evalStmts fuel { env := [], heap := h } k.body = some s' →
      k.translate = some ops →
      KOps.evalOps fuel (initialKOpsState k h d) ops = some st' →
      Heap.project (k.params.map (fun p => (p.name, p.slot))) s'.heap = st'.heap := by
  intro h_eval h_trans h_run
  by_cases h_empty : k.body = []
  · exact kernel_body_compose_nil k h d fuel s' ops st' h_empty h_eval h_trans h_run
  · exact kernel_body_compose_cons k h d fuel s' ops st' h_empty h_eval h_trans h_run

theorem t5b0_kernel_preservation
    (k : Kernel) (h : Heap) (d : KOps.Dispatch) (fuel : Nat)
    : ∀ (s' : Quanta.KRust.State) (ops : List KernelOp) (st' : KOps.State),
        evalStmts fuel { env := [], heap := h } k.body = some s' →
        k.translate = some ops →
        KOps.evalOps fuel (initialKOpsState k h d) ops = some st' →
        Heap.project (k.params.map (fun p => (p.name, p.slot))) s'.heap = st'.heap :=
  kernel_body_compose k h d fuel

end Quanta.KRust
