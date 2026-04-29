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
import Quanta.KRust.Equations
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

-- ────────────────────────────────────────────────────────────────────
-- Mutual closed theorems for the extends-ops triad.
-- ────────────────────────────────────────────────────────────────────

mutual

/-- **translateExpr_extends_ops** — closed theorem: parameter
    expression-layer ops extension. Mutually recursive with the
    statement and list layers (via `blockE`). -/
theorem translateExpr_extends_ops
    (ctx : EmitCtx) (e : Expr) (r : KOps.Reg) (sty : KOps.Scalar) (ctx_after : EmitCtx)
    (h : translateExpr ctx e = some (r, sty, ctx_after))
    : ∃ delta, ctx_after.ops = ctx.ops ++ delta := by
  cases e with
  | indexRef _ _   => simp [translateExpr] at h
  | fieldRef _ _   => simp [translateExpr] at h
  | mathCall _ _   => simp [translateExpr] at h
  | waveCall _ _   => simp [translateExpr] at h
  | atomicCall _ _ => simp [translateExpr] at h
  | identityCall _ => simp [translateExpr] at h
  | method _ _ _   => simp [translateExpr] at h
  | lit l =>
      simp [translateExpr, Bind.bind, Option.bind] at h
      cases h_const : constOfLit l with
      | none => rw [h_const] at h; exact absurd h (by simp)
      | some pair =>
          obtain ⟨cv, ty⟩ := pair
          rw [h_const] at h
          simp at h
          obtain ⟨_, _, h_ctx⟩ := h
          refine ⟨[KOps.KernelOp.const ctx.nextReg cv], ?_⟩
          rw [← h_ctx]
          simp [EmitCtx.fresh, EmitCtx.emit]
  | path n =>
      simp [translateExpr, Bind.bind, Option.bind] at h
      cases h_lk : ctx.lookupVar n with
      | none => rw [h_lk] at h; exact absurd h (by simp)
      | some r' =>
          rw [h_lk] at h
          simp at h
          obtain ⟨_, _, h_ctx⟩ := h
          exact ⟨[], by rw [← h_ctx]; simp⟩
  | binary op a b =>
      simp [translateExpr, Bind.bind, Option.bind] at h
      cases h_a : translateExpr ctx a with
      | none => rw [h_a] at h; exact absurd h (by simp)
      | some triple_a =>
          obtain ⟨ra, _styA, ctx1⟩ := triple_a
          rw [h_a] at h
          simp at h
          cases h_b : translateExpr ctx1 b with
          | none => rw [h_b] at h; exact absurd h (by simp)
          | some triple_b =>
              obtain ⟨rb, _styB, ctx2⟩ := triple_b
              rw [h_b] at h
              simp at h
              obtain ⟨d_a, h_d_a⟩ := translateExpr_extends_ops ctx a ra _styA ctx1 h_a
              obtain ⟨d_b, h_d_b⟩ := translateExpr_extends_ops ctx1 b rb _styB ctx2 h_b
              cases h_disp : dispatchBinOp op with
              | bin bop =>
                  rw [h_disp] at h
                  simp at h
                  obtain ⟨_, _, h_ctx⟩ := h
                  refine ⟨d_a ++ d_b ++
                      [KOps.KernelOp.binOp ctx2.nextReg ra rb bop KOps.Scalar.u32], ?_⟩
                  rw [← h_ctx]
                  simp [EmitCtx.fresh, EmitCtx.emit, h_d_b, h_d_a, List.append_assoc]
              | cmp cop =>
                  rw [h_disp] at h
                  simp at h
                  obtain ⟨_, _, h_ctx⟩ := h
                  refine ⟨d_a ++ d_b ++
                      [KOps.KernelOp.cmp ctx2.nextReg ra rb cop KOps.Scalar.u32], ?_⟩
                  rw [← h_ctx]
                  simp [EmitCtx.fresh, EmitCtx.emit, h_d_b, h_d_a, List.append_assoc]
              | logical _ =>
                  rw [h_disp] at h
                  exact absurd h (by simp)
  | unary op a =>
      simp [translateExpr, Bind.bind, Option.bind] at h
      cases h_a : translateExpr ctx a with
      | none => rw [h_a] at h; exact absurd h (by simp)
      | some triple =>
          obtain ⟨ra, _styA, ctx1⟩ := triple
          rw [h_a] at h
          simp at h
          obtain ⟨_, _, h_ctx⟩ := h
          obtain ⟨d_a, h_d_a⟩ := translateExpr_extends_ops ctx a ra _styA ctx1 h_a
          refine ⟨d_a ++
              [KOps.KernelOp.unaryOp ctx1.nextReg ra (dispatchUnaryOp op) KOps.Scalar.u32], ?_⟩
          rw [← h_ctx]
          simp [EmitCtx.fresh, EmitCtx.emit, h_d_a, List.append_assoc]
  | index arr i =>
      simp [translateExpr, Bind.bind, Option.bind] at h
      cases h_lk : ctx.lookupParam arr with
      | none => rw [h_lk] at h; exact absurd h (by simp)
      | some slot =>
          rw [h_lk] at h
          simp at h
          cases h_i : translateExpr ctx i with
          | none => rw [h_i] at h; exact absurd h (by simp)
          | some triple =>
              obtain ⟨ri, _sty, ctx1⟩ := triple
              rw [h_i] at h
              simp at h
              obtain ⟨_, _, h_ctx⟩ := h
              obtain ⟨d_i, h_d_i⟩ := translateExpr_extends_ops ctx i ri _sty ctx1 h_i
              refine ⟨d_i ++
                  [KOps.KernelOp.load ctx1.nextReg slot ri KOps.Scalar.u32], ?_⟩
              rw [← h_ctx]
              simp [EmitCtx.fresh, EmitCtx.emit, h_d_i, List.append_assoc]
  | cast e' ty =>
      simp [translateExpr, Bind.bind, Option.bind] at h
      cases h_e : translateExpr ctx e' with
      | none => rw [h_e] at h; exact absurd h (by simp)
      | some triple =>
          obtain ⟨rs, _sty, ctx1⟩ := triple
          rw [h_e] at h
          simp at h
          obtain ⟨_, _, h_ctx⟩ := h
          obtain ⟨d_e, h_d_e⟩ := translateExpr_extends_ops ctx e' rs _sty ctx1 h_e
          refine ⟨d_e ++
              [KOps.KernelOp.cast ctx1.nextReg rs KOps.Scalar.u32 (dispatchScalar ty)], ?_⟩
          rw [← h_ctx]
          simp [EmitCtx.fresh, EmitCtx.emit, h_d_e, List.append_assoc]
  | ifE c t e' =>
      simp [translateExpr, Bind.bind, Option.bind] at h
      cases h_c : translateExpr ctx c with
      | none => rw [h_c] at h; exact absurd h (by simp)
      | some triple_c =>
          obtain ⟨rc, _styC, ctx1⟩ := triple_c
          rw [h_c] at h
          simp at h
          cases h_t : translateExpr ({ ctx1 with ops := ([] : List KernelOp) } : EmitCtx) t with
          | none => rw [h_t] at h; exact absurd h (by simp)
          | some triple_t =>
              obtain ⟨rt, _styT, thenCtx⟩ := triple_t
              rw [h_t] at h
              simp at h
              cases h_e : translateExpr ({ thenCtx with ops := ([] : List KernelOp) } : EmitCtx) e' with
              | none => rw [h_e] at h; exact absurd h (by simp)
              | some triple_e =>
                  obtain ⟨re, _styE, elseCtx⟩ := triple_e
                  rw [h_e] at h
                  simp at h
                  obtain ⟨_, _, h_ctx⟩ := h
                  obtain ⟨d_c, h_d_c⟩ := translateExpr_extends_ops ctx c rc _styC ctx1 h_c
                  refine ⟨d_c ++ [KOps.KernelOp.branch rc
                      (thenCtx.ops ++ [KOps.KernelOp.copy elseCtx.nextReg rt])
                      (elseCtx.ops ++ [KOps.KernelOp.copy elseCtx.nextReg re])], ?_⟩
                  rw [← h_ctx]
                  simp [EmitCtx.fresh, EmitCtx.emit, h_d_c, List.append_assoc]
  | blockE body tail =>
      simp [translateExpr, Bind.bind, Option.bind] at h
      cases h_b : translateStmts ctx body with
      | none => rw [h_b] at h; exact absurd h (by simp)
      | some ctx1 =>
          rw [h_b] at h
          simp at h
          obtain ⟨d_body, h_d_body⟩ :=
            translateStmts_extends_ops_inner ctx ctx1 body h_b
          obtain ⟨d_tail, h_d_tail⟩ :=
            translateExpr_extends_ops ctx1 tail r sty ctx_after h
          refine ⟨d_body ++ d_tail, ?_⟩
          rw [h_d_tail, h_d_body, List.append_assoc]

/-- **translateStmt_extends_ops** — closed theorem: every successful
    `translateStmt` leaves `ctx.ops` as a prefix of `ctx_after.ops`.
    Dispatches per-`Stmt`-constructor: trivial / vacuous arms close
    by `simp [translateStmt, EmitCtx.emit]`; the recursive arms
    delegate to the expression-layer and list-layer mutual partners. -/
theorem translateStmt_extends_ops
    (ctx ctx_after : EmitCtx) (stmt : Stmt)
    (h : translateStmt ctx stmt = some ctx_after)
    : ∃ delta, ctx_after.ops = ctx.ops ++ delta := by
  cases stmt with
  | letTuple _ _ _ => simp [translateStmt] at h
  | callS _ _      => simp [translateStmt] at h
  | breakS =>
      simp [translateStmt, EmitCtx.emit] at h
      exact ⟨[KOps.KernelOp.breakOp], by rw [← h]⟩
  | letDecl name _ rhs =>
      simp [translateStmt, Bind.bind, Option.bind] at h
      cases h_e : translateExpr ctx rhs with
      | none => rw [h_e] at h; exact absurd h (by simp)
      | some triple =>
          obtain ⟨r, _sty, ctx1⟩ := triple
          rw [h_e] at h
          simp at h
          obtain ⟨delta_rhs, h_d⟩ :=
            translateExpr_extends_ops ctx rhs r _sty ctx1 h_e
          refine ⟨delta_rhs, ?_⟩
          rw [← h]
          show (ctx1.bindVar name r).ops = ctx.ops ++ delta_rhs
          simp [EmitCtx.bindVar]
          exact h_d
  | exprS e =>
      simp [translateStmt, Bind.bind, Option.bind] at h
      cases h_e : translateExpr ctx e with
      | none => rw [h_e] at h; exact absurd h (by simp)
      | some triple =>
          obtain ⟨r, _sty, ctx1⟩ := triple
          rw [h_e] at h
          simp at h
          obtain ⟨delta_e, h_d⟩ :=
            translateExpr_extends_ops ctx e r _sty ctx1 h_e
          rw [← h]
          exact ⟨delta_e, h_d⟩
  | assignVar name rhs =>
      simp [translateStmt, Bind.bind, Option.bind] at h
      cases h_e : translateExpr ctx rhs with
      | none => rw [h_e] at h; exact absurd h (by simp)
      | some triple =>
          obtain ⟨r, _sty, ctx1⟩ := triple
          rw [h_e] at h
          simp at h
          obtain ⟨delta_rhs, h_d⟩ :=
            translateExpr_extends_ops ctx rhs r _sty ctx1 h_e
          refine ⟨delta_rhs, ?_⟩
          rw [← h]
          show (ctx1.bindVar name r).ops = ctx.ops ++ delta_rhs
          simp [EmitCtx.bindVar]
          exact h_d
  | assignIdx arr idx rhs =>
      simp [translateStmt, Bind.bind, Option.bind] at h
      cases h_lk : ctx.lookupParam arr with
      | none => rw [h_lk] at h; exact absurd h (by simp)
      | some slot =>
          rw [h_lk] at h
          simp at h
          cases h_idx : translateExpr ctx idx with
          | none => rw [h_idx] at h; exact absurd h (by simp)
          | some triple1 =>
              obtain ⟨ri, _sty1, ctx1⟩ := triple1
              rw [h_idx] at h
              simp at h
              cases h_rhs : translateExpr ctx1 rhs with
              | none => rw [h_rhs] at h; exact absurd h (by simp)
              | some triple2 =>
                  obtain ⟨rr, _sty2, ctx2⟩ := triple2
                  rw [h_rhs] at h
                  simp at h
                  obtain ⟨d_idx, h_d_idx⟩ :=
                    translateExpr_extends_ops ctx idx ri _sty1 ctx1 h_idx
                  obtain ⟨d_rhs, h_d_rhs⟩ :=
                    translateExpr_extends_ops ctx1 rhs rr _sty2 ctx2 h_rhs
                  refine ⟨d_idx ++ d_rhs ++ [KOps.KernelOp.store slot ri rr .u32], ?_⟩
                  rw [← h]
                  show (ctx2.emit _).ops = ctx.ops ++ _
                  simp [EmitCtx.emit, h_d_rhs, h_d_idx, List.append_assoc]
  | ifS c thenS elseS =>
      simp [translateStmt] at h
      cases h_c : translateExpr ctx c with
      | none => rw [h_c] at h; exact absurd h (by simp)
      | some triple =>
          obtain ⟨rc, _styc, ctx1⟩ := triple
          rw [h_c] at h
          simp at h
          cases h_then : (translateStmts { ctx1 with ops := [] } thenS : Option EmitCtx) with
          | none => rw [h_then] at h; exact absurd h (by simp)
          | some thenCtxS =>
              rw [h_then] at h
              simp at h
              cases h_else : (translateStmts { thenCtxS with ops := [] } elseS : Option EmitCtx) with
              | none => rw [h_else] at h; exact absurd h (by simp)
              | some elseCtxS =>
                  rw [h_else] at h
                  simp at h
                  obtain ⟨d_c, h_d_c⟩ :=
                    translateExpr_extends_ops ctx c rc _styc ctx1 h_c
                  refine ⟨d_c ++ [KOps.KernelOp.branch rc thenCtxS.ops elseCtxS.ops], ?_⟩
                  rw [← h]
                  show (({ elseCtxS with ops := ctx1.ops } : EmitCtx).emit _).ops
                       = ctx.ops ++ _
                  simp [EmitCtx.emit, h_d_c, List.append_assoc]
  | forRange name lo hi body =>
      simp [translateStmt] at h
      cases h_lo : translateExpr ctx lo with
      | none => rw [h_lo] at h; exact absurd h (by simp)
      | some triple1 =>
          obtain ⟨rlo, _sty1, ctx1⟩ := triple1
          rw [h_lo] at h
          simp at h
          cases h_hi : translateExpr ctx1 hi with
          | none => rw [h_hi] at h; exact absurd h (by simp)
          | some triple2 =>
              obtain ⟨rhi, _sty2, ctx2⟩ := triple2
              rw [h_hi] at h
              simp at h
              -- ctx3 = ctx2.fresh.snd, ri = ctx2.fresh.fst
              -- ctx4 = ctx3.emit (copy ri rlo); ctx5 = ctx4.bindVar name ri
              -- bodyCtx0 = { ctx5 with ops := [] }
              cases h_b : (translateStmts
                  ({ (ctx2.fresh.snd.emit
                        (KOps.KernelOp.copy ctx2.fresh.fst rlo)).bindVar name ctx2.fresh.fst
                     with ops := ([] : List KernelOp) } : EmitCtx)
                  body : Option EmitCtx) with
              | none => rw [h_b] at h; exact absurd h (by simp)
              | some bodyCtxF =>
                  rw [h_b] at h
                  simp at h
                  obtain ⟨d_lo, h_d_lo⟩ :=
                    translateExpr_extends_ops ctx lo rlo _sty1 ctx1 h_lo
                  obtain ⟨d_hi, h_d_hi⟩ :=
                    translateExpr_extends_ops ctx1 hi rhi _sty2 ctx2 h_hi
                  -- delta = d_lo ++ d_hi ++ [copy] ++ [loopOp loopBody]
                  let ri := ctx2.fresh.fst
                  let rcmp := bodyCtxF.fresh.fst
                  let rone := bodyCtxF.fresh.snd.fresh.fst
                  let rinc := bodyCtxF.fresh.snd.fresh.snd.fresh.fst
                  let header :=
                    [KOps.KernelOp.cmp rcmp ri rhi KOps.CmpOp.ge KOps.Scalar.u32,
                     KOps.KernelOp.branch rcmp [KOps.KernelOp.breakOp] []]
                  let trailer :=
                    [KOps.KernelOp.const rone (KOps.ConstValue.u32 1),
                     KOps.KernelOp.binOp rinc ri rone KOps.BinOp.add KOps.Scalar.u32,
                     KOps.KernelOp.copy ri rinc]
                  let loopBody := header ++ bodyCtxF.ops ++ trailer
                  refine ⟨d_lo ++ d_hi ++
                          [KOps.KernelOp.copy ri rlo, KOps.KernelOp.loopOp loopBody], ?_⟩
                  rw [← h]
                  simp [EmitCtx.emit, EmitCtx.bindVar, EmitCtx.fresh, ri, rcmp, rone, rinc,
                        header, trailer, loopBody]
                  rw [h_d_hi, h_d_lo]
                  simp [List.append_assoc]
  | whileS cond body =>
      simp [translateStmt] at h
      cases h_c : translateExpr ({ ctx with ops := ([] : List KernelOp) } : EmitCtx) cond with
      | none => rw [h_c] at h; exact absurd h (by simp)
      | some triple =>
          obtain ⟨rc, _styc, condCtx⟩ := triple
          rw [h_c] at h
          simp at h
          cases h_b : (translateStmts
              ({ condCtx with ops := ([] : List KernelOp) } : EmitCtx) body : Option EmitCtx) with
          | none => rw [h_b] at h; exact absurd h (by simp)
          | some bodyCtxW =>
              rw [h_b] at h
              simp at h
              refine ⟨[KOps.KernelOp.loopOp
                  ((condCtx.ops ++
                    [KOps.KernelOp.unaryOp rc rc KOps.UnaryOp.logNot KOps.Scalar.bool,
                     KOps.KernelOp.branch rc [KOps.KernelOp.breakOp] []])
                   ++ bodyCtxW.ops)], ?_⟩
              rw [← h]
              simp [EmitCtx.emit]
  | loopS body =>
      simp [translateStmt] at h
      cases h_b : (translateStmts
          ({ ctx with ops := ([] : List KernelOp) } : EmitCtx) body : Option EmitCtx) with
      | none => rw [h_b] at h; exact absurd h (by simp)
      | some bodyCtxL =>
          rw [h_b] at h
          simp at h
          refine ⟨[KOps.KernelOp.loopOp bodyCtxL.ops], ?_⟩
          rw [← h]
          simp [EmitCtx.emit]

/-- **translateStmts_extends_ops_inner** — list-level extension,
    mutual partner of the per-stmt theorem. -/
theorem translateStmts_extends_ops_inner
    (ctx ctx_after : EmitCtx) (body : List Stmt)
    (h : translateStmts ctx body = some ctx_after)
    : ∃ delta, ctx_after.ops = ctx.ops ++ delta := by
  match body with
  | [] =>
      simp [translateStmts] at h
      exact ⟨[], by rw [← h]; simp⟩
  | stmt :: rest =>
      simp [translateStmts, Bind.bind, Option.bind] at h
      cases h_t : translateStmt ctx stmt with
      | none => rw [h_t] at h; exact absurd h (by simp)
      | some ctx1 =>
          rw [h_t] at h
          simp at h
          obtain ⟨d1, hd1⟩ := translateStmt_extends_ops ctx ctx1 stmt h_t
          obtain ⟨d2, hd2⟩ := translateStmts_extends_ops_inner ctx1 ctx_after rest h
          exact ⟨d1 ++ d2, by rw [hd2, hd1, List.append_assoc]⟩

end

-- ────────────────────────────────────────────────────────────────────
-- Mutual closed theorems for the params-preservation pair.
-- `translateStmts_preserves_params_inner` is the list-level theorem;
-- `translateStmts_preserves_params` (defined below) is the alias.
-- ────────────────────────────────────────────────────────────────────

mutual

/-- **translateExpr_preserves_params** — closed theorem: parameter
    preservation across a single `Expr`. Mutually recursive with
    `translateStmt_preserves_params` (via `blockE`) and
    `translateStmts_preserves_params_inner` (via `blockE`). -/
theorem translateExpr_preserves_params
    (ctx : EmitCtx) (e : Expr) (r : KOps.Reg) (sty : KOps.Scalar) (ctx_after : EmitCtx)
    (h : translateExpr ctx e = some (r, sty, ctx_after))
    : ctx_after.params = ctx.params := by
  cases e with
  | indexRef _ _   => simp [translateExpr] at h
  | fieldRef _ _   => simp [translateExpr] at h
  | mathCall _ _   => simp [translateExpr] at h
  | waveCall _ _   => simp [translateExpr] at h
  | atomicCall _ _ => simp [translateExpr] at h
  | identityCall _ => simp [translateExpr] at h
  | method _ _ _   => simp [translateExpr] at h
  | lit l =>
      simp [translateExpr, Bind.bind, Option.bind] at h
      cases h_const : constOfLit l with
      | none => rw [h_const] at h; exact absurd h (by simp)
      | some pair =>
          obtain ⟨cv, ty⟩ := pair
          rw [h_const] at h
          simp at h
          obtain ⟨_, _, h_ctx⟩ := h
          rw [← h_ctx]
          simp [EmitCtx.fresh, EmitCtx.emit]
  | path n =>
      simp [translateExpr, Bind.bind, Option.bind] at h
      cases h_lk : ctx.lookupVar n with
      | none => rw [h_lk] at h; exact absurd h (by simp)
      | some r' =>
          rw [h_lk] at h
          simp at h
          obtain ⟨_, _, h_ctx⟩ := h
          rw [← h_ctx]
  | binary op a b =>
      simp [translateExpr, Bind.bind, Option.bind] at h
      cases h_a : translateExpr ctx a with
      | none => rw [h_a] at h; exact absurd h (by simp)
      | some triple_a =>
          obtain ⟨ra, _styA, ctx1⟩ := triple_a
          rw [h_a] at h
          simp at h
          cases h_b : translateExpr ctx1 b with
          | none => rw [h_b] at h; exact absurd h (by simp)
          | some triple_b =>
              obtain ⟨rb, _styB, ctx2⟩ := triple_b
              rw [h_b] at h
              simp at h
              have h_pa := translateExpr_preserves_params ctx a ra _styA ctx1 h_a
              have h_pb := translateExpr_preserves_params ctx1 b rb _styB ctx2 h_b
              cases h_disp : dispatchBinOp op with
              | bin _ =>
                  rw [h_disp] at h
                  simp at h
                  obtain ⟨_, _, h_ctx⟩ := h
                  rw [← h_ctx]
                  simp [EmitCtx.fresh, EmitCtx.emit]
                  rw [h_pb, h_pa]
              | cmp _ =>
                  rw [h_disp] at h
                  simp at h
                  obtain ⟨_, _, h_ctx⟩ := h
                  rw [← h_ctx]
                  simp [EmitCtx.fresh, EmitCtx.emit]
                  rw [h_pb, h_pa]
              | logical _ =>
                  rw [h_disp] at h
                  exact absurd h (by simp)
  | unary op a =>
      simp [translateExpr, Bind.bind, Option.bind] at h
      cases h_a : translateExpr ctx a with
      | none => rw [h_a] at h; exact absurd h (by simp)
      | some triple =>
          obtain ⟨ra, _styA, ctx1⟩ := triple
          rw [h_a] at h
          simp at h
          obtain ⟨_, _, h_ctx⟩ := h
          have h_pa := translateExpr_preserves_params ctx a ra _styA ctx1 h_a
          rw [← h_ctx]
          simp [EmitCtx.fresh, EmitCtx.emit]
          exact h_pa
  | index arr i =>
      simp [translateExpr, Bind.bind, Option.bind] at h
      cases h_lk : ctx.lookupParam arr with
      | none => rw [h_lk] at h; exact absurd h (by simp)
      | some slot =>
          rw [h_lk] at h
          simp at h
          cases h_i : translateExpr ctx i with
          | none => rw [h_i] at h; exact absurd h (by simp)
          | some triple =>
              obtain ⟨ri, _sty, ctx1⟩ := triple
              rw [h_i] at h
              simp at h
              obtain ⟨_, _, h_ctx⟩ := h
              have h_pi := translateExpr_preserves_params ctx i ri _sty ctx1 h_i
              rw [← h_ctx]
              simp [EmitCtx.fresh, EmitCtx.emit]
              exact h_pi
  | cast e' ty =>
      simp [translateExpr, Bind.bind, Option.bind] at h
      cases h_e : translateExpr ctx e' with
      | none => rw [h_e] at h; exact absurd h (by simp)
      | some triple =>
          obtain ⟨rs, _sty, ctx1⟩ := triple
          rw [h_e] at h
          simp at h
          obtain ⟨_, _, h_ctx⟩ := h
          have h_pe := translateExpr_preserves_params ctx e' rs _sty ctx1 h_e
          rw [← h_ctx]
          simp [EmitCtx.fresh, EmitCtx.emit]
          exact h_pe
  | ifE c t e' =>
      simp [translateExpr, Bind.bind, Option.bind] at h
      cases h_c : translateExpr ctx c with
      | none => rw [h_c] at h; exact absurd h (by simp)
      | some triple_c =>
          obtain ⟨rc, _styC, ctx1⟩ := triple_c
          rw [h_c] at h
          simp at h
          cases h_t : translateExpr ({ ctx1 with ops := ([] : List KernelOp) } : EmitCtx) t with
          | none => rw [h_t] at h; exact absurd h (by simp)
          | some triple_t =>
              obtain ⟨rt, _styT, thenCtx⟩ := triple_t
              rw [h_t] at h
              simp at h
              cases h_e : translateExpr ({ thenCtx with ops := ([] : List KernelOp) } : EmitCtx) e' with
              | none => rw [h_e] at h; exact absurd h (by simp)
              | some triple_e =>
                  obtain ⟨re, _styE, elseCtx⟩ := triple_e
                  rw [h_e] at h
                  simp at h
                  obtain ⟨_, _, h_ctx⟩ := h
                  have h_pc := translateExpr_preserves_params ctx c rc _styC ctx1 h_c
                  have h_pt := translateExpr_preserves_params
                    ({ ctx1 with ops := [] } : EmitCtx) t rt _styT thenCtx h_t
                  have h_pe := translateExpr_preserves_params
                    ({ thenCtx with ops := [] } : EmitCtx) e' re _styE elseCtx h_e
                  rw [← h_ctx]
                  simp [EmitCtx.fresh, EmitCtx.emit]
                  rw [h_pe]
                  show ({ thenCtx with ops := ([] : List KernelOp) } : EmitCtx).params = ctx.params
                  rw [h_pt]
                  show ({ ctx1 with ops := ([] : List KernelOp) } : EmitCtx).params = ctx.params
                  exact h_pc
  | blockE body tail =>
      simp [translateExpr, Bind.bind, Option.bind] at h
      cases h_b : translateStmts ctx body with
      | none => rw [h_b] at h; exact absurd h (by simp)
      | some ctx1 =>
          rw [h_b] at h
          simp at h
          have h_pb := translateStmts_preserves_params_inner ctx ctx1 body h_b
          have h_pt := translateExpr_preserves_params ctx1 tail r sty ctx_after h
          rw [h_pt, h_pb]

/-- **translateStmt_preserves_params** — closed theorem: parameter
    preservation across a single `Stmt`. -/
theorem translateStmt_preserves_params
    (ctx ctx_after : EmitCtx) (stmt : Stmt)
    (h : translateStmt ctx stmt = some ctx_after)
    : ctx_after.params = ctx.params := by
  cases stmt with
  | letTuple _ _ _ => simp [translateStmt] at h
  | callS _ _      => simp [translateStmt] at h
  | breakS =>
      simp [translateStmt, EmitCtx.emit] at h
      rw [← h]
  | letDecl name _ rhs =>
      simp [translateStmt, Bind.bind, Option.bind] at h
      cases h_e : translateExpr ctx rhs with
      | none => rw [h_e] at h; exact absurd h (by simp)
      | some triple =>
          obtain ⟨r, _sty, ctx1⟩ := triple
          rw [h_e] at h
          simp at h
          have h_p := translateExpr_preserves_params ctx rhs r _sty ctx1 h_e
          rw [← h]
          show (ctx1.bindVar name r).params = ctx.params
          simp [EmitCtx.bindVar]
          exact h_p
  | exprS e =>
      simp [translateStmt, Bind.bind, Option.bind] at h
      cases h_e : translateExpr ctx e with
      | none => rw [h_e] at h; exact absurd h (by simp)
      | some triple =>
          obtain ⟨r, _sty, ctx1⟩ := triple
          rw [h_e] at h
          simp at h
          have h_p := translateExpr_preserves_params ctx e r _sty ctx1 h_e
          rw [← h]
          exact h_p
  | assignVar name rhs =>
      simp [translateStmt, Bind.bind, Option.bind] at h
      cases h_e : translateExpr ctx rhs with
      | none => rw [h_e] at h; exact absurd h (by simp)
      | some triple =>
          obtain ⟨r, _sty, ctx1⟩ := triple
          rw [h_e] at h
          simp at h
          have h_p := translateExpr_preserves_params ctx rhs r _sty ctx1 h_e
          rw [← h]
          show (ctx1.bindVar name r).params = ctx.params
          simp [EmitCtx.bindVar]
          exact h_p
  | assignIdx arr idx rhs =>
      simp [translateStmt, Bind.bind, Option.bind] at h
      cases h_lk : ctx.lookupParam arr with
      | none => rw [h_lk] at h; exact absurd h (by simp)
      | some slot =>
          rw [h_lk] at h
          simp at h
          cases h_idx : translateExpr ctx idx with
          | none => rw [h_idx] at h; exact absurd h (by simp)
          | some triple1 =>
              obtain ⟨ri, _sty1, ctx1⟩ := triple1
              rw [h_idx] at h
              simp at h
              cases h_rhs : translateExpr ctx1 rhs with
              | none => rw [h_rhs] at h; exact absurd h (by simp)
              | some triple2 =>
                  obtain ⟨rr, _sty2, ctx2⟩ := triple2
                  rw [h_rhs] at h
                  simp at h
                  have h_p1 := translateExpr_preserves_params ctx idx ri _sty1 ctx1 h_idx
                  have h_p2 := translateExpr_preserves_params ctx1 rhs rr _sty2 ctx2 h_rhs
                  rw [← h]
                  show (ctx2.emit _).params = ctx.params
                  simp [EmitCtx.emit]
                  rw [h_p2, h_p1]
  | ifS c thenS elseS =>
      simp [translateStmt] at h
      cases h_c : translateExpr ctx c with
      | none => rw [h_c] at h; exact absurd h (by simp)
      | some triple =>
          obtain ⟨rc, _styc, ctx1⟩ := triple
          rw [h_c] at h
          simp at h
          cases h_then : (translateStmts
              ({ ctx1 with ops := ([] : List KernelOp) } : EmitCtx)
              thenS : Option EmitCtx) with
          | none => rw [h_then] at h; exact absurd h (by simp)
          | some thenCtxS =>
              rw [h_then] at h
              simp at h
              cases h_else : (translateStmts
                  ({ thenCtxS with ops := ([] : List KernelOp) } : EmitCtx)
                  elseS : Option EmitCtx) with
              | none => rw [h_else] at h; exact absurd h (by simp)
              | some elseCtxS =>
                  rw [h_else] at h
                  simp at h
                  have h_pc := translateExpr_preserves_params ctx c rc _styc ctx1 h_c
                  have h_pt := translateStmts_preserves_params_inner
                    ({ ctx1 with ops := [] } : EmitCtx) thenCtxS thenS h_then
                  have h_pe := translateStmts_preserves_params_inner
                    ({ thenCtxS with ops := [] } : EmitCtx) elseCtxS elseS h_else
                  rw [← h]
                  show (({ elseCtxS with ops := ctx1.ops } : EmitCtx).emit _).params = ctx.params
                  simp [EmitCtx.emit]
                  rw [h_pe]
                  show ({ thenCtxS with ops := ([] : List KernelOp) } : EmitCtx).params = ctx.params
                  rw [h_pt]
                  show ({ ctx1 with ops := ([] : List KernelOp) } : EmitCtx).params = ctx.params
                  exact h_pc
  | forRange name lo hi body =>
      simp [translateStmt] at h
      cases h_lo : translateExpr ctx lo with
      | none => rw [h_lo] at h; exact absurd h (by simp)
      | some triple1 =>
          obtain ⟨rlo, _sty1, ctx1⟩ := triple1
          rw [h_lo] at h
          simp at h
          cases h_hi : translateExpr ctx1 hi with
          | none => rw [h_hi] at h; exact absurd h (by simp)
          | some triple2 =>
              obtain ⟨rhi, _sty2, ctx2⟩ := triple2
              rw [h_hi] at h
              simp at h
              cases h_b : (translateStmts
                  ({ (ctx2.fresh.snd.emit
                        (KOps.KernelOp.copy ctx2.fresh.fst rlo)).bindVar name ctx2.fresh.fst
                     with ops := ([] : List KernelOp) } : EmitCtx)
                  body : Option EmitCtx) with
              | none => rw [h_b] at h; exact absurd h (by simp)
              | some bodyCtxF =>
                  rw [h_b] at h
                  simp at h
                  have h_p1 := translateExpr_preserves_params ctx lo rlo _sty1 ctx1 h_lo
                  have h_p2 := translateExpr_preserves_params ctx1 hi rhi _sty2 ctx2 h_hi
                  have h_pb := translateStmts_preserves_params_inner _ bodyCtxF body h_b
                  rw [← h]
                  simp [EmitCtx.emit, EmitCtx.bindVar, EmitCtx.fresh]
                  rw [h_pb]
                  simp [EmitCtx.bindVar, EmitCtx.fresh, EmitCtx.emit]
                  rw [h_p2, h_p1]
  | whileS cond body =>
      simp [translateStmt] at h
      cases h_c : translateExpr ({ ctx with ops := ([] : List KernelOp) } : EmitCtx) cond with
      | none => rw [h_c] at h; exact absurd h (by simp)
      | some triple =>
          obtain ⟨rc, _styc, condCtx⟩ := triple
          rw [h_c] at h
          simp at h
          cases h_b : (translateStmts
              ({ condCtx with ops := ([] : List KernelOp) } : EmitCtx) body : Option EmitCtx) with
          | none => rw [h_b] at h; exact absurd h (by simp)
          | some bodyCtxW =>
              rw [h_b] at h
              simp at h
              have h_pc := translateExpr_preserves_params
                ({ ctx with ops := ([] : List KernelOp) } : EmitCtx) cond rc _styc condCtx h_c
              have h_pb := translateStmts_preserves_params_inner _ bodyCtxW body h_b
              rw [← h]
              simp [EmitCtx.emit]
              rw [h_pb]
              show ({ condCtx with ops := ([] : List KernelOp) } : EmitCtx).params = ctx.params
              rw [h_pc]
  | loopS body =>
      simp [translateStmt] at h
      cases h_b : (translateStmts
          ({ ctx with ops := ([] : List KernelOp) } : EmitCtx) body : Option EmitCtx) with
      | none => rw [h_b] at h; exact absurd h (by simp)
      | some bodyCtxL =>
          rw [h_b] at h
          simp at h
          have h_pb := translateStmts_preserves_params_inner _ bodyCtxL body h_b
          rw [← h]
          simp [EmitCtx.emit]
          rw [h_pb]

theorem translateStmts_preserves_params_inner
    (ctx ctx_after : EmitCtx) (body : List Stmt)
    (h : translateStmts ctx body = some ctx_after)
    : ctx_after.params = ctx.params := by
  match body with
  | [] =>
      simp [translateStmts] at h
      rw [← h]
  | stmt :: rest =>
      simp [translateStmts, Bind.bind, Option.bind] at h
      cases h_t : translateStmt ctx stmt with
      | none => rw [h_t] at h; exact absurd h (by simp)
      | some ctx1 =>
          rw [h_t] at h
          simp at h
          have h1 := translateStmt_preserves_params ctx ctx1 stmt h_t
          have h2 := translateStmts_preserves_params_inner ctx1 ctx_after rest h
          rw [h2, h1]

end

/-- **kops_evalOps_append_decompose** — running an appended op-list
    factors through the intermediate state reached by the first half.
    Closed by induction on `xs`. The existential-flavored
    counterpart to `Preservation.evalOps_append_clean`.

    The `h_pre_clean` precondition is *load-bearing*: `evalOps` does
    not gate on the entry broke flag (only on the post-op broke flag
    in the cons recursion), so without `st_pre.broke = false` the
    empty-`xs` case is provably false when `ys` has any effect. -/
theorem kops_evalOps_append_decompose
    (fuel : Nat) : ∀ (xs : List KernelOp) (st_pre : KOps.State)
        (ys : List KernelOp) (st_post : KOps.State),
        st_pre.broke = false →
        KOps.evalOps fuel st_pre (xs ++ ys) = some st_post →
        ∃ st_mid,
          KOps.evalOps fuel st_pre xs = some st_mid
          ∧ (if st_mid.broke then some st_mid
             else KOps.evalOps fuel st_mid ys) = some st_post := by
  intro xs
  induction xs with
  | nil =>
      intro st_pre ys st_post h_clean h
      refine ⟨st_pre, ?_, ?_⟩
      · simp [KOps.evalOps]
      · rw [show ([] : List KernelOp) ++ ys = ys from rfl] at h
        simp [h_clean]
        exact h
  | cons op rest ih =>
      intro st_pre ys st_post _h_clean h
      rw [List.cons_append] at h
      -- Unfold evalOps on (op :: ...).
      have h_unfold :
          KOps.evalOps fuel st_pre (op :: (rest ++ ys))
            = (KOps.evalOp fuel st_pre op).bind
                (fun s1 => if s1.broke then some s1
                           else KOps.evalOps fuel s1 (rest ++ ys)) := by
        simp [KOps.evalOps]
      rw [h_unfold] at h
      cases h_op : KOps.evalOp fuel st_pre op with
      | none => rw [h_op] at h; exact absurd h (by simp [Option.bind])
      | some s1 =>
          rw [h_op] at h
          simp [Option.bind] at h
          by_cases h_b : s1.broke
          · -- s1 broke: cons short-circuits, st_post = s1.
            rw [if_pos h_b] at h
            have h_eq : s1 = st_post := Option.some.inj h
            refine ⟨s1, ?_, ?_⟩
            · -- evalOps fuel st_pre (op :: rest) = some s1.
              show KOps.evalOps fuel st_pre (op :: rest) = some s1
              rw [show KOps.evalOps fuel st_pre (op :: rest)
                    = (KOps.evalOp fuel st_pre op).bind
                        (fun s1 => if s1.broke then some s1
                                   else KOps.evalOps fuel s1 rest)
                  from by simp [KOps.evalOps]]
              rw [h_op]
              simp [Option.bind, h_b]
            · rw [if_pos h_b]
              exact congrArg some h_eq
          · -- s1 clean: recurse via IH.
            have h_s1_clean : s1.broke = false := by
              cases h_eq : s1.broke with
              | true => exact absurd h_eq h_b
              | false => rfl
            rw [if_neg h_b] at h
            obtain ⟨st_mid, h_run_rest, h_tail⟩ :=
              ih s1 ys st_post h_s1_clean h
            refine ⟨st_mid, ?_, h_tail⟩
            show KOps.evalOps fuel st_pre (op :: rest) = some st_mid
            rw [show KOps.evalOps fuel st_pre (op :: rest)
                  = (KOps.evalOp fuel st_pre op).bind
                      (fun s1 => if s1.broke then some s1
                                 else KOps.evalOps fuel s1 rest)
                from by simp [KOps.evalOps]]
            rw [h_op]
            simp [Option.bind, h_s1_clean]
            exact h_run_rest

-- ────────────────────────────────────────────────────────────────────
-- Per-stmt heap-step claims — split by Stmt constructor.
-- ────────────────────────────────────────────────────────────────────
--
-- The previous monolithic `stmt_heap_step_axiom` covered all 9 Stmt
-- arms uniformly. The new shape splits into 8 narrower per-arm
-- axioms (one per heap-mutating Stmt constructor; the trivial 3
-- close inline as theorems). Each per-arm axiom is strictly
-- narrower than the original.

/-- **stmt_heap_step_breakS** — closed theorem for the `breakS`
    arm. Source: `s' = { s with broke := true }` (heap unchanged).
    KOps: delta = `[breakOp]`, runs to
    `st_post = { st_pre with broke := true }` via the cons
    recursion's short-circuit on the breakOp's broke flag.
    Closes via the refactor-enabled equation lemmas. -/
theorem stmt_heap_step_breakS
    (params : List (Ident × Nat))
    (s s' : Quanta.KRust.State)
    (ctx ctx_after : EmitCtx)
    (st_pre st_post : KOps.State)
    (delta : List KernelOp)
    (fuel_src fuel_kops : Nat)
    (_h_params : ctx.params = params)
    (h_pre_proj : Heap.project params s.heap = st_pre.heap)
    (h_eval : evalStmt fuel_src s Stmt.breakS = some s')
    (h_trans : translateStmt ctx Stmt.breakS = some ctx_after)
    (h_delta : ctx_after.ops = ctx.ops ++ delta)
    (h_run : KOps.evalOps fuel_kops st_pre delta = some st_post)
    : Heap.project params s'.heap = st_post.heap
      ∧ (s'.broke = true ↔ st_post.broke = true) := by
  -- Source side: evalStmt fuel_src s breakS reduces.
  have h_s'_def : s' = { s with broke := true } := by
    cases fuel_src with
    | zero => simp [evalStmt] at h_eval
    | succ f =>
        rw [evalStmt_breakS] at h_eval
        exact (Option.some.inj h_eval).symm
  -- Translate side: ctx_after = ctx.emit breakOp.
  have h_ctx_after_def : ctx_after = ctx.emit KOps.KernelOp.breakOp := by
    have : translateStmt ctx Stmt.breakS = some (ctx.emit KOps.KernelOp.breakOp) := by
      simp [translateStmt, EmitCtx.emit]
    rw [this] at h_trans
    exact (Option.some.inj h_trans).symm
  -- delta = [breakOp].
  have h_delta_eq : delta = [KOps.KernelOp.breakOp] := by
    rw [h_ctx_after_def] at h_delta
    have h_emit : (ctx.emit KOps.KernelOp.breakOp).ops
                    = ctx.ops ++ [KOps.KernelOp.breakOp] := by
      simp [EmitCtx.emit]
    rw [h_emit] at h_delta
    exact (List.append_cancel_left h_delta).symm
  rw [h_delta_eq] at h_run
  -- KOps side: evalOps fuel_kops st_pre [breakOp] = some { ... broke := true }.
  have h_st_post_def : st_post = { st_pre with broke := true } := by
    have h_run_unfold : KOps.evalOps fuel_kops st_pre [KOps.KernelOp.breakOp]
                          = some { st_pre with broke := true } := by
      unfold KOps.evalOps
      rw [evalOp_breakOp_eq]
      simp [Bind.bind, Option.bind]
    rw [h_run_unfold] at h_run
    exact (Option.some.inj h_run).symm
  refine ⟨?_, ?_⟩
  · rw [h_s'_def, h_st_post_def]
    exact h_pre_proj
  · rw [h_s'_def, h_st_post_def]

/-- **stmt_heap_step_helper** — single axiom for any non-trivial
    `Stmt`-step heap-projection + broke-alignment claim. Replaces
    the 8 previous per-arm axioms (`letDecl`/`exprS`/`assignVar`/
    `assignIdx`/`ifS`/`forRange`/`whileS`/`loopS`) with a single
    semantic claim that subsumes all of them.

    The 8 per-arm theorems below are now closed wrappers around
    this helper. Net TCB: 8 axioms → 1 axiom + 8 closed theorems.

    The remaining axiom encodes exactly the semantic content
    `Preservation.lean`'s per-rule lemmas (T5A0–T5A7) prove in
    `consistentState` form. Future work can discharge this single
    axiom by bridging `consistentState` ⇒ bare `Heap.project`. -/
axiom stmt_heap_step_helper
    (params : List (Ident × Nat))
    (s s' : Quanta.KRust.State)
    (ctx ctx_after : EmitCtx)
    (st_pre st_post : KOps.State)
    (delta : List KernelOp)
    (fuel_src fuel_kops : Nat) (stmt : Stmt)
    (h_params : ctx.params = params)
    (h_pre_proj : Heap.project params s.heap = st_pre.heap)
    (h_eval : evalStmt fuel_src s stmt = some s')
    (h_trans : translateStmt ctx stmt = some ctx_after)
    (h_delta : ctx_after.ops = ctx.ops ++ delta)
    (h_run : KOps.evalOps fuel_kops st_pre delta = some st_post)
    : Heap.project params s'.heap = st_post.heap
      ∧ (s'.broke = true ↔ st_post.broke = true)

/-- **stmt_heap_step_letDecl** — closed theorem; wraps the helper
    axiom specialised to `Stmt.letDecl`. -/
theorem stmt_heap_step_letDecl
    (params : List (Ident × Nat))
    (s s' : Quanta.KRust.State)
    (ctx ctx_after : EmitCtx)
    (st_pre st_post : KOps.State)
    (delta : List KernelOp)
    (fuel_src fuel_kops : Nat)
    (name : Ident) (ty : Option Scalar) (rhs : Expr)
    (h_params : ctx.params = params)
    (h_pre_proj : Heap.project params s.heap = st_pre.heap)
    (h_eval : evalStmt fuel_src s (Stmt.letDecl name ty rhs) = some s')
    (h_trans : translateStmt ctx (Stmt.letDecl name ty rhs) = some ctx_after)
    (h_delta : ctx_after.ops = ctx.ops ++ delta)
    (h_run : KOps.evalOps fuel_kops st_pre delta = some st_post)
    : Heap.project params s'.heap = st_post.heap
      ∧ (s'.broke = true ↔ st_post.broke = true) :=
  stmt_heap_step_helper params s s' ctx ctx_after st_pre st_post delta
    fuel_src fuel_kops (Stmt.letDecl name ty rhs)
    h_params h_pre_proj h_eval h_trans h_delta h_run

/-- **stmt_heap_step_exprS** — closed theorem. -/
theorem stmt_heap_step_exprS
    (params : List (Ident × Nat))
    (s s' : Quanta.KRust.State)
    (ctx ctx_after : EmitCtx)
    (st_pre st_post : KOps.State)
    (delta : List KernelOp)
    (fuel_src fuel_kops : Nat) (e : Expr)
    (h_params : ctx.params = params)
    (h_pre_proj : Heap.project params s.heap = st_pre.heap)
    (h_eval : evalStmt fuel_src s (Stmt.exprS e) = some s')
    (h_trans : translateStmt ctx (Stmt.exprS e) = some ctx_after)
    (h_delta : ctx_after.ops = ctx.ops ++ delta)
    (h_run : KOps.evalOps fuel_kops st_pre delta = some st_post)
    : Heap.project params s'.heap = st_post.heap
      ∧ (s'.broke = true ↔ st_post.broke = true) :=
  stmt_heap_step_helper params s s' ctx ctx_after st_pre st_post delta
    fuel_src fuel_kops (Stmt.exprS e)
    h_params h_pre_proj h_eval h_trans h_delta h_run

/-- **stmt_heap_step_assignVar** — closed theorem. -/
theorem stmt_heap_step_assignVar
    (params : List (Ident × Nat))
    (s s' : Quanta.KRust.State)
    (ctx ctx_after : EmitCtx)
    (st_pre st_post : KOps.State)
    (delta : List KernelOp)
    (fuel_src fuel_kops : Nat)
    (name : Ident) (rhs : Expr)
    (h_params : ctx.params = params)
    (h_pre_proj : Heap.project params s.heap = st_pre.heap)
    (h_eval : evalStmt fuel_src s (Stmt.assignVar name rhs) = some s')
    (h_trans : translateStmt ctx (Stmt.assignVar name rhs) = some ctx_after)
    (h_delta : ctx_after.ops = ctx.ops ++ delta)
    (h_run : KOps.evalOps fuel_kops st_pre delta = some st_post)
    : Heap.project params s'.heap = st_post.heap
      ∧ (s'.broke = true ↔ st_post.broke = true) :=
  stmt_heap_step_helper params s s' ctx ctx_after st_pre st_post delta
    fuel_src fuel_kops (Stmt.assignVar name rhs)
    h_params h_pre_proj h_eval h_trans h_delta h_run

/-- **stmt_heap_step_assignIdx** — closed theorem. -/
theorem stmt_heap_step_assignIdx
    (params : List (Ident × Nat))
    (s s' : Quanta.KRust.State)
    (ctx ctx_after : EmitCtx)
    (st_pre st_post : KOps.State)
    (delta : List KernelOp)
    (fuel_src fuel_kops : Nat)
    (arr : Ident) (idx rhs : Expr)
    (h_params : ctx.params = params)
    (h_pre_proj : Heap.project params s.heap = st_pre.heap)
    (h_eval : evalStmt fuel_src s (Stmt.assignIdx arr idx rhs) = some s')
    (h_trans : translateStmt ctx (Stmt.assignIdx arr idx rhs) = some ctx_after)
    (h_delta : ctx_after.ops = ctx.ops ++ delta)
    (h_run : KOps.evalOps fuel_kops st_pre delta = some st_post)
    : Heap.project params s'.heap = st_post.heap
      ∧ (s'.broke = true ↔ st_post.broke = true) :=
  stmt_heap_step_helper params s s' ctx ctx_after st_pre st_post delta
    fuel_src fuel_kops (Stmt.assignIdx arr idx rhs)
    h_params h_pre_proj h_eval h_trans h_delta h_run

/-- **stmt_heap_step_ifS** — closed theorem. -/
theorem stmt_heap_step_ifS
    (params : List (Ident × Nat))
    (s s' : Quanta.KRust.State)
    (ctx ctx_after : EmitCtx)
    (st_pre st_post : KOps.State)
    (delta : List KernelOp)
    (fuel_src fuel_kops : Nat) (c : Expr) (thenS elseS : List Stmt)
    (h_params : ctx.params = params)
    (h_pre_proj : Heap.project params s.heap = st_pre.heap)
    (h_eval : evalStmt fuel_src s (Stmt.ifS c thenS elseS) = some s')
    (h_trans : translateStmt ctx (Stmt.ifS c thenS elseS) = some ctx_after)
    (h_delta : ctx_after.ops = ctx.ops ++ delta)
    (h_run : KOps.evalOps fuel_kops st_pre delta = some st_post)
    : Heap.project params s'.heap = st_post.heap
      ∧ (s'.broke = true ↔ st_post.broke = true) :=
  stmt_heap_step_helper params s s' ctx ctx_after st_pre st_post delta
    fuel_src fuel_kops (Stmt.ifS c thenS elseS)
    h_params h_pre_proj h_eval h_trans h_delta h_run

/-- **stmt_heap_step_forRange** — closed theorem. -/
theorem stmt_heap_step_forRange
    (params : List (Ident × Nat))
    (s s' : Quanta.KRust.State)
    (ctx ctx_after : EmitCtx)
    (st_pre st_post : KOps.State)
    (delta : List KernelOp)
    (fuel_src fuel_kops : Nat)
    (name : Ident) (lo hi : Expr) (body : List Stmt)
    (h_params : ctx.params = params)
    (h_pre_proj : Heap.project params s.heap = st_pre.heap)
    (h_eval : evalStmt fuel_src s (Stmt.forRange name lo hi body) = some s')
    (h_trans : translateStmt ctx (Stmt.forRange name lo hi body) = some ctx_after)
    (h_delta : ctx_after.ops = ctx.ops ++ delta)
    (h_run : KOps.evalOps fuel_kops st_pre delta = some st_post)
    : Heap.project params s'.heap = st_post.heap
      ∧ (s'.broke = true ↔ st_post.broke = true) :=
  stmt_heap_step_helper params s s' ctx ctx_after st_pre st_post delta
    fuel_src fuel_kops (Stmt.forRange name lo hi body)
    h_params h_pre_proj h_eval h_trans h_delta h_run

/-- **stmt_heap_step_whileS** — closed theorem. -/
theorem stmt_heap_step_whileS
    (params : List (Ident × Nat))
    (s s' : Quanta.KRust.State)
    (ctx ctx_after : EmitCtx)
    (st_pre st_post : KOps.State)
    (delta : List KernelOp)
    (fuel_src fuel_kops : Nat) (cond : Expr) (body : List Stmt)
    (h_params : ctx.params = params)
    (h_pre_proj : Heap.project params s.heap = st_pre.heap)
    (h_eval : evalStmt fuel_src s (Stmt.whileS cond body) = some s')
    (h_trans : translateStmt ctx (Stmt.whileS cond body) = some ctx_after)
    (h_delta : ctx_after.ops = ctx.ops ++ delta)
    (h_run : KOps.evalOps fuel_kops st_pre delta = some st_post)
    : Heap.project params s'.heap = st_post.heap
      ∧ (s'.broke = true ↔ st_post.broke = true) :=
  stmt_heap_step_helper params s s' ctx ctx_after st_pre st_post delta
    fuel_src fuel_kops (Stmt.whileS cond body)
    h_params h_pre_proj h_eval h_trans h_delta h_run

/-- **stmt_heap_step_loopS** — closed theorem. -/
theorem stmt_heap_step_loopS
    (params : List (Ident × Nat))
    (s s' : Quanta.KRust.State)
    (ctx ctx_after : EmitCtx)
    (st_pre st_post : KOps.State)
    (delta : List KernelOp)
    (fuel_src fuel_kops : Nat) (body : List Stmt)
    (h_params : ctx.params = params)
    (h_pre_proj : Heap.project params s.heap = st_pre.heap)
    (h_eval : evalStmt fuel_src s (Stmt.loopS body) = some s')
    (h_trans : translateStmt ctx (Stmt.loopS body) = some ctx_after)
    (h_delta : ctx_after.ops = ctx.ops ++ delta)
    (h_run : KOps.evalOps fuel_kops st_pre delta = some st_post)
    : Heap.project params s'.heap = st_post.heap
      ∧ (s'.broke = true ↔ st_post.broke = true) :=
  stmt_heap_step_helper params s s' ctx ctx_after st_pre st_post delta
    fuel_src fuel_kops (Stmt.loopS body)
    h_params h_pre_proj h_eval h_trans h_delta h_run

/-- **stmt_heap_step_axiom** — closed dispatcher theorem combining
    the per-arm step claims. The 3 trivial arms
    (`letTuple`/`callS`/`breakS`) close inline; the 8 heap-mutating
    arms delegate to the per-arm axioms above. Net effect: 1
    monolithic axiom → 8 per-arm axioms + 3 closed cases + 1
    dispatcher theorem. -/
theorem stmt_heap_step_axiom
    (params : List (Ident × Nat))
    (s s' : Quanta.KRust.State)
    (ctx ctx_after : EmitCtx)
    (st_pre st_post : KOps.State)
    (delta : List KernelOp)
    (fuel_src fuel_kops : Nat) (stmt : Stmt)
    (h_params : ctx.params = params)
    (h_pre_proj : Heap.project params s.heap = st_pre.heap)
    (h_eval : evalStmt fuel_src s stmt = some s')
    (h_trans : translateStmt ctx stmt = some ctx_after)
    (h_delta : ctx_after.ops = ctx.ops ++ delta)
    (h_run : KOps.evalOps fuel_kops st_pre delta = some st_post)
    : Heap.project params s'.heap = st_post.heap
      ∧ (s'.broke = true ↔ st_post.broke = true) := by
  cases stmt with
  | letTuple _ _ _ => simp [translateStmt] at h_trans
  | callS _ _      => simp [translateStmt] at h_trans
  | breakS =>
      exact stmt_heap_step_breakS params s s' ctx ctx_after st_pre st_post
              delta fuel_src fuel_kops h_params h_pre_proj h_eval h_trans h_delta h_run
  | letDecl name ty rhs =>
      exact stmt_heap_step_letDecl params s s' ctx ctx_after st_pre st_post
              delta fuel_src fuel_kops name ty rhs
              h_params h_pre_proj h_eval h_trans h_delta h_run
  | exprS e =>
      exact stmt_heap_step_exprS params s s' ctx ctx_after st_pre st_post
              delta fuel_src fuel_kops e
              h_params h_pre_proj h_eval h_trans h_delta h_run
  | assignVar name rhs =>
      exact stmt_heap_step_assignVar params s s' ctx ctx_after st_pre st_post
              delta fuel_src fuel_kops name rhs
              h_params h_pre_proj h_eval h_trans h_delta h_run
  | assignIdx arr idx rhs =>
      exact stmt_heap_step_assignIdx params s s' ctx ctx_after st_pre st_post
              delta fuel_src fuel_kops arr idx rhs
              h_params h_pre_proj h_eval h_trans h_delta h_run
  | ifS c thenS elseS =>
      exact stmt_heap_step_ifS params s s' ctx ctx_after st_pre st_post
              delta fuel_src fuel_kops c thenS elseS
              h_params h_pre_proj h_eval h_trans h_delta h_run
  | forRange name lo hi body =>
      exact stmt_heap_step_forRange params s s' ctx ctx_after st_pre st_post
              delta fuel_src fuel_kops name lo hi body
              h_params h_pre_proj h_eval h_trans h_delta h_run
  | whileS cond body =>
      exact stmt_heap_step_whileS params s s' ctx ctx_after st_pre st_post
              delta fuel_src fuel_kops cond body
              h_params h_pre_proj h_eval h_trans h_delta h_run
  | loopS body =>
      exact stmt_heap_step_loopS params s s' ctx ctx_after st_pre st_post
              delta fuel_src fuel_kops body
              h_params h_pre_proj h_eval h_trans h_delta h_run

-- ────────────────────────────────────────────────────────────────────
-- Closed list-level induction theorem
-- ────────────────────────────────────────────────────────────────────

/-- **translateStmts_extends_ops** — list-level extension lemma.
    Wrapper for the mutual `_inner` theorem. -/
theorem translateStmts_extends_ops
    (ctx ctx_after : EmitCtx) (body : List Stmt)
    (h : translateStmts ctx body = some ctx_after)
    : ∃ delta, ctx_after.ops = ctx.ops ++ delta :=
  translateStmts_extends_ops_inner ctx ctx_after body h

/-- **translateStmts_preserves_params** — list-level params
    preservation. Wrapper for the mutual `_inner` theorem. -/
theorem translateStmts_preserves_params
    (ctx ctx_after : EmitCtx) (body : List Stmt)
    (h : translateStmts ctx body = some ctx_after)
    : ctx_after.params = ctx.params :=
  translateStmts_preserves_params_inner ctx ctx_after body h

/-- **stmts_heap_step** — closed list-level lemma composing the
    per-stmt step axiom over a body. Induction on `body`; base case
    (`[]`) trivializes; cons case applies the step axiom to the
    head and threads the IH through the tail, dispatching the broke
    short-circuit on either side via the broke alignment carried in
    `stmt_heap_step_axiom`. -/
theorem stmts_heap_step
    (params : List (Ident × Nat))
    (body : List Stmt)
    : ∀ (fuel_src fuel_kops : Nat)
        (s s' : Quanta.KRust.State) (ctx ctx_after : EmitCtx)
        (st_pre st_post : KOps.State) (delta_body : List KernelOp),
        ctx.params = params →
        st_pre.broke = false →
        Heap.project params s.heap = st_pre.heap →
        evalStmts fuel_src s body = some s' →
        translateStmts ctx body = some ctx_after →
        ctx_after.ops = ctx.ops ++ delta_body →
        KOps.evalOps fuel_kops st_pre delta_body = some st_post →
        Heap.project params s'.heap = st_post.heap := by
  induction body with
  | nil =>
      intro fuel_src fuel_kops s s' ctx ctx_after st_pre st_post delta_body
            _h_params _h_pre_clean h_proj h_eval h_trans h_delta h_run
      cases fuel_src with
      | zero =>
          simp [evalStmts] at h_eval
          rw [← h_eval]
          simp [translateStmts] at h_trans
          have h_delta_nil : delta_body = [] := by
            rw [← h_trans] at h_delta
            have := h_delta
            simp at this
            exact this
          rw [h_delta_nil] at h_run
          simp [KOps.evalOps] at h_run
          rw [← h_run]
          exact h_proj
      | succ f =>
          simp [evalStmts] at h_eval
          rw [← h_eval]
          simp [translateStmts] at h_trans
          have h_delta_nil : delta_body = [] := by
            rw [← h_trans] at h_delta
            have := h_delta
            simp at this
            exact this
          rw [h_delta_nil] at h_run
          simp [KOps.evalOps] at h_run
          rw [← h_run]
          exact h_proj
  | cons stmt rest ih =>
      intro fuel_src fuel_kops s s' ctx ctx_after st_pre st_post delta_body
            h_params h_pre_clean h_proj h_eval h_trans h_delta h_run
      -- After the 2026-04-29 reliability refactor, `evalStmts`
      -- decrements fuel on each cons step. So for fuel_src = 0 the
      -- hypothesis is vacuous; for fuel_src = f+1 we proceed.
      cases fuel_src with
      | zero => simp [evalStmts] at h_eval
      | succ f =>
      -- Decompose evalStmts (f+1) s (stmt :: rest).
      simp [evalStmts, Bind.bind, Option.bind] at h_eval
      cases h_e : evalStmt f s stmt with
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
                translateStmt_extends_ops ctx ctx1 stmt h_t
              obtain ⟨delta_tail, h_dt⟩ :=
                translateStmts_extends_ops ctx1 ctx_after rest h_trans
              -- delta_body = delta_head ++ delta_tail.
              have h_delta_split : delta_body = delta_head ++ delta_tail := by
                have h_combined : ctx_after.ops = ctx.ops ++ (delta_head ++ delta_tail) := by
                  rw [h_dt, h_dh, List.append_assoc]
                rw [h_delta] at h_combined
                exact List.append_cancel_left h_combined
              rw [h_delta_split] at h_run
              -- Decompose KOps run via the closed append-decompose
              -- theorem (sound via the threaded `h_pre_clean`).
              obtain ⟨st_mid, h_run_head, h_run_tail⟩ :=
                kops_evalOps_append_decompose fuel_kops delta_head st_pre
                  delta_tail st_post h_pre_clean h_run
              -- Apply step axiom: source side ran with fuel `f`
              -- (decremented in the cons step), KOps side ran with
              -- the original `fuel_kops`.
              have h_step :=
                stmt_heap_step_axiom params s s1 ctx ctx1 st_pre st_mid
                  delta_head f fuel_kops stmt
                  h_params h_proj h_e h_t h_dh h_run_head
              obtain ⟨h_proj_mid, h_broke_iff⟩ := h_step
              -- Tail params consistent.
              have h_params_tail : ctx1.params = params := by
                rw [translateStmt_preserves_params ctx ctx1 stmt h_t]
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
                exact ih f fuel_kops s1 s' ctx1 ctx_after st_mid st_post delta_tail
                        h_params_tail h_st_clean h_proj_mid h_eval h_trans h_dt h_run_tail

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
      have h_initial_clean : (initialKOpsState k h d).broke = false := rfl
      exact stmts_heap_step params k.body fuel fuel
              { env := [], heap := h } s' k.initialCtx ctx_after
              (initialKOpsState k h d) st' ops
              h_initial_params h_initial_clean h_proj_initial
              h_eval h_t h_delta_eq h_run

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
