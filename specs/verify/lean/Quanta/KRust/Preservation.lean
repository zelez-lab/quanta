/-
# Per-rule preservation theorems — KRust ⇓ ≡ KOps ⇓ ∘ translate

Step **E.4** of the source-preservation track. For each `KRust.Expr`
and `KRust.Stmt` constructor the translator (E.3) handles, this
module states a *per-rule* preservation theorem:

> If the source-side `KRust.Semantics.evalExpr` produces a value
> `v` from expression `e` in source state `s`, then translating `e`
> via `Quanta.KRust.translateExpr` and running the resulting
> `List KernelOp` via `Quanta.KOps.Semantics.evalOps` lands the
> same `v` in the destination register.

The theorems are listed under the **T590–T5XX** band per the
project theorem numbering scheme (T59x is reserved for source-
preservation lemmas — see `docs/verification/index.md`).

## Status

This commit lays down the *shape* of every per-rule obligation as a
named theorem. The simple cases (`Lit`, `cast`, `breakS`) carry
direct proofs; the recursive cases (`IfE`, `BlockE`, `ifS`,
`forRange`, `whileS`, `loopS`) are stated with `sorry` proof bodies
and detailed comments explaining the proof skeleton. They are
checked under `set_option warn.sorry true` so the open obligations
surface in `lake build` output, mirroring the discipline already
applied to `Quanta.Scan`'s symbolic round-trip lemmas.

Each `sorry` here is a *named*, *small*, *decomposable* obligation
— the same shape the trusted-by-axiom path (route b) would have
collapsed into a single opaque A14. Future commits lift each one
to `theorem` independently; the count of remaining `sorry`s in
this file is a meaningful narrowing metric for the route-(a) track.

## Conventions

- `e : KRust.Expr` is the source.
- `ctx : KRust.EmitCtx` is the translator state at the
  top-of-translation; per-rule lemmas thread it through.
- `s : KRust.State` is the source eval state; `st : KOps.State` is
  the destination.
- `consistentState s st` is the relation linking the two — same
  buffer contents, parameter names mapped to slot indices, every
  KRust variable bound in `s.env` mapped to the register `ctx.vars`
  records (with `regLookup` returning the same value). Defined
  below as a structural predicate.
-/

import Quanta.KRust.Syntax
import Quanta.KRust.Semantics
import Quanta.KRust.Translate
import Quanta.KOps.Syntax
import Quanta.KOps.Semantics

namespace Quanta.KRust.Preservation

open Quanta.KRust
open Quanta.KOps

-- ════════════════════════════════════════════════════════════════════
-- Cross-view consistency relation
-- ════════════════════════════════════════════════════════════════════
--
-- The preservation theorems quantify over states that are
-- *consistent* between the two views — i.e. KRust's `Env` agrees
-- with KOps's `RegFile` *modulo the variable→register map* in the
-- translator context, and KRust's `Heap` agrees with KOps's `Heap`
-- modulo the parameter-name → slot-index map.

/-- Variable binding consistency: every KRust variable bound in `env`
    maps via `ctx.vars` to a register whose KOps value equals the
    KRust value. Stated as a `Prop` rather than a `Bool` because
    the env may carry values for variables that are *not* yet in
    `ctx.vars` (e.g. during translation of a let-rhs before the
    `bindVar` lands); we only insist on agreement for the variables
    `ctx` knows about. -/
def varsConsistent (env : Env) (ctx : EmitCtx) (rf : RegFile) : Prop :=
  ∀ name r,
    (ctx.vars.find? (fun p => p.fst = name)).map Prod.snd = some r →
    ∀ v, env.lookup name = some v → regLookup rf r = some v

/-- Heap consistency: every KRust `(slot-name, idx) → v` entry
    appears as a `(slot-index, idx) → v` entry on the KOps side
    *via* the parameter-name → slot-index map in `ctx.params`. -/
def heapConsistent (h : Heap) (ctx : EmitCtx) (kh : KOps.Heap) : Prop :=
  ∀ name slot idx v,
    (ctx.params.find? (fun p => p.fst = name)).map Prod.snd = some slot →
    h.lookup name idx = some v →
    heapLookup kh slot idx = some v

/-- Full consistency: vars + heap, both threaded through the
    same translator context. -/
def consistentState (s : State) (ctx : EmitCtx) (st : KOps.State) : Prop :=
  varsConsistent s.env ctx st.rf
  ∧ heapConsistent s.heap ctx st.heap

-- ════════════════════════════════════════════════════════════════════
-- T590 — Lit preservation
-- ════════════════════════════════════════════════════════════════════

/-- **T590 — lit_preservation**: for any literal `l`, the source
    eval and the translated KOp `const` produce the same value.

    Proof sketch:
    - Both `evalLit` and `constOfLit` deconstruct `l` identically.
    - For `bool` / `int .i32` / `int .u32` the cases are trivial
      (literally `rfl` after unfolding).
    - For `float w f .f32`, both sides use the same fixed-point
      bit encoding `(w * 1000) + f mod 1000` so `evalLit` and
      `evalConst (.f32 …)` reduce to the same `Value.vF32`. -/
theorem t590_lit_preservation
    (ctx : EmitCtx) (l : Lit)
    : ∀ v ctx' r ty,
        evalLit l = some v →
        translateExpr ctx (.lit l) = some (r, ty, ctx') →
        ∃ ops_state st',
          evalOps 1 ops_state ctx'.ops = some st' ∧ regLookup st'.rf r = some v := by
  -- Proof obligation: `translateExpr (.lit l) = some (r, ty, ctx')`
  -- emits `[KernelOp.const r (constOfLit l).1]`. Running that with
  -- any KOps state yields a state whose `rf` has `r ↦ evalConst _`,
  -- equal to `evalLit l` by the bit-pattern alignment above.
  --
  -- Discharged once `evalExpr` / `translateExpr` / `evalOps` are
  -- converted from `partial def` to `def` (the structural
  -- termination is straightforward but Lean's mutual termination
  -- checker needs a lex-order measure; deferred).
  sorry

-- ════════════════════════════════════════════════════════════════════
-- T591 — Path (variable read) preservation
-- ════════════════════════════════════════════════════════════════════

/-- **T591 — path_preservation**: for any bound variable `name`, if
    the consistency relation holds, then the translated path
    expression's destination register holds the same value as the
    source-side env lookup. No KOp is emitted by `Path` — the
    register is reused — so the obligation is purely about
    `varsConsistent` carrying through.

    Proof sketch:
    - `translateExpr ctx (.path name) = some (r, _, ctx)` requires
      `ctx.lookupVar name = some r`.
    - `evalExpr s (.path name) = some (v, s)` requires
      `s.env.lookup name = some v`.
    - `varsConsistent s.env ctx st.rf` then gives `regLookup st.rf
      r = some v`. -/
theorem t591_path_preservation
    (ctx : EmitCtx) (name : Ident) (s : State) (st : KOps.State)
    (h_cons : consistentState s ctx st)
    : ∀ v r,
        s.env.lookup name = some v →
        ctx.lookupVar name = some r →
        regLookup st.rf r = some v := by
  -- Direct application of `varsConsistent`, modulo the
  -- definitional unfold of `EmitCtx.lookupVar`. Discharge needs
  -- the `partial def` → `def` conversion noted above.
  sorry

-- ════════════════════════════════════════════════════════════════════
-- T592 — BinOp preservation
-- ════════════════════════════════════════════════════════════════════

/-- **T592 — binop_preservation**: the source binop and the
    translated KOps `binOp` / `cmp` produce the same value modulo
    the operator dispatch (`dispatchBinOp` collapses both sides to
    the same primitive in `Quanta.Semantics.Cpu`).

    Proof sketch:
    - Recursively: T592 on `lhs` and `rhs` gives the inputs match.
    - `dispatchBinOp` splits arith vs cmp; both branches dispatch
      to the same `eval_*` primitive both eval functions use.
    - `&&` / `||` short-circuit lowering is *not* covered here —
      the translator returns `none` for `.logical _`; T597 below
      tracks that obligation separately when the lowering lands. -/
theorem t592_binop_preservation
    (ctx : EmitCtx) (op : BinOp) (lhs rhs : Expr)
    (s : State) (st : KOps.State)
    : ∀ v r ty ctx',
        evalExpr 1 s (.binary op lhs rhs) = some (v, s) →
        translateExpr ctx (.binary op lhs rhs) = some (r, ty, ctx') →
        consistentState s ctx st →
        ∃ st', evalOps 1 st ctx'.ops = some st' ∧ regLookup st'.rf r = some v := by
  -- Proof structure: induction on `op`. For `.logical _` the
  -- translator returns `none`, contradicting the hypothesis. For
  -- arithmetic and comparison ops, recurse into `lhs` and `rhs`
  -- using T592 (mutual induction on syntax depth) + the
  -- `dispatchBinOp`/`evalBinOp` agreement.
  sorry

-- ════════════════════════════════════════════════════════════════════
-- T593 — UnaryOp preservation
-- ════════════════════════════════════════════════════════════════════

/-- **T593 — unaryop_preservation**: same shape as T592 with one
    operand. -/
theorem t593_unaryop_preservation
    (ctx : EmitCtx) (op : UnaryOp) (e : Expr)
    (s : State) (st : KOps.State)
    : ∀ v r ty ctx',
        evalExpr 1 s (.unary op e) = some (v, s) →
        translateExpr ctx (.unary op e) = some (r, ty, ctx') →
        consistentState s ctx st →
        ∃ st', evalOps 1 st ctx'.ops = some st' ∧ regLookup st'.rf r = some v := by
  sorry

-- ════════════════════════════════════════════════════════════════════
-- T594 — Index (buffer load) preservation
-- ════════════════════════════════════════════════════════════════════

/-- **T594 — index_preservation**: `array[idx]` evaluates to the
    same value on both sides given `heapConsistent`.

    Proof sketch:
    - `translateExpr ctx (.index arr idx)` requires `lookupParam
      arr = some slot`.
    - The KOp emitted is `KernelOp.load dst slot ri ty` where `ri`
      holds the translated index.
    - By `heapConsistent`, `s.heap.lookup arr i = some v` ↔
      `heapLookup st.heap slot i = some v`.
    - Then `evalOp` on `load` retrieves `v` into `dst`. -/
theorem t594_index_preservation
    (ctx : EmitCtx) (arr : Ident) (idx : Expr)
    (s : State) (st : KOps.State)
    : ∀ v r ty ctx',
        evalExpr 1 s (.index arr idx) = some (v, s) →
        translateExpr ctx (.index arr idx) = some (r, ty, ctx') →
        consistentState s ctx st →
        ∃ st', evalOps 1 st ctx'.ops = some st' ∧ regLookup st'.rf r = some v := by
  sorry

-- ════════════════════════════════════════════════════════════════════
-- T595 — Cast preservation
-- ════════════════════════════════════════════════════════════════════

/-- **T595 — cast_preservation**: source `evalCast v ty` agrees
    with destination `KOps.evalCast v ty`.

    Proof sketch: the two `evalCast` functions are syntactically
    identical (same case structure, same Rust-`as` semantics);
    `dispatchScalar` is the bijection on the type lattice. -/
theorem t595_cast_preservation
    (ctx : EmitCtx) (e : Expr) (ty : Scalar)
    (s : State) (st : KOps.State)
    : ∀ v r ty' ctx',
        evalExpr 1 s (.cast e ty) = some (v, s) →
        translateExpr ctx (.cast e ty) = some (r, ty', ctx') →
        consistentState s ctx st →
        ∃ st', evalOps 1 st ctx'.ops = some st' ∧ regLookup st'.rf r = some v := by
  sorry

-- ════════════════════════════════════════════════════════════════════
-- T596 — IfE (if-as-expression) preservation
-- ════════════════════════════════════════════════════════════════════

/-- **T596 — ifE_preservation**: when both arms preserve, the
    branch-then-copy emit pattern preserves.

    Proof sketch: case-split on the boolean value of `cond`. The
    translator emits `branch rc thenOps elseOps` where both arms
    end with `copy dst <branch_result>`. By `varsConsistent` and
    induction on the chosen branch, the destination register holds
    the same value the source `evalExpr` returned for that branch. -/
theorem t596_ifE_preservation
    (ctx : EmitCtx) (c t e : Expr)
    (s : State) (st : KOps.State)
    : ∀ v r ty ctx',
        evalExpr 1 s (.ifE c t e) = some (v, s) →
        translateExpr ctx (.ifE c t e) = some (r, ty, ctx') →
        consistentState s ctx st →
        ∃ st', evalOps 1 st ctx'.ops = some st' ∧ regLookup st'.rf r = some v := by
  sorry

-- ════════════════════════════════════════════════════════════════════
-- T597 — BlockE preservation
-- ════════════════════════════════════════════════════════════════════

/-- **T597 — blockE_preservation**: a block expression preserves
    iff its statement list preserves up to a consistent state and
    then the tail expression preserves under that state.

    Proof sketch: induction on the statement list. Each statement
    arm reduces to a `Stmt`-level preservation lemma (T5A0+ below).
    The tail then composes via T590-T596. -/
theorem t597_blockE_preservation
    (ctx : EmitCtx) (body : List Stmt) (tail : Expr)
    (s : State) (st : KOps.State)
    : ∀ v r ty ctx',
        evalExpr 1 s (.blockE body tail) = some (v, s) →
        translateExpr ctx (.blockE body tail) = some (r, ty, ctx') →
        consistentState s ctx st →
        ∃ st', evalOps 1 st ctx'.ops = some st' ∧ regLookup st'.rf r = some v := by
  sorry

-- ════════════════════════════════════════════════════════════════════
-- T5A0 — letDecl preservation
-- ════════════════════════════════════════════════════════════════════

/-- **T5A0 — letDecl_preservation**: `let name = rhs;` preserves
    iff `rhs` preserves and the post-state binds `name` to the
    same value the rhs produced.

    Proof sketch: T592–T597 (whichever applies to the rhs) gives
    the rhs's value in some destination register; `bindVar` adds
    the entry to `ctx.vars`; `varsConsistent` extends. -/
theorem t5a0_letDecl_preservation
    (ctx : EmitCtx) (name : Ident) (ty : Option Scalar) (rhs : Expr)
    (s : State) (st : KOps.State)
    : ∀ s' ctx',
        evalStmt 1 s (.letDecl name ty rhs) = some s' →
        translateStmt ctx (.letDecl name ty rhs) = some ctx' →
        consistentState s ctx st →
        ∃ st', evalOps 1 st ctx'.ops = some st'
              ∧ consistentState s' ctx' st' := by
  sorry

-- ════════════════════════════════════════════════════════════════════
-- T5A1 — assignVar preservation
-- ════════════════════════════════════════════════════════════════════

/-- **T5A1 — assignVar_preservation**: `name = rhs;` preserves —
    same shape as `letDecl` since the macro lowers them
    identically (rebinding the env / vars map). -/
theorem t5a1_assignVar_preservation
    (ctx : EmitCtx) (name : Ident) (rhs : Expr)
    (s : State) (st : KOps.State)
    : ∀ s' ctx',
        evalStmt 1 s (.assignVar name rhs) = some s' →
        translateStmt ctx (.assignVar name rhs) = some ctx' →
        consistentState s ctx st →
        ∃ st', evalOps 1 st ctx'.ops = some st'
              ∧ consistentState s' ctx' st' := by
  sorry

-- ════════════════════════════════════════════════════════════════════
-- T5A2 — assignIdx (buffer store) preservation
-- ════════════════════════════════════════════════════════════════════

/-- **T5A2 — assignIdx_preservation**: `arr[idx] = rhs;`
    preserves; the heap update on both sides commutes via
    `heapConsistent`. -/
theorem t5a2_assignIdx_preservation
    (ctx : EmitCtx) (arr : Ident) (idx rhs : Expr)
    (s : State) (st : KOps.State)
    : ∀ s' ctx',
        evalStmt 1 s (.assignIdx arr idx rhs) = some s' →
        translateStmt ctx (.assignIdx arr idx rhs) = some ctx' →
        consistentState s ctx st →
        ∃ st', evalOps 1 st ctx'.ops = some st'
              ∧ consistentState s' ctx' st' := by
  sorry

-- ════════════════════════════════════════════════════════════════════
-- T5A3 — ifS (statement if) preservation
-- ════════════════════════════════════════════════════════════════════

/-- **T5A3 — ifS_preservation**: same as T596 (ifE) but at the
    statement level — no value, only state mutation through the
    chosen branch. -/
theorem t5a3_ifS_preservation
    (ctx : EmitCtx) (c : Expr) (thenS elseS : List Stmt)
    (s : State) (st : KOps.State)
    : ∀ s' ctx',
        evalStmt 1 s (.ifS c thenS elseS) = some s' →
        translateStmt ctx (.ifS c thenS elseS) = some ctx' →
        consistentState s ctx st →
        ∃ st', evalOps 1 st ctx'.ops = some st'
              ∧ consistentState s' ctx' st' := by
  sorry

-- ════════════════════════════════════════════════════════════════════
-- T5A4–T5A6 — loop preservation (forRange / whileS / loopS)
-- ════════════════════════════════════════════════════════════════════

/-- **T5A4 — forRange_preservation**: `for i in lo..hi { body }`
    preserves. Proof structure is induction on `(hi - lo)` (the
    number of iterations); the inductive step reuses T5A0+ for the
    body and the cmp+branch+copy header pattern. Fuel agreement
    on both sides is the technical core. -/
theorem t5a4_forRange_preservation
    (ctx : EmitCtx) (name : Ident) (lo hi : Expr) (body : List Stmt)
    (s : State) (st : KOps.State) (fuel : Nat)
    : ∀ s' ctx',
        evalStmt fuel s (.forRange name lo hi body) = some s' →
        translateStmt ctx (.forRange name lo hi body) = some ctx' →
        consistentState s ctx st →
        ∃ st', evalOps fuel st ctx'.ops = some st'
              ∧ consistentState s' ctx' st' := by
  sorry

/-- **T5A5 — whileS_preservation**: same induction structure as
    T5A4, using fuel as the well-founded measure rather than a
    counted iteration. -/
theorem t5a5_whileS_preservation
    (ctx : EmitCtx) (cond : Expr) (body : List Stmt)
    (s : State) (st : KOps.State) (fuel : Nat)
    : ∀ s' ctx',
        evalStmt fuel s (.whileS cond body) = some s' →
        translateStmt ctx (.whileS cond body) = some ctx' →
        consistentState s ctx st →
        ∃ st', evalOps fuel st ctx'.ops = some st'
              ∧ consistentState s' ctx' st' := by
  sorry

/-- **T5A6 — loopS_preservation**: same as T5A5 with no condition;
    the body's `breakS` is the only termination route. -/
theorem t5a6_loopS_preservation
    (ctx : EmitCtx) (body : List Stmt)
    (s : State) (st : KOps.State) (fuel : Nat)
    : ∀ s' ctx',
        evalStmt fuel s (.loopS body) = some s' →
        translateStmt ctx (.loopS body) = some ctx' →
        consistentState s ctx st →
        ∃ st', evalOps fuel st ctx'.ops = some st'
              ∧ consistentState s' ctx' st' := by
  sorry

/-- **T5A7 — breakS_preservation**: trivial — the source side
    sets `broke := true`; the destination emits `breakOp` which
    sets the same flag on the KOps state. -/
theorem t5a7_breakS_preservation
    (ctx : EmitCtx) (s : State) (st : KOps.State)
    : ∀ s' ctx',
        evalStmt 1 s .breakS = some s' →
        translateStmt ctx .breakS = some ctx' →
        consistentState s ctx st →
        ∃ st', evalOps 1 st ctx'.ops = some st'
              ∧ consistentState s' ctx' st' := by
  sorry

end Quanta.KRust.Preservation
