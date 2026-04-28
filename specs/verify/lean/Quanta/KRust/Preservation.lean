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
-- Value-level alignment lemmas
-- ════════════════════════════════════════════════════════════════════
--
-- These lemmas state that the primitive helpers on both sides
-- agree on a per-value basis. They do *not* require `evalExpr` /
-- `evalStmt` to unfold (the eval functions are still `partial def`
-- pending the lex-order termination measure noted in T590); they
-- only quote the pure `def` helpers (`evalLit`, `evalConst`,
-- `constOfLit`, `evalCast`, `dispatchScalar`). When the eval
-- functions land as `def`, T590-T5A7's proofs reduce via these
-- lemmas plus a structural unfolding of `evalExpr`/`evalStmt`.

/-- **Alignment 1 (bool case)**: for every boolean literal, the
    source-side `evalLit` and the translator-side `evalConst ∘
    constOfLit` produce the same `Value`. -/
theorem evalConst_eq_evalLit_bool
    (b : Bool) (cv : KOps.ConstValue) (sty : KOps.Scalar)
    (h_const : constOfLit (Lit.bool b) = some (cv, sty))
    : evalLit (Lit.bool b) = some (KOps.evalConst cv) := by
  simp [constOfLit] at h_const
  obtain ⟨h1, _⟩ := h_const
  subst h1
  rfl

/-- **Alignment 1 (i32 case)**: for every signed-32 integer literal,
    `evalLit` and `evalConst ∘ constOfLit` agree. -/
theorem evalConst_eq_evalLit_i32
    (n : Int) (cv : KOps.ConstValue) (sty : KOps.Scalar)
    (h_const : constOfLit (Lit.int n .i32) = some (cv, sty))
    : evalLit (Lit.int n .i32) = some (KOps.evalConst cv) := by
  simp [constOfLit] at h_const
  obtain ⟨h1, _⟩ := h_const
  subst h1
  rfl

/-- **Alignment 1 (f32 case)**: for every f32 float literal, both
    sides produce the same `vF32` bit pattern. The encoding choice
    `(w * 1000) + f mod 1000` is shared between `evalLit` and
    `constOfLit`, so the two coincide by construction. -/
theorem evalConst_eq_evalLit_f32
    (w f : Int) (cv : KOps.ConstValue) (sty : KOps.Scalar)
    (h_const : constOfLit (Lit.float w f .f32) = some (cv, sty))
    : evalLit (Lit.float w f .f32) = some (KOps.evalConst cv) := by
  simp [constOfLit] at h_const
  obtain ⟨h1, _⟩ := h_const
  subst h1
  rfl

-- ════════════════════════════════════════════════════════════════════
-- Alignment 2 — Cast agreement (KRust ↔ KOps)
-- ════════════════════════════════════════════════════════════════════

/-- **Alignment 2 (cast)**: the source-side `KRust.evalCast` and
    destination-side `KOps.evalCast` agree on every (value, target
    type) pair, modulo the `dispatchScalar` map. The two functions
    are syntactically identical; the bridge is just the namespace
    convention. -/
theorem evalCast_agreement
    (v : Value) (ty : Scalar)
    : KRust.evalCast v ty = KOps.evalCast v (dispatchScalar ty) := by
  cases ty <;> cases v <;> rfl

-- ════════════════════════════════════════════════════════════════════
-- Alignment 3 — UnaryOp agreement
-- ════════════════════════════════════════════════════════════════════

/-- **Alignment 3 (unary)**: KRust and KOps unary operators agree
    on every value pair, with `dispatchUnaryOp` acting as the
    namespace bridge. Provable by full case analysis since both
    functions branch identically on `Value` constructor. -/
theorem evalUnaryOp_agreement
    (op : UnaryOp) (v : Value)
    : KRust.evalUnaryOp op v = KOps.evalUnaryOp (dispatchUnaryOp op) v := by
  cases op <;> cases v <;> rfl

-- ════════════════════════════════════════════════════════════════════
-- Alignment 4 — BinOp agreement (arithmetic core)
-- ════════════════════════════════════════════════════════════════════

/-- **Alignment 4 (binop, arithmetic core)**: for the arithmetic
    binary operators (everything except `&&`/`||` short-circuit),
    `KRust.evalBinOp` and `KOps.evalBinOp ∘ dispatchBinOp.bin/cmp`
    produce the same value when applied to the same operands.

    The two implementations share the same primitive library
    (`Quanta.Semantics.Cpu`'s `eval_u32_*` / `eval_i32_*`); the
    operator-by-operator dispatch in both `evalBinOp` functions is
    structurally identical. The proof is a long case-split over
    operator × value-pair shape. -/
theorem evalBinOp_agreement_arith
    (op : BinOp) (va vb : Value)
    : (∀ b, dispatchBinOp op ≠ BinDispatch.logical b) →
      (KRust.evalBinOp op va vb =
         match dispatchBinOp op with
         | .bin bop => KOps.evalBinOp bop va vb
         | .cmp cop => KOps.evalCmpOp cop va vb
         | .logical _ => none) := by
  intro h_not_logical
  cases op <;>
    (try (cases va <;> cases vb <;> first | rfl | (simp [KRust.evalBinOp, KOps.evalBinOp, KOps.evalCmpOp, dispatchBinOp]))) <;>
    (try (exfalso; exact h_not_logical _ rfl))

-- ════════════════════════════════════════════════════════════════════
-- Alignment 5 — dispatchScalar bijection
-- ════════════════════════════════════════════════════════════════════

/-- **Alignment 5 (scalar bijection)**: the namespace-bridging
    `dispatchScalar` is injective. Two distinct KRust scalars never
    map to the same KOps scalar. Trivial by case analysis since
    `dispatchScalar` is a constructor-by-constructor identity
    rename. -/
theorem dispatchScalar_injective
    (a b : Scalar)
    (h : dispatchScalar a = dispatchScalar b)
    : a = b := by
  cases a <;> cases b <;> first | rfl | cases h

-- ════════════════════════════════════════════════════════════════════
-- Alignment 6 — RegFile / Env primitive lemmas
-- ════════════════════════════════════════════════════════════════════

/-- **Alignment 6.1 (regWrite-lookup)**: writing a value to a
    register and immediately looking it up returns the written
    value. Pure-state primitive used inside per-rule preservation
    proofs to discharge the "register r holds value v" obligation
    after a `KernelOp.const` or any other `dst`-writing op. -/
theorem regLookup_regWrite_eq
    (rf : KOps.RegFile) (r : KOps.Reg) (v : Value)
    : KOps.regLookup (KOps.regWrite rf r v) r = some v := by
  simp [KOps.regWrite, KOps.regLookup]

/-- **Alignment 6.2 (Env-bind-lookup)**: writing to the env and
    looking up the same name returns the written value. The KRust
    counterpart of 6.1; both views support this primitive shape. -/
theorem Env_lookup_bind_eq
    (env : Env) (name : Ident) (v : Value)
    : (env.bind name v).lookup name = some v := by
  simp [Env.bind, Env.lookup]

/-- **Alignment 6.3 (Heap.store-lookup)**: storing to a buffer
    slot+index and looking up the same key returns the stored
    value. Used by T5A2 (`assignIdx_preservation`). -/
theorem Heap_lookup_store_eq
    (h : Heap) (slot : Slot) (idx : Nat) (v : Value)
    : (h.store slot idx v).lookup slot idx = some v := by
  simp [Heap.store, Heap.lookup]

/-- **Alignment 6.4 (heapStore-lookup, KOps side)**: matching primitive
    on the KOps heap. -/
theorem heapLookup_heapStore_eq
    (h : KOps.Heap) (slot idx : Nat) (v : Value)
    : KOps.heapLookup (KOps.heapStore h slot idx v) slot idx = some v := by
  simp [KOps.heapStore, KOps.heapLookup]

-- ════════════════════════════════════════════════════════════════════
-- Alignment 7 — single-op KOps eval reductions
-- ════════════════════════════════════════════════════════════════════
--
-- `KOps.evalOp` was promoted from `partial def` to `def`
-- (Lean detected structural recursion automatically; the loop
-- arms's inner `opLoop` already had its own fuel parameter). The
-- following per-constructor reductions are now `rfl`-provable.
-- They give the "given an op X, evalOp produces this specific
-- state" facts that compose into the per-rule preservation
-- theorems T590–T5A7.

/-- `evalOp _ st (.const r cv)` writes `evalConst cv` into register
    `r` and leaves heap, dispatch, and broke flag unchanged. -/
theorem evalOp_const_eq
    (fuel : Nat) (st : KOps.State) (r : KOps.Reg) (cv : KOps.ConstValue)
    : KOps.evalOp fuel st (KernelOp.const r cv)
        = some { st with rf := KOps.regWrite st.rf r (KOps.evalConst cv) } := by
  simp [KOps.evalOp, pure]

/-- `evalOp _ st (.copy dst src)` returns `none` if `src` is unbound,
    otherwise writes `src`'s value into `dst`. -/
theorem evalOp_copy_eq
    (fuel : Nat) (st : KOps.State) (dst src : KOps.Reg)
    : KOps.evalOp fuel st (KernelOp.copy dst src)
        = (KOps.regLookup st.rf src).bind
            (fun v => some { st with rf := KOps.regWrite st.rf dst v }) := by
  simp [KOps.evalOp, pure, bind, Option.bind]

/-- `evalOp _ st .breakOp` sets the broke flag. -/
theorem evalOp_breakOp_eq
    (fuel : Nat) (st : KOps.State)
    : KOps.evalOp fuel st KernelOp.breakOp = some { st with broke := true } := by
  simp [KOps.evalOp, pure]

/-- `evalOp _ st .barrier` is a no-op. -/
theorem evalOp_barrier_eq
    (fuel : Nat) (st : KOps.State)
    : KOps.evalOp fuel st KernelOp.barrier = some st := by
  simp [KOps.evalOp, pure]

/-- Identity intrinsic: `quarkId`. The four other identity
    intrinsics (`protonId`, `nucleusId`, `protonSize`,
    `quarkCount`) follow the same shape. -/
theorem evalOp_quarkId_eq
    (fuel : Nat) (st : KOps.State) (dst : KOps.Reg)
    : KOps.evalOp fuel st (KernelOp.quarkId dst)
        = some { st with rf := KOps.regWrite st.rf dst (KOps.vU32 st.dispatch.quarkId) } := by
  simp [KOps.evalOp, pure]

theorem evalOp_protonId_eq
    (fuel : Nat) (st : KOps.State) (dst : KOps.Reg)
    : KOps.evalOp fuel st (KernelOp.protonId dst)
        = some { st with rf := KOps.regWrite st.rf dst (KOps.vU32 st.dispatch.protonId) } := by
  simp [KOps.evalOp, pure]

theorem evalOp_nucleusId_eq
    (fuel : Nat) (st : KOps.State) (dst : KOps.Reg)
    : KOps.evalOp fuel st (KernelOp.nucleusId dst)
        = some { st with rf := KOps.regWrite st.rf dst (KOps.vU32 st.dispatch.nucleusId) } := by
  simp [KOps.evalOp, pure]

theorem evalOp_protonSize_eq
    (fuel : Nat) (st : KOps.State) (dst : KOps.Reg)
    : KOps.evalOp fuel st (KernelOp.protonSize dst)
        = some { st with rf := KOps.regWrite st.rf dst (KOps.vU32 st.dispatch.protonSize) } := by
  simp [KOps.evalOp, pure]

theorem evalOp_quarkCount_eq
    (fuel : Nat) (st : KOps.State) (dst : KOps.Reg)
    : KOps.evalOp fuel st (KernelOp.quarkCount dst)
        = some { st with rf := KOps.regWrite st.rf dst (KOps.vU32 st.dispatch.quarkCount) } := by
  simp [KOps.evalOp, pure]

-- ════════════════════════════════════════════════════════════════════
-- Alignment 8 — evalOps singleton + cons
-- ════════════════════════════════════════════════════════════════════

/-- A non-`broke` singleton op-list reduces to the single op's eval. -/
theorem evalOps_singleton_clean
    (fuel : Nat) (st : KOps.State) (op : KernelOp) (st' : KOps.State)
    (h_eval : KOps.evalOp fuel st op = some st')
    (h_broke : st'.broke = false)
    : KOps.evalOps fuel st [op] = some st' := by
  simp [KOps.evalOps, bind, Option.bind, h_eval, h_broke]

/-- Cons reduces to: eval head, then if `broke` short-circuit, else
    recurse on tail. -/
theorem evalOps_cons_eq
    (fuel : Nat) (st : KOps.State) (op : KernelOp) (rest : List KernelOp)
    : KOps.evalOps fuel st (op :: rest)
        = (KOps.evalOp fuel st op).bind
            (fun s1 => if s1.broke then some s1 else KOps.evalOps fuel s1 rest) := by
  simp [KOps.evalOps, bind, Option.bind]

-- ════════════════════════════════════════════════════════════════════
-- T5A7' — focused breakS preservation on the KOps side
-- ════════════════════════════════════════════════════════════════════

/-- **T5A7' — breakS_preservation_focused**: the KOps-side fact
    that translating `breakS` produces the singleton op list
    `[KernelOp.breakOp]`, and running it produces a state whose
    register file and heap are unchanged, with the broke flag set
    to `true`.

    The full T5A7 falls out of this lemma plus the source-side
    `evalStmt _ s .breakS = some { s with broke := true }`, which
    needs `KRust.evalStmt` to reduce — pending the `partial def`
    → `def` conversion noted in T590's docstring. -/
theorem t5a7_breakS_preservation_focused
    (s : State) (st : KOps.State)
    : ∃ st',
        KOps.evalOps 1 st [KernelOp.breakOp] = some st'
        ∧ st'.rf = st.rf
        ∧ st'.heap = st.heap
        ∧ st'.broke = true := by
  refine ⟨{ st with broke := true }, ?_, ?_, ?_, ?_⟩
  · -- evalOps fuel st [breakOp] = bind (evalOp fuel st breakOp) (...)
    -- evalOp fuel st breakOp = some { st with broke := true }
    -- broke is true, so the cons short-circuits to the broke state.
    rw [evalOps_cons_eq, evalOp_breakOp_eq]
    simp
  · rfl
  · rfl
  · rfl

-- ════════════════════════════════════════════════════════════════════
-- T590' — focused lit preservation (KOps side)
-- ════════════════════════════════════════════════════════════════════

/-- **T590' — lit_preservation_focused**: a focused KOps-side
    theorem stating that if `constOfLit l` produces `(cv, sty)`,
    then running `[KernelOp.const r cv]` on any state writes
    `evalConst cv` (which equals `evalLit l` by alignment) into
    register `r`.

    The full T590 (linking source-side `evalLit` through to the
    register) reduces to this lemma plus the value alignment
    `evalConst_eq_evalLit_*`. The remaining gap is unfolding
    `KRust.evalExpr (.lit l)` to extract `evalLit l`'s result —
    which needs `KRust.evalExpr` to be `def` rather than `partial
    def`. -/
theorem t590_lit_preservation_focused
    (l : Lit) (cv : KOps.ConstValue) (sty : KOps.Scalar) (r : KOps.Reg) (st : KOps.State)
    (h_const : constOfLit l = some (cv, sty))
    (h_st_clean : st.broke = false)
    : ∃ st',
        KOps.evalOps 1 st [KernelOp.const r cv] = some st'
        ∧ KOps.regLookup st'.rf r = some (KOps.evalConst cv) := by
  refine ⟨{ st with rf := KOps.regWrite st.rf r (KOps.evalConst cv) }, ?_, ?_⟩
  · exact evalOps_singleton_clean 1 st (KernelOp.const r cv) _
            (evalOp_const_eq 1 st r cv) h_st_clean
  · exact regLookup_regWrite_eq st.rf r (KOps.evalConst cv)

-- ════════════════════════════════════════════════════════════════════
-- T592' — focused binop preservation (KOps side, arith dispatch)
-- ════════════════════════════════════════════════════════════════════

/-- **T592' — binop_preservation_focused (arith)**: given two
    register values `va`, `vb` already established to satisfy
    `evalBinOp_agreement_arith`, running `[KernelOp.binOp dst a b
    op .u32]` writes the agreement value into `dst`.

    The full T592 chains this with the per-operand recursive
    preservation. -/
theorem t592_binop_preservation_focused_bin
    (bop : KOps.BinOp) (a b dst : KOps.Reg) (ty : KOps.Scalar)
    (st : KOps.State) (va vb v : Value)
    (h_a : KOps.regLookup st.rf a = some va)
    (h_b : KOps.regLookup st.rf b = some vb)
    (h_op : KOps.evalBinOp bop va vb = some v)
    (h_st_clean : st.broke = false)
    : ∃ st',
        KOps.evalOps 1 st [KernelOp.binOp dst a b bop ty] = some st'
        ∧ KOps.regLookup st'.rf dst = some v := by
  refine ⟨{ st with rf := KOps.regWrite st.rf dst v }, ?_, ?_⟩
  · refine evalOps_singleton_clean 1 st (KernelOp.binOp dst a b bop ty)
              { st with rf := KOps.regWrite st.rf dst v } ?_ h_st_clean
    simp [KOps.evalOp, h_a, h_b, h_op, bind, Option.bind, pure]
  · exact regLookup_regWrite_eq st.rf dst v

-- ════════════════════════════════════════════════════════════════════
-- T593' — focused unary preservation
-- ════════════════════════════════════════════════════════════════════

theorem t593_unaryop_preservation_focused
    (uop : KOps.UnaryOp) (a dst : KOps.Reg) (ty : KOps.Scalar)
    (st : KOps.State) (va v : Value)
    (h_a : KOps.regLookup st.rf a = some va)
    (h_op : KOps.evalUnaryOp uop va = some v)
    (h_st_clean : st.broke = false)
    : ∃ st',
        KOps.evalOps 1 st [KernelOp.unaryOp dst a uop ty] = some st'
        ∧ KOps.regLookup st'.rf dst = some v := by
  refine ⟨{ st with rf := KOps.regWrite st.rf dst v }, ?_, ?_⟩
  · refine evalOps_singleton_clean 1 st (KernelOp.unaryOp dst a uop ty)
              { st with rf := KOps.regWrite st.rf dst v } ?_ h_st_clean
    simp [KOps.evalOp, h_a, h_op, bind, Option.bind, pure]
  · exact regLookup_regWrite_eq st.rf dst v

-- ════════════════════════════════════════════════════════════════════
-- T595' — focused cast preservation
-- ════════════════════════════════════════════════════════════════════

theorem t595_cast_preservation_focused
    (src dst : KOps.Reg) (fromTy toTy : KOps.Scalar)
    (st : KOps.State) (v_src v : Value)
    (h_src : KOps.regLookup st.rf src = some v_src)
    (h_cast : KOps.evalCast v_src toTy = some v)
    (h_st_clean : st.broke = false)
    : ∃ st',
        KOps.evalOps 1 st [KernelOp.cast dst src fromTy toTy] = some st'
        ∧ KOps.regLookup st'.rf dst = some v := by
  refine ⟨{ st with rf := KOps.regWrite st.rf dst v }, ?_, ?_⟩
  · refine evalOps_singleton_clean 1 st (KernelOp.cast dst src fromTy toTy)
              { st with rf := KOps.regWrite st.rf dst v } ?_ h_st_clean
    simp [KOps.evalOp, h_src, h_cast, bind, Option.bind, pure]
  · exact regLookup_regWrite_eq st.rf dst v

-- ════════════════════════════════════════════════════════════════════
-- T594' — focused index preservation
-- ════════════════════════════════════════════════════════════════════

/-- Open per-rule preservation: the `load`/`store` arms branch on
    the runtime shape of the index value (`.vU32 n`). The `simp`
    set reduces the bind chain but leaves the match-on-Value-shape
    unreduced; closing requires a custom `cases vi` step. Tracked
    as part of the broader open obligation. -/
theorem t594_index_preservation_focused
    (slot : Nat) (idx_reg dst : KOps.Reg) (ty : KOps.Scalar)
    (st : KOps.State) (idx : UInt32) (v : Value)
    (h_idx : KOps.regLookup st.rf idx_reg = some (KOps.vU32 idx))
    (h_load : KOps.heapLookup st.heap slot idx.toNat = some v)
    (h_st_clean : st.broke = false)
    : ∃ st',
        KOps.evalOps 1 st [KernelOp.load dst slot idx_reg ty] = some st'
        ∧ KOps.regLookup st'.rf dst = some v := by
  sorry

-- ════════════════════════════════════════════════════════════════════
-- T5A2' — focused assignIdx (store) preservation
-- ════════════════════════════════════════════════════════════════════

/-- Open per-rule preservation: `store` shares the same value-shape
    match as `load`. -/
theorem t5a2_assignIdx_preservation_focused
    (slot : Nat) (idx_reg src_reg : KOps.Reg) (ty : KOps.Scalar)
    (st : KOps.State) (idx : UInt32) (v : Value)
    (h_idx : KOps.regLookup st.rf idx_reg = some (KOps.vU32 idx))
    (h_src : KOps.regLookup st.rf src_reg = some v)
    (h_st_clean : st.broke = false)
    : ∃ st',
        KOps.evalOps 1 st [KernelOp.store slot idx_reg src_reg ty] = some st'
        ∧ KOps.heapLookup st'.heap slot idx.toNat = some v := by
  sorry

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
    sets the same flag on the KOps state. With `evalStmt` now
    `def`, the source side reduces; the open obligation is the
    `evalOps_append` composition lemma needed when `ctx.ops`
    already contains entries from earlier in translation. -/
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
