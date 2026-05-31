/-
# Scope-validity of WASM-route lowering output

First end-to-end consumer of `Quanta.KOps.scopeValid` against the
Lean model of the WASM-route translator (`Quanta.Wasm.Translate`).

**Claim shape**: for every successful `lowerInstr s instr = some (s', ops)`
where `s` is well-scoped, the emitted `ops` are scope-valid against
the scope env of `s`. The scope env is `List.range s.nextReg` — every
register the allocator has ever issued.

This file ships:

1. `LowerState.scopeEnv` — the env list `[0, 1, …, nextReg-1]`.
2. `LowerState.wellScoped` — invariant: every register reachable from
   the state (stack regs, localReg, currentReg) is `< nextReg`.
3. `LowerState.alloc_*` — sanity lemmas: `alloc` bumps `nextReg` by 1,
   the emitted reg equals `s.nextReg`, and `scopeEnv` grows by one.
4. `scopeEnv_mono_alloc` — the post-alloc `scopeEnv` contains the
   pre-alloc one (just the membership face of (3)).
5. Per-arm theorems for the empty-emit instructions:
   `i32Const`, `nop`, `wreturn`, `drop` — they emit `[]`, vacuously
   `scopeValid` against any env.
6. `lowerInstr_localGet_nonbuffer_scopeValid` — first non-trivial
   arm: when `lookupBufferSlot i = none` and the lookup chain
   succeeds, the emitted `[.copy fresh source]` is scope-valid
   against the post-state's scope env, given `wellScoped s`.

The remaining arms (binOp, cmp, shl, add, load, store, localSet/Tee)
are queued for follow-up — the harder ones because they thread
multiple `alloc`s and `commit`s and need the full induction over
`Wasm/Preservation`-style helper lemmas. This module establishes the
predicate vocabulary they will all use.
-/

import Quanta.Wasm.Translate
import Quanta.Wasm.LowerInvariants
import Quanta.KOps.Scope

namespace Quanta.Wasm

open Quanta.KOps (KernelOp Reg Scalar)
open Quanta.KOps.KernelOp (scopeValid scopeValidOps usedRegs extendEnv extendEnvOps)

-- ════════════════════════════════════════════════════════════════════
-- Scope env: every register `< nextReg`
-- ════════════════════════════════════════════════════════════════════

/-- The env list `[0, 1, …, s.nextReg - 1]`. Every register the
    allocator has issued so far. Emitted ops are required to be
    scope-valid against this env. -/
def LowerState.scopeEnv (s : LowerState) : List Reg :=
  List.range s.nextReg

/-- `scopeEnv` membership iff `r < nextReg`. -/
theorem LowerState.mem_scopeEnv (s : LowerState) (r : Reg) :
    r ∈ s.scopeEnv ↔ r < s.nextReg := by
  unfold LowerState.scopeEnv
  exact List.mem_range

-- ════════════════════════════════════════════════════════════════════
-- alloc lemmas
-- ════════════════════════════════════════════════════════════════════

/-- `alloc` issues `s.nextReg` as the new register. -/
theorem LowerState.alloc_fst (s : LowerState) :
    s.alloc.fst = s.nextReg := rfl

/-- `alloc` bumps `nextReg` by 1. -/
theorem LowerState.alloc_snd_nextReg (s : LowerState) :
    s.alloc.snd.nextReg = s.nextReg + 1 := rfl

/-- The pre-alloc `scopeEnv` is a sublist (⊆) of the post-alloc one
    — the post-alloc env adds exactly `s.nextReg` at the end. -/
theorem LowerState.scopeEnv_subset_alloc (s : LowerState) :
    s.scopeEnv ⊆ s.alloc.snd.scopeEnv := by
  intro r hr
  rw [LowerState.mem_scopeEnv] at hr
  rw [LowerState.mem_scopeEnv, LowerState.alloc_snd_nextReg]
  exact Nat.lt_succ_of_lt hr

/-- The freshly-allocated register sits in the post-alloc `scopeEnv`. -/
theorem LowerState.fresh_mem_scopeEnv_alloc (s : LowerState) :
    s.nextReg ∈ s.alloc.snd.scopeEnv := by
  rw [LowerState.mem_scopeEnv, LowerState.alloc_snd_nextReg]
  exact Nat.lt_succ_self _

/-- Monotonicity of `scopeEnv` in `nextReg`. -/
theorem LowerState.scopeEnv_subset_of_nextReg_le {s s' : LowerState}
    (h : s.nextReg ≤ s'.nextReg) : s.scopeEnv ⊆ s'.scopeEnv := by
  intro r hr
  rw [LowerState.mem_scopeEnv] at hr ⊢
  exact Nat.lt_of_lt_of_le hr h

-- ════════════════════════════════════════════════════════════════════
-- well-scoped invariant
-- ════════════════════════════════════════════════════════════════════

/-- A `LowerState` is well-scoped when every register reachable from
    its mutable fields (`stack`, `localReg`, `currentReg`) lives below
    `nextReg`. This is the invariant the lowering pass maintains and
    that downstream arms need to argue their inputs satisfy.

    `stack` regs are read by `pop`/`popSym` and emitted as operands of
    binops/cmps/loads/stores. `localReg` and `currentReg` are read by
    `localGet` and emitted as Copy sources. All three feed `usedRegs`
    of the emitted ops — so each must lie in `scopeEnv`. -/
def LowerState.wellScoped (s : LowerState) : Prop :=
  (∀ sv ∈ s.stack, ∀ r ∈ sv.regs, r < s.nextReg) ∧
  (∀ p ∈ s.localReg, p.snd < s.nextReg) ∧
  (∀ p ∈ s.currentReg, p.snd < s.nextReg)

/-- The empty state is trivially well-scoped. -/
theorem LowerState.empty_wellScoped : LowerState.empty.wellScoped := by
  refine ⟨?_, ?_, ?_⟩ <;> intro _ h <;> exact absurd h (by simp [LowerState.empty])

-- ════════════════════════════════════════════════════════════════════
-- popSym / commit helpers
--
-- The slice-1 binop and cmp arms pop two SymVals, commit each into a
-- real register, then alloc a destination. Several scope-validity
-- properties of `popSym` and `commit` are reused across both arms;
-- they live here as named lemmas.
-- ════════════════════════════════════════════════════════════════════

/-- `popSym` is structural: nextReg, localReg, currentReg untouched,
    stack reduced by one. -/
theorem LowerState.popSym_nextReg {s s' : LowerState} {sv : SymVal}
    (h : s.popSym = some (sv, s')) : s'.nextReg = s.nextReg := by
  unfold LowerState.popSym at h
  rcases hs : s.stack with _ | ⟨sv', rs⟩
  · rw [hs] at h; simp at h
  · rw [hs] at h; simp at h
    obtain ⟨_, hs_eq⟩ := h
    rw [← hs_eq]

theorem LowerState.popSym_localReg {s s' : LowerState} {sv : SymVal}
    (h : s.popSym = some (sv, s')) : s'.localReg = s.localReg := by
  unfold LowerState.popSym at h
  rcases hs : s.stack with _ | ⟨sv', rs⟩
  · rw [hs] at h; simp at h
  · rw [hs] at h; simp at h
    obtain ⟨_, hs_eq⟩ := h
    rw [← hs_eq]

theorem LowerState.popSym_currentReg {s s' : LowerState} {sv : SymVal}
    (h : s.popSym = some (sv, s')) : s'.currentReg = s.currentReg := by
  unfold LowerState.popSym at h
  rcases hs : s.stack with _ | ⟨sv', rs⟩
  · rw [hs] at h; simp at h
  · rw [hs] at h; simp at h
    obtain ⟨_, hs_eq⟩ := h
    rw [← hs_eq]

/-- The popped SymVal was on `s.stack` before the pop. -/
theorem LowerState.popSym_sv_mem {s s' : LowerState} {sv : SymVal}
    (h : s.popSym = some (sv, s')) : sv ∈ s.stack := by
  unfold LowerState.popSym at h
  rcases hs : s.stack with _ | ⟨sv', rs⟩
  · rw [hs] at h; simp at h
  · rw [hs] at h; simp at h
    obtain ⟨hsv_eq, _⟩ := h
    -- After `rcases hs : s.stack`, the rcases substituted s.stack → sv'::rs
    -- in the GOAL. So the goal is now `sv ∈ sv' :: rs`. After subst hsv_eq:
    subst hsv_eq
    exact List.mem_cons_self _ _

/-- Post-pop stack is a tail of pre-pop stack. -/
theorem LowerState.popSym_stack_subset {s s' : LowerState} {sv : SymVal}
    (h : s.popSym = some (sv, s')) : ∀ x ∈ s'.stack, x ∈ s.stack := by
  unfold LowerState.popSym at h
  rcases hs : s.stack with _ | ⟨sv', rs⟩
  · rw [hs] at h; simp at h
  · rw [hs] at h; simp at h
    obtain ⟨_, hs_eq⟩ := h
    intro x hx
    rw [← hs_eq] at hx
    -- Same as above: rcases substituted the GOAL; goal is `x ∈ sv' :: rs`.
    exact List.mem_cons_of_mem _ hx

/-- `popSym` preserves wellScoped. -/
theorem LowerState.popSym_preserves_wellScoped {s s' : LowerState} {sv : SymVal}
    (hws : s.wellScoped) (h : s.popSym = some (sv, s')) : s'.wellScoped := by
  obtain ⟨hstk, hloc, hcur⟩ := hws
  have hnr := LowerState.popSym_nextReg h
  have hlr := LowerState.popSym_localReg h
  have hcr := LowerState.popSym_currentReg h
  refine ⟨?_, ?_, ?_⟩
  · intro sv' hsv' r hr
    rw [hnr]
    exact hstk sv' (LowerState.popSym_stack_subset h sv' hsv') r hr
  · intro p hp
    rw [hlr] at hp
    rw [hnr]
    exact hloc p hp
  · intro p hp
    rw [hcr] at hp
    rw [hnr]
    exact hcur p hp

/-- The popped SymVal's registers all lie below `s.nextReg`. -/
theorem LowerState.popSym_sv_regs_lt {s s' : LowerState} {sv : SymVal}
    (hws : s.wellScoped) (h : s.popSym = some (sv, s')) :
    ∀ r ∈ sv.regs, r < s.nextReg := by
  obtain ⟨hstk, _, _⟩ := hws
  exact hstk sv (LowerState.popSym_sv_mem h)

/-- `commit` either leaves nextReg alone (.reg case) or bumps it by 1
    (.i32ConstSym case). Either way, post-state's nextReg ≥ pre-state's. -/
theorem LowerState.commit_nextReg_mono {s s' : LowerState} {sv : SymVal}
    {r : Reg} {ops : List KernelOp}
    (h : s.commit sv = some (r, s', ops)) : s.nextReg ≤ s'.nextReg := by
  unfold LowerState.commit at h
  cases sv with
  | reg r' _ =>
      simp at h
      obtain ⟨_, h_s_eq, _⟩ := h
      subst h_s_eq
      exact Nat.le_refl _
  | i32ConstSym n =>
      simp [LowerState.alloc] at h
      obtain ⟨_, h_s_eq, _⟩ := h
      rw [← h_s_eq]
      exact Nat.le_succ _
  | bufferPtr _ => simp at h
  | scaledIdx _ _ => simp at h
  | bufferAccess _ _ _ => simp at h

/-- `commit` preserves `localReg`. -/
theorem LowerState.commit_localReg {s s' : LowerState} {sv : SymVal}
    {r : Reg} {ops : List KernelOp}
    (h : s.commit sv = some (r, s', ops)) : s'.localReg = s.localReg := by
  unfold LowerState.commit at h
  cases sv with
  | reg r' _ =>
      simp at h
      obtain ⟨_, h_s_eq, _⟩ := h
      rw [← h_s_eq]
  | i32ConstSym n =>
      simp [LowerState.alloc] at h
      obtain ⟨_, h_s_eq, _⟩ := h
      rw [← h_s_eq]
  | bufferPtr _ => simp at h
  | scaledIdx _ _ => simp at h
  | bufferAccess _ _ _ => simp at h

/-- `commit` preserves `currentReg`. -/
theorem LowerState.commit_currentReg {s s' : LowerState} {sv : SymVal}
    {r : Reg} {ops : List KernelOp}
    (h : s.commit sv = some (r, s', ops)) : s'.currentReg = s.currentReg := by
  unfold LowerState.commit at h
  cases sv with
  | reg r' _ =>
      simp at h
      obtain ⟨_, h_s_eq, _⟩ := h
      rw [← h_s_eq]
  | i32ConstSym n =>
      simp [LowerState.alloc] at h
      obtain ⟨_, h_s_eq, _⟩ := h
      rw [← h_s_eq]
  | bufferPtr _ => simp at h
  | scaledIdx _ _ => simp at h
  | bufferAccess _ _ _ => simp at h

/-- `commit` preserves `stack`. -/
theorem LowerState.commit_stack {s s' : LowerState} {sv : SymVal}
    {r : Reg} {ops : List KernelOp}
    (h : s.commit sv = some (r, s', ops)) : s'.stack = s.stack := by
  unfold LowerState.commit at h
  cases sv with
  | reg r' _ =>
      simp at h
      obtain ⟨_, h_s_eq, _⟩ := h
      rw [← h_s_eq]
  | i32ConstSym n =>
      simp [LowerState.alloc] at h
      obtain ⟨_, h_s_eq, _⟩ := h
      rw [← h_s_eq]
  | bufferPtr _ => simp at h
  | scaledIdx _ _ => simp at h
  | bufferAccess _ _ _ => simp at h

/-- `commit` preserves wellScoped. Every field either stays the same
    (stack, localReg, currentReg) or grows (nextReg). -/
theorem LowerState.commit_preserves_wellScoped {s s' : LowerState} {sv : SymVal}
    {r : Reg} {ops : List KernelOp}
    (hws : s.wellScoped) (h : s.commit sv = some (r, s', ops)) :
    s'.wellScoped := by
  obtain ⟨hstk, hloc, hcur⟩ := hws
  have hnr := LowerState.commit_nextReg_mono h
  have hlr := LowerState.commit_localReg h
  have hcr := LowerState.commit_currentReg h
  have hst := LowerState.commit_stack h
  refine ⟨?_, ?_, ?_⟩
  · intro sv' hsv' r' hr'
    rw [hst] at hsv'
    exact Nat.lt_of_lt_of_le (hstk sv' hsv' r' hr') hnr
  · intro p hp
    rw [hlr] at hp
    exact Nat.lt_of_lt_of_le (hloc p hp) hnr
  · intro p hp
    rw [hcr] at hp
    exact Nat.lt_of_lt_of_le (hcur p hp) hnr

/-- Ops emitted by `commit` have no operand reads — `.const`'s
    `usedRegs = []` — so they're scope-valid against any env. -/
theorem LowerState.commit_ops_scopeValid {s s' : LowerState} {sv : SymVal}
    {r : Reg} {ops : List KernelOp}
    (env : List Reg) (h : s.commit sv = some (r, s', ops)) :
    scopeValidOps env ops := by
  unfold LowerState.commit at h
  cases sv with
  | reg r' _ =>
      simp at h
      obtain ⟨_, _, h_ops⟩ := h
      subst h_ops; exact trivial
  | i32ConstSym n =>
      simp [LowerState.alloc] at h
      obtain ⟨_, _, h_ops⟩ := h
      subst h_ops
      refine ⟨?_, trivial⟩
      intro r hr
      simp [KernelOp.usedRegs] at hr
  | bufferPtr _ => simp at h
  | scaledIdx _ _ => simp at h
  | bufferAccess _ _ _ => simp at h

/-- The committed register `r` lies in the post-commit `scopeEnv`,
    assuming the input SymVal came from a wellScoped stack. -/
theorem LowerState.commit_r_mem_scopeEnv {s s' : LowerState} {sv : SymVal}
    {r : Reg} {ops : List KernelOp}
    (hsv : ∀ r' ∈ sv.regs, r' < s.nextReg)
    (h : s.commit sv = some (r, s', ops)) : r ∈ s'.scopeEnv := by
  unfold LowerState.commit at h
  cases sv with
  | reg r' _ =>
      simp at h
      obtain ⟨hreq, hs_eq, _⟩ := h
      subst hreq; rw [← hs_eq]
      rw [LowerState.mem_scopeEnv]
      exact hsv r' (by simp [SymVal.regs])
  | i32ConstSym n =>
      simp [LowerState.alloc] at h
      obtain ⟨hreq, hs_eq, _⟩ := h
      -- r = s.nextReg; s'.nextReg = s.nextReg + 1.
      subst hreq; rw [← hs_eq]
      rw [LowerState.mem_scopeEnv]
      exact Nat.lt_succ_self _
  | bufferPtr _ => simp at h
  | scaledIdx _ _ => simp at h
  | bufferAccess _ _ _ => simp at h

/-- The ops emitted by `commit` extend the env by at most one register,
    namely `r` (the committed register itself). For `.reg`, no extension;
    for `.i32ConstSym`, extends by `r`. In both cases `r ∈ extendEnvOps env ops`. -/
theorem LowerState.commit_r_in_extendEnvOps {s s' : LowerState} {sv : SymVal}
    {r : Reg} {ops : List KernelOp} (env : List Reg)
    (hsv : ∀ r' ∈ sv.regs, r' ∈ env)
    (h : s.commit sv = some (r, s', ops)) :
    r ∈ extendEnvOps env ops := by
  unfold LowerState.commit at h
  cases sv with
  | reg r' _ =>
      simp at h
      obtain ⟨hreq, _, h_ops⟩ := h
      subst hreq; subst h_ops
      -- extendEnvOps env [] = env. Need r' ∈ env from hsv.
      exact hsv r' (by simp [SymVal.regs])
  | i32ConstSym n =>
      simp [LowerState.alloc] at h
      obtain ⟨hreq, _, h_ops⟩ := h
      subst hreq; subst h_ops
      -- extendEnvOps env [.const s.nextReg _] = [s.nextReg].extendEnv env (.const …)
      -- = s.nextReg :: env. Membership trivial.
      show s.nextReg ∈ extendEnvOps env [.const s.nextReg (.u32 (UInt32.ofNat n.toNat))]
      simp [extendEnvOps, extendEnv, KernelOp.definedReg]
  | bufferPtr _ => simp at h
  | scaledIdx _ _ => simp at h
  | bufferAccess _ _ _ => simp at h

-- ════════════════════════════════════════════════════════════════════
-- Primitive state-mutator wellScoped preservation
--
-- `push`, `pushSym`, `alloc`, `setLocalReg`, `setCurrentReg` —
-- each shown to preserve `wellScoped` under the appropriate
-- precondition (reg or sv.regs in scope). Used to chain wellScoped
-- through the multi-step lowering arms (binop/cmp/load/store/
-- localSet/Tee).
-- ════════════════════════════════════════════════════════════════════

/-- `alloc` only bumps nextReg; all prior regs stay in scope under the
    widened bound. -/
theorem LowerState.alloc_preserves_wellScoped {s : LowerState}
    (hws : s.wellScoped) : s.alloc.snd.wellScoped := by
  obtain ⟨hstk, hloc, hcur⟩ := hws
  refine ⟨?_, ?_, ?_⟩
  · intro sv hsv r hr
    show r < s.nextReg + 1
    exact Nat.lt_succ_of_lt (hstk sv hsv r hr)
  · intro p hp
    show p.snd < s.nextReg + 1
    exact Nat.lt_succ_of_lt (hloc p hp)
  · intro p hp
    show p.snd < s.nextReg + 1
    exact Nat.lt_succ_of_lt (hcur p hp)

/-- `push r` preserves wellScoped iff `r < nextReg`. -/
theorem LowerState.push_preserves_wellScoped {s : LowerState} {r : Reg}
    (hws : s.wellScoped) (hr : r < s.nextReg) : (s.push r).wellScoped := by
  obtain ⟨hstk, hloc, hcur⟩ := hws
  refine ⟨?_, hloc, hcur⟩
  intro sv hsv r' hr'
  -- s.push r := { s with stack := SymVal.reg r .u32 :: s.stack }
  show r' < s.nextReg
  simp [LowerState.push] at hsv
  rcases hsv with rfl | hsv
  · -- sv = .reg r .u32; sv.regs = [r]; hr' : r' ∈ [r] ⇒ r' = r
    simp [SymVal.regs] at hr'
    subst hr'; exact hr
  · exact hstk sv hsv r' hr'

/-- `pushSym sv` preserves wellScoped iff every reg in `sv.regs` is in scope. -/
theorem LowerState.pushSym_preserves_wellScoped {s : LowerState} {sv : SymVal}
    (hws : s.wellScoped) (hsv : ∀ r ∈ sv.regs, r < s.nextReg) :
    (s.pushSym sv).wellScoped := by
  obtain ⟨hstk, hloc, hcur⟩ := hws
  refine ⟨?_, hloc, hcur⟩
  intro sv' hsv' r' hr'
  show r' < s.nextReg
  simp [LowerState.pushSym] at hsv'
  rcases hsv' with rfl | hsv'
  · exact hsv r' hr'
  · exact hstk sv' hsv' r' hr'

/-- `setLocalReg i r ty` preserves wellScoped iff `r < nextReg`. -/
theorem LowerState.setLocalReg_preserves_wellScoped {s : LowerState}
    {i : Nat} {r : Reg} {ty : Scalar}
    (hws : s.wellScoped) (hr : r < s.nextReg) :
    (s.setLocalReg i r ty).wellScoped := by
  obtain ⟨hstk, hloc, hcur⟩ := hws
  refine ⟨hstk, ?_, hcur⟩
  intro p hp
  show p.snd < s.nextReg
  -- (s.setLocalReg i r ty).localReg = (i, r) :: s.localReg.filter (·.fst ≠ i)
  -- so p is either (i, r) or comes from the filtered tail (membership in original).
  have hp' : p = (i, r) ∨ p ∈ s.localReg.filter (fun q => q.fst ≠ i) := by
    have : p ∈ (i, r) :: s.localReg.filter (fun q => q.fst ≠ i) := hp
    exact List.mem_cons.mp this
  rcases hp' with rfl | hp'
  · exact hr
  · exact hloc p (List.mem_filter.mp hp').1

/-- `setCurrentReg i r` preserves wellScoped iff `r < nextReg`. -/
theorem LowerState.setCurrentReg_preserves_wellScoped {s : LowerState}
    {i : Nat} {r : Reg}
    (hws : s.wellScoped) (hr : r < s.nextReg) :
    (s.setCurrentReg i r).wellScoped := by
  obtain ⟨hstk, hloc, hcur⟩ := hws
  refine ⟨hstk, hloc, ?_⟩
  intro p hp
  show p.snd < s.nextReg
  have hp' : p = (i, r) ∨ p ∈ s.currentReg.filter (fun q => q.fst ≠ i) := by
    have : p ∈ (i, r) :: s.currentReg.filter (fun q => q.fst ≠ i) := hp
    exact List.mem_cons.mp this
  rcases hp' with rfl | hp'
  · exact hr
  · exact hcur p (List.mem_filter.mp hp').1

-- ════════════════════════════════════════════════════════════════════
-- Per-arm: empty-emit arms
-- ════════════════════════════════════════════════════════════════════

/-- `i32Const n` emits no IR ops — trivially scope-valid against any env. -/
theorem lowerInstr_i32Const_scopeValid (s : LowerState) (n : Int) :
    ∀ {s' : LowerState} {ops : List KernelOp},
      lowerInstr s (.i32Const n) = some (s', ops) →
      scopeValidOps s'.scopeEnv ops := by
  intro s' ops h
  simp [lowerInstr] at h
  rcases h with ⟨_, h_ops⟩
  subst h_ops
  exact trivial

/-- `nop` emits no IR ops. -/
theorem lowerInstr_nop_scopeValid (s : LowerState) :
    ∀ {s' : LowerState} {ops : List KernelOp},
      lowerInstr s .nop = some (s', ops) →
      scopeValidOps s'.scopeEnv ops := by
  intro s' ops h
  simp [lowerInstr] at h
  rcases h with ⟨_, h_ops⟩
  subst h_ops
  exact trivial

/-- `wreturn` emits no IR ops. -/
theorem lowerInstr_wreturn_scopeValid (s : LowerState) :
    ∀ {s' : LowerState} {ops : List KernelOp},
      lowerInstr s .wreturn = some (s', ops) →
      scopeValidOps s'.scopeEnv ops := by
  intro s' ops h
  simp [lowerInstr] at h
  rcases h with ⟨_, h_ops⟩
  subst h_ops
  exact trivial

/-- `drop` emits no IR ops (the popped SymVal is discarded). -/
theorem lowerInstr_drop_scopeValid (s : LowerState) :
    ∀ {s' : LowerState} {ops : List KernelOp},
      lowerInstr s .drop = some (s', ops) →
      scopeValidOps s'.scopeEnv ops := by
  intro s' ops h
  simp [lowerInstr] at h
  rcases hpop : s.popSym with _ | ⟨sv, s1⟩
  · simp [hpop] at h
  · simp [hpop] at h
    rcases h with ⟨_, h_ops⟩
    subst h_ops
    exact trivial

-- ════════════════════════════════════════════════════════════════════
-- Per-arm: `localGet` non-buffer path
--
-- Emits `[.copy fresh source]` where `source = currentReg[i]` (with
-- `localReg[i]` fallback) and `fresh = s.nextReg` (from alloc). Under
-- `wellScoped s`, `source < s.nextReg`, so `source ∈ s'.scopeEnv`.
-- ════════════════════════════════════════════════════════════════════

/-- Looking up a key in an association list returns a value that came
    from one of the list's pairs. -/
private theorem List.find?_map_snd_mem {α β}
    {l : List (α × β)} {p : α × β → Bool} {b : β}
    (h : (l.find? p).map Prod.snd = some b) :
    ∃ a, (a, b) ∈ l := by
  -- Pull out the pair `find?` succeeded with.
  rcases hf : l.find? p with _ | ⟨a, b'⟩
  · rw [hf] at h; simp at h
  · rw [hf] at h
    simp at h
    subst h
    exact ⟨a, List.mem_of_find?_eq_some hf⟩

/-- If `wellScoped s`, then `lookupLocal i = some r` implies `r ∈ scopeEnv s`. -/
theorem LowerState.lookupLocal_mem_scopeEnv {s : LowerState} {i : Nat} {r : Reg}
    (hws : s.wellScoped) (h : s.lookupLocal i = some r) :
    r ∈ s.scopeEnv := by
  obtain ⟨_, hloc, _⟩ := hws
  unfold LowerState.lookupLocal at h
  obtain ⟨_, hmem⟩ := List.find?_map_snd_mem h
  rw [LowerState.mem_scopeEnv]
  exact hloc _ hmem

/-- Same for `lookupCurrentReg`. -/
theorem LowerState.lookupCurrentReg_mem_scopeEnv {s : LowerState} {i : Nat} {r : Reg}
    (hws : s.wellScoped) (h : s.lookupCurrentReg i = some r) :
    r ∈ s.scopeEnv := by
  obtain ⟨_, _, hcur⟩ := hws
  unfold LowerState.lookupCurrentReg at h
  obtain ⟨_, hmem⟩ := List.find?_map_snd_mem h
  rw [LowerState.mem_scopeEnv]
  exact hcur _ hmem

/-- `localGet i` non-buffer path: emits `[.copy fresh source]`. Given
    `wellScoped s`, the `source` is in `s.scopeEnv` (hence in the
    post-state's scopeEnv by monotonicity), so the emitted single op
    is scope-valid against the post-state's scopeEnv. -/
theorem lowerInstr_localGet_nonbuffer_scopeValid
    {s s' : LowerState} {i : Nat} {ops : List KernelOp}
    (hws : s.wellScoped)
    (hbuf : s.lookupBufferSlot i = none)
    (h : lowerInstr s (.localGet i) = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops := by
  simp [lowerInstr, hbuf, Option.bind_eq_bind] at h
  -- After hbuf simp, h is the match on `(currentReg i).orElse fun _ => lookupLocal i`.
  rcases hcur : s.lookupCurrentReg i with _ | curReg
  · -- currentReg miss → fall back to localReg
    simp [hcur, Option.orElse] at h
    rcases hloc : s.lookupLocal i with _ | stable
    · simp [hloc] at h
    · -- localReg hit: source = stable, fresh = s.nextReg.
      -- ops := [.copy s.nextReg stable]
      have hmem_s : stable ∈ s.scopeEnv :=
        LowerState.lookupLocal_mem_scopeEnv hws hloc
      have hmem_alloc : stable ∈ s.alloc.snd.scopeEnv :=
        LowerState.scopeEnv_subset_alloc s hmem_s
      simp [hloc, LowerState.alloc, LowerState.push] at h
      obtain ⟨h_s_eq, h_ops⟩ := h
      subst h_ops
      subst h_s_eq
      refine ⟨?_, trivial⟩
      intro r hr
      simp [KernelOp.usedRegs] at hr
      subst hr
      -- Goal: stable ∈ ({ s.alloc.snd with stack := … }).scopeEnv
      -- scopeEnv depends only on nextReg, which equals s.alloc.snd.nextReg.
      exact hmem_alloc
  · -- currentReg hit → source = curReg, fresh = s.nextReg.
    have hmem_s : curReg ∈ s.scopeEnv :=
      LowerState.lookupCurrentReg_mem_scopeEnv hws hcur
    have hmem_alloc : curReg ∈ s.alloc.snd.scopeEnv :=
      LowerState.scopeEnv_subset_alloc s hmem_s
    simp [hcur, Option.orElse, LowerState.alloc, LowerState.push] at h
    obtain ⟨h_s_eq, h_ops⟩ := h
    subst h_ops
    subst h_s_eq
    refine ⟨?_, trivial⟩
    intro r hr
    simp [KernelOp.usedRegs] at hr
    subst hr
    exact hmem_alloc

-- ════════════════════════════════════════════════════════════════════
-- wellScoped preservation for closed arms
--
-- The eventual list-level scope-validity theorem inducts over an
-- instruction stream. To chain step N+1's `wellScoped` precondition
-- from step N's post-state, every closed arm must show wellScoped is
-- preserved. The five lemmas below cover the same five arms whose
-- scope-validity of *emitted ops* was proved above.
-- ════════════════════════════════════════════════════════════════════

/-- `i32Const n` only mutates `stack` (pushes `.i32ConstSym n`, regs = []),
    leaving `nextReg`, `localReg`, `currentReg` unchanged. -/
theorem lowerInstr_i32Const_preserves_wellScoped
    {s s' : LowerState} {n : Int} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerInstr s (.i32Const n) = some (s', ops)) :
    s'.wellScoped := by
  simp [lowerInstr] at h
  obtain ⟨h_s_eq, _⟩ := h
  subst h_s_eq
  obtain ⟨hstk, hloc, hcur⟩ := hws
  refine ⟨?_, hloc, hcur⟩
  intro sv hsv r hr
  simp at hsv
  rcases hsv with rfl | hsv
  · exact absurd hr (by simp [SymVal.regs])
  · exact hstk sv hsv r hr

/-- `nop` doesn't mutate state. -/
theorem lowerInstr_nop_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerInstr s .nop = some (s', ops)) :
    s'.wellScoped := by
  simp [lowerInstr] at h
  obtain ⟨h_s_eq, _⟩ := h
  subst h_s_eq
  exact hws

/-- `wreturn` doesn't mutate state. -/
theorem lowerInstr_wreturn_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerInstr s .wreturn = some (s', ops)) :
    s'.wellScoped := by
  simp [lowerInstr] at h
  obtain ⟨h_s_eq, _⟩ := h
  subst h_s_eq
  exact hws

/-- `drop` strictly shrinks `stack` (pops the head) and leaves all
    other fields untouched. Regs in the smaller stack are still bounded. -/
theorem lowerInstr_drop_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerInstr s .drop = some (s', ops)) :
    s'.wellScoped := by
  simp [lowerInstr] at h
  rcases hpop : s.popSym with _ | ⟨sv, s1⟩
  · simp [hpop] at h
  · simp [hpop] at h
    obtain ⟨h_s_eq, _⟩ := h
    subst h_s_eq
    -- s1 := {s with stack := tail of s.stack}. wellScoped on a smaller
    -- stack follows from wellScoped on the full one.
    unfold LowerState.popSym at hpop
    rcases hs : s.stack with _ | ⟨svh, rs⟩
    · rw [hs] at hpop; simp at hpop
    · rw [hs] at hpop
      simp at hpop
      obtain ⟨_, hs1_eq⟩ := hpop
      obtain ⟨hstk, hloc, hcur⟩ := hws
      refine ⟨?_, ?_, ?_⟩
      · intro sv' hsv' r hr
        -- s1.stack ⊆ s.stack as a list (tail of it).
        rw [← hs1_eq] at hsv'
        -- hsv' : sv' ∈ rs
        have : sv' ∈ s.stack := by rw [hs]; exact List.mem_cons_of_mem _ hsv'
        -- s1.nextReg = s.nextReg
        rw [← hs1_eq]
        exact hstk sv' this r hr
      · intro p hp
        rw [← hs1_eq] at hp
        rw [← hs1_eq]
        exact hloc p hp
      · intro p hp
        rw [← hs1_eq] at hp
        rw [← hs1_eq]
        exact hcur p hp

/-- `localGet i` (buffer arm): pushes `.bufferPtr slot` (regs = [])
    onto `stack`, no other mutation. -/
theorem lowerInstr_localGet_buffer_preserves_wellScoped
    {s s' : LowerState} {i slot : Nat} {ops : List KernelOp}
    (hws : s.wellScoped)
    (hbuf : s.lookupBufferSlot i = some slot)
    (h : lowerInstr s (.localGet i) = some (s', ops)) :
    s'.wellScoped := by
  simp [lowerInstr, hbuf, LowerState.pushSym] at h
  obtain ⟨h_s_eq, _⟩ := h
  subst h_s_eq
  obtain ⟨hstk, hloc, hcur⟩ := hws
  refine ⟨?_, hloc, hcur⟩
  intro sv hsv r hr
  simp at hsv
  rcases hsv with rfl | hsv
  · exact absurd hr (by simp [SymVal.regs])
  · exact hstk sv hsv r hr

/-- `localGet i` (non-buffer arm): allocates a fresh reg and pushes
    `.reg fresh .u32`. `nextReg` bumps by 1; fresh = old nextReg fits
    the new bound; pre-existing regs lift through the +1 widening. -/
theorem lowerInstr_localGet_nonbuffer_preserves_wellScoped
    {s s' : LowerState} {i : Nat} {ops : List KernelOp}
    (hws : s.wellScoped)
    (hbuf : s.lookupBufferSlot i = none)
    (h : lowerInstr s (.localGet i) = some (s', ops)) :
    s'.wellScoped := by
  simp [lowerInstr, hbuf, Option.bind_eq_bind] at h
  rcases hcur : s.lookupCurrentReg i with _ | curReg
  · simp [hcur, Option.orElse] at h
    rcases hloc : s.lookupLocal i with _ | stable
    · simp [hloc] at h
    · simp [hloc, LowerState.alloc, LowerState.push] at h
      obtain ⟨h_s_eq, _⟩ := h
      subst h_s_eq
      obtain ⟨hstk, hlocws, hcurws⟩ := hws
      refine ⟨?_, ?_, ?_⟩
      · intro sv hsv r hr
        simp at hsv
        rcases hsv with rfl | hsv
        · -- head is `.reg s.nextReg .u32`; regs = [s.nextReg]
          simp [SymVal.regs] at hr
          subst hr
          exact Nat.lt_succ_self _
        · -- tail: from `hstk`, regs < s.nextReg < s.nextReg + 1
          exact Nat.lt_succ_of_lt (hstk sv hsv r hr)
      · intro p hp
        exact Nat.lt_succ_of_lt (hlocws p hp)
      · intro p hp
        exact Nat.lt_succ_of_lt (hcurws p hp)
  · simp [hcur, Option.orElse, LowerState.alloc, LowerState.push] at h
    obtain ⟨h_s_eq, _⟩ := h
    subst h_s_eq
    obtain ⟨hstk, hlocws, hcurws⟩ := hws
    refine ⟨?_, ?_, ?_⟩
    · intro sv hsv r hr
      simp at hsv
      rcases hsv with rfl | hsv
      · simp [SymVal.regs] at hr
        subst hr
        exact Nat.lt_succ_self _
      · exact Nat.lt_succ_of_lt (hstk sv hsv r hr)
    · intro p hp
      exact Nat.lt_succ_of_lt (hlocws p hp)
    · intro p hp
      exact Nat.lt_succ_of_lt (hcurws p hp)

/-- Unified `localGet` wellScoped-preservation: combines the two arms. -/
theorem lowerInstr_localGet_preserves_wellScoped
    {s s' : LowerState} {i : Nat} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerInstr s (.localGet i) = some (s', ops)) :
    s'.wellScoped := by
  rcases hbuf : s.lookupBufferSlot i with _ | slot
  · exact lowerInstr_localGet_nonbuffer_preserves_wellScoped hws hbuf h
  · exact lowerInstr_localGet_buffer_preserves_wellScoped hws hbuf h

/-- `localGet i` buffer-typed path: when `lookupBufferSlot i = some slot`,
    the arm emits no IR — it just pushes `SymVal.bufferPtr slot` onto
    the stack symbolically. Trivially scope-valid against any env. -/
theorem lowerInstr_localGet_buffer_scopeValid
    {s s' : LowerState} {i slot : Nat} {ops : List KernelOp}
    (hbuf : s.lookupBufferSlot i = some slot)
    (h : lowerInstr s (.localGet i) = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops := by
  simp [lowerInstr, hbuf] at h
  obtain ⟨_, h_ops⟩ := h
  subst h_ops
  exact trivial

/-- Unified `localGet i` theorem: regardless of whether the local is a
    buffer parameter or a plain scalar, the emitted ops are scope-valid
    against the post-state's scopeEnv, given `wellScoped s`. -/
theorem lowerInstr_localGet_scopeValid
    {s s' : LowerState} {i : Nat} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerInstr s (.localGet i) = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops := by
  rcases hbuf : s.lookupBufferSlot i with _ | slot
  · exact lowerInstr_localGet_nonbuffer_scopeValid hws hbuf h
  · exact lowerInstr_localGet_buffer_scopeValid hbuf h

-- ════════════════════════════════════════════════════════════════════
-- Per-arm: lowerI32Bin (binop family)
--
-- Emits opsA ++ opsB ++ [.binOp dst ra rb op .u32] where:
--   opsA, opsB come from the two `commit` calls (each either [] or a
--     single .const with no operand reads).
--   ra, rb are committed registers — both lie in s'.scopeEnv via
--     commit_r_mem_scopeEnv + monotonicity.
--   dst is the freshly-allocated result reg.
--
-- This unlocks i32Sub/Mul/And/Or/Xor/ShrU/DivU/RemU (8 arms that
-- dispatch directly to lowerI32Bin) and the non-buffer fallthroughs
-- of lowerI32Add and lowerI32Shl.
-- ════════════════════════════════════════════════════════════════════

theorem lowerI32Bin_scopeValid
    {s s' : LowerState} {op : Quanta.KOps.BinOp} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerI32Bin s op = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops := by
  unfold lowerI32Bin at h
  -- Decompose the do-block step by step.
  rcases hb : s.popSym with _ | ⟨svb, s1⟩
  · simp [hb] at h
  simp only [hb, Option.bind_eq_bind, Option.some_bind] at h
  rcases ha : s1.popSym with _ | ⟨sva, s2⟩
  · simp [ha] at h
  simp only [ha, Option.bind_eq_bind, Option.some_bind] at h
  rcases hca : s2.commit sva with _ | ⟨ra, s3, opsA⟩
  · simp [hca] at h
  simp only [hca, Option.some_bind] at h
  rcases hcb : s3.commit svb with _ | ⟨rb, s4, opsB⟩
  · simp [hcb] at h
  simp only [hcb, Option.some_bind] at h
  -- h : some (let (dst, s5) := s4.alloc; ({s5 with stack := .reg dst .u32 :: s5.stack},
  --                                       opsA ++ opsB ++ [.binOp dst ra rb op .u32]))
  --       = some (s', ops)
  simp [LowerState.alloc, LowerState.push] at h
  obtain ⟨h_s_eq, h_ops⟩ := h
  -- Threading wellScoped through the four primitive steps.
  have hws1 : s1.wellScoped := LowerState.popSym_preserves_wellScoped hws hb
  have hws2 : s2.wellScoped := LowerState.popSym_preserves_wellScoped hws1 ha
  -- nextReg chain.
  have hnr_pop1 : s1.nextReg = s.nextReg := LowerState.popSym_nextReg hb
  have hnr_pop2 : s2.nextReg = s1.nextReg := LowerState.popSym_nextReg ha
  have hnr_ca : s2.nextReg ≤ s3.nextReg := LowerState.commit_nextReg_mono hca
  have hnr_cb : s3.nextReg ≤ s4.nextReg := LowerState.commit_nextReg_mono hcb
  -- sva came from s1.stack; sva.regs < s1.nextReg.
  have hsva_lt_s1 : ∀ r ∈ sva.regs, r < s1.nextReg :=
    LowerState.popSym_sv_regs_lt hws1 ha
  -- Lift to s2.nextReg via hnr_pop2.
  have hsva_lt_s2 : ∀ r ∈ sva.regs, r < s2.nextReg := by
    intro r hr; rw [hnr_pop2]; exact hsva_lt_s1 r hr
  -- svb came from s.stack; svb.regs < s.nextReg.
  have hsvb_lt_s : ∀ r ∈ svb.regs, r < s.nextReg :=
    LowerState.popSym_sv_regs_lt hws hb
  -- ra ∈ s3.scopeEnv via commit_r_mem_scopeEnv at the s2 → s3 step.
  have hra_s3 : ra ∈ s3.scopeEnv :=
    LowerState.commit_r_mem_scopeEnv hsva_lt_s2 hca
  -- svb.regs < s3.nextReg via s.nextReg = s1.nextReg = s2.nextReg ≤ s3.nextReg.
  have hsvb_lt_s3 : ∀ r ∈ svb.regs, r < s3.nextReg := by
    intro r hr
    have h0 := hsvb_lt_s r hr
    have h1 : r < s1.nextReg := by rw [hnr_pop1]; exact h0
    have h2 : r < s2.nextReg := by rw [hnr_pop2]; exact h1
    exact Nat.lt_of_lt_of_le h2 hnr_ca
  -- rb ∈ s4.scopeEnv from commit_r_mem_scopeEnv at s3 → s4.
  have hrb_s4 : rb ∈ s4.scopeEnv :=
    LowerState.commit_r_mem_scopeEnv hsvb_lt_s3 hcb
  -- ra lifts to s4.scopeEnv via the scope monotonicity through hnr_cb.
  have hra_s4 : ra ∈ s4.scopeEnv :=
    LowerState.scopeEnv_subset_of_nextReg_le hnr_cb hra_s3
  -- Now finalize the goal.
  subst h_ops
  subst h_s_eq
  -- Goal:
  --   scopeValidOps {…, nextReg := s4.nextReg + 1, …}.scopeEnv
  --     (opsA ++ (opsB ++ [.binOp s4.nextReg ra rb op .u32]))
  -- Abbreviate the env once to keep the proof readable.
  let envS' : List Reg := List.range (s4.nextReg + 1)
  have hra_envS' : ra ∈ envS' := by
    rw [LowerState.mem_scopeEnv] at hra_s4
    show ra ∈ List.range (s4.nextReg + 1)
    rw [List.mem_range]
    exact Nat.lt_succ_of_lt hra_s4
  have hrb_envS' : rb ∈ envS' := by
    rw [LowerState.mem_scopeEnv] at hrb_s4
    show rb ∈ List.range (s4.nextReg + 1)
    rw [List.mem_range]
    exact Nat.lt_succ_of_lt hrb_s4
  -- The goal's scopeEnv unfolds to envS' (both are List.range (s4.nextReg + 1)).
  show scopeValidOps envS' (opsA ++ (opsB ++ [KernelOp.binOp s4.nextReg ra rb op .u32]))
  apply Quanta.KOps.KernelOp.scopeValidOps_append
  · exact LowerState.commit_ops_scopeValid envS' hca
  · apply Quanta.KOps.KernelOp.scopeValidOps_append
    · exact LowerState.commit_ops_scopeValid (extendEnvOps envS' opsA) hcb
    · refine ⟨?_, trivial⟩
      intro r hr
      simp [KernelOp.usedRegs] at hr
      have hsuper_a : envS' ⊆ extendEnvOps envS' opsA :=
        Quanta.KOps.KernelOp.extendEnvOps_super envS' opsA
      have hsuper_b :
          extendEnvOps envS' opsA ⊆ extendEnvOps (extendEnvOps envS' opsA) opsB :=
        Quanta.KOps.KernelOp.extendEnvOps_super (extendEnvOps envS' opsA) opsB
      rcases hr with rfl | rfl
      · exact hsuper_b (hsuper_a hra_envS')
      · exact hsuper_b (hsuper_a hrb_envS')

-- ════════════════════════════════════════════════════════════════════
-- Per-arm wrappers for the binop family
--
-- These dispatch directly to lowerI32Bin in lowerInstr. Each is a
-- one-line `exact lowerI32Bin_scopeValid hws h`.
-- ════════════════════════════════════════════════════════════════════

theorem lowerInstr_i32Sub_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32Sub = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops :=
  lowerI32Bin_scopeValid hws h

theorem lowerInstr_i32Mul_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32Mul = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops :=
  lowerI32Bin_scopeValid hws h

theorem lowerInstr_i32And_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32And = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops :=
  lowerI32Bin_scopeValid hws h

theorem lowerInstr_i32Or_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32Or = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops :=
  lowerI32Bin_scopeValid hws h

theorem lowerInstr_i32Xor_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32Xor = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops :=
  lowerI32Bin_scopeValid hws h

theorem lowerInstr_i32ShrU_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32ShrU = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops :=
  lowerI32Bin_scopeValid hws h

theorem lowerInstr_i32DivU_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32DivU = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops :=
  lowerI32Bin_scopeValid hws h

theorem lowerInstr_i32RemU_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32RemU = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops :=
  lowerI32Bin_scopeValid hws h

-- ════════════════════════════════════════════════════════════════════
-- Per-arm: i32Shl (buffer-pattern arm + lowerI32Bin fallback)
--
-- lowerI32Shl pattern-matches on stack shape:
--   .i32ConstSym k :: .reg base _ :: rest  →  emit [], push scaledIdx
--   otherwise                              →  lowerI32Bin .shl
-- The buffer-pattern arm emits no ops (trivially scope-valid); the
-- fallback inherits from lowerI32Bin_scopeValid.
-- ════════════════════════════════════════════════════════════════════

theorem lowerI32Shl_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerI32Shl s = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops := by
  unfold lowerI32Shl at h
  rcases hs : s.stack with _ | ⟨sv1, rs⟩
  · rw [hs] at h
    exact lowerI32Bin_scopeValid hws h
  · rw [hs] at h
    cases sv1 with
    | i32ConstSym k =>
        cases rs with
        | nil => exact lowerI32Bin_scopeValid hws h
        | cons sv2 rs2 =>
            cases sv2 with
            | reg base ty =>
                -- Buffer-pattern arm: emit [], push scaledIdx.
                simp at h
                obtain ⟨_, h_ops⟩ := h
                subst h_ops; exact trivial
            | i32ConstSym _ => exact lowerI32Bin_scopeValid hws h
            | bufferPtr _ => exact lowerI32Bin_scopeValid hws h
            | scaledIdx _ _ => exact lowerI32Bin_scopeValid hws h
            | bufferAccess _ _ _ => exact lowerI32Bin_scopeValid hws h
    | reg _ _ => exact lowerI32Bin_scopeValid hws h
    | bufferPtr _ => exact lowerI32Bin_scopeValid hws h
    | scaledIdx _ _ => exact lowerI32Bin_scopeValid hws h
    | bufferAccess _ _ _ => exact lowerI32Bin_scopeValid hws h

theorem lowerInstr_i32Shl_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32Shl = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops :=
  lowerI32Shl_scopeValid hws h

-- ════════════════════════════════════════════════════════════════════
-- Per-arm: i32Add (two buffer-pattern arms + lowerI32Bin fallback)
-- ════════════════════════════════════════════════════════════════════

theorem lowerI32Add_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerI32Add s = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops := by
  unfold lowerI32Add at h
  rcases hs : s.stack with _ | ⟨sv1, rs⟩
  · rw [hs] at h; exact lowerI32Bin_scopeValid hws h
  · rw [hs] at h
    cases sv1 with
    | scaledIdx base scale =>
        cases rs with
        | nil => exact lowerI32Bin_scopeValid hws h
        | cons sv2 rs2 =>
            cases sv2 with
            | bufferPtr slot =>
                simp at h
                obtain ⟨_, h_ops⟩ := h
                subst h_ops; exact trivial
            | reg _ _ => exact lowerI32Bin_scopeValid hws h
            | i32ConstSym _ => exact lowerI32Bin_scopeValid hws h
            | scaledIdx _ _ => exact lowerI32Bin_scopeValid hws h
            | bufferAccess _ _ _ => exact lowerI32Bin_scopeValid hws h
    | bufferPtr slot =>
        cases rs with
        | nil => exact lowerI32Bin_scopeValid hws h
        | cons sv2 rs2 =>
            cases sv2 with
            | scaledIdx base scale =>
                simp at h
                obtain ⟨_, h_ops⟩ := h
                subst h_ops; exact trivial
            | reg _ _ => exact lowerI32Bin_scopeValid hws h
            | i32ConstSym _ => exact lowerI32Bin_scopeValid hws h
            | bufferPtr _ => exact lowerI32Bin_scopeValid hws h
            | bufferAccess _ _ _ => exact lowerI32Bin_scopeValid hws h
    | reg _ _ => exact lowerI32Bin_scopeValid hws h
    | i32ConstSym _ => exact lowerI32Bin_scopeValid hws h
    | bufferAccess _ _ _ => exact lowerI32Bin_scopeValid hws h

theorem lowerInstr_i32Add_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32Add = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops :=
  lowerI32Add_scopeValid hws h

-- ════════════════════════════════════════════════════════════════════
-- Per-arm: lowerI32Cmp (compare family)
--
-- Same structure as lowerI32Bin but emits TWO trailing ops:
--   [.cmp boolReg ra rb op .bool, .cast dst boolReg .bool .u32]
-- The cast's `boolReg` operand is the immediately-preceding cmp's
-- defined reg, so scopeValid through extendEnv on the cmp op.
-- Unlocks all six i32 compare arms.
-- ════════════════════════════════════════════════════════════════════

theorem lowerI32Cmp_scopeValid
    {s s' : LowerState} {op : Quanta.KOps.CmpOp} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerI32Cmp s op = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops := by
  unfold lowerI32Cmp at h
  rcases hb : s.popSym with _ | ⟨svb, s1⟩
  · simp [hb] at h
  simp only [hb, Option.bind_eq_bind, Option.some_bind] at h
  rcases ha : s1.popSym with _ | ⟨sva, s2⟩
  · simp [ha] at h
  simp only [ha, Option.bind_eq_bind, Option.some_bind] at h
  rcases hca : s2.commit sva with _ | ⟨ra, s3, opsA⟩
  · simp [hca] at h
  simp only [hca, Option.some_bind] at h
  rcases hcb : s3.commit svb with _ | ⟨rb, s4, opsB⟩
  · simp [hcb] at h
  simp only [hcb, Option.some_bind] at h
  simp [LowerState.alloc, LowerState.push] at h
  obtain ⟨h_s_eq, h_ops⟩ := h
  -- Thread wellScoped.
  have hws1 : s1.wellScoped := LowerState.popSym_preserves_wellScoped hws hb
  have hws2 : s2.wellScoped := LowerState.popSym_preserves_wellScoped hws1 ha
  -- nextReg chain.
  have hnr_pop1 : s1.nextReg = s.nextReg := LowerState.popSym_nextReg hb
  have hnr_pop2 : s2.nextReg = s1.nextReg := LowerState.popSym_nextReg ha
  have hnr_ca : s2.nextReg ≤ s3.nextReg := LowerState.commit_nextReg_mono hca
  have hnr_cb : s3.nextReg ≤ s4.nextReg := LowerState.commit_nextReg_mono hcb
  -- ra ∈ s3.scopeEnv → s4.scopeEnv.
  have hsva_lt_s1 : ∀ r ∈ sva.regs, r < s1.nextReg :=
    LowerState.popSym_sv_regs_lt hws1 ha
  have hsva_lt_s2 : ∀ r ∈ sva.regs, r < s2.nextReg := by
    intro r hr; rw [hnr_pop2]; exact hsva_lt_s1 r hr
  have hsvb_lt_s : ∀ r ∈ svb.regs, r < s.nextReg :=
    LowerState.popSym_sv_regs_lt hws hb
  have hra_s3 : ra ∈ s3.scopeEnv :=
    LowerState.commit_r_mem_scopeEnv hsva_lt_s2 hca
  have hsvb_lt_s3 : ∀ r ∈ svb.regs, r < s3.nextReg := by
    intro r hr
    have h0 := hsvb_lt_s r hr
    have h1 : r < s1.nextReg := by rw [hnr_pop1]; exact h0
    have h2 : r < s2.nextReg := by rw [hnr_pop2]; exact h1
    exact Nat.lt_of_lt_of_le h2 hnr_ca
  have hrb_s4 : rb ∈ s4.scopeEnv :=
    LowerState.commit_r_mem_scopeEnv hsvb_lt_s3 hcb
  have hra_s4 : ra ∈ s4.scopeEnv :=
    LowerState.scopeEnv_subset_of_nextReg_le hnr_cb hra_s3
  subst h_ops
  subst h_s_eq
  -- s'.nextReg = s4.nextReg + 1 + 1 = s4.nextReg + 2 (two allocs).
  -- boolReg = s4.nextReg; dst = s4.nextReg + 1; both < s4.nextReg + 2.
  let envS' : List Reg := List.range (s4.nextReg + 1 + 1)
  have hra_envS' : ra ∈ envS' := by
    rw [LowerState.mem_scopeEnv] at hra_s4
    show ra ∈ List.range _
    rw [List.mem_range]
    exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt hra_s4)
  have hrb_envS' : rb ∈ envS' := by
    rw [LowerState.mem_scopeEnv] at hrb_s4
    show rb ∈ List.range _
    rw [List.mem_range]
    exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt hrb_s4)
  show scopeValidOps envS'
    (opsA ++ (opsB ++
      [KernelOp.cmp s4.nextReg ra rb op .bool,
       KernelOp.cast (s4.nextReg + 1) s4.nextReg .bool .u32]))
  apply Quanta.KOps.KernelOp.scopeValidOps_append
  · exact LowerState.commit_ops_scopeValid envS' hca
  · apply Quanta.KOps.KernelOp.scopeValidOps_append
    · exact LowerState.commit_ops_scopeValid (extendEnvOps envS' opsA) hcb
    · -- Two trailing ops. First .cmp: uses ra, rb (both in env).
      -- Second .cast: uses s4.nextReg, which was just defined by the cmp.
      have hsuper_a : envS' ⊆ extendEnvOps envS' opsA :=
        Quanta.KOps.KernelOp.extendEnvOps_super envS' opsA
      have hsuper_b :
          extendEnvOps envS' opsA ⊆ extendEnvOps (extendEnvOps envS' opsA) opsB :=
        Quanta.KOps.KernelOp.extendEnvOps_super (extendEnvOps envS' opsA) opsB
      let envCmp : List Reg := extendEnvOps (extendEnvOps envS' opsA) opsB
      have hra_cmp : ra ∈ envCmp := hsuper_b (hsuper_a hra_envS')
      have hrb_cmp : rb ∈ envCmp := hsuper_b (hsuper_a hrb_envS')
      refine ⟨?_, ?_, trivial⟩
      · -- .cmp s4.nextReg ra rb op .bool: usedRegs = [ra, rb].
        intro r hr
        simp [KernelOp.usedRegs] at hr
        rcases hr with rfl | rfl
        · exact hra_cmp
        · exact hrb_cmp
      · -- .cast (s4.nextReg + 1) s4.nextReg .bool .u32 against env extended
        -- by the cmp. cmp's definedReg = some s4.nextReg, so the env
        -- becomes s4.nextReg :: envCmp. cast uses [s4.nextReg].
        intro r hr
        simp [KernelOp.usedRegs] at hr
        subst hr
        -- Goal: s4.nextReg ∈ extendEnv envCmp (.cmp s4.nextReg …).
        show s4.nextReg ∈ extendEnv envCmp (.cmp s4.nextReg ra rb op .bool)
        simp [extendEnv, KernelOp.definedReg]

-- ════════════════════════════════════════════════════════════════════
-- Per-arm: lowerI32Load (single buffer-pattern arm)
--
-- Only succeeds when top of stack is `bufferAccess slot base 4`.
-- Emits [.load dst slot base .u32] where dst = s.nextReg (just
-- allocated). usedRegs = [base]; base was a SymVal reg on s.stack,
-- so wellScoped s ⇒ base < s.nextReg ⇒ base < s'.nextReg.
-- ════════════════════════════════════════════════════════════════════

theorem lowerI32Load_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerI32Load s = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops := by
  unfold lowerI32Load at h
  rcases hs : s.stack with _ | ⟨sv, rs⟩
  · rw [hs] at h; simp at h
  · rw [hs] at h
    cases sv with
    | bufferAccess slot base scale =>
        by_cases hscale : scale = 4
        · subst hscale
          simp [LowerState.alloc] at h
          obtain ⟨h_s_eq, h_ops⟩ := h
          -- base ∈ s.scopeEnv from wellScoped + sv on stack.
          obtain ⟨hstk, _, _⟩ := hws
          have hbase_in : base < s.nextReg := by
            have hsv_mem : SymVal.bufferAccess slot base 4 ∈ s.stack := by
              rw [hs]; exact List.mem_cons_self _ _
            exact hstk _ hsv_mem base (by simp [SymVal.regs])
          subst h_ops; subst h_s_eq
          -- Goal: scopeValidOps s'.scopeEnv [.load s.nextReg slot base .u32]
          -- where s'.nextReg = s.nextReg + 1.
          have hbase_s' : base < s.nextReg + 1 := Nat.lt_succ_of_lt hbase_in
          refine ⟨?_, trivial⟩
          intro r hr
          simp [KernelOp.usedRegs] at hr
          -- hr : r = base (or base = r). Use it both ways.
          show r ∈ List.range (s.nextReg + 1)
          rw [List.mem_range]
          rcases hr with rfl
          exact hbase_s'
        · -- scale ≠ 4: the match arm doesn't fire; lowerI32Load returns none.
          exfalso
          split at h
          · rename_i _ _ _ _ hp
            cases hp
            exact hscale rfl
          · exact Option.noConfusion h
    | reg _ _ => simp at h
    | i32ConstSym _ => simp at h
    | bufferPtr _ => simp at h
    | scaledIdx _ _ => simp at h

theorem lowerInstr_i32Load_scopeValid
    {s s' : LowerState} {offset align : Nat} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s (.i32Load offset align) = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops :=
  lowerI32Load_scopeValid hws h

-- ════════════════════════════════════════════════════════════════════
-- Per-arm: lowerI32Store
--
-- pops val (top) then addr; commits val into src; emits
--   opsCommit ++ [.store slot base src .u32]
-- against `addr = .bufferAccess slot base 4`. opsCommit is
-- scope-valid against any env (commit_ops_scopeValid); the trailing
-- .store needs base + src in env. base came from a SymVal on
-- s1.stack ⊆ s.stack; src came out of commit at s3.
-- ════════════════════════════════════════════════════════════════════

theorem lowerI32Store_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerI32Store s = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops := by
  unfold lowerI32Store at h
  rcases h1 : s.popSym with _ | ⟨sv_val, s1⟩
  · simp [h1] at h
  simp only [h1, Option.bind_eq_bind, Option.some_bind] at h
  rcases h2 : s1.popSym with _ | ⟨sv_addr, s2⟩
  · simp [h2] at h
  simp only [h2, Option.bind_eq_bind, Option.some_bind] at h
  rcases hc : s2.commit sv_val with _ | ⟨src, s3, opsCommit⟩
  · simp [hc] at h
  simp only [hc, Option.bind_eq_bind, Option.some_bind] at h
  -- Thread wellScoped + nextReg chain.
  have hws1 : s1.wellScoped := LowerState.popSym_preserves_wellScoped hws h1
  have hws2 : s2.wellScoped := LowerState.popSym_preserves_wellScoped hws1 h2
  have hnr1 : s1.nextReg = s.nextReg := LowerState.popSym_nextReg h1
  have hnr2 : s2.nextReg = s1.nextReg := LowerState.popSym_nextReg h2
  have hnr_c : s2.nextReg ≤ s3.nextReg := LowerState.commit_nextReg_mono hc
  -- sv_val came from s.stack; lift to s2.
  have hsv_val_s : ∀ r ∈ sv_val.regs, r < s.nextReg :=
    LowerState.popSym_sv_regs_lt hws h1
  have hsv_val_s2 : ∀ r ∈ sv_val.regs, r < s2.nextReg := by
    intro r hr; rw [hnr2, hnr1]; exact hsv_val_s r hr
  have hsrc_s3 : src ∈ s3.scopeEnv :=
    LowerState.commit_r_mem_scopeEnv hsv_val_s2 hc
  -- sv_addr came from s1.stack; lift to s3 via the chain.
  have hsv_addr_s1 : ∀ r ∈ sv_addr.regs, r < s1.nextReg :=
    LowerState.popSym_sv_regs_lt hws1 h2
  cases sv_addr with
  | bufferAccess slot base scale =>
      by_cases hscale : scale = 4
      · subst hscale
        simp at h
        obtain ⟨h_s_eq, h_ops⟩ := h
        subst h_s_eq
        subst h_ops
        -- base ∈ sv_addr.regs = [base], so base < s1.nextReg = s.nextReg
        have hbase_s1 : base < s1.nextReg :=
          hsv_addr_s1 base (by simp [SymVal.regs])
        have hbase_s3 : base < s3.nextReg := by
          have : base < s2.nextReg := by rw [hnr2]; exact hbase_s1
          exact Nat.lt_of_lt_of_le this hnr_c
        -- Goal: scopeValidOps s3.scopeEnv (opsCommit ++ [.store slot base src .u32])
        apply Quanta.KOps.KernelOp.scopeValidOps_append
        · exact LowerState.commit_ops_scopeValid s3.scopeEnv hc
        · refine ⟨?_, trivial⟩
          intro r hr
          simp [KernelOp.usedRegs] at hr
          -- hr : r = base ∨ r = src
          have hsuper : s3.scopeEnv ⊆ extendEnvOps s3.scopeEnv opsCommit :=
            Quanta.KOps.KernelOp.extendEnvOps_super _ _
          rcases hr with rfl | rfl
          · apply hsuper
            rw [LowerState.mem_scopeEnv]; exact hbase_s3
          · exact hsuper hsrc_s3
      · -- scale ≠ 4: match fails; lowerI32Store returns none.
        exfalso
        simp only [pure, Pure.pure] at h
        split at h
        · rename_i _ _ _ hp
          cases hp
          exact hscale rfl
        · exact Option.noConfusion h
  | reg _ _ => simp at h
  | i32ConstSym _ => simp at h
  | bufferPtr _ => simp at h
  | scaledIdx _ _ => simp at h

theorem lowerInstr_i32Store_scopeValid
    {s s' : LowerState} {offset align : Nat} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s (.i32Store offset align) = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops := by
  unfold lowerInstr at h
  exact lowerI32Store_scopeValid hws h

-- ════════════════════════════════════════════════════════════════════
-- Per-arm: localSet (dual-Copy)
--
-- Emits opsCommit ++ [.copy fresh src, .copy stable fresh] in two
-- variants (existing stable, fresh stable). Either way:
--   * src came from commit at s1→s2, so src ∈ s2.scopeEnv.
--   * fresh = s2.nextReg (freshly allocated, defined by .copy fresh src).
--   * stable lives below s'.nextReg either via wellScoped (existing
--     case) or via being just allocated (new case).
-- ════════════════════════════════════════════════════════════════════

theorem lowerInstr_localSet_scopeValid
    {s s' : LowerState} {i : Nat} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerInstr s (.localSet i) = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops := by
  simp [lowerInstr] at h
  rcases hpop : s.popSym with _ | ⟨sv, s1⟩
  · simp [hpop] at h
  simp only [hpop, Option.bind_eq_bind, Option.some_bind] at h
  rcases hc : s1.commit sv with _ | ⟨src, s2, opsCommit⟩
  · simp [hc] at h
  simp only [hc, Option.some_bind] at h
  -- Thread wellScoped.
  have hws1 : s1.wellScoped := LowerState.popSym_preserves_wellScoped hws hpop
  have hws2 : s2.wellScoped := LowerState.commit_preserves_wellScoped hws1 hc
  have hnr_pop : s1.nextReg = s.nextReg := LowerState.popSym_nextReg hpop
  have hnr_c : s1.nextReg ≤ s2.nextReg := LowerState.commit_nextReg_mono hc
  have hsv_s : ∀ r ∈ sv.regs, r < s.nextReg :=
    LowerState.popSym_sv_regs_lt hws hpop
  have hsv_s1 : ∀ r ∈ sv.regs, r < s1.nextReg := by
    intro r hr; rw [hnr_pop]; exact hsv_s r hr
  have hsrc_s2 : src ∈ s2.scopeEnv :=
    LowerState.commit_r_mem_scopeEnv hsv_s1 hc
  -- After hc simp, h is the alloc-and-match.
  simp only [LowerState.alloc] at h
  -- The match in production is on lookupLocal of the bumped state; that
  -- equals s2.lookupLocal since alloc only touches nextReg.
  rcases hlk : ({ nextReg := s2.nextReg + 1, stack := s2.stack,
                   localReg := s2.localReg, localTy := s2.localTy,
                   bufferSlots := s2.bufferSlots, currentReg := s2.currentReg }
                   : LowerState).lookupLocal i with _ | stable
  · -- New stable: another alloc. s'.nextReg = s2.nextReg + 2.
    simp [hlk, LowerState.setLocalReg, LowerState.setCurrentReg] at h
    obtain ⟨h_s_eq, h_ops⟩ := h
    subst h_s_eq; subst h_ops
    -- Goal: scopeValidOps s'.scopeEnv (opsCommit ++ [.copy fresh src, .copy stable fresh])
    --   where fresh = s2.nextReg, stable = s2.nextReg + 1, s'.nextReg = s2.nextReg + 2.
    let envS' : List Reg := List.range (s2.nextReg + 1 + 1)
    have hsrc_envS' : src ∈ envS' := by
      rw [LowerState.mem_scopeEnv] at hsrc_s2
      show src ∈ List.range _
      rw [List.mem_range]
      exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt hsrc_s2)
    show scopeValidOps envS'
      (opsCommit ++
        [KernelOp.copy s2.nextReg src,
         KernelOp.copy (s2.nextReg + 1) s2.nextReg])
    apply Quanta.KOps.KernelOp.scopeValidOps_append
    · exact LowerState.commit_ops_scopeValid envS' hc
    · -- Trailing two ops.
      have hsuper : envS' ⊆ extendEnvOps envS' opsCommit :=
        Quanta.KOps.KernelOp.extendEnvOps_super envS' opsCommit
      refine ⟨?_, ?_, trivial⟩
      · -- .copy s2.nextReg src: usedRegs = [src].
        intro r hr
        simp [KernelOp.usedRegs] at hr
        rcases hr with rfl
        exact hsuper hsrc_envS'
      · -- .copy (s2.nextReg + 1) s2.nextReg against env extended by previous copy.
        intro r hr
        simp [KernelOp.usedRegs] at hr
        rcases hr with rfl
        -- Need s2.nextReg in env extended by .copy s2.nextReg src.
        show s2.nextReg ∈ extendEnv (extendEnvOps envS' opsCommit) (.copy s2.nextReg src)
        simp [extendEnv, KernelOp.definedReg]
  · -- Existing stable: only one alloc. s'.nextReg = s2.nextReg + 1.
    simp [hlk, LowerState.setLocalReg, LowerState.setCurrentReg] at h
    obtain ⟨h_s_eq, h_ops⟩ := h
    -- stable came from lookupLocal at the bumped state. lookupLocal reads
    -- only localReg, which is the same as s2.localReg. So stable ∈
    -- (∃ p, p.snd = stable in s2.localReg), hence stable < s2.nextReg by hws2.
    have hstable_lt_s2 : stable < s2.nextReg := by
      obtain ⟨_, hlocws, _⟩ := hws2
      unfold LowerState.lookupLocal at hlk
      simp at hlk
      -- hlk : ∃ a, find? (·.fst = i) s2.localReg = some (a, stable)
      obtain ⟨a, hfind⟩ := hlk
      have hmem : (a, stable) ∈ s2.localReg := List.mem_of_find?_eq_some hfind
      exact hlocws _ hmem
    subst h_s_eq; subst h_ops
    -- Goal: scopeValidOps s'.scopeEnv (opsCommit ++ [.copy fresh src, .copy stable fresh])
    --   where fresh = s2.nextReg, stable as bound, s'.nextReg = s2.nextReg + 1.
    let envS' : List Reg := List.range (s2.nextReg + 1)
    have hsrc_envS' : src ∈ envS' := by
      rw [LowerState.mem_scopeEnv] at hsrc_s2
      show src ∈ List.range _
      rw [List.mem_range]
      exact Nat.lt_succ_of_lt hsrc_s2
    have hstable_envS' : stable ∈ envS' := by
      show stable ∈ List.range _
      rw [List.mem_range]
      exact Nat.lt_succ_of_lt hstable_lt_s2
    show scopeValidOps envS'
      (opsCommit ++
        [KernelOp.copy s2.nextReg src,
         KernelOp.copy stable s2.nextReg])
    apply Quanta.KOps.KernelOp.scopeValidOps_append
    · exact LowerState.commit_ops_scopeValid envS' hc
    · have hsuper : envS' ⊆ extendEnvOps envS' opsCommit :=
        Quanta.KOps.KernelOp.extendEnvOps_super envS' opsCommit
      refine ⟨?_, ?_, trivial⟩
      · intro r hr
        simp [KernelOp.usedRegs] at hr
        rcases hr with rfl
        exact hsuper hsrc_envS'
      · intro r hr
        simp [KernelOp.usedRegs] at hr
        rcases hr with rfl
        show s2.nextReg ∈ extendEnv (extendEnvOps envS' opsCommit) (.copy s2.nextReg src)
        simp [extendEnv, KernelOp.definedReg]

-- ════════════════════════════════════════════════════════════════════
-- Per-arm: localTee (dual-Copy + tee re-read)
--
-- Like localSet but with one more alloc (post_fresh) and one more
-- trailing op: [.copy fresh src, .copy stable fresh, .copy post_fresh fresh].
-- ════════════════════════════════════════════════════════════════════

theorem lowerInstr_localTee_scopeValid
    {s s' : LowerState} {i : Nat} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerInstr s (.localTee i) = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops := by
  simp [lowerInstr] at h
  rcases hpop : s.popSym with _ | ⟨sv, s1⟩
  · simp [hpop] at h
  simp only [hpop, Option.bind_eq_bind, Option.some_bind] at h
  rcases hc : s1.commit sv with _ | ⟨src, s2, opsCommit⟩
  · simp [hc] at h
  simp only [hc, Option.some_bind] at h
  have hws1 : s1.wellScoped := LowerState.popSym_preserves_wellScoped hws hpop
  have hws2 : s2.wellScoped := LowerState.commit_preserves_wellScoped hws1 hc
  have hnr_pop : s1.nextReg = s.nextReg := LowerState.popSym_nextReg hpop
  have hnr_c : s1.nextReg ≤ s2.nextReg := LowerState.commit_nextReg_mono hc
  have hsv_s : ∀ r ∈ sv.regs, r < s.nextReg :=
    LowerState.popSym_sv_regs_lt hws hpop
  have hsv_s1 : ∀ r ∈ sv.regs, r < s1.nextReg := by
    intro r hr; rw [hnr_pop]; exact hsv_s r hr
  have hsrc_s2 : src ∈ s2.scopeEnv :=
    LowerState.commit_r_mem_scopeEnv hsv_s1 hc
  simp only [LowerState.alloc, LowerState.push] at h
  rcases hlk : ({ nextReg := s2.nextReg + 1, stack := s2.stack,
                   localReg := s2.localReg, localTy := s2.localTy,
                   bufferSlots := s2.bufferSlots, currentReg := s2.currentReg }
                   : LowerState).lookupLocal i with _ | stable
  · -- New stable: extra alloc. fresh = s2.nextReg, stable = s2.nextReg + 1,
    -- post_fresh = s2.nextReg + 2, s'.nextReg = s2.nextReg + 3.
    simp [hlk, LowerState.setLocalReg, LowerState.setCurrentReg] at h
    obtain ⟨h_s_eq, h_ops⟩ := h
    subst h_s_eq; subst h_ops
    let envS' : List Reg := List.range (s2.nextReg + 1 + 1 + 1)
    have hsrc_envS' : src ∈ envS' := by
      rw [LowerState.mem_scopeEnv] at hsrc_s2
      show src ∈ List.range _
      rw [List.mem_range]
      exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt (Nat.lt_succ_of_lt hsrc_s2))
    show scopeValidOps envS'
      (opsCommit ++
        [KernelOp.copy s2.nextReg src,
         KernelOp.copy (s2.nextReg + 1) s2.nextReg,
         KernelOp.copy (s2.nextReg + 1 + 1) s2.nextReg])
    apply Quanta.KOps.KernelOp.scopeValidOps_append
    · exact LowerState.commit_ops_scopeValid envS' hc
    · have hsuper : envS' ⊆ extendEnvOps envS' opsCommit :=
        Quanta.KOps.KernelOp.extendEnvOps_super envS' opsCommit
      refine ⟨?_, ?_, ?_, trivial⟩
      · intro r hr
        simp [KernelOp.usedRegs] at hr
        rcases hr with rfl
        exact hsuper hsrc_envS'
      · intro r hr
        simp [KernelOp.usedRegs] at hr
        rcases hr with rfl
        show s2.nextReg ∈ extendEnv (extendEnvOps envS' opsCommit) (.copy s2.nextReg src)
        simp [extendEnv, KernelOp.definedReg]
      · intro r hr
        simp [KernelOp.usedRegs] at hr
        rcases hr with rfl
        -- Need s2.nextReg ∈ extendEnv (extendEnv … (.copy s2.nextReg src)) (.copy (s2.nextReg+1) s2.nextReg)
        -- The inner .copy s2.nextReg src adds s2.nextReg to env (already there
        -- structurally via extendEnv); the outer .copy (s2.nextReg+1) s2.nextReg
        -- adds (s2.nextReg+1) but doesn't remove s2.nextReg. So s2.nextReg
        -- is still in env.
        show s2.nextReg ∈
          extendEnv
            (extendEnv (extendEnvOps envS' opsCommit) (.copy s2.nextReg src))
            (.copy (s2.nextReg + 1) s2.nextReg)
        simp [extendEnv, KernelOp.definedReg]
  · -- Existing stable: only the fresh + post_fresh allocs (no new stable).
    simp [hlk, LowerState.setLocalReg, LowerState.setCurrentReg] at h
    obtain ⟨h_s_eq, h_ops⟩ := h
    have hstable_lt_s2 : stable < s2.nextReg := by
      obtain ⟨_, hlocws, _⟩ := hws2
      unfold LowerState.lookupLocal at hlk
      simp at hlk
      obtain ⟨a, hfind⟩ := hlk
      have hmem : (a, stable) ∈ s2.localReg := List.mem_of_find?_eq_some hfind
      exact hlocws _ hmem
    subst h_s_eq; subst h_ops
    -- fresh = s2.nextReg, post_fresh = s2.nextReg + 1, s'.nextReg = s2.nextReg + 2.
    let envS' : List Reg := List.range (s2.nextReg + 1 + 1)
    have hsrc_envS' : src ∈ envS' := by
      rw [LowerState.mem_scopeEnv] at hsrc_s2
      show src ∈ List.range _
      rw [List.mem_range]
      exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt hsrc_s2)
    have hstable_envS' : stable ∈ envS' := by
      show stable ∈ List.range _
      rw [List.mem_range]
      exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt hstable_lt_s2)
    show scopeValidOps envS'
      (opsCommit ++
        [KernelOp.copy s2.nextReg src,
         KernelOp.copy stable s2.nextReg,
         KernelOp.copy (s2.nextReg + 1) s2.nextReg])
    apply Quanta.KOps.KernelOp.scopeValidOps_append
    · exact LowerState.commit_ops_scopeValid envS' hc
    · have hsuper : envS' ⊆ extendEnvOps envS' opsCommit :=
        Quanta.KOps.KernelOp.extendEnvOps_super envS' opsCommit
      refine ⟨?_, ?_, ?_, trivial⟩
      · intro r hr
        simp [KernelOp.usedRegs] at hr
        rcases hr with rfl
        exact hsuper hsrc_envS'
      · intro r hr
        simp [KernelOp.usedRegs] at hr
        rcases hr with rfl
        show s2.nextReg ∈ extendEnv (extendEnvOps envS' opsCommit) (.copy s2.nextReg src)
        simp [extendEnv, KernelOp.definedReg]
      · intro r hr
        simp [KernelOp.usedRegs] at hr
        rcases hr with rfl
        show s2.nextReg ∈
          extendEnv
            (extendEnv (extendEnvOps envS' opsCommit) (.copy s2.nextReg src))
            (.copy stable s2.nextReg)
        simp [extendEnv, KernelOp.definedReg]

-- Per-arm wrappers for the cmp family.
theorem lowerInstr_i32Eq_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32Eq = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops :=
  lowerI32Cmp_scopeValid hws h

theorem lowerInstr_i32Ne_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32Ne = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops :=
  lowerI32Cmp_scopeValid hws h

theorem lowerInstr_i32LtU_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32LtU = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops :=
  lowerI32Cmp_scopeValid hws h

theorem lowerInstr_i32LeU_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32LeU = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops :=
  lowerI32Cmp_scopeValid hws h

theorem lowerInstr_i32GtU_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32GtU = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops :=
  lowerI32Cmp_scopeValid hws h

theorem lowerInstr_i32GeU_scopeValid
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32GeU = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops :=
  lowerI32Cmp_scopeValid hws h

-- ════════════════════════════════════════════════════════════════════
-- Master theorem
--
-- Every successful `lowerInstr s instr = some (s', ops)` against a
-- wellScoped `s` produces `ops` that are scope-valid against
-- `s'.scopeEnv`. Each supported arm delegates to its per-arm theorem;
-- the unsupported arms refuse with `none` and the precondition
-- contradicts the hypothesis.
--
-- This is the per-instruction step lemma the eventual list-level
-- theorem will induct over (chained with the wellScoped-preservation
-- counterpart that lives below).
-- ════════════════════════════════════════════════════════════════════

theorem lowerInstr_scopeValid
    {s s' : LowerState} {instr : WasmInstr} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerInstr s instr = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops := by
  cases instr with
  | i32Const n      => exact lowerInstr_i32Const_scopeValid s n h
  | nop             => exact lowerInstr_nop_scopeValid s h
  | wreturn         => exact lowerInstr_wreturn_scopeValid s h
  | drop            => exact lowerInstr_drop_scopeValid s h
  | localGet _      => exact lowerInstr_localGet_scopeValid hws h
  | localSet _      => exact lowerInstr_localSet_scopeValid hws h
  | localTee _      => exact lowerInstr_localTee_scopeValid hws h
  | i32Add          => exact lowerInstr_i32Add_scopeValid hws h
  | i32Sub          => exact lowerInstr_i32Sub_scopeValid hws h
  | i32Mul          => exact lowerInstr_i32Mul_scopeValid hws h
  | i32And          => exact lowerInstr_i32And_scopeValid hws h
  | i32Or           => exact lowerInstr_i32Or_scopeValid hws h
  | i32Xor          => exact lowerInstr_i32Xor_scopeValid hws h
  | i32Shl          => exact lowerInstr_i32Shl_scopeValid hws h
  | i32ShrU         => exact lowerInstr_i32ShrU_scopeValid hws h
  | i32DivU         => exact lowerInstr_i32DivU_scopeValid hws h
  | i32RemU         => exact lowerInstr_i32RemU_scopeValid hws h
  | i32Eq           => exact lowerInstr_i32Eq_scopeValid hws h
  | i32Ne           => exact lowerInstr_i32Ne_scopeValid hws h
  | i32LtU          => exact lowerInstr_i32LtU_scopeValid hws h
  | i32LeU          => exact lowerInstr_i32LeU_scopeValid hws h
  | i32GtU          => exact lowerInstr_i32GtU_scopeValid hws h
  | i32GeU          => exact lowerInstr_i32GeU_scopeValid hws h
  | i32Load _ _     => exact lowerInstr_i32Load_scopeValid hws h
  | i32Store _ _    => exact lowerInstr_i32Store_scopeValid hws h
  -- Unsupported arms: lowerInstr returns none, contradicting h.
  | i64Const _      => simp [lowerInstr] at h
  | f32Const _      => simp [lowerInstr] at h
  | f64Const _      => simp [lowerInstr] at h
  | i32DivS         => simp [lowerInstr] at h
  | i32RemS         => simp [lowerInstr] at h
  | i32ShrS         => simp [lowerInstr] at h
  | i32LtS          => simp [lowerInstr] at h
  | i32GtS          => simp [lowerInstr] at h
  | i32LeS          => simp [lowerInstr] at h
  | i32GeS          => simp [lowerInstr] at h
  | i32Eqz          => simp [lowerInstr] at h
  | f32Add | f32Sub | f32Mul | f32Div => all_goals simp [lowerInstr] at h
  | f32Eq | f32Ne | f32Lt | f32Gt | f32Le | f32Ge =>
      all_goals simp [lowerInstr] at h
  | f32Neg | f32Abs | f32Sqrt | f32Min | f32Max =>
      all_goals simp [lowerInstr] at h
  | i32WrapI64 | f32ConvertI32S | f32ConvertI32U =>
      all_goals simp [lowerInstr] at h
  | i32TruncF32S | i32TruncF32U =>
      all_goals simp [lowerInstr] at h
  | f32ReinterpretI32 | i32ReinterpretF32 =>
      all_goals simp [lowerInstr] at h
  | f32Load _ _ | f32Store _ _ =>
      all_goals simp [lowerInstr] at h
  | i32Load8U _ _ | i32Load8S _ _ | i32Store8 _ _ =>
      all_goals simp [lowerInstr] at h
  | block _ | wloop _ | wif _ =>
      all_goals simp [lowerInstr] at h
  | welse | wend    => all_goals simp [lowerInstr] at h
  | br _ | brIf _   => all_goals simp [lowerInstr] at h
  | call _          => simp [lowerInstr] at h
  | wselect         => simp [lowerInstr] at h
  | unreachable     => simp [lowerInstr] at h
  | unsupported _   => simp [lowerInstr] at h

-- ════════════════════════════════════════════════════════════════════
-- wellScoped preservation for the rest of the closed arms
--
-- popSym + commit + alloc + push are each wellScoped-preserving
-- under the right precondition. Each per-arm theorem threads them
-- through the do-block.
-- ════════════════════════════════════════════════════════════════════

theorem lowerI32Bin_preserves_wellScoped
    {s s' : LowerState} {op : Quanta.KOps.BinOp} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerI32Bin s op = some (s', ops)) : s'.wellScoped := by
  unfold lowerI32Bin at h
  rcases hb : s.popSym with _ | ⟨svb, s1⟩
  · simp [hb] at h
  simp only [hb, Option.bind_eq_bind, Option.some_bind] at h
  rcases ha : s1.popSym with _ | ⟨sva, s2⟩
  · simp [ha] at h
  simp only [ha, Option.bind_eq_bind, Option.some_bind] at h
  rcases hca : s2.commit sva with _ | ⟨ra, s3, opsA⟩
  · simp [hca] at h
  simp only [hca, Option.some_bind] at h
  rcases hcb : s3.commit svb with _ | ⟨rb, s4, opsB⟩
  · simp [hcb] at h
  simp only [hcb, Option.some_bind] at h
  simp [LowerState.alloc, LowerState.push] at h
  obtain ⟨h_s_eq, _⟩ := h
  -- Thread wellScoped.
  have hws1 : s1.wellScoped := LowerState.popSym_preserves_wellScoped hws hb
  have hws2 : s2.wellScoped := LowerState.popSym_preserves_wellScoped hws1 ha
  have hws3 : s3.wellScoped := LowerState.commit_preserves_wellScoped hws2 hca
  have hws4 : s4.wellScoped := LowerState.commit_preserves_wellScoped hws3 hcb
  subst h_s_eq
  -- Goal: { s4 with nextReg := s4.nextReg + 1,
  --                  stack := .reg s4.nextReg .u32 :: s4.stack }.wellScoped
  -- This is `(s4.alloc.snd).push s4.nextReg`.
  obtain ⟨hstk, hloc, hcur⟩ := hws4
  refine ⟨?_, ?_, ?_⟩
  · intro sv hsv r hr
    show r < s4.nextReg + 1
    simp at hsv
    rcases hsv with rfl | hsv
    · simp [SymVal.regs] at hr
      subst hr; exact Nat.lt_succ_self _
    · exact Nat.lt_succ_of_lt (hstk sv hsv r hr)
  · intro p hp
    show p.snd < s4.nextReg + 1
    exact Nat.lt_succ_of_lt (hloc p hp)
  · intro p hp
    show p.snd < s4.nextReg + 1
    exact Nat.lt_succ_of_lt (hcur p hp)

theorem lowerInstr_i32Sub_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32Sub = some (s', ops)) :
    s'.wellScoped :=
  lowerI32Bin_preserves_wellScoped hws h

theorem lowerInstr_i32Mul_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32Mul = some (s', ops)) :
    s'.wellScoped :=
  lowerI32Bin_preserves_wellScoped hws h

theorem lowerInstr_i32And_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32And = some (s', ops)) :
    s'.wellScoped :=
  lowerI32Bin_preserves_wellScoped hws h

theorem lowerInstr_i32Or_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32Or = some (s', ops)) :
    s'.wellScoped :=
  lowerI32Bin_preserves_wellScoped hws h

theorem lowerInstr_i32Xor_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32Xor = some (s', ops)) :
    s'.wellScoped :=
  lowerI32Bin_preserves_wellScoped hws h

theorem lowerInstr_i32ShrU_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32ShrU = some (s', ops)) :
    s'.wellScoped :=
  lowerI32Bin_preserves_wellScoped hws h

theorem lowerInstr_i32DivU_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32DivU = some (s', ops)) :
    s'.wellScoped :=
  lowerI32Bin_preserves_wellScoped hws h

theorem lowerInstr_i32RemU_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32RemU = some (s', ops)) :
    s'.wellScoped :=
  lowerI32Bin_preserves_wellScoped hws h

-- ════════════════════════════════════════════════════════════════════
-- lowerI32Cmp preservation + 6 cmp arms
-- ════════════════════════════════════════════════════════════════════

theorem lowerI32Cmp_preserves_wellScoped
    {s s' : LowerState} {op : Quanta.KOps.CmpOp} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerI32Cmp s op = some (s', ops)) : s'.wellScoped := by
  unfold lowerI32Cmp at h
  rcases hb : s.popSym with _ | ⟨svb, s1⟩
  · simp [hb] at h
  simp only [hb, Option.bind_eq_bind, Option.some_bind] at h
  rcases ha : s1.popSym with _ | ⟨sva, s2⟩
  · simp [ha] at h
  simp only [ha, Option.bind_eq_bind, Option.some_bind] at h
  rcases hca : s2.commit sva with _ | ⟨ra, s3, opsA⟩
  · simp [hca] at h
  simp only [hca, Option.some_bind] at h
  rcases hcb : s3.commit svb with _ | ⟨rb, s4, opsB⟩
  · simp [hcb] at h
  simp only [hcb, Option.some_bind] at h
  simp [LowerState.alloc, LowerState.push] at h
  obtain ⟨h_s_eq, _⟩ := h
  have hws1 : s1.wellScoped := LowerState.popSym_preserves_wellScoped hws hb
  have hws2 : s2.wellScoped := LowerState.popSym_preserves_wellScoped hws1 ha
  have hws3 : s3.wellScoped := LowerState.commit_preserves_wellScoped hws2 hca
  have hws4 : s4.wellScoped := LowerState.commit_preserves_wellScoped hws3 hcb
  subst h_s_eq
  -- Goal state has nextReg = s4.nextReg + 1 + 1, stack = .reg (s4.nextReg+1) .u32 :: s4.stack.
  obtain ⟨hstk, hloc, hcur⟩ := hws4
  refine ⟨?_, ?_, ?_⟩
  · intro sv hsv r hr
    show r < s4.nextReg + 1 + 1
    simp at hsv
    rcases hsv with rfl | hsv
    · simp [SymVal.regs] at hr
      subst hr
      -- r = s4.nextReg + 1 < s4.nextReg + 2
      exact Nat.lt_succ_self _
    · -- r < s4.nextReg < s4.nextReg + 2
      exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt (hstk sv hsv r hr))
  · intro p hp
    show p.snd < s4.nextReg + 1 + 1
    exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt (hloc p hp))
  · intro p hp
    show p.snd < s4.nextReg + 1 + 1
    exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt (hcur p hp))

theorem lowerInstr_i32Eq_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32Eq = some (s', ops)) :
    s'.wellScoped :=
  lowerI32Cmp_preserves_wellScoped hws h

theorem lowerInstr_i32Ne_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32Ne = some (s', ops)) :
    s'.wellScoped :=
  lowerI32Cmp_preserves_wellScoped hws h

theorem lowerInstr_i32LtU_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32LtU = some (s', ops)) :
    s'.wellScoped :=
  lowerI32Cmp_preserves_wellScoped hws h

theorem lowerInstr_i32LeU_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32LeU = some (s', ops)) :
    s'.wellScoped :=
  lowerI32Cmp_preserves_wellScoped hws h

theorem lowerInstr_i32GtU_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32GtU = some (s', ops)) :
    s'.wellScoped :=
  lowerI32Cmp_preserves_wellScoped hws h

theorem lowerInstr_i32GeU_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32GeU = some (s', ops)) :
    s'.wellScoped :=
  lowerI32Cmp_preserves_wellScoped hws h

-- ════════════════════════════════════════════════════════════════════
-- i32Shl + i32Add (buffer-pattern arms + lowerI32Bin fallback)
--
-- Buffer-pattern arms rewrite `stack` to a new SymVal whose regs
-- come from a SymVal already on s.stack; nextReg unchanged.
-- ════════════════════════════════════════════════════════════════════

theorem lowerI32Shl_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerI32Shl s = some (s', ops)) : s'.wellScoped := by
  unfold lowerI32Shl at h
  rcases hs : s.stack with _ | ⟨sv1, rs⟩
  · rw [hs] at h
    exact lowerI32Bin_preserves_wellScoped hws h
  · rw [hs] at h
    cases sv1 with
    | i32ConstSym k =>
        cases rs with
        | nil => exact lowerI32Bin_preserves_wellScoped hws h
        | cons sv2 rs2 =>
            cases sv2 with
            | reg base ty =>
                -- Buffer-pattern arm: s' = {s with stack := scaledIdx base (1<<<k) :: rs2}.
                simp at h
                obtain ⟨h_s_eq, _⟩ := h
                subst h_s_eq
                obtain ⟨hstk, hloc, hcur⟩ := hws
                -- Need base < s.nextReg (from .reg base ty ∈ s.stack).
                have hbase : base < s.nextReg := by
                  have hmem : SymVal.reg base ty ∈ s.stack := by
                    rw [hs]
                    exact List.mem_cons_of_mem _ (List.mem_cons_self _ _)
                  exact hstk _ hmem base (by simp [SymVal.regs])
                refine ⟨?_, hloc, hcur⟩
                intro sv hsv r hr
                simp at hsv
                rcases hsv with rfl | hsv
                · -- sv = .scaledIdx base (1 <<< k.toNat); regs = [base]
                  simp [SymVal.regs] at hr
                  subst hr; exact hbase
                · -- sv ∈ rs2 ⊆ s.stack
                  have hmem : sv ∈ s.stack := by
                    rw [hs]
                    exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hsv)
                  exact hstk sv hmem r hr
            | i32ConstSym _ => exact lowerI32Bin_preserves_wellScoped hws h
            | bufferPtr _ => exact lowerI32Bin_preserves_wellScoped hws h
            | scaledIdx _ _ => exact lowerI32Bin_preserves_wellScoped hws h
            | bufferAccess _ _ _ => exact lowerI32Bin_preserves_wellScoped hws h
    | reg _ _ => exact lowerI32Bin_preserves_wellScoped hws h
    | bufferPtr _ => exact lowerI32Bin_preserves_wellScoped hws h
    | scaledIdx _ _ => exact lowerI32Bin_preserves_wellScoped hws h
    | bufferAccess _ _ _ => exact lowerI32Bin_preserves_wellScoped hws h

theorem lowerInstr_i32Shl_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32Shl = some (s', ops)) :
    s'.wellScoped :=
  lowerI32Shl_preserves_wellScoped hws h

theorem lowerI32Add_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerI32Add s = some (s', ops)) : s'.wellScoped := by
  unfold lowerI32Add at h
  rcases hs : s.stack with _ | ⟨sv1, rs⟩
  · rw [hs] at h; exact lowerI32Bin_preserves_wellScoped hws h
  · rw [hs] at h
    cases sv1 with
    | scaledIdx base scale =>
        cases rs with
        | nil => exact lowerI32Bin_preserves_wellScoped hws h
        | cons sv2 rs2 =>
            cases sv2 with
            | bufferPtr slot =>
                -- s' = {s with stack := bufferAccess slot base scale :: rs2}.
                simp at h
                obtain ⟨h_s_eq, _⟩ := h
                subst h_s_eq
                obtain ⟨hstk, hloc, hcur⟩ := hws
                have hbase : base < s.nextReg := by
                  have hmem : SymVal.scaledIdx base scale ∈ s.stack := by
                    rw [hs]; exact List.mem_cons_self _ _
                  exact hstk _ hmem base (by simp [SymVal.regs])
                refine ⟨?_, hloc, hcur⟩
                intro sv hsv r hr
                simp at hsv
                rcases hsv with rfl | hsv
                · simp [SymVal.regs] at hr
                  subst hr; exact hbase
                · have hmem : sv ∈ s.stack := by
                    rw [hs]
                    exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hsv)
                  exact hstk sv hmem r hr
            | reg _ _ => exact lowerI32Bin_preserves_wellScoped hws h
            | i32ConstSym _ => exact lowerI32Bin_preserves_wellScoped hws h
            | scaledIdx _ _ => exact lowerI32Bin_preserves_wellScoped hws h
            | bufferAccess _ _ _ => exact lowerI32Bin_preserves_wellScoped hws h
    | bufferPtr slot =>
        cases rs with
        | nil => exact lowerI32Bin_preserves_wellScoped hws h
        | cons sv2 rs2 =>
            cases sv2 with
            | scaledIdx base scale =>
                simp at h
                obtain ⟨h_s_eq, _⟩ := h
                subst h_s_eq
                obtain ⟨hstk, hloc, hcur⟩ := hws
                have hbase : base < s.nextReg := by
                  have hmem : SymVal.scaledIdx base scale ∈ s.stack := by
                    rw [hs]; exact List.mem_cons_of_mem _ (List.mem_cons_self _ _)
                  exact hstk _ hmem base (by simp [SymVal.regs])
                refine ⟨?_, hloc, hcur⟩
                intro sv hsv r hr
                simp at hsv
                rcases hsv with rfl | hsv
                · simp [SymVal.regs] at hr
                  subst hr; exact hbase
                · have hmem : sv ∈ s.stack := by
                    rw [hs]
                    exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hsv)
                  exact hstk sv hmem r hr
            | reg _ _ => exact lowerI32Bin_preserves_wellScoped hws h
            | i32ConstSym _ => exact lowerI32Bin_preserves_wellScoped hws h
            | bufferPtr _ => exact lowerI32Bin_preserves_wellScoped hws h
            | bufferAccess _ _ _ => exact lowerI32Bin_preserves_wellScoped hws h
    | reg _ _ => exact lowerI32Bin_preserves_wellScoped hws h
    | i32ConstSym _ => exact lowerI32Bin_preserves_wellScoped hws h
    | bufferAccess _ _ _ => exact lowerI32Bin_preserves_wellScoped hws h

theorem lowerInstr_i32Add_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s .i32Add = some (s', ops)) :
    s'.wellScoped :=
  lowerI32Add_preserves_wellScoped hws h

-- ════════════════════════════════════════════════════════════════════
-- i32Load + i32Store preservation
-- ════════════════════════════════════════════════════════════════════

theorem lowerI32Load_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerI32Load s = some (s', ops)) : s'.wellScoped := by
  unfold lowerI32Load at h
  rcases hs : s.stack with _ | ⟨sv, rs⟩
  · rw [hs] at h; simp at h
  · rw [hs] at h
    cases sv with
    | bufferAccess slot base scale =>
        by_cases hscale : scale = 4
        · subst hscale
          simp [LowerState.alloc] at h
          obtain ⟨h_s_eq, _⟩ := h
          subst h_s_eq
          -- Goal state: { s with nextReg := s.nextReg + 1,
          --                        stack := .reg s.nextReg .u32 :: rs }.
          obtain ⟨hstk, hloc, hcur⟩ := hws
          refine ⟨?_, ?_, ?_⟩
          · intro sv' hsv' r hr
            show r < s.nextReg + 1
            simp at hsv'
            rcases hsv' with rfl | hsv'
            · simp [SymVal.regs] at hr
              subst hr; exact Nat.lt_succ_self _
            · have hmem : sv' ∈ s.stack := by
                rw [hs]; exact List.mem_cons_of_mem _ hsv'
              exact Nat.lt_succ_of_lt (hstk sv' hmem r hr)
          · intro p hp
            show p.snd < s.nextReg + 1
            exact Nat.lt_succ_of_lt (hloc p hp)
          · intro p hp
            show p.snd < s.nextReg + 1
            exact Nat.lt_succ_of_lt (hcur p hp)
        · exfalso
          split at h
          · rename_i _ _ _ _ hp
            cases hp; exact hscale rfl
          · exact Option.noConfusion h
    | reg _ _ => simp at h
    | i32ConstSym _ => simp at h
    | bufferPtr _ => simp at h
    | scaledIdx _ _ => simp at h

theorem lowerInstr_i32Load_preserves_wellScoped
    {s s' : LowerState} {offset align : Nat} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s (.i32Load offset align) = some (s', ops)) :
    s'.wellScoped :=
  lowerI32Load_preserves_wellScoped hws h

theorem lowerI32Store_preserves_wellScoped
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerI32Store s = some (s', ops)) : s'.wellScoped := by
  unfold lowerI32Store at h
  rcases h1 : s.popSym with _ | ⟨sv_val, s1⟩
  · simp [h1] at h
  simp only [h1, Option.bind_eq_bind, Option.some_bind] at h
  rcases h2 : s1.popSym with _ | ⟨sv_addr, s2⟩
  · simp [h2] at h
  simp only [h2, Option.bind_eq_bind, Option.some_bind] at h
  rcases hc : s2.commit sv_val with _ | ⟨src, s3, opsCommit⟩
  · simp [hc] at h
  simp only [hc, Option.bind_eq_bind, Option.some_bind] at h
  have hws1 : s1.wellScoped := LowerState.popSym_preserves_wellScoped hws h1
  have hws2 : s2.wellScoped := LowerState.popSym_preserves_wellScoped hws1 h2
  have hws3 : s3.wellScoped := LowerState.commit_preserves_wellScoped hws2 hc
  cases sv_addr with
  | bufferAccess slot base scale =>
      by_cases hscale : scale = 4
      · subst hscale
        simp at h
        obtain ⟨h_s_eq, _⟩ := h
        subst h_s_eq
        exact hws3
      · exfalso
        simp only [pure, Pure.pure] at h
        split at h
        · rename_i _ _ _ hp; cases hp; exact hscale rfl
        · exact Option.noConfusion h
  | reg _ _ => simp at h
  | i32ConstSym _ => simp at h
  | bufferPtr _ => simp at h
  | scaledIdx _ _ => simp at h

theorem lowerInstr_i32Store_preserves_wellScoped
    {s s' : LowerState} {offset align : Nat} {ops : List KernelOp}
    (hws : s.wellScoped) (h : lowerInstr s (.i32Store offset align) = some (s', ops)) :
    s'.wellScoped := by
  unfold lowerInstr at h
  exact lowerI32Store_preserves_wellScoped hws h

-- ════════════════════════════════════════════════════════════════════
-- localSet + localTee preservation (dual-Copy pattern)
-- ════════════════════════════════════════════════════════════════════

theorem lowerInstr_localSet_preserves_wellScoped
    {s s' : LowerState} {i : Nat} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerInstr s (.localSet i) = some (s', ops)) : s'.wellScoped := by
  simp [lowerInstr] at h
  rcases hpop : s.popSym with _ | ⟨sv, s1⟩
  · simp [hpop] at h
  simp only [hpop, Option.bind_eq_bind, Option.some_bind] at h
  rcases hc : s1.commit sv with _ | ⟨src, s2, opsCommit⟩
  · simp [hc] at h
  simp only [hc, Option.some_bind] at h
  have hws1 : s1.wellScoped := LowerState.popSym_preserves_wellScoped hws hpop
  have hws2 : s2.wellScoped := LowerState.commit_preserves_wellScoped hws1 hc
  -- s3 = s2.alloc.snd; wellScoped via alloc_preserves.
  have hws3 : s2.alloc.snd.wellScoped :=
    LowerState.alloc_preserves_wellScoped hws2
  -- The match on the bumped state's lookupLocal == s2.lookupLocal (alloc
  -- doesn't touch localReg). The bumped state IS s2.alloc.snd ≡ s3.
  simp only [LowerState.alloc] at h
  -- Notation: s3.nextReg = s2.nextReg + 1 (post-alloc).
  rcases hlk : ({ nextReg := s2.nextReg + 1, stack := s2.stack,
                   localReg := s2.localReg, localTy := s2.localTy,
                   bufferSlots := s2.bufferSlots, currentReg := s2.currentReg }
                   : LowerState).lookupLocal i with _ | stable
  · -- New stable case: extra alloc inside. Final s'.nextReg = s2.nextReg + 2.
    simp [hlk, LowerState.setLocalReg, LowerState.setCurrentReg] at h
    obtain ⟨h_s_eq, _⟩ := h
    subst h_s_eq
    -- stable = s2.nextReg + 1, fresh = s2.nextReg.
    obtain ⟨hstk, hloc, hcur⟩ := hws2
    refine ⟨?_, ?_, ?_⟩
    · intro sv' hsv' r hr
      show r < s2.nextReg + 1 + 1
      exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt (hstk sv' hsv' r hr))
    · intro p hp
      show p.snd < s2.nextReg + 1 + 1
      have hp' : p = (i, s2.nextReg + 1) ∨
                  p ∈ s2.localReg.filter (fun q => !decide (q.fst = i)) := by
        have : p ∈ (i, s2.nextReg + 1) ::
                    s2.localReg.filter (fun q => !decide (q.fst = i)) := hp
        exact List.mem_cons.mp this
      rcases hp' with rfl | hp'
      · exact Nat.lt_succ_self _
      · have hpsnd_s2 : p.snd < s2.nextReg :=
          hloc p (List.mem_filter.mp hp').1
        exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt hpsnd_s2)
    · intro p hp
      show p.snd < s2.nextReg + 1 + 1
      have hp' : p = (i, s2.nextReg) ∨
                  p ∈ s2.currentReg.filter (fun q => !decide (q.fst = i)) := by
        have : p ∈ (i, s2.nextReg) ::
                    s2.currentReg.filter (fun q => !decide (q.fst = i)) := hp
        exact List.mem_cons.mp this
      rcases hp' with rfl | hp'
      · exact Nat.lt_succ_of_lt (Nat.lt_succ_self _)
      · have hpsnd_s2 : p.snd < s2.nextReg :=
          hcur p (List.mem_filter.mp hp').1
        exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt hpsnd_s2)
  · -- Existing stable case. Final s'.nextReg = s2.nextReg + 1.
    simp [hlk, LowerState.setLocalReg, LowerState.setCurrentReg] at h
    obtain ⟨h_s_eq, _⟩ := h
    have hstable_lt_s2 : stable < s2.nextReg := by
      obtain ⟨_, hlocws, _⟩ := hws2
      unfold LowerState.lookupLocal at hlk
      simp at hlk
      obtain ⟨a, hfind⟩ := hlk
      have hmem : (a, stable) ∈ s2.localReg := List.mem_of_find?_eq_some hfind
      exact hlocws _ hmem
    subst h_s_eq
    obtain ⟨hstk, hloc, hcur⟩ := hws2
    refine ⟨?_, ?_, ?_⟩
    · intro sv' hsv' r hr
      show r < s2.nextReg + 1
      exact Nat.lt_succ_of_lt (hstk sv' hsv' r hr)
    · intro p hp
      show p.snd < s2.nextReg + 1
      have hp' : p = (i, stable) ∨
                  p ∈ s2.localReg.filter (fun q => !decide (q.fst = i)) := by
        have : p ∈ (i, stable) ::
                    s2.localReg.filter (fun q => !decide (q.fst = i)) := hp
        exact List.mem_cons.mp this
      rcases hp' with rfl | hp'
      · exact Nat.lt_succ_of_lt hstable_lt_s2
      · exact Nat.lt_succ_of_lt (hloc p (List.mem_filter.mp hp').1)
    · intro p hp
      show p.snd < s2.nextReg + 1
      have hp' : p = (i, s2.nextReg) ∨
                  p ∈ s2.currentReg.filter (fun q => !decide (q.fst = i)) := by
        have : p ∈ (i, s2.nextReg) ::
                    s2.currentReg.filter (fun q => !decide (q.fst = i)) := hp
        exact List.mem_cons.mp this
      rcases hp' with rfl | hp'
      · exact Nat.lt_succ_self _
      · exact Nat.lt_succ_of_lt (hcur p (List.mem_filter.mp hp').1)

theorem lowerInstr_localTee_preserves_wellScoped
    {s s' : LowerState} {i : Nat} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerInstr s (.localTee i) = some (s', ops)) : s'.wellScoped := by
  simp [lowerInstr] at h
  rcases hpop : s.popSym with _ | ⟨sv, s1⟩
  · simp [hpop] at h
  simp only [hpop, Option.bind_eq_bind, Option.some_bind] at h
  rcases hc : s1.commit sv with _ | ⟨src, s2, opsCommit⟩
  · simp [hc] at h
  simp only [hc, Option.some_bind] at h
  have hws1 : s1.wellScoped := LowerState.popSym_preserves_wellScoped hws hpop
  have hws2 : s2.wellScoped := LowerState.commit_preserves_wellScoped hws1 hc
  have hws3 : s2.alloc.snd.wellScoped :=
    LowerState.alloc_preserves_wellScoped hws2
  simp only [LowerState.alloc, LowerState.push] at h
  rcases hlk : ({ nextReg := s2.nextReg + 1, stack := s2.stack,
                   localReg := s2.localReg, localTy := s2.localTy,
                   bufferSlots := s2.bufferSlots, currentReg := s2.currentReg }
                   : LowerState).lookupLocal i with _ | stable
  · -- New stable + post_fresh alloc. Final s'.nextReg = s2.nextReg + 3.
    simp [hlk, LowerState.setLocalReg, LowerState.setCurrentReg] at h
    obtain ⟨h_s_eq, _⟩ := h
    subst h_s_eq
    -- Direct destructuring against the simp-flattened record.
    -- stable = s2.nextReg + 1, fresh = s2.nextReg, post_fresh = s2.nextReg + 2.
    obtain ⟨hstk, hloc, hcur⟩ := hws2
    refine ⟨?_, ?_, ?_⟩
    · intro sv' hsv' r hr
      show r < s2.nextReg + 1 + 1 + 1
      simp at hsv'
      rcases hsv' with rfl | hsv'
      · simp [SymVal.regs] at hr
        subst hr; exact Nat.lt_succ_self _
      · have hr_s2 : r < s2.nextReg := hstk sv' hsv' r hr
        exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt (Nat.lt_succ_of_lt hr_s2))
    · intro p hp
      show p.snd < s2.nextReg + 1 + 1 + 1
      have hp' : p = (i, s2.nextReg + 1) ∨
                  p ∈ s2.localReg.filter (fun q => !decide (q.fst = i)) := by
        have : p ∈ (i, s2.nextReg + 1) ::
                    s2.localReg.filter (fun q => !decide (q.fst = i)) := hp
        exact List.mem_cons.mp this
      rcases hp' with rfl | hp'
      · show s2.nextReg + 1 < s2.nextReg + 1 + 1 + 1
        exact Nat.lt_succ_of_lt (Nat.lt_succ_self _)
      · have hpsnd_s2 : p.snd < s2.nextReg :=
          hloc p (List.mem_filter.mp hp').1
        exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt (Nat.lt_succ_of_lt hpsnd_s2))
    · intro p hp
      show p.snd < s2.nextReg + 1 + 1 + 1
      have hp' : p = (i, s2.nextReg) ∨
                  p ∈ s2.currentReg.filter (fun q => !decide (q.fst = i)) := by
        have : p ∈ (i, s2.nextReg) ::
                    s2.currentReg.filter (fun q => !decide (q.fst = i)) := hp
        exact List.mem_cons.mp this
      rcases hp' with rfl | hp'
      · show s2.nextReg < s2.nextReg + 1 + 1 + 1
        exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt (Nat.lt_succ_self _))
      · have hpsnd_s2 : p.snd < s2.nextReg :=
          hcur p (List.mem_filter.mp hp').1
        exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt (Nat.lt_succ_of_lt hpsnd_s2))
  · -- Existing stable + post_fresh alloc.
    simp [hlk, LowerState.setLocalReg, LowerState.setCurrentReg] at h
    obtain ⟨h_s_eq, _⟩ := h
    have hstable_lt_s2 : stable < s2.nextReg := by
      obtain ⟨_, hlocws, _⟩ := hws2
      unfold LowerState.lookupLocal at hlk
      simp at hlk
      obtain ⟨a, hfind⟩ := hlk
      have hmem : (a, stable) ∈ s2.localReg := List.mem_of_find?_eq_some hfind
      exact hlocws _ hmem
    subst h_s_eq
    -- Goal: the constructed record (nextReg = s2.nextReg + 2, etc.) is wellScoped.
    -- Prove by direct destructuring + lift through the +2 widening.
    obtain ⟨hstk, hloc, hcur⟩ := hws2
    refine ⟨?_, ?_, ?_⟩
    · intro sv' hsv' r hr
      show r < s2.nextReg + 1 + 1
      simp at hsv'
      rcases hsv' with rfl | hsv'
      · simp [SymVal.regs] at hr
        subst hr; exact Nat.lt_succ_self _
      · exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt (hstk sv' hsv' r hr))
    · intro p hp
      show p.snd < s2.nextReg + 1 + 1
      -- p ∈ (i, stable) :: filter ... s2.localReg
      have hp' : p = (i, stable) ∨
                  p ∈ s2.localReg.filter (fun q => !decide (q.fst = i)) := by
        have : p ∈ (i, stable) ::
                    s2.localReg.filter (fun q => !decide (q.fst = i)) := hp
        exact List.mem_cons.mp this
      rcases hp' with rfl | hp'
      · exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt hstable_lt_s2)
      · exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt
          (hloc p (List.mem_filter.mp hp').1))
    · intro p hp
      show p.snd < s2.nextReg + 1 + 1
      have hp' : p = (i, s2.nextReg) ∨
                  p ∈ s2.currentReg.filter (fun q => !decide (q.fst = i)) := by
        have : p ∈ (i, s2.nextReg) ::
                    s2.currentReg.filter (fun q => !decide (q.fst = i)) := hp
        exact List.mem_cons.mp this
      rcases hp' with rfl | hp'
      · exact Nat.lt_succ_of_lt (Nat.lt_succ_self _)
      · exact Nat.lt_succ_of_lt (Nat.lt_succ_of_lt
          (hcur p (List.mem_filter.mp hp').1))

-- ════════════════════════════════════════════════════════════════════
-- Master wellScoped-preservation theorem
--
-- Every successful `lowerInstr s instr = some (s', ops)` against a
-- wellScoped `s` produces a wellScoped post-state `s'`. Pair this
-- with `lowerInstr_scopeValid` to chain step-level theorems into a
-- list-level `lowerInstrs_scopeValid_ops`.
-- ════════════════════════════════════════════════════════════════════

theorem lowerInstr_preserves_wellScoped
    {s s' : LowerState} {instr : WasmInstr} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerInstr s instr = some (s', ops)) : s'.wellScoped := by
  cases instr with
  | i32Const _      => exact lowerInstr_i32Const_preserves_wellScoped hws h
  | nop             => exact lowerInstr_nop_preserves_wellScoped hws h
  | wreturn         => exact lowerInstr_wreturn_preserves_wellScoped hws h
  | drop            => exact lowerInstr_drop_preserves_wellScoped hws h
  | localGet _      => exact lowerInstr_localGet_preserves_wellScoped hws h
  | localSet _      => exact lowerInstr_localSet_preserves_wellScoped hws h
  | localTee _      => exact lowerInstr_localTee_preserves_wellScoped hws h
  | i32Add          => exact lowerInstr_i32Add_preserves_wellScoped hws h
  | i32Sub          => exact lowerInstr_i32Sub_preserves_wellScoped hws h
  | i32Mul          => exact lowerInstr_i32Mul_preserves_wellScoped hws h
  | i32And          => exact lowerInstr_i32And_preserves_wellScoped hws h
  | i32Or           => exact lowerInstr_i32Or_preserves_wellScoped hws h
  | i32Xor          => exact lowerInstr_i32Xor_preserves_wellScoped hws h
  | i32Shl          => exact lowerInstr_i32Shl_preserves_wellScoped hws h
  | i32ShrU         => exact lowerInstr_i32ShrU_preserves_wellScoped hws h
  | i32DivU         => exact lowerInstr_i32DivU_preserves_wellScoped hws h
  | i32RemU         => exact lowerInstr_i32RemU_preserves_wellScoped hws h
  | i32Eq           => exact lowerInstr_i32Eq_preserves_wellScoped hws h
  | i32Ne           => exact lowerInstr_i32Ne_preserves_wellScoped hws h
  | i32LtU          => exact lowerInstr_i32LtU_preserves_wellScoped hws h
  | i32LeU          => exact lowerInstr_i32LeU_preserves_wellScoped hws h
  | i32GtU          => exact lowerInstr_i32GtU_preserves_wellScoped hws h
  | i32GeU          => exact lowerInstr_i32GeU_preserves_wellScoped hws h
  | i32Load _ _     => exact lowerInstr_i32Load_preserves_wellScoped hws h
  | i32Store _ _    => exact lowerInstr_i32Store_preserves_wellScoped hws h
  -- Unsupported arms: lowerInstr returns none, contradicting h.
  | i64Const _      => simp [lowerInstr] at h
  | f32Const _      => simp [lowerInstr] at h
  | f64Const _      => simp [lowerInstr] at h
  | i32DivS         => simp [lowerInstr] at h
  | i32RemS         => simp [lowerInstr] at h
  | i32ShrS         => simp [lowerInstr] at h
  | i32LtS          => simp [lowerInstr] at h
  | i32GtS          => simp [lowerInstr] at h
  | i32LeS          => simp [lowerInstr] at h
  | i32GeS          => simp [lowerInstr] at h
  | i32Eqz          => simp [lowerInstr] at h
  | f32Add | f32Sub | f32Mul | f32Div => all_goals simp [lowerInstr] at h
  | f32Eq | f32Ne | f32Lt | f32Gt | f32Le | f32Ge =>
      all_goals simp [lowerInstr] at h
  | f32Neg | f32Abs | f32Sqrt | f32Min | f32Max =>
      all_goals simp [lowerInstr] at h
  | i32WrapI64 | f32ConvertI32S | f32ConvertI32U =>
      all_goals simp [lowerInstr] at h
  | i32TruncF32S | i32TruncF32U =>
      all_goals simp [lowerInstr] at h
  | f32ReinterpretI32 | i32ReinterpretF32 =>
      all_goals simp [lowerInstr] at h
  | f32Load _ _ | f32Store _ _ =>
      all_goals simp [lowerInstr] at h
  | i32Load8U _ _ | i32Load8S _ _ | i32Store8 _ _ =>
      all_goals simp [lowerInstr] at h
  | block _ | wloop _ | wif _ =>
      all_goals simp [lowerInstr] at h
  | welse | wend    => all_goals simp [lowerInstr] at h
  | br _ | brIf _   => all_goals simp [lowerInstr] at h
  | call _          => simp [lowerInstr] at h
  | wselect         => simp [lowerInstr] at h
  | unreachable     => simp [lowerInstr] at h
  | unsupported _   => simp [lowerInstr] at h

-- ════════════════════════════════════════════════════════════════════
-- lowerInstr nextReg monotonicity
--
-- A side effect of wellScoped-preservation we extract as its own
-- lemma: lowerInstr only ever grows nextReg. Used by the list-level
-- theorems to lift scope-validity across the recursive boundary.
-- ════════════════════════════════════════════════════════════════════

theorem lowerI32Bin_nextReg_mono
    {s s' : LowerState} {op : Quanta.KOps.BinOp} {ops : List KernelOp}
    (h : lowerI32Bin s op = some (s', ops)) : s.nextReg ≤ s'.nextReg := by
  unfold lowerI32Bin at h
  rcases hb : s.popSym with _ | ⟨svb, s1⟩
  · simp [hb] at h
  simp only [hb, Option.bind_eq_bind, Option.some_bind] at h
  rcases ha : s1.popSym with _ | ⟨sva, s2⟩
  · simp [ha] at h
  simp only [ha, Option.bind_eq_bind, Option.some_bind] at h
  rcases hca : s2.commit sva with _ | ⟨ra, s3, opsA⟩
  · simp [hca] at h
  simp only [hca, Option.some_bind] at h
  rcases hcb : s3.commit svb with _ | ⟨rb, s4, opsB⟩
  · simp [hcb] at h
  simp only [hcb, Option.some_bind] at h
  simp [LowerState.alloc, LowerState.push] at h
  obtain ⟨h_s_eq, _⟩ := h
  subst h_s_eq
  -- chain: s = s1 (pop), s1 = s2 (pop), s2 ≤ s3 (commit), s3 ≤ s4 (commit), s4 + 1.
  have h1 := LowerState.popSym_nextReg hb
  have h2 := LowerState.popSym_nextReg ha
  have h3 := LowerState.commit_nextReg_mono hca
  have h4 := LowerState.commit_nextReg_mono hcb
  show s.nextReg ≤ s4.nextReg + 1
  omega

theorem lowerI32Cmp_nextReg_mono
    {s s' : LowerState} {op : Quanta.KOps.CmpOp} {ops : List KernelOp}
    (h : lowerI32Cmp s op = some (s', ops)) : s.nextReg ≤ s'.nextReg := by
  unfold lowerI32Cmp at h
  rcases hb : s.popSym with _ | ⟨svb, s1⟩
  · simp [hb] at h
  simp only [hb, Option.bind_eq_bind, Option.some_bind] at h
  rcases ha : s1.popSym with _ | ⟨sva, s2⟩
  · simp [ha] at h
  simp only [ha, Option.bind_eq_bind, Option.some_bind] at h
  rcases hca : s2.commit sva with _ | ⟨ra, s3, opsA⟩
  · simp [hca] at h
  simp only [hca, Option.some_bind] at h
  rcases hcb : s3.commit svb with _ | ⟨rb, s4, opsB⟩
  · simp [hcb] at h
  simp only [hcb, Option.some_bind] at h
  simp [LowerState.alloc, LowerState.push] at h
  obtain ⟨h_s_eq, _⟩ := h
  subst h_s_eq
  have h1 := LowerState.popSym_nextReg hb
  have h2 := LowerState.popSym_nextReg ha
  have h3 := LowerState.commit_nextReg_mono hca
  have h4 := LowerState.commit_nextReg_mono hcb
  show s.nextReg ≤ s4.nextReg + 1 + 1
  omega

theorem lowerInstr_nextReg_mono
    {s s' : LowerState} {instr : WasmInstr} {ops : List KernelOp}
    (h : lowerInstr s instr = some (s', ops)) : s.nextReg ≤ s'.nextReg := by
  -- Most arms either don't change nextReg or bump it by a known amount.
  -- We delegate via the wellScoped chain when straightforward.
  cases instr with
  | i32Const _ =>
      simp [lowerInstr] at h
      obtain ⟨h_s_eq, _⟩ := h
      subst h_s_eq
      exact Nat.le_refl _
  | nop =>
      simp [lowerInstr] at h
      obtain ⟨h_s_eq, _⟩ := h
      subst h_s_eq
      exact Nat.le_refl _
  | wreturn =>
      simp [lowerInstr] at h
      obtain ⟨h_s_eq, _⟩ := h
      subst h_s_eq
      exact Nat.le_refl _
  | drop =>
      simp [lowerInstr] at h
      rcases hpop : s.popSym with _ | ⟨sv, s1⟩
      · simp [hpop] at h
      simp [hpop] at h
      obtain ⟨h_s_eq, _⟩ := h
      subst h_s_eq
      rw [LowerState.popSym_nextReg hpop]
      exact Nat.le_refl _
  | localGet i =>
      simp [lowerInstr] at h
      rcases hbuf : s.lookupBufferSlot i with _ | slot
      · rw [hbuf] at h
        simp [Option.bind_eq_bind] at h
        rcases hcur : s.lookupCurrentReg i with _ | curReg
        · simp [hcur, Option.orElse] at h
          rcases hloc : s.lookupLocal i with _ | stable
          · simp [hloc] at h
          · simp [hloc, LowerState.alloc, LowerState.push] at h
            obtain ⟨h_s_eq, _⟩ := h
            subst h_s_eq
            exact Nat.le_succ _
        · simp [hcur, Option.orElse, LowerState.alloc, LowerState.push] at h
          obtain ⟨h_s_eq, _⟩ := h
          subst h_s_eq
          exact Nat.le_succ _
      · rw [hbuf] at h
        simp [LowerState.pushSym] at h
        obtain ⟨h_s_eq, _⟩ := h
        subst h_s_eq
        exact Nat.le_refl _
  | localSet i =>
      simp [lowerInstr] at h
      rcases hpop : s.popSym with _ | ⟨sv, s1⟩
      · simp [hpop] at h
      simp only [hpop, Option.bind_eq_bind, Option.some_bind] at h
      rcases hc : s1.commit sv with _ | ⟨src, s2, opsCommit⟩
      · simp [hc] at h
      simp only [hc, Option.some_bind] at h
      simp only [LowerState.alloc] at h
      rcases hlk : ({ nextReg := s2.nextReg + 1, stack := s2.stack,
                       localReg := s2.localReg, localTy := s2.localTy,
                       bufferSlots := s2.bufferSlots, currentReg := s2.currentReg }
                       : LowerState).lookupLocal i with _ | stable
      · simp [hlk, LowerState.setLocalReg, LowerState.setCurrentReg] at h
        obtain ⟨h_s_eq, _⟩ := h
        subst h_s_eq
        have h1 := LowerState.popSym_nextReg hpop
        have h2 := LowerState.commit_nextReg_mono hc
        show s.nextReg ≤ s2.nextReg + 1 + 1
        omega
      · simp [hlk, LowerState.setLocalReg, LowerState.setCurrentReg] at h
        obtain ⟨h_s_eq, _⟩ := h
        subst h_s_eq
        have h1 := LowerState.popSym_nextReg hpop
        have h2 := LowerState.commit_nextReg_mono hc
        show s.nextReg ≤ s2.nextReg + 1
        omega
  | localTee i =>
      simp [lowerInstr] at h
      rcases hpop : s.popSym with _ | ⟨sv, s1⟩
      · simp [hpop] at h
      simp only [hpop, Option.bind_eq_bind, Option.some_bind] at h
      rcases hc : s1.commit sv with _ | ⟨src, s2, opsCommit⟩
      · simp [hc] at h
      simp only [hc, Option.some_bind] at h
      simp only [LowerState.alloc, LowerState.push] at h
      rcases hlk : ({ nextReg := s2.nextReg + 1, stack := s2.stack,
                       localReg := s2.localReg, localTy := s2.localTy,
                       bufferSlots := s2.bufferSlots, currentReg := s2.currentReg }
                       : LowerState).lookupLocal i with _ | stable
      · simp [hlk, LowerState.setLocalReg, LowerState.setCurrentReg] at h
        obtain ⟨h_s_eq, _⟩ := h
        subst h_s_eq
        have h1 := LowerState.popSym_nextReg hpop
        have h2 := LowerState.commit_nextReg_mono hc
        show s.nextReg ≤ s2.nextReg + 1 + 1 + 1
        omega
      · simp [hlk, LowerState.setLocalReg, LowerState.setCurrentReg] at h
        obtain ⟨h_s_eq, _⟩ := h
        subst h_s_eq
        have h1 := LowerState.popSym_nextReg hpop
        have h2 := LowerState.commit_nextReg_mono hc
        show s.nextReg ≤ s2.nextReg + 1 + 1
        omega
  | i32Add =>
      unfold lowerInstr at h
      unfold lowerI32Add at h
      rcases hs : s.stack with _ | ⟨sv1, rs⟩
      · rw [hs] at h; exact lowerI32Bin_nextReg_mono h
      · rw [hs] at h
        cases sv1 with
        | scaledIdx base scale =>
            cases rs with
            | nil => exact lowerI32Bin_nextReg_mono h
            | cons sv2 rs2 =>
                cases sv2 with
                | bufferPtr slot =>
                    simp at h
                    obtain ⟨h_s_eq, _⟩ := h
                    subst h_s_eq; exact Nat.le_refl _
                | reg _ _ => exact lowerI32Bin_nextReg_mono h
                | i32ConstSym _ => exact lowerI32Bin_nextReg_mono h
                | scaledIdx _ _ => exact lowerI32Bin_nextReg_mono h
                | bufferAccess _ _ _ => exact lowerI32Bin_nextReg_mono h
        | bufferPtr slot =>
            cases rs with
            | nil => exact lowerI32Bin_nextReg_mono h
            | cons sv2 rs2 =>
                cases sv2 with
                | scaledIdx base scale =>
                    simp at h
                    obtain ⟨h_s_eq, _⟩ := h
                    subst h_s_eq; exact Nat.le_refl _
                | reg _ _ => exact lowerI32Bin_nextReg_mono h
                | i32ConstSym _ => exact lowerI32Bin_nextReg_mono h
                | bufferPtr _ => exact lowerI32Bin_nextReg_mono h
                | bufferAccess _ _ _ => exact lowerI32Bin_nextReg_mono h
        | reg _ _ => exact lowerI32Bin_nextReg_mono h
        | i32ConstSym _ => exact lowerI32Bin_nextReg_mono h
        | bufferAccess _ _ _ => exact lowerI32Bin_nextReg_mono h
  | i32Sub => exact lowerI32Bin_nextReg_mono h
  | i32Mul => exact lowerI32Bin_nextReg_mono h
  | i32And => exact lowerI32Bin_nextReg_mono h
  | i32Or  => exact lowerI32Bin_nextReg_mono h
  | i32Xor => exact lowerI32Bin_nextReg_mono h
  | i32Shl =>
      unfold lowerInstr at h
      unfold lowerI32Shl at h
      rcases hs : s.stack with _ | ⟨sv1, rs⟩
      · rw [hs] at h; exact lowerI32Bin_nextReg_mono h
      · rw [hs] at h
        cases sv1 with
        | i32ConstSym k =>
            cases rs with
            | nil => exact lowerI32Bin_nextReg_mono h
            | cons sv2 rs2 =>
                cases sv2 with
                | reg base ty =>
                    simp at h
                    obtain ⟨h_s_eq, _⟩ := h
                    subst h_s_eq; exact Nat.le_refl _
                | i32ConstSym _ => exact lowerI32Bin_nextReg_mono h
                | bufferPtr _ => exact lowerI32Bin_nextReg_mono h
                | scaledIdx _ _ => exact lowerI32Bin_nextReg_mono h
                | bufferAccess _ _ _ => exact lowerI32Bin_nextReg_mono h
        | reg _ _ => exact lowerI32Bin_nextReg_mono h
        | bufferPtr _ => exact lowerI32Bin_nextReg_mono h
        | scaledIdx _ _ => exact lowerI32Bin_nextReg_mono h
        | bufferAccess _ _ _ => exact lowerI32Bin_nextReg_mono h
  | i32ShrU => exact lowerI32Bin_nextReg_mono h
  | i32DivU => exact lowerI32Bin_nextReg_mono h
  | i32RemU => exact lowerI32Bin_nextReg_mono h
  | i32Eq  => exact lowerI32Cmp_nextReg_mono h
  | i32Ne  => exact lowerI32Cmp_nextReg_mono h
  | i32LtU => exact lowerI32Cmp_nextReg_mono h
  | i32LeU => exact lowerI32Cmp_nextReg_mono h
  | i32GtU => exact lowerI32Cmp_nextReg_mono h
  | i32GeU => exact lowerI32Cmp_nextReg_mono h
  | i32Load _ _ =>
      -- Reuse existing wellScoped-preservation proof's load-arm structure.
      -- The success case bumps nextReg by 1 via alloc; failure cases
      -- contradict h. We delegate via unfold and direct unfolding.
      unfold lowerInstr at h
      unfold lowerI32Load at h
      rcases hs : s.stack with _ | ⟨sv, rs⟩
      · rw [hs] at h; simp at h
      · rw [hs] at h
        cases sv with
        | bufferAccess slot base scale =>
            by_cases hscale : scale = 4
            · subst hscale
              simp [LowerState.alloc] at h
              obtain ⟨h_s_eq, _⟩ := h
              subst h_s_eq
              exact Nat.le_succ _
            · exfalso
              simp only at h
              split at h
              · rename_i _ _ _ _ hp
                cases hp
                exact hscale rfl
              · exact Option.noConfusion h
        | reg _ _ => simp at h
        | i32ConstSym _ => simp at h
        | bufferPtr _ => simp at h
        | scaledIdx _ _ => simp at h
  | i32Store _ _ =>
      unfold lowerInstr at h
      unfold lowerI32Store at h
      rcases h1 : s.popSym with _ | ⟨sv_val, s1⟩
      · simp [h1] at h
      simp only [h1, Option.bind_eq_bind, Option.some_bind] at h
      rcases h2 : s1.popSym with _ | ⟨sv_addr, s2⟩
      · simp [h2] at h
      simp only [h2, Option.bind_eq_bind, Option.some_bind] at h
      rcases hc : s2.commit sv_val with _ | ⟨src, s3, opsCommit⟩
      · simp [hc] at h
      simp only [hc, Option.bind_eq_bind, Option.some_bind] at h
      cases sv_addr with
      | bufferAccess slot base scale =>
          by_cases hscale : scale = 4
          · subst hscale
            simp at h
            obtain ⟨h_s_eq, _⟩ := h
            subst h_s_eq
            have hh1 := LowerState.popSym_nextReg h1
            have hh2 := LowerState.popSym_nextReg h2
            have hh3 := LowerState.commit_nextReg_mono hc
            omega
          · exfalso
            simp only [pure, Pure.pure] at h
            split at h
            · rename_i _ _ _ hp; cases hp; exact hscale rfl
            · exact Option.noConfusion h
      | reg _ _ => simp at h
      | i32ConstSym _ => simp at h
      | bufferPtr _ => simp at h
      | scaledIdx _ _ => simp at h
  -- Unsupported arms: contradiction.
  | i64Const _ => simp [lowerInstr] at h
  | f32Const _ => simp [lowerInstr] at h
  | f64Const _ => simp [lowerInstr] at h
  | i32DivS    => simp [lowerInstr] at h
  | i32RemS    => simp [lowerInstr] at h
  | i32ShrS    => simp [lowerInstr] at h
  | i32LtS     => simp [lowerInstr] at h
  | i32GtS     => simp [lowerInstr] at h
  | i32LeS     => simp [lowerInstr] at h
  | i32GeS     => simp [lowerInstr] at h
  | i32Eqz     => simp [lowerInstr] at h
  | f32Add | f32Sub | f32Mul | f32Div => all_goals simp [lowerInstr] at h
  | f32Eq | f32Ne | f32Lt | f32Gt | f32Le | f32Ge =>
      all_goals simp [lowerInstr] at h
  | f32Neg | f32Abs | f32Sqrt | f32Min | f32Max =>
      all_goals simp [lowerInstr] at h
  | i32WrapI64 | f32ConvertI32S | f32ConvertI32U =>
      all_goals simp [lowerInstr] at h
  | i32TruncF32S | i32TruncF32U =>
      all_goals simp [lowerInstr] at h
  | f32ReinterpretI32 | i32ReinterpretF32 =>
      all_goals simp [lowerInstr] at h
  | f32Load _ _ | f32Store _ _ =>
      all_goals simp [lowerInstr] at h
  | i32Load8U _ _ | i32Load8S _ _ | i32Store8 _ _ =>
      all_goals simp [lowerInstr] at h
  | block _ | wloop _ | wif _ =>
      all_goals simp [lowerInstr] at h
  | welse | wend    => all_goals simp [lowerInstr] at h
  | br _ | brIf _   => all_goals simp [lowerInstr] at h
  | call _          => simp [lowerInstr] at h
  | wselect         => simp [lowerInstr] at h
  | unreachable     => simp [lowerInstr] at h
  | unsupported _   => simp [lowerInstr] at h

-- ════════════════════════════════════════════════════════════════════
-- List-level theorem: straight-line subset
--
-- For instruction lists containing ONLY straight-line ops (no
-- block/wloop/wif/br/brIf), lowerInstrs dispatches uniformly through
-- the catch-all arm: lowerInstr s i, then recursive call on rest.
-- The list-level scope-validity follows by induction.
--
-- Structured-control arms are queued for follow-up.
-- ════════════════════════════════════════════════════════════════════

/-- The catch-all arm subset: every WASM instruction that
    `lowerInstrs`'s outer `match` routes to the generic chain
    (instead of a structured-control arm). -/
def WasmInstr.straightLine : WasmInstr → Prop
  | .block _ => False
  | .wloop _ => False
  | .wif _   => False
  | .br _    => False
  | .brIf _  => False
  | _        => True

/-- A list of WasmInstr is straight-line when every instruction is. -/
def WasmInstrs.straightLine (instrs : List WasmInstr) : Prop :=
  ∀ i ∈ instrs, WasmInstr.straightLine i

/-- Helper: when `i` is a straight-line instruction, `lowerInstrs` on
    `i :: rest` reduces to the catch-all `do (s1, ops1) ← lowerInstr s i;
    (s2, ops2) ← lowerInstrs fuel frames s1 rest; pure (s2, ops1 ++ ops2)`. -/
private theorem lowerInstrs_cons_straightLine_eq
    (fuel : Nat) (frames : List FrameKind) (s : LowerState)
    (i : WasmInstr) (rest : List WasmInstr)
    (hi : WasmInstr.straightLine i) :
    lowerInstrs fuel frames s (i :: rest) =
      (do let (s1, ops1) ← lowerInstr s i
          let (s2, ops2) ← lowerInstrs fuel frames s1 rest
          pure (s2, ops1 ++ ops2)) := by
  cases i <;> first
    | exact absurd hi id  -- structured arms (block/wloop/wif/br/brIf)
    | (simp only [lowerInstrs])  -- catch-all arms reduce by simp/unfold

/-- For straight-line instr lists, lowerInstrs preserves wellScoped. -/
theorem lowerInstrs_preserves_wellScoped_straightLine :
    ∀ (fuel : Nat) (frames : List FrameKind) (instrs : List WasmInstr)
      (s s' : LowerState) (ops : List KernelOp),
    WasmInstrs.straightLine instrs →
    s.wellScoped →
    lowerInstrs fuel frames s instrs = some (s', ops) →
    s'.wellScoped := by
  intro fuel frames instrs
  induction instrs generalizing fuel frames with
  | nil =>
    intro s s' ops _ hws h
    simp [lowerInstrs] at h
    obtain ⟨h_s_eq, _⟩ := h
    subst h_s_eq; exact hws
  | cons i rest ih =>
    intro s s' ops hsl hws h
    have hi : WasmInstr.straightLine i := hsl i (List.mem_cons_self _ _)
    have hrest : WasmInstrs.straightLine rest := fun j hj =>
      hsl j (List.mem_cons_of_mem _ hj)
    rw [lowerInstrs_cons_straightLine_eq fuel frames s i rest hi] at h
    rcases hi1 : lowerInstr s i with _ | ⟨s1, ops1⟩
    · simp [hi1] at h
    simp only [hi1, Option.bind_eq_bind, Option.some_bind] at h
    rcases hi2 : lowerInstrs fuel frames s1 rest with _ | ⟨s2, ops2⟩
    · simp [hi2] at h
    simp only [hi2, Option.some_bind, pure, Pure.pure] at h
    -- h : some (s2, ops1 ++ ops2) = some (s', ops)
    have h_eq : (s2, ops1 ++ ops2) = (s', ops) := Option.some.inj h
    have h_s_eq : s2 = s' := (Prod.mk.inj h_eq).1
    subst h_s_eq
    have hws1 : s1.wellScoped := lowerInstr_preserves_wellScoped hws hi1
    exact ih fuel frames _ _ _ hrest hws1 hi2

/-- For straight-line instr lists, lowerInstrs has nextReg monotonicity. -/
theorem lowerInstrs_nextReg_mono_straightLine :
    ∀ (fuel : Nat) (frames : List FrameKind) (instrs : List WasmInstr)
      (s s' : LowerState) (ops : List KernelOp),
    WasmInstrs.straightLine instrs →
    lowerInstrs fuel frames s instrs = some (s', ops) →
    s.nextReg ≤ s'.nextReg := by
  intro fuel frames instrs
  induction instrs generalizing fuel frames with
  | nil =>
    intro s s' ops _ h
    simp [lowerInstrs] at h
    obtain ⟨h_s_eq, _⟩ := h
    subst h_s_eq; exact Nat.le_refl _
  | cons i rest ih =>
    intro s s' ops hsl h
    have hi : WasmInstr.straightLine i := hsl i (List.mem_cons_self _ _)
    have hrest : WasmInstrs.straightLine rest := fun j hj =>
      hsl j (List.mem_cons_of_mem _ hj)
    rw [lowerInstrs_cons_straightLine_eq fuel frames s i rest hi] at h
    rcases hi1 : lowerInstr s i with _ | ⟨s1, ops1⟩
    · simp [hi1] at h
    simp only [hi1, Option.bind_eq_bind, Option.some_bind] at h
    rcases hi2 : lowerInstrs fuel frames s1 rest with _ | ⟨s2, ops2⟩
    · simp [hi2] at h
    simp only [hi2, Option.some_bind, pure, Pure.pure] at h
    have h_eq : (s2, ops1 ++ ops2) = (s', ops) := Option.some.inj h
    have h_s_eq : s2 = s' := (Prod.mk.inj h_eq).1
    subst h_s_eq
    have h1 : s.nextReg ≤ s1.nextReg := lowerInstr_nextReg_mono hi1
    have h2 : s1.nextReg ≤ s2.nextReg := ih fuel frames _ _ _ hrest hi2
    exact Nat.le_trans h1 h2

/-- For straight-line instr lists, lowerInstrs produces scope-valid
    output against the post-state's scopeEnv. -/
theorem lowerInstrs_scopeValid_ops_straightLine :
    ∀ (fuel : Nat) (frames : List FrameKind) (instrs : List WasmInstr)
      (s s' : LowerState) (ops : List KernelOp),
    WasmInstrs.straightLine instrs →
    s.wellScoped →
    lowerInstrs fuel frames s instrs = some (s', ops) →
    scopeValidOps s'.scopeEnv ops := by
  intro fuel frames instrs
  induction instrs generalizing fuel frames with
  | nil =>
    intro s s' ops _ _ h
    simp [lowerInstrs] at h
    obtain ⟨_, h_ops⟩ := h
    subst h_ops; exact trivial
  | cons i rest ih =>
    intro s s' ops hsl hws h
    have hi : WasmInstr.straightLine i := hsl i (List.mem_cons_self _ _)
    have hrest : WasmInstrs.straightLine rest := fun j hj =>
      hsl j (List.mem_cons_of_mem _ hj)
    rw [lowerInstrs_cons_straightLine_eq fuel frames s i rest hi] at h
    rcases hi1 : lowerInstr s i with _ | ⟨s1, ops1⟩
    · simp [hi1] at h
    simp only [hi1, Option.bind_eq_bind, Option.some_bind] at h
    rcases hi2 : lowerInstrs fuel frames s1 rest with _ | ⟨s2, ops2⟩
    · simp [hi2] at h
    simp only [hi2, Option.some_bind, pure, Pure.pure] at h
    have h_eq : (s2, ops1 ++ ops2) = (s', ops) := Option.some.inj h
    have h_s_eq : s2 = s' := (Prod.mk.inj h_eq).1
    have h_ops : ops1 ++ ops2 = ops := (Prod.mk.inj h_eq).2
    subst h_s_eq; subst h_ops
    have hops1_s1 : scopeValidOps s1.scopeEnv ops1 :=
      lowerInstr_scopeValid hws hi1
    have hws1 : s1.wellScoped := lowerInstr_preserves_wellScoped hws hi1
    have hops2 : scopeValidOps s2.scopeEnv ops2 :=
      ih fuel frames _ _ _ hrest hws1 hi2
    have hmono : s1.nextReg ≤ s2.nextReg :=
      lowerInstrs_nextReg_mono_straightLine fuel frames _ _ _ _ hrest hi2
    have hsub : s1.scopeEnv ⊆ s2.scopeEnv :=
      LowerState.scopeEnv_subset_of_nextReg_le hmono
    have hops1_s' : scopeValidOps s2.scopeEnv ops1 :=
      Quanta.KOps.KernelOp.scopeValidOps_mono ops1 hsub hops1_s1
    have hsuper : s2.scopeEnv ⊆ extendEnvOps s2.scopeEnv ops1 :=
      Quanta.KOps.KernelOp.extendEnvOps_super _ _
    have hops2_ext : scopeValidOps (extendEnvOps s2.scopeEnv ops1) ops2 :=
      Quanta.KOps.KernelOp.scopeValidOps_mono ops2 hsuper hops2
    exact Quanta.KOps.KernelOp.scopeValidOps_append _ _ _ hops1_s' hops2_ext

-- ════════════════════════════════════════════════════════════════════
-- Structured-control arms: br
--
-- `.br depth` is an unconditional jump; code after is dead, so the
-- arm doesn't recurse on `rest`. Emits either [] or [.breakOp]
-- against the unmodified state.
-- ════════════════════════════════════════════════════════════════════

/-- Key shape lemma: every successful `.br` arm leaves `s` unchanged
    and emits ops with no operand reads (either `[]` or `[.breakOp]`).
    Extracting this once dispatches all three (nextReg / wellScoped /
    scopeValid) consequences. -/
private theorem lowerInstrs_br_shape
    {fuel : Nat} {frames : List FrameKind} {depth : Nat} {rest : List WasmInstr}
    {s s' : LowerState} {ops : List KernelOp}
    (h : lowerInstrs fuel frames s (.br depth :: rest) = some (s', ops)) :
    s' = s ∧ (ops = [] ∨ ops = [.breakOp]) := by
  simp only [lowerInstrs] at h
  -- Outer split: none vs some loopK vs some block vs some wif (4 cases).
  split at h
  · -- none case
    exact absurd h Option.noConfusion
  · -- some .loopK: nested if depth = 0 / hasLoopAbove
    split at h
    · obtain ⟨hs, hops⟩ := Prod.mk.inj (Option.some.inj h)
      exact ⟨hs.symm, Or.inl hops.symm⟩
    · split at h
      · obtain ⟨hs, hops⟩ := Prod.mk.inj (Option.some.inj h)
        exact ⟨hs.symm, Or.inr hops.symm⟩
      · obtain ⟨hs, hops⟩ := Prod.mk.inj (Option.some.inj h)
        exact ⟨hs.symm, Or.inl hops.symm⟩
  · -- some _ (block or wif): single if hasLoopAbove then [.breakOp] else none
    split at h
    · obtain ⟨hs, hops⟩ := Prod.mk.inj (Option.some.inj h)
      exact ⟨hs.symm, Or.inr hops.symm⟩
    · exact absurd h Option.noConfusion

theorem lowerInstrs_br_nextReg_mono
    {fuel : Nat} {frames : List FrameKind} {depth : Nat} {rest : List WasmInstr}
    {s s' : LowerState} {ops : List KernelOp}
    (h : lowerInstrs fuel frames s (.br depth :: rest) = some (s', ops)) :
    s.nextReg ≤ s'.nextReg := by
  obtain ⟨hs_eq, _⟩ := lowerInstrs_br_shape h
  rw [hs_eq]; exact Nat.le_refl _

theorem lowerInstrs_br_preserves_wellScoped
    {fuel : Nat} {frames : List FrameKind} {depth : Nat} {rest : List WasmInstr}
    {s s' : LowerState} {ops : List KernelOp}
    (hws : s.wellScoped)
    (h : lowerInstrs fuel frames s (.br depth :: rest) = some (s', ops)) :
    s'.wellScoped := by
  obtain ⟨hs_eq, _⟩ := lowerInstrs_br_shape h
  rw [hs_eq]; exact hws

theorem lowerInstrs_br_scopeValid
    {fuel : Nat} {frames : List FrameKind} {depth : Nat} {rest : List WasmInstr}
    {s s' : LowerState} {ops : List KernelOp}
    (h : lowerInstrs fuel frames s (.br depth :: rest) = some (s', ops)) :
    scopeValidOps s'.scopeEnv ops := by
  obtain ⟨_, hops⟩ := lowerInstrs_br_shape h
  rcases hops with rfl | rfl
  · exact trivial
  · refine ⟨?_, trivial⟩
    intro r hr; simp [KernelOp.usedRegs] at hr

-- ════════════════════════════════════════════════════════════════════
-- Master nextReg monotonicity for lowerInstrs (ALL arms) — start
--
-- Built via lowerInstrs.induct, dispatching all 18 cases. Each
-- recursive case (block/wloop/wif/brIf/catch-all) chains through
-- its IH(s).
-- ════════════════════════════════════════════════════════════════════

theorem lowerInstrs_nextReg_mono :
    ∀ (fuel : Nat) (frames : List FrameKind) (s : LowerState) (instrs : List WasmInstr)
      {s' : LowerState} {ops : List KernelOp},
      lowerInstrs fuel frames s instrs = some (s', ops) →
      s.nextReg ≤ s'.nextReg := by
  intro fuel frames s instrs
  induction fuel, frames, s, instrs using lowerInstrs.induct with
  | case1 _ _ _ =>
    intro s' ops h
    simp [lowerInstrs] at h
    obtain ⟨h_s_eq, _⟩ := h
    subst h_s_eq; exact Nat.le_refl _
  | case2 _ _ _ _ =>
    intro s' ops h; simp [lowerInstrs] at h
  | case3 _ _ _ _ _ hsplit =>
    intro s' ops h
    simp [lowerInstrs, hsplit] at h
  | case4 frames s rest arity f body post hsplit ih2 ih1 =>
    intro s' ops h
    simp [lowerInstrs, hsplit] at h
    rcases hb : lowerInstrs f (.block :: frames) s body with _ | ⟨s1, innerOps⟩
    · simp [hb] at h
    simp only [hb, Option.bind_eq_bind, Option.some_bind] at h
    rcases hp : lowerInstrs f frames s1 post with _ | ⟨s2, postOps⟩
    · simp [hp] at h
    simp only [hp, Option.some_bind, pure, Pure.pure] at h
    have hpair := Option.some.inj h
    have hs : s2 = s' := (Prod.mk.inj hpair).1
    subst hs
    exact Nat.le_trans (ih2 hb) (ih1 _ hp)
  | case5 _ _ _ _ =>
    intro s' ops h; simp [lowerInstrs] at h
  | case6 _ _ _ _ _ hsplit =>
    intro s' ops h
    simp [lowerInstrs, hsplit] at h
  | case7 frames s rest arity f body post hsplit =>
    rename_i ih2 ih1
    intro s' ops h
    simp [lowerInstrs, hsplit] at h
    rcases hb : lowerInstrs f (.loopK :: frames) s body with _ | ⟨s1, bodyOps⟩
    · simp [hb] at h
    simp only [hb, Option.bind_eq_bind, Option.some_bind] at h
    rcases hp : lowerInstrs f frames
      { nextReg := s1.nextReg, stack := s1.stack,
        localReg := s.localReg, localTy := s.localTy,
        bufferSlots := s1.bufferSlots, currentReg := s.currentReg }
      post with _ | ⟨s2, postOps⟩
    · simp [hp] at h
    simp only [hp, Option.some_bind, pure, Pure.pure] at h
    have hpair := Option.some.inj h
    have hs : s2 = s' := (Prod.mk.inj hpair).1
    subst hs
    have h1 : s.nextReg ≤ s1.nextReg := ih2 hb
    have h2 : s1.nextReg ≤ s2.nextReg := ih1 _ hp
    exact Nat.le_trans h1 h2
  | case8 _ _ _ _ =>
    intro s' ops h; simp [lowerInstrs] at h
  | case9 _ _ _ _ _ hsplit =>
    intro s' ops h
    simp [lowerInstrs, hsplit] at h
  | case10 frames s rest arity f thenBody elseBody post hsplit =>
    rename_i ih3 ih2 ih1
    intro s' ops h
    simp only [lowerInstrs, hsplit] at h
    rcases hpop : s.popSym with _ | ⟨svCond, s0⟩
    · simp [hpop] at h
    simp only [hpop, Option.bind_eq_bind, Option.some_bind] at h
    rcases hc : s0.commit svCond with _ | ⟨cond, s1, opsCommit⟩
    · simp [hc] at h
    simp only [hc, Option.some_bind] at h
    simp only [LowerState.alloc] at h
    -- Recursively lower thenBody on s_cast.
    rcases hth : lowerInstrs f (.wif :: frames)
        { nextReg := s1.nextReg + 1, stack := s1.stack,
          localReg := s1.localReg, localTy := s1.localTy,
          bufferSlots := s1.bufferSlots, currentReg := s1.currentReg }
        thenBody with _ | ⟨s2, thenOps⟩
    · simp [hth] at h
    simp only [hth, Option.some_bind] at h
    -- Recursively lower elseBody on s2_restored.
    rcases hel : lowerInstrs f (.wif :: frames)
        { nextReg := s2.nextReg, stack := s2.stack,
          localReg := s1.localReg, localTy := s1.localTy,
          bufferSlots := s2.bufferSlots, currentReg := s1.currentReg }
        elseBody with _ | ⟨s3, elseOps⟩
    · simp [hel] at h
    simp only [hel, Option.some_bind] at h
    -- Recursively lower post on s3_restored.
    rcases hpo : lowerInstrs f frames
        { nextReg := s3.nextReg, stack := s3.stack,
          localReg := s1.localReg, localTy := s1.localTy,
          bufferSlots := s3.bufferSlots, currentReg := s1.currentReg }
        post with _ | ⟨s4, postOps⟩
    · simp [hpo] at h
    simp only [hpo, Option.some_bind, pure, Pure.pure] at h
    have hpair := Option.some.inj h
    have hs : s4 = s' := (Prod.mk.inj hpair).1
    subst hs
    have hp0 : s.nextReg = s0.nextReg := (LowerState.popSym_nextReg hpop).symm
    have hp1 : s0.nextReg ≤ s1.nextReg := LowerState.commit_nextReg_mono hc
    have hp3 : s1.nextReg + 1 ≤ s2.nextReg := ih3 _ hth
    -- s_cast for ih2 is the alloc'd state; its localReg = s1.localReg
    -- so the s2_restored expected by ih2 matches the one we used in hel.
    let s_cast : LowerState :=
      { nextReg := s1.nextReg + 1, stack := s1.stack,
        localReg := s1.localReg, localTy := s1.localTy,
        bufferSlots := s1.bufferSlots, currentReg := s1.currentReg }
    have hp5 : s2.nextReg ≤ s3.nextReg := ih2 s_cast s2 hel
    have hp7 : s3.nextReg ≤ s4.nextReg := ih1 s_cast s3 hpo
    omega
  | case11 _ _ _ _ _ hg =>
    intro s' ops h
    simp only [lowerInstrs] at h
    rw [hg] at h
    simp at h
  | case12 _ _ _ _ hg =>
    intro s' ops h
    simp only [lowerInstrs] at h
    rw [hg] at h
    simp at h
    obtain ⟨h_s_eq, _⟩ := h
    subst h_s_eq; exact Nat.le_refl _
  | case13 _ _ _ _ _ hg hnz hla =>
    intro s' ops h
    simp only [lowerInstrs] at h
    rw [hg] at h
    simp [hnz, hla] at h
    obtain ⟨h_s_eq, _⟩ := h
    subst h_s_eq; exact Nat.le_refl _
  | case14 _ _ _ _ _ hg hnz hnla =>
    intro s' ops h
    simp only [lowerInstrs] at h
    rw [hg] at h
    simp [hnz, hnla] at h
    obtain ⟨h_s_eq, _⟩ := h
    subst h_s_eq; exact Nat.le_refl _
  | case15 _ _ _ _ _ val hne hg hla =>
    intro s' ops h
    simp only [lowerInstrs] at h
    rw [hg] at h
    cases val with
    | loopK => exact absurd rfl hne
    | block | wif =>
      all_goals (
        simp [hla] at h
        obtain ⟨h_s_eq, _⟩ := h
        subst h_s_eq; exact Nat.le_refl _)
  | case16 _ _ _ _ _ val hne hg hnla =>
    intro s' ops h
    simp only [lowerInstrs] at h
    rw [hg] at h
    cases val with
    | loopK => exact absurd rfl hne
    | block | wif => all_goals (simp [hnla] at h)
  | case17 fuel frames s rest depth ih1 =>
    intro s' ops h
    simp only [lowerInstrs] at h
    rcases hpop : s.popSym with _ | ⟨svCond, s0⟩
    · simp [hpop] at h
    simp only [hpop, Option.bind_eq_bind, Option.some_bind] at h
    rcases hc : s0.commit svCond with _ | ⟨cond, s1, opsCommit⟩
    · simp [hc] at h
    simp only [hc, Option.some_bind] at h
    have hp0 : s.nextReg = s0.nextReg := (LowerState.popSym_nextReg hpop).symm
    have hp1 : s0.nextReg ≤ s1.nextReg := LowerState.commit_nextReg_mono hc
    -- Dispatch on frames.get? depth
    rcases hg : frames.get? depth with _ | fk
    · rw [hg] at h; simp at h
    rw [hg] at h
    cases fk with
    | loopK =>
      simp at h
      by_cases h0 : depth = 0
      · simp [h0, LowerState.alloc] at h
        rcases hr : lowerInstrs fuel frames
          { nextReg := s1.nextReg + 1, stack := s1.stack,
            localReg := s1.localReg, localTy := s1.localTy,
            bufferSlots := s1.bufferSlots, currentReg := s1.currentReg }
          rest with _ | ⟨s2, postOps⟩
        · simp [hr] at h
        simp only [hr, Option.some_bind, pure, Pure.pure] at h
        have hpair := Option.some.inj h
        have hs : s2 = s' := (Prod.mk.inj hpair).1
        subst hs
        have hp3 : s1.nextReg + 1 ≤ s2.nextReg := ih1 _ hr
        omega
      · simp [h0] at h
        by_cases hla : hasLoopAbove frames depth
        · simp [hla, LowerState.alloc] at h
          rcases hr : lowerInstrs fuel frames
            { nextReg := s1.nextReg + 1, stack := s1.stack,
              localReg := s1.localReg, localTy := s1.localTy,
              bufferSlots := s1.bufferSlots, currentReg := s1.currentReg }
            rest with _ | ⟨s2, postOps⟩
          · simp [hr] at h
          simp only [hr, Option.some_bind, pure, Pure.pure] at h
          have hpair := Option.some.inj h
          have hs : s2 = s' := (Prod.mk.inj hpair).1
          subst hs
          have hp3 : s1.nextReg + 1 ≤ s2.nextReg := ih1 _ hr
          omega
        · simp [hla] at h
          rcases hr : lowerInstrs fuel frames s1 rest with _ | ⟨s2, postOps⟩
          · simp [hr] at h
          simp only [hr, Option.some_bind, pure, Pure.pure] at h
          have hpair := Option.some.inj h
          have hs : s2 = s' := (Prod.mk.inj hpair).1
          subst hs
          have hp3 : s1.nextReg ≤ s2.nextReg := ih1 _ hr
          omega
    | block | wif =>
      all_goals (
        simp at h
        by_cases hla : hasLoopAbove frames depth
        · simp [hla, LowerState.alloc] at h
          rcases hr : lowerInstrs fuel frames
            { nextReg := s1.nextReg + 1, stack := s1.stack,
              localReg := s1.localReg, localTy := s1.localTy,
              bufferSlots := s1.bufferSlots, currentReg := s1.currentReg }
            rest with _ | ⟨s2, postOps⟩
          · simp [hr] at h
          simp only [hr, Option.some_bind, pure, Pure.pure] at h
          have hpair := Option.some.inj h
          have hs : s2 = s' := (Prod.mk.inj hpair).1
          subst hs
          have hp3 : s1.nextReg + 1 ≤ s2.nextReg := ih1 _ hr
          omega
        · simp [hla] at h)
  | case18 fuel frames s i rest hnb hnl hnw hnbr hnbi ih1 =>
    -- catch-all: lowerInstr s i + lowerInstrs fuel frames s1 rest.
    intro s' ops h
    -- Dispatch on i. The 5 structured constructors are ruled out by
    -- the case18 hypotheses; the other ~42 fall through to the same
    -- catch-all unfolding.
    cases i <;>
      first
        | (first
            | (exact absurd rfl (hnb _))
            | (exact absurd rfl (hnl _))
            | (exact absurd rfl (hnw _))
            | (exact absurd rfl (hnbr _))
            | (exact absurd rfl (hnbi _)))
        | (rename_i x
           simp only [lowerInstrs] at h
           rcases hi1 : lowerInstr s x with _ | ⟨s1, ops1⟩
           · rw [hi1] at h; simp at h
           rw [hi1] at h
           simp only [Option.bind_eq_bind, Option.some_bind] at h
           rcases hi2 : lowerInstrs fuel frames s1 rest with _ | ⟨s2, ops2⟩
           · rw [hi2] at h; simp at h
           rw [hi2] at h
           simp only [Option.some_bind, pure, Pure.pure] at h
           have hpair := Option.some.inj h
           have hs : s2 = s' := (Prod.mk.inj hpair).1
           subst hs
           exact Nat.le_trans (lowerInstr_nextReg_mono hi1) (ih1 _ hi2))
        | (simp only [lowerInstrs] at h
           rcases hi1 : lowerInstr s _ with _ | ⟨s1, ops1⟩
           · rw [hi1] at h; simp at h
           rw [hi1] at h
           simp only [Option.bind_eq_bind, Option.some_bind] at h
           rcases hi2 : lowerInstrs fuel frames s1 rest with _ | ⟨s2, ops2⟩
           · rw [hi2] at h; simp at h
           rw [hi2] at h
           simp only [Option.some_bind, pure, Pure.pure] at h
           have hpair := Option.some.inj h
           have hs : s2 = s' := (Prod.mk.inj hpair).1
           subst hs
           exact Nat.le_trans (lowerInstr_nextReg_mono hi1) (ih1 _ hi2))

end Quanta.Wasm
