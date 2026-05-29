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

end Quanta.Wasm
