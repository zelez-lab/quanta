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

end Quanta.Wasm
