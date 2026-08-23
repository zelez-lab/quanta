/-
# Seeded locals (L14): the stable layer is list-invariant

Production allocates every declared local's stable register at function
entry (lower.rs `pre-allocate stable registers`, default-zero init), so
a kernel body never FIRST-binds a local mid-stream. This file gives the
model the same discipline as a hypothesis: `LocalsSeeded s instrs` says
the entry state already binds every local `instrs` writes. Under it —
plus `KeysNodup`, which every state reachable from a duplicate-free
entry keeps — lowering a straight-line stream leaves `s.localReg`
LIST-IDENTICAL: a seeded `local.set` takes the `some` arm and upserts
the same register in place. That equality is exactly the exit-side
condition of the while apex, so the seeded corollary discharges it.
-/

import Quanta.Wasm.LowerScopeValid
import Quanta.Wasm.TranslatePendingAgree

namespace Quanta.Wasm

open Quanta.KOps (KernelOp Reg)

-- ════════════════════════════════════════════════════════════════════
-- Written locals, seeding, key uniqueness
-- ════════════════════════════════════════════════════════════════════

/-- Local indices one instruction writes. -/
def writtenLocalsInstr : WasmInstr → List Nat
  | .localSet i => [i]
  | .localTee i => [i]
  | _ => []

/-- Local indices an instruction list writes. The stream is flat —
    `block`/`wloop`/`wif` are markers, not nested lists — so no
    recursion into bodies is needed. -/
def writtenLocals (instrs : List WasmInstr) : List Nat :=
  instrs.foldr (fun i acc => writtenLocalsInstr i ++ acc) []

@[simp] theorem writtenLocals_nil : writtenLocals [] = [] := rfl

@[simp] theorem writtenLocals_cons (i : WasmInstr) (rest : List WasmInstr) :
    writtenLocals (i :: rest) = writtenLocalsInstr i ++ writtenLocals rest := rfl

/-- The entry state binds every local `instrs` writes. -/
def LocalsSeeded (s : LowerState) (instrs : List WasmInstr) : Prop :=
  ∀ i ∈ writtenLocals instrs, (s.lookupLocal i).isSome

theorem LocalsSeeded.head {s : LowerState} {i : WasmInstr} {rest : List WasmInstr}
    (h : LocalsSeeded s (i :: rest)) :
    ∀ idx ∈ writtenLocalsInstr i, (s.lookupLocal idx).isSome := by
  intro idx hmem
  exact h idx (by rw [writtenLocals_cons]; exact List.mem_append_left _ hmem)

theorem LocalsSeeded.tail {s : LowerState} {i : WasmInstr} {rest : List WasmInstr}
    (h : LocalsSeeded s (i :: rest)) : LocalsSeeded s rest := by
  intro idx hmem
  exact h idx (by rw [writtenLocals_cons]; exact List.mem_append_right _ hmem)

/-- Seeding only reads `localReg`, so it transports along equal maps. -/
theorem LocalsSeeded.of_localReg_eq {s s' : LowerState} {instrs : List WasmInstr}
    (h_eq : s'.localReg = s.localReg) (h : LocalsSeeded s instrs) :
    LocalsSeeded s' instrs := by
  intro idx hmem
  rw [lookupLocal_find?, h_eq, ← lookupLocal_find?]
  exact h idx hmem

/-- Assoc-list key uniqueness — what makes a same-value upsert the
    identity. -/
def KeysNodup {α : Type} (xs : List (Nat × α)) : Prop :=
  (xs.map Prod.fst).Nodup

/-- Two entries with the same key in a duplicate-free list are the
    same entry. -/
theorem KeysNodup.eq_of_mem {α : Type} {xs : List (Nat × α)}
    (hnd : KeysNodup xs) {p q : Nat × α}
    (hp : p ∈ xs) (hq : q ∈ xs) (hf : p.fst = q.fst) : p = q := by
  induction xs with
  | nil => cases hp
  | cons hd tl ih =>
      have hnd' : (hd.fst :: tl.map Prod.fst).Nodup := hnd
      have h_head : hd.fst ∉ tl.map Prod.fst := (List.nodup_cons.mp hnd').1
      have h_tl : KeysNodup tl := (List.nodup_cons.mp hnd').2
      rcases List.mem_cons.mp hp with hp_hd | hp_tl <;>
      rcases List.mem_cons.mp hq with hq_hd | hq_tl
      · rw [hp_hd, hq_hd]
      · exact absurd (List.mem_map.mpr ⟨q, hq_tl, by rw [← hp_hd]; exact hf.symm⟩) h_head
      · exact absurd (List.mem_map.mpr ⟨p, hp_tl, by rw [← hq_hd]; exact hf⟩) h_head
      · exact ih h_tl hp_tl hq_tl

/-- Upserting the value an entry already holds is the identity. -/
theorem upsertAssoc_id_of_find? {α : Type} {xs : List (Nat × α)} {i : Nat}
    {v : α} (hnd : KeysNodup xs)
    (h : xs.find? (fun p => p.fst = i) = some (i, v)) :
    LowerState.upsertAssoc xs i v = xs := by
  rw [LowerState.upsertAssoc_of_find?_some v h]
  have h_entry : ((i, v) : Nat × α) ∈ xs := List.mem_of_find?_eq_some h
  have h_ptwise : xs.map (fun p => if p.fst = i then (i, v) else p) = xs.map id := by
    apply List.map_congr_left
    intro p hp
    by_cases hpi : p.fst = i
    · have hpe : p = (i, v) := hnd.eq_of_mem hp h_entry (by simpa using hpi)
      rw [if_pos hpi, hpe]
      rfl
    · rw [if_neg hpi]
      rfl
  rw [h_ptwise, List.map_id]

/-- The upsert preserves key uniqueness: the in-place branch keeps the
    key list, the cons branch adds an absent key. -/
theorem KeysNodup.upsertAssoc {α : Type} {xs : List (Nat × α)} {i : Nat}
    {v : α} (hnd : KeysNodup xs) :
    KeysNodup (LowerState.upsertAssoc xs i v) := by
  unfold LowerState.upsertAssoc
  cases hf : xs.find? (fun p => p.fst = i) with
  | none =>
      have h_absent : i ∉ xs.map Prod.fst := by
        intro hmem
        obtain ⟨p, hp, hfst⟩ := List.mem_map.mp hmem
        have := List.find?_eq_none.mp hf p hp
        simp [hfst] at this
      exact List.nodup_cons.mpr ⟨h_absent, hnd⟩
  | some e =>
      have h_keys : (xs.map (fun p => if p.fst = i then (i, v) else p)).map Prod.fst
          = xs.map Prod.fst := by
        rw [List.map_map]
        apply List.map_congr_left
        intro p _
        by_cases hpi : p.fst = i
        · simp [hpi]
        · simp [hpi]
      show ((xs.map _).map Prod.fst).Nodup
      rw [h_keys]
      exact hnd

/-- A `lookupLocal` hit names the full entry. -/
theorem find?_eq_of_lookupLocal {s : LowerState} {i : Nat} {r : Reg}
    (h : s.lookupLocal i = some r) :
    s.localReg.find? (fun p => p.fst = i) = some (i, r) := by
  rw [lookupLocal_find?] at h
  cases hf : s.localReg.find? (fun p => p.fst = i) with
  | none => rw [hf] at h; cases h
  | some e =>
      rw [hf] at h
      have he_i : e.fst = i := by simpa using List.find?_some hf
      have he_r : e.snd = r := by simpa using h
      cases e
      simp_all

-- ════════════════════════════════════════════════════════════════════
-- Per-instruction: a seeded write leaves the stable layer alone
-- ════════════════════════════════════════════════════════════════════

/-- One straight-line instruction under seeding: `localReg` is
    unchanged. Non-writing instructions already carry this in their
    `LowerFrame`; a seeded `local.set`/`local.tee` takes the `some`
    arm and upserts the register the map already holds. -/
theorem lowerInstr_localReg_seeded {s s' : LowerState} {i : WasmInstr}
    {ops : List KernelOp}
    (h_sl : StraightLineInstr i)
    (hnd : KeysNodup s.localReg)
    (h_seed : ∀ idx ∈ writtenLocalsInstr i, (s.lookupLocal idx).isSome)
    (h : lowerInstr s i = some (s', ops)) :
    s'.localReg = s.localReg := by
  by_cases h_nw : NoLocalWrite i
  · exact (lowerInstr_frame h_sl h_nw h).2.2.2.1
  · -- localSet / localTee: replay the arm with the seeded lookup.
    cases i with
    | localSet idx =>
        have h_bound : (s.lookupLocal idx).isSome :=
          h_seed idx (by simp [writtenLocalsInstr])
        simp only [lowerInstr] at h
        rcases hs : s.stack with _ | ⟨sv, rs⟩
        · simp [hs, LowerState.popSym] at h
        simp only [hs, LowerState.popSym, Option.bind_eq_bind, Option.some_bind] at h
        rcases hc : ({ s with stack := rs } : LowerState).commit sv with _ | ⟨src, s2, opsC⟩
        · simp [hc] at h
        simp only [hc, Option.some_bind, LowerState.alloc] at h
        have h_lr : s2.localReg = s.localReg := by
          have := LowerState.commit_localReg hc
          simpa using this
        rcases hlk : ({ s2 with nextReg := s2.nextReg + 1 } : LowerState).lookupLocal idx
            with _ | stable
        · -- Contradicts seeding: the lookup only reads localReg.
          exfalso
          rw [lookupLocal_find?] at hlk
          have : s.localReg.find? (fun p => p.fst = idx) = none := by
            have := hlk
            simp only [Option.map_eq_none'] at this
            rw [← h_lr]
            exact this
          rw [lookupLocal_find?, this] at h_bound
          simp at h_bound
        · simp only [hlk, LowerState.setLocalReg, LowerState.setCurrentReg, pure,
                     Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨h_s, _⟩ := h
          subst h_s
          show LowerState.upsertAssoc s2.localReg idx stable = s.localReg
          have h_lk_s : s.lookupLocal idx = some stable := by
            rw [lookupLocal_find?]
            rw [lookupLocal_find?] at hlk
            show ((s.localReg.find? (fun p => p.fst = idx)).map Prod.snd) = some stable
            rw [← h_lr]
            exact hlk
          have hf := find?_eq_of_lookupLocal h_lk_s
          rw [h_lr]
          exact upsertAssoc_id_of_find? hnd hf
    | localTee idx =>
        have h_bound : (s.lookupLocal idx).isSome :=
          h_seed idx (by simp [writtenLocalsInstr])
        simp only [lowerInstr] at h
        rcases hs : s.stack with _ | ⟨sv, rs⟩
        · simp [hs, LowerState.popSym] at h
        simp only [hs, LowerState.popSym, Option.bind_eq_bind, Option.some_bind] at h
        rcases hc : ({ s with stack := rs } : LowerState).commit sv with _ | ⟨src, s2, opsC⟩
        · simp [hc] at h
        simp only [hc, Option.some_bind, LowerState.alloc] at h
        have h_lr : s2.localReg = s.localReg := by
          have := LowerState.commit_localReg hc
          simpa using this
        rcases hlk : ({ s2 with nextReg := s2.nextReg + 1 } : LowerState).lookupLocal idx
            with _ | stable
        · exfalso
          rw [lookupLocal_find?] at hlk
          have : s.localReg.find? (fun p => p.fst = idx) = none := by
            have := hlk
            simp only [Option.map_eq_none'] at this
            rw [← h_lr]
            exact this
          rw [lookupLocal_find?, this] at h_bound
          simp at h_bound
        · simp only [hlk, LowerState.setLocalReg, LowerState.setCurrentReg,
                     LowerState.alloc, LowerState.pushSym, pure,
                     Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨h_s, _⟩ := h
          subst h_s
          show LowerState.upsertAssoc s2.localReg idx stable = s.localReg
          have h_lk_s : s.lookupLocal idx = some stable := by
            rw [lookupLocal_find?]
            rw [lookupLocal_find?] at hlk
            show ((s.localReg.find? (fun p => p.fst = idx)).map Prod.snd) = some stable
            rw [← h_lr]
            exact hlk
          have hf := find?_eq_of_lookupLocal h_lk_s
          rw [h_lr]
          exact upsertAssoc_id_of_find? hnd hf
    | nop => exact absurd trivial h_nw
    | _ => exact absurd trivial h_nw

-- ════════════════════════════════════════════════════════════════════
-- Straight-line traversals
-- ════════════════════════════════════════════════════════════════════

/-- A straight-line stream under seeding: `localReg` survives the whole
    Stage-A lowering unchanged. -/
theorem lowerInstrs_localReg_seeded {fuel : Nat} {frames : List FrameKind}
    {instrs : List WasmInstr} {s s1 : LowerState} {ops : List KernelOp}
    (h_sl : StraightLineInstrs instrs)
    (hnd : KeysNodup s.localReg)
    (h_seed : LocalsSeeded s instrs)
    (h : lowerInstrs fuel frames s instrs = some (s1, ops)) :
    s1.localReg = s.localReg := by
  induction instrs generalizing s ops with
  | nil =>
      simp only [lowerInstrs, Option.some.injEq, Prod.mk.injEq] at h
      rw [← h.1]
  | cons i rest ih =>
      obtain ⟨h_i, h_rest⟩ := h_sl
      rw [lowerInstrs_cons_default fuel frames s i rest
          (straightLine_not_structured_lower h_i)] at h
      cases hli : lowerInstr s i with
      | none => rw [hli] at h; simp at h
      | some q =>
          rw [hli] at h
          obtain ⟨s_m, ops_i⟩ := q
          simp only [Option.bind_eq_bind, Option.some_bind] at h
          cases hlr : lowerInstrs fuel frames s_m rest with
          | none => rw [hlr] at h; simp at h
          | some q2 =>
              rw [hlr] at h
              obtain ⟨s_e, ops_r⟩ := q2
              simp only [Option.some_bind, pure, Option.some.injEq, Prod.mk.injEq] at h
              obtain ⟨h_s, _⟩ := h
              subst h_s
              have h_step : s_m.localReg = s.localReg :=
                lowerInstr_localReg_seeded h_i hnd (h_seed.head) hli
              have h_rest_eq : s_e.localReg = s_m.localReg :=
                ih h_rest (by rw [h_step]; exact hnd)
                  (h_seed.tail.of_localReg_eq h_step) hlr
              rw [h_rest_eq, h_step]

/-- The Stage-B (pending) route inherits the invariance on
    straight-line streams: the pending machinery is inert there. -/
theorem lowerInstrsP_localReg_seeded {fuel : Nat} {frames : List FrameKind}
    {instrs : List WasmInstr} {s : LowerState} {p : List PendingWrap}
    {sp' : LowerStateP} {ops : List KernelOp}
    (h_sl : StraightLineInstrs instrs)
    (hnd : KeysNodup s.localReg)
    (h_seed : LocalsSeeded s instrs)
    (h : lowerInstrsP fuel frames ⟨s, p⟩ instrs = some (sp', ops)) :
    sp'.base.localReg = s.localReg := by
  have h' : lowerInstrsP fuel frames ⟨s, p⟩ (instrs ++ []) = some (sp', ops) := by
    simpa using h
  obtain ⟨s_m, ops1, ops2, h_pref, h_rest, _⟩ :=
    lowerInstrsP_straightLine_append h_sl h'
  have h_m : s_m.localReg = s.localReg :=
    lowerInstrs_localReg_seeded h_sl hnd h_seed h_pref
  -- The empty rest: sp' is s_m with the pending list untouched.
  simp only [lowerInstrsP, Option.some.injEq, Prod.mk.injEq] at h_rest
  rw [← h_rest.1, h_m]

end Quanta.Wasm
