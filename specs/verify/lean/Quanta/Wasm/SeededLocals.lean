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

open Quanta.KOps (KernelOp Reg evalOps regWrite regLookup)

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

@[simp] theorem writtenLocalsInstr_block (a : Nat) :
    writtenLocalsInstr (.block a) = [] := rfl
@[simp] theorem writtenLocalsInstr_wloop (a : Nat) :
    writtenLocalsInstr (.wloop a) = [] := rfl
@[simp] theorem writtenLocalsInstr_wend : writtenLocalsInstr .wend = [] := rfl
@[simp] theorem writtenLocalsInstr_br (d : Nat) :
    writtenLocalsInstr (.br d) = [] := rfl
@[simp] theorem writtenLocalsInstr_brIf (d : Nat) :
    writtenLocalsInstr (.brIf d) = [] := rfl

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

theorem LocalsSeeded.of_subset {s : LowerState} {a b : List WasmInstr}
    (h_sub : ∀ i, i ∈ writtenLocals a → i ∈ writtenLocals b)
    (h : LocalsSeeded s b) : LocalsSeeded s a :=
  fun i hi => h i (h_sub i hi)

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

-- ════════════════════════════════════════════════════════════════════
-- Structure plumbing: splitters reconstruct, writtenLocals distributes
-- ════════════════════════════════════════════════════════════════════

@[simp] theorem writtenLocals_append (a b : List WasmInstr) :
    writtenLocals (a ++ b) = writtenLocals a ++ writtenLocals b := by
  induction a with
  | nil => simp
  | cons i rest ih =>
      rw [List.cons_append, writtenLocals_cons, writtenLocals_cons, ih,
          List.append_assoc]

theorem LocalsSeeded.of_append_left {s : LowerState} {a b : List WasmInstr}
    (h : LocalsSeeded s (a ++ b)) : LocalsSeeded s a := by
  intro idx hmem
  exact h idx (by rw [writtenLocals_append]; exact List.mem_append_left _ hmem)

theorem LocalsSeeded.of_append_right {s : LowerState} {a b : List WasmInstr}
    (h : LocalsSeeded s (a ++ b)) : LocalsSeeded s b := by
  intro idx hmem
  exact h idx (by rw [writtenLocals_append]; exact List.mem_append_right _ hmem)

/-- The walker reconstructs its input: what it took, the closer it
    stopped at, and the rest are exactly the accumulator and the list. -/
theorem walkUntilCloser_append :
    ∀ (l : List WasmInstr) (n : Nat) (acc taken : List WasmInstr)
      (marker : WasmInstr) (rest : List WasmInstr),
    walkUntilCloser l n acc = some (taken, marker, rest) →
    taken ++ marker :: rest = acc.reverse ++ l := by
  intro l
  induction l with
  | nil => intro n acc taken marker rest h; cases h
  | cons i tl ih =>
      intro n acc taken marker rest h
      rw [walkUntilCloser.eq_def] at h
      split at h
      · exact absurd h (by simp)
      · -- depth-0 `wend`: the walker stops here.
        rename_i heq
        injection heq with h_i h_tl
        subst h_i; subst h_tl
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨h_t, h_m, h_r⟩ := h
        subst h_t; subst h_m; subst h_r
        rfl
      · -- depth-0 `welse`: same shape.
        rename_i heq
        injection heq with h_i h_tl
        subst h_i; subst h_tl
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨h_t, h_m, h_r⟩ := h
        subst h_t; subst h_m; subst h_r
        rfl
      · -- Every other instruction: consume and recurse.
        injections
        subst_vars
        have h_rec := ih _ _ _ _ _ h
        rw [h_rec]
        simp

/-- `splitAtEnd` reconstructs: the input is the body, its closer, and
    the tail. -/
theorem splitAtEnd_append {l body post : List WasmInstr}
    (h : splitAtEnd l = some (body, post)) :
    body ++ .wend :: post = l := by
  unfold splitAtEnd at h
  cases hw : walkUntilCloser l 0 [] with
  | none => rw [hw] at h; cases h
  | some t =>
      rw [hw] at h
      obtain ⟨taken, marker, rest⟩ := t
      simp only [Option.bind_eq_bind, Option.some_bind] at h
      cases marker with
      | wend =>
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨h_b, h_p⟩ := h
          subst h_b; subst h_p
          simpa using walkUntilCloser_append l 0 [] taken .wend rest hw
      | _ => cases h

-- ════════════════════════════════════════════════════════════════════
-- The do-while body under seeding
-- ════════════════════════════════════════════════════════════════════

/-- The `WhileBody` lowering leaves the stable layer alone — the seeded
    analog of `whileBody_lowering_frame`, local writes admitted. -/
theorem whileBody_localReg_of_seeded
    {fuel : Nat} {frames : List FrameKind} {pref : List WasmInstr}
    (h_sl : StraightLineInstrs pref)
    {s s' : LowerState} {ops : List KernelOp}
    (hnd : KeysNodup s.localReg) (h_seed : LocalsSeeded s pref)
    (hl : lowerInstrs fuel (.loopK :: frames) s (pref ++ [.brIf 0]) = some (s', ops)) :
    s'.localReg = s.localReg := by
  obtain ⟨s_m, ops1, ops2, hl_pref, hl_br, _⟩ :=
    lowerInstrs_straightLine_append h_sl hl
  have h_lr : s_m.localReg = s.localReg :=
    lowerInstrs_localReg_seeded h_sl hnd h_seed hl_pref
  rw [lowerInstrs_brIf0_loop_empty_tail fuel (.loopK :: frames) s_m rfl] at hl_br
  rcases h_pop : s_m.popSym with _ | ⟨svCond, s0⟩
  · rw [h_pop] at hl_br; simp at hl_br
  rw [h_pop] at hl_br
  simp only [Option.bind_eq_bind, Option.some_bind] at hl_br
  rcases h_commit : s0.commit svCond with _ | ⟨cond, s1, opsCommit⟩
  · rw [h_commit] at hl_br; simp at hl_br
  rw [h_commit] at hl_br
  simp only [Option.some_bind, LowerState.alloc, pure, Option.some.injEq,
             Prod.mk.injEq] at hl_br
  obtain ⟨h_s', _⟩ := hl_br
  subst h_s'
  simp only
  rw [LowerState.commit_localReg h_commit, LowerState.popSym_localReg h_pop, h_lr]

-- ════════════════════════════════════════════════════════════════════
-- Function-entry seeding (the model of production's pre-allocation)
-- ════════════════════════════════════════════════════════════════════

/-- Function-entry seeding of declared locals (production lower.rs,
    "Pre-allocate stable registers for every value-typed declared
    local"): one fresh register and one default-zero `Const` per
    declared local, in declaration order, bound as the local's stable
    register. Pinned against production by
    `crates/gpu/quanta-wasm-lowering/tests/lower_entry_seed.rs` — the
    per-write frame-0 declarations production hoists to the function
    head are NOT part of this stream (the model keeps them inline at
    the write sites; each is overwritten before its first read). -/
def seedLocals : List (Nat × Quanta.KOps.Scalar) → LowerState →
    LowerState × List KernelOp
  | [], s => (s, [])
  | (i, ty) :: rest, s =>
      let (r, s1) := s.alloc
      let s2 := s1.setLocalReg i r ty
      let (s3, ops) := seedLocals rest s2
      (s3, .const r (LowerState.zeroConst ty) :: ops)

/-- `KernelOp` is a nested inductive, so the pin compares through
    `Repr` — same device as the `TranslatePending` pins. -/
private def seedPinEq {α : Type} [Repr α] (a b : α) : Bool :=
  toString (repr a) == toString (repr b)

/-- The two-local witness's seed stream: from the params-bound entry
    (`n` at register 0), locals 2 and 3 seed to registers 1 and 2 with
    unsigned zeros — ops `[5]`/`[6]` of the production pin
    (`lower_entry_seed.rs`). -/
example :
    seedPinEq
      (seedLocals [(2, .u32), (3, .u32)]
        { LowerState.empty with nextReg := 1,
                                localReg := [(1, 0)], localTy := [(1, .u32)] })
      ({ LowerState.empty with nextReg := 3,
                               localReg := [(3, 2), (2, 1), (1, 0)],
                               localTy := [(3, .u32), (2, .u32), (1, .u32)] },
       [.const 1 (.u32 0), .const 2 (.u32 0)]) = true := by
  native_decide

/-- Seeding binds every declared local and keeps every prior binding. -/
theorem seedLocals_binds :
    ∀ (decls : List (Nat × Quanta.KOps.Scalar)) (s : LowerState) (j : Nat),
    ((∃ ty, (j, ty) ∈ decls) ∨ (s.lookupLocal j).isSome) →
    (((seedLocals decls s).1).lookupLocal j).isSome := by
  intro decls
  induction decls with
  | nil =>
      intro s j h
      rcases h with ⟨ty, h⟩ | h
      · cases h
      · exact h
  | cons d rest ih =>
      intro s j h
      obtain ⟨i, ty⟩ := d
      show (((seedLocals rest ((s.alloc).2.setLocalReg i (s.alloc).1 ty)).1).lookupLocal j).isSome
      apply ih
      by_cases hji : j = i
      · subst hji
        right
        rw [lookupLocal_find?]
        show ((LowerState.upsertAssoc _ j _).find? (fun p => p.fst = j)).map Prod.snd |>.isSome
        rw [LowerState.find?_upsertAssoc_self]
        rfl
      · rcases h with ⟨ty', h_mem⟩ | h_bound
        · rcases List.mem_cons.mp h_mem with h_hd | h_tl
          · exact absurd (congrArg Prod.fst h_hd) hji
          · exact Or.inl ⟨ty', h_tl⟩
        · right
          rw [lookupLocal_find?]
          show ((LowerState.upsertAssoc _ i _).find? (fun p => p.fst = j)).map Prod.snd |>.isSome
          rw [LowerState.find?_upsertAssoc_ne _ i j _ hji]
          rw [lookupLocal_find?] at h_bound
          exact h_bound

/-- Seeding preserves key uniqueness. -/
theorem seedLocals_keysNodup :
    ∀ (decls : List (Nat × Quanta.KOps.Scalar)) (s : LowerState),
    KeysNodup s.localReg → KeysNodup (((seedLocals decls s).1).localReg) := by
  intro decls
  induction decls with
  | nil => intro s h; exact h
  | cons d rest ih =>
      intro s h
      obtain ⟨i, ty⟩ := d
      exact ih _ (h.upsertAssoc)

/-- A kernel whose written locals are all declared is seeded after the
    entry stream (locals already bound at entry — the params — count). -/
theorem seedLocals_seeded
    (decls : List (Nat × Quanta.KOps.Scalar)) (s : LowerState)
    (instrs : List WasmInstr)
    (h : ∀ j ∈ writtenLocals instrs,
        (∃ ty, (j, ty) ∈ decls) ∨ (s.lookupLocal j).isSome) :
    LocalsSeeded ((seedLocals decls s).1) instrs :=
  fun j hj => seedLocals_binds decls s j (h j hj)

/-- Running the seed stream extends the refinement: from a function
    entry (empty stack and per-frame map, params already refined), each
    `Const r 0` writes a fresh register and binds a zero-initialised
    WASM local to it — the model-side face of production's entry
    pre-allocation meeting WASM's own zero-init of locals. -/
theorem seedLocals_refines
    (decls : List (Nat × Quanta.KOps.Scalar))
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_tys : ∀ p ∈ decls, p.snd = Quanta.KOps.Scalar.u32)
    (h_zero : ∀ p ∈ decls, ws.locals.get? p.fst = some (.wI32 0))
    (h_stack : s.stack = []) (h_creg : s.currentReg = [])
    (h_kst : kst.broke = false) :
    ∃ kst' : Quanta.KOps.State,
      (∀ F, evalOps F kst (seedLocals decls s).2 = some kst') ∧
      Refines ws (seedLocals decls s).1 kst' layout ∧
      kst'.broke = false ∧ kst'.heap = kst.heap := by
  induction decls generalizing s kst with
  | nil => exact ⟨kst, fun F => by simp [seedLocals, evalOps], R, h_kst, rfl⟩
  | cons d rest ih =>
      obtain ⟨i, ty⟩ := d
      have h_ty : ty = Quanta.KOps.Scalar.u32 := h_tys (i, ty) (List.mem_cons_self _ _)
      subst h_ty
      have h_zero_i : ws.locals.get? i = some (.wI32 0) :=
        h_zero (i, .u32) (List.mem_cons_self _ _)
      let s1 : LowerState := (s.alloc).2.setLocalReg i (s.alloc).1 .u32
      let kst1 : Quanta.KOps.State :=
        { kst with rf := regWrite kst.rf s.nextReg (Quanta.KOps.Value.vU32 0) }
      have h_b1 : kst1.broke = false := h_kst
      -- One `Const` evaluates to the register write.
      have h_step : ∀ F, Quanta.KOps.evalOp F kst
          (KernelOp.const s.nextReg (LowerState.zeroConst .u32)) = some kst1 := by
        intro F
        show _ = some ({ kst with
          rf := regWrite kst.rf s.nextReg (Quanta.KOps.Value.vU32 0) } : Quanta.KOps.State)
        simp [Quanta.KOps.evalOp, Quanta.KOps.evalConst, LowerState.zeroConst]
        rfl
      -- The refinement survives the step.
      have R1 : Refines ws s1 kst1 layout := by
        refine ⟨?_, ?_, ?_, ?_, ?_, R.heapRefines, ?_, ?_, ?_⟩
        · -- StackRefines: the stack is untouched; lift past the fresh write.
          refine ⟨R.stk.left, ?_⟩
          intro j v hv
          obtain ⟨svj, h_get, henc⟩ := R.stk.right j v hv
          refine ⟨svj, h_get, ?_⟩
          apply WasmValue.encodes_preserved_of_fresh _ henc
          intro r' hr'
          exact R.fresh.left svj (List.mem_of_get? h_get) r' hr'
        · -- LocalsRefines: the new binding reads its zero; the old ones lift.
          intro k q hfind v hv
          by_cases hki : k = i
          · subst hki
            rw [show s1.localReg = LowerState.upsertAssoc s.localReg k (s.alloc).1 from rfl,
                LowerState.find?_upsertAssoc_self] at hfind
            injection hfind with h_pair
            have hq : (s.alloc).1 = q := ((Prod.mk.injEq _ _ _ _).mp h_pair).2
            have hv0 : v = .wI32 0 := by
              rw [h_zero_i] at hv
              exact ((Option.some.injEq _ _).mp hv).symm
            subst hv0
            have h_tyk : localTyOf s1.localTy k = .u32 := by
              show localTyOf (LowerState.upsertAssoc s.localTy k .u32) k = _
              unfold localTyOf
              rw [LowerState.find?_upsertAssoc_self]
              rfl
            rw [h_tyk, ← hq]
            apply encodes_wI32_reg_of_tagVal (Or.inl rfl)
            show regLookup (regWrite kst.rf s.nextReg (Quanta.KOps.Value.vU32 0))
                (s.alloc).1 = some (tagVal 0 .u32)
            rw [show ((s.alloc).1 : Reg) = s.nextReg from rfl,
                regLookup_regWrite_self]
            rfl
          · rw [show s1.localReg = LowerState.upsertAssoc s.localReg i (s.alloc).1 from rfl,
                LowerState.find?_upsertAssoc_ne _ i k _ hki] at hfind
            have henc := R.locs k q hfind v hv
            have h_ty_ne : localTyOf s1.localTy k = localTyOf s.localTy k := by
              show localTyOf (LowerState.upsertAssoc s.localTy i .u32) k = _
              unfold localTyOf
              rw [LowerState.find?_upsertAssoc_ne _ i k _ hki]
            rw [h_ty_ne]
            apply WasmValue.encodes_preserved_of_fresh _ henc
            intro r' hr'
            have hrq : r' = q := by simpa [SymVal.regs] using hr'
            subst hrq
            exact R.fresh.right (k, r') (List.mem_of_find?_eq_some hfind)
        · -- Fresh: nextReg bumps; the new pair sits at the old nextReg.
          refine ⟨?_, ?_⟩
          · intro sv hsv r' hr'
            exact Nat.lt_succ_of_lt (R.fresh.left sv hsv r' hr')
          · intro ir hir
            rw [show s1.localReg = LowerState.upsertAssoc s.localReg i (s.alloc).1
                  from rfl] at hir
            rcases LowerState.mem_upsertAssoc_iff.mp hir with h_eq | ⟨h_in, _⟩
            · rw [h_eq]
              exact Nat.lt_succ_self _
            · exact Nat.lt_succ_of_lt (R.fresh.right ir h_in)
        · -- AliasFree: the fresh register is above every stack register.
          intro ir hir sv hsv
          rw [show s1.localReg = LowerState.upsertAssoc s.localReg i (s.alloc).1
                from rfl] at hir
          rcases LowerState.mem_upsertAssoc_iff.mp hir with h_eq | ⟨h_in, _⟩
          · rw [h_eq]
            intro hcontra
            exact absurd (R.fresh.left sv hsv _ hcontra) (Nat.lt_irrefl _)
          · exact R.aliasFree ir h_in sv hsv
        · -- InjectiveLocals: the new register is above every old one.
          intro p q hp hq
          rw [show s1.localReg = LowerState.upsertAssoc s.localReg i (s.alloc).1
                from rfl] at hp hq
          rcases LowerState.mem_upsertAssoc_iff.mp hp with hp_eq | ⟨hp_in, _⟩ <;>
          rcases LowerState.mem_upsertAssoc_iff.mp hq with hq_eq | ⟨hq_in, _⟩
          · rw [hp_eq, hq_eq]
            exact Or.inl rfl
          · right
            rw [hp_eq]
            exact Ne.symm (Nat.ne_of_lt (R.fresh.right q hq_in))
          · right
            rw [hq_eq]
            exact Nat.ne_of_lt (R.fresh.right p hp_in)
          · exact R.injLocals p q hp_in hq_in
        · -- CurrentRegRefines: the per-frame map is empty at entry.
          intro k q hfind v _
          rw [show s1.currentReg = s.currentReg from rfl, h_creg] at hfind
          cases hfind
        · -- FreshCurrent: empty.
          intro ir hir
          rw [show s1.currentReg = s.currentReg from rfl, h_creg] at hir
          cases hir
        · -- CurrentLocalDisjoint: empty.
          intro p q hp _
          rw [show s1.currentReg = s.currentReg from rfl, h_creg] at hp
          cases hp
      -- Recurse over the tail from the extended state.
      obtain ⟨kst', h_ops, R', h_b', h_h'⟩ :=
        ih s1 kst1 R1
          (fun p hp => h_tys p (List.mem_cons_of_mem _ hp))
          (fun p hp => h_zero p (List.mem_cons_of_mem _ hp))
          (show s1.stack = [] from h_stack)
          (show s1.currentReg = [] from h_creg)
          h_b1
      refine ⟨kst', ?_, R', h_b', by rw [h_h']⟩
      intro F
      show evalOps F kst
          (KernelOp.const (s.alloc).1 (LowerState.zeroConst .u32)
            :: (seedLocals rest s1).2) = some kst'
      simp only [evalOps, Option.bind_eq_bind]
      rw [show ((s.alloc).1 : Quanta.KOps.Reg) = s.nextReg from rfl, h_step F]
      simp only [Option.some_bind, h_b1]
      rw [if_neg (by simp [h_b1])]
      exact h_ops F

end Quanta.Wasm
