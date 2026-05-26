/-
# State-evolution invariants for `lowerInstr`

Every successful `lowerInstr s instr = some (s', ops)` step preserves
certain `LowerState` fields untouched. These invariants are the
architectural prerequisite for composing cons-case preservation
theorems into multi-step chain theorems: the next step's
preconditions (e.g., `s_mid.lookupBufferSlot idxIdx = none`) lift
from the entry state through prior steps without re-derivation.

The universal invariant for the slice-1 instruction subset is
**`bufferSlots` preservation** — no `lowerInstr` arm modifies
`s.bufferSlots`. The supporting invariants (`localReg` preservation
for non-local-write instructions, `nextReg` monotonicity) follow
the same case-by-case unfold.

This module is the foundation for chain theorems like
`preservation_evalInstrs_chain_buffer_address_prelude` in
`PreservationList`.
-/

import Quanta.Wasm.Translate

namespace Quanta.Wasm

open Quanta.KOps (Reg KernelOp)

-- ════════════════════════════════════════════════════════════════════
-- bufferSlots preservation
-- ════════════════════════════════════════════════════════════════════

/-- `LowerState.alloc` preserves `bufferSlots`. -/
theorem LowerState.alloc_preserves_bufferSlots (s : LowerState) :
    s.alloc.snd.bufferSlots = s.bufferSlots := rfl

/-- `LowerState.push` (plain reg push) preserves `bufferSlots`. -/
theorem LowerState.push_preserves_bufferSlots (s : LowerState) (r : Reg) :
    (s.push r).bufferSlots = s.bufferSlots := rfl

/-- `LowerState.pushSym` preserves `bufferSlots`. -/
theorem LowerState.pushSym_preserves_bufferSlots (s : LowerState) (sv : SymVal) :
    (s.pushSym sv).bufferSlots = s.bufferSlots := rfl

/-- `LowerState.pop` (any successful pop) preserves `bufferSlots`. -/
theorem LowerState.pop_preserves_bufferSlots {s s' : LowerState} {r : Reg}
    (h : s.pop = some (r, s')) : s'.bufferSlots = s.bufferSlots := by
  unfold LowerState.pop at h
  rcases hs : s.stack with _ | ⟨sv, rs⟩
  · rw [hs] at h; simp at h
  · rw [hs] at h
    cases sv with
    | reg r' ty =>
        simp at h
        rcases h with ⟨_, h_s_eq⟩
        rw [← h_s_eq]
    | i32ConstSym _ => simp at h
    | bufferPtr _ => simp at h
    | scaledIdx _ _ => simp at h
    | bufferAccess _ _ _ => simp at h

/-- `LowerState.popSym` (any successful pop) preserves `bufferSlots`. -/
theorem LowerState.popSym_preserves_bufferSlots {s s' : LowerState} {sv : SymVal}
    (h : s.popSym = some (sv, s')) : s'.bufferSlots = s.bufferSlots := by
  unfold LowerState.popSym at h
  rcases hs : s.stack with _ | ⟨sv', rs⟩
  · rw [hs] at h; simp at h
  · rw [hs] at h
    simp at h
    rcases h with ⟨_, h_s_eq⟩
    rw [← h_s_eq]

/-- `LowerState.setLocalReg` preserves `bufferSlots`. -/
theorem LowerState.setLocalReg_preserves_bufferSlots
    (s : LowerState) (i : Nat) (r : Reg) (ty : Quanta.KOps.Scalar) :
    (s.setLocalReg i r ty).bufferSlots = s.bufferSlots := rfl

/-- `LowerState.commit` preserves `bufferSlots` on the materialized state. -/
theorem LowerState.commit_preserves_bufferSlots
    {s : LowerState} {sv : SymVal} {r : Reg} {s' : LowerState} {ops : List KernelOp}
    (h : s.commit sv = some (r, s', ops)) : s'.bufferSlots = s.bufferSlots := by
  unfold LowerState.commit at h
  cases sv with
  | reg r' _ =>
      simp at h
      rcases h with ⟨_, h_s_eq, _⟩
      rw [← h_s_eq]
  | i32ConstSym _ =>
      simp [LowerState.alloc] at h
      rcases h with ⟨_, h_s_eq, _⟩
      rw [← h_s_eq]
  | bufferPtr _ => simp at h
  | scaledIdx _ _ => simp at h
  | bufferAccess _ _ _ => simp at h

/-- `lowerI32Bin` preserves `bufferSlots`. -/
theorem lowerI32Bin_preserves_bufferSlots
    {s s' : LowerState} {op : Quanta.KOps.BinOp} {ops : List KernelOp}
    (h : lowerI32Bin s op = some (s', ops)) :
    s'.bufferSlots = s.bufferSlots := by
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
  -- h : some (let (dst, s5) := s4.alloc; ({s5 with stack := .reg dst .u32 :: s5.stack}, ...)) = some (s', ops)
  simp [LowerState.alloc] at h
  rcases h with ⟨h_s_eq, _⟩
  -- Chain the primitives.
  have hb_inv := LowerState.popSym_preserves_bufferSlots hb
  have ha_inv := LowerState.popSym_preserves_bufferSlots ha
  have hca_inv := LowerState.commit_preserves_bufferSlots hca
  have hcb_inv := LowerState.commit_preserves_bufferSlots hcb
  rw [← h_s_eq]; show s4.bufferSlots = _
  rw [hcb_inv, hca_inv, ha_inv, hb_inv]

/-- `lowerI32Cmp` preserves `bufferSlots`. Same shape as the binop;
    the cmp emits cast op but the state mutation is structurally
    identical. -/
theorem lowerI32Cmp_preserves_bufferSlots
    {s s' : LowerState} {op : Quanta.KOps.CmpOp} {ops : List KernelOp}
    (h : lowerI32Cmp s op = some (s', ops)) :
    s'.bufferSlots = s.bufferSlots := by
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
  simp [LowerState.alloc] at h
  rcases h with ⟨h_s_eq, _⟩
  have hb_inv := LowerState.popSym_preserves_bufferSlots hb
  have ha_inv := LowerState.popSym_preserves_bufferSlots ha
  have hca_inv := LowerState.commit_preserves_bufferSlots hca
  have hcb_inv := LowerState.commit_preserves_bufferSlots hcb
  rw [← h_s_eq]; show s4.bufferSlots = _
  rw [hcb_inv, hca_inv, ha_inv, hb_inv]

/-- `lowerI32Shl` preserves `bufferSlots`. Buffer-pattern arm rewrites
    only `s.stack`; non-buffer arm dispatches to `lowerI32Bin`. -/
theorem lowerI32Shl_preserves_bufferSlots
    {s s' : LowerState} {ops : List KernelOp}
    (h : lowerI32Shl s = some (s', ops)) :
    s'.bufferSlots = s.bufferSlots := by
  unfold lowerI32Shl at h
  rcases hs : s.stack with _ | ⟨sv1, rs⟩
  · rw [hs] at h
    -- Bin path on empty stack — refused inside lowerI32Bin.
    exact lowerI32Bin_preserves_bufferSlots h
  · rw [hs] at h
    cases sv1 with
    | i32ConstSym k =>
        cases rs with
        | nil => exact lowerI32Bin_preserves_bufferSlots h
        | cons sv2 rs2 =>
            cases sv2 with
            | reg base ty =>
                -- Buffer-pattern arm: state mutation is { s with stack := ... }
                -- which preserves bufferSlots structurally.
                simp at h
                rcases h with ⟨h_s_eq, _⟩
                rw [← h_s_eq]
            | i32ConstSym _ => exact lowerI32Bin_preserves_bufferSlots h
            | bufferPtr _ => exact lowerI32Bin_preserves_bufferSlots h
            | scaledIdx _ _ => exact lowerI32Bin_preserves_bufferSlots h
            | bufferAccess _ _ _ => exact lowerI32Bin_preserves_bufferSlots h
    | reg _ _ => exact lowerI32Bin_preserves_bufferSlots h
    | bufferPtr _ => exact lowerI32Bin_preserves_bufferSlots h
    | scaledIdx _ _ => exact lowerI32Bin_preserves_bufferSlots h
    | bufferAccess _ _ _ => exact lowerI32Bin_preserves_bufferSlots h

/-- `lowerI32Add` preserves `bufferSlots`. Same shape as `lowerI32Shl` —
    buffer-pattern arms rewrite only `s.stack`; non-buffer arm
    dispatches to `lowerI32Bin`. -/
theorem lowerI32Add_preserves_bufferSlots
    {s s' : LowerState} {ops : List KernelOp}
    (h : lowerI32Add s = some (s', ops)) :
    s'.bufferSlots = s.bufferSlots := by
  unfold lowerI32Add at h
  rcases hs : s.stack with _ | ⟨sv1, rs⟩
  · rw [hs] at h
    exact lowerI32Bin_preserves_bufferSlots h
  · rw [hs] at h
    cases sv1 with
    | scaledIdx base scale =>
        cases rs with
        | nil => exact lowerI32Bin_preserves_bufferSlots h
        | cons sv2 rs2 =>
            cases sv2 with
            | bufferPtr slot =>
                simp at h
                rcases h with ⟨h_s_eq, _⟩
                rw [← h_s_eq]
            | reg _ _ => exact lowerI32Bin_preserves_bufferSlots h
            | i32ConstSym _ => exact lowerI32Bin_preserves_bufferSlots h
            | scaledIdx _ _ => exact lowerI32Bin_preserves_bufferSlots h
            | bufferAccess _ _ _ => exact lowerI32Bin_preserves_bufferSlots h
    | bufferPtr slot =>
        cases rs with
        | nil => exact lowerI32Bin_preserves_bufferSlots h
        | cons sv2 rs2 =>
            cases sv2 with
            | scaledIdx base scale =>
                simp at h
                rcases h with ⟨h_s_eq, _⟩
                rw [← h_s_eq]
            | reg _ _ => exact lowerI32Bin_preserves_bufferSlots h
            | i32ConstSym _ => exact lowerI32Bin_preserves_bufferSlots h
            | bufferPtr _ => exact lowerI32Bin_preserves_bufferSlots h
            | bufferAccess _ _ _ => exact lowerI32Bin_preserves_bufferSlots h
    | reg _ _ => exact lowerI32Bin_preserves_bufferSlots h
    | i32ConstSym _ => exact lowerI32Bin_preserves_bufferSlots h
    | bufferAccess _ _ _ => exact lowerI32Bin_preserves_bufferSlots h

/-- `lowerI32Load` preserves `bufferSlots`. Only the `bufferAccess`
    arm succeeds; the resulting state is `{s1 with stack := ...}`
    where `s1 = s.alloc.snd` — both alloc and stack-rewrite preserve
    `bufferSlots`. -/
theorem lowerI32Load_preserves_bufferSlots
    {s s' : LowerState} {ops : List KernelOp}
    (h : lowerI32Load s = some (s', ops)) :
    s'.bufferSlots = s.bufferSlots := by
  unfold lowerI32Load at h
  rcases hs : s.stack with _ | ⟨sv, rs⟩
  · rw [hs] at h; simp at h
  · rw [hs] at h
    cases sv with
    | bufferAccess slot base scale =>
        by_cases hscale : scale = 4
        · subst hscale
          simp [LowerState.alloc] at h
          rcases h with ⟨h_s_eq, _⟩
          rw [← h_s_eq]
        · exfalso
          split at h
          · rename_i _ _ _ _ hp
            -- hp : SymVal.bufferAccess slot base scale :: rs
            --      = SymVal.bufferAccess _ _ 4 :: _
            cases hp
            exact hscale rfl
          · exact Option.noConfusion h
    | reg _ _ => simp at h
    | i32ConstSym _ => simp at h
    | bufferPtr _ => simp at h
    | scaledIdx _ _ => simp at h

/-- **Master invariant**: every successful `lowerInstr` step preserves
    `bufferSlots`. Covers every instruction in the slice-1 subset; the
    remaining catch-all `_ => none` arms refuse, so the precondition
    `h` rules them out. -/
theorem lowerInstr_preserves_bufferSlots
    {s s' : LowerState} {instr : WasmInstr} {ops : List KernelOp}
    (h : lowerInstr s instr = some (s', ops)) :
    s'.bufferSlots = s.bufferSlots := by
  cases instr with
  | i32Const n =>
      simp [lowerInstr] at h
      rcases h with ⟨h_s_eq, _⟩
      rw [← h_s_eq]
  | localGet i =>
      simp [lowerInstr] at h
      rcases hbuf : s.lookupBufferSlot i with _ | slot
      · rw [hbuf] at h
        simp [Option.bind_eq_bind] at h
        -- Stage 3: localGet consults currentReg first, then localReg.
        -- Either way, the lookup result (a Reg) doesn't change bufferSlots.
        rcases hcur : s.lookupCurrentReg i with _ | curReg
        · simp [hcur, Option.orElse] at h
          rcases hloc : s.lookupLocal i with _ | stable
          · simp [hloc] at h
          · simp [hloc, LowerState.alloc, LowerState.push] at h
            rcases h with ⟨h_s_eq, _⟩
            rw [← h_s_eq]
        · simp [hcur, Option.orElse, LowerState.alloc, LowerState.push] at h
          rcases h with ⟨h_s_eq, _⟩
          rw [← h_s_eq]
      · rw [hbuf] at h
        simp [LowerState.pushSym] at h
        rcases h with ⟨h_s_eq, _⟩
        rw [← h_s_eq]
  | localSet i =>
      simp [lowerInstr] at h
      rcases hpop : s.popSym with _ | ⟨sv, s1⟩
      · simp [hpop] at h
      simp only [hpop, Option.bind_eq_bind, Option.some_bind] at h
      rcases hc : s1.commit sv with _ | ⟨src, s2, opsCommit⟩
      · simp [hc] at h
      simp only [hc, Option.some_bind] at h
      have hpop_inv := LowerState.popSym_preserves_bufferSlots hpop
      have hc_inv := LowerState.commit_preserves_bufferSlots hc
      -- Stage 3 dual-Copy: alloc fresh, then match lookupLocal on
      -- the bumped state. The alloc only bumps nextReg; lookupLocal
      -- only depends on localReg, so the result equals s2.lookupLocal.
      simp only [LowerState.alloc] at h
      -- After simp, h's match is on `{s2 with nextReg := s2.nextReg + 1}.lookupLocal`.
      -- That equals s2.lookupLocal since lookupLocal only reads localReg.
      rcases hlk : ({ nextReg := s2.nextReg + 1, stack := s2.stack,
                       localReg := s2.localReg, localTy := s2.localTy,
                       bufferSlots := s2.bufferSlots, currentReg := s2.currentReg }
                       : LowerState).lookupLocal i with _ | stable
      · simp [hlk, LowerState.setLocalReg, LowerState.setCurrentReg] at h
        rcases h with ⟨h_s_eq, _⟩
        rw [← h_s_eq]
        show s2.bufferSlots = _
        rw [hc_inv, hpop_inv]
      · simp [hlk, LowerState.setLocalReg, LowerState.setCurrentReg] at h
        rcases h with ⟨h_s_eq, _⟩
        rw [← h_s_eq]
        show s2.bufferSlots = _
        rw [hc_inv, hpop_inv]
  | localTee i =>
      simp [lowerInstr] at h
      rcases hpop : s.popSym with _ | ⟨sv, s1⟩
      · simp [hpop] at h
      simp only [hpop, Option.bind_eq_bind, Option.some_bind] at h
      rcases hc : s1.commit sv with _ | ⟨src, s2, opsCommit⟩
      · simp [hc] at h
      simp only [hc, Option.some_bind] at h
      have hpop_inv := LowerState.popSym_preserves_bufferSlots hpop
      have hc_inv := LowerState.commit_preserves_bufferSlots hc
      simp only [LowerState.alloc] at h
      rcases hlk : ({ nextReg := s2.nextReg + 1, stack := s2.stack,
                       localReg := s2.localReg, localTy := s2.localTy,
                       bufferSlots := s2.bufferSlots, currentReg := s2.currentReg }
                       : LowerState).lookupLocal i with _ | stable
      · simp [hlk, LowerState.setLocalReg, LowerState.setCurrentReg,
              LowerState.push] at h
        rcases h with ⟨h_s_eq, _⟩
        rw [← h_s_eq]
        show s2.bufferSlots = _
        rw [hc_inv, hpop_inv]
      · simp [hlk, LowerState.setLocalReg, LowerState.setCurrentReg,
              LowerState.push] at h
        rcases h with ⟨h_s_eq, _⟩
        rw [← h_s_eq]
        show s2.bufferSlots = _
        rw [hc_inv, hpop_inv]
  | i32Add  =>
      exact lowerI32Add_preserves_bufferSlots h
  | i32Sub  =>
      exact lowerI32Bin_preserves_bufferSlots h
  | i32Mul  =>
      exact lowerI32Bin_preserves_bufferSlots h
  | i32And  =>
      exact lowerI32Bin_preserves_bufferSlots h
  | i32Or   =>
      exact lowerI32Bin_preserves_bufferSlots h
  | i32Xor  =>
      exact lowerI32Bin_preserves_bufferSlots h
  | i32Shl  =>
      exact lowerI32Shl_preserves_bufferSlots h
  | i32ShrU =>
      exact lowerI32Bin_preserves_bufferSlots h
  | i32DivU =>
      exact lowerI32Bin_preserves_bufferSlots h
  | i32RemU =>
      exact lowerI32Bin_preserves_bufferSlots h
  | i32Load _ _ =>
      exact lowerI32Load_preserves_bufferSlots h
  | i32Store _ _ =>
      -- lowerI32Store: popSym + popSym + commit + match on bufferAccess.
      unfold lowerInstr at h
      unfold lowerI32Store at h
      rcases h1 : s.popSym with _ | ⟨svv, s1⟩
      · simp [h1] at h
      simp only [h1, Option.bind_eq_bind, Option.some_bind] at h
      rcases h2 : s1.popSym with _ | ⟨sva, s2⟩
      · simp [h2] at h
      simp only [h2, Option.bind_eq_bind, Option.some_bind] at h
      rcases hc : s2.commit svv with _ | ⟨src, s3, opsCommit⟩
      · simp [hc] at h
      simp only [hc, Option.bind_eq_bind, Option.some_bind] at h
      have h1_inv := LowerState.popSym_preserves_bufferSlots h1
      have h2_inv := LowerState.popSym_preserves_bufferSlots h2
      have hc_inv := LowerState.commit_preserves_bufferSlots hc
      cases sva with
      | bufferAccess slot base scale =>
          -- Only scale = 4 produces some(...); other scales yield none.
          by_cases hscale : scale = 4
          · subst hscale
            simp at h
            rcases h with ⟨h_s_eq, _⟩
            rw [← h_s_eq]; rw [hc_inv, h2_inv, h1_inv]
          · exfalso
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
  | i32Eq  =>
      exact lowerI32Cmp_preserves_bufferSlots h
  | i32Ne  =>
      exact lowerI32Cmp_preserves_bufferSlots h
  | i32LtU =>
      exact lowerI32Cmp_preserves_bufferSlots h
  | i32LeU =>
      exact lowerI32Cmp_preserves_bufferSlots h
  | i32GtU =>
      exact lowerI32Cmp_preserves_bufferSlots h
  | i32GeU =>
      exact lowerI32Cmp_preserves_bufferSlots h
  | wreturn =>
      simp [lowerInstr] at h
      rcases h with ⟨h_s_eq, _⟩
      rw [← h_s_eq]
  | nop =>
      simp [lowerInstr] at h
      rcases h with ⟨h_s_eq, _⟩
      rw [← h_s_eq]
  | drop =>
      simp [lowerInstr] at h
      rcases hpop : s.popSym with _ | ⟨sv, s1⟩
      · simp [hpop] at h
      simp [hpop] at h
      rcases h with ⟨h_s_eq, _⟩
      rw [← h_s_eq]
      exact LowerState.popSym_preserves_bufferSlots hpop
  -- Catch-all for unsupported / structured-control / float / load8 etc.
  -- arms — they all refuse with `none` in lowerInstr, contradicting h.
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
  | i32WrapI64 | f32ConvertI32S | f32ConvertI32U => all_goals simp [lowerInstr] at h
  | i32TruncF32S | i32TruncF32U => all_goals simp [lowerInstr] at h
  | f32ReinterpretI32 | i32ReinterpretF32 => all_goals simp [lowerInstr] at h
  | f32Load _ _ | f32Store _ _ => all_goals simp [lowerInstr] at h
  | i32Load8U _ _ | i32Load8S _ _ | i32Store8 _ _ => all_goals simp [lowerInstr] at h
  | block _ | wloop _ | wif _ => all_goals simp [lowerInstr] at h
  | welse | wend => all_goals simp [lowerInstr] at h
  | br _ | brIf _ => all_goals simp [lowerInstr] at h
  | call _ => simp [lowerInstr] at h
  | wselect => simp [lowerInstr] at h
  | unreachable => simp [lowerInstr] at h
  | unsupported _ => simp [lowerInstr] at h

-- ════════════════════════════════════════════════════════════════════
-- List-level lift
-- ════════════════════════════════════════════════════════════════════

/-- `lookupBufferSlot` reads only `s.bufferSlots`, so it's invariant
    along any `bufferSlots`-preserving step. -/
theorem LowerState.lookupBufferSlot_of_bufferSlots_eq
    {s s' : LowerState} (i : Nat) (h : s'.bufferSlots = s.bufferSlots) :
    s'.lookupBufferSlot i = s.lookupBufferSlot i := by
  unfold LowerState.lookupBufferSlot
  rw [h]

end Quanta.Wasm
