/-
# The loop exit flag — what it does at the site, at the loop's end, and after

rustc's `while` lowers (`emit_loop_crossing_exit`, pinned in
`TranslatePending.lean` against
`crates/gpu/quanta-wasm-lowering/tests/lower_while_exit_flag.rs`) to a
flag register threaded around a loop op:

    const flag (.bool false)                       -- the declaration
    loopOp [ …,
             cast cb c .u32 .bool,
             branch cb [const flag (.bool true), breakOp] [],   -- the exit site
             … ]
    branch flag [] tailOps                         -- the wrapped block tail
    …

This file states, one op at a time, what the KOps semantics does with
each of those pieces — the exit site sets the flag and breaks, skipping
the rest of the body; a non-firing site falls through; the declaration
is one register write; the tail wrap runs its payload exactly when the
flag is `false`; the loop's `reset_broke` keeps the register file — and
the refinement fact the loop proof threads past all of it: `Refines`
survives a write to a register the lowering does not hold, and never
reads `broke`.

Everything here is about `evalOps` / `evalOp` on concrete op shapes and
the `Refines` record; nothing is about the translator.
-/

import Quanta.Wasm.PreservationWhile

namespace Quanta.Wasm

open Quanta.KOps (KernelOp Reg ConstValue Value State evalOps evalOp evalConst
                  regLookup regWrite vBool)

-- ════════════════════════════════════════════════════════════════════
-- Single-op steps and how `evalOps` sequences them
-- ════════════════════════════════════════════════════════════════════

/-- The boolean constant evaluates to the boolean value. Stated so the
    flag lemmas can speak in `vBool` without unfolding `evalConst`. -/
theorem evalConst_bool (b : Bool) : evalConst (.bool b) = vBool b := rfl

/-- A `const` is one register write; `broke` is untouched. -/
theorem evalOp_const (F : Nat) (kst : State) (dst : Reg) (c : ConstValue) :
    evalOp F kst (.const dst c) = some { kst with rf := regWrite kst.rf dst (evalConst c) } := by
  simp only [evalOp, Option.pure_def]

/-- A `breakOp` raises `broke` and touches nothing else. -/
theorem evalOp_breakOp (F : Nat) (kst : State) :
    evalOp F kst .breakOp = some { kst with broke := true } := by
  simp only [evalOp, Option.pure_def]

/-- The empty op list is the identity. -/
theorem evalOps_nil (F : Nat) (kst : State) : evalOps F kst [] = some kst := by
  simp only [evalOps]

/-- A `branch` whose condition register reads `true` is its then-arm. -/
theorem evalOp_branch_true {F : Nat} {kst : State} {c : Reg} {t e : List KernelOp}
    (h_c : regLookup kst.rf c = some (vBool true)) :
    evalOp F kst (.branch c t e) = evalOps F kst t := by
  simp only [evalOp, h_c, vBool, Option.bind_eq_bind, Option.some_bind]

/-- A `branch` whose condition register reads `false` is its else-arm. -/
theorem evalOp_branch_false {F : Nat} {kst : State} {c : Reg} {t e : List KernelOp}
    (h_c : regLookup kst.rf c = some (vBool false)) :
    evalOp F kst (.branch c t e) = evalOps F kst e := by
  simp only [evalOp, h_c, vBool, Option.bind_eq_bind, Option.some_bind]

/-- `evalOps` past an op that leaves `broke` clear: continue with the
    rest from the op's output state. -/
theorem evalOps_cons_continue {F : Nat} {kst kst1 : State} {op : KernelOp}
    {rest : List KernelOp}
    (h_op : evalOp F kst op = some kst1) (h_ok : kst1.broke = false) :
    evalOps F kst (op :: rest) = evalOps F kst1 rest := by
  simp only [evalOps, h_op, Option.bind_eq_bind, Option.some_bind, h_ok,
             Bool.false_eq_true, ↓reduceIte]

/-- `evalOps` at an op that raises `broke`: stop there, the rest is
    skipped. -/
theorem evalOps_cons_stop {F : Nat} {kst kst1 : State} {op : KernelOp}
    {rest : List KernelOp}
    (h_op : evalOp F kst op = some kst1) (h_br : kst1.broke = true) :
    evalOps F kst (op :: rest) = some kst1 := by
  simp only [evalOps, h_op, Option.bind_eq_bind, Option.some_bind, h_br, ↓reduceIte]

-- ════════════════════════════════════════════════════════════════════
-- The exit site: `branch cb [const flag true, breakOp] []`
-- ════════════════════════════════════════════════════════════════════

/-- A `breakOp` at the head of an op list ends the list: the state
    comes back with `broke` raised and nothing after it runs. -/
theorem evalOps_breakOp_cons (F : Nat) (kst : State) (rest : List KernelOp) :
    evalOps F kst (.breakOp :: rest) = some { kst with broke := true } :=
  evalOps_cons_stop (evalOp_breakOp F kst) rfl

/-- The exit site's then-arm: set the flag, then break. The result
    holds the flag at `true` with `broke` raised, whatever `rest` is.
    Unconditional in `kst.broke`: when it was already raised the
    `const` step returns early, and that early state IS the stated one
    (its `broke` field was `true` to begin with). -/
theorem evalOps_setFlag_break (F : Nat) (kst : State) (flag : Reg) (rest : List KernelOp) :
    evalOps F kst (.const flag (.bool true) :: .breakOp :: rest)
      = some { kst with rf := regWrite kst.rf flag (vBool true), broke := true } := by
  cases h_br : kst.broke with
  | false =>
      rw [evalOps_cons_continue (evalOp_const F kst flag (.bool true)) h_br,
          evalOps_breakOp_cons, evalConst_bool]
  | true =>
      rw [evalOps_cons_stop (evalOp_const F kst flag (.bool true)) h_br, evalConst_bool]
      -- `{ kst with broke := true }` is `kst` itself once `kst.broke = true`.
      cases kst
      simp only at h_br
      subst h_br
      rfl

/-- The exit site with `broke` already raised is the same early return,
    now spelled without the `broke := true` override. -/
theorem evalOps_setFlag_break_of_broke (F : Nat) {kst : State} (flag : Reg)
    (rest : List KernelOp) (h_br : kst.broke = true) :
    evalOps F kst (.const flag (.bool true) :: .breakOp :: rest)
      = some { kst with rf := regWrite kst.rf flag (vBool true) } := by
  rw [evalOps_cons_stop (evalOp_const F kst flag (.bool true)) h_br, evalConst_bool]

/-- The exit site FIRES: the condition register reads `true`, so the
    flag is set, the loop body is broken out of, and `rest` — the part
    of the body after the site — is skipped. No `kst.broke = false`
    needed: the then-arm's output has `broke` raised whatever it was on
    entry (`evalOps_setFlag_break`), and that is what stops the list. -/
theorem evalOps_exitSite_fires {F : Nat} {kst : State} {cb flag : Reg}
    {rest : List KernelOp}
    (h_cb : regLookup kst.rf cb = some (vBool true)) :
    evalOps F kst (.branch cb [.const flag (.bool true), .breakOp] [] :: rest)
      = some { kst with rf := regWrite kst.rf flag (vBool true), broke := true } := by
  have h_site : evalOp F kst (.branch cb [.const flag (.bool true), .breakOp] [])
      = some { kst with rf := regWrite kst.rf flag (vBool true), broke := true } := by
    rw [evalOp_branch_true h_cb, evalOps_setFlag_break]
  exact evalOps_cons_stop h_site rfl

/-- The exit site FALLS THROUGH: the condition register reads `false`,
    the empty else-arm runs, and evaluation continues with `rest` from
    the unchanged state. -/
theorem evalOps_exitSite_falls_through {F : Nat} {kst : State} {cb flag : Reg}
    {rest : List KernelOp}
    (h_ok : kst.broke = false) (h_cb : regLookup kst.rf cb = some (vBool false)) :
    evalOps F kst (.branch cb [.const flag (.bool true), .breakOp] [] :: rest)
      = evalOps F kst rest := by
  have h_site : evalOp F kst (.branch cb [.const flag (.bool true), .breakOp] []) = some kst := by
    rw [evalOp_branch_false h_cb, evalOps_nil]
  exact evalOps_cons_continue h_site h_ok

-- ════════════════════════════════════════════════════════════════════
-- The declaration and the tail wrap
-- ════════════════════════════════════════════════════════════════════

/-- The declaration `const flag false` ahead of the loop: one register
    write, then on with `rest`. -/
theorem evalOps_decl {F : Nat} {kst : State} {flag : Reg} {rest : List KernelOp}
    (h_ok : kst.broke = false) :
    evalOps F kst (.const flag (.bool false) :: rest)
      = evalOps F { kst with rf := regWrite kst.rf flag (vBool false) } rest := by
  rw [evalOps_cons_continue (evalOp_const F kst flag (.bool false)) h_ok, evalConst_bool]

/-- The tail wrap `branch flag [] tail` SKIPS its payload when the flag
    reads `true` (the loop was exited through the site): the empty
    then-arm runs and `rest` continues from the unchanged state. -/
theorem evalOps_tailWrap_skips {F : Nat} {kst : State} {flag : Reg}
    {tail rest : List KernelOp}
    (h_ok : kst.broke = false) (h_flag : regLookup kst.rf flag = some (vBool true)) :
    evalOps F kst (.branch flag [] tail :: rest) = evalOps F kst rest := by
  have h_wrap : evalOp F kst (.branch flag [] tail) = some kst := by
    rw [evalOp_branch_true h_flag, evalOps_nil]
  exact evalOps_cons_continue h_wrap h_ok

/-- The tail wrap RUNS its payload when the flag reads `false` (the
    loop was left some other way): `tail` runs to `kst1`, and provided
    it did not break, `rest` continues from there. Only `kst1.broke` is
    consulted — `evalOps` checks the flag after the wrap op, i.e. on
    the payload's output, never on `kst` itself. -/
theorem evalOps_tailWrap_runs {F : Nat} {kst kst1 : State} {flag : Reg}
    {tail rest : List KernelOp}
    (h_flag : regLookup kst.rf flag = some (vBool false))
    (h_tail : evalOps F kst tail = some kst1) (h_ok1 : kst1.broke = false) :
    evalOps F kst (.branch flag [] tail :: rest) = evalOps F kst1 rest := by
  have h_wrap : evalOp F kst (.branch flag [] tail) = some kst1 := by
    rw [evalOp_branch_false h_flag]; exact h_tail
  exact evalOps_cons_continue h_wrap h_ok1

-- ════════════════════════════════════════════════════════════════════
-- `reset_broke` — what the loop op returns
-- ════════════════════════════════════════════════════════════════════

/-- `reset_broke` keeps the register file. -/
@[simp] theorem State.reset_broke_rf (s : State) : s.reset_broke.rf = s.rf := rfl

/-- `reset_broke` keeps the heap. -/
@[simp] theorem State.reset_broke_heap (s : State) : s.reset_broke.heap = s.heap := rfl

/-- `reset_broke` clears `broke`. -/
@[simp] theorem State.reset_broke_broke (s : State) : s.reset_broke.broke = false := rfl

/-- After the exit site fired and the loop op returned: the flag still
    reads `true` — `reset_broke` touched only `broke`, and the site's
    write to the flag is the last write to that register. -/
theorem regLookup_after_exit (kst : State) (flag : Reg) :
    regLookup ({ kst with rf := regWrite kst.rf flag (vBool true), broke := true }.reset_broke).rf
        flag
      = some (vBool true) :=
  regLookup_regWrite_self kst.rf flag (vBool true)

-- ════════════════════════════════════════════════════════════════════
-- Refinement survives the flag: a write to a register the lowering does
-- not hold, and any change to `broke`
-- ════════════════════════════════════════════════════════════════════

/-- Encoding survives a write at or above a bound every register of the
    SymVal is below. The flag register is allocated from `nextReg`, so
    with `n := s.nextReg` this is exactly "the flag is not a register
    any stack or local encoding reads". -/
theorem WasmValue.encodes_regWrite_of_fresh
    {v : WasmValue} {layout : BufferLayout} {rf : Quanta.KOps.RegFile} {sv : SymVal}
    {n r : Reg} {w : Value}
    (h_lt : ∀ r' ∈ sv.regs, r' < n) (h_le : n ≤ r)
    (h : v.encodes layout rf sv) :
    v.encodes layout (regWrite rf r w) sv :=
  WasmValue.encodes_preserved_of_fresh
    (fun r' hr' => Nat.lt_of_lt_of_le (h_lt r' hr') h_le) h

/-- `StackRefines` survives a write at or above `nextReg`: every stack
    register is below `nextReg` by `Fresh`. -/
theorem StackRefines_regWrite_fresh
    {layout : BufferLayout} {ws : WasmState} {s : LowerState} {rf : Quanta.KOps.RegFile}
    (h_stk : StackRefines layout ws.stack s.stack rf) (h_fresh : Fresh s)
    {r : Reg} (h_le : s.nextReg ≤ r) (w : Value) :
    StackRefines layout ws.stack s.stack (regWrite rf r w) := by
  refine ⟨h_stk.left, ?_⟩
  intro i v hv
  obtain ⟨sv, hsv, henc⟩ := h_stk.right i v hv
  refine ⟨sv, hsv, ?_⟩
  have hsv_in : sv ∈ s.stack := List.mem_of_get? hsv
  exact WasmValue.encodes_regWrite_of_fresh (h_fresh.left sv hsv_in) h_le henc

/-- `LocalsRefines` survives a write at or above `nextReg`: every
    stable register is below `nextReg` by `Fresh`. -/
theorem LocalsRefines_regWrite_fresh
    {layout : BufferLayout} {ws : WasmState} {s : LowerState} {rf : Quanta.KOps.RegFile}
    (h_locs : LocalsRefines layout ws.locals s.localReg s.localTy rf) (h_fresh : Fresh s)
    {r : Reg} (h_le : s.nextReg ≤ r) (w : Value) :
    LocalsRefines layout ws.locals s.localReg s.localTy (regWrite rf r w) := by
  intro i q hfind v hv
  have henc := h_locs i q hfind v hv
  have hpair : (i, q) ∈ s.localReg := List.mem_of_find?_eq_some hfind
  have hq_lt : q < s.nextReg := h_fresh.right (i, q) hpair
  apply WasmValue.encodes_regWrite_of_fresh _ h_le henc
  intro r' hr'
  simp only [SymVal.regs, List.mem_singleton] at hr'
  rw [hr']; exact hq_lt

/-- `CurrentRegRefines` survives a write at or above `nextReg`: every
    per-frame binding register is below `nextReg` by `FreshCurrent`. -/
theorem CurrentRegRefines_regWrite_fresh
    {layout : BufferLayout} {ws : WasmState} {s : LowerState} {rf : Quanta.KOps.RegFile}
    (h_cur : CurrentRegRefines layout ws.locals s.currentReg s.localTy rf)
    (h_fresh : FreshCurrent s)
    {r : Reg} (h_le : s.nextReg ≤ r) (w : Value) :
    CurrentRegRefines layout ws.locals s.currentReg s.localTy (regWrite rf r w) :=
  CurrentRegRefines_preserved_fresh h_cur
    (fun ir hir => Nat.lt_of_lt_of_le (h_fresh ir hir) h_le) w

/-- THE refinement fact behind the flag: writing a register at or above
    `nextReg` — one the lowering has not handed to any stack slot or
    local — changes nothing `Refines` reads. The flag's declaration,
    its set at the exit site, and the condition casts feeding the site
    are all such writes. -/
theorem Refines.regWrite_fresh
    {ws : WasmState} {s : LowerState} {kst : State} {layout : BufferLayout}
    (R : Refines ws s kst layout) {r : Reg} (h_le : s.nextReg ≤ r) (v : Value) :
    Refines ws s { kst with rf := regWrite kst.rf r v } layout :=
  ⟨StackRefines_regWrite_fresh R.stk R.fresh h_le v,
   LocalsRefines_regWrite_fresh R.locs R.fresh h_le v,
   R.fresh, R.aliasFree, R.injLocals, R.heapRefines,
   CurrentRegRefines_regWrite_fresh R.currentReg R.freshCurrent h_le v,
   R.freshCurrent, R.curLocDisj⟩

/-- `Refines` never reads `broke`: none of its nine fields mentions it. -/
theorem Refines.set_broke
    {ws : WasmState} {s : LowerState} {kst : State} {layout : BufferLayout}
    (R : Refines ws s kst layout) (b : Bool) :
    Refines ws s { kst with broke := b } layout :=
  ⟨R.stk, R.locs, R.fresh, R.aliasFree, R.injLocals, R.heapRefines, R.currentReg,
   R.freshCurrent, R.curLocDisj⟩

/-- `Refines` survives the loop op's `reset_broke` (the `false` case of
    `set_broke`, spelled the way `opLoop` returns it). -/
theorem Refines.reset_broke
    {ws : WasmState} {s : LowerState} {kst : State} {layout : BufferLayout}
    (R : Refines ws s kst layout) :
    Refines ws s kst.reset_broke layout :=
  R.set_broke false

/-- The exit site's output state, in one step: the flag written above
    `nextReg` and `broke` raised — `Refines` still holds. -/
theorem Refines.regWrite_fresh_set_broke
    {ws : WasmState} {s : LowerState} {kst : State} {layout : BufferLayout}
    (R : Refines ws s kst layout) {r : Reg} (h_le : s.nextReg ≤ r)
    (v : Value) (b : Bool) :
    Refines ws s { kst with rf := regWrite kst.rf r v, broke := b } layout :=
  (R.regWrite_fresh h_le v).set_broke b

end Quanta.Wasm
