/-
# Well-formed kernel predicate

`wellFormedInstr` is a Bool recognizer over `WasmInstr` that's true
for every constructor the lowering pass (per-op `lowerInstr` or
list-level `lowerInstrs`) accepts. Concretely, it admits the union
of:

* The 20 instructions `closedInstr` already recognizes (nop /
  i32Const / drop / localSet / localTee / 8 unsigned i32 binops /
  6 unsigned i32 cmps).
* `localGet` for any local — whether buffer-typed or not, both
  arms succeed.
* `i32Add`, `i32Shl` — buffer-pattern fast-path plus the generic
  binop fallback.
* `i32Load`, `i32Store` — buffer-pattern address arms (the only
  shape `lowerInstr` accepts on the load/store side).
* `wreturn` — emits no IR; closes the function-level scope.
* Structured-control constructors that `lowerInstrs` handles at
  the list level: `block`, `wloop`, `wif`, `br`, `brIf`.

The deliberate refusals are the constructors the lowering pass
returns `none` for under any input:

* `i64Const`, `f32Const`, `f64Const` — outside the slice-1 type
  set.
* Signed-i32 ops: `i32DivS`, `i32RemS`, `i32ShrS`, `i32LtS`,
  `i32GtS`, `i32LeS`, `i32GeS`, `i32Eqz` — refused (slice-2).
* All `f32` arithmetic / comparison / unary ops — refused.
* Type conversions (`i32WrapI64`, `f32ConvertI32{S,U}`,
  `i32TruncF32{S,U}`, reinterpret pair) — refused.
* Byte-level memory (`i32Load8U`, `i32Load8S`, `i32Store8`) and
  `f32` memory — refused.
* Stream terminators `welse`, `wend` — only valid as inner
  markers consumed by `splitAtEnd` / `splitAtElseOrEnd`, never
  as top-level instructions in a well-formed kernel body.
* `call` (non-imported) — slice-2.
* `wselect`, `unreachable`, `unsupported` — refused.

`wellFormedInstrs` extends this element-wise to instruction
lists. Used by the L10 top-level framework theorem to scope the
preservation claim to lists the lowering can actually handle.
-/

import Quanta.Wasm.Syntax
import Quanta.Wasm.PreservationInduction

namespace Quanta.Wasm

-- ════════════════════════════════════════════════════════════════════
-- wellFormedInstr / wellFormedInstrs
-- ════════════════════════════════════════════════════════════════════

/-- Bool recognizer for every `WasmInstr` constructor that `lowerInstr`
    or `lowerInstrs` accepts. The complement is the set of refused
    constructors (slice-1 scope boundary plus the dangling
    `welse`/`wend` terminators). -/
def wellFormedInstr : WasmInstr → Bool
  -- Constants
  | .i32Const _   => true
  -- Locals: both buffer and non-buffer arms succeed.
  | .localGet _   => true
  | .localSet _   => true
  | .localTee _   => true
  -- i32 arithmetic — unsigned (signed deferred to slice 2).
  | .i32Add       => true
  | .i32Sub       => true
  | .i32Mul       => true
  | .i32And       => true
  | .i32Or        => true
  | .i32Xor       => true
  | .i32Shl       => true
  | .i32ShrU      => true
  | .i32DivU      => true
  | .i32RemU      => true
  -- i32 comparisons — unsigned.
  | .i32Eq        => true
  | .i32Ne        => true
  | .i32LtU       => true
  | .i32LeU       => true
  | .i32GtU       => true
  | .i32GeU       => true
  -- Memory — buffer-pattern address arms only.
  | .i32Load _ _  => true
  | .i32Store _ _ => true
  -- Misc
  | .nop          => true
  | .drop         => true
  | .wreturn      => true
  -- Structured control — handled by `lowerInstrs`, not per-op.
  | .block _      => true
  | .wloop _      => true
  | .wif _        => true
  | .br _         => true
  | .brIf _       => true
  -- Everything else is refused.
  | _             => false

/-- Well-formed instruction list: every element is `wellFormedInstr`. -/
def wellFormedInstrs : List WasmInstr → Bool
  | []        => true
  | i :: rest => wellFormedInstr i && wellFormedInstrs rest

-- ════════════════════════════════════════════════════════════════════
-- Cons destructors
-- ════════════════════════════════════════════════════════════════════

theorem wellFormedInstrs_cons {i : WasmInstr} {rest : List WasmInstr} :
    wellFormedInstrs (i :: rest) = (wellFormedInstr i && wellFormedInstrs rest) := rfl

theorem wellFormedInstrs_head {i : WasmInstr} {rest : List WasmInstr}
    (h : wellFormedInstrs (i :: rest) = true) : wellFormedInstr i = true :=
  (Bool.and_eq_true _ _).mp h |>.left

theorem wellFormedInstrs_tail {i : WasmInstr} {rest : List WasmInstr}
    (h : wellFormedInstrs (i :: rest) = true) : wellFormedInstrs rest = true :=
  (Bool.and_eq_true _ _).mp h |>.right

-- ════════════════════════════════════════════════════════════════════
-- Bridges from the narrower closed recognizers
-- ════════════════════════════════════════════════════════════════════

/-- Every `closedInstr`-recognized arm is well-formed. Closed is a
    strict subset (it excludes localGet, i32Add, i32Shl, mem ops,
    structured-control, wreturn). -/
theorem wellFormedInstr_of_closedInstr
    {i : WasmInstr} (h : closedInstr i = true) :
    wellFormedInstr i = true := by
  cases i with
  | nop          => rfl
  | i32Const _   => rfl
  | drop         => rfl
  | localSet _   => rfl
  | localTee _   => rfl
  | i32Sub       => rfl
  | i32Mul       => rfl
  | i32And       => rfl
  | i32Or        => rfl
  | i32Xor       => rfl
  | i32ShrU      => rfl
  | i32DivU      => rfl
  | i32RemU      => rfl
  | i32Eq        => rfl
  | i32Ne        => rfl
  | i32LtU       => rfl
  | i32LeU       => rfl
  | i32GtU       => rfl
  | i32GeU       => rfl
  -- All other arms have closedInstr = false; h is False.
  | localGet _ => simp [closedInstr] at h
  | i64Const _ => simp [closedInstr] at h
  | f32Const _ => simp [closedInstr] at h
  | f64Const _ => simp [closedInstr] at h
  | i32Add => simp [closedInstr] at h
  | i32DivS => simp [closedInstr] at h
  | i32RemS => simp [closedInstr] at h
  | i32Shl => simp [closedInstr] at h
  | i32ShrS => simp [closedInstr] at h
  | i32LtS => simp [closedInstr] at h
  | i32GtS => simp [closedInstr] at h
  | i32LeS => simp [closedInstr] at h
  | i32GeS => simp [closedInstr] at h
  | i32Eqz => simp [closedInstr] at h
  | f32Add => simp [closedInstr] at h
  | f32Sub => simp [closedInstr] at h
  | f32Mul => simp [closedInstr] at h
  | f32Div => simp [closedInstr] at h
  | f32Eq => simp [closedInstr] at h
  | f32Ne => simp [closedInstr] at h
  | f32Lt => simp [closedInstr] at h
  | f32Gt => simp [closedInstr] at h
  | f32Le => simp [closedInstr] at h
  | f32Ge => simp [closedInstr] at h
  | f32Neg => simp [closedInstr] at h
  | f32Abs => simp [closedInstr] at h
  | f32Sqrt => simp [closedInstr] at h
  | f32Min => simp [closedInstr] at h
  | f32Max => simp [closedInstr] at h
  | i32WrapI64 => simp [closedInstr] at h
  | f32ConvertI32S => simp [closedInstr] at h
  | f32ConvertI32U => simp [closedInstr] at h
  | i32TruncF32S => simp [closedInstr] at h
  | i32TruncF32U => simp [closedInstr] at h
  | f32ReinterpretI32 => simp [closedInstr] at h
  | i32ReinterpretF32 => simp [closedInstr] at h
  | i32Load _ _ => simp [closedInstr] at h
  | i32Store _ _ => simp [closedInstr] at h
  | f32Load _ _ => simp [closedInstr] at h
  | f32Store _ _ => simp [closedInstr] at h
  | i32Load8U _ _ => simp [closedInstr] at h
  | i32Load8S _ _ => simp [closedInstr] at h
  | i32Store8 _ _ => simp [closedInstr] at h
  | block _ => simp [closedInstr] at h
  | wloop _ => simp [closedInstr] at h
  | wif _ => simp [closedInstr] at h
  | welse => simp [closedInstr] at h
  | wend => simp [closedInstr] at h
  | br _ => simp [closedInstr] at h
  | brIf _ => simp [closedInstr] at h
  | wreturn => simp [closedInstr] at h
  | call _ => simp [closedInstr] at h
  | wselect => simp [closedInstr] at h
  | unreachable => simp [closedInstr] at h
  | unsupported _ => simp [closedInstr] at h

/-- Every `closedInstrAt s` -recognized arm is well-formed. The
    state-aware admission for `localGet` lands in the `localGet`
    arm of `wellFormedInstr`, which is unconditionally true. -/
theorem wellFormedInstr_of_closedInstrAt
    {s : LowerState} {i : WasmInstr} (h : closedInstrAt s i = true) :
    wellFormedInstr i = true := by
  cases i with
  | localGet _ => rfl
  | _ => exact wellFormedInstr_of_closedInstr h

/-- List-level bridge from `closedInstrs`. -/
theorem wellFormedInstrs_of_closedInstrs
    {instrs : List WasmInstr} (h : closedInstrs instrs = true) :
    wellFormedInstrs instrs = true := by
  induction instrs with
  | nil => rfl
  | cons i rest ih =>
      have h_head : closedInstr i = true := closedInstrs_head h
      have h_rest : closedInstrs rest = true := closedInstrs_tail h
      show (wellFormedInstr i && wellFormedInstrs rest) = true
      rw [wellFormedInstr_of_closedInstr h_head, ih h_rest]
      rfl

/-- List-level bridge from `closedInstrsAt`. -/
theorem wellFormedInstrs_of_closedInstrsAt
    {s : LowerState} {instrs : List WasmInstr}
    (h : closedInstrsAt s instrs = true) :
    wellFormedInstrs instrs = true := by
  induction instrs with
  | nil => rfl
  | cons i rest ih =>
      have h_head : closedInstrAt s i = true := closedInstrsAt_head h
      have h_rest : closedInstrsAt s rest = true := closedInstrsAt_tail h
      show (wellFormedInstr i && wellFormedInstrs rest) = true
      rw [wellFormedInstr_of_closedInstrAt h_head, ih h_rest]
      rfl

end Quanta.Wasm
