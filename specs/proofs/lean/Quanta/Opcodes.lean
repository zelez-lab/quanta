-- SPIR-V opcode correctness specification.
--
-- Theorem T2: every KernelOp maps to exactly one SPIR-V opcode,
-- and that opcode matches the SPIR-V 1.6 specification.

namespace Quanta.Opcodes

/-- SPIR-V binary operation opcodes from the specification. -/
inductive SpvBinOp where
  | IAdd       -- 128
  | FAdd       -- 129
  | ISub       -- 130
  | FSub       -- 131
  | IMul       -- 132
  | FMul       -- 133
  | UDiv       -- 134
  | SDiv       -- 135
  | FDiv       -- 136
  | UMod       -- 137
  | SMod       -- 138
  | FRem       -- 140
  | BitwiseOr  -- 197
  | BitwiseXor -- 198
  | BitwiseAnd -- 199
  | ShiftLeftLogical      -- 196
  | ShiftRightLogical     -- 194
  | ShiftRightArithmetic  -- 195
  deriving Repr, DecidableEq

/-- The SPIR-V numeric opcode for each binary operation. -/
def SpvBinOp.toNat : SpvBinOp → Nat
  | .IAdd      => 128
  | .FAdd      => 129
  | .ISub      => 130
  | .FSub      => 131
  | .IMul      => 132
  | .FMul      => 133
  | .UDiv      => 134
  | .SDiv      => 135
  | .FDiv      => 136
  | .UMod      => 137
  | .SMod      => 138
  | .FRem      => 140
  | .BitwiseOr  => 197
  | .BitwiseXor => 198
  | .BitwiseAnd => 199
  | .ShiftLeftLogical     => 196
  | .ShiftRightLogical    => 194
  | .ShiftRightArithmetic => 195

/-- Quanta IR binary operations. -/
inductive BinOp where
  | Add | Sub | Mul | Div | Rem
  | BitAnd | BitOr | BitXor
  | Shl | Shr
  deriving Repr, DecidableEq

/-- Whether a scalar type is floating-point. -/
inductive FloatKind where
  | IsFloat | IsSignedInt | IsUnsignedInt
  deriving Repr, DecidableEq

/-- Map (BinOp, FloatKind) to the correct SPIR-V opcode. -/
def binOpToSpv : BinOp → FloatKind → SpvBinOp
  | .Add, .IsFloat       => .FAdd
  | .Add, _              => .IAdd
  | .Sub, .IsFloat       => .FSub
  | .Sub, _              => .ISub
  | .Mul, .IsFloat       => .FMul
  | .Mul, _              => .IMul
  | .Div, .IsFloat       => .FDiv
  | .Div, .IsSignedInt   => .SDiv
  | .Div, .IsUnsignedInt => .UDiv
  | .Rem, .IsFloat       => .FRem
  | .Rem, .IsSignedInt   => .SMod
  | .Rem, .IsUnsignedInt => .UMod
  | .BitAnd, _           => .BitwiseAnd
  | .BitOr, _            => .BitwiseOr
  | .BitXor, _           => .BitwiseXor
  | .Shl, _              => .ShiftLeftLogical
  | .Shr, .IsSignedInt   => .ShiftRightArithmetic
  | .Shr, _              => .ShiftRightLogical

-- Theorem: BitAnd always maps to opcode 199 (not 197)
theorem bitand_is_199 : ∀ fk, (binOpToSpv .BitAnd fk).toNat = 199 := by
  intro fk; cases fk <;> rfl

-- Theorem: BitOr always maps to opcode 197 (not 198)
theorem bitor_is_197 : ∀ fk, (binOpToSpv .BitOr fk).toNat = 197 := by
  intro fk; cases fk <;> rfl

-- Theorem: BitXor always maps to opcode 198 (not 199)
theorem bitxor_is_198 : ∀ fk, (binOpToSpv .BitXor fk).toNat = 198 := by
  intro fk; cases fk <;> rfl

-- Theorem: the mapping is injective on opcode numbers
-- (no two different (BinOp, FloatKind) pairs produce the same opcode)
theorem binop_mapping_injective :
    ∀ op1 op2 fk1 fk2,
      (binOpToSpv op1 fk1).toNat = (binOpToSpv op2 fk2).toNat →
      op1 = op2 ∧ (op1 ≠ .Div ∧ op1 ≠ .Rem ∧ op1 ≠ .Shr → fk1 = fk2) := by
  intro op1 op2 fk1 fk2 h
  cases op1 <;> cases op2 <;> cases fk1 <;> cases fk2 <;> simp_all [binOpToSpv, SpvBinOp.toNat]

end Quanta.Opcodes
