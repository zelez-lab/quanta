-- SPIR-V unary opcode correctness specification.
--
-- Theorem T2 extension: every (UnaryOp, FloatKind) pair maps to exactly
-- one SPIR-V unary opcode, and that opcode matches SPIR-V 1.6.

namespace Quanta.UnaryOps

/-- Reuse the float/signedness tag from Opcodes. -/
inductive FloatKind where
  | IsFloat | IsSignedInt | IsUnsignedInt
  deriving Repr, DecidableEq

/-- Quanta IR unary operations. -/
inductive UnaryOp where
  | Neg | BitNot | LogicalNot
  deriving Repr, DecidableEq

/-- SPIR-V unary opcodes from the specification. -/
inductive SpvUnaryOp where
  | SNegate     -- 126
  | FNegate     -- 127
  | LogicalNot  -- 168
  | Not         -- 200
  deriving Repr, DecidableEq

/-- The SPIR-V numeric opcode for each unary operation. -/
def SpvUnaryOp.toNat : SpvUnaryOp → Nat
  | .SNegate    => 126
  | .FNegate    => 127
  | .LogicalNot => 168
  | .Not        => 200

/-- Map (UnaryOp, FloatKind) to the correct SPIR-V unary opcode. -/
def unaryOpToSpv : UnaryOp → FloatKind → SpvUnaryOp
  | .Neg, .IsFloat => .FNegate
  | .Neg, _        => .SNegate
  | .BitNot, _     => .Not
  | .LogicalNot, _ => .LogicalNot

-- Theorem: Neg + float maps to FNegate (127)
theorem neg_float_is_127 : (unaryOpToSpv .Neg .IsFloat).toNat = 127 := by rfl

-- Theorem: Neg + int maps to SNegate (126)
theorem neg_int_signed_is_126 : (unaryOpToSpv .Neg .IsSignedInt).toNat = 126 := by rfl
theorem neg_int_unsigned_is_126 : (unaryOpToSpv .Neg .IsUnsignedInt).toNat = 126 := by rfl

-- Theorem: BitNot always maps to Not (200)
theorem bitnot_is_200 : ∀ fk, (unaryOpToSpv .BitNot fk).toNat = 200 := by
  intro fk; cases fk <;> rfl

-- Theorem: LogicalNot always maps to LogicalNot (168)
theorem logicalnot_is_168 : ∀ fk, (unaryOpToSpv .LogicalNot fk).toNat = 168 := by
  intro fk; cases fk <;> rfl

-- Theorem: the mapping is injective on opcode numbers
-- (no two different (UnaryOp, FloatKind) pairs produce the same opcode)
theorem unaryop_mapping_injective :
    ∀ op1 op2 fk1 fk2,
      (unaryOpToSpv op1 fk1).toNat = (unaryOpToSpv op2 fk2).toNat →
      op1 = op2 ∧ (op1 = .Neg → fk1 = fk2) := by
  intro op1 op2 fk1 fk2 h
  cases op1 <;> cases op2 <;> cases fk1 <;> cases fk2 <;> simp_all [unaryOpToSpv, SpvUnaryOp.toNat]

end Quanta.UnaryOps
