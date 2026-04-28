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

-- The original sketch claimed `op1 = op2 ∧ (op1 = .Neg → fk1 = fk2)`,
-- but `Neg IsSignedInt` and `Neg IsUnsignedInt` both map to `.SNegate`,
-- so `fk1 = fk2` doesn't follow. Right pair of theorems mirrors the
-- BinOp story in `Opcodes.lean`: numeric tags are injective on
-- `SpvUnaryOp`, and `unaryOpToSpv` is consistent under `toNat`
-- equality. Round-tripping the wire format only needs that the
-- `SpvUnaryOp` is recoverable, not the input `(op, fk)` pair.

/-- `SpvUnaryOp.toNat` is injective on the inductive variants. -/
theorem spv_unary_toNat_injective :
    ∀ x y : SpvUnaryOp, x.toNat = y.toNat → x = y := by
  intro x y h
  cases x <;> cases y <;> simp_all [SpvUnaryOp.toNat]

/-- Two `(op, fk)` inputs that produce the same numeric opcode produce
    the same `SpvUnaryOp` value. -/
theorem unaryop_mapping_consistent :
    ∀ op1 op2 fk1 fk2,
      (unaryOpToSpv op1 fk1).toNat = (unaryOpToSpv op2 fk2).toNat →
      unaryOpToSpv op1 fk1 = unaryOpToSpv op2 fk2 := by
  intro op1 op2 fk1 fk2 h
  exact spv_unary_toNat_injective _ _ h

end Quanta.UnaryOps
